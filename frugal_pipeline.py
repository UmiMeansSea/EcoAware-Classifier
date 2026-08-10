# -*- coding: utf-8 -*-
"""
frugal_pipeline.py
==================
End-to-end Frugal AI (sobriété numérique) compression pipeline for CIFAR-10.

Pipeline stages
---------------
  Phase A — FP32 Baseline training with early stopping + regularisation
  Phase B — Iterative L1 magnitude pruning (5 steps → 70 % sparsity)
  Phase C — Quantisation-Aware Training (QAT) → INT8 (FX graph mode)

All three phases are wrapped inside a single CodeCarbon tracker so the
final audit reports the *total* operational carbon footprint.

Outputs
-------
  pipeline_fp32.pth    — best FP32 checkpoint (float weights)
  pipeline_pruned.pth  — permanently pruned FP32 state_dict
  pipeline_int8.pth    — full quantised GraphModule object
"""

# =============================================================================
# 0. Imports & global settings
# =============================================================================
import os
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
# Eager Mode QAT — no FX graph imports needed
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. Model — Eager-Mode QAT-ready SimpleCNN
# =============================================================================
class SimpleCNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10 (32 × 32 RGB → 10 classes).

    Architecture
    ~~~~~~~~~~~~
    conv1 → BN → ReLU → MaxPool(2)   : 3 → 16 channels
    conv2 → BN → ReLU → MaxPool(2)   : 16 → 32 channels
    flatten                           : 32 × 8 × 8 = 2 048 features
    fc1   → ReLU → Dropout(0.5)      : 2 048 → 256
    fc2   (logits)                    : 256  → 10

    Eager-Mode QAT additions vs original
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    • QuantStub / DeQuantStub  — marks FP32→INT8 / INT8→FP32 boundaries.
      These are identity ops during Phase A & B; only Phase C activates them.
    • nn.ReLU modules instead of F.relu — required so fuse_modules() can
      collapse Conv→BN→ReLU into a single ConvBnReLU2d intrinsic module.
    • fuse() method — called once before prepare_qat to enable INT8 fusion.
    """

    def __init__(self):
        super().__init__()

        self.quant = torch.quantization.QuantStub()

        # ── Feature extractor ──────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)   # named module for fuse_modules()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Classifier head ────────────────────────────────────────────────
        self.fc1     = nn.Linear(32 * 8 * 8, 256)
        self.relu3   = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # 1. FP32 Input Block (Highly sensitive edge layer)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))

        # 2. INT8 Heavy Processing Block
        x = self.quant(x)
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.dequant(x)

        # 3. FP32 Classification Head (Highly sensitive logits)
        x = self.fc2(x)
        return x

    def fuse(self) -> None:
        """Fuse Conv→BN→ReLU pairs. Call once on CPU in eval() before prepare_qat."""
        torch.quantization.fuse_modules(
            self, [["conv1", "bn1", "relu1"],
                   ["conv2", "bn2", "relu2"]],
            inplace=True,
        )


# =============================================================================
# 2. Data pipelines
# =============================================================================
# CIFAR-10 channel statistics
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ---------------------------------------------------------------------------
# Helper — build all loaders in one place
# ---------------------------------------------------------------------------
def build_loaders(data_root: str = "./data", batch_size: int = 128, num_workers: int = 2):
    """Return (trainloader, valloader, testloader, calib_loader)."""

    full_trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )

    # 80 / 20 train-val split (deterministic seed for reproducibility)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.80 * len(full_trainset))
    val_size   = len(full_trainset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_trainset, [train_size, val_size], generator=generator
    )

    # Val uses eval transforms — wrap the subset
    val_subset.dataset = copy.deepcopy(full_trainset)
    val_subset.dataset.transform = transform_eval

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_eval
    )

    # Calibration set: first 512 images from the training split (clean eval transforms)
    calib_indices = list(range(512))
    calib_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform_eval
    )
    calib_subset = torch.utils.data.Subset(calib_set, calib_indices)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,  **loader_kwargs
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,   batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    testloader = torch.utils.data.DataLoader(
        testset,      batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    calib_loader = torch.utils.data.DataLoader(
        calib_subset, batch_size=32,         shuffle=False, **loader_kwargs
    )

    return trainloader, valloader, testloader, calib_loader


# =============================================================================
# 3. Evaluation helper
# =============================================================================
def evaluate(model: nn.Module, dataloader, target_device=None) -> float:
    """Return top-1 accuracy (%) on *dataloader*."""
    if target_device is None:
        target_device = next(model.parameters(), torch.tensor(0)).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(target_device), labels.to(target_device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


# =============================================================================
# 4. Phase A — Baseline FP32 training with early stopping
# =============================================================================
def _train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def _val_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            total_loss += criterion(model(images), labels).item() * images.size(0)
    return total_loss / len(loader.dataset)


def train_baseline(
    trainloader,
    valloader,
    max_epochs: int = 35,
    patience: int = 5,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    checkpoint_path: str = "pipeline_fp32.pth",
) -> SimpleCNN:
    """
    Train a regularised SimpleCNN baseline.

    Optimiser : SGD with Nesterov momentum + L2 weight_decay
    Scheduler : CosineAnnealingLR (T_max = max_epochs)
    Stopping  : Early stopping on *validation loss* with patience epochs;
                best weights are restored before returning.
    """
    print("\n" + "=" * 60)
    print("  PHASE A — Baseline FP32 Training")
    print("=" * 60)

    model     = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,   # L2 regularisation
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss   = float("inf")
    epochs_no_improve = 0
    best_state      = None

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(model, trainloader, criterion, optimizer)
        val_loss   = _val_loss(model, valloader, criterion)
        val_acc    = evaluate(model, valloader)
        scheduler.step()

        print(
            f"  Epoch {epoch:02d}/{max_epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.5f}"
        )

        # ── Checkpoint best model ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            best_state        = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            print(f"    ✓ New best val_loss={best_val_loss:.4f} — checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"    · No improvement ({epochs_no_improve}/{patience})")

        # ── Early stopping ─────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"\n  Best FP32 model restored (val_loss={best_val_loss:.4f}).")
    return model


# =============================================================================
# 5. Phase B — Iterative magnitude pruning → 70 % sparsity
# =============================================================================
# Layers to prune (parameter name, layer object)
def _get_prunable_layers(model: SimpleCNN):
    return [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc1,   "weight"),
        (model.fc2,   "weight"),
    ]


def _compute_global_sparsity(model: SimpleCNN) -> float:
    """Fraction of exactly-zero weights across all prunable layers."""
    total = zeros = 0
    for layer, _ in _get_prunable_layers(model):
        w = layer.weight
        total += w.numel()
        zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def _fine_tune(model, trainloader, valloader, epochs: int, lr: float = 0.005):
    """Fine-tune pruned model for a few epochs using a lower learning rate."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Only optimise non-zero (active) parameters
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for ep in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, trainloader, criterion, optimizer)
        val_acc    = evaluate(model, valloader)
        scheduler.step()
        print(f"      fine-tune ep {ep}/{epochs} | loss={train_loss:.4f}  val_acc={val_acc:.2f}%")


def iterative_prune(
    model: SimpleCNN,
    trainloader,
    valloader,
    target_sparsity: float = 0.70,
) -> SimpleCNN:
    """
    Gradually prune convolutional and linear layers to *target_sparsity* using
    L1 unstructured magnitude pruning.

    Each step applies l1_unstructured with an *incremental* amount so that the
    cumulative zero-fraction reaches the step's target. After every step the
    model is fine-tuned to allow accuracy recovery before the next mask is applied.

    At the end, prune.remove() is called on every layer to make the zero masks
    permanent (the weight buffer is overwritten with the sparse tensor).
    """
    print("\n" + "=" * 60)
    print("  PHASE B — Iterative Magnitude Pruning")
    print("=" * 60)

    # ── Pruning schedule ───────────────────────────────────────────────────
    # Each tuple: (cumulative_target_sparsity, fine_tune_epochs, fine_tune_lr)
    schedule = [
        (0.20, 4, 0.010),
        (0.40, 4, 0.008),
        (0.55, 4, 0.006),
        (0.65, 4, 0.005),
        (0.70, 5, 0.004),
    ]

    for step_idx, (cum_target, ft_epochs, ft_lr) in enumerate(schedule, start=1):
        current_sparsity = _compute_global_sparsity(model)

        # Incremental amount needed to jump from current → cum_target
        # Formula: new_zeros / remaining_weights = incremental_amount
        # new_zeros  = (cum_target - current_sparsity) * total_weights
        # remaining  = (1 - current_sparsity) * total_weights
        if current_sparsity >= cum_target:
            print(f"\n  Step {step_idx}: already at {current_sparsity*100:.1f}%, skipping.")
            continue

        incremental_amount = (cum_target - current_sparsity) / (1.0 - current_sparsity)
        incremental_amount = max(0.0, min(incremental_amount, 1.0))  # clamp to [0, 1]

        print(
            f"\n  Step {step_idx}/{len(schedule)}: "
            f"current={current_sparsity*100:.1f}%  target={cum_target*100:.0f}%  "
            f"incremental_amount={incremental_amount*100:.1f}%"
        )

        for layer, param_name in _get_prunable_layers(model):
            prune.l1_unstructured(layer, name=param_name, amount=incremental_amount)

        achieved = _compute_global_sparsity(model)
        print(f"  Achieved sparsity after masking: {achieved*100:.1f}%")

        # ── Fine-tuning recovery ───────────────────────────────────────────
        print(f"  Fine-tuning for {ft_epochs} epochs (lr={ft_lr})…")
        _fine_tune(model, trainloader, valloader, epochs=ft_epochs, lr=ft_lr)

        post_acc = evaluate(model, valloader)
        print(f"  Post-fine-tune val_acc: {post_acc:.2f}%")

    # ── Make pruning permanent ─────────────────────────────────────────────
    print("\n  Making pruning permanent (prune.remove)…")
    for layer, param_name in _get_prunable_layers(model):
        try:
            prune.remove(layer, param_name)
        except ValueError:
            pass  # layer was never pruned (edge case)

    final_sparsity = _compute_global_sparsity(model)
    print(f"  Final permanent sparsity: {final_sparsity*100:.1f}%")

    torch.save(model.state_dict(), "pipeline_pruned.pth")
    print("  Pruned state_dict saved → pipeline_pruned.pth")

    return model


# =============================================================================
# 6. Phase C — Quantisation-Aware Training (QAT)
# =============================================================================
def _select_backend() -> str:
    supported = torch.backends.quantized.supported_engines
    for engine in ("onednn", "fbgemm", "x86", "qnnpack"):
        if engine in supported:
            return engine
    return "fbgemm"   # safe fallback


def quantise_qat(
    model: SimpleCNN,
    trainloader,
    valloader,
    save_path: str = "pipeline_int8.pth",
    epochs: int = 5,
    lr: float = 1e-3,
):
    """
    Apply Eager-Mode Quantisation-Aware Training (QAT) to recover accuracy
    after heavy pruning.

    QAT Fixes & Recovery Strategy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    • Force Dropout OFF (`model_qat.dropout.eval()`) during QAT training:
      Dropout randomly zeroes 50% of activations, which violently skews fake-quant
      observer min/max stats. Evaluation mode keeps activations stable for accurate
      quantization scale observer calibration.
    • Unfreeze all layers (`conv1` / `bn1` trainable): Allows all parameters to adapt
      to discrete INT8 grid rounding rather than forcing rigid continuous weights into
      quantized buckets.
    • Zero-mask enforcement after every step — preserves 70 % sparsity.
    • 5 epochs with lr=1e-3 — sufficient time and learning rate for weights to shift.
    """
    print("\n" + "=" * 60)
    print("  PHASE C — Quantisation-Aware Training (QAT, Eager Mode)")
    print("=" * 60)

    # 1. Deep-copy, fuse, then move to training device
    #    fuse() must happen on CPU while in eval() so fuse_modules() can
    #    inspect the float conv/BN/ReLU intrinsics before fake-quant insertion.
    model_qat = copy.deepcopy(model).cpu().eval()
    model_qat.fuse()
    model_qat.train().to(device)
    print("  Conv–BN–ReLU blocks fused.")

    # 2. Snapshot zero-weight masks BEFORE prepare_qat
    #    Eager Mode preserves module attribute names exactly, so zero_masks
    #    keys will match model_qat.named_parameters() throughout training.
    zero_masks: dict[str, torch.Tensor] = {
        name: (param.data == 0.0)
        for name, param in model_qat.named_parameters()
        if "weight" in name and param.requires_grad
    }
    total_masked = sum(m.sum().item() for m in zero_masks.values())
    total_params = sum(m.numel() for m in zero_masks.values())
    print(
        f"  Sparsity masks captured: {len(zero_masks)} tensors, "
        f"{total_masked:,.0f}/{total_params:,.0f} zeros "
        f"({100*total_masked/total_params:.1f}% sparse)."
    )

    # 3. Assign QAT qconfig and prepare (inserts fake-quant observers)
    backend = _select_backend()
    torch.backends.quantized.engine = backend
    model_qat.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # --- NEW MIXED PRECISION FIX ---
    # Disable quantization for the sensitive edges
    model_qat.conv1.qconfig = None
    model_qat.fc2.qconfig = None
    # -------------------------------

    torch.quantization.prepare_qat(model_qat, inplace=True)
    print(f"  Backend: {backend}  |  prepare_qat done.")

    # 4. Optimiser — all parameters trainable
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model_qat.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── QAT Fine-Tuning Loop ───────────────────────────────────────────────
    print(f"  Running QAT for {epochs} epochs (lr={lr})…")
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model_qat.train()

        # CRITICAL FIX: Force dropout off so observers calibrate correctly!
        model_qat.dropout.eval()

        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model_qat(images), labels)
            loss.backward()
            optimizer.step()

            # Enforce 70% sparsity — force pruned zeros back after every step
            with torch.no_grad():
                for name, param in model_qat.named_parameters():
                    if name in zero_masks:
                        param.masked_fill_(zero_masks[name].to(param.device), 0.0)

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        val_acc    = evaluate(model_qat, valloader, target_device=device)
        print(f"    QAT Epoch {epoch}/{epochs} | loss={train_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model_qat.state_dict())
            print(f"      ✓ New best val_acc={best_val_acc:.2f}% — checkpoint saved.")

    # Restore best weights
    if best_state is not None:
        model_qat.load_state_dict(best_state)
    print(f"  Best QAT val_acc: {best_val_acc:.2f}%")

    # 5. Convert fake-quant → real INT8, JIT-trace, and save
    print("  Converting QAT model to static INT8…")
    model_qat.eval().cpu()
    quantised = torch.quantization.convert(model_qat, inplace=False)
    quantised.eval()

    dummy_input  = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(quantised, dummy_input)
    traced_model.save(save_path)
    print(f"  INT8 JIT model saved → {save_path}")

    return quantised


# =============================================================================
# 7. Inference helper for quantised model (always CPU)
# =============================================================================
def evaluate_quantised(model, dataloader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            # Quantised models run on CPU
            outputs = model(images.cpu())
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels.cpu()).sum().item()
    return 100.0 * correct / total


# =============================================================================
# 8. Main — wire everything together
# =============================================================================
def main():
    print(f"\n{'='*60}")
    print("  FRUGAL AI COMPRESSION PIPELINE")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ── Loaders ───────────────────────────────────────────────────────────
    print("Preparing data loaders…")
    trainloader, valloader, testloader, calib_loader = build_loaders(
        data_root="./data", batch_size=128, num_workers=2
    )
    print(
        f"  Train: {len(trainloader.dataset):,} images  |  "
        f"Val: {len(valloader.dataset):,} images  |  "
        f"Test: {len(testloader.dataset):,} images\n"
    )

    # ── Start CodeCarbon (wraps ALL three phases) ──────────────────────────
    tracker = EmissionsTracker(
        project_name="frugal_pipeline",
        output_dir=".",
        log_level="warning",      # suppress verbose progress bars
    )
    tracker.start()

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A — Baseline FP32 Training
    # ══════════════════════════════════════════════════════════════════════
    baseline_model = train_baseline(
        trainloader=trainloader,
        valloader=valloader,
        max_epochs=35,
        patience=5,
        lr=0.05,
        weight_decay=1e-4,
        checkpoint_path="pipeline_fp32.pth",
    )
    fp32_acc = evaluate(baseline_model, testloader)
    fp32_size_kb = os.path.getsize("pipeline_fp32.pth") / 1024
    print(f"\n  ► FP32 Baseline test accuracy : {fp32_acc:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B — Iterative Magnitude Pruning → 70 % sparsity
    # ══════════════════════════════════════════════════════════════════════
    pruned_model = iterative_prune(
        model=baseline_model,
        trainloader=trainloader,
        valloader=valloader,
        target_sparsity=0.70,
    )
    pruned_acc = evaluate(pruned_model, testloader)
    pruned_size_kb = os.path.getsize("pipeline_pruned.pth") / 1024
    print(f"\n  ► Pruned model test accuracy  : {pruned_acc:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE C — Quantisation-Aware Training (QAT) → INT8
    # ══════════════════════════════════════════════════════════════════════
    int8_model = quantise_qat(
        model=pruned_model,
        trainloader=trainloader,
        valloader=valloader,
        save_path="pipeline_int8.pth",
        epochs=5,      # Increased to 5 for grid adaptation
        lr=1e-3,       # Increased to 1e-3 so weights can shift
    )
    int8_acc = evaluate_quantised(int8_model, testloader)
    int8_size_kb = os.path.getsize("pipeline_int8.pth") / 1024
    print(f"\n  ► INT8 QAT test accuracy: {int8_acc:.2f}%")

    # ── Stop CodeCarbon ───────────────────────────────────────────────────
    emissions_kg = tracker.stop()                  # returns kg CO₂e
    emissions_g  = emissions_kg * 1000            # convert to grams

    # ══════════════════════════════════════════════════════════════════════
    # FINAL AUDIT
    # ══════════════════════════════════════════════════════════════════════
    compression_ratio = fp32_size_kb / int8_size_kb if int8_size_kb > 0 else float("inf")

    print("\n")
    print("╔" + "═" * 54 + "╗")
    print("║       FRUGAL AI PIPELINE — FINAL AUDIT               ║")
    print("╠" + "═" * 54 + "╣")
    print(f"║  Model Accuracy                                      ║")
    print(f"║  {'FP32 Baseline':30s}  {fp32_acc:>7.2f} %       ║")
    print(f"║  {'Pruned (70 % sparse)':30s}  {pruned_acc:>7.2f} %       ║")
    print(f"║  {'INT8 Quantised':30s}  {int8_acc:>7.2f} %       ║")
    print("╠" + "═" * 54 + "╣")
    print(f"║  File Sizes                                          ║")
    print(f"║  {'Unoptimised (FP32 baseline)':30s}  {fp32_size_kb:>7.2f} KB      ║")
    print(f"║  {'Final compressed (INT8)':30s}  {int8_size_kb:>7.2f} KB      ║")
    print(f"║  {'Compression ratio':30s}  {compression_ratio:>7.2f} ×       ║")
    print("╠" + "═" * 54 + "╣")
    print(f"║  Carbon Footprint (CodeCarbon)                       ║")
    print(f"║  {'Total operational CO₂e':30s}  {emissions_g:>7.4f} g       ║")
    print("╚" + "═" * 54 + "╝")
    print()

    # ── Accuracy deltas for quick sanity check ────────────────────────────
    print("  Accuracy deltas")
    print(f"  FP32 → Pruned   : {pruned_acc - fp32_acc:+.2f} %")
    print(f"  Pruned → INT8   : {int8_acc - pruned_acc:+.2f} %")
    print(f"  FP32 → INT8     : {int8_acc - fp32_acc:+.2f} %  (end-to-end loss)")
    print()


# =============================================================================
if __name__ == "__main__":
    main()
