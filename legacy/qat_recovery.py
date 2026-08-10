# -*- coding: utf-8 -*-
"""
qat_recovery.py
===============
Quantization-Aware Training (QAT) recovery script for the Frugal AI pipeline.

PROBLEM SOLVED
--------------
The pipeline's 70%-pruned model suffered an 11.74 % accuracy drop when
converted to static INT8 via Post-Training Quantization (PTQ).  PTQ is
"calibration-only" — the pruned weights were never trained to tolerate fake-
quantization noise, causing severe quantization shock.

QAT SOLUTION
------------
We fine-tune the pruned model with *fake quantization* nodes active during
the forward pass.  The weights gradually adapt to INT8 grid precision.
After training we call torch.quantization.convert() to make the quantization
real (identical to PTQ convert, but the weights are now INT8-ready).

ENERGY-EFFICIENCY MEASURES (Frugal ML / Sobriété Numérique)
------------------------------------------------------------
  • Layer freezing  — conv1+BN block frozen from the start; the early
                       texture filters don't need to change.
  • Max 15 epochs   — QAT converges much faster than full training.
  • Early stopping  — patience = 2 epochs on val accuracy.
  • Warmup epoch    — fake quant disabled for epoch 1; observers collect
                       range statistics before introducing noise.
  • Low LR (5e-5)   — avoids catastrophic forgetting of the pruned topology.
  • CodeCarbon       — tracks the CO₂e of this recovery run.

MIXED-PRECISION FALLBACK
-------------------------
If the model still fails to converge to ~75 %, see the section labelled
"MIXED-PRECISION FALLBACK" below.  You can keep the first and last layers
in FP32 by setting their qconfig to None before prepare_qat().

OUTPUTS
-------
  qat_int8_model.pth   — final INT8 model (torch.save of full module)
  qat_fp32_backup.pth  — FP32 QAT-tuned weights (state_dict) as a safety copy
"""

import os
import sys
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker

# =============================================================================
# 0. Hyper-parameters — tune here without touching the rest of the script
# =============================================================================
CFG = dict(
    pruned_checkpoint = "pipeline_pruned.pth",  # source: 70% sparse FP32 weights
    qat_model_out     = "qat_int8_model.pth",   # final INT8 artifact
    fp32_backup_out   = "qat_fp32_backup.pth",  # FP32 QAT weights (safety copy)

    data_root         = "./data",
    batch_size        = 128,
    num_workers       = 2,

    max_epochs        = 15,       # hard cap; early stopping will fire first
    warmup_epochs     = 1,        # fake-quant OFF during these first epochs
    lr                = 5e-5,     # low LR — avoids forgetting pruned topology
    weight_decay      = 1e-4,     # L2 regularisation
    es_patience       = 2,        # early stopping: plateau threshold in epochs
    freeze_layers     = True,     # freeze conv1+BN1 block (early features)
    label_smoothing   = 0.05,

    # CIFAR-10 channel statistics
    mean = (0.4914, 0.4822, 0.4465),
    std  = (0.2023, 0.1994, 0.2010),
)


# =============================================================================
# 1. QAT-compatible model
#
# CRITICAL DESIGN NOTE
# --------------------
# Eager-mode QAT (QuantStub / prepare_qat / convert) requires *nn.Module*
# ReLU instances (not torch.functional.relu) so that fuse_modules() can
# collapse Conv → BN → ReLU into a single ConvBnReLU2d intrinsic module.
# The *weight key names* (conv1, bn1, conv2, bn2, fc1, fc2) are kept
# identical to frugal_pipeline.py's SimpleCNN so that pipeline_pruned.pth
# loads with a clean strict=False match.
# =============================================================================
class QATNet(nn.Module):
    """
    Drop-in QAT wrapper around the PipelineCNN topology.

    Structural changes vs frugal_pipeline.py SimpleCNN
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    • QuantStub  inserted before first conv  (marks FP32 → INT8 boundary)
    • DeQuantStub inserted after fc2         (marks INT8 → FP32 boundary)
    • F.relu replaced by nn.ReLU modules    (required for fuse_modules)
    • Dropout kept — some regularisation during QAT fine-tuning is useful
    """

    def __init__(self):
        super().__init__()

        self.quant = torch.quantization.QuantStub()

        # ── Block 1: early feature extractor (will be frozen) ──────────────
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)   # named for fuse_modules()
        self.pool  = nn.MaxPool2d(2, 2)

        # ── Block 2: deeper feature extractor (trainable) ──────────────────
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)

        # ── Classifier head (trainable) ────────────────────────────────────
        self.fc1     = nn.Linear(32 * 8 * 8, 256)
        self.relu3   = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse(self) -> None:
        """
        Fuse Conv → BN → ReLU into ConvBnReLU2d intrinsic modules.
        Must be called *before* prepare_qat().
        """
        torch.quantization.fuse_modules(
            self, [["conv1", "bn1", "relu1"],
                   ["conv2", "bn2", "relu2"]],
            inplace=True,
        )


# =============================================================================
# 2. Data loaders
# =============================================================================
def build_loaders(cfg: dict):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=True, download=True, transform=transform_train
    )
    # 80/20 train-val split (same seed as frugal_pipeline.py for reproducibility)
    gen        = torch.Generator().manual_seed(42)
    n_train    = int(0.80 * len(full_train))
    n_val      = len(full_train) - n_train
    train_sub, val_sub = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=gen
    )
    # Val uses eval transforms
    val_sub.dataset           = copy.deepcopy(full_train)
    val_sub.dataset.transform = transform_eval

    testset = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=False, download=True, transform=transform_eval
    )

    kw = dict(num_workers=cfg["num_workers"], pin_memory=False)
    return (
        torch.utils.data.DataLoader(train_sub, batch_size=cfg["batch_size"], shuffle=True,  **kw),
        torch.utils.data.DataLoader(val_sub,   batch_size=cfg["batch_size"], shuffle=False, **kw),
        torch.utils.data.DataLoader(testset,   batch_size=cfg["batch_size"], shuffle=False, **kw),
    )


# =============================================================================
# 3. Evaluation helper (CPU-only — quantised models run on CPU)
# =============================================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    """Return top-1 accuracy (%) on *loader*."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.cpu(), labels.cpu()
        preds  = model(images).argmax(dim=1)
        total  += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return 100.0 * correct / total


# =============================================================================
# 4. Model preparation
# =============================================================================
def _select_backend() -> str:
    for engine in ("onednn", "fbgemm", "x86", "qnnpack"):
        if engine in torch.backends.quantized.supported_engines:
            return engine
    return "fbgemm"


def prepare_for_qat(cfg: dict) -> tuple:
    """
    Load pruned weights → fuse → assign QAT qconfigs → prepare_qat.

    Returns
    -------
    model    : prepared QATNet (fake-quant nodes inserted, still FP32)
    backend  : str — the selected quantisation backend
    """
    if not os.path.exists(cfg["pruned_checkpoint"]):
        raise FileNotFoundError(
            f"{cfg['pruned_checkpoint']} not found. "
            "Run frugal_pipeline.py first."
        )

    # ── Instantiate & load pruned weights ─────────────────────────────────
    model = QATNet()

    pruned_sd   = torch.load(cfg["pruned_checkpoint"], weights_only=True)
    missing, unexpected = model.load_state_dict(pruned_sd, strict=False)

    if missing:
        # Expected: quant, dequant, relu1/2/3 have no parameters — fine.
        param_missing = [k for k in missing
                         if any(s in k for s in ("weight", "bias", "running"))]
        if param_missing:
            print(f"  [WARN] Parameters not found in checkpoint: {param_missing}")
    print(f"  Pruned weights loaded from '{cfg['pruned_checkpoint']}'")

    # ── Fuse Conv → BN → ReLU before qconfig assignment ──────────────────
    model.cpu().eval()
    model.fuse()
    print("  Conv–BN–ReLU blocks fused.")

    # ── Select hardware backend ────────────────────────────────────────────
    backend = _select_backend()
    torch.backends.quantized.engine = backend
    print(f"  QAT backend: {backend}")

    # ── Assign QAT qconfig ─────────────────────────────────────────────────
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # ══════════════════════════════════════════════════════════════════════
    # MIXED-PRECISION FALLBACK
    # ══════════════════════════════════════════════════════════════════════
    # If the fully-INT8 model still fails to converge (accuracy < 72 %),
    # uncomment the lines below to keep the input quant stub and the final
    # classification layer in FP32.  This preserves the most numerically
    # sensitive operations at full precision while all hidden layers remain
    # INT8 — a common "selective quantisation" strategy.
    #
    #   model.quant.qconfig = None   # skip input quantisation (keep FP32 input)
    #   model.fc2.qconfig   = None   # skip output layer (keep FP32 logits)
    #
    # After uncommenting, re-run the script.  Expect ~0.5 KB larger artifact.
    # ══════════════════════════════════════════════════════════════════════

    # ── Insert fake-quant observers ───────────────────────────────────────
    torch.quantization.prepare_qat(model, inplace=True)
    print("  Fake-quantisation observers inserted (prepare_qat done).")

    # ── Freeze early feature extraction block ─────────────────────────────
    # conv1 (now fused into ConvBnReLU2d) captures low-level textures that
    # don't need updating during QAT recovery.  Freezing it saves ~30 % of
    # the per-step gradient computation and keeps the frugal carbon budget low.
    if cfg["freeze_layers"]:
        model.conv1.requires_grad_(False)   # freeze all params in fused block 1
        print("  Block 1 (conv1+BN1+ReLU1) frozen — gradients disabled.")

    return model, backend


# =============================================================================
# 5. QAT training loop (carbon-tracked)
# =============================================================================
def qat_train(model: nn.Module, trainloader, valloader, cfg: dict) -> nn.Module:
    """
    Fine-tune *model* with fake quantisation active.

    Strategy
    --------
    Epoch 1 (warmup)   : fake-quant DISABLED — observers accumulate activation
                          ranges without injecting noise.  The pruned weights
                          stabilise first.
    Epochs 2 → N       : fake-quant ENABLED  — weights adapt to INT8 grid.
    Early stopping      : stops if val accuracy doesn't improve for 2 epochs.

    Returns the model with the best validation accuracy weights restored.
    """
    print("\n" + "=" * 62)
    print("  QAT FINE-TUNING  (CodeCarbon tracking active)")
    print("=" * 62)

    criterion  = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # Only optimise parameters that require grad
    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.Adam(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    best_val_acc     = 0.0
    best_state       = None
    epochs_no_improve = 0

    # ── Start CodeCarbon ───────────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name="qat_recovery",
        output_dir=".",
        log_level="warning",
    )
    tracker.start()

    for epoch in range(1, cfg["max_epochs"] + 1):

        # ── Warmup: disable fake-quant for the first N epochs ─────────────
        if epoch <= cfg["warmup_epochs"]:
            model.apply(torch.quantization.disable_fake_quant)
            model.apply(torch.quantization.enable_observer)
            warmup_tag = " [WARMUP — fake-quant OFF, observers ON]"
        else:
            model.apply(torch.quantization.enable_fake_quant)
            model.apply(torch.quantization.disable_observer)   # freeze ranges
            warmup_tag = ""

        # ── Training step ─────────────────────────────────────────────────
        model.train()

        # Frozen layers must stay in eval mode even during model.train()
        if cfg["freeze_layers"]:
            model.conv1.eval()   # BN running stats frozen; no grad

        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.cpu(), labels.cpu()
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)

        # ── Validation ────────────────────────────────────────────────────
        val_acc = evaluate(model, valloader)
        scheduler.step()

        print(
            f"  Ep {epoch:02d}/{cfg['max_epochs']}  "
            f"train_loss={train_loss:.4f}  val_acc={val_acc:.2f}%"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"{warmup_tag}"
        )
        sys.stdout.flush()

        # ── Checkpoint best model ──────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_state        = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"    ✓ New best val_acc={best_val_acc:.2f}% — checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"    · No improvement ({epochs_no_improve}/{cfg['es_patience']})")

        # ── Early stopping ─────────────────────────────────────────────────
        if epochs_no_improve >= cfg["es_patience"] and epoch > cfg["warmup_epochs"]:
            print(f"\n  Early stopping fired after epoch {epoch}.")
            break

    # ── Stop CodeCarbon ────────────────────────────────────────────────────
    emissions_kg = tracker.stop()
    qat_co2_g    = emissions_kg * 1000

    print(f"\n  QAT training complete.")
    print(f"  Best val_acc = {best_val_acc:.2f}%")
    print(f"  QAT carbon footprint: {qat_co2_g:.5f} g CO₂e")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, qat_co2_g


# =============================================================================
# 6. Convert QAT model → real INT8 and save
# =============================================================================
def convert_and_save(model: nn.Module, testloader, cfg: dict):
    """
    Convert the QAT-trained model to a real INT8 quantised model,
    evaluate on the test set, and save the artifact.

    Returns (accuracy_pct, size_kb).
    """
    print("\n" + "=" * 62)
    print("  CONVERTING TO INT8")
    print("=" * 62)

    model.eval()
    model.cpu()

    # Convert fake-quant nodes to real INT8 quantised ops
    int8_model = torch.quantization.convert(model, inplace=False)
    int8_model.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("  Evaluating INT8 model on CIFAR-10 test set…")
    test_acc = evaluate(int8_model, testloader)
    print(f"  INT8 accuracy: {test_acc:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(int8_model, cfg["qat_model_out"])
    size_kb = os.path.getsize(cfg["qat_model_out"]) / 1024
    print(f"  Saved INT8 model → {cfg['qat_model_out']}  ({size_kb:.1f} KB)")

    # Safety copy: save the FP32 QAT-tuned state_dict before conversion
    # (useful to re-run convert without retraining)
    torch.save(model.state_dict(), cfg["fp32_backup_out"])
    print(f"  FP32 QAT backup → {cfg['fp32_backup_out']}")

    return test_acc, size_kb


# =============================================================================
# 7. Main
# =============================================================================
def main():
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║        FRUGAL AI — QAT RECOVERY SCRIPT                    ║")
    print("╚" + "═" * 60 + "╝\n")
    print("  Goal   : recover INT8 accuracy from 63.64 % → ~75 %")
    print("  Method : Quantization-Aware Training on 70%-pruned model")
    print("  Budget : energy-efficient (layer freeze + early stopping)\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Preparing data loaders…")
    trainloader, valloader, testloader = build_loaders(CFG)
    print(
        f"  Train: {len(trainloader.dataset):,}  "
        f"Val: {len(valloader.dataset):,}  "
        f"Test: {len(testloader.dataset):,}\n"
    )

    # ── Prepare ───────────────────────────────────────────────────────────
    print("Preparing QAT model…")
    model, backend = prepare_for_qat(CFG)

    # ── Sanity check: evaluate the loaded pruned weights before QAT ───────
    print("\n  Sanity check — pruned FP32 accuracy before QAT:")
    pre_qat_acc = evaluate(model, testloader)
    print(f"  Pre-QAT accuracy: {pre_qat_acc:.2f}%  (target after QAT: ~75 %)\n")

    # ── QAT Training ──────────────────────────────────────────────────────
    model, qat_co2_g = qat_train(model, trainloader, valloader, CFG)

    # ── Convert & Save ────────────────────────────────────────────────────
    int8_acc, int8_size_kb = convert_and_save(model, testloader, CFG)

    # ── Reference numbers from frugal_pipeline.py ─────────────────────────
    baseline_fp32_acc  = 76.28
    ptq_int8_acc       = 63.64
    pipeline_co2_g     = 11.7836   # from the pipeline run

    # ── Final Audit ───────────────────────────────────────────────────────
    recovered_pct = int8_acc - ptq_int8_acc

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║       QAT RECOVERY — FINAL AUDIT                        ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  {'FP32 Pruned Baseline':35s}  {baseline_fp32_acc:>7.2f} %    ║")
    print(f"║  {'PTQ INT8 (pipeline, before QAT)':35s}  {ptq_int8_acc:>7.2f} %    ║")
    print(f"║  {'QAT INT8 (this run)':35s}  {int8_acc:>7.2f} %    ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  {'Accuracy recovered vs PTQ':35s}  {recovered_pct:>+7.2f} %    ║")
    print(f"║  {'QAT INT8 model size':35s}  {int8_size_kb:>7.1f} KB   ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  {'Pipeline total CO₂e':35s}  {pipeline_co2_g:>7.4f} g    ║")
    print(f"║  {'QAT recovery CO₂e (this run)':35s}  {qat_co2_g:>7.4f} g    ║")
    print(f"║  {'QAT as % of pipeline budget':35s}  {100*qat_co2_g/pipeline_co2_g:>7.1f} %    ║")
    print("╚" + "═" * 58 + "╝")
    print()

    if int8_acc >= 73.0:
        print("  ✅  QAT recovery SUCCESSFUL — target accuracy achieved.")
    elif int8_acc >= 68.0:
        print("  ⚠️   Partial recovery. Consider uncommenting MIXED-PRECISION FALLBACK.")
    else:
        print("  ❌  Recovery below target. Enable mixed-precision fallback and re-run.")
        print("      See the MIXED-PRECISION FALLBACK section in prepare_for_qat().")


if __name__ == "__main__":
    main()
