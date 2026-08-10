# -*- coding: utf-8 -*-
"""
evaluate_all.py
===============
Benchmark all models produced by the frugal_ai_project.

  Legacy models (pre-pipeline)
  ─────────────────────────────
  1. Baseline SimpleCNN — FP32          (heavy_model.pth)
  2. Dynamic INT8 SimpleCNN             (quantized_model.pth)
  3. FrugalNet Dynamic INT8             (recycled_model.pth / static_quantized_model.pth)

  New pipeline models (frugal_pipeline.py)
  ─────────────────────────────────────────
  4. Pipeline FP32 Baseline             (pipeline_fp32.pth)
  5. Pipeline Pruned 70% FP32           (pipeline_pruned.pth)
  6. Pipeline Pruned + Static INT8      (rebuilt from pipeline_pruned.pth)

Results are saved to all_models_comparison.json for the Streamlit dashboard.

NOTE on Model 6 quantisation:
  PyTorch 2.13 has a bug where deserialising a saved convert_fx GraphModule
  produces ConvReLU2d objects whose _modules attribute is missing, crashing
  eval() / named_modules().  We therefore REBUILD the INT8 model from the
  pruned state_dict every time, using the same prepare_fx→calibrate→convert_fx
  path that ran successfully inside frugal_pipeline.py.  File size is still
  read from pipeline_int8.pth (the artifact written by the pipeline).
"""

import os
import sys
import time
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# Ensure legacy models can be imported from legacy/ subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legacy"))
from frugal_net import FrugalNet  # legacy model used by models 1–3

# =============================================================================
# Model definitions
# =============================================================================

class SimpleCNN(nn.Module):
    """Original baseline — no BatchNorm, no Dropout."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PipelineCNN(nn.Module):
    """
    Must exactly match frugal_pipeline.py's SimpleCNN
    (BatchNorm + Dropout, fc1=256 units).
    """
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3,  16, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(16)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(32)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(32 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# =============================================================================
# Data loaders
# =============================================================================
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

print("Loading CIFAR-10 test set…")
testset    = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_eval
)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Calibration subset (512 images) — needed for static INT8 quantisation (model 6)
calib_set    = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_eval
)
calib_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(calib_set, list(range(512))),
    batch_size=32, shuffle=False,
)
print(f"Test set: {len(testset):,} images  |  Calibration: 512 images\n")


# =============================================================================
# Helpers
# =============================================================================
def evaluate(model: nn.Module, dataloader, use_cpu: bool = False):
    """Return (top-1 accuracy %, latency ms/image)."""
    model.eval()
    correct = total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            if use_cpu:
                images, labels = images.cpu(), labels.cpu()
            outputs  = model(images)
            _, preds = outputs.max(1)
            total   += labels.size(0)
            correct += preds.eq(labels).sum().item()
    elapsed_ms = (time.time() - start) * 1000
    return 100.0 * correct / total, elapsed_ms / total


def _select_backend() -> str:
    for engine in ("onednn", "fbgemm", "x86", "qnnpack"):
        if engine in torch.backends.quantized.supported_engines:
            return engine
    return "fbgemm"


def _resolve_path(path: str) -> str:
    """Check root directory first, then legacy/ subfolder."""
    if os.path.exists(path):
        return path
    legacy_path = os.path.join("legacy", path)
    if os.path.exists(legacy_path):
        return legacy_path
    return path


def _file_kb(path: str) -> float:
    resolved = _resolve_path(path)
    return os.path.getsize(resolved) / 1024 if os.path.exists(resolved) else 0.0


def _sep(label: str):
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    sys.stdout.flush()


# =============================================================================
# Benchmarks
# =============================================================================
rows = []

print("=" * 60)
print("  BENCHMARKING ALL MODELS")
print("=" * 60)

# -----------------------------------------------------------------------------
# MODEL 1 - Legacy Baseline SimpleCNN (FP32)
# -----------------------------------------------------------------------------
heavy_path = _resolve_path("heavy_model.pth")
if os.path.exists(heavy_path):
    _sep("[1/6] Legacy Baseline SimpleCNN - FP32")
    m1 = SimpleCNN()
    m1.load_state_dict(torch.load(heavy_path, weights_only=True))
    m1.eval()
    acc1, lat1 = evaluate(m1, testloader)
    rows.append({
        "Model"            : "1. Legacy Baseline (FP32)",
        "Accuracy (%)"     : round(acc1, 2),
        "Size (KB)"        : round(_file_kb("heavy_model.pth"), 2),
        "Latency (ms/img)" : round(lat1, 3),
        "Stage"            : "Legacy",
    })
    print(f"  acc={acc1:.2f}%  size={_file_kb('heavy_model.pth'):.1f} KB  lat={lat1:.3f} ms/img")
else:
    print("\n[1/6] heavy_model.pth not found - skipping.")

# -----------------------------------------------------------------------------
# MODEL 2 - Legacy Dynamic INT8 SimpleCNN
# -----------------------------------------------------------------------------
quant_path = _resolve_path("quantized_model.pth")
if os.path.exists(heavy_path) and os.path.exists(quant_path):
    _sep("[2/6] Legacy Dynamic INT8 SimpleCNN")
    m2_fp = SimpleCNN()
    m2_fp.load_state_dict(torch.load(heavy_path, weights_only=True))
    m2 = torch.quantization.quantize_dynamic(m2_fp, {nn.Linear}, dtype=torch.qint8)
    m2.eval()
    acc2, lat2 = evaluate(m2, testloader, use_cpu=True)
    rows.append({
        "Model"            : "2. Legacy Dynamic INT8",
        "Accuracy (%)"     : round(acc2, 2),
        "Size (KB)"        : round(_file_kb("quantized_model.pth"), 2),
        "Latency (ms/img)" : round(lat2, 3),
        "Stage"            : "Legacy",
    })
    print(f"  acc={acc2:.2f}%  size={_file_kb('quantized_model.pth'):.1f} KB  lat={lat2:.3f} ms/img")
else:
    print("\n[2/6] heavy_model.pth or quantized_model.pth not found - skipping.")

# -----------------------------------------------------------------------------
# MODEL 3 - Legacy FrugalNet Dynamic INT8
# -----------------------------------------------------------------------------
_sep("[3/6] Legacy FrugalNet - Dynamic INT8")

# Prefer the trained weights; fall back to random init if file missing
frugalnet_weights = next(
    (p for p in (_resolve_path("recycled_model.pth"), _resolve_path("frugalnet_base.pth")) if os.path.exists(p)),
    None,
)
m3 = FrugalNet()
if frugalnet_weights:
    m3.load_state_dict(torch.load(frugalnet_weights, weights_only=True))
    print(f"  Loaded FrugalNet weights from {frugalnet_weights}")
else:
    print("  WARNING: No FrugalNet weights found - using random init.")
m3.eval()

# Dynamic INT8 on both Conv and Linear layers
m3_int8 = torch.quantization.quantize_dynamic(
    m3, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
m3_int8.eval()
acc3, lat3 = evaluate(m3_int8, testloader, use_cpu=True)

# Size: prefer the saved static INT8 file; fall back to recycled weights file
size3_kb = _file_kb("static_quantized_model.pth") or _file_kb(frugalnet_weights or "")
rows.append({
    "Model"            : "3. Legacy FrugalNet Dyn-INT8",
    "Accuracy (%)"     : round(acc3, 2),
    "Size (KB)"        : round(size3_kb, 2),
    "Latency (ms/img)" : round(lat3, 3),
    "Stage"            : "Legacy",
})
print(f"  acc={acc3:.2f}%  size~{size3_kb:.1f} KB  lat={lat3:.3f} ms/img")

# -----------------------------------------------------------------------------
# MODEL 4 - Pipeline FP32 Baseline
# -----------------------------------------------------------------------------
if os.path.exists("pipeline_fp32.pth"):
    _sep("[4/6] Pipeline FP32 Baseline (Dropout + L2 + Early Stopping)")
    m4 = PipelineCNN()
    m4.load_state_dict(torch.load("pipeline_fp32.pth", weights_only=True))
    m4.eval()
    acc4, lat4 = evaluate(m4, testloader)
    rows.append({
        "Model"            : "4. Pipeline FP32 Baseline",
        "Accuracy (%)"     : round(acc4, 2),
        "Size (KB)"        : round(_file_kb("pipeline_fp32.pth"), 2),
        "Latency (ms/img)" : round(lat4, 3),
        "Stage"            : "Pipeline",
    })
    print(f"  acc={acc4:.2f}%  size={_file_kb('pipeline_fp32.pth'):.1f} KB  lat={lat4:.3f} ms/img")
else:
    print("\n[4/6] pipeline_fp32.pth not found - run frugal_pipeline.py first.")

# -----------------------------------------------------------------------------
# MODEL 5 - Pipeline Pruned 70% FP32
# -----------------------------------------------------------------------------
if os.path.exists("pipeline_pruned.pth"):
    _sep("[5/6] Pipeline Pruned FP32 (70% L1 sparsity, permanent)")
    m5 = PipelineCNN()
    m5.load_state_dict(torch.load("pipeline_pruned.pth", weights_only=True))
    m5.eval()
    acc5, lat5 = evaluate(m5, testloader)
    rows.append({
        "Model"            : "5. Pipeline Pruned 70% (FP32)",
        "Accuracy (%)"     : round(acc5, 2),
        "Size (KB)"        : round(_file_kb("pipeline_pruned.pth"), 2),
        "Latency (ms/img)" : round(lat5, 3),
        "Stage"            : "Pipeline",
    })
    print(f"  acc={acc5:.2f}%  size={_file_kb('pipeline_pruned.pth'):.1f} KB  lat={lat5:.3f} ms/img")
else:
    print("\n[5/6] pipeline_pruned.pth not found - run frugal_pipeline.py first.")

# -----------------------------------------------------------------------------
# MODEL 6 - Pipeline Pruned + INT8 QAT (Eager Mode, JIT TorchScript)
#
# pipeline_int8.pth is saved via torch.jit.trace inside frugal_pipeline.py.
# Load with torch.jit.load — no class definitions, no FX graph rebuilding,
# no ConvReLU2d _modules deserialization bug.
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists("pipeline_int8.pth"):
    _sep("[6/6] Pipeline Pruned 70% + Static INT8 (QAT Eager Mode)")

    m6 = torch.jit.load("pipeline_int8.pth", map_location="cpu")
    m6.eval()

    acc6, lat6 = evaluate(m6, testloader, use_cpu=True)
    size6_kb = _file_kb("pipeline_int8.pth")
    rows.append({
        "Model"            : "6. Pipeline Pruned + Static INT8",
        "Accuracy (%)"     : round(acc6, 2),
        "Size (KB)"        : round(size6_kb, 2),
        "Latency (ms/img)" : round(lat6, 3),
        "Stage"            : "Pipeline",
    })
    print(f"  acc={acc6:.2f}%  size~{size6_kb:.1f} KB  lat={lat6:.3f} ms/img")
else:
    print("\n[6/6] pipeline_int8.pth not found - run frugal_pipeline.py first.")


# =============================================================================
# Save results & print summary
# =============================================================================
if not rows:
    print("\nNo models evaluated — nothing saved.")
    sys.exit(0)

benchmark_data = {
    "Model"            : [r["Model"]             for r in rows],
    "Accuracy (%)"     : [r["Accuracy (%)"]      for r in rows],
    "Size (KB)"        : [r["Size (KB)"]          for r in rows],
    "Latency (ms/img)" : [r["Latency (ms/img)"]  for r in rows],
    "Stage"            : [r["Stage"]              for r in rows],
}

with open("all_models_comparison.json", "w") as f:
    json.dump(benchmark_data, f, indent=4)

print("\n" + "=" * 60)
print("  FINAL BENCHMARK SUMMARY")
print("=" * 60)
print(f"  {'Model':<36} {'Acc%':>6}  {'KB':>8}  {'ms/img':>8}")
print("  " + "-" * 56)
for r in rows:
    tag = "[P]" if r["Stage"] == "Pipeline" else "[L]"
    print(
        f"  {tag} {r['Model']:<34} {r['Accuracy (%)']:>6.2f}"
        f"  {r['Size (KB)']:>8.1f}  {r['Latency (ms/img)']:>8.3f}"
    )
print("=" * 60)
print("\n  [L] Legacy   [P] Pipeline")
print(f"\n[OK] Results saved -> all_models_comparison.json")
print("     Launch dashboard: streamlit run dashboard.py\n")