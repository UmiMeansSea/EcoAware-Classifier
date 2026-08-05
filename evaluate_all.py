import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
import os
import time
import json
from frugal_net import FrugalNet

# --- 1. Model Definitions ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data Loaders (All using native 32x32 images)
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading dataset...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def evaluate(model, dataloader):
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    elapsed_ms = (time.time() - start_time) * 1000
    accuracy = 100 * correct / total
    latency_per_img = elapsed_ms / total
    return accuracy, latency_per_img

print("--- BENCHMARKING ALL 3 MODELS ---")

# --- Model 1: Baseline FP32 CNN ---
print("\n1. Evaluating Baseline CNN (FP32)...")
m1 = SimpleCNN()
m1.load_state_dict(torch.load("heavy_model.pth", weights_only=True))
m1.eval()
acc1, lat1 = evaluate(m1, testloader)
size1_kb = os.path.getsize("heavy_model.pth") / 1024

# --- Model 2: Dynamic INT8 CNN ---
print("2. Evaluating Dynamic INT8 CNN...")
m2 = torch.quantization.quantize_dynamic(m1, {nn.Linear}, dtype=torch.qint8)
acc2, lat2 = evaluate(m2, testloader)
size2_kb = os.path.getsize("quantized_model.pth") / 1024

# --- Model 3: Custom FrugalNet Static INT8 ---
print("3. Evaluating Custom FrugalNet Static INT8...")
m3_base = FrugalNet()
m3_base.eval()

# Select quantization backend
supported = torch.backends.quantized.supported_engines
if 'onednn' in supported:
    engine = 'onednn'
elif 'fbgemm' in supported:
    engine = 'fbgemm'
elif 'x86' in supported:
    engine = 'x86'
elif 'qnnpack' in supported:
    engine = 'qnnpack'
else:
    engine = 'none'

if engine != 'none':
    torch.backends.quantized.engine = engine
    qconfig_mapping = get_default_qconfig_mapping(engine)
else:
    qconfig_mapping = get_default_qconfig_mapping('fbgemm')

example_input = torch.randn(1, 3, 32, 32)
prepared_m3 = prepare_fx(m3_base, qconfig_mapping, example_input)
m3 = convert_fx(prepared_m3)
m3.load_state_dict(torch.load("static_quantized_model.pth", weights_only=True))
m3.eval()

acc3, lat3 = evaluate(m3, testloader)
size3_kb = os.path.getsize("static_quantized_model.pth") / 1024

# Save Results
benchmark_data = {
    "Model": [
        "1. Baseline CNN (FP32)",
        "2. Dynamic INT8 CNN",
        "3. Pruned + Static INT8 (FrugalNet)"
    ],
    "Accuracy (%)": [round(acc1, 2), round(acc2, 2), round(acc3, 2)],
    "Size (KB)": [round(size1_kb, 2), round(size2_kb, 2), round(size3_kb, 2)],
    "Latency (ms/img)": [round(lat1, 3), round(lat2, 3), round(lat3, 3)]
}

with open("all_models_comparison.json", "w") as f:
    json.dump(benchmark_data, f, indent=4)

print("\nBenchmark completed and saved to 'all_models_comparison.json'!")