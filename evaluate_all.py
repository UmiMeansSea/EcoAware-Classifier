import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import os
import time
import json

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

# Data Loader for Simple CNN
transform_simple = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data Loader for MobileNetV2 (Resized to 224)
transform_mobilenet = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset_simple = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_simple)
testloader_simple = torch.utils.data.DataLoader(testset_simple, batch_size=64, shuffle=False)

testset_mobile = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_mobilenet)
testloader_mobile = torch.utils.data.DataLoader(testset_mobile, batch_size=64, shuffle=False)

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
acc1, lat1 = evaluate(m1, testloader_simple)
size1_kb = os.path.getsize("heavy_model.pth") / 1024

# --- Model 2: Dynamic INT8 CNN ---
print("2. Evaluating Dynamic INT8 CNN...")
m2 = torch.quantization.quantize_dynamic(m1, {nn.Linear}, dtype=torch.qint8)
acc2, lat2 = evaluate(m2, testloader_simple)
size2_kb = os.path.getsize("quantized_model.pth") / 1024

# --- Model 3: MobileNetV2 Pruned + Static INT8 ---
print("3. Evaluating Pruned + Static INT8 MobileNetV2...")
m3_base = mobilenet_v2()
m3_base.classifier[1] = nn.Linear(m3_base.last_channel, 10)
m3_base.eval()

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


# Tracing example input updated to 224x224
example_input = torch.randn(1, 3, 224, 224) 
prepared_m3 = prepare_fx(m3_base, qconfig_mapping, example_input)
m3 = convert_fx(prepared_m3)

m3.load_state_dict(torch.load("static_quantized_model.pth", weights_only=True))
m3.eval()

acc3, lat3 = evaluate(m3, testloader_mobile)
size3_kb = os.path.getsize("static_quantized_model.pth") / 1024

# Save Results
benchmark_data = {
    "Model": [
        "1. Baseline CNN (FP32)",
        "2. Dynamic INT8 CNN",
        "3. Pruned + Static INT8 (MobileNet)"
    ],
    "Accuracy (%)": [round(acc1, 2), round(acc2, 2), round(acc3, 2)],
    "Size (KB)": [round(size1_kb, 2), round(size2_kb, 2), round(size3_kb, 2)],
    "Latency (ms/img)": [round(lat1, 3), round(lat2, 3), round(lat3, 3)]
}

with open("all_models_comparison.json", "w") as f:
    json.dump(benchmark_data, f, indent=4)

print("\nBenchmark completed and saved to 'all_models_comparison.json'!")