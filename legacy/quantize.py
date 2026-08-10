import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import json

# 1. Define Architecture
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

# 2. Prepare Test Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading CIFAR-10 Test Dataset...")
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Evaluation helper function
def evaluate(model):
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    elapsed_ms = (time.time() - start_time) * 1000
    accuracy = 100 * correct / total
    latency_per_img = elapsed_ms / total # Average ms per image
    return accuracy, latency_per_img

# 3. Load & Evaluate Heavy Model (FP32)
heavy_model = SimpleCNN()
heavy_model.load_state_dict(torch.load("heavy_model.pth", weights_only=True))
heavy_model.eval()

print("Evaluating Heavy Model (FP32) on test set...")
heavy_acc, heavy_lat = evaluate(heavy_model)

# 4. Quantize & Evaluate Frugal Model (INT8)
print("Quantizing model...")
quantized_model = torch.quantization.quantize_dynamic(heavy_model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized_model.state_dict(), "quantized_model.pth")

print("Evaluating Quantized Model (INT8) on test set...")
tiny_acc, tiny_lat = evaluate(quantized_model)

# 5. Measure File Sizes & Export Comparison
heavy_size = os.path.getsize("heavy_model.pth") / 1024
tiny_size = os.path.getsize("quantized_model.pth") / 1024

comparison_data = {
    "Model": ["Heavy Model (FP32)", "Frugal Model (INT8)"],
    "Accuracy (%)": [round(heavy_acc, 2), round(tiny_acc, 2)],
    "Size (KB)": [round(heavy_size, 2), round(tiny_size, 2)],
    "Latency (ms/img)": [round(heavy_lat, 3), round(tiny_lat, 3)]
}

with open("model_comparison.json", "w") as f:
    json.dump(comparison_data, f, indent=4)

print("\n--- EVALUATION COMPLETE ---")
print(f"Accuracy Loss: {heavy_acc - tiny_acc:.2f}%")
print("Saved comparison metrics to 'model_comparison.json'")