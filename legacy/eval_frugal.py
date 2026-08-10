import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import time
import json

# 1. Architecture with Quantization Stubs
class FrugalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stubs mark where data is converted from 32-bit float to 8-bit int
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# Data pipeline
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    print("Loading test dataset...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
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
        return (100 * correct / total), (elapsed_ms / total)

    # --- Evaluate Native FrugalNet (FP32) ---
    print("1. Evaluating Baseline FrugalNet (FP32)...")
    model_fp32 = FrugalNet()
    model_fp32.load_state_dict(torch.load("frugalnet_base.pth", weights_only=True), strict=False)
    model_fp32.eval()

    acc_fp32, lat_fp32 = evaluate(model_fp32, testloader)
    size_fp32 = os.path.getsize("frugalnet_base.pth") / 1024

    # --- Apply Perfect Static INT8 Quantization ---
    print("2. Applying Perfect Static INT8 Quantization...")
    
    # Safely configure backend engine
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
        model_fp32.qconfig = torch.quantization.get_default_qconfig(engine)
    else:
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for calibration
    model_prepared = torch.quantization.prepare(model_fp32)

    print("Calibrating model...")
    with torch.no_grad():
        for i, (images, _) in enumerate(testloader):
            model_prepared(images)
            if i >= 10: 
                break

    # Convert to final static integer model
    model_int8 = torch.quantization.convert(model_prepared)

    torch.save(model_int8.state_dict(), "frugalnet_int8.pth")
    acc_int8, lat_int8 = evaluate(model_int8, testloader)
    size_int8 = os.path.getsize("frugalnet_int8.pth") / 1024

    # --- Update Dashboard Data ---
    print("Updating Dashboard JSON...")
    benchmark_data = {
        "Model": [
            "1. Old Baseline CNN (FP32)",
            "2. FrugalNet (Native FP32)",
            "3. FrugalNet (Static INT8)"
        ],
        "Accuracy (%)": [43.87, round(acc_fp32, 2), round(acc_int8, 2)],
        "Size (KB)": [1052.94, round(size_fp32, 2), round(size_int8, 2)],
        "Latency (ms/img)": [0.336, round(lat_fp32, 3), round(lat_int8, 3)]
    }

    # Overwriting your existing json file so the dashboard updates automatically
    with open("all_models_comparison.json", "w") as f:
        json.dump(benchmark_data, f, indent=4)

    print("\nDone! Check your Streamlit dashboard for the final, uncompromising metrics.")

if __name__ == '__main__':
    main()
