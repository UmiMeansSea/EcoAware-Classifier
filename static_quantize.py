import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torchvision
import torchvision.transforms as transforms
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
import os

print("Loading Recycled Model...")
model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 10)

# Load your trained weights
model.load_state_dict(torch.load("recycled_model.pth", weights_only=True))
model.eval()

print("Configuring Static Quantization...")

# Hardware-aware backend selection
supported = torch.backends.quantized.supported_engines
if 'onednn' in supported:
    backend_engine = 'onednn'
elif 'fbgemm' in supported:
    backend_engine = 'fbgemm'
elif 'x86' in supported:
    backend_engine = 'x86'
elif 'qnnpack' in supported:
    backend_engine = 'qnnpack'
else:
    backend_engine = 'none'

if backend_engine != 'none':
    torch.backends.quantized.engine = backend_engine
    qconfig_mapping = get_default_qconfig_mapping(backend_engine)
else:
    qconfig_mapping = get_default_qconfig_mapping('fbgemm')


# Tracing shape updated to 224x224
example_input = torch.randn(1, 3, 224, 224)
prepared_model = prepare_fx(model, qconfig_mapping, example_input)

# The Calibration Phase
print("Calibrating observers with real data (Crucial for INT8 Accuracy)...")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

calib_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_set, batch_size=32, shuffle=True)

prepared_model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(calib_loader):
        prepared_model(images)
        if i >= 5: # 5 batches for calibration
            break
            
print("Calibration complete!")

print("Converting model to INT8...")
quantized_model = convert_fx(prepared_model)

torch.save(quantized_model.state_dict(), "static_quantized_model.pth")
print(f"Quantized Model saved! Size: {os.path.getsize('static_quantized_model.pth') / 1024:.2f} KB")