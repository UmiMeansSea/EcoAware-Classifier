import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# 1. Load the Recycled Model (FP32)
print("Loading Recycled Model...")
model = mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("recycled_model.pth", weights_only=True))
model.eval() # MUST be in eval mode for quantization

# 2. Prepare Calibration Data
# We only need a tiny subset (e.g., 500 images) to calibrate the integer ranges
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

subset_indices = torch.randperm(len(trainset))[:500]
calib_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
calib_loader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=calib_sampler)

def main():
    # 3. FX Graph Mode Static Quantization Setup
    print("Configuring Static Quantization...")
    
    # Dynamically select the correct engine supported by PyTorch build
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


    # 1. FIX: Update tracing shape to 224
    example_input = torch.randn(1, 3, 224, 224)
    prepared_model = prepare_fx(model, qconfig_mapping, example_input)

    # 2. FIX: Calibration Phase
    print("Calibrating observers with real data (Crucial for INT8 Accuracy)...")
    prepared_model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            prepared_model(images)
            if i >= 5: # 5 batches are plenty for the observers to calculate min/max ranges
                break
    print("Calibration complete!")

    # 3. Convert to INT8
    print("Converting model...")
    quantized_model = convert_fx(prepared_model)


    # 6. Save and Compare Sizes
    torch.save(quantized_model.state_dict(), "static_quantized_model.pth")

    fp32_size = os.path.getsize("recycled_model.pth") / (1024 * 1024)
    int8_size = os.path.getsize("static_quantized_model.pth") / (1024 * 1024)

    print("\n--- STATIC QUANTIZATION COMPLETE ---")
    print(f"Original FP32 Size: {fp32_size:.2f} MB")
    print(f"Static INT8 Size:   {int8_size:.2f} MB")
    print(f"Compression:        {fp32_size/int8_size:.1f}x smaller")
    print("The model math is now strictly integers. Perfect for edge devices!")

if __name__ == '__main__':
    main()