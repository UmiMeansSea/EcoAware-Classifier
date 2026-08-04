import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn.utils.prune as prune
from codecarbon import EmissionsTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The Data (Upscaling CIFAR-10 to MobileNet's expected 224x224 size)
transform = transforms.Compose([
    transforms.Resize(224), # FIX: Upscale the tiny images!
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def main():
    print(f"Using device: {device}")
    print("Downloading CIFAR-10 data (Upscaled to 224x224)...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Optional: Subset the data to make training faster on CPU
    subset_indices = torch.randperm(len(trainset))[:5000] 
    train_subset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)

    print("Building the Frugal Recycled Model...")

    # 2. TRANSFER LEARNING
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, 10)

    # 3. PRUNING
    prune.l1_unstructured(model.classifier[1], name='weight', amount=0.2)
    prune.remove(model.classifier[1], 'weight')

    model = model.to(device)
    print(f"Model ready! Frozen heavy layers, pruned 20% of classifier, moved to {device}.")

    # 4. Frugal Training Loop
    criterion = nn.CrossEntropyLoss()
    # FIX: Swap to Adam Optimizer for faster convergence
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    tracker = EmissionsTracker(project_name="advanced_eco_classifier")
    tracker.start()

    print("Starting training (1 Epoch only)...")
    for epoch in range(1): 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"  -> Training batch {i}...")

    emissions = tracker.stop()
    print(f"\nTraining finished! Total Carbon Emissions: {emissions * 1000:.5f} grams of CO2.")

    torch.save(model.state_dict(), "recycled_model.pth")
    print("Saved to 'recycled_model.pth'")

if __name__ == '__main__':
    main()