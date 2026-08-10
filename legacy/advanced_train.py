import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from codecarbon import EmissionsTracker
from frugal_net import FrugalNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Native 32x32 Data Pipeline with Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    print(f"Using device: {device}")
    print("Preparing native 32x32 training data with augmentation...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Train on 15,000 images for better convergence of custom network on CPU/GPU
    subset_indices = torch.randperm(len(trainset))[:15000]
    train_subset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)

    print("Building Custom FrugalNet Model...")
    model = FrugalNet().to(device)

    # Apply 20% Pruning on FC1 layers
    prune.l1_unstructured(model.fc1, name='weight', amount=0.2)
    prune.remove(model.fc1, 'weight')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    tracker = EmissionsTracker(project_name="frugal_net_training")
    tracker.start()

    print("Starting FrugalNet training (5 Epochs)...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                print(f"  [Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 50:.3f}")
                running_loss = 0.0

    emissions = tracker.stop()
    print(f"\nTraining finished! Total Carbon Emissions: {emissions * 1000:.5f} grams of CO2.")

    torch.save(model.state_dict(), "recycled_model.pth")
    print("Saved trained weights to 'recycled_model.pth'")

if __name__ == '__main__':
    main()