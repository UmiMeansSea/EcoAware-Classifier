import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from codecarbon import EmissionsTracker

# Set device (Use GPU / CUDA if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Building the Baseline Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # DEPARTMENT 1: Feature Extractors
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # DEPARTMENT 2: Decision Makers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Transform definitions
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading and preparing the training data...")

    # 2. Download Training Data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # 3. Create DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    print("Data is ready! We have", len(trainset), "training images.")

    # Initialize model and move to GPU/CPU device
    model = SimpleCNN().to(device)
    print(f"Model built and moved to {device}.")

    # 5. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 6. Carbon Tracker
    tracker = EmissionsTracker(project_name="eco_classifier_training")
    tracker.start()

    print(f"Starting to train on {device}. Tracking emissions...")

    # 7. Training Loop
    for epoch in range(2): 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    emissions = tracker.stop()

    print("Training finished!")
    print(f"Total Carbon Emissions: {emissions * 1000:.5f} grams of CO2.")

    # 8. Save Model
    torch.save(model.state_dict(), "heavy_model.pth")
    print("Saved the model to 'heavy_model.pth'")

if __name__ == '__main__':
    main()