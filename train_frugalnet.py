import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The Native FrugalNet Architecture
class FrugalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard convolutions are highly stable for static quantization
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Data Pipeline (Native 32x32 with Augmentation!)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    print(f"Using device: {device}")
    print("Loading native 32x32 CIFAR-10 data...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 3. Training Setup
    model = FrugalNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    tracker = EmissionsTracker(project_name="frugalnet_training")
    tracker.start()

    # We train for 10 epochs because the model is tiny and trains very fast!
    print("Starting training (10 Epochs)...")

    # 4. Training Loop
    model.train()
    for epoch in range(10):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f"--- Epoch {epoch + 1}/10 completed ---")

    emissions = tracker.stop()
    print(f"\nTraining finished! Total Carbon Emissions: {emissions * 1000:.5f} grams of CO2.")

    torch.save(model.state_dict(), "frugalnet_base.pth")
    print("Saved baseline model to 'frugalnet_base.pth'")

if __name__ == '__main__':
    main()
