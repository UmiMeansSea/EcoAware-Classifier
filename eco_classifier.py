import torch
import torchvision
import torchvision.transforms as transforms

# 1. How we want to change our images
transform = transforms.Compose([
    transforms.ToTensor(), # Converts the image into a grid of numbers (a Tensor)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Squishes the numbers to make math easier
])

print("Downloading and preparing the training data...")

# 2. Download the Training Data (The textbook the model learns from)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 3. Create a DataLoader to feed data in 'batches'
# We feed 64 images at a time instead of all 50,000 at once so we don't crash our computer's memory!
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

print("Data is ready! We have", len(trainset), "training images.")

import torch.nn as nn
import torch.nn.functional as F

# 4. Building the Baseline Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # DEPARTMENT 1: Feature Extractors
        # Takes 3 channels (Red, Green, Blue) and outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Takes those 16 maps and finds even more complex patterns (32 maps)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # The "Shrinker" - cuts the image size in half to save memory
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # DEPARTMENT 2: Decision Makers
        # 32 channels * 8 height * 8 width (shrunk down from original 32x32 image)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10) # 10 because CIFAR-10 has exactly 10 categories

    def forward(self, x):
        # Step A: Slide the first magnifying glass, activate (ReLU), and shrink
        x = self.pool(F.relu(self.conv1(x)))
        # Step B: Slide the second magnifying glass, activate, and shrink again
        x = self.pool(F.relu(self.conv2(x)))
        
        # Step C: Flatten the 2D grids into a single 1D list of numbers
        x = torch.flatten(x, 1) 
        
        # Step D: Make the final decision
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN()
print("Model built! It currently relies on heavy, 32-bit floating-point math.")

import torch.optim as optim
from codecarbon import EmissionsTracker

# 5. Prepare the Teacher (Loss function) and the Optimizer (Gradient Descent)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6. The Carbon Tracker - Start the meter!
tracker = EmissionsTracker(project_name="eco_classifier_training")
tracker.start()

print("Starting to train the heavy model. Tracking emissions...")

# 7. Training Loop (Just 2 passes through the dataset to keep it fast)
for epoch in range(2): 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Clear the memory of the last step
        optimizer.zero_grad()

        # Forward pass (guess), Backward pass (learn), Optimize (adjust weights)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Stop the meter!
emissions = tracker.stop()

print("Training finished!")
print(f"Total Carbon Emissions: {emissions * 1000:.5f} grams of CO2.")

# 8. Save the heavy FP32 model to our hard drive
torch.save(model.state_dict(), "heavy_model.pth")
print("Saved the heavy 32-bit model to 'heavy_model.pth'")