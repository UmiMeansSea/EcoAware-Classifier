import torch
import torch.nn as nn
import torch.nn.functional as F

class FrugalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: Conv -> BN -> ReLU -> Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Conv -> BN -> ReLU -> Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Conv -> BN -> ReLU -> Pool
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected layers
        # Input size to first linear layer: 64 channels * 4 height * 4 width = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Image shrunken: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Image shrunken: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Image shrunken: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x
