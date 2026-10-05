import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_of_classes, p): # 'p' probability of an element being zeroed out
        # Initialize the parent class (nn.Module)
        super(LeNet5, self).__init__()

        # Layer 1: Convolutional + BatchNorm2D  +  Pooling
        ''' we are using three channels as input (instead of 1) for some experiments that will be done later '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        # Layer 2: Convolutional + BatchNorm2D + Pooling
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # 14x14 -> 10x10
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
        # Layer 3: Fully Connected + BatchNorm1D
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.bn3 = nn.BatchNorm1d(120)
        # Layer 4: Fully Connected + BatchNorm1D
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.bn4 = nn.BatchNorm1d(84)
        # Layer 5: Output Layer
        self.fc3 = nn.Linear(in_features=84, out_features=num_of_classes)
        # Dropout layers
        self.dropout = nn.Dropout(p)  # dropout layer defined in __init__

    def forward(self, x):
        # Layer 1: Conv2 -> BatchNorm -> Pooling -> Activation
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))

        # Layer 2: Conv2 -> BatchNorm -> Pooling -> Activation
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))

        # Flatten the output from the convolutional layers
        x = x.view(-1, 16 * 5 * 5)
        # Layer 3: Fully Connected + BatchNorm + Activation
        x = F.relu(self.bn3(self.fc1(x)))
        # Apply dropout after first FC layer
        x = self.dropout(x)
        # Layer 4: Fully Connected +  BatchNorm + Activation
        x = F.relu(self.bn4(self.fc2(x)))
        # Apply dropout after second FC layer
        x = self.dropout(x)
        # Layer 5: Fully Connected (Output)
        x = self.fc3(x)
        return x
