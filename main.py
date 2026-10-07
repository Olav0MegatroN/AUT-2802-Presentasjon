import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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

'''  Decide if the calculations are perfomed by GPU or CPU  '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''.to(device): moves tensors and models to the specified device (either GPU or CPU).
This ensures that all computations happen on the same device, avoiding errors or inefficiencies.'''
model = model.to(device)
# Cross Entropy Loss for multi-class classification
criterion = nn.CrossEntropyLoss()
''' L2 Regularization adds a penalty proportional to the square of the magnitude
 of the weights to the loss function. Discourages the model from assigning
  too much importance to any individual feature by keeping the weights small.
 The larger the weight, the higher the penalty.'''
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)  # L2 regularization via weight_decay

class EarlyStopping:
    """
    Initializes the early stopping mechanism.

    Args:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.

    """
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None  # Tracks the best loss (lower is better)
        self.early_stop = False  # Flag for early stopping
        self.counter = 0  # Counter for epochs without improvement
        self.best_loss = np.Inf  # Best loss starts as infinity

    ''' special method that allows us to use the object as if it were a function. '''
    def __call__(self, val_loss, model):
        """
        Training should be stopped early based on validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        score = -val_loss  # Loss is used as the score (lower is better, so we negate it)

        # If this is the first epoch or loss improved significantly
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            # No significant improvement (i..e, loss hasn't decreased enough)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement in loss
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset the counter if loss improves

    def save_checkpoint(self, val_loss, model):
        """
        Save the model when validation loss improves.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        if val_loss < self.best_loss:  # We save if loss decreased
            self.best_loss = val_loss
            checkpoint_name = "checkpoint.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f'Model saved as {checkpoint_name}')

early_stopping = EarlyStopping(patience=5, delta=0.0001)


# Initialize lists to store training and validation metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# number of epochs
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        ''' torch.max(outputs, 1): The torch.max() function returns the maximum
        value of all elements in the tensor along a specified dimension.
         (_) is used for variables that we don't need to use   '''
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    # append the list
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    # append the list
    train_accuracies.append(train_accuracy)

    # .eval indicates that the model is used for prediction.
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
      print("Early stopping")
      break

# Plot training and validation loss
plt.figure(figsize=(8,4))
plt.plot(range(1, len(val_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(8,4))
plt.plot(range(1, len(val_losses)+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_losses)+1), val_accuracies, label='Validation Accuracy')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()