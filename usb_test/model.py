import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN2D(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN2D, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        # Adjust the size of the linear layer to match the output of the last pooling layer
        self.fc1 = nn.Linear(32 * 2 * 2, 96)  # Example size, adjust based on your input and architecture
        self.fc2 = nn.Linear(96, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # Flattening the output for the fully connected layers
        x = x.view(-1, 32 * 2 * 2)  # Adjust the size here to match the output of the last pooling layer

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
# model = SimpleCNN2D(4)
# if __name__ == "__main__":
#     input = torch.randn(1,10,9)
#     output = model(input)
#     print(output.shape)