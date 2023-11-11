import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN2D

import torch.nn.functional as F
from dataloder import dataset


# Define the model
num_classes = 6  # Example number of output classes
model = SimpleCNN2D(num_classes)

# Example input (batch size, channels, height, width)
# Here, batch size = 1, channels = 1, height = 9, width = 10
example_input = torch.randn(1, 1, 9, 10)


# Forward pass
output = model(example_input)
# print("Output shape:", output.shape)

# n_samples = 5120
# Assuming you have training data in 'train_x' and 'train_y'
# train_x: input data, shape: (n_samples, 1, 9, 10)
# train_y: labels, shape: (n_samples,)
#Example: 
# train_x = torch.randn(n_samples, 1, 9, 10)
# train_y = torch.randint(0, 4, (n_samples,))

train_x, train_y = dataset('label0.csv','left.csv','right.csv','jump.csv','label4.csv','still.csv')
# Create dataset and dataloader for batch processing
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = SimpleCNN2D(num_classes=6)
# model.load_state_dict(torch.load('cnn_model_3.pth'))
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100  # Number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # print(outputs,labels)
        labels = labels.float()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'cnn_model_6.pth')