import torch
import torch.nn as nn
import torch.optim as optim
from dataloder import dataloader
# Define the GRU model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_dim = 9
hidden_dim = 256
output_dim = 1  # Adjust according to your needs
num_layers = 2
learning_rate = 0.001
num_epochs = 50

# Initialize the model
model = GRUNet(input_dim, hidden_dim, output_dim, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy dataset (batch_size, sequence_length, input_dim)
# Replace this with your real dataset
# train_data = torch.randn(32, 10, input_dim)  # example data
# train_labels = torch.randn(32, output_dim)   # example labels
p1 = 'label0.csv'
p2 = 'label1.csv'
p3 = 'label2.csv'
p4 = 'label3.csv'
train_data,train_labels = dataloader(p1,p2,p3,p4)
# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'gru_model.pth')