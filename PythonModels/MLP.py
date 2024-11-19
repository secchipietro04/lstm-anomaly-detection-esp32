import torch
import torch.nn as nn
import torch.optim as optim
import json

# Define the MLP model with a configurable number of layers
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Define the layers dynamically based on the list of layer sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU for all layers except the last one
                x = torch.relu(x)
        return x

# Hyperparameters
layer_sizes = [64, 128, 256, 1]  # Number of layers and sizes: 32 -> 64 -> 128 -> 1
learning_rate = 0.01
num_epochs = 1000
targetLoss=0.05
# Create the model
model = MLP(layer_sizes)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate synthetic data
def generate_data(batch_size):
    x = torch.randn(batch_size, layer_sizes[0])  # Random input with size of first layer
    y = x.sum(dim=1, keepdim=True)  # Target is the sum of the input values
    return x, y

# Training loop
epoch=0
loss=targetLoss
while loss>=targetLoss:
    x_train, y_train = generate_data(batch_size=64)
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}")
    epoch+=1

# Test the model
x_test, y_test = generate_data(batch_size=10)
y_pred = model(x_test)

# Print test results
print("\nTest Results:")
for i in range(10):
    print(f"True Sum: {y_test[i].item():.4f}, Predicted Sum: {y_pred[i].item():.4f}\n")

# Export model to JSON
def export_model_to_json(model, filename="mlp_model.json"):
    model_data = {
        "layers": [],
        "weights": [],
        "biases": []
    }

    # Collect the layer sizes and weights
    for i, layer in enumerate(model.layers):
        layer_data = {
            "input_size": layer.in_features,
            "output_size": layer.out_features,
            "weights": layer.weight.tolist(),
            "biases": layer.bias.tolist()
        }
        model_data["layers"].append(layer_data)
        model_data["biases"].append(layer.bias.tolist())

    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(model_data, f, indent=4)

# Export model to file
export_model_to_json(model)

