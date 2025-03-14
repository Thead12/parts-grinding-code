import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd

def assign_class_fft(amplitude_fft, min_freq, max_freq, threshold):
    
    mean_freq = np.mean(amplitude_fft[min_freq:max_freq])
    if mean_freq > threshold:
        return 1
    else:
        return 0    

def extract_features(buffer, buffer_size):
    amplitude_features = buffer
    amplitude_fft = np.abs(np.fft.fft(buffer)[:buffer_size//2])
    frequency_features = amplitude_fft

    labels = assign_class_fft(amplitude_fft, 400, 500, 0.5)

    features = np.concatenate((amplitude_features, frequency_features))
    return features, labels

# Step 1: Prepare your dataset
# Load your data
df = pd.read_csv('vibration_data_20250313_104518.csv')
data = df.to_numpy()
time = data[:, 0]
amplitude = data[:, 1]

# Convert into buffers
buffer_size = 1000
buffers = [amplitude[i:i+buffer_size] for i in range(0, len(amplitude) - buffer_size, buffer_size)]

# Extract features and labels from each buffer
features_labels = [extract_features(buffer, buffer_size) for buffer in buffers]
features, labels = zip(*features_labels)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Convert to PyTorch tensors
X_tensor = torch.tensor(features, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Create a dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Define the neural network model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = features.shape[1]
model = BinaryClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Step 4: Evaluate the model on the test set
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader)
accuracy = correct / total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Save the model parameters
torch.save(model.state_dict(), 'binary_classifier.pth')