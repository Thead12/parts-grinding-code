import torch
import torch.nn as nn
import numpy as np

# Define the neural network model (same as before)
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

# Function to extract features from a buffer
def extract_features(buffer):
    buffer_size = len(buffer)
    amplitude_features = buffer
    amplitude_fft = np.abs(np.fft.fft(buffer)[:buffer_size//2])
    frequency_features = amplitude_fft
    features = np.concatenate((amplitude_features, frequency_features))
    return features

# Class to handle model loading and prediction
class RealTimeClassifier:
    def __init__(self, model_path, buffer_size):
        self.buffer_size = buffer_size
        input_size = buffer_size + buffer_size // 2  # Adjust input size based on buffer size
        self.model = BinaryClassifier(input_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, buffer):
        # Preprocess the buffer to extract features
        features = extract_features(buffer)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Use the model for inference
        with torch.no_grad():
            output = self.model(features_tensor)
            prediction = (output > 0.5).float()
        
        return prediction.item()  # 1 indicates part is being ground, 0 indicates not being ground

# Example usage
if __name__ == "__main__":
    # Initialize the classifier
    classifier = RealTimeClassifier('binary_classifier.pth', buffer_size=1000)

    # Example buffer input (replace with actual data)
    example_buffer = np.random.randn(1000)  # Replace with actual buffer data

    # Make a prediction
    prediction = classifier.predict(example_buffer)
    print(f'Prediction: {prediction}')  # 1 indicates part is being ground, 0 indicates not being ground