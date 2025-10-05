import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Define the same 1D-CNN model structure as during training
class OptimizedCNN1D(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(OptimizedCNN1D, self).__init__()
        
        # First convolutional block - Extract local spectral features
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, padding=4)
        self.bn1_1 = nn.BatchNorm1d(32)
        self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Second convolutional block - Extract mid-level spectral features
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third convolutional block - Extract high-level spectral features
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Fourth convolutional block - Extract semantic-level spectral features
        self.conv4_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm1d(256)
        self.conv4_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(0.4)
        
        # Calculate flattened size
        self.flattened_size = (input_size // 16) * 256  # After four pooling operations, size becomes 1/16 of original
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout7 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third convolutional block
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth convolutional block
        x = torch.relu(self.bn4_1(self.conv4_1(x)))
        x = torch.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout6(x)
        
        x = torch.relu(self.fc3(x))
        x = self.dropout7(x)
        
        x = self.fc4(x)
        return x

# CNN wrapper class for using PyTorch models in sklearn
class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path='best_1D_CNN_model_optimized.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2, 3])  # Add classes_ attribute required by scikit-learn
        self._load_model()
    
    def _load_model(self):
        # Load checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # Compatible with older PyTorch versions without weights_only parameter
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model parameters
        input_size = checkpoint.get('input_size', 2048)  # Default value 2048, adjust according to actual situation
        num_classes = checkpoint.get('num_classes', 4)
        
        # Create model instance
        self.model = OptimizedCNN1D(input_size, num_classes).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, X):
        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Convert to tensor and add channel dimension
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Return prediction results
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Convert to tensor and add channel dimension
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Return probabilities
        return probabilities.cpu().numpy()
    
    def fit(self, X, y):
        # This is a pre-trained model that does not require training
        # But to be compatible with sklearn interface, this method needs to be implemented
        return self