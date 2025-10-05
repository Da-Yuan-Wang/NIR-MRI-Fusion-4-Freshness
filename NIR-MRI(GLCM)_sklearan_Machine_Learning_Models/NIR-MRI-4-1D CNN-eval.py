# Import necessary libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import openpyxl
from sklearn.preprocessing import StandardScaler

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Read training data
train_data = pd.read_csv(r'data\train_data_Raw_GLCM.csv')
X_train = train_data.iloc[:, 1:-1]  # Exclude SampleID and Label columns, keep only feature data
y_train = train_data.iloc[:, -1]     # Label column as target variable

# Read test data
test_data = pd.read_csv(r'data\test_data_Raw_GLCM.csv')
X_test = test_data.iloc[:, 1:-1]     # Exclude SampleID and Label columns, keep only feature data
y_test = test_data.iloc[:, -1]       # Label column as target variable

# Feature names (excluding SampleID column)
feature_names = list(train_data.columns[1:-1])

# Class names

target_names = ['Class-S-Freshness', 'Class-A-Freshness', 'Class-B-Freshness', 'Class-C-Freshness']


print(f"→ Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data to fit 1D-CNN (samples, features)
# For spectral data, each sample is a one-dimensional sequence containing multiple wavelength points
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Move data to specified device (CUDA or CPU)
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Optimized 1D-CNN model, designed for NIR spectral data characteristics
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

# Load pre-trained model
print("Loading pre-trained model...")
checkpoint = torch.load('NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/best_1D_CNN_NIR-MRI-GLCM_model_optimized.pth', map_location=device)
num_features = checkpoint['input_size']
num_classes = checkpoint['num_classes']
seed = checkpoint['seed']

print(f"Model information:")
print(f"  Random seed: {seed}")
print(f"  Input size: {num_features}")
print(f"  Number of classes: {num_classes}")
print(f"  Test set accuracy: {checkpoint['test_accuracy']:.4f}")  # Modified display information val_accuracy -> test_accuracy
print(f"  Training loss: {checkpoint['train_loss']:.4f}")
print(f"  Test set loss: {checkpoint['test_loss']:.4f}")  # Modified display information val_loss -> test_loss
print(f"  Training epochs: {checkpoint['epoch']}")

# Create model instance and move to specified device
model = OptimizedCNN1D(num_features, num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model loaded successfully!")

# Make predictions
model.eval()
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Ensure data is on the correct device
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())  # Move results back to CPU for processing
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Evaluate model performance
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=target_names)

# Plot charts
plt.figure(figsize=(12, 6))

# Plot confusion matrix
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
plt.title("1D-CNN (NIR-MRI(GLCM))", fontsize=18)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=16)
tick_marks = np.arange(len(target_names))
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

plt.xticks(tick_marks, target_names_multiline, rotation=45, fontsize=16)
plt.yticks(tick_marks, target_names_multiline, fontsize=16)
plt.xlabel("Predicted Freshness Grades", fontsize=16)
plt.ylabel("True Freshness Grades", fontsize=16)

# Add values in matrix
thresh = conf_matrix.max() / 2.
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                fontsize=16,
                color="white" if conf_matrix[i, j] > thresh else "black")

# Calculate F1-score and plot bar chart
plt.subplot(122)
# Calculate F1-score for each class
f1_scores = []
for i in range(len(target_names)):
    tp = conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

x = range(len(target_names))
plt.bar(x, f1_scores, color='purple')
plt.xticks(x, target_names_multiline, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Class', fontsize=16)
plt.ylabel('F1-Score', fontsize=16)
plt.title('F1-Score by Class', fontsize=18)
plt.ylim(0, 1)

# Adjust subplot spacing
plt.tight_layout()

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

plt.show()