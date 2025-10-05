# Import necessary libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import openpyxl
from sklearn.preprocessing import StandardScaler
import random
import os

# Set random seed to ensure experiment reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed
seed = 42
set_seed(seed)
print(f"Used random seed: {seed}")

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print PyTorch and CUDA version information
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")


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
target_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']

print(f"→ Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"→ Number of features: {X_train.shape[1]}, Number of classes: {len(np.unique(y_train))}")

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduce batch size
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

# Create model instance and move to specified device
num_features = X_train.shape[1]
num_classes = 4  # Four-class classification task
model = OptimizedCNN1D(num_features, num_classes).to(device)

# Print model parameter information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)  # Use AdamW optimizer with larger weight decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)  # Cosine annealing learning rate scheduling

# Train model
num_epochs = 600 # Adjust number of training epochs
train_losses = []
test_accuracies = []
test_losses = []
train_accuracies = []

best_test_accuracy = 0.0
best_epoch = 0

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Test phase
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Save best model
    if accuracy > best_test_accuracy:
        best_test_accuracy = accuracy
        best_epoch = epoch
        # Save best model, including input size information
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'input_size': num_features,  # Add input size information
            'num_classes': num_classes,   # Add number of classes information
            'seed': seed                  # Add random seed information
        }, 'NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/best_1D_CNN_NIR-MRI-GLCM_model_optimized.pth')
    
    # Print training information for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}')

print(f"Best test accuracy: {best_test_accuracy:.4f} at epoch {best_epoch+1}")

# Load best model
checkpoint = torch.load('NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/best_1D_CNN_NIR-MRI-GLCM_model_optimized.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# Print model information
print(f"Loaded best model information:")
print(f"  Epoch: {checkpoint['epoch']+1}")
print(f"  Training loss: {checkpoint['train_loss']:.4f}")
print(f"  Test loss: {checkpoint['test_loss']:.4f}")
print(f"  Test accuracy: {checkpoint['test_accuracy']:.4f}")

# Make predictions
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Ensure data is on the correct device
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())  # Move results back to CPU for processing
        all_labels.extend(labels.cpu().numpy())

# Evaluate model performance
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=target_names)

# Save training history data to Excel
history_df = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Test Loss': test_losses,
    'Test Accuracy': test_accuracies
})

# Save to current directory
history_df.to_excel('NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/training_NIR-MRI-1D_CNN_history_optimized.xlsx', index=False)
print("Training history saved to NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/training_NIR-MRI-1D_CNN_history_optimized.xlsx")

# Plot charts
plt.figure(figsize=(12, 6))

# Plot training history
plt.subplot(121)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Create new figure for confusion matrix and F1-score
plt.figure(figsize=(12, 6))

# Plot confusion matrix
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
plt.title("1D-CNN(NIR-MRI(GLCM))", fontsize=14)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=12)
tick_marks = np.arange(len(target_names))
target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
plt.xticks(tick_marks, target_names_multiline, rotation=45, fontsize=12)
plt.yticks(tick_marks, target_names_multiline, fontsize=12)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)

# Adjust subplot spacing to prevent label overlap
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Add values in confusion matrix
thresh = conf_matrix.max() / 2.
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                fontsize=12,
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
plt.xticks(x, target_names_multiline, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score by Class', fontsize=14)
plt.tight_layout()

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

plt.tight_layout()
plt.show()