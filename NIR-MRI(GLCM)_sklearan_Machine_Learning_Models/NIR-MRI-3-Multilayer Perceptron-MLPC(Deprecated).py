# Import required libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

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
print(f"→ Number of features: {X_train.shape[1]}, Number of classes: {len(np.unique(y_train))}")

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train optimized MLP neural network model

# Use a smaller sample for grid search to save time
mlp_model = MLPClassifier(hidden_layer_sizes=(4,),
                          activation='relu',
                          solver='lbfgs',
                          alpha=0.0001,
                          max_iter=10000, 
                          random_state=42)

mlp_model.fit(X_train_scaled, y_train)

# Predict test data
y_pred = mlp_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, mlp_model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print classification report
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Save the trained MLP model
model_data = {
    'input_size': X_train.shape[1],
    'coefs_': mlp_model.coefs_,
    'intercepts_': mlp_model.intercepts_,
    'classes_': mlp_model.classes_,
    'scaler': scaler
}

with open('NIR-MRI(GLCM)_sklearan_Machine_Learning_Models\mlpc_model-NIR-MRI-GLCM.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("MLP model has been saved as NIR-MRI(GLCM)_sklearan_Machine_Learning_Models\mlpc_model-NIR-MRI-GLCM.pkl file")


# Plot charts
plt.figure(figsize=(12, 6))

# Plot figure
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
plt.title("Confusion Matrix of Multilayer Perceptron", fontsize=14)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=12)
tick_marks = np.arange(len(target_names))
# Modify label display to two lines
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S\nFreshness', 'Class-A\nFreshness', 'Class-B\nFreshness', 'Class-C\nFreshness']

plt.xticks(tick_marks, target_names_multiline, rotation=45, fontsize=12)
plt.yticks(tick_marks, target_names_multiline, fontsize=12)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)

# Add text in matrix
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
# Use multi-line labels
plt.xticks(x, target_names_multiline, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('F1-Score by Class', fontsize=14)

print(f"Train_Accuracy: {train_accuracy:.4f}")
print(f"Test_Accuracy: {test_accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

plt.tight_layout()
plt.show()