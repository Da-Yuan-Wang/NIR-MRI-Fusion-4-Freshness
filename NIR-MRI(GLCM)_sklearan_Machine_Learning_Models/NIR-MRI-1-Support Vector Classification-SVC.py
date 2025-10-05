# Import required libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Add joblib library for saving models

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

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train SVM model
#svm_model = SVC(kernel='linear', C=300, probability=True) 
#svm_model = SVC(kernel='linear', C=600, probability=True)  0.9036 #Increase C value, weak regularization, similar to MLPC
svm_model = SVC(kernel='linear', C=1000, probability=True)  

svm_model.fit(X_train, y_train)

# Save the trained SVC model
joblib.dump(svm_model, 'NIR-MRI(GLCM)_sklearan_Machine_Learning_Models\svc_model-NIR-MRI-GLCM.pkl')
print("SVC model saved as 'NIR-MRI(GLCM)_sklearan_Machine_Learning_Models\svc_model-NIR-MRI-GLCM.pkl'")

# Predict test data
y_pred = svm_model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, svm_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print classification report
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Plot charts
plt.figure(figsize=(12, 6))

# Plot figure
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
plt.title("SVC (NIR-MRI(GLCM))", fontsize=18)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=16)
tick_marks = np.arange(len(target_names))
# Modify label display to two lines
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

plt.xticks(tick_marks, target_names_multiline, rotation=45, fontsize=16)
plt.yticks(tick_marks, target_names_multiline, fontsize=16)
plt.xlabel("Predicted Freshness Grades", fontsize=16)
plt.ylabel("True Freshness Grades", fontsize=16)

# Add text in matrix
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
# Use multi-line labels
plt.xticks(x, target_names_multiline, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Class', fontsize=16)
plt.ylabel('F1-Score', fontsize=16)
plt.title('F1-Score by Class', fontsize=18)

print(f"Train_Accuracy: {train_accuracy:.4f}")
print(f"Test_Accuracy: {test_accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

plt.tight_layout()
plt.show()