# Import required libraries
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Create and train logistic regression model - Using parameters more suitable for linear data
logistic_model = LogisticRegression(
    solver='liblinear',         # Solver suitable for small datasets
    multi_class='ovr',          # One-vs-rest strategy
    C=20000,                     # Increase C value, reduce regularization to fit linearly separable data
    random_state=42,
    max_iter=1000               # Increase maximum iterations to ensure convergence
)
logistic_model.fit(X_train, y_train)

# Predict test data
y_pred = logistic_model.predict(X_test)

# Evaluate model performance
# Calculate accuracy
train_accuracy = accuracy_score(y_train, logistic_model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print classification report
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Get feature importance (logistic regression coefficients)
# For multi-class problems, coef_ has shape (number of classes, number of features)
feature_importance = np.abs(logistic_model.coef_).mean(axis=0)  # Calculate average importance of each feature across all classes

# Plot charts
plt.figure(figsize=(12, 6))

# Plot figure
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Purples)
plt.title("LR (NIR-MRI(GLCM))", fontsize=18)
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

# Plot feature importance chart (in a separate window)
plt.figure(figsize=(6, 6.5))
# Get top 20 most important features, sorted by importance in descending order ([::-1] achieves reverse sorting)
top_features = np.argsort(feature_importance)[-20:][::-1]
top_importance = feature_importance[top_features]
top_feature_names = [feature_names[i] for i in top_features]

# Vertical bar chart
plt.bar(range(len(top_importance)), top_importance, color='skyblue')
plt.xticks(range(len(top_feature_names)), top_feature_names, rotation=45, ha='right', fontsize=12)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Feature Importance', fontsize=16)
plt.title('Top 20 Feature Importances (NIR-MRI(GLCM))', fontsize=16)

plt.tight_layout()
plt.show()