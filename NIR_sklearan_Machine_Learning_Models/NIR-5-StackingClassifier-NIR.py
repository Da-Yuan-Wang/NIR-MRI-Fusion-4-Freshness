import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
# Import CNNWrapper from external library
from cnn_wrapper import CNNWrapper
# Import MLPWrapper from external library
from mlp_wrapper import MLPWrapper
# Import SVCWrapper from external library
from svc_wrapper import SVCWrapper
import os

# Read training data
train_data = pd.read_csv('data/nir_train.csv')
X_train = train_data.iloc[:, 1:-1]  # Exclude SampleID and Label columns, keep only feature data
y_train = train_data.iloc[:, -1]     # Label column as target variable

# Read test data
test_data = pd.read_csv('data/nir_test.csv')
X_test = test_data.iloc[:, 1:-1]     # Exclude SampleID and Label columns, keep only feature data
y_test = test_data.iloc[:, -1]       # Label column as target variable

# Feature names (excluding SampleID column)
feature_names = list(train_data.columns[1:-1])

# Class names

target_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

print(f"→ Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"→ Number of features: {X_train.shape[1]}, Number of classes: {len(np.unique(y_train))}") #For 2D tensors, shape[0] represents the number of rows, shape[1] represents the number of columns

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base classifiers
base_classifiers = [
    # ('lda', LinearDiscriminantAnalysis(solver='svd', 
    #                                    shrinkage=None, 
    #                                    priors=None, 
    #                                    n_components=None,  #Will be set to min(n_classes - 1, n_features)
    #                                    store_covariance=False, 
    #                                    tol=0.0001, 
    #                                    covariance_estimator=None)),
    ('lr', LogisticRegression(solver='liblinear',         # Solver suitable for small datasets
                              multi_class='ovr',          # One-vs-rest strategy
                              C=20000,                     # Increase C value, reduce regularization to fit linearly separable data
                              random_state=42,
                              max_iter=1000)),               # Increase maximum iterations to ensure convergence),  
    ('svc', SVCWrapper(model_path='NIR_sklearan_Machine_Learning_Models/svc_model.pkl')),  # Use saved SVC model
    #('mlp', MLPWrapper(model_path='NIR_sklearan_Machine_Learning_Models/mlpc_model.pkl')),  # Use saved MLP model
    ("cnn", CNNWrapper(model_path='NIR_sklearan_Machine_Learning_Models/best_NIR-1D_CNN_model_optimized.pth'))     # Use custom CNNWrapper as base classifier
]

# Train Stacking classifier (meta-classifier: SVM, enable probability prediction)
# Use SVM that supports probability prediction as meta-classifier
stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=SVC(kernel='rbf', probability=True), cv=5)

stacking_clf.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Get prediction probabilities
y_proba = stacking_clf.predict_proba(X_test)

# Evaluate model performance
train_accuracy = accuracy_score(y_train, stacking_clf.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# Plot charts
plt.figure(figsize=(12, 6))

# Plot figure
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Stacking (NIR)", fontsize=18)
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
plt.bar(x, f1_scores, color='blue')
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

# Save prediction results to CSV file
try:
    # Create results directory
    os.makedirs('stacking_results', exist_ok=True)
    
    # Prepare results data
    results_data = {
        'true_label': y_test,
        'true_label_name': [target_names[i] for i in y_test],
        'prediction': y_pred,
        'prediction_name': [target_names[i] for i in y_pred]
    }
    
    # Add probabilities for each class
    for i, class_name in enumerate(target_names):
        results_data[f'prob_{class_name}'] = y_proba[:, i]
    
    # Create DataFrame and save to CSV, filename includes data source information
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('NIR_sklearan_Machine_Learning_Models/stacking_results/predictions_nir.csv', index=False)
    print(f"\nPrediction results saved to: NIR_sklearan_Machine_Learning_Models/stacking_results/predictions_nir.csv")
except Exception as e:
    print(f"Error saving prediction results: {e}")

plt.tight_layout()
plt.show()