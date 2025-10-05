import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression  # For meta-classifier
import os
import sys
import random

# Add model path
sys.path.append('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9375-concat/Featurefusionmodel')
sys.path.append('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9635-add/Featurefusionmodel')
sys.path.append('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9740-weighted/Featurefusionmodel')

# Import necessary modules
# Modified import path, removed Featurefusionmodel prefix
from utils.data_utils import get_data_loaders
from models.fusion_model import FusionModel


def set_seed(seed=42):
    """
    Set random seed to ensure experiment reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model(model_path, nir_feature_count, fusion_type, device):
    """
    Load trained model
    
    Args:
        model_path: Model file path
        nir_feature_count: NIR feature count
        fusion_type: Fusion type
        device: Device
        
    Returns:
        model: Loaded model
    """
    # Create model
    model = FusionModel(
        nir_input_size=nir_feature_count,
        num_classes=4,
        fusion_type=fusion_type
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


class MetaClassifier(nn.Module):
    """
    Meta-classifier for stacking ensemble method
    """
    def __init__(self, n_features, n_classes):
        super(MetaClassifier, self).__init__()
        self.fc = nn.Linear(n_features, n_classes)
        
    def forward(self, x):
        return self.fc(x)


def get_base_predictions(models, data_loader, device, model_accuracies=None, threshold=0.85):
    """
    Get base classifier predictions as input features for meta-classifier
    
    Args:
        models: Base classifier model dictionary
        data_loader: Data loader
        device: Device
        model_accuracies: Model accuracy dictionary, used to filter low-performance models
        threshold: Accuracy threshold, models below this value will be excluded
        
    Returns:
        features: Base classifier predictions as features
        labels: True labels
    """
    for model in models.values():
        model.eval()
    
    all_features = []
    all_labels = []
    
    # Determine which models should be used
    selected_models = {}
    if model_accuracies:
        for name, model in models.items():
            if name in model_accuracies and model_accuracies[name] >= threshold:
                selected_models[name] = model
                print(f"Using model {name} (accuracy: {model_accuracies[name]:.4f})")
            elif name not in model_accuracies:
                selected_models[name] = model  # Use by default if accuracy is not provided
                print(f"Using model {name} (accuracy: unknown)")
        if len(selected_models) != len(models):
            print(f"Warning: {len(models) - len(selected_models)} low-performance models excluded")
    else:
        selected_models = models
        print("Using all base classifier models")
    
    with torch.no_grad():
        for nir_data, mri_data, labels in data_loader:
            nir_data = nir_data.to(device)
            mri_data = mri_data.to(device)
            labels = labels.to(device)
            
            # Get prediction probabilities for each base classifier
            model_probs = []
            for name, model in selected_models.items():
                outputs = model(nir_data, mri_data)
                probs = torch.softmax(outputs, dim=1)
                model_probs.append(probs)
            
            # Concatenate all base classifier probabilities as features
            features = torch.cat(model_probs, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Merge results from all batches
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    return features, labels


def train_meta_classifier(X_train, y_train, X_val, y_val, max_iter=1000):
    """
    Train meta-classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        max_iter: Maximum iterations
        
    Returns:
        meta_model: Trained meta-classifier
        val_accuracy: Validation accuracy
    """
    # Use logistic regression as meta-classifier
    meta_model = LogisticRegression(max_iter=max_iter, random_state=42)
    meta_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = meta_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    return meta_model, val_accuracy


def evaluate_stacking_ensemble(models, meta_model, test_loader, device, model_accuracies=None, threshold=0.85):
    """
    Evaluate stacking ensemble model
    
    Args:
        models: Base classifier model dictionary
        meta_model: Meta-classifier
        test_loader: Test data loader
        device: Device
        model_accuracies: Model accuracy dictionary, used to filter low-performance models
        threshold: Accuracy threshold
        
    Returns:
        predictions: Prediction results
        labels: True labels
    """
    # Get test set features
    X_test, y_test = get_base_predictions(models, test_loader, device, model_accuracies, threshold)
    
    # Predict using meta-classifier
    predictions = meta_model.predict(X_test)
    
    return predictions, y_test


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        title: Figure title
        save_path: Save path
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create new figure for confusion matrix and F1-score
    plt.figure(figsize=(12, 6))
    
    # Handle class name line breaks
    #class_names_multiline = [name.replace('-', '\n') for name in class_names]
    class_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']
    class_names_multiline = ['Class-S', 'Class-A', 'Class-B', 'Class-C']
    
    # Plot confusion matrix
    plt.subplot(121)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Stacking (NIR(Conv)-MRI(Conv))", fontsize=18)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names_multiline, rotation=45, fontsize=16)
    plt.yticks(tick_marks, class_names_multiline, fontsize=16)
    plt.xlabel("Predicted Freshness Grades", fontsize=16)
    plt.ylabel("True Freshness Grades", fontsize=16)
    
    # Adjust subplot spacing to prevent label overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Add values in confusion matrix
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    fontsize=16,
                    color="white" if cm[i, j] > thresh else "black")
    
    # Calculate F1-score and plot bar chart
    plt.subplot(122)
    # Calculate F1-score for each class
    f1_scores = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    x = range(len(class_names))
    plt.bar(x, f1_scores, color='orange')
    plt.xticks(x, class_names_multiline, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('F1-Score', fontsize=16)
    plt.title('F1-Score by Class', fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_ensemble_comparison(accuracies, save_path=None):
    """
    Plot performance comparison between ensemble model and individual models
    
    Args:
        accuracies: Accuracy dictionary
        save_path: Save path
    """
    models = list(accuracies.keys())
    values = list(accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Set random seed to ensure reproducible results
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model paths
    model_paths = {
        'concat': 'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9375-concat/best_fusion_model_concat-epoch984-trainacc0.9643-testacc0.9375.pth',
        'add': 'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9635-add/best_fusion_model_add-epoch789-trainacc0.9542-testacc0.9635.pth',
        'weighted': 'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/checkpoints/Fusionmodel-testacc0.9740-weighted/best_fusion_model_weighted-epoch586-trainacc0.9562-testacc0.9740.pth'
    }
    
    # Data paths
    data_root = './data'
    nir_train_path = os.path.join(data_root, 'nir_train.csv')
    nir_test_path = os.path.join(data_root, 'nir_test.csv')
    mri_data_dir = data_root
    
    # Class names
    #class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
    class_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']
    
    # Get NIR feature count
    sample_nir_data = pd.read_csv(nir_train_path, nrows=1)
    nir_feature_count = len(sample_nir_data.columns) - 2  # Subtract SampleID and Label columns
    print(f"NIR feature count: {nir_feature_count}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        nir_train_path=nir_train_path,
        nir_test_path=nir_test_path,
        mri_data_dir=mri_data_dir,
        batch_size=16,
        seed=42
    )
    
    # Load base classifier models
    print("Loading base classifier models...")
    models = {}
    for fusion_type, model_path in model_paths.items():
        models[fusion_type] = load_model(model_path, nir_feature_count, fusion_type, device)
        print(f"Successfully loaded {fusion_type} base classifier model")
    
    # Evaluate performance of each base classifier separately for comparison
    print("\n\n=== Base Classifier Performance Comparison ===")
    base_accuracies = {}
    for fusion_type, model in models.items():
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for nir_data, mri_data, labels in test_loader:
                nir_data = nir_data.to(device)
                mri_data = mri_data.to(device)
                labels = labels.to(device)
                
                outputs = model(nir_data, mri_data)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_predictions)
        base_accuracies[fusion_type] = acc
        print(f"{fusion_type.upper()} base classifier test accuracy: {acc:.4f}")
    
    # Get features (base classifier predictions) from training and test sets
    print("\nGetting base classifier predictions on training set...")
    X_train, y_train = get_base_predictions(models, train_loader, device, base_accuracies, threshold=0.90)
    print(f"Training feature shape: {X_train.shape}")
    
    print("\nGetting base classifier predictions on test set...")
    X_test, y_test = get_base_predictions(models, test_loader, device, base_accuracies, threshold=0.90)
    print(f"Test feature shape: {X_test.shape}")
    
    # To ensure consistent data splitting for each run, fix random seed before splitting
    np.random.seed(42)
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    # Use fixed indices for data splitting
    val_split = int(0.2 * len(X_train))
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train_meta = X_train[train_indices]
    y_train_meta = y_train[train_indices]
    
    print(f"\nValidation set size: {X_val.shape[0]}")
    print(f"Meta-training set size: {X_train_meta.shape[0]}")
    
    # Train meta-classifier
    print("\nTraining meta-classifier...")
    meta_model, val_accuracy = train_meta_classifier(
        X_train_meta, y_train_meta, X_val, y_val, max_iter=1000
    )
    print(f"Meta-classifier accuracy on validation set: {val_accuracy:.4f}")
    
    # Evaluate stacking ensemble model
    print("\nEvaluating stacking ensemble model...")
    stacking_predictions, stacking_labels = evaluate_stacking_ensemble(
        models, meta_model, test_loader, device, base_accuracies, threshold=0.90
    )
    
    stacking_accuracy = accuracy_score(stacking_labels, stacking_predictions)
    print(f'Stacking ensemble model test accuracy: {stacking_accuracy:.4f}')
    
    # Generate detailed classification report
    print('\n=== Stacking Ensemble Model Detailed Classification Report ===')
    print(classification_report(stacking_labels, stacking_predictions, target_names=class_names))
    
    # Generate confusion matrix
    print('\n=== Stacking Ensemble Model Confusion Matrix ===')
    cm = confusion_matrix(stacking_labels, stacking_predictions)
    print(cm)
    
    # Save prediction results
    try:
        # Ensure directory exists
        os.makedirs('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results', exist_ok=True)
        
        results_data = {
            'true_label': stacking_labels,
            'true_label_name': [class_names[i] for i in stacking_labels],
            'stacking_prediction': stacking_predictions,
            'stacking_prediction_name': [class_names[i] for i in stacking_predictions]
        }
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/stacking_ensemble_predictions.csv', index=False)
        print(f"\nPrediction results saved to: NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/stacking_ensemble_predictions.csv")
    except Exception as e:
        print(f"Error saving prediction results: {e}")
        print("Skipping saving prediction results step")
    
    # Generate prediction results CSV file consistent with other models
    try:
        # Get test set features (base classifier prediction probabilities)
        X_test, y_test = get_base_predictions(models, test_loader, device, base_accuracies, threshold=0.90)
        
        # Use logistic regression meta-classifier to get prediction probabilities
        y_proba = meta_model.predict_proba(X_test)
        
        # Create results data consistent with other models
        consistent_results_data = {
            'true_label': stacking_labels,
            'true_label_name': [class_names[i] for i in stacking_labels],
            'prediction': stacking_predictions,
            'prediction_name': [class_names[i] for i in stacking_predictions]
        }
        
        # Add probabilities for each class
        for i, class_name in enumerate(class_names):
            consistent_results_data[f'prob_{class_name}'] = y_proba[:, i]
        
        # Save in format consistent with other models, filename includes data source information
        consistent_results_df = pd.DataFrame(consistent_results_data)
        consistent_results_df.to_csv('NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/predictions_nir_mri_conv.csv', index=False)
        print(f"Prediction results saved to: NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/predictions_nir_mri_conv.csv")
    except Exception as e:
        print(f"Error saving consistent format prediction results: {e}")
        print("Skipping saving consistent format prediction results step")
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(
            stacking_labels, stacking_predictions, class_names, 
            'Stacking Ensemble Confusion Matrix',
            'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/confusion_matrix_stacking.png'
        )
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        print("Skipping plotting confusion matrix")
    
    # Output summary
    print("\n=== Performance Summary ===")
    for fusion_type, acc in base_accuracies.items():
        print(f"{fusion_type.upper()} base classifier accuracy: {acc:.4f}")
    print(f"Stacking ensemble model accuracy: {stacking_accuracy:.4f}")
    
    # Determine best model
    all_accuracies = {**base_accuracies, 
                      'stacking': stacking_accuracy}
    
    best_model_name = max(all_accuracies, key=all_accuracies.get)
    best_accuracy = all_accuracies[best_model_name]
    
    print(f"\nBest model: {best_model_name.upper()} (accuracy: {best_accuracy:.4f})")
    
    # Plot performance comparison
    try:
        plot_ensemble_comparison(all_accuracies, 'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/model_comparison.png')
    except Exception as e:
        print(f"Error plotting performance comparison: {e}")


if __name__ == '__main__':
    main()