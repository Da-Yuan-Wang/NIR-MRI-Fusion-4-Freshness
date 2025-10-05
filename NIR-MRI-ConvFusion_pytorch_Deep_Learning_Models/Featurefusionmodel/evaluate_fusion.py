import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models.fusion_model import FusionModel
from utils.data_utils import get_data_loaders
import os


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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        class_names: Class name list
        
    Returns:
        predictions: Prediction results
        labels: True labels
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for nir_data, mri_data, labels in test_loader:
            nir_data = nir_data.to(device)
            mri_data = mri_data.to(device)
            labels = labels.to(device)
            
            outputs = model(nir_data, mri_data)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


# def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
#     """
#     Plot confusion matrix
    
#     Args:
#         y_true: True labels
#         y_pred: Predicted labels
#         class_names: Class names
#         save_path: Image save path
#     """
#     cm = confusion_matrix(y_true, y_pred)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return cm


def plot_confusion_matrix_and_f1(conf_matrix, class_names, save_path=None):
    """
    Plot confusion matrix and F1-score chart
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Class name list
        save_path: Image save path
    """
    # Create new figure for confusion matrix and F1-score
    plt.figure(figsize=(12, 6))
    
    # Handle class name line breaks
    class_names_multiline = [name.replace('-', '\n') for name in class_names]
    
    # Plot confusion matrix
    plt.subplot(121)
    im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Confusion Matrix of 'weighted' Fusion", fontsize=14)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names_multiline, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names_multiline, fontsize=12)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    
    # Adjust subplot spacing to prevent label overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Add numbers to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    fontsize=12,
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    # Calculate F1-score and plot bar chart
    plt.subplot(122)
    # Calculate F1-score for each class
    f1_scores = []
    for i in range(len(class_names)):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    x = range(len(class_names))
    plt.bar(x, f1_scores, color='orange')
    plt.xticks(x, class_names_multiline, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score by Class', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_classification_report(class_report_dict, save_path=None):
    """
    Plot classification report (precision, recall, and F1-score for each class)
    
    Args:
        class_report_dict: Classification report dictionary
        save_path: Image save path
    """
    # Extract metrics for each class
    classes = list(class_report_dict.keys())[:-3]  # Remove 'accuracy', 'macro avg', 'weighted avg'
    precision = [class_report_dict[c]['precision'] for c in classes]
    recall = [class_report_dict[c]['recall'] for c in classes]
    f1_score = [class_report_dict[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-Score')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history_df, save_path=None):
    """
    Plot training curves
    
    Args:
        history_df: Training history data
        save_path: Image save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss curve
    ax1.plot(history_df['train_loss'], label='Training Loss')
    ax1.plot(history_df['test_loss'], label='Test Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curve
    ax2.plot(history_df['train_acc'], label='Training Accuracy')
    ax2.plot(history_df['test_acc'], label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Directly specify model path and parameters in the code
    model_path = r'D:\Yuan_Code\code\Fruit-Vegetable-LFNMR-NIR-Fusion\NIR-MRI-Fusion-ALL\NIR-MRI-ConvCat_pytorch_Deep_Learning_Models\checkpoints\最高的测试集准确率是-0.9740-weighted\best_fusion_model_weighted-epoch586-trainacc0.9562-testacc0.9740.pth'
    fusion_type = 'weighted'  # Determine weighted based on the filename in the model path
    batch_size = 16
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    data_root = '../data'
    nir_train_path = os.path.join(data_root, 'nir_train.csv')
    nir_test_path = os.path.join(data_root, 'nir_test.csv')
    mri_data_dir = data_root
    
    # Class names
    class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
    
    # Get data loaders
    _, test_loader = get_data_loaders(
        nir_train_path=nir_train_path,
        nir_test_path=nir_test_path,
        mri_data_dir=mri_data_dir,
        batch_size=batch_size
    )
    
    # NIR feature count
    sample_nir_data = pd.read_csv(nir_train_path, nrows=1)
    nir_feature_count = len(sample_nir_data.columns) - 2  # Subtract SampleID and Label columns
    
    # Load model
    model = load_model(model_path, nir_feature_count, fusion_type, device)
    print(f"Model loaded: {model_path}")
    
    # Evaluate model
    print("Evaluating model...")
    predictions, labels, probabilities = evaluate_model(model, test_loader, device, class_names)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f'Test accuracy: {accuracy:.4f}')
    
    # Generate classification report
    class_report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    print('\nClassification report:')
    print(classification_report(labels, predictions, target_names=class_names))
    
    # Plot confusion matrix
    save_dir = './Feature-fusionmodel-results'
    os.makedirs(save_dir, exist_ok=True)
    cm_save_path = os.path.join(save_dir, f'confusion_matrix_{fusion_type}.png')
    #plot_confusion_matrix(labels, predictions, class_names, cm_save_path)
    
    # Plot classification report chart
    cr_save_path = os.path.join(save_dir, f'classification_report_{fusion_type}.png')
    plot_classification_report(class_report, cr_save_path)
    
    # Plot confusion matrix and F1-score chart
    cm_f1_save_path = os.path.join(save_dir, f'confusion_matrix_and_f1_{fusion_type}.png')
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix_and_f1(cm, class_names, cm_f1_save_path)
    
    # Plot training curves (if training history exists)
    history_csv_path = os.path.join('./checkpoints', f'training_history_{fusion_type}.csv')
    if os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)
        curves_save_path = os.path.join(save_dir, f'training_curves_{fusion_type}.png')
        plot_training_curves(history_df, curves_save_path)
        print(f"Training curves saved to: {curves_save_path}")
    
    # Save prediction results
    results_df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'true_label_name': [class_names[i] for i in labels],
        'predicted_label_name': [class_names[i] for i in predictions]
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probabilities[:, i]
    
    results_csv_path = os.path.join(save_dir, f'predictions_{fusion_type}.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\nEvaluation complete!")
    print(f"Confusion matrix image saved to: {cm_save_path}")
    print(f"Classification report image saved to: {cr_save_path}")
    print(f"Prediction results saved to: {results_csv_path}")


if __name__ == '__main__':
    main()