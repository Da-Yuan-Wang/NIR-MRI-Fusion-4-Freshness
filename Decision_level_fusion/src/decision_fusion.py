import pandas as pd
import numpy as np
from collections import Counter
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Define weights for four base classifiers (based on decimal places of accuracy)
CLASSIFIER_WEIGHTS = {
    'nir': 0.0792,            # NIR-Stacking     0.9792
    'mri': 0.0557,            # MRI-Stacking   0.9557
    'nir_mri_glcm': 0.0896,   # NIR-MRI-GLCM-Stacking   0.9896
    'nir_mri_conv': 0.0557    # NIR-MRI-ConvFusion-Stacking 0.9557
}

# Class names

target_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

# Modify label display to two lines
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

def load_predictions():
    """Load prediction results from four base classifiers"""
    # Define file paths
    nir_file = 'NIR_sklearan_Machine_Learning_Models/stacking_results/predictions_nir.csv'
    mri_file = 'MRI_pytorch_Deep_Learning_Models/stacking_results/predictions_mri.csv'
    glcm_file = 'NIR-MRI(GLCM)_sklearan_Machine_Learning_Models/stacking_results/predictions_nir_mri_glcm.csv'
    conv_file = 'NIR-MRI-ConvFusion_pytorch_Deep_Learning_Models/stacking_results/predictions_nir_mri_conv.csv'
    
    # Load data
    nir_df = pd.read_csv(nir_file)
    mri_df = pd.read_csv(mri_file)
    glcm_df = pd.read_csv(glcm_file)
    conv_df = pd.read_csv(conv_file)
    
    return nir_df, mri_df, glcm_df, conv_df

def plot_confusion_matrix_and_f1(true_labels, predictions, method_name):
    """Plot confusion matrix and F1-score bar chart"""
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot confusion matrix
    plt.subplot(121)
    im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.YlGn)
    plt.title(f"{method_name} of Stacking Results", fontsize=18)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(target_names))
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
    plt.bar(x, f1_scores, color='pink')
    plt.xticks(x, target_names_multiline, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('F1-Score', fontsize=16)
    plt.title(f'F1-Score by Class of {method_name}', fontsize=18)
    
    # Save figure
    output_dir = 'decision_level_fusion/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decision_fusion_{method_name}.png')
    
    # Show figure
    plt.show()
    
    # Close figure to release memory
    plt.close()
    
    # Print accuracy and confusion matrix
    acc = accuracy_score(true_labels, predictions)
    print(f"{method_name} Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

def voting_fusion(nir_df, mri_df, glcm_df, conv_df):
    """Hard voting fusion method (based on voting of prediction labels)"""
    # Get true labels
    true_labels = nir_df['true_label'].values
    true_label_names = nir_df['true_label_name'].values
    
    # Get prediction results
    nir_pred = nir_df['prediction'].values
    mri_pred = mri_df['prediction'].values
    glcm_pred = glcm_df['prediction'].values
    conv_pred = conv_df['prediction'].values
    
    # Hard voting fusion
    fused_predictions = []
    fused_prediction_names = []
    
    # Class name mapping
    class_name_map = {}
    for i in range(len(nir_df)):
        class_name_map[nir_df['prediction'].iloc[i]] = nir_df['prediction_name'].iloc[i]
    
    for i in range(len(true_labels)):
        votes = [nir_pred[i], mri_pred[i], glcm_pred[i], conv_pred[i]]
        # Count votes
        vote_count = Counter(votes)
        # Select the most voted class
        fused_pred = vote_count.most_common(1)[0][0]
        fused_predictions.append(fused_pred)
        
        # Get class name
        fused_prediction_names.append(class_name_map.get(fused_pred, 'Unknown'))
    
    return np.array(fused_predictions), np.array(fused_prediction_names)

def averaging_fusion(nir_df, mri_df, glcm_df, conv_df):
    """Averaging fusion method"""
    # Get true labels
    true_labels = nir_df['true_label'].values
    true_label_names = nir_df['true_label_name'].values
    
    # Get probabilities of each class
    #class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
    class_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']
    
    # Calculate average probabilities
    avg_probs = []
    for i in range(len(true_labels)):
        avg_prob = []
        for class_name in class_names:
            prob_col = f'prob_{class_name}'
            avg_p = (nir_df[prob_col].iloc[i] + 
                     mri_df[prob_col].iloc[i] + 
                     glcm_df[prob_col].iloc[i] + 
                     conv_df[prob_col].iloc[i]) / 4
            avg_prob.append(avg_p)
        avg_probs.append(avg_prob)
    
    # Determine predicted class based on average probabilities
    fused_predictions = np.argmax(avg_probs, axis=1)
    
    # Get predicted class names
    fused_prediction_names = []
    for pred in fused_predictions:
        fused_prediction_names.append(class_names[pred])
    
    return fused_predictions, np.array(fused_prediction_names)

def weighted_fusion(nir_df, mri_df, glcm_df, conv_df):
    """Weighted fusion method"""
    # Get true labels
    true_labels = nir_df['true_label'].values
    true_label_names = nir_df['true_label_name'].values
    
    # Calculate total weight
    total_weight = sum(CLASSIFIER_WEIGHTS.values())
    
    # Normalize weights
    normalized_weights = {k: v/total_weight for k, v in CLASSIFIER_WEIGHTS.items()}
    
    # Get probabilities of each class
    #class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
    class_names =['Class-S', 'Class-A', 'Class-B', 'Class-C']
    
    # Calculate weighted average probabilities
    weighted_probs = []
    for i in range(len(true_labels)):
        weighted_prob = []
        for class_name in class_names:
            prob_col = f'prob_{class_name}'
            weighted_p = (normalized_weights['nir'] * nir_df[prob_col].iloc[i] +
                          normalized_weights['mri'] * mri_df[prob_col].iloc[i] + 
                          normalized_weights['nir_mri_glcm'] * glcm_df[prob_col].iloc[i] + 
                          normalized_weights['nir_mri_conv'] * conv_df[prob_col].iloc[i])
            weighted_prob.append(weighted_p)
        weighted_probs.append(weighted_prob)
    
    # Determine predicted class based on weighted average probabilities
    fused_predictions = np.argmax(weighted_probs, axis=1)
    
    # Get predicted class names
    fused_prediction_names = []
    for pred in fused_predictions:
        fused_prediction_names.append(class_names[pred])
    
    return fused_predictions, np.array(fused_prediction_names)

def save_results(true_labels, true_label_names, predictions, prediction_names, method_name):
    """Save fusion results"""
    result_df = pd.DataFrame({
        'true_label': true_labels,
        'true_label_name': true_label_names,
        'fusion_prediction': predictions,
        'fusion_prediction_name': prediction_names
    })
    
    # Create output directory
    output_dir = 'decision_level_fusion/results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    output_file = f'{output_dir}/decision_fusion_{method_name}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"{method_name} fusion results saved to: {output_file}")

def calculate_accuracy(true_labels, predictions):
    """Calculate accuracy"""
    correct = np.sum(true_labels == predictions)
    total = len(true_labels)
    return correct / total

def main():
    """Main function"""
    print("Loading prediction results...")
    nir_df, mri_df, glcm_df, conv_df = load_predictions()
    
    print("Performing hard voting fusion...")
    voting_pred, voting_pred_names = voting_fusion(nir_df, mri_df, glcm_df, conv_df)
    save_results(nir_df['true_label'].values, nir_df['true_label_name'].values, 
                 voting_pred, voting_pred_names, "voting")
    plot_confusion_matrix_and_f1(nir_df['true_label'].values, voting_pred, "Hard Voting")
    
    print("Performing averaging probability fusion...")
    avg_pred, avg_pred_names = averaging_fusion(nir_df, mri_df, glcm_df, conv_df)
    save_results(nir_df['true_label'].values, nir_df['true_label_name'].values, 
                 avg_pred, avg_pred_names, "averaging")
    plot_confusion_matrix_and_f1(nir_df['true_label'].values, avg_pred, "Averaging")
    
    print("Performing weighted fusion...")
    weighted_pred, weighted_pred_names = weighted_fusion(nir_df, mri_df, glcm_df, conv_df)
    save_results(nir_df['true_label'].values, nir_df['true_label_name'].values, 
                 weighted_pred, weighted_pred_names, "weighted")
    plot_confusion_matrix_and_f1(nir_df['true_label'].values, weighted_pred, "Weighted")
    
    # Calculate and print accuracy
    true_labels = nir_df['true_label'].values
    print("\nFusion method accuracy:")
    print(f"Hard voting fusion: {calculate_accuracy(true_labels, voting_pred):.4f}")
    print(f"Averaging fusion: {calculate_accuracy(true_labels, avg_pred):.4f}")
    print(f"Weighted fusion: {calculate_accuracy(true_labels, weighted_pred):.4f}")

if __name__ == "__main__":
    main()