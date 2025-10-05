import os
import numpy as np
import sys
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score

# Add project root directory to system path to correctly import modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MRI_pytorch_Deep_Learning_Models.stacking_ensemble import load_images_and_labels, create_base_classifiers, create_meta_classifiers, \
    manual_stacking_ensemble, fit_manual_stacking_ensemble, predict_proba_manual_stacking_ensemble
from MRI_pytorch_Deep_Learning_Models.utils.utils_metrics import evaluteTop1_5

#------------------------------------------------------#
#   test_annotation_path    Test image paths and labels
#------------------------------------------------------#
test_annotation_path    = 'MRI_pytorch_Deep_Learning_Models/cls_test.txt'
#------------------------------------------------------#
#   metrics_out_path        Folder to save metrics
#------------------------------------------------------#
metrics_out_path        = "MRI_pytorch_Deep_Learning_Models/metrics_out_stacking"

class StackingClassification:
    def __init__(self):
        # Load training data for training meta-classifier
        print("Loading training data...")
        train_images, train_labels = load_images_and_labels("MRI_pytorch_Deep_Learning_Models/cls_train.txt")
        
        if len(train_images) == 0:
            print("No training images loaded, please check data paths and annotation files")
            return
        
        print(f"Successfully loaded {len(train_images)} training images")
        
        # Load test data for final testing
        print("Loading test data...")
        test_images, test_labels = load_images_and_labels("MRI_pytorch_Deep_Learning_Models/cls_test.txt")
        
        if len(test_images) == 0:
            print("No test images loaded, please check data paths and annotation files")
            return
        
        print(f"Successfully loaded {len(test_images)} test images")
        
        # Ensure label count matches image count
        if len(train_labels) != len(train_images):
            print("Training image and label counts do not match")
            return
            
        if len(test_labels) != len(test_images):
            print("Test image and label counts do not match")
            return
            
        # Create base classifiers
        print("Creating base classifiers...")
        base_classifiers = create_base_classifiers()
        
        # Create meta-classifiers
        print("Creating meta-classifiers...")
        meta_classifiers = create_meta_classifiers()
        
        # Use SVM as meta-classifier
        meta_clf = meta_classifiers['SVM']
        
        # Create manual stacking ensemble model
        print("Creating stacking ensemble model...")
        self.stacking_ensemble = manual_stacking_ensemble(base_classifiers, meta_clf)
        
        # Train stacking model's meta-classifier using training set data
        print("Training stacking ensemble model using training set data...")
        self.stacking_ensemble = fit_manual_stacking_ensemble(self.stacking_ensemble, train_images, train_labels)
        
        # Get class names
        self.class_names = []
        for _, clf in base_classifiers:
            self.class_names = clf.classes_.tolist()
            break
            
    def detect_image(self, image):
        """
        Predict a single image
        
        Parameters:
        image : PIL Image
            Input image
            
        Returns:
        preds : array
            Prediction probability distribution
        """
        # Wrap single image in a list
        image_list = [image]
        
        # Use stacking model for probability prediction
        proba = predict_proba_manual_stacking_ensemble(self.stacking_ensemble, image_list)
        
        # Return prediction result (squeeze handles single image case)
        return proba[0] if proba.ndim > 1 else proba

def evalute_stacking():
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
        
    print("Initializing Stacking ensemble model...")
    stacking_classification = StackingClassification()
    
    # Check if model was successfully initialized
    if not hasattr(stacking_classification, 'class_names') or not stacking_classification.class_names:
        print("Stacking ensemble model initialization failed")
        return
    
    print("Evaluating Stacking ensemble model...")
    with open(test_annotation_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        
    top1, top5, Recall, Precision = evaluteTop1_5(stacking_classification, lines, metrics_out_path)
    
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))
    
    # Generate prediction results CSV file
    print("Generating prediction results CSV file...")
    try:
        # Load test data
        test_images, test_labels = load_images_and_labels("MRI_pytorch_Deep_Learning_Models/cls_test.txt")
        
        # Get prediction probabilities
        y_proba = predict_proba_manual_stacking_ensemble(stacking_classification.stacking_ensemble, test_images)
        
        # Get prediction labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Class names
        #class_names = stacking_classification.class_names
        class_names = ['Class-S', 'Class-A', 'Class-B', 'Class-C']
        
        # Create results directory
        os.makedirs('stacking_results', exist_ok=True)
        
        # Prepare results data
        results_data = {
            'true_label': test_labels,
            'true_label_name': [class_names[i] for i in test_labels],
            'prediction': y_pred,
            'prediction_name': [class_names[i] for i in y_pred]
        }
        
        # Add probabilities for each class
        for i, class_name in enumerate(class_names):
            results_data[f'prob_{class_name}'] = y_proba[:, i]
        
        # Create DataFrame and save to CSV, file name includes data source information
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('MRI_pytorch_Deep_Learning_Models/stacking_results/predictions_mri.csv', index=False)
        print(f"Prediction results saved to: MRI_pytorch_Deep_Learning_Models/stacking_results/predictions_mri.csv")
        
        # Calculate and print accuracy
        accuracy = accuracy_score(test_labels, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error generating prediction results CSV file: {e}")

if __name__ == "__main__":
    evalute_stacking()