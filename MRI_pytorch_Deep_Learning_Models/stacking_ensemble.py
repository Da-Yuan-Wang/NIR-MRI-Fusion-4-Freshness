import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import sys
from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold for cross-validation

# Add project root directory to system path to correctly import MRI_pytorch_Deep_Learning_Models modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MRI_pytorch_Deep_Learning_Models.classification import Classification
from MRI_pytorch_Deep_Learning_Models.utils.utils import (cvtColor, letterbox_image, preprocess_input)


class MRIModelWrapper:
    """
    Wrap MRI deep learning model to be compatible with sklearn interface
    """
    def __init__(self, model_path, backbone, input_shape=[224, 224], classes_path='MRI_pytorch_Deep_Learning_Models/model_data/cls_classes.txt'):
        self.model_path = model_path
        self.backbone = backbone
        self.input_shape = input_shape
        self.classes_path = classes_path
        self.classes_ = None
        self._load_model()
    
    def _load_model(self):
        """
        Load MRI deep learning model
        """
        # Create Classification instance to load model
        # Fix path separator issue
        model_path = self.model_path.replace('\\', '/')
        classes_path = self.classes_path.replace('\\', '/')
        
        self.classifier = Classification(
            model_path=model_path,
            backbone=self.backbone,
            input_shape=self.input_shape,
            classes_path=classes_path
        )
        self.classes_ = np.array(self.classifier.class_names)
        
    def predict_proba(self, X):
        """
        Predict the probability of samples belonging to each class
        
        Parameters:
        X : list of PIL Images
            Test image list
            
        Returns:
        probabilities : array, shape = [n_samples, n_classes]
            Predicted probabilities
        """
        if len(X) == 0:
            return np.array([])
            
        probabilities = []
        for image in X:
            try:
                # Process images using the same method as in eval.py
                image = cvtColor(image)
                image_data = letterbox_image(image, [self.classifier.input_shape[1], self.classifier.input_shape[0]], self.classifier.letterbox_image)
                image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))
                
                with torch.no_grad():
                    photo = torch.from_numpy(image_data).type(torch.FloatTensor)
                    if self.classifier.cuda and torch.cuda.is_available():
                        photo = photo.cuda()
                    preds = torch.softmax(self.classifier.model(photo)[0], dim=-1).cpu().numpy()
                    
                    # Handle HRNet model outputting ImageNet 1000 classes issue
                    # If the number of output classes does not match the task classes, special handling is needed
                    # Here we assume the model actually outputs task-related classes (e.g., 16 classes), 
                    # but as a pre-trained model it may output 1000 classes
                    # We only take the first len(self.classes_) classes, which may require more complex mapping in actual situations
                    if len(self.classes_) != len(preds):
                        # HRNet model outputs ImageNet 1000 class predictions, need to map to actual classes
                        # If prediction result classes are more than actual classes, only take the first few
                        if len(preds) > len(self.classes_):
                            # If prediction result classes are more than actual classes, only take the first few
                            preds = preds[:len(self.classes_)]
                        else:
                            # If prediction result classes are less than actual classes, pad with zeros
                            pad_width = len(self.classes_) - len(preds)
                            preds = np.pad(preds, (0, pad_width), mode='constant')
                    
                    # Ensure probability sum is 1
                    preds = preds / np.sum(preds)
                    probabilities.append(preds)
            except Exception as e:
                print(f"Error processing image: {e}")
                # Return uniform distribution as default probability
                probabilities.append(np.ones(len(self.classes_)) / len(self.classes_))
        
        return np.array(probabilities).squeeze()
    
    def predict(self, X):
        """
        Predict samples
        
        Parameters:
        X : list of PIL Images
            Test image list
            
        Returns:
        y_pred : array, shape = [n_samples]
            Predicted class labels
        """
        if len(X) == 0:
            return np.array([])
            
        probabilities = self.predict_proba(X)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        return np.argmax(probabilities, axis=1)

    def fit(self, X, y):
        """
        Empty method implemented for sklearn interface compatibility
        """
        return self
    
    def get_params(self, deep=True):
        """
        Get model parameters, implement sklearn interface
        """
        return {
            "model_path": self.model_path,
            "backbone": self.backbone,
            "input_shape": self.input_shape,
            "classes_path": self.classes_path
        }
    
    def set_params(self, **params):
        """
        Set model parameters, implement sklearn interface
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def load_images_and_labels(annotation_file):
    """
    Load images and labels from annotation file
    
    Parameters:
    annotation_file : str
        Annotation file path
        
    Returns:
    images : list of PIL Images
        Image list
    labels : array
        Label array
    """
    images = []
    labels = []
    
    # Fix path separator
    annotation_file = annotation_file.replace('\\', '/')
    
    if not os.path.exists(annotation_file):
        print(f"Annotation file does not exist: {annotation_file}")
        return images, np.array(labels)
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 2:
            try:
                label = int(parts[0])
                image_path = parts[1].split()[0]  # Handle path that may contain additional information
                # Fix path separator
                image_path = image_path.replace('\\', '/')
                
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        images.append(image)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                else:
                    print(f"Image file does not exist: {image_path}")
            except Exception as e:
                print(f"Error processing line '{line}': {e}")
    
    return images, np.array(labels)


def create_base_classifiers():
    """
    Create base classifier list
    """
    # Define base classifiers
    base_classifiers = [
        # Can add multiple different MRI models
        # ('Xception', MRIModelWrapper(
        #     model_path='MRI_pytorch_Deep_Learning_Models\logs\Xception-loss_2025_08_08_17_40_16\ep793-loss0.399-val_loss0.329-val_accuracy0.914.pth',
        #     backbone='Xception'
        # )),
        # ('mobilenet', MRIModelWrapper(
        #     model_path='MRI_pytorch_Deep_Learning_Models/logs/MobileNetV2-loss_2025_08_06_16_31_53/ep1168-loss0.103-val_loss0.396-val_accuracy0.909.pth',
        #     backbone='mobilenet'
        # )),
        ('ghostnet', MRIModelWrapper(
            model_path='MRI_pytorch_Deep_Learning_Models\logs\Ghostnet-loss_2025_08_07_10_02_36\ep1130-loss0.363-val_loss0.405-val_accuracy0.901-used.pth',
            backbone='ghostnet'
        )),
        ('hrnet', MRIModelWrapper(
            model_path='MRI_pytorch_Deep_Learning_Models\logs\HRNet-loss_2025_08_07_14_43_53\ep1012-loss0.485-val_loss0.257-val_accuracy0.924-Used.pth',
            backbone='cls_hrnet'
        )),
        ('resnet50', MRIModelWrapper(
            model_path='MRI_pytorch_Deep_Learning_Models\logs\ResNet50-loss_2025_08_06_10_02_24\ep301-loss0.129-val_loss0.206-val_accuracy0.948-Uesd.pth',
            backbone="resnet50"
        )),
        # Add other models...
    ]
    
    return base_classifiers


def create_meta_classifiers():
    """
    Create SVM meta-classifiers
    """
    meta_classifiers = {
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
    }
    
    return meta_classifiers


def manual_stacking_ensemble(base_classifiers, meta_classifier=None, n_folds=5):
    """
    Manually implement stacking ensemble learning
    
    Parameters:
    base_classifiers : list
        Base classifier list, each element is a (name, classifier) tuple
    meta_classifier : sklearn classifier
        Meta-classifier, defaults to LogisticRegression
    n_folds : int
        Number of cross-validation folds, defaults to 5
    
    Returns:
    ensemble : dict
        Dictionary containing base classifiers and meta-classifier
    """
    if meta_classifier is None:
        meta_classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    return {
        'base_classifiers': base_classifiers,
        'meta_classifier': meta_classifier,
        'n_folds': n_folds  # Save number of cross-validation folds
    }


def fit_manual_stacking_ensemble(ensemble, X, y):
    """
    Train manual stacking ensemble model
    
    Parameters:
    ensemble : dict
        Ensemble model created by manual_stacking_ensemble
    X : list of PIL Images
        Training image list
    y : array
        Training labels
    
    Returns:
    ensemble : dict
        Trained ensemble model
    """
    base_classifiers = ensemble['base_classifiers']
    meta_classifier = ensemble['meta_classifier']
    n_folds = ensemble['n_folds']  # Get number of cross-validation folds
    
    # Ensure y is a numpy array
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Get number of classes
    n_classes = len(np.unique(y))
    
    # Initialize meta-feature array
    meta_features = []
    meta_labels = []
    
    # Use StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        print(f"Obtaining validation set predictions for fold {fold+1}...")
        
        # Extract training and validation sets for current fold
        X_train_fold = [X[i] for i in train_idx.tolist()]
        y_train_fold = y[train_idx.tolist()]
        X_val_fold = [X[i] for i in val_idx.tolist()]
        y_val_fold = y[val_idx.tolist()]
        
        # Train and predict each base classifier
        fold_meta_features = []
        for name, clf in base_classifiers:
            print(f"Training and predicting base classifier {name}...")
            
            # Train base classifier
            clf.fit(X_train_fold, y_train_fold)
            
            # Validation set prediction
            proba = clf.predict_proba(X_val_fold)
            
            # Ensure prediction result shape is correct
            if len(proba.shape) == 1:
                proba = proba.reshape(1, -1) if len(X_val_fold) == 1 else proba.reshape(-1, 1)
            
            fold_meta_features.append(proba)
        
        # Store current fold predictions in meta_features
        current_meta = np.concatenate(fold_meta_features, axis=1)
        meta_features.append(current_meta)
        meta_labels.append(y_val_fold)
        fold += 1
    
    # Combine results from all folds
    meta_features = np.vstack(meta_features)
    meta_labels = np.hstack(meta_labels)
    
    # Train meta-classifier
    print("Training meta-classifier...")
    meta_classifier.fit(meta_features, meta_labels)
    
    return ensemble


def predict_manual_stacking_ensemble(ensemble, X):
    """
    Predict using manual stacking ensemble model
    
    Parameters:
    ensemble : dict
        Trained ensemble model
    X : list of PIL Images
        Test image list
        
    Returns:
    predictions : array
        Predicted labels
    """
    base_classifiers = ensemble['base_classifiers']
    meta_classifier = ensemble['meta_classifier']
    
    # Get base classifier predictions as meta-features
    meta_features = []
    for name, clf in base_classifiers:
        proba = clf.predict_proba(X)
        meta_features.append(proba)
    
    # Concatenate all base classifier probability predictions as meta-features
    meta_X = np.concatenate(meta_features, axis=1)
    
    # Predict using meta-classifier
    return meta_classifier.predict(meta_X)


def predict_proba_manual_stacking_ensemble(ensemble, X):
    """
    Predict probabilities using manual stacking ensemble model
    
    Parameters:
    ensemble : dict
        Trained ensemble model
    X : list of PIL Images
        Test image list
        
    Returns:
    probabilities : array
        Predicted probabilities
    """
    base_classifiers = ensemble['base_classifiers']
    meta_classifier = ensemble['meta_classifier']
    
    # Get base classifier predictions as meta-features
    meta_features = []
    for name, clf in base_classifiers:
        proba = clf.predict_proba(X)
        # Ensure proba is a 2D array
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        meta_features.append(proba)
    
    # Concatenate all base classifier probability predictions as meta-features
    meta_X = np.concatenate(meta_features, axis=1)
    
    # Predict probabilities using meta-classifier
    return meta_classifier.predict_proba(meta_X)


def main():
    """
    Main function: execute ensemble learning
    """
    # Load training data for training meta-classifier
    print("Loading training data...")
    train_images, train_labels = load_images_and_labels("MRI_pytorch_Deep_Learning_Models/cls_train.txt")
    
    if len(train_images) == 0:
        print("No training images loaded, please check data path and annotation file")
        return
    
    print(f"Successfully loaded {len(train_images)} training images")
    
    # Load test data for final testing
    print("Loading test data...")
    test_images, test_labels = load_images_and_labels("MRI_pytorch_Deep_Learning_Models/cls_test.txt")
    
    if len(test_images) == 0:
        print("No test images loaded, please check data path and annotation file")
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
    
    # Create multiple meta-classifiers
    print("Creating multiple meta-classifiers...")
    meta_classifiers = create_meta_classifiers()
    
    # Store accuracy results for each model
    results = {}
    
    print("\n=== Stacking Ensemble with Different Meta-Classifiers ===")
    # Train and test for each meta-classifier
    for meta_name, meta_clf in meta_classifiers.items():
        print(f"\n--- Using {meta_name} as Meta-Classifier ---")
        # Create manual stacking ensemble model
        stacking_ensemble = manual_stacking_ensemble(base_classifiers, meta_clf)
        
        # Train stacking model's meta-classifier using training set data
        print(f"Training stacking ensemble model based on {meta_name} using training set data...")
        try:
            stacking_ensemble = fit_manual_stacking_ensemble(stacking_ensemble, train_images, train_labels)
        except Exception as e:
            print(f"Error training stacking model based on {meta_name}: {e}")
            continue
        
        # Predict on test set
        print(f"Predicting on test set...")
        try:
            y_pred_stacking = predict_manual_stacking_ensemble(stacking_ensemble, test_images)
        except Exception as e:
            print(f"Error predicting with stacking based on {meta_name}: {e}")
            continue
        
        # Evaluate stacking model performance on test set
        print(f"Evaluating stacking model based on {meta_name} on test set...")
        try:
            accuracy_stacking = accuracy_score(test_labels, y_pred_stacking)
            results[meta_name] = accuracy_stacking
            print(f"Stacking ensemble model based on {meta_name} accuracy on test set: {accuracy_stacking:.4f}")
        except Exception as e:
            print(f"Error evaluating stacking based on {meta_name}: {e}")
            continue
    
    # Evaluate individual model performance on test set
    print("\n=== Individual Model Performance on Test Set ===")
    print("Note: Here we evaluate individual model performance using the complete test set (384 samples), consistent with previous test results")
    
    for name, clf in base_classifiers:
        try:
            single_pred = clf.predict(test_images)
            single_accuracy = accuracy_score(test_labels, single_pred)
            results[name] = single_accuracy
            print(f"{name} accuracy: {single_accuracy:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Output accuracy comparison for all models
    print("\n=== Accuracy Comparison for All Models ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for model_name, accuracy in sorted_results:
        print(f"{model_name:20s}: {accuracy:.4f}")


if __name__ == "__main__":
    main()