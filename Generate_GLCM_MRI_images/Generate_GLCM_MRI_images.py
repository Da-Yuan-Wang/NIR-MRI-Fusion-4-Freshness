import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) features using 10 feature variables
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate GLCM features using multiple angles and distances
        distances = [1, 2]  # Use multiple distances
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Multiple angles
        glcm = graycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract 10 GLCM features
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()  # Correctly use energy property
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        asm = graycoprops(glcm, 'ASM').mean()  # ASM (Angular Second Moment)
        # Add more features
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Add small value to avoid log(0)
        variance = np.var(glcm)
        cluster_shade = np.sum((glcm - np.mean(glcm))**3)
        cluster_prominence = np.sum((glcm - np.mean(glcm))**4)
        
        return [contrast, energy, homogeneity, correlation, dissimilarity, asm, 
                entropy, variance, cluster_shade, cluster_prominence]
    except Exception as e:
        print(f"Error extracting GLCM features: {e}")
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Return 10 zero features

def find_image_path(data_dir, sample_id):
    """
    Find image path for specified sample ID in data directory
    """
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                # Extract sample ID from filename
                file_sample_id = file.replace('.jpg', '')
                if file_sample_id == sample_id:
                    return os.path.join(root, file)
    return None

def load_and_extract_mri_features(data_dir, sample_ids):
    """
    Load MRI images and extract features
    """
    glcm_features = []
    
    for sample_id in sample_ids:
        # Build image path
        img_path = find_image_path(data_dir, f"sample_id_{sample_id}")
        
        if img_path and os.path.exists(img_path):
            # Read image
            try:
                image = cv2.imread(img_path)
                
                # Extract GLCM features
                glcm_feat = extract_glcm_features(image)
                glcm_features.append(glcm_feat)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Add zero features
                glcm_features.append([0]*10)
        else:
            print(f"Image not found: sample_id_{sample_id}")
            # If image does not exist, add zero features
            glcm_features.append([0]*10)
    
    return np.array(glcm_features)

def load_nir_data():
    """
    Load NIR spectral data
    """
    train_data = pd.read_csv("data/nir_train.csv")
    test_data = pd.read_csv("data/nir_test.csv")
    
    # Separate features and labels
    X_train_nir = train_data.drop(['SampleID', 'Label'], axis=1)
    y_train = train_data['Label']
    X_eval_nir = test_data.drop(['SampleID', 'Label'], axis=1)
    y_eval = test_data['Label']
    
    return X_train_nir, y_train, X_eval_nir, y_eval

def extract_raw_features(X_train, X_eval):
    """
    Do not perform feature selection, keep all original wavelengths
    """
    # Return raw data directly
    return X_train, X_eval

def main():
    print("Starting NIR raw spectrum and GLCM feature extraction and concatenation...")
    
    # Load NIR data
    X_train_nir, y_train, X_eval_nir, y_eval = load_nir_data()
    print(f"NIR training set shape: {X_train_nir.shape}")
    print(f"NIR test set shape: {X_eval_nir.shape}")
    
    # Standardize NIR spectral data preprocessing
    print("Standardizing NIR spectral data...")
    nir_scaler = StandardScaler()
    X_train_nir_scaled = nir_scaler.fit_transform(X_train_nir)
    X_eval_nir_scaled = nir_scaler.transform(X_eval_nir)
    
    # Convert standardized data to DataFrame, keeping column names
    X_train_nir_scaled = pd.DataFrame(X_train_nir_scaled, columns=X_train_nir.columns)
    X_eval_nir_scaled = pd.DataFrame(X_eval_nir_scaled, columns=X_eval_nir.columns)
    
    print(f"Standardized NIR training set shape: {X_train_nir_scaled.shape}")
    print(f"Standardized NIR test set shape: {X_eval_nir_scaled.shape}")
    
    # Extract NIR raw spectral features
    print("Extracting NIR raw spectral features...")
    X_train_raw, X_eval_raw = extract_raw_features(X_train_nir_scaled, X_eval_nir_scaled)
    
    print(f"Raw feature count: {X_train_raw.shape[1]}")
    
    # Get sample IDs
    train_sample_ids = pd.read_csv("data/nir_train.csv")['SampleID'].str.replace('sample_id_', '')
    eval_sample_ids = pd.read_csv("data/nir_test.csv")['SampleID'].str.replace('sample_id_', '')
    
    # Extract MRI GLCM features
    print("Extracting MRI GLCM image features...")
    train_glcm = load_and_extract_mri_features("data/train", train_sample_ids)
    eval_glcm = load_and_extract_mri_features("data/test", eval_sample_ids)
    
    print(f"GLCM feature count: {train_glcm.shape[1]}")
    
    # Feature-level fusion
    print("Performing feature-level fusion...")
    # Define feature combination
    key = "Raw+GLCM"
    train_combined = np.concatenate([X_train_raw.values, train_glcm], axis=1)
    eval_combined = np.concatenate([X_eval_raw.values, eval_glcm], axis=1)
    
    # Generate feature names
    nir_feature_names = [f"Raw_Wavelength_{i+1}" for i in range(X_train_raw.shape[1])]
    mri_feature_names = ["GLCM_Contrast", "GLCM_Energy", "GLCM_Homogeneity", "GLCM_Correlation", 
                         "GLCM_Dissimilarity", "GLCM_ASM", "GLCM_Entropy", "GLCM_Variance",
                         "GLCM_Cluster_Shade", "GLCM_Cluster_Prominence"]
    combined_feature_names = nir_feature_names + mri_feature_names
    
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_combined)
    eval_scaled = scaler.transform(eval_combined)
    
    # Save fused feature data to CSV files
    print("Saving fused feature data to CSV files...")
    output_dir = "Generate_GLCM_MRI_images/Fused_Feature_data"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    
    # Get feature names
    feature_names = combined_feature_names
    
    # Create training set DataFrame
    train_df = pd.DataFrame(train_scaled, columns=feature_names)
    train_df['Label'] = y_train
    
    # Create test set DataFrame
    eval_df = pd.DataFrame(eval_scaled, columns=feature_names)
    eval_df['Label'] = y_eval
    
    # Save to CSV files
    train_file_path = os.path.join(output_dir, "train_data_Raw_GLCM.csv")
    eval_file_path = os.path.join(output_dir, "test_data_Raw_GLCM.csv")
    
    train_df.to_csv(train_file_path, index=False)
    eval_df.to_csv(eval_file_path, index=False)
    
    print(f"Saved feature data of {key} combination to CSV files")
    print("Processing completed!")

if __name__ == "__main__":
    main()