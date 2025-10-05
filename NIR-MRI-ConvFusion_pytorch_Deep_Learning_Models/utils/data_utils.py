import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms
import os


class FusionDataset(Dataset):
    """
    Fusion dataset class that handles both NIR spectral data and MRI image data
    """
    def __init__(self, nir_data_path, mri_data_dir, transform=None, mode='train', augment_times=1):
        """
        Initialize dataset
        
        Args:
            nir_data_path: NIR data CSV file path
            mri_data_dir: MRI image data directory path
            transform: Image transformations
            mode: Data mode ('train' or 'test')
            augment_times: Data augmentation times
        """
        self.nir_data_path = nir_data_path
        self.mri_data_dir = mri_data_dir
        self.transform = transform
        self.mode = mode
        self.augment_times = augment_times if mode == 'train' else 1  # Only perform data augmentation in training mode
        
        # Load NIR data
        self.nir_data = pd.read_csv(nir_data_path)
        
        # Extract sample IDs, features, and labels
        self.sample_ids = self.nir_data.iloc[:, 0].values  # First column is SampleID
        self.nir_features = self.nir_data.iloc[:, 1:-1].values  # Middle columns are features
        self.labels = self.nir_data.iloc[:, -1].values  # Last column is label
        
        # Standardize NIR features
        self.scaler = StandardScaler()
        self.nir_features = self.scaler.fit_transform(self.nir_features)
        
        # Image preprocessing
        if self.transform is None:
            if self.mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # Uniformly resize to 224x224
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # Uniformly resize to 224x224
                    transforms.ToTensor(),
                ])
        
        # If in training mode and augmentation is needed, expand the dataset
        if self.mode == 'train' and self.augment_times > 1:
            # Expand dataset
            original_size = len(self.sample_ids)
            self.sample_ids = np.tile(self.sample_ids, self.augment_times)
            self.nir_features = np.tile(self.nir_features, (self.augment_times, 1))
            self.labels = np.tile(self.labels, self.augment_times)
            
            print(f"Data augmentation: Original training set sample count {original_size}, augmented sample count {len(self.sample_ids)}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        # Get sample ID
        sample_id = self.sample_ids[idx]
        
        # Get NIR feature
        nir_feature = self.nir_features[idx]
        
        # Add minor noise to NIR data for augmentation (only in training mode)
        if self.mode == 'train':
            noise = np.random.normal(0, 0.01, nir_feature.shape)  # Add Gaussian noise
            nir_feature = nir_feature + noise
            
        nir_feature = torch.FloatTensor(nir_feature).unsqueeze(0)  # Add channel dimension (1, features)
        
        # Get label
        label = self.labels[idx]
        label = torch.LongTensor([label]).squeeze()
        
        # Determine image directory based on mode (train or test)
        mri_mode_dir = 'train' if self.mode == 'train' else 'test'
        full_mri_dir = os.path.join(self.mri_data_dir, mri_mode_dir)
        
        # Construct image path
        class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
        
        # Try to get class directory from label
        try:
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            class_dir = class_names[label_value]
            img_path = os.path.join(full_mri_dir, class_dir, f"{sample_id}.jpg")
        except IndexError:
            # If label is out of class range, use default directory
            img_path = os.path.join(full_mri_dir, 'Fresh', f"{sample_id}.jpg")
        
        # If not found by standard path, try to find in all subdirectories
        if not os.path.exists(img_path):
            for root, dirs, files in os.walk(full_mri_dir):
                if f"{sample_id}.jpg" in files:
                    img_path = os.path.join(root, f"{sample_id}.jpg")
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as a substitute
            image = Image.new('RGB', (224, 224))
        
        # Apply basic transformations
        if self.transform:
            image = self.transform(image)
        
        # Perform MRI image data augmentation in training mode
        if self.mode == 'train':
            # Random rotation (-15 degrees to 15 degrees)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
            
            # Add minor noise
            if np.random.random() > 0.5:
                noise = torch.randn_like(image) * 0.01  # Add small Gaussian noise
                image = image + noise
            
            # Ensure pixel values are within reasonable range
            image = torch.clamp(image, 0, 1)
        
        # Normalize (after data augmentation)
        if self.mode == 'train':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        
        return nir_feature, image, label


def get_data_loader(nir_csv_file, mri_data_dir, batch_size=32, mode='train', shuffle=True, augment_times=1, seed=None):
    """
    Get data loader
    
    Args:
        nir_csv_file (str): NIR spectral data CSV file path
        mri_data_dir (str): MRI image data directory path
        batch_size (int): Batch size
        mode (str): Data mode ('train' or 'test')
        shuffle (bool): Whether to shuffle data
        augment_times (int): Data augmentation times
        seed (int): Random seed, for ensuring consistency in data loading
        
    Returns:
        DataLoader: PyTorch data loader
    """
    dataset = FusionDataset(nir_csv_file, mri_data_dir, mode=mode, augment_times=augment_times)
    
    # If seed is provided, use fixed seed generator
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        
    return data_loader


def get_data_loaders(nir_train_path, nir_test_path, mri_data_dir, batch_size=32, augment_times=4, seed=None):
    """
    Get training and testing data loaders
    
    Args:
        nir_train_path (str): Training set NIR spectral data path
        nir_test_path (str): Testing set NIR spectral data path
        mri_data_dir (str): MRI image data root directory
        batch_size (int): Batch size
        augment_times (int): Data augmentation times
        seed (int): Random seed, for ensuring consistency in data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = get_data_loader(nir_train_path, mri_data_dir, batch_size, mode='train', shuffle=True, augment_times=augment_times, seed=seed)
    test_loader = get_data_loader(nir_test_path, mri_data_dir, batch_size, mode='test', shuffle=False, augment_times=1, seed=seed)
    return train_loader, test_loader


def load_pretrained_extractors(nir_model_path, mri_model_path, device):
    """
    Load pretrained feature extractors
    
    Args:
        nir_model_path: NIR model path
        mri_model_path: MRI model path
        device: Device
        
    Returns:
        nir_extractor, mri_extractor: NIR and MRI feature extractors
    """
    # Import models
    from Featurefusionmodel.models.fusion_model import NIRFeatureExtractor, MRIFeatureExtractor
    
    # Load NIR model checkpoint
    nir_checkpoint = torch.load(nir_model_path, map_location=device)
    input_size = nir_checkpoint['input_size']
    
    # Create NIR feature extractor
    nir_extractor = NIRFeatureExtractor(input_size).to(device)
    
    # Load NIR model weights
    # Extract feature extractor part weights from the full model
    nir_state_dict = {}
    for key, value in nir_checkpoint['model_state_dict'].items():
        if key.startswith('conv') or key.startswith('bn') or key.startswith('fc'):
            # Remove classifier part weights
            if not key.startswith(('fc4')):
                nir_state_dict[key] = value
    
    nir_extractor.load_state_dict(nir_state_dict, strict=False)
    
    # Create MRI feature extractor
    mri_extractor = MRIFeatureExtractor(mri_model_path).to(device)
    
    return nir_extractor, mri_extractor