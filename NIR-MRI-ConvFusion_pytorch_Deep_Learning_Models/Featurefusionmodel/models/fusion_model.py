import torch
import torch.nn as nn
import torch.nn.functional as F


class NIRFeatureExtractor(nn.Module):
    """
    1D-CNN model for extracting features from NIR spectral data with attention mechanism
    Based on model structure in NIR-4-1D CNN-train-optimized version.py
    """
    def __init__(self, input_size):
        super(NIRFeatureExtractor, self).__init__()
        
        # Further simplified convolutional block - extract spectral features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.4)  # Increase dropout rate
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.5)  # Increase dropout rate
        
        # Calculate flattened size
        self.flattened_size = (input_size // 4) * 32  # Size becomes 1/4 of original after two pooling operations
        
        # Fully connected layer for feature extraction
        self.fc1 = nn.Linear(self.flattened_size, 64)  # Reduce number of neurons
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.6)    # Increase dropout rate
        
    def forward(self, x):
        # First convolutional block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        
        return x


class MRIFeatureExtractor(nn.Module):
    """
    Simplified model for extracting features from MRI images
    """
    def __init__(self, pretrained_model_path=None):
        super(MRIFeatureExtractor, self).__init__()
        # Use a more simplified CNN as feature extractor
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce output size
        )
        
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten features
        x = self.classifier(x)
        return x


class FusionModel(nn.Module):
    """
    Feature fusion model supporting multiple fusion strategies
    """
    def __init__(self, nir_input_size, num_classes=4, fusion_type='concat', pretrained_mri_path=None):
        super(FusionModel, self).__init__()
        
        self.fusion_type = fusion_type
        self.nir_input_size = nir_input_size  # Add this line to save nir_input_size attribute
        self.num_classes = num_classes  # Also save num_classes attribute
        
        # NIR feature extractor
        self.nir_extractor = NIRFeatureExtractor(nir_input_size)
        
        # MRI feature extractor
        self.mri_extractor = MRIFeatureExtractor(pretrained_mri_path)
        
        # Calculate NIR feature dimension, use eval mode to avoid BatchNorm errors
        self.nir_extractor.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, nir_input_size)
            nir_features = self.nir_extractor(dummy_input)
            nir_feature_dim = nir_features.size(1)
        self.nir_extractor.train()
        
        # Define classifier based on fusion strategy
        if fusion_type == 'concat':
            # Concatenation fusion: NIR features + MRI features (128 dimensions)
            self.classifier = nn.Sequential(
                nn.Linear(nir_feature_dim + 128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(128, num_classes)
            )
        elif fusion_type == 'add':
            # Addition fusion: need to align feature dimensions first
            self.nir_projection = nn.Linear(nir_feature_dim, 128)
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(128, num_classes)
            )
        elif fusion_type == 'weighted':
            # Weighted fusion
            self.nir_projection = nn.Linear(nir_feature_dim, 128)
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize weights to 0.5
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.7),
                nn.Linear(128, num_classes)
            )
        elif fusion_type == 'bilinear':
            # Bilinear fusion
            self.bilinear_layer = nn.Bilinear(nir_feature_dim, 128, 64)
            self.classifier = nn.Sequential(
                nn.Linear(64, num_classes)
            )
            
    def forward(self, nir_data, mri_data):
        # Extract NIR features
        nir_features = self.nir_extractor(nir_data)
        
        # Extract MRI features
        mri_features = self.mri_extractor(mri_data)
        
        # Feature fusion
        if self.fusion_type == 'concat':
            # Concatenation fusion
            fused_features = torch.cat((nir_features, mri_features), dim=1)
        elif self.fusion_type == 'add':
            # Addition fusion
            nir_proj = self.nir_projection(nir_features)
            fused_features = nir_proj + mri_features
        elif self.fusion_type == 'weighted':
            # Weighted fusion
            nir_proj = self.nir_projection(nir_features)
            weights = F.softmax(self.fusion_weight, dim=0)
            fused_features = weights[0] * nir_proj + weights[1] * mri_features
        elif self.fusion_type == 'bilinear':
            # Bilinear fusion
            fused_features = self.bilinear_layer(nir_features, mri_features)
            
        # Classification
        output = self.classifier(fused_features)
        return output