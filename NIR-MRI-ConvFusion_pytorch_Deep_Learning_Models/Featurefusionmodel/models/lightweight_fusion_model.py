import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightNIRFeatureExtractor(nn.Module):
    """
    Lightweight NIR spectral feature extractor
    Uses fewer parameters to extract effective features
    """
    def __init__(self, input_size):
        super(LightweightNIRFeatureExtractor, self).__init__()
        
        # Simplified convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        self.flattened_size = (input_size // 8) * 64  # After three pooling operations, size becomes 1/8 of original
        
        # Feature compression
        self.fc = nn.Linear(self.flattened_size, 64)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and compress features
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return x


class LightweightMRIFeatureExtractor(nn.Module):
    """
    Lightweight MRI image feature extractor
    Uses pre-trained MobileNetV2 as feature extractor with fewer parameters
    """
    def __init__(self, pretrained_model_path=None):
        super(LightweightMRIFeatureExtractor, self).__init__()
        import torchvision.models as models
        
        # Use MobileNetV2, which has significantly fewer parameters than ResNet50
        self.mobilenet = models.mobilenet_v2(pretrained=True if pretrained_model_path is None else False)
        
        # Remove the final classification layer, keep only the feature extraction part
        self.features = self.mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Compress features to smaller dimensions
        self.compress = nn.Linear(1280, 256)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.compress(x)
        return x


class LightweightFusionModel(nn.Module):
    """
    Lightweight feature fusion model supporting multiple fusion strategies
    Has fewer parameters and smaller file size compared to the original model
    """
    def __init__(self, nir_input_size, num_classes=4, fusion_type='concat', pretrained_mri_path=None):
        super(LightweightFusionModel, self).__init__()
        
        self.fusion_type = fusion_type
        
        # NIR feature extractor
        self.nir_extractor = LightweightNIRFeatureExtractor(nir_input_size)
        
        # MRI feature extractor
        self.mri_extractor = LightweightMRIFeatureExtractor(pretrained_mri_path)
        
        # Calculate NIR feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, nir_input_size)
            nir_features = self.nir_extractor(dummy_input)
            nir_feature_dim = nir_features.size(1)
            
            dummy_mri = torch.zeros(1, 3, 224, 224)
            mri_features = self.mri_extractor(dummy_mri)
            mri_feature_dim = mri_features.size(1)
        
        # Define classifier based on fusion strategy
        if fusion_type == 'concat':
            # Concatenation fusion
            self.classifier = nn.Sequential(
                nn.Linear(nir_feature_dim + mri_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        elif fusion_type == 'add':
            # Addition fusion: need to align feature dimensions first
            self.nir_projection = nn.Linear(nir_feature_dim, mri_feature_dim)
            self.classifier = nn.Sequential(
                nn.Linear(mri_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        elif fusion_type == 'weighted':
            # Weighted fusion
            self.nir_projection = nn.Linear(nir_feature_dim, mri_feature_dim)
            self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize weights to 0.5
            self.classifier = nn.Sequential(
                nn.Linear(mri_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        elif fusion_type == 'bilinear':
            # Bilinear fusion
            self.bilinear_layer = nn.Bilinear(nir_feature_dim, mri_feature_dim, 64)
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes)
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