# NIR-MRI Feature-level Fusion Classification Model

This project implements a feature-level fusion classification model based on NIR spectral data and MRI image data for fruit and vegetable quality detection.

## Project Overview

This project uses deep learning methods to extract features from NIR spectral data and MRI image data, and employs multiple fusion strategies to fuse features from the two modalities, ultimately achieving a four-classification task:

## Model Architecture

### NIR Feature Extractor

Based on 1D-CNN architecture, containing 4 convolutional blocks:
1. First convolutional block: 32 filters, kernel size 9
2. Second convolutional block: 64 filters, kernel size 7
3. Third convolutional block: 128 filters, kernel size 5
4. Fourth convolutional block: 256 filters, kernel size 3

Each block is followed by batch normalization, activation function, and pooling layer, ultimately outputting a 128-dimensional feature vector.

### MRI Feature Extractor

Based on ResNet50 architecture, using pre-trained weights, removing the final fully connected layer, and outputting a 2048-dimensional feature vector.

### Feature Fusion Strategies

Supports multiple feature fusion strategies:

1. **Concatenation fusion (concat)**: Directly concatenating NIR and MRI features (128 + 2048 = 2176 dimensions)
2. **Additive fusion (add)**: Projecting NIR features to 2048 dimensions and adding to MRI features
3. **Weighted fusion (weighted)**: Learnable weight feature fusion
4. **Bilinear fusion (bilinear)**: Using bilinear pooling for fusion

## Project Structure

```
fusion_model/
├── models/
│   └── fusion_model.py      # Model definition
├── utils/
│   └── data_utils.py        # Data processing tools
├── train_fusion.py          # Model training script
├── evaluate_fusion.py       # Model evaluation script
└── README.md                # Project documentation
```

## Usage

### Training the Model

```bash
# Training the model using concatenation fusion strategy
python train_fusion.py --fusion_type concat --batch_size 32 --num_epochs 100 --lr 0.001

# Training the model using weighted fusion strategy
python train_fusion.py --fusion_type weighted --batch_size 32 --num_epochs 100 --lr 0.001

# Training the model using bilinear fusion strategy
python train_fusion.py --fusion_type bilinear --batch_size 32 --num_epochs 100 --lr 0.001
```

### Evaluating the Model

```bash
# Evaluating the concatenation fusion model
python evaluate_fusion.py --model_path checkpoints/best_fusion_model_concat.pth --fusion_type concat

# Evaluating the weighted fusion model
python evaluate_fusion.py --model_path checkpoints/best_fusion_model_weighted.pth --fusion_type weighted
```

## Parameter Description

### Training Parameters

- `--fusion_type`: Feature fusion type (concat, add, weighted, bilinear)
- `--batch_size`: Batch size
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--seed`: Random seed
- `--weight_decay`: Weight decay
- `--pretrained_mri_path`: Pre-trained MRI model path

### Evaluation Parameters

- `--model_path`: Model file path
- `--fusion_type`: Feature fusion type
- `--batch_size`: Batch size
- `--pretrained_mri_path`: Pre-trained MRI model path

## Output Results

The training and evaluation process will generate the following outputs:

1. **Model files**: Saved in the `checkpoints/` directory
2. **Training history**: Including loss and accuracy curves, saved in CSV and PNG formats
3. **Evaluation results**: 
   - Confusion matrix image
   - Classification report image
   - Prediction results CSV file

## Performance Metrics

Based on test results from individual modalities:
- NIR spectral data classification accuracy: 97.92%
- MRI image data classification accuracy: 94.79%

The fusion model is expected to achieve better classification performance.