import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from models.fusion_model import FusionModel
from utils.data_utils import get_data_loaders
import argparse
import math


def set_seed(seed=42):
    """Set random seed to ensure experiment reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path, patience=30, args=None):
    """
    Train fusion model
    
    Args:
        model: Fusion model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device
        save_path: Model save path
        patience: Early stopping patience
        args: Training arguments
        
    Returns:
        history: Training history records
    """
    # Record training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        for batch_idx, (nir_data, mri_data, labels) in enumerate(train_loader):
            nir_data = nir_data.to(device)
            mri_data = mri_data.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(nir_data, mri_data)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Collect training statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_corrects += (predicted == labels).sum().item()
            
            # Debugging information, only shown in the first batch
            if epoch == 0 and batch_idx == 0:
                print(f"第一轮训练 - Batch损失: {loss.item():.4f}")
                print(f"输出形状: {outputs.shape}")
                print(f"标签形状: {labels.shape}")
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader)
        train_acc = train_corrects / train_total
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        test_total = 0
        
        with torch.no_grad():
            for nir_data, mri_data, labels in test_loader:
                nir_data = nir_data.to(device)
                mri_data = mri_data.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(nir_data, mri_data)
                loss = criterion(outputs, labels)
                
                # Collect testing statistics
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_corrects += (predicted == labels).sum().item()
        
        # Calculate average testing loss and accuracy
        test_loss = test_loss / len(test_loader)
        test_acc = test_corrects / test_total
        
        # Learning rate scheduling - based on validation loss
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_loss)
        elif scheduler:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Early stopping mechanism
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Dynamically generate filename containing epoch and accuracy
            model_save_path = os.path.join(
                os.path.dirname(save_path),
                f'best_fusion_model_{model.fusion_type}_epoch_{epoch+1}_acc_{test_acc:.4f}.pth'
            )
            
            # Save more complete model information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'fusion_type': model.fusion_type,
                'model_architecture': {
                    'nir_input_size': model.nir_input_size,
                    'num_classes': model.num_classes
                },
                'training_config': {
                    'fusion_type': args.fusion_type if args else None,
                    'batch_size': args.batch_size if args else None,
                    'num_epochs': num_epochs,
                    'learning_rate': args.lr if args else None,
                    'min_lr': args.min_lr if args else None,
                    'lr_decay_type': args.lr_decay_type if args else None,
                    'weight_decay': args.weight_decay if args else None,
                    'optimizer_type': args.optimizer_type if args else None,
                    'momentum': args.momentum if args else None,
                    'patience': patience
                },
                'training_history': {
                    'train_loss': history['train_loss'].copy(),
                    'train_acc': history['train_acc'].copy(),
                    'test_loss': history['test_loss'].copy(),
                    'test_acc': history['test_acc'].copy()
                },
                'final_metrics': {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'current_lr': current_lr
                }
            }
            
            # Add scheduler-specific parameters
            if args:
                if args.lr_decay_type == 'step':
                    checkpoint['training_config']['step_size'] = args.step_size
                    checkpoint['training_config']['gamma'] = args.gamma
                elif args.lr_decay_type == 'cos':
                    checkpoint['training_config']['T_max'] = args.num_epochs
                
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1
            
        # Print information every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                  f'LR: {current_lr:.6f}')
        
        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"早停机制触发，最佳Epoch: {best_epoch+1}，最佳准确率: {best_acc:.4f}")
            break
    
    if epochs_no_improve < patience:
        print(f'训练完成，最佳测试准确率: {best_acc:.4f} (Epoch: {best_epoch+1})')
    return history


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        class_names: List of class names
    """
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
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    print(f'测试准确率: {accuracy:.4f}')
    print('混淆矩阵:')
    print(conf_matrix)
    print('分类报告:')
    print(class_report)
    
    return accuracy, conf_matrix, class_report


def plot_training_history(history, save_path=None):
    """
    Plot training history curves
    
    Args:
        history: Training history records
        save_path: Image save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss curve
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy curve
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix_and_f1(conf_matrix, class_names, save_path=None):
    """
    Plot confusion matrix and F1-score charts
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Image save path
    """
    # Create new figure for confusion matrix and F1-score
    plt.figure(figsize=(12, 6))
    
    # Handle line breaks in class names
    class_names_multiline = [name.replace('-', '\n') for name in class_names]
    
    # Plot confusion matrix
    plt.subplot(121)
    im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of Fusion Model", fontsize=14)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names_multiline, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names_multiline, fontsize=12)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    
    # Adjust subplot spacing to prevent labels from being covered
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Add numerical values to confusion matrix
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
    plt.bar(x, f1_scores, color='blue')
    plt.xticks(x, class_names_multiline, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score by Class', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='训练NIR-MRI融合分类模型')
    parser.add_argument('--fusion_type', type=str, default='weighted', 
                        choices=['concat', 'add', 'weighted', 'bilinear'],
                        help='特征融合类型: concat(连接), add(加法), weighted(加权), bilinear(双线性)')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')  # 增加批大小
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率 (Init_lr)')  # 进一步降低学习率
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率 (Min_lr)')
    parser.add_argument('--lr_decay_type', type=str, default='step', 
                        choices=['step', 'cos', 'reduce'], help='学习率下降方式: step, cos或reduce')
    parser.add_argument('--step_size', type=int, default=50, help='学习率下降步长')
    parser.add_argument('--gamma', type=float, default=0.7, help='学习率下降比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')  # 增加L2正则化
    parser.add_argument('--optimizer_type', type=str, default='adam', 
                        choices=['sgd', 'adam'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='优化器动量参数')
    parser.add_argument('--patience', type=int, default=900, help='早停耐心值')
    parser.add_argument('--augment_times', type=int, default=4, help='数据增强倍数')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # Data paths
    data_root = '../data'
    nir_train_path = os.path.join(data_root, 'nir_train.csv')
    nir_test_path = os.path.join(data_root, 'nir_test.csv')
    mri_data_dir = data_root
    
    # Class names
    class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        nir_train_path=nir_train_path,
        nir_test_path=nir_test_path,
        mri_data_dir=mri_data_dir,
        batch_size=args.batch_size,  # 增加批大小以获得更稳定的梯度估计
        augment_times=args.augment_times  # 添加数据增强倍数参数
    )
    
    # Create fusion model
    # NIR feature count determined from training data
    sample_nir_data = pd.read_csv(nir_train_path, nrows=1)
    nir_feature_count = len(sample_nir_data.columns) - 2  # Subtract SampleID and Label columns
    
    model = FusionModel(
        nir_input_size=nir_feature_count,
        num_classes=4,
        fusion_type=args.fusion_type
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.lr_decay_type == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_decay_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    else:  # reduce
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=args.min_lr)
    
    # Model save path
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    # Use temporary filename first, rename to include accuracy after training
    model_save_path = os.path.join(save_dir, f'best_fusion_model_{args.fusion_type}.pth')
    
    # Print training configuration
    print(f"训练配置:")
    print(f"- 融合类型: {args.fusion_type}")
    print(f"- 批大小: {args.batch_size}")
    print(f"- 训练轮数: {args.num_epochs}")
    print(f"- 学习率: {args.lr}")
    print(f"- 权重衰减: {args.weight_decay}")
    print(f"- 优化器: {args.optimizer_type}")
    print(f"- 学习率调度: {args.lr_decay_type}")
    print(f"- 早停耐心值: {args.patience}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_path=model_save_path,
        patience=args.patience,
        args=args
    )
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Do not load scheduler state
    
    # Get epoch and accuracy information of the best model
    best_epoch = checkpoint.get('epoch', 0)
    best_test_acc = checkpoint.get('best_acc', 0.0)
    final_train_acc = checkpoint['final_metrics'].get('train_acc', 0.0) if 'final_metrics' in checkpoint else 0.0
    
    # Rename model file to include epoch and accuracy information
    new_model_save_path = os.path.join(
        save_dir, 
        f'best_fusion_model_{args.fusion_type}-epoch{best_epoch+1}-trainacc{final_train_acc:.4f}-testacc{best_test_acc:.4f}.pth'
    )
    
    # Rename file
    os.rename(model_save_path, new_model_save_path)
    
    # Update model save path
    model_save_path = new_model_save_path
    
    # Evaluate model
    print("\n评估最佳模型:")
    accuracy, conf_matrix, class_report = evaluate_model(model, test_loader, device, class_names)
    
    # Plot training history
    plot_save_path = os.path.join(save_dir, f'training_history_{args.fusion_type}.png')
    plot_training_history(history, plot_save_path)
    
    # Plot confusion matrix and F1-score charts
    cm_f1_save_path = os.path.join(save_dir, f'confusion_matrix_and_f1_{args.fusion_type}.png')
    plot_confusion_matrix_and_f1(conf_matrix, class_names, cm_f1_save_path)
    
    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(save_dir, f'training_history_{args.fusion_type}.csv')
    history_df.to_csv(history_csv_path, index=False)
    
    print(f"\n训练完成!")
    print(f"最佳模型已保存到: {model_save_path}")
    print(f"训练历史图表已保存到: {plot_save_path}")
    print(f"混淆矩阵和F1-score图表已保存到: {cm_f1_save_path}")
    print(f"训练历史数据已保存到: {history_csv_path}")


if __name__ == '__main__':
    main()