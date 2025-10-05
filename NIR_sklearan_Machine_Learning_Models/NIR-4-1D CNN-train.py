# 引入必要的库
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import openpyxl
from sklearn.preprocessing import StandardScaler
import random

# 设置随机种子以确保实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
seed = 42
set_seed(seed)
print(f"使用的随机种子: {seed}")

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 打印PyTorch和CUDA版本信息
print(f"PyTorch版本: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name()}")

# 读取训练数据
train_data = pd.read_csv('data/nir_train.csv')
X_train = train_data.iloc[:, 1:-1]  # 排除SampleID和Label列，只保留特征数据
y_train = train_data.iloc[:, -1]     # Label列作为目标变量

# 读取测试数据
test_data = pd.read_csv('data/nir_test.csv')
X_test = test_data.iloc[:, 1:-1]     # 排除SampleID和Label列，只保留特征数据
y_test = test_data.iloc[:, -1]       # Label列作为目标变量

# 特征名称（去除SampleID列）
feature_names = list(train_data.columns[1:-1])

# 类别名称

target_names = ['Class-S-Freshness', 'Class-A-Freshness', 'Class-B-Freshness', 'Class-C-Freshness']

print(f"→ Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"→ Number of features: {X_train.shape[1]}, Number of classes: {len(np.unique(y_train))}")

# 特征缩放（标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 重塑数据以适应1D-CNN (samples, features)
# 对于光谱数据，每个样本是一个包含多个波长点的一维序列
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # 增加通道维度
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# 将数据移动到指定设备（CUDA或CPU）
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch size
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 优化的1D-CNN模型，针对NIR光谱数据特点设计
class OptimizedCNN1D(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(OptimizedCNN1D, self).__init__()
        
        # 第一个卷积块 - 提取局部光谱特征
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, padding=4)
        self.bn1_1 = nn.BatchNorm1d(32)
        self.conv1_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, padding=4)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.1)
        
        # 第二个卷积块 - 提取中级光谱特征
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # 第三个卷积块 - 提取高级光谱特征
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn3_2 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)
        
        # 第四个卷积块 - 提取语义级光谱特征
        self.conv4_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm1d(256)
        self.conv4_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(0.4)
        
        # 计算展平后的尺寸
        self.flattened_size = (input_size // 16) * 256  # 经过四次池化后尺寸变为原来的1/16
        
        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.dropout7 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 第一个卷积块
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 第四个卷积块
        x = torch.relu(self.bn4_1(self.conv4_1(x)))
        x = torch.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout6(x)
        
        x = torch.relu(self.fc3(x))
        x = self.dropout7(x)
        
        x = self.fc4(x)
        return x

# 创建模型实例并移动到指定设备
num_features = X_train.shape[1]
num_classes = 4  # 四分类任务
model = OptimizedCNN1D(num_features, num_classes).to(device)

# 打印模型参数信息
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)  # 使用AdamW优化器和更大的权重衰减
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)  # 余弦退火学习率调度

# 训练模型
num_epochs = 600 # 调整训练轮数
train_losses = []
test_accuracies = []
test_losses = []
train_accuracies = []

best_test_accuracy = 0.0
best_epoch = 0

print("开始训练...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 测试阶段
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    
    # 学习率调度
    scheduler.step()
    
    # 保存最佳模型
    if accuracy > best_test_accuracy:
        best_test_accuracy = accuracy
        best_epoch = epoch
        # 保存最佳模型，包含输入大小信息
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'input_size': num_features,  # 添加输入大小信息
            'num_classes': num_classes,   # 添加类别数信息
            'seed': seed                  # 添加随机种子信息
        }, 'NIR_sklearan_Machine_Learning_Models/best_NIR-1D_CNN_model_optimized.pth')
    
    # 每个epoch都打印训练信息
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}')

print(f"最佳测试准确率: {best_test_accuracy:.4f} at epoch {best_epoch+1}")

# 加载最佳模型
checkpoint = torch.load('NIR_sklearan_Machine_Learning_Models/best_NIR-1D_CNN_model_optimized.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# 打印模型信息
print(f"加载的最佳模型信息:")
print(f"  Epoch: {checkpoint['epoch']+1}")
print(f"  训练集损失: {checkpoint['train_loss']:.4f}")
print(f"  测试集损失: {checkpoint['test_loss']:.4f}")
print(f"  测试集准确率: {checkpoint['test_accuracy']:.4f}")

# 进行预测
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在正确的设备上
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())  # 将结果移回CPU进行处理
        all_labels.extend(labels.cpu().numpy())

# 评估模型性能
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=target_names)

# 计算特征重要性（基于梯度的方法）
def compute_feature_importance(model, X_test_tensor, device):
    model.eval()
    # 只使用一部分数据计算特征重要性以提高效率
    sample_size = min(100, X_test_tensor.size(0))
    sample_indices = torch.randperm(X_test_tensor.size(0))[:sample_size]
    X_sample = X_test_tensor[sample_indices].clone().requires_grad_(True)
    
    # 前向传播
    outputs = model(X_sample)
    # 计算所有类别的平均概率作为损失
    probs = torch.softmax(outputs, dim=1)
    loss = probs.mean()
    
    # 反向传播计算梯度
    loss.backward()
    
    # 计算特征重要性（梯度的绝对值）
    feature_importance = X_sample.grad.abs().mean(dim=0).squeeze().detach().cpu().numpy()
    return feature_importance

# 计算特征重要性
feature_importance = compute_feature_importance(model, X_test_tensor, device)

# 将训练过程数据保存到Excel
history_df = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Train Loss': train_losses,
    'Train Accuracy': train_accuracies,
    'Test Loss': test_losses,
    'Test Accuracy': test_accuracies
})

# 保存到当前目录
history_df.to_excel('NIR_sklearan_Machine_Learning_Models/training-NIR_1D_CNN_history_optimized.xlsx', index=False)
print("训练历史已保存到 NIR_sklearan_Machine_Learning_Models/training-NIR_1D_CNN_history_optimized.xlsx")

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制训练历史
plt.subplot(121)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 创建新图形用于混淆矩阵和F1-score
plt.figure(figsize=(18, 6))

# 绘制混淆矩阵
plt.subplot(121)
im = plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("1D-CNN (NIR)", fontsize=18)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=16)
tick_marks = np.arange(len(target_names))
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S', 'Class-A', 'Class-B', 'Class-C']

plt.xticks(tick_marks, target_names_multiline, rotation=45, fontsize=16)
plt.yticks(tick_marks, target_names_multiline, fontsize=16)
plt.xlabel("Predicted Freshness Grades", fontsize=16)
plt.ylabel("True Freshness Grades", fontsize=16)

# 调整子图间距，防止标签被覆盖
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 在混淆矩阵中添加数值
thresh = conf_matrix.max() / 2.
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                fontsize=16,
                color="white" if conf_matrix[i, j] > thresh else "black")

# 计算F1-score并绘制柱状图
plt.subplot(122)
# 计算每个类别的F1-score
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
plt.bar(x, f1_scores, color='blue')
plt.xticks(x, target_names_multiline, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Class', fontsize=16)
plt.ylabel('F1-Score', fontsize=16)
plt.title('F1-Score by Class', fontsize=18)
plt.tight_layout()
plt.show()

# 单独绘制特征重要性图（前15个重要特征）
plt.figure(figsize=(12, 6))
# 计算特征重要性
feature_importance = compute_feature_importance(model, X_test_tensor, device)
# 获取前15个最重要的特征，按重要性从高到低排序
top_features = np.argsort(feature_importance)[-15:][::-1]
top_importance = feature_importance[top_features]
top_feature_names = [feature_names[i] for i in top_features]

# 竖向柱状图
plt.bar(range(len(top_importance)), top_importance, color='skyblue')
plt.xticks(range(len(top_feature_names)), top_feature_names, rotation=45, ha='right', fontsize=12)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Feature Importance', fontsize=16)
plt.title('Top 15 Feature Importances (1D CNN)', fontsize=16)
plt.tight_layout()
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

# 绘制训练历史
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
