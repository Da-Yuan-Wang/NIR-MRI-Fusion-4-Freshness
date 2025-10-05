# 引入必要的库
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import openpyxl
from sklearn.preprocessing import StandardScaler

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

# 加载已训练好的模型
print("加载已训练好的模型...")
checkpoint = torch.load(r'NIR_sklearan_Machine_Learning_Models\best_NIR-1D_CNN_model_optimized.pth', map_location=device)
num_features = checkpoint['input_size']
num_classes = checkpoint['num_classes']
seed = checkpoint['seed']

print(f"模型信息:")
print(f"  随机种子: {seed}")
print(f"  输入大小: {num_features}")
print(f"  类别数: {num_classes}")
print(f"  验证准确率: {checkpoint['val_accuracy']:.4f}")
print(f"  训练损失: {checkpoint['train_loss']:.4f}")
print(f"  验证损失: {checkpoint['val_loss']:.4f}")
print(f"  训练轮数: {checkpoint['epoch']}")

# 创建模型实例并移动到指定设备
model = OptimizedCNN1D(num_features, num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"模型加载成功!")

# 进行预测
model.eval()
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在正确的设备上
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())  # 将结果移回CPU进行处理
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# 评估模型性能
val_accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=target_names)

# 绘制图表
plt.figure(figsize=(12, 6))

# 绘制混淆矩阵
plt.subplot(121)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix of 1D-CNN")
plt.colorbar()
tick_marks = np.arange(len(target_names))
# target_names_multiline = ['Fresh', 'Slight\nShriveling', 'Moderate\nShriveling', 'Severe\nShriveling']
target_names_multiline = ['Class-S\nFreshness', 'Class-A\nFreshness', 'Class-B\nFreshness', 'Class-C\nFreshness']

plt.xticks(tick_marks, target_names_multiline, rotation=45)
plt.yticks(tick_marks, target_names_multiline)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# 在混淆矩阵中添加数值
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

# 绘制F1-score by Class
plt.subplot(122)
x = range(len(target_names))
f1_scores = []
for i in range(len(target_names)):
    # 计算每个类别的F1分数在classification report中的位置
    f1_scores.append(float(class_report.split()[5*(i+1)]))

plt.bar(x, f1_scores)
plt.xticks(x, target_names_multiline)
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('F1-Score by Class')
plt.ylim(0, 1)

# 调整子图间距
plt.tight_layout()

print(f"Accuracy: {val_accuracy:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print("Classification Report:\n", class_report)

plt.show()