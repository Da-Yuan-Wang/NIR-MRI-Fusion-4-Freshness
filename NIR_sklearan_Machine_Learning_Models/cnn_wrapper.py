import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# 定义与训练时相同的1D-CNN模型结构
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

# CNN包装器类，用于在sklearn中使用PyTorch模型
class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path='NIR_sklearan_Machine_Learning_Models/best_NIR-1D_CNN_model_optimized.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2, 3])  # 添加scikit-learn所需的classes_属性
        self._load_model()
    
    def _load_model(self):
        # 加载检查点
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            # 兼容旧版本PyTorch没有weights_only参数
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型参数
        input_size = checkpoint.get('input_size', 2048)  # 默认值2048，根据实际情况调整
        num_classes = checkpoint.get('num_classes', 4)
        
        # 创建模型实例
        self.model = OptimizedCNN1D(input_size, num_classes).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 设置为评估模式
        self.model.eval()
    
    def predict(self, X):
        # 确保X是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 转换为tensor并增加通道维度
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # 返回预测结果
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        # 确保X是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 转换为tensor并增加通道维度
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # 返回概率
        return probabilities.cpu().numpy()
    
    def fit(self, X, y):
        # 这是一个预训练模型，不需要训练
        # 但为了兼容sklearn接口，需要实现这个方法
        return self