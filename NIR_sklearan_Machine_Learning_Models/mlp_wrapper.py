import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import os

# 定义与训练时相同的MLP模型结构
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=(4,), num_classes=4, activation='relu'):
        super(MLPModel, self).__init__()
        self.activation = activation
        
        # 创建层列表
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation())
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        # 将所有层组合成一个序列
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'logistic':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def forward(self, x):
        return self.network(x)

# MLP包装器类，用于在sklearn中使用已训练的MLP模型
class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path='NIR_sklearan_Machine_Learning_Models\mlpc_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2, 3])  # 添加scikit-learn所需的classes_属性
        self.input_size = None
        self.scaler = None
        if os.path.exists(self.model_path):
            self._load_model()
    
    def _load_model(self):
        # 加载保存的模型和参数
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        # 获取模型参数
        self.input_size = data['input_size']
        self.coefs_ = data['coefs_']
        self.intercepts_ = data['intercepts_']
        self.classes_ = data['classes_']
        self.scaler = data['scaler']
        
        # 创建模型实例
        hidden_layer_sizes = tuple([coef.shape[1] for coef in self.coefs_[:-1]])
        self.model = MLPModel(self.input_size, hidden_layer_sizes, len(self.classes_)).to(self.device)
        
        # 将sklearn MLP的权重转换为PyTorch格式
        state_dict = {}
        for i, (coef, intercept) in enumerate(zip(self.coefs_, self.intercepts_)):
            if i == len(self.coefs_) - 1:  # 输出层
                state_dict[f'network.{i*2}.weight'] = torch.FloatTensor(coef.T)
                state_dict[f'network.{i*2}.bias'] = torch.FloatTensor(intercept)
            else:  # 隐藏层
                state_dict[f'network.{i*2}.weight'] = torch.FloatTensor(coef.T)
                state_dict[f'network.{i*2}.bias'] = torch.FloatTensor(intercept)
        
        # 加载权重
        self.model.load_state_dict(state_dict)
        
        # 设置为评估模式
        self.model.eval()
    
    def save_model(self, mlp_sklearn, input_size, scaler, classes_):
        """保存sklearn MLP模型为PyTorch格式"""
        self.input_size = input_size
        self.scaler = scaler
        self.classes_ = classes_
        
        # 提取sklearn MLP的权重
        self.coefs_ = mlp_sklearn.coefs_
        self.intercepts_ = mlp_sklearn.intercepts_
        
        # 保存到文件
        data = {
            'input_size': input_size,
            'coefs_': self.coefs_,
            'intercepts_': self.intercepts_,
            'classes_': classes_,
            'scaler': scaler
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        # 同时加载模型
        self._load_model()
    
    def predict(self, X):
        # 确保X是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 如果有scaler，则进行标准化
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
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
        
        # 如果有scaler，则进行标准化
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
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