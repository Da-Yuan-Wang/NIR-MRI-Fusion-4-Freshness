import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SVCWrapper(BaseEstimator, ClassifierMixin):
    """
    SVC模型包装器，用于加载预训练的SVC模型并在StackingClassifier中使用
    """
    
    def __init__(self, model_path='NIR_sklearan_Machine_Learning_Models\svc_model.pkl'):
        """
        初始化SVC包装器
        
        Parameters:
        model_path (str): 预训练SVC模型的路径
        """
        self.model_path = model_path
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        加载预训练的SVC模型
        注意：由于是预训练模型，X和y参数仅用于兼容sklearn接口，实际不会用于训练
        
        Parameters:
        X : array-like, shape = [n_samples, n_features]
            训练样本
        y : array-like, shape = [n_samples]
            目标值
        """
        self.model = joblib.load(self.model_path)
        # 从加载的模型中获取classes_属性
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X):
        """
        对样本进行预测
        
        Parameters:
        X : array-like, shape = [n_samples, n_features]
            测试样本
            
        Returns:
        y_pred : array, shape = [n_samples]
            预测的类别标签
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用fit方法")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测样本属于各个类别的概率
        
        Parameters:
        X : array-like, shape = [n_samples, n_features]
            测试样本
            
        Returns:
        y_proba : array, shape = [n_samples, n_classes]
            预测的概率
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用fit方法")
        return self.model.predict_proba(X)