import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import os

# Define the same MLP model structure as during training
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=(4,), num_classes=4, activation='relu'):
        super(MLPModel, self).__init__()
        self.activation = activation
        
        # Create layer list
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation())
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers into a sequence
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

# MLP wrapper class for using trained MLP models in sklearn
class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path='mlp_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2, 3])  # Add classes_ attribute required by scikit-learn
        self.input_size = None
        self.scaler = None
        if os.path.exists(self.model_path):
            self._load_model()
    
    def _load_model(self):
        # Load saved model and parameters
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get model parameters
        self.input_size = data['input_size']
        self.coefs_ = data['coefs_']
        self.intercepts_ = data['intercepts_']
        self.classes_ = data['classes_']
        self.scaler = data['scaler']
        
        # Create model instance
        hidden_layer_sizes = tuple([coef.shape[1] for coef in self.coefs_[:-1]])
        self.model = MLPModel(self.input_size, hidden_layer_sizes, len(self.classes_)).to(self.device)
        
        # Convert sklearn MLP weights to PyTorch format
        state_dict = {}
        for i, (coef, intercept) in enumerate(zip(self.coefs_, self.intercepts_)):
            if i == len(self.coefs_) - 1:  # Output layer
                state_dict[f'network.{i*2}.weight'] = torch.FloatTensor(coef.T)
                state_dict[f'network.{i*2}.bias'] = torch.FloatTensor(intercept)
            else:  # Hidden layers
                state_dict[f'network.{i*2}.weight'] = torch.FloatTensor(coef.T)
                state_dict[f'network.{i*2}.bias'] = torch.FloatTensor(intercept)
        
        # Load weights
        self.model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.model.eval()
    
    def save_model(self, mlp_sklearn, input_size, scaler, classes_):
        """Save sklearn MLP model in PyTorch format"""
        self.input_size = input_size
        self.scaler = scaler
        self.classes_ = classes_
        
        # Extract weights from sklearn MLP
        self.coefs_ = mlp_sklearn.coefs_
        self.intercepts_ = mlp_sklearn.intercepts_
        
        # Save to file
        data = {
            'input_size': input_size,
            'coefs_': self.coefs_,
            'intercepts_': self.intercepts_,
            'classes_': classes_,
            'scaler': scaler
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Also load the model
        self._load_model()
    
    def predict(self, X):
        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # If scaler exists, perform standardization
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Return prediction results
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        # Ensure X is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # If scaler exists, perform standardization
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Return probabilities
        return probabilities.cpu().numpy()
    
    def fit(self, X, y):
        # This is a pre-trained model that does not require training
        # But to be compatible with sklearn interface, this method needs to be implemented
        return self