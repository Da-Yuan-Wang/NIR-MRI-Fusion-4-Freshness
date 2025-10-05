import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms
import os


class FusionDataset(Dataset):
    """
    融合数据集类，同时处理NIR光谱数据和MRI图像数据
    """
    def __init__(self, nir_data_path, mri_data_dir, transform=None, mode='train', augment_times=1):
        """
        初始化数据集
        
        Args:
            nir_data_path: NIR数据CSV文件路径
            mri_data_dir: MRI图像数据目录路径
            transform: 图像变换
            mode: 数据模式 ('train' 或 'test')
            augment_times: 数据增强倍数
        """
        self.nir_data_path = nir_data_path
        self.mri_data_dir = mri_data_dir
        self.transform = transform
        self.mode = mode
        self.augment_times = augment_times if mode == 'train' else 1  # 只在训练模式下进行数据增强
        
        # 加载NIR数据
        self.nir_data = pd.read_csv(nir_data_path)
        
        # 提取样本ID、特征和标签
        self.sample_ids = self.nir_data.iloc[:, 0].values  # 第一列是SampleID
        self.nir_features = self.nir_data.iloc[:, 1:-1].values  # 中间的列是特征
        self.labels = self.nir_data.iloc[:, -1].values  # 最后一列是标签
        
        # 标准化NIR特征
        self.scaler = StandardScaler()
        self.nir_features = self.scaler.fit_transform(self.nir_features)
        
        # 图像预处理
        if self.transform is None:
            if self.mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 统一调整为224x224
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 统一调整为224x224
                    transforms.ToTensor(),
                ])
        
        # 如果是训练模式且需要数据增强，则扩展数据集
        if self.mode == 'train' and self.augment_times > 1:
            # 扩展数据集
            original_size = len(self.sample_ids)
            self.sample_ids = np.tile(self.sample_ids, self.augment_times)
            self.nir_features = np.tile(self.nir_features, (self.augment_times, 1))
            self.labels = np.tile(self.labels, self.augment_times)
            
            print(f"数据增强: 原始训练集样本数量 {original_size}, 增强后样本数量 {len(self.sample_ids)}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        # 获取样本ID
        sample_id = self.sample_ids[idx]
        
        # 获取NIR特征
        nir_feature = self.nir_features[idx]
        
        # 对NIR数据添加轻微噪声进行增强（仅在训练模式下）
        if self.mode == 'train':
            noise = np.random.normal(0, 0.01, nir_feature.shape)  # 添加高斯噪声
            nir_feature = nir_feature + noise
            
        nir_feature = torch.FloatTensor(nir_feature).unsqueeze(0)  # 增加通道维度 (1, features)
        
        # 获取标签
        label = self.labels[idx]
        label = torch.LongTensor([label]).squeeze()
        
        # 根据mode确定图像目录 (train或test)
        mri_mode_dir = 'train' if self.mode == 'train' else 'test'
        full_mri_dir = os.path.join(self.mri_data_dir, mri_mode_dir)
        
        # 构建图像路径
        class_names = ['Fresh', 'Slight-Shriveling', 'Moderate-Shriveling', 'Severe-Shriveling']
        
        # 尝试从标签中获取类别目录
        try:
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            class_dir = class_names[label_value]
            img_path = os.path.join(full_mri_dir, class_dir, f"{sample_id}.jpg")
        except IndexError:
            # 如果标签超出类别范围，使用默认目录
            img_path = os.path.join(full_mri_dir, 'Fresh', f"{sample_id}.jpg")
        
        # 如果按标准路径找不到，尝试在所有子目录中查找
        if not os.path.exists(img_path):
            for root, dirs, files in os.walk(full_mri_dir):
                if f"{sample_id}.jpg" in files:
                    img_path = os.path.join(root, f"{sample_id}.jpg")
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个空白图像作为替代
            image = Image.new('RGB', (224, 224))
        
        # 应用基础变换
        if self.transform:
            image = self.transform(image)
        
        # 在训练模式下进行MRI图像数据增强
        if self.mode == 'train':
            # 随机旋转（-15度到15度之间）
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
            
            # 添加轻微噪声
            if np.random.random() > 0.5:
                noise = torch.randn_like(image) * 0.01  # 添加小幅度高斯噪声
                image = image + noise
            
            # 确保像素值在合理范围内
            image = torch.clamp(image, 0, 1)
        
        # 标准化（在数据增强之后进行）
        if self.mode == 'train':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
        
        return nir_feature, image, label


def get_data_loader(nir_csv_file, mri_data_dir, batch_size=32, mode='train', shuffle=True, augment_times=1):
    """
    获取数据加载器
    
    Args:
        nir_csv_file (str): NIR光谱数据CSV文件路径
        mri_data_dir (str): MRI图像数据目录路径
        batch_size (int): 批次大小
        mode (str): 数据模式 ('train' 或 'test')
        shuffle (bool): 是否打乱数据
        augment_times (int): 数据增强倍数
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = FusionDataset(nir_csv_file, mri_data_dir, mode=mode, augment_times=augment_times)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader


def get_data_loaders(nir_train_path, nir_test_path, mri_data_dir, batch_size=32, augment_times=4):
    """
    获取训练和测试数据加载器
    
    Args:
        nir_train_path (str): 训练集NIR光谱数据路径
        nir_test_path (str): 测试集NIR光谱数据路径
        mri_data_dir (str): MRI图像数据根目录
        batch_size (int): 批次大小
        augment_times (int): 数据增强倍数
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = get_data_loader(nir_train_path, mri_data_dir, batch_size, mode='train', shuffle=True, augment_times=augment_times)
    test_loader = get_data_loader(nir_test_path, mri_data_dir, batch_size, mode='test', shuffle=False, augment_times=1)
    return train_loader, test_loader


def load_pretrained_extractors(nir_model_path, mri_model_path, device):
    """
    加载预训练的特征提取器
    
    Args:
        nir_model_path: NIR模型路径
        mri_model_path: MRI模型路径
        device: 设备
        
    Returns:
        nir_extractor, mri_extractor: NIR和MRI特征提取器
    """
    # 导入模型
    from Featurefusionmodel.models.fusion_model import NIRFeatureExtractor, MRIFeatureExtractor
    
    # 加载NIR模型检查点
    nir_checkpoint = torch.load(nir_model_path, map_location=device)
    input_size = nir_checkpoint['input_size']
    
    # 创建NIR特征提取器
    nir_extractor = NIRFeatureExtractor(input_size).to(device)
    
    # 加载NIR模型权重
    # 从完整模型中提取特征提取器部分的权重
    nir_state_dict = {}
    for key, value in nir_checkpoint['model_state_dict'].items():
        if key.startswith('conv') or key.startswith('bn') or key.startswith('fc'):
            # 移除分类器部分的权重
            if not key.startswith(('fc4')):
                nir_state_dict[key] = value
    
    nir_extractor.load_state_dict(nir_state_dict, strict=False)
    
    # 创建MRI特征提取器
    mri_extractor = MRIFeatureExtractor(mri_model_path).to(device)
    
    return nir_extractor, mri_extractor