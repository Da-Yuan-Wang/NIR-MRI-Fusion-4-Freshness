# NIR-MRI 融合网络结构详解

## 1. 整体架构概述

本模型采用双流网络结构，分别处理NIR光谱数据和MRI图像数据，然后通过多种融合策略将两种模态的特征进行融合，最终进行分类。

```
NIR数据 --> NIR特征提取器 --> \
                                --> 融合模块 --> 分类器 --> 输出
MRI数据 --> MRI特征提取器 --> /
```

## 2. NIR特征提取网络 (NIRFeatureExtractor)

NIR光谱数据通过一个1D-CNN网络提取特征，网络结构如下：

### 网络结构
```
Input: (batch_size, 1, nir_input_size)
    |
    | Conv1: Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
    |        -> BatchNorm1d(16)
    |        -> ReLU
    |        -> MaxPool1d(kernel_size=2)
    |        -> Dropout(0.4)
    |
    | Conv2: Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
    |        -> BatchNorm1d(32)
    |        -> ReLU
    |        -> MaxPool1d(kernel_size=2)
    |        -> Dropout(0.5)
    |
    | Flatten: (batch_size, flattened_size)
    |          where flattened_size = (nir_input_size // 4) * 32
    |
    | FC1: Linear(flattened_size, 64)
    |      -> BatchNorm1d(64)
    |      -> ReLU
    |      -> Dropout(0.6)
    |
Output: (batch_size, 64)
```

### 特点
- 使用1D卷积处理一维光谱数据
- 两层卷积层，逐步提取高级光谱特征
- 每层后接BatchNorm和Dropout，防止过拟合
- 最终输出64维特征向量

## 3. MRI特征提取网络 (MRIFeatureExtractor)

MRI图像通过一个简化CNN网络提取特征，网络结构如下：

### 网络结构
```
Input: (batch_size, 3, H, W)  # H, W为图像高宽
    |
    | Conv Block 1:
    | Conv2d(3, 16, kernel_size=3, padding=1)
    | -> BatchNorm2d(16)
    | -> ReLU
    | -> MaxPool2d(kernel_size=2, stride=2)
    |
    | Conv Block 2:
    | Conv2d(16, 32, kernel_size=3, padding=1)
    | -> BatchNorm2d(32)
    | -> ReLU
    | -> MaxPool2d(kernel_size=2, stride=2)
    |
    | Conv Block 3:
    | Conv2d(32, 64, kernel_size=3, padding=1)
    | -> BatchNorm2d(64)
    | -> ReLU
    | -> AdaptiveAvgPool2d((4, 4))
    |
    | Flatten: (batch_size, 64*4*4) = (batch_size, 1024)
    |
    | FC Block:
    | Linear(1024, 128)
    | -> BatchNorm1d(128)
    | -> ReLU
    | -> Dropout(0.7)
    |
Output: (batch_size, 128)
```

### 特点
- 使用2D卷积处理二维图像数据
- 三层卷积层，逐步提取图像特征
- 使用AdaptiveAvgPool2d固定输出尺寸
- 最终输出128维特征向量

## 4. 特征融合模块 (FusionModel)

支持四种不同的融合策略：

### 4.1 连接融合 (Concatenation Fusion)
```
NIR特征 (64-dim) --\
                    --> Concat --> (192-dim)
MRI特征 (128-dim) -/
    |
    | Linear(192, 128)
    | -> BatchNorm1d(128)
    | -> ReLU
    | -> Dropout(0.7)
    | -> Linear(128, num_classes)
    |
Output: (batch_size, num_classes)
```

### 4.2 加法融合 (Addition Fusion)
```
NIR特征 (64-dim) --> Linear(64, 128) -->\
                                        --> Add --> (128-dim)
MRI特征 (128-dim) ----------------------/
    |
    | Linear(128, 128)
    | -> BatchNorm1d(128)
    | -> ReLU
    | -> Dropout(0.7)
    | -> Linear(128, num_classes)
    |
Output: (batch_size, num_classes)
```

### 4.3 加权融合 (Weighted Fusion)
```
NIR特征 (64-dim) --> Linear(64, 128) -->\
                                        --> Weighted Sum --> (128-dim)
MRI特征 (128-dim) ----------------------/
    |
    | Linear(128, 128)
    | -> BatchNorm1d(128)
    | -> ReLU
    | -> Dropout(0.7)
    | -> Linear(128, num_classes)
    |
Output: (batch_size, num_classes)
```

其中权重通过softmax归一化：
```
weights = softmax([w1, w2])  # 初始值[0.5, 0.5]
fused = weights[0] * nir_proj + weights[1] * mri_features
```

### 4.4 双线性融合 (Bilinear Fusion)
```
NIR特征 (64-dim) --\
                   --> Bilinear(64, 128, 64) --> (64-dim)
MRI特征 (128-dim) -/
    |
    | BatchNorm1d(64)
    | -> ReLU
    | -> Dropout(0.5)
    | -> Linear(64, 32)
    | -> BatchNorm1d(32)
    | -> ReLU
    | -> Dropout(0.5)
    | -> Linear(32, num_classes)
    |
Output: (batch_size, num_classes)
```

## 5. 完整前向传播流程

```
NIR数据 -------------> NIR特征提取器 -----> 64维特征 \
                                                    --> 融合模块 --> 分类器 --> 分类结果
MRI数据 -------------> MRI特征提取器 -----> 128维特征 /
```

## 6. 各模块参数量估算

1. NIR特征提取器:
   - Conv1: 1×16×9 + 16 = 160
   - Conv2: 16×32×7 + 32 = 3,616
   - FC1: 64×((nir_input_size//4)×32) + 64
   - 总计: 约数千到数万个参数（取决于输入大小）

2. MRI特征提取器:
   - Conv层: 约3,000参数
   - FC层: 1024×128 + 128 = 131,200
   - 总计: 约135,000参数

3. 融合和分类模块:
   - Concat融合: (192+1)×128 + 128 + 128×num_classes + num_classes = 约30,000参数（4类）
   - 其他融合方式: 约20,000参数（4类）

## 7. 设计特点

1. **模块化设计**: NIR和MRI特征提取器独立设计，便于调试和优化
2. **多种融合策略**: 支持连接、加法、加权和双线性四种融合方式
3. **正则化技术**: 广泛使用BatchNorm和Dropout防止过拟合
4. **灵活配置**: 融合方式可通过参数选择