import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_network_architecture():
    """
    绘制NIR-MRI融合网络结构图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 设置背景
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制NIR数据流
    # 输入
    nir_input = patches.Rectangle((1, 7.5), 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(nir_input)
    ax.text(1.5, 8, 'NIR Input\n(1×L)', ha='center', va='center', fontsize=10, weight='bold')
    
    # NIR特征提取器
    nir_extractor = patches.Rectangle((3, 7), 2, 2, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_extractor)
    ax.text(4, 8, 'NIR Feature\nExtractor\n(1D-CNN)', ha='center', va='center', fontsize=9)
    
    # MRI数据流
    # 输入
    mri_input = patches.Rectangle((1, 2.5), 1, 1, linewidth=1, edgecolor='red', facecolor='lightcoral')
    ax.add_patch(mri_input)
    ax.text(1.5, 3, 'MRI Input\n(3×H×W)', ha='center', va='center', fontsize=10, weight='bold')
    
    # MRI特征提取器
    mri_extractor = patches.Rectangle((3, 2), 2, 2, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_extractor)
    ax.text(4, 3, 'MRI Feature\nExtractor\n(CNN)', ha='center', va='center', fontsize=9)
    
    # 特征向量
    nir_features = patches.Rectangle((6, 7.5), 1, 0.5, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_features)
    ax.text(6.5, 7.75, 'NIR Features\n(64-dim)', ha='center', va='center', fontsize=8)
    
    mri_features = patches.Rectangle((6, 2.75), 1, 0.5, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_features)
    ax.text(6.5, 3, 'MRI Features\n(128-dim)', ha='center', va='center', fontsize=8)
    
    # 融合模块（以连接融合为例）
    fusion_module = patches.Rectangle((8, 4.5), 2, 1, linewidth=1, edgecolor='purple', facecolor='plum')
    ax.add_patch(fusion_module)
    ax.text(9, 5, 'Feature Fusion\n(Concat/Add/Weighted/\nBilinear)', ha='center', va='center', fontsize=8)
    
    # 箭头连接
    ax.annotate('', xy=(3, 8), xytext=(2, 8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(6, 8), xytext=(5, 8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(3, 3), xytext=(2, 3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(6, 3), xytext=(5, 3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(8, 5.5), xytext=(7, 7.75),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(8, 4.5), xytext=(7, 3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    # 分类器
    classifier = patches.Rectangle((11, 4.5), 2, 1, linewidth=1, edgecolor='brown', facecolor='tan')
    ax.add_patch(classifier)
    ax.text(12, 5, 'Classifier\n(MLP)', ha='center', va='center', fontsize=9)
    
    # 输出
    output = patches.Rectangle((14, 4.5), 1, 1, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(output)
    ax.text(14.5, 5, 'Output\n(4 classes)', ha='center', va='center', fontsize=9)
    
    # 更多箭头
    ax.annotate('', xy=(11, 5), xytext=(10, 5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(14, 5), xytext=(13, 5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    # 添加详细信息框
    info_text = ("NIR Feature Extractor:\n"
                 "- Conv1: 1->16 channels, kernel=9\n"
                 "- Conv2: 16->32 channels, kernel=7\n"
                 "- FC: 64 units\n\n"
                 "MRI Feature Extractor:\n"
                 "- 3 Conv blocks (16, 32, 64 ch)\n"
                 "- Adaptive pooling (4×4)\n"
                 "- FC: 128 units\n\n"
                 "Fusion Methods:\n"
                 "- Concatenation\n"
                 "- Addition\n"
                 "- Weighted Sum\n"
                 "- Bilinear")
    
    ax.text(0.5, 0.5, info_text, ha='left', va='bottom', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.title("NIR-MRI Fusion Network Architecture", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_fusion_methods():
    """
    绘制不同的融合方法示意图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. 连接融合
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # 输入特征
    nir_feat = patches.Rectangle((1, 3), 1, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_feat)
    ax.text(1.5, 3.5, 'NIR\n(64)', ha='center', va='center', fontsize=8)
    
    mri_feat = patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_feat)
    ax.text(1.5, 1.5, 'MRI\n(128)', ha='center', va='center', fontsize=8)
    
    # 连接操作
    concat_op = patches.Rectangle((3, 1.5), 1, 2, linewidth=1, edgecolor='purple', facecolor='plum')
    ax.add_patch(concat_op)
    ax.text(3.5, 2.5, 'Concat', ha='center', va='center', fontsize=9)
    
    # 输出特征
    output_feat = patches.Rectangle((5, 2), 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(output_feat)
    ax.text(5.5, 2.5, 'Fused\n(192)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(3, 3), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(3, 2), xytext=(2, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(5, 2.5), xytext=(4, 2.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    ax.set_title('Concatenation Fusion', fontsize=12, weight='bold')
    
    # 2. 加法融合
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # 输入特征
    nir_feat = patches.Rectangle((1, 3), 1, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_feat)
    ax.text(1.5, 3.5, 'NIR\n(64)', ha='center', va='center', fontsize=8)
    
    mri_feat = patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_feat)
    ax.text(1.5, 1.5, 'MRI\n(128)', ha='center', va='center', fontsize=8)
    
    # 投影层
    proj_layer = patches.Rectangle((3, 3), 1, 1, linewidth=1, edgecolor='gray', facecolor='lightgray')
    ax.add_patch(proj_layer)
    ax.text(3.5, 3.5, 'Proj\n(128)', ha='center', va='center', fontsize=8)
    
    # 加法操作
    add_op = patches.Circle((5, 2.5), 0.3, linewidth=1, edgecolor='purple', facecolor='plum')
    ax.add_patch(add_op)
    ax.text(5, 2.5, '+', ha='center', va='center', fontsize=12, weight='bold')
    
    # 输出特征
    output_feat = patches.Rectangle((6.5, 2), 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(output_feat)
    ax.text(7, 2.5, 'Fused\n(128)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(3, 3.5), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(5.7, 1.5), xytext=(2, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(4.7, 3), xytext=(4, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(6.5, 2.5), xytext=(5.3, 2.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    ax.set_title('Addition Fusion', fontsize=12, weight='bold')
    
    # 3. 加权融合
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # 输入特征
    nir_feat = patches.Rectangle((1, 3), 1, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_feat)
    ax.text(1.5, 3.5, 'NIR\n(64)', ha='center', va='center', fontsize=8)
    
    mri_feat = patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_feat)
    ax.text(1.5, 1.5, 'MRI\n(128)', ha='center', va='center', fontsize=8)
    
    # 投影层
    proj_layer = patches.Rectangle((3, 3), 1, 1, linewidth=1, edgecolor='gray', facecolor='lightgray')
    ax.add_patch(proj_layer)
    ax.text(3.5, 3.5, 'Proj\n(128)', ha='center', va='center', fontsize=8)
    
    # 权重
    w1 = patches.Rectangle((4, 3.5), 0.5, 0.3, linewidth=1, edgecolor='red', facecolor='lightcoral')
    ax.add_patch(w1)
    ax.text(4.25, 3.65, 'w1', ha='center', va='center', fontsize=8)
    
    w2 = patches.Rectangle((4, 1.5), 0.5, 0.3, linewidth=1, edgecolor='red', facecolor='lightcoral')
    ax.add_patch(w2)
    ax.text(4.25, 1.65, 'w2', ha='center', va='center', fontsize=8)
    
    # 加权操作
    weighted_op = patches.Circle((6, 2.5), 0.3, linewidth=1, edgecolor='purple', facecolor='plum')
    ax.add_patch(weighted_op)
    ax.text(6, 2.5, '+', ha='center', va='center', fontsize=12, weight='bold')
    
    # 输出特征
    output_feat = patches.Rectangle((7.5, 2), 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(output_feat)
    ax.text(8, 2.5, 'Fused\n(128)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(3, 3.5), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(5.7, 1.5), xytext=(2, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(4, 2.8), xytext=(3.5, 3.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(5.7, 3), xytext=(4.5, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(5.7, 2), xytext=(4.5, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(7.5, 2.5), xytext=(6.3, 2.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    ax.set_title('Weighted Fusion', fontsize=12, weight='bold')
    
    # 4. 双线性融合
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # 输入特征
    nir_feat = patches.Rectangle((1, 3), 1, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(nir_feat)
    ax.text(1.5, 3.5, 'NIR\n(64)', ha='center', va='center', fontsize=8)
    
    mri_feat = patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='orange', facecolor='moccasin')
    ax.add_patch(mri_feat)
    ax.text(1.5, 1.5, 'MRI\n(128)', ha='center', va='center', fontsize=8)
    
    # 双线性操作
    bilinear_op = patches.Rectangle((3, 1.5), 2, 2, linewidth=1, edgecolor='purple', facecolor='plum')
    ax.add_patch(bilinear_op)
    ax.text(4, 2.5, 'Bilinear\nLayer', ha='center', va='center', fontsize=9)
    
    # 输出特征
    output_feat = patches.Rectangle((6.5, 2), 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(output_feat)
    ax.text(7, 2.5, 'Fused\n(64)', ha='center', va='center', fontsize=8)
    
    # 箭头
    ax.annotate('', xy=(3, 3), xytext=(2, 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(3, 2), xytext=(2, 1.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate('', xy=(6.5, 2.5), xytext=(5, 2.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    
    ax.set_title('Bilinear Fusion', fontsize=12, weight='bold')
    
    plt.suptitle('Different Fusion Methods', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('fusion_methods.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_network_architecture()
    draw_fusion_methods()