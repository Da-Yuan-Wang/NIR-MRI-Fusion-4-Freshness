import torch
import os
import argparse


def check_model_info(model_path):
    """
    检查模型文件的信息
    
    Args:
        model_path: 模型文件路径
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return
    
    # 加载模型文件
    print(f"正在加载模型文件: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 显示模型信息
    print("\n" + "="*50)
    print("模型文件信息")
    print("="*50)
    
    # 显示检查点中的所有键
    print(f"检查点包含的键: {list(checkpoint.keys())}")
    
    # 显示模型训练的epoch
    if 'epoch' in checkpoint:
        print(f"训练的Epoch: {checkpoint['epoch']}")
    else:
        print("Epoch信息: 未找到")
    
    # 显示最佳准确率
    if 'best_acc' in checkpoint:
        print(f"最佳准确率: {checkpoint['best_acc']:.4f}")
    else:
        print("最佳准确率: 未找到")
    
    # 显示融合类型
    if 'fusion_type' in checkpoint:
        print(f"融合类型: {checkpoint['fusion_type']}")
    else:
        print("融合类型: 未找到")
    
    # 显示模型状态字典的键（层名称）
    if 'model_state_dict' in checkpoint:
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        print(f"\n模型层的数量: {len(state_dict_keys)}")
        print("模型层名称:")
        for i, key in enumerate(state_dict_keys):
            print(f"  {i+1:3d}. {key}")
            
        # 显示一些参数统计信息
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        trainable_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() if p.requires_grad)
        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
    else:
        print("\n模型状态字典: 未找到")
    
    # 显示优化器状态（如果存在）
    if 'optimizer_state_dict' in checkpoint:
        opt_state_keys = list(checkpoint['optimizer_state_dict'].keys())
        print(f"\n优化器状态键:")
        for key in opt_state_keys:
            print(f"  - {key}")
    else:
        print("\n优化器状态: 未找到")
    
    print("="*50)


def main():
    # 模型路径（可以按实际情况修改）
    model_path = r'checkpoints\best_fusion_model_weighted-epoch944-testacc0.9661.pth'
    
    # 检查模型信息
    check_model_info(model_path)
    
    # 同时检查其他可能的模型文件
    checkpoints_dir = './checkpoints'
    if os.path.exists(checkpoints_dir):
        print(f"\n检查 '{checkpoints_dir}' 目录中的其他模型文件:")
        for file in os.listdir(checkpoints_dir):
            if file.endswith('.pth'):
                print(f"\n--- {file} ---")
                full_path = os.path.join(checkpoints_dir, file)
                try:
                    checkpoint = torch.load(full_path, map_location='cpu')
                    if 'epoch' in checkpoint:
                        print(f"  Epoch: {checkpoint['epoch']}")
                    if 'best_acc' in checkpoint:
                        print(f"  最佳准确率: {checkpoint['best_acc']:.4f}")
                    if 'fusion_type' in checkpoint:
                        print(f"  融合类型: {checkpoint['fusion_type']}")
                except Exception as e:
                    print(f"  读取失败: {str(e)}")


if __name__ == '__main__':
    main()