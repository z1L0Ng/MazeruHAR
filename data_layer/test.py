#!/usr/bin/env python3
"""
数据层测试脚本 - 修复导入错误版本
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np

# 确保能够导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 现在导入应该正常工作
from base_parser import DataParser
from shl_parser import SHLDataParser
from universal_dataset import UniversalHARDataset


def test_shl_data_parser():
    """测试SHL数据解析器"""
    print("=" * 50)
    print("测试SHL数据解析器")
    print("=" * 50)
    
    # 创建测试配置
    config = {
        'name': 'SHL',
        'data_path': './datasets/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        }
    }
    
    # 创建数据解析器
    parser = SHLDataParser(config)
    
    # 测试模态信息获取
    modality_info = parser.get_modality_info()
    print("模态信息:")
    for modality, info in modality_info.items():
        print(f"  {modality}: {info}")
    
    # 测试数据解析
    try:
        print("\n开始解析训练数据...")
        train_data, train_labels = parser.parse_data('train')
        print(f"训练数据样本数: {len(train_data)}")
        print(f"训练标签数: {len(train_labels)}")
        
        if train_data:
            print("第一个样本的数据形状:")
            for modality, data in train_data[0].items():
                print(f"  {modality}: {data.shape}")
                
    except Exception as e:
        print(f"数据解析失败: {e}")
        import traceback
        traceback.print_exc()


def test_universal_dataset():
    """测试通用数据集类"""
    print("\n" + "=" * 50)
    print("测试通用数据集类")
    print("=" * 50)
    
    # 创建测试配置
    config = {
        'name': 'SHL',
        'data_path': './datasets/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        }
    }
    
    # 创建数据解析器和数据集
    parser = SHLDataParser(config)
    dataset = UniversalHARDataset(parser, split='train')
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数: {dataset.get_num_classes()}")
    print(f"模态信息: {dataset.get_modality_info()}")
    
    # 测试数据加载
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("\n测试数据加载:")
    for batch_idx, (data_dict, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Labels: {labels.shape}, {labels}")
        for modality, data in data_dict.items():
            print(f"  {modality}: {data.shape}")
        
        if batch_idx == 1:  # 只显示前两个batch
            break


def test_data_dict_output():
    """测试数据字典输出格式"""
    print("\n" + "=" * 50)
    print("测试数据字典输出格式")
    print("=" * 50)
    
    # 创建配置
    config = {
        'name': 'SHL',
        'data_path': './datasets/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        }
    }
    
    # 创建数据集
    parser = SHLDataParser(config)
    dataset = UniversalHARDataset(parser, split='train')
    
    # 获取单个样本
    data_dict, label = dataset[0]
    
    print("单个样本数据字典格式:")
    print(f"标签: {label}")
    for modality, data in data_dict.items():
        print(f"  {modality}:")
        print(f"    形状: {data.shape}")
        print(f"    数据类型: {data.dtype}")
        print(f"    数值范围: [{data.min():.3f}, {data.max():.3f}]")


def test_batch_processing():
    """测试批处理功能"""
    print("\n" + "=" * 50)
    print("测试批处理功能")
    print("=" * 50)
    
    # 创建配置
    config = {
        'name': 'SHL',
        'data_path': './datasets/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        }
    }
    
    # 创建数据集
    parser = SHLDataParser(config)
    dataset = UniversalHARDataset(parser, split='train')
    
    # 测试不同的批处理大小
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\n批处理大小: {batch_size}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        data_dict, labels = next(iter(dataloader))
        print(f"  标签形状: {labels.shape}")
        
        for modality, data in data_dict.items():
            print(f"  {modality}形状: {data.shape}")
            
        # 验证批处理维度
        expected_batch_size = min(batch_size, len(dataset))
        assert labels.shape[0] == expected_batch_size, f"批处理大小不匹配: {labels.shape[0]} != {expected_batch_size}"
        
        for modality, data in data_dict.items():
            assert data.shape[0] == expected_batch_size, f"{modality}批处理大小不匹配"


def main():
    """主函数"""
    print("开始测试重构后的数据层...")
    
    try:
        # 运行各种测试
        test_shl_data_parser()
        test_universal_dataset()
        test_data_dict_output()
        test_batch_processing()
        
        print("\n" + "=" * 50)
        print("所有测试完成!")
        print("=" * 50)
        
        # 显示任务1.2完成情况
        print("\n任务1.2完成情况:")
        print("✓ 实现了抽象数据解析器基类 (DataParser)")
        print("✓ 实现了SHL数据集的具体解析器 (SHLDataParser)")
        print("✓ 实现了通用的数据集类 (UniversalHARDataset)")
        print("✓ 确保输出数据字典格式")
        print("✓ 支持多模态数据处理")
        print("✓ 支持配置驱动的数据加载")
        print("✓ 包含错误处理和验证")
        
        print("\n下一步 (任务1.3):")
        print("- 实现DynamicHarModel的骨架")
        print("- 实现基于nn.ModuleDict的动态专家实例化")
        print("- 实现动态forward流程")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()