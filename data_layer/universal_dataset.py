"""
通用数据集类实现 - 修复导入错误版本
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# 修复导入错误 - 使用绝对导入
try:
    from base_parser import DataParser
except ImportError:
    # 如果在包内运行，尝试相对导入
    from .base_parser import DataParser


class UniversalHARDataset(Dataset):
    """
    通用的人体活动识别数据集类
    支持多模态数据输入，输出数据字典
    """
    
    def __init__(self, data_parser: DataParser, split: str = 'train', 
                 transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            data_parser: 数据解析器实例
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据变换函数（可选）
        """
        self.data_parser = data_parser
        self.split = split
        self.transform = transform
        
        # 解析数据
        self.data_list, self.labels = self.data_parser.parse_data(split)
        
        # 获取模态信息
        self.modality_info = self.data_parser.get_modality_info()
        
        print(f"Dataset initialized with {len(self.data_list)} samples")
        print(f"Available modalities: {list(self.modality_info.keys())}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Tuple of (data_dict, label):
                - data_dict: 包含不同模态数据的字典
                - label: 对应的标签
        """
        # 获取原始数据
        data_dict = self.data_list[idx]
        label = self.labels[idx]
        
        # 转换为张量
        tensor_dict = {}
        for modality, data in data_dict.items():
            tensor_dict[modality] = torch.FloatTensor(data)
            
            # 应用变换（如果有）
            if self.transform:
                tensor_dict[modality] = self.transform(tensor_dict[modality])
        
        return tensor_dict, label
    
    def get_modality_info(self) -> Dict[str, Dict[str, Any]]:
        """获取模态信息"""
        return self.modality_info
    
    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(set(self.labels))