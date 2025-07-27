"""
SHL数据解析器 - 修正版本（基于真实传感器）
"""

import os
import numpy as np
import pandas as pd
import hickle as hkl
from typing import Dict, List, Tuple, Any
import logging

try:
    from .base_parser import DataParser
except ImportError:
    try:
        from data_layer.base_parser import DataParser
    except ImportError:
        from base_parser import DataParser


class SHLDataParser(DataParser):
    """
    SHL数据集解析器 - 基于真实传感器配置
    支持6种传感器类型：IMU、磁力计、方向、重力、线性加速度
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('window_size', 128)
        self.step_size = config.get('step_size', 64)
        self.sample_rate = config.get('sample_rate', 100)
        self.normalize_per_sample = config.get('normalize_per_sample', True)
        
        # SHL数据集的活动标签映射 (处理后的0-based)
        self.activity_labels = {
            0: 'Still', 1: 'Walking', 2: 'Run', 3: 'Bike',
            4: 'Car', 5: 'Bus', 6: 'Train', 7: 'Subway'
        }
        
        # 真实的SHL传感器配置
        self.sensor_column_mapping = {
            'acc': [1, 2, 3],           # 加速度计
            'gyro': [4, 5, 6],          # 陀螺仪
            'mag': [7, 8, 9],           # 磁力计
            'ori': [10, 11, 12, 13],    # 方向四元数
            'gra': [14, 15, 16],        # 重力
            'lacc': [17, 18, 19]        # 线性加速度
        }
        
        # 模态配置
        self.modalities_config = config.get('modalities', {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 数据集划分比例
        self.split_ratios = {
            'train': 0.7,
            'val': 0.15, 
            'test': 0.15
        }

    def load_preprocessed_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """从预处理的HKL文件加载数据"""
        data_file = os.path.join(self.data_path, 'clientsData.hkl')
        label_file = os.path.join(self.data_path, 'clientsLabel.hkl')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"标签文件不存在: {label_file}")
        
        self.logger.info(f"从 {self.data_path} 加载预处理数据...")
        
        clients_data = hkl.load(data_file)
        clients_labels = hkl.load(label_file)
        
        self.logger.info(f"成功加载 {len(clients_data)} 个客户端的数据")
        
        return clients_data, clients_labels

    def split_modalities(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将19维数据拆分为不同的传感器模态
        
        Args:
            data: 形状为 (time_steps, 19) 的数据
            
        Returns:
            模态字典，每个键对应一种传感器类型
        """
        modalities = {}
        
        # 根据配置文件中启用的模态进行拆分
        for modality_name, modality_config in self.modalities_config.items():
            if not modality_config.get('enabled', False):
                continue
                
            # 获取列索引
            column_indices = modality_config.get('column_indices', [])
            if not column_indices:
                # 如果没有直接指定列索引，根据传感器名称推断
                sensors = modality_config.get('sensors', [])
                column_indices = []
                for sensor in sensors:
                    if sensor in self.sensor_column_mapping:
                        column_indices.extend(self.sensor_column_mapping[sensor])
            
            if column_indices:
                # 调整索引（从1-based转为0-based）
                adjusted_indices = [idx - 1 for idx in column_indices if idx > 0]
                
                # 提取对应列的数据
                if max(adjusted_indices) < data.shape[1]:
                    modality_data = data[:, adjusted_indices]
                    modalities[modality_name] = modality_data
                    self.logger.debug(f"提取 {modality_name}: 形状 {modality_data.shape}")
                else:
                    self.logger.warning(f"模态 {modality_name} 的列索引超出数据范围")
        
        return modalities

    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        解析数据并返回模态字典格式
        """
        self.logger.info(f"🚀 开始解析 {split} 数据集...")
        
        # 加载预处理数据
        all_data, all_labels = self.load_preprocessed_data()
        
        if not all_data:
            raise ValueError("没有加载到任何数据")
        
        # 数据集划分
        total_samples = len(all_data)
        if split == 'train':
            start_idx = 0
            end_idx = int(total_samples * self.split_ratios['train'])
        elif split == 'val':
            start_idx = int(total_samples * self.split_ratios['train'])
            end_idx = start_idx + int(total_samples * self.split_ratios['val'])
        else:  # test
            start_idx = int(total_samples * (self.split_ratios['train'] + self.split_ratios['val']))
            end_idx = total_samples
        
        split_data = all_data[start_idx:end_idx]
        split_labels = all_labels[start_idx:end_idx]
        
        self.logger.info(f"{split} 数据集: {len(split_data)} 样本")
        
        # 分离多模态数据
        self.logger.info("分离多模态数据...")
        processed_data = []
        
        for i, data in enumerate(split_data):
            try:
                modalities = self.split_modalities(data)
                processed_data.append(modalities)
            except Exception as e:
                self.logger.error(f"处理样本 {i} 时出错: {e}")
                raise
        
        self.logger.info(f"成功解析 {len(processed_data)} 个样本")
        
        # 打印第一个样本的模态信息
        if processed_data:
            first_sample = processed_data[0]
            self.logger.info("第一个样本的模态信息:")
            for modality, data in first_sample.items():
                self.logger.info(f"  {modality}: {data.shape}")
        
        return processed_data, split_labels

    def get_num_classes(self) -> int:
        """返回类别数"""
        return len(self.activity_labels)

    def get_class_names(self) -> List[str]:
        """返回类别名称列表"""
        return list(self.activity_labels.values())

    def get_modality_info(self) -> Dict[str, Any]:
        """返回模态信息"""
        modality_info = {}
        for modality_name, config in self.modalities_config.items():
            if config.get('enabled', False):
                modality_info[modality_name] = {
                    'channels': config.get('channels', 0),
                    'sensors': config.get('sensors', []),
                    'enabled': True
                }
        return modality_info