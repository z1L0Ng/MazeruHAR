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
        
        # 从嵌套的配置中正确获取参数
        dataset_config = config.get('dataset', {})
        preprocessing_config = dataset_config.get('preprocessing', {})
        
        self.window_size = preprocessing_config.get('window_size', 128)
        self.step_size = preprocessing_config.get('step_size', 64)
        self.sample_rate = preprocessing_config.get('sample_rate', 100)
        self.normalize_per_sample = preprocessing_config.get('normalize_per_sample', True)
        
        self.activity_labels = {
            0: 'Still', 1: 'Walking', 2: 'Run', 3: 'Bike',
            4: 'Car', 5: 'Bus', 6: 'Train', 7: 'Subway'
        }
        
        self.sensor_column_mapping = {
            'acc': [1, 2, 3], 'gyro': [4, 5, 6], 'mag': [7, 8, 9],
            'ori': [10, 11, 12, 13], 'gra': [14, 15, 16], 'lacc': [17, 18, 19]
        }
        
        self.logger = logging.getLogger(__name__)
        
        self.split_ratios = dataset_config.get('data_split', {
            'train': 0.7, 'validation': 0.15, 'test': 0.15
        })

    def load_preprocessed_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """从预处理的HKL文件加载数据"""
        data_file = os.path.join(self.data_path, 'clientsData.hkl')
        label_file = os.path.join(self.data_path, 'clientsLabel.hkl')
        
        if not os.path.exists(data_file) or not os.path.exists(label_file):
            raise FileNotFoundError(f"数据或标签文件在路径 {self.data_path} 中不存在")
        
        self.logger.info(f"从 {self.data_path} 加载预处理数据...")
        clients_data = hkl.load(data_file)
        clients_labels = hkl.load(label_file)
        self.logger.info(f"成功加载 {len(clients_data)} 个客户端的数据")
        return clients_data, clients_labels

    def split_modalities(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """将19维数据拆分为不同的传感器模态"""
        modalities_data = {}
        # **修正点**: 使用由基类正确初始化的 self.modalities
        for modality_name, modality_config in self.modalities.items():
            if not modality_config.get('enabled', False):
                continue
            
            column_indices = modality_config.get('column_indices', [])
            if not column_indices:
                sensors = modality_config.get('sensors', [])
                column_indices = [idx for s in sensors if s in self.sensor_column_mapping for idx in self.sensor_column_mapping[s]]
            
            if column_indices:
                adjusted_indices = [idx - 1 for idx in column_indices if 0 < idx <= data.shape[1]]
                if adjusted_indices:
                    modalities_data[modality_name] = data[:, adjusted_indices]
        return modalities_data

    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """解析数据并返回模态字典格式 (已修复数据划分逻辑)"""
        self.logger.info(f"🚀 开始解析 {split} 数据集...")
        all_clients_data, all_clients_labels = self.load_preprocessed_data()

        combined_data = np.vstack(all_clients_data)
        combined_labels = np.hstack(all_clients_labels)
        self.logger.info(f"所有客户端数据已合并，总样本数: {len(combined_data)}")

        total_samples = len(combined_data)
        train_end_idx = int(total_samples * self.split_ratios['train'])
        val_end_idx = train_end_idx + int(total_samples * self.split_ratios.get('validation', self.split_ratios.get('val', 0.15)))

        if split == 'train':
            split_data, split_labels = combined_data[:train_end_idx], combined_labels[:train_end_idx]
        elif split in ['validation', 'val']:
            split_data, split_labels = combined_data[train_end_idx:val_end_idx], combined_labels[train_end_idx:val_end_idx]
        elif split == 'test':
            split_data, split_labels = combined_data[val_end_idx:], combined_labels[val_end_idx:]
        else:
            raise ValueError(f"未知的 split 类型: {split}")

        self.logger.info(f"{split} 数据集: {len(split_data)} 样本")
        
        processed_data = [self.split_modalities(sample) for sample in split_data]
        self.logger.info(f"成功解析 {len(processed_data)} 个样本")

        if processed_data:
            self.logger.info(f"第一个样本的模态信息: { {k: v.shape for k, v in processed_data[0].items()} }")
        
        return processed_data, split_labels.tolist()

    def get_modality_info(self) -> Dict[str, Any]:
        """返回模态信息"""
        modality_info = {}
        # **修正点**: 使用由基类正确初始化的 self.modalities
        for modality_name, config in self.modalities.items():
            if config.get('enabled', False):
                modality_info[modality_name] = {
                    'channels': config.get('channels', 0),
                    'sensors': config.get('sensors', []),
                    'enabled': True
                }
        return modality_info