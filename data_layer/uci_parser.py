import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

try:
    from .base_parser import DataParser
except ImportError:
    from data_layer.base_parser import DataParser

class UCIHarParser(DataParser):
    """
    UCI HAR 数据集解析器
    读取 'Inertial Signals' 文件夹下的原始窗口数据 (128 time steps)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # UCI 数据集特有配置
        self.signal_files = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"
        ]
        
        # 默认映射：如果没有在config里指定indices，则使用此默认值
        # 0-2: Body Acc, 3-5: Body Gyro, 6-8: Total Acc
        self.default_map = {
            'body_acc': [0, 1, 2],
            'gyro': [3, 4, 5],
            'acc': [6, 7, 8]
        }

    def load_signal_file(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        return pd.read_csv(file_path, sep='\s+', header=None).values

    def load_raw_group(self, group: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载 train 或 test 的完整数据"""
        # 假设路径结构: dataset_path/train/Inertial Signals/
        path_group = 'test' if group == 'test' else 'train'
        folder = os.path.join(self.data_path, path_group, "Inertial Signals")
        
        if not os.path.exists(folder):
            # 尝试不带 Inertial Signals 的路径，或者提示错误
            raise FileNotFoundError(f"找不到信号文件夹: {folder}. 请确保 UCI 数据集已解压且保持原始结构。")

        self.logger.info(f"Loading UCI {group} signals from {folder}...")
        
        signals = []
        for name in self.signal_files:
            filename = f"{name}_{path_group}.txt"
            signals.append(self.load_signal_file(os.path.join(folder, filename)))
            
        # 堆叠 (N, 128, 9)
        X = np.stack(signals, axis=-1)
        
        # 加载标签
        y_path = os.path.join(self.data_path, path_group, f"y_{path_group}.txt")
        y = pd.read_csv(y_path, sep='\s+', header=None).values.flatten()
        y = y - 1 # 转换 1-6 为 0-5
        
        return X, y

    def split_modalities(self, data_sample: np.ndarray) -> Dict[str, np.ndarray]:
        """将 (128, 9) 的样本切分为配置中启用的模态"""
        res = {}
        for mod_name, mod_cfg in self.modalities.items():
            if not mod_cfg.get('enabled', False):
                continue
            
            # 获取列索引
            indices = mod_cfg.get('column_indices', [])
            if not indices:
                # 尝试使用默认映射匹配
                for i, j in self.default_map.items():
                    if i in mod_name: 
                        indices = j
                        break
            
            if indices:
                # 确保索引有效
                valid_idx = [i for i in indices if i < data_sample.shape[1]]
                if valid_idx:
                    res[mod_name] = data_sample[:, valid_idx]
        return res

    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """解析数据并切分模态"""
        # UCI 只有 train 和 test。Validation 通常从 train 切分
        if split == 'test':
            X, y = self.load_raw_group('test')
        else:
            X, y = self.load_raw_group('train')
            
            # 简单的 8:2 切分用于 validation
            val_split_idx = int(len(X) * 0.8)
            if split == 'train':
                X, y = X[:val_split_idx], y[:val_split_idx]
            elif split == 'validation':
                X, y = X[val_split_idx:], y[val_split_idx:]
                
        self.logger.info(f"UCI {split} set: {len(X)} samples")
        
        # 转换为字典列表
        processed_data = [self.split_modalities(sample) for sample in X]
        return processed_data, y.tolist()

    def get_modality_info(self) -> Dict[str, Any]:
        info = {}
        for k, v in self.modalities.items():
            if v.get('enabled'):
                info[k] = {'channels': v.get('channels', 3), 'enabled': True}
        return info