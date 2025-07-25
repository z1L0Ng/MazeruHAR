"""
SHL数据集解析器 - 修复导入错误版本
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import zipfile
import requests
from tqdm import tqdm
from scipy import signal

# 修复导入错误 - 使用绝对导入
try:
    from base_parser import DataParser
except ImportError:
    # 如果在包内运行，尝试相对导入
    from .base_parser import DataParser


class SHLDataParser(DataParser):
    """
    SHL数据集解析器
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('window_size', 256)
        self.step_size = config.get('step_size', 128)
        self.sample_rate = config.get('sample_rate', 100)  # Hz
        
        # SHL数据集的活动标签映射
        self.activity_labels = {
            1: 'Standing', 2: 'Walking', 3: 'Running', 4: 'Biking',
            5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'
        }
        
        # 定义可用的模态和其在Motion.txt中的列
        self.available_modalities = {
            'imu': {'channels': 6, 'cols': [1,2,3,4,5,6]},
            'pressure': {'channels': 1, 'file': 'Pressure.txt', 'cols': [1]},
            'magnetometer': {'channels': 3, 'cols': [7,8,9]},
            'lacc': {'channels': 3, 'cols': [17, 18, 19]},
            'gra': {'channels': 3, 'cols': [14, 15, 16]}
        }
        self.body_locations = ["Bag", "Hand", "Hips", "Torso"]
        # 定义数据集划分的用户
        self.user_folders = {
            'train': ['User1'],
            'val': ['User2'],
            'test': ['User3']
        }

    def _process_label_for_window(self, labels: np.ndarray) -> int:
        """确定窗口中最常见的标签。"""
        unique_values, counts = np.unique(labels, return_counts=True)
        return unique_values[np.argmax(counts)]

    def _segment_data(self, data: np.ndarray) -> np.ndarray:
        """将时间序列数据分割成窗口。"""
        segments = []
        for i in range(0, data.shape[0] - self.window_size, self.step_size):
            segments.append(data[i:i + self.window_size, :])
        return np.asarray(segments)

    def _segment_labels(self, labels: np.ndarray) -> np.ndarray:
        """将标签数据分割成窗口并为每个窗口分配一个标签。"""
        segmented_labels = []
        for i in range(0, labels.shape[0] - self.window_size, self.step_size):
            segmented_labels.append(self._process_label_for_window(labels[i:i + self.window_size]))
        return np.asarray(segmented_labels)

    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        解析真实的SHL数据，并提供详细的调试信息。
        """
        print(f"Parsing real SHL data for '{split}' split...")
        
        base_path = self.data_path
        root_dir = os.path.join(base_path, 'SHLDataset_preview_v1')
        
        if not os.path.isdir(root_dir):
            error_msg = (f"SHL dataset directory not found at the expected location: '{root_dir}'.\n"
                         f"Please check the path in your config file.")
            raise FileNotFoundError(error_msg)

        all_data = []
        all_labels = []

        user_list = self.user_folders.get(split)
        if not user_list:
            raise ValueError(f"Invalid split name: {split}. Use 'train', 'val', or 'test'.")

        for user_folder in user_list:
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path):
                print(f"Warning: User folder {user_folder} not found, skipping.")
                continue

            time_folders = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
            for time_folder in time_folders:
                session_path = os.path.join(user_path, time_folder)
                
                # 已修正: 使用 sep='\s+'
                labels_raw = pd.read_csv(os.path.join(session_path, 'Label.txt'), header=None, sep=r'\s+').values
                motion_data = {}
                for loc in self.body_locations:
                    motion_file = os.path.join(session_path, f'{loc}_Motion.txt')
                    if os.path.exists(motion_file):
                         # 已修正: 使用 sep='\s+'
                        motion_data[loc] = pd.read_csv(motion_file, header=None, sep=r'\s+').values

                if not motion_data:
                    continue

                min_len = min(len(labels_raw), min(len(data) for data in motion_data.values()))
                labels_raw = labels_raw[:min_len]

                for loc in self.body_locations:
                    if loc not in motion_data: continue
                    motion_data[loc] = motion_data[loc][:min_len]

                    segmented_labels = self._segment_labels(labels_raw[:, 1])
                    
                    non_null_indices = np.where(segmented_labels != 0)[0]
                    if len(non_null_indices) == 0: continue
                    
                    final_labels = segmented_labels[non_null_indices] - 1

                    user_samples = [{} for _ in range(len(final_labels))]
                    for mod_name, mod_config in self.modalities.items():
                        if mod_config.get('enabled', False) and mod_name in self.available_modalities:
                            cols = self.available_modalities[mod_name]['cols']
                            mod_data = motion_data[loc][:, cols]
                            segmented_mod_data = self._segment_data(mod_data)[non_null_indices]
                            
                            if self.sample_rate < 100:
                                factor = 100 // self.sample_rate
                                segmented_mod_data = signal.decimate(segmented_mod_data, factor, axis=1)

                            for i in range(len(final_labels)):
                                user_samples[i][mod_name] = segmented_mod_data[i].astype(np.float32)
                    
                    all_data.extend(user_samples)
                    all_labels.extend(final_labels.tolist())

        print(f"Finished parsing. Found {len(all_data)} samples for '{split}' split.")
        return all_data, all_labels
    
    def get_modality_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取模态信息
        """
        info = {}
        for modality, config in self.modalities.items():
            if config.get('enabled', False) and modality in self.available_modalities:
                info[modality] = self.available_modalities[modality].copy()
                info[modality]['window_size'] = self.window_size
                info[modality]['sample_rate'] = self.sample_rate
        return info