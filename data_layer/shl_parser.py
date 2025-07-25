"""
SHL数据集解析器 - 修复导入错误和多模态加载逻辑的版本
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
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
        
        # 定义可用的模态及其来源
        self.available_modalities = {
            'imu': {'channels': 6, 'file': '_Motion.txt', 'cols': [1,2,3,4,5,6]},
            'pressure': {'channels': 1, 'file': 'Pressure.txt', 'cols': [1]},
            'magnetometer': {'channels': 3, 'file': '_Motion.txt', 'cols': [7,8,9]},
            'lacc': {'channels': 3, 'file': '_Motion.txt', 'cols': [17, 18, 19]},
            'gra': {'channels': 3, 'file': '_Motion.txt', 'cols': [14, 15, 16]}
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
        print(f"正在为 '{split}' 分割解析真实的SHL数据...")
        
        base_path = self.data_path
        root_dir = os.path.join(base_path, 'SHLDataset_preview_v1')
        
        if not os.path.isdir(root_dir):
            error_msg = (f"在预期位置找不到SHL数据集目录: '{root_dir}'.\n"
                         f"请检查您的配置文件中的路径。")
            raise FileNotFoundError(error_msg)

        all_data = []
        all_labels = []

        user_list = self.user_folders.get(split)
        if not user_list:
            raise ValueError(f"无效的分割名称: {split}。请使用 'train', 'val', 或 'test'。")

        for user_folder in user_list:
            user_path = os.path.join(root_dir, user_folder)
            if not os.path.isdir(user_path):
                print(f"警告: 用户文件夹 {user_folder} 未找到，跳过。")
                continue

            time_folders = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
            for time_folder in time_folders:
                session_path = os.path.join(user_path, time_folder)
                
                # --- BUG修复：为每个模态加载各自的文件 ---
                # 1. 加载所有需要的原始数据文件
                try:
                    labels_raw_df = pd.read_csv(os.path.join(session_path, 'Label.txt'), header=None, sep=r'\\s+')
                    
                    raw_data_dfs = {}
                    # 加载所有位置的Motion.txt
                    for loc in self.body_locations:
                        motion_file = os.path.join(session_path, f'{loc}_Motion.txt')
                        if os.path.exists(motion_file):
                            raw_data_dfs[f'{loc}_Motion'] = pd.read_csv(motion_file, header=None, sep=r'\\s+')
                    
                    # 加载Pressure.txt
                    pressure_file = os.path.join(session_path, 'Pressure.txt')
                    if os.path.exists(pressure_file):
                        raw_data_dfs['Pressure'] = pd.read_csv(pressure_file, header=None, sep=r'\\s+')

                except FileNotFoundError as e:
                    print(f"警告: 在 {session_path} 中缺少文件，跳过该会话: {e}")
                    continue
                
                # 2. 对齐所有数据帧的时间戳
                all_dfs = {'Label': labels_raw_df, **raw_data_dfs}
                # 使用第一个时间戳列作为对齐基准
                for name, df in all_dfs.items():
                    df.set_index(0, inplace=True)
                
                # 合并所有数据帧，内连接以保证时间戳对齐
                aligned_df = pd.concat(all_dfs.values(), axis=1, join='inner')
                
                # 3. 移除任何仍然存在的NaN值
                aligned_df.dropna(inplace=True)
                if aligned_df.empty:
                    continue

                # 4. 分离出对齐后的数据
                labels_aligned = aligned_df.iloc[:, 0].values # 第一列是标签
                
                # 为每个位置和文件类型重新构建数据数组
                motion_data = {}
                pressure_data = None
                
                col_idx = 1 # 0是标签
                for name, df in raw_data_dfs.items():
                    num_cols = df.shape[1]
                    data_slice = aligned_df.iloc[:, col_idx : col_idx + num_cols].values
                    if 'Pressure' in name:
                        pressure_data = data_slice
                    else:
                        loc = name.split('_')[0]
                        motion_data[loc] = data_slice
                    col_idx += num_cols

                if not motion_data:
                    continue

                for loc in self.body_locations:
                    if loc not in motion_data: continue

                    segmented_labels = self._segment_labels(labels_aligned)
                    
                    non_null_indices = np.where(segmented_labels != 0)[0]
                    if len(non_null_indices) == 0: continue
                    
                    final_labels = segmented_labels[non_null_indices] - 1

                    user_samples = [{} for _ in range(len(final_labels))]
                    for mod_name, mod_config in self.modalities.items():
                        if mod_config.get('enabled', False) and mod_name in self.available_modalities:
                            mod_info = self.available_modalities[mod_name]
                            cols = mod_info['cols']
                            
                            # --- 核心修复：根据模态选择正确的数据源 ---
                            if 'Pressure' in mod_info['file'] and pressure_data is not None:
                                source_data = pressure_data
                            else:
                                source_data = motion_data[loc]
                            
                            mod_data = source_data[:, cols]
                            segmented_mod_data = self._segment_data(mod_data)[non_null_indices]
                            
                            if self.sample_rate < 100:
                                factor = 100 // self.sample_rate
                                segmented_mod_data = signal.decimate(segmented_mod_data, factor, axis=1)

                            for i in range(len(final_labels)):
                                user_samples[i][mod_name] = segmented_mod_data[i].astype(np.float32)
                    
                    all_data.extend(user_samples)
                    all_labels.extend(final_labels.tolist())

        print(f"解析完成。为 '{split}' 分割找到 {len(all_data)} 个样本。")
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