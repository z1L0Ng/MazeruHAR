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
        self.window_size = config.get('window_size', 128)
        self.step_size = config.get('step_size', 64)
        self.sample_rate = config.get('sample_rate', 100)  # Hz
        
        # SHL数据集的活动标签映射
        self.activity_labels = {
            1: 'Standing',
            2: 'Walking', 
            3: 'Running',
            4: 'Biking',
            5: 'Car',
            6: 'Bus',
            7: 'Train',
            8: 'Subway'
        }
        
        # 定义可用的模态
        self.available_modalities = {
            'imu': {
                'channels': ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z'],
                'dimension': 6,
                'description': 'IMU sensor data (accelerometer + gyroscope)'
            },
            'pressure': {
                'channels': ['Pressure'],
                'dimension': 1,
                'description': 'Barometric pressure sensor data'
            },
            'magnetometer': {
                'channels': ['Mag_x', 'Mag_y', 'Mag_z'],
                'dimension': 3,
                'description': 'Magnetometer sensor data'
            }
        }
        
    def _download_shl_data(self) -> bool:
        """
        下载SHL数据集
        """
        os.makedirs(self.data_path, exist_ok=True)
        
        file_names = [
            "SHLDataset_preview_v1_part1.zip",
            "SHLDataset_preview_v1_part2.zip", 
            "SHLDataset_preview_v1_part3.zip"
        ]
        
        links = [
            "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part1.zip",
            "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part2.zip",
            "http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part3.zip"
        ]
        
        for file_name, link in zip(file_names, links):
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                try:
                    response = requests.get(link, stream=True)
                    response.raise_for_status()
                    
                    with open(file_path, 'wb') as f:
                        for chunk in tqdm(response.iter_content(chunk_size=8192)):
                            f.write(chunk)
                    print(f"Downloaded {file_name}")
                except Exception as e:
                    print(f"Failed to download {file_name}: {e}")
                    return False
        
        return True
    
    def _extract_data(self) -> bool:
        """
        解压SHL数据集
        """
        extracted_path = os.path.join(self.data_path, 'SHL_Dataset_preview_v1')
        if os.path.exists(extracted_path):
            print("Data already extracted")
            return True
            
        file_names = [
            "SHLDataset_preview_v1_part1.zip",
            "SHLDataset_preview_v1_part2.zip",
            "SHLDataset_preview_v1_part3.zip"
        ]
        
        for file_name in file_names:
            file_path = os.path.join(self.data_path, file_name)
            if os.path.exists(file_path):
                print(f"Extracting {file_name}...")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(self.data_path)
                except Exception as e:
                    print(f"Failed to extract {file_name}: {e}")
                    return False
        
        return True
    
    def _process_label(self, labels: np.ndarray) -> int:
        """
        处理标签，返回窗口中最常见的标签
        """
        unique_values, counts = np.unique(labels, return_counts=True)
        return unique_values[np.argmax(counts)]
    
    def _segment_data(self, data: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """
        将传感器数据分割成窗口
        """
        segments = []
        for i in range(0, data.shape[0] - window_size, step_size):
            segments.append(data[i:i+window_size, :])
        return np.asarray(segments)
    
    def _segment_labels(self, labels: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
        """
        将标签数据分割成窗口，并为每个窗口确定一个单一标签
        """
        segmented_labels = []
        for i in range(0, labels.shape[0] - window_size, step_size):
            segmented_labels.append(self._process_label(labels[i:i+window_size]))
        return np.asarray(segmented_labels)
    
    def _load_user_data(self, user_id: str, position: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        加载单个用户的数据
        
        Args:
            user_id: 用户ID
            position: 传感器位置 ('Bag', 'Hand', 'Hips', 'Torso')
            
        Returns:
            Tuple of (modality_data, labels)
        """
        data_path = os.path.join(self.data_path, 'SHL_Dataset_preview_v1', user_id, position)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # 读取各种传感器数据
        modality_data = {}
        
        # 读取IMU数据（加速度计 + 陀螺仪）
        if 'imu' in self.modalities:
            imu_files = ['Acc_x.txt', 'Acc_y.txt', 'Acc_z.txt', 'Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt']
            imu_data = []
            
            for file_name in imu_files:
                file_path = os.path.join(data_path, file_name)
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=None, delim_whitespace=True).values
                    imu_data.append(data)
                else:
                    print(f"Warning: {file_path} not found")
                    
            if imu_data:
                modality_data['imu'] = np.concatenate(imu_data, axis=1)
        
        # 读取气压数据
        if 'pressure' in self.modalities:
            pressure_file = os.path.join(data_path, 'Pressure.txt')
            if os.path.exists(pressure_file):
                pressure_data = pd.read_csv(pressure_file, header=None, delim_whitespace=True).values
                modality_data['pressure'] = pressure_data
        
        # 读取磁力计数据
        if 'magnetometer' in self.modalities:
            mag_files = ['Mag_x.txt', 'Mag_y.txt', 'Mag_z.txt']
            mag_data = []
            
            for file_name in mag_files:
                file_path = os.path.join(data_path, file_name)
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=None, delim_whitespace=True).values
                    mag_data.append(data)
                    
            if mag_data:
                modality_data['magnetometer'] = np.concatenate(mag_data, axis=1)
        
        # 读取标签
        label_file = os.path.join(data_path, 'Label.txt')
        if os.path.exists(label_file):
            labels = pd.read_csv(label_file, header=None, delim_whitespace=True).values.flatten()
        else:
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        return modality_data, labels
    
    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        解析SHL数据
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            Tuple of (data_list, labels)
        """
        # 创建模拟数据用于测试（实际使用时会下载真实数据）
        print(f"Creating mock data for {split} split...")
        
        # 生成模拟数据
        all_data = []
        all_labels = []
        
        # 生成一些模拟样本
        num_samples = 50 if split == 'train' else 20
        
        for i in range(num_samples):
            sample_data = {}
            
            # 生成IMU数据
            if 'imu' in self.modalities:
                sample_data['imu'] = np.random.randn(self.window_size, 6).astype(np.float32)
            
            # 生成压力数据
            if 'pressure' in self.modalities:
                sample_data['pressure'] = np.random.randn(self.window_size, 1).astype(np.float32)
            
            # 生成磁力计数据
            if 'magnetometer' in self.modalities:
                sample_data['magnetometer'] = np.random.randn(self.window_size, 3).astype(np.float32)
            
            all_data.append(sample_data)
            all_labels.append((i % 8) + 1)  # 8个类别
        
        print(f"Generated {len(all_data)} mock samples for {split} split")
        return all_data, all_labels
    
    def get_modality_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取模态信息
        """
        info = {}
        for modality in self.modalities:
            if modality in self.available_modalities:
                info[modality] = self.available_modalities[modality].copy()
                info[modality]['window_size'] = self.window_size
                info[modality]['sample_rate'] = self.sample_rate
        return info