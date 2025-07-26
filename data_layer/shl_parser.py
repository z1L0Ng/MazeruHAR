# 完整的SHL数据解析器修复
# 文件路径: data_layer/shl_parser.py

import os
import numpy as np
import pandas as pd
import hickle as hkl
from typing import Dict, List, Tuple, Any
from scipy import signal
import logging

# 修复导入错误 - 使用绝对导入
try:
    from base_parser import DataParser
except ImportError:
    # 如果在包内运行，尝试相对导入
    from .base_parser import DataParser


class SHLDataParser(DataParser):
    """
    SHL数据集解析器 - 修复版本
    支持从预处理的HKL文件加载数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get('window_size', 128)
        self.step_size = config.get('step_size', 64)
        self.sample_rate = config.get('sample_rate', 100)
        self.normalize_per_sample = config.get('normalize_per_sample', True)
        
        # SHL数据集的活动标签映射 (0-based)
        self.activity_labels = {
            0: 'Standing', 1: 'Walking', 2: 'Running', 3: 'Biking',
            4: 'Car', 5: 'Bus', 6: 'Train', 7: 'Subway'
        }
        
        # 模态配置
        self.modalities_config = config.get('modalities', {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        })
        
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

    def validate_and_clean_data(self, clients_data: List[np.ndarray], 
                              clients_labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """验证和清理数据"""
        self.logger.info("验证和清理数据...")
        
        # 合并所有客户端数据
        all_data = np.vstack(clients_data)
        all_labels = np.hstack(clients_labels)
        
        self.logger.info(f"合并后数据形状: {all_data.shape}")
        self.logger.info(f"合并后标签形状: {all_labels.shape}")
        
        # 过滤有效标签（0-7，排除无效值）
        valid_indices = (all_labels >= 0) & (all_labels <= 7)
        all_data = all_data[valid_indices]
        all_labels = all_labels[valid_indices]
        
        self.logger.info(f"过滤后数据形状: {all_data.shape}")
        
        # 检查标签分布
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        self.logger.info(f"标签分布: {label_distribution}")
        
        # 检查数据质量
        if np.any(np.isnan(all_data)):
            self.logger.warning("数据中包含NaN值，将被替换为0")
            all_data = np.nan_to_num(all_data)
        
        if np.any(np.isinf(all_data)):
            self.logger.warning("数据中包含Inf值，将被替换为有限值")
            all_data = np.nan_to_num(all_data)
        
        return all_data, all_labels

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        if self.normalize_per_sample:
            # 样本级标准化：每个样本独立标准化
            normalized_data = []
            for sample in data:
                # 沿时间轴计算均值和标准差
                mean = np.mean(sample, axis=0, keepdims=True)
                std = np.std(sample, axis=0, keepdims=True)
                std = np.where(std == 0, 1, std)  # 避免除零
                normalized_sample = (sample - mean) / std
                normalized_data.append(normalized_sample)
            return np.array(normalized_data)
        else:
            # 全局标准化
            original_shape = data.shape
            reshaped_data = data.reshape(-1, original_shape[-1])
            
            mean = np.mean(reshaped_data, axis=0)
            std = np.std(reshaped_data, axis=0)
            std = np.where(std == 0, 1, std)
            
            normalized_reshaped = (reshaped_data - mean) / std
            return normalized_reshaped.reshape(original_shape)

    def split_modalities(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """将多模态数据分离"""
        modalities = {}
        
        # 根据SHL数据集的结构分离模态
        if self.modalities_config.get('imu', {}).get('enabled', False):
            if data.shape[-1] >= 6:
                modalities['imu'] = data[:, :, :6]  # 前6列：IMU数据
            else:
                self.logger.warning("数据维度不足，无法提取IMU模态")
        
        if self.modalities_config.get('pressure', {}).get('enabled', False):
            if data.shape[-1] >= 7:
                modalities['pressure'] = data[:, :, 6:7]  # 第7列：压力数据
            else:
                # 如果只有6列，可能压力数据在其他位置或不存在
                self.logger.warning("数据维度不足，无法提取压力模态")
                # 创建虚拟压力数据作为占位符
                modalities['pressure'] = np.zeros((data.shape[0], data.shape[1], 1))
        
        return modalities

    def split_dataset(self, data: np.ndarray, labels: np.ndarray, 
                     split: str) -> Tuple[np.ndarray, np.ndarray]:
        """按比例划分数据集"""
        total_samples = len(labels)
        
        # 设置随机种子以确保可重现的划分
        np.random.seed(42)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # 计算划分点
        train_end = int(total_samples * self.split_ratios['train'])
        val_end = train_end + int(total_samples * self.split_ratios['val'])
        
        if split == 'train':
            split_indices = indices[:train_end]
        elif split == 'val':
            split_indices = indices[train_end:val_end]
        elif split == 'test':
            split_indices = indices[val_end:]
        else:
            raise ValueError(f"无效的split参数: {split}")
        
        split_data = data[split_indices]
        split_labels = labels[split_indices]
        
        self.logger.info(f"{split} 数据集: {len(split_labels)} 样本")
        
        # 打印标签分布
        unique_labels, counts = np.unique(split_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        self.logger.info(f"{split} 标签分布: {label_distribution}")
        
        return split_data, split_labels

    def parse_data(self, split: str) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        解析SHL数据的主要方法
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            Tuple of (data_list, labels):
                - data_list: 包含数据字典的列表
                - labels: 对应的标签列表
        """
        self.logger.info(f"开始解析 {split} 数据集...")
        
        try:
            # 1. 加载预处理数据
            clients_data, clients_labels = self.load_preprocessed_data()
            
            # 2. 验证和清理数据
            all_data, all_labels = self.validate_and_clean_data(clients_data, clients_labels)
            
            # 3. 数据标准化
            self.logger.info("执行数据标准化...")
            all_data = self.normalize_data(all_data)
            
            # 4. 划分数据集
            split_data, split_labels = self.split_dataset(all_data, all_labels, split)
            
            # 5. 分离多模态数据
            self.logger.info("分离多模态数据...")
            split_modalities = self.split_modalities(split_data)
            
            # 6. 构建数据列表
            data_list = []
            labels_list = []
            
            num_samples = len(split_labels)
            for i in range(num_samples):
                sample_dict = {}
                for modality_name, modality_data in split_modalities.items():
                    sample_dict[modality_name] = modality_data[i]
                
                data_list.append(sample_dict)
                labels_list.append(int(split_labels[i]))
            
            self.logger.info(f"成功解析 {len(data_list)} 个样本")
            
            # 验证数据完整性
            if len(data_list) != len(labels_list):
                raise ValueError("数据和标签数量不匹配")
            
            if len(data_list) == 0:
                raise ValueError(f"没有找到有效的 {split} 数据")
            
            # 打印第一个样本的信息用于调试
            first_sample = data_list[0]
            self.logger.info(f"第一个样本的模态: {list(first_sample.keys())}")
            for modality, data in first_sample.items():
                self.logger.info(f"  {modality}: 形状 {data.shape}")
            
            return data_list, labels_list
            
        except Exception as e:
            self.logger.error(f"解析 {split} 数据时发生错误: {e}")
            raise e

    def get_modality_info(self) -> Dict[str, Dict[str, Any]]:
        """获取模态信息"""
        modality_info = {}
        
        for modality_name, config in self.modalities_config.items():
            if config.get('enabled', False):
                modality_info[modality_name] = {
                    'channels': config.get('channels', 1),
                    'sequence_length': self.window_size,
                    'enabled': True
                }
        
        return modality_info

    def get_activity_labels(self) -> Dict[int, str]:
        """获取活动标签映射"""
        return self.activity_labels

    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.activity_labels)


# 快速测试函数
def test_shl_parser():
    """测试SHL解析器"""
    print("测试SHL数据解析器...")
    
    config = {
        'name': 'SHL',
        'data_path': 'datasets/datasetStandardized/SHL_Multimodal/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'normalize_per_sample': True,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        }
    }
    
    try:
        parser = SHLDataParser(config)
        
        # 测试训练数据解析
        train_data, train_labels = parser.parse_data('train')
        print(f"✓ 训练数据解析成功: {len(train_data)} 样本")
        
        # 测试验证数据解析
        val_data, val_labels = parser.parse_data('val')
        print(f"✓ 验证数据解析成功: {len(val_data)} 样本")
        
        # 测试测试数据解析
        test_data, test_labels = parser.parse_data('test')
        print(f"✓ 测试数据解析成功: {len(test_data)} 样本")
        
        # 获取模态信息
        modality_info = parser.get_modality_info()
        print(f"✓ 模态信息: {modality_info}")
        
        print("✓ SHL数据解析器测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ SHL数据解析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    
    # 运行测试
    test_shl_parser()