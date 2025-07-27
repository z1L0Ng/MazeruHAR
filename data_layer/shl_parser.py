# 扩展的SHL数据解析器 - 最小化修改版本
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
    SHL数据集解析器 - 扩展版本
    支持从预处理的HKL文件加载数据，并支持更多传感器模态
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
        
        # 扩展的模态配置
        self.modalities_config = config.get('modalities', {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1}
        })
        
        # SHL数据集中传感器的列索引映射（基于_Motion.txt文件格式）
        self.sensor_column_mapping = {
            'accelerometer': [1, 2, 3],      # 加速度计 x, y, z
            'gyroscope': [4, 5, 6],          # 陀螺仪 x, y, z  
            'magnetometer': [7, 8, 9],       # 磁力计 x, y, z
            'orientation': [10, 11, 12, 13], # 方向四元数 w, x, y, z
            'gravity': [14, 15, 16],         # 重力 x, y, z
            'linear_acceleration': [17, 18, 19], # 线性加速度 x, y, z
            'pressure': [20]                 # 气压（假设在第20列）
        }
        
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
        
        self.logger.info(f"从 {data_file} 加载数据...")
        clients_data = hkl.load(data_file)
        
        self.logger.info(f"从 {label_file} 加载标签...")
        clients_labels = hkl.load(label_file)
        
        self.logger.info(f"加载了 {len(clients_data)} 个客户端的数据")
        
        return clients_data, clients_labels

    def validate_and_clean_data(self, clients_data: List[np.ndarray], 
                               clients_labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """验证和清理数据"""
        self.logger.info("验证和清理数据...")
        
        valid_data = []
        valid_labels = []
        
        for i, (data, labels) in enumerate(zip(clients_data, clients_labels)):
            if data is None or labels is None:
                self.logger.warning(f"跳过空数据客户端 {i}")
                continue
            
            if len(data) != len(labels):
                self.logger.warning(f"客户端 {i} 数据和标签长度不匹配，跳过")
                continue
            
            if len(data) == 0:
                self.logger.warning(f"客户端 {i} 数据为空，跳过")
                continue
            
            # 检查数据形状
            if data.ndim != 3:
                self.logger.warning(f"客户端 {i} 数据维度错误: {data.shape}")
                continue
                
            valid_data.append(data)
            valid_labels.append(labels)
        
        if not valid_data:
            raise ValueError("没有找到有效的数据")
        
        # 合并所有客户端的数据
        all_data = np.vstack(valid_data)
        all_labels = np.hstack(valid_labels)
        
        self.logger.info(f"有效数据形状: {all_data.shape}")
        self.logger.info(f"有效标签形状: {all_labels.shape}")
        
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

    # 立即替换 data_layer/shl_parser.py 中的 split_modalities 函数

def split_modalities(self, data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    将多模态数据分离 - 修复版本，支持磁力计
    """
    modalities = {}
    total_channels = data.shape[-1]
    self.logger.info(f"🔍 DEBUG: 输入数据形状={data.shape}, 总通道数={total_channels}")
    self.logger.info(f"🔍 DEBUG: 配置的模态={list(self.modalities_config.keys())}")

    # 打印每个模态的启用状态
    for modality_name, modality_config in self.modalities_config.items():
        enabled = modality_config.get('enabled', False)
        self.logger.info(f"🔍 DEBUG: 模态 {modality_name} - enabled={enabled}")

    # 检查启用的模态并分离数据
    for modality_name, modality_config in self.modalities_config.items():
        if not modality_config.get('enabled', False):
            self.logger.info(f"⏭️  跳过未启用的模态: {modality_name}")
            continue

        self.logger.info(f"🔄 正在处理启用的模态: {modality_name}")

        if modality_name == 'imu':
            # IMU模态：加速度计(3) + 陀螺仪(3) = 6通道
            if total_channels >= 6:
                acc_data = data[:, :, 0:3]    # 加速度计
                gyro_data = data[:, :, 3:6]   # 陀螺仪
                imu_data = np.concatenate([acc_data, gyro_data], axis=-1)
                modalities['imu'] = imu_data
                self.logger.info(f"✅ 提取IMU模态: {imu_data.shape}")
            else:
                self.logger.warning("❌ 数据通道不足，无法提取IMU模态")

        elif modality_name == 'pressure':
            # 压力模态：使用最后一列作为压力数据
            if total_channels >= 7:
                # 使用倒数第一列作为压力数据
                modalities['pressure'] = data[:, :, -1:]
                self.logger.info(f"✅ 提取压力模态: {modalities['pressure'].shape}")
            else:
                # 创建虚拟压力数据
                self.logger.warning("⚠️  数据通道不足，创建虚拟压力数据")
                modalities['pressure'] = np.zeros((data.shape[0], data.shape[1], 1))

        elif modality_name == 'magnetometer':
            # 磁力计模态：列6-8 (基于SHL数据集标准格式)
            self.logger.info(f"🔍 处理磁力计: total_channels={total_channels}, 需要>=9")
            if total_channels >= 9:
                magnetometer_data = data[:, :, 6:9]
                modalities['magnetometer'] = magnetometer_data
                self.logger.info(f"✅ 成功提取磁力计模态: {magnetometer_data.shape}")
                # 额外调试信息
                self.logger.info(f"🔍 磁力计数据统计: min={magnetometer_data.min():.3f}, max={magnetometer_data.max():.3f}, mean={magnetometer_data.mean():.3f}")
            else:
                self.logger.warning(f"❌ 数据通道不足({total_channels})，无法提取磁力计模态(需要9通道)")

        elif modality_name == 'orientation':
            # 方向模态：列9-12 (四元数)
            if total_channels >= 13:
                modalities['orientation'] = data[:, :, 9:13]
                self.logger.info(f"✅ 提取方向模态: {modalities['orientation'].shape}")
            else:
                self.logger.warning(f"❌ 数据通道不足({total_channels})，无法提取方向模态(需要13通道)")

        elif modality_name == 'gravity':
            # 重力模态：列13-15
            if total_channels >= 16:
                modalities['gravity'] = data[:, :, 13:16]
                self.logger.info(f"✅ 提取重力模态: {modalities['gravity'].shape}")
            else:
                self.logger.warning(f"❌ 数据通道不足({total_channels})，无法提取重力模态(需要16通道)")

        elif modality_name == 'linear_acceleration':
            # 线性加速度模态：列16-18
            if total_channels >= 19:
                modalities['linear_acceleration'] = data[:, :, 16:19]
                self.logger.info(f"✅ 提取线性加速度模态: {modalities['linear_acceleration'].shape}")
            else:
                self.logger.warning(f"❌ 数据通道不足({total_channels})，无法提取线性加速度模态(需要19通道)")

        else:
            self.logger.warning(f"⚠️  未知的模态类型: {modality_name}")

    if not modalities:
        raise ValueError("❌ 没有成功提取任何模态数据")

    # 打印最终提取的模态总结
    self.logger.info(f"🎯 最终成功提取的模态: {list(modalities.keys())}")
    for modality_name, modality_data in modalities.items():
        self.logger.info(f"   📊 {modality_name}: {modality_data.shape}")

    return modalities


# 快速测试函数
def test_shl_parser():
    """测试SHL解析器"""
    print("测试扩展的SHL数据解析器...")
    
    # 测试配置 - 包含更多传感器
    config = {
        'name': 'SHL',
        'data_path': 'datasets/datasetStandardized/SHL_Multimodal/',
        'window_size': 128,
        'step_size': 64,
        'sample_rate': 100,
        'normalize_per_sample': True,
        'modalities': {
            'imu': {'enabled': True, 'channels': 6},
            'pressure': {'enabled': True, 'channels': 1},
            'magnetometer': {'enabled': True, 'channels': 3},
            'orientation': {'enabled': False, 'channels': 4},  # 可选启用
            'gravity': {'enabled': False, 'channels': 3},      # 可选启用
            'linear_acceleration': {'enabled': False, 'channels': 3}  # 可选启用
        }
    }
    
    try:
        parser = SHLDataParser(config)
        
        # 测试训练数据解析
        train_data, train_labels = parser.parse_data('train')
        print(f"✓ 训练数据解析成功: {len(train_data)} 样本")
        
        # 获取模态信息
        modality_info = parser.get_modality_info()
        print(f"✓ 支持的模态: {list(modality_info.keys())}")
        
        # 打印第一个样本的模态信息
        if train_data:
            first_sample = train_data[0]
            print("✓ 第一个样本包含的模态:")
            for modality, data in first_sample.items():
                print(f"    {modality}: {data.shape}")
        
        print("✓ 扩展的SHL数据解析器测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 扩展的SHL数据解析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    
    # 运行测试
    test_shl_parser()