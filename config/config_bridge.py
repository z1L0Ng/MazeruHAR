# config/config_bridge.py
"""
配置桥接器 - 用于桥接现有硬编码配置和新的配置系统
提供向后兼容性，支持渐进式迁移
"""

import os
import sys
from pathlib import Path
from .config_loader import ConfigLoader, create_sample_config

class ConfigBridge:
    """配置桥接器 - 使现有代码能够逐步迁移到新配置系统"""
    
    def __init__(self, config_path=None):
        """
        初始化配置桥接器
        
        Args:
            config_path: YAML配置文件路径，如果为None则使用传统的硬编码方式
        """
        self.config_path = config_path
        self.config = None
        self.use_new_config = config_path is not None
        
        if self.use_new_config:
            try:
                self.config = ConfigLoader.load_from_yaml(config_path)
                print(f"✓ 成功加载配置文件: {config_path}")
            except Exception as e:
                print(f"✗ 配置文件加载失败: {e}")
                print("  回退到硬编码配置")
                self.use_new_config = False
                self.config = None
    
    def get_dataset_config(self):
        """获取数据集配置"""
        if self.use_new_config:
            return {
                'name': self.config.dataset.name,
                'path': self.config.dataset.path,
                'activity_labels': self.config.dataset.activity_labels,
                'modalities': self.config.dataset.modalities,
                'train_split': self.config.dataset.train_split,
                'val_split': self.config.dataset.val_split,
                'test_split': self.config.dataset.test_split
            }
        else:
            # 返回None，让现有代码使用硬编码配置
            return None
    
    def get_training_config(self):
        """获取训练配置"""
        if self.use_new_config:
            return {
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'epochs': self.config.training.epochs,
                'optimizer': self.config.training.optimizer,
                'scheduler': self.config.training.scheduler,
                'weight_decay': self.config.training.weight_decay,
                'label_smoothing': self.config.training.label_smoothing,
                'gradient_clip_norm': self.config.training.gradient_clip_norm,
                'early_stopping_patience': self.config.training.early_stopping_patience
            }
        else:
            return None
    
    def get_architecture_config(self):
        """获取架构配置"""
        if self.use_new_config:
            return {
                'experts': self.config.architecture.experts,
                'fusion': self.config.architecture.fusion,
                'fusion_output_dim': self.config.architecture.fusion_output_dim,
                'dropout_rate': self.config.architecture.dropout_rate
            }
        else:
            return None
    
    def get_experiment_config(self):
        """获取实验配置"""
        if self.use_new_config:
            return {
                'name': self.config.name,
                'device': self.config.device,
                'seed': self.config.seed,
                'output_dir': self.config.output_dir,
                'save_checkpoints': self.config.save_checkpoints,
                'verbose': self.config.verbose
            }
        else:
            return None
    
    def create_compatible_config_from_notebook_vars(self, 
                                                   dataset_name, 
                                                   architecture, 
                                                   batch_size, 
                                                   learning_rate, 
                                                   local_epoch,
                                                   projection_dim=192,
                                                   frame_length=16,
                                                   time_step=16,
                                                   dropout_rate=0.1,
                                                   token_based=True,
                                                   random_seed=42,
                                                   **kwargs):
        """
        从notebook变量创建兼容的配置
        这个方法帮助从现有的硬编码变量创建配置对象
        """
        # 创建基本配置字典
        config_dict = {
            'name': f'{dataset_name}_{architecture}_experiment',
            'dataset': {
                'name': dataset_name,
                'path': f'./datasets/{dataset_name}',
                'activity_labels': self._get_activity_labels(dataset_name),
                'modalities': self._get_default_modalities(dataset_name, architecture, projection_dim)
            },
            'architecture': {
                'experts': self._get_default_experts(dataset_name, architecture, projection_dim),
                'fusion': {'strategy': 'concatenate', 'params': {}},
                'fusion_output_dim': self._calculate_fusion_output_dim(dataset_name, architecture, projection_dim),
                'dropout_rate': dropout_rate
            },
            'training': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': local_epoch,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'weight_decay': 1e-4,
                'label_smoothing': 0.1,
                'gradient_clip_norm': 1.0,
                'early_stopping_patience': 10
            },
            'device': 'auto',
            'seed': random_seed,
            'output_dir': f'./results/{dataset_name}_{architecture}',
            'save_checkpoints': True,
            'verbose': True
        }
        
        # 从字典创建配置对象
        self.config = ConfigLoader.load_from_dict(config_dict)
        self.use_new_config = True
        
        return self.config
    
    def _get_activity_labels(self, dataset_name):
        """获取数据集的活动标签"""
        activity_labels = {
            'UCI': ['Walking', 'Upstair', 'Downstair', 'Sitting', 'Standing', 'Lying'],
            'RealWorld': ['Downstairs', 'Upstairs', 'Jumping', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking'],
            'MotionSense': ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'],
            'HHAR': ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstairs', 'Biking'],
            'SHL': ['Standing', 'Walking', 'Running', 'Biking', 'Car', 'Bus', 'Train', 'Subway']
        }
        return activity_labels.get(dataset_name, ['Unknown'])
    
    def _get_default_modalities(self, dataset_name, architecture, projection_dim):
        """获取默认的模态配置"""
        # 根据数据集和架构创建默认模态
        if dataset_name == 'SHL':
            return [
                {
                    'name': 'imu',
                    'channels': 6,
                    'sequence_length': 200,
                    'expert_type': self._normalize_architecture_name(architecture),
                    'expert_params': {
                        'projection_dim': projection_dim,
                        'num_heads': 4,
                        'num_layers': 3
                    },
                    'preprocessing': {'normalize': True}
                },
                {
                    'name': 'pressure',
                    'channels': 1,
                    'sequence_length': 200,
                    'expert_type': 'lstm',
                    'expert_params': {
                        'hidden_dim': 64,
                        'num_layers': 2
                    },
                    'preprocessing': {'normalize': True}
                }
            ]
        else:
            return [
                {
                    'name': 'accelerometer',
                    'channels': 6,  # 通常是加速度计+陀螺仪
                    'sequence_length': 200,
                    'expert_type': self._normalize_architecture_name(architecture),
                    'expert_params': {
                        'projection_dim': projection_dim,
                        'num_heads': 4,
                        'num_layers': 3
                    },
                    'preprocessing': {'normalize': True}
                }
            ]
    
    def _get_default_experts(self, dataset_name, architecture, projection_dim):
        """获取默认的专家配置"""
        expert_type = self._normalize_architecture_name(architecture)
        
        if dataset_name == 'SHL':
            return {
                'imu': {
                    'type': expert_type,
                    'params': {
                        'projection_dim': projection_dim,
                        'num_heads': 4,
                        'num_layers': 3,
                        'output_dim': projection_dim
                    }
                },
                'pressure': {
                    'type': 'lstm',
                    'params': {
                        'hidden_dim': 64,
                        'num_layers': 2,
                        'output_dim': 64
                    }
                }
            }
        else:
            return {
                'accelerometer': {
                    'type': expert_type,
                    'params': {
                        'projection_dim': projection_dim,
                        'num_heads': 4,
                        'num_layers': 3,
                        'output_dim': projection_dim
                    }
                }
            }
    
    def _normalize_architecture_name(self, architecture):
        """标准化架构名称"""
        arch_map = {
            'HART': 'cbranchformer',
            'MobileHART': 'mobilenet',
            'Transformer': 'transformer',
            'LSTM': 'lstm',
            'CNN': 'cnn'
        }
        return arch_map.get(architecture, architecture.lower())
    
    def _calculate_fusion_output_dim(self, dataset_name, architecture, projection_dim):
        """计算融合输出维度"""
        if dataset_name == 'SHL':
            return projection_dim + 64  # IMU + pressure
        else:
            return projection_dim  # 单个模态
    
    def get_legacy_variables(self):
        """获取传统变量格式（用于向后兼容）"""
        if not self.use_new_config:
            return {}
        
        # 从配置中提取传统变量
        legacy_vars = {
            'dataset_name': self.config.dataset.name,
            'batch_size': self.config.training.batch_size,
            'learning_rate': self.config.training.learning_rate,
            'local_epoch': self.config.training.epochs,
            'random_seed': self.config.seed,
            'dropout_rate': self.config.architecture.dropout_rate,
            'activity_labels': self.config.dataset.activity_labels,
            'output_dir': self.config.output_dir
        }
        
        # 从第一个专家获取架构参数
        if self.config.architecture.experts:
            first_expert = next(iter(self.config.architecture.experts.values()))
            legacy_vars.update({
                'projection_dim': first_expert.params.get('projection_dim', 192),
                'num_heads': first_expert.params.get('num_heads', 4),
                'num_layers': first_expert.params.get('num_layers', 3),
                'architecture': first_expert.type.upper()
            })
        
        return legacy_vars


# 配置转换和兼容性函数
def convert_notebook_to_config(notebook_vars):
    """
    将notebook中的变量转换为配置文件
    
    Args:
        notebook_vars: 包含notebook变量的字典
    
    Returns:
        配置对象
    """
    bridge = ConfigBridge()
    return bridge.create_compatible_config_from_notebook_vars(**notebook_vars)


def load_config_or_use_defaults(config_path=None, **notebook_vars):
    """
    加载配置文件或使用默认值
    
    Args:
        config_path: 配置文件路径（可选）
        **notebook_vars: notebook中的变量
    
    Returns:
        ConfigBridge对象
    """
    if config_path and os.path.exists(config_path):
        print(f"✓ 使用配置文件: {config_path}")
        return ConfigBridge(config_path)
    else:
        if config_path:
            print(f"⚠ 配置文件不存在: {config_path}")
        print("使用硬编码配置模式")
        bridge = ConfigBridge()
        if notebook_vars:
            print("从notebook变量创建配置...")
            bridge.create_compatible_config_from_notebook_vars(**notebook_vars)
        return bridge


def create_config_from_current_notebook():
    """
    从当前notebook环境创建配置
    这个函数会尝试读取notebook中的全局变量
    """
    import inspect
    
    # 获取调用者的框架
    frame = inspect.currentframe().f_back
    
    # 尝试从全局变量中获取配置参数
    try:
        notebook_vars = {}
        
        # 必需的变量
        required_vars = ['dataset_name', 'architecture', 'batch_size', 'learning_rate', 'local_epoch']
        for var in required_vars:
            if var in frame.f_globals:
                notebook_vars[var] = frame.f_globals[var]
        
        # 可选的变量
        optional_vars = ['projection_dim', 'frame_length', 'time_step', 'dropout_rate', 
                        'token_based', 'random_seed', 'filter_attention_head', 'conv_kernels']
        for var in optional_vars:
            if var in frame.f_globals:
                notebook_vars[var] = frame.f_globals[var]
        
        # 检查是否有足够的变量
        if len(notebook_vars) >= len(required_vars):
            bridge = ConfigBridge()
            bridge.create_compatible_config_from_notebook_vars(**notebook_vars)
            return bridge
        else:
            print("⚠ 无法从notebook环境获取足够的配置变量")
            return ConfigBridge()
            
    except Exception as e:
        print(f"⚠ 从notebook环境创建配置失败: {e}")
        return ConfigBridge()


# 配置文件生成器
def generate_config_file(dataset_name, architecture, output_path=None, **kwargs):
    """
    生成配置文件模板
    
    Args:
        dataset_name: 数据集名称
        architecture: 架构名称
        output_path: 输出文件路径
        **kwargs: 其他配置参数
    """
    if output_path is None:
        output_path = f"config/generated/{dataset_name}_{architecture.lower()}_config.yaml"
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 默认参数
    default_params = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'local_epoch': 100,
        'projection_dim': 192,
        'dropout_rate': 0.1,
        'random_seed': 42
    }
    
    # 合并参数
    params = {**default_params, **kwargs}
    
    # 创建配置
    bridge = ConfigBridge()
    config = bridge.create_compatible_config_from_notebook_vars(
        dataset_name=dataset_name,
        architecture=architecture,
        **params
    )
    
    # 保存配置
    ConfigLoader.save_to_yaml(config, output_path)
    print(f"✓ 配置文件已生成: {output_path}")
    
    return output_path


# 批量配置生成器
def generate_batch_configs(base_config_path, param_grid, output_dir="config/experiments"):
    """
    批量生成实验配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        param_grid: 参数网格字典
        output_dir: 输出目录
    
    Returns:
        生成的配置文件路径列表
    """
    import itertools
    
    # 加载基础配置
    base_config = ConfigLoader.load_from_yaml(base_config_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = list(itertools.product(*values))
    
    generated_configs = []
    
    for i, params in enumerate(param_combinations):
        # 创建参数字典
        param_dict = dict(zip(keys, params))
        
        # 复制基础配置
        config_dict = ConfigLoader._config_to_dict(base_config)
        
        # 更新参数
        for key, value in param_dict.items():
            if key in ['batch_size', 'learning_rate', 'epochs']:
                config_dict['training'][key] = value
            elif key in ['dropout_rate']:
                config_dict['architecture'][key] = value
            elif key in ['projection_dim']:
                # 更新所有专家的projection_dim
                for expert_name in config_dict['architecture']['experts']:
                    config_dict['architecture']['experts'][expert_name]['params']['projection_dim'] = value
        
        # 更新实验名称
        param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
        config_dict['name'] = f"{base_config.name}_{param_str}"
        
        # 保存配置
        output_path = os.path.join(output_dir, f"exp_{i:03d}_{param_str}.yaml")
        config = ConfigLoader.load_from_dict(config_dict)
        ConfigLoader.save_to_yaml(config, output_path)
        
        generated_configs.append(output_path)
    
    print(f"✓ 批量生成了 {len(generated_configs)} 个配置文件")
    return generated_configs


# 使用示例
if __name__ == '__main__':
    # 示例1: 使用新的配置系统
    print("=== 示例1: 使用配置文件 ===")
    bridge = ConfigBridge('config/default_configs/shl_config.yaml')
    if bridge.use_new_config:
        dataset_config = bridge.get_dataset_config()
        print(f"数据集: {dataset_config['name']}")
        print(f"活动标签: {dataset_config['activity_labels']}")
    
    # 示例2: 从notebook变量创建配置
    print("\n=== 示例2: 从notebook变量创建配置 ===")
    notebook_vars = {
        'dataset_name': 'SHL',
        'architecture': 'HART',
        'batch_size': 32,
        'learning_rate': 0.001,
        'local_epoch': 100,
        'projection_dim': 192
    }
    
    bridge = ConfigBridge()
    config = bridge.create_compatible_config_from_notebook_vars(**notebook_vars)
    print(f"创建的配置: {config.name}")
    
    # 示例3: 向后兼容模式
    print("\n=== 示例3: 向后兼容模式 ===")
    bridge = load_config_or_use_defaults(**notebook_vars)
    legacy_vars = bridge.get_legacy_variables()
    print(f"传统变量: {legacy_vars}")
    
    # 示例4: 生成配置文件
    print("\n=== 示例4: 生成配置文件 ===")
    config_path = generate_config_file(
        dataset_name='UCI',
        architecture='HART',
        output_path='config/generated/uci_hart_config.yaml',
        batch_size=64,
        learning_rate=0.0001
    )
    
    # 示例5: 批量生成实验配置
    print("\n=== 示例5: 批量生成实验配置 ===")
    param_grid = {
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'dropout_rate': [0.1, 0.2]
    }
    
    if os.path.exists('config/generated/uci_hart_config.yaml'):
        batch_configs = generate_batch_configs(
            'config/generated/uci_hart_config.yaml',
            param_grid,
            'config/experiments'
        )
        print(f"生成的配置文件: {batch_configs[:3]}...")  # 显示前3个
    
    print("\n✓ 配置桥接器演示完成!")