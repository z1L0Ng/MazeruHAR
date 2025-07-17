# config/config_loader.py
"""
配置层实现 - 任务1.1
MazeruHAR项目的配置驱动框架核心组件
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ModalityConfig:
    """单个模态配置"""
    name: str
    channels: int
    sequence_length: int
    expert_type: str
    expert_params: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 验证必要字段
        if self.channels <= 0:
            raise ValueError(f"Modality '{self.name}' channels must be > 0")
        if self.sequence_length <= 0:
            raise ValueError(f"Modality '{self.name}' sequence_length must be > 0")


@dataclass
class ExpertConfig:
    """专家模型配置"""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 验证专家类型
        valid_types = ['transformer', 'cnn', 'lstm', 'gru', 'cbranchformer', 'hart']
        if self.type not in valid_types:
            raise ValueError(f"Expert type '{self.type}' not supported. Valid types: {valid_types}")


@dataclass
class FusionConfig:
    """融合策略配置"""
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 验证融合策略
        valid_strategies = ['concatenate', 'attention', 'weighted_sum', 'average', 'max_pooling']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Fusion strategy '{self.strategy}' not supported. Valid strategies: {valid_strategies}")


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    path: str
    modalities: List[ModalityConfig]
    activity_labels: List[str]
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    def __post_init__(self):
        # 验证数据集分割比例
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Dataset splits must sum to 1.0, got {total_split}")
        
        # 验证数据集路径（如果路径存在的话）
        if os.path.exists(self.path):
            pass  # 路径存在，验证通过
        else:
            # 路径不存在，给出警告但不抛出异常（可能是相对路径或稍后创建）
            print(f"Warning: Dataset path '{self.path}' does not exist")


@dataclass
class ArchitectureConfig:
    """模型架构配置"""
    experts: Dict[str, ExpertConfig]
    fusion: FusionConfig
    fusion_output_dim: int
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.fusion_output_dim <= 0:
            raise ValueError("fusion_output_dim must be > 0")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be in [0, 1]")


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    name: str
    dataset: DatasetConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    device: str = 'auto'
    seed: int = 42
    output_dir: str = './results'
    save_checkpoints: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 验证模态与专家的映射
        dataset_modalities = {mod.name for mod in self.dataset.modalities}
        expert_modalities = set(self.architecture.experts.keys())
        
        if dataset_modalities != expert_modalities:
            raise ValueError(f"Modality-Expert mapping mismatch. "
                           f"Dataset modalities: {dataset_modalities}, "
                           f"Expert modalities: {expert_modalities}")


class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> ExperimentConfig:
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """从字典加载配置"""
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """将配置字典转换为ExperimentConfig对象"""
        
        # 解析模态配置
        modalities = []
        for mod_config in config_dict['dataset']['modalities']:
            modalities.append(ModalityConfig(**mod_config))
        
        # 解析数据集配置
        dataset_config = DatasetConfig(
            name=config_dict['dataset']['name'],
            path=config_dict['dataset']['path'],
            modalities=modalities,
            activity_labels=config_dict['dataset']['activity_labels'],
            train_split=config_dict['dataset'].get('train_split', 0.7),
            val_split=config_dict['dataset'].get('val_split', 0.15),
            test_split=config_dict['dataset'].get('test_split', 0.15)
        )
        
        # 解析专家配置
        experts = {}
        for expert_name, expert_config in config_dict['architecture']['experts'].items():
            experts[expert_name] = ExpertConfig(**expert_config)
        
        # 解析融合配置
        fusion_config = FusionConfig(**config_dict['architecture']['fusion'])
        
        # 解析架构配置
        architecture_config = ArchitectureConfig(
            experts=experts,
            fusion=fusion_config,
            fusion_output_dim=config_dict['architecture']['fusion_output_dim'],
            dropout_rate=config_dict['architecture'].get('dropout_rate', 0.1)
        )
        
        # 解析训练配置
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        # 构建完整配置
        experiment_config = ExperimentConfig(
            name=config_dict['name'],
            dataset=dataset_config,
            architecture=architecture_config,
            training=training_config,
            device=config_dict.get('device', 'auto'),
            seed=config_dict.get('seed', 42),
            output_dir=config_dict.get('output_dir', './results'),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            verbose=config_dict.get('verbose', True)
        )
        
        return experiment_config
    
    @staticmethod
    def save_to_yaml(config: ExperimentConfig, yaml_path: str):
        """保存配置到YAML文件"""
        config_dict = ConfigLoader._config_to_dict(config)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        
        # 转换模态配置
        modalities_dict = []
        for mod in config.dataset.modalities:
            modalities_dict.append({
                'name': mod.name,
                'channels': mod.channels,
                'sequence_length': mod.sequence_length,
                'expert_type': mod.expert_type,
                'expert_params': mod.expert_params,
                'preprocessing': mod.preprocessing
            })
        
        # 转换专家配置
        experts_dict = {}
        for expert_name, expert in config.architecture.experts.items():
            experts_dict[expert_name] = {
                'type': expert.type,
                'params': expert.params
            }
        
        return {
            'name': config.name,
            'dataset': {
                'name': config.dataset.name,
                'path': config.dataset.path,
                'modalities': modalities_dict,
                'activity_labels': config.dataset.activity_labels,
                'train_split': config.dataset.train_split,
                'val_split': config.dataset.val_split,
                'test_split': config.dataset.test_split
            },
            'architecture': {
                'experts': experts_dict,
                'fusion': {
                    'strategy': config.architecture.fusion.strategy,
                    'params': config.architecture.fusion.params
                },
                'fusion_output_dim': config.architecture.fusion_output_dim,
                'dropout_rate': config.architecture.dropout_rate
            },
            'training': {
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'epochs': config.training.epochs,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler,
                'weight_decay': config.training.weight_decay,
                'label_smoothing': config.training.label_smoothing,
                'gradient_clip_norm': config.training.gradient_clip_norm,
                'early_stopping_patience': config.training.early_stopping_patience
            },
            'device': config.device,
            'seed': config.seed,
            'output_dir': config.output_dir,
            'save_checkpoints': config.save_checkpoints,
            'verbose': config.verbose
        }


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_config(config: ExperimentConfig) -> List[str]:
        """验证配置的完整性和一致性"""
        errors = []
        
        # 验证模态与专家匹配
        for modality in config.dataset.modalities:
            if modality.name not in config.architecture.experts:
                errors.append(f"No expert defined for modality '{modality.name}'")
        
        # 验证融合输出维度
        if config.architecture.fusion.strategy == 'concatenate':
            total_expert_output_dim = 0
            for expert_name, expert in config.architecture.experts.items():
                # 获取专家输出维度
                expert_output_dim = expert.params.get('output_dim', 128)
                total_expert_output_dim += expert_output_dim
            
            if config.architecture.fusion_output_dim != total_expert_output_dim:
                errors.append(f"Fusion output dim mismatch for concatenate strategy. "
                            f"Expected {total_expert_output_dim}, got {config.architecture.fusion_output_dim}")
        
        # 验证设备配置
        if config.device not in ['auto', 'cpu', 'cuda', 'mps']:
            errors.append(f"Invalid device '{config.device}'. Valid options: auto, cpu, cuda, mps")
        
        return errors


def create_sample_config() -> ExperimentConfig:
    """创建示例配置"""
    
    # 创建模态配置
    modalities = [
        ModalityConfig(
            name='imu',
            channels=6,
            sequence_length=200,
            expert_type='cbranchformer',
            expert_params={'projection_dim': 128, 'num_heads': 4},
            preprocessing={'normalize': True, 'filter_freq': 20}
        ),
        ModalityConfig(
            name='pressure',
            channels=1,
            sequence_length=200,
            expert_type='lstm',
            expert_params={'hidden_dim': 64, 'num_layers': 2},
            preprocessing={'normalize': True}
        )
    ]
    
    # 创建数据集配置
    dataset_config = DatasetConfig(
        name='SHL',
        path='./datasets/datasetStandardized/SHL_Multimodal',
        modalities=modalities,
        activity_labels=['Standing', 'Walking', 'Running', 'Biking', 'Car', 'Bus', 'Train', 'Subway']
    )
    
    # 创建专家配置
    experts = {
        'imu': ExpertConfig(
            type='cbranchformer',
            params={
                'projection_dim': 128,
                'num_heads': 4,
                'num_layers': 3,
                'output_dim': 128
            }
        ),
        'pressure': ExpertConfig(
            type='lstm',
            params={
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 64
            }
        )
    }
    
    # 创建融合配置
    fusion_config = FusionConfig(
        strategy='concatenate',
        params={}
    )
    
    # 创建架构配置
    architecture_config = ArchitectureConfig(
        experts=experts,
        fusion=fusion_config,
        fusion_output_dim=192,  # 128 + 64
        dropout_rate=0.1
    )
    
    # 创建训练配置
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        epochs=100,
        optimizer='adam',
        scheduler='cosine'
    )
    
    # 创建完整配置
    experiment_config = ExperimentConfig(
        name='SHL_MultiModal_Experiment',
        dataset=dataset_config,
        architecture=architecture_config,
        training=training_config,
        device='auto',
        seed=42,
        output_dir='./results/shl_multimodal'
    )
    
    return experiment_config


if __name__ == '__main__':
    # 示例使用
    
    # 创建示例配置
    config = create_sample_config()
    
    # 验证配置
    validator = ConfigValidator()
    errors = validator.validate_config(config)
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过!")
    
    # 保存配置到YAML文件
    ConfigLoader.save_to_yaml(config, 'sample_config.yaml')
    print("示例配置已保存到 sample_config.yaml")
    
    # 从YAML文件加载配置
    loaded_config = ConfigLoader.load_from_yaml('sample_config.yaml')
    print(f"从YAML加载的配置: {loaded_config.name}")
    
    # 打印配置摘要
    print("\n配置摘要:")
    print(f"  实验名称: {loaded_config.name}")
    print(f"  数据集: {loaded_config.dataset.name}")
    print(f"  模态数量: {len(loaded_config.dataset.modalities)}")
    print(f"  专家数量: {len(loaded_config.architecture.experts)}")
    print(f"  融合策略: {loaded_config.architecture.fusion.strategy}")
    print(f"  训练轮数: {loaded_config.training.epochs}")
    print(f"  批大小: {loaded_config.training.batch_size}")