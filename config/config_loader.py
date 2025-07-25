# config/enhanced_config_loader.py
"""
增强的配置加载器 - 支持数据集路径配置和所有可用专家模型
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class EnhancedDatasetConfig:
    """增强的数据集配置类"""
    name: str
    path: str  # 数据集目录路径
    activity_labels: List[str]
    modalities: List[Dict[str, Any]]
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    data_split: Dict[str, float] = field(default_factory=lambda: {
        'train': 0.7, 'validation': 0.15, 'test': 0.15
    })


@dataclass
class ExpertConfig:
    """专家配置类"""
    type: str  # transformer, rnn, cnn, hybrid
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedArchitectureConfig:
    """增强的架构配置类"""
    experts: Dict[str, ExpertConfig]
    fusion: Dict[str, Any] = field(default_factory=dict)
    fusion_output_dim: int = 192
    dropout_rate: float = 0.3


@dataclass
class EnhancedTrainingConfig:
    """增强的训练配置类"""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.0001
    early_stopping: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedExperimentConfig:
    """增强的实验配置类"""
    name: str
    description: str = ""
    seed: int = 42
    device: str = "auto"
    output_dir: str = "./results"
    save_checkpoints: bool = True
    verbose: bool = True
    
    dataset: EnhancedDatasetConfig = None
    architecture: EnhancedArchitectureConfig = None
    training: EnhancedTrainingConfig = None
    visualization: Dict[str, Any] = field(default_factory=dict)


class EnhancedConfigLoader:
    """增强的配置加载器"""
    
    # 支持的专家类型映射
    SUPPORTED_EXPERTS = {
        'transformer': {
            'presets': ['transformer_small', 'transformer_medium', 'transformer_large'],
            'description': 'Transformer-based expert for attention mechanisms'
        },
        'rnn': {
            'presets': ['rnn_lstm', 'rnn_gru', 'rnn_channel_specific', 'rnn_deep'],
            'description': 'RNN-based expert (LSTM/GRU)'
        },
        'cnn': {
            'presets': ['cnn_simple', 'cnn_multiscale', 'cnn_deep', 'cnn_lightweight'],
            'description': 'CNN-based expert for spatial-temporal patterns'
        },
        'hybrid': {
            'presets': ['hybrid_small', 'hybrid_medium', 'hybrid_large', 'hybrid_attention_fusion'],
            'description': 'Hybrid expert combining multiple architectures'
        }
    }
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> EnhancedExperimentConfig:
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return EnhancedConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> EnhancedExperimentConfig:
        """将配置字典转换为配置对象"""
        
        # 验证数据集路径
        dataset_path = config_dict['dataset']['path']
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集路径不存在: {dataset_path}")
        
        # 解析数据集配置
        dataset_config = EnhancedDatasetConfig(
            name=config_dict['dataset']['name'],
            path=dataset_path,
            activity_labels=config_dict['dataset']['activity_labels'],
            modalities=config_dict['dataset']['modalities'],
            preprocessing=config_dict['dataset'].get('preprocessing', {}),
            data_split=config_dict['dataset'].get('data_split', {
                'train': 0.7, 'validation': 0.15, 'test': 0.15
            })
        )
        
        # 解析专家配置
        experts = {}
        for expert_name, expert_config in config_dict['architecture']['experts'].items():
            experts[expert_name] = ExpertConfig(
                type=expert_config['type'],
                params=expert_config.get('params', {})
            )
        
        # 解析架构配置
        architecture_config = EnhancedArchitectureConfig(
            experts=experts,
            fusion=config_dict['architecture'].get('fusion', {}),
            fusion_output_dim=config_dict['architecture'].get('fusion_output_dim', 192),
            dropout_rate=config_dict['architecture'].get('dropout_rate', 0.3)
        )
        
        # 解析训练配置
        training_config = EnhancedTrainingConfig(
            epochs=config_dict['training'].get('epochs', 100),
            batch_size=config_dict['training'].get('batch_size', 64),
            learning_rate=config_dict['training'].get('learning_rate', 0.001),
            optimizer=config_dict['training'].get('optimizer', 'adamw'),
            scheduler=config_dict['training'].get('scheduler', 'cosine'),
            weight_decay=config_dict['training'].get('weight_decay', 0.0001),
            early_stopping=config_dict['training'].get('early_stopping', {})
        )
        
        # 创建完整的实验配置
        experiment_config = EnhancedExperimentConfig(
            name=config_dict['name'],
            description=config_dict.get('description', ''),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'auto'),
            output_dir=config_dict.get('output_dir', './results'),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            verbose=config_dict.get('verbose', True),
            dataset=dataset_config,
            architecture=architecture_config,
            training=training_config,
            visualization=config_dict.get('visualization', {})
        )
        
        return experiment_config
    
    @staticmethod
    def validate_config(config: EnhancedExperimentConfig) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证数据集路径
        if not os.path.exists(config.dataset.path):
            errors.append(f"数据集路径不存在: {config.dataset.path}")
        
        # 验证专家配置
        for expert_name, expert in config.architecture.experts.items():
            if expert.type not in EnhancedConfigLoader.SUPPORTED_EXPERTS:
                errors.append(f"不支持的专家类型: {expert.type}")
        
        # 验证模态配置
        if not config.dataset.modalities:
            errors.append("至少需要一个模态配置")
        
        return errors
    
    @staticmethod
    def list_supported_experts():
        """列出所有支持的专家类型"""
        print("支持的专家类型:")
        for expert_type, info in EnhancedConfigLoader.SUPPORTED_EXPERTS.items():
            print(f"  {expert_type}: {info['description']}")
            print(f"    预设: {', '.join(info['presets'])}")
    
    @staticmethod
    def create_shl_config(dataset_path: str = "./datasets/datasetStandardized/SHL_Multimodal") -> EnhancedExperimentConfig:
        """创建SHL数据集的配置"""
        
        # 创建数据集配置
        dataset_config = EnhancedDatasetConfig(
            name="SHL",
            path=dataset_path,
            activity_labels=["Standing", "Walking", "Running", "Biking", "Car", "Bus", "Train", "Subway"],
            modalities=[
                {
                    "name": "imu",
                    "channels": 6,
                    "sequence_length": 128,
                    "expert_type": "transformer",
                    "expert_params": {
                        "projection_dim": 128,
                        "num_heads": 8,
                        "num_layers": 4
                    }
                },
                {
                    "name": "pressure",
                    "channels": 1,
                    "sequence_length": 128,
                    "expert_type": "rnn",
                    "expert_params": {
                        "hidden_dim": 64,
                        "num_layers": 2,
                        "rnn_type": "gru"
                    }
                }
            ],
            preprocessing={
                "window_size": 128,
                "step_size": 64,
                "normalize": True
            }
        )
        
        # 创建专家配置
        experts = {
            "imu_expert": ExpertConfig(
                type="transformer",
                params={
                    "projection_dim": 128,
                    "num_heads": 8,
                    "num_layers": 4,
                    "dropout": 0.1,
                    "output_dim": 128
                }
            ),
            "pressure_expert": ExpertConfig(
                type="rnn",
                params={
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "rnn_type": "gru",
                    "bidirectional": True,
                    "output_dim": 64
                }
            )
        }
        
        # 创建架构配置
        architecture_config = EnhancedArchitectureConfig(
            experts=experts,
            fusion={
                "strategy": "concatenate",
                "params": {"fusion_dim": 192}
            },
            fusion_output_dim=192,
            dropout_rate=0.3
        )
        
        # 创建训练配置
        training_config = EnhancedTrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            optimizer="adamw",
            scheduler="cosine",
            weight_decay=0.0001,
            early_stopping={
                "enabled": True,
                "patience": 15,
                "monitor": "val_f1"
            }
        )
        
        # 创建完整配置
        config = EnhancedExperimentConfig(
            name="SHL_MultiModal_Experiment",
            description="SHL数据集多模态融合实验",
            seed=42,
            device="auto",
            output_dir="./results/shl_multimodal",
            dataset=dataset_config,
            architecture=architecture_config,
            training=training_config,
            visualization={
                "enabled": True,
                "plots": {
                    "learning_curves": True,
                    "confusion_matrix": True,
                    "attention_maps": True
                }
            }
        )
        
        return config
    
    @staticmethod
    def save_to_yaml(config: EnhancedExperimentConfig, output_path: str):
        """保存配置到YAML文件"""
        # 转换为字典
        config_dict = {
            'name': config.name,
            'description': config.description,
            'seed': config.seed,
            'device': config.device,
            'output_dir': config.output_dir,
            'save_checkpoints': config.save_checkpoints,
            'verbose': config.verbose,
            'dataset': {
                'name': config.dataset.name,
                'path': config.dataset.path,
                'activity_labels': config.dataset.activity_labels,
                'modalities': config.dataset.modalities,
                'preprocessing': config.dataset.preprocessing,
                'data_split': config.dataset.data_split
            },
            'architecture': {
                'experts': {
                    name: {
                        'type': expert.type,
                        'params': expert.params
                    } for name, expert in config.architecture.experts.items()
                },
                'fusion': config.architecture.fusion,
                'fusion_output_dim': config.architecture.fusion_output_dim,
                'dropout_rate': config.architecture.dropout_rate
            },
            'training': {
                'epochs': config.training.epochs,
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler,
                'weight_decay': config.training.weight_decay,
                'early_stopping': config.training.early_stopping
            },
            'visualization': config.visualization
        }
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存YAML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"✓ 配置已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 列出支持的专家类型
    EnhancedConfigLoader.list_supported_experts()
    
    # 创建SHL配置
    shl_config = EnhancedConfigLoader.create_shl_config(
        dataset_path="./datasets/datasetStandardized/SHL_Multimodal"
    )
    
    # 验证配置
    errors = EnhancedConfigLoader.validate_config(shl_config)
    if errors:
        print("配置验证错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ 配置验证通过")
    
    # 保存配置
    EnhancedConfigLoader.save_to_yaml(shl_config, "config/shl_multimodal_config.yaml")