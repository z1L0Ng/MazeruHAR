# config/__init__.py
"""
MazeruHAR 配置系统
提供配置驱动的实验框架支持
"""

from .config_loader import (
    ConfigLoader,
    ConfigValidator,
    ExperimentConfig,
    DatasetConfig,
    ModalityConfig,
    ExpertConfig,
    FusionConfig,
    ArchitectureConfig,
    TrainingConfig,
    create_sample_config
)

from .config_bridge import (
    ConfigBridge,
    load_config_or_use_defaults,
    convert_notebook_to_config
)

__version__ = "1.0.0"
__author__ = "MazeruHAR Team"

__all__ = [
    'ConfigLoader',
    'ConfigValidator', 
    'ExperimentConfig',
    'DatasetConfig',
    'ModalityConfig',
    'ExpertConfig',
    'FusionConfig',
    'ArchitectureConfig',
    'TrainingConfig',
    'create_sample_config',
    'ConfigBridge',
    'load_config_or_use_defaults',
    'convert_notebook_to_config'
]