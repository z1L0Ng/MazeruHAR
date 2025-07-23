# fusion_layer/__init__.py
"""
融合层模块 - 任务2.2
提供多模态特征融合策略
"""

from .fusion_strategies import (
    FusionStrategy,
    ConcatenateFusion,
    WeightedSumFusion,
    AverageFusion,
    create_fusion_strategy
)

__all__ = [
    'FusionStrategy',
    'ConcatenateFusion', 
    'WeightedSumFusion',
    'AverageFusion',
    'create_fusion_strategy'
]

__version__ = "1.0.0"