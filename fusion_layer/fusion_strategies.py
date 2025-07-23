# fusion_layer/fusion_strategies.py
"""
融合策略实现 - 任务2.2
实现拼接融合策略
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class FusionStrategy(nn.Module, ABC):
    """融合策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合多个专家输出
        
        Args:
            expert_outputs: 专家输出字典 {modality_name: tensor}
        
        Returns:
            融合后的特征张量
        """
        pass


class ConcatenateFusion(FusionStrategy):
    """拼接融合策略 - 任务2.2核心实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dim = config.get('dim', -1)  # 拼接维度，默认最后一维
        
        # 可选的降维层
        if 'output_dim' in config:
            self.projection = nn.Linear(
                config.get('input_dim', -1), 
                config['output_dim']
            )
        else:
            self.projection = None
    
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        拼接融合实现
        
        Args:
            expert_outputs: {
                'imu': tensor[batch, feature_dim1],
                'pressure': tensor[batch, feature_dim2], 
                ...
            }
        
        Returns:
            tensor[batch, sum(feature_dims)]
        """
        if not expert_outputs:
            raise ValueError("Empty expert outputs")
        
        # 按模态名排序确保一致性
        sorted_outputs = [expert_outputs[key] for key in sorted(expert_outputs.keys())]
        
        # 拼接所有专家输出
        fused_features = torch.cat(sorted_outputs, dim=self.dim)
        
        # 可选的投影降维
        if self.projection is not None:
            fused_features = self.projection(fused_features)
        
        return fused_features
    
    def get_output_dim(self, expert_dims: Dict[str, int]) -> int:
        """计算输出维度"""
        if self.projection is not None:
            return self.projection.out_features
        return sum(expert_dims.values())


class WeightedSumFusion(FusionStrategy):
    """加权求和融合策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_experts = config['num_experts']
        
        # 学习权重参数
        if config.get('learnable_weights', True):
            self.weights = nn.Parameter(torch.ones(self.num_experts) / self.num_experts)
        else:
            fixed_weights = config.get('weights', [1.0] * self.num_experts)
            self.register_buffer('weights', torch.tensor(fixed_weights))
    
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """加权求和融合"""
        sorted_outputs = [expert_outputs[key] for key in sorted(expert_outputs.keys())]
        
        # 应用softmax确保权重和为1
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # 加权求和
        fused_features = sum(w * output for w, output in zip(normalized_weights, sorted_outputs))
        
        return fused_features


class AverageFusion(FusionStrategy):
    """平均融合策略"""
    
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """平均融合"""
        sorted_outputs = [expert_outputs[key] for key in sorted(expert_outputs.keys())]
        return torch.mean(torch.stack(sorted_outputs), dim=0)


def create_fusion_strategy(fusion_config: Dict[str, Any]) -> FusionStrategy:
    """
    融合策略工厂函数
    
    Args:
        fusion_config: 融合配置
        
    Returns:
        融合策略实例
    """
    strategy_type = fusion_config['strategy']
    params = fusion_config.get('params', {})
    
    strategy_map = {
        'concatenate': ConcatenateFusion,
        'weighted_sum': WeightedSumFusion,
        'average': AverageFusion,
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unsupported fusion strategy: {strategy_type}")
    
    return strategy_map[strategy_type](params)