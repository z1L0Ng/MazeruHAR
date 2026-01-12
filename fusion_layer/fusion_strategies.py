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
            # 初始化为均匀分布
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

class MoEFusion(FusionStrategy):
    """MoE融合策略"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.common_dim = config.get('common_dim', 64)
        
        # 获取专家维度信息
        self.expert_dims = config.get('expert_dims', {})

        # 确保有专家维度信息
        if not self.expert_dims:
            raise ValueError("MoEFusion requires 'expert_dims' in config")

        # 将每个专家映射到common_dim
        self.projections = nn.ModuleDict()
        for name, dim in self.expert_dims.items():
            self.projections[name] = nn.Linear(dim, self.common_dim)

        # 输入维度是所有专家输出维度的总和
        total_input_dim = sum(self.expert_dims.values())
        num_experts = len(self.expert_dims)
        
        if total_input_dim > 0:
            self.gate_net = nn.Sequential(
                nn.Linear(total_input_dim, total_input_dim // 2),
                nn.ReLU(),
                nn.Linear(total_input_dim // 2, num_experts),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError("Total expert output dimension must be greater than zero.")

    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not expert_outputs:
            raise ValueError("Empty expert outputs")
            
        # 确保按固定顺序处理
        sorted_keys = sorted(expert_outputs.keys())
        
        # 验证是否所有预定义的专家都有输出
        if self.expert_dims and len(expert_outputs) != len(self.expert_dims):
            # 可能会有模态缺失的情况，后续看具体数据和情况处理
            pass

        # 拼接所有原始特征
        raw_outputs_list = [expert_outputs[k] for k in sorted_keys]
        concatenated_input = torch.cat(raw_outputs_list, dim=-1)
        
        # 计算权重
        gate_weights = self.gate_net(concatenated_input)
        
        # 投影并加权
        weighted_experts = []
        for i, key in enumerate(sorted_keys):
            # 投影到common_dim
            proj = self.projections[key](expert_outputs[key])
            
            # 加权
            w = gate_weights[:, i:i+1]
            weighted_experts.append(w * proj)
            
        # 求和聚合
        fused_features = torch.stack(weighted_experts, dim=0).sum(dim=0)
        
        return fused_features

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
        'MoE': MoEFusion, 
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unsupported fusion strategy: {strategy_type}")
    
    return strategy_map[strategy_type](params)