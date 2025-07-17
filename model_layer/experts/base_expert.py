# model_layer/experts/base_expert.py
"""
基础专家类定义
提供所有专家模块的标准化接口和基础功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class ExpertModel(nn.Module, ABC):
    """
    专家模型基类
    所有专家模块必须继承此类并实现required方法
    """
    
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int, **kwargs):
        """
        初始化专家模型
        
        Args:
            input_shape: 输入数据形状 (time_steps, features)
            output_dim: 输出特征维度
            **kwargs: 其他配置参数
        """
        super(ExpertModel, self).__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.expert_type = self.__class__.__name__
        
        # 存储中间特征用于可视化和分析
        self.intermediate_features = None
        
        # 初始化模型
        self._build_model(**kwargs)
        
        # 应用权重初始化
        self.apply(self._init_weights)
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """
        构建具体的模型结构
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, time_steps, features)
            
        Returns:
            特征张量 (batch_size, output_dim)
        """
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征的标准接口
        默认等同于forward，子类可以重写以提供特定的特征提取逻辑
        
        Args:
            x: 输入张量
            
        Returns:
            特征张量
        """
        return self.forward(x)
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置信息
        
        Returns:
            配置字典
        """
        return {
            'expert_type': self.expert_type,
            'input_shape': self.input_shape,
            'output_dim': self.output_dim
        }
    
    def _init_weights(self, module):
        """
        权重初始化策略
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
    
    def get_output_dim(self) -> int:
        """
        获取输出维度
        
        Returns:
            输出特征维度
        """
        return self.output_dim
    
    def get_parameter_count(self) -> int:
        """
        获取参数数量
        
        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self):
        """
        冻结模型参数
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        """
        解冻模型参数
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def set_dropout_training(self, training: bool):
        """
        设置dropout层的训练状态
        
        Args:
            training: 是否为训练模式
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.training = training
    
    def __repr__(self):
        return f"{self.expert_type}(input_shape={self.input_shape}, output_dim={self.output_dim})"


class DummyExpert(ExpertModel):
    """
    虚拟专家模型，用于测试和占位
    """
    
    def _build_model(self, **kwargs):
        """
        构建简单的线性模型
        """
        input_size = self.input_shape[0] * self.input_shape[1]
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, self.output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 存储中间特征
        self.intermediate_features = x.detach()
        
        return x


def create_expert_from_config(expert_config: Dict[str, Any]) -> ExpertModel:
    """
    根据配置创建专家模型
    
    Args:
        expert_config: 专家配置字典，包含:
            - type: 专家类型 (transformer, rnn, cnn, hybrid)
            - input_shape: 输入形状
            - output_dim: 输出维度
            - params: 其他参数
    
    Returns:
        专家模型实例
    """
    expert_type = expert_config['type'].lower()
    input_shape = expert_config['input_shape']
    output_dim = expert_config['output_dim']
    params = expert_config.get('params', {})
    
    # 动态导入专家类
    if expert_type == 'transformer':
        try:
            from .transformer_expert import TransformerExpert
            return TransformerExpert(input_shape, output_dim, **params)
        except ImportError as e:
            raise ImportError(f"Failed to import TransformerExpert: {e}")
    elif expert_type == 'rnn':
        try:
            from .rnn_expert import RNNExpert
            return RNNExpert(input_shape, output_dim, **params)
        except ImportError as e:
            raise ImportError(f"Failed to import RNNExpert: {e}")
    elif expert_type == 'cnn':
        try:
            from .cnn_expert import CNNExpert
            return CNNExpert(input_shape, output_dim, **params)
        except ImportError as e:
            raise ImportError(f"Failed to import CNNExpert: {e}")
    elif expert_type == 'hybrid':
        try:
            from .hybrid_expert import HybridExpert
            return HybridExpert(input_shape, output_dim, **params)
        except ImportError as e:
            print(f"Warning: HybridExpert not available: {e}")
            # 回退到RNN专家
            from .rnn_expert import RNNExpert
            return RNNExpert(input_shape, output_dim, **params)
    elif expert_type == 'dummy':
        return DummyExpert(input_shape, output_dim, **params)
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")