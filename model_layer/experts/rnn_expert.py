# model_layer/experts/rnn_expert.py
"""
RNN专家模块
基于现有RNN-HART模型的实现，封装为独立的专家模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .base_expert import ExpertModel


class RNNBlock(nn.Module):
    """
    RNN基础块
    """
    def __init__(self, input_dim: int, hidden_dim: int, rnn_type: str = 'gru', 
                 dropout_rate: float = 0.1, bidirectional: bool = True):
        super(RNNBlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout_rate if dropout_rate > 0 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 输出维度
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 输出层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            输出张量 (batch_size, seq_len, hidden_dim * num_directions)
        """
        # RNN前向传播
        output, _ = self.rnn(x)
        
        # 层归一化和dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class SensorChannelRNN(nn.Module):
    """
    对特定传感器通道应用RNN处理
    """
    def __init__(self, start_index: int, stop_index: int, hidden_dim: int = 64, 
                 rnn_type: str = 'gru', dropout_rate: float = 0.1, bidirectional: bool = True):
        super(SensorChannelRNN, self).__init__()
        
        self.start_index = start_index
        self.stop_index = stop_index
        self.input_dim = stop_index - start_index
        self.hidden_dim = hidden_dim
        
        # RNN层
        self.rnn_block = RNNBlock(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, features)
            
        Returns:
            输出张量 (batch_size, seq_len, hidden_dim * num_directions)
        """
        # 提取相关通道
        channel_data = x[:, :, self.start_index:self.stop_index]
        
        # 应用RNN
        output = self.rnn_block(channel_data)
        
        return output


class MultiLayerRNN(nn.Module):
    """
    多层RNN结构
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 rnn_type: str = 'gru', dropout_rate: float = 0.1, bidirectional: bool = True):
        super(MultiLayerRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 构建多层RNN
        self.rnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim * (2 if bidirectional else 1)
            
            self.rnn_layers.append(
                RNNBlock(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout_rate,
                    bidirectional=bidirectional
                )
            )
        
        # 输出维度
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)
        
        return x


class RNNExpert(ExpertModel):
    """
    RNN专家模块
    基于RNN-HART模型的实现
    """
    
    def _build_model(self, **kwargs):
        """
        构建RNN模型
        """
        # 从kwargs中获取参数，设置默认值
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.num_layers = kwargs.get('num_layers', 2)
        self.rnn_type = kwargs.get('rnn_type', 'gru')
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.use_channel_specific = kwargs.get('use_channel_specific', False)
        self.pooling_method = kwargs.get('pooling_method', 'last')  # 'last', 'mean', 'max'
        
        # 输入特征数
        input_features = self.input_shape[1]
        
        if self.use_channel_specific:
            # 使用通道特定的RNN处理
            self._build_channel_specific_model(input_features)
        else:
            # 使用标准多层RNN
            self._build_standard_model(input_features)
        
        # 输出投影层
        rnn_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        if rnn_output_dim != self.output_dim:
            self.output_projection = nn.Linear(rnn_output_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def _build_standard_model(self, input_features: int):
        """
        构建标准多层RNN模型
        """
        self.rnn_encoder = MultiLayerRNN(
            input_dim=input_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional
        )
    
    def _build_channel_specific_model(self, input_features: int):
        """
        构建通道特定的RNN模型
        """
        # 假设输入特征可以分为不同的传感器通道
        # 这里简化为三个相等的部分（可根据实际需求调整）
        channel_size = input_features // 3
        
        # 第一通道 (加速度计)
        self.acc_rnn = SensorChannelRNN(
            start_index=0,
            stop_index=channel_size,
            hidden_dim=self.hidden_dim,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional
        )
        
        # 第二通道 (陀螺仪)
        self.gyro_rnn = SensorChannelRNN(
            start_index=channel_size,
            stop_index=channel_size * 2,
            hidden_dim=self.hidden_dim,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional
        )
        
        # 第三通道 (其他传感器)
        self.other_rnn = SensorChannelRNN(
            start_index=channel_size * 2,
            stop_index=input_features,
            hidden_dim=self.hidden_dim,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate,
            bidirectional=self.bidirectional
        )
        
        # 融合层
        fusion_input_dim = self.hidden_dim * (2 if self.bidirectional else 1) * 3
        self.fusion_layer = nn.Linear(fusion_input_dim, self.hidden_dim * (2 if self.bidirectional else 1))
        self.fusion_dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, time_steps, features)
            
        Returns:
            特征张量 (batch_size, output_dim)
        """
        if self.use_channel_specific:
            return self._forward_channel_specific(x)
        else:
            return self._forward_standard(x)
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准RNN前向传播
        """
        # RNN编码
        rnn_output = self.rnn_encoder(x)
        
        # 池化操作
        if self.pooling_method == 'last':
            # 使用最后一个时间步的输出
            features = rnn_output[:, -1, :]
        elif self.pooling_method == 'mean':
            # 使用平均池化
            features = rnn_output.mean(dim=1)
        elif self.pooling_method == 'max':
            # 使用最大池化
            features, _ = rnn_output.max(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # 存储中间特征
        self.intermediate_features = features.detach()
        
        # 输出投影
        output = self.output_projection(features)
        
        return output
    
    def _forward_channel_specific(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道特定RNN前向传播
        """
        # 分别处理每个通道
        acc_output = self.acc_rnn(x)
        gyro_output = self.gyro_rnn(x)
        other_output = self.other_rnn(x)
        
        # 池化操作
        if self.pooling_method == 'last':
            acc_features = acc_output[:, -1, :]
            gyro_features = gyro_output[:, -1, :]
            other_features = other_output[:, -1, :]
        elif self.pooling_method == 'mean':
            acc_features = acc_output.mean(dim=1)
            gyro_features = gyro_output.mean(dim=1)
            other_features = other_output.mean(dim=1)
        elif self.pooling_method == 'max':
            acc_features, _ = acc_output.max(dim=1)
            gyro_features, _ = gyro_output.max(dim=1)
            other_features, _ = other_output.max(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # 融合特征
        fused_features = torch.cat([acc_features, gyro_features, other_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.fusion_dropout(fused_features)
        
        # 存储中间特征
        self.intermediate_features = fused_features.detach()
        
        # 输出投影
        output = self.output_projection(fused_features)
        
        return output
    
    def get_config(self):
        """
        获取模型配置
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'rnn_type': self.rnn_type,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'use_channel_specific': self.use_channel_specific,
            'pooling_method': self.pooling_method
        })
        return config


# 创建不同配置的RNN专家的便捷函数
def create_rnn_expert_lstm(input_shape: Tuple[int, int], output_dim: int) -> RNNExpert:
    """创建LSTM专家"""
    return RNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        rnn_type='lstm',
        dropout_rate=0.1,
        bidirectional=True,
        pooling_method='last'
    )


def create_rnn_expert_gru(input_shape: Tuple[int, int], output_dim: int) -> RNNExpert:
    """创建GRU专家"""
    return RNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        rnn_type='gru',
        dropout_rate=0.1,
        bidirectional=True,
        pooling_method='last'
    )


def create_rnn_expert_channel_specific(input_shape: Tuple[int, int], output_dim: int) -> RNNExpert:
    """创建通道特定的RNN专家"""
    return RNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        rnn_type='gru',
        dropout_rate=0.1,
        bidirectional=True,
        use_channel_specific=True,
        pooling_method='last'
    )


def create_rnn_expert_deep(input_shape: Tuple[int, int], output_dim: int) -> RNNExpert:
    """创建深层RNN专家"""
    return RNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=128,
        num_layers=4,
        rnn_type='gru',
        dropout_rate=0.2,
        bidirectional=True,
        pooling_method='mean'
    )