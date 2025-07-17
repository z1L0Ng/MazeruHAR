# model_layer/experts/cnn_expert.py
"""
CNN专家模块
1D CNN实现用于时序数据处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .base_expert import ExpertModel


class Conv1DBlock(nn.Module):
    """
    1D卷积块
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 dropout_rate: float = 0.1, activation: str = 'relu'):
        super(Conv1DBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, channels, length)
            
        Returns:
            输出张量 (batch_size, out_channels, new_length)
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, channels: int, kernel_size: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = Conv1DBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = Conv1DBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dropout_rate=dropout_rate
        )
        
        self.layer_norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        residual = x
        
        # 第一个卷积
        out = self.conv1(x)
        
        # 第二个卷积
        out = self.conv2(out)
        
        # 残差连接
        out = out + residual
        
        # 层归一化（转换维度）
        out = out.transpose(1, 2)  # (batch, length, channels)
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # (batch, channels, length)
        
        return out


class MultiScaleCNN(nn.Module):
    """
    多尺度CNN
    """
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_sizes: List[int], dropout_rate: float = 0.1):
        super(MultiScaleCNN, self).__init__()
        
        self.branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = Conv1DBlock(
                in_channels=input_channels,
                out_channels=output_channels // len(kernel_sizes),
                kernel_size=kernel_size,
                padding=padding,
                dropout_rate=dropout_rate
            )
            self.branches.append(branch)
        
        # 如果输出通道数不能被分支数整除，添加一个1x1卷积来调整
        total_branch_channels = (output_channels // len(kernel_sizes)) * len(kernel_sizes)
        if total_branch_channels != output_channels:
            self.adjust_conv = nn.Conv1d(total_branch_channels, output_channels, 1)
        else:
            self.adjust_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        branch_outputs = []
        
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # 连接所有分支
        out = torch.cat(branch_outputs, dim=1)
        
        # 调整通道数
        out = self.adjust_conv(out)
        
        return out


class CNNExpert(ExpertModel):
    """
    CNN专家模块
    1D CNN实现用于时序数据处理
    """
    
    def _build_model(self, **kwargs):
        """
        构建CNN模型
        """
        # 从kwargs中获取参数，设置默认值
        self.num_layers = kwargs.get('num_layers', 4)
        self.base_channels = kwargs.get('base_channels', 64)
        self.kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7])
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.use_residual = kwargs.get('use_residual', True)
        self.use_multiscale = kwargs.get('use_multiscale', True)
        self.pooling_method = kwargs.get('pooling_method', 'adaptive')  # 'adaptive', 'max', 'avg'
        self.channel_multiplier = kwargs.get('channel_multiplier', 2)
        
        # 输入特征数
        input_features = self.input_shape[1]
        
        # 构建CNN层
        self.cnn_layers = nn.ModuleList()
        
        # 第一层：输入投影
        if self.use_multiscale:
            self.cnn_layers.append(
                MultiScaleCNN(
                    input_channels=input_features,
                    output_channels=self.base_channels,
                    kernel_sizes=self.kernel_sizes,
                    dropout_rate=self.dropout_rate
                )
            )
        else:
            self.cnn_layers.append(
                Conv1DBlock(
                    in_channels=input_features,
                    out_channels=self.base_channels,
                    kernel_size=self.kernel_sizes[0],
                    padding=self.kernel_sizes[0] // 2,
                    dropout_rate=self.dropout_rate
                )
            )
        
        # 中间层
        current_channels = self.base_channels
        for i in range(1, self.num_layers):
            next_channels = current_channels * self.channel_multiplier
            
            if self.use_residual and current_channels == next_channels:
                # 使用残差块
                self.cnn_layers.append(
                    ResidualBlock(
                        channels=current_channels,
                        kernel_size=self.kernel_sizes[0],
                        dropout_rate=self.dropout_rate
                    )
                )
            else:
                # 使用普通卷积块
                if self.use_multiscale:
                    self.cnn_layers.append(
                        MultiScaleCNN(
                            input_channels=current_channels,
                            output_channels=next_channels,
                            kernel_sizes=self.kernel_sizes,
                            dropout_rate=self.dropout_rate
                        )
                    )
                else:
                    self.cnn_layers.append(
                        Conv1DBlock(
                            in_channels=current_channels,
                            out_channels=next_channels,
                            kernel_size=self.kernel_sizes[0],
                            padding=self.kernel_sizes[0] // 2,
                            dropout_rate=self.dropout_rate
                        )
                    )
                current_channels = next_channels
        
        # 全局池化
        if self.pooling_method == 'adaptive':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        elif self.pooling_method == 'max':
            self.global_pooling = nn.AdaptiveMaxPool1d(1)
        elif self.pooling_method == 'avg':
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # 输出投影
        if current_channels != self.output_dim:
            self.output_projection = nn.Linear(current_channels, self.output_dim)
        else:
            self.output_projection = nn.Identity()
        
        # 最终dropout
        self.final_dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, time_steps, features)
            
        Returns:
            特征张量 (batch_size, output_dim)
        """
        # 转换为CNN输入格式 (batch_size, features, time_steps)
        x = x.transpose(1, 2)
        
        # 通过CNN层
        for layer in self.cnn_layers:
            x = layer(x)
        
        # 全局池化
        x = self.global_pooling(x)  # (batch_size, channels, 1)
        
        # 展平
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # 存储中间特征
        self.intermediate_features = x.detach()
        
        # 输出投影
        x = self.output_projection(x)
        x = self.final_dropout(x)
        
        return x
    
    def get_config(self):
        """
        获取模型配置
        """
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'base_channels': self.base_channels,
            'kernel_sizes': self.kernel_sizes,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'use_multiscale': self.use_multiscale,
            'pooling_method': self.pooling_method,
            'channel_multiplier': self.channel_multiplier
        })
        return config


# 创建不同配置的CNN专家的便捷函数
def create_cnn_expert_simple(input_shape: Tuple[int, int], output_dim: int) -> CNNExpert:
    """创建简单CNN专家"""
    return CNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        num_layers=3,
        base_channels=32,
        kernel_sizes=[3, 5],
        dropout_rate=0.1,
        use_residual=False,
        use_multiscale=False
    )


def create_cnn_expert_multiscale(input_shape: Tuple[int, int], output_dim: int) -> CNNExpert:
    """创建多尺度CNN专家"""
    return CNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        num_layers=4,
        base_channels=64,
        kernel_sizes=[3, 5, 7],
        dropout_rate=0.1,
        use_residual=True,
        use_multiscale=True
    )


def create_cnn_expert_deep(input_shape: Tuple[int, int], output_dim: int) -> CNNExpert:
    """创建深层CNN专家"""
    return CNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        num_layers=6,
        base_channels=64,
        kernel_sizes=[3, 5, 7, 9],
        dropout_rate=0.2,
        use_residual=True,
        use_multiscale=True,
        channel_multiplier=2
    )


def create_cnn_expert_lightweight(input_shape: Tuple[int, int], output_dim: int) -> CNNExpert:
    """创建轻量级CNN专家"""
    return CNNExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        num_layers=2,
        base_channels=32,
        kernel_sizes=[3],
        dropout_rate=0.1,
        use_residual=False,
        use_multiscale=False,
        channel_multiplier=1
    )