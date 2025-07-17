# model_layer/experts/transformer_expert.py
"""
Transformer专家模块
基于现有HART模型的Transformer实现，封装为独立的专家模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base_expert import ExpertModel


class SensorPatches(nn.Module):
    """
    传感器数据补丁提取
    """
    def __init__(self, projection_dim: int, patch_size: int, time_step: int):
        super(SensorPatches, self).__init__()
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        
        # 1D卷积用于补丁提取
        self.conv = nn.Conv1d(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=patch_size,
            stride=time_step,
            padding=0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, features)
        # 转换为 (batch_size, features, time_steps) 用于1D卷积
        x = x.transpose(1, 2)
        
        # 应用1D卷积
        patches = self.conv(x)
        
        # 转换回 (batch_size, num_patches, features)
        patches = patches.transpose(1, 2)
        
        return patches


class PatchEncoder(nn.Module):
    """
    补丁编码器，添加位置编码
    """
    def __init__(self, num_patches: int, projection_dim: int):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        
        # 位置编码
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0, self.num_patches, device=patch.device)
        encoded = patch + self.position_embedding(positions).unsqueeze(0)
        return encoded


class ClassToken(nn.Module):
    """
    可学习的分类token
    """
    def __init__(self, projection_dim: int):
        super(ClassToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_broadcasted = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_broadcasted, x], dim=1)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.out_linear(attention_output)
        
        return output
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output


class FeedForward(nn.Module):
    """
    前馈网络
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerExpert(ExpertModel):
    """
    Transformer专家模块
    基于HART模型的Transformer实现
    """
    
    def _build_model(self, **kwargs):
        """
        构建Transformer模型
        """
        # 从kwargs中获取参数，设置默认值
        self.projection_dim = kwargs.get('projection_dim', 192)
        self.patch_size = kwargs.get('patch_size', 16)
        self.time_step = kwargs.get('time_step', 16)
        self.num_heads = kwargs.get('num_heads', 4)
        self.num_layers = kwargs.get('num_layers', 4)
        self.d_ff = kwargs.get('d_ff', 768)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.use_cls_token = kwargs.get('use_cls_token', True)
        
        # 确保输出维度匹配
        if self.projection_dim != self.output_dim:
            self.projection_dim = self.output_dim
        
        # 1. 输入投影层
        input_features = self.input_shape[1]
        self.input_projection = nn.Linear(input_features, self.projection_dim)
        
        # 2. 补丁提取
        self.patches = SensorPatches(self.projection_dim, self.patch_size, self.time_step)
        
        # 3. 分类token（可选）
        if self.use_cls_token:
            self.cls_token = ClassToken(self.projection_dim)
        
        # 4. 位置编码
        time_steps = self.input_shape[0]
        n_patches = (time_steps - self.patch_size) // self.time_step + 1
        if self.use_cls_token:
            n_patches += 1
        
        self.patch_encoder = PatchEncoder(n_patches, self.projection_dim)
        
        # 5. Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.projection_dim, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        
        # 6. 最终层归一化
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # 7. 输出投影（如果需要）
        if self.projection_dim != self.output_dim:
            self.output_projection = nn.Linear(self.projection_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, time_steps, features)
            
        Returns:
            特征张量 (batch_size, output_dim)
        """
        # 1. 输入投影
        x = self.input_projection(x)
        
        # 2. 补丁提取
        x = self.patches(x)
        
        # 3. 添加分类token（可选）
        if self.use_cls_token:
            x = self.cls_token(x)
        
        # 4. 位置编码
        x = self.patch_encoder(x)
        
        # 5. Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 6. 层归一化
        x = self.layer_norm(x)
        
        # 7. 全局平均池化或使用CLS token
        if self.use_cls_token:
            # 使用CLS token（第一个token）
            features = x[:, 0]
        else:
            # 全局平均池化
            features = x.mean(dim=1)
        
        # 存储中间特征
        self.intermediate_features = features.detach()
        
        # 8. 输出投影
        output = self.output_projection(features)
        
        return output
    
    def get_config(self):
        """
        获取模型配置
        """
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'patch_size': self.patch_size,
            'time_step': self.time_step,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'use_cls_token': self.use_cls_token
        })
        return config


# 创建不同配置的Transformer专家的便捷函数
def create_transformer_expert_small(input_shape: Tuple[int, int], output_dim: int) -> TransformerExpert:
    """创建小型Transformer专家"""
    return TransformerExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=128,
        patch_size=16,
        time_step=8,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout_rate=0.1
    )


def create_transformer_expert_medium(input_shape: Tuple[int, int], output_dim: int) -> TransformerExpert:
    """创建中型Transformer专家"""
    return TransformerExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=192,
        patch_size=16,
        time_step=16,
        num_heads=6,
        num_layers=4,
        d_ff=768,
        dropout_rate=0.1
    )


def create_transformer_expert_large(input_shape: Tuple[int, int], output_dim: int) -> TransformerExpert:
    """创建大型Transformer专家"""
    return TransformerExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=256,
        patch_size=16,
        time_step=16,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout_rate=0.1
    )