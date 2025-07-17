# model_layer/experts/hybrid_expert.py
"""
混合专家模块
基于现有model_hybrid.py的RNN-Attention混合实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from .base_expert import ExpertModel


class RNNEncoder(nn.Module):
    """
    RNN编码器
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 rnn_type: str = 'gru', bidirectional: bool = True, dropout_rate: float = 0.1):
        super(RNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 输出维度
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            输出张量 (batch_size, seq_len, output_dim)
        """
        output, _ = self.rnn(x)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


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
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 保存残差连接
        residual = query
        
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
        
        # 残差连接和层归一化
        output = self.layer_norm(residual + self.dropout(output))
        
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
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # 残差连接和层归一化
        x = self.layer_norm(residual + self.dropout(x))
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        x = self.self_attention(x, x, x, mask)
        
        # 前馈网络
        x = self.feed_forward(x)
        
        return x


class HybridExpert(ExpertModel):
    """
    混合专家模块
    结合RNN和Transformer的优势
    """
    
    def _build_model(self, **kwargs):
        """
        构建混合模型
        """
        # 从kwargs中获取参数，设置默认值
        self.projection_dim = kwargs.get('projection_dim', 192)
        self.rnn_hidden_dim = kwargs.get('rnn_hidden_dim', 128)
        self.rnn_num_layers = kwargs.get('rnn_num_layers', 2)
        self.rnn_type = kwargs.get('rnn_type', 'gru')
        self.rnn_bidirectional = kwargs.get('rnn_bidirectional', True)
        self.num_heads = kwargs.get('num_heads', 4)
        self.num_transformer_layers = kwargs.get('num_transformer_layers', 2)
        self.transformer_dim_feedforward = kwargs.get('transformer_dim_feedforward', 768)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.fusion_method = kwargs.get('fusion_method', 'concat')  # 'concat', 'add', 'attention'
        
        # 输入特征数
        input_features = self.input_shape[1]
        
        # 确保投影维度与输出维度兼容
        if self.projection_dim != self.output_dim:
            self.projection_dim = self.output_dim
        
        # 1. 输入投影层
        self.input_projection = nn.Linear(input_features, self.projection_dim)
        
        # 2. RNN编码器
        self.rnn_encoder = RNNEncoder(
            input_dim=self.projection_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            bidirectional=self.rnn_bidirectional,
            dropout_rate=self.dropout_rate
        )
        
        # 3. RNN输出维度调整
        self.rnn_output_dim = self.rnn_hidden_dim * (2 if self.rnn_bidirectional else 1)
        if self.rnn_output_dim != self.projection_dim:
            self.rnn_projection = nn.Linear(self.rnn_output_dim, self.projection_dim)
        else:
            self.rnn_projection = nn.Identity()
        
        # 4. Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.projection_dim,
                num_heads=self.num_heads,
                d_ff=self.transformer_dim_feedforward,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_transformer_layers)
        ])
        
        # 5. 融合层
        if self.fusion_method == 'concat':
            self.fusion_layer = nn.Linear(self.projection_dim * 2, self.projection_dim)
        elif self.fusion_method == 'add':
            self.fusion_layer = nn.Identity()
        elif self.fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.projection_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.fusion_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # 6. 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.projection_dim)
        
        # 7. 输出投影（如果需要）
        if self.projection_dim != self.output_dim:
            self.output_projection = nn.Linear(self.projection_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()
        
        # 8. 最终dropout
        self.final_dropout = nn.Dropout(self.dropout_rate)
    
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
        
        # 2. RNN编码
        rnn_output = self.rnn_encoder(x)
        rnn_output = self.rnn_projection(rnn_output)
        
        # 3. Transformer编码
        transformer_output = rnn_output
        for layer in self.transformer_layers:
            transformer_output = layer(transformer_output)
        
        # 4. 融合RNN和Transformer特征
        if self.fusion_method == 'concat':
            # 全局平均池化
            rnn_pooled = rnn_output.mean(dim=1)
            transformer_pooled = transformer_output.mean(dim=1)
            
            # 拼接融合
            fused_features = torch.cat([rnn_pooled, transformer_pooled], dim=1)
            fused_features = self.fusion_layer(fused_features)
            
        elif self.fusion_method == 'add':
            # 加法融合
            fused_output = rnn_output + transformer_output
            fused_features = fused_output.mean(dim=1)
            
        elif self.fusion_method == 'attention':
            # 注意力融合
            # 使用transformer输出作为query，RNN输出作为key和value
            attn_output, _ = self.fusion_attention(
                query=transformer_output,
                key=rnn_output,
                value=rnn_output
            )
            fused_features = attn_output.mean(dim=1)
        
        # 5. 最终层归一化
        fused_features = self.final_layer_norm(fused_features)
        
        # 存储中间特征
        self.intermediate_features = fused_features.detach()
        
        # 6. 输出投影
        output = self.output_projection(fused_features)
        output = self.final_dropout(output)
        
        return output
    
    def get_config(self):
        """
        获取模型配置
        """
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'rnn_num_layers': self.rnn_num_layers,
            'rnn_type': self.rnn_type,
            'rnn_bidirectional': self.rnn_bidirectional,
            'num_heads': self.num_heads,
            'num_transformer_layers': self.num_transformer_layers,
            'transformer_dim_feedforward': self.transformer_dim_feedforward,
            'dropout_rate': self.dropout_rate,
            'fusion_method': self.fusion_method
        })
        return config


# 创建不同配置的混合专家的便捷函数
def create_hybrid_expert_small(input_shape: Tuple[int, int], output_dim: int) -> HybridExpert:
    """创建小型混合专家"""
    return HybridExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=128,
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        rnn_type='gru',
        rnn_bidirectional=True,
        num_heads=4,
        num_transformer_layers=1,
        transformer_dim_feedforward=512,
        dropout_rate=0.1,
        fusion_method='add'
    )


def create_hybrid_expert_medium(input_shape: Tuple[int, int], output_dim: int) -> HybridExpert:
    """创建中型混合专家"""
    return HybridExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=192,
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        rnn_type='gru',
        rnn_bidirectional=True,
        num_heads=4,
        num_transformer_layers=2,
        transformer_dim_feedforward=768,
        dropout_rate=0.1,
        fusion_method='concat'
    )


def create_hybrid_expert_large(input_shape: Tuple[int, int], output_dim: int) -> HybridExpert:
    """创建大型混合专家"""
    return HybridExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=256,
        rnn_hidden_dim=128,
        rnn_num_layers=3,
        rnn_type='lstm',
        rnn_bidirectional=True,
        num_heads=8,
        num_transformer_layers=4,
        transformer_dim_feedforward=1024,
        dropout_rate=0.1,
        fusion_method='attention'
    )


def create_hybrid_expert_attention_fusion(input_shape: Tuple[int, int], output_dim: int) -> HybridExpert:
    """创建使用注意力融合的混合专家"""
    return HybridExpert(
        input_shape=input_shape,
        output_dim=output_dim,
        projection_dim=192,
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        rnn_type='gru',
        rnn_bidirectional=True,
        num_heads=6,
        num_transformer_layers=2,
        transformer_dim_feedforward=768,
        dropout_rate=0.1,
        fusion_method='attention'
    )