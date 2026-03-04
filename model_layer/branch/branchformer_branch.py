# model_layer/branch/branchformer_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .base_branch import ExpertModel
from .transformer_branch import SensorPatches, PatchEncoder, ClassToken

class BranchformerBlock(nn.Module):
    """
    Branchformer 核心块：并行处理全局注意力分支和局部卷积分支。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, kernel_size: int = 31):
        super().__init__()
        
        # FFN 1 (Sandwich 结构)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        
        # 全局分支：多头注意力
        self.attn_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 局部分支：Conformer 风格的卷积模块
        self.conv_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.conv_pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.conv_dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.dw_bn = nn.BatchNorm1d(d_model)
        self.conv_pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FFN 2
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        
        self.final_ln = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN 1
        x = x + 0.5 * self.ff1(x)
        
        # 全局分支
        attn_in = self.attn_ln(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        
        # 局部分支 (B, T, C) -> (B, C, T)
        conv_in = self.conv_ln(x).transpose(1, 2)
        # GLU 激活
        conv_pw = self.conv_pw1(conv_in)
        u, v = torch.chunk(conv_pw, 2, dim=1)
        conv_glu = u * torch.sigmoid(v)
        # Depthwise Conv
        conv_dw = self.conv_dw(conv_glu)
        conv_dw = self.dw_bn(conv_dw)
        conv_dw = F.silu(conv_dw)
        # PW 2
        conv_out = self.conv_pw2(conv_dw).transpose(1, 2)
        
        # 合并分支
        x = x + attn_out + self.dropout(conv_out)
        
        # FFN 2
        x = x + 0.5 * self.ff2(x)
        return self.final_ln(x)

class BranchformerExpert(ExpertModel):
    """
    Branchformer 专家模块：参数与 TransformerExpert 对齐，支持 UCI 数据集。
    """
    def _build_model(self, **kwargs):
        # 参数对齐：支持从 config 中读取 d_model/nhead 等别名
        self.projection_dim = kwargs.get('projection_dim', 192)
        self.num_heads = kwargs.get('num_heads', 4)
        self.d_ff = kwargs.get('d_ff', 768)
        self.num_layers = kwargs.get('num_layers', 4)
        self.dropout_rate = kwargs.get('dropout', 0.1)
        
        self.patch_size = kwargs.get('patch_size', 16)
        self.time_step = kwargs.get('time_step', 16)
        self.use_cls_token = kwargs.get('use_cls_token', True)
        self.kernel_size = kwargs.get('kernel_size', 3)

        # 1. 输入投影层
        input_features = self.input_shape[1]
        self.input_projection = nn.Linear(input_features, self.projection_dim)
        
        # 2. 补丁提取与编码
        self.patches = SensorPatches(self.projection_dim, self.patch_size, self.time_step)
        
        if self.use_cls_token:
            self.cls_token = ClassToken(self.projection_dim)
        
        num_patches = (self.input_shape[0] - self.patch_size) // self.time_step + 1
        if self.use_cls_token: num_patches += 1
        self.patch_encoder = PatchEncoder(num_patches, self.projection_dim)
        
        # 3. Branchformer 层堆叠
        self.layers = nn.ModuleList([
            BranchformerBlock(self.projection_dim, self.num_heads, self.d_ff, self.dropout_rate, self.kernel_size)
            for _ in range(self.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.projection_dim)
        
        # 4. 输出投影
        if self.projection_dim != self.output_dim:
            self.output_projection = nn.Linear(self.projection_dim, self.output_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, time_steps, features)
        x = self.input_projection(x)
        x = self.patches(x)
        
        if self.use_cls_token:
            x = self.cls_token(x)
            
        x = self.patch_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.layer_norm(x)
        
        # 提取特征：CLS token 或平均池化
        features = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        self.intermediate_features = features.detach()
        
        return self.output_projection(features)

        
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