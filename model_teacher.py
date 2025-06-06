#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Teacher Model for Human Activity Recognition
针对HART架构优化的Teacher模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union


# ============================================
# 基础组件
# ============================================

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) 正则化
    在训练期间随机丢弃整个残差分支
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output


class LayerScale(nn.Module):
    """
    Layer Scale用于稳定深层网络训练
    """
    def __init__(self, dim: int, init_values: float = 1e-4):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PatchEncoder(nn.Module):
    """
    为输入补丁添加可学习的位置编码
    """
    def __init__(self, num_patches: int, projection_dim: int):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = patch.shape
        positions = torch.arange(0, seq_len, device=patch.device)
        position_encoding = self.position_embedding(positions)
        return patch + position_encoding.unsqueeze(0)


class ClassToken(nn.Module):
    """
    添加可学习的分类token到序列开头
    """
    def __init__(self, hidden_size: int):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        cls_broadcasted = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_broadcasted, inputs], dim=1)


class SensorPatches(nn.Module):
    """
    将原始传感器数据转换为补丁表示
    针对HART架构优化，分别处理加速度计和陀螺仪数据
    """
    def __init__(self, projection_dim: int, patch_size: int, time_step: int):
        super(SensorPatches, self).__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim
        
        # HART使用分离的加速度计和陀螺仪投影
        self.acc_projection = nn.Conv1d(
            in_channels=3,  # 加速度计 x, y, z
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step,
            padding=0
        )
        
        self.gyro_projection = nn.Conv1d(
            in_channels=3,  # 陀螺仪 x, y, z
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step,
            padding=0
        )
        
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # 输入形状: [batch_size, time_steps, channels]
        # 分割加速度计和陀螺仪数据
        acc_data = input_data[:, :, :3].permute(0, 2, 1)
        gyro_data = input_data[:, :, 3:6].permute(0, 2, 1)
        
        # 应用卷积投影
        acc_projections = self.acc_projection(acc_data)
        gyro_projections = self.gyro_projection(gyro_data)
        
        # 转换回序列格式
        acc_projections = acc_projections.permute(0, 2, 1)
        gyro_projections = gyro_projections.permute(0, 2, 1)
        
        # 拼接投影
        projections = torch.cat((acc_projections, gyro_projections), dim=2)
        
        return projections


# ============================================
# 增强的注意力组件
# ============================================

class FilterBasedMultiHeadAttention(nn.Module):
    """
    HART的过滤器基础多头注意力机制
    """
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int, dropout_rate: float = 0.0):
        super(FilterBasedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        
        # 深度卷积用于局部注意力
        self.depthwise_conv = nn.Conv1d(
            embed_dim, embed_dim, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim
        )
        
        # 标准多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, return_attention_scores: bool = False):
        # 局部特征提取
        conv_out = x.transpose(1, 2)  # [B, D, T]
        conv_out = self.depthwise_conv(conv_out)
        conv_out = conv_out.transpose(1, 2)  # [B, T, D]
        
        # 多头注意力
        if return_attention_scores:
            attn_out, attn_weights = self.multihead_attn(x, x, x, need_weights=True)
            return attn_out + conv_out, attn_weights
        else:
            attn_out, _ = self.multihead_attn(x, x, x, need_weights=False)
            return attn_out + conv_out


class FeedForward(nn.Module):
    """
    增强的前馈网络，使用GELU激活函数
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class HARTBlock(nn.Module):
    """
    HART块，包含分离的加速度计和陀螺仪注意力
    """
    def __init__(self, d_model: int, num_heads: int, kernel_size: int, 
                 dim_feedforward: int, dropout_rate: float = 0.1, 
                 drop_path_rate: float = 0.0):
        super(HARTBlock, self).__init__()
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 分离的注意力头
        self.acc_attention = FilterBasedMultiHeadAttention(
            d_model // 2, num_heads // 2, kernel_size, dropout_rate
        )
        self.gyro_attention = FilterBasedMultiHeadAttention(
            d_model // 2, num_heads // 2, kernel_size, dropout_rate
        )
        
        # 前馈网络
        self.ffn = FeedForward(d_model, dim_feedforward, dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout和DropPath
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, return_attention_scores: bool = False):
        # 归一化
        normed = self.norm(x)
        
        # 分离加速度计和陀螺仪特征
        acc_features = normed[:, :, :normed.size(2)//2]
        gyro_features = normed[:, :, normed.size(2)//2:]
        
        # 分别应用注意力
        if return_attention_scores:
            acc_out, acc_weights = self.acc_attention(acc_features, return_attention_scores=True)
            gyro_out, gyro_weights = self.gyro_attention(gyro_features, return_attention_scores=True)
        else:
            acc_out = self.acc_attention(acc_features)
            gyro_out = self.gyro_attention(gyro_features)
        
        # 合并输出
        attn_out = torch.cat([acc_out, gyro_out], dim=2)
        
        # 残差连接
        x = x + self.drop_path(self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.dropout(ffn_out))
        
        if return_attention_scores:
            return x, (acc_weights, gyro_weights)
        return x


# ============================================
# 主模型 - Enhanced HART Teacher
# ============================================

class EnhancedHARTTeacher(nn.Module):
    """
    增强版HART Teacher模型
    基于原始HART架构，增加深度和宽度用于知识蒸馏
    """
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 activity_count: int,
                 # 基础参数 - 增强版本
                 projection_dim: int = 384,  # 从192增加到384
                 patch_size: int = 16,
                 time_step: int = 16,
                 # 注意力参数
                 num_heads: int = 8,  # 从4增加到8
                 conv_kernels: List[int] = None,  # [3, 7, 15, 31, 31, 31, 63, 63]
                 # 正则化
                 dropout_rate: float = 0.2,  # 从0.3降低到0.2
                 drop_path_max_rate: float = 0.3,
                 # 其他
                 use_tokens: bool = False,
                 mlp_head_units: List[int] = None,
                 temperature_init: float = 4.0):
        
        super(EnhancedHARTTeacher, self).__init__()
        
        # 默认使用更深的架构
        if conv_kernels is None:
            conv_kernels = [3, 7, 15, 31, 31, 31, 63, 63]  # 8层而不是6层
        
        if mlp_head_units is None:
            mlp_head_units = [768, 512]  # 更大的MLP头
        
        # 保存配置
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.num_heads = num_heads
        self.conv_kernels = conv_kernels
        self.num_blocks = len(conv_kernels)
        self.use_tokens = use_tokens
        
        # 计算补丁数量
        self.num_patches = (input_shape[0] - patch_size) // time_step + 1
        if use_tokens:
            self.num_patches += 1
        
        # 构建模型组件
        # 1. 补丁嵌入
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 2. 分类Token（可选）
        if use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 3. 位置编码
        self.patch_encoder = PatchEncoder(self.num_patches, projection_dim)
        
        # 4. HART块
        self.hart_blocks = nn.ModuleList()
        for i, kernel_size in enumerate(conv_kernels):
            # 渐进式DropPath
            drop_path_rate = drop_path_max_rate * (i / max(1, len(conv_kernels) - 1))
            
            block = HARTBlock(
                d_model=projection_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dim_feedforward=projection_dim * 4,  # 标准的4倍扩展
                dropout_rate=dropout_rate,
                drop_path_rate=drop_path_rate
            )
            self.hart_blocks.append(block)
            
            # 注册命名以保持与原始HART的兼容性
            self.register_module(f"normalizedInputs_{i}", block.norm)
            self.register_module(f"AccMHA_{i}", block.acc_attention)
            self.register_module(f"GyroMHA_{i}", block.gyro_attention)
        
        # 5. 最终层归一化
        self.final_norm = nn.LayerNorm(projection_dim)
        
        # 6. 分类头
        self.GAP = nn.AdaptiveAvgPool1d(1) if not use_tokens else None
        
        # MLP头
        layers = []
        in_features = projection_dim
        for units in mlp_head_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = units
        
        self.mlp_head = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, activity_count)
        
        # 7. 知识蒸馏温度
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        # 存储中间特征
        self.intermediate_features = {}
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取"""
        # 1. 创建补丁
        x = self.patches(x)
        self.intermediate_features['patches'] = x.clone()
        
        # 2. 添加分类token
        if self.use_tokens:
            x = self.class_token(x)
        
        # 3. 位置编码
        x = self.patch_encoder(x)
        self.intermediate_features['encoded'] = x.clone()
        
        # 4. 通过HART块
        for i, block in enumerate(self.hart_blocks):
            x = block(x)
            self.intermediate_features[f'block_{i}'] = x.clone()
        
        # 5. 最终归一化
        x = self.final_norm(x)
        self.intermediate_features['normalized'] = x.clone()
        
        # 6. 池化
        if self.use_tokens:
            features = x[:, 0]
        else:
            # 全局平均池化
            features = x.mean(dim=1)
        
        self.intermediate_features['pooled'] = features.clone()
        
        return features
    
    def forward(self, inputs: torch.Tensor, 
                return_features: bool = False,
                return_logits: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            inputs: 输入张量 [batch_size, time_steps, 6]
            return_features: 是否返回特征字典（用于知识蒸馏）
            return_logits: 是否返回原始logits
        """
        # 特征提取
        features = self.extract_features(inputs)
        
        # MLP头
        mlp_features = self.mlp_head(features)
        
        # 分类
        logits = self.classifier(mlp_features)
        
        if return_features:
            # 返回用于知识蒸馏的特征
            return {
                'logits': logits,
                'probabilities': F.softmax(logits / self.temperature, dim=-1),
                'features': features,
                'mlp_features': mlp_features,
                'temperature': self.temperature,
                'intermediate': self.intermediate_features
            }
        elif return_logits:
            return logits
        else:
            return F.softmax(logits, dim=-1)
    
    def get_attention_weights(self, inputs: torch.Tensor, block_idx: int = -1):
        """获取指定块的注意力权重"""
        # 前向传播到指定块
        x = self.patches(inputs)
        if self.use_tokens:
            x = self.class_token(x)
        x = self.patch_encoder(x)
        
        # 如果block_idx为-1，使用最后一个块
        if block_idx == -1:
            block_idx = len(self.hart_blocks) - 1
        
        # 通过块直到目标块
        for i in range(block_idx + 1):
            if i == block_idx:
                x, (acc_weights, gyro_weights) = self.hart_blocks[i](x, return_attention_scores=True)
                return acc_weights, gyro_weights
            else:
                x = self.hart_blocks[i](x)
        
        return None, None


# ============================================
# 创建不同规模的Teacher模型
# ============================================

def create_enhanced_teacher(input_shape: Tuple[int, int], 
                          activity_count: int,
                          model_scale: str = 'xxlarge') -> EnhancedHARTTeacher:
    """
    创建增强版Teacher模型
    
    Args:
        input_shape: (time_steps, channels) 例如 (128, 6)
        activity_count: 活动类别数
        model_scale: 模型规模 'large', 'xlarge', 'xxlarge'
    """
    
    configs = {
        'large': {
            'projection_dim': 256,
            'num_heads': 8,
            'conv_kernels': [3, 7, 15, 31, 31, 31],  # 6层
            'mlp_head_units': [512, 384],
            'dropout_rate': 0.2,
            'drop_path_max_rate': 0.2
        },
        'xlarge': {
            'projection_dim': 384,
            'num_heads': 12,
            'conv_kernels': [3, 7, 15, 31, 31, 31, 63, 63],  # 8层
            'mlp_head_units': [768, 512],
            'dropout_rate': 0.15,
            'drop_path_max_rate': 0.3
        },
        'xxlarge': {
            'projection_dim': 512,
            'num_heads': 16,
            'conv_kernels': [3, 7, 15, 31, 31, 31, 63, 63, 127, 127],  # 10层
            'mlp_head_units': [1024, 768, 512],
            'dropout_rate': 0.1,
            'drop_path_max_rate': 0.4
        }
    }
    
    config = configs[model_scale]
    
    return EnhancedHARTTeacher(
        input_shape=input_shape,
        activity_count=activity_count,
        **config
    )


# ============================================
# 工具函数
# ============================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return total_params, trainable_params


def prepare_training_config(model_scale: str = 'xxlarge'):
    """
    准备训练配置
    
    Returns:
        包含优化器、调度器等的配置字典
    """
    
    configs = {
        'large': {
            'lr': 5e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 5,
            'total_epochs': 100,
            'batch_size': 128
        },
        'xlarge': {
            'lr': 3e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 10,
            'total_epochs': 150,
            'batch_size': 64
        },
        'xxlarge': {
            'lr': 1e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 15,
            'total_epochs': 200,
            'batch_size': 32
        }
    }
    
    return configs[model_scale]


# ============================================
# 示例和测试
# ============================================

# 设置随机种子
def set_seed(seed=1):
    """设置所有随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # 设置随机种子
    set_seed(1)
    
    # 模型参数（与您的训练脚本一致）
    input_shape = (128, 6)  # 128时间步，6个通道
    activity_count = 6      # UCI数据集的6个活动
    
    print("Creating Enhanced HART Teacher Model")
    print("="*60)
    
    # 创建最大规模的Teacher模型
    teacher = create_enhanced_teacher(
        input_shape=input_shape,
        activity_count=activity_count,
        model_scale='xxlarge'
    )
    
    # 统计参数
    total_params, trainable_params = count_parameters(teacher)
    
    # 测试前向传播
    batch_size = 4
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # 标准输出
    with torch.no_grad():
        output = teacher(dummy_input)
        print(f"\nOutput shape: {output.shape}")
        
        # 知识蒸馏输出
        kd_features = teacher(dummy_input, return_features=True)
        print("\nKnowledge Distillation outputs:")
        for key, value in kd_features.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  - {key}: {len(value)} intermediate features")
        
        # 获取注意力权重
        acc_weights, gyro_weights = teacher.get_attention_weights(dummy_input)
        print(f"\nAttention weights shape:")
        print(f"  - Accelerometer: {acc_weights.shape}")
        print(f"  - Gyroscope: {gyro_weights.shape}")
    
    # 训练配置建议
    print("\n" + "="*60)
    print("Recommended Training Configuration:")
    config = prepare_training_config('xxlarge')
    for key, value in config.items():
        print(f"  - {key}: {value}")
    
    print("\nModel ready for knowledge distillation!")