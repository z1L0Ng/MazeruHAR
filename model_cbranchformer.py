#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 设置随机种子以保持一致性
randomSeed = 1
torch.manual_seed(randomSeed)


class ConvModule(nn.Module):
    """
    卷积模块 - C-Branchformer中的卷积分支
    包含深度卷积、逐点卷积和激活函数
    """
    def __init__(self, channels, kernel_size=15, expansion_factor=2, dropout_rate=0.1):
        super(ConvModule, self).__init__()
        
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(channels)
        
        # 逐点卷积扩展
        self.pointwise_conv1 = nn.Conv1d(channels, channels * expansion_factor, kernel_size=1)
        
        # GLU激活
        self.glu = nn.GLU(dim=1)
        
        # 深度卷积
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=channels
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(channels)
        
        # Swish激活
        self.activation = nn.SiLU()
        
        # 逐点卷积压缩
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        # 转换为卷积格式: [batch_size, channels, seq_len]
        x = self.norm1(x)
        x = x.transpose(1, 2)
        
        # 逐点卷积扩展
        x = self.pointwise_conv1(x)
        
        # GLU激活
        x = self.glu(x)
        
        # 深度卷积
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # 逐点卷积压缩
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 转回序列格式: [batch_size, seq_len, channels]
        return x.transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 层归一化
        x_norm = self.norm(x)
        
        # 多头注意力
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        attn_output = self.dropout(attn_output)
        
        return attn_output


class CBranchformerLayer(nn.Module):
    """
    C-Branchformer层实现
    包含并行的MHA和ConvModule分支，以及Merge操作
    """
    def __init__(self, d_model, num_heads, conv_kernel_size=15, 
                 conv_expansion_factor=2, ffn_expansion_factor=4,
                 dropout_rate=0.1, gate_activation='sigmoid'):
        super(CBranchformerLayer, self).__init__()
        
        self.d_model = d_model
        self.gate_activation = gate_activation
        
        # MHA分支
        self.mha_branch = MultiHeadSelfAttention(d_model, num_heads, dropout_rate)
        
        # ConvModule分支
        self.conv_branch = ConvModule(
            channels=d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_rate=dropout_rate
        )
        
        # Merge门控机制
        self.merge_proj = nn.Linear(d_model, d_model * 2)
        
        # FFN层
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ffn_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * ffn_expansion_factor, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # 层归一化
        self.norm_final = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 保存残差连接的输入
        residual = x
        
        # 并行处理两个分支
        # MHA分支
        mha_out = self.mha_branch(x)
        
        # ConvModule分支  
        conv_out = self.conv_branch(x)
        
        # Merge操作 - 门控融合
        # 计算门控权重
        gate_input = mha_out + conv_out  # 简单相加作为门控输入
        gate_scores = self.merge_proj(gate_input)  # [batch, seq, 2*d_model]
        gate_scores = gate_scores.view(gate_scores.size(0), gate_scores.size(1), 2, self.d_model)
        
        if self.gate_activation == 'sigmoid':
            gate_weights = torch.sigmoid(gate_scores)
        else:
            gate_weights = torch.softmax(gate_scores, dim=2)
        
        # 应用门控权重 (×1/2的效果通过权重实现)
        merged = gate_weights[:, :, 0] * mha_out + gate_weights[:, :, 1] * conv_out
        
        # 第一个残差连接
        x = residual + merged
        
        # FFN层
        ffn_out = self.ffn(x)
        
        # 第二个残差连接和最终归一化
        x = x + ffn_out
        x = self.norm_final(x)
        
        return x


class SensorPatches(nn.Module):
    """
    将原始传感器数据转换为补丁表示
    """
    def __init__(self, projection_dim, patch_size, time_step):
        super(SensorPatches, self).__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim
        
        # 加速度计和陀螺仪数据的卷积投影
        self.acc_projection = nn.Conv1d(
            in_channels=3,  # 加速度计x, y, z
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.gyro_projection = nn.Conv1d(
            in_channels=3,  # 陀螺仪x, y, z
            out_channels=projection_dim // 2,
            kernel_size=patch_size,
            stride=time_step
        )
        
    def forward(self, input_data):
        # 输入形状: [batch_size, time_steps, channels]
        # 分割加速度计和陀螺仪数据
        acc_data = input_data[:, :, :3].permute(0, 2, 1)
        gyro_data = input_data[:, :, 3:].permute(0, 2, 1)
        
        # 应用卷积投影
        acc_projections = self.acc_projection(acc_data)
        gyro_projections = self.gyro_projection(gyro_data)
        
        # 组合投影并转换回序列格式
        acc_projections = acc_projections.permute(0, 2, 1)
        gyro_projections = gyro_projections.permute(0, 2, 1)
        
        # 拼接加速度计和陀螺仪投影
        projections = torch.cat((acc_projections, gyro_projections), dim=2)
        
        return projections


class PatchEncoder(nn.Module):
    """
    为输入补丁添加位置编码
    """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, device=patch.device)
        encoded = patch + self.position_embedding(positions).unsqueeze(0)
        return encoded


class ClassToken(nn.Module):
    """
    添加可学习的分类token到序列开头
    """
    def __init__(self, hidden_size):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        cls_broadcasted = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_broadcasted, inputs], dim=1)


class CBranchformerHAR(nn.Module):
    """
    基于C-Branchformer的人体活动识别模型
    结合了多头注意力和卷积模块的优势
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, 
                 patch_size=16, time_step=16, num_heads=4, num_layers=4,
                 conv_kernel_size=15, conv_expansion_factor=2, 
                 ffn_expansion_factor=4, dropout_rate=0.1, 
                 use_tokens=False, mlp_head_units=None,
                 # 为兼容原有模型添加的参数（会被忽略但不报错）
                 filter_attention_head=None, conv_kernels=None, 
                 rnn_hidden_dim=None, rnn_num_layers=None, 
                 rnn_type=None, rnn_bidirectional=None,
                 num_transformer_layers=None, transformer_dim_feedforward=None,
                 **kwargs):
        super(CBranchformerHAR, self).__init__()
        
        if mlp_head_units is None:
            mlp_head_units = [384]
        
        # 兼容性处理：如果传入了conv_kernels，使用其长度作为层数
        if conv_kernels is not None:
            num_layers = len(conv_kernels)
        
        # 兼容性处理：如果传入了num_transformer_layers，使用它作为层数
        if num_transformer_layers is not None:
            num_layers = num_transformer_layers
        
        # 保存参数
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 1. 补丁提取
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 2. 分类token（可选）
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 3. 位置编码
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        if self.use_tokens:
            n_patches += 1
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # 4. C-Branchformer层堆叠
        self.cbranchformer_layers = nn.ModuleList([
            CBranchformerLayer(
                d_model=projection_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                conv_expansion_factor=conv_expansion_factor,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # 5. 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 6. 分类头
        self.intermediate_features = None
        self.mlp_head = nn.Sequential()
        in_features = projection_dim
        for i, units in enumerate(mlp_head_units):
            self.mlp_head.add_module(f"dense_{i}", nn.Linear(in_features, units))
            self.mlp_head.add_module(f"activation_{i}", nn.SiLU())
            self.mlp_head.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            in_features = units
        
        # 7. 输出层
        self.classifier = nn.Linear(in_features, activity_count)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
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
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def extract_features(self, x):
        """
        特征提取方法
        """
        # 创建补丁
        x = self.patches(x)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        x = self.patch_encoder(x)
        
        # 通过C-Branchformer层
        for layer in self.cbranchformer_layers:
            x = layer(x)
        
        # 特征提取
        if self.use_tokens:
            # 使用分类token
            representation = x[:, 0]
        else:
            # 使用全局平均池化
            representation = x.mean(dim=1)
        
        # 存储中间特征
        self.intermediate_features = representation
        
        return representation
    
    def forward(self, inputs):
        # 特征提取
        representation = self.extract_features(inputs)
        
        # 通过MLP头
        for layer in self.mlp_head:
            representation = layer(representation)
        
        # 分类
        logits = self.classifier(representation)
        output = F.softmax(logits, dim=-1)
        
        return output


class LightweightCBranchformerHAR(nn.Module):
    """
    轻量级C-Branchformer HAR模型
    适用于资源受限的环境
    """
    def __init__(self, input_shape, activity_count, **kwargs):
        super(LightweightCBranchformerHAR, self).__init__()
        
        # 设置轻量级模型的默认参数
        default_params = {
            'projection_dim': 128,
            'patch_size': 16,
            'time_step': 16,
            'num_heads': 4,
            'num_layers': 2,
            'conv_kernel_size': 7,
            'conv_expansion_factor': 1,  # 减少膨胀因子
            'ffn_expansion_factor': 2,   # 减少FFN膨胀因子
            'dropout_rate': 0.1,
            'use_tokens': False,
            'mlp_head_units': [256]      # 减少MLP头的大小
        }
        
        # 处理兼容性参数
        if 'conv_kernels' in kwargs and kwargs['conv_kernels'] is not None:
            default_params['num_layers'] = min(len(kwargs['conv_kernels']), 3)  # 轻量级模型限制层数
        
        if 'num_transformer_layers' in kwargs and kwargs['num_transformer_layers'] is not None:
            default_params['num_layers'] = min(kwargs['num_transformer_layers'], 3)
        
        # 更新默认参数（自动忽略不支持的参数）
        for key, value in kwargs.items():
            if key in default_params:
                default_params[key] = value
        
        # 使用更小的配置
        self.model = CBranchformerHAR(
            input_shape=input_shape,
            activity_count=activity_count,
            **default_params
        )
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def extract_features(self, inputs):
        return self.model.extract_features(inputs)


# 辅助函数
def cbranchformer_har_base(input_shape, activity_count, **kwargs):
    """创建基础版本的C-Branchformer HAR模型"""
    # 设置默认参数，如果kwargs中有相同参数则使用kwargs中的值
    default_params = {
        'projection_dim': 192,
        'num_layers': 4,
        'num_heads': 4,
        'patch_size': 16,
        'time_step': 16,
        'conv_kernel_size': 15,
        'dropout_rate': 0.1
    }
    
    # 更新默认参数
    default_params.update(kwargs)
    
    return CBranchformerHAR(
        input_shape=input_shape,
        activity_count=activity_count,
        **default_params
    )


def cbranchformer_har_large(input_shape, activity_count, **kwargs):
    """创建大型版本的C-Branchformer HAR模型"""
    # 设置默认参数
    default_params = {
        'projection_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'mlp_head_units': [512, 256],
        'patch_size': 16,
        'time_step': 16,
        'conv_kernel_size': 15,
        'dropout_rate': 0.1
    }
    
    # 更新默认参数
    default_params.update(kwargs)
    
    return CBranchformerHAR(
        input_shape=input_shape,
        activity_count=activity_count,
        **default_params
    )


def cbranchformer_har_small(input_shape, activity_count, **kwargs):
    """创建小型版本的C-Branchformer HAR模型"""
    # 设置默认参数
    default_params = {
        'projection_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'conv_kernel_size': 7,
        'dropout_rate': 0.1
    }
    
    # 更新默认参数
    default_params.update(kwargs)
    
    return LightweightCBranchformerHAR(
        input_shape=input_shape,
        activity_count=activity_count,
        **default_params
    )


# 特征提取函数
def get_cbranchformer_features(model, dataloader, device):
    """
    提取C-Branchformer模型的特征表示用于分析
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # 提取特征
            features = model.extract_features(inputs).cpu().numpy()
            all_features.append(features)
            all_labels.extend(labels.numpy())
    
    # 合并所有特征
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels