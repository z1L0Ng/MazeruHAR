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


class DropPath(nn.Module):
    """
    DropPath操作，也被称为Stochastic Depth，在训练期间随机丢弃整个路径(层)
    """
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output


class MobileMambaHART(nn.Module):
    """
    移动版MambaHART，适合在资源受限设备上部署
    使用向量化操作提高效率
    """
    def __init__(self, input_shape, activity_count, projection_dim=128, patch_size=16, time_step=16, 
                 d_state=8, expand=2, dropout_rate=0.1, use_tokens=False):
        super(MobileMambaHART, self).__init__()
        
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.d_state = d_state
        self.expand = expand
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 定义模型组件
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 如果使用分类token，则添加
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 计算补丁数量
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        if self.use_tokens:
            n_patches += 1  # 额外添加分类token
            
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # Mamba块
        self.mamba_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(projection_dim),
                MobileMamba(
                    d_model=projection_dim,
                    d_state=d_state,
                    expand=expand,
                    dropout_rate=dropout_rate
                ),
                ResidualAdd()
            )
            for _ in range(3)  # 更少的层
        ])
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(projection_dim)
        
        # 分类头
        self.head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim * 2, activity_count)
        )
    
    def forward(self, inputs):
        # 创建补丁
        x = self.patches(inputs)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        x = self.patch_encoder(x)
        
        # 通过Mamba块
        for block in self.mamba_blocks:
            # 模块内部已有残差连接
            x = block((block[0:2](x), x))
        
        # 最终层归一化
        x = self.final_layer_norm(x)
        
        # 提取表示
        if self.use_tokens:
            # 使用分类token作为表示
            x = x[:, 0]
        else:
            # 使用全局平均池化作为表示
            x = x.mean(dim=1)
        
        # 分类头
        logits = self.head(x)
        output = F.softmax(logits, dim=-1)
        
        return output


def mamba_hart_small(input_shape, activity_count, **kwargs):
    """
    小型MambaHART模型工厂函数
    """
    return MambaHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=128,  # 较小的投影维度
        patch_size=16, 
        time_step=16,
        d_state=12,  # 较小的状态维度
        expand=2,  # 较小的扩展因子
        conv_kernels=[3, 7, 15],  # 更少的层
        mlp_head_units=[512],  # 更小的MLP头
        dropout_rate=0.2,
        **kwargs
    )


def mamba_hart_tiny(input_shape, activity_count, **kwargs):
    """
    微型MambaHART模型工厂函数，适用于非常受限的设备
    """
    return MambaHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=64,  # 很小的投影维度
        patch_size=16, 
        time_step=16,
        d_state=8,  # 很小的状态维度
        expand=2,  # 最小的扩展因子
        conv_kernels=[3, 7],  # 只有两层
        mlp_head_units=[256],  # 很小的MLP头
        dropout_rate=0.1,
        **kwargs
    )


def mamba_hart_micro(input_shape, activity_count, **kwargs):
    """
    微型MambaHART模型，极简版本用于快速验证
    """
    return MambaHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=32,  # 极小的投影维度
        patch_size=16, 
        time_step=16,
        d_state=4,  # 极小的状态维度
        expand=1,  # 无扩展
        conv_kernels=[3],  # 只有一层
        mlp_head_units=[128],  # 很小的MLP头
        dropout_rate=0.0,  # 禁用dropout以加速训练
        **kwargs
    )


def mobile_mamba_hart(input_shape, activity_count, **kwargs):
    """
    移动版MambaHART模型工厂函数
    """
    return MobileMambaHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=96,
        patch_size=16,
        time_step=16,
        d_state=8,
        expand=2,
        dropout_rate=0.1,
        **kwargs
    )

class GatedLinearUnit(nn.Module):
    """
    门控线性单元，将输入分成两部分，一部分通过sigmoid门控另一部分
    """
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.units = units
        self.linear = nn.Linear(units, units * 2)
        
    def forward(self, inputs):
        linear_projection = self.linear(inputs)
        # 将输出分为两半
        gate, value = torch.split(linear_projection, self.units, dim=-1)
        # 对门应用sigmoid激活函数
        gate = torch.sigmoid(gate)
        # 返回门控输出
        return value * gate


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
    添加可学习的分类token到序列开头，类似于BERT的[CLS]token或ViT的class token
    """
    def __init__(self, hidden_size):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        cls_broadcasted = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_broadcasted, inputs], dim=1)


class Prompts(nn.Module):
    """
    在序列末尾添加可学习的提示向量
    """
    def __init__(self, projection_dims, prompt_count=1):
        super(Prompts, self).__init__()
        self.projection_dims = projection_dims
        self.prompt_count = prompt_count
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, projection_dims))
            for _ in range(prompt_count)
        ])
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        prompt_broadcasted = torch.cat([
            prompt.expand(batch_size, 1, -1) for prompt in self.prompts
        ], dim=1)
        return torch.cat([inputs, prompt_broadcasted], dim=1)


class SimpleMamba(nn.Module):
    """
    优化版简化Mamba状态空间模型，使用向量化操作
    """
    def __init__(self, projection_quarter, start_index, stop_index, d_state=16, dropout_rate=0.0, drop_path_rate=0.0, num_heads=None):
        super(SimpleMamba, self).__init__()
        self.start_index = start_index
        self.stop_index = stop_index
        self.dropout_rate = dropout_rate
        self.drop_path = DropPath(drop_path_rate)
        
        # 计算输入维度
        d_model = stop_index - start_index
        self.d_model = d_model
        self.d_state = d_state
        
        # S4/Mamba核心参数
        # A参数 - 状态转移矩阵的对角线元素
        self.A_log = nn.Parameter(torch.randn(self.d_model, self.d_state))
        # B参数 - 输入投影
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state))
        # C参数 - 输出投影
        self.C = nn.Parameter(torch.randn(self.d_model, self.d_state))
        # D参数 - 跳跃连接
        self.D = nn.Parameter(torch.randn(self.d_model))
        
        # 输入和输出投影
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 卷积层，模拟Mamba的局部依赖获取
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model  # 深度卷积
        )
        
        # 门控机制
        self.gate_proj = nn.Linear(d_model, d_model)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 参数初始化为稳定的值
        nn.init.normal_(self.A_log, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        nn.init.normal_(self.D, mean=0.0, std=0.1)
    
    def forward(self, input_data, return_attention_scores=False):
        # 提取特定通道的输入数据
        x = input_data[:, :, self.start_index:self.stop_index]
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.in_proj(x)
        
        # 转换为卷积格式
        x_conv = x.transpose(1, 2)  # [B, C, L]
        
        # 应用卷积获取局部信息
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, C]
        
        # 计算SSM的A矩阵 (对数域转换为实数域)
        # 使用负指数确保状态稳定
        A = -torch.exp(self.A_log)  # [D, N]
        
        # 门控网络
        gate = torch.sigmoid(self.gate_proj(x))
        x = x * gate
        
        # 向量化实现状态更新
        # 将A转换为批量广播格式 [1, 1, D, N]
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        
        # 将输入x变形为 [B, L, D, 1]
        x_expanded = x.unsqueeze(-1)
        
        # 将B转换为 [1, 1, D, N]
        B_expanded = self.B.unsqueeze(0).unsqueeze(0)
        
        # 初始化隐藏状态 [B, D, N]
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        
        # 预计算B*u_t为所有时间步 [B, L, D, N]
        Bu = x_expanded * B_expanded
        
        # 声明输出张量
        output = torch.zeros_like(x)
        
        # 高效状态更新
        for t in range(seq_len):
            # 更新状态: h_t = A * h_{t-1} + B * u_t
            h = A_expanded[:, 0] * h + Bu[:, t]
            
            # 计算输出: y_t = C * h_t + D * u_t
            # C: [D, N], h: [B, D, N] -> [B, D]
            Ch = torch.sum(self.C.unsqueeze(0) * h, dim=2)
            
            # D: [D], x[:, t]: [B, D] -> [B, D]
            Dx = self.D * x[:, t]
            
            # 合并为最终输出
            output[:, t] = Ch + Dx
        
        # 融合局部卷积特征和SSM输出
        output = output + x_conv
        
        # 输出投影
        output = self.out_proj(output)
        
        # DropPath
        output = self.drop_path(output)
        
        if return_attention_scores:
            # 仅作为接口兼容，不返回实际注意力分数
            dummy_attention = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            return output, dummy_attention
        else:
            return output


class MambaFormer(nn.Module):
    """
    优化版Mamba状态空间模型替代LiteFormer，使用向量化操作
    """
    def __init__(self, start_index, stop_index, projection_size, kernel_size=16, d_state=16, expand=2, 
                 attention_head=None, use_bias=False, drop_path_rate=0.0, dropout_rate=0):
        super(MambaFormer, self).__init__()
        self.use_bias = use_bias
        self.start_index = start_index
        self.stop_index = stop_index
        self.kernel_size = kernel_size
        self.projection_size = projection_size
        self.drop_path = DropPath(drop_path_rate)
        
        # 计算输入维度
        d_model = stop_index - start_index
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # 输入扩展
        self.expand_proj = nn.Linear(d_model, d_model * expand)
        
        # SiLU激活
        self.act = nn.SiLU()
        
        # 卷积层捕获局部上下文
        self.conv1d = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model * expand,
            kernel_size=kernel_size if kernel_size % 2 == 1 else kernel_size + 1,  # 确保奇数
            padding='same',
            groups=d_model * expand  # 深度卷积
        )
        
        # SSM核心参数
        self.A_log = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.B = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.C = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.D = nn.Parameter(torch.randn(d_model * expand))
        
        # 输出投影
        self.out_proj = nn.Linear(d_model * expand, d_model)
        
        # 门控机制
        self.gate = nn.Linear(d_model * expand, d_model * expand)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 参数初始化为稳定的值
        nn.init.normal_(self.A_log, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        nn.init.normal_(self.D, mean=0.0, std=0.1)
    
    def forward(self, inputs):
        # 提取相关的输入通道
        x = inputs[:, :, self.start_index:self.stop_index]
        batch_size, seq_len, _ = x.shape
        
        # 输入扩展
        x = self.expand_proj(x)
        x = self.act(x)
        
        # 应用卷积获取局部信息
        x_conv = x.transpose(1, 2)  # [B, C, L]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, L, C]
        
        # 门控机制
        gate_output = torch.sigmoid(self.gate(x_conv))
        x_gated = x_conv * gate_output
        
        # 计算SSM的A矩阵 (对数域转换为实数域)
        A = -torch.exp(self.A_log)  # [D*expand, N]
        
        # 向量化状态更新实现
        d_expanded = self.d_model * self.expand
        
        # 将A转换为批量可广播格式 [1, 1, D*expand, N]
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        
        # 将输入x_gated变形为 [B, L, D*expand, 1]
        x_expanded = x_gated.unsqueeze(-1)
        
        # 将B转换为 [1, 1, D*expand, N]
        B_expanded = self.B.unsqueeze(0).unsqueeze(0)
        
        # 初始化隐藏状态 [B, D*expand, N]
        h = torch.zeros(batch_size, d_expanded, self.d_state, device=x.device)
        
        # 预计算B*u_t为所有时间步 [B, L, D*expand, N]
        Bu = x_expanded * B_expanded
        
        # 声明输出张量
        output = torch.zeros_like(x_gated)
        
        # 高效状态更新
        for t in range(seq_len):
            # 更新状态: h_t = A * h_{t-1} + B * u_t
            h = A_expanded[:, 0] * h + Bu[:, t]
            
            # 计算输出: y_t = C * h_t + D * u_t
            # C: [D*expand, N], h: [B, D*expand, N] -> [B, D*expand]
            Ch = torch.sum(self.C.unsqueeze(0) * h, dim=2)
            
            # D: [D*expand], x_gated[:, t]: [B, D*expand] -> [B, D*expand]
            Dx = self.D * x_gated[:, t]
            
            # 合并为最终输出
            output[:, t] = Ch + Dx
        
        # 输出投影，降维回原始大小
        output = self.out_proj(output)  # [B, L, D]
        
        # DropPath
        output = self.drop_path(output)
        
        return output


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
        # 转换为卷积格式: [batch_size, channels, time_steps]
        
        # 分割加速度计和陀螺仪数据
        acc_data = input_data[:, :, :3].permute(0, 2, 1)
        gyro_data = input_data[:, :, 3:].permute(0, 2, 1)
        
        # 应用卷积投影
        acc_projections = self.acc_projection(acc_data)
        gyro_projections = self.gyro_projection(gyro_data)
        
        # 组合投影并转换回序列格式
        # [batch_size, channels, patches] -> [batch_size, patches, channels]
        acc_projections = acc_projections.permute(0, 2, 1)
        gyro_projections = gyro_projections.permute(0, 2, 1)
        
        # 拼接加速度计和陀螺仪投影
        projections = torch.cat((acc_projections, gyro_projections), dim=2)
        
        return projections


class MambaHART(nn.Module):
    """
    使用优化版自定义Mamba实现的HART模型
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16, 
                 d_state=16, expand=4, filter_attention_head=4, num_heads=3, conv_kernels=None, 
                 mlp_head_units=None, dropout_rate=0.3, use_tokens=False):
        super(MambaHART, self).__init__()
        
        if conv_kernels is None:
            conv_kernels = [3, 7, 15, 31, 31, 31]
        
        if mlp_head_units is None:
            mlp_head_units = [1024]
        
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.d_state = d_state
        self.expand = expand
        self.num_heads = num_heads
        self.filter_attention_head = filter_attention_head
        self.conv_kernels = conv_kernels
        self.mlp_head_units = mlp_head_units
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 计算投影维度的一半和四分之一
        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        
        # 计算线性增加的drop path rate
        self.drop_path_rates = torch.linspace(0, dropout_rate * 10, len(conv_kernels)) * 0.1
        
        # Transformer单元大小
        self.transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]
        
        # 定义模型组件
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 如果使用分类token，则添加
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 添加位置编码
        # 注意：补丁数量需要根据输入形状和补丁大小计算
        # 假设input_shape = (time_steps, channels)
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        if self.use_tokens:
            n_patches += 1  # 额外添加分类token
            
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # 创建Transformer块的序列
        self.transformer_blocks = nn.ModuleList()
        
        # 为每个卷积核创建层归一化和Mamba块
        for i, kernel_length in enumerate(conv_kernels):
            # 层归一化
            self.transformer_blocks.append(nn.LayerNorm(projection_dim))
            
            # 使用MambaFormer替代LiteFormer
            self.transformer_blocks.append(
                MambaFormer(
                    start_index=self.projection_quarter,
                    stop_index=self.projection_quarter + self.projection_half,
                    projection_size=self.projection_half,
                    kernel_size=kernel_length,
                    attention_head=filter_attention_head,
                    d_state=d_state,
                    expand=expand,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 使用SimpleMamba替代加速度计MHA分支
            self.transformer_blocks.append(
                SimpleMamba(
                    projection_quarter=self.projection_quarter,
                    start_index=0,
                    stop_index=self.projection_quarter,
                    d_state=d_state,
                    num_heads=num_heads,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 使用SimpleMamba替代陀螺仪MHA分支
            self.transformer_blocks.append(
                SimpleMamba(
                    projection_quarter=self.projection_quarter,
                    start_index=self.projection_quarter + self.projection_half,
                    stop_index=projection_dim,
                    d_state=d_state,
                    num_heads=num_heads,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 层归一化和MLP
            self.transformer_blocks.append(nn.LayerNorm(projection_dim))
            
            # MLP块
            self.transformer_blocks.append(
                nn.Sequential(
                    nn.Linear(projection_dim, self.transformer_units[0]),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.transformer_units[0], projection_dim)
                )
            )
            
            # DropPath
            self.transformer_blocks.append(DropPath(self.drop_path_rates[i].item()))
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(projection_dim)
        
        # 分类头
        self.mlp_head = nn.ModuleList()
        in_features = projection_dim
        for units in mlp_head_units:
            self.mlp_head.append(nn.Linear(in_features, units))
            self.mlp_head.append(nn.SiLU())
            self.mlp_head.append(nn.Dropout(dropout_rate))
            in_features = units
        
        # 输出层
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
    
    def forward(self, inputs):
        # 创建补丁
        x = self.patches(inputs)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        encoded_patches = self.patch_encoder(x)
        
        # 应用Transformer块
        for i in range(0, len(self.transformer_blocks), 7):
            # 应用层归一化
            x1 = self.transformer_blocks[i](encoded_patches)
            
            # 计算预期维度
            expected_half = self.projection_half
            expected_quarter = self.projection_quarter
            
            # MambaFormer分支
            branch1 = self.transformer_blocks[i + 1](x1)
            
            # 确保branch1维度与预期相同
            if branch1.size(-1) != expected_half:
                branch1 = F.interpolate(
                    branch1.permute(0, 2, 1),
                    size=branch1.size(1),
                    mode='linear'
                ).permute(0, 2, 1)
            
            # 加速度计Mamba分支
            branch2_acc = self.transformer_blocks[i + 2](x1)
            
            # 陀螺仪Mamba分支
            branch2_gyro = self.transformer_blocks[i + 3](x1)
            
            # 确保所有分支维度正确
            if branch2_acc.size(-1) != expected_quarter:
                branch2_acc = F.interpolate(
                    branch2_acc.permute(0, 2, 1),
                    size=branch2_acc.size(1),
                    mode='linear'
                ).permute(0, 2, 1)
            
            if branch2_gyro.size(-1) != expected_quarter:
                branch2_gyro = F.interpolate(
                    branch2_gyro.permute(0, 2, 1),
                    size=branch2_gyro.size(1),
                    mode='linear'
                ).permute(0, 2, 1)
            
            # 拼接所有分支
            concat_attention = torch.cat((branch2_acc, branch1, branch2_gyro), dim=2)
            
            # 残差连接
            x2 = concat_attention + encoded_patches
            
            # 层归一化
            x3 = self.transformer_blocks[i + 4](x2)
            
            # MLP
            mlp_output = self.transformer_blocks[i + 5](x3)
            
            # DropPath
            mlp_output = self.transformer_blocks[i + 6](mlp_output)
            
            # 最终残差连接
            encoded_patches = mlp_output + x2
        
        # 最终层归一化
        representation = self.final_layer_norm(encoded_patches)
        
        # 提取表示
        if self.use_tokens:
            # 使用分类token作为表示
            representation = representation[:, 0]
        else:
            # 使用全局平均池化作为表示
            representation = representation.mean(dim=1)
        
        # 应用MLP头
        for layer in self.mlp_head:
            representation = layer(representation)
        
        # 分类层
        logits = self.classifier(representation)
        output = F.softmax(logits, dim=-1)
        
        return output


class ThreeSensorPatches(nn.Module):
    """
    用于三个传感器（加速度计、陀螺仪和磁力计）的补丁提取
    """
    def __init__(self, projection_dim, patch_size, time_step):
        super(ThreeSensorPatches, self).__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim
        
        # 三个传感器数据的卷积投影
        self.acc_projection = nn.Conv1d(
            in_channels=3,  # 加速度计x, y, z
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.gyro_projection = nn.Conv1d(
            in_channels=3,  # 陀螺仪x, y, z
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.mag_projection = nn.Conv1d(
            in_channels=3,  # 磁力计x, y, z
            out_channels=projection_dim // 3,
            kernel_size=patch_size,
            stride=time_step
        )
        
    def forward(self, input_data):
        # 分割三个传感器的数据
        acc_data = input_data[:, :, :3].permute(0, 2, 1)
        gyro_data = input_data[:, :, 3:6].permute(0, 2, 1)
        mag_data = input_data[:, :, 6:].permute(0, 2, 1)
        
        # 应用卷积投影
        acc_projections = self.acc_projection(acc_data)
        gyro_projections = self.gyro_projection(gyro_data)
        mag_projections = self.mag_projection(mag_data)
        
        # 转换回序列格式
        acc_projections = acc_projections.permute(0, 2, 1)
        gyro_projections = gyro_projections.permute(0, 2, 1)
        mag_projections = mag_projections.permute(0, 2, 1)
        
        # 拼接所有投影
        projections = torch.cat((acc_projections, gyro_projections, mag_projections), dim=2)
        
        return projections


class FourSensorPatches(nn.Module):
    """
    用于四个传感器的补丁提取
    """
    def __init__(self, projection_dim, patch_size, time_step):
        super(FourSensorPatches, self).__init__()
        self.patch_size = patch_size
        self.time_step = time_step
        self.projection_dim = projection_dim
        
        # 四个传感器数据的卷积投影
        self.acc_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.gyro_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.mag_projection = nn.Conv1d(
            in_channels=3,
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        
        self.alt_projection = nn.Conv1d(
            in_channels=3,  # 假设第四个传感器也有三个轴
            out_channels=projection_dim // 4,
            kernel_size=patch_size,
            stride=time_step
        )
        
    def forward(self, input_data):
        # 分割四个传感器的数据
        acc_data = input_data[:, :, :3].permute(0, 2, 1)
        gyro_data = input_data[:, :, 3:6].permute(0, 2, 1)
        mag_data = input_data[:, :, 6:9].permute(0, 2, 1)
        alt_data = input_data[:, :, 9:].permute(0, 2, 1)
        
        # 应用卷积投影
        acc_projections = self.acc_projection(acc_data)
        gyro_projections = self.gyro_projection(gyro_data)
        mag_projections = self.mag_projection(mag_data)
        alt_projections = self.alt_projection(alt_data)
        
        # 转换回序列格式
        acc_projections = acc_projections.permute(0, 2, 1)
        gyro_projections = gyro_projections.permute(0, 2, 1)
        mag_projections = mag_projections.permute(0, 2, 1)
        alt_projections = alt_projections.permute(0, 2, 1)
        
        # 拼接所有投影
        projections = torch.cat((acc_projections, gyro_projections, mag_projections, alt_projections), dim=2)
        
        return projections


class ResidualAdd(nn.Module):
    """残差连接辅助类"""
    def forward(self, x):
        input_tensor, residual_tensor = x
        return input_tensor + residual_tensor


def extract_intermediate_model_from_base_model(model, layer_idx=-4):
    """从模型中提取中间特征的钩子机制
    
    Args:
        model: 基础模型
        layer_idx: 需要提取特征的层索引或名称
        
    Returns:
        一个接收输入并返回特定层输出的函数
    """
    class IntermediateModel(torch.nn.Module):
        def __init__(self, base_model, target_layer):
            super().__init__()
            self.base_model = base_model
            self.target_layer = target_layer
            self.features = None
            
            # 用于临时保存特征的钩子函数
            def hook_fn(module, input, output):
                self.features = output
            
            # 找到目标层并注册钩子
            if isinstance(target_layer, int):
                # 如果是索引，找到相应位置的层
                for i, (name, module) in enumerate(model.named_modules()):
                    if i == target_layer:
                        self.hook = module.register_forward_hook(hook_fn)
                        break
            else:
                # 如果是名称，通过名称查找层
                for name, module in model.named_modules():
                    if name == target_layer:
                        self.hook = module.register_forward_hook(hook_fn)
                        break
        
        def forward(self, x):
            # 运行整个模型但只返回目标层的输出
            self.base_model(x)
            features = self.features
            self.features = None  # 清空以避免内存泄漏
            return features
    
    return IntermediateModel(model, layer_idx)


# 创建移动版本的MambaHART (MobileHART版本)
class MobileMamba(nn.Module):
    """
    轻量级Mamba模块，适用于移动部署，使用向量化操作提高效率
    """
    def __init__(self, d_model, d_state=8, expand=2, dropout_rate=0.0, drop_path_rate=0.0):
        super(MobileMamba, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.drop_path = DropPath(drop_path_rate)
        
        # 点卷积扩展通道
        self.point_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expand,
            kernel_size=1
        )
        
        # 深度卷积捕获局部信息
        self.depth_conv = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model * expand,
            kernel_size=5,
            padding=2,
            groups=d_model * expand
        )
        
        # 简化版SSM参数
        self.A = nn.Parameter(-torch.ones(d_model * expand) * 0.5)  # 直接使用负值初始化
        self.B = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.C = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.D = nn.Parameter(torch.randn(d_model * expand))
        
        # 点卷积压缩通道
        self.point_conv2 = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model,
            kernel_size=1
        )
        
        # 激活函数与归一化
        self.act = nn.SiLU()
        self.norm = nn.BatchNorm1d(d_model * expand)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        nn.init.normal_(self.D, mean=0.0, std=0.1)
    
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # 转为卷积格式
        x_conv = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        
        # 点卷积扩展通道
        x_conv = self.point_conv1(x_conv)
        x_conv = self.act(x_conv)
        
        # 深度卷积捕获局部信息
        x_local = self.depth_conv(x_conv)
        x_local = self.norm(x_local)
        x_local = self.act(x_local)
        
        # 转回序列格式
        x_local = x_local.transpose(1, 2)  # [batch_size, seq_len, d_model*expand]
        
        # 获取扩展后的通道数
        expanded_dim = self.d_model * 2
        
        # 向量化SSM处理
        # 初始化隐状态
        h = torch.zeros(batch_size, expanded_dim, self.d_state, device=x.device)
        
        # A矩阵 - 广播为每个样本的形状 [batch_size, expanded_dim, 1]
        A_exp = torch.exp(self.A).unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 预计算B*x_t
        x_seq = x_local.unsqueeze(-1)  # [batch_size, seq_len, expanded_dim, 1]
        B_exp = self.B.unsqueeze(0).unsqueeze(0)  # [1, 1, expanded_dim, d_state]
        Bx = x_seq * B_exp  # [batch_size, seq_len, expanded_dim, d_state]
        
        # 输出张量
        output = torch.zeros_like(x_local)
        
        # 高效状态更新
        for t in range(seq_len):
            # 状态更新: h_t = A * h_{t-1} + B * x_t
            h = A_exp * h + Bx[:, t]
            
            # 输出计算: y_t = C * h_t + D * x_t
            y = torch.sum(self.C.unsqueeze(0) * h, dim=-1) + self.D * x_local[:, t]
            output[:, t] = y
        
        # 转回卷积格式
        output = output.transpose(1, 2)  # [batch_size, expanded_dim, seq_len]
        
        # 点卷积压缩通道
        output = self.point_conv2(output)
        
        # 转回序列格式
        output = output.transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        # DropPath
        output = self.drop_path(output)
        
        return output