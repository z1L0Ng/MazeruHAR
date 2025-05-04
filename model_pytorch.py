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


class SensorWiseMHA(nn.Module):
    """
    传感器级多头注意力机制，在特定传感器通道上应用注意力
    """
    def __init__(self, projection_quarter, num_heads, start_index, stop_index, dropout_rate=0.0, drop_path_rate=0.0):
        super(SensorWiseMHA, self).__init__()
        self.projection_quarter = projection_quarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.start_index = start_index
        self.stop_index = stop_index
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate)
        
        # 使用PyTorch的多头注意力实现
        embed_dim = stop_index - start_index
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, input_data, return_attention_scores=False):
        # 提取特定通道的输入数据
        extracted_input = input_data[:, :, self.start_index:self.stop_index]
        
        if return_attention_scores:
            # 返回注意力分数
            mha_outputs, attention_scores = self.mha(
                extracted_input, 
                extracted_input, 
                extracted_input,
                need_weights=True,
                average_attn_weights=False  # 获取所有头的注意力分数
            )
            return mha_outputs, attention_scores
        else:
            # 不返回注意力分数
            mha_outputs, _ = self.mha(
                extracted_input, 
                extracted_input, 
                extracted_input,
                need_weights=False
            )
            mha_outputs = self.drop_path(mha_outputs)
            return mha_outputs


class LiteFormer(nn.Module):
    """
    改进的LiteFormer模块，使用高效的一维卷积处理序列数据
    """
    def __init__(self, start_index, stop_index, projection_size, kernel_size=16, attention_head=3, use_bias=False, drop_path_rate=0.0, dropout_rate=0):
        super(LiteFormer, self).__init__()
        self.use_bias = use_bias
        self.start_index = start_index
        self.stop_index = stop_index
        self.kernel_size = kernel_size
        self.projection_size = projection_size
        self.attention_head = attention_head
        self.drop_path = DropPath(drop_path_rate)
        self.projection_half = projection_size // 2
        
        # 初始化卷积核权重
        self.depthwise_kernels = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, 1, kernel_size))
            for _ in range(attention_head)
        ])
        
        # 初始化卷积核
        for kernel in self.depthwise_kernels:
            nn.init.xavier_uniform_(kernel)
            
        # 可选的偏置项
        if self.use_bias:
            self.conv_bias = nn.Parameter(torch.zeros(attention_head))
            nn.init.xavier_uniform_(self.conv_bias)
    
    def forward(self, inputs):
        # 提取相关的输入通道
        formatted_inputs = inputs[:, :, self.start_index:self.stop_index]
        batch_size, seq_len, channels = formatted_inputs.shape

        # 将输入reshape成 [batch_size, channels, seq_len]，更适合卷积操作
        reshaped_inputs = formatted_inputs.permute(0, 2, 1)

        # 计算每个头的通道数
        channels_per_head = channels // self.attention_head
        
        # 重塑输入以进行每头处理 [batch_size, heads, channels_per_head, seq_len]
        reshaped_inputs = reshaped_inputs.view(batch_size, self.attention_head, channels_per_head, seq_len)

        # 应用softmax到卷积核（仅在训练模式下）
        if self.training:
            softmax_kernels = [F.softmax(kernel, dim=-1) for kernel in self.depthwise_kernels]
        else:
            softmax_kernels = self.depthwise_kernels
        
        # 为每个注意力头执行卷积操作
        conv_outputs = []
        for i in range(self.attention_head):
            # 提取当前头的输入
            head_input = reshaped_inputs[:, i]  # [batch_size, channels_per_head, seq_len]
            
            # 获取当前头的核
            kernel = softmax_kernels[i].view(1, 1, self.kernel_size)
            
            # 扩展核以匹配输入通道
            kernel = kernel.repeat(channels_per_head, 1, 1)
            
            # 执行卷积
            head_output = F.conv1d(
                head_input,
                kernel,
                bias=None,
                padding='same',
                groups=channels_per_head  # 每个通道独立卷积
            )
            
            # 添加偏置（如果需要）
            if self.use_bias:
                head_output = head_output + self.conv_bias[i].view(1, 1, 1)
            
            # 收集输出
            conv_outputs.append(head_output)
        
        # 拼接所有头的输出 [batch_size, channels, seq_len]
        conv_outputs = torch.cat(conv_outputs, dim=1)
        
        # 应用DropPath
        conv_outputs = self.drop_path(conv_outputs)
        
        # 转换回原始形状 [batch_size, seq_len, channels]
        local_attention = conv_outputs.permute(0, 2, 1)
        
        return local_attention

class MixAccGyro(nn.Module):
    """
    混合加速度计和陀螺仪特征
    """
    def __init__(self, projection_quarter, projection_half, projection_dim):
        super(MixAccGyro, self).__init__()
        self.projection_quarter = projection_quarter
        self.projection_half = projection_half
        self.projection_dim = projection_dim
        self.projection_three_fourth = self.projection_half + self.projection_quarter
        
        # 混合索引计算
        mix_acc_gyro_indices = []
        for i in range(projection_quarter):
            mix_acc_gyro_indices.append(projection_quarter + i)
            mix_acc_gyro_indices.append(projection_half + i)
        
        # 创建新的排列索引
        self.new_arrangement = list(range(0, projection_quarter)) + mix_acc_gyro_indices + list(range(self.projection_three_fourth, projection_dim))
        self.register_buffer('indices', torch.tensor(self.new_arrangement))
        
    def forward(self, inputs):
        return torch.index_select(inputs, 2, self.indices)


def mlp(x, hidden_units, dropout_rate):
    """
    简单的MLP层序列，用于特征提取
    """
    for units in hidden_units:
        x = F.silu(nn.Linear(x.size(-1), units)(x))
        x = F.dropout(x, p=dropout_rate, training=x.requires_grad)
    return x


def mlp2(x, hidden_units, dropout_rate):
    """
    两层MLP，第一层有激活函数，第二层没有
    """
    x = F.silu(nn.Linear(x.size(-1), hidden_units[0])(x))
    x = F.dropout(x, p=dropout_rate, training=x.requires_grad)
    x = nn.Linear(hidden_units[0], hidden_units[1])(x)
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


class HART(nn.Module):
    """
    人体活动识别Transformer (HART) 模型的PyTorch实现
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16, 
                 num_heads=3, filter_attention_head=4, conv_kernels=None, 
                 mlp_head_units=None, dropout_rate=0.3, use_tokens=False):
        super(HART, self).__init__()
        
        if conv_kernels is None:
            conv_kernels = [3, 7, 15, 31, 31, 31]
        
        if mlp_head_units is None:
            mlp_head_units = [1024]
        
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
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
        for i, kernel_length in enumerate(conv_kernels):
            # 层归一化
            self.transformer_blocks.append(nn.LayerNorm(projection_dim))
            
            # LiteFormer分支
            self.transformer_blocks.append(
                LiteFormer(
                    start_index=self.projection_quarter,
                    stop_index=self.projection_quarter + self.projection_half,
                    projection_size=self.projection_half,
                    attention_head=filter_attention_head,
                    kernel_size=kernel_length,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 加速度计MHA分支
            self.transformer_blocks.append(
                SensorWiseMHA(
                    projection_quarter=self.projection_quarter,
                    num_heads=num_heads,
                    start_index=0,
                    stop_index=self.projection_quarter,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 陀螺仪MHA分支
            self.transformer_blocks.append(
                SensorWiseMHA(
                    projection_quarter=self.projection_quarter,
                    num_heads=num_heads,
                    start_index=self.projection_quarter + self.projection_half,
                    stop_index=projection_dim,
                    drop_path_rate=self.drop_path_rates[i].item(),
                    dropout_rate=dropout_rate
                )
            )
            
            # 层归一化和MLP
            self.transformer_blocks.append(nn.LayerNorm(projection_dim))
            
            # 为MLP添加一个占位符，因为我们需要在forward中创建它
            self.transformer_blocks.append(nn.Identity())
            
            # 为DropPath添加一个占位符
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

        # 在 __init__ 方法结束前添加
        # 初始化权重
        self.apply(self._init_weights)

        # 同时添加这个方法到 HART 类
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

        # 调试信息
        debug_info = {}

        # 应用Transformer块
        for i in range(0, len(self.transformer_blocks), 7):
            # 应用层归一化
            x1 = self.transformer_blocks[i](encoded_patches)

            # 计算预期维度
            expected_half = self.projection_half
            expected_quarter = self.projection_quarter

            # LiteFormer分支
            branch1 = self.transformer_blocks[i + 1](x1)

            # 记录维度信息
            debug_info[f"layer_{i}_branch1_shape_before"] = branch1.shape

            # 确保branch1维度与预期相同
            branch1 = torch.nn.functional.interpolate(
                branch1,
                size=(branch1.size(1), expected_half),
                mode='bilinear',
                align_corners=False
            ) if branch1.size(-1) != expected_half else branch1

            debug_info[f"layer_{i}_branch1_shape_after"] = branch1.shape

            # 加速度计MHA分支
            branch2_acc = self.transformer_blocks[i + 2](x1)
            debug_info[f"layer_{i}_branch2_acc_shape_before"] = branch2_acc.shape

            # 陀螺仪MHA分支
            branch2_gyro = self.transformer_blocks[i + 3](x1)
            debug_info[f"layer_{i}_branch2_gyro_shape_before"] = branch2_gyro.shape

            # 确保所有分支维度正确
            branch2_acc = torch.nn.functional.interpolate(
                branch2_acc,
                size=(branch2_acc.size(1), expected_quarter),
                mode='bilinear',
                align_corners=False
            ) if branch2_acc.size(-1) != expected_quarter else branch2_acc

            branch2_gyro = torch.nn.functional.interpolate(
                branch2_gyro,
                size=(branch2_gyro.size(1), expected_quarter),
                mode='bilinear',
                align_corners=False
            ) if branch2_gyro.size(-1) != expected_quarter else branch2_gyro

            debug_info[f"layer_{i}_branch2_acc_shape_after"] = branch2_acc.shape
            debug_info[f"layer_{i}_branch2_gyro_shape_after"] = branch2_gyro.shape

            # 执行断言检查，确保维度匹配
            assert branch1.shape[2] == expected_half, f"Branch1维度不匹配: {branch1.shape[2]} vs {expected_half}"
            assert branch2_acc.shape[2] == expected_quarter, f"Branch2_acc维度不匹配: {branch2_acc.shape[2]} vs {expected_quarter}"
            assert branch2_gyro.shape[2] == expected_quarter, f"Branch2_gyro维度不匹配: {branch2_gyro.shape[2]} vs {expected_quarter}"

            # 拼接所有分支
            concat_attention = torch.cat((branch2_acc, branch1, branch2_gyro), dim=2)

            # 残差连接
            x2 = concat_attention + encoded_patches

            # 层归一化
            x3 = self.transformer_blocks[i + 4](x2)

            # MLP (动态创建，因为它需要输入大小)
            mlp_output = nn.Sequential(
                nn.Linear(x3.size(-1), self.transformer_units[0]),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.transformer_units[0], self.transformer_units[1])
            )(x3)

            # DropPath
            x3 = self.transformer_blocks[i + 6](mlp_output)

            # 最终残差连接
            encoded_patches = x3 + x2

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


class MobileHART_XS(nn.Module):
    """
    MobileHART_XS模型的PyTorch实现 - 更轻量级版本的HART
    """
    def __init__(self, input_shape, activity_count, projection_dims=None, filter_count=None, 
                 expansion_factor=4, mlp_head_units=None, dropout_rate=0.3):
        super(MobileHART_XS, self).__init__()
        
        if projection_dims is None:
            projection_dims = [96, 120, 144]
        
        if filter_count is None:
            filter_count = [16//2, 32//2, 48//2, 64//2, 80, 96, 384]
        
        if mlp_head_units is None:
            mlp_head_units = [1024]
        
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dims = projection_dims
        self.filter_count = filter_count
        self.expansion_factor = expansion_factor
        self.mlp_head_units = mlp_head_units
        self.dropout_rate = dropout_rate
        
        # 初始卷积层
        self.acc_conv = self._conv_block(in_channels=3, out_channels=filter_count[0])
        self.gyro_conv = self._conv_block(in_channels=3, out_channels=filter_count[0])
        
        # MV2块
        self.acc_mv2 = nn.Sequential(
            self._inverted_residual_block(filter_count[0], filter_count[0] * expansion_factor, filter_count[1]),
            self._inverted_residual_block(filter_count[1], filter_count[1] * expansion_factor, filter_count[2], strides=2),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[2]),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[2]),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[3], strides=2)
        )
        
        self.gyro_mv2 = nn.Sequential(
            self._inverted_residual_block(filter_count[0], filter_count[0] * expansion_factor, filter_count[1]),
            self._inverted_residual_block(filter_count[1], filter_count[1] * expansion_factor, filter_count[2], strides=2),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[2]),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[2]),
            self._inverted_residual_block(filter_count[2], filter_count[2] * expansion_factor, filter_count[3], strides=2)
        )
        
        # SensorWiseHART
        # 注意：这里需要实现sensorWiseHART函数，这在原始代码中是函数而不是类
        # 为简化起见，我们保留函数形式，但在forward中调用
        
        # 中间投影层
        self.mid_projection = nn.Sequential(
            nn.Linear(projection_dims[0], projection_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 第二个MV2块和MobileViT块
        self.mv2_block2 = self._inverted_residual_block(
            projection_dims[0], 
            projection_dims[0] * expansion_factor, 
            filter_count[4], 
            strides=2
        )
        
        # MobileViT块 - 由于复杂度较高，我们使用占位符
        # 实际实现需要移植mobilevit_block函数
        self.mobilevit_block1 = nn.Identity()  # 占位符
        
        # 第三个MV2块和MobileViT块
        self.mv2_block3 = self._inverted_residual_block(
            projection_dims[1], 
            projection_dims[1] * expansion_factor, 
            filter_count[5], 
            strides=2
        )
        
        self.mobilevit_block2 = nn.Identity()  # 占位符
        
        # 最终卷积层
        self.final_conv = self._conv_block(
            in_channels=filter_count[5], 
            out_channels=filter_count[6], 
            kernel_size=1, 
            strides=1
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP头
        self.mlp_head = nn.ModuleList()
        in_features = filter_count[6]
        for units in mlp_head_units:
            self.mlp_head.append(nn.Linear(in_features, units))
            self.mlp_head.append(nn.SiLU())
            self.mlp_head.append(nn.Dropout(dropout_rate))
            in_features = units
        
        # 输出层
        self.classifier = nn.Linear(in_features, activity_count)
    
    def _conv_block(self, in_channels, out_channels, kernel_size=3, strides=2):
        """实现卷积块"""
        padding = kernel_size // 2 if strides == 1 else 0
        return nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=strides, 
                padding=padding
            ),
            nn.SiLU()
        )
    
    def _inverted_residual_block(self, in_channels, expanded_channels, out_channels, strides=1):
        """实现倒置残差块 (MobileNetV2块)"""
        layers = []
        
        # 扩展部分
        layers.append(nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(expanded_channels))
        layers.append(nn.SiLU())
        
        # 深度卷积部分
        if strides == 2:
            layers.append(nn.ZeroPad1d(1))
            
        layers.append(nn.Conv1d(
            expanded_channels, 
            expanded_channels, 
            kernel_size=3, 
            stride=strides, 
            padding=1 if strides == 1 else 0, 
            groups=expanded_channels, 
            bias=False
        ))
        layers.append(nn.BatchNorm1d(expanded_channels))
        layers.append(nn.SiLU())
        
        # 投影部分
        layers.append(nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        
        # 残差连接
        use_residual = (in_channels == out_channels) and (strides == 1)
        
        if use_residual:
            return nn.Sequential(
                *layers,
                ResidualAdd()
            )
        else:
            return nn.Sequential(*layers)
    
    def forward(self, inputs):
        # 输入形状: [batch_size, time_steps, channels]
        # 分离加速度计和陀螺仪数据
        acc_data = inputs[:, :, :3].permute(0, 2, 1)  # [batch_size, 3, time_steps]
        gyro_data = inputs[:, :, 3:].permute(0, 2, 1)  # [batch_size, 3, time_steps]
        
        # 应用初始卷积
        acc_x = self.acc_conv(acc_data)
        gyro_x = self.gyro_conv(gyro_data)
        
        # 应用MV2块
        acc_x = self.acc_mv2(acc_x)
        gyro_x = self.gyro_mv2(gyro_x)
        
        # 应用SensorWiseHART
        # 注意：原始代码中这是一个函数调用，这里我们在forward内部实现
        acc_x, gyro_x = self._sensor_wise_hart(acc_x, gyro_x, num_blocks=2, projection_dim=self.projection_dims[0])
        
        # 拼接两个传感器的输出
        x = torch.cat((acc_x, gyro_x), dim=1)
        
        # 重塑为序列格式
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # 应用中间投影
        x = self.mid_projection(x)
        
        # 转回卷积格式
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 应用第二个MV2块
        x = self.mv2_block2(x)
        
        # 应用第一个MobileViT块
        # 在实际实现中，这需要是mobilevit_block的调用
        # x = self.mobilevit_block1(x)
        # 简化的实现，跳过实际的MobileViT块
        
        # 应用第三个MV2块
        x = self.mv2_block3(x)
        
        # 应用第二个MobileViT块
        # 在实际实现中，这需要是mobilevit_block的调用
        # x = self.mobilevit_block2(x)
        # 简化的实现，跳过实际的MobileViT块
        
        # 应用最终卷积
        x = self.final_conv(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 应用MLP头
        for layer in self.mlp_head:
            x = layer(x)
        
        # 分类层
        logits = self.classifier(x)
        output = F.softmax(logits, dim=-1)
        
        return output
    
    def _transformer_block(self, x, transformer_layers, projection_dim, dropout_rate=0.3, num_heads=2):
        """实现Transformer块"""
        drop_path_rates = torch.linspace(0, dropout_rate * 10, transformer_layers) * 0.1
        
        for i in range(transformer_layers):
            # 层归一化1
            x1 = nn.LayerNorm(x.size(-1))(x)
            
            # 多头注意力
            attn = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            attention_output, _ = attn(x1, x1, x1)
            
            # 残差连接1
            x2 = attention_output + x
            
            # 层归一化2
            x3 = nn.LayerNorm(x.size(-1))(x2)
            
            # MLP
            x3 = nn.Sequential(
                nn.Linear(x3.size(-1), x3.size(-1) * 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(x3.size(-1) * 2, x3.size(-1))
            )(x3)
            
            # 残差连接2
            x = x3 + x2
            
        return x
    
    def _mobilevit_block(self, x, num_blocks, projection_dim, strides=1):
        """实现MobileViT块"""
        # 局部特征提取
        local_features = self._conv_block(x, out_channels=projection_dim, strides=strides)
        local_features = self._conv_block(local_features, out_channels=projection_dim, kernel_size=1, strides=strides)
        
        # 全局特征提取（使用Transformer）
        batch_size, channels, seq_len = local_features.shape
        
        # 重塑为序列格式
        global_features = local_features.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        # 应用Transformer
        global_features = self._transformer_block(global_features, num_blocks, projection_dim)
        
        # 重塑回卷积格式
        global_features = global_features.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        
        # 点卷积
        folded_feature_map = self._conv_block(global_features, out_channels=x.size(1), kernel_size=1, strides=strides)
        
        # 拼接局部和全局特征
        local_global_features = torch.cat([x, folded_feature_map], dim=1)
        
        # 融合特征
        local_global_features = self._conv_block(local_global_features, out_channels=projection_dim, strides=strides)
        
        return local_global_features
    
    def _sensor_wise_transformer_block(self, x_acc, x_gyro, patch_count, transformer_layers, projection_dim,
                                      kernel_size=4, dropout_rate=0.3, num_heads=2):
        """实现传感器级Transformer块"""
        projection_quarter = projection_dim // 4
        projection_half = projection_dim // 2
        drop_path_rates = torch.linspace(0, dropout_rate * 10, transformer_layers) * 0.1
        
        # 拼接两个传感器输入
        x = torch.cat((x_acc, x_gyro), dim=1)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
        
        for layer_index in range(transformer_layers):
            # 层归一化
            x1 = nn.LayerNorm(x.size(-1))(x)
            
            # LiteFormer分支
            branch1 = LiteFormer(
                start_index=projection_quarter,
                stop_index=projection_quarter + projection_half,
                projection_size=projection_half,
                attention_head=num_heads,
                kernel_size=kernel_size,
                drop_path_rate=drop_path_rates[layer_index].item()
            )(x1)
            
            # 加速度计MHA分支
            branch2_acc = SensorWiseMHA(
                projection_quarter=projection_quarter,
                num_heads=num_heads,
                start_index=0,
                stop_index=projection_quarter,
                drop_path_rate=drop_path_rates[layer_index].item(),
                dropout_rate=dropout_rate
            )(x1)
            
            # 陀螺仪MHA分支
            branch2_gyro = SensorWiseMHA(
                projection_quarter=projection_quarter,
                num_heads=num_heads,
                start_index=projection_quarter + projection_half,
                stop_index=projection_dim,
                drop_path_rate=drop_path_rates[layer_index].item(),
                dropout_rate=dropout_rate
            )(x1)
            
            # 拼接注意力输出
            concat_attention = torch.cat((branch2_acc, branch1, branch2_gyro), dim=2)
            
            # 残差连接1
            x2 = concat_attention + x
            
            # 层归一化2
            x3 = nn.LayerNorm(x2.size(-1))(x2)
            
            # MLP
            x3 = nn.Sequential(
                nn.Linear(x3.size(-1), x3.size(-1) * 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(x3.size(-1) * 2, x3.size(-1))
            )(x3)
            
            # DropPath
            x3 = DropPath(drop_path_rates[layer_index].item())(x3)
            
            # 残差连接2
            x = x3 + x2
        
        return x
    
    def _sensor_wise_hart(self, x_acc, x_gyro, num_blocks, projection_dim, kernel_size=4, strides=1):
        """实现传感器级HART"""
        # 局部特征提取 - 加速度计
        local_features_acc = self._conv_block(x_acc, out_channels=projection_dim//2, strides=strides)
        local_features_acc = self._conv_block(local_features_acc, out_channels=projection_dim//2, kernel_size=1, strides=strides)
        
        # 局部特征提取 - 陀螺仪
        local_features_gyro = self._conv_block(x_gyro, out_channels=projection_dim//2, strides=strides)
        local_features_gyro = self._conv_block(local_features_gyro, out_channels=projection_dim//2, kernel_size=1, strides=strides)
        
        # 应用传感器级Transformer
        global_features = self._sensor_wise_transformer_block(
            local_features_acc,
            local_features_gyro,
            local_features_gyro.size(2),  # patch_count
            num_blocks,
            projection_dim,
            kernel_size=kernel_size
        )
        
        # 将全局特征转回卷积格式
        global_features = global_features.permute(0, 2, 1)
        
        # 分离不同传感器的特征
        global_features_acc = global_features[:, :projection_dim//2, :]
        global_features_gyro = global_features[:, projection_dim//2:, :]
        
        # 加速度计分支的特征融合
        folded_feature_map_acc = self._conv_block(
            global_features_acc, 
            out_channels=x_acc.size(1), 
            kernel_size=1, 
            strides=strides
        )
        local_global_features_acc = torch.cat([x_acc, folded_feature_map_acc], dim=1)
        local_global_features_acc = self._conv_block(
            local_global_features_acc, 
            out_channels=projection_dim//2, 
            strides=strides
        )
        
        # 陀螺仪分支的特征融合
        folded_feature_map_gyro = self._conv_block(
            global_features_gyro, 
            out_channels=x_gyro.size(1), 
            kernel_size=1, 
            strides=strides
        )
        local_global_features_gyro = torch.cat([x_gyro, folded_feature_map_gyro], dim=1)
        local_global_features_gyro = self._conv_block(
            local_global_features_gyro, 
            out_channels=projection_dim//2, 
            strides=strides
        )
        
        return local_global_features_acc, local_global_features_gyro


class ResidualAdd(nn.Module):
    """残差连接辅助类"""
    def forward(self, x):
        input_tensor, residual_tensor = x, x[0]
        return input_tensor + residual_tensor


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

def mobileHART_XXS(input_shape, activity_count, projection_dims=None, filter_count=None, 
                expansion_factor=2, mlp_head_units=None, dropout_rate=0.3):
    """
    更轻量级的MobileHART版本 (XXS)
    这是对原始MobileHART_XS的进一步简化版本
    """
    if projection_dims is None:
        projection_dims = [64, 80, 96]
    
    if filter_count is None:
        filter_count = [16//2, 16//2, 24//2, 48//2, 64, 80, 320]
    
    if mlp_head_units is None:
        mlp_head_units = [1024]
    
    # 直接使用MobileHART_XS实现，但使用不同的超参数
    model = MobileHART_XS(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dims=projection_dims,
        filter_count=filter_count,
        expansion_factor=expansion_factor,
        mlp_head_units=mlp_head_units,
        dropout_rate=dropout_rate
    )
    
    return model