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


class RNNBlock(nn.Module):
    """
    使用RNN处理传感器通道的块
    """
    def __init__(self, input_dim, hidden_dim, rnn_type='gru', bidirectional=True, dropout_rate=0.1):
        super(RNNBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
        else:  # 默认使用GRU
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if dropout_rate > 0 else 0
            )
            
        # 输出投影层
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(output_dim, input_dim)
        
        # 归一化层
        self.norm = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 应用层归一化
        residual = x
        x = self.norm(x)
        
        # 通过RNN
        if self.rnn_type == 'lstm':
            outputs, (_, _) = self.rnn(x)
        else:
            outputs, _ = self.rnn(x)
        
        # 投影回原始维度
        outputs = self.projection(outputs)
        
        # Dropout
        outputs = self.dropout(outputs)
        
        # 残差连接
        return outputs + residual


class SensorChannelRNN(nn.Module):
    """
    对特定传感器通道应用RNN处理
    """
    def __init__(self, start_index, stop_index, hidden_dim=64, rnn_type='gru', dropout_rate=0.1):
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
            dropout_rate=dropout_rate
        )
        
    def forward(self, inputs):
        # 提取相关通道
        channel_data = inputs[:, :, self.start_index:self.stop_index]
        
        # 应用RNN
        outputs = self.rnn_block(channel_data)
        
        return outputs


class RNNHART(nn.Module):
    """
    基于RNN的人体活动识别模型 (RNN-HART)
    使用RNN替代原始HART模型中的Transformer组件
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16, 
                 num_layers=3, hidden_dim=64, rnn_type='gru', dropout_rate=0.3, use_tokens=False,
                 num_heads=None, filter_attention_head=None, conv_kernels=None, drop_path_rate=None,
                 d_state=None, expand=None, **kwargs):  # 添加额外参数以保持兼容性
        super(RNNHART, self).__init__()
        
        # 记录主要参数
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 记录兼容性参数（不使用）
        self.num_heads = num_heads
        self.filter_attention_head = filter_attention_head
        self.conv_kernels = [3, 7, 15] if conv_kernels is None else conv_kernels
        
        # 计算投影维度的一半和四分之一
        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        
        # 定义模型组件
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 如果使用分类token，则添加
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 添加位置编码
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        if self.use_tokens:
            n_patches += 1  # 额外添加分类token
            
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # 创建RNN块的序列
        self.rnn_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            # 加速度计RNN
            self.rnn_blocks.append(
                SensorChannelRNN(
                    start_index=0,
                    stop_index=self.projection_quarter,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout_rate
                )
            )
            
            # 中心通道RNN
            self.rnn_blocks.append(
                SensorChannelRNN(
                    start_index=self.projection_quarter,
                    stop_index=self.projection_quarter + self.projection_half,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout_rate
                )
            )
            
            # 陀螺仪RNN
            self.rnn_blocks.append(
                SensorChannelRNN(
                    start_index=self.projection_quarter + self.projection_half,
                    stop_index=projection_dim,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout_rate
                )
            )
            
            # 全通道整合RNN
            self.rnn_blocks.append(
                RNNBlock(
                    input_dim=projection_dim,
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout_rate
                )
            )
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(projection_dim)
        
        # 分类头
        self.mlp_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim * 2, activity_count)
        )
        
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
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, inputs):
        # 创建补丁
        x = self.patches(inputs)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        x = self.patch_encoder(x)
        
        # 应用RNN块
        for i in range(0, len(self.rnn_blocks), 4):
            # 三个独立的传感器通道处理
            acc_output = self.rnn_blocks[i](x)
            mid_output = self.rnn_blocks[i+1](x)
            gyro_output = self.rnn_blocks[i+2](x)
            
            # 将处理后的通道拼接回原来的位置
            x_parts = []
            x_parts.append(acc_output)
            x_parts.append(mid_output)
            x_parts.append(gyro_output)
            
            # 用处理后的通道更新x
            x_updated = torch.cat(x_parts, dim=2)
            
            # 全通道整合
            x = self.rnn_blocks[i+3](x_updated)
        
        # 最终层归一化
        x = self.final_layer_norm(x)
        
        # 提取表示
        if self.use_tokens:
            # 使用分类token作为表示
            representation = x[:, 0]
        else:
            # 使用全局平均池化作为表示
            representation = x.mean(dim=1)
        
        # 通过分类头
        logits = self.mlp_head(representation)
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


class FastRNNHART(nn.Module):
    """
    轻量级RNN-HART变体，用于快速训练和评估
    """
    def __init__(self, input_shape, activity_count, projection_dim=128, patch_size=16, time_step=16, 
                 hidden_dim=64, rnn_type='gru', dropout_rate=0.2, num_heads=None, 
                 filter_attention_head=None, conv_kernels=None, drop_path_rate=None,
                 d_state=None, expand=None, **kwargs):  # 添加额外参数以保持兼容性
        super(FastRNNHART, self).__init__()
        
        self.input_shape = input_shape
        self.activity_count = activity_count
        
        # 简化的补丁提取 - 使用较小的投影维度
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 计算序列长度
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        # 简化的位置编码
        self.position_encoder = nn.Embedding(n_patches, projection_dim)
        
        # 单层双向GRU/LSTM
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=projection_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
        else:
            self.rnn = nn.GRU(
                input_size=projection_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
            
        # 输出投影
        self.projection = nn.Linear(hidden_dim * 2, projection_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(projection_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, activity_count),
            nn.Dropout(dropout_rate)
        )
        
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
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, inputs):
        # 创建补丁
        x = self.patches(inputs)
        
        # 添加位置编码
        positions = torch.arange(0, x.size(1), device=inputs.device)
        x = x + self.position_encoder(positions).unsqueeze(0)
        
        # 通过RNN
        if isinstance(self.rnn, nn.LSTM):
            x, _ = self.rnn(x)
        else:
            x, _ = self.rnn(x)
        
        # 投影
        x = self.projection(x)
        
        # 归一化
        x = self.norm(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(x)
        output = F.softmax(logits, dim=-1)
        
        return output


def rnn_hart_small(input_shape, activity_count, **kwargs):
    """
    创建小型RNNHART模型
    """
    return RNNHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=128,
        patch_size=16,
        time_step=16,
        num_layers=2,
        hidden_dim=64,
        rnn_type='gru',
        dropout_rate=0.2,
        **kwargs
    )


def rnn_hart_tiny(input_shape, activity_count, **kwargs):
    """
    创建超小型RNNHART模型
    """
    return RNNHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=64,
        patch_size=16,
        time_step=16,
        num_layers=1,
        hidden_dim=32,
        rnn_type='gru',
        dropout_rate=0.1,
        **kwargs
    )


def fast_rnn_hart(input_shape, activity_count, **kwargs):
    """
    创建FastRNNHART模型
    """
    return FastRNNHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=96,
        patch_size=16,
        time_step=16,
        hidden_dim=48,
        rnn_type='gru',
        dropout_rate=0.1,
        **kwargs
    )


# 演示使用方法
if __name__ == "__main__":
    # 输入形状: [time_steps, channels]
    input_shape = (128, 6)  # 128个时间步，6个通道(3个加速度计，3个陀螺仪)
    activity_count = 6  # 活动类别数量
    
    # 创建随机输入数据
    batch_size = 8
    x = torch.randn(batch_size, input_shape[0], input_shape[1])
    
    # 测试FastRNNHART模型 - 最高效的版本
    print("\n创建FastRNNHART模型...")
    # 测试兼容性 - 添加num_heads参数
    fast_model = fast_rnn_hart(input_shape, activity_count, num_heads=3, filter_attention_head=4)
    
    # 前向传播
    print("执行前向传播...")
    try:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = fast_model(x)
        end_time.record()
        
        # 同步CUDA操作
        torch.cuda.synchronize()
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数: {sum(p.numel() for p in fast_model.parameters())}")
        print(f"前向传播时间: {start_time.elapsed_time(end_time):.2f} ms")
    except:
        import time
        start = time.time()
        output = fast_model(x)
        end = time.time()
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数: {sum(p.numel() for p in fast_model.parameters())}")
        print(f"前向传播时间: {(end-start)*1000:.2f} ms")
    
    # 测试RNN-HART模型
    print("\n创建RNN-HART模型...")
    # 测试兼容性 - 添加num_heads参数
    rnn_model = RNNHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=128,
        patch_size=16,
        time_step=16,
        num_layers=2,
        hidden_dim=64,
        rnn_type='gru',
        dropout_rate=0.2,
        num_heads=3,
        filter_attention_head=4
    )
    
    # 前向传播
    print("执行前向传播...")
    try:
        start_time.record()
        output = rnn_model(x)
        end_time.record()
        
        # 同步CUDA操作
        torch.cuda.synchronize()
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数: {sum(p.numel() for p in rnn_model.parameters())}")
        print(f"前向传播时间: {start_time.elapsed_time(end_time):.2f} ms")
    except:
        import time
        start = time.time()
        output = rnn_model(x)
        end = time.time()
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数: {sum(p.numel() for p in rnn_model.parameters())}")
        print(f"前向传播时间: {(end-start)*1000:.2f} ms")