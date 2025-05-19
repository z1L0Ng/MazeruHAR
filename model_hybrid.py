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


class RNNEncoder(nn.Module):
    """
    使用RNN对序列数据进行编码
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, rnn_type='gru', bidirectional=True, dropout_rate=0.1):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # 选择RNN类型
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        else:  # 默认使用GRU
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        
        # 输出维度
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
    def forward(self, x):
        # 通过RNN
        if self.rnn_type == 'lstm':
            outputs, (_, _) = self.rnn(x)
        else:
            outputs, _ = self.rnn(x)
        
        return outputs


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 使用PyTorch的多头注意力实现
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.self_attn(x, x, x, need_weights=True)
            return attn_output, attn_weights
        else:
            attn_output, _ = self.self_attn(x, x, x, need_weights=False)
            return attn_output


class FeedForward(nn.Module):
    """
    前馈网络，通常用在Transformer块中
    """
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)  # 使用SiLU激活函数，也称为Swish
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层，包含自注意力和前馈网络
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout_rate=0.1, drop_path_rate=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout_rate)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path = DropPath(drop_path_rate)
        
    def forward(self, src):
        # 自注意力部分
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.drop_path(self.dropout(src2))
        
        # 前馈网络部分
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.drop_path(self.dropout(src2))
        
        return src


class RNNAttentionHART(nn.Module):
    """
    结合RNN编码器和多头注意力机制的人体活动识别模型
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16, 
                 rnn_hidden_dim=128, rnn_num_layers=2, rnn_type='gru', rnn_bidirectional=True,
                 num_heads=4, num_transformer_layers=2, transformer_dim_feedforward=768,
                 dropout_rate=0.3, use_tokens=False, mlp_head_units=None, 
                 # 兼容HART模型的参数
                 filter_attention_head=None, conv_kernels=None):
        super(RNNAttentionHART, self).__init__()
        
        if mlp_head_units is None:
            mlp_head_units = [384]
        
        # 保存主要参数
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self.rnn_bidirectional = rnn_bidirectional
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 设置Transformer层数量
        if conv_kernels is not None:
            self.num_transformer_layers = len(conv_kernels)
        else:
            self.num_transformer_layers = num_transformer_layers
        
        # 计算RNN输出维度
        self.rnn_output_dim = rnn_hidden_dim * 2 if rnn_bidirectional else rnn_hidden_dim
        
        # 定义模型组件
        # 1. 补丁提取
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 2. 如果使用分类token，则添加
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 3. 添加位置编码
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        if self.use_tokens:
            n_patches += 1  # 额外添加分类token
            
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # 4. RNN编码器
        self.rnn_encoder = RNNEncoder(
            input_dim=projection_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            bidirectional=rnn_bidirectional,
            dropout_rate=dropout_rate
        )
        
        # 5. 通过投影层将RNN输出维度调整为Transformer输入维度
        if self.rnn_output_dim != projection_dim:
            self.projection = nn.Linear(self.rnn_output_dim, projection_dim)
        else:
            self.projection = nn.Identity()
        
        # 6. Transformer编码器层
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_transformer_layers):
            # 添加各种层，确保命名与原始代码一致
            self.transformer_layers.append(nn.LayerNorm(projection_dim))
            self.transformer_layers.append(TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=num_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout_rate=dropout_rate,
                drop_path_rate=dropout_rate * (i + 1) / self.num_transformer_layers
            ))
            
            # 使用register_module确保命名一致性，以便可视化
            self.register_module(f"normalizedInputs_{i}", self.transformer_layers[-2])
            self.register_module(f"AccMHA_{i}", self.transformer_layers[-1])
            self.register_module(f"GyroMHA_{i}", nn.Identity())  # 占位符保持命名一致
        
        # 7. 最终层归一化
        self.final_layer_norm = nn.LayerNorm(projection_dim)
        
        # 8. 分类头 - 确保存储中间表示
        self.intermediate_features = None
        self.mlp_head = nn.Sequential()
        in_features = projection_dim
        for i, units in enumerate(mlp_head_units):
            self.mlp_head.add_module(f"dense_{i}", nn.Linear(in_features, units))
            self.mlp_head.add_module(f"activation_{i}", nn.SiLU())
            self.mlp_head.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            in_features = units
        
        # 9. 输出层
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
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def extract_features(self, x):
        """
        明确定义的特征提取方法
        """
        # 创建补丁
        x = self.patches(x)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        x = self.patch_encoder(x)
        
        # 应用RNN编码器
        x = self.rnn_encoder(x)
        
        # 投影到所需维度
        x = self.projection(x)
        
        # 应用Transformer层
        for i in range(0, len(self.transformer_layers), 2):
            # 层归一化
            norm_x = self.transformer_layers[i](x)
            # Transformer层
            x = self.transformer_layers[i+1](norm_x) + x
        
        # 最终层归一化
        x = self.final_layer_norm(x)
        
        # 提取表示
        if self.use_tokens:
            # 使用分类token作为表示
            representation = x[:, 0]
        else:
            # 使用全局平均池化作为表示
            representation = x.mean(dim=1)
        
        # 存储提取的特征
        self.intermediate_features = representation
        
        return representation
    
    def forward(self, inputs):
        # 提取特征
        representation = self.extract_features(inputs)
        
        # 通过MLP头
        for layer in self.mlp_head:
            representation = layer(representation)
        
        # 分类层
        logits = self.classifier(representation)
        output = F.softmax(logits, dim=-1)
        
        return output


class MixedAttention(nn.Module):
    """
    线性注意力机制，结合卷积和自注意力
    """
    def __init__(self, dim, kernel_size=15, num_heads=4, dropout=0.0):
        super(MixedAttention, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "维度必须能被头数整除"
        
        # 1x1卷积层作为投影
        self.project = nn.Conv1d(dim, dim, kernel_size=1)
        
        # 深度卷积层用于局部注意力
        self.depthwise_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,  # 深度卷积
            bias=False
        )
        
        # 最终1x1卷积混合特征
        self.output_proj = nn.Conv1d(dim, dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.shape
        
        # 转换为卷积格式 [batch_size, dim, seq_len]
        x_conv = x.permute(0, 2, 1)
        
        # 初始投影
        x_proj = self.project(x_conv)
        
        # 深度卷积
        x_conv = self.depthwise_conv(x_proj)
        
        # 输出投影
        x_out = self.output_proj(x_conv)
        x_out = self.dropout(x_out)
        
        # 转回序列格式 [batch_size, seq_len, dim]
        return x_out.permute(0, 2, 1)


class RNNLinearAttentionHART(nn.Module):
    """
    一个更高效的版本，使用线性注意力替代标准注意力
    """
    def __init__(self, input_shape, activity_count, projection_dim=192, patch_size=16, time_step=16, 
                 num_heads=4, filter_attention_head=4, conv_kernels=None, dropout_rate=0.3, use_tokens=False):
        super(RNNLinearAttentionHART, self).__init__()
        
        if conv_kernels is None:
            conv_kernels = [3, 7, 15, 31, 31, 31]
        
        # 保存主要参数
        self.input_shape = input_shape
        self.activity_count = activity_count
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.time_step = time_step
        self.num_heads = num_heads
        self.filter_attention_head = filter_attention_head
        self.conv_kernels = conv_kernels
        self.dropout_rate = dropout_rate
        self.use_tokens = use_tokens
        
        # 计算投影维度的一半和四分之一
        self.projection_half = projection_dim // 2
        self.projection_quarter = projection_dim // 4
        
        # 定义模型组件
        # 1. 补丁提取
        self.patches = SensorPatches(projection_dim, patch_size, time_step)
        
        # 2. 如果使用分类token，则添加
        if self.use_tokens:
            self.class_token = ClassToken(projection_dim)
        
        # 3. 添加位置编码
        time_steps = input_shape[0]
        n_patches = (time_steps - patch_size) // time_step + 1
        
        if self.use_tokens:
            n_patches += 1  # 额外添加分类token
            
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        
        # 4. RNN编码器
        rnn_hidden_dim = projection_dim // 2
        self.rnn_encoder = RNNEncoder(
            input_dim=projection_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=2,
            rnn_type='gru',
            bidirectional=True,
            dropout_rate=dropout_rate
        )
        
        # 5. 线性注意力层
        self.attention_layers = nn.ModuleList()
        for i, kernel_size in enumerate(conv_kernels):
            # 创建混合注意力层
            self.attention_layers.append(nn.LayerNorm(projection_dim))
            self.attention_layers.append(MixedAttention(
                dim=projection_dim,
                kernel_size=kernel_size,
                num_heads=num_heads,
                dropout=dropout_rate
            ))
            
            # 注册模块以保持与原始代码的兼容性
            self.register_module(f"normalizedInputs_{i}", self.attention_layers[-2])
            self.register_module(f"AccMHA_{i}", self.attention_layers[-1])
            self.register_module(f"GyroMHA_{i}", nn.Identity())  # 占位符保持命名一致
        
        # 存储中间特征
        self.intermediate_features = None
        
        # 6. 最终层归一化
        self.final_layer_norm = nn.LayerNorm(projection_dim)
        
        # 7. 分类头
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
    
    def extract_features(self, inputs):
        """
        明确定义的特征提取方法
        """
        # 创建补丁
        x = self.patches(inputs)
        
        # 添加分类token（如果需要）
        if self.use_tokens:
            x = self.class_token(x)
        
        # 添加位置编码
        x = self.patch_encoder(x)
        
        # 应用RNN编码器
        x = self.rnn_encoder(x)
        
        # 应用线性注意力层
        for i in range(0, len(self.attention_layers), 2):
            # 层归一化
            norm_x = self.attention_layers[i](x)
            # 注意力层
            attn_x = self.attention_layers[i+1](norm_x)
            # 残差连接
            x = attn_x + x
        
        # 最终层归一化
        x = self.final_layer_norm(x)
        
        # 提取表示
        if self.use_tokens:
            # 使用分类token作为表示
            representation = x[:, 0]
        else:
            # 使用全局平均池化作为表示
            representation = x.mean(dim=1)
        
        # 存储中间特征
        self.intermediate_features = representation
        
        return representation
    
    def forward(self, inputs):
        # 提取特征
        representation = self.extract_features(inputs)
        
        # 通过分类头
        logits = self.mlp_head(representation)
        
        # 输出softmax概率
        return F.softmax(logits, dim=-1)


# 修复t-SNE可视化的钩子函数
def get_model_features(model, dataloader, device):
    """
    提取模型的特征表示用于t-SNE可视化
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # 使用提取特征的方法
            if hasattr(model, 'extract_features'):
                features = model.extract_features(inputs).cpu().numpy()
                all_features.append(features)
                all_labels.extend(labels.numpy())
            # 备用方法：如果没有extract_features方法，则尝试从中间特征获取
            elif hasattr(model, 'intermediate_features') and model.intermediate_features is not None:
                # 先运行前向传播
                _ = model(inputs)
                # 然后获取存储的中间特征
                features = model.intermediate_features.cpu().numpy()
                all_features.append(features)
                all_labels.extend(labels.numpy())
            else:
                print("警告: 模型没有提供特征提取方法或中间特征，t-SNE可能无法正常工作")
    
    # 确保有特征提取
    if not all_features:
        raise ValueError("无法提取特征。请确保模型有extract_features方法或存储中间特征。")
    
    # 合并所有特征
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels


# 辅助函数，创建不同大小的模型
def rnn_attention_hart_small(input_shape, activity_count, **kwargs):
    """
    创建小型RNN-Attention HART模型
    """
    return RNNAttentionHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=128,
        patch_size=16,
        time_step=16,
        rnn_hidden_dim=64,
        rnn_num_layers=2,
        rnn_type='gru',
        rnn_bidirectional=True,
        num_heads=4,
        num_transformer_layers=2,
        transformer_dim_feedforward=512,
        dropout_rate=0.2,
        **kwargs
    )


def rnn_attention_hart_tiny(input_shape, activity_count, **kwargs):
    """
    创建超小型RNN-Attention HART模型
    """
    return RNNAttentionHART(
        input_shape=input_shape,
        activity_count=activity_count,
        projection_dim=64,
        patch_size=16,
        time_step=16,
        rnn_hidden_dim=32,
        rnn_num_layers=1,
        rnn_type='gru',
        rnn_bidirectional=True,
        num_heads=2,
        num_transformer_layers=1,
        transformer_dim_feedforward=256,
        dropout_rate=0.1,
        **kwargs
    )