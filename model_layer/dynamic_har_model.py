"""
DynamicHarModel - 任务1.3
动态模型容器实现，支持基于nn.ModuleDict的动态专家实例化和动态forward流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import importlib


class ExpertModel(nn.Module, ABC):
    """
    专家模型的抽象基类
    所有专家模型都应该继承这个类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 6)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('output_dim', 128)
        self.dropout = config.get('dropout', 0.1)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播抽象方法
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        pass
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim


class TransformerExpert(ExpertModel):
    """
    基于Transformer的专家模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.ff_dim = config.get('ff_dim', self.hidden_dim * 4)
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, self.hidden_dim) * 0.02
        )
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 输出投影
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 添加位置编码
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        # 应用Transformer
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # 全局平均池化
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 层归一化和dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # 输出投影
        x = self.output_projection(x)  # [batch_size, output_dim]
        
        return x


class RNNExpert(ExpertModel):
    """
    基于RNN的专家模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.num_layers = config.get('num_layers', 2)
        self.bidirectional = config.get('bidirectional', True)
        
        # RNN层
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # 输出投影
        rnn_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.output_projection = nn.Linear(rnn_output_dim, self.output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        # 通过RNN
        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(x)
        else:  # GRU
            output, hidden = self.rnn(x)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 拼接前向和后向的最后隐藏状态
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # 应用dropout
        hidden = self.dropout_layer(hidden)
        
        # 输出投影
        output = self.output_projection(hidden)
        
        return output


class CNNExpert(ExpertModel):
    """
    基于CNN的专家模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        self.num_filters = config.get('num_filters', 64)
        self.pool_size = config.get('pool_size', 2)
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.input_dim,
                    out_channels=self.num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(self.num_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pool_size),
                nn.Dropout(self.dropout)
            )
            self.conv_layers.append(conv_layer)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出投影
        self.output_projection = nn.Linear(
            len(self.kernel_sizes) * self.num_filters, 
            self.output_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, output_dim]
        """
        # 转换为卷积格式
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        
        # 多尺度卷积
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)  # [batch_size, num_filters, seq_len']
            conv_out = self.global_pool(conv_out)  # [batch_size, num_filters, 1]
            conv_out = conv_out.squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(conv_out)
        
        # 拼接所有尺度的特征
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]
        
        # 输出投影
        x = self.output_projection(x)
        
        return x


class FusionLayer(nn.Module):
    """
    融合层基类
    """
    
    def __init__(self, strategy: str, config: Dict[str, Any]):
        super().__init__()
        self.strategy = strategy
        self.config = config
        
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合多个专家的输出
        
        Args:
            expert_outputs: 专家输出字典 {expert_name: tensor}
            
        Returns:
            融合后的特征张量
        """
        if self.strategy == 'concatenate':
            return self._concatenate_fusion(expert_outputs)
        elif self.strategy == 'average':
            return self._average_fusion(expert_outputs)
        elif self.strategy == 'attention':
            return self._attention_fusion(expert_outputs)
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.strategy}")
    
    def _concatenate_fusion(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接融合"""
        outputs = list(expert_outputs.values())
        return torch.cat(outputs, dim=1)
    
    def _average_fusion(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """平均融合"""
        outputs = list(expert_outputs.values())
        stacked = torch.stack(outputs, dim=1)  # [batch_size, num_experts, feature_dim]
        return torch.mean(stacked, dim=1)
    
    def _attention_fusion(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """注意力融合"""
        outputs = list(expert_outputs.values())
        stacked = torch.stack(outputs, dim=1)  # [batch_size, num_experts, feature_dim]
        
        # 简单的注意力机制
        attention_weights = torch.mean(stacked, dim=2, keepdim=True)  # [batch_size, num_experts, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        fused = torch.sum(stacked * attention_weights, dim=1)
        return fused


class DynamicHarModel(nn.Module):
    """
    动态HAR模型容器
    支持配置驱动的专家模型实例化和动态前向传播
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 获取架构配置
        self.architecture_config = config.get('architecture', {})
        self.experts_config = self.architecture_config.get('experts', {})
        self.fusion_config = self.architecture_config.get('fusion', {})
        self.classifier_config = self.architecture_config.get('classifier', {})
        
        # 获取数据集信息
        self.num_classes = config.get('labels', {}).get('num_classes', 8)
        
        # 专家模型注册表
        self.expert_registry = {
            'TransformerExpert': TransformerExpert,
            'RNNExpert': RNNExpert,
            'CNNExpert': CNNExpert,
        }
        
        # 动态创建专家模型
        self.experts = nn.ModuleDict()
        self._create_experts()
        
        # 创建融合层
        self.fusion_layer = self._create_fusion_layer()
        
        # 创建分类器
        self.classifier = self._create_classifier()
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _create_experts(self):
        """
        根据配置动态创建专家模型
        """
        for expert_name, expert_config in self.experts_config.items():
            expert_type = expert_config.get('type', 'TransformerExpert')
            expert_params = expert_config.get('params', {})
            
            # 获取专家类
            if expert_type in self.expert_registry:
                expert_class = self.expert_registry[expert_type]
            else:
                # 尝试动态导入
                try:
                    module_name, class_name = expert_type.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    expert_class = getattr(module, class_name)
                except (ImportError, AttributeError):
                    raise ValueError(f"Unknown expert type: {expert_type}")
            
            # 创建专家实例
            expert_instance = expert_class(expert_params)
            self.experts[expert_name] = expert_instance
            
            print(f"Created expert: {expert_name} ({expert_type})")
    
    def _create_fusion_layer(self) -> FusionLayer:
        """
        创建融合层
        """
        strategy = self.fusion_config.get('strategy', 'concatenate')
        return FusionLayer(strategy, self.fusion_config)
    
    def _create_classifier(self) -> nn.Module:
        """
        创建分类器
        """
        classifier_type = self.classifier_config.get('type', 'MLP')
        
        if classifier_type == 'MLP':
            return self._create_mlp_classifier()
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def _create_mlp_classifier(self) -> nn.Module:
        """
        创建MLP分类器
        """
        layers = self.classifier_config.get('layers', [256, 128, self.num_classes])
        activation = self.classifier_config.get('activation', 'relu')
        dropout = self.classifier_config.get('dropout', 0.2)
        
        # 计算输入维度
        input_dim = self._calculate_fusion_output_dim()
        
        # 构建MLP层
        mlp_layers = []
        prev_dim = input_dim
        
        for i, layer_dim in enumerate(layers[:-1]):
            mlp_layers.append(nn.Linear(prev_dim, layer_dim))
            
            if activation == 'relu':
                mlp_layers.append(nn.ReLU())
            elif activation == 'gelu':
                mlp_layers.append(nn.GELU())
            elif activation == 'silu':
                mlp_layers.append(nn.SiLU())
            
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        
        # 输出层
        mlp_layers.append(nn.Linear(prev_dim, layers[-1]))
        
        return nn.Sequential(*mlp_layers)
    
    def _calculate_fusion_output_dim(self) -> int:
        """
        计算融合层输出维度
        """
        strategy = self.fusion_config.get('strategy', 'concatenate')
        
        if strategy == 'concatenate':
            # 拼接所有专家的输出维度
            total_dim = 0
            for expert_name, expert in self.experts.items():
                total_dim += expert.get_output_dim()
            return total_dim
        elif strategy in ['average', 'attention']:
            # 假设所有专家输出维度相同
            first_expert = next(iter(self.experts.values()))
            return first_expert.get_output_dim()
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
    
    def _init_weights(self, module):
        """
        初始化模型权重
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        动态前向传播
        
        Args:
            data_dict: 数据字典 {modality: tensor}
            
        Returns:
            分类结果 [batch_size, num_classes]
        """
        # 1. 专家特征提取
        expert_outputs = {}
        
        for expert_name, expert_model in self.experts.items():
            # 获取专家对应的模态数据
            modality = self._get_expert_modality(expert_name)
            
            if modality in data_dict:
                modality_data = data_dict[modality]
                expert_output = expert_model(modality_data)
                expert_outputs[expert_name] = expert_output
            else:
                print(f"Warning: Modality {modality} not found in data_dict for expert {expert_name}")
        
        # 2. 融合特征
        if not expert_outputs:
            raise ValueError("No expert outputs available for fusion")
        
        fused_features = self.fusion_layer(expert_outputs)
        
        # 3. 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def _get_expert_modality(self, expert_name: str) -> str:
        """
        获取专家对应的模态名称
        """
        expert_config = self.experts_config.get(expert_name, {})
        return expert_config.get('modality', expert_name.split('_')[0])  # 默认使用专家名称前缀
    
    def get_expert_info(self) -> Dict[str, Any]:
        """
        获取专家信息
        """
        info = {}
        for expert_name, expert in self.experts.items():
            info[expert_name] = {
                'type': type(expert).__name__,
                'output_dim': expert.get_output_dim(),
                'modality': self._get_expert_modality(expert_name),
                'parameters': sum(p.numel() for p in expert.parameters())
            }
        return info
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_experts': len(self.experts),
            'fusion_strategy': self.fusion_config.get('strategy', 'concatenate'),
            'num_classes': self.num_classes,
            'experts': self.get_expert_info()
        }


def create_dynamic_har_model(config: Dict[str, Any]) -> DynamicHarModel:
    """
    创建动态HAR模型的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        DynamicHarModel实例
    """
    return DynamicHarModel(config)


# 示例配置
def get_example_config():
    """
    获取示例配置
    """
    return {
        'labels': {
            'num_classes': 8
        },
        'architecture': {
            'experts': {
                'imu_expert': {
                    'type': 'TransformerExpert',
                    'modality': 'imu',
                    'params': {
                        'input_dim': 6,
                        'hidden_dim': 128,
                        'output_dim': 128,
                        'num_heads': 8,
                        'num_layers': 4,
                        'dropout': 0.1
                    }
                },
                'pressure_expert': {
                    'type': 'RNNExpert',
                    'modality': 'pressure',
                    'params': {
                        'input_dim': 1,
                        'hidden_dim': 64,
                        'output_dim': 64,
                        'rnn_type': 'LSTM',
                        'num_layers': 2,
                        'bidirectional': True,
                        'dropout': 0.1
                    }
                }
            },
            'fusion': {
                'strategy': 'concatenate'
            },
            'classifier': {
                'type': 'MLP',
                'layers': [192, 128, 64, 8],
                'activation': 'relu',
                'dropout': 0.2
            }
        }
    }


if __name__ == "__main__":
    # 测试DynamicHarModel
    config = get_example_config()
    model = create_dynamic_har_model(config)
    
    print("Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 128
    test_data = {
        'imu': torch.randn(batch_size, seq_len, 6),
        'pressure': torch.randn(batch_size, seq_len, 1)
    }
    
    # 测试前向传播
    output = model(test_data)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")