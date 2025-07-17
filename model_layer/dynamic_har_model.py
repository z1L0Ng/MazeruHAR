# model_layer/dynamic_har_model.py
"""
动态HAR模型 - 任务1.3核心模块
基于配置动态实例化专家模块和融合策略的主模型
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from .experts import create_expert_from_config, ExpertModel, validate_expert_config


class SimpleConcatFusion(nn.Module):
    """
    简单拼接融合策略
    """
    def __init__(self, expert_output_dims: List[int], output_dim: int, dropout_rate: float = 0.1):
        super(SimpleConcatFusion, self).__init__()
        
        self.expert_output_dims = expert_output_dims
        self.input_dim = sum(expert_output_dims)
        self.output_dim = output_dim
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            expert_outputs: 专家输出字典 {expert_name: tensor}
            
        Returns:
            融合后的特征张量
        """
        # 按顺序拼接所有专家输出
        outputs = []
        for expert_name, output in expert_outputs.items():
            outputs.append(output)
        
        # 拼接
        fused_features = torch.cat(outputs, dim=1)
        
        # 通过融合层
        result = self.fusion_layer(fused_features)
        
        return result


class WeightedSumFusion(nn.Module):
    """
    加权求和融合策略
    """
    def __init__(self, expert_output_dims: List[int], output_dim: int, dropout_rate: float = 0.1):
        super(WeightedSumFusion, self).__init__()
        
        self.expert_output_dims = expert_output_dims
        self.output_dim = output_dim
        self.num_experts = len(expert_output_dims)
        
        # 确保所有专家输出维度相同
        if not all(dim == output_dim for dim in expert_output_dims):
            # 如果维度不同，需要投影层
            self.projection_layers = nn.ModuleDict()
            for i, dim in enumerate(expert_output_dims):
                if dim != output_dim:
                    self.projection_layers[f'expert_{i}'] = nn.Linear(dim, output_dim)
        else:
            self.projection_layers = None
        
        # 可学习的权重
        self.weights = nn.Parameter(torch.ones(self.num_experts))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, expert_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        """
        outputs = []
        expert_names = list(expert_outputs.keys())
        
        for i, (expert_name, output) in enumerate(expert_outputs.items()):
            # 投影到相同维度（如果需要）
            if self.projection_layers and f'expert_{i}' in self.projection_layers:
                output = self.projection_layers[f'expert_{i}'](output)
            
            outputs.append(output)
        
        # 堆叠所有输出
        stacked_outputs = torch.stack(outputs, dim=1)  # (batch_size, num_experts, output_dim)
        
        # 应用softmax权重
        weights = torch.softmax(self.weights, dim=0)
        
        # 加权求和
        weighted_output = torch.sum(stacked_outputs * weights.view(1, -1, 1), dim=1)
        
        # Dropout
        result = self.dropout(weighted_output)
        
        return result


class DynamicHarModel(nn.Module):
    """
    动态HAR模型
    基于配置动态实例化专家模块和融合策略
    """
    
    def __init__(self, config: Any):
        """
        初始化动态HAR模型
        
        Args:
            config: 配置对象，包含architecture、experts、fusion等配置
        """
        super(DynamicHarModel, self).__init__()
        
        self.config = config
        
        # 验证配置
        self._validate_config()
        
        # 动态创建专家模块
        self.experts = nn.ModuleDict()
        self._create_experts()
        
        # 创建融合层
        self.fusion_layer = self._create_fusion_layer()
        
        # 创建分类器
        self.classifier = self._create_classifier()
        
        # 存储中间特征
        self.intermediate_features = {}
        
        # 应用权重初始化
        self.apply(self._init_weights)
    
    def _validate_config(self):
        """
        验证配置有效性
        """
        required_sections = ['architecture']
        for section in required_sections:
            if not hasattr(self.config, section):
                raise ValueError(f"Missing configuration section: {section}")
        
        if not hasattr(self.config.architecture, 'experts'):
            raise ValueError("Missing experts configuration in architecture")
        
        if not hasattr(self.config.architecture, 'fusion'):
            raise ValueError("Missing fusion configuration in architecture")
    
    def _create_experts(self):
        """
        动态创建专家模块
        """
        experts_config = self.config.architecture.experts
        
        for expert_name, expert_config in experts_config.items():
            # 验证专家配置
            if not validate_expert_config(expert_config):
                raise ValueError(f"Invalid configuration for expert: {expert_name}")
            
            # 创建专家实例
            try:
                expert = create_expert_from_config(expert_config)
                self.experts[expert_name] = expert
                print(f"✓ Created expert: {expert_name} ({expert_config['type']})")
            except Exception as e:
                print(f"✗ Failed to create expert {expert_name}: {e}")
                raise
    
    def _create_fusion_layer(self):
        """
        创建融合层
        """
        fusion_config = self.config.architecture.fusion
        strategy = fusion_config.get('strategy', 'concat')
        
        # 获取专家输出维度
        expert_output_dims = []
        for expert_name, expert in self.experts.items():
            expert_output_dims.append(expert.get_output_dim())
        
        # 获取融合输出维度
        fusion_output_dim = fusion_config.get('output_dim', 256)
        dropout_rate = fusion_config.get('dropout_rate', 0.1)
        
        # 创建融合层
        if strategy == 'concat':
            return SimpleConcatFusion(expert_output_dims, fusion_output_dim, dropout_rate)
        elif strategy == 'weighted_sum':
            return WeightedSumFusion(expert_output_dims, fusion_output_dim, dropout_rate)
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
    
    def _create_classifier(self):
        """
        创建分类器
        """
        fusion_output_dim = self.config.architecture.fusion.get('output_dim', 256)
        num_classes = self.config.dataset.get('num_classes', 8)
        
        classifier_config = self.config.architecture.get('classifier', {})
        hidden_dims = classifier_config.get('hidden_dims', [])
        dropout_rate = classifier_config.get('dropout_rate', 0.1)
        
        # 构建分类器
        layers = []
        
        # 隐藏层
        current_dim = fusion_output_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data_dict: 数据字典，键为模态名称，值为数据张量
            
        Returns:
            分类预测结果
        """
        # 字典推导，并行提取所有模态的特征
        expert_outputs = {}
        
        for expert_name, expert in self.experts.items():
            # 获取对应的数据
            # 这里假设数据字典的键与专家名称对应
            # 实际实现中可能需要更复杂的映射逻辑
            if expert_name in data_dict:
                data = data_dict[expert_name]
            else:
                # 如果没有直接对应，使用第一个可用的数据
                # 这是一个简化的实现，实际中需要更好的映射策略
                data = next(iter(data_dict.values()))
            
            # 专家特征提取
            expert_output = expert(data)
            expert_outputs[expert_name] = expert_output
            
            # 存储中间特征
            if hasattr(expert, 'intermediate_features'):
                self.intermediate_features[expert_name] = expert.intermediate_features
        
        # 融合特征
        fused_features = self.fusion_layer(expert_outputs)
        
        # 存储融合后的特征
        self.intermediate_features['fused'] = fused_features.detach()
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def extract_features(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        提取特征
        
        Args:
            data_dict: 数据字典
            
        Returns:
            特征字典
        """
        # 执行前向传播
        _ = self.forward(data_dict)
        
        # 返回中间特征
        return self.intermediate_features.copy()
    
    def get_expert_names(self) -> List[str]:
        """
        获取专家名称列表
        
        Returns:
            专家名称列表
        """
        return list(self.experts.keys())
    
    def get_expert_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取专家配置信息
        
        Returns:
            专家配置字典
        """
        configs = {}
        for expert_name, expert in self.experts.items():
            configs[expert_name] = expert.get_config()
        
        return configs
    
    def freeze_expert(self, expert_name: str):
        """
        冻结特定专家的参数
        
        Args:
            expert_name: 专家名称
        """
        if expert_name in self.experts:
            self.experts[expert_name].freeze_parameters()
            print(f"Frozen expert: {expert_name}")
        else:
            print(f"Expert not found: {expert_name}")
    
    def unfreeze_expert(self, expert_name: str):
        """
        解冻特定专家的参数
        
        Args:
            expert_name: 专家名称
        """
        if expert_name in self.experts:
            self.experts[expert_name].unfreeze_parameters()
            print(f"Unfrozen expert: {expert_name}")
        else:
            print(f"Expert not found: {expert_name}")
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取参数数量统计
        
        Returns:
            参数数量字典
        """
        counts = {}
        
        # 专家参数
        for expert_name, expert in self.experts.items():
            counts[f"expert_{expert_name}"] = expert.get_parameter_count()
        
        # 融合层参数
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters() if p.requires_grad)
        counts["fusion_layer"] = fusion_params
        
        # 分类器参数
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        counts["classifier"] = classifier_params
        
        # 总参数
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts
    
    def _init_weights(self, module):
        """
        权重初始化
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def __repr__(self):
        return f"DynamicHarModel(experts={list(self.experts.keys())}, fusion={self.config.architecture.fusion.get('strategy', 'concat')})"


def create_dynamic_har_model(config: Any) -> DynamicHarModel:
    """
    创建动态HAR模型的工厂函数
    
    Args:
        config: 配置对象
        
    Returns:
        DynamicHarModel实例
    """
    return DynamicHarModel(config)


# 便捷函数：创建基于字典配置的模型
def create_model_from_dict(config_dict: Dict[str, Any]) -> DynamicHarModel:
    """
    从字典配置创建模型
    
    Args:
        config_dict: 配置字典
        
    Returns:
        DynamicHarModel实例
    """
    from types import SimpleNamespace
    
    # 递归转换字典为对象
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    config = dict_to_namespace(config_dict)
    return DynamicHarModel(config)


# 示例配置函数
def get_example_config() -> Dict[str, Any]:
    """
    获取示例配置
    
    Returns:
        示例配置字典
    """
    return {
        "dataset": {
            "num_classes": 8,
            "activity_labels": ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
        },
        "architecture": {
            "experts": {
                "transformer_expert": {
                    "type": "transformer",
                    "input_shape": [500, 6],
                    "output_dim": 128,
                    "params": {
                        "projection_dim": 128,
                        "patch_size": 16,
                        "time_step": 16,
                        "num_heads": 4,
                        "num_layers": 2,
                        "d_ff": 512,
                        "dropout_rate": 0.1,
                        "use_cls_token": True
                    }
                },
                "rnn_expert": {
                    "type": "rnn",
                    "input_shape": [500, 6],
                    "output_dim": 128,
                    "params": {
                        "hidden_dim": 64,
                        "num_layers": 2,
                        "rnn_type": "gru",
                        "dropout_rate": 0.1,
                        "bidirectional": True,
                        "pooling_method": "last"
                    }
                },
                "cnn_expert": {
                    "type": "cnn",
                    "input_shape": [500, 6],
                    "output_dim": 128,
                    "params": {
                        "num_layers": 3,
                        "base_channels": 64,
                        "kernel_sizes": [3, 5, 7],
                        "dropout_rate": 0.1,
                        "use_residual": True,
                        "use_multiscale": True
                    }
                }
            },
            "fusion": {
                "strategy": "concat",
                "output_dim": 256,
                "dropout_rate": 0.1
            },
            "classifier": {
                "hidden_dims": [128],
                "dropout_rate": 0.1
            }
        }
    }


def create_example_model() -> DynamicHarModel:
    """
    创建示例模型
    
    Returns:
        DynamicHarModel实例
    """
    config_dict = get_example_config()
    return create_model_from_dict(config_dict)


# 模型测试函数
def test_dynamic_model():
    """
    测试动态模型
    """
    print("Testing DynamicHarModel...")
    
    # 创建示例模型
    model = create_example_model()
    
    # 打印模型信息
    print(f"Model: {model}")
    print(f"Experts: {model.get_expert_names()}")
    print(f"Parameter counts: {model.get_parameter_count()}")
    
    # 测试前向传播
    batch_size = 4
    time_steps = 500
    features = 6
    
    # 创建测试数据
    test_data = {
        "transformer_expert": torch.randn(batch_size, time_steps, features),
        "rnn_expert": torch.randn(batch_size, time_steps, features),
        "cnn_expert": torch.randn(batch_size, time_steps, features)
    }
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_data)
        print(f"Output shape: {outputs.shape}")
        
        # 测试特征提取
        features = model.extract_features(test_data)
        print(f"Extracted features keys: {list(features.keys())}")
        for key, feature in features.items():
            if feature is not None:
                print(f"  {key}: {feature.shape}")
    
    print("✓ DynamicHarModel test passed!")


if __name__ == "__main__":
    test_dynamic_model()