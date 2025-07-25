import torch
import torch.nn as nn
from typing import Dict, Any

from .experts import EXPERT_TYPES
from fusion_layer import create_fusion_strategy

class DynamicHarModel(nn.Module):
    """
    动态HAR模型容器
    支持配置驱动的专家模型实例化和动态前向传播
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.architecture_config = config.get('architecture', {})
        self.experts_config = self.architecture_config.get('experts', {})
        self.fusion_config = self.architecture_config.get('fusion', {})
        self.classifier_config = self.architecture_config.get('classifier', {})
        self.num_classes = self.architecture_config.get('num_classes', 8)
        
        self.experts = nn.ModuleDict()
        self._create_experts()
        
        self.fusion_layer = create_fusion_strategy(self.fusion_config)
        
        self.classifier = self._create_classifier()
        self.apply(self._init_weights)
        
    def _create_experts(self):
        """根据配置动态创建专家模型"""
        for expert_name, expert_config in self.experts_config.items():
            expert_type_name = expert_config.get('type')
            expert_params = expert_config.get('params', {})
            
            if expert_type_name in EXPERT_TYPES:
                expert_class = EXPERT_TYPES[expert_type_name]
                
                # --- FIX START ---
                # 复制参数字典以安全地修改它
                params_for_expert = expert_params.copy()
                
                # 弹出并传递 'input_shape' 和 'output_dim'
                input_shape = params_for_expert.pop('input_shape', None)
                output_dim = params_for_expert.pop('output_dim', None)
                
                if input_shape is None or output_dim is None:
                    raise ValueError(f"Expert '{expert_name}' config is missing 'input_shape' or 'output_dim'.")

                # 将剩余的参数作为kwargs传递
                expert_instance = expert_class(
                    input_shape=input_shape,
                    output_dim=output_dim,
                    **params_for_expert  # 现在这个字典不包含重复的键
                )
                # --- FIX END ---

                self.experts[expert_name] = expert_instance
                print(f"Created expert: {expert_name} ({expert_type_name})")
            else:
                raise ValueError(f"Unsupported expert type: {expert_type_name}")

    def _create_classifier(self):
        """创建分类器"""
        fusion_output_dim = self._calculate_fusion_output_dim()
        
        classifier_layers_config = self.classifier_config.get('layers', [fusion_output_dim, self.num_classes])
        if not classifier_layers_config or classifier_layers_config[0] != fusion_output_dim:
             classifier_layers_config.insert(0, fusion_output_dim)

        layers = []
        for i in range(len(classifier_layers_config) - 1):
            layers.append(nn.Linear(classifier_layers_config[i], classifier_layers_config[i+1]))
            if i < len(classifier_layers_config) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.classifier_config.get('dropout', 0.2)))
        
        return nn.Sequential(*layers)

    def _calculate_fusion_output_dim(self) -> int:
        """计算融合后的输出维度"""
        strategy = self.fusion_config.get('strategy', 'concatenate')
        expert_dims = {name: expert.output_dim for name, expert in self.experts.items()}
        
        if strategy == 'concatenate':
            return sum(expert_dims.values())
        else:
            return next(iter(expert_dims.values()))

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """动态前向传播"""
        expert_outputs = {}
        for expert_name, expert_model in self.experts.items():
            modality = self.experts_config[expert_name].get('modality')
            if modality and modality in data_dict:
                expert_outputs[expert_name] = expert_model(data_dict[modality])
        
        if not expert_outputs:
            raise ValueError("No expert outputs available for fusion. Check modality names in config and data.")
            
        fused_features = self.fusion_layer(expert_outputs)
        logits = self.classifier(fused_features)
        return logits

def create_dynamic_har_model(config: Dict[str, Any]) -> DynamicHarModel:
    """创建动态HAR模型的工厂函数"""
    return DynamicHarModel(config)