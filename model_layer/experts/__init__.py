# model_layer/experts/__init__.py
"""
专家模块包
提供所有专家模型的导入和工厂函数
"""

from .base_expert import ExpertModel, DummyExpert, create_expert_from_config
from .transformer_expert import (
    TransformerExpert,
    create_transformer_expert_small,
    create_transformer_expert_medium,
    create_transformer_expert_large
)
from .rnn_expert import (
    RNNExpert,
    create_rnn_expert_lstm,
    create_rnn_expert_gru,
    create_rnn_expert_channel_specific,
    create_rnn_expert_deep
)
from .cnn_expert import (
    CNNExpert,
    create_cnn_expert_simple,
    create_cnn_expert_multiscale,
    create_cnn_expert_deep,
    create_cnn_expert_lightweight
)

# 尝试导入混合专家，如果失败则跳过
try:
    from .hybrid_expert import (
        HybridExpert,
        create_hybrid_expert_small,
        create_hybrid_expert_medium,
        create_hybrid_expert_large,
        create_hybrid_expert_attention_fusion
    )
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HybridExpert not available: {e}")
    # 创建占位符类和函数
    HybridExpert = None
    create_hybrid_expert_small = None
    create_hybrid_expert_medium = None
    create_hybrid_expert_large = None
    create_hybrid_expert_attention_fusion = None
    HYBRID_AVAILABLE = False

# 专家类型映射
EXPERT_TYPES = {
    'transformer': TransformerExpert,
    'rnn': RNNExpert,
    'cnn': CNNExpert,
    'dummy': DummyExpert
}

# 只有在混合专家可用时才添加
if HYBRID_AVAILABLE:
    EXPERT_TYPES['hybrid'] = HybridExpert

# 预定义专家配置
EXPERT_PRESETS = {
    # Transformer预设
    'transformer_small': {
        'type': 'transformer',
        'factory': create_transformer_expert_small,
        'description': 'Small Transformer expert for lightweight tasks'
    },
    'transformer_medium': {
        'type': 'transformer',
        'factory': create_transformer_expert_medium,
        'description': 'Medium Transformer expert for balanced performance'
    },
    'transformer_large': {
        'type': 'transformer',
        'factory': create_transformer_expert_large,
        'description': 'Large Transformer expert for complex tasks'
    },
    
    # RNN预设
    'rnn_lstm': {
        'type': 'rnn',
        'factory': create_rnn_expert_lstm,
        'description': 'LSTM-based RNN expert'
    },
    'rnn_gru': {
        'type': 'rnn',
        'factory': create_rnn_expert_gru,
        'description': 'GRU-based RNN expert'
    },
    'rnn_channel_specific': {
        'type': 'rnn',
        'factory': create_rnn_expert_channel_specific,
        'description': 'Channel-specific RNN expert for multi-sensor data'
    },
    'rnn_deep': {
        'type': 'rnn',
        'factory': create_rnn_expert_deep,
        'description': 'Deep RNN expert with multiple layers'
    },
    
    # CNN预设
    'cnn_simple': {
        'type': 'cnn',
        'factory': create_cnn_expert_simple,
        'description': 'Simple CNN expert for basic feature extraction'
    },
    'cnn_multiscale': {
        'type': 'cnn',
        'factory': create_cnn_expert_multiscale,
        'description': 'Multi-scale CNN expert for diverse temporal patterns'
    },
    'cnn_deep': {
        'type': 'cnn',
        'factory': create_cnn_expert_deep,
        'description': 'Deep CNN expert with residual connections'
    },
    'cnn_lightweight': {
        'type': 'cnn',
        'factory': create_cnn_expert_lightweight,
        'description': 'Lightweight CNN expert for resource-constrained environments'
    }
}

# 只有在混合专家可用时才添加混合预设
if HYBRID_AVAILABLE:
    EXPERT_PRESETS.update({
        'hybrid_small': {
            'type': 'hybrid',
            'factory': create_hybrid_expert_small,
            'description': 'Small hybrid expert combining RNN and Transformer'
        },
        'hybrid_medium': {
            'type': 'hybrid',
            'factory': create_hybrid_expert_medium,
            'description': 'Medium hybrid expert with balanced architecture'
        },
        'hybrid_large': {
            'type': 'hybrid',
            'factory': create_hybrid_expert_large,
            'description': 'Large hybrid expert for complex temporal modeling'
        },
        'hybrid_attention_fusion': {
            'type': 'hybrid',
            'factory': create_hybrid_expert_attention_fusion,
            'description': 'Hybrid expert with attention-based fusion'
        }
    })


def create_expert_from_preset(preset_name: str, input_shape, output_dim) -> ExpertModel:
    """
    从预设创建专家模型
    
    Args:
        preset_name: 预设名称
        input_shape: 输入形状
        output_dim: 输出维度
        
    Returns:
        专家模型实例
    """
    if preset_name not in EXPERT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(EXPERT_PRESETS.keys())}")
    
    preset = EXPERT_PRESETS[preset_name]
    factory_func = preset['factory']
    
    return factory_func(input_shape, output_dim)


def list_expert_types():
    """
    列出所有可用的专家类型
    
    Returns:
        专家类型列表
    """
    return list(EXPERT_TYPES.keys())


def list_expert_presets():
    """
    列出所有可用的专家预设
    
    Returns:
        专家预设字典
    """
    return {name: preset['description'] for name, preset in EXPERT_PRESETS.items()}


def get_expert_class(expert_type: str):
    """
    获取专家类
    
    Args:
        expert_type: 专家类型
        
    Returns:
        专家类
    """
    if expert_type not in EXPERT_TYPES:
        raise ValueError(f"Unknown expert type: {expert_type}. Available types: {list(EXPERT_TYPES.keys())}")
    
    return EXPERT_TYPES[expert_type]


def validate_expert_config(config: dict) -> bool:
    """
    验证专家配置
    
    Args:
        config: 专家配置字典
        
    Returns:
        是否有效
    """
    required_keys = ['type', 'input_shape', 'output_dim']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    if config['type'] not in EXPERT_TYPES:
        print(f"Invalid expert type: {config['type']}")
        return False
    
    return True


# 导出所有内容
__all__ = [
    # 基础类
    'ExpertModel',
    'DummyExpert',
    
    # 专家类
    'TransformerExpert',
    'RNNExpert',
    'CNNExpert',
    
    # 工厂函数
    'create_expert_from_config',
    'create_expert_from_preset',
    
    # Transformer工厂函数
    'create_transformer_expert_small',
    'create_transformer_expert_medium',
    'create_transformer_expert_large',
    
    # RNN工厂函数
    'create_rnn_expert_lstm',
    'create_rnn_expert_gru',
    'create_rnn_expert_channel_specific',
    'create_rnn_expert_deep',
    
    # CNN工厂函数
    'create_cnn_expert_simple',
    'create_cnn_expert_multiscale',
    'create_cnn_expert_deep',
    'create_cnn_expert_lightweight',
    
    # 工具函数
    'list_expert_types',
    'list_expert_presets',
    'get_expert_class',
    'validate_expert_config',
    
    # 常量
    'EXPERT_TYPES',
    'EXPERT_PRESETS',
    'HYBRID_AVAILABLE'
]

# 只有在混合专家可用时才导出相关内容
if HYBRID_AVAILABLE:
    __all__.extend([
        'HybridExpert',
        'create_hybrid_expert_small',
        'create_hybrid_expert_medium',
        'create_hybrid_expert_large',
        'create_hybrid_expert_attention_fusion'
    ])