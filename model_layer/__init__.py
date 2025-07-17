# model_layer/__init__.py
"""
模型层 - 任务1.3核心模块
提供动态HAR模型和专家系统的完整实现
"""

# 导入动态HAR模型
from .dynamic_har_model import (
    DynamicHarModel,
    create_dynamic_har_model,
    create_model_from_dict,
    get_example_config,
    create_example_model,
    SimpleConcatFusion,
    WeightedSumFusion
)

# 导入专家模块
from .experts import (
    # 基础类
    ExpertModel,
    DummyExpert,
    
    # 专家类
    TransformerExpert,
    RNNExpert,
    CNNExpert,
    
    # 工厂函数
    create_expert_from_config,
    create_expert_from_preset,
    
    # Transformer工厂函数
    create_transformer_expert_small,
    create_transformer_expert_medium,
    create_transformer_expert_large,
    
    # RNN工厂函数
    create_rnn_expert_lstm,
    create_rnn_expert_gru,
    create_rnn_expert_channel_specific,
    create_rnn_expert_deep,
    
    # CNN工厂函数
    create_cnn_expert_simple,
    create_cnn_expert_multiscale,
    create_cnn_expert_deep,
    create_cnn_expert_lightweight,
    
    # 工具函数
    list_expert_types,
    list_expert_presets,
    get_expert_class,
    validate_expert_config,
    
    # 常量
    EXPERT_TYPES,
    EXPERT_PRESETS,
    HYBRID_AVAILABLE
)

# 尝试导入混合专家相关内容
try:
    from .experts import (
        HybridExpert,
        create_hybrid_expert_small,
        create_hybrid_expert_medium,
        create_hybrid_expert_large,
        create_hybrid_expert_attention_fusion
    )
    _hybrid_exports = [
        'HybridExpert',
        'create_hybrid_expert_small',
        'create_hybrid_expert_medium',
        'create_hybrid_expert_large',
        'create_hybrid_expert_attention_fusion'
    ]
except ImportError:
    _hybrid_exports = []

# 版本信息
__version__ = "1.0.0"

# 导出所有主要组件
__all__ = [
    # 动态HAR模型
    'DynamicHarModel',
    'create_dynamic_har_model',
    'create_model_from_dict',
    'get_example_config',
    'create_example_model',
    
    # 融合策略
    'SimpleConcatFusion',
    'WeightedSumFusion',
    
    # 专家基础类
    'ExpertModel',
    'DummyExpert',
    
    # 专家实现类
    'TransformerExpert',
    'RNNExpert',
    'CNNExpert',
    
    # 专家工厂函数
    'create_expert_from_config',
    'create_expert_from_preset',
    
    # Transformer专家工厂
    'create_transformer_expert_small',
    'create_transformer_expert_medium',
    'create_transformer_expert_large',
    
    # RNN专家工厂
    'create_rnn_expert_lstm',
    'create_rnn_expert_gru',
    'create_rnn_expert_channel_specific',
    'create_rnn_expert_deep',
    
    # CNN专家工厂
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

# 添加混合专家相关导出（如果可用）
__all__.extend(_hybrid_exports)


def print_model_layer_info():
    """
    打印模型层信息
    """
    print("=" * 50)
    print("模型层 (Model Layer) - 任务1.3核心模块")
    print("=" * 50)
    print(f"版本: {__version__}")
    print(f"支持的专家类型: {list_expert_types()}")
    print(f"预设专家配置: {len(EXPERT_PRESETS)} 个")
    print(f"融合策略: SimpleConcatFusion, WeightedSumFusion")
    print("=" * 50)
    
    # 打印专家预设信息
    print("\n可用的专家预设:")
    presets = list_expert_presets()
    for name, description in presets.items():
        print(f"  • {name}: {description}")
    
    print("\n使用示例:")
    print("  from model_layer import create_example_model")
    print("  model = create_example_model()")
    print("  print(model)")


def create_minimal_example():
    """
    创建最小示例
    """
    print("创建最小动态HAR模型示例...")
    
    # 最小配置
    minimal_config = {
        "dataset": {
            "num_classes": 8
        },
        "architecture": {
            "experts": {
                "simple_rnn": {
                    "type": "rnn",
                    "input_shape": [100, 6],
                    "output_dim": 64,
                    "params": {
                        "hidden_dim": 32,
                        "num_layers": 1,
                        "rnn_type": "gru"
                    }
                }
            },
            "fusion": {
                "strategy": "concat",
                "output_dim": 64
            }
        }
    }
    
    # 创建模型
    model = create_model_from_dict(minimal_config)
    
    print(f"✓ 创建成功: {model}")
    print(f"✓ 专家数量: {len(model.get_expert_names())}")
    print(f"✓ 参数数量: {model.get_parameter_count()}")
    
    return model


def demo_expert_creation():
    """
    演示专家创建
    """
    print("演示专家创建...")
    
    input_shape = (100, 6)
    output_dim = 64
    
    # 创建不同类型的专家
    experts = {
        "Transformer (Small)": create_transformer_expert_small(input_shape, output_dim),
        "RNN (GRU)": create_rnn_expert_gru(input_shape, output_dim),
        "CNN (Multi-scale)": create_cnn_expert_multiscale(input_shape, output_dim)
    }
    
    # 如果混合专家可用，添加到演示中
    if HYBRID_AVAILABLE:
        try:
            experts["Hybrid (Medium)"] = create_hybrid_expert_medium(input_shape, output_dim)
        except Exception as e:
            print(f"Warning: Could not create hybrid expert: {e}")
    
    for name, expert in experts.items():
        params = expert.get_parameter_count()
        print(f"✓ {name}: {params:,} parameters")
    
    return experts


if __name__ == "__main__":
    # 打印模型层信息
    print_model_layer_info()
    
    print("\n" + "="*50)
    print("运行示例...")
    print("="*50)
    
    # 创建最小示例
    print("\n1. 最小示例:")
    minimal_model = create_minimal_example()
    
    # 演示专家创建
    print("\n2. 专家创建示例:")
    experts = demo_expert_creation()
    
    # 创建完整示例
    print("\n3. 完整示例:")
    full_model = create_example_model()
    print(f"✓ 完整模型: {full_model}")
    
    print("\n✓ 所有示例运行成功！")