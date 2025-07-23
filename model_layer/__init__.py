# model_layer/__init__.py
"""
模型层 - 完整实现包含任务1.3和任务2.2
支持动态专家模型和多种融合策略
"""

from .dynamic_har_model import (
    # 核心模型类
    DynamicHarModel,
    create_dynamic_har_model,
    
    # 专家模型基类和实现
    ExpertModel,
    TransformerExpert,
    RNNExpert,
    CNNExpert,
    
    # 融合层
    FusionLayer,
    
    # 配置和工具函数
    get_example_config
)

# 版本信息
__version__ = "1.0.0"
__author__ = "MazeruHAR Team"

# 导出的公共接口
__all__ = [
    # 主要模型类
    'DynamicHarModel',
    'create_dynamic_har_model',
    
    # 专家模型
    'ExpertModel',
    'TransformerExpert',
    'RNNExpert', 
    'CNNExpert',
    
    # 融合层
    'FusionLayer',
    
    # 工具函数
    'get_example_config'
]

# 模块级别的配置
SUPPORTED_EXPERT_TYPES = [
    'TransformerExpert',
    'RNNExpert',
    'CNNExpert'
]

SUPPORTED_FUSION_STRATEGIES = [
    'concatenate',      # 任务2.2核心 - 拼接融合
    'average',          # 平均融合
    'weighted_sum',     # 加权求和融合
    'attention'         # 注意力融合
]

# 默认配置模板
DEFAULT_CONFIG_TEMPLATE = {
    'labels': {
        'num_classes': 8
    },
    'architecture': {
        'experts': {
            'default_expert': {
                'type': 'TransformerExpert',
                'modality': 'sensor_data',
                'params': {
                    'input_dim': 6,
                    'hidden_dim': 128,
                    'output_dim': 128,
                    'num_heads': 8,
                    'num_layers': 4,
                    'dropout': 0.1
                }
            }
        },
        'fusion': {
            'strategy': 'concatenate',
            'params': {}
        },
        'classifier': {
            'type': 'MLP',
            'layers': [128, 64, 8],
            'activation': 'relu',
            'dropout': 0.2
        }
    }
}


def get_supported_expert_types():
    """获取支持的专家模型类型"""
    return SUPPORTED_EXPERT_TYPES.copy()


def get_supported_fusion_strategies():
    """获取支持的融合策略"""
    return SUPPORTED_FUSION_STRATEGIES.copy()


def validate_config(config: dict) -> bool:
    """
    验证配置文件的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    try:
        # 检查必要的键
        required_keys = ['labels', 'architecture']
        for key in required_keys:
            if key not in config:
                print(f"Missing required key: {key}")
                return False
        
        # 检查专家配置
        experts = config['architecture'].get('experts', {})
        if not experts:
            print("No experts defined in configuration")
            return False
        
        for expert_name, expert_config in experts.items():
            expert_type = expert_config.get('type')
            if expert_type not in SUPPORTED_EXPERT_TYPES:
                print(f"Unsupported expert type: {expert_type} for expert: {expert_name}")
                return False
        
        # 检查融合策略
        fusion_strategy = config['architecture'].get('fusion', {}).get('strategy', 'concatenate')
        if fusion_strategy not in SUPPORTED_FUSION_STRATEGIES:
            print(f"Unsupported fusion strategy: {fusion_strategy}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Config validation error: {e}")
        return False


def create_model_from_config_file(config_path: str):
    """
    从配置文件创建模型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DynamicHarModel实例
    """
    import yaml
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not validate_config(config):
            raise ValueError("Invalid configuration file")
        
        return create_dynamic_har_model(config)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create model from config file {config_path}: {e}")


# 模块初始化时的信息输出
def _print_module_info():
    """打印模块信息（仅在调试时使用）"""
    info = f"""
MazeruHAR Model Layer v{__version__}
=====================================
支持的专家类型: {', '.join(SUPPORTED_EXPERT_TYPES)}
支持的融合策略: {', '.join(SUPPORTED_FUSION_STRATEGIES)}
核心功能: 动态多模态HAR模型构建
"""
    return info


# 仅供调试使用
if __name__ == "__main__":
    print(_print_module_info())
    
    # 测试默认配置
    print("测试默认配置:")
    default_config = get_example_config()
    if validate_config(default_config):
        print("✅ 默认配置有效")
        model = create_dynamic_har_model(default_config)
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("❌ 默认配置无效")