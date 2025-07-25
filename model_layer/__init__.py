# model_layer/__init__.py

# 从正确的子模块中导入
from .dynamic_har_model import DynamicHarModel, create_dynamic_har_model
from .experts import ExpertModel, TransformerExpert, RNNExpert, CNNExpert, HybridExpert
from fusion_layer import FusionStrategy # 已修正

# 将 get_example_config 函数移到这里
def get_example_config():
    """获取示例配置"""
    return {
        'architecture': {
            'num_classes': 8,
            'experts': {
                'imu_expert': {
                    'type': 'TransformerExpert',
                    'modality': 'imu',
                    'params': {
                        'input_shape': (128, 6),
                        'output_dim': 128,
                        'num_heads': 8,
                        'num_layers': 4,
                        'dropout_rate': 0.1
                    }
                },
                'pressure_expert': {
                    'type': 'RNNExpert',
                    'modality': 'pressure',
                    'params': {
                        'input_shape': (128, 1),
                        'output_dim': 64,
                        'rnn_type': 'LSTM',
                        'num_layers': 2
                    }
                }
            },
            'fusion': {
                'strategy': 'concatenate',
                'params': {}
            },
            'classifier': {
                'layers': [192, 128, 8], # 128(IMU) + 64(Pressure) = 192
                'dropout': 0.2
            }
        }
    }

# 定义此模块的公共API
__all__ = [
    'DynamicHarModel',
    'create_dynamic_har_model',
    'ExpertModel',
    'TransformerExpert',
    'RNNExpert', 
    'CNNExpert',
    'HybridExpert',
    'FusionStrategy', # 已修正
    'get_example_config'
]