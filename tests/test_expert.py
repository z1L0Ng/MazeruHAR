# tests/test_experts.py
"""
专家模块单元测试
验证各个专家模块的功能和接口一致性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Tuple

# 尝试导入pytest，如果不可用则提供简单的替代
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    print("Warning: pytest not available, using simple test runner")
    PYTEST_AVAILABLE = False
    
    # 简单的pytest替代
    class MockPytest:
        def skip(self, reason):
            def decorator(func):
                return func
            return decorator
    
    pytest = MockPytest()

from model_layer.experts import (
    ExpertModel,
    DummyExpert,
    TransformerExpert,
    RNNExpert,
    CNNExpert,
    create_expert_from_config,
    create_expert_from_preset,
    validate_expert_config,
    EXPERT_TYPES,
    EXPERT_PRESETS,
    HYBRID_AVAILABLE
)

# 尝试导入混合专家
if HYBRID_AVAILABLE:
    from model_layer.experts import HybridExpert
else:
    HybridExpert = None


class TestExpertBase:
    """
    专家基类测试
    """
    
    def test_dummy_expert(self):
        """测试虚拟专家"""
        input_shape = (100, 6)
        output_dim = 64
        
        expert = DummyExpert(input_shape, output_dim)
        
        # 测试基本属性
        assert expert.input_shape == input_shape
        assert expert.output_dim == output_dim
        assert expert.get_output_dim() == output_dim
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            output = expert(x)
            assert output.shape == (batch_size, output_dim)
            
            # 测试特征提取
            features = expert.extract_features(x)
            assert features.shape == (batch_size, output_dim)
    
    def test_expert_config(self):
        """测试专家配置"""
        input_shape = (100, 6)
        output_dim = 64
        
        expert = DummyExpert(input_shape, output_dim)
        config = expert.get_config()
        
        assert config['expert_type'] == 'DummyExpert'
        assert config['input_shape'] == input_shape
        assert config['output_dim'] == output_dim
    
    def test_parameter_operations(self):
        """测试参数操作"""
        input_shape = (100, 6)
        output_dim = 64
        
        expert = DummyExpert(input_shape, output_dim)
        
        # 测试参数数量
        param_count = expert.get_parameter_count()
        assert param_count > 0
        
        # 测试参数冻结/解冻
        expert.freeze_parameters()
        for param in expert.parameters():
            assert not param.requires_grad
        
        expert.unfreeze_parameters()
        for param in expert.parameters():
            assert param.requires_grad


class TestTransformerExpert:
    """
    Transformer专家测试
    """
    
    def test_transformer_expert_creation(self):
        """测试Transformer专家创建"""
        input_shape = (500, 6)
        output_dim = 128
        
        expert = TransformerExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            projection_dim=128,
            patch_size=16,
            time_step=16,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            dropout_rate=0.1
        )
        
        assert expert.input_shape == input_shape
        assert expert.output_dim == output_dim
        assert expert.projection_dim == 128
        assert expert.num_heads == 4
        assert expert.num_layers == 2
    
    def test_transformer_expert_forward(self):
        """测试Transformer专家前向传播"""
        input_shape = (500, 6)
        output_dim = 128
        batch_size = 4
        
        expert = TransformerExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            projection_dim=128,
            num_layers=2
        )
        
        x = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            output = expert(x)
            assert output.shape == (batch_size, output_dim)
            
            # 检查中间特征
            assert expert.intermediate_features is not None
            assert expert.intermediate_features.shape == (batch_size, output_dim)


class TestRNNExpert:
    """
    RNN专家测试
    """
    
    def test_rnn_expert_creation(self):
        """测试RNN专家创建"""
        input_shape = (500, 6)
        output_dim = 128
        
        expert = RNNExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=64,
            num_layers=2,
            rnn_type='gru',
            bidirectional=True
        )
        
        assert expert.input_shape == input_shape
        assert expert.output_dim == output_dim
        assert expert.hidden_dim == 64
        assert expert.num_layers == 2
        assert expert.rnn_type == 'gru'
        assert expert.bidirectional == True
    
    def test_rnn_expert_different_types(self):
        """测试不同类型的RNN专家"""
        input_shape = (100, 6)
        output_dim = 64
        batch_size = 4
        
        rnn_types = ['lstm', 'gru', 'rnn']
        
        for rnn_type in rnn_types:
            expert = RNNExpert(
                input_shape=input_shape,
                output_dim=output_dim,
                hidden_dim=32,
                num_layers=1,
                rnn_type=rnn_type
            )
            
            x = torch.randn(batch_size, *input_shape)
            
            with torch.no_grad():
                output = expert(x)
                assert output.shape == (batch_size, output_dim)
    
    def test_rnn_expert_channel_specific(self):
        """测试通道特定的RNN专家"""
        input_shape = (100, 6)
        output_dim = 64
        batch_size = 4
        
        expert = RNNExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=1,
            rnn_type='gru',
            use_channel_specific=True
        )
        
        x = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            output = expert(x)
            assert output.shape == (batch_size, output_dim)


class TestCNNExpert:
    """
    CNN专家测试
    """
    
    def test_cnn_expert_creation(self):
        """测试CNN专家创建"""
        input_shape = (500, 6)
        output_dim = 128
        
        expert = CNNExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            num_layers=3,
            base_channels=64,
            kernel_sizes=[3, 5, 7],
            use_multiscale=True
        )
        
        assert expert.input_shape == input_shape
        assert expert.output_dim == output_dim
        assert expert.num_layers == 3
        assert expert.base_channels == 64
        assert expert.kernel_sizes == [3, 5, 7]
        assert expert.use_multiscale == True
    
    def test_cnn_expert_forward(self):
        """测试CNN专家前向传播"""
        input_shape = (100, 6)
        output_dim = 64
        batch_size = 4
        
        expert = CNNExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            num_layers=2,
            base_channels=32
        )
        
        x = torch.randn(batch_size, *input_shape)
        
        with torch.no_grad():
            output = expert(x)
            assert output.shape == (batch_size, output_dim)


class TestHybridExpert:
    """
    混合专家测试
    """
    
    def test_hybrid_expert_creation(self):
        """测试混合专家创建"""
        if not HYBRID_AVAILABLE:
            print("Skipping hybrid expert tests - not available")
            return
            
        input_shape = (500, 6)
        output_dim = 128
        
        expert = HybridExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            projection_dim=128,
            rnn_hidden_dim=64,
            rnn_num_layers=2,
            num_heads=4,
            num_transformer_layers=2,
            fusion_method='concat'
        )
        
        assert expert.input_shape == input_shape
        assert expert.output_dim == output_dim
        assert expert.projection_dim == 128
        assert expert.rnn_hidden_dim == 64
        assert expert.fusion_method == 'concat'
    
    def test_hybrid_expert_different_fusion(self):
        """测试不同融合方法的混合专家"""
        if not HYBRID_AVAILABLE:
            print("Skipping hybrid expert fusion tests - not available")
            return
            
        input_shape = (100, 6)
        output_dim = 64
        batch_size = 4
        
        fusion_methods = ['concat', 'add', 'attention']
        
        for fusion_method in fusion_methods:
            expert = HybridExpert(
                input_shape=input_shape,
                output_dim=output_dim,
                projection_dim=64,
                rnn_hidden_dim=32,
                rnn_num_layers=1,
                num_heads=2,
                num_transformer_layers=1,
                fusion_method=fusion_method
            )
            
            x = torch.randn(batch_size, *input_shape)
            
            with torch.no_grad():
                output = expert(x)
                assert output.shape == (batch_size, output_dim)


class TestExpertFactory:
    """
    专家工厂函数测试
    """
    
    def test_create_expert_from_config(self):
        """测试从配置创建专家"""
        config = {
            "type": "rnn",
            "input_shape": [100, 6],
            "output_dim": 64,
            "params": {
                "hidden_dim": 32,
                "num_layers": 1,
                "rnn_type": "gru"
            }
        }
        
        expert = create_expert_from_config(config)
        
        assert isinstance(expert, RNNExpert)
        assert expert.input_shape == (100, 6)
        assert expert.output_dim == 64
        assert expert.hidden_dim == 32
    
    def test_create_expert_from_preset(self):
        """测试从预设创建专家"""
        input_shape = (100, 6)
        output_dim = 64
        
        # 测试几个预设
        presets_to_test = ['transformer_small', 'rnn_gru', 'cnn_simple']
        
        for preset_name in presets_to_test:
            if preset_name in EXPERT_PRESETS:
                expert = create_expert_from_preset(preset_name, input_shape, output_dim)
                assert isinstance(expert, ExpertModel)
                assert expert.input_shape == input_shape
                assert expert.output_dim == output_dim
    
    def test_validate_expert_config(self):
        """测试专家配置验证"""
        # 有效配置
        valid_config = {
            "type": "rnn",
            "input_shape": [100, 6],
            "output_dim": 64
        }
        
        assert validate_expert_config(valid_config) == True
        
        # 无效配置 - 缺少必需字段
        invalid_config = {
            "type": "rnn",
            "input_shape": [100, 6]
            # 缺少 output_dim
        }
        
        assert validate_expert_config(invalid_config) == False
        
        # 无效配置 - 错误的类型
        invalid_type_config = {
            "type": "invalid_type",
            "input_shape": [100, 6],
            "output_dim": 64
        }
        
        assert validate_expert_config(invalid_type_config) == False


class TestExpertConsistency:
    """
    专家一致性测试
    """
    
    def test_all_experts_consistency(self):
        """测试所有专家的一致性"""
        input_shape = (100, 6)
        output_dim = 64
        batch_size = 4
        
        # 创建所有类型的专家
        experts = {
            'transformer': TransformerExpert(input_shape, output_dim, projection_dim=64, num_layers=1),
            'rnn': RNNExpert(input_shape, output_dim, hidden_dim=32, num_layers=1),
            'cnn': CNNExpert(input_shape, output_dim, num_layers=2, base_channels=32)
        }
        
        # 只有在混合专家可用时才测试
        if HYBRID_AVAILABLE:
            experts['hybrid'] = HybridExpert(
                input_shape, output_dim, projection_dim=64, rnn_hidden_dim=32, 
                rnn_num_layers=1, num_transformer_layers=1
            )
        
        # 测试输入
        x = torch.randn(batch_size, *input_shape)
        
        for expert_name, expert in experts.items():
            with torch.no_grad():
                # 测试前向传播
                output = expert(x)
                assert output.shape == (batch_size, output_dim), f"Expert {expert_name} output shape mismatch"
                
                # 测试特征提取
                features = expert.extract_features(x)
                assert features.shape == (batch_size, output_dim), f"Expert {expert_name} features shape mismatch"
                
                # 测试配置获取
                config = expert.get_config()
                assert 'expert_type' in config, f"Expert {expert_name} missing expert_type in config"
                assert 'input_shape' in config, f"Expert {expert_name} missing input_shape in config"
                assert 'output_dim' in config, f"Expert {expert_name} missing output_dim in config"
                
                # 测试参数数量
                param_count = expert.get_parameter_count()
                assert param_count > 0, f"Expert {expert_name} has no parameters"
                
                # 测试中间特征
                if hasattr(expert, 'intermediate_features'):
                    assert expert.intermediate_features is not None, f"Expert {expert_name} has no intermediate features"

    def test_performance(self):
        """
        性能测试
        """
        print("\nPerformance Tests...")
        print("=" * 30)
        
        import time
        
        input_shape = (500, 6)
        output_dim = 128
        batch_size = 32
        
        # 创建测试专家
        experts = {
            'transformer': TransformerExpert(input_shape, output_dim, projection_dim=128, num_layers=2),
            'rnn': RNNExpert(input_shape, output_dim, hidden_dim=64, num_layers=2),
            'cnn': CNNExpert(input_shape, output_dim, num_layers=3, base_channels=64)
        }
        
        # 只有在混合专家可用时才测试
        if HYBRID_AVAILABLE:
            experts['hybrid'] = HybridExpert(
                input_shape, output_dim, projection_dim=128, rnn_hidden_dim=64, 
                rnn_num_layers=1, num_transformer_layers=1
            )
        
        # 测试数据
        x = torch.randn(batch_size, *input_shape)
        
        for expert_name, expert in experts.items():
            # 参数数量
            param_count = expert.get_parameter_count()
            
            # 推理时间测试
            expert.eval()
            with torch.no_grad():
                # 预热
                for _ in range(5):
                    _ = expert(x)
                
                # 实际测试
                start_time = time.time()
                num_runs = 50
                for _ in range(num_runs):
                    _ = expert(x)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs * 1000  # ms
            
            print(f"{expert_name:12} | {param_count:8,} params | {avg_time:6.2f} ms/batch")

    def test_demo_expert_usage(self):
        """
        演示专家使用
        """
        print("\nExpert Usage Demo...")
        print("=" * 30)
        
        # 1. 基本使用
        print("1. Basic Usage:")
        input_shape = (100, 6)
        output_dim = 64
        
        expert = RNNExpert(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=1,
            rnn_type='gru'
        )
        
        print(f"   Created: {expert}")
        print(f"   Parameters: {expert.get_parameter_count():,}")
        
        # 2. 配置使用
        print("\n2. Config-based Usage:")
        config = {
            "type": "transformer",
            "input_shape": [100, 6],
            "output_dim": 64,
            "params": {
                "projection_dim": 64,
                "num_layers": 1,
                "num_heads": 2
            }
        }
        
        expert = create_expert_from_config(config)
        print(f"   Created: {expert}")
        print(f"   Config: {expert.get_config()}")
        
        # 3. 预设使用
        print("\n3. Preset Usage:")
        expert = create_expert_from_preset('cnn_simple', input_shape, output_dim)
        print(f"   Created: {expert}")
        
        # 4. 推理演示
        print("\n4. Inference Demo:")
        x = torch.randn(4, *input_shape)
        
        with torch.no_grad():
            output = expert(x)
            features = expert.extract_features(x)
            
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Features shape: {features.shape}")


def run_all_tests():
    """
    运行所有测试
    """
    print("Running Expert Module Tests...")
    print("=" * 50)
    
    # 测试类列表
    test_classes = [
        TestExpertBase,
        TestTransformerExpert,
        TestRNNExpert,
        TestCNNExpert,
        TestHybridExpert,
        TestExpertFactory,
        TestExpertConsistency
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # 获取测试方法
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # 创建测试实例并运行测试
                test_instance = test_class()
                getattr(test_instance, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()
    
    if success:
        # 运行额外的测试
        print("\nRunning additional tests...")
        test_consistency = TestExpertConsistency()
        test_consistency.test_performance()
        test_consistency.test_demo_expert_usage()
        
        print("\n" + "=" * 50)
        print("✓ All tests and demos completed successfully!")
    else:
        print("\n✗ Tests failed. Please check the implementation.")