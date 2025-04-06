# This is the test file for migration of the code from Tensorflow to PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model_pytorch
import model

# test_mixaccgyro.py

"""
测试 TensorFlow 和 PyTorch 版本的 mixAccGyro / MixAccGyro 模块是否一致
"""

# ----------------- TensorFlow 测试 -----------------
def test_tf_mixaccgyro():
    import tensorflow as tf
    # 从 model.py 中导入 TensorFlow 版本的 mixAccGyro
    from model import mixAccGyro
    import numpy as np

    # 定义参数
    projection_quarter = 48
    projection_half = 96
    projection_dim = 192

    # 创建一个 dummy 输入，形状为 (batch, seq_len, projection_dim)
    x = tf.random.normal((2, 10, projection_dim))
    
    # 实例化 mixAccGyro 层
    mix_layer = mixAccGyro(projectionQuarter=projection_quarter,
                           projectionHalf=projection_half,
                           projection_dim=projection_dim)
    
    # 前向传播
    y = mix_layer(x)
    print("TensorFlow mixAccGyro output shape:", y.shape)


# ----------------- PyTorch 测试 -----------------
def test_torch_mixaccgyro():
    import torch
    # 从 model_pytorch.py 中导入 PyTorch 版本的 MixAccGyro
    from model_pytorch import MixAccGyro

    # 定义参数
    projection_quarter = 48
    projection_half = 96
    projection_dim = 192

    # 创建一个 dummy 输入，形状为 (batch, seq_len, projection_dim)
    x = torch.randn(2, 10, projection_dim)
    
    # 实例化 MixAccGyro 层
    mix_layer = MixAccGyro(projection_quarter=projection_quarter,
                           projection_half=projection_half,
                           projection_dim=projection_dim)
    
    # 前向传播
    y = mix_layer(x)
    print("PyTorch MixAccGyro output shape:", y.shape)


if __name__ == "__main__":
    print("Testing TensorFlow mixAccGyro...")
    test_tf_mixaccgyro()
    print("\nTesting PyTorch MixAccGyro...")
    test_torch_mixaccgyro()