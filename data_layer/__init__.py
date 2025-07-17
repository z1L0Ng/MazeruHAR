"""
数据层模块 - 修复导入错误版本
"""

# 使用绝对导入避免相对导入问题
try:
    from base_parser import DataParser
    from shl_parser import SHLDataParser
    from universal_dataset import UniversalHARDataset
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from .base_parser import DataParser
    from .shl_parser import SHLDataParser
    from .universal_dataset import UniversalHARDataset

__all__ = [
    'DataParser',
    'SHLDataParser', 
    'UniversalHARDataset'
]

__version__ = '1.0.0'
__author__ = 'MazeruHAR Team'
__description__ = 'Data layer for MazeruHAR framework'