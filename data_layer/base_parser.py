"""
抽象数据解析器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class DataParser(ABC):
    """数据解析器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get('name', '')
        self.data_path = config.get('data_path', './datasets/')
        self.modalities = config.get('modalities', {})
        
    @abstractmethod
    def parse_data(self, split: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        """解析数据的抽象方法"""
        pass
    
    @abstractmethod
    def get_modality_info(self) -> Dict[str, Dict[str, Any]]:
        """获取模态信息的抽象方法"""
        pass