#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点基类
所有可拖拽的功能节点都应继承此基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseNode(ABC):
    """节点基类"""
    
    def __init__(self, node_id: str, node_type: str):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一ID
            node_type: 节点类型
        """
        self.node_id = node_id
        self.node_type = node_type
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行节点逻辑
        
        Args:
            inputs: 输入数据字典
        
        Returns:
            输出数据字典
        """
        pass
    
    @abstractmethod
    def get_display_name(self) -> str:
        """获取节点显示名称"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """获取节点描述"""
        pass
    
    def get_input_ports(self) -> List[str]:
        """获取输入端口列表"""
        return list(self.inputs.keys())
    
    def get_output_ports(self) -> List[str]:
        """获取输出端口列表"""
        return list(self.outputs.keys())
    
    def set_parameter(self, key: str, value: Any):
        """设置参数"""
        self.parameters[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取参数"""
        return self.parameters.get(key, default)
    
    def to_code(self) -> str:
        """
        将节点转换为代码
        
        Returns:
            生成的代码字符串
        """
        return f"# {self.get_display_name()}: {self.node_id}\n"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.node_id}', type='{self.node_type}')"
