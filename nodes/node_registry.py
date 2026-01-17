#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点注册表
管理所有可用的节点类型
"""

from typing import Dict, Type, List
from .base_node import BaseNode
from utils.logger import get_logger

logger = get_logger()

# 全局节点注册表
REGISTERED_NODES: Dict[str, Type[BaseNode]] = {}


def register_node(node_type: str, node_class: Type[BaseNode]):
    """
    注册节点类型
    
    Args:
        node_type: 节点类型标识
        node_class: 节点类
    """
    REGISTERED_NODES[node_type] = node_class
    logger.info(f"注册节点: {node_type} -> {node_class.__name__}")


def get_node_class(node_type: str) -> Type[BaseNode]:
    """
    获取节点类
    
    Args:
        node_type: 节点类型标识
    
    Returns:
        节点类，如果未找到则返回None
    """
    return REGISTERED_NODES.get(node_type)


def create_node(node_type: str, node_id: str) -> BaseNode:
    """
    创建节点实例
    
    Args:
        node_type: 节点类型标识
        node_id: 节点ID
    
    Returns:
        节点实例
    """
    node_class = get_node_class(node_type)
    if node_class is None:
        raise ValueError(f"未找到节点类型: {node_type}")
    
    return node_class(node_id)


def list_node_types() -> List[str]:
    """
    列出所有已注册的节点类型
    
    Returns:
        节点类型列表
    """
    return list(REGISTERED_NODES.keys())


# ========== 自动注册所有节点 ==========

# 导入并注册动作节点
try:
    from .action_nodes import ActionExecutionNode, StopNode
    register_node("action_execution", ActionExecutionNode)
    register_node("stop", StopNode)
except ImportError as e:
    logger.warning(f"无法导入动作节点: {e}")

# 导入并注册逻辑节点
try:
    from .logic_nodes import IfNode, WhileLoopNode, ComparisonNode
    register_node("if", IfNode)
    register_node("while_loop", WhileLoopNode)
    register_node("comparison", ComparisonNode)
except ImportError as e:
    logger.warning(f"无法导入逻辑节点: {e}")

# 导入并注册传感器节点
try:
    from .sensor_nodes import SensorInputNode
    register_node("sensor_input", SensorInputNode)
except ImportError as e:
    logger.warning(f"无法导入传感器节点: {e}")

logger.info(f"已注册 {len(REGISTERED_NODES)} 个节点类型")

__all__ = [
    'register_node',
    'get_node_class',
    'create_node',
    'list_node_types',
    'REGISTERED_NODES'
]
