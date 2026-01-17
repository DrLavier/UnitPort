#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人模型注册中心
支持热插拔的机器人模型管理
"""

from typing import Dict, Type, Optional
from .base import BaseRobotModel
from utils.logger import get_logger

logger = get_logger()

# 全局模型注册表
REGISTERED_MODELS: Dict[str, Type[BaseRobotModel]] = {}


def register_model(name: str, model_class: Type[BaseRobotModel]):
    """
    注册机器人模型
    
    Args:
        name: 模型名称（如 'unitree'）
        model_class: 模型类
    """
    REGISTERED_MODELS[name] = model_class
    logger.info(f"注册模型: {name} -> {model_class.__name__}")


def get_model(name: str) -> Optional[Type[BaseRobotModel]]:
    """
    获取机器人模型类
    
    Args:
        name: 模型名称
    
    Returns:
        模型类，如果未找到则返回None
    """
    model_class = REGISTERED_MODELS.get(name)
    if model_class is None:
        logger.warning(f"未找到模型: {name}")
    return model_class


def list_models() -> list:
    """
    列出所有已注册的模型
    
    Returns:
        模型名称列表
    """
    return list(REGISTERED_MODELS.keys())


# ========== 自动注册所有模型 ==========

# 注册Unitree模型
try:
    from .unitree import UnitreeModel
    register_model("unitree", UnitreeModel)
except ImportError as e:
    logger.warning(f"无法导入Unitree模型: {e}")

# 注册其他模型（示例）
# try:
#     from .other_robot import OtherRobotModel
#     register_model("other", OtherRobotModel)
# except ImportError:
#     pass

logger.info(f"已注册 {len(REGISTERED_MODELS)} 个机器人模型")

__all__ = [
    'BaseRobotModel',
    'register_model',
    'get_model',
    'list_models',
    'REGISTERED_MODELS'
]
