#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Model Registry
Hot-pluggable robot model management
"""

from typing import Dict, Type, Optional
from .base import BaseRobotModel

# Global model registry
REGISTERED_MODELS: Dict[str, Type[BaseRobotModel]] = {}


def register_model(name: str, model_class: Type[BaseRobotModel]):
    """
    Register robot model

    Args:
        name: Model name (e.g., 'unitree')
        model_class: Model class
    """
    REGISTERED_MODELS[name] = model_class


def get_model(name: str) -> Optional[Type[BaseRobotModel]]:
    """
    Get robot model class

    Args:
        name: Model name

    Returns:
        Model class, or None if not found
    """
    return REGISTERED_MODELS.get(name)


def list_models() -> list:
    """
    List all registered models

    Returns:
        List of model names
    """
    return list(REGISTERED_MODELS.keys())


# ========== Auto-register All Models ==========

# Register Unitree model
try:
    from .unitree import UnitreeModel
    register_model("unitree", UnitreeModel)
except ImportError:
    pass

# Register other models (example)
# try:
#     from .other_robot import OtherRobotModel
#     register_model("other", OtherRobotModel)
# except ImportError:
#     pass

__all__ = [
    'BaseRobotModel',
    'register_model',
    'get_model',
    'list_models',
    'REGISTERED_MODELS'
]
