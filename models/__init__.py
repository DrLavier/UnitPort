#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot model registry with dynamic brand loading."""

import importlib
from typing import Dict, Type, Optional

from .base import BaseRobotModel
from system.model_registry import list_brand_items, normalize_brand_id

# Global model registry
REGISTERED_MODELS: Dict[str, Type[BaseRobotModel]] = {}


def register_model(name: str, model_class: Type[BaseRobotModel]):
    """
    Register robot model

    Args:
        name: Model name (e.g., 'unitree')
        model_class: Model class
    """
    REGISTERED_MODELS[name.lower()] = model_class


def get_model(name: str) -> Optional[Type[BaseRobotModel]]:
    """
    Get robot model class

    Args:
        name: Model name

    Returns:
        Model class, or None if not found
    """
    key = name.lower()
    if not REGISTERED_MODELS:
        refresh_models()
    if key not in REGISTERED_MODELS:
        _autoload_model(key)
    return REGISTERED_MODELS.get(key)


def list_models() -> list:
    """
    List all registered models

    Returns:
        List of model names
    """
    return list(REGISTERED_MODELS.keys())


def refresh_models() -> None:
    """Attempt to import and register all discoverable brands."""
    for brand_id, _display_name in list_brand_items():
        _autoload_model(brand_id)


def _autoload_model(name: str) -> None:
    """Import the brand package on demand and register its model class if exported."""
    canonical_name = normalize_brand_id(name)
    brand_map = {brand_id: _module_key(brand_id) for brand_id, _display_name in list_brand_items()}
    module_name = brand_map.get(canonical_name)
    if module_name is None:
        return

    try:
        module = importlib.import_module(f"system.brand_packages.{module_name}")
    except Exception:
        return

    # Import BaseRobotModel fresh to avoid stale stub bindings from test isolation.
    try:
        _base_mod = importlib.import_module("models.base")
        _BaseModel = getattr(_base_mod, "BaseRobotModel", BaseRobotModel)
    except Exception:
        _BaseModel = BaseRobotModel

    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if (
            isinstance(attr, type)
            and issubclass(attr, _BaseModel)
            and attr is not _BaseModel
        ):
            register_model(canonical_name, attr)
            break


def _module_key(brand_name: str) -> str:
    return brand_name.strip().lower().replace("-", "").replace("_", "")


__all__ = [
    'BaseRobotModel',
    'register_model',
    'get_model',
    'list_models',
    'REGISTERED_MODELS',
]
