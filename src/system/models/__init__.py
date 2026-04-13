#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot model registry with dynamic brand loading."""

from typing import Dict, Type, Optional

from .base import BaseRobotModel
from src.system.brand_packages import get_brand_model_class, list_brands
from src.system.models.model_registry import list_brand_items, normalize_brand_id

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
    """Resolve a brand to its model class via the central brand registry.

    Brands declared in ``model_registry`` but absent from
    ``src.system.brand_packages._BRAND_MODULES`` (e.g. unimplemented brands
    like xiaomi) are silently skipped.
    """
    canonical_name = normalize_brand_id(name)
    if canonical_name not in list_brands():
        return
    try:
        model_class = get_brand_model_class(canonical_name)
    except (KeyError, ImportError):
        return
    if isinstance(model_class, type) and issubclass(model_class, BaseRobotModel):
        register_model(canonical_name, model_class)


__all__ = [
    'BaseRobotModel',
    'register_model',
    'get_model',
    'list_models',
    'REGISTERED_MODELS',
]
