#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree机器人模块
"""

from .unitree_model import UnitreeModel, UNITREE_AVAILABLE, MUJOCO_AVAILABLE

SUPPORTED_MODELS = ["go2", "a1", "b1", "b2", "h1"]

__all__ = [
    'UnitreeModel',
    'UNITREE_AVAILABLE',
    'MUJOCO_AVAILABLE',
    'SUPPORTED_MODELS',
]
