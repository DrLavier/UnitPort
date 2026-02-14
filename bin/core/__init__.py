#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Module
"""

from .config_manager import ConfigManager
from .localisation import get_localisation, tr, tr_list, LocalisationManager


def __getattr__(name):
    """Lazy-load runtime bridge symbols to avoid import cycles."""
    if name == "SimulationThread":
        from .simulation_thread import SimulationThread as _SimulationThread
        return _SimulationThread
    if name == "NodeExecutor":
        from .node_executor import NodeExecutor as _NodeExecutor
        return _NodeExecutor
    raise AttributeError(name)

__all__ = [
    'ConfigManager',
    'SimulationThread',
    'NodeExecutor',
    'get_localisation',
    'tr',
    'tr_list',
    'LocalisationManager'
]
