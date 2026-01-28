#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node Module - Unified Node Registry
Loads nodes from sys_nodes (built-in) and custom_nodes (community)
"""

from typing import Dict, Type, List, Optional

# Import base node
from sys_nodes.base_node import BaseNode

# Import system nodes
from sys_nodes import (
    ActionExecutionNode,
    StopNode,
    IfNode,
    WhileLoopNode,
    ComparisonNode,
    SensorInputNode
)

# Import custom nodes interface
from custom_nodes import CUSTOM_NODES, discover_custom_nodes, get_custom_nodes

# ============================================================================
# Global Node Registry
# ============================================================================

REGISTERED_NODES: Dict[str, Type[BaseNode]] = {}


def register_node(node_type: str, node_class: Type[BaseNode]):
    """
    Register a node type

    Args:
        node_type: Node type identifier
        node_class: Node class
    """
    REGISTERED_NODES[node_type] = node_class


def get_node_class(node_type: str) -> Optional[Type[BaseNode]]:
    """
    Get node class by type

    Args:
        node_type: Node type identifier

    Returns:
        Node class, or None if not found
    """
    return REGISTERED_NODES.get(node_type)


def create_node(node_type: str, node_id: str) -> BaseNode:
    """
    Create node instance

    Args:
        node_type: Node type identifier
        node_id: Node ID

    Returns:
        Node instance

    Raises:
        ValueError: If node type not found
    """
    node_class = get_node_class(node_type)
    if node_class is None:
        raise ValueError(f"Node type not found: {node_type}")

    return node_class(node_id)


def list_node_types() -> List[str]:
    """
    List all registered node types

    Returns:
        List of node type identifiers
    """
    return list(REGISTERED_NODES.keys())


def list_system_nodes() -> List[str]:
    """List only system (built-in) node types"""
    system_types = [
        'action_execution', 'stop',
        'if', 'while_loop', 'comparison',
        'sensor_input'
    ]
    return [t for t in system_types if t in REGISTERED_NODES]


def list_custom_nodes() -> List[str]:
    """List only custom node types"""
    system_types = set(list_system_nodes())
    return [t for t in REGISTERED_NODES.keys() if t not in system_types]


# ============================================================================
# Auto-register System Nodes
# ============================================================================

# Action nodes
register_node("action_execution", ActionExecutionNode)
register_node("stop", StopNode)

# Logic nodes
register_node("if", IfNode)
register_node("while_loop", WhileLoopNode)
register_node("comparison", ComparisonNode)

# Sensor nodes
register_node("sensor_input", SensorInputNode)


# ============================================================================
# Auto-register Custom Nodes
# ============================================================================

def load_custom_nodes():
    """Load and register all custom nodes"""
    # First, discover any auto-discoverable nodes
    try:
        discover_custom_nodes()
    except Exception:
        pass

    # Then register all custom nodes
    custom = get_custom_nodes()
    for node_type, node_class in custom.items():
        if node_type not in REGISTERED_NODES:
            register_node(node_type, node_class)


# Load custom nodes on import
load_custom_nodes()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base
    'BaseNode',

    # Registry functions
    'register_node',
    'get_node_class',
    'create_node',
    'list_node_types',
    'list_system_nodes',
    'list_custom_nodes',
    'load_custom_nodes',
    'REGISTERED_NODES',

    # System node classes
    'ActionExecutionNode',
    'StopNode',
    'IfNode',
    'WhileLoopNode',
    'ComparisonNode',
    'SensorInputNode'
]
