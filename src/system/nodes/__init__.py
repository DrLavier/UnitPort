#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node Module - Unified Node Registry
Loads nodes from sys_nodes (built-in) and discovers custom nodes from
custom_mods/nodes/ at runtime.

Custom node discovery is owned by the system — custom_mods/nodes/ is a pure
data directory (no __init__.py required).  Users drop .py files there; the
system scans, validates, and registers them on startup.
"""

import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Type, List, Optional

# Import base node
from .sys_nodes.base_node import BaseNode

# Import system nodes
from .sys_nodes import (
    ActionExecutionNode,
    StopNode,
    BehaviorCallNode,
    IfNode,
    WhileLoopNode,
    ComparisonNode,
    SwitchNode,
    TryCatchNode,
    ParallelBranchNode,
    JoinNode,
    SensorInputNode,
    ProtocolEmitNode,
    MathNode,
    TimerNode,
    VariableNode,
    StartNode,
    EndNode,
    WaitNode,
    GateNode,
    TimeoutNode,
    RetryNode,
    CancelNode,
    AbortNode,
    BreakNode,
    CheckpointNode,
    BehaviorNode,
    # PolicyNode / ConductorNode are DEPRECATED — not imported here.
    # Training nodes (EnvConfigNode / TrainConfigNode / TrainNode) are
    # intentionally NOT imported here.  They live in
    # nodes.sys_nodes.training_nodes and are registered only by the
    # Training Ground canvas — never by the Mission Canvas registry.
)
from .sys_nodes.reactive_loco_node import ReactiveLocomotionNode

_logger = logging.getLogger(__name__)

# ============================================================================
# Project root resolution
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/system/nodes/__init__.py -> project root
_CONFIG_DIR = _PROJECT_ROOT / "src" / "config"
_NODE_REGISTRY_PATH = _CONFIG_DIR / "node_registry.json"

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
        'start', 'end',
        'action_execution', 'stop', 'behavior_call',
        'if', 'while_loop', 'comparison', 'switch', 'try_catch', 'parallel', 'join',
        'sensor_input',
        'protocol_emit',
        'math', 'timer', 'variable',
        'wait', 'gate', 'timeout', 'retry', 'cancel', 'abort', 'break',
        'checkpoint', 'behavior', 'reactive_loco',
    ]
    return [t for t in system_types if t in REGISTERED_NODES]


def list_custom_nodes() -> List[str]:
    """List only custom node types"""
    system_types = set(list_system_nodes())
    return [t for t in REGISTERED_NODES.keys() if t not in system_types]


# ============================================================================
# Auto-register System Nodes
# ============================================================================

# Workflow boundary nodes
register_node("start", StartNode)
register_node("end", EndNode)

# Action nodes
register_node("action_execution", ActionExecutionNode)
register_node("stop", StopNode)
register_node("behavior_call", BehaviorCallNode)

# Logic nodes
register_node("if", IfNode)
register_node("while_loop", WhileLoopNode)
register_node("comparison", ComparisonNode)
register_node("switch", SwitchNode)
register_node("try_catch", TryCatchNode)
register_node("parallel", ParallelBranchNode)
register_node("join", JoinNode)

# Sensor nodes
register_node("sensor_input", SensorInputNode)
register_node("protocol_emit", ProtocolEmitNode)

# Utility nodes
register_node("math", MathNode)
register_node("timer", TimerNode)
register_node("variable", VariableNode)

# Base control nodes
register_node("wait", WaitNode)
register_node("gate", GateNode)
register_node("timeout", TimeoutNode)
register_node("retry", RetryNode)
register_node("cancel", CancelNode)
register_node("abort", AbortNode)
register_node("break", BreakNode)

# Control pipeline nodes
register_node("checkpoint", CheckpointNode)
register_node("behavior", BehaviorNode)
register_node("reactive_loco", ReactiveLocomotionNode)
# policy / conductor are DEPRECATED — not registered.
# env_config / train_config / train are NOT registered here.
# The Training Ground canvas manages its own separate node table.


# ============================================================================
# Custom Node Discovery (system-owned)
# ============================================================================

def _load_registry_config() -> dict:
    """Load node_registry.json from src/config/."""
    if not _NODE_REGISTRY_PATH.exists():
        return {"system": {}, "custom": {"scan_dirs": ["custom_mods/nodes"], "registered": {}}}
    try:
        with open(_NODE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"system": {}, "custom": {"scan_dirs": ["custom_mods/nodes"], "registered": {}}}


def _save_registry_config(config: dict) -> None:
    """Persist updated custom node registrations back to node_registry.json."""
    try:
        with open(_NODE_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as exc:
        _logger.warning(f"[node_registry] Failed to save registry config: {exc}")


def _discover_custom_nodes_from_dir(scan_dir: Path) -> Dict[str, Type[BaseNode]]:
    """
    Scan a directory for .py files containing BaseNode subclasses.

    This is the system-side discovery logic.  The scanned directory (e.g.
    custom_mods/nodes/) does NOT need to be a Python package — each .py file
    is loaded as an isolated module via importlib.util.
    """
    discovered: Dict[str, Type[BaseNode]] = {}
    if not scan_dir.is_dir():
        return discovered

    for py_file in sorted(scan_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"_custom_node_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, BaseNode)
                        and attr is not BaseNode):
                    try:
                        temp = attr("__probe__")
                        ntype = getattr(temp, "node_type", None)
                        if ntype and ntype not in discovered:
                            discovered[ntype] = attr
                    except Exception:
                        pass
        except Exception as exc:
            _logger.warning(f"[node_registry] Failed to load custom node file {py_file.name}: {exc}")

    return discovered


def load_custom_nodes() -> None:
    """
    Discover and register custom nodes from all configured scan directories.

    Reads scan_dirs from node_registry.json, scans each directory, registers
    discovered nodes, and persists the updated custom registration list back
    to the config file.
    """
    config = _load_registry_config()
    custom_cfg = config.get("custom", {})
    scan_dirs = custom_cfg.get("scan_dirs", ["custom_mods/nodes"])

    all_custom: Dict[str, str] = {}

    for rel_dir in scan_dirs:
        abs_dir = _PROJECT_ROOT / rel_dir
        discovered = _discover_custom_nodes_from_dir(abs_dir)
        for ntype, nclass in discovered.items():
            if ntype not in REGISTERED_NODES:
                register_node(ntype, nclass)
                all_custom[ntype] = f"{rel_dir}/{nclass.__name__}"
                _logger.info(f"[node_registry] Registered custom node: {ntype} ({nclass.__name__})")

    # Persist discovered custom nodes to config
    custom_cfg["registered"] = all_custom
    config["custom"] = custom_cfg
    _save_registry_config(config)


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
    'StartNode',
    'EndNode',
    'ActionExecutionNode',
    'StopNode',
    'BehaviorCallNode',
    'IfNode',
    'WhileLoopNode',
    'ComparisonNode',
    'SwitchNode',
    'TryCatchNode',
    'ParallelBranchNode',
    'JoinNode',
    'SensorInputNode',
    'ProtocolEmitNode',
    'MathNode',
    'TimerNode',
    'VariableNode',
    'WaitNode',
    'GateNode',
    'TimeoutNode',
    'RetryNode',
    'CancelNode',
    'AbortNode',
    'BreakNode',
    'CheckpointNode',
    'BehaviorNode',
    'ReactiveLocomotionNode',
    # PolicyNode / ConductorNode: DEPRECATED, not exported.
    # EnvConfigNode / TrainConfigNode / TrainNode intentionally excluded.
]
