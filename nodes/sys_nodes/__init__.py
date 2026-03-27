#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Nodes Module
Built-in functional nodes for the visual programming system
"""

from .base_node import BaseNode
from .action_nodes import ActionExecutionNode, StopNode, BehaviorCallNode
from .logic_nodes import (
    IfNode, WhileLoopNode, ComparisonNode,
    SwitchNode, TryCatchNode, ParallelBranchNode, JoinNode,
)
from .sensor_nodes import SensorInputNode
from .protocol_emit_node import ProtocolEmitNode
from .utility_nodes import MathNode, TimerNode, VariableNode
from .workflow_boundary_nodes import StartNode, EndNode
from .base_control_nodes import (
    WaitNode, GateNode, TimeoutNode, RetryNode, CancelNode, AbortNode, BreakNode,
)
from .checkpoint_node import CheckpointNode
from .behavior_node import BehaviorNode
# PolicyNode / ConductorNode are DEPRECATED and no longer registered.
# Training nodes are NOT exported here — they belong to the Training Ground
# canvas only and must not appear in the Mission Canvas node registry.
# Import directly from nodes.sys_nodes.training_nodes when needed by the
# Training Ground canvas.

__all__ = [
    'BaseNode',
    # Action nodes
    'ActionExecutionNode',
    'StopNode',
    'BehaviorCallNode',
    # Logic nodes
    'IfNode',
    'WhileLoopNode',
    'ComparisonNode',
    'SwitchNode',
    'TryCatchNode',
    'ParallelBranchNode',
    'JoinNode',
    # Sensor nodes
    'SensorInputNode',
    'ProtocolEmitNode',
    # Utility nodes
    'MathNode',
    'TimerNode',
    'VariableNode',
    # Workflow boundary nodes
    'StartNode',
    'EndNode',
    # Base control nodes
    'WaitNode',
    'GateNode',
    'TimeoutNode',
    'RetryNode',
    'CancelNode',
    'AbortNode',
    'BreakNode',
    # Control pipeline nodes
    'CheckpointNode',
    'BehaviorNode',
]
