#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Execution Nodes

These nodes execute robot actions. They automatically adapt to different robot
brands/models by using the global RobotContext.

Design:
    - Action nodes don't directly depend on any specific robot SDK
    - They use RobotContext to get the appropriate robot model
    - The robot model (e.g., UnitreeModel) handles the actual hardware/simulation
"""

from typing import Dict, Any
from .base_node import BaseNode


class ActionExecutionNode(BaseNode):
    """
    Action execution node - executes robot actions.

    This node automatically adapts to the currently selected robot type
    by using RobotContext. The actual action implementation is delegated
    to the appropriate brand-specific model (e.g., models/unitree/unitree_model.py).
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "action_execution")
        self.inputs = {'flow_in': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'action': 'stand',
            'robot_model': None  # Can be set explicitly or obtained from RobotContext
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action on the robot via RobotContext lifecycle routing.
        """
        action = self.get_parameter('action', 'stand')
        try:
            from bin.core.robot_context import RobotContext
            success = RobotContext.run_action(action)
            return {
                'flow_out': {
                    'status': 'success' if success else 'failed',
                    'action': action,
                    'robot_type': RobotContext.get_robot_type(),
                }
            }
        except ImportError:
            return {'flow_out': {'status': 'error', 'message': 'RobotContext not available'}}
        except Exception as e:
            return {'flow_out': {'status': 'error', 'message': str(e)}}

    def get_display_name(self) -> str:
        return "Action Execution"

    def get_description(self) -> str:
        return "Execute robot action (stand, sit, walk, etc.). Adapts to selected robot type."

    def to_code(self) -> str:
        """
        Generate code that uses RobotContext for robot-agnostic execution.
        """
        action = self.get_parameter('action', 'stand')
        return (
            f"# Action: {action}\n"
            f"# Uses RobotContext to automatically select correct robot model\n"
            f"from bin.core.robot_context import RobotContext\n"
            f"RobotContext.run_action('{action}')\n"
        )


class StopNode(BaseNode):
    """Stop node - stops robot motion."""

    def __init__(self, node_id: str):
        super().__init__(node_id, "stop")
        self.inputs = {'flow_in': None}
        self.outputs = {'flow_out': None}
        self.parameters = {
            'robot_model': None
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stop command via RobotContext lifecycle routing."""
        try:
            from bin.core.robot_context import RobotContext
            RobotContext.stop()
        except ImportError:
            pass
        return {'flow_out': {'status': 'stopped'}}

    def get_display_name(self) -> str:
        return "Stop"

    def get_description(self) -> str:
        return "Stop robot motion"

    def to_code(self) -> str:
        return (
            "# Stop robot\n"
            "from bin.core.robot_context import RobotContext\n"
            "RobotContext.stop()\n"
        )


class BehaviorCallNode(BaseNode):
    """Behavior subgraph invocation node.

    Runtime execution is handled by ``NodeExecutor``'s behavior fast-path.
    This class exists so the editor/registry can construct and serialize
    behavior_call nodes without treating them as raw action nodes.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "behavior_call")
        self.inputs = {"flow_in": None, "control_pipe": None}
        self.outputs = {"flow_out": None}
        self.parameters = {
            "behavior_ref": "",
            "behavior_intent_id": "",
            "action": "",
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # control_pipe: absorbed silently in Circle 2; acted on in Circle 7
        _control_pipe = inputs.get("control_pipe")  # noqa: F841  (used in Circle 7)
        return {
            "flow_out": {
                "status": "skipped",
                "reason": "behavior_call_requires_runtime_executor",
            }
        }

    def get_display_name(self) -> str:
        return "Behavior"

    def get_description(self) -> str:
        return "Invoke a compiled behavior subgraph."

    def to_code(self) -> str:
        behavior_ref = self.get_parameter("behavior_ref", "")
        return f"# Behavior: {behavior_ref}\n"
