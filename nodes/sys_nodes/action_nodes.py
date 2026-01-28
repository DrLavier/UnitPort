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
        Execute action on the robot.

        The robot model is obtained in this priority:
        1. From parameters['robot_model'] if explicitly set
        2. From global RobotContext (based on UI robot selection)
        """
        action = self.get_parameter('action', 'stand')
        robot_model = self.get_parameter('robot_model')

        # If no robot model provided, try to get from RobotContext
        if robot_model is None:
            try:
                from bin.core.robot_context import RobotContext
                robot_model = RobotContext.get_robot_model()
            except ImportError:
                pass

        if robot_model is None:
            return {'flow_out': {'status': 'error', 'message': 'Robot model not available'}}

        try:
            success = robot_model.run_action(action)
            return {
                'flow_out': {
                    'status': 'success' if success else 'failed',
                    'action': action,
                    'robot_type': getattr(robot_model, 'robot_type', 'unknown')
                }
            }
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
        """Execute stop command."""
        robot_model = self.get_parameter('robot_model')

        # If no robot model provided, try to get from RobotContext
        if robot_model is None:
            try:
                from bin.core.robot_context import RobotContext
                robot_model = RobotContext.get_robot_model()
            except ImportError:
                pass

        if robot_model:
            robot_model.stop()

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
