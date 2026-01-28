#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Execution Nodes
"""

from typing import Dict, Any
from .base_node import BaseNode


class ActionExecutionNode(BaseNode):
    """Action execution node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "action_execution")
        self.inputs = {'in': None}
        self.outputs = {'out': None}
        self.parameters = {
            'action': 'stand',
            'robot_model': None
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action"""
        action = self.get_parameter('action', 'stand')
        robot_model = self.get_parameter('robot_model')

        if robot_model is None:
            return {'out': {'status': 'error', 'message': 'Robot model not set'}}

        try:
            success = robot_model.run_action(action)
            return {
                'out': {
                    'status': 'success' if success else 'failed',
                    'action': action
                }
            }
        except Exception as e:
            return {'out': {'status': 'error', 'message': str(e)}}

    def get_display_name(self) -> str:
        return "Action Execution"

    def get_description(self) -> str:
        return "Execute robot action (stand, lift leg, walk, etc.)"

    def to_code(self) -> str:
        action = self.get_parameter('action', 'stand')
        return f"# Action execution: {action}\nrobot.run_action('{action}')\n"


class StopNode(BaseNode):
    """Stop node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "stop")
        self.inputs = {'in': None}
        self.outputs = {'out': None}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stop"""
        robot_model = self.get_parameter('robot_model')

        if robot_model:
            robot_model.stop()

        return {'out': {'status': 'stopped'}}

    def get_display_name(self) -> str:
        return "Stop"

    def get_description(self) -> str:
        return "Stop robot motion"

    def to_code(self) -> str:
        return "# Stop\nrobot.stop()\n"
