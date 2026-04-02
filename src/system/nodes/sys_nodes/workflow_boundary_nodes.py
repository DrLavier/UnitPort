#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Boundary Nodes
Start and End nodes that define workflow entry/exit points.
"""

from typing import Dict, Any, List


from .base_node import BaseNode


class StartNode(BaseNode):
    """Start node - workflow entry point with robot type selection."""

    def __init__(self, node_id: str):
        super().__init__(node_id, "start")
        self.parameters['robot_brand'] = 'unitree'
        self.parameters['robot_type'] = 'go2'

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from src.system.core.robot_context import RobotContext
        RobotContext.set_robot_type(self.parameters.get('robot_type', 'go2'))
        return {}

    def get_display_name(self) -> str:
        return "Start"

    def get_description(self) -> str:
        return "Workflow entry point - sets the robot type"

    def get_output_ports(self) -> List[str]:
        return ["flow_out"]

    def get_input_ports(self) -> List[str]:
        return []

    def to_code(self) -> str:
        robot_type = self.parameters.get('robot_type', 'go2')
        return f"RobotContext.set_robot_type('{robot_type}')"


class EndNode(BaseNode):
    """End node - workflow exit point.

    Collects all upstream input data and produces a workflow completion
    summary.  The ``status`` parameter allows the canvas author to label
    the exit condition (e.g. "success", "error", "timeout").
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "end")
        self.parameters = {
            'status': 'success',       # user-defined exit label
            'message': '',             # optional exit message
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        status = self.get_parameter('status', 'success')
        message = self.get_parameter('message', '')
        return {
            'workflow_status': status,
            'workflow_message': message,
            'collected_inputs': {k: v for k, v in inputs.items() if v is not None},
        }

    def get_display_name(self) -> str:
        return "End"

    def get_description(self) -> str:
        return "Workflow exit point"

    def get_output_ports(self) -> List[str]:
        return []

    def get_input_ports(self) -> List[str]:
        return ["flow_in"]

    def to_code(self) -> str:
        return "# End of workflow"
