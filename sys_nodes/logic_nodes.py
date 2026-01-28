#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logic Control Nodes
"""

from typing import Dict, Any
from .base_node import BaseNode


class IfNode(BaseNode):
    """Conditional branch node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "if")
        self.inputs = {'condition': None}
        self.outputs = {'out_true': None, 'out_false': None}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional branch"""
        condition = inputs.get('condition', False)

        if condition:
            return {'out_true': {'value': True}, 'out_false': None}
        else:
            return {'out_true': None, 'out_false': {'value': False}}

    def get_display_name(self) -> str:
        return "If"

    def get_description(self) -> str:
        return "Select execution path based on condition"

    def to_code(self) -> str:
        return "# Conditional branch\nif condition:\n    # true branch\nelse:\n    # false branch\n"


class WhileLoopNode(BaseNode):
    """While loop node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "while_loop")
        self.inputs = {'condition': None}
        self.outputs = {'loop_body': None, 'loop_end': None}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute loop"""
        condition = inputs.get('condition', False)

        if condition:
            return {'loop_body': {'continue': True}, 'loop_end': None}
        else:
            return {'loop_body': None, 'loop_end': {'finished': True}}

    def get_display_name(self) -> str:
        return "While Loop"

    def get_description(self) -> str:
        return "Repeat execution while condition is true"

    def to_code(self) -> str:
        return "# While loop\nwhile condition:\n    # loop body\n"


class ComparisonNode(BaseNode):
    """Comparison node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "comparison")
        self.inputs = {'value_in': None}
        self.outputs = {'result': None}
        self.parameters = {
            'operator': '==',  # ==, !=, >, <, >=, <=
            'compare_value': 0
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comparison"""
        value = inputs.get('value_in', 0)
        operator = self.get_parameter('operator', '==')
        compare_value = self.get_parameter('compare_value', 0)

        operators = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b
        }

        result = operators.get(operator, operators['=='])(value, compare_value)

        return {'result': {'value': result}}

    def get_display_name(self) -> str:
        return "Comparison"

    def get_description(self) -> str:
        return "Compare two values"

    def to_code(self) -> str:
        operator = self.get_parameter('operator', '==')
        compare_value = self.get_parameter('compare_value', 0)
        return f"# Comparison\nresult = value {operator} {compare_value}\n"
