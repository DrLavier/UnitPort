#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logic Control Nodes
"""

from typing import Dict, Any, List
from .base_node import BaseNode


class IfNode(BaseNode):
    """Conditional branch node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "if")
        self.inputs = {'condition': None}
        self.outputs = {'out_true': None, 'out_false': None}
        self.parameters = {
            'condition_expr': '',
            'elif_conditions': []  # list of condition expressions
        }

    def add_elif(self, condition_expr: str = ""):
        """Add an elif branch"""
        elif_index = len(self.parameters.get('elif_conditions', []))
        self.parameters.setdefault('elif_conditions', []).append(condition_expr)
        self.inputs[f'elif_{elif_index}'] = None
        self.outputs[f'out_elif_{elif_index}'] = None

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional branch"""
        condition = inputs.get('condition', False)

        if condition:
            return {'out_true': {'value': True}, 'out_false': None}

        # Check elif conditions (if any)
        for idx in range(len(self.parameters.get('elif_conditions', []))):
            elif_cond = inputs.get(f'elif_{idx}', False)
            if elif_cond:
                return {f'out_elif_{idx}': {'value': True}, 'out_false': None}

        return {'out_true': None, 'out_false': {'value': False}}

    def get_display_name(self) -> str:
        return "If"

    def get_description(self) -> str:
        return "Select execution path based on condition"

    def to_code(self) -> str:
        condition_expr = self.get_parameter('condition_expr', '') or 'condition'
        elif_blocks = ""
        for idx, expr in enumerate(self.parameters.get('elif_conditions', [])):
            cond_expr = expr or f"elif_condition_{idx}"
            elif_blocks += f"elif {cond_expr}:\n    # elif branch {idx}\n    pass\n"
        return (
            f"# Conditional branch\n"
            f"if {condition_expr}:\n"
            f"    # true branch\n"
            f"    pass\n"
            f"{elif_blocks}"
            f"else:\n"
            f"    # false branch\n"
            f"    pass\n"
        )


class WhileLoopNode(BaseNode):
    """While loop node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "while_loop")
        self.inputs = {
            'condition': None,
            'for_start': None,
            'for_end': None,
            'for_step': None
        }
        self.outputs = {'loop_body': None, 'loop_end': None}
        self.parameters = {
            'loop_type': 'while',  # while | for
            'for_start': 0,
            'for_end': 1,
            'for_step': 1
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute loop"""
        loop_type = self.get_parameter('loop_type', 'while')
        if loop_type == 'for':
            start = inputs.get('for_start', self.get_parameter('for_start', 0))
            end = inputs.get('for_end', self.get_parameter('for_end', 1))
            step = inputs.get('for_step', self.get_parameter('for_step', 1))
            should_run = start < end if step > 0 else start > end
        else:
            should_run = inputs.get('condition', False)

        if should_run:
            return {'loop_body': {'continue': True}, 'loop_end': None}
        return {'loop_body': None, 'loop_end': {'finished': True}}

    def get_display_name(self) -> str:
        return "While Loop"

    def get_description(self) -> str:
        return "Repeat execution while condition is true"

    def to_code(self) -> str:
        loop_type = self.get_parameter('loop_type', 'while')
        if loop_type == 'for':
            start = self.get_parameter('for_start', 0)
            end = self.get_parameter('for_end', 1)
            step = self.get_parameter('for_step', 1)
            return f"# For loop\nfor i in range({start}, {end}, {step}):\n    # loop body\n    pass\n"
        condition_expr = self.get_parameter('condition_expr', '') or 'condition'
        return f"# While loop\nwhile {condition_expr}:\n    # loop body\n    pass\n"


class ComparisonNode(BaseNode):
    """Comparison node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "comparison")
        self.inputs = {'left': None, 'right': None, 'value_in': None, 'compare_value': None}
        self.outputs = {'result': None}
        self.parameters = {
            'operator': '==',  # ==, !=, >, <, >=, <=
            'compare_value': 0,
            'input_expr': '',
            'output_name': ''
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comparison"""
        value = inputs.get('left', inputs.get('value_in', 0))
        operator = self.get_parameter('operator', '==')
        compare_value = inputs.get('right', inputs.get('compare_value', self.get_parameter('compare_value', 0)))

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
        input_expr = self.get_parameter('input_expr', '') or 'left_value'
        output_name = self.get_parameter('output_name', '') or 'result'
        return f"# Comparison: {input_expr} {operator} {compare_value}\n{output_name} = {input_expr} {operator} {compare_value}\n"
