#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逻辑控制相关节点
"""

from typing import Dict, Any
from .base_node import BaseNode


class IfNode(BaseNode):
    """条件判断节点"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "if")
        self.inputs = {'condition': None}
        self.outputs = {'out_true': None, 'out_false': None}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行条件判断"""
        condition = inputs.get('condition', False)
        
        if condition:
            return {'out_true': {'value': True}, 'out_false': None}
        else:
            return {'out_true': None, 'out_false': {'value': False}}
    
    def get_display_name(self) -> str:
        return "如果"
    
    def get_description(self) -> str:
        return "根据条件选择执行路径"
    
    def to_code(self) -> str:
        return "# 条件判断\nif condition:\n    # true分支\nelse:\n    # false分支\n"


class WhileLoopNode(BaseNode):
    """while循环节点"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "while_loop")
        self.inputs = {'condition': None}
        self.outputs = {'loop_body': None, 'loop_end': None}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行循环"""
        condition = inputs.get('condition', False)
        
        if condition:
            return {'loop_body': {'continue': True}, 'loop_end': None}
        else:
            return {'loop_body': None, 'loop_end': {'finished': True}}
    
    def get_display_name(self) -> str:
        return "当循环"
    
    def get_description(self) -> str:
        return "当条件为真时重复执行"
    
    def to_code(self) -> str:
        return "# while循环\nwhile condition:\n    # 循环体\n"


class ComparisonNode(BaseNode):
    """比较节点"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "comparison")
        self.inputs = {'value_in': None}
        self.outputs = {'result': None}
        self.parameters = {
            'operator': '==',  # ==, !=, >, <, >=, <=
            'compare_value': 0
        }
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行比较"""
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
        return "条件判断"
    
    def get_description(self) -> str:
        return "比较两个值"
    
    def to_code(self) -> str:
        operator = self.get_parameter('operator', '==')
        compare_value = self.get_parameter('compare_value', 0)
        return f"# 比较\nresult = value {operator} {compare_value}\n"
