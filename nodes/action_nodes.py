#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作执行相关节点
"""

from typing import Dict, Any
from .base_node import BaseNode


class ActionExecutionNode(BaseNode):
    """动作执行节点"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "action_execution")
        self.inputs = {'in': None}
        self.outputs = {'out': None}
        self.parameters = {
            'action': 'stand',
            'robot_model': None
        }
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行动作"""
        action = self.get_parameter('action', 'stand')
        robot_model = self.get_parameter('robot_model')
        
        if robot_model is None:
            return {'out': {'status': 'error', 'message': '未设置机器人模型'}}
        
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
        return "动作执行"
    
    def get_description(self) -> str:
        return "执行机器人动作（如站立、抬腿、行走等）"
    
    def to_code(self) -> str:
        action = self.get_parameter('action', 'stand')
        return f"# 动作执行: {action}\nrobot.run_action('{action}')\n"


class StopNode(BaseNode):
    """停止节点"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "stop")
        self.inputs = {'in': None}
        self.outputs = {'out': None}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行停止"""
        robot_model = self.get_parameter('robot_model')
        
        if robot_model:
            robot_model.stop()
        
        return {'out': {'status': 'stopped'}}
    
    def get_display_name(self) -> str:
        return "停止"
    
    def get_description(self) -> str:
        return "停止机器人运动"
    
    def to_code(self) -> str:
        return "# 停止\nrobot.stop()\n"
