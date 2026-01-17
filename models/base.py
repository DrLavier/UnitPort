#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人模型基类
所有机器人模型都应继承此基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRobotModel(ABC):
    """机器人模型基类"""
    
    def __init__(self, robot_type: str):
        """
        初始化机器人模型
        
        Args:
            robot_type: 机器人型号（如 go2, a1, b1）
        """
        self.robot_type = robot_type
        self.is_available = False
        self._actions = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化机器人模型
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def load_model(self) -> bool:
        """
        加载机器人模型文件
        
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    def run_action(self, action_name: str, **kwargs) -> bool:
        """
        执行指定动作
        
        Args:
            action_name: 动作名称
            **kwargs: 动作参数
        
        Returns:
            是否执行成功
        """
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """
        获取可用动作列表
        
        Returns:
            动作名称列表
        """
        pass
    
    def get_action_info(self, action_name: str) -> Optional[Dict[str, Any]]:
        """
        获取动作信息
        
        Args:
            action_name: 动作名称
        
        Returns:
            动作信息字典，如果不存在则返回None
        """
        return self._actions.get(action_name)
    
    def register_action(self, action_name: str, 
                       action_func: callable,
                       description: str = "",
                       parameters: Dict[str, Any] = None):
        """
        注册动作
        
        Args:
            action_name: 动作名称
            action_func: 动作执行函数
            description: 动作描述
            parameters: 动作参数定义
        """
        self._actions[action_name] = {
            'name': action_name,
            'function': action_func,
            'description': description,
            'parameters': parameters or {}
        }
    
    def is_action_available(self, action_name: str) -> bool:
        """
        检查动作是否可用
        
        Args:
            action_name: 动作名称
        
        Returns:
            是否可用
        """
        return action_name in self._actions
    
    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        获取传感器数据
        
        Returns:
            传感器数据字典
        """
        pass
    
    @abstractmethod
    def stop(self):
        """停止机器人运行"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(robot_type='{self.robot_type}')"
