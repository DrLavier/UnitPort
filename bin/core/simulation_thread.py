#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真线程模块
在独立线程中运行MuJoCo仿真
"""

from PySide6.QtCore import QThread, Signal
from typing import Optional
from bin.core.logger import log_info, log_error, log_debug

# 移除: from utils.logger import get_logger
# logger = get_logger()


class SimulationThread(QThread):
    """仿真线程基类"""
    
    # 信号定义
    simulation_started = Signal(str)
    simulation_finished = Signal(str)
    error_occurred = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, robot_model, action: str, **kwargs):
        """
        初始化仿真线程
        
        Args:
            robot_model: 机器人模型实例
            action: 要执行的动作
            **kwargs: 其他参数
        """
        super().__init__()
        self.robot_model = robot_model
        self.action = action
        self.kwargs = kwargs
        self.running = False
        self._stop_requested = False
    
    def run(self):
        """运行仿真"""
        try:
            self.running = True
            self._stop_requested = False
            
            # 发射开始信号
            start_msg = f"开始仿真: {self.robot_model.robot_type} - {self.action}"
            self.simulation_started.emit(start_msg)
            log_info(start_msg)
            
            # 执行实际仿真
            self._run_simulation()
            
        except Exception as e:
            error_msg = f"仿真错误: {str(e)}"
            log_error(error_msg)
            self.error_occurred.emit(error_msg)
            
        finally:
            self.running = False
            finish_msg = "仿真完成" if not self._stop_requested else "仿真已停止"
            self.simulation_finished.emit(finish_msg)
            log_info(finish_msg)
    
    def _run_simulation(self):
        """执行仿真（子类应重写此方法）"""
        # 调用机器人模型的动作执行方法
        success = self.robot_model.run_action(self.action, **self.kwargs)
        
        if not success:
            raise RuntimeError(f"动作执行失败: {self.action}")
    
    def stop(self):
        """停止仿真"""
        log_info("收到停止仿真请求")
        self._stop_requested = True
        
        # 停止机器人
        if self.robot_model:
            self.robot_model.stop()
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running and not self._stop_requested