#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree机器人模型
支持 Go2, A1, B1 等型号的MuJoCo仿真
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from models.base import BaseRobotModel
from bin.core.logger import log_info, log_error, log_debug, log_warning
from bin.core.config_manager import ConfigManager

# ========== Unitree MuJoCo 导入部分 ==========
UNITREE_AVAILABLE = False
MUJOCO_AVAILABLE = False

try:
    # 获取配置
    config = ConfigManager()
    project_root = config.get_path('project_root')
    
    # 添加Unitree SDK路径
    unitree_sdk_path = config.get_path('unitree_sdk')
    unitree_mujoco_path = config.get_path('unitree_mujoco')
    
    possible_paths = [unitree_sdk_path, unitree_mujoco_path]
    
    added_paths = []
    for path in possible_paths:
        if path.exists():
            sys.path.insert(0, str(path))
            added_paths.append(str(path))
            log_info(f"✅ 添加路径: {path}")
    
    if added_paths:
        log_info(f"已添加 {len(added_paths)} 个路径到 sys.path")
    
    # 导入 MuJoCo
    try:
        import mujoco
        log_info(f"✅ mujoco 版本: {mujoco.__version__}")
        MUJOCO_AVAILABLE = True
        
        try:
            import mujoco.viewer
            log_info("✅ mujoco.viewer 导入成功")
        except ImportError as e:
            log_warning(f"⚠️ mujoco.viewer 导入失败: {e}")
            MUJOCO_AVAILABLE = False
            
    except ImportError as e:
        log_error(f"❌ 无法导入 mujoco: {e}")
        MUJOCO_AVAILABLE = False
    
    # 导入 Unitree SDK
    try:
        import importlib.util
        
        sdk_spec = importlib.util.find_spec("unitree_sdk2py")
        if sdk_spec is None:
            log_warning("⚠️ 未找到 unitree_sdk2py 模块")
        else:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
            log_info("✅ Unitree SDK 导入成功")
            UNITREE_AVAILABLE = True
    except ImportError as e:
        log_warning(f"⚠️ 无法导入 Unitree SDK: {e}")

except Exception as e:
    log_error(f"⚠️ 导入过程中发生错误: {e}")
    UNITREE_AVAILABLE = False
    MUJOCO_AVAILABLE = False

# 如果MuJoCo可用，则认为Unitree可用（至少可以仿真）
if MUJOCO_AVAILABLE:
    UNITREE_AVAILABLE = True

if not UNITREE_AVAILABLE:
    log_warning("⚠️ Unitree/MuJoCo 导入失败，启用模拟模式")


class UnitreeModel(BaseRobotModel):
    """Unitree机器人模型类"""
    
    def __init__(self, robot_type: str = "go2"):
        """
        初始化Unitree机器人模型
        
        Args:
            robot_type: 机器人型号 (go2, a1, b1)
        """
        super().__init__(robot_type)
        self.config = ConfigManager()
        self.is_available = UNITREE_AVAILABLE
        self.mujoco_available = MUJOCO_AVAILABLE
        
        # MuJoCo模型相关
        self.model = None
        self.data = None
        self.viewer = None
        
        # 仿真控制
        self.running = False
        self.stop_requested = False
        
        # 注册可用动作
        self._register_actions()
        
        log_info(f"UnitreeModel 初始化: robot_type={robot_type}, available={self.is_available}")
    
    def _register_actions(self):
        """注册可用动作"""
        self.register_action(
            "lift_right_leg",
            self._lift_right_leg_action,
            "抬起右前腿",
            {}
        )
        
        self.register_action(
            "stand",
            self._stand_action,
            "站立姿势",
            {}
        )
        
        self.register_action(
            "sit",
            self._sit_action,
            "坐下姿势",
            {}
        )
        
        self.register_action(
            "walk",
            self._walk_action,
            "行走",
            {}
        )
        
        self.register_action(
            "stop",
            self._stop_action,
            "停止运动",
            {}
        )
    
    def initialize(self) -> bool:
        """初始化机器人模型"""
        if not self.mujoco_available:
            log_warning("MuJoCo不可用，使用模拟模式")
            return True  # 模拟模式下也返回True
        
        try:
            # 设置MuJoCo环境变量
            gl_backend = self.config.get('MUJOCO', 'gl_backend', fallback='glfw')
            os.environ['MUJOCO_GL'] = gl_backend
            log_info(f"MuJoCo GL后端: {gl_backend}")
            return True
        except Exception as e:
            log_error(f"初始化失败: {e}")
            return False
    
    def load_model(self) -> bool:
        """加载MuJoCo机器人模型文件"""
        if not self.mujoco_available:
            log_warning("MuJoCo不可用，跳过模型加载")
            return True
        
        try:
            # 查找模型文件
            model_file = self._find_model_file()
            
            if model_file is None:
                log_error(f"未找到 {self.robot_type} 的模型文件")
                return False
            
            # 加载模型
            log_info(f"加载模型: {model_file}")
            self.model = mujoco.MjModel.from_xml_path(str(model_file))
            self.data = mujoco.MjData(self.model)
            
            # 打印模型信息
            log_info(f"模型加载成功:")
            log_info(f"  - 位置变量个数 nq = {self.model.nq}")
            log_info(f"  - 速度变量个数 nv = {self.model.nv}")
            log_info(f"  - 执行器个数 nu = {self.model.nu}")
            log_info(f"  - 关节数量 njnt = {self.model.njnt}")
            
            return True
            
        except Exception as e:
            log_error(f"加载模型失败: {e}")
            return False
    
    def _find_model_file(self) -> Optional[Path]:
        """查找机器人模型文件"""
        unitree_robots_path = self.config.get_path('unitree_robots')
        
        # 可能的模型文件路径
        possible_paths = [
            unitree_robots_path / self.robot_type / "scene.xml",
            unitree_robots_path / "data" / self.robot_type / "scene.xml",
        ]
        
        for path in possible_paths:
            if path.exists():
                log_info(f"✅ 找到模型文件: {path}")
                return path
        
        # 调试：列出目录结构
        self._debug_directory_structure()
        return None
    
    def _debug_directory_structure(self):
        """调试目录结构"""
        unitree_robots_path = self.config.get_path('unitree_robots')
        logger_debug(f"🔍 调试目录结构: {unitree_robots_path}")
        
        if unitree_robots_path.exists():
            for item in unitree_robots_path.iterdir():
                if item.is_dir():
                    logger_debug(f"📁 {item.name}")
        else:
            log_warning(f"⚠️ 目录不存在: {unitree_robots_path}")
    
    def run_action(self, action_name: str, **kwargs) -> bool:
        """
        执行指定动作
        
        Args:
            action_name: 动作名称
            **kwargs: 动作参数
        
        Returns:
            是否执行成功
        """
        if not self.is_available:
            log_warning(f"Unitree不可用，模拟执行动作: {action_name}")
            time.sleep(2)  # 模拟延迟
            return True
        
        action_info = self.get_action_info(action_name)
        if not action_info:
            log_error(f"未找到动作: {action_name}")
            return False
        
        try:
            # 初始化（如果还未初始化）
            if not self.initialize():
                return False
            
            # 加载模型（如果还未加载）
            if self.model is None:
                if not self.load_model():
                    return False
            
            # 执行动作
            action_func = action_info['function']
            return action_func(**kwargs)
            
        except Exception as e:
            log_error(f"执行动作失败: {action_name}, 错误: {e}")
            return False
    
    def get_available_actions(self) -> List[str]:
        """获取可用动作列表"""
        return list(self._actions.keys())
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据"""
        if not self.mujoco_available or self.data is None:
            return {
                'simulated': True,
                'message': '模拟模式，无真实传感器数据'
            }
        
        return {
            'qpos': self.data.qpos.tolist() if hasattr(self.data, 'qpos') else [],
            'qvel': self.data.qvel.tolist() if hasattr(self.data, 'qvel') else [],
            'time': self.data.time if hasattr(self.data, 'time') else 0.0
        }
    
    def stop(self):
        """停止机器人运行"""
        self.running = False
        self.stop_requested = True
        log_info("停止请求已发送")
    
    def _set_initial_pose(self):
        """设置初始姿势"""
        if not self.mujoco_available or self.model is None:
            return
        
        log_info("设置初始姿势...")
        
        # 重置所有状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置站立姿势
        if "go2" in self.robot_type:
            if self.model.nq >= 19:
                # 身体位置和朝向
                self.data.qpos[0] = 0.0  # x
                self.data.qpos[1] = 0.0  # y
                self.data.qpos[2] = 0.3  # z (高度)
                
                # 四元数 (姿态)
                self.data.qpos[3] = 1.0  # w
                self.data.qpos[4] = 0.0  # x
                self.data.qpos[5] = 0.0  # y
                self.data.qpos[6] = 0.0  # z
                
                # 关节角度 - 站立姿势
                if self.model.nu >= 12:
                    # 右前腿
                    self.data.qpos[7] = 0.0    # 髋外展
                    self.data.qpos[8] = 0.67   # 髋屈曲
                    self.data.qpos[9] = -1.3   # 膝关节
                    
                    # 左前腿
                    self.data.qpos[10] = 0.0
                    self.data.qpos[11] = 0.67
                    self.data.qpos[12] = -1.3
                    
                    # 右后腿
                    self.data.qpos[13] = 0.0
                    self.data.qpos[14] = 0.67
                    self.data.qpos[15] = -1.3
                    
                    # 左后腿
                    self.data.qpos[16] = 0.0
                    self.data.qpos[17] = 0.67
                    self.data.qpos[18] = -1.3
                    
                    # 设置对应的控制输入
                    self.data.ctrl[0] = 0.0   # 右前髋外展
                    self.data.ctrl[1] = 0.67  # 右前髋屈曲
                    self.data.ctrl[2] = -1.3  # 右前膝
                    
                    self.data.ctrl[3] = 0.0   # 左前髋外展
                    self.data.ctrl[4] = 0.67  # 左前髋屈曲
                    self.data.ctrl[5] = -1.3  # 左前膝
                    
                    self.data.ctrl[6] = 0.0   # 右后髋外展
                    self.data.ctrl[7] = 0.67  # 右后髋屈曲
                    self.data.ctrl[8] = -1.3  # 右后膝
                    
                    self.data.ctrl[9] = 0.0   # 左后髋外展
                    self.data.ctrl[10] = 0.67 # 左后髋屈曲
                    self.data.ctrl[11] = -1.3 # 左后膝
                    
                    log_info(f"设置 Go2 标准姿势 (nu={self.model.nu})")
        
        # 应用初始状态
        mujoco.mj_forward(self.model, self.data)
    
    def _lift_right_leg_action(self, **kwargs) -> bool:
        """抬起右前腿动作"""
        if not self.mujoco_available:
            log_warning("模拟模式：抬右腿动作")
            return True
        
        try:
            log_info("执行抬右腿动作...")
            
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置初始姿势
                self._set_initial_pose()
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                
                # 执行抬腿动作
                self._lift_right_leg_simulation(viewer)
                
                # 保持窗口打开一段时间
                keep_time = self.config.get_float('MUJOCO', 'keep_window_time', fallback=5.0)
                keep_steps = int(keep_time / self.model.opt.timestep)
                
                for i in range(keep_steps):
                    if self.stop_requested:
                        break
                    mujoco.mj_forward(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
            
            log_info("抬右腿动作完成")
            return True
            
        except Exception as e:
            log_error(f"抬右腿动作失败: {e}")
            return False
    
    def _lift_right_leg_simulation(self, viewer):
        """抬右腿仿真过程"""
        self.running = True
        self.stop_requested = False
        
        # 安全检查
        if self.model.nu < 12:
            log_warning(f"模型控制输入不足 (nu={self.model.nu})，无法执行完整动作")
            return
        
        # 第一阶段：确保站立姿势
        log_info("第一步：确保站立姿势...")
        stand_duration = 1.0
        stand_steps = int(stand_duration / self.model.opt.timestep)
        stand_hip_angle = 0.67
        
        for step in range(stand_steps):
            if self.stop_requested:
                return
            
            # 设置所有腿的站立角度
            if self.model.nu > 1:
                self.data.ctrl[1] = stand_hip_angle  # 右前腿髋屈曲
            if self.model.nu > 4:
                self.data.ctrl[4] = stand_hip_angle  # 左前腿髋屈曲
            if self.model.nu > 7:
                self.data.ctrl[7] = stand_hip_angle  # 右后腿髋屈曲
            if self.model.nu > 10:
                self.data.ctrl[10] = stand_hip_angle  # 左后腿髋屈曲
            
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)
        
        log_info("站立姿势就绪，开始抬右腿...")
        
        # 第二阶段：抬起右前腿
        lift_duration = 1.5
        hold_duration = 1.0
        lower_duration = 1.5
        
        total_steps = int((lift_duration + hold_duration + lower_duration) / self.model.opt.timestep)
        
        target_hip_angle = 1.2  # 抬起时的角度
        original_hip_angle = 0.67  # 站立时的角度
        
        log_info(f"抬右前腿: {original_hip_angle} -> {target_hip_angle}")
        
        for step_count in range(total_steps):
            if self.stop_requested:
                return
            
            if step_count < lift_duration / self.model.opt.timestep:
                # 抬腿阶段
                progress = step_count / (lift_duration / self.model.opt.timestep)
                current_angle = original_hip_angle + (target_hip_angle - original_hip_angle) * progress
                if self.model.nu > 1:
                    self.data.ctrl[1] = current_angle
                
            elif step_count < (lift_duration + hold_duration) / self.model.opt.timestep:
                # 保持阶段
                if self.model.nu > 1:
                    self.data.ctrl[1] = target_hip_angle
                
            else:
                # 放下阶段
                progress = (step_count - (lift_duration + hold_duration) / self.model.opt.timestep) / (lower_duration / self.model.opt.timestep)
                current_angle = target_hip_angle - (target_hip_angle - original_hip_angle) * progress
                if self.model.nu > 1:
                    self.data.ctrl[1] = current_angle
            
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)
        
        # 第三阶段：返回站立姿势
        log_info("返回站立姿势...")
        for step in range(min(stand_steps, 50)):
            if self.stop_requested:
                return
            
            if self.model.nu > 1:
                self.data.ctrl[1] = original_hip_angle
            
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)
        
        log_info("动作执行完毕")
    
    def _stand_action(self, **kwargs) -> bool:
        """站立姿势动作"""
        if not self.mujoco_available:
            log_warning("模拟模式：站立动作")
            return True
        
        try:
            log_info("执行站立动作...")
            
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._set_initial_pose()
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                
                # 保持站立姿势
                duration = kwargs.get('duration', 5.0)
                steps = int(duration / self.model.opt.timestep)
                
                for _ in range(steps):
                    if self.stop_requested:
                        break
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)
            
            log_info("站立动作完成")
            return True
            
        except Exception as e:
            log_error(f"站立动作失败: {e}")
            return False
    
    def _sit_action(self, **kwargs) -> bool:
        """坐下姿势动作"""
        log_warning("坐下动作尚未实现")
        return True
    
    def _walk_action(self, **kwargs) -> bool:
        """行走动作"""
        log_warning("行走动作尚未实现")
        return True
    
    def _stop_action(self, **kwargs) -> bool:
        """停止运动"""
        self.stop()
        return True
