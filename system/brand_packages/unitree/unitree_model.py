#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unitree robot model with MuJoCo simulation support."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import math

from models.base import BaseRobotModel
from bin.core.logger import log_info, log_error, log_debug, log_warning
from bin.core.config_manager import ConfigManager

# Unitree / MuJoCo import bootstrap.
UNITREE_AVAILABLE = False
MUJOCO_AVAILABLE = False

try:
    # Load path configuration
    config = ConfigManager()
    project_root = config.get_path('project_root')
    
    # Add Unitree SDK paths
    unitree_sdk_path = config.get_path('unitree_sdk')
    unitree_mujoco_path = config.get_path('unitree_mujoco')
    
    possible_paths = [unitree_sdk_path, unitree_mujoco_path]
    
    added_paths = []
    for path in possible_paths:
        if path.exists():
            sys.path.insert(0, str(path))
            added_paths.append(str(path))
            log_info(f"Added path: {path}")
    
    if added_paths:
        log_info(f"Added {len(added_paths)} paths to sys.path")
    
    # Import MuJoCo
    try:
        import mujoco
        log_info(f"mujoco version: {mujoco.__version__}")
        MUJOCO_AVAILABLE = True
        
        try:
            import mujoco.viewer
            log_info("mujoco.viewer imported successfully")
        except ImportError as e:
            log_warning(f"mujoco.viewer import failed: {e}")
            MUJOCO_AVAILABLE = False
            
    except ImportError as e:
        log_error(f"Failed to import mujoco: {e}")
        MUJOCO_AVAILABLE = False
    
    # Import Unitree SDK
    try:
        import importlib.util
        
        sdk_spec = importlib.util.find_spec("unitree_sdk2py")
        if sdk_spec is None:
            log_warning("unitree_sdk2py module not found")
        else:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
            log_info("Unitree SDK imported successfully")
            UNITREE_AVAILABLE = True
    except ImportError as e:
        log_warning(f"Failed to import Unitree SDK: {e}")

except Exception as e:
    log_error(f"Error occurred during import: {e}")
    UNITREE_AVAILABLE = False
    MUJOCO_AVAILABLE = False

# If MuJoCo is available, keep simulation path available.
if MUJOCO_AVAILABLE:
    UNITREE_AVAILABLE = True

if not UNITREE_AVAILABLE:
    log_warning("Unitree/MuJoCo import failed; simulation mode enabled")


class UnitreeModel(BaseRobotModel):
    """Unitree机器人模型类"""

    # Class-level persistent viewer (shared across instances)
    _persistent_viewer = None
    _viewer_model = None
    _viewer_data = None

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
        self.pause_requested = False

        # 注册可用动作
        self._register_actions()

        # SDK clients (lazy init)
        self._sport_client = None
        self._sdk_channel_inited = False

        log_info(f"UnitreeModel initialized: robot_type={robot_type}, available={self.is_available}")
    
    def _register_actions(self):
        """注册可用动作"""
        self.register_action(
            "lift_right_leg",
            self._lift_right_leg_action,
            "抬起右前腿",
            {}
        )

        self.register_action(
            "lift_left_leg",
            self._lift_left_leg_action,
            "抬起左前腿",
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
            log_warning("MuJoCo unavailable; using simulation mode")
            return True  # 模拟模式下也返回True
        
        try:
            # 设置MuJoCo环境变量
            gl_backend = self.config.get('MUJOCO', 'gl_backend', fallback='glfw')
            os.environ['MUJOCO_GL'] = gl_backend
            log_info(f"MuJoCo GL backend: {gl_backend}")
            return True
        except Exception as e:
            log_error(f"Initialization failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """加载MuJoCo机器人模型文件"""
        if not self.mujoco_available:
            log_warning("MuJoCo unavailable; skipping model loading")
            return True
        
        try:
            # 查找模型文件
            model_file = self._find_model_file()
            
            if model_file is None:
                log_error(f"Model file for {self.robot_type} not found")
                return False
            
            # 加载模型
            log_info(f"Loading model: {model_file}")
            self.model = mujoco.MjModel.from_xml_path(str(model_file))
            self.data = mujoco.MjData(self.model)
            
            # 打印模型信息
            log_info(f"Model loaded successfully:")
            log_info(f"  - Position variables (nq) = {self.model.nq}")
            log_info(f"  - Velocity variables (nv) = {self.model.nv}")
            log_info(f"  - Actuators (nu) = {self.model.nu}")
            log_info(f"  - Joints (njnt) = {self.model.njnt}")
            
            return True
            
        except Exception as e:
            log_error(f"Model loading failed: {e}")
            return False
    
    def _find_model_file(self) -> Optional[Path]:
        """查找机器人模型文件"""
        unitree_robots_path = self.config.get_path('unitree_robots')
        project_root = self.config.get_path('project_root')

        # Robust fallbacks in case PATH.unitree_robots is missing/misconfigured.
        # Include multiple models/<brand> layouts used by historical builds.
        candidate_roots = [
            unitree_robots_path,
            project_root / "models" / "Unitree",
            project_root / "models" / "unitree",
            project_root / "models" / "Unitree" / "unitree_robots",
            project_root / "models" / "unitree" / "unitree_robots",
            project_root / "models" / "Unitree" / "unitree_mujoco" / "unitree_robots",
            project_root / "models" / "unitree" / "unitree_mujoco" / "unitree_robots",
        ]

        possible_paths = []
        for root in candidate_roots:
            possible_paths.append(root / self.robot_type / "scene.xml")
            possible_paths.append(root / "data" / self.robot_type / "scene.xml")
            possible_paths.append(root / "unitree_robots" / self.robot_type / "scene.xml")
            possible_paths.append(root / "unitree_robots" / "data" / self.robot_type / "scene.xml")
            possible_paths.append(root / f"{self.robot_type}.xml")
            possible_paths.append(root / self.robot_type / f"{self.robot_type}.xml")
        
        for path in possible_paths:
            if path.exists():
                log_info(f"Model file found: {path}")
                return path

        # Last-resort recursive search under known model roots.
        search_roots = [
            project_root / "models" / "Unitree",
            project_root / "models" / "unitree",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for path in root.rglob("scene.xml"):
                    path_str = str(path).replace("\\", "/").lower()
                    token = f"/{self.robot_type}/"
                    if token in path_str or f"{self.robot_type}.xml" in path_str:
                        log_info(f"Model file found via recursive search: {path}")
                        return path
            except Exception as e:
                log_warning(f"Recursive model search failed under {root}: {e}")
        
        # Debug fallback: list candidate directories for diagnostics.
        self._debug_directory_structure(candidate_roots)
        return None
    
    def _debug_directory_structure(self, roots=None):
        """Debug helper: print directory structure for model discovery."""
        if roots is None:
            roots = [self.config.get_path('unitree_robots')]
        for root in roots:
            log_debug(f"Debug directory structure: {root}")
            if root.exists():
                for item in root.iterdir():
                    if item.is_dir():
                        log_debug(f"  [dir] {item.name}")
            else:
                log_warning(f"Directory does not exist: {root}")
    
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
            log_warning(f"Unitree unavailable; simulating action: {action_name}")
            time.sleep(2)  # 模拟延迟
            return True
        
        action_info = self.get_action_info(action_name)
        if not action_info:
            log_error(f"Action not found: {action_name}")
            return False

        # Reset control flags for a fresh action run.
        self.stop_requested = False
        self.pause_requested = False
        self.running = True

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
            log_error(f"Action execution failed: {action_name}, error: {e}")
            return False
        finally:
            self.running = False
    
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
        self.pause_requested = False
        log_info("Stop request sent")

    def pause(self):
        """Pause current simulation loop (best effort)."""
        self.pause_requested = True
        log_info("Pause request sent")

    def resume(self):
        """Resume a paused simulation loop."""
        self.pause_requested = False
        log_info("Resume request sent")

    def cancel_action(self):
        """RuntimeEngine-compatible cancel hook."""
        self.stop()

    def _should_stop(self, viewer=None) -> bool:
        """
        检查是否应该停止仿真

        Returns:
            True if should stop (stop requested or viewer closed)
        """
        if self.stop_requested:
            return True
        # Cooperative pause: block loop progression until resumed or stopped.
        while self.pause_requested and not self.stop_requested:
            if viewer is not None:
                try:
                    if not viewer.is_running():
                        self.stop_requested = True
                        log_info("Viewer closed while paused, stopping simulation")
                        return True
                except Exception:
                    self.stop_requested = True
                    return True
            time.sleep(0.02)
        if self.stop_requested:
            return True
        # 检查 viewer 是否仍在运行
        if viewer is not None:
            try:
                if not viewer.is_running():
                    self.stop_requested = True
                    log_info("Viewer closed, stopping simulation")
                    return True
            except Exception:
                self.stop_requested = True
                return True
        return False

    def ensure_viewer(self) -> bool:
        """
        Ensure MuJoCo viewer is open. Creates new one if needed, reuses if exists.

        Returns:
            True if viewer is ready, False otherwise
        """
        if not self.mujoco_available:
            return False

        # A fresh ensure call represents a new action/reset attempt.
        # Clear stale stop/pause flags from a previous cancelled run.
        self.stop_requested = False
        self.pause_requested = False

        # Check if model is loaded
        if self.model is None:
            if not self.load_model():
                return False

        # Check if persistent viewer exists and is valid
        if UnitreeModel._persistent_viewer is not None:
            try:
                # Check if viewer is still running
                if not UnitreeModel._persistent_viewer.is_running():
                    raise Exception("Viewer not running")
                # Try to sync - if it fails, viewer is closed
                UnitreeModel._persistent_viewer.sync()
                # Viewer exists and is valid, update reference
                self.viewer = UnitreeModel._persistent_viewer
                # Update model/data if changed
                if UnitreeModel._viewer_model != self.model:
                    log_info("Model changed, recreating viewer...")
                    self.close_viewer()
                else:
                    log_info("Reusing existing viewer")
                    return True
            except Exception as e:
                # Viewer was closed, need to recreate
                log_info(f"Viewer was closed ({e}), recreating...")
                UnitreeModel._persistent_viewer = None
                UnitreeModel._viewer_model = None
                UnitreeModel._viewer_data = None

        # Create new persistent viewer
        try:
            log_info("Creating new MuJoCo viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            UnitreeModel._persistent_viewer = self.viewer
            UnitreeModel._viewer_model = self.model
            UnitreeModel._viewer_data = self.data

            # Set initial pose
            self._set_initial_pose()
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

            log_info("Viewer created successfully")
            return True

        except Exception as e:
            log_error(f"Failed to create viewer: {e}")
            return False

    def close_viewer(self):
        """Close the persistent viewer."""
        if UnitreeModel._persistent_viewer is not None:
            try:
                UnitreeModel._persistent_viewer.close()
            except Exception:
                pass
            UnitreeModel._persistent_viewer = None
            UnitreeModel._viewer_model = None
            UnitreeModel._viewer_data = None
            self.viewer = None
            log_info("Viewer closed")

    def is_viewer_open(self) -> bool:
        """Check if viewer is currently open."""
        if UnitreeModel._persistent_viewer is None:
            return False
        try:
            UnitreeModel._persistent_viewer.sync()
            return True
        except Exception:
            return False

    def reset_simulation(self) -> bool:
        """Reset MuJoCo state and ensure the viewer is ready."""
        if not self.mujoco_available:
            return True

        # Reset stale control flags so a prior quit/stop does not block restart.
        self.stop_requested = False
        self.pause_requested = False
        if not self.ensure_viewer():
            return False

        try:
            self._set_initial_pose()
            mujoco.mj_forward(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
            return True
        except Exception as e:
            log_error(f"Reset simulation failed: {e}")
            return False
    
    def _set_initial_pose(self):
        """设置初始姿势"""
        if not self.mujoco_available or self.model is None:
            return
        
        log_info("Setting initial posture...")
        
        # 重置所有状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置站立姿势
        if "go2" in self.robot_type:
            if self.model.nq >= 19:
                # 身体位置和朝向
                self.data.qpos[0] = 0.0  # x
                self.data.qpos[1] = 0.0  # y
                self.data.qpos[2] = 0.445  # z (高度) — 与 go2.xml base_link 定义一致
                
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
                    
                    log_info(f"Set Go2 default posture (nu={self.model.nu})")
        
        # 应用初始状态
        mujoco.mj_forward(self.model, self.data)
    
    def _lift_right_leg_action(self, **kwargs) -> bool:
        """抬起右前腿动作"""
        if not self.mujoco_available:
            log_warning("Simulation mode: lift right leg action")
            return True

        try:
            log_info("Executing right-leg lift action...")

            # Use persistent viewer
            if not self.ensure_viewer():
                log_error("Unable to create/get viewer")
                return False

            viewer = self.viewer

            # Reset to initial pose before action
            self._set_initial_pose()
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()

            # 执行抬腿动作
            ok = self._lift_right_leg_simulation(viewer)
            if ok:
                log_info("Right-leg lift action completed (viewer remains open)")
            else:
                log_warning("Right-leg lift action interrupted")
            return ok

        except Exception as e:
            log_error(f"Right-leg lift action failed: {e}")
            return False
    
    def _lift_right_leg_simulation(self, viewer) -> bool:
        """抬右腿仿真过程"""
        self.running = True
        self.stop_requested = False
        
        # 安全检查
        if self.model.nu < 12:
            log_warning(f"Insufficient model control inputs (nu={self.model.nu}); cannot execute full action")
            return False
        
        # 如果是 Go2 模型，使用 PD 位置控制保持稳定站姿并抬腿
        if self.robot_type.lower() == "go2":
            return self._lift_right_leg_simulation_go2(viewer)

        # 兜底逻辑（非 Go2）
        # 第一阶段：确保站立姿势
        log_info("Step 1: ensure standing posture...")
        stand_duration = 1.0
        stand_steps = int(stand_duration / self.model.opt.timestep)
        stand_hip_angle = 0.67
        
        for step in range(stand_steps):
            if self._should_stop(viewer):
                return False

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
        
        log_info("Standing posture ready, starting right-leg lift...")
        
        # 第二阶段：抬起右前腿
        lift_duration = 1.5
        hold_duration = 1.0
        lower_duration = 1.5
        
        total_steps = int((lift_duration + hold_duration + lower_duration) / self.model.opt.timestep)
        
        target_hip_angle = 1.2  # 抬起时的角度
        original_hip_angle = 0.67  # 站立时的角度
        
        log_info(f"Lift front-right leg: {original_hip_angle} -> {target_hip_angle}")
        
        for step_count in range(total_steps):
            if self._should_stop(viewer):
                return False
            
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
        log_info("Returning to standing posture...")
        for step in range(min(stand_steps, 50)):
            if self._should_stop(viewer):
                return False
            
            if self.model.nu > 1:
                self.data.ctrl[1] = original_hip_angle
            
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)
        
        log_info("Action completed")
        return True

    def _lift_right_leg_simulation_go2(self, viewer) -> bool:
        """Go2 抬右腿（PD 位置控制 + 支撑侧重心偏移）"""
        stand_targets = self._get_go2_stand_targets()

        # 站立稳态
        log_info("Step 1: ensure standing posture (Go2 PD)...")
        stand_duration = 1.0
        stand_steps = int(stand_duration / self.model.opt.timestep)
        for _ in range(stand_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(stand_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        # 预下蹲：降低重心
        log_info("Step 2: crouch to lower center of mass (Go2)...")
        crouch_targets = dict(stand_targets)
        crouch_targets["FR_thigh_joint"] = 0.85
        crouch_targets["FR_calf_joint"] = -1.75
        crouch_targets["FL_thigh_joint"] = 0.85
        crouch_targets["FL_calf_joint"] = -1.75
        crouch_targets["RR_thigh_joint"] = 0.85
        crouch_targets["RR_calf_joint"] = -1.75
        crouch_targets["RL_thigh_joint"] = 0.85
        crouch_targets["RL_calf_joint"] = -1.75
        crouch_duration = 0.8
        crouch_steps = int(crouch_duration / self.model.opt.timestep)
        for _ in range(crouch_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(crouch_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        # 支撑相：轻微向左侧偏重心，避免抬腿侧翻
        # 注意：不同模型关节正负方向可能相反，这里自动选择更稳定的偏移方向
        log_info("Step 3: shift center of mass left (Go2)...")
        support_sign = self._choose_support_abd_sign(crouch_targets, viewer)
        support_targets = dict(crouch_targets)
        # 保持抬腿侧(右前)髋外展不动，主要通过支撑腿调整重心
        support_targets["FR_hip_joint"] = stand_targets["FR_hip_joint"]
        support_targets["RR_hip_joint"] = 0.12 * support_sign
        support_targets["FL_hip_joint"] = -0.28 * support_sign
        support_targets["RL_hip_joint"] = -0.22 * support_sign
        # 让左侧更稳定一些，右后略支撑
        support_targets["FL_thigh_joint"] = 0.95
        support_targets["FL_calf_joint"] = -1.90
        support_targets["RL_thigh_joint"] = 0.95
        support_targets["RL_calf_joint"] = -1.90
        support_targets["RR_thigh_joint"] = 0.90
        support_targets["RR_calf_joint"] = -1.85

        support_duration = 1.2
        support_steps = int(support_duration / self.model.opt.timestep)
        for _ in range(support_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(support_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        log_info("Center of mass stabilized, starting front-right leg lift (Go2)...")
        lift_duration = 1.8
        hold_duration = 1.0
        lower_duration = 1.6

        total_steps = int((lift_duration + hold_duration + lower_duration) / self.model.opt.timestep)

        # 目标角度
        original_thigh = support_targets["FR_thigh_joint"]
        original_calf = support_targets["FR_calf_joint"]
        original_abd = support_targets["FR_hip_joint"]
        target_thigh = 1.45
        target_calf = -2.45
        target_abd = original_abd

        for step_count in range(total_steps):
            if self._should_stop(viewer):
                return False

            targets = dict(support_targets)
            if step_count < lift_duration / self.model.opt.timestep:
                progress = step_count / (lift_duration / self.model.opt.timestep)
                targets["FR_thigh_joint"] = original_thigh + (target_thigh - original_thigh) * progress
                targets["FR_calf_joint"] = original_calf + (target_calf - original_calf) * progress
                targets["FR_hip_joint"] = original_abd
            elif step_count < (lift_duration + hold_duration) / self.model.opt.timestep:
                targets["FR_thigh_joint"] = target_thigh
                targets["FR_calf_joint"] = target_calf
                targets["FR_hip_joint"] = original_abd
            else:
                progress = (step_count - (lift_duration + hold_duration) / self.model.opt.timestep) / (
                    lower_duration / self.model.opt.timestep
                )
                targets["FR_thigh_joint"] = target_thigh - (target_thigh - original_thigh) * progress
                targets["FR_calf_joint"] = target_calf - (target_calf - original_calf) * progress
                targets["FR_hip_joint"] = original_abd

            gain_scale = {
                "FR_hip_joint": 1.0,
                "FR_thigh_joint": 1.2,
                "FR_calf_joint": 1.2,
                "FL_hip_joint": 1.6,
                "FL_thigh_joint": 1.6,
                "FL_calf_joint": 1.6,
                "RR_hip_joint": 1.4,
                "RR_thigh_joint": 1.4,
                "RR_calf_joint": 1.4,
                "RL_hip_joint": 1.6,
                "RL_thigh_joint": 1.6,
                "RL_calf_joint": 1.6,
            }
            self._apply_pd_control(targets, gain_scale=gain_scale)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        log_info("Step 3: return to standing posture (Go2)...")
        return_steps = int(0.8 / self.model.opt.timestep)
        for _ in range(return_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(stand_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        log_info("Action completed (Go2)")
        return True

    # ------------------------------------------------------------------
    # 左前腿抬腿动作（镜像右前腿）
    # ------------------------------------------------------------------

    def _lift_left_leg_action(self, **kwargs) -> bool:
        """抬起左前腿动作"""
        if not self.mujoco_available:
            log_warning("Simulation mode: lift left leg action")
            return True

        try:
            log_info("Executing left-leg lift action...")

            if not self.ensure_viewer():
                log_error("Unable to create/get viewer")
                return False

            viewer = self.viewer

            self._set_initial_pose()
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()

            ok = self._lift_left_leg_simulation(viewer)
            if ok:
                log_info("Left-leg lift action completed (viewer remains open)")
            else:
                log_warning("Left-leg lift action interrupted")
            return ok

        except Exception as e:
            log_error(f"Left-leg lift action failed: {e}")
            return False

    def _lift_left_leg_simulation(self, viewer) -> bool:
        """抬左腿仿真过程"""
        self.running = True
        self.stop_requested = False

        if self.model.nu < 12:
            log_warning(f"Insufficient model control inputs (nu={self.model.nu}); cannot execute full action")
            return False

        if self.robot_type.lower() == "go2":
            return self._lift_left_leg_simulation_go2(viewer)

        # 兜底逻辑（非 Go2）——与右腿对称，使用 ctrl[4] (左前髋屈曲)
        stand_duration = 1.0
        stand_steps = int(stand_duration / self.model.opt.timestep)
        stand_hip_angle = 0.67
        for step in range(stand_steps):
            if self._should_stop(viewer):
                return False
            for idx in [1, 4, 7, 10]:
                if self.model.nu > idx:
                    self.data.ctrl[idx] = stand_hip_angle
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        lift_duration, hold_duration, lower_duration = 1.5, 1.0, 1.5
        total_steps = int((lift_duration + hold_duration + lower_duration) / self.model.opt.timestep)
        target_hip_angle = 1.2
        for step_count in range(total_steps):
            if self._should_stop(viewer):
                return False
            if step_count < lift_duration / self.model.opt.timestep:
                progress = step_count / (lift_duration / self.model.opt.timestep)
                angle = stand_hip_angle + (target_hip_angle - stand_hip_angle) * progress
            elif step_count < (lift_duration + hold_duration) / self.model.opt.timestep:
                angle = target_hip_angle
            else:
                progress = (step_count - (lift_duration + hold_duration) / self.model.opt.timestep) / (lower_duration / self.model.opt.timestep)
                angle = target_hip_angle - (target_hip_angle - stand_hip_angle) * progress
            if self.model.nu > 4:
                self.data.ctrl[4] = angle  # 左前髋屈曲
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        for _ in range(min(stand_steps, 50)):
            if self._should_stop(viewer):
                return False
            if self.model.nu > 4:
                self.data.ctrl[4] = stand_hip_angle
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        return True

    def _lift_left_leg_simulation_go2(self, viewer) -> bool:
        """Go2 抬左前腿（PD 位置控制，镜像右腿逻辑）"""
        stand_targets = self._get_go2_stand_targets()

        # 站立稳态
        log_info("Step 1: ensure standing posture (Go2 PD)...")
        stand_steps = int(1.0 / self.model.opt.timestep)
        for _ in range(stand_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(stand_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        # 预下蹲
        log_info("Step 2: crouch to lower center of mass (Go2)...")
        crouch_targets = dict(stand_targets)
        for leg in ("FR", "FL", "RR", "RL"):
            crouch_targets[f"{leg}_thigh_joint"] = 0.85
            crouch_targets[f"{leg}_calf_joint"] = -1.75
        crouch_steps = int(0.8 / self.model.opt.timestep)
        for _ in range(crouch_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(crouch_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        # 重心右移（抬左腿时向右侧支撑）
        log_info("Step 3: shift center of mass right (Go2)...")
        support_sign = self._choose_support_abd_sign(crouch_targets, viewer)
        support_targets = dict(crouch_targets)
        support_targets["FL_hip_joint"] = stand_targets["FL_hip_joint"]   # 抬腿侧保持
        support_targets["RL_hip_joint"] = -0.12 * support_sign
        support_targets["FR_hip_joint"] = 0.28 * support_sign             # 支撑侧右前
        support_targets["RR_hip_joint"] = 0.22 * support_sign
        support_targets["FR_thigh_joint"] = 0.95
        support_targets["FR_calf_joint"] = -1.90
        support_targets["RR_thigh_joint"] = 0.90
        support_targets["RR_calf_joint"] = -1.85
        support_targets["RL_thigh_joint"] = 0.95
        support_targets["RL_calf_joint"] = -1.90

        support_steps = int(1.2 / self.model.opt.timestep)
        for _ in range(support_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(support_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        # 抬左前腿
        log_info("Center of mass stabilized, starting front-left leg lift (Go2)...")
        lift_duration, hold_duration, lower_duration = 1.8, 1.0, 1.6
        total_steps = int((lift_duration + hold_duration + lower_duration) / self.model.opt.timestep)

        original_thigh = support_targets["FL_thigh_joint"]
        original_calf = support_targets["FL_calf_joint"]
        original_abd = support_targets["FL_hip_joint"]
        target_thigh = 1.45
        target_calf = -2.45

        for step_count in range(total_steps):
            if self._should_stop(viewer):
                return False

            targets = dict(support_targets)
            if step_count < lift_duration / self.model.opt.timestep:
                progress = step_count / (lift_duration / self.model.opt.timestep)
                targets["FL_thigh_joint"] = original_thigh + (target_thigh - original_thigh) * progress
                targets["FL_calf_joint"] = original_calf + (target_calf - original_calf) * progress
                targets["FL_hip_joint"] = original_abd
            elif step_count < (lift_duration + hold_duration) / self.model.opt.timestep:
                targets["FL_thigh_joint"] = target_thigh
                targets["FL_calf_joint"] = target_calf
                targets["FL_hip_joint"] = original_abd
            else:
                progress = (step_count - (lift_duration + hold_duration) / self.model.opt.timestep) / (lower_duration / self.model.opt.timestep)
                targets["FL_thigh_joint"] = target_thigh - (target_thigh - original_thigh) * progress
                targets["FL_calf_joint"] = target_calf - (target_calf - original_calf) * progress
                targets["FL_hip_joint"] = original_abd

            gain_scale = {
                "FL_hip_joint": 1.0, "FL_thigh_joint": 1.2, "FL_calf_joint": 1.2,
                "FR_hip_joint": 1.6, "FR_thigh_joint": 1.6, "FR_calf_joint": 1.6,
                "RL_hip_joint": 1.4, "RL_thigh_joint": 1.4, "RL_calf_joint": 1.4,
                "RR_hip_joint": 1.6, "RR_thigh_joint": 1.6, "RR_calf_joint": 1.6,
            }
            self._apply_pd_control(targets, gain_scale=gain_scale)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        log_info("Step 4: return to standing posture (Go2)...")
        return_steps = int(0.8 / self.model.opt.timestep)
        for _ in range(return_steps):
            if self._should_stop(viewer):
                return False
            self._apply_pd_control(stand_targets)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(self.model.opt.timestep)

        log_info("Left-leg action completed (Go2)")
        return True

    def _get_go2_stand_targets(self) -> Dict[str, float]:
        """Go2 站立目标角度"""
        return {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.67,
            "FR_calf_joint": -1.3,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.67,
            "FL_calf_joint": -1.3,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.67,
            "RR_calf_joint": -1.3,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.67,
            "RL_calf_joint": -1.3,
        }

    def _choose_support_abd_sign(self, stand_targets: Dict[str, float], viewer) -> int:
        """自动选择更稳定的髋外展偏移方向（返回 +1 或 -1）"""
        best_sign = 1
        best_roll = None
        test_steps = int(0.3 / self.model.opt.timestep)
        for sign in (1, -1):
            targets = dict(stand_targets)
            targets["FR_hip_joint"] = stand_targets["FR_hip_joint"]
            targets["RR_hip_joint"] = 0.08 * sign
            targets["FL_hip_joint"] = -0.16 * sign
            targets["RL_hip_joint"] = -0.12 * sign

            accum = 0.0
            for _ in range(test_steps):
                if self._should_stop(viewer):
                    return best_sign
                self._apply_pd_control(targets)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(self.model.opt.timestep)
                accum += abs(self._get_base_roll())

            avg_roll = accum / max(test_steps, 1)
            if best_roll is None or avg_roll < best_roll:
                best_roll = avg_roll
                best_sign = sign

        return best_sign

    def _get_base_roll(self) -> float:
        """获取机身 roll 角（弧度）"""
        if self.model.nq < 7:
            return 0.0
        qw, qx, qy, qz = self.data.qpos[3:7]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        return math.atan2(sinr_cosp, cosr_cosp)

    def _apply_pd_control(self, targets: Dict[str, float], gain_scale: Optional[Dict[str, float]] = None):
        """基于关节目标角度的 PD 控制（用于 Go2）"""
        actuator_names = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
        ]

        for act_name in actuator_names:
            joint_name = f"{act_name}_joint"
            if joint_name not in targets:
                continue

            try:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            except Exception:
                continue

            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            q = self.data.qpos[qpos_adr]
            qd = self.data.qvel[qvel_adr]

            # 关节类型分配不同增益
            if "calf" in joint_name:
                kp, kd = 80.0, 4.0
            elif "thigh" in joint_name:
                kp, kd = 60.0, 3.0
            else:
                kp, kd = 40.0, 2.5

            if gain_scale and joint_name in gain_scale:
                scale = gain_scale[joint_name]
                kp *= scale
                kd *= scale

            tau = kp * (targets[joint_name] - q) - kd * qd

            # 扭矩限幅
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[aid]
            if tau < ctrl_min:
                tau = ctrl_min
            elif tau > ctrl_max:
                tau = ctrl_max

            self.data.ctrl[aid] = tau
    
    def _stand_action(self, **kwargs) -> bool:
        """站立姿势动作"""
        if not self.mujoco_available:
            log_warning("Simulation mode: stand action")
            return True

        try:
            log_info("Executing stand action...")

            # Use persistent viewer
            if not self.ensure_viewer():
                log_error("Unable to create/get viewer")
                return False

            viewer = self.viewer

            # Set standing pose
            self._set_initial_pose()
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()

            # Hold standing pose for a short time
            duration = kwargs.get('duration', 1.0)  # Reduced default duration
            steps = int(duration / self.model.opt.timestep)

            for _ in range(steps):
                if self._should_stop(viewer):
                    return False
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(self.model.opt.timestep)

            log_info("Stand action completed (viewer remains open)")
            return True

        except Exception as e:
            log_error(f"Stand action failed: {e}")
            return False

    def _sit_action(self, **kwargs) -> bool:
        """坐下姿势动作"""
        if not self.mujoco_available:
            log_warning("Simulation mode: sit action")
            return True

        try:
            log_info("Executing sit action...")

            if not self.ensure_viewer():
                log_error("Unable to create/get viewer")
                return False

            viewer = self.viewer

            # Sit pose targets (lower body)
            if self.robot_type.lower() == "go2":
                sit_targets = self._get_go2_stand_targets()
                # Modify for sitting - bend legs more
                sit_targets["FR_thigh_joint"] = 1.2
                sit_targets["FR_calf_joint"] = -2.4
                sit_targets["FL_thigh_joint"] = 1.2
                sit_targets["FL_calf_joint"] = -2.4
                sit_targets["RR_thigh_joint"] = 1.2
                sit_targets["RR_calf_joint"] = -2.4
                sit_targets["RL_thigh_joint"] = 1.2
                sit_targets["RL_calf_joint"] = -2.4

                duration = kwargs.get('duration', 1.5)
                steps = int(duration / self.model.opt.timestep)

                for _ in range(steps):
                    if self._should_stop(viewer):
                        return False
                    self._apply_pd_control(sit_targets)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(self.model.opt.timestep)

            log_info("Sit action completed (viewer remains open)")
            return True

        except Exception as e:
            log_error(f"Sit action failed: {e}")
            return False

    def _walk_action(self, **kwargs) -> bool:
        """
        Walk action - Go2 prefers official SDK high-level control.

        In MuJoCo, use a simplified trot gait (thigh/calf swing + foot lift).
        """
        # Prefer official SDK walk for real robot (no MuJoCo)
        if self.robot_type.lower() == "go2" and not self.mujoco_available and UNITREE_AVAILABLE:
            return self._walk_sdk_go2(**kwargs)

        if not self.mujoco_available:
            log_warning("Simulation mode: walk action")
            return True

        try:
            log_info("Running walk action...")

            if not self.ensure_viewer():
                log_error("Failed to create/get viewer")
                return False

            viewer = self.viewer

            if self.robot_type.lower() == "go2":
                ok = self._walk_trot_gait_go2(viewer, **kwargs)
                if not ok:
                    log_warning("Walk action interrupted")
                    return False

            log_info("Walk action completed (viewer kept open)")
            return True

        except Exception as e:
            log_error(f"Walk action failed: {e}")
            return False

    def _walk_trot_gait_go2(self, viewer, **kwargs) -> bool:
        """
        Go2 trot gait using joint order from unitree_sdk2_python.
        - hip: abduction/adduction (keep standing)
        - thigh/calf: fore-aft swing + foot lift
        """
        import math

        num_cycles = kwargs.get("cycles", 6)
        gait_period = kwargs.get("gait_period", 0.5)
        thigh_swing = kwargs.get("thigh_swing", 0.22)
        calf_lift = kwargs.get("calf_lift", 0.35)

        stand_targets = self._get_go2_stand_targets()

        dt = self.model.opt.timestep
        steps_per_cycle = max(2, int(gait_period / dt))
        half_cycle_steps = max(1, steps_per_cycle // 2)

        def apply_targets(targets, gain_scale=None):
            self._apply_pd_control(targets, gain_scale=gain_scale)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(dt)

        # Stabilize standing
        log_info("Stabilizing standing pose...")
        for _ in range(int(1.0 / dt)):
            if self._should_stop(viewer):
                return False
            apply_targets(stand_targets)

        log_info(f"Start trot gait walk ({num_cycles} cycles)...")

        for _ in range(num_cycles):
            if self._should_stop(viewer):
                return False

            # Phase 1: FR + RL swing forward, FL + RR support
            for step in range(half_cycle_steps):
                if self._should_stop(viewer):
                    return False

                progress = step / half_cycle_steps
                swing = math.sin(progress * math.pi)

                targets = dict(stand_targets)

                for leg in ("FR", "RL"):
                    targets[f"{leg}_thigh_joint"] = stand_targets[f"{leg}_thigh_joint"] + thigh_swing * swing
                    targets[f"{leg}_calf_joint"] = stand_targets[f"{leg}_calf_joint"] + calf_lift * swing

                for leg in ("FL", "RR"):
                    targets[f"{leg}_thigh_joint"] = stand_targets[f"{leg}_thigh_joint"] - 0.6 * thigh_swing * swing
                    targets[f"{leg}_calf_joint"] = stand_targets[f"{leg}_calf_joint"] - 0.3 * calf_lift * swing

                apply_targets(targets)

            # Phase 2: FL + RR swing forward, FR + RL support
            for step in range(half_cycle_steps):
                if self._should_stop(viewer):
                    return False

                progress = step / half_cycle_steps
                swing = math.sin(progress * math.pi)

                targets = dict(stand_targets)

                for leg in ("FL", "RR"):
                    targets[f"{leg}_thigh_joint"] = stand_targets[f"{leg}_thigh_joint"] + thigh_swing * swing
                    targets[f"{leg}_calf_joint"] = stand_targets[f"{leg}_calf_joint"] + calf_lift * swing

                for leg in ("FR", "RL"):
                    targets[f"{leg}_thigh_joint"] = stand_targets[f"{leg}_thigh_joint"] - 0.6 * thigh_swing * swing
                    targets[f"{leg}_calf_joint"] = stand_targets[f"{leg}_calf_joint"] - 0.3 * calf_lift * swing

                apply_targets(targets)

        # Return to standing
        log_info("Returning to standing pose...")
        for _ in range(int(0.5 / dt)):
            if self._should_stop(viewer):
                return False
            apply_targets(stand_targets)
        return True

    def _walk_sdk_go2(self, **kwargs) -> bool:
        """
        Go2 walk via official SDK Move/StopMove high-level control.
        Reference: unitree_sdk2_python/example/go2/high_level/go2_sport_client.py
        """
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except Exception as e:
            log_warning(f"SDK import failed: {e}")
            return False

        iface = kwargs.get("iface") or kwargs.get("network_interface")
        if not self._sdk_channel_inited:
            try:
                if iface:
                    ChannelFactoryInitialize(0, iface)
                else:
                    ChannelFactoryInitialize(0)
                self._sdk_channel_inited = True
            except Exception as e:
                log_error(f"SDK channel init failed: {e}")
                return False

        if self._sport_client is None:
            self._sport_client = SportClient()
            self._sport_client.SetTimeout(5.0)
            self._sport_client.Init()

        vx = kwargs.get("vx", 0.3)
        vy = kwargs.get("vy", 0.0)
        vyaw = kwargs.get("vyaw", 0.0)
        duration = kwargs.get("duration", 2.0)

        log_info(f"SDK walk: vx={vx}, vy={vy}, vyaw={vyaw}, duration={duration}s")

        try:
            self._sport_client.Move(vx, vy, vyaw)
            time.sleep(max(0.0, duration))
            self._sport_client.StopMove()
            return True
        except Exception as e:
            log_error(f"SDK walk failed: {e}")
            return False

    def _stop_action(self, **kwargs) -> bool:
        """停止运动"""
        self.stop()
        return True
