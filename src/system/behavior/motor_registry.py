#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor Registry — 真实机器人电机型号注册表

定义每个机器人型号对应的真实电机硬件信息，包括：
- 电机型号和类型
- 物理限制（扭矩、转速、位置）
- 通信通道配置
- 与抽象轨道的映射关系

Usage:
    from src.system.behavior.motor_registry import (
        get_motor_by_joint_name,
        get_motors_for_robot,
        RobotMotorSpec,
        MOTOR_REGISTRY,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 电机硬件规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorSpec:
    """单个电机的硬件规格"""
    
    # 电机标识
    motor_id: str              # 唯一标识，如 "FL_hip_motor"
    motor_type: str            # 电机型号，如 "GO2_MOTOR_V1"
    brand: str                 # 品牌，如 "unitree"
    
    # 物理限制
    max_torque: float          # 最大扭矩 (N·m)
    max_velocity: float        # 最大转速 (rad/s)
    position_min: float        # 最小位置 (rad)
    position_max: float        # 最大位置 (rad)
    
    # 通信配置
    channel_id: int            # CAN/总线通道ID
    motor_index: int           # 电机索引号
    
    # 映射到抽象轨道
    abstract_track: str         # 对应的抽象轨道名，如 "leg_fl"
    joint_name: str            # 关节名称，如 "FL_hip"


@dataclass
class RobotMotorSpec:
    """机器人型号的电机配置"""
    
    robot_type: str            # 机器人型号，如 "go2"
    brand: str                 # 品牌，如 "unitree"
    motors: List[MotorSpec]    # 电机列表
    total_dof: int             # 总自由度
    
    def get_motor_by_joint(self, joint_name: str) -> Optional[MotorSpec]:
        """通过关节名称查找电机"""
        for motor in self.motors:
            if motor.joint_name == joint_name:
                return motor
        return None
    
    def get_motors_by_track(self, track_name: str) -> List[MotorSpec]:
        """通过轨道名称查找所有相关电机"""
        return [m for m in self.motors if m.abstract_track == track_name]


# ---------------------------------------------------------------------------
# 具体机器人电机配置
# ---------------------------------------------------------------------------

# Unitree GO2 电机配置 (12个关节)
GO2_MOTORS: List[MotorSpec] = [
    # Front Left Leg
    MotorSpec(
        motor_id="go2_fl_hip",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hip",
    ),
    MotorSpec(
        motor_id="go2_fl_thigh",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_thigh",
    ),
    MotorSpec(
        motor_id="go2_fl_calf",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_calf",
    ),
    # Front Right Leg
    MotorSpec(
        motor_id="go2_fr_hip",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hip",
    ),
    MotorSpec(
        motor_id="go2_fr_thigh",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_thigh",
    ),
    MotorSpec(
        motor_id="go2_fr_calf",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_calf",
    ),
    # Rear Left Leg
    MotorSpec(
        motor_id="go2_rl_hip",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hip",
    ),
    MotorSpec(
        motor_id="go2_rl_thigh",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_thigh",
    ),
    MotorSpec(
        motor_id="go2_rl_calf",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_calf",
    ),
    # Rear Right Leg
    MotorSpec(
        motor_id="go2_rr_hip",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hip",
    ),
    MotorSpec(
        motor_id="go2_rr_thigh",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_thigh",
    ),
    MotorSpec(
        motor_id="go2_rr_calf",
        motor_type="GO2_JOINT_MOTOR",
        brand="unitree",
        max_torque=23.5,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_calf",
    ),
]


# Unitree A1 电机配置 (12个关节)
A1_MOTORS: List[MotorSpec] = [
    MotorSpec(
        motor_id="a1_fl_hip",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hip",
    ),
    MotorSpec(
        motor_id="a1_fl_thigh",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.2,
        position_max=2.5,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_thigh",
    ),
    MotorSpec(
        motor_id="a1_fl_calf",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-2.6,
        position_max=-0.6,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_calf",
    ),
    # 其他关节类似...
    MotorSpec(
        motor_id="a1_fr_hip",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hip",
    ),
    MotorSpec(
        motor_id="a1_fr_thigh",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.2,
        position_max=2.5,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_thigh",
    ),
    MotorSpec(
        motor_id="a1_fr_calf",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-2.6,
        position_max=-0.6,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_calf",
    ),
    MotorSpec(
        motor_id="a1_rl_hip",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hip",
    ),
    MotorSpec(
        motor_id="a1_rl_thigh",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.2,
        position_max=2.5,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_thigh",
    ),
    MotorSpec(
        motor_id="a1_rl_calf",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-2.6,
        position_max=-0.6,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_calf",
    ),
    MotorSpec(
        motor_id="a1_rr_hip",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hip",
    ),
    MotorSpec(
        motor_id="a1_rr_thigh",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-0.2,
        position_max=2.5,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_thigh",
    ),
    MotorSpec(
        motor_id="a1_rr_calf",
        motor_type="A1_JOINT_MOTOR",
        brand="unitree",
        max_torque=33.5,
        max_velocity=21.0,
        position_min=-2.6,
        position_max=-0.6,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_calf",
    ),
]


# Unitree B2 电机配置 (12个关节, 大型四足)
B2_MOTORS: List[MotorSpec] = [
    # Front Left Leg
    MotorSpec(
        motor_id="b2_fl_hip",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hip",
    ),
    MotorSpec(
        motor_id="b2_fl_thigh",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_thigh",
    ),
    MotorSpec(
        motor_id="b2_fl_calf",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_calf",
    ),
    # Front Right Leg
    MotorSpec(
        motor_id="b2_fr_hip",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hip",
    ),
    MotorSpec(
        motor_id="b2_fr_thigh",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_thigh",
    ),
    MotorSpec(
        motor_id="b2_fr_calf",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_calf",
    ),
    # Rear Left Leg
    MotorSpec(
        motor_id="b2_rl_hip",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hip",
    ),
    MotorSpec(
        motor_id="b2_rl_thigh",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_thigh",
    ),
    MotorSpec(
        motor_id="b2_rl_calf",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_calf",
    ),
    # Rear Right Leg
    MotorSpec(
        motor_id="b2_rr_hip",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hip",
    ),
    MotorSpec(
        motor_id="b2_rr_thigh",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_thigh",
    ),
    MotorSpec(
        motor_id="b2_rr_calf",
        motor_type="B2_JOINT_MOTOR",
        brand="unitree",
        max_torque=60.0,
        max_velocity=20.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_calf",
    ),
]


# Unitree G1 电机配置 (23个关节, 人形)
G1_MOTORS: List[MotorSpec] = [
    # Left Leg — joint_name must match _SPEC_G1 track names exactly
    MotorSpec(
        motor_id="g1_l_hip_yaw",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.09,
        position_max=2.09,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_l",
        joint_name="L_hip_yaw",
    ),
    MotorSpec(
        motor_id="g1_l_hip_roll",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-0.52,
        position_max=2.53,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_l",
        joint_name="L_hip_roll",
    ),
    MotorSpec(
        motor_id="g1_l_hip_pitch",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.87,
        position_max=2.87,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_l",
        joint_name="L_hip_pitch",
    ),
    MotorSpec(
        motor_id="g1_l_knee",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=139.0,
        max_velocity=20.0,
        position_min=-0.26,
        position_max=2.87,
        channel_id=0,
        motor_index=3,
        abstract_track="leg_l",
        joint_name="L_knee",
    ),
    MotorSpec(
        motor_id="g1_l_ankle",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=50.0,
        max_velocity=20.0,
        position_min=-0.87,
        position_max=0.52,
        channel_id=0,
        motor_index=4,
        abstract_track="leg_l",
        joint_name="L_ankle",
    ),
    # Right Leg
    MotorSpec(
        motor_id="g1_r_hip_yaw",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.09,
        position_max=2.09,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_r",
        joint_name="R_hip_yaw",
    ),
    MotorSpec(
        motor_id="g1_r_hip_roll",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.53,
        position_max=0.52,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_r",
        joint_name="R_hip_roll",
    ),
    MotorSpec(
        motor_id="g1_r_hip_pitch",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.87,
        position_max=2.87,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_r",
        joint_name="R_hip_pitch",
    ),
    MotorSpec(
        motor_id="g1_r_knee",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=139.0,
        max_velocity=20.0,
        position_min=-0.26,
        position_max=2.87,
        channel_id=1,
        motor_index=3,
        abstract_track="leg_r",
        joint_name="R_knee",
    ),
    MotorSpec(
        motor_id="g1_r_ankle",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=50.0,
        max_velocity=20.0,
        position_min=-0.87,
        position_max=0.52,
        channel_id=1,
        motor_index=4,
        abstract_track="leg_r",
        joint_name="R_ankle",
    ),
    # Left Arm
    MotorSpec(
        motor_id="g1_l_shoulder_pitch",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-3.14,
        position_max=3.14,
        channel_id=2,
        motor_index=0,
        abstract_track="arm_l",
        joint_name="L_shoulder_pitch",
    ),
    MotorSpec(
        motor_id="g1_l_shoulder_roll",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-0.52,
        position_max=2.87,
        channel_id=2,
        motor_index=1,
        abstract_track="arm_l",
        joint_name="L_shoulder_roll",
    ),
    MotorSpec(
        motor_id="g1_l_elbow",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-1.57,
        position_max=1.57,
        channel_id=2,
        motor_index=2,
        abstract_track="arm_l",
        joint_name="L_elbow",
    ),
    # Right Arm
    MotorSpec(
        motor_id="g1_r_shoulder_pitch",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-3.14,
        position_max=3.14,
        channel_id=3,
        motor_index=0,
        abstract_track="arm_r",
        joint_name="R_shoulder_pitch",
    ),
    MotorSpec(
        motor_id="g1_r_shoulder_roll",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-2.87,
        position_max=0.52,
        channel_id=3,
        motor_index=1,
        abstract_track="arm_r",
        joint_name="R_shoulder_roll",
    ),
    MotorSpec(
        motor_id="g1_r_elbow",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=25.0,
        max_velocity=20.0,
        position_min=-1.57,
        position_max=1.57,
        channel_id=3,
        motor_index=2,
        abstract_track="arm_r",
        joint_name="R_elbow",
    ),
    # Torso
    MotorSpec(
        motor_id="g1_torso_yaw",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-2.09,
        position_max=2.09,
        channel_id=4,
        motor_index=0,
        abstract_track="torso",
        joint_name="torso_yaw",
    ),
    MotorSpec(
        motor_id="g1_torso_pitch",
        motor_type="G1_JOINT_MOTOR",
        brand="unitree",
        max_torque=88.0,
        max_velocity=20.0,
        position_min=-1.05,
        position_max=1.05,
        channel_id=4,
        motor_index=1,
        abstract_track="torso",
        joint_name="torso_pitch",
    ),
]


# Unitree H1 电机配置 (20个关节)
H1_MOTORS: List[MotorSpec] = [
    # H1有更复杂的电机配置，这里列出主要关节
    # Waist
    MotorSpec(
        motor_id="h1_waist_yaw",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.05,
        position_max=1.05,
        channel_id=0,
        motor_index=0,
        abstract_track="body_waist",
        joint_name="waist_yaw",
    ),
    # Left Arm
    MotorSpec(
        motor_id="h1_l_arm_shoulder_pitch",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-2.96,
        position_max=2.96,
        channel_id=1,
        motor_index=0,
        abstract_track="arm_l",
        joint_name="L_arm_shoulder_pitch",
    ),
    MotorSpec(
        motor_id="h1_l_arm_shoulder_roll",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-0.52,
        position_max=2.61,
        channel_id=1,
        motor_index=1,
        abstract_track="arm_l",
        joint_name="L_arm_shoulder_roll",
    ),
    MotorSpec(
        motor_id="h1_l_arm_elbow",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-2.96,
        position_max=0.0,
        channel_id=1,
        motor_index=2,
        abstract_track="arm_l",
        joint_name="L_arm_elbow",
    ),
    # Right Arm
    MotorSpec(
        motor_id="h1_r_arm_shoulder_pitch",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-2.96,
        position_max=2.96,
        channel_id=2,
        motor_index=0,
        abstract_track="arm_r",
        joint_name="R_arm_shoulder_pitch",
    ),
    MotorSpec(
        motor_id="h1_r_arm_shoulder_roll",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-2.61,
        position_max=0.52,
        channel_id=2,
        motor_index=1,
        abstract_track="arm_r",
        joint_name="R_arm_shoulder_roll",
    ),
    MotorSpec(
        motor_id="h1_r_arm_elbow",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=20.0,
        max_velocity=10.0,
        position_min=-2.96,
        position_max=0.0,
        channel_id=2,
        motor_index=2,
        abstract_track="arm_r",
        joint_name="R_arm_elbow",
    ),
    # Left Leg
    MotorSpec(
        motor_id="h1_l_leg_hip_yaw",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.05,
        position_max=1.05,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_l",
        joint_name="L_hip_yaw",
    ),
    MotorSpec(
        motor_id="h1_l_leg_hip_roll",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-0.52,
        position_max=1.57,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_l",
        joint_name="L_hip_roll",
    ),
    MotorSpec(
        motor_id="h1_l_leg_hip_pitch",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-2.79,
        position_max=2.79,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_l",
        joint_name="L_hip_pitch",
    ),
    MotorSpec(
        motor_id="h1_l_leg_knee",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-0.17,
        position_max=2.61,
        channel_id=3,
        motor_index=3,
        abstract_track="leg_l",
        joint_name="L_knee",
    ),
    MotorSpec(
        motor_id="h1_l_leg_ankle",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.31,
        position_max=1.31,
        channel_id=3,
        motor_index=4,
        abstract_track="leg_l",
        joint_name="L_ankle",
    ),
    # Right Leg
    MotorSpec(
        motor_id="h1_r_leg_hip_yaw",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.05,
        position_max=1.05,
        channel_id=4,
        motor_index=0,
        abstract_track="leg_r",
        joint_name="R_hip_yaw",
    ),
    MotorSpec(
        motor_id="h1_r_leg_hip_roll",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.57,
        position_max=0.52,
        channel_id=4,
        motor_index=1,
        abstract_track="leg_r",
        joint_name="R_hip_roll",
    ),
    MotorSpec(
        motor_id="h1_r_leg_hip_pitch",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-2.79,
        position_max=2.79,
        channel_id=4,
        motor_index=2,
        abstract_track="leg_r",
        joint_name="R_hip_pitch",
    ),
    MotorSpec(
        motor_id="h1_r_leg_knee",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-2.61,
        position_max=0.17,
        channel_id=4,
        motor_index=3,
        abstract_track="leg_r",
        joint_name="R_knee",
    ),
    MotorSpec(
        motor_id="h1_r_leg_ankle",
        motor_type="H1_JOINT_MOTOR",
        brand="unitree",
        max_torque=55.0,
        max_velocity=10.0,
        position_min=-1.31,
        position_max=1.31,
        channel_id=4,
        motor_index=4,
        abstract_track="leg_r",
        joint_name="R_ankle",
    ),
]


# Boston Dynamics Spot 电机配置
SPOT_MOTORS: List[MotorSpec] = [
    # Spot有更复杂的配置，这里列出主要关节
    MotorSpec(
        motor_id="spot_fl_hx",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-1.34,
        position_max=2.3,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hx",
    ),
    MotorSpec(
        motor_id="spot_fl_hy",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=0.0,
        position_max=2.5,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_hy",
    ),
    MotorSpec(
        motor_id="spot_fl_kn",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.6,
        position_max=-0.5,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_kn",
    ),
    # 其他腿类似...
    MotorSpec(
        motor_id="spot_fr_hx",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.3,
        position_max=1.34,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hx",
    ),
    MotorSpec(
        motor_id="spot_fr_hy",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.5,
        position_max=0.0,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_hy",
    ),
    MotorSpec(
        motor_id="spot_fr_kn",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.6,
        position_max=-0.5,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_kn",
    ),
    MotorSpec(
        motor_id="spot_rl_hx",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-1.34,
        position_max=2.3,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hx",
    ),
    MotorSpec(
        motor_id="spot_rl_hy",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=0.0,
        position_max=2.5,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_hy",
    ),
    MotorSpec(
        motor_id="spot_rl_kn",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.6,
        position_max=-0.5,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_kn",
    ),
    MotorSpec(
        motor_id="spot_rr_hx",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.3,
        position_max=1.34,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hx",
    ),
    MotorSpec(
        motor_id="spot_rr_hy",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.5,
        position_max=0.0,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_hy",
    ),
    MotorSpec(
        motor_id="spot_rr_kn",
        motor_type="SPOT_JOINT_MOTOR",
        brand="boston_dynamics",
        max_torque=26.0,
        max_velocity=10.0,
        position_min=-2.6,
        position_max=-0.5,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_kn",
    ),
]


# Xiaomi Cyberdog 电机配置 (12个关节)
CYBERDOG_MOTORS: List[MotorSpec] = [
    # Front Left Leg
    MotorSpec(
        motor_id="cyberdog_fl_hip",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hip",
    ),
    MotorSpec(
        motor_id="cyberdog_fl_thigh",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog_fl_calf",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_calf",
    ),
    # Front Right Leg
    MotorSpec(
        motor_id="cyberdog_fr_hip",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hip",
    ),
    MotorSpec(
        motor_id="cyberdog_fr_thigh",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog_fr_calf",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_calf",
    ),
    # Rear Left Leg
    MotorSpec(
        motor_id="cyberdog_rl_hip",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hip",
    ),
    MotorSpec(
        motor_id="cyberdog_rl_thigh",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog_rl_calf",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_calf",
    ),
    # Rear Right Leg
    MotorSpec(
        motor_id="cyberdog_rr_hip",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hip",
    ),
    MotorSpec(
        motor_id="cyberdog_rr_thigh",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog_rr_calf",
        motor_type="CYBERDOG_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=20.0,
        max_velocity=30.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_calf",
    ),
]


# Xiaomi Cyberdog2 电机配置 (12个关节，性能更强)
CYBERDOG2_MOTORS: List[MotorSpec] = [
    # Front Left Leg
    MotorSpec(
        motor_id="cyberdog2_fl_hip",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=0,
        motor_index=0,
        abstract_track="leg_fl",
        joint_name="FL_hip",
    ),
    MotorSpec(
        motor_id="cyberdog2_fl_thigh",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=0,
        motor_index=1,
        abstract_track="leg_fl",
        joint_name="FL_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog2_fl_calf",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=0,
        motor_index=2,
        abstract_track="leg_fl",
        joint_name="FL_calf",
    ),
    # Front Right Leg
    MotorSpec(
        motor_id="cyberdog2_fr_hip",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=1,
        motor_index=0,
        abstract_track="leg_fr",
        joint_name="FR_hip",
    ),
    MotorSpec(
        motor_id="cyberdog2_fr_thigh",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=1,
        motor_index=1,
        abstract_track="leg_fr",
        joint_name="FR_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog2_fr_calf",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=1,
        motor_index=2,
        abstract_track="leg_fr",
        joint_name="FR_calf",
    ),
    # Rear Left Leg
    MotorSpec(
        motor_id="cyberdog2_rl_hip",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=2,
        motor_index=0,
        abstract_track="leg_rl",
        joint_name="RL_hip",
    ),
    MotorSpec(
        motor_id="cyberdog2_rl_thigh",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=2,
        motor_index=1,
        abstract_track="leg_rl",
        joint_name="RL_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog2_rl_calf",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=2,
        motor_index=2,
        abstract_track="leg_rl",
        joint_name="RL_calf",
    ),
    # Rear Right Leg
    MotorSpec(
        motor_id="cyberdog2_rr_hip",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-0.8,
        position_max=0.8,
        channel_id=3,
        motor_index=0,
        abstract_track="leg_rr",
        joint_name="RR_hip",
    ),
    MotorSpec(
        motor_id="cyberdog2_rr_thigh",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=0.0,
        position_max=2.6,
        channel_id=3,
        motor_index=1,
        abstract_track="leg_rr",
        joint_name="RR_thigh",
    ),
    MotorSpec(
        motor_id="cyberdog2_rr_calf",
        motor_type="CYBERDOG2_JOINT_MOTOR",
        brand="xiaomi",
        max_torque=25.0,
        max_velocity=35.0,
        position_min=-2.7,
        position_max=-0.5,
        channel_id=3,
        motor_index=2,
        abstract_track="leg_rr",
        joint_name="RR_calf",
    ),
]


# ---------------------------------------------------------------------------
# 机器人电机注册表
# ---------------------------------------------------------------------------

MOTOR_REGISTRY: Dict[str, RobotMotorSpec] = {
    # Unitree robots
    "go2": RobotMotorSpec(
        robot_type="go2",
        brand="unitree",
        motors=GO2_MOTORS,
        total_dof=12,
    ),
    "go2w": RobotMotorSpec(
        robot_type="go2w",
        brand="unitree",
        motors=GO2_MOTORS,  # go2w与go2相同的电机配置
        total_dof=12,
    ),
    "a1": RobotMotorSpec(
        robot_type="a1",
        brand="unitree",
        motors=A1_MOTORS,
        total_dof=12,
    ),
    "b2": RobotMotorSpec(
        robot_type="b2",
        brand="unitree",
        motors=B2_MOTORS,
        total_dof=12,
    ),
    "g1": RobotMotorSpec(
        robot_type="g1",
        brand="unitree",
        motors=G1_MOTORS,
        total_dof=23,
    ),
    "h1": RobotMotorSpec(
        robot_type="h1",
        brand="unitree",
        motors=H1_MOTORS,
        total_dof=20,
    ),
    "h1_2": RobotMotorSpec(
        robot_type="h1_2",
        brand="unitree",
        motors=H1_MOTORS,  # h1_2与h1相同的抽象轨道结构
        total_dof=20,
    ),
    # Boston Dynamics robots
    "spot": RobotMotorSpec(
        robot_type="spot",
        brand="boston_dynamics",
        motors=SPOT_MOTORS,
        total_dof=12,
    ),
    # Xiaomi robots
    "cyberdog": RobotMotorSpec(
        robot_type="cyberdog",
        brand="xiaomi",
        motors=CYBERDOG_MOTORS,
        total_dof=12,
    ),
    "cyberdog2": RobotMotorSpec(
        robot_type="cyberdog2",
        brand="xiaomi",
        motors=CYBERDOG2_MOTORS,
        total_dof=12,
    ),
}


# ---------------------------------------------------------------------------
# 公开API
# ---------------------------------------------------------------------------

def get_motor_registry(brand: str, robot_type: str) -> Optional[RobotMotorSpec]:
    """获取机器人电机注册表"""
    key = str(robot_type or "").lower()
    return MOTOR_REGISTRY.get(key)


def get_motors_for_robot(brand: str, robot_type: str) -> List[MotorSpec]:
    """获取机器人的所有电机"""
    spec = get_motor_registry(brand, robot_type)
    return spec.motors if spec else []


def get_motor_by_joint_name(brand: str, robot_type: str, joint_name: str) -> Optional[MotorSpec]:
    """通过关节名称查找电机"""
    spec = get_motor_registry(brand, robot_type)
    if spec:
        return spec.get_motor_by_joint(joint_name)
    return None


def get_motor_by_track(brand: str, robot_type: str, track_name: str) -> List[MotorSpec]:
    """通过轨道名称查找电机"""
    spec = get_motor_registry(brand, robot_type)
    if spec:
        return spec.get_motors_by_track(track_name)
    return []


def validate_motor_params(
    brand: str,
    robot_type: str,
    joint_name: str,
    position: Optional[float] = None,
    velocity: Optional[float] = None,
    torque: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    验证电机参数是否在安全范围内
    
    Returns:
        (is_valid, error_messages)
    """
    motor = get_motor_by_joint_name(brand, robot_type, joint_name)
    if not motor:
        return False, [f"Unknown joint: {joint_name} for {brand}/{robot_type}"]
    
    errors = []
    
    if position is not None:
        if position < motor.position_min or position > motor.position_max:
            errors.append(
                f"Position {position:.3f} out of range [{motor.position_min:.3f}, {motor.position_max:.3f}]"
            )
    
    if velocity is not None:
        if abs(velocity) > motor.max_velocity:
            errors.append(
                f"Velocity {velocity:.3f} exceeds max {motor.max_velocity:.3f}"
            )
    
    if torque is not None:
        if abs(torque) > motor.max_torque:
            errors.append(
                f"Torque {torque:.3f} exceeds max {motor.max_torque:.3f}"
            )
    
    return len(errors) == 0, errors
