#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D 闂?Real Gymnasium environment for Unitree Go2 locomotion training.

UnitreeGymEnv
    Headless-first MuJoCo-backed Gymnasium environment.
    Observation follows the DEFAULT_COMPONENT_ORDER from src.system.policy.obs_builder
    so that trained networks deploy directly through PolicyRunner without remapping:

        base_angular_velocity (3)
        projected_gravity     (3)
        joint_positions       (12)
        joint_velocities      (12)
        last_action           (12)
        velocity_command      (3)
        闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜鹃梺鍐插帨閸嬫捇鏌嶉崗澶婁壕闂佸啿鍘滈崑鎾绘煃閸忓浜?        total obs_dim          45

    Action space: Box(-1, 1, shape=(12,)) 闂?normalised torque targets.
    Torques are scaled to physical limits before writing to mujoco ctrl.

    ``gymnasium.utils.env_checker.check_env(env)`` passes on this class.
"""

from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from src.system.training.task_module_registry import (
    default_reward_terms,
    default_termination_conditions,
)


# ---------------------------------------------------------------------------
# Go2 constants
# ---------------------------------------------------------------------------

GO2_JOINT_NAMES = [
    "FL_hip_joint",   "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",   "FR_thigh_joint",  "FR_calf_joint",
    "RL_hip_joint",   "RL_thigh_joint",  "RL_calf_joint",
    "RR_hip_joint",   "RR_thigh_joint",  "RR_calf_joint",
]

# Default (standing) joint angles in radians [FL_hip, FL_thigh, FL_calf, ...]
GO2_DEFAULT_QPOS = np.array([
     0.0,  0.9, -1.8,   # FL
     0.0,  0.9, -1.8,   # FR
     0.0,  0.9, -1.8,   # RL
     0.0,  0.9, -1.8,   # RR
], dtype=np.float32)

# Maximum torques (N閻犺櫣妫? per joint class
_TORQUE_LIMITS = np.array([
    33.5, 33.5, 45.0,   # FL
    33.5, 33.5, 45.0,   # FR
    33.5, 33.5, 45.0,   # RL
    33.5, 33.5, 45.0,   # RR
], dtype=np.float32)

# Software PD gains (Nm/rad, Nm·s/rad).
# Matched to Unitree Go2 hardware defaults and Isaac Gym / legged_gym standard.
# Old values (20.0, 0.5) were far too low for the 15 kg Go2 — the robot could
# not generate enough joint torque to maintain standing height.
_PD_KP = 70.0   # position gain
_PD_KD = 1.5    # velocity (damping) gain
_VEL_KP = 20.0  # gain for joint_velocity action mode

# Observation and action dimensions
OBS_DIM    = 45
ACTION_DIM = 12

_OBS_COMPONENT_DIM_ALIASES = {
    "base_angular_velocity": 3,
    "base_ang_vel": 3,
    "base_linear_velocity": 3,
    "base_lin_vel": 3,
    "projected_gravity": 3,
    "gravity_vec": 3,
    "joint_positions": "num_joints",
    "joint_pos": "num_joints",
    "joint_velocities": "num_joints",
    "joint_vel": "num_joints",
    "last_action": "action_dim",
    "previous_action": "action_dim",
    "velocity_command": 3,
    "command": 4,
    "imu": 6,
    "phase_sin_cos": 2,
    "phase": 2,
    # Reference motion obs (DeepMimic / AMP standard — policy sees the target)
    "reference_joint_positions": "num_joints",
    "ref_joint_pos": "num_joints",
    "reference_joint_velocities": "num_joints",
    "ref_joint_vel": "num_joints",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_go2_mjcf() -> Optional[str]:
    """Return absolute path to go2.xml if available in the repo, else None."""
    candidates = [
        "runtime/sdk/unitree/unitree_mujoco/unitree_robots/go2/go2.xml",
        str(Path(__file__).parent.parent.parent /
            "runtime/sdk/unitree/unitree_mujoco/unitree_robots/go2/go2.xml"),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None


def _find_go2_scene_xml(scene_type: str = "flat") -> Optional[str]:
    """
    Return absolute path to the go2 scene XML for the given scene_type.

    scene_type: 'flat'    闂?scene.xml
                'terrain' 闂?scene_terrain.xml
    Falls back to go2.xml (robot-only) if neither scene file is found.
    """
    filename = "scene_terrain.xml" if scene_type == "terrain" else "scene.xml"
    candidates = [
        f"runtime/sdk/unitree/unitree_mujoco/unitree_robots/go2/{filename}",
        str(Path(__file__).parent.parent.parent /
            f"runtime/sdk/unitree/unitree_mujoco/unitree_robots/go2/{filename}"),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    # Fallback: robot-only MJCF (no scene)
    return _find_go2_mjcf()


def _resolve_registered_scene_xml(robot_type: str, scene_type: str = "flat") -> Optional[str]:
    robot_key = str(robot_type or "").strip().lower()
    brand_map = {
        "go2": ("unitree", "go2"),
        "go2w": ("unitree", "go2w"),
        "go1": ("unitree", "go1"),
        "a1": ("unitree", "a1"),
        "g1": ("unitree", "g1"),
        "h1": ("unitree", "h1"),
        "h1_2": ("unitree", "h1_2"),
        "spot": ("boston_dynamics", "spot"),
    }
    brand_info = brand_map.get(robot_key)
    if brand_info is None:
        return None
    try:
        from src.system.models.mujoco_asset_registry import resolve_mujoco_asset
        loc = resolve_mujoco_asset(brand_info[0], brand_info[1])
        if loc is None:
            return None
        filename = "scene_terrain.xml" if scene_type == "terrain" else "scene.xml"
        candidate = loc.scene_path.parent / filename
        if candidate.exists():
            return str(candidate)
        if scene_type == "flat" and loc.scene_path.exists():
            return str(loc.scene_path)
    except Exception:
        return None
    return None


def _resolve_scene_xml(
    scene_config,       # Optional[SceneConfig] - may be None
    robot_mjcf_path: str = "",
    robot_type: str = "",
) -> Optional[str]:
    """
    Resolve the final scene XML path from a SceneConfig (or None).

    Priority:
    1. scene_type='custom' + valid custom_scene_path  闂?use directly
    2. scene_type='flat'/'terrain'                    闂?auto-detect from robot dir
    3. scene_config is None                           闂?auto-detect scene.xml (flat)
    Fallback: None (UnitreeGymEnv will use _MINIMAL_MJCF).
    """
    if scene_config is None:
        resolved = _resolve_registered_scene_xml(robot_type, "flat")
        if resolved:
            return resolved
        if robot_mjcf_path:
            sibling = Path(os.path.abspath(robot_mjcf_path)).parent / "scene.xml"
            if sibling.exists():
                return str(sibling)
        # Only fall back to Go2 scene when robot_type is explicitly go2 or empty
        # (unknown).  Non-Go2 robots must not silently load a Go2 scene.
        rt = str(robot_type or "").strip().lower()
        if not rt or rt == "go2":
            return _find_go2_scene_xml("flat")
        return None

    scene_type = getattr(scene_config, "scene_type", "flat")

    if scene_type == "custom":
        custom = str(getattr(scene_config, "custom_scene_path", "") or "")
        if custom:
            abs_custom = os.path.abspath(custom)
            if os.path.exists(abs_custom):
                return abs_custom
        # custom path invalid — fall through to auto-detect
        scene_type = "flat"

    # Try robot mjcf sibling directory first
    filename = "scene_terrain.xml" if scene_type == "terrain" else "scene.xml"
    if robot_mjcf_path:
        sibling = Path(os.path.abspath(robot_mjcf_path)).parent / filename
        if sibling.exists():
            return str(sibling)

    resolved = _resolve_registered_scene_xml(robot_type, scene_type)
    if resolved:
        return resolved

    # Only fall back to Go2 scene for go2 or unspecified robot type
    rt = str(robot_type or "").strip().lower()
    if not rt or rt == "go2":
        return _find_go2_scene_xml(scene_type)
    return None


def _extract_joint_names_from_model(model, action_dim: int) -> List[str]:
    names: List[str] = []
    try:
        import mujoco
        for actuator_id in range(int(getattr(model, "nu", 0) or 0)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if name:
                names.append(str(name))
    except Exception:
        names = []

    if names:
        return names[: max(0, int(action_dim or len(names)))]

    try:
        import mujoco
        for joint_id in range(int(getattr(model, "njnt", 0) or 0)):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not joint_name or str(joint_name).lower() == "root":
                continue
            names.append(str(joint_name))
    except Exception:
        names = []

    if names:
        return names[: max(0, int(action_dim or len(names)))]

    # Last resort: generate generic names rather than misleading Go2 names
    return [f"joint_{i}" for i in range(max(0, int(action_dim or 12)))]


# Minimal 12-DOF MJCF used when go2.xml is not available.
_MINIMAL_MJCF = """<mujoco model="unitree_minimal">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.1" armature="0.01"/>
    <geom contype="1" conaffinity="1" friction="1 0.005 0.0001"/>
  </default>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 1" rgba=".9 .9 .9 1"/>
    <body name="torso" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom name="torso_geom" type="box" size="0.12 0.06 0.05" mass="6.0"
            rgba=".8 .3 .3 1"/>
      <body name="FL_hip" pos="0.17 0.085 0">
        <joint name="FL_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="FL_thigh" pos="0.07 0 0">
          <joint name="FL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="FL_calf" pos="0 0 -0.21">
            <joint name="FL_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="FR_hip" pos="0.17 -0.085 0">
        <joint name="FR_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="FR_thigh" pos="0.07 0 0">
          <joint name="FR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="FR_calf" pos="0 0 -0.21">
            <joint name="FR_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="RL_hip" pos="-0.17 0.085 0">
        <joint name="RL_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="RL_thigh" pos="0.07 0 0">
          <joint name="RL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="RL_calf" pos="0 0 -0.21">
            <joint name="RL_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
      <body name="RR_hip" pos="-0.17 -0.085 0">
        <joint name="RR_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.07 0 0" mass="0.6"/>
        <body name="RR_thigh" pos="0.07 0 0">
          <joint name="RR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.4"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.21" mass="0.6"/>
          <body name="RR_calf" pos="0 0 -0.21">
            <joint name="RR_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.8"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 -0.19" mass="0.3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="FL_hip"    joint="FL_hip_joint"    gear="33.5"/>
    <motor name="FL_thigh"  joint="FL_thigh_joint"  gear="33.5"/>
    <motor name="FL_calf"   joint="FL_calf_joint"   gear="45.0"/>
    <motor name="FR_hip"    joint="FR_hip_joint"    gear="33.5"/>
    <motor name="FR_thigh"  joint="FR_thigh_joint"  gear="33.5"/>
    <motor name="FR_calf"   joint="FR_calf_joint"   gear="45.0"/>
    <motor name="RL_hip"    joint="RL_hip_joint"    gear="33.5"/>
    <motor name="RL_thigh"  joint="RL_thigh_joint"  gear="33.5"/>
    <motor name="RL_calf"   joint="RL_calf_joint"   gear="45.0"/>
    <motor name="RR_hip"    joint="RR_hip_joint"    gear="33.5"/>
    <motor name="RR_thigh"  joint="RR_thigh_joint"  gear="33.5"/>
    <motor name="RR_calf"   joint="RR_calf_joint"   gear="45.0"/>
  </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Gravity projection  (mirrors system/policy/obs_builder._project_gravity)
# ---------------------------------------------------------------------------

def _project_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity (0, 0, -1) into the body frame.

    Parameters
    ----------
    quat_wxyz:
        Unit quaternion in MuJoCo convention [w, x, y, z].

    Returns
    -------
    float32 vector of shape (3,).
    """
    w, x, y, z = quat_wxyz.astype(float)
    # Body-to-world rotation matrix R: v_world = R @ v_body
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # Gravity in body frame: R^T @ g_world
    return (R.T @ gravity_world).astype(np.float32)


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> Tuple[float, float]:
    """Return roll/pitch angles from quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz.astype(float)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


def _normalize_term_values(
    values: Optional[Dict[str, float]],
    defaults: Dict[str, float],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    source = values if isinstance(values, dict) and values else defaults
    for key, value in (source or {}).items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def resolve_task_commands(task_config) -> np.ndarray:
    """Return the effective target command vector [vx, vy, yaw_rate]."""
    if task_config is None:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return np.array(
        [
            float(getattr(task_config, "target_vx", 0.5) or 0.0),
            float(getattr(task_config, "target_vy", 0.0) or 0.0),
            float(getattr(task_config, "target_wz", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )


def resolve_obs_components(obs_action_config) -> List[str]:
    if obs_action_config is None:
        return [
            "base_angular_velocity",
            "projected_gravity",
            "joint_positions",
            "joint_velocities",
            "last_action",
            "velocity_command",
        ]
    raw = getattr(obs_action_config, "obs_components", None)
    if isinstance(raw, (list, tuple)):
        components = [str(item).strip() for item in raw if str(item).strip()]
        if components:
            return components
    return [
        "base_angular_velocity",
        "projected_gravity",
        "joint_positions",
        "joint_velocities",
        "last_action",
        "velocity_command",
    ]


def resolve_obs_dim_from_components(
    components: List[str],
    num_joints: int,
    action_dim: int,
    frame_stack: int = 1,
) -> int:
    total = 0
    for name in components:
        dim = _OBS_COMPONENT_DIM_ALIASES.get(str(name).strip().lower())
        if dim == "num_joints":
            total += int(num_joints)
        elif dim == "action_dim":
            total += int(action_dim)
        elif isinstance(dim, int):
            total += dim
    return total * max(1, int(frame_stack))


def resolve_action_clip_range(obs_action_config, env_config) -> float:
    value = None
    if obs_action_config is not None:
        value = getattr(obs_action_config, "action_clip", None)
    if value in (None, 0, 0.0):
        value = getattr(env_config, "action_clip_range", None) if env_config is not None else None
    try:
        return max(1e-6, float(value))
    except (TypeError, ValueError):
        return 1.0


def _normalize_joint_config(joint_config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    if not isinstance(joint_config, dict):
        return normalized
    for key, raw in joint_config.items():
        joint_name = str(key).strip()
        if not joint_name:
            continue
        entry: Dict[str, float] = {}
        if isinstance(raw, (int, float)):
            entry["qpos"] = float(raw)
        elif isinstance(raw, dict):
            for src_key, dst_key in (
                ("qpos", "qpos"),
                ("default_qpos", "qpos"),
                ("position", "qpos"),
                ("damping", "damping"),
                ("armature", "armature"),
            ):
                if src_key in raw:
                    try:
                        entry[dst_key] = float(raw[src_key])
                    except (TypeError, ValueError):
                        pass
        if entry:
            normalized[joint_name] = entry
    return normalized


# ---------------------------------------------------------------------------
# UnitreeGymEnv
# ---------------------------------------------------------------------------

class UnitreeGymEnv(gymnasium.Env):
    """
    Headless MuJoCo-based Gymnasium environment for Unitree Go2.

    Observation layout (45-dim) matches system.policy.obs_builder DEFAULT_COMPONENT_ORDER:
        base_angular_velocity  3
        projected_gravity      3
        joint_positions        12
        joint_velocities       12
        last_action            12
        velocity_command        3
                                45

    Action space: Box(-1, 1, (12,)) 闂?normalised joint torques.

    Parameters
    ----------
    obs_dim:
        Observation dimension; must be 45 unless a custom component order is used.
    action_dim:
        Action dimension; must equal the number of actuators.
    max_episode_steps:
        Episode truncation length (steps at ``sim_dt``).
    mjcf_path:
        Optional path to an MJCF XML file.  If None or non-existent the
        built-in go2.xml is used; if that is also absent a minimal
        12-DOF fallback model is loaded from string.
    commands:
        Target velocity command [vx, vy, yaw_rate].  Defaults to forward walk.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        max_episode_steps: int = 1_000,
        mjcf_path: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        sim_dt: Optional[float] = None,
        control_dt: Optional[float] = None,
        gravity_z: Optional[float] = None,
        commands: Optional[np.ndarray] = None,
        reward_terms: Optional[Dict[str, float]] = None,
        termination_conditions: Optional[Dict[str, float]] = None,
        obs_components: Optional[List[str]] = None,
        frame_stack: int = 1,
        action_type: str = "joint_position",
        action_scale: float = 1.0,
        action_clip: float = 1.0,
        joint_config: Optional[Dict[str, Any]] = None,
        domain_rand_config: Optional[Any] = None,
        command_mode: str = "fixed",
        curriculum: bool = False,
        success_threshold: float = 0.8,
        truncation_max_steps: int = 0,
        curriculum_schedule: Optional[Dict[str, Any]] = None,
        reference_motion_config: Optional[Any] = None,
        init_pose_config: Optional[Any] = None,
        gated_reward_config: Optional[Any] = None,
    ) -> None:
        import mujoco

        self._act_dim    = action_dim
        self._obs_components = [str(item).strip() for item in (obs_components or []) if str(item).strip()]
        if not self._obs_components:
            self._obs_components = resolve_obs_components(None)
        self._frame_stack = max(1, int(frame_stack))
        self._single_obs_dim = resolve_obs_dim_from_components(
            self._obs_components,
            num_joints=self._act_dim,
            action_dim=self._act_dim,
            frame_stack=1,
        )
        self._obs_dim = int(obs_dim or (self._single_obs_dim * self._frame_stack))
        self._action_type = str(action_type or "joint_position").strip().lower()
        self._action_scale = float(action_scale or 1.0)
        self._action_clip = max(1e-6, float(action_clip or 1.0))
        self._joint_config = _normalize_joint_config(joint_config)
        self._domain_rand = domain_rand_config
        self._init_pose_config = init_pose_config
        self._command_mode = str(command_mode or "fixed").strip().lower()
        self._curriculum = bool(curriculum)
        self._success_threshold = float(success_threshold or 0.8)
        self._truncation_max_steps = max(0, int(truncation_max_steps or 0))
        self._curriculum_schedule = dict(curriculum_schedule or {})
        self._max_steps  = (
            self._truncation_max_steps
            if self._truncation_max_steps > 0
            else int(max_episode_steps)
        )
        self._step_count = 0
        self._episode_index = 0
        self._success_streak = 0
        self._reward_terms = _normalize_term_values(
            reward_terms,
            default_reward_terms(),
        )
        self._termination_conditions = _normalize_term_values(
            termination_conditions,
            default_termination_conditions(),
        )

        # ── Gated reward curriculum state ─────────────────────────────────────
        self._gated_config = gated_reward_config   # Optional[GatedRewardConfig]
        self._base_reward_terms: Dict[str, float] = dict(self._reward_terms)
        _grc_window = (
            int(getattr(gated_reward_config, "ep_reward_window", 20) or 20)
            if gated_reward_config is not None else 20
        )
        self._global_step: int = 0
        self._ep_rewards: Deque[float] = deque(maxlen=_grc_window)
        self._best_ep_reward: float = -float("inf")
        self._current_stage_idx: int = 0
        self._blend_origin_weights: Dict[str, float] = {}
        self._blend_start_step: int = -1
        self._current_ep_reward: float = 0.0
        # ── Stage-local episode tracking for plateau detection ─────────────
        self._stage_ep_rewards: List[float] = []   # rewards since stage start
        self._stage_ep_count: int = 0              # episodes since stage start

        # Load MuJoCo model
        # Priority: scene_xml_path > mjcf_path > auto-detect scene.xml > _MINIMAL_MJCF
        resolved_path = None
        if scene_xml_path and os.path.exists(scene_xml_path):
            resolved_path = scene_xml_path
        elif mjcf_path and os.path.exists(mjcf_path):
            resolved_path = mjcf_path
        else:
            resolved_path = _find_go2_scene_xml("flat")  # default: flat scene

        if resolved_path:
            self._model = mujoco.MjModel.from_xml_path(resolved_path)
        else:
            self._model = mujoco.MjModel.from_xml_string(_MINIMAL_MJCF)

        try:
            from src.system.training.training_config import resolve_control_timing
            timing = resolve_control_timing(sim_dt, control_dt)
        except Exception:
            timing = {
                "sim_dt": 0.002,
                "control_dt": 0.02,
                "decimation": 10,
                "control_frequency_hz": 50.0,
                "timing_exact": True,
            }

        self._sim_dt = float(timing["sim_dt"])
        self._control_dt = float(timing["control_dt"])
        self._decimation = int(timing["decimation"])
        self.control_frequency_hz = float(timing["control_frequency_hz"])

        try:
            self._model.opt.timestep = self._sim_dt
        except Exception:
            pass

        if gravity_z is not None:
            try:
                self._model.opt.gravity[2] = float(gravity_z)
            except Exception:
                pass

        self._data = mujoco.MjData(self._model)

        # ── Build qpos→ctrl permutation index ────────────────────────
        # The MJCF may define joints and actuators in different order
        # (e.g. Go2: qpos = FL,FR,RL,RR but ctrl = FR,FL,RR,RL).
        # _ctrl_to_qpos_idx[actuator_i] = qpos joint index (relative to
        # the joint block, i.e. 0-based from qpos[7] for free-joint).
        # This lets _action_to_ctrl correctly route each actuator's PD
        # error from the matching joint.
        _jnt_offset = 0
        for _ji in range(self._model.njnt):
            if self._model.jnt_type[_ji] == 0:  # mjJNT_FREE
                _jnt_offset = 7
                break
        n_ctrl = min(self._act_dim, self._model.nu)
        self._ctrl_to_qpos_idx = np.arange(n_ctrl, dtype=np.intp)
        for _ai in range(n_ctrl):
            _trnid = int(self._model.actuator_trnid[_ai, 0])
            _qadr = int(self._model.jnt_qposadr[_trnid])
            self._ctrl_to_qpos_idx[_ai] = _qadr - _jnt_offset

        # ── Default standing pose (qpos order, full joint block) ─────
        # Covers ALL joints in the model (not just act_dim), so that
        # joints beyond the policy's control range are held at the
        # correct default pose by _action_to_ctrl's PD controller.
        # The first act_dim entries are also used as the residual
        # action baseline.
        _n_all_joints = max(self._act_dim, self._model.nq - _jnt_offset)
        self._default_qpos_full = np.zeros(_n_all_joints, dtype=np.float32)
        _got_keyframe = False
        if self._model.nkey > 0:
            try:
                kf_qpos = np.asarray(self._model.key_qpos[0], dtype=np.float32)
                n_kf = min(_n_all_joints, len(kf_qpos) - _jnt_offset)
                if n_kf > 0:
                    self._default_qpos_full[:n_kf] = kf_qpos[_jnt_offset: _jnt_offset + n_kf]
                    _got_keyframe = True
            except Exception:
                pass
        if not _got_keyframe:
            n_def = min(_n_all_joints, len(GO2_DEFAULT_QPOS))
            self._default_qpos_full[:n_def] = GO2_DEFAULT_QPOS[:n_def]
        # Convenience view: policy-controlled subset
        self._default_qpos = self._default_qpos_full[:self._act_dim]

        # ── Per-joint torque limits (from MJCF actuator ctrl range) ────
        # Replaces the hardcoded GO2-only _TORQUE_LIMITS for multi-robot support.
        # Only consider actuators whose target joint falls within act_dim.
        self._torque_limits = np.full(self._act_dim, 33.5, dtype=np.float32)
        for _ai in range(n_ctrl):
            qi = int(self._ctrl_to_qpos_idx[_ai])
            if 0 <= qi < self._act_dim:
                cr = self._model.actuator_ctrlrange[_ai]
                self._torque_limits[qi] = max(abs(float(cr[0])), abs(float(cr[1])), 1.0)

        self.joint_names = _extract_joint_names_from_model(self._model, self._act_dim)
        self._joint_names = list(self.joint_names)
        self._renderer = None
        self._frame_history: Deque[np.ndarray] = deque(maxlen=self._frame_stack)
        self._base_body_id = None
        try:
            self._base_body_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
        except Exception:
            self._base_body_id = 1 if self._model.nbody > 1 else 0
        self._foot_probe_body_ids = self._resolve_probe_body_ids(
            preferred_keywords=("foot", "toe", "ankle", "wheel", "calf"),
        )

        # Velocity command [vx, vy, yaw_rate]
        if commands is not None:
            self._base_commands = np.array(commands, dtype=np.float32).flatten()[:3]
        else:
            self._base_commands = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        self._commands = self._base_commands.copy()
        self._base_body_mass = np.array(self._model.body_mass, copy=True)
        self._base_geom_friction = np.array(self._model.geom_friction, copy=True)
        self._base_dof_damping = np.array(self._model.dof_damping, copy=True)
        self._base_actuator_gear = np.array(self._model.actuator_gear, copy=True)
        self._pending_external_force = np.zeros(6, dtype=np.float64)

        self._prev_action = np.zeros(action_dim, dtype=np.float32)
        # Per-step reward component snapshot (raw scores, un-weighted) — written
        # by _compute_reward() and exposed via step() info dict.
        self._last_reward_components: Dict[str, float] = {}

        # Foot air-time tracking (per-foot consecutive steps in swing phase)
        n_feet = len(self._foot_probe_body_ids)
        self._foot_air_steps: List[int] = [0] * n_feet
        self._foot_was_in_air: List[bool] = [False] * n_feet

        # Nominal foot positions for foot_pos_tracking (no-reference fallback).
        # Captured from the default MuJoCo model pose via a single mj_forward call.
        self._nominal_foot_positions: Optional[np.ndarray] = None
        if self._foot_probe_body_ids:
            try:
                mujoco.mj_forward(self._model, self._data)
                self._nominal_foot_positions = np.array(
                    [self._data.xpos[bid].copy() for bid in self._foot_probe_body_ids],
                    dtype=np.float32,
                )
            except Exception:
                pass

        # Derived reference-motion arrays (populated by _init_reference_motion).
        self._ref_joint_vels: Optional[np.ndarray] = None    # (T, act_dim) finite-diff velocities
        self._ref_foot_positions: Optional[np.ndarray] = None  # (T, n_feet, 3) FK foot positions

        # Reference motion tracking (Plan A motion imitation)
        self._ref_frames: Optional[np.ndarray] = None
        self._ref_num_frames: int = 0
        self._ref_phase_f: float = 0.0   # float phase counter (supports resampling)
        self._ref_phase_step: float = 1.0  # frames to advance per control step
        self._ref_phase_mode: str = "loop"
        self._ref_tracking_weight: float = 1.0
        self._ref_tracking_sigma: float = 5.0
        self._ref_random_start: bool = False
        if reference_motion_config is not None:
            self._init_reference_motion(reference_motion_config)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self._action_clip,
            high=self._action_clip,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Needed by gymnasium.utils.env_checker
        self.render_mode = None

    def _resolve_probe_body_ids(
        self,
        preferred_keywords: Tuple[str, ...],
    ) -> List[int]:
        try:
            import mujoco
        except Exception:
            return []

        matches: List[int] = []
        fallback: List[int] = []
        for body_id in range(1, int(getattr(self._model, "nbody", 0) or 0)):
            try:
                body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            except Exception:
                body_name = None
            name = str(body_name or "").strip().lower()
            if not name:
                continue
            if any(token in name for token in preferred_keywords):
                matches.append(body_id)
            elif any(token in name for token in ("shank", "leg", "lower", "end", "tip")):
                fallback.append(body_id)
        return matches or fallback

    # ------------------------------------------------------------------
    # Init pose
    # ------------------------------------------------------------------

    def _apply_init_pose(self, rng: np.random.Generator) -> None:
        """
        Place the robot in its starting pose according to _init_pose_config.

        Called at the top of reset() before mj_forward.  Falls back to the
        classic GO2_DEFAULT_QPOS + noise behaviour when no config is provided
        or when the selected mode cannot be fulfilled.
        """
        if self._model.nq < 19:
            return  # model too small / not a legged robot

        cfg = self._init_pose_config
        mode = str(getattr(cfg, "mode", "default") or "default").strip() if cfg is not None else "default"
        noise_scale = float(getattr(cfg, "noise_scale", 0.05) or 0.05) if cfg is not None else 0.05
        base_height_override = float(getattr(cfg, "base_height", -1.0) if cfg is not None else -1.0)

        n_joints = min(self._act_dim, self._model.nq - 7)
        target_joints: Optional[np.ndarray] = None
        target_height: Optional[float] = None

        # ── mode: reference_frame_0 ────────────────────────────────────────
        if mode == "reference_frame_0":
            rsi_prob = float(getattr(cfg, "rsi_prob", 1.0) if cfg is not None else 1.0)
            use_ref = rsi_prob >= 1.0 or float(rng.random()) < rsi_prob
            if use_ref and self._ref_frames is not None and len(self._ref_frames) > 0:
                sample_mode = str(
                    getattr(cfg, "rsi_sample_mode", "frame_0") if cfg is not None else "frame_0"
                ).strip().lower()
                if sample_mode == "uniform_phase":
                    idx = int(rng.integers(0, len(self._ref_frames)))
                else:
                    idx = 0
                frame = self._ref_frames[idx]
                n = min(n_joints, len(frame))
                target_joints = np.zeros(n_joints, dtype=np.float32)
                target_joints[:n] = frame[:n]
            # When the coin flip rejects RSI, target_joints stays None
            # and the fallback section below will use default_qpos.

        # ── mode: keyframe ─────────────────────────────────────────────────
        elif mode == "keyframe":
            keyframe_name = str(getattr(cfg, "keyframe_name", "home") or "home").strip()
            try:
                # MuJoCo >= 3.0: model.keyframe is a named collection
                kf_id = -1
                for k in range(int(getattr(self._model, "nkey", 0) or 0)):
                    name_bytes = self._model.key(k).name
                    name_str = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)
                    if name_str == keyframe_name:
                        kf_id = k
                        break
                if kf_id >= 0:
                    kf_qpos = np.asarray(self._model.key_qpos[kf_id], dtype=np.float32)
                    # Full qpos from keyframe — extract joint block
                    if len(kf_qpos) >= 7 + n_joints:
                        target_joints = kf_qpos[7: 7 + n_joints].copy()
                        if base_height_override < 0.0:
                            target_height = float(kf_qpos[2])
                    elif len(kf_qpos) >= n_joints:
                        target_joints = kf_qpos[:n_joints].copy()
            except Exception:
                pass  # keyframe lookup failed → fallback to default

        # ── mode: custom ───────────────────────────────────────────────────
        elif mode == "custom":
            custom_list = getattr(cfg, "custom_qpos", []) or []
            if isinstance(custom_list, (list, np.ndarray)) and len(custom_list) > 0:
                arr = np.asarray(custom_list, dtype=np.float32)
                target_joints = np.zeros(n_joints, dtype=np.float32)
                n = min(n_joints, len(arr))
                target_joints[:n] = arr[:n]

        # ── fallback / default ─────────────────────────────────────────────
        # (also covers mode == "default" and any failed modes above)
        if target_joints is None:
            target_joints = self._default_qpos[:n_joints].copy()

        # Apply noise
        if noise_scale > 0.0:
            noise = rng.normal(0.0, noise_scale, n_joints).astype(np.float32)
            target_joints = target_joints + noise

        # Base height: use keyframe height if available, else Go2 default
        _kf_height = None
        if self._model.nkey > 0:
            try:
                _kf_height = float(self._model.key_qpos[0][2])
            except Exception:
                pass
        height = target_height if target_height is not None else (_kf_height if _kf_height is not None else 0.32)
        if base_height_override >= 0.0:
            height = base_height_override

        self._data.qpos[2] = height
        self._data.qpos[7: 7 + n_joints] = target_joints

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        import mujoco

        # Let gymnasium.Env initialise self.np_random from seed
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        self._episode_index += 1
        self._success_streak = 0
        self._apply_joint_config()
        self._apply_domain_randomization(rng)
        self._commands = self._resolve_episode_commands(rng)

        self._apply_init_pose(rng)

        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        if self._ref_num_frames > 0 and self._ref_random_start:
            self._ref_phase_f = float(rng.integers(0, self._ref_num_frames))
        else:
            self._ref_phase_f = 0.0
        self._prev_action = np.zeros(self._act_dim, dtype=np.float32)
        self._frame_history.clear()
        n_feet = len(self._foot_probe_body_ids)
        self._foot_air_steps = [0] * n_feet
        self._foot_was_in_air = [False] * n_feet

        obs = self._get_obs()
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        action = np.clip(
            np.asarray(action, dtype=np.float32).flatten(),
            -self._action_clip, self._action_clip,
        )
        if self._action_scale != 1.0:
            action = action * np.float32(self._action_scale)
        prev_action = self._prev_action.copy()

        # Translate the configured action representation into actuator ctrl.
        n_act = min(self._act_dim, self._model.nu)
        self._data.ctrl[:n_act] = self._action_to_ctrl(action[:n_act]).astype(np.float64)

        for _ in range(self._decimation):
            self._apply_pending_push()
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        self._global_step += 1
        self._ref_phase_f += self._ref_phase_step

        # ── Gated reward: resolve active weights for this step ────────────
        if self._gated_config is not None:
            try:
                _gated_valid = bool(self._gated_config.is_valid())
            except Exception:
                _gated_valid = False
            if _gated_valid:
                self._reward_terms = self._resolve_reward_weights()

        reward     = float(self._compute_reward(action, prev_action))
        self._current_ep_reward += reward
        self._prev_action = action.copy()
        obs        = self._get_obs()
        terminated = self._is_terminated()
        task_success = self._task_success_score() >= self._success_threshold
        self._success_streak = self._success_streak + 1 if task_success else 0
        truncated  = self._step_count >= self._max_steps
        if self._success_streak >= max(10, self._decimation):
            truncated = True

        # ── Episode end: record reward for gated stage tracking ───────────
        if terminated or truncated:
            self._ep_rewards.append(self._current_ep_reward)
            self._stage_ep_rewards.append(self._current_ep_reward)
            self._stage_ep_count += 1
            if self._current_ep_reward > self._best_ep_reward:
                self._best_ep_reward = self._current_ep_reward
            self._current_ep_reward = 0.0

        return obs, reward, terminated, truncated, {
            "task_success": task_success,
            "command": self._commands.copy(),
            "reward_components": self._last_reward_components.copy(),
        }

    def render(self) -> None:
        """Headless env rendering path; offscreen when available."""
        self.render_frame()

    def close(self) -> None:
        """Release any held resources."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        self._renderer = None

    def render_frame(self) -> Optional[np.ndarray]:
        """Render the current frame offscreen when MuJoCo rendering is available."""
        try:
            import mujoco
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, 640, 480)
            self._renderer.update_scene(self._data)
            return np.asarray(self._renderer.render(), dtype=np.uint8)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector from configured components and frame stack."""
        single = self._build_single_frame_obs()
        if single.shape[0] != self._single_obs_dim:
            single = np.resize(single.astype(np.float32), self._single_obs_dim)
        while len(self._frame_history) < self._frame_stack:
            self._frame_history.append(np.zeros(self._single_obs_dim, dtype=np.float32))
        self._frame_history.append(single.astype(np.float32))
        stacked = np.concatenate(list(self._frame_history), dtype=np.float32)
        if stacked.shape[0] != self._obs_dim:
            if stacked.shape[0] < self._obs_dim:
                stacked = np.pad(stacked, (0, self._obs_dim - stacked.shape[0]))
            else:
                stacked = stacked[: self._obs_dim]
        obs_noise_std = 0.0
        if self._domain_rand is not None:
            try:
                obs_noise_std = float(getattr(self._domain_rand, "obs_noise_std", 0.0) or 0.0)
            except Exception:
                obs_noise_std = 0.0
        if obs_noise_std > 0:
            stacked = stacked + np.random.normal(0.0, obs_noise_std, size=stacked.shape).astype(np.float32)
        return stacked.astype(np.float32)

    def _build_single_frame_obs(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        for name in self._obs_components:
            component = self._build_obs_component(name)
            if component.size > 0:
                parts.append(component.astype(np.float32))
        if not parts:
            return np.zeros(self._single_obs_dim, dtype=np.float32)
        return np.concatenate(parts, dtype=np.float32)

    def _build_obs_component(self, name: str) -> np.ndarray:
        key = str(name).strip().lower()
        if key in ("base_angular_velocity", "base_ang_vel"):
            return np.asarray(self._data.qvel[3:6], dtype=np.float32)
        if key in ("base_linear_velocity", "base_lin_vel"):
            return np.asarray(self._data.qvel[0:3], dtype=np.float32)
        if key == "imu":
            return np.concatenate([
                np.asarray(self._data.qvel[3:6], dtype=np.float32),
                self._projected_gravity(),
            ], dtype=np.float32)
        if key in ("projected_gravity", "gravity_vec"):
            return self._projected_gravity()
        if key in ("joint_positions", "joint_pos"):
            end = min(self._model.nq, 7 + self._act_dim)
            values = np.asarray(self._data.qpos[7:end], dtype=np.float32)
            if values.shape[0] < self._act_dim:
                values = np.pad(values, (0, self._act_dim - values.shape[0]))
            return values[: self._act_dim]
        if key in ("joint_velocities", "joint_vel"):
            end = min(self._model.nv, 6 + self._act_dim)
            values = np.asarray(self._data.qvel[6:end], dtype=np.float32)
            if values.shape[0] < self._act_dim:
                values = np.pad(values, (0, self._act_dim - values.shape[0]))
            return values[: self._act_dim]
        if key in ("last_action", "previous_action"):
            return self._prev_action.copy().astype(np.float32)
        if key == "velocity_command":
            return self._commands.copy().astype(np.float32)
        if key == "command":
            return np.array([self._commands[0], self._commands[1], self._commands[2], 0.0], dtype=np.float32)
        if key in ("phase_sin_cos", "phase"):
            # 2-D phase encoding: [sin(2π * phase_f / T), cos(2π * phase_f / T)]
            # Provides the policy a smooth, periodic clock signal so the MDP
            # stays Markovian when a reference trajectory is active.
            # When no reference motion is loaded, T defaults to 1 (constant 0).
            T = float(self._ref_num_frames) if self._ref_num_frames > 0 else 1.0
            angle = 2.0 * np.pi * (self._ref_phase_f / T)
            return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

        # ── Reference motion observations (DeepMimic / AMP standard) ─────
        # The policy must *see* the reference target to imitate it.  Without
        # these components the agent can only infer the target indirectly
        # from the reward signal — catastrophically inefficient in 12-D
        # continuous action space.
        if key in ("reference_joint_positions", "ref_joint_pos"):
            return self._get_ref_joint_positions()
        if key in ("reference_joint_velocities", "ref_joint_vel"):
            return self._get_ref_joint_velocities()

        return np.zeros(0, dtype=np.float32)

    def _projected_gravity(self) -> np.ndarray:
        quat = np.array(self._data.qpos[3:7], dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            quat = quat / norm
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return _project_gravity(quat)

    def _current_ref_frame_index(self) -> int:
        """Return the reference frame index for the current phase."""
        if self._ref_num_frames <= 0:
            return 0
        phase_int = int(self._ref_phase_f)
        if self._ref_phase_mode == "loop":
            return phase_int % self._ref_num_frames
        return min(phase_int, self._ref_num_frames - 1)

    def _get_ref_joint_positions(self) -> np.ndarray:
        """Current reference frame joint positions (num_joints,).

        Returns the reference keyframe joint angles for the current phase.
        When no reference is loaded, returns the model's default standing pose
        so the observation dimension stays consistent.
        """
        if self._ref_frames is not None:
            idx = self._current_ref_frame_index()
            ref = self._ref_frames[idx]
            n = min(self._act_dim, len(ref))
            out = np.zeros(self._act_dim, dtype=np.float32)
            out[:n] = ref[:n]
            return out
        return self._default_qpos.copy()

    def set_default_qpos(self, values_in_qpos_order: np.ndarray) -> None:
        """Override the env's reset/standing pose for the joint block.

        Public contract used by ``PolicyRunner._align_env_default_pose``
        when an IL bundle ships its own training-time standing pose.
        ``values_in_qpos_order`` is the per-joint default vector aligned
        to MuJoCo's ``qpos[7:]`` slot order (NOT bundle order — the
        permutation is the caller's responsibility, since only the
        caller knows the bundle joint space).

        This is the supported way for an out-of-process loader to set
        the standing pose. Reaching directly into ``_default_qpos`` /
        ``_default_qpos_full`` still works for legacy callers but is no
        longer the recommended path; new code should call this method
        so that any future changes to the internal storage layout
        remain transparent.
        """
        try:
            arr = np.asarray(values_in_qpos_order, dtype=np.float32).flatten()
        except Exception:
            return
        if arr.size == 0:
            return
        # _default_qpos_full is the full joint block (length n_qpos);
        # _default_qpos is the policy-controlled prefix (length act_dim).
        # We update both so reset() and any internal helper that reads
        # either field stays consistent.
        existing_full = getattr(self, "_default_qpos_full", None)
        if existing_full is not None:
            full = np.asarray(existing_full, dtype=np.float32).copy()
            n = min(arr.shape[0], full.shape[0])
            full[:n] = arr[:n]
            self._default_qpos_full = full
            self._default_qpos = full[:int(self._act_dim)].astype(np.float32)
        else:
            n = min(arr.shape[0], int(self._act_dim))
            new_sub = np.zeros(int(self._act_dim), dtype=np.float32)
            new_sub[:n] = arr[:n]
            self._default_qpos = new_sub

    def _get_ref_joint_velocities(self) -> np.ndarray:
        """Current reference frame joint velocities (num_joints,).

        Finite-difference velocities precomputed in ``_init_reference_motion``.
        When no reference is loaded, returns zeros (standing = zero velocity).
        """
        if self._ref_joint_vels is not None:
            idx = self._current_ref_frame_index()
            ref = self._ref_joint_vels[idx]
            n = min(self._act_dim, len(ref))
            out = np.zeros(self._act_dim, dtype=np.float32)
            out[:n] = ref[:n]
            return out
        return np.zeros(self._act_dim, dtype=np.float32)

    def get_reference_action_for_bc(self) -> np.ndarray:
        """Return the reference joint position target as a normalised action.

        Used by the behavioral cloning pipeline to construct (obs, action_ref)
        supervision pairs.  The returned action is in the same space as the
        policy output: residual from GO2_DEFAULT_QPOS, divided by action_scale.

        For ``joint_position`` action type (default):
            action_ref = (q_ref - q_default) / action_scale

        For ``torque`` / ``joint_velocity`` types a PD-derived approximation is
        used since direct torque supervision from kinematic reference is
        ill-defined.
        """
        ref_pos = self._get_ref_joint_positions()
        default_pos = self._default_qpos

        if self._action_type == "joint_position":
            # Residual action: inverse of _action_to_ctrl's pos_target = default + action
            raw_action = ref_pos - default_pos
            if self._action_scale != 0.0:
                raw_action = raw_action / np.float32(self._action_scale)
            return np.clip(raw_action, -self._action_clip, self._action_clip)

        if self._action_type == "joint_velocity":
            # Use reference velocity directly, normalised
            ref_vel = self._get_ref_joint_velocities()
            if self._action_scale != 0.0:
                ref_vel = ref_vel / np.float32(self._action_scale)
            return np.clip(ref_vel, -self._action_clip, self._action_clip)

        # torque mode: PD-derived target torque from position error
        qpos_end = min(self._model.nq, 7 + self._act_dim)
        cur_pos = np.asarray(self._data.qpos[7:qpos_end], dtype=np.float32)
        if cur_pos.shape[0] < self._act_dim:
            cur_pos = np.pad(cur_pos, (0, self._act_dim - cur_pos.shape[0]))
        qvel_end = min(self._model.nv, 6 + self._act_dim)
        cur_vel = np.asarray(self._data.qvel[6:qvel_end], dtype=np.float32)
        if cur_vel.shape[0] < self._act_dim:
            cur_vel = np.pad(cur_vel, (0, self._act_dim - cur_vel.shape[0]))
        ref_vel = self._get_ref_joint_velocities()
        desired_torque = _PD_KP * (ref_pos - cur_pos) + _PD_KD * (ref_vel - cur_vel)
        normalised = desired_torque[:self._act_dim] / self._torque_limits
        return np.clip(normalised, -self._action_clip, self._action_clip)

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Translate a normalised network action into MuJoCo ``ctrl`` values.

        The policy's action space is in **qpos joint order** (FL, FR, RL, RR
        for Go2).  Each ``ctrl[i]`` drives ``actuator[i]`` which may map to a
        different qpos joint (e.g. Go2 actuators are FR, FL, RR, RL).
        ``_ctrl_to_qpos_idx`` provides the correct per-actuator mapping.

        All position/velocity PD paths divide by the actuator gear ratio so
        that the *effective* joint-space gains stay at the nominal values
        (Kp = 20 Nm/rad, Kd = 0.5 Nm·s/rad) regardless of the gear in the
        MJCF.

        ``joint_position`` mode uses *residual* actions: the network output is
        added to the default standing pose (GO2_DEFAULT_QPOS) rather than
        being used as an absolute target.
        """
        n_act = min(len(action), self._model.nu)
        c2q = self._ctrl_to_qpos_idx   # actuator[i] → qpos joint index

        # Per-actuator gear (first column of the gear matrix, scalar per actuator).
        raw_gear = np.asarray(self._model.actuator_gear[:n_act, 0], dtype=np.float32)
        gear = np.where(raw_gear > 0.0, raw_gear, 1.0)

        # Full joint-block state — read ALL actuated joints, not just act_dim,
        # because c2q mapping may point to qpos indices beyond act_dim when
        # model.nu > act_dim (e.g. H1: 20 actuators, act_dim=10).
        n_qpos_joints = max(self._act_dim, self._model.nq - 7)
        n_qvel_joints = max(self._act_dim, self._model.nv - 6)
        joint_pos = np.asarray(self._data.qpos[7:7 + n_qpos_joints], dtype=np.float32)
        joint_vel = np.asarray(self._data.qvel[6:6 + n_qvel_joints], dtype=np.float32)

        # Policy action is in qpos order, dimensioned to act_dim
        act_arr = np.asarray(action[:self._act_dim], dtype=np.float32)
        # Full default pose for PD target (includes joints beyond act_dim)
        default_full = self._default_qpos_full[:n_qpos_joints]
        ctrl = np.zeros(n_act, dtype=np.float32)

        if self._action_type == "torque":
            for ai in range(n_act):
                qi = int(c2q[ai])
                if qi < self._act_dim:
                    tl = self._torque_limits[qi] if qi < len(self._torque_limits) else 33.5
                    ctrl[ai] = (act_arr[qi] * tl) / gear[ai]
                # else: actuator beyond policy control → ctrl stays 0
            return ctrl

        if self._action_type == "joint_velocity":
            for ai in range(n_act):
                qi = int(c2q[ai])
                if qi < self._act_dim:
                    ctrl[ai] = (_VEL_KP * (act_arr[qi] - joint_vel[qi])) / gear[ai]
            return ctrl

        # joint_position (default): residual from default standing pose.
        # pos_target covers the full joint block; joints beyond act_dim
        # are held at their default pose (no policy control).
        pos_target = default_full.copy()
        pos_target[:self._act_dim] += act_arr

        for ai in range(n_act):
            qi = int(c2q[ai])
            ctrl[ai] = (_PD_KP * (pos_target[qi] - joint_pos[qi])
                        - _PD_KD * joint_vel[qi]) / gear[ai]
        return ctrl

    def _apply_joint_config(self) -> None:
        if not self._joint_config:
            return
        for joint_id in range(self._model.njnt):
            try:
                import mujoco
                joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            except Exception:
                joint_name = None
            if not joint_name or joint_name not in self._joint_config:
                continue
            cfg = self._joint_config[joint_name]
            qpos_adr = int(self._model.jnt_qposadr[joint_id])
            dof_adr = int(self._model.jnt_dofadr[joint_id])
            if "qpos" in cfg and qpos_adr < self._model.nq:
                self._data.qpos[qpos_adr] = float(cfg["qpos"])
            if "damping" in cfg and dof_adr < self._model.nv:
                self._model.dof_damping[dof_adr] = float(cfg["damping"])
            if "armature" in cfg and dof_adr < self._model.nv:
                self._model.dof_armature[dof_adr] = float(cfg["armature"])

    def _apply_domain_randomization(self, rng: np.random.Generator) -> None:
        self._model.body_mass[:] = self._base_body_mass
        self._model.geom_friction[:] = self._base_geom_friction
        self._model.dof_damping[:] = self._base_dof_damping
        self._model.actuator_gear[:] = self._base_actuator_gear
        self._pending_external_force[:] = 0.0
        cfg = self._domain_rand
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return
        # Delayed DR activation (RSI-AMP staged training): skip DR
        # until the global step reaches rand_schedule_start_step.
        start_step = int(getattr(cfg, "rand_schedule_start_step", 0) or 0)
        if start_step > 0 and self._global_step < start_step:
            return

        # Linear intensity ramp: when rand_schedule == "linear", DR
        # range scales from 0→1 between start_step and end_step.
        sched = str(getattr(cfg, "rand_schedule", "none") or "none").strip().lower()
        dr_intensity = 1.0
        if sched == "linear":
            end_step = int(getattr(cfg, "rand_schedule_end_step", 500000) or 500000)
            span = max(1, end_step - start_step)
            dr_intensity = min(1.0, max(0.0,
                (self._global_step - start_step) / float(span)))

        def _sample_range(value, default_low=1.0, default_high=1.0):
            try:
                low = float(value[0])
                high = float(value[1])
            except Exception:
                low, high = default_low, default_high
            # Apply intensity: lerp the range toward [1.0, 1.0] (neutral)
            # when dr_intensity < 1.0.
            if dr_intensity < 1.0:
                mid = (low + high) * 0.5
                neutral = 1.0
                low = neutral + (low - neutral) * dr_intensity
                high = neutral + (high - neutral) * dr_intensity
            return float(rng.uniform(low, high))

        self._model.body_mass[:] = self._base_body_mass * _sample_range(getattr(cfg, "mass_range", [1.0, 1.0]))
        self._model.geom_friction[:, :] = self._base_geom_friction * _sample_range(getattr(cfg, "friction_range", [1.0, 1.0]))
        self._model.dof_damping[:] = self._base_dof_damping * _sample_range(getattr(cfg, "joint_damping_range", [1.0, 1.0]))
        self._model.actuator_gear[:, :] = self._base_actuator_gear * _sample_range(getattr(cfg, "motor_strength_range", [1.0, 1.0]))
        if bool(getattr(cfg, "push_robot", False)):
            interval = max(1, int(getattr(cfg, "push_interval_steps", 200) or 200))
            if self._episode_index % interval == 0:
                force_mag = _sample_range(getattr(cfg, "push_force_range", [0.0, 0.0]), 0.0, 0.0)
                direction = rng.normal(size=2).astype(np.float64)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-8:
                    direction = direction / norm
                self._pending_external_force[:2] = direction * force_mag

    def _apply_pending_push(self) -> None:
        if not np.any(self._pending_external_force):
            return
        try:
            self._data.xfrc_applied[self._base_body_id, :6] = self._pending_external_force
        except Exception:
            return
        self._pending_external_force[:] = 0.0

    def _resolve_episode_commands(self, rng: np.random.Generator) -> np.ndarray:
        base = self._base_commands.copy()
        if self._curriculum:
            ramp = max(1, int(self._curriculum_schedule.get("ramp_episodes", 100) or 100))
            start_scale = float(self._curriculum_schedule.get("start_scale", 0.25) or 0.25)
            end_scale = float(self._curriculum_schedule.get("end_scale", 1.0) or 1.0)
            alpha = min(1.0, self._episode_index / float(ramp))
            scale = start_scale + (end_scale - start_scale) * alpha
            base = base * np.float32(scale)
        if self._command_mode not in ("fixed", "", "none"):
            high = np.maximum(np.abs(base), np.array([0.2, 0.2, 0.2], dtype=np.float32))
            base = rng.uniform(-high, high).astype(np.float32)
        return base.astype(np.float32)

    def _task_success_score(self) -> float:
        lin_vel_x = float(self._data.qvel[0]) if self._model.nv > 0 else 0.0
        lin_vel_y = float(self._data.qvel[1]) if self._model.nv > 1 else 0.0
        ang_vel_z = float(self._data.qvel[5]) if self._model.nv > 5 else 0.0
        vx_tgt, vy_tgt, yaw_tgt = map(float, self._commands[:3])
        r_vx = float(np.exp(-((lin_vel_x - vx_tgt) ** 2) / 0.25))
        r_vy = float(np.exp(-((lin_vel_y - vy_tgt) ** 2) / 0.25))
        r_yaw = float(np.exp(-((ang_vel_z - yaw_tgt) ** 2) / 0.25))
        return (r_vx + r_vy + r_yaw) / 3.0

    # ------------------------------------------------------------------
    # Reference motion
    # ------------------------------------------------------------------

    def _init_reference_motion(self, cfg: Any) -> None:
        """Load and validate the reference keyframe array from *cfg*."""
        motion_file = str(getattr(cfg, "motion_file", "") or "")

        # If motion_file is empty, try resolving motion_source via the bridge
        if not motion_file:
            motion_source = str(getattr(cfg, "motion_source", "") or "")
            if motion_source:
                motion_file = self._resolve_motion_source_fallback(motion_source)
        if not motion_file:
            return
        try:
            frames = np.load(motion_file)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"ReferenceMotion: could not load '{motion_file}': {exc}. "
                "Reference tracking reward will be disabled.",
                stacklevel=4,
            )
            return

        # Accept (T, num_joints) or (T, full_qpos) where full_qpos >= 7+num_joints
        if frames.ndim != 2:
            import warnings
            warnings.warn(
                f"ReferenceMotion: expected 2-D array (T, joints), got shape {frames.shape}. "
                "Reference tracking reward will be disabled.",
                stacklevel=4,
            )
            return

        T, cols = frames.shape
        if cols >= 7 + self._act_dim:
            # Full qpos layout — slice out joint columns
            frames = frames[:, 7 : 7 + self._act_dim].astype(np.float32)
        elif cols >= self._act_dim:
            frames = frames[:, : self._act_dim].astype(np.float32)
        else:
            import warnings
            warnings.warn(
                f"ReferenceMotion: array has only {cols} columns but env has "
                f"{self._act_dim} joints. Reference tracking reward will be disabled.",
                stacklevel=4,
            )
            return

        self._ref_frames = frames
        self._ref_num_frames = T
        self._ref_phase_mode = str(getattr(cfg, "phase_mode", "loop") or "loop")
        self._ref_tracking_weight = float(getattr(cfg, "tracking_weight", 1.0) or 1.0)
        self._ref_tracking_sigma = float(getattr(cfg, "tracking_sigma", 5.0) or 5.0)
        self._ref_random_start = bool(getattr(cfg, "random_start_phase", False))

        # Compute phase advance per control step from motion_fps.
        # motion_fps == 0  →  advance 1 frame per step (legacy, no resampling).
        # motion_fps > 0   →  advance motion_fps / control_frequency_hz frames
        #                     so the trajectory plays at its original speed.
        motion_fps = float(getattr(cfg, "motion_fps", 0.0) or 0.0)
        if motion_fps > 0.0:
            self._ref_phase_step = motion_fps / max(self.control_frequency_hz, 1.0)
        else:
            self._ref_phase_step = 1.0

        # Precompute finite-difference joint velocities (joint_vel_tracking).
        # qdot[t] ≈ (q[t+1] - q[t]) / control_dt; last frame copies second-to-last.
        vels = np.zeros_like(frames)
        if T > 1:
            vels[:-1] = (frames[1:] - frames[:-1]) / max(self._control_dt, 1e-6)
            vels[-1] = vels[-2]
        self._ref_joint_vels = vels

        # Precompute reference foot positions via MuJoCo FK (foot_pos_tracking).
        # For each reference frame: set joint qpos → mj_kinematics → read xpos.
        # qpos is saved and restored so the simulation state is unchanged.
        if self._foot_probe_body_ids:
            try:
                import mujoco as _mj
                saved_qpos = self._data.qpos.copy()
                n_feet_ref = len(self._foot_probe_body_ids)
                ref_foot = np.zeros((T, n_feet_ref, 3), dtype=np.float32)
                n_jnt = min(self._act_dim, frames.shape[1])
                for t_idx in range(T):
                    self._data.qpos[7: 7 + n_jnt] = frames[t_idx, :n_jnt]
                    _mj.mj_kinematics(self._model, self._data)
                    for fi, bid in enumerate(self._foot_probe_body_ids):
                        ref_foot[t_idx, fi] = self._data.xpos[bid]
                self._data.qpos[:] = saved_qpos
                _mj.mj_kinematics(self._model, self._data)
                self._ref_foot_positions = ref_foot
            except Exception:
                self._ref_foot_positions = None

        # Auto-inject reference_tracking reward term so the user does not have
        # to add it manually in the RewardsNode.  If the user already set a
        # non-zero weight there, that value wins; here we only provide the
        # default of 1.0 when the key is absent or zero.
        if self._reward_terms.get("reference_tracking", 0.0) == 0.0:
            self._reward_terms["reference_tracking"] = 1.0

    @staticmethod
    def _resolve_motion_source_fallback(source: str) -> str:
        """Resolve a motion_source string to a .npy path (env-level fallback)."""
        try:
            from src.system.training.loco_mujoco_bridge import (
                is_available, load_trajectory_as_npy,
                generate_standing_reference, generate_simple_walk_reference,
            )
            source = source.strip()
            if source.startswith("loco:"):
                parts = source.split(":", 2)
                if len(parts) >= 3 and is_available():
                    result = load_trajectory_as_npy(parts[1], task=parts[2])
                    return str(result) if result else ""
            elif source.startswith("generate:"):
                gen_type = source.split(":", 1)[1].strip().lower()
                if gen_type == "standing":
                    return str(generate_standing_reference("go2"))
                if gen_type == "walk":
                    return str(generate_simple_walk_reference("go2"))
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _resolve_reward_weights(self) -> Dict[str, float]:
        """Return the active reward-term weights for the current training step.

        When no GatedRewardConfig is attached, returns the base weights unchanged.

        Gate condition for advancing to stage N+1:
          - global_step >= stage[N+1].min_step   (minimum step count)
          - rolling episode mean >= best_ever * threshold_ratio  (stability guard)
          - OR global_step >= stage[N].max_step   (hard timeout — force advance)

        While the gate is open a linear blend over ``blend_steps`` is applied
        between the old and new weight sets.  ``stage_behavior = "accumulate"``
        stacks weights from all prior stages; ``"replace"`` (default) uses only
        the current stage's weights on top of the static base.
        """
        cfg = self._gated_config
        if cfg is None:
            return self._base_reward_terms

        stages = cfg.stages
        n_stages = len(stages)
        if n_stages == 0:
            return self._base_reward_terms

        idx = min(self._current_stage_idx, n_stages - 1)

        # ── Check whether the next stage gate should open ──────────────────
        next_idx = idx + 1
        if next_idx < n_stages and self._blend_start_step < 0:
            cur_stage = stages[idx]

            # --- Primary readiness guard: minimum episodes in this stage ---
            min_ep = int(getattr(cfg, "min_ep_window", 10) or 10)
            min_ep_ok = self._stage_ep_count >= min_ep

            # --- Stability guard: rolling mean vs best-ever ----------------
            ep_list = list(self._ep_rewards)
            rolling_mean = float(np.mean(ep_list)) if ep_list else -float("inf")
            ratio = float(getattr(cur_stage, "reward_threshold_ratio", 0.75) or 0.75)
            stability_ok = (
                self._best_ep_reward > -float("inf")
                and rolling_mean >= self._best_ep_reward * ratio
            )

            # --- Plateau detection: reward slope has converged -------------
            p_window = int(getattr(cfg, "plateau_window", 10) or 10)
            p_eps    = float(getattr(cfg, "plateau_eps", 0.005) or 0.005)
            plateau_detected = self._is_plateau(p_window, p_eps)

            # --- Hard timeout: absolute step ceiling (safety net) ----------
            hard_max = int(getattr(cur_stage, "max_step", 0) or 0)
            hard_timeout = hard_max > 0 and self._global_step >= hard_max

            if min_ep_ok and (stability_ok or plateau_detected or hard_timeout):
                # Capture current weights as blend origin, open blend
                self._blend_origin_weights = dict(self._reward_terms)
                self._blend_start_step = self._global_step
                self._current_stage_idx = next_idx
                idx = next_idx
                # Reset stage-local counters for the new stage
                self._stage_ep_rewards = []
                self._stage_ep_count = 0

        def _stage_weights(target_idx: int) -> Dict[str, float]:
            """Build full weight dict for the given stage index."""
            w = dict(self._base_reward_terms)
            if cfg.stage_behavior == "accumulate":
                for si in range(min(target_idx + 1, n_stages)):
                    for k, v in (stages[si].reward_terms or {}).items():
                        w[k] = w.get(k, 0.0) + float(v)
            else:  # "replace"
                w.update(stages[min(target_idx, n_stages - 1)].reward_terms or {})
            return w

        # ── Active blend ───────────────────────────────────────────────────
        if self._blend_start_step >= 0:
            blend_steps = max(1, int(getattr(cfg, "blend_steps", 3000) or 3000))
            t = min(1.0, (self._global_step - self._blend_start_step) / blend_steps)
            target_w = _stage_weights(self._current_stage_idx)

            if t >= 1.0:
                # Blend finished
                self._blend_start_step = -1
                self._blend_origin_weights = {}
                return target_w

            # Linear interpolation between origin and target weights
            all_keys = set(self._blend_origin_weights) | set(target_w)
            blended: Dict[str, float] = {}
            for k in all_keys:
                w0 = self._blend_origin_weights.get(k, 0.0)
                w1 = target_w.get(k, 0.0)
                blended[k] = w0 + t * (w1 - w0)
            return blended

        # ── Stable at current stage ────────────────────────────────────────
        return _stage_weights(self._current_stage_idx)

    def _is_plateau(self, window: int, eps: float) -> bool:
        """Return True when the stage-local reward curve has converged.

        Uses a simple linear regression over the last ``window`` episode
        rewards recorded since the current stage started.  The slope is
        normalised by the absolute rolling mean so the threshold is
        scale-independent.

        Returns False when there are fewer episodes than ``window`` to
        avoid false-positive plateaus during warm-up.
        """
        if len(self._stage_ep_rewards) < window:
            return False
        recent = self._stage_ep_rewards[-window:]
        n = len(recent)
        x = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        # slope via least-squares (avoids importing scipy)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = float(np.sum((x - x_mean) ** 2))
        if denom == 0.0:
            return True  # all x identical → no variance → treat as flat
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        normaliser = abs(y_mean) if abs(y_mean) > 1e-6 else 1.0
        return abs(slope) / normaliser < eps

    def _compute_reward(
        self,
        action: np.ndarray,
        prev_action: Optional[np.ndarray] = None,
    ) -> float:
        """Config-driven reward composition; unknown keys are ignored safely."""
        if not self._reward_terms:
            return 0.0

        lin_vel_x = float(self._data.qvel[0])
        lin_vel_y = float(self._data.qvel[1])
        ang_vel_z = float(self._data.qvel[5])

        vx_tgt, vy_tgt, yaw_tgt = (
            float(self._commands[0]),
            float(self._commands[1]),
            float(self._commands[2]),
        )

        r_vx  = float(np.exp(-((lin_vel_x - vx_tgt)  ** 2) / 0.25))
        r_vy  = float(np.exp(-((lin_vel_y - vy_tgt)  ** 2) / 0.25))
        r_yaw = float(np.exp(-((ang_vel_z - yaw_tgt) ** 2) / 0.25))
        velocity_score = r_vx + 0.5 * r_vy

        quat = np.array(self._data.qpos[3:7], dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            quat = quat / norm
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        projected_gravity = _project_gravity(quat)
        upright_score = float(np.clip(-projected_gravity[2], 0.0, 1.0))

        alive_score = 1.0
        energy_cost = float(np.sum(action ** 2))
        lateral_error = abs(lin_vel_y - vy_tgt)
        if prev_action is None:
            prev_action = np.zeros_like(action)
        smoothness_cost = float(np.sum((action - prev_action) ** 2))
        foot_clearance_score = self._estimate_foot_clearance_score()
        slip_cost = self._estimate_slip_cost()
        collision_cost = self._estimate_collision_cost()

        # base_height_penalty: squared deviation from nominal standing height (0.32 m)
        _STANDING_HEIGHT = 0.32
        base_height = float(self._data.qpos[2]) if self._model.nq >= 3 else _STANDING_HEIGHT
        base_height_err_sq = (base_height - _STANDING_HEIGHT) ** 2

        # angular_rate_penalty: squared roll + pitch angular rates (qvel[3], qvel[4])
        roll_rate  = float(self._data.qvel[3]) if self._model.nv > 3 else 0.0
        pitch_rate = float(self._data.qvel[4]) if self._model.nv > 4 else 0.0
        angular_rate_cost = roll_rate ** 2 + pitch_rate ** 2

        # foot_air_time: landing bonus accumulated via per-foot swing-phase counter
        foot_air_time_score = self._update_foot_air_time()

        reward = 0.0
        reward += self._reward_terms.get("velocity_tracking", 0.0) * velocity_score
        reward += self._reward_terms.get("yaw_tracking", 0.0) * r_yaw
        reward += self._reward_terms.get("alive", 0.0) * alive_score
        reward += self._reward_terms.get("upright", 0.0) * upright_score
        reward += self._reward_terms.get("energy", 0.0) * energy_cost
        reward += self._reward_terms.get("lateral_penalty", 0.0) * lateral_error
        reward += self._reward_terms.get("action_smoothness", 0.0) * smoothness_cost
        reward += self._reward_terms.get("foot_clearance", 0.0) * foot_clearance_score
        reward += self._reward_terms.get("slip_penalty", 0.0) * slip_cost
        reward += self._reward_terms.get("collision_penalty", 0.0) * collision_cost
        reward += self._reward_terms.get("goal_distance", 0.0) * self._task_success_score()
        reward += self._reward_terms.get("grasp_success", 0.0) * float(self._success_streak > 0)
        reward += self._reward_terms.get("base_height_penalty", 0.0) * base_height_err_sq
        reward += self._reward_terms.get("angular_rate_penalty", 0.0) * angular_rate_cost
        reward += self._reward_terms.get("foot_air_time", 0.0) * foot_air_time_score

        # Reference motion tracking reward (Plan A motion imitation)
        _ref_score = 0.0
        ref_term_weight = self._reward_terms.get("reference_tracking", 0.0)
        if ref_term_weight != 0.0 and self._ref_frames is not None:
            phase_int = int(self._ref_phase_f)
            if self._ref_phase_mode == "loop":
                ref_idx = phase_int % self._ref_num_frames
            else:
                ref_idx = min(phase_int, self._ref_num_frames - 1)
            ref_joints = self._ref_frames[ref_idx]
            cur_joints = np.asarray(
                self._data.qpos[7 : 7 + self._act_dim], dtype=np.float32
            )
            n = min(len(cur_joints), len(ref_joints))
            sq_err = float(np.sum((cur_joints[:n] - ref_joints[:n]) ** 2))
            ref_score = self._ref_tracking_weight * float(
                np.exp(-self._ref_tracking_sigma * sq_err)
            )
            _ref_score = ref_score
            reward += ref_term_weight * ref_score

        # ------------------------------------------------------------------
        # joint_pose_tracking  exp(-5 * ||q_joints - q_ref||²)
        # q_ref: current reference frame joints (ref loaded) or GO2_DEFAULT_QPOS.
        # ------------------------------------------------------------------
        _jpt_score = 0.0
        jpt_weight = self._reward_terms.get("joint_pose_tracking", 0.0)
        if jpt_weight != 0.0:
            cur_joints = np.asarray(
                self._data.qpos[7: 7 + self._act_dim], dtype=np.float32
            )
            if self._ref_frames is not None:
                _ri = int(self._ref_phase_f)
                _ri = (_ri % self._ref_num_frames if self._ref_phase_mode == "loop"
                       else min(_ri, self._ref_num_frames - 1))
                q_ref = self._ref_frames[_ri]
            else:
                q_ref = self._default_qpos[:self._act_dim]
            n = min(len(cur_joints), len(q_ref))
            sq_err = float(np.sum((cur_joints[:n] - q_ref[:n]) ** 2))
            _jpt_score = float(np.exp(-5.0 * sq_err))
            reward += jpt_weight * _jpt_score

        # ------------------------------------------------------------------
        # joint_vel_tracking  exp(-0.5 * ||qvel_joints - qdot_ref||²)
        # qdot_ref: precomputed finite-diff velocities (ref loaded) or zeros.
        # ------------------------------------------------------------------
        _jvt_score = 0.0
        jvt_weight = self._reward_terms.get("joint_vel_tracking", 0.0)
        if jvt_weight != 0.0:
            end_v = min(self._model.nv, 6 + self._act_dim)
            cur_vel = np.asarray(self._data.qvel[6:end_v], dtype=np.float32)
            if cur_vel.shape[0] < self._act_dim:
                cur_vel = np.pad(cur_vel, (0, self._act_dim - cur_vel.shape[0]))
            if self._ref_joint_vels is not None:
                _ri = int(self._ref_phase_f)
                _ri = (_ri % self._ref_num_frames if self._ref_phase_mode == "loop"
                       else min(_ri, self._ref_num_frames - 1))
                qdot_ref = self._ref_joint_vels[_ri]
            else:
                qdot_ref = np.zeros(self._act_dim, dtype=np.float32)
            n = min(len(cur_vel), len(qdot_ref))
            sq_err = float(np.sum((cur_vel[:n] - qdot_ref[:n]) ** 2))
            _jvt_score = float(np.exp(-0.5 * sq_err))
            reward += jvt_weight * _jvt_score

        # ------------------------------------------------------------------
        # foot_pos_tracking  exp(-10 * mean(||foot_pos - foot_ref||²))
        # foot_ref: precomputed FK positions (ref loaded) or nominal standing positions.
        # ------------------------------------------------------------------
        _fpt_score = 0.0
        fpt_weight = self._reward_terms.get("foot_pos_tracking", 0.0)
        if fpt_weight != 0.0 and self._foot_probe_body_ids:
            nbody = int(getattr(self._model, "nbody", 0) or 0)
            foot_ref_frame: Optional[np.ndarray] = None
            if self._ref_foot_positions is not None:
                _ri = int(self._ref_phase_f)
                _ri = (_ri % self._ref_num_frames if self._ref_phase_mode == "loop"
                       else min(_ri, self._ref_num_frames - 1))
                foot_ref_frame = self._ref_foot_positions[_ri]
            elif self._nominal_foot_positions is not None:
                foot_ref_frame = self._nominal_foot_positions
            if foot_ref_frame is not None:
                sq_errs: List[float] = []
                for fi, bid in enumerate(self._foot_probe_body_ids):
                    if not (0 <= int(bid) < nbody) or fi >= len(foot_ref_frame):
                        continue
                    try:
                        cur_fp = np.asarray(self._data.xpos[bid], dtype=np.float32)
                        sq_errs.append(float(np.sum((cur_fp - foot_ref_frame[fi]) ** 2)))
                    except Exception:
                        pass
                if sq_errs:
                    _fpt_score = float(np.exp(-10.0 * float(np.mean(sq_errs))))
                    reward += fpt_weight * _fpt_score

        # ── Snapshot raw component values (un-weighted) for metrics reporting ──
        self._last_reward_components = {
            "velocity":           velocity_score,
            "yaw":                r_yaw,
            "alive":              alive_score,
            "upright":            upright_score,
            "energy":             energy_cost,
            "smoothness":         smoothness_cost,
            "foot_clearance":     foot_clearance_score,
            "slip":               slip_cost,
            "collision":          collision_cost,
            "base_height":        base_height,
            "base_height_err":    base_height_err_sq,
            "angular_rate":       angular_rate_cost,
            "foot_air_time":      foot_air_time_score,
            "reference_tracking": _ref_score,
            "joint_pose_tracking":_jpt_score,
            "joint_vel_tracking": _jvt_score,
            "foot_pos_tracking":  _fpt_score,
        }

        return float(reward)

    def _estimate_foot_clearance_score(self) -> float:
        body_ids = list(getattr(self, "_foot_probe_body_ids", []) or [])
        if not body_ids:
            return 0.0
        try:
            heights = [
                float(self._data.xpos[body_id][2])
                for body_id in body_ids
                if 0 <= int(body_id) < int(getattr(self._model, "nbody", 0) or 0)
            ]
        except Exception:
            return 0.0
        if not heights:
            return 0.0
        avg_height = float(np.mean(heights))
        return float(np.clip((avg_height - 0.02) / 0.18, 0.0, 1.0))

    def _estimate_slip_cost(self) -> float:
        body_ids = list(getattr(self, "_foot_probe_body_ids", []) or [])
        if not body_ids:
            return 0.0
        slip_speeds: List[float] = []
        try:
            for body_id in body_ids:
                if not (0 <= int(body_id) < int(getattr(self._model, "nbody", 0) or 0)):
                    continue
                height = float(self._data.xpos[body_id][2])
                if height > 0.08:
                    continue
                lin_vel = np.asarray(self._data.cvel[body_id][3:6], dtype=np.float32)
                slip_speeds.append(float(np.linalg.norm(lin_vel[:2])))
        except Exception:
            return 0.0
        if not slip_speeds:
            return 0.0
        return float(np.clip(np.mean(slip_speeds) / 1.0, 0.0, 1.0))

    def _estimate_collision_cost(self) -> float:
        base_body_id = int(getattr(self, "_base_body_id", 0) or 0)
        try:
            base_force = np.asarray(self._data.cfrc_ext[base_body_id][:3], dtype=np.float32)
        except Exception:
            return 0.0
        force_mag = float(np.linalg.norm(base_force))
        return float(np.clip(force_mag / 150.0, 0.0, 1.0))

    def _update_foot_air_time(self) -> float:
        """Return per-step foot-air-time bonus.

        At each step, for every tracked foot body:
        - Height > 0.05 m  →  foot is in swing (air); increment air counter.
        - Height ≤ 0.05 m  →  foot is in stance (ground).
          · If the foot just transitioned from swing to stance (touchdown event),
            add a landing bonus = clip(air_duration_seconds, 0, 0.5).
          · Reset the air counter.

        The total score is summed across all feet and normalised by foot count,
        producing a value in [0, 0.5] at touchdown frames and 0 otherwise.
        This encourages each leg to spend an appropriate amount of time in the
        air (~0.2–0.5 s) during each gait cycle.
        """
        body_ids = list(getattr(self, "_foot_probe_body_ids", []) or [])
        n_feet = len(body_ids)
        if n_feet == 0:
            return 0.0

        nbody = int(getattr(self._model, "nbody", 0) or 0)
        score = 0.0
        for i, body_id in enumerate(body_ids):
            if not (0 <= int(body_id) < nbody):
                continue
            try:
                h = float(self._data.xpos[body_id][2])
            except Exception:
                continue

            in_air = h > 0.05
            if in_air:
                self._foot_air_steps[i] += 1
            else:
                if self._foot_was_in_air[i]:  # touchdown event
                    air_duration = self._foot_air_steps[i] * self._control_dt
                    score += float(np.clip(air_duration, 0.0, 0.5))
                self._foot_air_steps[i] = 0
            self._foot_was_in_air[i] = in_air

        return score / n_feet

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def get_termination_debug_info(self) -> Dict[str, Any]:
        """Return the current termination decision plus the triggering threshold."""
        info: Dict[str, Any] = {
            "terminated": False,
            "reason": "",
            "step_count": int(getattr(self, "_step_count", 0) or 0),
        }
        if not self._termination_conditions:
            return info
        if self._model.nq < 7:
            return info

        base_height = float(self._data.qpos[2])
        min_height = self._termination_conditions.get("min_height", -float("inf"))
        info["base_height"] = base_height
        info["min_height"] = min_height
        if base_height < min_height:
            info["terminated"] = True
            info["reason"] = "min_height"
            return info

        quat = np.array(self._data.qpos[3:7], dtype=np.float32)
        norm = np.linalg.norm(quat)
        roll = 0.0
        pitch = 0.0
        if norm > 1e-8:
            quat = quat / norm
            roll, pitch = _quat_to_roll_pitch(quat)
        roll_threshold = self._termination_conditions.get(
            "fall_threshold_roll", float("inf")
        )
        pitch_threshold = self._termination_conditions.get(
            "fall_threshold_pitch", float("inf")
        )
        info["roll"] = float(roll)
        info["pitch"] = float(pitch)
        info["fall_threshold_roll"] = roll_threshold
        info["fall_threshold_pitch"] = pitch_threshold
        if abs(roll) > roll_threshold:
            info["terminated"] = True
            info["reason"] = "fall_threshold_roll"
            return info
        if abs(pitch) > pitch_threshold:
            info["terminated"] = True
            info["reason"] = "fall_threshold_pitch"
            return info

        timeout_threshold = self._termination_conditions.get("timeout", float("inf"))
        info["timeout"] = timeout_threshold
        if self._step_count >= timeout_threshold:
            info["terminated"] = True
            info["reason"] = "timeout"
            return info

        max_contact_impulse = self._termination_conditions.get("max_contact_impulse", float("inf"))
        info["max_contact_impulse"] = max_contact_impulse
        if np.isfinite(max_contact_impulse):
            try:
                contact_impulse = float(np.max(np.abs(self._data.cfrc_ext)))
                info["contact_impulse"] = contact_impulse
                if contact_impulse > float(max_contact_impulse):
                    info["terminated"] = True
                    info["reason"] = "max_contact_impulse"
                    return info
            except Exception:
                pass

        # joint_limit_violation: terminate when >= threshold joints exceed their MuJoCo range.
        # Threshold value (default 1.0) is a count: 1 = any violation triggers termination.
        jlv_threshold = self._termination_conditions.get("joint_limit_violation", float("inf"))
        if np.isfinite(jlv_threshold):
            try:
                violated = 0
                for jnt_id in range(int(getattr(self._model, "njnt", 0) or 0)):
                    if int(self._model.jnt_limited[jnt_id]) == 0:
                        continue  # joint has no range limits defined
                    qpos_adr = int(self._model.jnt_qposadr[jnt_id])
                    lo = float(self._model.jnt_range[jnt_id, 0])
                    hi = float(self._model.jnt_range[jnt_id, 1])
                    val = float(self._data.qpos[qpos_adr])
                    if val < lo or val > hi:
                        violated += 1
                info["joint_limit_violated_count"] = violated
                info["joint_limit_violation_threshold"] = jlv_threshold
                if violated >= jlv_threshold:
                    info["terminated"] = True
                    info["reason"] = "joint_limit_violation"
                    return info
            except Exception:
                pass

        # self_collision: terminate when >= threshold non-adjacent robot body contacts are detected.
        # Parent-child (adjacent limb) contacts are excluded as they are physically normal.
        # World/floor contacts (body_id == 0) are excluded.
        sc_threshold = self._termination_conditions.get("self_collision", float("inf"))
        if np.isfinite(sc_threshold):
            try:
                sc_count = 0
                ncon = int(getattr(self._data, "ncon", 0) or 0)
                ngeom = int(getattr(self._model, "ngeom", 0) or 0)
                for c_idx in range(ncon):
                    contact = self._data.contact[c_idx]
                    g1 = int(contact.geom1)
                    g2 = int(contact.geom2)
                    if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
                        continue
                    b1 = int(self._model.geom_bodyid[g1])
                    b2 = int(self._model.geom_bodyid[g2])
                    if b1 == 0 or b2 == 0:
                        continue  # world/floor contact
                    if b1 == b2:
                        continue  # same body
                    # exclude parent-child adjacent links (touching during normal motion)
                    if (int(self._model.body_parentid[b1]) == b2
                            or int(self._model.body_parentid[b2]) == b1):
                        continue
                    sc_count += 1
                info["self_collision_count"] = sc_count
                info["self_collision_threshold"] = sc_threshold
                if sc_count >= sc_threshold:
                    info["terminated"] = True
                    info["reason"] = "self_collision"
                    return info
            except Exception:
                pass

        return info

    def _is_terminated(self) -> bool:
        """Config-driven termination thresholds; unknown keys are ignored safely."""
        return bool(self.get_termination_debug_info().get("terminated", False))

