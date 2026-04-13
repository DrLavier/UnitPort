"""Isaac Lab environment presets — parameter templates for common tasks.

Each preset is a dict mapping ``node_type`` → ``{param_key: value}``.
Applying a preset batch-updates every matching node on the canvas.

Preset values are derived from the official Isaac Lab source:
  isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
  - velocity_env_cfg.py  (base)
  - rough_env_cfg.py     (Go2 rough override)
  - flat_env_cfg.py      (Go2 flat override)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Type alias:  preset_name → { node_type → { param_key: param_value } }
# ---------------------------------------------------------------------------
ILPreset = Dict[str, Dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════
# Go2 — Flat Terrain Velocity Tracking
# ═══════════════════════════════════════════════════════════════════════════
# Merged: base → Go2 rough → Go2 flat

_GO2_FLAT: ILPreset = {
    "play_ground_setting": {
        "sim_dt": "0.005",
        "gravity_z": "-9.81",
        "gpu_max_rigid_contact_count": "524288",
        "gpu_max_rigid_patch_count": "327680",
    },
    "il_terrain_config": {
        "terrain_type": "flat",
        "size_x": "100.0",
        "size_y": "100.0",
        "curriculum_enabled": "false",
        "num_rows": "10",
        "num_cols": "20",
        "max_difficulty": "1.0",
    },
    "il_robot_asset": {
        "usd_path": "isaac_lab_assets://robots/unitree/go2/go2.usd",
        "num_envs": "4096",
        "env_spacing": "2.5",
        "init_pos_x": "0.0",
        "init_pos_y": "0.0",
        "init_pos_z": "0.4",
        "init_joint_angles": "{}",
        "activate_contact_sensors": "true",
    },
    "il_actuator_config": {
        "joint_names_expr": ".*",
        "model": "implicit_pd",
        "stiffness": "25.0",
        "damping": "0.5",
        "effort_limit": "23.5",
        "velocity_limit": "30.0",
    },
    "il_contact_sensor": {
        "body_names": '[".*_foot"]',
        "history_length": "3",
        "track_air_time": "true",
    },
    "il_observation": {
        "obs_terms": json.dumps({
            "base_lin_vel": 1.0,
            "base_ang_vel": 1.0,
            "projected_gravity": 1.0,
            "velocity_command": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "last_action": 1.0,
            # height_scan removed in flat variant
        }),
        "group_name": "policy",
        "enable_corruption": "true",
        "corruption_noise_std": "0.05",
    },
    "il_action_config": {
        "joint_names_expr": ".*",
        "scale": "0.25",
        "use_default_offset": "true",
    },
    "il_velocity_cmd": {
        "lin_vel_x_range": "[-1.0, 1.0]",
        "lin_vel_y_range": "[-1.0, 1.0]",
        "ang_vel_z_range": "[-1.0, 1.0]",
        "resampling_time_range": "[10.0, 10.0]",
        "zero_command_probability": "0.0",
    },
    # Unified nodes — backend set to isaac_lab
    "rewards": {
        "backend": "isaac_lab",
        "reward_terms": json.dumps({
            "track_lin_vel_xy": 1.5,
            "track_ang_vel_z": 0.75,
            "lin_vel_z_penalty": -2.0,
            "ang_vel_xy_penalty": -0.05,
            "joint_torque_penalty": -0.0002,
            "joint_accel_penalty": -2.5e-7,
            "action_rate_penalty": -0.01,
            "feet_air_time": 0.25,
            "flat_orientation": -2.5,
        }),
        "std": "0.25",
        "threshold": "0.5",
    },
    "terminations": {
        "backend": "isaac_lab",
        # Unified items list — same widget as SB3, populated with the IL
        # registry keys (time_out / illegal_contact / base_height). The
        # IL compiler reads these values directly to emit DoneTerm
        # entries; see isaac_lab_config_compiler._terminations_cfg.
        # base_height is intentionally omitted so it stays "off" by
        # default (mirrors the previous enable_base_height=false toggle).
        "termination_conditions": json.dumps({
            "time_out": 20.0,
            "illegal_contact": 1.0,
        }),
    },
    "domain_rand": {
        "backend": "isaac_lab",
        "enable_friction_rand": "true",
        "friction_mode": "startup",
        "static_friction_range": "[0.5, 1.5]",
        "dynamic_friction_range": "[0.4, 1.2]",
        "restitution_range": "[0.0, 0.2]",
        "enable_mass_rand": "true",
        "mass_mode": "startup",
        "mass_target_body": "base",
        "mass_offset_range": "[-1.0, 3.0]",
        "enable_external_push": "false",
        "push_mode": "interval",
        "push_interval_range_s": "[10.0, 15.0]",
        "push_velocity_x_range": "[-0.5, 0.5]",
        "push_velocity_y_range": "[-0.5, 0.5]",
        "enable_init_pose_rand": "true",
        "init_pose_mode": "reset",
        "init_pos_x_range": "[-0.5, 0.5]",
        "init_pos_y_range": "[-0.5, 0.5]",
        "init_yaw_range": "[-3.14, 3.14]",
        "enable_joint_noise": "true",
        "joint_noise_mode": "reset",
        "joint_pos_noise": "0.0",
        "joint_vel_noise": "0.0",
    },
    "il_policy_network": {
        "actor_hidden_dims": "[128, 64, 32]",
        "critic_hidden_dims": "[128, 64, 32]",
        "activation": "elu",
        "init_noise_std": "-1.0",
        "disc_hidden_dims": "[1024, 512]",
    },
    "il_ppo_trainer": {
        "training_mode": "PPO",
        "max_iterations": "1500",
        "num_steps_per_env": "24",
        "num_learning_epochs": "5",
        "num_minibatches": "4",
        "learning_rate": "0.001",
        "discount_factor": "0.99",
        "gae_lambda": "0.95",
        "clip_param": "0.2",
        "entropy_coef": "0.01",
        "value_loss_coef": "1.0",
        "max_grad_norm": "1.0",
        "schedule": "adaptive",
        "desired_kl": "0.01",
        "save_interval": "100",
        "seed": "42",
    },
    "il_policy_exporter": {
        "export_onnx": "true",
        "export_torchscript": "false",
        "bundle_name": "",
        "auto_import": "true",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Go2 — Rough Terrain Velocity Tracking
# ═══════════════════════════════════════════════════════════════════════════
# Merged: base → Go2 rough (keeps height_scan, curriculum, rough terrain)

_GO2_ROUGH: ILPreset = {
    **{k: dict(v) for k, v in _GO2_FLAT.items()},  # start from flat as base
}
# Override terrain → rough with curriculum
_GO2_ROUGH["il_terrain_config"] = {
    "terrain_type": "generator",
    "size_x": "100.0",
    "size_y": "100.0",
    "curriculum_enabled": "true",
    "num_rows": "10",
    "num_cols": "20",
    "max_difficulty": "5.0",
}
# Add height_scan back to obs
_GO2_ROUGH["il_observation"] = {
    "obs_terms": json.dumps({
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "projected_gravity": 1.0,
        "velocity_command": 1.0,
        "joint_pos": 1.0,
        "joint_vel": 1.0,
        "last_action": 1.0,
        "height_scan": 1.0,
    }),
    "group_name": "policy",
    "enable_corruption": "true",
    "corruption_noise_std": "0.05",
}
# Rough-specific reward weights (no flat_orientation boost, lower feet_air_time)
# The rough variant reuses the flat trainer params verbatim — task_name
# is sourced from the _meta dict below, not from a trainer param.
_GO2_ROUGH["il_ppo_trainer"] = dict(_GO2_FLAT["il_ppo_trainer"])
_GO2_ROUGH["rewards"] = {
    "backend": "isaac_lab",
    "reward_terms": json.dumps({
        "track_lin_vel_xy": 1.5,
        "track_ang_vel_z": 0.75,
        "lin_vel_z_penalty": -2.0,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -0.0002,
        "joint_accel_penalty": -2.5e-7,
        "action_rate_penalty": -0.01,
        "feet_air_time": 0.01,
        "flat_orientation": 0.0,
    }),
    "std": "0.25",
    "threshold": "0.5",
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

# Meta key stores non-node-param metadata (task_name for Isaac Lab CLI).
# It's NOT applied to any canvas node — only read by the launch pipeline.
_GO2_FLAT["_meta"] = {"task_name": "Isaac-Velocity-Flat-Unitree-Go2-v0"}
_GO2_ROUGH["_meta"] = {"task_name": "Isaac-Velocity-Rough-Unitree-Go2-v0"}


IL_PRESETS: Dict[str, ILPreset] = {
    "Go2 Flat Velocity":  _GO2_FLAT,
    "Go2 Rough Velocity": _GO2_ROUGH,
}


def list_il_presets() -> List[str]:
    """Return available preset names."""
    return list(IL_PRESETS.keys())


def get_il_preset(name: str) -> ILPreset:
    """Return preset dict by name. Raises KeyError if not found."""
    return IL_PRESETS[name]


def get_il_preset_task_name(name: str) -> str:
    """Return the Isaac Lab registered task name for a preset."""
    preset = IL_PRESETS[name]
    meta = preset.get("_meta", {})
    return meta.get("task_name", "Isaac-Velocity-Flat-Unitree-Go2-v0")
