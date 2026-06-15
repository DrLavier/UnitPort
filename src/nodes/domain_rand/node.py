# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""DomainRandNode — 统一域随机化节点 (SB3 + Isaac Lab).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:DomainRandNode``.

Backend binding：每张画布与单一训练后端绑定，本节点不暴露 backend ParamSpec —
其行为契约由所在 canvas 自动决定（``CanvasPage.apply_canvas_backend`` 注入
``params['backend']`` 给 ``conditional_on``）：
    sb3        — SB3 DR 参数 (mass_range / friction_range / motor / push / schedule)；
                 actor_pipe 端口隐藏
    isaac_lab  — IL DR events (friction / mass / push / init_pose / joint_noise，
                 每组带 mode 切换)；actor_pipe 端口可见

``normalize_parameters()`` 在画布加载后裁掉错误 backend 的 keys，
保证「保存的画布 → 加载后的 params → execute 后的 config」一一一对齐。

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


# ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----

# 仅 IL 模式可见的端口集合
_IL_PORTS = ("actor_pipe",)

# 共享参数集合（两个 backend 都保留）
_SHARED_KEYS = (
    "rand_schedule",
    "rand_schedule_start_step",
    "rand_schedule_end_step",
)

# SB3 模式专属参数集合
_SB3_KEYS = (
    "enabled",
    "mass_range",
    "friction_range",
    # Stage H — replaces motor_strength_range / joint_damping_range.
    "omega_n_log_uniform",
    "zeta_log_uniform",
    "obs_noise_std",
    "push_robot",
    "push_interval_steps",
    "push_force_range",
    # P2 — reset-time state randomization (feature parity with IsaacLab DR).
    "sb3_enable_init_pose_rand",
    "sb3_init_pos_x_range",
    "sb3_init_pos_y_range",
    "sb3_init_yaw_range",
    "sb3_enable_joint_noise",
    "sb3_joint_pos_noise",
    "sb3_joint_vel_noise",
)

# Isaac Lab 模式专属参数集合
_IL_KEYS = (
    # friction
    "enable_friction_rand", "friction_mode",
    "static_friction_range", "dynamic_friction_range", "restitution_range",
    # mass
    "enable_mass_rand", "mass_mode", "mass_target_scope", "mass_target_body",
    "mass_offset_range",
    # push
    "enable_external_push", "push_mode", "push_interval_range_s",
    "push_velocity_x_range", "push_velocity_y_range",
    # init_pose
    "enable_init_pose_rand", "init_pose_mode",
    "init_pos_x_range", "init_pos_y_range", "init_yaw_range",
    # joint noise
    "enable_joint_noise", "joint_noise_mode",
    "joint_pos_noise", "joint_vel_noise",
)


# Per-backend default parameter sets — mirrors DEMO ``_SB3_DEFAULTS`` / ``_IL_DEFAULTS``.
_SB3_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "mass_range": "[0.8, 1.2]",
    "friction_range": "[0.5, 1.5]",
    # Stage H mass-matrix-adaptive PD: log-uniform multiplicative jitter
    # on (omega_n, zeta) per training step. Bounds chosen so a 1.25x omega_n
    # excursion remains within (0.1, 2.0) zeta dynamic range.
    "omega_n_log_uniform": "[0.8, 1.25]",
    "zeta_log_uniform": "[0.9, 1.11]",
    "obs_noise_std": 0.01,
    "push_robot": False,
    "push_interval_steps": 200,
    "push_force_range": "[50, 150]",
    # P2 — reset-time state randomization (disabled by default → legacy parity).
    "sb3_enable_init_pose_rand": False,
    "sb3_init_pos_x_range": "[-0.0, 0.0]",
    "sb3_init_pos_y_range": "[-0.0, 0.0]",
    "sb3_init_yaw_range": "[-0.0, 0.0]",
    "sb3_enable_joint_noise": False,
    "sb3_joint_pos_noise": 0.0,
    "sb3_joint_vel_noise": 0.0,
}

_IL_DEFAULTS: Dict[str, Any] = {
    "enable_friction_rand": True,
    "friction_mode": "reset",
    "static_friction_range": "[0.5, 1.25]",
    "dynamic_friction_range": "[0.5, 1.25]",
    "restitution_range": "[0.0, 0.4]",
    "enable_mass_rand": True,
    "mass_mode": "reset",
    "mass_target_scope": "root",
    "mass_target_body": "torso",
    "mass_offset_range": "[-0.5, 0.5]",
    "enable_external_push": True,
    "push_mode": "interval",
    "push_interval_range_s": "[10.0, 15.0]",
    "push_velocity_x_range": "[-0.5, 0.5]",
    "push_velocity_y_range": "[-0.5, 0.5]",
    "enable_init_pose_rand": True,
    "init_pose_mode": "reset",
    "init_pos_x_range": "[-0.5, 0.5]",
    "init_pos_y_range": "[-0.5, 0.5]",
    "init_yaw_range": "[-3.14, 3.14]",
    "enable_joint_noise": True,
    "joint_noise_mode": "reset",
    "joint_pos_noise": 0.25,
    "joint_vel_noise": 0.1,
}

_SHARED_DEFAULTS: Dict[str, Any] = {
    "rand_schedule": "none",
    "rand_schedule_start_step": 0,
    "rand_schedule_end_step": 500000,
}


class DomainRandNode(BaseNode):
    """Layer A — Unified domain randomisation node (SB3 + Isaac Lab)."""

    # ---- DEMO 行为契约（class-level 提示） ----
    _IL_PORTS: tuple = _IL_PORTS
    _SB3_KEYS: tuple = _SB3_KEYS
    _IL_KEYS: tuple = _IL_KEYS
    _SHARED_KEYS: tuple = _SHARED_KEYS

    MANIFEST = manifest_from_toml(__file__)

    # ---- 画布 UI 钩子（动态隐藏端口与参数） ----

    @property
    def hidden_ports(self) -> set:
        """SB3 backend 下隐藏 IL 专属端口；IL backend 全部可见."""
        backend = str(self.get_parameter("backend", "sb3_mujoco") or "sb3_mujoco").strip()
        if backend == "isaac_lab":
            return set()
        return set(self._IL_PORTS)

    @property
    def hidden_params(self) -> set:
        """根据 backend 隐藏对侧 backend 的参数 keys."""
        backend = str(self.get_parameter("backend", "sb3_mujoco") or "sb3_mujoco").strip()
        if backend == "isaac_lab":
            return set(self._SB3_KEYS)
        return set(self._IL_KEYS)

    # ---- normalize_parameters：画布加载后补齐默认值 ----

    def normalize_parameters(self) -> None:
        """补齐当前 backend 的默认值与共享默认值.

        Strict-mode contract:
        Wrong-backend keys are **NOT** silently dropped (CLAUDE.md §1, plan
        invariant: "canvas 永远不静默改用户参数"). If a canvas carries SB3
        keys while ``backend=isaac_lab`` (or vice versa) the keys stay
        untouched in ``self.params``; spec_compiler reads only the active
        side, so the stray keys are inert at compile time. To strip them
        permanently, run ``bootstrap/migrate_canvas_strict_v1.py``.

        Default补齐 (``setdefault``) is harmless: it only fills keys the
        user never set, never overwrites user values, never deletes user
        values. This preserves the canvas-as-truth invariant.
        """
        backend = str(self.params.get("backend", "sb3_mujoco") or "sb3_mujoco").strip() or "sb3_mujoco"
        self.params["backend"] = backend
        if backend == "isaac_lab":
            active_defaults = _IL_DEFAULTS
        else:
            active_defaults = _SB3_DEFAULTS
        # 1) 补齐当前 backend 的默认值（仅当用户未设）
        for k, v in active_defaults.items():
            self.params.setdefault(k, v)
        # 2) 补齐共享默认值（仅当用户未设）
        for k, v in _SHARED_DEFAULTS.items():
            self.params.setdefault(k, v)

    # ---- 校验 ----

    def validate(self) -> List[str]:
        return super().validate()