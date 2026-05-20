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
)

# Isaac Lab 模式专属参数集合
_IL_KEYS = (
    # friction
    "enable_friction_rand", "friction_mode",
    "static_friction_range", "dynamic_friction_range", "restitution_range",
    # mass
    "enable_mass_rand", "mass_mode", "mass_target_body", "mass_offset_range",
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
    # Stage H defaults mirror knowledge_base/sim2sim_mass-matrix-adaptive.yaml
    # lines 103-105.
    "omega_n_log_uniform": "[0.8, 1.25]",
    "zeta_log_uniform": "[0.9, 1.11]",
    "obs_noise_std": 0.01,
    "push_robot": False,
    "push_interval_steps": 200,
    "push_force_range": "[50, 150]",
}

_IL_DEFAULTS: Dict[str, Any] = {
    "enable_friction_rand": True,
    "friction_mode": "reset",
    "static_friction_range": "[0.5, 1.25]",
    "dynamic_friction_range": "[0.5, 1.25]",
    "restitution_range": "[0.0, 0.4]",
    "enable_mass_rand": True,
    "mass_mode": "reset",
    "mass_target_body": "base",
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

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="domain_rand",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.domain_rand.display",
        description_key="node.domain_rand.desc",
        icon="domain_rand.svg",
        inputs=[
            PortSpec(name="actor_pipe", type="actor_pipe", optional=True,
                     description="Actor stream (IL only — auto-hidden in SB3 mode)"),
        ],
        outputs=[
            PortSpec(name="domain_rand_config", type="domain_rand_config"),
        ],
        parameters=[
            # ---- shared schedule ----
            ParamSpec(key="rand_schedule", type="enum", default="none",
                      choices=["none", "linear"],
                      description="DR 强度调度"),
            ParamSpec(key="rand_schedule_start_step", type="int", default=0,
                      widget="index", description="DR 激活起始全局步",
                      meta={"min": 0, "max": 1_000_000_000}),
            ParamSpec(key="rand_schedule_end_step", type="int", default=500_000,
                      widget="index", description="DR 达到满强度的全局步",
                      meta={"min": 0, "max": 1_000_000_000}),
            # ---- SB3 backend ----
            ParamSpec(key="enabled", type="bool", default=True,
                      description="[SB3] 启用",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"}}),
            ParamSpec(key="mass_range", type="list", default="[0.8, 1.2]",
                      widget="range_list",
                      description="[SB3] 质量缩放范围",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="friction_range", type="list", default="[0.5, 1.5]",
                      widget="range_list",
                      description="[SB3] 摩擦缩放范围",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 0.0, "max": 5.0, "step": 0.01}),
            # Stage H — (omega_n, zeta) log-uniform DR replaces the
            # legacy motor_strength_range / joint_damping_range knobs.
            # Both engines (SB3 + IL) get the same scaling: SB3 re-solves
            # via mujoco_gain_solver at episode reset, IL writes the
            # multipliers into an event term in env_cfg.py.
            ParamSpec(key="omega_n_log_uniform", type="list", default="[0.8, 1.25]",
                      widget="range_list",
                      description="[SB3] ωn 对数均匀缩放范围 (×nominal)",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 0.5, "max": 2.0, "step": 0.01}),
            ParamSpec(key="zeta_log_uniform", type="list", default="[0.9, 1.11]",
                      widget="range_list",
                      description="[SB3] ζ 对数均匀缩放范围 (×nominal)",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 0.7, "max": 1.5, "step": 0.01}),
            ParamSpec(key="obs_noise_std", type="float", default=0.01,
                      description="[SB3] 观测高斯噪声 std",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"}}),
            ParamSpec(key="push_robot", type="bool", default=False,
                      description="[SB3] 启用外力推扰",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"}}),
            ParamSpec(key="push_interval_steps", type="int", default=200,
                      widget="index", description="[SB3] 推扰间隔（步）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 1, "max": 1_000_000}),
            ParamSpec(key="push_force_range", type="list", default="[50, 150]",
                      widget="range_list",
                      description="[SB3] 推扰力范围（N）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "sb3"},
                            "min": 0.0, "max": 1000.0, "step": 1.0}),
            # ---- Isaac Lab backend ----
            # Friction event
            ParamSpec(key="enable_friction_rand", type="bool", default=True,
                      description="[IL] 启用摩擦随机化",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="friction_mode", type="enum", default="reset",
                      choices=["reset", "interval", "startup"],
                      description="[IL] 摩擦事件触发模式",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="static_friction_range", type="list", default="[0.5, 1.25]",
                      widget="range_list",
                      description="[IL] 静摩擦范围",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="dynamic_friction_range", type="list", default="[0.5, 1.25]",
                      widget="range_list",
                      description="[IL] 动摩擦范围",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="restitution_range", type="list", default="[0.0, 0.4]",
                      widget="range_list",
                      description="[IL] 弹性系数范围",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            # Mass event
            ParamSpec(key="enable_mass_rand", type="bool", default=True,
                      description="[IL] 启用质量随机化",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="mass_mode", type="enum", default="reset",
                      choices=["reset", "interval", "startup"],
                      description="[IL] 质量事件触发模式",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="mass_target_body", type="string", default="base",
                      description="[IL] 质量随机化目标 body",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="mass_offset_range", type="list", default="[-0.5, 0.5]",
                      widget="range_list",
                      description="[IL] 质量偏移范围（kg）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -10.0, "max": 10.0, "step": 0.05}),
            # External push
            ParamSpec(key="enable_external_push", type="bool", default=True,
                      description="[IL] 启用外力推扰",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="push_mode", type="enum", default="interval",
                      choices=["reset", "interval", "startup"],
                      description="[IL] 推扰事件触发模式",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="push_interval_range_s", type="list", default="[10.0, 15.0]",
                      widget="range_list",
                      description="[IL] 推扰时间间隔范围（s）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": 0.0, "max": 60.0, "step": 0.5}),
            ParamSpec(key="push_velocity_x_range", type="list", default="[-0.5, 0.5]",
                      widget="range_list",
                      description="[IL] 推扰 X 速度增量范围（m/s）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -5.0, "max": 5.0, "step": 0.05}),
            ParamSpec(key="push_velocity_y_range", type="list", default="[-0.5, 0.5]",
                      widget="range_list",
                      description="[IL] 推扰 Y 速度增量范围（m/s）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -5.0, "max": 5.0, "step": 0.05}),
            # Init pose event
            ParamSpec(key="enable_init_pose_rand", type="bool", default=True,
                      description="[IL] 启用初始位姿随机化",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="init_pose_mode", type="enum", default="reset",
                      choices=["reset", "interval", "startup"],
                      description="[IL] 初始位姿事件触发模式",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="init_pos_x_range", type="list", default="[-0.5, 0.5]",
                      widget="range_list",
                      description="[IL] 初始 X 位置范围（m）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -10.0, "max": 10.0, "step": 0.1}),
            ParamSpec(key="init_pos_y_range", type="list", default="[-0.5, 0.5]",
                      widget="range_list",
                      description="[IL] 初始 Y 位置范围（m）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -10.0, "max": 10.0, "step": 0.1}),
            ParamSpec(key="init_yaw_range", type="list", default="[-3.14, 3.14]",
                      widget="range_list",
                      description="[IL] 初始 Yaw 范围（rad）",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"},
                            "min": -3.14159, "max": 3.14159, "step": 0.01}),
            # Joint noise event
            ParamSpec(key="enable_joint_noise", type="bool", default=True,
                      description="[IL] 启用关节噪声",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="joint_noise_mode", type="enum", default="reset",
                      choices=["reset", "interval", "startup"],
                      description="[IL] 关节噪声事件触发模式",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="joint_pos_noise", type="float", default=0.25,
                      description="[IL] 关节位置噪声幅度",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
            ParamSpec(key="joint_vel_noise", type="float", default=0.1,
                      description="[IL] 关节速度噪声幅度",
                      meta={"conditional_on": {"key": "backend", "op": "==", "value": "isaac_lab"}}),
        ],
    )

    # ---- 画布 UI 钩子（动态隐藏端口与参数） ----

    @property
    def hidden_ports(self) -> set:
        """SB3 backend 下隐藏 IL 专属端口；IL backend 全部可见."""
        backend = str(self.get_parameter("backend", "sb3") or "sb3").strip()
        if backend == "isaac_lab":
            return set()
        return set(self._IL_PORTS)

    @property
    def hidden_params(self) -> set:
        """根据 backend 隐藏对侧 backend 的参数 keys."""
        backend = str(self.get_parameter("backend", "sb3") or "sb3").strip()
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
        backend = str(self.params.get("backend", "sb3") or "sb3").strip() or "sb3"
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