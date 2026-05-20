"""ActorSettingNode — Actor 初始姿态 / 接触传感器 / 执行器 / 动作空间 / 课程.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:ActorSettingNode``.

Partner of :class:`RobotNode` — together they form the complete Actor canvas
block. Owns EVERY user-editable knob that attaches to the robot:

    §1 Init pose           世界系 X/Y/Z spawn
    §2 Init joint angles   关节角度可视编辑器 (slider table)
    §2.5 Init-pose strategy mode/noise/keyframe/RSI (从已删除的 init_pose
                            节点迁移过来)
    §3 Contact sensors     接触 body 名 / history / air-time
    §4 Actuator overrides  stiffness / damping / effort / velocity
    §5 Action space        joint_names_expr / scale / use_default_offset
    §6 Action-scale 课程   start → end over ramp_iters

Reads ``robot_pipe`` (structural body list for the contact picker) and
re-emits a superset ``actor_pipe`` consumed by rewards / domain rand /
observation / action / velocity command nodes.

Phase 5 — IR-only joint contract:
    * ``init_joint_angles`` JSON dict keys MUST be IR role names from the
      bound robot's ``joint_ir_roles`` (e.g. ``"hip_FL"``, ``"thigh_FL"``).
      Physical USD names (``"FL_hip_joint"``) and vendor abbreviations
      (``"fl_hx"``) are rejected by both ``validate()`` (canvas-edit time)
      and ``spec_validator.R8`` (compile time).  See
      ``application.training.joint_ir.JointIRResolver`` for the canonical
      translation layer.
    * ``action_joint_names_expr`` items MUST be either a regex pattern
      (``".*"``, ``"hip_.*"``) or an exact IR role.  Literal physical
      names are rejected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class ActorSettingNode(BaseNode):
    """Layer A — Actor-layer state, spawn pose, and actuation settings."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----
    _OPTIONAL_INPUTS: set = set()

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="actor_setting",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.actor_setting.display",
        description_key="node.actor_setting.desc",
        icon="actor_setting.svg",
        inputs=[
            PortSpec(
                name="robot_pipe", type="robot_pipe",
                description="Structural body list / joint names / PD parameterization from the paired Robot node",
            ),
        ],
        outputs=[
            PortSpec(
                name="actor_pipe", type="actor_pipe",
                description="Superset Actor block consumed by rewards / DR / obs / action / cmd",
            ),
        ],
        parameters=[
            # §1 Init pose
            ParamSpec(key="init_pos_x", type="float", default=0.0, widget="range",
                      description="初始 X 位置 / Init world-frame X (m)",
                      meta={"min": -100.0, "max": 100.0, "step": 0.01}),
            ParamSpec(key="init_pos_y", type="float", default=0.0, widget="range",
                      description="初始 Y 位置 / Init world-frame Y (m)",
                      meta={"min": -100.0, "max": 100.0, "step": 0.01}),
            ParamSpec(key="init_pos_z", type="float", default=0.4, widget="range",
                      description="初始 Z 位置 / Init world-frame Z (m)",
                      meta={"min": 0.0, "max": 2.4, "step": 0.01}),
            # §2 Init joint angles — slider editor (widget="joint_pose_table")
            ParamSpec(key="init_joint_angles", type="json", default="{}",
                      widget="joint_pose_table",
                      description="初始关节角 (IR-role → radians) / Init joint angles (slider editor)",
                      meta={"language": "json"}),
            # §2.5 Episode init-pose strategy (migrated from init_pose node)
            ParamSpec(key="init_pose_mode", type="enum", default="default",
                      choices=["default", "reference_frame_0", "keyframe", "custom"],
                      description="起始姿态模式 / Episode init-pose mode"),
            ParamSpec(key="init_pose_noise_scale", type="float", default=0.05, widget="range",
                      description="起始姿态噪声 (rad) / Init-pose joint noise",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001}),
            ParamSpec(key="init_pose_keyframe_name", type="string", default="home",
                      description="MJCF keyframe 名 (mode=keyframe) / Keyframe name",
                      meta={"conditional_on": {"key": "init_pose_mode", "op": "==", "value": "keyframe"}}),
            ParamSpec(key="init_pose_rsi_prob", type="float", default=1.0, widget="range",
                      description="RSI 概率 (mode=reference_frame_0) / RSI probability",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01,
                            "conditional_on": {"key": "init_pose_mode", "op": "==", "value": "reference_frame_0"}}),
            ParamSpec(key="init_pose_rsi_sample_mode", type="enum", default="frame_0",
                      choices=["frame_0", "uniform_phase"],
                      description="RSI 采样模式 / RSI sample mode",
                      meta={"conditional_on": {"key": "init_pose_mode", "op": "==", "value": "reference_frame_0"}}),
            # §3 Contact sensors
            ParamSpec(key="contact_body_names", type="json", default="[]",
                      widget="ir_body_picker",
                      description="接触 body 名 / Contact-sensor body IR roles (multi-select; from upstream Robot Node)",
                      meta={"source": "body", "default_select_all": False, "language": "json"}),
            ParamSpec(key="contact_history_length", type="int", default=3, widget="index",
                      description="接触历史长度 / Contact history length",
                      meta={"min": 1, "max": 64}),
            ParamSpec(key="contact_track_air_time", type="bool", default=True,
                      description="追踪 air time / Track air time"),
            # §4 Actuator overrides
            ParamSpec(key="stiffness", type="float", default=25.0,
                      description="刚度 / Joint stiffness (PD P gain)"),
            ParamSpec(key="damping", type="float", default=0.5,
                      description="阻尼 / Joint damping (PD D gain)"),
            ParamSpec(key="effort_limit", type="float", default=30.0,
                      description="扭矩上限 / Effort (torque) limit"),
            ParamSpec(key="velocity_limit", type="float", default=30.0,
                      description="速度上限 / Joint velocity limit"),
            # §5 Action space
            # Empty list ``[]`` = "all joints" (matches spec_validator
            # _check_action_joints, which skips the check when expr is
            # empty). The IR joint picker pre-selects every upstream IR
            # role at open time; user deselects to suspend specific
            # joints from the action space.
            ParamSpec(key="action_joint_names_expr", type="json", default="[]",
                      widget="ir_joint_picker",
                      description="动作关节 IR 角色 / Action joint IR roles (multi-select; empty = all)",
                      meta={"source": "joint", "default_select_all": True, "language": "json"}),
            ParamSpec(key="action_scale", type="float", default=0.25, widget="range",
                      description="动作缩放 / Action scale",
                      meta={"min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="action_use_default_offset", type="bool", default=True,
                      description="动作使用默认 offset / Use default joint offset"),
            # §6 Action-scale curriculum
            ParamSpec(key="action_scale_curriculum_enabled", type="bool", default=False,
                      description="启用 action_scale 课程 / Enable action-scale curriculum"),
            ParamSpec(key="action_scale_curriculum_start", type="float", default=0.15,
                      widget="range",
                      description="课程起始 scale / Curriculum start scale",
                      meta={"min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="action_scale_curriculum_end", type="float", default=0.25,
                      widget="range",
                      description="课程结束 scale / Curriculum end scale",
                      meta={"min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="action_scale_curriculum_ramp_iters", type="int", default=300,
                      widget="index",
                      description="课程 ramp 迭代数 / Curriculum ramp iterations",
                      meta={"min": 1, "max": 1_000_000}),
            # §7 Review Pose — engine picker + full-width teal button.
            # The button submits a ReviewTask in the selected engine
            # (MuJoCo passive viewer in-process vs Isaac Sim Kit
            # subprocess) against the upstream RobotNode's resolved SKU.
            ParamSpec(key="review_pose_engine", type="enum", default="mujoco",
                      choices=["mujoco", "isaac_sim"],
                      description="Review 引擎 / Review engine (mujoco = MJCF, isaac_sim = USD)"),
            ParamSpec(key="_review_pose", type="string", default="",
                      widget="actor_review_pose_button",
                      description="▶ Review init pose in selected engine",
                      meta={"full_width_widget": True}),
        ],
    )

    # ------------------------------------------------------------------
    # Phase 5 IR-only joint contract — canvas-edit-time validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Static validation — IR-only joint contract.

        Checks performed beyond the base ``BaseNode.validate()`` (enum
        choices):

        * ``init_joint_angles`` parses as a JSON dict ``{str: number}`` and
          every key is a recognised IR joint role across *any* canonical
          family (quadruped / biped / humanoid / manipulator). Per-robot
          family scoping happens at compile time in
          ``spec_validator._check_joint_ir_canonical`` (R8).
        * ``action_joint_names_expr`` parses as a JSON list of strings.
          Literal items (no regex metachars) must also be recognised IR
          roles. Regex patterns (``.*``, ``hip_.*``) pass through.

        Vendor abbreviations (``fl_hx``, ``fl_hy``) and physical USD/MJCF
        names (``FL_hip_joint``) are rejected here.
        """
        errors: List[str] = list(super().validate())

        all_joint_ir_roles = self._all_joint_ir_roles()

        init_raw = self.get_parameter("init_joint_angles", "{}")
        init_errors = self._validate_init_joint_angles(init_raw, all_joint_ir_roles)
        errors.extend(init_errors)

        expr_raw = self.get_parameter("action_joint_names_expr", "[]")
        expr_errors = self._validate_action_joint_names_expr(expr_raw, all_joint_ir_roles)
        errors.extend(expr_errors)

        return errors

    @staticmethod
    def _all_joint_ir_roles() -> set:
        """Union of joint IR roles across every canonical morphology family.

        Used to flag obvious non-IR tokens (``fl_hx`` / ``FL_hip_joint``)
        without needing the bound robot's family. The strict per-robot
        family check is enforced at compile time by spec_validator R8.
        """
        from application.training.body_ir import (
            get_joint_ir_roles,
            list_known_families,
        )
        roles: set = set()
        for fam in list_known_families():
            roles.update(get_joint_ir_roles(fam))
        return roles

    @staticmethod
    def _validate_init_joint_angles(raw: Any, all_joint_ir_roles: set) -> List[str]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return [f"actor_setting.init_joint_angles: invalid JSON ({exc})"]
        else:
            parsed = raw
        if not isinstance(parsed, dict):
            return [
                f"actor_setting.init_joint_angles: expected JSON object "
                f"(IR-role → radians), got {type(parsed).__name__}"
            ]
        errors: List[str] = []
        bad_keys: List[str] = []
        for key, value in parsed.items():
            if not isinstance(key, str):
                errors.append(
                    f"actor_setting.init_joint_angles: key {key!r} is not a string"
                )
                continue
            if not isinstance(value, (int, float)):
                errors.append(
                    f"actor_setting.init_joint_angles[{key!r}]: value "
                    f"{value!r} is not a number"
                )
            if all_joint_ir_roles and key not in all_joint_ir_roles:
                bad_keys.append(key)
        if bad_keys:
            sample = sorted(all_joint_ir_roles)[:8]
            errors.append(
                f"actor_setting.init_joint_angles: keys {bad_keys} are not IR "
                f"joint roles. RELEASE accepts ONLY IR role names "
                f"(e.g. {sample!r}). Vendor abbreviations like 'fl_hx' and "
                f"physical USD/MJCF names like 'FL_hip_joint' are rejected — "
                f"see application.training.joint_ir.JointIRResolver for the "
                f"translation layer."
            )
        return errors

    @staticmethod
    def _validate_action_joint_names_expr(raw: Any, all_joint_ir_roles: set) -> List[str]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return [f"actor_setting.action_joint_names_expr: invalid JSON ({exc})"]
        else:
            parsed = raw
        if not isinstance(parsed, list):
            return [
                f"actor_setting.action_joint_names_expr: expected JSON list "
                f"of regex/IR-role strings, got {type(parsed).__name__}"
            ]
        errors: List[str] = []
        bad_literals: List[str] = []
        regex_metachars = set(".^$*+?()[]{}|\\")
        for item in parsed:
            if not isinstance(item, str):
                errors.append(
                    f"actor_setting.action_joint_names_expr: item {item!r} is not a string"
                )
                continue
            if any(ch in regex_metachars for ch in item):
                continue  # treat as regex pattern; matched against IR roles at compile time
            if all_joint_ir_roles and item not in all_joint_ir_roles:
                bad_literals.append(item)
        if bad_literals:
            sample = sorted(all_joint_ir_roles)[:8]
            errors.append(
                f"actor_setting.action_joint_names_expr: literal items {bad_literals} "
                f"are not IR joint roles. Use IR role names (e.g. {sample!r}) or a "
                f"regex pattern (e.g. '.*', 'hip_.*'). Vendor abbreviations and "
                f"physical USD/MJCF names are rejected."
            )
        return errors