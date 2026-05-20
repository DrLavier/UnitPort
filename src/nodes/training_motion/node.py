"""TrainingMotionNode — 统一 training-motion 节点 (替代 TrainingCommandsNode + ReferenceMotionNode).

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:TrainingMotionNode``.

行为契约：
    - 单节点持有「training item → motion_tag → 命令 envelope → reference clip」
      全链路绑定，保证 train = reference = execute 三者一致
    - command_pipe 输出 CommandSchema.to_dict() + task_items list
    - reference_motion_config 输出 entries / task_item_to_clip(s) /
      tag_map / pack_refs / format_mixed
    - gait_* 仅在 gait_enabled == true 可见
    - command_curriculum_* 仅在 command_curriculum_enabled == true 可见

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


# Prefix for the dynamic per-item reward input port created on a training_motion
# NodeItem. Canvas UI uses this to (a) generate ports from the ``training_items``
# dict, (b) recognize/clean up orphaned ports when items are removed, and (c)
# let the training-spec compiler reverse-resolve ``port_name → item_id``.
REWARD_PORT_PREFIX = "reward_in__"


def reward_port_name(item_id: str) -> str:
    """Return the canonical canvas port name for a per-item reward input."""
    return f"{REWARD_PORT_PREFIX}{item_id}"


def item_id_from_reward_port(port_name: str) -> str | None:
    """Reverse of :func:`reward_port_name` — None if the port name doesn't match."""
    if isinstance(port_name, str) and port_name.startswith(REWARD_PORT_PREFIX):
        return port_name[len(REWARD_PORT_PREFIX):]
    return None


def _parse_training_items(value: Any) -> Dict[str, dict]:
    """Decode the ``training_items`` param value into a dict[str, dict]."""
    if isinstance(value, dict):
        d = value
    elif isinstance(value, str) and value.strip():
        try:
            d = json.loads(value)
        except Exception:
            return {}
    else:
        return {}
    if not isinstance(d, dict):
        return {}
    return {
        str(k): (v if isinstance(v, dict) else {})
        for k, v in d.items()
    }


def iter_reward_port_specs(params: Dict[str, Any]) -> List[PortSpec]:
    """Build dynamic reward input PortSpecs from the current ``training_items``.

    One PortSpec per **enabled** item — disabled items don't get a port so the
    rewards-node fan-out stays uncluttered. Called by ``NodeItem._compute_layout``
    via the inline-row anchor hook (see canvas/items.py).
    """
    items = _parse_training_items(params.get("training_items"))
    specs: List[PortSpec] = []
    for item_id, entry in items.items():
        if not bool(entry.get("enabled", True)):
            continue
        specs.append(PortSpec(
            name=reward_port_name(item_id),
            type="reward_pipe",
            optional=True,
            multi=True,
            description=f"Per-item reward feed for '{item_id}'",
        ))
    return specs


# H4-A: ``registers.commands`` is the canonical catalog of training items.
# An empty default means "let the compiler seed from registers.commands.
# default_items_for_families(robot.families)" — the 11 builtin templates
# (stand / walk / turn / pace + 7 directional) flow in at compile time, so
# this node param no longer carries inline JSON that drifts from the registry.
_DEFAULT_TRAINING_ITEMS = "{}"


class TrainingMotionNode(BaseNode):
    """Layer B — Unified training-motion authoring node."""

    # ---- DEMO 行为契约（class-level 提示，画布 UI 读取） ----

    # 条件可见参数集合（与 manifest meta.conditional_on 一一对应）
    _GAIT_ONLY_KEYS: tuple = (
        "gait_frequency_range", "body_height_range",
        "step_height_range", "gait_presets",
    )
    _CMD_CURRICULUM_ONLY_KEYS: tuple = (
        "command_curriculum_start",
        "command_curriculum_end",
        "command_curriculum_ramp_iters",
    )

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="training_motion",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="B",
        display_name_key="node.training_motion.display",
        description_key="node.training_motion.desc",
        icon="training_motion.svg",
        inputs=[
            PortSpec(name="actor_pipe", type="actor_pipe", optional=True,
                     description="Actor stream (from actor_setting)"),
        ],
        outputs=[
            PortSpec(name="command_pipe", type="command_pipe"),
            PortSpec(name="reference_motion_config", type="reference_motion_config"),
        ],
        parameters=[
            # §1 Training items
            ParamSpec(key="training_items", type="json",
                      default=_DEFAULT_TRAINING_ITEMS,
                      widget="training_items",
                      description="训练项字典（item_id → {enabled, speed, clip, advanced}）",
                      meta={"pin_on_collapse": True}),
            # §2 Global stick-shaping
            ParamSpec(key="mapping_mode", type="enum", default="linear",
                      choices=["linear", "deadzone", "exponential"],
                      description="命令映射模式 [deploy-only: 不影响 sim 训练]"),
            ParamSpec(key="deadzone", type="float", default=0.10,
                      widget="range",
                      description="死区 [deploy-only: 仅 deploy CommandBus 使用]",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01,
                            "conditional_on": {"key": "mapping_mode", "op": "==", "value": "deadzone"}}),
            ParamSpec(key="curve_exponent", type="float", default=2.00,
                      widget="range",
                      description="曲线指数 [deploy-only: 仅 deploy CommandBus 使用]",
                      meta={"min": 0.1, "max": 10.0, "step": 0.01,
                            "conditional_on": {"key": "mapping_mode", "op": "==", "value": "exponential"}}),
            ParamSpec(key="runtime_clip", type="bool", default=True,
                      description="运行时裁剪命令 [deploy-only: sim 训练 CommandTerm 用 ranges 内部约束]"),
            # §3 Walk-These-Ways gait parameterisation
            ParamSpec(key="gait_enabled", type="bool", default=False,
                      description="启用 gait 参数化"),
            ParamSpec(key="gait_frequency_range", type="list", default="[1.5, 3.5]",
                      widget="range_list",
                      description="步频范围（Hz）",
                      meta={"conditional_on": {"key": "gait_enabled", "op": "==", "value": True},
                            "min": 0.0, "max": 10.0, "step": 0.1}),
            ParamSpec(key="body_height_range", type="list", default="[0.28, 0.40]",
                      widget="range_list",
                      description="身体高度范围（m）",
                      meta={"conditional_on": {"key": "gait_enabled", "op": "==", "value": True},
                            "min": 0.0, "max": 2.0, "step": 0.01}),
            ParamSpec(key="step_height_range", type="list", default="[0.03, 0.15]",
                      widget="range_list",
                      description="抬腿高度范围（m）",
                      meta={"conditional_on": {"key": "gait_enabled", "op": "==", "value": True},
                            "min": 0.0, "max": 0.5, "step": 0.01}),
            ParamSpec(key="gait_presets", type="string", default="",
                      description="gait 预设名",
                      meta={"conditional_on": {"key": "gait_enabled", "op": "==", "value": True}}),
            # §4 Episode sampler
            ParamSpec(key="resampling_time_range", type="list", default="[4.0, 12.0]",
                      widget="range_list",
                      description="命令重采样时间范围（s）",
                      meta={"min": 0.0, "max": 30.0, "step": 0.1}),
            ParamSpec(key="zero_command_probability", type="float", default=0.1,
                      widget="range", description="零命令概率",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="cmd_step_change_prob", type="float", default=0.01,
                      widget="range", description="每步命令切换概率",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001}),
            # §5 Reference-motion consumption
            ParamSpec(key="consumer_mode", type="enum", default="amp",
                      choices=["tracking", "amp", "both"],
                      description="Reference clip 消费模式"),
            ParamSpec(key="phase_mode", type="enum", default="loop",
                      choices=["loop", "once", "clamp"],
                      description="Phase 模式"),
            ParamSpec(key="motion_fps", type="float", default=50.0,
                      description="Motion clip FPS"),
            ParamSpec(key="random_start_phase", type="bool", default=True,
                      description="随机起始 phase"),
            # §6 Command velocity curriculum
            ParamSpec(key="command_curriculum_enabled", type="bool", default=False,
                      description="启用命令速度 curriculum"),
            ParamSpec(key="command_curriculum_start", type="float", default=0.25,
                      widget="range", description="Curriculum 起始倍率",
                      meta={"conditional_on": {"key": "command_curriculum_enabled", "op": "==", "value": True},
                            "min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="command_curriculum_end", type="float", default=1.0,
                      widget="range", description="Curriculum 终止倍率",
                      meta={"conditional_on": {"key": "command_curriculum_enabled", "op": "==", "value": True},
                            "min": 0.0, "max": 5.0, "step": 0.01}),
            ParamSpec(key="command_curriculum_ramp_iters", type="int", default=800,
                      widget="index", description="Curriculum ramp 外层 iter 数",
                      meta={"conditional_on": {"key": "command_curriculum_enabled", "op": "==", "value": True},
                            "min": 1, "max": 1_000_000}),
        ],
    )

    # ---- 画布 UI 钩子（动态隐藏条件参数） ----

    @property
    def hidden_params(self) -> set:
        """根据 gait_enabled / command_curriculum_enabled 隐藏对应参数."""
        hidden: set = set()
        gait_on = self.get_parameter("gait_enabled", False)
        if not (gait_on is True or str(gait_on).strip().lower() == "true"):
            hidden.update(self._GAIT_ONLY_KEYS)
        cur_on = self.get_parameter("command_curriculum_enabled", False)
        if not (cur_on is True or str(cur_on).strip().lower() == "true"):
            hidden.update(self._CMD_CURRICULUM_ONLY_KEYS)
        return hidden