# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
    manifest_from_toml,
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

    MANIFEST = manifest_from_toml(__file__)

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