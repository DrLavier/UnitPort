"""ObsActionConfigNode — Observation / action space configuration.

DEMO 对应：``training_nodes.py:ObsActionConfigNode``.
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class ObsActionConfigNode(BaseNode):
    """Layer A — Observation / action space configuration."""

    PRESET_NAMES = ("custom", "unitport_go2_v1", "community_go2_sac_34d")

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="obs_action_config",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.obs_action_config.display",
        description_key="node.obs_action_config.desc",
        icon="obs_action_config.svg",
        inputs=[],
        outputs=[PortSpec(name="obs_action_config", type="obs_action_config")],
        parameters=[
            ParamSpec(key="contract_preset", type="enum", default="custom",
                      choices=list(PRESET_NAMES),
                      description="契约预设（非 custom 时锁定下方字段）"),
            ParamSpec(key="obs_components", type="string",
                      default="joint_pos joint_vel imu command previous_action",
                      widget="picker_obs_components",
                      description="观测组件（多选）",
                      meta={"choices": ["joint_pos", "joint_vel", "imu", "command",
                                        "previous_action", "height_scan",
                                        "contact", "phase"]}),
            ParamSpec(key="obs_clip_range", type="float", default=100.0,
                      description="观测裁剪范围"),
            ParamSpec(key="frame_stack", type="int", default=1, widget="index",
                      description="帧栈深度",
                      meta={"min": 1, "max": 32}),
            ParamSpec(key="action_type", type="enum", default="joint_position",
                      choices=["joint_position", "torque"],
                      description="动作类型"),
            ParamSpec(key="action_scale", type="float", default=1.0,
                      description="动作缩放"),
            ParamSpec(key="action_clip", type="float", default=1.0,
                      description="动作裁剪上限"),
        ],
    )