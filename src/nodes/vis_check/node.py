# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""VisCheckNode — 训练里程碑 MuJoCo viewer.

DEMO 对应：``training_nodes.py:VisCheckNode``.
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


class VisCheckNode(BaseNode):
    """Layer D — Milestone visualization."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="vis_check",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="D",
        display_name_key="node.vis_check.display",
        description_key="node.vis_check.desc",
        icon="vis_check.svg",
        inputs=[
            PortSpec(name="vis_check", type="vis_check", optional=True),
        ],
        outputs=[],
        parameters=[
            ParamSpec(key="trigger_mode", type="enum", default="step_interval",
                      choices=["step_interval", "episode_split"],
                      description="触发模式 / Trigger mode"),
            ParamSpec(key="vis_start_step", type="int", default=50000, widget="index",
                      description="起始步 (step_interval mode)",
                      meta={"min": 0, "max": 100_000_000}),
            ParamSpec(key="vis_interval_steps", type="int", default=50000, widget="index",
                      description="间隔步 (step_interval mode)",
                      meta={"min": 1, "max": 100_000_000}),
            ParamSpec(key="num_vis_checks", type="int", default=5, widget="index",
                      description="总检查次数 (episode_split mode)",
                      meta={"min": 1, "max": 1000}),
            ParamSpec(key="vis_episodes", type="int", default=3, widget="index",
                      description="每次回合数",
                      meta={"min": 1, "max": 100}),
            ParamSpec(key="deterministic", type="bool", default=True,
                      description="确定性策略 / Deterministic actions"),
        ],
    )