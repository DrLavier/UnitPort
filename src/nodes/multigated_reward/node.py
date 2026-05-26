# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MultiGatedRewardNode — Multi-stage gated reward curriculum.

DEMO 对应：``training_nodes.py:MultiGatedRewardNode``.
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


class MultiGatedRewardNode(BaseNode):
    """Layer A — Multi-stage gated reward router."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="multigated_reward",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.multigated_reward.display",
        description_key="node.multigated_reward.desc",
        icon="multigated_reward.svg",
        inputs=[
            PortSpec(name="stage_0", type="total_reward"),
            PortSpec(name="stage_1", type="total_reward"),
        ],
        outputs=[
            PortSpec(name="total_reward", type="total_reward"),
            PortSpec(name="total_steps", type="int"),
        ],
        parameters=[
            ParamSpec(key="max_step_stage0", type="int", default=0, widget="index",
                      description="stage 0 硬超时 (0=不限)",
                      meta={"min": 0, "max": 1_000_000_000}),
            ParamSpec(key="reward_threshold_ratio", type="float", default=0.75,
                      widget="range",
                      description="stage 切换阈值比例 (rolling mean / best_ever)",
                      meta={"min": 0.0, "max": 1.0, "step": 0.01}),
            ParamSpec(key="min_ep_window", type="int", default=10, widget="index",
                      description="最小回合窗口",
                      meta={"min": 1, "max": 1000}),
            ParamSpec(key="plateau_window", type="int", default=10, widget="index",
                      description="稳定性窗口",
                      meta={"min": 1, "max": 1000}),
            ParamSpec(key="plateau_eps", type="float", default=0.005,
                      description="稳定性 ε"),
            ParamSpec(key="blend_steps", type="int", default=3000, widget="index",
                      description="切换混合步数",
                      meta={"min": 0, "max": 1_000_000}),
            ParamSpec(key="stage_behavior", type="enum", default="replace",
                      choices=["replace", "accumulate"],
                      description="Stage 行为：替换 / 累加"),
            ParamSpec(key="ep_reward_window", type="int", default=20, widget="index",
                      description="回合 reward 窗口",
                      meta={"min": 1, "max": 10000}),
            ParamSpec(key="stage1_ratio", type="float", default=1.5, widget="range",
                      description="stage 1 budget = max_step_stage0 × stage1_ratio",
                      meta={"min": 0.0, "max": 10.0, "step": 0.1}),
        ],
    )