# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RewardsNode — Per-item 奖励配置 / Per-item reward configuration.

v2 颗粒度：rewards 节点输出 ``reward_pipe``（multi）—— 连到
training_motion 节点上每个 enabled item 行内独立的 reward 输入端口。
可放多个 rewards 节点，分别供应不同 item。

Backend binding：
    每张画布与单一训练后端绑定（``CanvasPage._backend_id``，see
    ``registers/backends.py``），本节点不暴露 ``backend`` ParamSpec —
    其 reward registry 由所在 canvas 自动决定：
        sb3        — REWARD_REGISTRY
        isaac_lab  — IL_REWARD_REGISTRY

    画布 ``spawn_node`` / ``bind_backend`` / ``from_workflow_dict`` 会通过
    ``NodeItem.apply_canvas_backend`` 把 canvas 解析后的 backend kind
    注入 ``self.params['backend']``。

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import List

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class RewardsNode(BaseNode):
    """Layer A — Per-item reward configuration node (registry-backed)."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="rewards",
        kind=NodeKind.CUSTOM,
        version="2.0.0",
        category="config",
        layer="A",
        display_name_key="node.rewards.display",
        description_key="node.rewards.desc",
        icon="rewards.svg",
        inputs=[],
        outputs=[
            PortSpec(
                name="reward_pipe", type="reward_pipe", multi=True,
                description="Per-item reward feed (connect to training_motion item rows)",
            ),
        ],
        parameters=[
            ParamSpec(
                key="reward_terms", type="json", default="{}",
                widget="registry_module",
                description="奖励项字典 / Reward terms (registry-backed editor)",
                meta={
                    "registry_id": "rewards",
                    "registry_id_il": "il_rewards",
                    "backend_keyed": True,
                    "pin_on_collapse": True,
                },
            ),
            ParamSpec(
                key="std", type="float", default=0.25, widget="range",
                description="全局 std (奖励项 default σ)",
                meta={"min": 0.0, "max": 5.0, "step": 0.01},
            ),
            ParamSpec(
                key="threshold", type="float", default=0.5, widget="range",
                description="全局 threshold",
                meta={"min": 0.0, "max": 1.0, "step": 0.01},
            ),
        ],
    )

    def validate(self) -> List[str]:
        return super().validate()
