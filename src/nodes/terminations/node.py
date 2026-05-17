"""TerminationsNode — Termination condition configuration.

DEMO 对应：``training_nodes.py:TerminationsNode``.

Backend binding：每张画布与单一训练后端绑定，本节点不暴露 backend ParamSpec —
其行为契约由所在 canvas 自动决定（见 RewardsNode 同款契约）：
    sb3        — TERMINATION_REGISTRY (fall_threshold, min_height, ...)
    isaac_lab  — IL_TERMINATION_REGISTRY (time_out, illegal_contact, base_height)
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


class TerminationsNode(BaseNode):
    """Layer A — Termination condition configuration."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="terminations",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.terminations.display",
        description_key="node.terminations.desc",
        icon="terminations.svg",
        inputs=[],
        outputs=[PortSpec(name="termination_config", type="termination_config")],
        parameters=[
            ParamSpec(key="termination_conditions", type="json", default="{}",
                      widget="registry_module",
                      description="终止条件 dict / Termination terms",
                      meta={"registry_id": "terminations",
                            "registry_id_il": "il_terminations",
                            "backend_keyed": True}),
            ParamSpec(key="termination_curriculum_enabled", type="bool", default=False,
                      description="启用 base_height termination curriculum"),
            ParamSpec(key="termination_curriculum_start", type="float", default=0.18,
                      description="curriculum 起始阈值 (m)"),
            ParamSpec(key="termination_curriculum_end", type="float", default=0.22,
                      description="curriculum 终止阈值 (m)"),
            ParamSpec(key="termination_curriculum_ramp_iters", type="int", default=500,
                      widget="index",
                      description="curriculum ramp 迭代数",
                      meta={"min": 1, "max": 1_000_000}),
        ],
    )