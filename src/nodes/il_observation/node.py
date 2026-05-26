# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ILObservationNode — Isaac Lab observation configuration.

DEMO 对应：``training_nodes.py:ILObservationNode``.
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


class ILObservationNode(BaseNode):
    """Layer IL — Isaac Lab observation configuration."""

    _OPTIONAL_INPUTS: set = {"actor_pipe", "command_pipe"}

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="il_observation",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="IL",
        display_name_key="node.il_observation.display",
        description_key="node.il_observation.desc",
        icon="il_observation.svg",
        inputs=[
            PortSpec(name="actor_pipe", type="actor_pipe", optional=True),
            PortSpec(name="command_pipe", type="command_pipe", optional=True),
        ],
        outputs=[PortSpec(name="obs_vector", type="obs_vector")],
        parameters=[
            ParamSpec(key="obs_terms", type="json", default="{}",
                      widget="registry_module",
                      description="观测项 dict / Observation terms (IL registry-backed)",
                      meta={"registry_id": "il_obs"}),
            ParamSpec(key="group_name", type="string", default="policy",
                      description="ObsGroup 名"),
            ParamSpec(key="enable_corruption", type="bool", default=True,
                      description="启用观测噪声"),
            ParamSpec(key="corruption_noise_std", type="float", default=0.05,
                      widget="range",
                      description="噪声标准差",
                      meta={"min": 0.0, "max": 1.0, "step": 0.001}),
            ParamSpec(key="obs_noise_curriculum_enabled", type="bool", default=False,
                      description="启用 obs noise curriculum"),
            ParamSpec(key="obs_noise_curriculum_start", type="float", default=0.25,
                      description="curriculum 起始倍率"),
            ParamSpec(key="obs_noise_curriculum_end", type="float", default=1.0,
                      description="curriculum 终止倍率"),
            ParamSpec(key="obs_noise_curriculum_ramp_iters", type="int", default=500,
                      widget="index",
                      description="curriculum ramp iter 数",
                      meta={"min": 1, "max": 1_000_000}),
        ],
    )