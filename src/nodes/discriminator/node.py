# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""DiscriminatorNode — 独立 AMP 判别器配置 / Standalone AMP discriminator.

DEMO 对应：``src/system/nodes/sys_nodes/training_nodes.py:DiscriminatorNode``.

Layer IL / category=config. 集中所有 adversarial motion prior 参数（先前散落
在 ILPolicyNetworkNode / ILPPOTrainerNode / TrainingMotionNode 三处），
让用户在一处推理 Rewards vs. Discriminator 的信号重叠。

Wiring::

    [TrainingMotion] ── reference_motion_config ──► [Discriminator]
                                                        │
                            discriminator_config ───────┘──► [IL Trainer]

Phase 2 frontend skeleton — ``execute()`` 留接口给 Stage C 后端 wiring。
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class DiscriminatorNode(BaseNode):
    """Layer IL — Standalone AMP discriminator configuration."""

    MANIFEST = manifest_from_toml(__file__)