# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class TerminationsNode(BaseNode):
    """Layer A — Termination condition configuration."""

    MANIFEST = manifest_from_toml(__file__)