# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MultiGatedRewardNode — Multi-stage gated reward curriculum.

DEMO 对应：``training_nodes.py:MultiGatedRewardNode``.
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


class MultiGatedRewardNode(BaseNode):
    """Layer A — Multi-stage gated reward router."""

    MANIFEST = manifest_from_toml(__file__)