# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""VisCheckNode — 训练里程碑 MuJoCo viewer.

DEMO 对应：``training_nodes.py:VisCheckNode``.
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


class VisCheckNode(BaseNode):
    """Layer D — Milestone visualization."""

    MANIFEST = manifest_from_toml(__file__)