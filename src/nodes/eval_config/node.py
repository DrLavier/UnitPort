# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""EvalConfigNode — 训练后评估配置.

DEMO 对应：``training_nodes.py:EvalConfigNode``.
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


class EvalConfigNode(BaseNode):
    """Layer D — Post-training evaluation configuration."""

    MANIFEST = manifest_from_toml(__file__)