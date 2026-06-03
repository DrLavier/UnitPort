# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ILObservationNode — Isaac Lab observation configuration.

DEMO 对应：``training_nodes.py:ILObservationNode``.
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


class ILObservationNode(BaseNode):
    """Layer IL — Isaac Lab observation configuration."""

    _OPTIONAL_INPUTS: set = {"actor_pipe", "command_pipe"}

    MANIFEST = manifest_from_toml(__file__)