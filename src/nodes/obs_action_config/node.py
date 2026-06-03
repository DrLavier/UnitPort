# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ObsActionConfigNode — Observation / action space configuration.

DEMO 对应：``training_nodes.py:ObsActionConfigNode``.
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


class ObsActionConfigNode(BaseNode):
    """Layer A — Observation / action space configuration."""

    PRESET_NAMES = ("custom", "unitport_go2_v1", "community_go2_sac_34d")

    MANIFEST = manifest_from_toml(__file__)