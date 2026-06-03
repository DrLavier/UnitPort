# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ILPolicyNetworkNode — Isaac Lab MLP actor-critic.

DEMO 对应：``training_nodes.py:ILPolicyNetworkNode``.

The legacy ``disc_hidden_dims`` field was removed in the strict-canvas
migration; discriminator dims now live exclusively on DiscriminatorNode.
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


class ILPolicyNetworkNode(BaseNode):
    """Layer IL — MLP actor-critic policy network."""

    MANIFEST = manifest_from_toml(__file__)