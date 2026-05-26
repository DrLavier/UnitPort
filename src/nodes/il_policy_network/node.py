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
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class ILPolicyNetworkNode(BaseNode):
    """Layer IL — MLP actor-critic policy network."""

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="il_policy_network",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="IL",
        display_name_key="node.il_policy_network.display",
        description_key="node.il_policy_network.desc",
        icon="il_policy_network.svg",
        inputs=[PortSpec(name="obs_vector", type="obs_vector")],
        outputs=[PortSpec(name="policy", type="policy")],
        parameters=[
            ParamSpec(key="actor_hidden_dims", type="list", default="[128, 64, 32]",
                      widget="code",
                      description="actor 隐藏层 (JSON list)",
                      meta={"language": "json"}),
            ParamSpec(key="critic_hidden_dims", type="list", default="[128, 64, 32]",
                      widget="code",
                      description="critic 隐藏层 (JSON list)",
                      meta={"language": "json"}),
            ParamSpec(key="activation", type="enum", default="elu",
                      choices=["relu", "elu", "tanh", "leaky_relu"],
                      description="激活函数"),
            ParamSpec(key="init_noise_std", type="float", default=-1.0,
                      description="初始噪声 std（-1=自动）"),
        ],
    )