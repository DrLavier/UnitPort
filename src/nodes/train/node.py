# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TrainNode — SB3 training execution (passthrough，无 user 参数).

DEMO 对应：``training_nodes.py:TrainNode``.
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    PortSpec,
)


class TrainNode(BaseNode):
    """Layer C — SB3 training execution."""

    _OPTIONAL_INPUTS: set = {"eval_config"}

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="train",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="training",
        layer="C",
        display_name_key="node.train.display",
        description_key="node.train.desc",
        icon="train.svg",
        inputs=[
            PortSpec(name="env_config", type="env_config"),
            PortSpec(name="algo_config", type="algo_config"),
            PortSpec(name="eval_config", type="eval_config", optional=True),
        ],
        outputs=[
            PortSpec(name="train_pipe", type="train_pipe"),
            PortSpec(name="vis_check", type="vis_check"),
        ],
        parameters=[],
    )

    @property
    def hidden_ports(self) -> set:
        return set(self._HIDDEN_PORTS)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """SB3 sink — reads ``algo_config.backend`` to dispatch via
        :func:`select_backend`. Stage 4 surface; Stage 10 lands the SB3
        subprocess launcher under the same selector contract.
        """
        from application.training.trainer_runtime import submit_sb3_trainer

        return submit_sb3_trainer(
            node_id=self.node_id,
            schema_id=self.MANIFEST.id,
            inputs=inputs or {},
        )