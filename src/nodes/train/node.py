# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TrainNode — SB3 training execution (passthrough，无 user 参数).

DEMO 对应：``training_nodes.py:TrainNode``.
"""

from __future__ import annotations

from typing import Any, Dict

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    PortSpec,
)


class TrainNode(BaseNode):
    """Layer C — SB3 training execution."""

    _OPTIONAL_INPUTS: set = {"eval_config"}

    MANIFEST = manifest_from_toml(__file__)

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