# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PhysicsConfigNode — 仿真时间步 + 执行器配置.

DEMO 对应：``training_nodes.py:PhysicsConfigNode``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
    manifest_from_toml,
    NODE_MANIFEST_SCHEMA,
    BaseNode,
    NodeKind,
    NodeManifest,
    ParamSpec,
    PortSpec,
)


class PhysicsConfigNode(BaseNode):
    """Layer A — Physics simulation timing + actuator configuration."""

    _OPTIONAL_INPUTS: set = {"robot_pipe", "robot_spec"}

    MANIFEST = manifest_from_toml(__file__)

    def validate(self) -> List[str]:
        errors = super().validate()
        sim_raw = self.get_parameter("sim_dt")
        ctrl_raw = self.get_parameter("control_dt")
        try:
            sim = float(sim_raw)
        except (TypeError, ValueError):
            errors.append(
                f"node={self.node_id}: sim_dt={sim_raw!r} cannot be parsed as float"
            )
            return errors
        try:
            ctrl = float(ctrl_raw)
        except (TypeError, ValueError):
            errors.append(
                f"node={self.node_id}: control_dt={ctrl_raw!r} cannot be parsed as float"
            )
            return errors
        if sim <= 0:
            errors.append(f"node={self.node_id}: sim_dt must be > 0 (got {sim})")
            return errors
        if ctrl <= 0:
            errors.append(f"node={self.node_id}: control_dt must be > 0 (got {ctrl})")
            return errors
        if abs(round(ctrl / sim) - (ctrl / sim)) > 1e-6:
            errors.append(
                f"node={self.node_id}: control_dt ({ctrl}) 必须是 sim_dt ({sim}) 的整数倍"
            )
        return errors