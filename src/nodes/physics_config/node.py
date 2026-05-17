"""PhysicsConfigNode — 仿真时间步 + 执行器配置.

DEMO 对应：``training_nodes.py:PhysicsConfigNode``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from application.compiler.nodes import (
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

    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="physics_config",
        kind=NodeKind.CUSTOM,
        version="1.0.0",
        category="config",
        layer="A",
        display_name_key="node.physics_config.display",
        description_key="node.physics_config.desc",
        icon="physics_config.svg",
        inputs=[
            PortSpec(name="robot_pipe", type="robot_pipe", optional=True),
            PortSpec(name="robot_spec", type="robot_spec", optional=True,
                     description="Legacy robot spec port — both inputs are sim-agnostic"),
        ],
        outputs=[PortSpec(name="physics_config", type="physics_config")],
        parameters=[
            ParamSpec(key="sim_dt", type="float", default=0.002,
                      description="物理步长 (s)"),
            ParamSpec(key="control_dt", type="float", default=0.02,
                      description="控制步长 (s) — 必须是 sim_dt 的整数倍"),
            ParamSpec(key="episode_max_steps", type="int", default=1000, widget="index",
                      description="回合最大步数",
                      meta={"min": 1, "max": 100_000_000}),
            ParamSpec(key="action_type", type="enum", default="joint_position",
                      choices=["joint_position", "torque", "joint_velocity"],
                      description="动作类型"),
        ],
    )

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