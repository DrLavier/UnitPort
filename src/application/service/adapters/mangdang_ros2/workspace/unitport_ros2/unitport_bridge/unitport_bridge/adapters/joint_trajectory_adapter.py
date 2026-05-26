# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint-trajectory adapter — manipulators + bipeds using ros2_control's
JointTrajectoryController."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unitport_bridge.adapters.base import UnitPortAdapterBase


class JointTrajectoryAdapter(UnitPortAdapterBase):
    CONTROLLER_TYPE = "joint_trajectory"

    DEFAULT_TOPIC = "/joint_trajectory_controller/joint_trajectory"
    POINT_DURATION_S = 0.1  # 10 Hz trajectory point spacing

    def output_topics(self) -> List[Tuple[str, str]]:
        return [(self._topic(), "trajectory_msgs/msg/JointTrajectory")]

    def convert(self, policy_action, joint_state=None) -> Dict[str, Any]:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        msg = JointTrajectory()
        joints = self.enabled_joints()
        msg.joint_names = [str(j["joint_name"]) for j in joints]

        space = getattr(policy_action, "space", "") or ""
        pt = JointTrajectoryPoint()
        if space == "joint_position":
            pt.positions = list(self.reorder_to_controller(
                list(getattr(policy_action, "joint_positions", []) or [])
            ))
        elif space == "joint_effort":
            pt.effort = list(self.reorder_to_controller(
                list(getattr(policy_action, "joint_efforts", []) or [])
            ))
        else:
            pt.positions = [0.0] * len(msg.joint_names)

        d = Duration()
        d.sec = int(self.POINT_DURATION_S)
        d.nanosec = int((self.POINT_DURATION_S - int(self.POINT_DURATION_S)) * 1e9)
        pt.time_from_start = d
        msg.points = [pt]
        return {self._topic(): msg}

    def _topic(self) -> str:
        for b in self.mapping.get("topic_bindings") or []:
            topic = str(b.get("robot_topic", ""))
            if "trajectory" in topic:
                return topic
        return self.DEFAULT_TOPIC
