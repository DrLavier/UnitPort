# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Diff-drive adapter — wheeled robots with a single /cmd_vel topic."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unitport_bridge.adapters.base import UnitPortAdapterBase


class DiffDriveAdapter(UnitPortAdapterBase):
    CONTROLLER_TYPE = "diff_drive"

    DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"

    def output_topics(self) -> List[Tuple[str, str]]:
        return [(self._cmd_vel_topic(), "geometry_msgs/msg/Twist")]

    def convert(self, policy_action, joint_state=None) -> Dict[str, Any]:
        from geometry_msgs.msg import Twist
        topic = self._cmd_vel_topic()
        space = getattr(policy_action, "space", "") or ""
        t = Twist()
        if space in ("velocity_2d", "joint_position"):
            # Treat either as vx/vy/vyaw. Many wheeled stacks only honour vx+vyaw.
            linear = getattr(policy_action, "linear", None)
            angular = getattr(policy_action, "angular", None)
            if linear is not None:
                t.linear.x = float(getattr(linear, "x", 0.0))
                t.linear.y = float(getattr(linear, "y", 0.0))
            if angular is not None:
                t.angular.z = float(getattr(angular, "z", 0.0))
        # space == "stop" / unknown → zero
        return {topic: t}

    def _cmd_vel_topic(self) -> str:
        for b in self.mapping.get("topic_bindings") or []:
            topic = str(b.get("robot_topic", ""))
            if "cmd_vel" in topic:
                return topic
        return self.DEFAULT_CMD_VEL_TOPIC
