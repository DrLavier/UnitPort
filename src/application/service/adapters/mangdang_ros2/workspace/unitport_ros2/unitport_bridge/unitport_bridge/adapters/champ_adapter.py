# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Champ adapter — 12-DOF quadrupeds running the champ_controller stack.

Publishes:
  * geometry_msgs/Twist on /cmd_vel             (base velocity)
  * std_msgs/Float64MultiArray on the position command topic (pose trim)

Conversion:
  * space=velocity_2d    → cmd_vel only
  * space=joint_position → position array, reordered per mapping
  * space=stop           → zero on both
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unitport_bridge.adapters.base import UnitPortAdapterBase


class ChampAdapter(UnitPortAdapterBase):
    CONTROLLER_TYPE = "champ"

    DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"
    DEFAULT_POSITION_TOPIC = "/joint_group_position_controller/command"

    def output_topics(self) -> List[Tuple[str, str]]:
        cmd_vel, position = self._topics()
        return [
            (cmd_vel, "geometry_msgs/msg/Twist"),
            (position, "std_msgs/msg/Float64MultiArray"),
        ]

    def convert(self, policy_action, joint_state=None) -> Dict[str, Any]:
        cmd_vel_topic, position_topic = self._topics()
        space = getattr(policy_action, "space", "") or ""
        out: Dict[str, Any] = {}

        if space == "velocity_2d":
            out[cmd_vel_topic] = self._to_twist(policy_action)
        elif space == "joint_position":
            out[position_topic] = self._to_float_array(
                list(getattr(policy_action, "joint_positions", []) or [])
            )
        elif space == "stop":
            out[cmd_vel_topic] = self._zero_twist()
            out[position_topic] = self._to_float_array([])
        else:
            # Unknown — emit zero cmd_vel as the safe default.
            out[cmd_vel_topic] = self._zero_twist()
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _topics(self) -> Tuple[str, str]:
        cmd_vel = self.DEFAULT_CMD_VEL_TOPIC
        position = self.DEFAULT_POSITION_TOPIC
        for b in self.mapping.get("topic_bindings") or []:
            topic = str(b.get("robot_topic", ""))
            if "cmd_vel" in topic:
                cmd_vel = topic
            elif "position" in topic or "/command" in topic:
                position = topic
        return cmd_vel, position

    @staticmethod
    def _to_twist(policy_action):
        from geometry_msgs.msg import Twist
        t = Twist()
        linear = getattr(policy_action, "linear", None)
        angular = getattr(policy_action, "angular", None)
        if linear is not None:
            t.linear.x = float(getattr(linear, "x", 0.0))
            t.linear.y = float(getattr(linear, "y", 0.0))
            t.linear.z = float(getattr(linear, "z", 0.0))
        if angular is not None:
            t.angular.x = float(getattr(angular, "x", 0.0))
            t.angular.y = float(getattr(angular, "y", 0.0))
            t.angular.z = float(getattr(angular, "z", 0.0))
        return t

    @staticmethod
    def _zero_twist():
        from geometry_msgs.msg import Twist
        return Twist()

    def _to_float_array(self, joint_positions: List[float]):
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        if joint_positions:
            msg.data = [float(v) for v in self.reorder_to_controller(joint_positions)]
        else:
            # Zero-fill to DOF count so a silent stop publishes the right shape.
            dof = int(self.profile.get("kinematics", {}).get("dof_count", 12) or 12)
            msg.data = [0.0] * dof
        return msg
