# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""estop_node — heartbeat watchdog.

Subscribes to /unitport/heartbeat (expected ≥1 Hz from the host). If the
heartbeat stops for longer than the timeout, publishes a zero Twist on
/cmd_vel and an Alert on /unitport/alerts, then keeps sending zeros until
either the heartbeat resumes or the node is shut down.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from unitport_msgs.msg import Alert, Heartbeat


HEARTBEAT_TOPIC = "/unitport/heartbeat"
CMD_VEL_TOPIC = "/cmd_vel"
ALERT_TOPIC = "/unitport/alerts"

HEARTBEAT_TIMEOUT_S = 2.0
SAFE_PUBLISH_HZ = 10.0


class EstopNode(Node):

    def __init__(self) -> None:
        super().__init__("unitport_estop")
        self._last_heartbeat = 0.0
        self._triggered = False

        self.create_subscription(Heartbeat, HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self._cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._alert_pub = self.create_publisher(Alert, ALERT_TOPIC, 10)
        self.create_timer(1.0 / SAFE_PUBLISH_HZ, self._tick)
        self.get_logger().info("estop_node up — waiting for heartbeat")

    def _on_heartbeat(self, msg: Heartbeat) -> None:
        # Arming is presence-based: any message inside the timeout window
        # keeps estop happy. Payload fields (robot_state, …) are observable
        # for diagnostics but the watchdog itself doesn't gate on them.
        self._last_heartbeat = time.monotonic()
        if self._triggered:
            self.get_logger().info("heartbeat resumed — estop cleared")
            self._triggered = False

    def _tick(self) -> None:
        now = time.monotonic()
        if self._last_heartbeat == 0.0:
            # Never saw one — hold safe-state zeros silently until we do.
            self._publish_zero()
            return
        if now - self._last_heartbeat > HEARTBEAT_TIMEOUT_S:
            if not self._triggered:
                self._triggered = True
                alert = Alert()
                alert.header.stamp = self.get_clock().now().to_msg()
                alert.kind = "estop_triggered"
                alert.severity = "critical"
                alert.detail = f"heartbeat timeout {HEARTBEAT_TIMEOUT_S}s"
                self._alert_pub.publish(alert)
                self.get_logger().error(alert.detail)
            self._publish_zero()

    def _publish_zero(self) -> None:
        try:
            self._cmd_vel_pub.publish(Twist())
        except Exception as exc:
            self.get_logger().warn(f"zero cmd_vel publish failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = EstopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
