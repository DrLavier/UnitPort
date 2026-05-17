"""HeartbeatMatchProbe — verify the robot-side estop watchdog hears us.

The robot-side ``unitport_estop`` watchdog subscribes to
``/unitport/heartbeat`` and silences /cmd_vel with zero Twists when the
heartbeat goes quiet. The host's :class:`HeartbeatPublisher` publishes at
2 Hz with QoS reliability=RELIABLE; the robot subscribes with default
RCLPY (also RELIABLE). DDS only matches the pair when reliability matches
exactly — a single mismatched policy keeps the writer alone forever and
the watchdog never sees the heartbeat -> robot stays in safe-zero mode
even though everything looks fine on the host.

Tri-state (matching :meth:`NativeDDSBridge.publication_matched_count`):

* ``None`` — HeartbeatPublisher hasn't started yet (bridge alive but
  adapter session not open). INFO, not a failure.
* ``0``   — heartbeat writer exists but no robot subscriber matched.
  WARNING with the QoS RELIABILITY-mismatch root cause language.
* ``>=1`` — matched. OK.

Severity tops out at WARNING because not every brand runs a
heartbeat-gated estop watchdog; missing matches on a brand without one
is harmless.
"""

from __future__ import annotations

from typing import List, Optional

from application.service.adapters.heartbeat_publisher import HeartbeatPublisher
from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


class HeartbeatMatchProbe(DiagnosticProbe):
    id = "heartbeat_match"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        bridge = ctx.bridge
        if bridge is None:
            return [self.info(
                "bridge not active; skipping heartbeat match check",
            )]

        topic = HeartbeatPublisher.TOPIC
        count: Optional[int]
        try:
            count = bridge.publication_matched_count(topic)
        except Exception as exc:
            return [self.warning(
                f"could not read publication_matched_count for {topic}",
                detail=str(exc),
            )]

        if count is None:
            return [self.info(
                f"no heartbeat writer on {topic} yet "
                "(adapter session may not be fully open)",
            )]

        if count > 0:
            return [self.ok(
                f"{topic} writer matched to {count} reader(s)",
            )]

        return [self.warning(
            f"{topic} writer has no matched subscriber",
            detail=(
                f"Our HeartbeatPublisher publishes '{topic}' at 2 Hz with "
                "QoS reliability=RELIABLE, but no robot-side DataReader is "
                "currently matched. If the robot runs an estop watchdog "
                "(unitport_estop on Mangdang), it will not see our heartbeat "
                "and will keep silencing /cmd_vel with zero Twists -> the "
                "robot won't respond to teleop even though the bridge looks "
                "healthy.\n\nMost common cause: QoS RELIABILITY mismatch "
                "between host writer and robot subscriber. Less common: the "
                "robot-side estop unit is not running at all (in which case "
                "this warning is benign and can be ignored)."
            ),
            requires_ssh=False,
        )]


__all__ = ["HeartbeatMatchProbe"]
