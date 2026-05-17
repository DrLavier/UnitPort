"""MangdangROS2Adapter — Mini Pupper v2 brand specialisation.

Thin subclass of :class:`BaseROS2Adapter`: every kinematics-agnostic plumbing
bit (bridge lifecycle, declared subscriptions, teleop pump, heartbeat) lives
in the base. This file only carries Mini Pupper v2 specifics:

- ``BRAND_ID = "mangdang"`` — matches the brand_id the topic_table registers
- ``DEFAULT_DOMAIN_ID = 42`` — matches DEMO's hard-coded
  ``ConnectionProfile.ros_domain_id`` and the ``ROS_DOMAIN_ID=42`` written
  to ``/etc/unitport/env`` on the robot by P4B brand-runtime push
- :meth:`capabilities` — declares 2D quadruped teleop (vx/vy/vyaw) +
  estop on /cmd_vel + per-joint PID tuning placeholder for Phase 6
- :meth:`_safety_clamp` — applies the v2 datasheet envelope from
  :mod:`safety_overlay`

The Flask HTTP control panel (/pupper/status/start, /walk/vel_x/{n}, ...) is
intentionally NOT wired here: the user's pupper is in unitport mode and the
factory Flask service is stopped. Phase 4's SSH transport will reintroduce
mode switching; if the user ever rolls back to Flask mode, that path can
re-enter this adapter as a separate ``activate_via_flask`` method.
"""

from __future__ import annotations

from typing import Any, Dict, List

from unitport_sdk import log_info

from application.service.adapters.base_ros2_adapter import BaseROS2Adapter
from application.service.adapters.capabilities import (
    CapabilitySet,
    EstopCapability,
    TeleopCapability,
)
from application.service.adapters.mangdang_ros2.probes import mangdang_probes
from application.service.adapters.mangdang_ros2.safety_overlay import (
    MAX_ANG_VEL_Z_RPS,
    MAX_LIN_VEL_X_MPS,
    MAX_LIN_VEL_Y_MPS,
    clamp_quadruped_twist,
)
from application.service.diagnostics.base_probe import DiagnosticProbe


class MangdangROS2Adapter(BaseROS2Adapter):
    """Host-side adapter for any Mini Pupper running UnitPort + CHAMP."""

    BRAND_ID: str = "mangdang"
    # 42 matches /etc/unitport/env on the robot side (DEMO P4B writes 42).
    # If you change this, you MUST also patch the robot-side env file or
    # discovery will silently fail (Phase 2 dark-bridge symptom).
    DEFAULT_DOMAIN_ID: int = 42
    # CHAMP / mini_pupper_bringup publishes everything on the root namespace.
    TOPIC_NAMESPACE: str = ""

    # Teleop pump runs at 50 Hz — matches CHAMP's expected control loop rate.
    TELEOP_RATE_HZ: float = 50.0
    # Deadman: zero twist after 0.5s of no input. CHAMP's own watchdog is
    # ~1s, so 0.5s gives the host first crack at safe-stop before robot-side
    # logic intervenes.
    TELEOP_DEADMAN_S: float = 0.5

    def capabilities(self) -> CapabilitySet:
        teleop = TeleopCapability(
            axes=("vx", "vy", "vyaw"),
            max_rates=(MAX_LIN_VEL_X_MPS, MAX_LIN_VEL_Y_MPS, MAX_ANG_VEL_Z_RPS),
            deadman_s=self.TELEOP_DEADMAN_S,
            cmd_topic="/cmd_vel",
            cmd_msg_type="geometry_msgs/msg/Twist",
            cmd_qos="reliable",
        )
        estop = EstopCapability(
            topic="/cmd_vel",
            msg_type="geometry_msgs/msg/Twist",
            qos="reliable",
        )
        return CapabilitySet(teleop=teleop, estop=estop)

    # ── Brand-specific safety envelope ────────────────────────────────────

    def _safety_clamp(self, twist: Dict[str, Any]) -> Dict[str, Any]:
        return clamp_quadruped_twist(twist)

    # ── Activation hooks ──────────────────────────────────────────────────

    def activate(self) -> None:
        """Mini Pupper in unitport mode has CHAMP already listening.

        The brownfield path (user's pupper) starts CHAMP via systemd at boot;
        no host-side activation message is needed. The Flask-mode "start
        controller" call (/pupper/status/start) is gated to Phase 4 because
        it requires the robot to NOT be in unitport mode (Conflicts= directive
        between unitport-minipupper.service and mini_pupper_web.service).
        """
        log_info(
            "[mangdang] activate: CHAMP already running via "
            "unitport-minipupper.service (brownfield path)"
        )

    def deactivate(self) -> None:
        """Stop teleop pump but leave CHAMP / unitport stack running.

        Real shutdown (systemctl stop unitport-minipupper) is robot-side and
        only needed when transitioning back to Flask mode — Phase 4 work.
        """
        self.disable_teleop()
        log_info("[mangdang] deactivate: teleop disabled, CHAMP stays alive")

    # ── Diagnostics (P9 brand probes) ─────────────────────────────────────

    def diagnostic_probes(self) -> List[DiagnosticProbe]:
        return mangdang_probes()


__all__ = ["MangdangROS2Adapter"]
