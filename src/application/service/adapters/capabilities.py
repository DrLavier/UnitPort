"""Capability descriptors — adapter-declared schema for what a robot supports.

Capabilities drive two consumers:

1. **UI** — the Controller sub-card (``ConnectionControllerCard``) renders
   different widgets per capability kind (2D teleop for quadrupeds, 6DOF arm
   for manipulators, PID tuning for joint-level interfaces).

2. **Connection orchestrator** — ``Ros2BrownfieldConnectTask`` P7 reads the
   teleop+estop topic / msg type / QoS from the active adapter's capability
   set so the bridge wires up exactly what the robot uses.

Adapters return a :class:`CapabilitySet` from :meth:`BaseAdapter.capabilities`;
unsupported capabilities are ``None`` (so consumers branch on ``if caps.teleop
is not None``, not on stringly-typed flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class TeleopCapability:
    """Schema for a robot's teleop surface.

    ``axes`` names the user-facing axis labels (e.g. ``("vx", "vy", "vyaw")``
    for a quadruped; ``("vx", "vy", "vz", "wx", "wy", "wz")`` for a 6DOF base).
    ``max_rates`` is a per-axis hard envelope applied by the adapter's
    ``_safety_clamp`` before publish.
    """

    axes: Tuple[str, ...]
    max_rates: Tuple[float, ...]
    deadman_s: float                 # publish zero twist + warn after this idle window
    cmd_topic: str                    # e.g. "/cmd_vel"
    cmd_msg_type: str                 # e.g. "geometry_msgs/msg/Twist"
    cmd_qos: str = "reliable"         # qos profile name resolved via qos_profiles


@dataclass(frozen=True)
class EstopCapability:
    """Estop = publish a zero command on a specific topic.

    For most ROS2 quadrupeds this is just a zero Twist on /cmd_vel; richer
    brands may publish a dedicated EstopState message. The connector card's
    Estop button calls ``adapter.disable_teleop()`` which already publishes
    zero, then optionally fires this if it points at a distinct topic.
    """

    topic: str
    msg_type: str
    qos: str = "reliable"


@dataclass(frozen=True)
class ArmCapability:
    """6DOF arm placeholder — full schema lands in Phase 6+ controller work."""

    dof: int
    joint_names: Tuple[str, ...]
    cmd_topic: str
    cmd_msg_type: str


@dataclass(frozen=True)
class PidCapability:
    """Per-joint PID tuning surface — drives Phase 6 sec3 PID sub-card.

    ``joints`` lists the IR-role names the adapter can tune online (e.g.
    ``("hip_fl_haa", "hip_fl_hfe", ...)``). Empty tuple means PID tuning is
    declared but no joints are currently exposed (transient).
    """

    joints: Tuple[str, ...]
    set_gains_topic: str = ""    # adapter-specific publish target
    set_gains_msg_type: str = ""


@dataclass(frozen=True)
class CapabilitySet:
    """Aggregate of every capability a single adapter declares.

    Consumers branch on ``capability is not None`` — never on string flags
    or duck-typing — so adding a new capability kind is a single dataclass
    field addition with no broken consumer code.
    """

    teleop: Optional[TeleopCapability] = None
    estop: Optional[EstopCapability] = None
    arm: Optional[ArmCapability] = None
    pid_tuning: Optional[PidCapability] = None

    def has_any(self) -> bool:
        """True iff at least one capability is declared (not all None)."""
        return any(
            getattr(self, f) is not None
            for f in ("teleop", "estop", "arm", "pid_tuning")
        )


__all__ = [
    "TeleopCapability",
    "EstopCapability",
    "ArmCapability",
    "PidCapability",
    "CapabilitySet",
]
