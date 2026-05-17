"""Topic surface declared by the Unitree Go2 Air WebRTC adapter.

Topic names are the literal strings the firmware accepts over the
WebRTC data channel (no DDS namespace munging). Roles drive the
:mod:`topic_registry` queries used by UI sparkline pickers and the
Phase 4 inspector.

Only the topics the adapter actively uses are registered (sport command
publish, low state subscription). Adding more is one-line below; the
registry de-duplicates same-(brand, topic, role) tuples on import.
"""

from __future__ import annotations

from typing import Tuple

from application.service.adapters.topic_registry import TopicSpec


ALL_SPECS: Tuple[TopicSpec, ...] = (
    # Outbound sport API command channel (StandUp, Move, Damp, ...).
    TopicSpec(
        topic="rt/api/sport/request",
        msg_type="unitree_sport/Request",
        qos_profile="reliable",
        role="cmd",
    ),
    # Low-frequency robot state stream (joint angles, IMU, battery).
    # Used by :meth:`Go2AirWebRTCAdapter.confirm_steady_state` and future
    # telemetry sparkline consumers.
    TopicSpec(
        topic="rt/lf/lowstate",
        msg_type="unitree_sport/LowState",
        qos_profile="best_effort",
        role="obs",
    ),
)


__all__ = ["ALL_SPECS"]
