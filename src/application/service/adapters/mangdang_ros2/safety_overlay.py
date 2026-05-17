"""Mangdang Mini Pupper v2 — teleop safety envelope.

Hard-clamps the user-supplied Twist before it hits the bridge so a jittery
controller cannot command past the servo-safe envelope. Values mirror DEMO's
``runtime/ros2/brands/mangdang_mini_pupper_v2/adapter_overlay.controller_to_action``
(MAX_LIN_VEL 0.35 m/s, MAX_ANG_VEL 2.0 rad/s, MAX_LAT_VEL 0.20 m/s),
documented as the v2 datasheet servo envelope.

Pure functions — no Qt, no DDS, no adapter state. Easy to unit-test in
isolation.
"""

from __future__ import annotations

from typing import Any, Dict


# Mini Pupper v2 datasheet envelope. Larger values risk servo stall + thermal
# shutdown under sustained input. DEMO has been operating these values since
# 2026-04 with no reported regressions.
MAX_LIN_VEL_X_MPS: float = 0.35    # forward / back
MAX_LIN_VEL_Y_MPS: float = 0.20    # side strafe (lateral gait is less stable)
MAX_LIN_VEL_Z_MPS: float = 0.0     # quadruped has no z translation
MAX_ANG_VEL_Z_RPS: float = 2.0     # yaw rate
MAX_ANG_VEL_XY_RPS: float = 0.0    # roll/pitch are not user-commandable


def _clamp(value: float, hi: float) -> float:
    """Symmetric clamp: ``value`` to ``[-hi, hi]``. ``hi`` non-negative."""
    if hi <= 0.0:
        return 0.0
    if value > hi:
        return hi
    if value < -hi:
        return -hi
    return value


def _f(value: Any, default: float = 0.0) -> float:
    """Defensive coerce-to-float; ``None`` / non-numeric maps to ``default``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_quadruped_twist(twist: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the Mini Pupper v2 envelope to a Twist dict.

    Input shape: ``{"linear": {"x", "y", "z"}, "angular": {"x", "y", "z"}}``
    matching the geometry_msgs/Twist IDL the bridge expects. Returns a fresh
    dict with the same shape; never mutates the input.
    """
    linear = twist.get("linear") or {}
    angular = twist.get("angular") or {}
    return {
        "linear": {
            "x": _clamp(_f(linear.get("x")), MAX_LIN_VEL_X_MPS),
            "y": _clamp(_f(linear.get("y")), MAX_LIN_VEL_Y_MPS),
            "z": _clamp(_f(linear.get("z")), MAX_LIN_VEL_Z_MPS),
        },
        "angular": {
            "x": _clamp(_f(angular.get("x")), MAX_ANG_VEL_XY_RPS),
            "y": _clamp(_f(angular.get("y")), MAX_ANG_VEL_XY_RPS),
            "z": _clamp(_f(angular.get("z")), MAX_ANG_VEL_Z_RPS),
        },
    }


__all__ = [
    "clamp_quadruped_twist",
    "MAX_LIN_VEL_X_MPS",
    "MAX_LIN_VEL_Y_MPS",
    "MAX_ANG_VEL_Z_RPS",
]
