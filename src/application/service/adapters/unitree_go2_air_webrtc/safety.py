# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Velocity envelope clamp for Go2 Air sport API.

Go2 Air datasheet absolute limits (factory firmware): vx <= 1.0 m/s,
vy <= 0.6 m/s, vyaw <= 2.0 rad/s. We apply a conservative ~60% derate
so the first teleop session cannot accidentally drive the robot into a
fall — operators can lift the cap once the loop has been tuned.

The clamp operates on the dict-Twist shape produced by
:class:`TeleopPump`:

    {"linear":  {"x": vx, "y": vy, "z": 0.0},
     "angular": {"x": 0.0, "y": 0.0, "z": vyaw}}

and returns the same shape. Non-numeric inputs are coerced to 0.0 so a
malformed CommandBus payload cannot crash the pump.
"""

from __future__ import annotations

from typing import Any, Dict


MAX_VX_MPS: float = 0.6
MAX_VY_MPS: float = 0.4
MAX_VYAW_RPS: float = 1.2


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, hi: float) -> float:
    if value > hi:
        return hi
    if value < -hi:
        return -hi
    return value


def clamp_go2_air_twist(twist: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the Go2 Air datasheet envelope to a dict-Twist."""
    zero = {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    if not isinstance(twist, dict):
        return zero
    linear_raw = twist.get("linear")
    angular_raw = twist.get("angular")
    linear = linear_raw if isinstance(linear_raw, dict) else {}
    angular = angular_raw if isinstance(angular_raw, dict) else {}
    vx = _clamp(_coerce_float(linear.get("x", 0.0)), MAX_VX_MPS)
    vy = _clamp(_coerce_float(linear.get("y", 0.0)), MAX_VY_MPS)
    vyaw = _clamp(_coerce_float(angular.get("z", 0.0)), MAX_VYAW_RPS)
    return {
        "linear": {"x": vx, "y": vy, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": vyaw},
    }


__all__ = [
    "MAX_VX_MPS",
    "MAX_VY_MPS",
    "MAX_VYAW_RPS",
    "clamp_go2_air_twist",
]
