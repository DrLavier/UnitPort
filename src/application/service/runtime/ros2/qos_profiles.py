"""Canonical QoS profile presets for ROS2 topic subscriptions/publications.

These are **descriptor dicts** — not rclpy or cyclonedds QoS objects. The
NativeDDSBridge translates them to ``cyclonedds.core.Qos`` at runtime via
``_build_qos`` in ``native_dds_bridge``.

Keys (all 6 recognised by NativeDDSBridge):
    reliability: "best_effort" | "reliable"
    history:     "keep_last"   | "keep_all"
    depth:       int  (history queue depth)
    durability:  "volatile"    | "transient_local"
    deadline:    float seconds | None
    liveliness:  "automatic"   | "manual_by_topic"
"""

from __future__ import annotations

from typing import Any, Dict, Optional


_DEFAULTS: Dict[str, Any] = {
    "reliability": "best_effort",
    "history": "keep_last",
    "depth": 5,
    "durability": "volatile",
    "deadline": None,
    "liveliness": "automatic",
}


SENSOR_DATA: Dict[str, Any] = {
    **_DEFAULTS,
    "reliability": "best_effort",
    "durability": "volatile",
    "history": "keep_last",
    "depth": 5,
}

RELIABLE_COMMAND: Dict[str, Any] = {
    **_DEFAULTS,
    "reliability": "reliable",
    "durability": "volatile",
    "history": "keep_last",
    "depth": 10,
}

BEST_EFFORT_STATE: Dict[str, Any] = {
    **_DEFAULTS,
    "reliability": "best_effort",
    "durability": "volatile",
    "history": "keep_last",
    "depth": 1,
}

PROFILES: Dict[str, Dict[str, Any]] = {
    "sensor_data": SENSOR_DATA,
    "reliable_command": RELIABLE_COMMAND,
    "best_effort_state": BEST_EFFORT_STATE,
    # Aliases — see DEMO note: short names used by adapters and per-topic
    # YAMLs ("reliable", "best_effort") must resolve to the right preset
    # instead of silently falling through to SENSOR_DATA. Without these,
    # CHAMP's reliable /cmd_vel subscriber rejected best-effort writes as
    # RELIABILITY_QOS_POLICY incompatible and the robot couldn't move.
    "reliable": RELIABLE_COMMAND,
    "best_effort": SENSOR_DATA,
}


_VALID_RELIABILITY = frozenset({"best_effort", "reliable"})
_VALID_HISTORY = frozenset({"keep_last", "keep_all"})
_VALID_DURABILITY = frozenset({"volatile", "transient_local"})
_VALID_LIVELINESS = frozenset({"automatic", "manual_by_topic"})


def resolve(name: str) -> Dict[str, Any]:
    """Return a QoS descriptor by name. Falls back to sensor_data."""
    return dict(PROFILES.get(name, SENSOR_DATA))


def merge(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow-merge ``overrides`` onto ``base`` and validate the result."""
    merged = {**_DEFAULTS, **base}
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    _validate(merged)
    return merged


def _validate(qos: Dict[str, Any]) -> None:
    reliability = qos.get("reliability")
    if reliability not in _VALID_RELIABILITY:
        raise ValueError(f"qos.reliability: unknown value {reliability!r}")
    history = qos.get("history")
    if history not in _VALID_HISTORY:
        raise ValueError(f"qos.history: unknown value {history!r}")
    durability = qos.get("durability")
    if durability not in _VALID_DURABILITY:
        raise ValueError(f"qos.durability: unknown value {durability!r}")
    liveliness = qos.get("liveliness")
    if liveliness not in _VALID_LIVELINESS:
        raise ValueError(f"qos.liveliness: unknown value {liveliness!r}")
    depth = qos.get("depth")
    if not isinstance(depth, int) or depth < 1:
        raise ValueError(f"qos.depth: must be int >= 1, got {depth!r}")
    deadline = qos.get("deadline")
    if deadline is not None and (not isinstance(deadline, (int, float)) or deadline <= 0):
        raise ValueError(f"qos.deadline: must be None or positive float, got {deadline!r}")
