#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HeartBeat node catalog contract — Circle 1 Step 1.4.

Defines a frontend-consumable catalog of HeartBeat node availability states,
keyed by brand / robot_type / capability profile.

Public surface
--------------
HBNodeAvailability      — availability state string constants
HBNodeCatalogEntry      — per-node catalog record (kind, label, availability, reason)
HBNodeCatalog           — catalog model; from_capability_dict() builds from adapter data

Design constraints
------------------
- Pure Python; no Qt, no SDK calls, no file I/O.
- Deterministic fallback: absent capability data → UNKNOWN_CAPABILITY (not silent hide).
- All HBNodeKind entries are always present in a catalog — nodes are never omitted.
- Extra capability keys beyond the known schema are silently ignored (forward compat).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Tuple


# ===========================================================================
# Availability Constants
# ===========================================================================

class HBNodeAvailability:
    """Canonical availability state strings for a heartbeat node catalog entry.

    States are mutually exclusive and always deterministic:

    AVAILABLE          — node is fully supported by the current capability profile.
    LIMITED            — node is partially supported; functionality may be degraded.
    UNSUPPORTED        — node is not supported by the current robot/capability profile.
    UNKNOWN_CAPABILITY — capability data is absent; support state cannot be determined.
    """

    AVAILABLE          = "available"
    LIMITED            = "limited"
    UNSUPPORTED        = "unsupported"
    UNKNOWN_CAPABILITY = "unknown_capability"

    ALL: Tuple[str, ...] = (AVAILABLE, LIMITED, UNSUPPORTED, UNKNOWN_CAPABILITY)


# ===========================================================================
# Catalog Entry
# ===========================================================================

@dataclass
class HBNodeCatalogEntry:
    """Per-node availability record in the HeartBeat node catalog.

    Attributes
    ----------
    kind         : Canonical node kind string (matches HBNodeKind constants).
    label        : Human-readable display name.
    availability : One of HBNodeAvailability.* constants.
    reason       : Human-readable explanation for non-AVAILABLE states;
                   empty string when availability is AVAILABLE.
    """

    kind: str
    label: str
    availability: str
    reason: str = ""

    def is_available(self) -> bool:
        """Return True if this node is fully available."""
        return self.availability == HBNodeAvailability.AVAILABLE

    def is_limited(self) -> bool:
        """Return True if this node has limited (partial) availability."""
        return self.availability == HBNodeAvailability.LIMITED

    def is_unsupported(self) -> bool:
        """Return True if this node is explicitly unsupported."""
        return self.availability == HBNodeAvailability.UNSUPPORTED

    def is_unknown(self) -> bool:
        """Return True if capability data is absent for this node."""
        return self.availability == HBNodeAvailability.UNKNOWN_CAPABILITY


# ===========================================================================
# Internal sensor / flag constants for capability evaluation
# ===========================================================================

# Primary sensor that makes SensorRead fully available (IMU-first design)
_SENSOR_READ_PRIMARY: str = "imu"

# Sensors that support SensorTransform computations
_SENSOR_TRANSFORM_COMPATIBLE: FrozenSet[str] = frozenset({
    "imu", "qpos", "qvel", "foot_contact", "odometry", "joint_state",
})

# Default human-readable labels keyed by HBNodeKind constant strings
_DEFAULT_LABELS: Dict[str, str] = {
    "heartbeat_source":   "Heartbeat Source",
    "sensor_read":        "Sensor Read",
    "sensor_transform":   "Transform",
    "movement_blend":     "Movement Blend",
    "safety_gate":        "Safety Gate",
}


# ===========================================================================
# Capability evaluation functions
# ===========================================================================

def _eval_heartbeat_source(
    sensors: List[str], flags: Dict[str, Any]
) -> Tuple[str, str]:
    """HeartbeatSource is the tick initiator — no sensor dependency."""
    return HBNodeAvailability.AVAILABLE, ""


def _eval_sensor_read(
    sensors: List[str], flags: Dict[str, Any]
) -> Tuple[str, str]:
    """SensorRead is fully available when the primary sensor (IMU) is present."""
    if _SENSOR_READ_PRIMARY in sensors:
        return HBNodeAvailability.AVAILABLE, ""
    if sensors:
        sensor_list = ", ".join(sorted(sensors))
        return (
            HBNodeAvailability.LIMITED,
            f"IMU sensor unavailable; sensor read is limited to: {sensor_list}",
        )
    return HBNodeAvailability.UNSUPPORTED, "No sensors available for this robot profile"


def _eval_sensor_transform(
    sensors: List[str], flags: Dict[str, Any]
) -> Tuple[str, str]:
    """SensorTransform needs at least one transform-compatible sensor (imu, joint, etc.)."""
    compatible = _SENSOR_TRANSFORM_COMPATIBLE & set(sensors)
    if compatible:
        return HBNodeAvailability.AVAILABLE, ""
    if sensors:
        sensor_list = ", ".join(sorted(sensors))
        return (
            HBNodeAvailability.LIMITED,
            f"No transform-compatible sensors in profile; available: {sensor_list}",
        )
    return HBNodeAvailability.UNSUPPORTED, "No sensors available for sensor transform"


def _eval_movement_blend(
    sensors: List[str], flags: Dict[str, Any]
) -> Tuple[str, str]:
    """MovementBlend requires the adapter to declare high-level movement control."""
    if "high_level_control" not in flags:
        return (
            HBNodeAvailability.UNKNOWN_CAPABILITY,
            "high_level_control flag not declared by adapter",
        )
    if flags["high_level_control"]:
        return HBNodeAvailability.AVAILABLE, ""
    return (
        HBNodeAvailability.LIMITED,
        "High-level movement control not available; movement blend has limited effect",
    )


def _eval_safety_gate(
    sensors: List[str], flags: Dict[str, Any]
) -> Tuple[str, str]:
    """SafetyGate applies a software-level constraint — always available."""
    return HBNodeAvailability.AVAILABLE, ""


# Map from node kind to its evaluation function
_EVAL_FNS: Dict[str, Any] = {
    "heartbeat_source":  _eval_heartbeat_source,
    "sensor_read":       _eval_sensor_read,
    "sensor_transform":  _eval_sensor_transform,
    "movement_blend":    _eval_movement_blend,
    "safety_gate":       _eval_safety_gate,
}


# ===========================================================================
# HBNodeCatalog
# ===========================================================================

class HBNodeCatalog:
    """Frontend-consumable HeartBeat node catalog keyed by capability profile.

    Built from a capability dict returned by ``adapter.capabilities()``.  All
    HBNodeKind entries are always present — nodes are never omitted.  When
    capability data is absent or incomplete, entries fall back to
    UNKNOWN_CAPABILITY rather than silently hiding the node.

    Usage
    -----
    ::

        cap = adapter.capabilities()
        catalog = HBNodeCatalog.from_capability_dict(cap)
        for entry in catalog.all_entries():
            render_node_item(entry)

    Serialization
    -------------
    The catalog is a transient view model; it is not persisted.  Rebuild it
    whenever the robot brand / settings change by calling
    ``from_capability_dict()`` again.
    """

    # Canonical display order for all_entries()
    _NODE_ORDER: List[str] = [
        "heartbeat_source",
        "sensor_read",
        "sensor_transform",
        "movement_blend",
        "safety_gate",
    ]

    def __init__(self, entries: List[HBNodeCatalogEntry]) -> None:
        # Index by kind for O(1) lookup; insertion order preserved (Python 3.7+)
        self._entries: Dict[str, HBNodeCatalogEntry] = {e.kind: e for e in entries}
        self._brand: str = ""
        self._robot_type: str = ""

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def for_kind(self, kind: str) -> HBNodeCatalogEntry:
        """Return the catalog entry for a node kind.

        Falls back to UNKNOWN_CAPABILITY if the kind is not registered.
        Never raises — deterministic for any input.

        Args
        ----
        kind : Node kind string (e.g. ``"sensor_read"``).
        """
        if kind in self._entries:
            return self._entries[kind]
        label = _DEFAULT_LABELS.get(kind, kind)
        return HBNodeCatalogEntry(
            kind=kind,
            label=label,
            availability=HBNodeAvailability.UNKNOWN_CAPABILITY,
            reason="Node kind not registered in catalog",
        )

    def all_entries(self) -> List[HBNodeCatalogEntry]:
        """Return all catalog entries in canonical display order.

        Entries for kinds in ``_NODE_ORDER`` come first, in that order.
        Any additional entries (forward-compat extras) are appended at the end.
        """
        ordered = [
            self._entries[k]
            for k in self._NODE_ORDER
            if k in self._entries
        ]
        canonical_set = set(self._NODE_ORDER)
        extras = [e for k, e in self._entries.items() if k not in canonical_set]
        return ordered + extras

    @property
    def brand(self) -> str:
        """Brand string from the originating capability dict (informational)."""
        return self._brand

    @property
    def robot_type(self) -> str:
        """Robot type string extracted from the capability dict.

        Extraction priority (first non-empty wins):
        1. Top-level ``cap["robot_type"]`` key.
        2. ``cap["flags"]["robot_type"]`` key.
        3. Empty string (deterministic fallback — never raises).
        """
        return self._robot_type

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_capability_dict(cls, cap: Any) -> "HBNodeCatalog":
        """Build a catalog from an adapter capability dict.

        Robustly handles malformed or incomplete capability dicts:

        - Non-dict input  → returns ``unknown()``
        - Missing ``sensors`` / ``flags`` keys → treated as empty
        - Non-list ``sensors`` value → treated as empty list
        - Non-dict ``flags`` value → treated as empty dict
        - Extra keys → silently ignored (forward compatibility)

        Args
        ----
        cap : The dict returned by ``adapter.capabilities()``, or any value.
        """
        if not isinstance(cap, dict):
            return cls.unknown()

        # Extract and normalise sensors list
        raw_sensors = cap.get("sensors", [])
        if not isinstance(raw_sensors, list):
            raw_sensors = []
        sensors: List[str] = [str(s) for s in raw_sensors if isinstance(s, str)]

        # Extract and normalise flags dict
        raw_flags = cap.get("flags", {})
        flags: Dict[str, Any] = raw_flags if isinstance(raw_flags, dict) else {}

        catalog = cls._build_entries(sensors, flags)
        catalog._brand = str(cap.get("brand", ""))
        # robot_type: top-level key takes priority; fall back to flags["robot_type"]
        rt_top   = str(cap.get("robot_type", "") or "")
        rt_flags = str((flags.get("robot_type", "") if isinstance(flags, dict) else "") or "")
        catalog._robot_type = rt_top or rt_flags
        return catalog

    @classmethod
    def unknown(cls) -> "HBNodeCatalog":
        """Return a catalog where every node has UNKNOWN_CAPABILITY state.

        Used as the default before any capability data is loaded.  Ensures
        deterministic rendering with no silent-hide behaviour.
        """
        entries = [
            HBNodeCatalogEntry(
                kind=kind,
                label=_DEFAULT_LABELS.get(kind, kind),
                availability=HBNodeAvailability.UNKNOWN_CAPABILITY,
                reason="No capability data loaded",
            )
            for kind in cls._NODE_ORDER
        ]
        catalog = cls(entries)
        catalog._brand = ""
        catalog._robot_type = ""
        return catalog

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_entries(
        cls,
        sensors: List[str],
        flags: Dict[str, Any],
    ) -> "HBNodeCatalog":
        """Build catalog entries from normalised sensor list and flags dict."""
        entries: List[HBNodeCatalogEntry] = []
        for kind in cls._NODE_ORDER:
            label = _DEFAULT_LABELS.get(kind, kind)
            eval_fn = _EVAL_FNS.get(kind)
            if eval_fn is None:
                availability = HBNodeAvailability.UNKNOWN_CAPABILITY
                reason = "No evaluation function registered for this node kind"
            else:
                availability, reason = eval_fn(sensors, flags)
            entries.append(HBNodeCatalogEntry(
                kind=kind,
                label=label,
                availability=availability,
                reason=reason,
            ))
        catalog = cls(entries)
        catalog._brand = ""
        return catalog
