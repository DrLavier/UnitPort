#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HeartBeat display-state helpers — Circle 1 Step 1.5.

Pure Python; no Qt, no SDK calls, no file I/O.

Public surface
--------------
HBEventStatus        — canonical event state string constants
HBIOStatus           — canonical IO signal status string constants
HBEventBadge         — (label, color_key) badge descriptor for an event state
HBIOBadge            — (label, color_key) badge descriptor for an IO status
HBConflictHint       — (signal, reason, severity) conflict annotation
HBEventSummary       — aggregate counts for an event state list
HBIOSummary          — aggregate counts for an IO mapping list

badge_for_event_state(state)   — map event state string → HBEventBadge
badge_for_io_status(status)    — map IO status string → HBIOBadge
detect_io_conflicts(mappings)  — detect duplicate-signal and bad-status conflicts
compute_event_summary(entries) — count totals/active/errors/warnings
compute_io_summary(mappings)   — count totals/inbound/outbound/conflicts

Design constraints
------------------
- color_key is a semantic token ("success" | "error" | "warning" | "info" | "neutral"),
  NOT a Qt QColor.  The Qt layer maps tokens to colours; this layer stays framework-free.
- All public functions are deterministic and never raise on arbitrary string inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type hints in this file.
    from bin.pages.layout.behavior_panel import HBEventStateEntry, HBIOMapping


# ===========================================================================
# Status Constants
# ===========================================================================

class HBEventStatus:
    """Canonical event state string constants.

    ACTIVE    — event is currently active / firing.
    INACTIVE  — event exists but is not active.
    PENDING   — event is scheduled / queued.
    ERROR     — event is in an error condition.
    WARNING   — event fired but with a degraded condition.
    UNKNOWN   — state cannot be determined.
    """

    ACTIVE   = "active"
    INACTIVE = "inactive"
    PENDING  = "pending"
    ERROR    = "error"
    WARNING  = "warning"
    UNKNOWN  = "unknown"

    ALL: Tuple[str, ...] = (ACTIVE, INACTIVE, PENDING, ERROR, WARNING, UNKNOWN)


class HBIOStatus:
    """Canonical IO signal status string constants.

    ACTIVE      — signal is actively bound and flowing.
    UNMAPPED    — signal has no target parameter binding.
    UNSUPPORTED — signal is not supported by the current robot capability.
    CONFLICT    — signal conflicts with another binding (e.g. dual-direction).
    UNKNOWN     — status cannot be determined.
    """

    ACTIVE      = "active"
    UNMAPPED    = "unmapped"
    UNSUPPORTED = "unsupported"
    CONFLICT    = "conflict"
    UNKNOWN     = "unknown"

    ALL: Tuple[str, ...] = (ACTIVE, UNMAPPED, UNSUPPORTED, CONFLICT, UNKNOWN)


# ===========================================================================
# Badge Descriptors
# ===========================================================================

@dataclass
class HBEventBadge:
    """Visual badge descriptor for an event state value.

    Attributes
    ----------
    label     : Human-readable display label (e.g. "Active", "Error").
    color_key : Semantic color token — one of
                "success" | "error" | "warning" | "info" | "neutral".
                The Qt layer maps this to a QColor; this type stays Qt-free.
    """

    label: str
    color_key: str


@dataclass
class HBIOBadge:
    """Visual badge descriptor for an IO signal status value.

    Attributes
    ----------
    label     : Human-readable display label (e.g. "OK", "Unmapped").
    color_key : Semantic color token — same palette as HBEventBadge.
    """

    label: str
    color_key: str


# ===========================================================================
# Conflict and Summary Descriptors
# ===========================================================================

@dataclass
class HBConflictHint:
    """A single conflict or warning annotation for an IO signal.

    Attributes
    ----------
    signal   : The signal name that caused the conflict.
    reason   : Human-readable explanation.
    severity : "error" (hard conflict) or "warning" (soft/degraded).
    """

    signal: str
    reason: str
    severity: str  # "error" | "warning"


@dataclass
class HBEventSummary:
    """Aggregate counts over an event state list.

    Attributes
    ----------
    total    : Total number of event entries.
    active   : Entries with state == ACTIVE.
    errors   : Entries with state == ERROR.
    warnings : Entries with state == WARNING.
    """

    total: int
    active: int
    errors: int
    warnings: int


@dataclass
class HBIOSummary:
    """Aggregate counts over an IO mapping list.

    Attributes
    ----------
    total     : Total number of IO mapping entries.
    inbound   : Entries with direction == "inbound".
    outbound  : Entries with direction == "outbound".
    conflicts : Number of conflict hints detected (from detect_io_conflicts).
    """

    total: int
    inbound: int
    outbound: int
    conflicts: int


# ===========================================================================
# Internal badge lookup tables
# ===========================================================================

_EVENT_BADGE_MAP: Dict[str, HBEventBadge] = {
    HBEventStatus.ACTIVE:   HBEventBadge(label="Active",   color_key="success"),
    HBEventStatus.INACTIVE: HBEventBadge(label="Inactive", color_key="neutral"),
    HBEventStatus.PENDING:  HBEventBadge(label="Pending",  color_key="info"),
    HBEventStatus.ERROR:    HBEventBadge(label="Error",    color_key="error"),
    HBEventStatus.WARNING:  HBEventBadge(label="Warning",  color_key="warning"),
    HBEventStatus.UNKNOWN:  HBEventBadge(label="Unknown",  color_key="neutral"),
}

_IO_BADGE_MAP: Dict[str, HBIOBadge] = {
    HBIOStatus.ACTIVE:      HBIOBadge(label="OK",          color_key="success"),
    HBIOStatus.UNMAPPED:    HBIOBadge(label="Unmapped",    color_key="warning"),
    HBIOStatus.UNSUPPORTED: HBIOBadge(label="Unsupported", color_key="error"),
    HBIOStatus.CONFLICT:    HBIOBadge(label="Conflict",    color_key="error"),
    HBIOStatus.UNKNOWN:     HBIOBadge(label="Unknown",     color_key="neutral"),
}


# ===========================================================================
# Badge Lookup Functions
# ===========================================================================

def badge_for_event_state(state: str) -> HBEventBadge:
    """Return the badge descriptor for an event state string.

    Falls back to a neutral badge for unknown states — never raises.

    Args
    ----
    state : Event state string (e.g. ``"active"``, ``"error"``).
    """
    if state in _EVENT_BADGE_MAP:
        return _EVENT_BADGE_MAP[state]
    return HBEventBadge(label=state, color_key="neutral")


def badge_for_io_status(status: str) -> HBIOBadge:
    """Return the badge descriptor for an IO signal status string.

    Falls back to a neutral badge for unknown statuses — never raises.

    Args
    ----
    status : IO status string (e.g. ``"active"``, ``"unmapped"``).
    """
    if status in _IO_BADGE_MAP:
        return _IO_BADGE_MAP[status]
    return HBIOBadge(label=status, color_key="neutral")


# ===========================================================================
# Conflict Detection
# ===========================================================================

def detect_io_conflicts(mappings: "List[HBIOMapping]") -> List[HBConflictHint]:
    """Detect conflicts and warnings in an IO mapping list.

    Three detection rules (evaluated independently; one signal may produce
    multiple hints):

    Rule 1 — Dual-direction conflict:
        If the same signal name appears as both ``"inbound"`` and ``"outbound"``,
        a conflict hint with ``severity="error"`` is emitted.

    Rule 2 — Unsupported signal:
        If any mapping has ``status == "unsupported"``, a conflict hint with
        ``severity="error"`` is emitted for that signal.

    Rule 3 — Unmapped signal:
        If any mapping has ``status == "unmapped"``, a conflict hint with
        ``severity="warning"`` is emitted for that signal.

    Args
    ----
    mappings : List of HBIOMapping objects.  Safe with empty list.

    Returns
    -------
    List of HBConflictHint objects.  Empty list when no conflicts found.
    Never raises on any input.
    """
    hints: List[HBConflictHint] = []
    if not mappings:
        return hints

    # Rule 1: dual-direction detection
    inbound_signals: set = set()
    outbound_signals: set = set()
    for m in mappings:
        if m.direction == "inbound":
            inbound_signals.add(m.signal)
        elif m.direction == "outbound":
            outbound_signals.add(m.signal)
    for sig in inbound_signals & outbound_signals:
        hints.append(HBConflictHint(
            signal=sig,
            reason="signal used as both inbound and outbound",
            severity="error",
        ))

    # Rules 2 & 3: status-based conflicts
    for m in mappings:
        if m.status == HBIOStatus.UNSUPPORTED:
            hints.append(HBConflictHint(
                signal=m.signal,
                reason="signal not supported by current robot capability",
                severity="error",
            ))
        elif m.status == HBIOStatus.UNMAPPED:
            hints.append(HBConflictHint(
                signal=m.signal,
                reason="signal has no target parameter binding",
                severity="warning",
            ))

    return hints


# ===========================================================================
# Summary Computation
# ===========================================================================

def compute_event_summary(entries: "List[HBEventStateEntry]") -> HBEventSummary:
    """Compute aggregate counts for a list of event state entries.

    Args
    ----
    entries : List of HBEventStateEntry objects.  Safe with empty list.
    """
    total = len(entries)
    active = sum(1 for e in entries if e.state == HBEventStatus.ACTIVE)
    errors = sum(1 for e in entries if e.state == HBEventStatus.ERROR)
    warnings = sum(1 for e in entries if e.state == HBEventStatus.WARNING)
    return HBEventSummary(total=total, active=active, errors=errors, warnings=warnings)


def compute_io_summary(
    mappings: "List[HBIOMapping]",
    conflicts: "List[HBConflictHint]",
) -> HBIOSummary:
    """Compute aggregate counts for a list of IO mappings.

    Args
    ----
    mappings  : List of HBIOMapping objects.  Safe with empty list.
    conflicts : Pre-computed conflict hint list (from detect_io_conflicts).
    """
    total = len(mappings)
    inbound = sum(1 for m in mappings if m.direction == "inbound")
    outbound = sum(1 for m in mappings if m.direction == "outbound")
    return HBIOSummary(
        total=total,
        inbound=inbound,
        outbound=outbound,
        conflicts=len(conflicts),
    )
