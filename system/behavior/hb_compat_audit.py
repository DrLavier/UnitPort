#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HeartBeat model-switch compatibility audit — Circle 1 Step 1.7.

Pure Python; no Qt, no SDK calls, no file I/O.

Public surface
--------------
HBCompatIssue        — per-node incompatibility record
HBCompatReport       — aggregate audit result for a brand/robot_type switch
audit_catalog_compatibility(catalog, brand, robot_type)
    → HBCompatReport

Design
------
- Severity mapping (fixed, deterministic):
    UNSUPPORTED        → "error"   (node cannot function at all)
    LIMITED            → "warning" (node may have degraded behaviour)
    UNKNOWN_CAPABILITY → "warning" (capability data absent; assume degraded)
    AVAILABLE          → no issue
- All public functions are pure — no mutation of input objects, never raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from system.behavior.hb_node_catalog import HBNodeCatalog

from system.behavior.hb_node_catalog import HBNodeAvailability


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class HBCompatIssue:
    """A single node-level incompatibility record from a compatibility audit.

    Attributes
    ----------
    node_kind    : Canonical HBNodeKind string for the affected node.
    node_label   : Human-readable display name of the affected node.
    availability : The availability state that triggered this issue
                   (``"limited"`` | ``"unsupported"`` | ``"unknown_capability"``).
    reason       : Human-readable explanation of the incompatibility.
    severity     : ``"error"`` (node unusable) or ``"warning"`` (degraded).
    """

    node_kind: str
    node_label: str
    availability: str
    reason: str
    severity: str  # "error" | "warning"


@dataclass
class HBCompatReport:
    """Aggregate compatibility audit result for a brand/robot_type switch.

    Attributes
    ----------
    brand      : Robot brand string (e.g. ``"unitree"``).
    robot_type : Robot type string (e.g. ``"go2"``).
    issues     : List of :class:`HBCompatIssue` objects — empty when clean.
    """

    brand: str
    robot_type: str
    issues: List[HBCompatIssue] = field(default_factory=list)

    # ------------------------------------------------------------------ queries

    def is_clean(self) -> bool:
        """Return True when no issues were found."""
        return len(self.issues) == 0

    @property
    def has_errors(self) -> bool:
        """True when at least one error-severity issue is present."""
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True when at least one warning-severity issue is present."""
        return any(i.severity == "warning" for i in self.issues)

    @property
    def error_count(self) -> int:
        """Number of error-severity issues."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of warning-severity issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary_text(self) -> str:
        """Short human-readable summary suitable for a status bar or label.

        Examples::

            "No issues"
            "1 error"
            "2 errors"
            "1 warning"
            "3 warnings"
            "1 error, 2 warnings"
        """
        if self.is_clean():
            return "No issues"
        parts: List[str] = []
        e = self.error_count
        w = self.warning_count
        if e:
            parts.append(f"{e} error" if e == 1 else f"{e} errors")
        if w:
            parts.append(f"{w} warning" if w == 1 else f"{w} warnings")
        return ", ".join(parts)


# ===========================================================================
# Audit function
# ===========================================================================

_SEVERITY_MAP = {
    HBNodeAvailability.UNSUPPORTED:        "error",
    HBNodeAvailability.LIMITED:            "warning",
    HBNodeAvailability.UNKNOWN_CAPABILITY: "warning",
}

_DEFAULT_REASON = {
    HBNodeAvailability.UNSUPPORTED:
        "node is not supported by the current robot capability profile",
    HBNodeAvailability.LIMITED:
        "node has limited support on this robot; behaviour may be degraded",
    HBNodeAvailability.UNKNOWN_CAPABILITY:
        "capability data is unavailable; support state cannot be determined",
}


def audit_catalog_compatibility(
    catalog: "HBNodeCatalog",
    brand: str = "",
    robot_type: str = "",
) -> HBCompatReport:
    """Audit a HeartBeat node catalog for compatibility issues.

    Iterates all catalog entries and maps each non-AVAILABLE availability
    state to a :class:`HBCompatIssue` using the fixed severity table:

    ========================== ========
    Availability               Severity
    ========================== ========
    ``UNSUPPORTED``            error
    ``LIMITED``                warning
    ``UNKNOWN_CAPABILITY``     warning
    ``AVAILABLE``              (no issue)
    ========================== ========

    Parameters
    ----------
    catalog    : Populated :class:`HBNodeCatalog`.  Empty catalog is safe.
    brand      : Robot brand used for default reason strings (optional).
    robot_type : Robot type used for default reason strings (optional).

    Returns
    -------
    :class:`HBCompatReport` — always returned; never raises.
    """
    issues: List[HBCompatIssue] = []

    context = f"{brand}/{robot_type}".strip("/") or "current robot"

    for entry in catalog.all_entries():
        severity = _SEVERITY_MAP.get(entry.availability)
        if severity is None:
            continue  # AVAILABLE → no issue

        reason = entry.reason or _DEFAULT_REASON.get(
            entry.availability,
            f"incompatible with {context}",
        )

        issues.append(HBCompatIssue(
            node_kind=entry.kind,
            node_label=entry.label,
            availability=entry.availability,
            reason=reason,
            severity=severity,
        ))

    return HBCompatReport(brand=brand, robot_type=robot_type, issues=issues)


def report_to_behavior_diagnostics(
    report: HBCompatReport,
) -> "List":
    """Convert a :class:`HBCompatReport` to :class:`BehaviorDiagnostic` records.

    Maps each :class:`HBCompatIssue` to a ``BehaviorDiagnostic`` so that
    compatibility audit results can flow through the IR semantic diagnostics
    pipeline (``DiagnosticsKey.COMPAT_DIAGNOSTICS``).

    Parameters
    ----------
    report : :class:`HBCompatReport` from :func:`audit_catalog_compatibility`.

    Returns
    -------
    ``List[BehaviorDiagnostic]`` — empty list when ``report.is_clean()``.
    """
    from system.behavior.behavior_artifact import BehaviorDiagnostic  # local — avoids circular

    result: List = []
    for issue in report.issues:
        code    = f"compat.{issue.availability}.{issue.node_kind}"
        message = f"[{issue.node_label}] {issue.reason}"
        diag = (
            BehaviorDiagnostic.error(code, message, location=issue.node_kind)
            if issue.severity == "error"
            else BehaviorDiagnostic.warning(code, message, location=issue.node_kind)
        )
        result.append(diag)
    return result
