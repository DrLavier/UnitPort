#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HeartBeat model-switch compatibility audit — Circle 1 Step 1.7.
Extended with semantic action-level audit — Circle 6 Steps 6.1 & 6.2.

Pure Python; no Qt, no SDK calls, no file I/O.

Public surface
--------------
HBCompatIssue        — per-issue incompatibility record (HB node or action)
HBCompatReport       — aggregate audit result for a brand/robot_type switch
audit_catalog_compatibility(catalog, brand, robot_type) → HBCompatReport
audit_action_compatibility(timeline, brand, robot_type) → List[HBCompatIssue]
report_to_behavior_diagnostics(report)                  → List[BehaviorDiagnostic]

Design
------
- Severity mapping (fixed, deterministic):
    UNSUPPORTED        → "error"   (node/action cannot function at all)
    LIMITED/DEGRADED   → "warning" (may have degraded behaviour)
    UNKNOWN_CAPABILITY → "warning" (capability data absent; assume degraded)
    AVAILABLE          → no issue
- All public functions are pure — no mutation of input objects, never raise.
- Circle 6: HBCompatIssue carries optional action-level context (intent_id,
  source_raw_action, target_raw_action) for richer diagnostics display.
- Circle 6: HBCompatReport carries an optional ActionProfileReport for
  machine-readable programmatic access to the action comparison result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from src.system.behavior.hb_node_catalog import HBNodeCatalog
    from src.system.behavior.action_profile import BehaviorTimeline
    from src.system.behavior.action_profile_report import ActionProfileReport

from src.system.behavior.hb_node_catalog import HBNodeAvailability


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class HBCompatIssue:
    """A single incompatibility record from a compatibility audit.

    Covers both HeartBeat node issues (Circle 1) and semantic action issues
    (Circle 6).  Action-level issues populate the optional fields below.

    Attributes
    ----------
    node_kind         : Canonical HBNodeKind or ``"action.<intent_id>"`` string.
    node_label        : Human-readable display name of the affected node/action.
    availability      : Availability state that triggered this issue.
    reason            : Human-readable explanation.
    severity          : ``"error"`` (unusable) or ``"warning"`` (degraded).
    intent_id         : [Circle 6] Semantic intent ID when this is an
                        action-level issue; ``None`` for HB node issues.
    source_raw_action : [Circle 6] Raw backend action on the source model;
                        ``None`` for HB node issues.
    target_raw_action : [Circle 6] Raw backend action on the target model
                        (``None`` when unsupported/absent or HB node issue).
    """

    node_kind: str
    node_label: str
    availability: str
    reason: str
    severity: str  # "error" | "warning"
    # Circle 6: action-level context (None for HB node issues)
    intent_id: Optional[str] = None
    source_raw_action: Optional[str] = None
    target_raw_action: Optional[str] = None


@dataclass
class HBCompatReport:
    """Aggregate compatibility audit result for a brand/robot_type switch.

    Attributes
    ----------
    brand         : Robot brand string (e.g. ``"unitree"``).
    robot_type    : Robot type string (e.g. ``"go2"``).
    issues        : List of :class:`HBCompatIssue` objects — empty when clean.
    action_report : [Circle 6] Full ``ActionProfileReport`` from semantic action
                    comparison; ``None`` when no timeline was audited or the
                    audit produced no action-level findings.
    """

    brand: str
    robot_type: str
    issues: List[HBCompatIssue] = field(default_factory=list)
    # Circle 6: machine-readable action comparison result
    action_report: "Optional[ActionProfileReport]" = field(default=None, repr=False)

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

    @property
    def action_issue_count(self) -> int:
        """Number of issues that are action-level (have an ``intent_id``)."""
        return sum(1 for i in self.issues if i.intent_id is not None)

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


def audit_action_compatibility(
    timeline: "Optional[BehaviorTimeline]",
    brand: str,
    robot_type: str,
) -> "Tuple[List[HBCompatIssue], Optional[ActionProfileReport]]":
    """Audit authored behavior action intents against the target model's capabilities.

    Extracts unique ``intent_id`` values from the timeline's movement segments,
    builds a source profile from them, and compares against the target model's
    semantic action profile using :func:`compare_action_profiles`.

    Comparison is performed at the ``intent_id`` level — a raw-name difference
    alone is **never** classified as incompatible (REMAPPED is transparent).

    Parameters
    ----------
    timeline   : :class:`BehaviorTimeline` authored on the source model.
                 ``None`` or empty timeline → returns ``([], None)`` safely.
    brand      : Target robot brand (e.g. ``"unitree"``).
    robot_type : Target robot type (e.g. ``"go2"``).

    Returns
    -------
    Tuple of:
        - ``List[HBCompatIssue]`` — only DEGRADED / UNSUPPORTED / AMBIGUOUS
          entries are returned; PRESERVED and REMAPPED produce no issues.
        - ``Optional[ActionProfileReport]`` — full comparison report for
          machine-readable access; ``None`` when no segments were audited.
    """
    from src.system.behavior.action_profile_report import (
        ActionProfileReport,
        CompatStatus,
        compare_action_profiles,
    )
    from src.system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
    from src.system.behavior.action_registry import get_semantic_actions

    if timeline is None or timeline.is_empty():
        return [], None

    # Build source profile: one descriptor per unique intent_id in movement segments.
    seen_intents: dict = {}
    for seg in timeline.action_segments:
        if seg.kind != "movement":
            continue
        intent_id = (seg.intent_id or "").strip()
        if not intent_id or intent_id in seen_intents:
            continue
        # Use segment name as source raw_action (the action as originally authored).
        seen_intents[intent_id] = SemanticActionDescriptor(
            intent_id=intent_id,
            display_name=intent_id,
            category="",
            raw_action=seg.name or intent_id,
            brand="",
            robot_type="",
            parameters={},
            availability=ActionAvailability.AVAILABLE,
            tags=(),
        )

    if not seen_intents:
        return [], None

    source_profile = list(seen_intents.values())
    target_profile = get_semantic_actions(brand, robot_type)

    action_report = compare_action_profiles(source_profile, target_profile)

    _STATUS_SEVERITY = {
        CompatStatus.UNSUPPORTED: "error",
        CompatStatus.DEGRADED:    "warning",
        CompatStatus.AMBIGUOUS:   "warning",
    }

    issues: List[HBCompatIssue] = []
    for entry in action_report.all_entries:
        severity = _STATUS_SEVERITY.get(entry.status)
        if severity is None:
            continue  # PRESERVED / REMAPPED → compatible, no issue
        reason = entry.reason or (
            f"action {entry.intent_id!r} is {entry.status} on target model"
        )
        issues.append(HBCompatIssue(
            node_kind=f"action.{entry.intent_id}",
            node_label=entry.intent_id,
            availability=entry.status,
            reason=reason,
            severity=severity,
            intent_id=entry.intent_id,
            source_raw_action=entry.source_raw_action or None,
            target_raw_action=entry.target_raw_action,
        ))

    return issues, action_report


def report_to_behavior_diagnostics(
    report: HBCompatReport,
) -> "List":
    """Convert a :class:`HBCompatReport` to :class:`BehaviorDiagnostic` records.

    Maps each :class:`HBCompatIssue` to a ``BehaviorDiagnostic`` so that
    compatibility audit results can flow through the IR semantic diagnostics
    pipeline (``DiagnosticsKey.COMPAT_DIAGNOSTICS``).

    Circle 6 Step 6.2: action-level issues (``intent_id`` is not None) produce
    richer ``code`` and ``message`` strings that include semantic intent,
    source raw action, and target raw action for machine-readable diagnostics.

    Parameters
    ----------
    report : :class:`HBCompatReport` from :func:`audit_catalog_compatibility`
             or a merged report that also includes action-level issues.

    Returns
    -------
    ``List[BehaviorDiagnostic]`` — empty list when ``report.is_clean()``.
    """
    from src.system.behavior.behavior_artifact import BehaviorDiagnostic  # local — avoids circular

    result: List = []
    for issue in report.issues:
        if issue.intent_id is not None:
            # Circle 6 Step 6.2: richer code + message for action-level issues.
            code = f"compat.action.{issue.availability}.{issue.intent_id}"
            parts = [f"[{issue.node_label}] {issue.reason}"]
            if issue.source_raw_action:
                parts.append(f"src={issue.source_raw_action!r}")
            if issue.target_raw_action:
                parts.append(f"tgt={issue.target_raw_action!r}")
            message = " | ".join(parts)
        else:
            code    = f"compat.{issue.availability}.{issue.node_kind}"
            message = f"[{issue.node_label}] {issue.reason}"
        diag = (
            BehaviorDiagnostic.error(code, message, location=issue.node_kind)
            if issue.severity == "error"
            else BehaviorDiagnostic.warning(code, message, location=issue.node_kind)
        )
        result.append(diag)
    return result
