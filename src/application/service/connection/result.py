"""ConnectionResult — final verdict of one AutoConnect run.

Phase 3.7 introduced an autonomous repair loop that runs inside
``Ros2BrownfieldConnectTask.P9_AUTOCONNECT``: P9 keeps re-running the
diagnostic + safe-repair cycle (up to N attempts) until either every
finding clears or the loop is starved of safe repairs. The single
:class:`ConnectionResult` instance shipped at the end captures everything
the UI needs to render :class:`ConnectionResultDialog` without re-querying.

Findings that arrived in the final report are classified once via
:func:`classify_findings`:

* ``unresolved_safe``     — has a safe repair but the loop tried it and
                            it failed (or didn't help). User can retry.
* ``unresolved_invasive`` — has a repair, but ``safe=False`` so the loop
                            never tried it. ResultDialog surfaces an
                            ``[Apply Invasive Fixes]`` button for these.
* ``unresolved_manual``   — no repair attached at all. User must act
                            outside the app (cable, power, manual SSH).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from application.service.diagnostics.results import (
    DiagnosticFinding,
    DiagnosticReport,
    RepairAction,
    Severity,
)


@dataclass(frozen=True)
class ConnectionResult:
    """Snapshot of one AutoConnect attempt's final state."""

    ok: bool
    """True iff the final diagnostic report passed (no >=WARNING findings)."""

    applied: List[RepairAction] = field(default_factory=list)
    """Repairs the loop ran successfully across all attempts."""

    failed: List[Tuple[RepairAction, str]] = field(default_factory=list)
    """(action, error_msg) for repairs the loop tried that raised."""

    unresolved_safe: List[DiagnosticFinding] = field(default_factory=list)
    """Findings with a safe repair the loop couldn't clear (worth retry)."""

    unresolved_invasive: List[DiagnosticFinding] = field(default_factory=list)
    """Findings whose only repair is invasive — held back for user opt-in."""

    unresolved_manual: List[DiagnosticFinding] = field(default_factory=list)
    """Findings with no repair attached — needs user action outside app."""

    duration_s: float = 0.0
    attempts: int = 0

    @property
    def has_partial(self) -> bool:
        """True when the bridge is up but findings still need attention."""
        return not self.ok and (
            bool(self.unresolved_safe)
            or bool(self.unresolved_invasive)
            or bool(self.unresolved_manual)
        )

    @property
    def total_unresolved(self) -> int:
        return (
            len(self.unresolved_safe)
            + len(self.unresolved_invasive)
            + len(self.unresolved_manual)
        )

    def invasive_actions(self) -> List[RepairAction]:
        """Deduplicated invasive RepairAction list for [Apply Invasive Fixes]."""
        out: List[RepairAction] = []
        seen: set = set()
        for f in self.unresolved_invasive:
            if f.repair is None:
                continue
            key = (f.repair.name, f.repair.describe)
            if key in seen:
                continue
            seen.add(key)
            out.append(f.repair)
        return out


def classify_findings(
    report: DiagnosticReport,
) -> Tuple[List[DiagnosticFinding], List[DiagnosticFinding], List[DiagnosticFinding]]:
    """Bucket findings into (safe, invasive, manual) for ConnectionResult.

    Only WARNING+ findings are considered — OK / INFO entries are noise
    here. A finding is bucketed by:

    * ``repair is None``         -> manual
    * ``repair.safe is True``    -> safe (loop will / did try it)
    * ``repair.safe is False``   -> invasive (loop never tries)
    """
    safe: List[DiagnosticFinding] = []
    invasive: List[DiagnosticFinding] = []
    manual: List[DiagnosticFinding] = []
    for f in report.findings:
        if f.severity < Severity.WARNING:
            continue
        if f.repair is None:
            manual.append(f)
        elif f.repair.safe:
            safe.append(f)
        else:
            invasive.append(f)
    return safe, invasive, manual


__all__ = ["ConnectionResult", "classify_findings"]
