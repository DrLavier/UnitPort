"""DiagnosticFinding / RepairAction / DiagnosticReport dataclasses.

Contract surface for the Phase 3.5 self-diagnose / self-repair framework.
Every probe returns a list of ``DiagnosticFinding``; the worst severity in a
report drives the P9 phase status (success/warning/error) and whether the
``ConnectionDiagnosticsDialog`` opens automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from application.service.diagnostics.context import DiagnosticContext


class Severity(IntEnum):
    """Five-level severity ordered worst-last so ``max(...)`` works.

    OK / INFO never block; WARNING surfaces in dialog without auto-popup;
    ERROR / CRITICAL trigger the dialog and (if ERROR) keep the connection
    in the connected state with a yellow phase chip — they do not roll back
    P0..P8 success.
    """

    OK = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class RepairAction:
    """A single fix the user can apply against one finding.

    ``run`` receives the same ``DiagnosticContext`` the probe ran with so it
    can reach SSH, the bridge, the active transport. Raise on failure; the
    RepairTask catches and aggregates into ``RepairReport.failed``.

    ``safe`` distinguishes idempotent host-only / restart-only fixes from
    invasive operations (apt install, persistent robot file writes). The
    dialog renders two buttons: ``[Apply All Safe Fixes]`` filters by this.
    """

    name: str
    describe: str
    run: Callable[["DiagnosticContext"], None]
    safe: bool = True


@dataclass
class DiagnosticFinding:
    """One probe verdict.

    ``probe_id`` is the producing probe's ``id`` class attribute (used to
    group findings in the dialog). ``repair`` may be None (informational
    only). ``requires_ssh`` lets the UI auto-expand the SSH credentials
    section in sec1 when any finding flags it.
    """

    probe_id: str
    severity: Severity
    summary: str
    detail: str = ""
    repair: Optional[RepairAction] = None
    requires_ssh: bool = False


@dataclass
class DiagnosticReport:
    """Aggregate of one DiagnosticsTask run."""

    findings: List[DiagnosticFinding] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def worst_severity(self) -> Severity:
        if not self.findings:
            return Severity.OK
        return max(f.severity for f in self.findings)

    @property
    def passed(self) -> bool:
        return self.worst_severity <= Severity.INFO

    @property
    def has_warnings(self) -> bool:
        return self.worst_severity == Severity.WARNING

    @property
    def has_errors(self) -> bool:
        return self.worst_severity >= Severity.ERROR

    def repairable(self, *, safe_only: bool = False) -> List[RepairAction]:
        out: List[RepairAction] = []
        seen: set = set()
        for f in self.findings:
            if f.repair is None:
                continue
            if safe_only and not f.repair.safe:
                continue
            key = (f.repair.name, f.repair.describe)
            if key in seen:
                continue
            seen.add(key)
            out.append(f.repair)
        return out


@dataclass
class RepairReport:
    """Aggregate of one RepairTask run."""

    applied: List[RepairAction] = field(default_factory=list)
    failed: List[Tuple[RepairAction, str]] = field(default_factory=list)
    skipped_unsafe: List[RepairAction] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        return not self.failed and bool(self.applied)


__all__ = [
    "Severity",
    "RepairAction",
    "DiagnosticFinding",
    "DiagnosticReport",
    "RepairReport",
]
