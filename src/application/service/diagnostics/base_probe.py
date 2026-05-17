"""DiagnosticProbe ABC — every probe is a stateless object that runs once.

A probe is defined by:

* A class-level ``id`` string (used to group findings and theme severity chips).
* A class-level ``requires_ssh`` flag — when True the probe is skipped with
  an INFO finding if ``ctx.ssh is None``.
* A ``run(ctx) -> List[DiagnosticFinding]`` method.

Probes never raise out — DiagnosticsTask catches and converts to ERROR
findings, so a buggy probe never aborts the rest of the run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import (
    DiagnosticFinding,
    RepairAction,
    Severity,
)


class DiagnosticProbe(ABC):
    """Abstract probe.

    Subclasses set the class attributes below and implement :meth:`run`.
    """

    id: str = ""
    """Stable string used as the probe id in findings; subclass MUST set."""

    requires_ssh: bool = False
    """When True the probe is skipped with INFO if ctx.ssh is None."""

    @abstractmethod
    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        """Inspect ``ctx`` and return a list of findings.

        Multiple findings per probe are allowed (e.g. one per missing
        dependency). Empty list = OK; equivalent to a single OK finding.
        """

    # ------------------------------------------------------------------
    # Convenience helpers — keeps subclass code terse.
    # ------------------------------------------------------------------

    def make_finding(
        self,
        severity: Severity,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return DiagnosticFinding(
            probe_id=self.id or self.__class__.__name__,
            severity=severity,
            summary=summary,
            detail=detail,
            repair=repair,
            requires_ssh=self.requires_ssh if requires_ssh is None else requires_ssh,
        )

    def ok(
        self,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return self.make_finding(
            Severity.OK, summary,
            detail=detail, repair=repair, requires_ssh=requires_ssh,
        )

    def info(
        self,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return self.make_finding(
            Severity.INFO, summary,
            detail=detail, repair=repair, requires_ssh=requires_ssh,
        )

    def warning(
        self,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return self.make_finding(
            Severity.WARNING, summary,
            detail=detail, repair=repair, requires_ssh=requires_ssh,
        )

    def error(
        self,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return self.make_finding(
            Severity.ERROR, summary,
            detail=detail, repair=repair, requires_ssh=requires_ssh,
        )

    def critical(
        self,
        summary: str,
        *,
        detail: str = "",
        repair: Optional[RepairAction] = None,
        requires_ssh: Optional[bool] = None,
    ) -> DiagnosticFinding:
        return self.make_finding(
            Severity.CRITICAL, summary,
            detail=detail, repair=repair, requires_ssh=requires_ssh,
        )


__all__ = ["DiagnosticProbe"]
