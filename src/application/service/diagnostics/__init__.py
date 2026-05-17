"""Diagnostics framework — Phase 3.5 self-diagnose / self-repair.

Public API surface (every name a caller should ever need):

* :class:`DiagnosticProbe` — base class for individual probes.
* :class:`DiagnosticContext` — per-run snapshot of profile / transport / bridge / SSH.
* :class:`DiagnosticFinding` / :class:`Severity` / :class:`RepairAction` —
  data carried across the framework.
* :class:`DiagnosticReport` / :class:`RepairReport` — aggregate outputs.
* :class:`DiagnosticsTask` / :class:`RepairTask` — SDK Task wrappers.

Probe implementations are not re-exported from this package — host probes
live under :mod:`application.service.diagnostics.probes`, brand probes under
``application/service/adapters/<brand>/probes.py``.
"""

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.diagnostics_task import DiagnosticsTask
from application.service.diagnostics.repair_task import RepairTask
from application.service.diagnostics.results import (
    DiagnosticFinding,
    DiagnosticReport,
    RepairAction,
    RepairReport,
    Severity,
)


__all__ = [
    "DiagnosticProbe",
    "DiagnosticContext",
    "DiagnosticFinding",
    "DiagnosticReport",
    "DiagnosticsTask",
    "RepairAction",
    "RepairReport",
    "RepairTask",
    "Severity",
]
