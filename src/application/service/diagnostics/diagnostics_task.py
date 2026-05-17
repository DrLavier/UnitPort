"""DiagnosticsTask — runs a probe pipeline and emits one report.

Probe execution rules:

* Probes run sequentially in given order. Errors thrown by a probe are
  caught and converted to a CRITICAL finding so a single buggy probe
  never aborts the rest of the run.
* SSH session is built lazily on the first probe with ``requires_ssh=True``,
  using credentials from :class:`SecureCredentialStore`. When credentials
  are missing or the connect fails, every SSH-required probe gets an
  ERROR finding tagged with summary ``ssh_required`` (Phase 3.7 changed
  this from a silent INFO skip — AutoRepairLoop pattern-matches on the
  summary to know when to prompt the user for credentials).
* The final :class:`DiagnosticReport` is broadcast on
  ``AppSignals.diagnostics_ready``; the SSH session (if any) is closed
  in ``run()`` finally.

Log levels (Phase 3.7 contract):

* probe start / per-probe trace -> ``log_debug`` (default cmd_log hides)
* probe outcome OK / INFO       -> ``log_debug``
* probe outcome WARNING         -> ``log_warning``
* probe outcome ERROR/CRITICAL  -> ``log_error``
* probe crashed                  -> ``log_error``
* task cancelled                 -> ``log_warning``
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from unitport_sdk import (
    Task,
    TaskCancelledException,
    log_debug,
    log_error,
    log_warning,
)

from application.service.connection.profile import ConnectionProfile
from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import (
    DiagnosticFinding,
    DiagnosticReport,
    Severity,
)
from application.service.signals import get_app_signals


# Type of factory that returns a fresh DiagnosticContext when called.
ContextFactory = Callable[[], DiagnosticContext]


# Sentinel summary used by the ssh_required finding so AutoRepairLoop can
# detect "we need credentials" without parsing free-form text. Kept here
# (and re-exported via auto_repair_loop) so probes never accidentally
# pick the same summary string.
SSH_REQUIRED_SUMMARY = "ssh_required"


def _level_for(severity: Severity):
    """Return the log_* function appropriate for ``severity``."""
    if severity >= Severity.ERROR:
        return log_error
    if severity == Severity.WARNING:
        return log_warning
    return log_debug


class DiagnosticsTask(Task):
    """Run ``probes`` against a context built by ``ctx_factory``.

    ``ssh_factory`` (optional) is called the first time an SSH-required
    probe runs. It returns either a connected :class:`SSHSession` or
    ``None`` if credentials are unavailable / connect failed.
    """

    def __init__(
        self,
        probes: List[DiagnosticProbe],
        ctx_factory: ContextFactory,
        ssh_factory: Optional[Callable[[ConnectionProfile], object]] = None,
        name: str = "Diagnostics",
    ) -> None:
        super().__init__(name)
        self._probes = list(probes)
        self._ctx_factory = ctx_factory
        self._ssh_factory = ssh_factory
        self._report: Optional[DiagnosticReport] = None

    @property
    def report(self) -> Optional[DiagnosticReport]:
        return self._report

    def run(self) -> str:
        signals = get_app_signals()
        ctx = self._ctx_factory()
        report = DiagnosticReport()
        ssh_obj: Optional[object] = None
        ssh_attempted = False
        t0 = time.monotonic()
        try:
            for probe in self._probes:
                self.check_cancelled()
                pid = probe.id or probe.__class__.__name__
                log_debug(f"[diagnose] {pid} start")

                local_ctx = ctx
                if probe.requires_ssh:
                    if ssh_obj is None and not ssh_attempted:
                        ssh_attempted = True
                        ssh_obj = self._try_open_ssh(ctx.profile)
                    local_ctx = ctx.with_ssh(ssh_obj)  # type: ignore[arg-type]
                    if ssh_obj is None:
                        # Phase 3.7: NEVER silently skip — AutoRepairLoop
                        # needs to see this as an ERROR so it can prompt
                        # for credentials and retry.
                        report.findings.append(
                            DiagnosticFinding(
                                probe_id=pid,
                                severity=Severity.ERROR,
                                summary=SSH_REQUIRED_SUMMARY,
                                detail=(
                                    "This probe needs SSH to inspect the "
                                    "robot but no credentials are saved for "
                                    "this server. Save SSH password (and "
                                    "sudo password if required) in the "
                                    "connection card and re-run."
                                ),
                                requires_ssh=True,
                            )
                        )
                        log_error(
                            f"[diagnose] {pid} ERROR: ssh_required "
                            "(no saved credentials)"
                        )
                        continue

                t = time.monotonic()
                try:
                    findings = probe.run(local_ctx) or []
                except TaskCancelledException:
                    raise
                except Exception as exc:  # noqa: BLE001 — guard host run
                    findings = [
                        DiagnosticFinding(
                            probe_id=pid,
                            severity=Severity.CRITICAL,
                            summary=f"probe crashed: {type(exc).__name__}",
                            detail=str(exc),
                        )
                    ]
                    log_error(f"[diagnose] {pid} crashed: {exc}")
                report.findings.extend(findings)
                worst = max(
                    (f.severity for f in findings), default=Severity.OK,
                )
                # Per-probe outcome line — level scales with severity so
                # OK/INFO traces hide under log_debug while ERROR/WARNING
                # surface in the default cmd_log view.
                worst_summary = next(
                    (f.summary for f in findings if f.severity == worst),
                    "",
                )
                duration = time.monotonic() - t
                log_fn = _level_for(worst)
                if worst >= Severity.WARNING and worst_summary:
                    log_fn(
                        f"[diagnose] {pid} {worst.label.upper()}: "
                        f"{worst_summary} ({duration:.2f}s)"
                    )
                else:
                    log_fn(
                        f"[diagnose] {pid} {worst.label} ({duration:.2f}s)"
                    )
        except TaskCancelledException:
            log_warning("[diagnose] cancelled")
            raise
        finally:
            report.duration_s = time.monotonic() - t0
            self._report = report
            try:
                signals.diagnostics_ready.emit(report)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[diagnose] diagnostics_ready emit failed: {exc}")
            if ssh_obj is not None:
                close = getattr(ssh_obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001
                        log_warning(f"[diagnose] ssh close raised: {exc}")
        return report.worst_severity.label

    def _try_open_ssh(self, profile: ConnectionProfile) -> Optional[object]:
        if self._ssh_factory is None:
            return None
        try:
            ssh = self._ssh_factory(profile)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[diagnose] ssh_factory raised: {exc}")
            return None
        return ssh


__all__ = [
    "DiagnosticsTask",
    "ContextFactory",
    "SSH_REQUIRED_SUMMARY",
]
