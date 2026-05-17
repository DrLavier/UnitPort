"""RepairTask — apply a list of RepairAction against a DiagnosticContext.

Filters by ``safe_only`` flag, runs sequentially (a repair may depend on
the previous one — e.g. installing a package before writing a service
drop-in), aggregates failures, broadcasts the resulting
:class:`RepairReport` on ``AppSignals.diagnostics_repair_finished``.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from unitport_sdk import (
    Task,
    TaskCancelledException,
    log_debug,
    log_error,
    log_success,
    log_warning,
)

from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import RepairAction, RepairReport
from application.service.signals import get_app_signals


class RepairTask(Task):
    """Apply ``actions`` against ``ctx_factory()`` output.

    The factory is called once at the start so the SSH session lives across
    the whole repair sequence (avoids re-prompting credentials between
    actions). Pass ``safe_only=True`` to skip ``RepairAction.safe == False``
    entries and report them under ``skipped_unsafe``.
    """

    def __init__(
        self,
        actions: List[RepairAction],
        ctx_factory: Callable[[], DiagnosticContext],
        ssh_factory: Optional[Callable[[object], object]] = None,
        *,
        safe_only: bool,
        name: str = "Diagnostics Repair",
    ) -> None:
        super().__init__(name)
        self._actions = list(actions)
        self._ctx_factory = ctx_factory
        self._ssh_factory = ssh_factory
        self._safe_only = bool(safe_only)
        self._report: Optional[RepairReport] = None

    @property
    def report(self) -> Optional[RepairReport]:
        return self._report

    def run(self) -> str:
        signals = get_app_signals()
        try:
            signals.diagnostics_repair_started.emit()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[repair] repair_started emit failed: {exc}")

        ctx = self._ctx_factory()
        report = RepairReport()
        ssh_obj: Optional[object] = None
        ssh_attempted = False
        t0 = time.monotonic()
        # Total only counts actions we'll actually attempt (skipped_unsafe
        # are reported separately so the dialog shows a clean "k/N" count).
        total = sum(1 for a in self._actions if a.safe or not self._safe_only)
        idx = 0
        try:
            for action in self._actions:
                self.check_cancelled()
                if self._safe_only and not action.safe:
                    report.skipped_unsafe.append(action)
                    log_debug(f"[repair] {action.name} skipped (unsafe)")
                    continue
                idx += 1
                try:
                    signals.diagnostics_repair_progress.emit(
                        action.name, idx, total,
                    )
                except Exception as exc:  # noqa: BLE001
                    log_warning(f"[repair] repair_progress emit failed: {exc}")

                # Lazy SSH on first action that needs it. Repair actions
                # don't declare requires_ssh on themselves; we open SSH
                # eagerly on the first action attempt because most repair
                # paths target the robot. Host-only safe actions (firewall
                # rules) don't touch ctx.ssh and so don't pay any cost.
                local_ctx = ctx
                if (
                    self._ssh_factory is not None
                    and ssh_obj is None
                    and not ssh_attempted
                ):
                    ssh_attempted = True
                    try:
                        ssh_obj = self._ssh_factory(ctx.profile)
                    except Exception as exc:  # noqa: BLE001
                        log_warning(f"[repair] ssh_factory raised: {exc}")
                if ssh_obj is not None:
                    local_ctx = ctx.with_ssh(ssh_obj)  # type: ignore[arg-type]

                t = time.monotonic()
                log_debug(f"[repair] {action.name} start")
                try:
                    action.run(local_ctx)
                except TaskCancelledException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    report.failed.append((action, str(exc)))
                    log_error(f"[repair] {action.name} FAILED: {exc}")
                    continue
                report.applied.append(action)
                log_success(
                    f"[repair] {action.name} ok ({time.monotonic() - t:.2f}s)"
                )
        except TaskCancelledException:
            log_warning("[repair] cancelled")
            raise
        finally:
            report.duration_s = time.monotonic() - t0
            self._report = report
            try:
                signals.diagnostics_repair_finished.emit(report)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[repair] repair_finished emit failed: {exc}")
            if ssh_obj is not None:
                close = getattr(ssh_obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001
                        log_warning(f"[repair] ssh close raised: {exc}")
        return "ok" if report.all_succeeded else "partial"


__all__ = ["RepairTask"]
