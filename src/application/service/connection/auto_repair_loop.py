"""AutoRepairLoop — autonomous diagnose+safe-repair loop for P9_AUTOCONNECT.

Replaces the user-driven ConnectionDiagnosticsDialog/RepairTask flow with
an in-task loop that:

1. runs :class:`DiagnosticsTask` synchronously,
2. classifies findings into safe / invasive / manual buckets,
3. if any SSH-required findings exist but no credentials are saved,
   emits :pyqtsig:`AppSignals.connection_needs_ssh` and blocks on a
   :class:`threading.Event` until UI hands credentials back via
   ``Ros2ConnectionController.submit_ssh_response``,
4. applies every safe repair sequentially (logging success/error per
   action with the appropriate ``log_*`` level),
5. re-runs diagnostics; loops up to ``max_attempts`` times.

The final :class:`ConnectionResult` is returned to the caller (the
brownfield connect task) which broadcasts it on
:pyqtsig:`AppSignals.connection_result` for the UI ResultDialog.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Callable, List, Optional, Tuple

from unitport_sdk import (
    Task,
    TaskCancelledException,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)

from application.service.connection.profile import ConnectionProfile
from application.service.connection.result import (
    ConnectionResult,
    classify_findings,
)
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.diagnostics_task import (
    SSH_REQUIRED_SUMMARY,
    DiagnosticsTask,
)
from application.service.diagnostics.results import (
    DiagnosticFinding,
    DiagnosticReport,
    RepairAction,
    Severity,
)
from application.service.signals import get_app_signals

# How long we wait for the user to fill / cancel the SSH prompt before
# giving up and treating the SSH attempt as cancelled.
_SSH_PROMPT_TIMEOUT_S = 120.0


# Type aliases for the factories the loop receives from the connect task.
ContextFactory = Callable[[], DiagnosticContext]
SshFactory = Callable[[ConnectionProfile], object]


class AutoRepairLoop:
    """Run diagnose+safe-repair in a closed loop until pass or starvation."""

    def __init__(
        self,
        owner_task: Task,
        probes_factory: Callable[[], List[object]],
        ctx_factory: ContextFactory,
        ssh_factory: SshFactory,
        profile: ConnectionProfile,
        server_key: str,
        suggested_user: str,
        *,
        max_attempts: int = 3,
    ) -> None:
        self._owner = owner_task
        self._probes_factory = probes_factory
        self._ctx_factory = ctx_factory
        self._ssh_factory = ssh_factory
        self._profile = profile
        self._server_key = server_key
        self._suggested_user = suggested_user
        self._max_attempts = max_attempts

        # SSH prompt round-trip — set by ``notify_ssh_response`` from the
        # controller when the user clicks Save & Continue / Cancel.
        self._ssh_event = threading.Event()
        self._ssh_response: Tuple[str, bool] = ("", False)

    # ------------------------------------------------------------------
    # Public API used by Ros2BrownfieldConnectTask
    # ------------------------------------------------------------------

    def run(self) -> ConnectionResult:
        """Drive the loop; returns the final ConnectionResult."""
        t0 = time.monotonic()
        applied: List[RepairAction] = []
        failed: List[Tuple[RepairAction, str]] = []
        attempts = 0
        last_report: Optional[DiagnosticReport] = None

        for attempt in range(1, self._max_attempts + 1):
            self._owner.check_cancelled()
            attempts = attempt

            log_info(
                f"[auto-connect] diagnose attempt {attempt}/{self._max_attempts}"
            )
            report = self._run_diagnose()
            last_report = report

            if report.passed:
                log_success(
                    f"[auto-connect] all {len(report.findings)} probe(s) ok "
                    f"on attempt {attempt}"
                )
                break

            # SSH-required findings without an active session -> prompt user.
            ssh_blocked = self._collect_ssh_blocked(report)
            if ssh_blocked:
                ok = self._prompt_ssh(reason=ssh_blocked[0].summary)
                if not ok:
                    log_error(
                        "[auto-connect] user declined SSH prompt; "
                        "leaving SSH-required findings unresolved"
                    )
                    break
                # Don't count this iteration against max_attempts — credentials
                # arrived, retry with full SSH context.
                attempts = attempt - 1 if attempt > 1 else 0
                continue

            # Apply every safe repair this report knows about.
            safe_actions = report.repairable(safe_only=True)
            if not safe_actions:
                log_warning(
                    f"[auto-connect] {len(report.findings)} finding(s) but no "
                    "safe repair available — escalating to user"
                )
                break

            new_applied, new_failed = self._apply_safe(safe_actions)
            applied.extend(new_applied)
            failed.extend(new_failed)

            # If every safe repair raised, the next attempt would just re-run
            # the same diagnose with the same broken state. Bail out so the
            # ResultDialog can list what failed.
            if not new_applied:
                log_error(
                    "[auto-connect] every safe repair failed this attempt; "
                    "escalating to user"
                )
                break

        # Loop exited — bucket whatever's still on the report (or empty if
        # the loop never produced one).
        if last_report is None:
            last_report = DiagnosticReport()
        passed = last_report.passed
        unresolved_safe, unresolved_invasive, unresolved_manual = (
            classify_findings(last_report)
        )

        result = ConnectionResult(
            ok=passed,
            applied=applied,
            failed=failed,
            unresolved_safe=unresolved_safe if not passed else [],
            unresolved_invasive=unresolved_invasive if not passed else [],
            unresolved_manual=unresolved_manual if not passed else [],
            duration_s=time.monotonic() - t0,
            attempts=attempts,
        )
        if passed:
            log_success(
                f"[auto-connect] connection ready "
                f"(auto-fixed {len(applied)} issue(s) in {result.duration_s:.1f}s)"
            )
        elif failed:
            log_error(
                f"[auto-connect] connection partial: "
                f"applied={len(applied)} failed={len(failed)} "
                f"unresolved={result.total_unresolved}"
            )
        else:
            log_warning(
                f"[auto-connect] connection partial: "
                f"unresolved={result.total_unresolved} (no auto fix available)"
            )
        return result

    def notify_ssh_response(self, server_key: str, ok: bool) -> None:
        """Wake the loop after the SshCredentialPromptDialog closes.

        Called by ``Ros2ConnectionController.submit_ssh_response`` from the
        UI thread; safe across threads because we just store + set an Event.
        """
        if server_key != self._server_key:
            return
        self._ssh_response = (server_key, ok)
        self._ssh_event.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_diagnose(self) -> DiagnosticReport:
        probes = self._probes_factory()
        diag = DiagnosticsTask(
            probes=probes,
            ctx_factory=self._ctx_factory,
            ssh_factory=self._ssh_factory,
            name="Diagnostics (auto-connect)",
        )
        try:
            diag.run()
        except TaskCancelledException:
            raise
        except Exception as exc:  # noqa: BLE001
            log_error(f"[auto-connect] DiagnosticsTask raised: {exc}")
            return DiagnosticReport()
        return diag.report or DiagnosticReport()

    @staticmethod
    def _collect_ssh_blocked(
        report: DiagnosticReport,
    ) -> List[DiagnosticFinding]:
        return [
            f for f in report.findings
            if f.severity >= Severity.WARNING
            and f.summary == SSH_REQUIRED_SUMMARY
        ]

    def _prompt_ssh(self, reason: str) -> bool:
        """Emit ``connection_needs_ssh`` and block on user response.

        Returns True when the user saved credentials, False on cancel /
        timeout.
        """
        log_warning(
            f"[auto-connect] SSH credentials required for {self._server_key} "
            f"({reason}); prompting user"
        )
        self._ssh_event.clear()
        self._ssh_response = ("", False)
        try:
            get_app_signals().connection_needs_ssh.emit(
                self._server_key, self._suggested_user, reason,
            )
        except Exception as exc:  # noqa: BLE001
            log_error(f"[auto-connect] connection_needs_ssh emit failed: {exc}")
            return False

        # Wait in short slices so we honour task cancellation promptly.
        deadline = time.monotonic() + _SSH_PROMPT_TIMEOUT_S
        while time.monotonic() < deadline:
            self._owner.check_cancelled()
            if self._ssh_event.wait(timeout=0.5):
                server, ok = self._ssh_response
                return bool(ok)
        log_error(
            "[auto-connect] SSH prompt timed out "
            f"({_SSH_PROMPT_TIMEOUT_S:.0f}s); treating as cancel"
        )
        return False

    def _apply_safe(
        self, actions: List[RepairAction],
    ) -> Tuple[List[RepairAction], List[Tuple[RepairAction, str]]]:
        applied: List[RepairAction] = []
        failed: List[Tuple[RepairAction, str]] = []
        ctx = self._ctx_factory()
        # Re-bind SSH on the local ctx so repairs that need it have a session.
        ssh_obj = self._ssh_factory(self._profile)
        if ssh_obj is not None:
            ctx = ctx.with_ssh(ssh_obj)

        try:
            for action in actions:
                self._owner.check_cancelled()
                if not action.safe:
                    log_debug(f"[auto-connect] skip non-safe repair {action.name}")
                    continue
                t = time.monotonic()
                log_debug(f"[auto-connect] repair {action.name} start")
                try:
                    action.run(ctx)
                except TaskCancelledException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log_error(
                        f"[auto-connect] repair {action.name} FAILED: {exc}"
                    )
                    failed.append((action, str(exc)))
                    continue
                applied.append(action)
                log_success(
                    f"[auto-connect] repair {action.name} ok "
                    f"({time.monotonic() - t:.2f}s)"
                )
        finally:
            if ssh_obj is not None:
                close = getattr(ssh_obj, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001
                        log_warning(
                            f"[auto-connect] ssh close raised: {exc}"
                        )
        return applied, failed


__all__ = [
    "AutoRepairLoop",
    "ContextFactory",
    "SshFactory",
]
