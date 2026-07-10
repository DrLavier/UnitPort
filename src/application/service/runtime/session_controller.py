# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SessionController — the system-action dispatcher for estop (Slice 0).

``skill_command_path_design.md`` §4.1: estop is a SYSTEM channel — unconditional
preemption of the active session, and it must NEVER travel through the policy or the
command vector (a zeroed command vector is a *soft* velocity-stop, still a policy
step; estop is a hard stop of the whole session).

This process-singleton is the single subscriber of ``AppSignals.system_estop`` for the
live-sim / review path: it tracks the active review task id (registered by every review
submit site) and, on estop, cancels it via ``TasksManager.cancel`` — a thread-safe
flag-flip honoured at the review loop's ``check_cancelled()`` boundaries. The tid is
cleared when the task finishes (``TaskSignal.task_finished``), covering normal exit,
fall-termination, and the estop-cancel itself.

Deploy / real-hardware estop is handled separately by
``ConnectionControllerCard`` (which also subscribes to ``system_estop`` and calls
``adapter.disable_teleop()``) — that owner already holds the adapter, so this
controller stays focused on the session/task side. Pressing estop with no active
session is a safe no-op.
"""

from __future__ import annotations

import threading
from typing import Optional, Set

from PyQt6.QtCore import QObject

from unitport_sdk import get_task_signal, get_tasks_manager, log_warning


class SessionController(QObject):
    """Tracks the active live-sim/review tasks and cancels them on system estop.

    A set (not a single id) because more than one review viewer can be open at
    once; estop is unconditional and must stop *all* active review sessions. The
    set drains as tasks finish (``task_finished``).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._review_task_ids: Set[str] = set()
        # Single subscriber of the system estop channel (sim/review side).
        from application.service.signals import get_app_signals
        get_app_signals().system_estop.connect(self._on_system_estop)
        # Drop tracked tids whenever a task finishes (any reason).
        get_task_signal().task_finished.connect(self._on_task_finished)

    # ----- registration ----------------------------------------------------

    def register_review_task(self, task_id: str) -> None:
        """Record an active live-sim/review task so estop can reach it.

        Called by every review submit site (Export-node launch + Mission Control
        card, policy + pose viewers). A blank id is ignored.
        """
        tid = str(task_id or "").strip()
        if tid:
            self._review_task_ids.add(tid)

    @property
    def active_review_task_ids(self) -> Set[str]:
        return set(self._review_task_ids)

    # ----- system estop ----------------------------------------------------

    def _on_system_estop(self, source: str = "") -> None:
        tids = sorted(self._review_task_ids)
        if not tids:
            return  # no active review/live-sim session — safe no-op
        for tid in tids:
            log_warning(
                f"[estop] cancelling live-sim/review task {tid} (source={source})"
            )
            try:
                get_tasks_manager().cancel(tid)
            except Exception as exc:
                log_warning(f"[estop] cancel review task {tid} raised: {exc}")

    def _on_task_finished(self, task_id: str, success: bool, result: object) -> None:
        self._review_task_ids.discard(str(task_id))


_singleton: Optional[SessionController] = None
_singleton_lock = threading.Lock()


def get_session_controller() -> SessionController:
    """Return the process-wide SessionController (lazy). Creating it subscribes it
    to the system-estop + task-finished signals, so it must be created on the GUI
    thread (which every review submit site already runs on)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = SessionController()
    return _singleton


__all__ = ["SessionController", "get_session_controller"]
