# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AutoSyncController — persist-time cloud auto-sync driver.

This is the application-side half of the SDK's persist observer
(:meth:`unitport_sdk.Storage.register_persist_observer`). When any user-config
file is written through ``DataManager.write`` (the single on-disk funnel:
``Storage.push(local)``, direct ``save_data`` for canvas/project files, and
``Config._flush`` for ``user.ini`` all pass through it), the SDK hands us the
rel-path. We decide whether it should sync and, if so, push it to the cloud —
so cloud-sync follows the user's edits at save-time instead of a batch upload on
exit.

Design
------
- **Eligibility is decided at persist-time, cheaply.** ``notify_persist`` runs
  on the writing thread and only enqueues when: the Auto-sync toggle
  (``user.ini[Cloud] auto_push``) is on, the user is signed in, and the file is
  in cloud-sync's include set (:func:`cloud_sync.is_syncable_rel`). Anything
  else is dropped immediately — no queue growth, no work.

- **Never block the write path.** ``notify_persist`` does not touch the network;
  it appends to a thread-safe pending set and kicks a debounce timer via a
  queued Qt signal, so the actual upload always happens on a TasksManager worker
  thread. A save of the canvas never stalls on a Supabase round-trip.

- **Debounce + coalesce.** A burst of writes (saving a project touches many
  files) collapses into one push after the window (``[CloudSync]
  auto_sync_debounce_ms``, default 1500). Re-writing the same file N times in
  the window uploads once. Only one auto-sync push runs at a time; files that
  arrive while a push is in flight flush in the next round.

The heavy lifting (transforms, size-limit, content-diff, state) lives in
:class:`CloudSyncService`; this class only gates + batches + schedules.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Set

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from unitport_sdk import (
    Config,
    get_task_signal,
    get_tasks_manager,
    log_debug,
    log_warning,
)

from application.service.cloud_sync import is_syncable_rel

_DEFAULT_DEBOUNCE_MS = 1500


class AutoSyncController(QObject):
    """Singleton. Bridges the SDK persist observer to a debounced cloud push."""

    # Cross-thread kick: ``notify_persist`` may run on any worker thread, but the
    # debounce QTimer must be (re)started on the thread that owns it (the GUI
    # thread this object lives on). A queued-connection signal marshals it.
    _persist_queued = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pending: Set[str] = set()
        self._lock = threading.Lock()
        # id of the in-flight AutoSyncPushTask, or None when idle. Guards
        # against overlapping auto-sync pushes (they serialize).
        self._task_id: Optional[str] = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_debounce_timeout)

        # Default connection type is Auto → queued when emitted from another
        # thread, so the timer is always armed on this object's own thread.
        self._persist_queued.connect(self._arm_timer)

        # Learn when our push task finishes so a follow-up batch can flush.
        try:
            get_task_signal().task_finished.connect(self._on_task_finished)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[auto-sync] task_finished wire failed: {exc}")

    # ----- SDK persist observer entrypoint --------------------------------

    def notify_persist(self, rel_posix: str) -> None:
        """Called by the SDK after a successful USER_CONFIG_DIR write.

        Thread-safe, non-blocking, and swallows its own errors: this runs on the
        write path and must never turn a successful persist into a failure.
        """
        try:
            rel = str(rel_posix).replace("\\", "/").lstrip("/")
            if not rel:
                return
            if not self._eligible(rel):
                return
            with self._lock:
                self._pending.add(rel)
            self._persist_queued.emit()
        except Exception as exc:                                  # noqa: BLE001
            log_debug(f"[auto-sync] notify_persist({rel_posix!r}) ignored: {exc}")

    # ----- eligibility -----------------------------------------------------

    def _eligible(self, rel: str) -> bool:
        """Cheap persist-time gate: auto-sync on + signed in + file is syncable.

        Returning False here is the intended "don't sync this write" outcome
        (feature off, guest, or a non-synced file) — NOT a swallowed failure, so
        it needs no §8 loud-raise. Any unexpected exception is treated as "skip"
        and logged at debug, because auto-sync is opportunistic: the Manual Push
        button remains the authoritative, fail-loud path.
        """
        try:
            if not Config.get_value(
                "Cloud", "auto_push", False, value_type=bool,
            ):
                return False
            from application.service.auth import get_auth_manager

            if not get_auth_manager().is_signed_in():
                return False
            return is_syncable_rel(rel)
        except Exception as exc:                                  # noqa: BLE001
            log_debug(f"[auto-sync] eligibility check failed for {rel!r}: {exc}")
            return False

    def _debounce_ms(self) -> int:
        raw = Config.get_value(
            "CloudSync", "auto_sync_debounce_ms", _DEFAULT_DEBOUNCE_MS,
            value_type=int,
        )
        try:
            ms = int(raw)
        except (TypeError, ValueError):
            ms = _DEFAULT_DEBOUNCE_MS
        # Clamp to a sane floor so a mis-set 0 can't busy-loop the queue.
        return max(200, ms)

    # ----- debounce + submit (own thread) ---------------------------------

    @pyqtSlot()
    def _arm_timer(self) -> None:
        self._timer.start(self._debounce_ms())

    @pyqtSlot()
    def _on_debounce_timeout(self) -> None:
        self._maybe_submit()

    def _maybe_submit(self) -> None:
        # Serialize: if a push is already running, re-arm so the newly-queued
        # files flush after it completes (also covered by _on_task_finished).
        if self._task_id is not None:
            self._timer.start(self._debounce_ms())
            return
        with self._lock:
            if not self._pending:
                return
            rels: List[str] = sorted(self._pending)
            self._pending.clear()
        try:
            from application.tools.auto_sync_task import AutoSyncPushTask

            self._task_id = get_tasks_manager().submit(AutoSyncPushTask(rels))
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[auto-sync] push submit failed: {exc}")
            # Requeue so the files aren't silently dropped (§8): they retry on
            # the next persist or the follow-up re-arm.
            with self._lock:
                self._pending.update(rels)
            self._task_id = None
            self._timer.start(self._debounce_ms())

    @pyqtSlot(str, bool, object)
    def _on_task_finished(self, task_id: str, _success: bool, _result: object) -> None:
        if task_id != self._task_id:
            return
        self._task_id = None
        # Drain anything that accumulated while the push ran.
        with self._lock:
            has_more = bool(self._pending)
        if has_more:
            self._arm_timer()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_instance: Optional[AutoSyncController] = None


def get_auto_sync_controller() -> AutoSyncController:
    """Return the process-wide AutoSyncController, lazy-constructed.

    Construct it once on the GUI thread at bootstrap (the wiring in
    ``main.py``) so its QTimer is owned by that thread; later ``notify_persist``
    calls from worker threads only touch the thread-safe pending set + queued
    signal.
    """
    global _instance
    if _instance is None:
        _instance = AutoSyncController()
    return _instance


__all__ = ["AutoSyncController", "get_auto_sync_controller"]
