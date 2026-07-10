# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BridgeMutationSink — marshal worker-thread AI Build mutations to the GUI.

The orchestration harness runs on a worker thread (network I/O off the GUI
thread), but :class:`~application.ui.canvas.agent_bridge.CanvasAgentBridge` and
the canvas/panel widgets are Qt objects that MUST be touched only on the GUI
thread (the bridge docstring is explicit). This object is the single seam that
bridges the two: it implements the service-side ``MutationSink`` Protocol and,
for every call, marshals a closure onto the GUI thread via a queued signal +
``QMutex``/``QWaitCondition``, blocking the worker until the GUI slot has run it
— the exact pattern of ``Task._request_input`` (``unitport_sdk/sys.py``).

This is the ONLY place Qt is touched on the orchestration path; the service
layer stays Qt-free and headless-testable.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QMutex, QObject, QThread, QWaitCondition, pyqtSignal

from unitport_sdk import log_warning

__all__ = ["BridgeMutationSink"]


class BridgeMutationSink(QObject):
    """A GUI-thread-affined MutationSink over a live CanvasPage + bridge."""

    # (request_id, zero-arg callable to run on the GUI thread)
    _request = pyqtSignal(str, object)

    def __init__(self, page: Any) -> None:
        super().__init__()  # affined to the creating (GUI) thread
        self._page = page
        self._bridge = page.ai_agent_bridge
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._results: Dict[str, tuple] = {}  # rid -> (ok, value_or_exc)
        self._cancelled = False
        # AutoConnection → queued when emitted from the worker thread.
        self._request.connect(self._on_request)

    # =====================================================================
    # MutationSink protocol (called on the WORKER thread)
    # =====================================================================

    def place_node(
        self, schema_id, *, params=None, x=0.0, y=0.0, instance_id=None
    ) -> Optional[str]:
        return self._invoke(
            lambda: self._bridge.place_node(
                schema_id, params=params, x=x, y=y, instance_id=instance_id
            )
        )

    def remove_node(self, instance_id) -> bool:
        return self._invoke(lambda: self._bridge.remove_node(instance_id))

    def connect(self, src_node, src_port, dst_node, dst_port) -> Optional[str]:
        return self._invoke(
            lambda: self._bridge.connect(src_node, src_port, dst_node, dst_port)
        )

    def disconnect(self, edge_id) -> bool:
        return self._invoke(lambda: self._bridge.disconnect(edge_id))

    def set_param(self, instance_id, key, value) -> bool:
        return self._invoke(lambda: self._bridge.set_param(instance_id, key, value))

    def snapshot(self) -> Dict[str, Any]:
        return self._invoke(self._page.to_workflow_dict)

    def log_markup(self, markup: str) -> None:
        self._invoke(lambda: self._bridge.log(markup))

    def set_progress(self, fraction: float, text: str = "") -> None:
        # Marshal the progress update to the GUI thread → the page forwards it
        # to the AI Build overlay's progress bar. Non-authoritative UI feedback.
        self._invoke(lambda: self._report_progress(fraction, text))

    def _report_progress(self, fraction: float, text: str) -> None:
        fn = getattr(self._page, "ai_build_report_progress", None)
        if callable(fn):
            fn(fraction, text)

    # -- structured-phase / token / highlight feedback (AI Build v2) ---------

    def begin_phase(self, phase_id: str, title: str) -> None:
        self._invoke(lambda: self._begin_phase(phase_id, title))

    def _begin_phase(self, phase_id: str, title: str) -> None:
        fn = getattr(self._page, "ai_build_begin_phase", None)
        if callable(fn):
            fn(phase_id, title)

    def set_phase_tokens(self, phase_total: int, cumulative: int) -> None:
        self._invoke(lambda: self._set_phase_tokens(phase_total, cumulative))

    def _set_phase_tokens(self, phase_total: int, cumulative: int) -> None:
        fn = getattr(self._page, "ai_build_set_phase_tokens", None)
        if callable(fn):
            fn(phase_total, cumulative)

    def notify_build_engaged(self) -> None:
        # The silent intent gate admitted the prompt as a real build run;
        # forward to the page so build-run affordances (checkpoint notice)
        # appear only now — never for a gate-declined direct answer.
        self._invoke(self._notify_build_engaged)

    def _notify_build_engaged(self) -> None:
        fn = getattr(self._page, "ai_build_on_engaged", None)
        if callable(fn):
            fn()

    def highlight_block(self, schema_ids, on: bool) -> None:
        self._invoke(lambda: self._bridge.highlight_block(schema_ids, on))

    def highlight_node(self, instance_id: str, on: bool) -> None:
        self._invoke(lambda: self._bridge.highlight_node(instance_id, on))

    def clear_highlights(self) -> None:
        self._invoke(self._bridge.clear_highlights)

    def apply_layout(self, plan) -> None:
        # Marshal the whole tidy-layout (move nodes + rebuild region boxes) as a
        # single GUI-thread closure — one hop, not one per node.
        self._invoke(lambda: self._bridge.apply_layout(plan))

    # =====================================================================
    # commit / cancel (harness-facing, also marshalled)
    # =====================================================================

    def commit(self) -> Optional[Any]:
        """Persist the canvas to the page's configured save target (GUI thread).

        Returns the saved path, or None when no project target is set (the user
        must save manually — surfaced as a warning, not a silent success)."""
        return self._invoke(self._commit_on_gui)

    def _commit_on_gui(self) -> Optional[Any]:
        target = getattr(self._page, "commit_to_save_target", None)
        if callable(target):
            return target()
        log_warning("[ai.marshaller] page has no commit_to_save_target; not saved")
        return None

    def cancel(self) -> None:
        """Unblock any worker waiting on a marshalled call (task cancelled)."""
        self._mutex.lock()
        self._cancelled = True
        self._cond.wakeAll()
        self._mutex.unlock()

    # =====================================================================
    # marshalling core
    # =====================================================================

    def _invoke(self, fn: Callable[[], Any]) -> Any:
        # Fast path: already on the GUI thread (direct calls in tests / reentry)
        # — run inline, never marshal (marshalling on the GUI thread would
        # deadlock, since the event loop that runs the slot is this thread).
        if QThread.currentThread() is self.thread():
            return fn()

        rid = str(uuid.uuid4())
        self._request.emit(rid, fn)

        self._mutex.lock()
        try:
            while rid not in self._results and not self._cancelled:
                self._cond.wait(self._mutex, 100)
            if rid not in self._results:
                raise RuntimeError("AI Build run cancelled while applying a mutation")
            ok, value = self._results.pop(rid)
        finally:
            self._mutex.unlock()

        if not ok:
            raise value  # fail-loud: propagate the GUI-side exception to the worker
        return value

    def _on_request(self, rid: str, fn: Callable[[], Any]) -> None:
        """Runs on the GUI thread (queued slot). Execute fn, store the result."""
        try:
            value = fn()
            ok = True
        except Exception as exc:  # capture and re-raise on the worker (§8)
            value = exc
            ok = False
        self._mutex.lock()
        self._results[rid] = (ok, value)
        self._cond.wakeAll()
        self._mutex.unlock()
