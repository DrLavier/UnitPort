# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AiBuildRunTask — the Qt worker entry for an AI Build run (blueprint §4.B2).

Runs on a :class:`~unitport_sdk.Task` worker thread: builds the B3 client from
the saved settings, drives the :class:`Scheduler` over the marshalled
:class:`MutationSink` (every canvas write hops to the GUI thread inside the
sink), and — on a deterministic clean compile — commits the canvas to the
project via the sink. The scheduler's silent intent gate may resolve the run as
a direct answer instead (``RunResult.chat_only``): nothing is committed and the
save/commit branch below is naturally skipped (``should_commit`` is False). Fail-loud: an unconfigured API or a transport failure is
logged to the panel and re-raised so the task is reported as failed; a bad run
is never silently persisted (§8).

The task itself touches NO Qt widgets — all GUI work goes through the sink — so
the service layer stays Qt-free.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from unitport_sdk import Task, tr

from ..contracts import LlmTransportError, MutationSink, Requirements
from .session import BuildSession

__all__ = ["AiBuildRunTask"]


class AiBuildRunTask(Task):
    """Drive one orchestration run off the GUI thread."""

    def __init__(
        self,
        sink: MutationSink,
        prompt: str,
        *,
        max_repair_rounds: int = 3,
        auto_commit: bool = True,
        requirements: Optional[Requirements] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        super().__init__(name="AI Build")
        self._sink = sink
        self._prompt = prompt
        self._max_repair = max_repair_rounds
        self._auto_commit = auto_commit
        self._requirements = requirements
        # The thread's in-chat conversation so far (OpenAI-style entries from
        # conversation_store.to_llm_history), captured on the GUI thread at
        # submit time. Rides into the gate / direct answer / Understanding.
        self._history = list(history or [])
        # Set = paused. Toggled from the GUI thread (panel Pause button); read on
        # this worker thread at every phase/block boundary via ``_pause_gate``.
        self._paused = threading.Event()

    # -- pause control (called from the GUI thread) --------------------------

    def request_pause(self) -> None:
        self._paused.set()

    def request_resume(self) -> None:
        self._paused.clear()

    def _pause_gate(self) -> None:
        """Block while paused; cancellation (check_cancelled/sleep) unblocks."""
        while self._paused.is_set():
            self.check_cancelled()
            self.sleep(0.1)

    def run(self) -> Any:
        self._emit("canvas.ai_build.run_connecting", "[[info:Connecting to the model…]]")
        self._progress(0.05, "connecting")
        try:
            session = BuildSession(
                self._sink,
                on_log=self._sink.log_markup,
                max_repair_rounds=self._max_repair,
                requirements=self._requirements,
                pause_gate=self._pause_gate,
            )
        except (LlmTransportError, ValueError) as exc:
            self._emit(
                "canvas.ai_build.run_api_failed",
                "[[error:Model API call failed — {detail}]]",
                detail=str(exc),
            )
            raise

        self._progress(0.15, "running")
        try:
            result = session.run(self._prompt, history=self._history)
        except LlmTransportError as exc:
            self._emit(
                "canvas.ai_build.run_api_failed",
                "[[error:Model API call failed — {detail}]]",
                detail=str(exc),
            )
            raise
        finally:
            # Always clear node-edit highlights — clean run, error, or cancel.
            self._clear_highlights()

        if result.should_commit:
            if not self._auto_commit:
                # User turned auto-save off — the canvas is built but not
                # persisted; surface it loudly rather than pretending it saved.
                self._emit(
                    "canvas.ai_build.run_autosave_off",
                    "[[warning:Auto-save is off — the canvas was built but not "
                    "saved. Save it manually.]]",
                )
            else:
                self._progress(0.9, "saving")
                saved = self._commit()
                if saved:
                    self._emit("canvas.ai_build.run_saved",
                               "[[success:Canvas saved to the project.]]")
                else:
                    self._emit(
                        "canvas.ai_build.run_not_saved",
                        "[[warning:Canvas built but not auto-saved (no project "
                        "target) — save it manually.]]",
                    )
        self._progress(1.0, "done")
        return result

    # -- helpers -------------------------------------------------------------

    def _progress(self, fraction: float, text: str) -> None:
        """Drive both the task-slot progress and the overlay progress bar.

        ``set_progress`` updates the standard task UI; the sink hop drives the
        AI Build overlay's bar. The sink call is best-effort UI feedback — a
        minimal sink without it must not abort the run.
        """
        self.set_progress(fraction, text)
        sink_progress = getattr(self._sink, "set_progress", None)
        if callable(sink_progress):
            sink_progress(fraction, text)

    def _commit(self) -> bool:
        commit = getattr(self._sink, "commit", None)
        if not callable(commit):
            return False
        return bool(commit())

    def _clear_highlights(self) -> None:
        clear = getattr(self._sink, "clear_highlights", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # cosmetic cleanup must never mask the real outcome
                pass

    def _emit(self, key: str, default: str, **fields: Any) -> None:
        template = tr(key, default)
        try:
            rendered = template.format(**fields) if fields else template
        except (KeyError, IndexError):
            rendered = default.format(**fields) if fields else default
        self._sink.log_markup(rendered)
