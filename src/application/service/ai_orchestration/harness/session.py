# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""B1-facing session seam (blueprint §4.B1 boundary).

A thin facade that the GUI runner (``run_task``) drives: build the B3 client from
the saved :class:`ApiConfig`, wire a :class:`Scheduler` over a
:class:`MutationSink`, run a prompt, and return the :class:`RunResult`. It keeps
provider construction + scheduler wiring in one place so the worker entry stays
small. Every run opens with the scheduler's silent intent gate — a prompt that
is not a canvas build/modify request is answered directly from the model's own
knowledge (``RunResult.chat_only``) instead of entering orchestration.
Fail-loud: an unconfigured API or a transport error raises (the runner
reports it; never a degraded canvas, §8).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..contracts import LlmClient, MutationSink, Requirements
from .scheduler import RunResult, Scheduler

__all__ = ["BuildSession"]

LogFn = Callable[[str], None]
PauseGate = Callable[[], None]


class BuildSession:
    """One AI-Build run over a live canvas sink."""

    def __init__(
        self,
        sink: MutationSink,
        *,
        llm: Optional[LlmClient] = None,
        on_log: Optional[LogFn] = None,
        max_repair_rounds: int = 3,
        requirements: Optional[Requirements] = None,
        pause_gate: Optional[PauseGate] = None,
    ) -> None:
        if llm is None:
            # Build the OpenAI-compatible client from the saved settings.
            from ..provider.openai_compat import build_client_from_config

            llm = build_client_from_config()
        self._scheduler = Scheduler(
            sink,
            llm,
            max_repair_rounds=max_repair_rounds,
            on_log=on_log,
            requirements=requirements,
            pause_gate=pause_gate,
        )

    def run(
        self,
        prompt: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RunResult:
        """Run the orchestration; raises LlmTransportError on a provider failure.

        ``history`` is the thread's in-chat conversation so far (from
        ``conversation_store.to_llm_history``); None/empty = first message.
        """
        if not str(prompt or "").strip():
            raise ValueError("AI Build: prompt is empty (§8)")
        return self._scheduler.run(prompt, history=history)
