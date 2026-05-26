# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.launcher — SB3 subprocess launcher (Stage 10).

Wraps :func:`application.training.sb3_trainer.train_sb3` in a
``subprocess.Popen``-managed child process so:

  * The Studio UI thread never freezes during ``model.learn``.
  * Cancellation routes through Windows ``taskkill /F /T /PID`` /
    POSIX ``killpg(SIGTERM)`` for prompt teardown.
  * Per-run isolation: each task has its own run_id, run_dir, and OS
    process — multiple submissions don't share state.

Public surface:
    * :class:`SB3SubprocessBackend` — Popen orchestrator (mirrors
      IsaacLabBackend's MSG_* line protocol).
    * :class:`SB3TrainingTask`     — SDK :class:`TrainingTask` wrapper
      submitted via ``get_tasks_manager().submit(task)``.
    * Entry script :mod:`sb3_entry` — what the child process runs.
"""
from __future__ import annotations

from application.training.launcher.sb3_subprocess import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_FINISHED,
    MSG_LOG,
    MSG_METRICS,
    MSG_PROGRESS,
    SB3SubprocessBackend,
)
from application.training.launcher.sb3_task import SB3TrainingTask

__all__ = [
    "SB3SubprocessBackend",
    "SB3TrainingTask",
    "MSG_LOG",
    "MSG_PROGRESS",
    "MSG_METRICS",
    "MSG_FINISHED",
    "MSG_ERROR",
    "MSG_CANCELLED",
]
