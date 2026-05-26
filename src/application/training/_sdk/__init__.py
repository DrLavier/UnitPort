# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training._sdk — internal SDK adapter shim for the training stack.

This subpackage is **not a node** and is **not user-facing**. It wraps
``unitport_sdk`` primitives (Task / DataManager / Paths / Storage / log_*) into
the patterns the training/policy/sim2sim layers reach for everywhere:

- ``task_runner.TrainingTask``  — base ``Task`` for long-running training jobs;
  enforces ``self.check_cancelled()`` + ``self.sleep()`` discipline. Training
  artifacts MUST live under ``<project>/training/...``; unbound training is
  rejected at submit time (RELEASE/CLAUDE.md §1.4).
- ``checkpoint_io``             — atomic file IO around ``DataManager`` and
  ``Storage`` for ``.pt`` / ``.zip`` / ``.onnx`` / ``manifest.yaml``.
- ``logging_bridge``            — ``[run_id]``-prefixed ``log_*`` helpers so
  CmdLogWidget can filter by run.

Phase 2 (Isaac Lab subprocess), Phase 3 (sim2sim playback) and Phase 4
(SB3 trainer) all sit on top of this shim.
"""

from __future__ import annotations
