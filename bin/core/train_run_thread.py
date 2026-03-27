#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrainRunThread 鈥?Circle 7 Phase D training worker.

Dispatches to either the real SB3 backend (default) or a lightweight mock
loop (set UNITPORT_TRAINING_MOCK=1 to force mock mode, e.g. for UI tests).

Signals
-------
progress(step, total, reward_mean, best_reward, status)
    Fired periodically during the run.
log_line(str)
    One text line to append to the training log.
finished(bundle_path)
    Fired on successful completion with the exported bundle path.
error(message)
    Fired when an unrecoverable error occurs.
cancelled()
    Fired when the thread is cancelled via cancel().
"""

from __future__ import annotations

import math
import os
import random
import threading
import uuid
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QThread, Signal
from utils.path_helper import get_project_root

if TYPE_CHECKING:
    from system.training.training_spec import TrainingJobSpec


class TrainRunThread(QThread):
    """Training worker thread for Circle 7 Phase D."""

    # Signal: step, total, reward_mean, best_reward, ep_len_mean, status_text
    progress = Signal(int, int, float, float, float, str)
    log_line = Signal(str)
    finished = Signal(str)   # bundle_path
    error = Signal(str)
    cancelled = Signal()
    # eval_completed(mean_reward, std_reward, success_rate, passed)
    eval_completed = Signal(float, float, float, bool)
    # vis check signals 鈥?emitted from training thread (cross-thread queued)
    vis_check_started = Signal(int)   # check_number (1-based)
    vis_check_ended   = Signal()

    # Default mock run parameters
    DEFAULT_TOTAL_STEPS = 100_000
    TICK_INTERVAL_MS = 80      # ms between progress ticks
    STEPS_PER_TICK = 2_000

    def __init__(
        self,
        policy_id_out: str = "trained_policy",
        total_timesteps: int = DEFAULT_TOTAL_STEPS,
        algorithm: str = "SAC",
        run_id: str = "",
        spec: Optional["TrainingJobSpec"] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._policy_id_out = policy_id_out
        self._total_timesteps = max(1, total_timesteps)
        self._algorithm = algorithm
        self._run_id = run_id
        self._spec = spec
        self._cancelled = False
        self._best_reward = -float("inf")
        self._vis_check_count = 0
        self._cache_run_id = str(run_id or f"run_{uuid.uuid4().hex[:8]}")
        self._last_step = 0
        self._last_total = self._total_timesteps
        self._last_reward_mean = 0.0
        self._last_ep_len_mean = 0.0
        self._last_status = ""

    # ------------------------------------------------------------------
    # Named constructor 鈥?build from a compiled TrainingJobSpec
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(
        cls,
        spec: "TrainingJobSpec",
        run_id: str = "",
        parent=None,
    ) -> "TrainRunThread":
        """
        Construct a TrainRunThread from a compiled TrainingJobSpec.

        Reads ``policy_id_out``, ``total_timesteps``, and ``algorithm``
        from ``spec.algorithm_config``.  Falls back to sensible defaults
        when fields are empty.

        Parameters
        ----------
        spec:
            Fully compiled TrainingJobSpec from TrainingSpecCompiler.
        run_id:
            Persistent run identifier for lifecycle tracking.
        parent:
            Optional QObject parent.
        """
        algo = spec.algorithm_config
        policy_id_out = (
            getattr(spec.export_config, "bundle_name", "") or
            algo.policy_id_out or
            f"{spec.policy_id}_trained"
        )
        return cls(
            policy_id_out=policy_id_out,
            total_timesteps=max(1, algo.total_timesteps),
            algorithm=algo.algorithm or "PPO",
            run_id=run_id,
            spec=spec,
            parent=parent,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation. The thread will stop at the next tick."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D102
        use_mock = (
            os.environ.get("UNITPORT_TRAINING_MOCK", "0") == "1"
            or self._spec is None
        )
        try:
            self._init_run_cache()
            if use_mock:
                self._run_mock()
            else:
                self._run_real()
        except Exception as exc:  # noqa: BLE001
            self._record_run_event("error", message=str(exc), status="error", finished=True)
            self.error.emit(str(exc))

    def _init_run_cache(self) -> None:
        from system.training.training_run_cache import get_training_run_cache

        spec = self._spec
        policy_id = getattr(spec, "policy_id", "") if spec is not None else ""
        algorithm = (
            getattr(getattr(spec, "algorithm_config", None), "algorithm", "")
            if spec is not None else self._algorithm
        ) or self._algorithm
        total_timesteps = (
            getattr(getattr(spec, "algorithm_config", None), "total_timesteps", 0)
            if spec is not None else self._total_timesteps
        ) or self._total_timesteps
        cache = get_training_run_cache()
        cache.ensure_run(
            self._cache_run_id,
            policy_id=policy_id,
            policy_id_out=self._policy_id_out,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            status="queued",
        )
        self._record_run_event("queued", policy_id_out=self._policy_id_out, status="queued")

    def _record_progress_sample(
        self,
        step: int,
        total: int,
        reward_mean: float,
        best_reward: float,
        ep_len_mean: float = 0.0,
        status: str = "",
    ) -> None:
        from system.training.training_run_cache import get_training_run_cache

        self._last_step = max(0, int(step or 0))
        self._last_total = max(0, int(total or 0))
        self._last_reward_mean = float(reward_mean or 0.0)
        self._best_reward = float(best_reward or 0.0)
        self._last_ep_len_mean = float(ep_len_mean or 0.0)
        self._last_status = str(status or "").strip()
        get_training_run_cache().record_progress(
            self._cache_run_id,
            step=self._last_step,
            total=self._last_total,
            reward_mean=self._last_reward_mean,
            best_reward=self._best_reward,
            ep_len_mean=self._last_ep_len_mean,
            status=self._last_status,
        )

    def _record_run_event(self, event_type: str, **payload) -> None:
        from system.training.training_run_cache import get_training_run_cache

        get_training_run_cache().record_event(self._cache_run_id, event_type, **payload)
        status = payload.get("status")
        if status:
            get_training_run_cache().update_status(
                self._cache_run_id,
                str(status),
                finished=bool(payload.get("finished")),
            )

    def _format_vis_milestone_delta(self, check_num: int) -> str:
        from system.training.training_run_cache import get_training_run_cache

        delta = get_training_run_cache().summarize_progress_delta(self._cache_run_id)
        self._record_run_event(
            "vis_check_milestone",
            check_num=int(check_num),
            step=delta.get("to_step", 0),
            delta=delta,
            status="vis_check",
        )
        return (
            f"[vis] Milestone #{check_num}: "
            f"step +{int(delta.get('step_delta', 0)):,} "
            f"({int(delta.get('from_step', 0)):,} -> {int(delta.get('to_step', 0)):,}), "
            f"reward_mean {float(delta.get('reward_mean_delta', 0.0)):+.3f} "
            f"-> {float(delta.get('latest_reward_mean', 0.0)):.3f}, "
            f"best {float(delta.get('best_reward_delta', 0.0)):+.3f} "
            f"-> {float(delta.get('latest_best_reward', 0.0)):.3f}"
        )

    # ------------------------------------------------------------------
    # Real SB3 backend — runs training in an isolated subprocess so the
    # Python GIL is never shared with the UI process.
    # ------------------------------------------------------------------

    def _run_real(self) -> None:
        import multiprocessing as mp
        import sys as _sys

        from system.training.training_process import (
            MSG_CANCELLED, MSG_ERROR, MSG_EVAL, MSG_FINISHED, MSG_LOG,
            MSG_PROGRESS, run_training_in_process,
        )

        spec = self._spec
        # Attach the run_id so the subprocess can embed it in the export.
        try:
            spec._run_id = self._run_id
        except Exception:
            pass

        # ── Spawn isolated training process ──────────────────────────────
        ctx = mp.get_context("spawn")
        msg_queue    = ctx.Queue(maxsize=512)
        cancel_event = ctx.Event()

        process = ctx.Process(
            target=run_training_in_process,
            args=(spec, msg_queue, cancel_event),
            kwargs={"extra_sys_path": list(_sys.path)},
            name="unitport_training",
            daemon=True,
        )
        process.start()

        # Raise OS priority of the training process (best-effort).
        self._boost_process_priority(process.pid)

        self._record_run_event("started", status="running", mode="real_process")
        self.log_line.emit(
            f"[training] Launched isolated training process (pid={process.pid})"
        )

        # ── Monitor loop — polls queue at ~50 ms ─────────────────────────
        _finished_bundle = ""
        try:
            while True:
                # Cancellation: signal the subprocess and wait for it.
                if self._cancelled:
                    cancel_event.set()
                    self.log_line.emit("[cancelled] Cancellation requested — stopping training process…")
                    process.join(timeout=8.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=3.0)
                    self._record_run_event("cancelled", status="cancelled", finished=True)
                    self.cancelled.emit()
                    return

                # Drain queue.
                try:
                    msg = msg_queue.get(timeout=0.05)
                except Exception:
                    # Timeout or Empty — check whether the process has exited
                    # without sending a terminal message (crash / OOM).
                    if not process.is_alive() and msg_queue.empty():
                        exit_code = process.exitcode
                        if exit_code != 0 and not _finished_bundle:
                            self.error.emit(
                                f"Training process exited unexpectedly (exit_code={exit_code})."
                            )
                        break
                    continue

                msg_type = msg.get("type")

                if msg_type == MSG_LOG:
                    self.log_line.emit(str(msg.get("data", "")))

                elif msg_type == MSG_PROGRESS:
                    step, total, reward_mean, best_reward, ep_len_mean, status = msg["data"]
                    self._record_progress_sample(
                        step, total, reward_mean, best_reward, ep_len_mean, status
                    )
                    self.progress.emit(
                        step, total,
                        float(reward_mean), float(best_reward),
                        float(ep_len_mean), str(status),
                    )

                elif msg_type == MSG_EVAL:
                    mean_r, std_r, success_rate, passed = msg["data"]
                    self._record_run_event(
                        "eval_completed", status="running",
                        result={"mean_reward": mean_r, "std_reward": std_r,
                                "success_rate": success_rate, "passed": passed},
                    )
                    self.eval_completed.emit(
                        float(mean_r), float(std_r), float(success_rate), bool(passed)
                    )

                elif msg_type == MSG_FINISHED:
                    data = msg.get("data", {})
                    _finished_bundle = str(data.get("bundle_path", ""))
                    total = self._total_timesteps
                    final_reward = self._last_reward_mean
                    final_best   = self._best_reward if self._best_reward != -float("inf") else final_reward
                    final_ep_len = self._last_ep_len_mean
                    self._record_progress_sample(
                        total, total, final_reward, final_best, final_ep_len, "Complete"
                    )
                    self._record_run_event(
                        "finished", status="finished", finished=True,
                        bundle_path=_finished_bundle,
                    )
                    self.progress.emit(total, total, final_reward, final_best, final_ep_len, "Complete")
                    self.finished.emit(_finished_bundle)
                    break

                elif msg_type == MSG_ERROR:
                    err_text = str(msg.get("data", "Unknown training error"))
                    self._record_run_event("error", message=err_text, status="error", finished=True)
                    self.error.emit(err_text)
                    break

                elif msg_type == MSG_CANCELLED:
                    self._record_run_event("cancelled", status="cancelled", finished=True)
                    self.cancelled.emit()
                    break

        finally:
            # Always clean up the process.
            if process.is_alive():
                cancel_event.set()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)

    @staticmethod
    def _boost_process_priority(pid: int) -> None:
        """Raise the training process to above-normal priority (best-effort)."""
        if pid is None:
            return
        try:
            if os.name == "nt":
                import ctypes
                ABOVE_NORMAL = 0x00008000
                handle = ctypes.windll.kernel32.OpenProcess(0x0200, False, pid)
                if handle:
                    ctypes.windll.kernel32.SetPriorityClass(handle, ABOVE_NORMAL)
                    ctypes.windll.kernel32.CloseHandle(handle)
            else:
                os.setpriority(os.PRIO_PROCESS, pid, -5)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mock loop (UI tests / UNITPORT_TRAINING_MOCK=1)
    # ------------------------------------------------------------------

    def _run_mock(self) -> None:
        total = self._total_timesteps
        algo = self._algorithm
        pid = self._policy_id_out

        self.log_line.emit(f"[{algo}] Starting mock training run: policy_id_out={pid}")
        self.log_line.emit(f"[{algo}] Total timesteps: {total:,}")
        self.log_line.emit(f"[{algo}] Initializing environment鈥?(mock)")
        self._record_run_event("started", status="running", mode="mock")

        step = 0
        reward_mean = -10.0
        best_reward = -10.0
        tick = 0

        while step < total:
            if self._cancelled:
                self.log_line.emit("[cancelled] Training run cancelled by user.")
                self._record_run_event("cancelled", status="cancelled", finished=True)
                self.cancelled.emit()
                return

            self.msleep(self.TICK_INTERVAL_MS)

            step = min(step + self.STEPS_PER_TICK, total)
            tick += 1

            # Simulate improving reward curve with noise
            t = step / total
            reward_mean = -10.0 + 15.0 * (1 - math.exp(-4 * t)) + random.gauss(0, 0.3)
            if reward_mean > best_reward:
                best_reward = reward_mean
            self._best_reward = best_reward

            status = f"{algo} step {step:,}"
            self._record_progress_sample(step, total, reward_mean, best_reward, 0.0, status)
            self.progress.emit(step, total, reward_mean, best_reward, 0.0, status)

        if self._cancelled:
            self.log_line.emit("[cancelled] Training run cancelled.")
            self._record_run_event("cancelled", status="cancelled", finished=True)
            self.cancelled.emit()
            return

        bundle_path = f"mock/custom_checkpoints/{pid}"
        self.log_line.emit(f"[{algo}] Training complete. Mock bundle: {bundle_path}")
        self._record_progress_sample(total, total, reward_mean, best_reward, 0.0, "Complete")
        self._record_run_event(
            "finished",
            status="finished",
            finished=True,
            bundle_path=bundle_path,
        )
        self.progress.emit(total, total, reward_mean, best_reward, 0.0, "Complete")
        self.finished.emit(bundle_path)
