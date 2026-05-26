# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""SB3TrainingTask — SDK Task wrapper around :class:`SB3SubprocessBackend`.

Submitted via ``get_tasks_manager().submit(task)`` from
:func:`application.training.trainer_runtime.submit_sb3_trainer` (Stage 4).
Inside ``run()`` we drive the subprocess synchronously on the SDK Task
thread so:

  * ``self.check_cancelled()`` cooperates with the subprocess tail loop
    (cancellation triggers ``backend.cancel()`` → ``taskkill /T``).
  * ``self.progress_line(...)`` updates the UI as MSG_PROGRESS arrives.
  * ``self.log_*`` is auto-prefixed with ``[task.name]`` and routed to
    ``CmdLogWidget``.
  * The eventual return value (run_dir + bundle path) is delivered
    through ``TaskSignal.task_finished(task_id, success, result)``.

Mirrors :class:`IsaacLabTrainingTask` so the UI's task panel can render
both backends with a single code path.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from unitport_sdk import log_info as _raw_log_info
from unitport_sdk import log_warning

from application.service.metrics_cache import get_metrics_cache
from application.service.projects import current_project_info
from application.service.signals import current_backend, get_app_signals
from application.training._sdk.task_runner import TrainingTask
from application.training.backend import BACKEND_SB3_MUJOCO
from application.training.launcher.sb3_subprocess import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_FINISHED,
    MSG_LOG,
    MSG_METRICS,
    MSG_PROGRESS,
    SB3SubprocessBackend,
)


class SB3TrainingTask(TrainingTask):
    """Long-running task wrapping :class:`SB3SubprocessBackend.run_blocking`.

    Args:
        spec: payload dict for the child process. Schema:
            ``{"spec": <TrainingSpec.to_dict()>, "run_id": str,
              "total_timesteps": int|None, "export_bundle": bool}``
        run_id: unique training run identifier.
        name: optional UI label; defaults to ``"sb3_mujoco:<algorithm>"``.
        python: Python interpreter for the child (defaults to
            ``sys.executable``).
    """

    def __init__(
        self,
        spec: Dict[str, Any],
        run_id: str,
        *,
        name: Optional[str] = None,
        python: Optional[str] = None,
    ) -> None:
        algo = "PPO"
        try:
            algo = (spec.get("spec", {}).get("algorithm", {}).get("algorithm")
                    or "PPO").upper()
        except Exception:
            pass
        display = name or f"sb3_mujoco:{algo}"

        # Resolve run_dir from the active project BEFORE constructing the base
        # class. Training artifacts MUST land at
        # ``<project>/training/runs/<backend_id>/<run_id>/`` — unbound training
        # is rejected so nothing leaks into ``Paths.USER_CONFIG_DIR``
        # (RELEASE/CLAUDE.md §1.4). Backend id: AppSignals.current_backend()
        # (the canvas-bound id) → fallback to this Task's backend constant.
        # No index file is written — History dropdown is a directory scan of
        # <project>/training/runs/<backend>/; the dir IS the source of truth.
        backend_id = (current_backend() or BACKEND_SB3_MUJOCO).strip()
        proj = current_project_info()
        if proj is None:
            raise RuntimeError(
                "SB3 training requires an open project — refusing to write "
                "outside the project tree."
            )
        from pathlib import Path as _P
        run_dir = _P(proj.path) / "training" / "runs" / backend_id / str(run_id)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        super().__init__(display, run_id, run_dir)
        self.backend_id = backend_id

        # Inject run_id into the payload so the child uses the same one.
        # Pass project_path + backend_id so the child's BundleExporter can
        # route the exported bundle into
        # <project>/training/exported/<backend_id>/<name>/.
        payload = dict(spec or {})
        payload.setdefault("run_id", run_id)
        payload["backend_id"] = backend_id
        payload["project_path"] = str(proj.path)

        self._backend = SB3SubprocessBackend(payload, run_id=run_id, python=python)
        self._backend.on_message = self._on_backend_message

        # Cache project + Export-Node intent so the run() head can wipe
        # the prior bundle when overwrite=True (see comment in run()).
        # Default overwrite=True matches the canvas Export Node default
        # (spec_compiler line ~1369: ``overwrite=_as_bool(..., True)``).
        self._proj = proj
        export_cfg = (
            (spec or {}).get("spec", {}).get("export") or {}
            if isinstance(spec, dict) else {}
        )
        self._bundle_name: str = str(
            (export_cfg.get("bundle_name") if isinstance(export_cfg, dict) else "")
            or ""
        ).strip()
        self._bundle_overwrite: bool = bool(
            export_cfg.get("overwrite", True)
            if isinstance(export_cfg, dict) else True
        )

        self._last_step: int = 0
        self._last_total: int = 0
        self._last_reward: float = float("-inf")
        self._bundle_info: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._cancelled_externally: bool = False

        # Display label for the run-source dropdown (algo + short run_id).
        # MetricsCache.mark_started keys off this; AppSignals.training_run_started
        # emits the same string so RunSourceSelector can render without a
        # second cache lookup.
        self._algo = algo
        self._run_label = f"{algo} #{str(run_id)[-8:]}"

    # ------------------------------------------------------------------
    # Backend → UI bridge
    # ------------------------------------------------------------------

    def _on_backend_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        data = msg.get("data")

        if mtype == MSG_LOG:
            # Subprocess passthrough — print raw lines without the
            # [task.name] wrapper Task.log_info would inject.
            try:
                _raw_log_info(str(data))
            except Exception:
                pass
            return

        if mtype == MSG_PROGRESS:
            try:
                step, total, reward, _best, _ep, _phase = data
            except (TypeError, ValueError):
                return
            self._last_step = int(step)
            self._last_total = int(total)
            if isinstance(reward, (int, float)) and reward > self._last_reward:
                self._last_reward = float(reward)
            ratio = (step / total) if total > 0 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            try:
                self.progress_line(
                    ratio,
                    f"step {step}/{total}  reward={reward:.3f}",
                )
            except Exception:
                pass
            # Synthesise a per-rollout sample for the chart panel. SB3's
            # entry script only emits MSG_PROGRESS today (not MSG_METRICS)
            # — without this bridge the chart would never receive data on
            # the SB3 path.
            self._record_sample({
                "step": int(step),
                "total_timesteps": int(total),
                "reward_mean": float(reward) if isinstance(reward, (int, float)) else 0.0,
                "algorithm": self._algo,
            })
            return

        if mtype == MSG_METRICS:
            if isinstance(data, dict):
                try:
                    rew = float(data.get("reward_mean", 0.0))
                    if rew > self._last_reward:
                        self._last_reward = rew
                except Exception:
                    pass
                # Forward the trainer's full metrics dict — let the chart
                # decide which keys to render. Inject the latest known step
                # if the trainer didn't include one (chart needs an X value).
                sample = dict(data)
                sample.setdefault("step", self._last_step)
                sample.setdefault("total_timesteps", self._last_total)
                sample.setdefault("algorithm", self._algo)
                self._record_sample(sample)
            return

        if mtype == MSG_FINISHED:
            if isinstance(data, dict):
                self._bundle_info = dict(data)
                try:
                    rew = float(data.get("best_reward", float("-inf")))
                    if rew > self._last_reward:
                        self._last_reward = rew
                except Exception:
                    pass
            return

        if mtype == MSG_ERROR:
            self._error = str(data)
            return

        if mtype == MSG_CANCELLED:
            self._cancelled_externally = True
            return

    # ------------------------------------------------------------------
    # Metrics fanout — MetricsCache + AppSignals
    # ------------------------------------------------------------------

    def _record_sample(self, sample: Dict[str, Any]) -> None:
        """Write to MetricsCache + emit on AppSignals.

        Called from the SDK Task thread (subprocess tail loop). Both sinks
        are thread-safe: MetricsCache uses an internal lock, and pyqtSignal
        marshals to the main thread via QueuedConnection.

        Augments the sample with the running-max ``best_reward`` so the
        chart panel and Mission Control status both see a monotonic series
        without each having to recompute it from ``reward_mean``.
        """
        enriched = dict(sample)
        if self._last_reward != float("-inf") and "best_reward" not in enriched:
            enriched["best_reward"] = float(self._last_reward)
        try:
            get_metrics_cache().record(self.run_id, enriched)
        except Exception:
            pass
        try:
            get_app_signals().training_metrics.emit(self.run_id, dict(enriched))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Task contract
    # ------------------------------------------------------------------

    def cancel(self) -> None:  # type: ignore[override]
        """Cooperative cancel — also kills the subprocess tree so the
        Task thread's tail loop unblocks immediately.
        """
        try:
            self._backend.cancel()
        finally:
            super().cancel()

    def run(self) -> Dict[str, Any]:
        """Drive the SB3 subprocess to completion (or cancellation)."""
        self.info(f"SB3 run starting in {self.run_dir}")

        # Mark the run live so the chart's data-source dropdown picks it
        # up immediately (even before the first MSG_PROGRESS arrives).
        try:
            get_metrics_cache().mark_started(self.run_id, self._run_label)
            get_app_signals().training_run_started.emit(self.run_id, self._run_label)
        except Exception:
            pass

        # ExportNode.overwrite contract: when the canvas Export Node has
        # overwrite=True, the prior bundle with the same name must be
        # gone the moment this training starts — so a failed run (cancel,
        # subprocess crash) cannot leave a stale bundle masquerading as
        # the new result. Mirror of IsaacLabTrainingTask.run() head.
        if self._bundle_overwrite and self._proj is not None and self._bundle_name:
            try:
                from application.training.bundle_exporter import BundleExporter
                wiped = BundleExporter.wipe_bundle_dir(
                    project=self._proj,
                    backend_id=self.backend_id,
                    bundle_name=self._bundle_name,
                )
                if wiped:
                    self.info(
                        f"[sb3_task] overwrite=True → wiped existing "
                        f"bundle '{self._bundle_name}' before training"
                    )
            except Exception as exc:
                log_warning(
                    f"[sb3_task] pre-training bundle wipe failed ({exc}); "
                    f"a failed training may leave the previous bundle "
                    f"on disk."
                )

        success = False
        try:
            self._backend.run_blocking()

            if self._error:
                raise RuntimeError(self._error)
            if self._cancelled_externally:
                self.check_cancelled()  # raises if SDK side also flagged cancel

            success = True
            return {
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "checkpoints_dir": str(self.run_dir / "checkpoints"),
                "last_step": self._last_step,
                "total_steps": self._last_total,
                "best_reward": (
                    self._last_reward if self._last_reward != float("-inf") else None
                ),
                "bundle": self._bundle_info,
                "exit_code": self._backend.exit_code,
            }
        finally:
            try:
                get_metrics_cache().mark_finished(self.run_id, success)
                get_app_signals().training_run_finished.emit(self.run_id, success)
            except Exception:
                pass


__all__ = ["SB3TrainingTask"]
