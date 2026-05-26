# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacLabTrainingTask — TrainingTask wrapper around IsaacLabBackend.

Submitted via ``get_tasks_manager().submit(task)`` from
``ILPPOTrainerNode.execute``. Inside ``run()`` we drive the subprocess
synchronously on the SDK Task thread so:

* ``self.check_cancelled()`` cooperates with the subprocess tail loop
  (cancellation triggers ``backend.cancel()`` → ``taskkill /T``).
* ``self.progress_line(...)`` updates the UI as iteration messages arrive.
* ``self.log_*`` is auto-prefixed with ``[task.name]`` and routed to
  CmdLogWidget.
* The eventual return value (run_dir + checkpoint info) is delivered
  through ``TaskSignal.task_finished(task_id, success, result)``.

When training succeeds, ``run()`` finalizes the bundle via
:func:`application.training.isaac_lab.bundle_finalizer.finalize_isaac_lab_bundle`
into ``<project>/training/exported/isaac_lab/<bundle_name>/``. The
finalizer reads ``params/env.yaml`` + the latest ``model_*.pt`` from
the training run_dir and converts ``.pt`` → ONNX in-process via
``export_rsl_rl_actor_to_onnx`` (pure torch, no Isaac Sim subprocess).
The IL task mirrors SB3TrainingTask.run() finally block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from unitport_sdk import log_error, log_info, log_warning
from unitport_sdk import log_info as _raw_log_info

from application.service.metrics_cache import get_metrics_cache
from application.service.projects import ProjectInfo, current_project_info
from application.service.signals import current_backend, get_app_signals
from application.training._sdk.task_runner import TrainingTask
from application.training.backend import BACKEND_ISAAC_LAB

from .backend import (
    MSG_CANCELLED,
    MSG_ERROR,
    MSG_FINISHED,
    MSG_LOG,
    MSG_METRICS,
    MSG_PROGRESS,
    IsaacLabBackend,
)
from .config import IsaacLabConfig


class IsaacLabTrainingTask(TrainingTask):
    """Long-running task wrapping ``IsaacLabBackend.run_blocking``.

    Args:
        config:  IsaacLabConfig — should already have isaac_lab_path /
                 python / launcher resolved (use IsaacLabConfig.from_registers).
                 ``log_dir`` is overridden to the run dir.
        run_id:  Unique training run identifier.
        name:    Optional UI label; defaults to ``"isaac_lab:<task_name>"``.
    """

    def __init__(
        self,
        config: IsaacLabConfig,
        run_id: str,
        name: Optional[str] = None,
        *,
        spec: Any = None,
        project: Optional[ProjectInfo] = None,
    ) -> None:
        display = name or f"isaac_lab:{config.task_name}"

        # IsaacLabBackendAdapter.build_task hands us TrainingSpec.to_dict()
        # output (backend.py:246) so the same payload can also be JSON-shipped
        # to a subprocess. The Isaac Lab task runs in-process and consumes
        # spec.* via attribute access (env_cfg compile, bundle finalize).
        # Reconstruct the dataclass tree once so every downstream
        # getattr(spec, "X", ...) call sees real attributes instead of
        # silently returning the default for a dict.
        if isinstance(spec, dict):
            from application.training.training_spec import TrainingSpec
            spec = TrainingSpec.from_dict(spec)

        # Resolve run_dir from the active project BEFORE constructing the base
        # class. Training artifacts MUST land at
        # ``<project>/training/runs/<backend_id>/<run_id>/`` — unbound training
        # is rejected so nothing leaks into ``Paths.USER_CONFIG_DIR``
        # (RELEASE/CLAUDE.md §1.4). Backend id: AppSignals.current_backend()
        # (canvas-bound id) → fallback to this Task's backend constant.
        # Caller (IsaacLabBackendAdapter.build_task) may pass ProjectInfo
        # explicitly so the same ProjectInfo travels through bundle
        # finalization without a second current_project_info() call.
        backend_id = (current_backend() or BACKEND_ISAAC_LAB).strip()
        proj = project if project is not None else current_project_info()
        if proj is None:
            raise RuntimeError(
                "Isaac Lab training requires an open project — refusing to "
                "write outside the project tree."
            )
        run_dir = proj.path / "training" / "runs" / backend_id / str(run_id)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        super().__init__(display, run_id, run_dir)
        self.backend_id = backend_id
        self._proj: Optional[ProjectInfo] = proj
        self._spec: Any = spec
        config.log_dir = str(self.run_dir)
        config.run_id = run_id

        # Phase 4: compile the @configclass env_cfg from the canvas dict
        # stashed on spec.meta by ``submit_canvas_training``.  Lands as
        # ``<run_dir>/unitport_env_cfg.py``; ``IsaacLabConfig.build_command``
        # then forces ``--task UnitPort-Custom-v0`` + ``--unitport_config <path>``
        # so the launcher (il_train_launcher.py:376-434) loads
        # ``UnitPortEnvCfg`` + ``PPORunnerCfg`` and registers them under the
        # gym id.
        #
        # Both inputs are MANDATORY. If either is missing we raise rather
        # than degrade to Isaac Lab's stock task — the silent fallback
        # caused 2026-05-10's "RELEASE training broken" debug session
        # because runs without a canvas dict were quietly executing
        # ``Isaac-Velocity-Flat-Unitree-Go2-v0`` instead of the user's
        # canvas, and nobody knew until env.yaml was inspected.
        canvas_dict = self._extract_canvas_dict(spec)
        robot_ref = self._extract_robot_ref(spec)

        if canvas_dict is None:
            raise RuntimeError(
                "[IsaacLabTrainingTask] spec carries no "
                "``__canvas_dict__`` in meta — cannot compile env_cfg. "
                "RELEASE refuses to silently fall back to Isaac Lab's "
                "stock task. The legitimate path is the top Play button "
                "(``MainWindow._on_start_training`` → "
                "``submit_canvas_training``), which always stashes the "
                "canvas onto spec.meta. If you reached this point via "
                "``submit_il_trainer`` (canvas IR execution of an "
                "il_ppo_trainer node), that path is no longer a valid "
                "training entry — use the Play button instead."
            )
        if not config.unitport_launcher_path:
            raise RuntimeError(
                "[IsaacLabTrainingTask] config.unitport_launcher_path is "
                "empty — cannot launch through UnitPort's "
                "il_train_launcher.py. Stock train.py fallback was "
                "removed because it ignores --unitport_config. "
                "``IsaacLabConfig.from_registers`` auto-wires the "
                "in-tree ``application/training/isaac_lab/launcher/"
                "il_train_launcher.py`` — make sure construction goes "
                "through that classmethod."
            )

        try:
            from application.training.isaac_lab.env_cfg_compiler import (
                compile_env_cfg_to_file,
            )
            cfg_path = compile_env_cfg_to_file(
                canvas_dict,
                out_dir=Path(self.run_dir),
                robot=robot_ref,                      # Phase 5: IR-only translation
            )
            config.config_file = str(cfg_path)
            log_info(
                f"[isaac_lab_task] env_cfg compiled → {cfg_path.name} "
                f"({cfg_path.stat().st_size} bytes; robot="
                f"{robot_ref.sku if robot_ref is not None else 'NONE'!r})"
            )
        except Exception as exc:
            log_error(
                f"[isaac_lab_task] env_cfg compile failed: {exc!r} — "
                f"training aborted. Fix the canvas (commonly: scene_type "
                f"must be 'flat' or 'rough'; init_joint_angles keys must "
                f"be IR roles; base_height reward target must be ≥ "
                f"termination minimum) and re-run."
            )
            raise
        self.config = config

        # Resolve bundle naming from spec.export (Export node's
        # bundle_name / version / overwrite).  When spec is absent
        # (smoke / direct construction without the adapter) fall back to
        # run_id so finalize_isaac_lab_bundle still has a deterministic
        # bundle dir to land in.
        export_cfg = getattr(spec, "export", None) if spec is not None else None
        bn = (getattr(export_cfg, "bundle_name", "") or "").strip()
        if not bn or bn == "<NEW>":
            bn = f"{run_id}"
        self._bundle_name: str = bn
        self._bundle_version: str = str(
            getattr(export_cfg, "version", "v1") or "v1"
        )
        self._bundle_overwrite: bool = bool(
            getattr(export_cfg, "overwrite", True)
        ) if export_cfg is not None else True

        self._backend = IsaacLabBackend(config, run_id=run_id)
        self._backend.on_message = self._on_backend_message

        self._last_iter: int = 0
        self._last_total: int = 0
        self._last_reward: float = float("-inf")
        self._last_checkpoint: Optional[str] = None
        self._exported_dir: Optional[str] = None
        self._error: Optional[str] = None
        self._cancelled_externally: bool = False

        # Display label for the run-source dropdown — algorithm tag from
        # config (defaults to PPO) + short run_id suffix. Mirrors
        # SB3TrainingTask so both backends render identically in the UI.
        self._algo = (getattr(config, "algorithm", None) or "PPO").upper()
        self._run_label = f"{self._algo} #{str(run_id)[-8:]}"

    @staticmethod
    def _extract_robot_ref(spec: Any):
        """Pull a usable :class:`RobotSpecRef` off the spec for IR translation.

        Tolerates both shapes the spec may take through the call chain:
          - TrainingSpec instance (``spec.robot`` is already a RobotSpecRef)
          - Dict (TrainingSpec.to_dict() output) — reconstruct via
            :meth:`RobotSpecRef.from_dict`-style walk through the nested dict
        Returns None when the spec carries no robot info; callers decide what
        to do (Phase 5 env_cfg_compiler raises if a joint dict needs
        translation but robot is None).
        """
        if spec is None:
            return None
        from application.training.training_spec import RobotSpecRef
        # Direct attribute access first (TrainingSpec instance path)
        robot = getattr(spec, "robot", None)
        if isinstance(robot, RobotSpecRef):
            return robot
        # Dict-shape spec: reconstruct via the dataclass walker that
        # TrainingSpec.from_dict already uses.
        if isinstance(spec, dict):
            robot_dict = spec.get("robot")
            if isinstance(robot_dict, dict):
                from application.training.training_spec import _spec_from_dict
                return _spec_from_dict(RobotSpecRef, robot_dict)
        return None

    @staticmethod
    def _extract_canvas_dict(spec: Any) -> Optional[Dict[str, Any]]:
        """Pull the source canvas dict from spec.meta if present.

        ``submit_canvas_training`` stashes the raw canvas (CanvasPage.to_workflow_dict
        shape) here so the IsaacLab backend can rebuild the @configclass env_cfg
        in the run dir. Handles both forms ``spec`` may take in the call chain
        (TrainingSpec instance or its asdict).
        """
        if spec is None:
            return None
        meta: Any = None
        if isinstance(spec, dict):
            meta = spec.get("meta")
        else:
            meta = getattr(spec, "meta", None)
        if not isinstance(meta, dict):
            return None
        cd = meta.get("__canvas_dict__")
        return cd if isinstance(cd, dict) else None

    # ------------------------------------------------------------------
    # Backend → UI bridge
    # ------------------------------------------------------------------

    def _on_backend_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        data = msg.get("data")

        if mtype == MSG_LOG:
            # Subprocess passthrough: print raw lines (no [task.name] wrapper)
            # — Task.log_info would prefix every line with the long display
            # name (e.g. ``[isaac_lab:Isaac-Velocity-Flat-Unitree-Go2-v0]``),
            # which is pure noise on top of rsl_rl's already-formatted output.
            # WHY KEPT (Rule 1.b) try/except around the raw-log call: this
            # is in the backend message bus and a logging-system fault
            # must not kill the training subprocess tail loop.
            try:
                _raw_log_info(str(data))
            except Exception:
                pass
            return

        if mtype == MSG_PROGRESS:
            try:
                step, total, reward, best_reward, ep_len, _phase = data
            except (TypeError, ValueError):
                return
            self._last_iter = int(step)
            self._last_total = int(total)
            if isinstance(best_reward, (int, float)):
                self._last_reward = float(best_reward)
            ratio = (step / total) if total > 0 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            try:
                self.progress_line(
                    ratio,
                    f"iter {step}/{total}  reward={best_reward:.3f}",
                )
            except Exception:
                pass
            # NOTE: do NOT push this to the chart. IsaacLabBackend emits
            # MSG_PROGRESS using ``best_reward`` (running max) in the
            # reward slot AND a separate MSG_METRICS each iter carrying
            # the real ``reward_mean``. Recording both produced two points
            # per step (mean vs running-max), which the chart connected as
            # a zig-zag back to ``best_reward`` between every real sample.
            # MSG_METRICS alone gives a clean per-iter curve.
            return

        if mtype == MSG_METRICS:
            # rsl_rl multi-line block parsed by IsaacLabBackend into a
            # rich dict — this is the ONE chart-data source for the
            # Isaac Lab path. Forward verbatim so reward_mean / reward_min /
            # reward_max / policy_loss / value_loss / kl / entropy /
            # ep_len_mean / amp_* all appear as their own series.
            if isinstance(data, dict):
                try:
                    rew = float(data.get("reward_mean", 0.0))
                    if rew > self._last_reward:
                        self._last_reward = rew
                except Exception:
                    pass
                sample = dict(data)
                sample.setdefault("step", self._last_iter)
                sample.setdefault("total_timesteps", self._last_total)
                sample.setdefault("algorithm", self._algo)
                self._record_sample(sample)
            return

        if mtype == MSG_FINISHED:
            try:
                self._last_checkpoint = str(data.get("last_checkpoint_path") or
                                            data.get("artifact_path") or "")
                # Phase 3: backend now also reports an ``exported_dir``
                # — Isaac Lab's exported/ subdir produced by the export
                # subprocess. bundle_finalizer scans this dir for
                # policy.onnx + env.yaml in run() finally.
                exp = str(data.get("exported_dir") or "")
                if exp:
                    self._exported_dir = exp
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
    # Metrics fanout — MetricsCache + AppSignals (mirror of SB3 path)
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
        """Drive the Isaac Lab subprocess to completion (or cancellation)."""
        self.info(f"Isaac Lab run starting in {self.run_dir}")
        self.info(f"task={self.config.task_name} num_envs={self.config.num_envs} "
                  f"max_iter={self.config.max_iterations} headless={self.config.headless}")

        # Mark the run live so the chart's data-source dropdown picks it
        # up immediately (even before the first MSG_PROGRESS arrives).
        try:
            get_metrics_cache().mark_started(self.run_id, self._run_label)
            get_app_signals().training_run_started.emit(self.run_id, self._run_label)
        except Exception:
            pass

        # ExportNode.overwrite contract: when the canvas Export Node has
        # overwrite=True, the user has explicitly opted in to replacing
        # the prior bundle with the same name. Wipe it NOW (at training
        # start) rather than at the final atomic swap — that way a failed
        # training (export crash, cancel, finalize error) can never leave
        # the previous bundle on disk pretending to be the new result,
        # which would silently re-introduce the SKU-mismatch chain
        # PolicyRunner.load is hardened against. No-op when bundle_name
        # is the run_id fallback (always unique → nothing to remove).
        if self._bundle_overwrite and self._proj is not None:
            try:
                from application.training.bundle_exporter import BundleExporter
                wiped = BundleExporter.wipe_bundle_dir(
                    project=self._proj,
                    backend_id="isaac_lab",
                    bundle_name=self._bundle_name,
                )
                if wiped:
                    self.info(
                        f"[isaac_lab_task] overwrite=True → wiped existing "
                        f"bundle '{self._bundle_name}' before training"
                    )
            except Exception as exc:
                # Wipe failure is non-fatal — finalize's atomic swap will
                # still try to replace at the end. Surface the warning so
                # the user knows mid-training cancel could leave stale state.
                log_warning(
                    f"[isaac_lab_task] pre-training bundle wipe failed "
                    f"({exc}); a failed training may leave the previous "
                    f"bundle on disk."
                )

        train_ok = False
        bundle_path: Optional[str] = None
        try:
            self._backend.run_blocking()

            if self._error:
                raise RuntimeError(self._error)
            if self._cancelled_externally:
                self.check_cancelled()  # raises if SDK side also flagged cancel

            train_ok = True
            return {
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "checkpoints_dir": str(self.run_dir / "checkpoints"),
                "last_iter": self._last_iter,
                "total_iter": self._last_total,
                "best_reward": self._last_reward if self._last_reward != float("-inf") else None,
                "last_checkpoint": self._last_checkpoint,
                "exported_dir": self._exported_dir or "",
                "bundle_path": bundle_path or "",
            }
        finally:
            # Bundle finalize: training succeeded AND we have a bound
            # project. ``_exported_dir`` is set by IsaacLabBackend on
            # MSG_FINISHED to ``config.log_dir`` (= run_dir), which now
            # always exists when training completes — the legacy
            # ``_run_export`` Isaac-Sim-subprocess path is gone (CLAUDE.md
            # §1.8), so the only way ``_exported_dir`` can be empty is
            # training crash before MSG_FINISHED, in which case
            # ``train_ok`` is False and we skip below.
            finalize_ok = False
            finalize_exc: Optional[BaseException] = None
            if train_ok and self._proj is not None:
                if not self._exported_dir:
                    raise RuntimeError(
                        "[isaac_lab_task] train_ok but no exported_dir from "
                        "backend MSG_FINISHED — backend contract violation"
                    )
                from application.training.isaac_lab.bundle_finalizer import (
                    finalize_isaac_lab_bundle,
                )
                is_amp = self._algo.upper() == "AMP_PPO"
                # CLAUDE.md §1.8: finalize MUST NOT be silently swallowed.
                # Catch only to interleave the training_run_finished signal
                # (so the UI's chart panel stops streaming and the user sees
                # the failure state) BEFORE re-raising so the SDK's
                # task_finished fires with success=False. No "warn and
                # continue" — the exception always propagates.
                try:
                    out = finalize_isaac_lab_bundle(
                        exported_dir=Path(self._exported_dir),
                        bundle_name=self._bundle_name,
                        version=self._bundle_version,
                        overwrite=self._bundle_overwrite,
                        project=self._proj,
                        spec=self._spec,
                        run_id=self.run_id,
                        is_amp=is_amp,
                    )
                    bundle_path = str(out.bundle_path)
                    finalize_ok = True
                    log_info(
                        f"[isaac_lab_task] bundle finalized → {bundle_path}"
                    )
                except BaseException as exc:
                    finalize_exc = exc
                    log_error(
                        f"[isaac_lab_task] bundle finalize failed: {exc}"
                    )

            pipeline_ok = train_ok and (finalize_ok or self._proj is None)
            try:
                get_metrics_cache().mark_finished(self.run_id, pipeline_ok)
                get_app_signals().training_run_finished.emit(self.run_id, pipeline_ok)
            except Exception:
                pass

            if finalize_exc is not None:
                raise finalize_exc


__all__ = ["IsaacLabTrainingTask"]
