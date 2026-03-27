from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .action_applier import ActionApplier
from .bundle_loader import BundleLoader
from .compatibility_checker import CompatibilityChecker, CompatReport, CompatStatus
from .inference_engine import InferenceEngine, JITEngine, ONNXEngine
from .manifest_schema import CheckpointBundle
from .normalizer import NormalizationStats, Normalizer
from .obs_builder import ObsBuilder
from .sim_env_context import SimEnvContext


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class IncompatibleWeightError(RuntimeError):
    """Raised when a bundle is incompatible with the current environment."""


@dataclass
class EpisodeResult:
    success: bool
    steps_run: int
    terminated: bool
    compat_status: CompatStatus
    last_action: Optional[np.ndarray]
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 2 fields 閳?present with defaults so Phase 1 consumers are unaffected
    engine_type: str = ""           # "onnx" | "jit"
    termination_reason: str = ""    # "completed" | "safety_stop" | "env_terminated" | "error"
    safety_status: str = ""         # "ok" | "safety_stop" | ""
    telemetry_emitted: bool = False


# ---------------------------------------------------------------------------
# PolicyRunner
# ---------------------------------------------------------------------------

class PolicyRunner:
    """Assembles Circle 3/4 components into an executable policy pipeline.

    Usage::

        runner = PolicyRunner()
        report = runner.load(bundle_path, env)   # setup + compat check
        result = runner.run_episode(env)          # decimation loop

    Control write contract:
      - run_episode() writes actions to env.mj_data.ctrl if available.
      - If ctrl is shorter than the action, only the matching prefix is written;
        if the action is longer than ctrl, a ValueError is raised.
      - Missing ctrl is surfaced immediately rather than silently skipped.
    """

    def __init__(
        self,
        loader: Optional[BundleLoader] = None,
        checker: Optional[CompatibilityChecker] = None,
        engine: Optional[InferenceEngine] = None,
        normalizer: Optional[Normalizer] = None,
        safety_checker: Optional[Any] = None,
        telemetry_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Parameters
        ----------
        safety_checker:
            Optional object with a ``check_policy_step(action, step_index,
            policy_id, context)`` method.  When provided, it is called once
            per policy step during ``run_episode()``.  A result with
            ``ok=False`` stops the episode with reason ``"safety_stop"``.
            Defaults to None (no per-step safety checking 閳?Phase 1 behavior).
        telemetry_fn:
            Optional callable ``fn(payload: dict) -> None`` invoked once at
            episode completion.  Receives a summary payload (no credentials).
            Defaults to None; when None, the built-in
            ``system.service.telemetry`` logger is used as best-effort.
        """
        self._loader = loader or BundleLoader()
        self._checker = checker or CompatibilityChecker()
        # engine may be overridden in load() based on bundle format; keep the
        # injected value as a fallback for tests that inject a mock engine.
        self._injected_engine: Optional[InferenceEngine] = engine
        self._engine: InferenceEngine = engine or ONNXEngine()
        self._normalizer: Normalizer = normalizer or Normalizer()
        self._safety_checker = safety_checker
        self._telemetry_fn = telemetry_fn

        # State set after load()
        self._bundle: Optional[CheckpointBundle] = None
        self._compat_report: Optional[CompatReport] = None
        self._obs_builder: Optional[ObsBuilder] = None
        self._action_applier: Optional[ActionApplier] = None
        self._env: Optional[SimEnvContext] = None
        self._loaded: bool = False
        self._policy_id: str = ""   # set in load() for telemetry / safety

    # ------------------------------------------------------------------
    # load()
    # ------------------------------------------------------------------

    def load(self, bundle_path: Path, env: SimEnvContext) -> CompatReport:
        """Load and validate a bundle against *env*.

        Assembly order (per spec):
          1. BundleLoader.load()
          2. Create ObsBuilder
          3. Create ActionApplier
          4. CompatibilityChecker.check()
          5. Raise IncompatibleWeightError on FAIL
          6. Retain Normalizer
          7. engine.load(bundle.policy_file)
          8. Store assembled components
          9. Return CompatReport

        Raises:
            FileNotFoundError           if bundle or model file is missing
            ManifestValidationError     if manifest is invalid
            IncompatibleWeightError     if CompatStatus is FAIL
        """
        self._loaded = False

        bundle = self._loader.load(Path(bundle_path))
        obs_builder = ObsBuilder(bundle)
        action_applier = ActionApplier(bundle)

        report = self._checker.check(bundle, env, obs_builder, action_applier)

        if report.status is CompatStatus.FAIL:
            fail_codes = [i.code for i in report.issues if i.severity is CompatStatus.FAIL]
            raise IncompatibleWeightError(
                f"Bundle '{bundle_path}' is incompatible with the current environment. "
                f"Failing checks: {fail_codes}. "
                f"Run CompatibilityChecker.check() for the full report."
            )

        # Select execution engine by bundle format (explicit, centralized).
        # If the caller injected an engine, honour it (test / override path).
        if self._injected_engine is None:
            self._engine = self._engine_for_format(bundle.policy_format)

        self._engine.load(bundle.policy_file)
        self._normalizer = self._load_bundle_normalizer(bundle)

        # Capture policy_id (directory name) for safety / telemetry context.
        self._policy_id = Path(bundle_path).name

        # Bind runtime timing to the loaded bundle so execution cadence matches
        # training/export timing rather than any adapter default.
        try:
            env.control_frequency_hz = float(bundle.control_frequency_hz)
        except Exception:
            pass
        try:
            setattr(env, "runtime_decimation", int(bundle.decimation))
        except Exception:
            pass
        try:
            adapter = getattr(env, "adapter", None)
            if adapter is not None:
                setattr(adapter, "control_frequency_hz", float(bundle.control_frequency_hz))
                setattr(adapter, "runtime_decimation", int(bundle.decimation))
        except Exception:
            pass

        self._bundle = bundle
        self._compat_report = report
        self._obs_builder = obs_builder
        self._action_applier = action_applier
        self._env = env
        self._loaded = True

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_stats_block(block: Any) -> Optional[NormalizationStats]:
        """Convert a JSON stats block into a typed NormalizationStats object."""
        if not isinstance(block, dict):
            return None
        mean = block.get("mean")
        std = block.get("std")
        if mean is None or std is None:
            return None
        return NormalizationStats(
            mean=np.asarray(mean, dtype=np.float32),
            std=np.asarray(std, dtype=np.float32),
            clip_min=(None if block.get("clip_min") is None else float(block.get("clip_min"))),
            clip_max=(None if block.get("clip_max") is None else float(block.get("clip_max"))),
        )

    @classmethod
    def _load_bundle_normalizer(cls, bundle: CheckpointBundle) -> Normalizer:
        """Load optional normalization stats from bundle metadata."""
        try:
            norm_meta = dict(bundle.raw_manifest.get("normalization") or {})
        except Exception:
            norm_meta = {}
        stats_file = str(norm_meta.get("file") or "").strip()
        if not stats_file:
            return Normalizer()

        stats_path = bundle.bundle_path / stats_file
        if not stats_path.exists():
            return Normalizer()

        import json

        with stats_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        obs_stats = cls._load_stats_block(raw.get("observation"))
        action_stats = cls._load_stats_block(raw.get("action"))
        return Normalizer(obs_stats=obs_stats, action_stats=action_stats)

    @staticmethod
    def _engine_for_format(fmt: str) -> InferenceEngine:
        """Return the correct InferenceEngine subclass for *fmt*.

        Raises ValueError for unsupported formats so failures are explicit.
        """
        if fmt == "onnx":
            return ONNXEngine()
        if fmt == "jit":
            return JITEngine()
        raise ValueError(
            f"PolicyRunner: unsupported policy format '{fmt}'. "
            f"Supported formats are: 'onnx', 'jit'."
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        mj_data: Any,
        last_action: np.ndarray,
        command: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Execute one policy tick and return the transformed action vector.

        Flow: obs build -> normalize obs -> predict -> denormalize action ->
              action apply -> return float32 vector.

        *mj_data* is passed explicitly so the caller controls which data
        object the observation reads from.  The env captured at load() time
        provides joint_names and bundle metadata; mj_data provides live state.

        Raises RuntimeError if called before load().
        """
        if not self._loaded:
            raise RuntimeError(
                "PolicyRunner.step() called before load(). "
                "Call load(bundle_path, env) first."
            )

        # Temporarily swap mj_data so obs_builder reads the live state.
        # We restore it after to avoid surprising the caller.
        original_mj_data = self._env.mj_data
        self._env.mj_data = mj_data

        try:
            obs = self._obs_builder.build(self._env, last_action=last_action, command=command)
            obs_norm = self._normalizer.normalize_obs(obs)
            raw_action = self._engine.predict(obs_norm)
            action = self._normalizer.denormalize_action(raw_action)
            final_action = self._action_applier.apply(action, self._env)
        finally:
            self._env.mj_data = original_mj_data

        return final_action

    # ------------------------------------------------------------------
    # run_episode()
    # ------------------------------------------------------------------

    def run_episode(
        self,
        env: SimEnvContext,
        max_steps: int = 1000,
        command: Optional[Sequence[float]] = None,
        render: bool = True,
    ) -> EpisodeResult:
        """Run the full episode loop with decimation.

        Steps per outer policy step = bundle.decimation physics substeps.
        steps_run counts policy decisions (outer steps), not physics substeps.

        Control write contract:
          - Writes action to env.mj_data.ctrl if present.
          - If ctrl length < action length, raises ValueError.
          - If ctrl length >= action length, writes the action into the first
            len(action) slots (prefix write 閳?documented choice for multi-actuator envs).

        Raises RuntimeError if called before load().
        """
        if not self._loaded:
            raise RuntimeError(
                "PolicyRunner.run_episode() called before load(). "
                "Call load(bundle_path, env) first."
            )

        assert self._bundle is not None

        try:
            env.reset()
        except NotImplementedError:
            pass
        try:
            self._obs_builder.reset()
        except Exception:
            pass

        last_action = np.zeros(self._bundle.action_dim, dtype=np.float32)
        steps_run = 0
        terminated = False
        engine_type = type(self._engine).__name__.replace("Engine", "").lower()  # "onnx" | "jit"
        termination_reason = "completed"
        safety_status = "ok"
        safety_stop = False

        try:
            for _ in range(max_steps):
                action = self.step(env.mj_data, last_action, command=command)
                last_action = action

                # Per-step safety check (Phase 2 hook 閳?no-op when no checker injected)
                if self._safety_checker is not None:
                    safety_result = self._safety_checker.check_policy_step(
                        action,
                        steps_run,
                        policy_id=self._policy_id,
                    )
                    if not safety_result.get("ok", True):
                        safety_stop = True
                        safety_status = "safety_stop"
                        termination_reason = "safety_stop"
                        break

                # Write to simulator control surface
                self._write_ctrl(env.mj_data, action)

                # Physics substeps (decimation). When rendering is enabled,
                # refresh the viewer on each substep so replay motion stays
                # visually continuous instead of updating only at control-rate.
                for _ in range(self._bundle.decimation):
                    env.sim_step()
                    if render:
                        env.render()

                steps_run += 1

                if env.is_terminated():
                    terminated = True
                    termination_reason = "env_terminated"
                    break

        except NotImplementedError as exc:
            termination_reason = "error"
            result = EpisodeResult(
                success=False,
                steps_run=steps_run,
                terminated=False,
                compat_status=self._compat_report.status,
                last_action=last_action if steps_run > 0 else None,
                message=f"Environment method not implemented: {exc}",
                engine_type=engine_type,
                termination_reason=termination_reason,
                safety_status=safety_status,
            )
            result.telemetry_emitted = self._emit_telemetry(result)
            return result
        except Exception as exc:
            termination_reason = "error"
            result = EpisodeResult(
                success=False,
                steps_run=steps_run,
                terminated=False,
                compat_status=self._compat_report.status,
                last_action=last_action if steps_run > 0 else None,
                message=f"Episode error: {exc}",
                engine_type=engine_type,
                termination_reason=termination_reason,
                safety_status=safety_status,
            )
            result.telemetry_emitted = self._emit_telemetry(result)
            return result

        if safety_stop:
            result = EpisodeResult(
                success=False,
                steps_run=steps_run,
                terminated=False,
                compat_status=self._compat_report.status,
                last_action=last_action if steps_run > 0 else None,
                message="Episode stopped by safety checker.",
                engine_type=engine_type,
                termination_reason=termination_reason,
                safety_status=safety_status,
            )
            result.telemetry_emitted = self._emit_telemetry(result)
            return result

        result = EpisodeResult(
            success=True,
            steps_run=steps_run,
            terminated=terminated,
            compat_status=self._compat_report.status,
            last_action=last_action if steps_run > 0 else None,
            message="Episode completed.",
            engine_type=engine_type,
            termination_reason=termination_reason,
            safety_status=safety_status,
        )
        result.telemetry_emitted = self._emit_telemetry(result)
        return result

    # ------------------------------------------------------------------
    # Telemetry emission
    # ------------------------------------------------------------------

    def _emit_telemetry(self, result: "EpisodeResult") -> bool:
        """Emit a lightweight summary telemetry event for the episode.

        Uses the injected ``telemetry_fn`` when provided; otherwise falls back
        to ``system.service.telemetry.TelemetryCollector`` (best-effort).
        Never raises 閳?telemetry failure must not affect the workflow.
        """
        payload = {
            "policy_id": self._policy_id,
            "engine_type": result.engine_type,
            "status": "ok" if result.success else "error",
            "compat_status": result.compat_status.value if result.compat_status is not None else "",
            "steps_run": result.steps_run,
            "termination_reason": result.termination_reason,
            "safety_status": result.safety_status,
        }

        try:
            if self._telemetry_fn is not None:
                self._telemetry_fn(payload)
                return True

            # Fallback: use existing service telemetry (best-effort)
            from system.service.telemetry import TelemetryCollector, new_trace_id  # lazy import
            trace_id = new_trace_id()
            collector = TelemetryCollector(
                trace_id=trace_id,
                adapter_name="policy_runner",
                brand="",
                operation="run_policy",
            )
            collector.start("episode")
            collector.end(
                "episode",
                status=payload["status"],
                reason=payload["termination_reason"],
                diagnostics={k: v for k, v in payload.items() if k not in ("status",)},
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _write_ctrl(mj_data: Any, action: np.ndarray) -> None:
        """Write *action* into mj_data.ctrl.

        Convention: prefix write.
          - If len(ctrl) >= len(action): write action into ctrl[0:len(action)].
          - If len(ctrl) <  len(action): raise ValueError.
          - If ctrl is absent: raise ValueError.
        """
        ctrl = getattr(mj_data, "ctrl", None)
        if ctrl is None:
            raise ValueError(
                "PolicyRunner: env.mj_data.ctrl is absent. "
                "Cannot write policy action to simulator."
            )
        ctrl_arr = np.asarray(ctrl)
        if ctrl_arr.shape[0] < action.shape[0]:
            raise ValueError(
                f"PolicyRunner: action length ({action.shape[0]}) exceeds "
                f"ctrl length ({ctrl_arr.shape[0]}). Cannot write action."
            )
        ctrl_arr[: action.shape[0]] = action
