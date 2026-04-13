from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.system.engine.contracts import DiagnosticsKey

from .action_applier import ActionApplier
from .bundle_loader import BundleLoader
from .compatibility_checker import CompatibilityChecker, CompatReport, CompatStatus
from .inference_engine import InferenceEngine, JITEngine, ONNXEngine
from .manifest_schema import CheckpointBundle
from .normalizer import NormalizationStats, Normalizer
from .obs_builder import ObsBuilder
from .sim_env_context import SimEnvContext

log = logging.getLogger(__name__)


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
        self._last_obs: Optional[np.ndarray] = None  # F3 per-step telemetry stash

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

        # ── H3 (AMP fix plan): algorithm-aware validation ─────────────
        # Per AMP-PPO_design.yaml §7.validation_at_load, bundles that
        # declare ``algorithm: AMP_PPO`` in their manifest are subject
        # to extra checks on top of the standard joint / obs_dim /
        # control_dt compat pass that ``CompatibilityChecker`` runs
        # below. We do NOT re-implement those generic checks here —
        # CompatibilityChecker remains the single source of truth for
        # shape/dim validation. What we add is:
        #
        #   * a positive confirmation log so users can see at a glance
        #     that they are loading an AMP bundle;
        #   * a warning if ``amp_metadata`` is missing (indicates the
        #     bundle was imported by a pre-B6 checkpoint_registry and
        #     will be missing discriminator provenance);
        #   * a warning if ``amp_metadata.spec_hash_verified == False``
        #     (meta file was edited after training finished — the
        #     actor weights are still usable but resume-training would
        #     be operating on a stale spec);
        #   * a reminder that the discriminator is NOT loaded into
        #     PolicyRunner — it is training-time state only, archived
        #     in the bundle for offline debugging via M6.
        #
        # The checks are advisory. A bundle with mangled amp_metadata
        # can still run inference because the policy.onnx is built
        # from the actor alone; amp_metadata only affects diagnostics.
        try:
            if bundle.algorithm == "AMP_PPO":
                amp_md = bundle.amp_metadata
                if amp_md is None:
                    log.warning(
                        "PolicyRunner.load: bundle '%s' declares algorithm="
                        "AMP_PPO but has no amp_metadata block in its "
                        "manifest. Discriminator archive / dataset "
                        "provenance are missing. The policy will still "
                        "run but offline AMP debugging tools will be "
                        "unable to consume this bundle.",
                        Path(bundle_path).name,
                    )
                else:
                    n_terms = len(amp_md.get("amp_obs_terms") or [])
                    amp_obs_dim = int(amp_md.get("amp_obs_dim") or 0)
                    n_datasets = len(amp_md.get("dataset_files") or [])
                    disc_ok = bool(amp_md.get("discriminator_archived"))
                    spec_ok = amp_md.get("spec_hash_verified")
                    log.info(
                        "PolicyRunner.load: AMP_PPO bundle — "
                        "%d obs term(s), amp_obs_dim=%d, "
                        "%d dataset file(s), discriminator_archived=%s, "
                        "spec_hash_verified=%s. Discriminator is training-"
                        "time only and will NOT be loaded into the runner.",
                        n_terms, amp_obs_dim, n_datasets, disc_ok, spec_ok,
                    )
                    if spec_ok is False:
                        log.warning(
                            "PolicyRunner.load: bundle '%s' has "
                            "spec_hash_verified=False — the training run "
                            "meta file was edited after training. Actor "
                            "weights are still usable for inference, but "
                            "resume-training from this bundle may be "
                            "operating on a stale spec.",
                            Path(bundle_path).name,
                        )
        except Exception:
            # Never let the AMP-specific validation block a load — the
            # generic compat pass below is the authoritative gate.
            log.exception("PolicyRunner.load: AMP metadata validation hook failed")

        # ── Joint reference frames ──────────────────────────────────────
        # The bundle's training joint order is the bundle space; the
        # MuJoCo model exposes its OWN qpos and ctrl orderings (which
        # may differ from each other on real-world MJCFs). All joint
        # vectors crossing layers go through explicit ``permute(source,
        # target)`` so the obs and action stacks never have to assume
        # two lists "happen to be in the same order".
        from src.system.policy.joint_space import (
            JointSpace,
            joint_spaces_from_deploy_contract,
            joint_spaces_from_mj_model,
        )

        bundle_space: Optional[JointSpace] = None
        qpos_space: Optional[JointSpace] = None
        ctrl_space: Optional[JointSpace] = None

        # ── Contract-first joint space construction ───────────────────
        # If the bundle ships a deploy_contract (sim2sim_design.yaml
        # Stage A2/B), build all three spaces from the explicit name list
        # via mj_name2id — strict, no canonicalize fallback. Falls back
        # to the legacy heuristic only when the contract is absent or
        # the MJCF doesn't match it (the latter is logged loudly).
        contract = None
        try:
            contract = bundle.deploy_contract
        except ValueError as exc:
            log.error(
                "PolicyRunner.load: bundle has a deploy_contract section but "
                "it failed validation (%s); falling back to legacy joint-space "
                "heuristic — sim2sim parity is NOT guaranteed.",
                exc,
            )
            contract = None

        mj_model = getattr(env, "mj_model", None)
        if contract is not None and mj_model is not None:
            try:
                bundle_space, qpos_space, ctrl_space = (
                    joint_spaces_from_deploy_contract(mj_model, contract)
                )
                log.info(
                    "PolicyRunner.load: built joint spaces from deploy_contract "
                    "(%d joints, identity_map=%s)",
                    contract.n_joints,
                    contract.is_identity_joint_map(),
                )
            except (ValueError, NotImplementedError) as exc:
                log.error(
                    "PolicyRunner.load: deploy_contract joint-space construction "
                    "failed (%s); falling back to legacy heuristic — sim2sim "
                    "parity is NOT guaranteed.",
                    exc,
                )
                bundle_space = None
                qpos_space = None
                ctrl_space = None

        # ── Legacy fallback ───────────────────────────────────────────
        if bundle_space is None:
            bundle_space = JointSpace(
                label="bundle", joints=tuple(bundle.joint_names)
            )
        if qpos_space is None or ctrl_space is None:
            try:
                if mj_model is not None:
                    qpos_space, ctrl_space = joint_spaces_from_mj_model(mj_model)
            except Exception:
                qpos_space = None
                ctrl_space = None

        obs_builder = ObsBuilder(
            bundle,
            bundle_space=bundle_space,
            qpos_space=qpos_space,
        )

        # ── PD controller construction ────────────────────────────────
        # Priority, top-down — the FIRST non-None wins:
        #   1. Isaac Lab deploy_contract — when the bundle ships a
        #      validated deploy_contract, its stiffness/damping/effort/
        #      default_joint_pos values ARE the authoritative training-
        #      time config parsed from env.yaml. The policy was trained
        #      against those exact gains; overriding them with MuJoCo-
        #      native defaults produces a 2-3× gain mismatch that makes
        #      the robot over-correct into thrashing / spasms even when
        #      every other pipeline stage is correct. This is the IL
        #      sim2sim path. Contract gains (e.g. Go2 Kp=25, Kd=0.5,
        #      effort=23.5) are per-joint and respect Isaac Lab's
        #      ImplicitActuator torque formula exactly.
        #   2. MuJoCo motor fallback (Kp=70, Kd=1.5, effort from MJCF
        #      ctrlrange) — for SB3 / UnitreeGymEnv-trained bundles
        #      whose training env had no deploy_contract. Those bundles
        #      WERE trained against these gains (UnitreeGymEnv's own
        #      _action_to_ctrl uses Kp=70 / Kd=1.5), so applying them at
        #      deploy matches training. Only triggers when action_type
        #      is joint_position AND every actuator is a plain motor.
        #   3. None — torque-mode bundles or envs with a native adapter
        #      that does its own PD.
        pd_controller = self._maybe_build_il_pd_controller(bundle)
        if pd_controller is None:
            pd_controller = self._build_motor_pd_fallback(bundle, env)
        action_applier = ActionApplier(
            bundle,
            pd_controller=pd_controller,
            bundle_space=bundle_space,
            ctrl_space=ctrl_space,
        )
        if qpos_space is not None:
            action_applier.attach_qpos_space(qpos_space)

        # ── Initial pose alignment ────────────────────────────────────
        # The MuJoCo Go2 keyframe / GO2_DEFAULT_QPOS pose is hip=0,
        # thigh=0.9, calf=-1.8 — completely different from the Isaac Lab
        # standing pose (hip=±0.1, thigh=0.8/1.0, calf=-1.5) the IL
        # policy was trained against. Without this alignment the very
        # first ``env.reset()`` in run_episode lands the robot in an
        # OOD pose, joint_pos_rel is huge from frame 0, and the PD
        # controller tries to slam the legs ~17° toward the IL default —
        # the "robot lurches at the start" symptom.
        #
        # We push the bundle's default_joint_pos (in **bundle order**)
        # into the env's _default_qpos via the qpos→bundle permutation
        # turned on its head, so the env reset uses the IL pose instead.
        try:
            self._align_env_default_pose(env, bundle, bundle_space, qpos_space)
        except Exception:
            log.exception("PolicyRunner.load: failed to align env default pose")

        # ── Physics timestep alignment ────────────────────────────────
        # The Isaac Lab bundle ships its training ``sim_dt`` (e.g. 0.005)
        # in the deploy_contract. The MuJoCo MJCF may use a different
        # default (menagerie Go2 scene.xml has no <option timestep> →
        # MuJoCo default 0.002). If we leave it alone, bundle.decimation
        # substeps execute at the wrong physics rate and the effective
        # control period is wrong (e.g. 4×0.002 = 8ms instead of the
        # trained 4×0.005 = 20ms). The policy was trained assuming
        # 20ms between decisions; running it at 8ms makes every learned
        # gait/torque response 2.5× too frequent — legs twitch then
        # diverge, producing the "twist into locked pose on init"
        # symptom.
        #
        # Force mj_model.opt.timestep to the contract's sim_dt so the
        # decimation loop in run_episode() produces the training-time
        # control period exactly. Only applies when the bundle ships
        # a contract; legacy bundles keep whatever the MJCF defines.
        try:
            self._align_env_physics_timestep(env, bundle)
        except Exception:
            log.exception("PolicyRunner.load: failed to align env physics timestep")

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
        #
        # NOTE on runtime_decimation: the actual decimation loop lives in
        # run_episode() below — see the `for _ in range(self._bundle.decimation)`
        # block. The setattr calls here are kept for telemetry/instrumentation
        # use ONLY (so external code can introspect the bound timing). The PD/
        # physics loop reads self._bundle.decimation directly and never queries
        # env.runtime_decimation. Do not rely on this attribute to drive
        # physics substeps — that contract is owned by run_episode().
        try:
            env.control_frequency_hz = float(bundle.control_frequency_hz)
        except Exception:
            pass
        try:
            setattr(env, "runtime_decimation", int(bundle.decimation))  # instrumentation only
        except Exception:
            pass
        try:
            adapter = getattr(env, "adapter", None)
            if adapter is not None:
                setattr(adapter, "control_frequency_hz", float(bundle.control_frequency_hz))
                setattr(adapter, "runtime_decimation", int(bundle.decimation))  # instrumentation only
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

    @staticmethod
    def _align_env_default_pose(
        env: SimEnvContext,
        bundle: "CheckpointBundle",
        bundle_space: "JointSpace",
        qpos_space: Optional["JointSpace"],
    ) -> None:
        """Patch the env's reset/default qpos to match the bundle's
        training pose.

        Only fires for IL bundles where ``robot.default_joint_pos`` is
        populated. The default pose lives in **bundle order**, so we
        permute it into qpos order before writing into the env.

        Resolution order for the write surface:
          1. ``env.set_default_qpos(values_in_qpos_order)`` — the public
             contract. UnitreeGymEnv implements it; new env classes
             should too.
          2. ``adapter.set_default_qpos(...)`` — same contract on the
             adapter when the env wraps one.
          3. Legacy private-attribute write to ``_default_qpos_full`` /
             ``_default_qpos``. Kept so older env classes keep working.
          4. Loud warning naming the env type — silent no-op was the
             P2 bug we're closing here.

        ``values_in_qpos_order`` is built once by permuting the
        bundle-order default through ``qpos_space`` so callers don't
        have to know the bundle layout.
        """
        raw_default = (
            (bundle.raw_manifest or {}).get("robot", {}) or {}
        ).get("default_joint_pos") or []
        if not raw_default:
            return

        try:
            default_in_bundle = np.asarray(raw_default, dtype=np.float32)
        except Exception:
            return

        if default_in_bundle.size == 0 or qpos_space is None:
            return

        # Build the qpos-ordered vector ONCE. For joints in qpos_space
        # that aren't in bundle_space (e.g. an actuator-only floating
        # wheel) we leave the env's existing default untouched, so we
        # need to start from whatever the env currently has and only
        # overwrite the bundle-controlled slots.
        adapter = getattr(env, "adapter", None)
        existing_source = None
        for candidate in (env, adapter):
            if candidate is None:
                continue
            existing_source = getattr(candidate, "_default_qpos_full", None)
            if existing_source is None:
                existing_source = getattr(candidate, "_default_qpos", None)
            if existing_source is not None:
                break

        n_qpos = len(qpos_space)
        if existing_source is not None:
            try:
                base = np.asarray(existing_source, dtype=np.float32).flatten().copy()
            except Exception:
                base = np.zeros(n_qpos, dtype=np.float32)
        else:
            base = np.zeros(n_qpos, dtype=np.float32)
        if base.shape[0] < n_qpos:
            base = np.pad(base, (0, n_qpos - base.shape[0]))

        for k, qname in enumerate(qpos_space.joints):
            if k >= n_qpos or k >= base.shape[0]:
                break
            idx = bundle_space.index_of_safe(qname)
            if idx is None or idx >= default_in_bundle.shape[0]:
                continue
            base[k] = float(default_in_bundle[idx])

        # Step 1+2: prefer the public ``set_default_qpos`` contract.
        for candidate in (env, adapter):
            if candidate is None:
                continue
            setter = getattr(candidate, "set_default_qpos", None)
            if callable(setter):
                try:
                    setter(base)
                    return
                except Exception:
                    log.exception(
                        "PolicyRunner._align_env_default_pose: %s.set_default_qpos failed",
                        type(candidate).__name__,
                    )
                    return

        # Step 3: legacy private-attribute path. Try the adapter first
        # (mirrors the v1 fall-through logic) so wrappers that proxy
        # state through ``env.adapter`` still get patched.
        for candidate in (adapter, env):
            if candidate is None:
                continue
            if not hasattr(candidate, "_default_qpos_full") and not hasattr(candidate, "_default_qpos"):
                continue
            try:
                if hasattr(candidate, "_default_qpos_full"):
                    candidate._default_qpos_full = base.astype(np.float32)
                if hasattr(candidate, "_default_qpos"):
                    existing_subset = getattr(candidate, "_default_qpos", None)
                    if existing_subset is not None:
                        n_sub = int(np.asarray(existing_subset).shape[0])
                        candidate._default_qpos = base[:n_sub].astype(np.float32)
                return
            except Exception:
                log.exception(
                    "PolicyRunner._align_env_default_pose: legacy write-back to %s failed",
                    type(candidate).__name__,
                )
                return

        # Step 4: nothing worked — surface this loudly so the next
        # robot integrator knows their env is missing the contract.
        log.warning(
            "PolicyRunner._align_env_default_pose: env %s exposes neither "
            "set_default_qpos() nor _default_qpos[_full]; IL standing pose "
            "alignment skipped — the policy will start from whatever pose "
            "the env's default reset() lands at, which may be OOD for "
            "Isaac Lab-trained bundles.",
            type(env).__name__,
        )

    @staticmethod
    def _align_env_physics_timestep(env: Any, bundle: "CheckpointBundle") -> None:
        """Force MuJoCo's physics timestep to match the bundle's training sim_dt.

        The Isaac Lab bundle carries its training-time physics timestep
        in ``deploy_contract.sim_dt`` (e.g. 0.005 for the stock locomotion
        configs). Most MuJoCo menagerie scenes do not declare an explicit
        ``<option timestep="..."/>`` and inherit the MuJoCo default of
        0.002 s. If we leave the model alone, every physics substep in
        ``run_episode``'s decimation loop advances simulated time by the
        wrong amount — e.g. at decimation=4 we cover 8 ms instead of the
        20 ms the policy was trained to expect. The dynamics integrate
        2.5× faster than training, every learned torque response is
        compressed in time, and the robot twists into a locked pose
        within a handful of substeps.

        This fix overrides ``mj_model.opt.timestep`` at bundle-load time
        so the decimation loop produces the exact training-time control
        period. Legacy bundles without a contract fall through untouched.
        """
        try:
            contract = bundle.deploy_contract
        except Exception:
            contract = None
        if contract is None:
            return
        sim_dt = float(getattr(contract, "sim_dt", 0.0) or 0.0)
        if sim_dt <= 0.0:
            return

        mj_model = getattr(env, "mj_model", None)
        if mj_model is None and getattr(env, "adapter", None) is not None:
            mj_model = getattr(env.adapter, "mj_model", None)
        if mj_model is None or not hasattr(mj_model, "opt"):
            log.warning(
                "PolicyRunner._align_env_physics_timestep: env %s has no "
                "mj_model — skipping timestep alignment (bundle sim_dt=%.4f).",
                type(env).__name__,
                sim_dt,
            )
            return

        old_dt = float(mj_model.opt.timestep)
        if abs(old_dt - sim_dt) < 1e-9:
            return
        try:
            mj_model.opt.timestep = sim_dt
        except Exception:
            log.exception(
                "PolicyRunner._align_env_physics_timestep: failed to assign "
                "mj_model.opt.timestep (was %.6f, wanted %.6f)",
                old_dt,
                sim_dt,
            )
            return
        log.info(
            "PolicyRunner._align_env_physics_timestep: mj_model.opt.timestep "
            "%.4f → %.4f (bundle sim_dt; matches training decimation %d × "
            "%.4f = %.4f s control period)",
            old_dt,
            sim_dt,
            int(getattr(contract, "decimation", 0) or 0),
            sim_dt,
            sim_dt * float(getattr(contract, "decimation", 0) or 0),
        )

    @staticmethod
    def _maybe_build_il_pd_controller(bundle: "CheckpointBundle"):
        """Construct a PDController for an Isaac Lab bundle, or return None.

        Two-tier resolution (sim2sim_design.yaml Stage C):

        1. **deploy_contract path (preferred)** — when the bundle's manifest
           ships a ``deploy_contract`` section, build the PDController from
           it via ``PDController.from_deploy_contract``. This is the single
           source of truth path: stiffness/damping/effort/default_joint_pos
           come from one validated record, no regex pattern expansion, no
           silent fallbacks. Works for IL AND SB3 bundles once both produce
           contracts.
        2. **env.yaml path (legacy)** — for IL bundles that don't yet ship a
           contract, fall back to parsing the bundled ``env.yaml``. Fix #2
           ensured the silent failures here are now log.error.
        """
        # Tier 1: contract path (works for any bundle that has one)
        try:
            contract = bundle.deploy_contract
        except ValueError as exc:
            log.error(
                "PolicyRunner._maybe_build_il_pd_controller: bundle has a "
                "deploy_contract section but it failed validation (%s); "
                "falling back to env.yaml path.",
                exc,
            )
            contract = None
        if contract is not None and contract.stiffness:
            try:
                from src.system.sim.pd_controller import PDController
                return PDController.from_deploy_contract(
                    contract, bundle.joint_names
                )
            except ValueError as exc:
                log.error(
                    "PolicyRunner._maybe_build_il_pd_controller: "
                    "from_deploy_contract failed (%s); falling back to "
                    "env.yaml path.",
                    exc,
                )

        # Tier 2: legacy env.yaml path (IL bundles only)
        try:
            convention = str(
                (bundle.raw_manifest or {}).get("inference_convention", "")
            ).strip().lower()
        except Exception:
            convention = ""
        if convention != "isaac_lab":
            return None
        try:
            import yaml as _yaml
            from src.system.skill.isaac_lab_manifest_parser import _IsaacLabEnvYamlLoader
            from src.system.sim.pd_controller import PDController
            env_yaml_path = bundle.bundle_path / "env.yaml"
            if not env_yaml_path.is_file():
                __import__("logging").getLogger(__name__).error(
                    "PolicyRunner: Isaac Lab bundle %s is marked "
                    "inference_convention=isaac_lab but has no env.yaml — "
                    "PD gains, default joint pose and decimation cannot be "
                    "recovered. The runner will fall back to MuJoCo's native "
                    "actuator config, which is the leading cause of sim2sim "
                    "divergence for IL bundles.",
                    bundle.bundle_path,
                )
                return None
            with env_yaml_path.open("r", encoding="utf-8") as fh:
                raw_env = _yaml.load(fh, Loader=_IsaacLabEnvYamlLoader) or {}
            return PDController.from_env_yaml(
                raw_env,
                num_joints=int(bundle.action_dim),
                joint_names=list(bundle.joint_names),
            )
        except Exception:
            log = __import__("logging").getLogger(__name__)
            log.exception("PolicyRunner: failed to build IL PDController")
            return None

    @staticmethod
    def _build_motor_pd_fallback(
        bundle: "CheckpointBundle",
        env: SimEnvContext,
    ) -> Optional["PDController"]:
        """Build a fallback PDController for MuJoCo motor-actuator envs.

        Fires when ``_maybe_build_il_pd_controller`` returns None (no
        deploy_contract, no env.yaml).  Covers SB3 and AMP_PPO bundles
        trained through UnitreeGymEnv, whose ``_action_to_ctrl`` provided
        PD conversion via the adapter in the old code path. MjSimEnv has
        no adapter, so we replicate that PD here.

        The gains match UnitreeGymEnv's training defaults:
          - Kp = 70.0  (``_PD_KP`` in unitree_gym_env.py)
          - Kd =  1.5  (``_PD_KD``)
          - effort_limit = actuator ctrlrange × gear (per-actuator)
          - default_pos  = bundle ``robot.default_joint_pos``

        Only builds when action_type is ``joint_position`` (the mode
        where raw actions are residuals added to the standing pose).
        Torque-mode bundles pass through to ``apply()`` as before.
        """
        try:
            action_type = str(
                (bundle.raw_manifest or {}).get("action_space", {}).get("type", "")
            ).strip().lower()
        except Exception:
            action_type = ""
        if action_type != "joint_position":
            return None

        mj_model = getattr(env, "mj_model", None)
        if mj_model is None:
            return None

        # Only build when every actuator is a basic motor (mjtTrn_JOINT=0).
        # Position / velocity / general actuators have their own semantics.
        try:
            import mujoco as _mj
            for ai in range(int(mj_model.nu)):
                if int(mj_model.actuator_trntype[ai]) != 0:  # mjTRN_JOINT
                    return None
        except Exception:
            return None

        n = int(bundle.action_dim)
        _KP = 70.0
        _KD = 1.5

        try:
            # Per-actuator effort limit = abs(ctrlrange.high) * gear.
            # For menagerie models with gear=1 this equals ctrlrange.
            gear = np.asarray(mj_model.actuator_gear[:n, 0], dtype=np.float32)
            gear = np.where(gear > 0, gear, 1.0)
            ctrlrange = np.asarray(mj_model.actuator_ctrlrange[:n, 1], dtype=np.float32)
            effort = np.abs(ctrlrange) * gear
        except Exception:
            effort = np.full(n, 33.5, dtype=np.float32)

        # Default standing pose from the bundle manifest.
        raw_default = (
            (bundle.raw_manifest or {}).get("robot", {}) or {}
        ).get("default_joint_pos") or []
        try:
            default_pos = np.asarray(raw_default, dtype=np.float32).flatten()
            if default_pos.shape[0] < n:
                default_pos = np.pad(default_pos, (0, n - default_pos.shape[0]))
        except Exception:
            default_pos = np.zeros(n, dtype=np.float32)

        try:
            from src.system.sim.pd_controller import PDController

            log.info(
                "PolicyRunner: built motor-actuator fallback PDController "
                "(Kp=%.1f Kd=%.1f n=%d effort=[%.1f..%.1f]) for bundle '%s'",
                _KP, _KD, n, float(effort.min()), float(effort.max()),
                getattr(bundle, "name", "?"),
            )
            return PDController(
                kp=_KP, kd=_KD,
                effort_limit=effort,
                default_pos=default_pos,
                num_joints=n,
            )
        except Exception:
            log.exception("PolicyRunner: failed to build motor-actuator fallback PDController")
            return None

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def _predict_policy(
        self,
        mj_data: Any,
        last_action: np.ndarray,
        command: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Run one policy forward pass at control rate.

        Returns the denormalized policy output **in bundle joint order**,
        *before* any PD / actuator conversion. The caller is expected to
        feed this vector into ``_apply_policy_output`` each physics
        substep so the PD torque stays fresh with live ``q/q̇``.

        Splitting the obs→policy step from the action→torque step is
        what lets us run policy decisions at 50 Hz (control rate) while
        refreshing PD torque at 200 Hz (sim rate). Isaac Lab's
        ``ImplicitActuator`` does the equivalent inside PhysX's
        constraint solver — by matching that structure on the MuJoCo
        side we remove a major sim2sim mismatch the user observed as
        "slippery / soft legs".
        """
        if not self._loaded:
            raise RuntimeError(
                "PolicyRunner._predict_policy called before load()."
            )

        original_mj_data = self._env.mj_data
        self._env.mj_data = mj_data
        try:
            obs = self._obs_builder.build(
                self._env, last_action=last_action, command=command
            )
            # Stash the un-normalized obs for per-step telemetry consumers
            # (sim2sim_design.yaml Stage F3). run_episode reads this after
            # the call to publish to the telemetry callback. We store the
            # PRE-normalize vector so external dashboards see the same
            # numbers the user wired up in the contract — the in-graph
            # rsl_rl normalizer is invisible to consumers by design.
            self._last_obs = obs
            obs_norm = self._normalizer.normalize_obs(obs)
            raw_action = self._engine.predict(obs_norm)
            return self._normalizer.denormalize_action(raw_action)
        finally:
            self._env.mj_data = original_mj_data

    def _apply_policy_output(
        self,
        policy_output: np.ndarray,
        mj_data: Any,
    ) -> np.ndarray:
        """Convert a (frozen) policy output into a fresh ctrl-space torque.

        Called once per physics substep inside the decimation loop so
        the PD controller sees the up-to-date ``mj_data.qpos/qvel``. The
        policy output itself (``policy_output``) does NOT change across
        the substeps of a single control decision — only the sim state
        does, and therefore only the PD's error/velocity terms update.

        Picks ``apply_with_pd`` when a PDController is attached (covers
        IL bundles, AMP_PPO bundles, and any SB3 bundle whose action
        type is ``joint_position`` — all of which need the PD torque
        path to produce correct forces for MuJoCo motor actuators).
        Falls through to ``apply`` only when no PDController exists
        (e.g. torque-mode bundles or envs with a native adapter that
        provides its own ``_action_to_ctrl``).
        """
        original_mj_data = self._env.mj_data
        self._env.mj_data = mj_data
        try:
            if self._action_applier._pd_controller is not None:
                return self._action_applier.apply_with_pd(policy_output, self._env)
            return self._action_applier.apply(policy_output, self._env)
        finally:
            self._env.mj_data = original_mj_data

    def step(
        self,
        mj_data: Any,
        last_action: np.ndarray,
        command: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Legacy single-shot step: predict + apply in one call.

        Kept for external callers that do their own substep loop or
        only want a single torque per policy decision. Internal
        ``run_episode`` uses the split ``_predict_policy`` /
        ``_apply_policy_output`` API so PD torque can refresh each
        decimation substep.

        Raises RuntimeError if called before load().
        """
        policy_output = self._predict_policy(mj_data, last_action, command)
        return self._apply_policy_output(policy_output, mj_data)

    # ------------------------------------------------------------------
    # run_episode()
    # ------------------------------------------------------------------

    def run_episode(
        self,
        env: SimEnvContext,
        max_steps: int = 1000,
        command: Optional[Sequence[float]] = None,
        render: bool = True,
        command_provider: Optional[Callable[[], Sequence[float]]] = None,
        step_telemetry_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> EpisodeResult:
        """Run the full episode loop with decimation.

        Steps per outer policy step = bundle.decimation physics substeps.
        steps_run counts policy decisions (outer steps), not physics substeps.

        Parameters
        ----------
        env, max_steps, command, render
            As before.
        command_provider:
            Optional zero-argument callable that returns the freshest
            command vector (e.g. ``[vx, vy, vyaw]``) every policy
            decision. When supplied, the command is **re-fetched once
            per outer iteration** so live keyboard / gamepad input from
            ``GlobalInputManager`` flows into the policy in real time.
            The fixed ``command`` argument is used as the initial value
            and as the fallback if ``command_provider()`` raises. When
            ``None`` the runtime keeps using ``command`` for every
            decision (legacy behaviour, used by static-command callers).

            The provider is called from the run_episode thread, NOT from
            inside the decimation substep loop, so the policy still sees
            a single command per 50 Hz decision (matching how Isaac Lab
            sends commands at the control rate).

        Control write contract:
          - Writes action to env.mj_data.ctrl if present.
          - If ctrl length < action length, raises ValueError.
          - If ctrl length >= action length, writes the action into the first
            len(action) slots (prefix write — documented choice for multi-actuator envs).

        Raises RuntimeError if called before load().
        """
        if not self._loaded:
            raise RuntimeError(
                "PolicyRunner.run_episode() called before load(). "
                "Call load(bundle_path, env) first."
            )

        assert self._bundle is not None

        # Resolve the active command vector for this iteration. When a
        # provider is configured we call it; on any failure we silently
        # fall back to the static ``command`` so a flaky live input
        # source can never crash the replay loop.
        def _current_command() -> Optional[Sequence[float]]:
            if command_provider is None:
                return command
            try:
                fresh = command_provider()
            except Exception:
                log.exception("PolicyRunner: command_provider raised; using static fallback")
                return command
            if fresh is None:
                return command
            return fresh

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
                # ── Policy decision (control rate, ~50 Hz) ─────────────
                # One obs build + ONNX/JIT forward pass per outer iter.
                # ``policy_output`` stays in bundle joint order —
                # ``_apply_policy_output`` below is what permutes it to
                # ctrl order and wraps it in PD torque each substep.
                #
                # ``_current_command()`` re-fetches from
                # ``command_provider`` if one is configured, so live
                # keyboard / gamepad input flows through here at exactly
                # the same rate the policy was trained against (50 Hz).
                # When no provider is configured this returns the
                # static ``command`` argument unchanged.
                live_command = _current_command()
                policy_output = self._predict_policy(
                    env.mj_data, last_action, command=live_command
                )

                # Feed the raw bundle-order policy output into the next
                # obs step's "actions" term. Previous behaviour leaked
                # the ctrl-order torque into that slot, which for IL
                # bundles (where bundle and ctrl orderings differ) was
                # silently wrong: the policy got its own previous
                # output permuted into the wrong joint order.
                #
                # INVARIANT I5 (sim2sim_design.yaml): last_action MUST be
                # the RAW denormalized network output — before action_scale,
                # before clip, before default_joint_pos offset, before PD.
                # The policy was trained to see its own previous raw output
                # as part of the obs. Do not let any post-PD value flow into
                # this slot.
                last_action = policy_output

                # ── Per-step telemetry (F3) ────────────────────────────
                # Fire the optional step_telemetry_fn with the obs/action/
                # command we just acted on. The callback runs SYNCHRONOUSLY
                # at the policy decision rate (~50 Hz), so it must not
                # block — typical implementation is a non-blocking
                # event_bus.publish from BehaviorNode. Wrapped in
                # try/except so a slow / broken telemetry consumer never
                # crashes the replay loop.
                if step_telemetry_fn is not None:
                    try:
                        payload: Dict[str, Any] = {
                            DiagnosticsKey.MUJOCO_SIM2SIM_STEP_INDEX: int(steps_run),
                            DiagnosticsKey.MUJOCO_SIM2SIM_LAST_ACTION: (
                                np.asarray(policy_output, dtype=np.float32).copy()
                            ),
                        }
                        if self._last_obs is not None:
                            payload[DiagnosticsKey.MUJOCO_SIM2SIM_LAST_OBS] = (
                                np.asarray(self._last_obs, dtype=np.float32).copy()
                            )
                        if live_command is not None:
                            payload[DiagnosticsKey.MUJOCO_SIM2SIM_COMMAND] = (
                                np.asarray(live_command, dtype=np.float32).copy()
                            )
                        step_telemetry_fn(payload)
                    except Exception:
                        log.exception(
                            "PolicyRunner.run_episode: step_telemetry_fn raised — "
                            "ignored to keep the replay loop alive."
                        )

                # Per-step safety check (Phase 2 hook — no-op when no
                # checker injected)
                if self._safety_checker is not None:
                    safety_result = self._safety_checker.check_policy_step(
                        policy_output,
                        steps_run,
                        policy_id=self._policy_id,
                    )
                    if not safety_result.get("ok", True):
                        safety_stop = True
                        safety_status = "safety_stop"
                        termination_reason = "safety_stop"
                        break

                # ── Physics substeps with per-substep PD refresh ───────
                # Previously we wrote ``ctrl`` once and called
                # ``sim_step`` ``decimation`` times with the SAME frozen
                # ctrl for the whole 20 ms window. Isaac Lab's
                # ImplicitActuator recomputes the joint drive torque
                # inside every PhysX substep using the latest q/q̇ —
                # so "freeze for 20 ms" is a structural mismatch that
                # manifests as stale PD error and drifting steady state
                # (the "slippery" feeling on replay). Recomputing
                # ``apply_with_pd`` each substep here closes that gap
                # without running the policy more often than 50 Hz.
                #
                # INVARIANT (sim2sim_design.yaml rule_4): the decimation
                # loop is owned HERE, in PolicyRunner.run_episode. The
                # policy is queried ONCE per outer step; the inner loop
                # runs `bundle.decimation` MuJoCo physics substeps with
                # a re-computed PD torque each substep. env.sim_step()
                # MUST execute exactly ONE mj_step per call — do not
                # move the loop into MjSimEnv or the contract breaks.
                for _ in range(self._bundle.decimation):
                    fresh_torque = self._apply_policy_output(
                        policy_output, env.mj_data
                    )
                    self._write_ctrl(env.mj_data, fresh_torque)
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
        finally:
            # Stop driving the simulator the moment the policy loop
            # exits — *for any reason*. Without this, the last frame's
            # ctrl values stay frozen in ``mj_data.ctrl`` and any
            # downstream loop that keeps calling ``mj_step`` (the
            # vis_check_runner ``hold_viewer_open`` viewer is the most
            # visible offender) re-applies the same torques forever,
            # producing the "robot suddenly snaps then locks at a weird
            # angle" symptom the user reported. Zero ctrl is the
            # conservative choice: the robot collapses under gravity
            # rather than fighting against frozen targets.
            #
            # ``finally`` runs even when an except handler above already
            # returned, so all four return paths from this method get
            # the cleanup automatically.
            try:
                mj_data = getattr(env, "mj_data", None)
                if mj_data is not None and hasattr(mj_data, "ctrl"):
                    mj_data.ctrl[:] = 0.0
            except Exception:
                pass

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
            from src.system.service.telemetry import TelemetryCollector, new_trace_id  # lazy import
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
