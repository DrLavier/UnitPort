"""PolicyRunner — bundle → ObsBuilder/ActionApplier → run_episode (port from DEMO).

Phase 3 RELEASE port. Adjustments:

* ``manifest_schema`` import → ``application.training.manifest_schema``.
* ``DiagnosticsKey`` import → ``application.engine.contracts`` (a minimal
  shim exposing only the four ``MUJOCO_SIM2SIM_*`` keys this file uses).
* ``pd_controller`` import → ``application.service.runtime.simulation.pd_controller``.
* The legacy env.yaml PD-controller path (DEMO Tier 2 in
  ``_maybe_build_il_pd_controller``) is dropped: it depended on
  ``src.system.skill.isaac_lab_manifest_parser`` which has not landed in
  RELEASE yet. New bundles ship a deploy_contract (Tier 1) and Phase 3
  acceptance only goes through that path. SB3 fallback (``_build_motor_pd_fallback``)
  is preserved.
* The telemetry fallback (DEMO ``src.system.service.telemetry``) is also
  dropped; ``telemetry_fn`` works as before, and absent that, no-op.

Critical invariants — byte-identical with DEMO:

* **R2 (last_action raw)** — ``run_episode`` line that sets
  ``last_action = policy_output`` AFTER the network output, BEFORE any
  scale/clip/PD. The "INVARIANT I5" comment block is preserved
  verbatim. PD-applied torques NEVER flow into the obs's last_action term.
* **R3 (timestep alignment)** — ``_align_env_physics_timestep`` forces
  ``mj_model.opt.timestep = contract.sim_dt`` at load time so the
  decimation loop produces the training-time control period.
* **R6 (default_pose alignment)** — ``_align_env_default_pose`` permutes
  the bundle's ``robot.default_joint_pos`` from bundle order into qpos
  order via the JointSpace map and writes via ``set_default_qpos`` /
  ``_default_qpos_full`` fallback.
* The decimation loop owns physics substeps; ``env.sim_step()`` MUST
  execute exactly one ``mj_step`` per call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from application.engine.contracts import DiagnosticsKey
from application.training.manifest_schema import CheckpointBundle

from .action_applier import ActionApplier
from .bundle_loader import BundleLoader
from .compatibility_checker import CompatibilityChecker, CompatReport, CompatStatus
from .inference_engine import InferenceEngine, JITEngine, ONNXEngine
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
    # Phase 2 fields — present with defaults so Phase 1 consumers are unaffected
    engine_type: str = ""           # "onnx" | "jit"
    termination_reason: str = ""    # "completed" | "safety_stop" | "env_terminated" | "error"
    safety_status: str = ""         # "ok" | "safety_stop" | ""
    telemetry_emitted: bool = False


# ---------------------------------------------------------------------------
# PolicyRunner
# ---------------------------------------------------------------------------

class PolicyRunner:
    """Assembles bundle → ObsBuilder/ActionApplier → episode loop."""

    def __init__(
        self,
        loader: Optional[BundleLoader] = None,
        checker: Optional[CompatibilityChecker] = None,
        engine: Optional[InferenceEngine] = None,
        normalizer: Optional[Normalizer] = None,
        safety_checker: Optional[Any] = None,
        telemetry_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._loader = loader or BundleLoader()
        self._checker = checker or CompatibilityChecker()
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
        self._policy_id: str = ""
        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # load()
    # ------------------------------------------------------------------

    def load(
        self,
        bundle_path: Path,
        env: SimEnvContext,
        *,
        robot_sku: Optional[str] = None,
    ) -> CompatReport:
        """Load and validate a bundle against *env*.

        ``robot_sku`` is the canonical robot identity. When supplied by the
        caller (UI / training-time round-trip), it wins. When None, falls
        back to ``bundle.robot_sku`` (the SKU stored in manifest.yaml).
        Reverse-resolving SKU from manifest brand/model strings is no
        longer supported — that was an antipattern that broke whenever
        the brand string was the human display name (e.g. "Unitree Go2").
        """
        self._loaded = False

        bundle = self._loader.load(Path(bundle_path))
        effective_sku = (robot_sku or bundle.robot_sku or "").strip()
        if not effective_sku:
            raise IncompatibleWeightError(
                f"Bundle '{bundle_path}' has no robot.sku and the caller "
                f"passed none. Re-export the bundle so manifest.yaml "
                f"carries the SKU, or pass robot_sku= explicitly."
            )

        # ── AMP_PPO advisory checks ───────────────────────────────────
        try:
            if bundle.algorithm == "AMP_PPO":
                amp_md = bundle.amp_metadata
                if amp_md is None:
                    log.warning(
                        "PolicyRunner.load: bundle '%s' declares algorithm="
                        "AMP_PPO but has no amp_metadata block.",
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
                        "spec_hash_verified=%s.",
                        n_terms, amp_obs_dim, n_datasets, disc_ok, spec_ok,
                    )
                    if spec_ok is False:
                        log.warning(
                            "PolicyRunner.load: bundle '%s' has "
                            "spec_hash_verified=False — actor weights are "
                            "still usable but resume-training may use a "
                            "stale spec.",
                            Path(bundle_path).name,
                        )
        except Exception:
            log.exception("PolicyRunner.load: AMP metadata validation hook failed")

        # ── Joint reference frames ──────────────────────────────────────
        from .joint_space import (
            JointSpace,
            joint_spaces_from_deploy_contract,
            joint_spaces_from_mj_model,
        )

        bundle_space: Optional[JointSpace] = None
        qpos_space: Optional[JointSpace] = None
        ctrl_space: Optional[JointSpace] = None

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

        # Normalize legacy IL bundles to the Isaac Lab USD articulation
        # joint order BEFORE any downstream consumer (JointSpace /
        # ObsBuilder / ActionApplier / PDController) reads the contract.
        # Bundles exported before bundle_finalizer learned about
        # ``RobotSpec.isaac_lab_joint_order`` ship per-joint arrays in
        # SDK canonical (按腿) order, but the trained policy expects
        # USD prim (按 type) order — mismatch silently corrupts every
        # joint-space obs/action term and produces twitching sim2sim
        # runs. The normalize is in-place on the cached
        # CheckpointBundle.deploy_contract instance, so all subsequent
        # bundle.deploy_contract reads see the corrected order.
        if contract is not None:
            try:
                PolicyRunner._normalize_il_contract_joint_order(
                    contract, bundle.raw_manifest or {}
                )
            except Exception:
                log.exception(
                    "PolicyRunner.load: contract joint-order normalize raised"
                )

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

        if bundle_space is None:
            # bundle.joint_names is IR roles (Phase 5+). Translate to
            # physical names via the SKU plumbed in by the caller; the
            # SKU is authoritative — no reverse-derivation from manifest
            # display strings.
            bundle_joints = tuple(bundle.joint_names)
            try:
                from application.service.runtime.policy.joint_name_utils import (
                    ir_roles_to_physical_names,
                )
                physical = ir_roles_to_physical_names(
                    list(bundle_joints), effective_sku
                )
                bundle_joints = tuple(physical)
                log.info(
                    "PolicyRunner.load: bundle.joint_names IR→physical "
                    "via robot_sku=%r (%d joints translated)",
                    effective_sku, len(physical),
                )
            except ValueError as exc:
                # Translation failure is a hard problem — the bundle's
                # IR roles don't match the SKU's joint catalog. Surface
                # via CompatibilityChecker (below) rather than guessing.
                log.error(
                    "PolicyRunner.load: bundle.joint_names IR→physical "
                    "translation failed (%s) for robot_sku=%r; deploy "
                    "compatibility check will report the mismatch.",
                    exc, effective_sku,
                )
            bundle_space = JointSpace(label="bundle", joints=bundle_joints)
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
            robot_sku=effective_sku,
        )

        # ── PD controller construction ────────────────────────────────
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

        # ── Initial pose alignment (R6) ───────────────────────────────
        try:
            self._align_env_default_pose(env, bundle, bundle_space, qpos_space)
        except Exception:
            log.exception("PolicyRunner.load: failed to align env default pose")

        # ── IL training init root position (authoritative spawn base z) ──
        # Bundles exported by the current pipeline ship contract.init_base_pos
        # (= spec.actor.init_pos_x/y/z, the same tuple Isaac Lab uses in
        # ArticulationCfg.InitialStateCfg.pos). Push it into the env so
        # MjSimEnv.reset can write the authoritative spawn base z instead of
        # the contact-based heuristic. Legacy bundles without the field leave
        # init_base_pos=None and the env falls back to the heuristic.
        try:
            init_base_pos = (
                getattr(contract, "init_base_pos", None)
                if contract is not None else None
            )
            for candidate in (env, getattr(env, "adapter", None)):
                if candidate is None:
                    continue
                setter = getattr(candidate, "set_il_init_base_pos", None)
                if callable(setter):
                    try:
                        setter(init_base_pos)
                    except Exception:
                        log.exception(
                            "PolicyRunner.load: set_il_init_base_pos failed "
                            "on %s",
                            type(candidate).__name__,
                        )
                    break
        except Exception:
            log.exception(
                "PolicyRunner.load: failed to push IL init_base_pos to env"
            )

        # ── Physics timestep alignment (R3) ───────────────────────────
        try:
            self._align_env_physics_timestep(env, bundle)
        except Exception:
            log.exception("PolicyRunner.load: failed to align env physics timestep")

        report = self._checker.check(
            bundle, env, obs_builder, action_applier, robot_sku=effective_sku,
        )

        if report.status is CompatStatus.FAIL:
            fail_codes = [i.code for i in report.issues if i.severity is CompatStatus.FAIL]
            raise IncompatibleWeightError(
                f"Bundle '{bundle_path}' is incompatible with the current environment. "
                f"Failing checks: {fail_codes}. "
                f"Run CompatibilityChecker.check() for the full report."
            )

        if self._injected_engine is None:
            self._engine = self._engine_for_format(bundle.policy_format)

        self._engine.load(bundle.policy_file)
        self._normalizer = self._load_bundle_normalizer(bundle)

        self._policy_id = Path(bundle_path).name

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_il_contract_joint_order(
        contract: "DeployContract",
        raw_manifest: Dict[str, Any],
    ) -> bool:
        """In-place reorder ``contract`` per-joint arrays to the Isaac Lab
        USD articulation joint order when the bundle ships them in a
        different order.

        Legacy IL bundles exported before ``bundle_finalizer`` learned
        about ``RobotSpec.isaac_lab_joint_order`` write ``joint_sdk_names``
        in SDK canonical (按腿) order, but the trained policy expects USD
        prim (按 type) order. Without normalization, every joint-keyed
        obs/action slot lands in the wrong position and the deploy run
        twitches catastrophically.

        Returns ``True`` when a reorder was applied, ``False`` otherwise.
        The mutation is in-place on the contract instance, which is the
        cached ``CheckpointBundle.deploy_contract`` — every downstream
        reader (JointSpace, ObsBuilder, ActionApplier, PDController) sees
        the corrected order.
        """
        inference_convention = str(
            (raw_manifest or {}).get("inference_convention", "") or ""
        ).strip().lower()
        if inference_convention != "isaac_lab":
            return False
        sku = str(getattr(contract, "robot_sku", "") or "")
        if not sku:
            return False
        try:
            from registers.robots import get_robot_spec
        except ImportError:
            return False
        spec = get_robot_spec(sku)
        if spec is None:
            return False
        expected = list(getattr(spec, "isaac_lab_joint_order", None) or [])
        if not expected:
            return False
        current = list(contract.joint_sdk_names or [])
        if not current or current == expected:
            return False
        idx_map = {role: i for i, role in enumerate(current)}
        missing = [r for r in expected if r not in idx_map]
        if missing:
            log.error(
                "PolicyRunner.load: cannot normalize contract joint order "
                "for SKU %r — isaac_lab_joint_order references roles "
                "missing from contract.joint_sdk_names: %s. Skipping; "
                "sim2sim parity NOT guaranteed.",
                sku, missing,
            )
            return False
        if len(expected) != len(current):
            log.error(
                "PolicyRunner.load: cannot normalize contract joint order "
                "for SKU %r — isaac_lab_joint_order len=%d ≠ "
                "contract.joint_sdk_names len=%d. Skipping.",
                sku, len(expected), len(current),
            )
            return False
        perm = [idx_map[r] for r in expected]
        n = len(perm)

        def _permute(arr: Any) -> Any:
            if isinstance(arr, list) and len(arr) == n:
                return [arr[i] for i in perm]
            return arr

        contract.joint_sdk_names = _permute(contract.joint_sdk_names)
        contract.stiffness = _permute(contract.stiffness)
        contract.damping = _permute(contract.damping)
        contract.effort_limit = _permute(contract.effort_limit)
        contract.default_joint_pos = _permute(contract.default_joint_pos)
        if contract.velocity_limit is not None:
            contract.velocity_limit = _permute(contract.velocity_limit)
        if contract.saturation_effort is not None:
            contract.saturation_effort = _permute(contract.saturation_effort)
        contract.joint_ids_map = list(range(n))
        log.info(
            "PolicyRunner.load: normalized deploy_contract to Isaac Lab "
            "USD articulation order for SKU %r (%d joints; "
            "permutation=%s)",
            sku, n, perm,
        )
        return True

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
        """Return the correct InferenceEngine subclass for *fmt*."""
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
        """Patch the env's reset/default qpos to match the bundle's training pose.

        R6 invariant: the bundle's ``robot.default_joint_pos`` is in BUNDLE
        order; permute into qpos order before writing.

        Resolution order for the write surface:
          1. ``env.set_default_qpos(values_in_qpos_order)``
          2. ``adapter.set_default_qpos(...)``
          3. Legacy ``_default_qpos_full`` / ``_default_qpos`` write
          4. Loud warning naming the env type
        """
        # Prefer the deploy_contract's default_joint_pos (post-normalize,
        # this is always in bundle order = Isaac Lab USD articulation
        # order, matches bundle_space.joints). The legacy
        # ``raw_manifest['robot']['default_joint_pos']`` is only consulted
        # for bundles without a contract — but its order may not line up
        # with the normalized bundle_space, so contract is the authority.
        raw_default: list = []
        try:
            contract = bundle.deploy_contract
        except Exception:
            contract = None
        if contract is not None:
            try:
                raw_default = list(contract.default_joint_pos or [])
            except Exception:
                raw_default = []
        if not raw_default:
            raw_default = list(
                ((bundle.raw_manifest or {}).get("robot", {}) or {}).get(
                    "default_joint_pos"
                ) or []
            )
        if not raw_default:
            return

        try:
            default_in_bundle = np.asarray(raw_default, dtype=np.float32)
        except Exception:
            return

        if default_in_bundle.size == 0 or qpos_space is None:
            return

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

        # Step 3: legacy private-attribute path (try adapter first).
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

        # Step 4: loud warning.
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
        """Force MuJoCo's physics timestep to match contract.sim_dt (R3).

        Without this, every physics substep advances by the wrong amount
        (MuJoCo default 0.002 vs Isaac Lab training 0.005) — a 2.5× mismatch
        that compresses every learned torque response in time.
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
        """Construct a PDController for an IL bundle from deploy_contract.

        RELEASE Phase 3 ports only the contract path. Legacy env.yaml
        fallback (DEMO Tier 2) is dropped; bundles must ship a contract.
        """
        try:
            contract = bundle.deploy_contract
        except ValueError as exc:
            log.error(
                "PolicyRunner._maybe_build_il_pd_controller: bundle has a "
                "deploy_contract section but it failed validation (%s).",
                exc,
            )
            contract = None
        if contract is not None and contract.stiffness:
            try:
                from application.service.runtime.simulation.pd_controller import (
                    PDController,
                )
                return PDController.from_deploy_contract(
                    contract, bundle.joint_names
                )
            except ValueError as exc:
                log.error(
                    "PolicyRunner._maybe_build_il_pd_controller: "
                    "from_deploy_contract failed (%s).",
                    exc,
                )
        return None

    @staticmethod
    def _build_motor_pd_fallback(
        bundle: "CheckpointBundle",
        env: SimEnvContext,
    ) -> Optional[Any]:
        """Build a fallback PDController for MuJoCo motor-actuator envs.

        Only triggers when contract path returned None AND action_type
        is ``joint_position`` AND every actuator is a basic motor.
        Uses Kp=70, Kd=1.5 — UnitreeGymEnv's training defaults.
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

        try:
            import mujoco as _mj  # noqa: F401
            for ai in range(int(mj_model.nu)):
                if int(mj_model.actuator_trntype[ai]) != 0:  # mjTRN_JOINT
                    return None
        except Exception:
            return None

        n = int(bundle.action_dim)
        _KP = 70.0
        _KD = 1.5

        try:
            gear = np.asarray(mj_model.actuator_gear[:n, 0], dtype=np.float32)
            gear = np.where(gear > 0, gear, 1.0)
            ctrlrange = np.asarray(mj_model.actuator_ctrlrange[:n, 1], dtype=np.float32)
            effort = np.abs(ctrlrange) * gear
        except Exception:
            effort = np.full(n, 33.5, dtype=np.float32)

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
            from application.service.runtime.simulation.pd_controller import (
                PDController,
            )
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
        before any PD / actuator conversion. Splits obs→policy from
        action→torque so PD torque can refresh each physics substep
        (matches Isaac Lab's ImplicitActuator).
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
        """Convert a (frozen) policy output into a fresh ctrl-space torque."""
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
        """Legacy single-shot step: predict + apply in one call."""
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

        INVARIANT (rule_4): the decimation loop is owned HERE, in
        PolicyRunner.run_episode. The policy is queried ONCE per outer
        step; the inner loop runs `bundle.decimation` MuJoCo physics
        substeps with a re-computed PD torque each substep.
        env.sim_step() MUST execute exactly ONE mj_step per call.
        """
        if not self._loaded:
            raise RuntimeError(
                "PolicyRunner.run_episode() called before load(). "
                "Call load(bundle_path, env) first."
            )

        assert self._bundle is not None

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
        engine_type = type(self._engine).__name__.replace("Engine", "").lower()
        termination_reason = "completed"
        safety_status = "ok"
        safety_stop = False

        try:
            for _ in range(max_steps):
                # ── Policy decision (control rate, ~50 Hz) ─────────────
                live_command = _current_command()
                policy_output = self._predict_policy(
                    env.mj_data, last_action, command=live_command
                )

                # INVARIANT I5 (R2): last_action MUST be the RAW denormalized
                # network output — before action_scale, before clip, before
                # default_joint_pos offset, before PD. The policy was trained
                # to see its own previous raw output as part of the obs. Do
                # not let any post-PD value flow into this slot.
                last_action = policy_output

                # ── Per-step telemetry ────────────────────────────────
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
                            "PolicyRunner.run_episode: step_telemetry_fn raised"
                        )

                # Per-step safety check
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
                for _ in range(self._bundle.decimation):
                    fresh_torque = self._apply_policy_output(
                        policy_output, env.mj_data
                    )
                    self._write_ctrl(env.mj_data, fresh_torque)
                    env.sim_step()
                    if render:
                        env.render()

                steps_run += 1

                # First-step kinematics diagnostic (one-shot per class).
                # Anchors "limbs too fast" reports to a concrete number:
                #   max |joint_vel| < ~1.5 rad/s + base_lin_vel ≈ 0 → standing
                #   max |joint_vel| > 5 rad/s at step 1 → physics / PD bug
                #   base_lin_vel > 0.5 m/s at step 1 → a command leaked in
                if steps_run == 1 and not getattr(
                    type(self), "_logged_first_step_kin", False
                ):
                    try:
                        qpos = np.asarray(
                            env.mj_data.qpos, dtype=np.float32
                        ).flatten()
                        qvel = np.asarray(
                            env.mj_data.qvel, dtype=np.float32
                        ).flatten()
                        n_act = int(self._bundle.action_dim)
                        joint_qpos = (
                            qpos[7 : 7 + n_act] if qpos.size > 7 else qpos
                        )
                        joint_qvel = (
                            qvel[6 : 6 + n_act] if qvel.size > 6 else qvel
                        )
                        log.info(
                            "PolicyRunner.run_episode: after first step · "
                            "base_z=%.4f · max |joint_vel|=%.3f rad/s · "
                            "base_lin_vel=[%.3f %.3f %.3f] m/s",
                            float(qpos[2]) if qpos.size > 2 else float("nan"),
                            float(np.max(np.abs(joint_qvel)))
                            if joint_qvel.size else 0.0,
                            float(qvel[0]) if qvel.size > 0 else 0.0,
                            float(qvel[1]) if qvel.size > 1 else 0.0,
                            float(qvel[2]) if qvel.size > 2 else 0.0,
                        )
                        log.info(
                            "PolicyRunner.run_episode: after first step · "
                            "qpos[7:7+n]=%s",
                            ["%.3f" % v for v in joint_qpos.tolist()],
                        )
                        log.info(
                            "PolicyRunner.run_episode: after first step · "
                            "qvel[6:6+n]=%s",
                            ["%.3f" % v for v in joint_qvel.tolist()],
                        )
                    except Exception:
                        pass
                    type(self)._logged_first_step_kin = True

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
            # Zero ctrl on every exit path so a downstream loop that keeps
            # calling mj_step doesn't re-apply frozen torques forever.
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
    # Telemetry (best-effort; ``telemetry_fn`` injection is the only path)
    # ------------------------------------------------------------------

    def _emit_telemetry(self, result: "EpisodeResult") -> bool:
        """Emit a lightweight summary telemetry event for the episode.

        Uses the injected ``telemetry_fn`` when provided. RELEASE Phase 3
        does not wire the DEMO ``system.service.telemetry`` fallback;
        without an injection this method is a no-op that returns False.
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

        if self._telemetry_fn is None:
            return False
        try:
            self._telemetry_fn(payload)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _write_ctrl(mj_data: Any, action: np.ndarray) -> None:
        """Write *action* into mj_data.ctrl. Prefix-write convention."""
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

    # ------------------------------------------------------------------
    # Convenience: build runner directly from a bundle path
    # ------------------------------------------------------------------

    @classmethod
    def load_runner(
        cls,
        bundle_path: Path,
        env: SimEnvContext,
        *,
        robot_sku: Optional[str] = None,
    ) -> "PolicyRunner":
        """Convenience — construct + load() in one call.

        Pass ``robot_sku=`` when the caller knows the canonical SKU
        (Studio UI, training-time round-trip). When omitted, the SKU is
        read from ``manifest.robot.sku``; missing → IncompatibleWeightError.
        """
        runner = cls()
        runner.load(bundle_path, env, robot_sku=robot_sku)
        return runner
