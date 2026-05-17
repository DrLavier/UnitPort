"""BundleExporter — write v1 policy bundles to ``<project>/training/exported/``.

Phase 3 RELEASE port — minimal subset of DEMO ``bundle_exporter.py`` (1077 LoC)
focused on the format-side responsibilities:

* Compose a v1 ``manifest.yaml`` (12 required fields) optionally carrying a
  ``deploy_contract`` block, ``normalization`` reference, and arbitrary
  metadata blocks (algorithm, amp_metadata, source).
* Write ``policy.onnx`` (caller supplies bytes / source path).
* Write ``source.json`` lineage stub.
* Atomic move from staging dir to
  ``<project>/training/exported/<backend_id>/<name>/``.

Two SB3-specific paths from DEMO are intentionally dropped here:
``_build_ppo_onnx_net`` / ``_build_sac_onnx_net`` and the
``export_bundle(model, spec, ...)`` end-to-end pipeline. Those depend on
``stable_baselines3`` model objects + ``TrainingJobSpec`` and land alongside
the SB3 trainer port in Phase 4. The Phase 3 round-trip uses
``export_from_artifacts(...)`` which takes already-prepared ONNX + manifest
parts directly.

I/O discipline:

* Bundles MUST live inside the active project; unbound export is rejected
  so nothing leaks into ``Paths.USER_CONFIG_DIR`` (RELEASE/CLAUDE.md §1.4).
* Manifest YAML uses ``yaml.safe_dump(..., sort_keys=False)`` so
  ``deploy_contract.observations`` insertion order is preserved on disk
  (the same constraint BundleLoader relies on at read time).
* ``policy.onnx`` is binary and written via the staging dir + ``os.replace``
  atomic-rename pattern (temp dir is created adjacent to the final target).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from unitport_sdk import log_warning

from application.training.manifest_schema import (
    ManifestValidationError,
    validate_manifest,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """Returned by ``BundleExporter.export_from_artifacts``."""
    bundle_path: Path                # <project>/training/exported/<backend_id>/<name>/
    manifest_path: Path              # bundle_path / "manifest.yaml"
    policy_path: Path                # bundle_path / "policy.onnx"
    source_path: Path                # bundle_path / "source.json"

    def __str__(self) -> str:
        return str(self.bundle_path)


# ---------------------------------------------------------------------------
# Helpers (verbatim subset of DEMO bundle_exporter helpers)
# ---------------------------------------------------------------------------

def _identity_norm_stats(dim: int, clip: Optional[float] = None) -> dict:
    """Return a no-op normalization payload for a vector of *dim* values."""
    if dim <= 0:
        raise ValueError(f"Normalization dim must be positive, got {dim}")
    clip_val = float(clip) if clip is not None else None
    return {
        "mean": [0.0] * int(dim),
        "std": [1.0] * int(dim),
        "clip_min": (-clip_val if clip_val is not None else None),
        "clip_max": clip_val,
    }


# ---------------------------------------------------------------------------
# VecNormalize stats extraction
# ---------------------------------------------------------------------------


def _extract_vec_normalize_stats(vec_env: Any, *, obs_dim: int) -> Optional[Dict[str, Any]]:
    """Walk an SB3 VecEnv wrapper chain and snapshot VecNormalize obs stats.

    Returns a JSON-serialisable payload in the shape PolicyRunner expects::

        {"observation": {"mean": [...], "std": [...],
                          "clip_min": -clip, "clip_max": clip}}

    Returns ``None`` when no VecNormalize is found (e.g. canvas had
    ``obs_normalize=False``) — caller is responsible for omitting the
    manifest ``normalization`` block in that case.
    """
    if vec_env is None:
        return None
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except Exception:
        return None

    # Walk down ``venv`` references — VecNormalize wraps a (Dummy|Subproc)VecEnv
    # which may itself wrap something else.
    layer: Any = vec_env
    seen: set = set()
    norm_layer: Optional[Any] = None
    while layer is not None and id(layer) not in seen:
        seen.add(id(layer))
        if isinstance(layer, VecNormalize):
            norm_layer = layer
            break
        layer = getattr(layer, "venv", None)
    if norm_layer is None:
        return None

    try:
        import numpy as np
        obs_rms = getattr(norm_layer, "obs_rms", None)
        if obs_rms is None:
            return None
        mean = np.asarray(obs_rms.mean, dtype=np.float64).reshape(-1).tolist()
        var = np.asarray(obs_rms.var, dtype=np.float64).reshape(-1)
        std = np.sqrt(np.maximum(var, 1e-12)).tolist()
        if len(mean) != obs_dim or len(std) != obs_dim:
            log_warning(
                f"[bundle_exporter] VecNormalize stats dim {len(mean)} != "
                f"manifest obs_dim {obs_dim}; skipping normalization round-trip"
            )
            return None
        clip = float(getattr(norm_layer, "clip_obs", 10.0))
        return {
            "observation": {
                "mean": mean,
                "std": std,
                "clip_min": -clip,
                "clip_max": clip,
            }
        }
    except Exception as exc:                                  # pragma: no cover
        log_warning(f"[bundle_exporter] VecNormalize stats extract failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# SB3 deploy_contract construction
# ---------------------------------------------------------------------------


def _build_sb3_deploy_contract(
    spec: Any,
    *,
    action_dim: int,
    obs_dim: int,
    joint_ir_roles: List[str],
    sim_dt: float,
    control_dt: float,
) -> Dict[str, Any]:
    """Compose a schema-v1 deploy_contract for an SB3-trained bundle.

    Phase 5 IR-only joint contract: ``joint_sdk_names`` carries IR role
    names; the deploy stack resolves to physical names via ``robot_sku``.
    Per-joint scalar lists (stiffness / damping / effort_limit) broadcast
    the scalar canvas knobs from ``spec.actor.actuator``. ``default_joint_pos``
    is translated from ``spec.actor.joint_init`` (IR-keyed → IR-order list)
    so the deploy spawn pose matches the policy's training distribution.

    The observation block mirrors :class:`GenericMujocoEnv._build_obs`
    layout: ``base_ang_vel(3) + projected_gravity(3) + joint_pos(N) +
    joint_vel(N) + last_action(N) + commands(3)``. Per-component clip
    is taken from ``spec.obs_action.obs_clip_range`` when non-zero.
    """
    n = int(action_dim)
    actor = getattr(spec, "actor", None)
    actuator = getattr(actor, "actuator", None) if actor is not None else None
    obs_action = getattr(spec, "obs_action", None)

    stiffness_val = float(getattr(actuator, "stiffness", 25.0) or 25.0) if actuator else 25.0
    damping_val = float(getattr(actuator, "damping", 0.5) or 0.5) if actuator else 0.5
    effort_val = float(getattr(actuator, "effort_limit", 30.0) or 30.0) if actuator else 30.0
    velocity_val = float(getattr(actuator, "velocity_limit", 0.0) or 0.0) if actuator else 0.0

    stiffness_arr = [stiffness_val] * n
    damping_arr = [damping_val] * n
    effort_arr = [effort_val] * n

    # default_joint_pos in IR-role order. spec.actor.joint_init is an
    # IR-keyed dict; missing roles fall to zero so the deploy spawn
    # mirrors the training default (env's _default_qpos_actuated is also
    # zeros for unspecified joints).
    joint_init_raw = getattr(actor, "joint_init", None) if actor is not None else None
    joint_init: Dict[str, float] = {}
    if isinstance(joint_init_raw, dict):
        for k, v in joint_init_raw.items():
            try:
                joint_init[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    default_joint_pos = [float(joint_init.get(role, 0.0)) for role in joint_ir_roles]

    # Timing — recompute step_dt from decimation * sim_dt so the strict
    # consistency check (decimation == round(step_dt/sim_dt)) cannot trip
    # on floating-point drift from canvas control_dt.
    sim_dt_f = float(sim_dt) if sim_dt and float(sim_dt) > 0.0 else 2e-3
    control_dt_f = float(control_dt) if control_dt and float(control_dt) > 0.0 else 0.02
    decimation = max(1, int(round(control_dt_f / max(sim_dt_f, 1e-6))))
    step_dt = decimation * sim_dt_f

    # Action knobs from canvas. scale is broadcastable scalar so
    # ActionApplier reads it from either deploy_contract.action.scale or
    # the mirrored action_space.scale. clip is None at deploy_contract
    # level (per-joint list-of-pairs is overkill for SB3's scalar canvas
    # knob); the scalar mirror lands on manifest.action_space.clip
    # outside this function.
    action_scale = (
        float(getattr(obs_action, "action_scale", 1.0) or 1.0) if obs_action else 1.0
    )
    use_default_offset = bool(
        getattr(actor, "action_use_default_offset", True)
    ) if actor is not None else True
    action_block: Dict[str, Any] = {
        "scale": action_scale,
        "clip": None,
        "offset_mode": "default_joint_pos" if use_default_offset else "zero",
    }

    # Observations — mirror GenericMujocoEnv._build_obs layout exactly.
    obs_clip = (
        float(getattr(obs_action, "obs_clip_range", 0.0) or 0.0) if obs_action else 0.0
    )
    obs_clip_pair: Optional[List[float]] = (
        [-obs_clip, obs_clip] if obs_clip > 0.0 else None
    )

    def _obs_term(dim: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {"dim": int(dim), "scale": 1.0, "history_length": 1}
        if obs_clip_pair is not None:
            out["clip"] = list(obs_clip_pair)
        return out

    observations: Dict[str, Any] = {
        "base_ang_vel": _obs_term(3),
        "projected_gravity": _obs_term(3),
        "joint_pos": _obs_term(n),
        "joint_vel": _obs_term(n),
        "last_action": _obs_term(n),
        "commands": _obs_term(3),
    }

    contract: Dict[str, Any] = {
        "schema_version": 1,
        "robot_sku": str(getattr(spec.robot, "sku", "") or ""),
        "joint_sdk_names": list(joint_ir_roles),
        "joint_ids_map": list(range(n)),
        "stiffness": stiffness_arr,
        "damping": damping_arr,
        "effort_limit": effort_arr,
        "default_joint_pos": default_joint_pos,
        "sim_dt": sim_dt_f,
        "step_dt": step_dt,
        "decimation": decimation,
        "observations": observations,
        "action": action_block,
        "commands": {},
        "base_body_name": "",
    }
    if velocity_val > 0.0:
        contract["velocity_limit"] = [velocity_val] * n

    # init_base_pos: only carry when the canvas set a non-default spawn
    # height. init_pos_z == 0 means "fall back to RobotSpec.target_height"
    # in the training env; we mirror that fallback by omitting the field.
    init_z = float(getattr(actor, "init_pos_z", 0.0) or 0.0) if actor is not None else 0.0
    if init_z > 0.0:
        init_x = float(getattr(actor, "init_pos_x", 0.0) or 0.0)
        init_y = float(getattr(actor, "init_pos_y", 0.0) or 0.0)
        contract["init_base_pos"] = [init_x, init_y, init_z]

    return contract


# ---------------------------------------------------------------------------
# BundleExporter
# ---------------------------------------------------------------------------

class BundleExporter:
    """Phase 3 minimal exporter — writes v1 bundles from prepared artifacts."""

    @staticmethod
    def build_manifest(
        *,
        name: str,
        version: str,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        control_frequency_hz: float,
        decimation: int,
        robot_sku: str,
        joint_names: List[str],
        policy_file: str = "policy.onnx",
        policy_format: str = "onnx",
        deploy_contract: Optional[Dict[str, Any]] = None,
        normalization: Optional[Dict[str, Any]] = None,
        algorithm: str = "",
        amp_metadata: Optional[Dict[str, Any]] = None,
        inference_convention: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compose a v1 manifest dict from explicit fields.

        ``robot_sku`` is the **only** robot identifier required from the
        caller; the registered ``brand`` / ``model`` / ``name`` are looked
        up via ``registers.robots.get_robot_spec(sku)`` and written into
        the manifest as informational fields (the deploy stack ignores
        them — see manifest_schema.py module docstring).

        All required ``MANIFEST_REQUIRED_FIELDS`` are populated; the
        result is passed through ``validate_manifest`` so callers see
        structural errors at compose time, not at first BundleLoader.load.
        """
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"obs_dim/action_dim must be positive (got {obs_dim}/{action_dim})"
            )
        if control_frequency_hz <= 0 or decimation <= 0:
            raise ValueError(
                f"control_frequency_hz/decimation must be positive "
                f"(got {control_frequency_hz}/{decimation})"
            )
        if len(joint_names) != action_dim:
            raise ValueError(
                f"len(joint_names)={len(joint_names)} != action_dim={action_dim}"
            )
        if not isinstance(robot_sku, str) or not robot_sku.strip():
            raise ValueError(
                "BundleExporter.build_manifest: robot_sku is required "
                "(the canonical key from registers.robots; lowering must "
                "produce a registered robot)"
            )
        from registers.robots import get_robot_spec, resolve_id
        canonical_sku = resolve_id(robot_sku.strip()) or robot_sku.strip()
        rs = get_robot_spec(canonical_sku)
        if rs is None:
            raise ValueError(
                f"BundleExporter.build_manifest: robot_sku {robot_sku!r} "
                f"(canonical {canonical_sku!r}) is not registered. "
                f"Lowering produced an unregistered robot — fix at the "
                f"source before exporting."
            )

        manifest: Dict[str, Any] = {
            "name": str(name),
            "version": str(version),
            "policy": {
                "file": str(policy_file),
                "format": str(policy_format),
            },
            "observation_space": {
                "dim": int(obs_dim),
            },
            "action_space": {
                "dim": int(action_dim),
                "type": str(action_type),
            },
            "runtime": {
                "control_frequency_hz": float(control_frequency_hz),
                "decimation": int(decimation),
            },
            "robot": {
                "sku": canonical_sku,
                # Informational only — derived from the registry; the
                # deploy stack NEVER consumes these. They live in the
                # YAML so a human grepping artifacts can read which
                # robot a bundle targets.
                "brand": rs.brand,
                "model": rs.model,
                "name": rs.name,
                "num_joints": int(action_dim),
                "joint_names": list(joint_names),
            },
        }

        if deploy_contract is not None:
            # Forward as-is — caller is responsible for the dict already
            # being yaml-dumpable. ObsTermSpec.to_dict / DeployContract.to_dict
            # both produce yaml-friendly output.
            manifest["deploy_contract"] = deploy_contract

            # Mirror the contract's action.scale / action.clip into the
            # top-level ``action_space`` block so legacy ActionApplier
            # readers see them too. Without this, an IL bundle with a
            # 0.25× action scale buried only in deploy_contract.action
            # quietly degrades to scale=1.0 at deploy, producing 4×
            # over-amplified joint targets and PD wind-up. See plan
            # release-demo-sim2sim-policy-...md §1.
            contract_action = (
                deploy_contract.get("action")
                if isinstance(deploy_contract, dict) else None
            )
            if isinstance(contract_action, dict):
                top_action = manifest["action_space"]
                if "scale" not in top_action and contract_action.get("scale") is not None:
                    top_action["scale"] = contract_action["scale"]
                if "clip" not in top_action and contract_action.get("clip") is not None:
                    top_action["clip"] = contract_action["clip"]

        if normalization is not None:
            manifest["normalization"] = normalization

        if algorithm:
            manifest["algorithm"] = str(algorithm)

        if inference_convention:
            # IL bundles MUST carry this so deploy-side ObsBuilder rotates
            # base_lin_vel / base_ang_vel from MuJoCo world frame into the
            # robot body frame (Isaac Lab training convention). Absence
            # silently degrades to SB3 world-frame semantics and produces
            # the classic "robot drifts + joints twitch with no command"
            # sim2sim failure mode. See plan
            # release-demo-sim2sim-policy-a-unitport-a-nifty-micali.md.
            manifest["inference_convention"] = str(inference_convention)

        if amp_metadata is not None:
            manifest["amp_metadata"] = dict(amp_metadata)

        if extra:
            for k, v in extra.items():
                if k in manifest:
                    raise ValueError(
                        f"BundleExporter.build_manifest: extra key {k!r} "
                        f"would clobber an existing manifest field"
                    )
                manifest[k] = v

        validate_manifest(manifest)
        return manifest

    @staticmethod
    def export_from_artifacts(
        *,
        name: str,
        version: str,
        manifest: Dict[str, Any],
        onnx_source: Union[bytes, Path, str],
        source_meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        project: Optional[Any] = None,
        backend_id: str = "",
        extra_files: Optional[Dict[str, Union[bytes, str]]] = None,
    ) -> ExportResult:
        """Write a v1 bundle to the current project (or user-global fallback).

        **Backend-agnostic entry (rule B2).** This is the single atomic-write
        sink for both training chains. Composition strategy differs:

          * SB3 chain: ``export_bundle(model, spec, ...)`` builds the
            manifest from ``spec.*`` + ``sb3_onnx_export.export_to_onnx_bytes``
            and forwards into here.
          * Isaac Lab chain:
            ``application.training.isaac_lab.bundle_finalizer.finalize_isaac_lab_bundle``
            takes RSL-RL play.py's ``exported/policy.onnx`` + reads
            ``env.yaml`` via ``manifest_parser.parse_isaac_lab_env_yaml``
            and forwards into here.

        Both callers pass ``backend_id`` explicitly so cross-process
        AppSignals defaults can never silently misroute the bundle.

        When a ``ProjectInfo`` is bound (passed via ``project=`` or resolved
        via ``current_project_info()``), the bundle lands at
        ``<project>/training/exported/<backend_id>/<name>/`` so on-disk
        artifacts are partitioned by training backend (matching the
        runs/ layout). ``backend_id`` defaults to
        ``AppSignals.current_backend()`` (the canvas-bound backend); if
        that is also empty we fall back to ``"unknown"``.

        Unbound export is rejected — when no project is open the call raises
        ``RuntimeError`` rather than leaking bundles into the user-config tree
        (RELEASE/CLAUDE.md §1.4).

        Parameters
        ----------
        name:
            Bundle directory name. Lands under
            ``<project>/training/exported/<backend_id>/<name>/``.
        version:
            Bundle version (mirrored into ``source.json``).
        manifest:
            Already-composed v1 manifest dict (use ``build_manifest`` to
            compose). The dict is validated again at write time.
        onnx_source:
            Either raw bytes or a Path/string pointing at an existing
            ``.onnx`` file. Copied verbatim into the bundle.
        source_meta:
            Optional ``source.json`` body. Defaults to a minimal
            ``{"type": "training", "name": ..., "version": ...,
              "created_utc": ...}`` payload.
        overwrite:
            When False (default) and the destination bundle dir already
            exists, raise FileExistsError. When True, the dir is wiped
            before the staged copy.

        Returns
        -------
        ExportResult with absolute paths to the written files.
        """
        validate_manifest(manifest)

        # Resolve target dir — when a project is bound the bundle lands at
        # ``<project>/training/exported/<backend_id>/<name>/`` (pure path
        # math, no index file involved). The Policies dropdown discovers
        # bundles by scanning that dir, so this on-disk layout IS the
        # source of truth.
        proj = project
        if proj is None:
            try:
                from application.service.projects import current_project_info
                proj = current_project_info()
            except Exception:
                proj = None
        if proj is None:
            raise RuntimeError(
                "Bundle export requires an open project — refusing to write "
                "outside the project tree."
            )

        # backend_id resolution: explicit arg wins; fall back to the
        # canvas-bound backend in AppSignals; finally "unknown".
        bid = (backend_id or "").strip()
        if not bid:
            try:
                from application.service.signals import current_backend
                bid = (current_backend() or "").strip()
            except Exception:
                bid = ""
        if not bid:
            bid = "unknown"

        target_dir = proj.path / "training" / "exported" / bid / name

        if target_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Bundle dir already exists: {target_dir} "
                    f"(pass overwrite=True to replace)"
                )

        # Stage in a temp dir adjacent to the final location so the rename
        # is on the same filesystem (cheap, atomic).
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f"unitport_bundle_{name}_",
            dir=str(target_dir.parent),
        ))

        try:
            # 1) policy.onnx
            policy_dst = staging / str(manifest["policy"]["file"])
            policy_dst.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(onnx_source, (bytes, bytearray)):
                policy_dst.write_bytes(bytes(onnx_source))
            else:
                src = Path(onnx_source)
                if not src.exists():
                    raise FileNotFoundError(f"ONNX source not found: {src}")
                shutil.copyfile(src, policy_dst)

            # 2) manifest.yaml — sort_keys=False so insertion order is
            #    preserved (deploy_contract.observations relies on this).
            manifest_dst = staging / "manifest.yaml"
            manifest_dst.write_text(
                yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            # 3) source.json — lineage stub
            if source_meta is None:
                source_meta = {
                    "type": "training",
                    "name": str(name),
                    "version": str(version),
                }
            payload = dict(source_meta)
            payload.setdefault("created_utc", time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ))
            source_dst = staging / "source.json"
            source_dst.write_text(
                json.dumps(payload, indent=2, sort_keys=False),
                encoding="utf-8",
            )

            # 3.5) Optional extra files (e.g. normalization.json for SB3
            # VecNormalize stats round-trip). Each entry is written into
            # the staging dir at its relative path; the bundle's manifest
            # is responsible for referencing them by file name.
            if extra_files:
                for rel_path, content in extra_files.items():
                    rel_clean = str(rel_path).strip().lstrip("/\\")
                    if not rel_clean:
                        continue
                    extra_dst = staging / rel_clean
                    extra_dst.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, (bytes, bytearray)):
                        extra_dst.write_bytes(bytes(content))
                    else:
                        extra_dst.write_text(str(content), encoding="utf-8")

            # 4) Atomic move into place
            if target_dir.exists() and overwrite:
                shutil.rmtree(target_dir)
            os.replace(str(staging), str(target_dir))
            staging = None  # don't try to clean it up below

        finally:
            if staging is not None and staging.exists():
                try:
                    shutil.rmtree(staging)
                except Exception:
                    pass

        return ExportResult(
            bundle_path=target_dir,
            manifest_path=target_dir / "manifest.yaml",
            policy_path=target_dir / str(manifest["policy"]["file"]),
            source_path=target_dir / "source.json",
        )


    # ------------------------------------------------------------------
    # Stage 7 — end-to-end wrapper (model + spec → bundle on disk)
    # ------------------------------------------------------------------

    @staticmethod
    def export_bundle(
        model: Any,
        spec: Any,
        run_id: str,
        *,
        bundle_name: Optional[str] = None,
        version: Optional[str] = None,
        overwrite: Optional[bool] = None,
        project: Optional[Any] = None,
        backend_id: str = "",
        vec_env: Any = None,
    ) -> ExportResult:
        """SB3-only end-to-end export: trained SB3 model → v1 bundle on disk.

        **Backend boundary (rule B1):** ``model`` MUST be a
        ``stable_baselines3`` PPO / SAC / TD3 instance — internally this
        function calls ``sb3_onnx_export.export_to_onnx_bytes(model, obs_dim)``
        which extracts SB3's actor network. Other backends (Isaac Lab,
        future Newton, ...) MUST NOT call this entry. They compose their
        own manifest dict + obtain ONNX bytes / path through their own
        toolchain (e.g. Isaac Lab uses RSL-RL's stock play.py to produce
        ``exported/policy.onnx`` and then calls
        :meth:`export_from_artifacts` directly via
        ``application.training.isaac_lab.bundle_finalizer``).

        Pipeline:
            1. ``sb3_onnx_export.export_to_onnx_bytes`` — actor → ONNX bytes.
            2. ``BundleExporter.build_manifest`` — derive 12 required fields
               from ``spec.robot`` / ``spec.obs_action`` / ``spec.physics``.
            3. ``BundleExporter.export_from_artifacts`` — atomic write to
               ``<project>/training/exported/<backend_id>/<name>/`` (or the
               user-global fallback when no project is bound).

        Args:
            model: trained SB3 PPO/SAC/TD3 model. **Not** a ``.pt`` path —
                this entry refuses non-SB3 callers by virtue of the
                ``export_to_onnx_bytes`` contract.
            spec: :class:`TrainingSpec` consumed for manifest fields +
                obs/action contract.
            run_id: training run id (mirrored into ``source.json``).
            bundle_name: defaults to ``spec.export.bundle_name`` (with the
                ``"<NEW>"`` sentinel rejected — see canva_todo R6).
            version: defaults to ``spec.export.version``.
            overwrite: defaults to ``spec.export.overwrite``.
            backend_id: partition under ``training/exported/`` — typically
                ``"sb3_mujoco"`` for this entry. Caller (sb3_entry) passes
                explicitly so cross-process AppSignals defaults can't drift.

        Returns:
            :class:`ExportResult` pointing at the on-disk bundle.

        Raises:
            ValueError: bundle_name is the ``<NEW>`` sentinel, robot is
                unresolved, or ONNX export fails for the algorithm.
        """
        if spec is None or not getattr(spec, "robot", None):
            raise ValueError("export_bundle: spec.robot missing")
        if not spec.robot.sku:
            raise ValueError("export_bundle: spec.robot.sku is empty")

        export_cfg = getattr(spec, "export", None)
        if bundle_name is None:
            bundle_name = (
                getattr(export_cfg, "bundle_name", None) or ""
            ).strip()
        if not bundle_name or bundle_name == "<NEW>":
            raise ValueError(
                "export_bundle: bundle_name is required (got '<NEW>' sentinel "
                "or empty). Set ExportNode.bundle_name on the canvas."
            )
        if version is None:
            version = str(getattr(export_cfg, "version", "v1") or "v1")
        if overwrite is None:
            overwrite = bool(getattr(export_cfg, "overwrite", True))

        # ── Resolve obs/action dims ──
        obs_dim = 0
        action_dim = int(spec.robot.num_joints if hasattr(spec.robot, "num_joints") else len(spec.robot.joint_order))
        oa = getattr(spec, "obs_action", None)
        if oa is not None:
            preset = (getattr(oa, "contract_preset", "") or "").strip()
            if preset and preset != "custom":
                from application.training.obs_contracts import get_obs_contract
                contract = get_obs_contract(preset)
                if contract is not None:
                    obs_dim = int(contract["obs_dim"])
                    action_dim = int(contract["action_dim"])
        if obs_dim <= 0:
            # Fall back to env-shape inference: ang_vel(3) + grav(3) + qpos(N)
            # + qvel(N) + last_action(N) + cmd(3); same as GenericMujocoEnv.
            obs_dim = 3 + 3 + action_dim + action_dim + action_dim + 3

        action_type = str(
            getattr(oa, "action_type", "joint_position") or "joint_position"
        )

        # Control freq from physics_config
        physics = getattr(spec, "physics", None)
        sim_dt = float(getattr(physics, "sim_dt", 2e-3) or 2e-3) if physics else 2e-3
        control_dt = (
            float(getattr(physics, "control_dt", 0.02) or 0.02)
            if physics else 0.02
        )
        control_freq_hz = 1.0 / max(control_dt, 1e-6)
        decimation = max(1, int(round(control_dt / max(sim_dt, 1e-6))))

        # ── ONNX export ──
        from application.training.sb3_onnx_export import export_to_onnx_bytes

        onnx_bytes = export_to_onnx_bytes(model, obs_dim)

        # ── Compose manifest ──
        # Phase 5 IR-only contract: manifest.robot.joint_names = IR roles
        # (the substrate-side translation happens deploy-time via
        # JointIRResolver(robot_sku) lookup so the same bundle can be
        # deployed across same-family robots with different physical names).
        joint_names = list(spec.robot.joint_ir_roles or [])
        if not joint_names:
            # Legacy spec with no joint_ir_roles populated → fall back to
            # joint_order so the manifest is at least self-consistent on
            # length. Should not occur in Phase 5+ flows.
            joint_names = list(spec.robot.joint_order or [])
        if len(joint_names) != action_dim:
            # Re-pad / re-trim — manifest validator requires exact match.
            if len(joint_names) < action_dim:
                joint_names = joint_names + [
                    f"joint_{i}" for i in range(len(joint_names), action_dim)
                ]
            else:
                joint_names = joint_names[:action_dim]

        algorithm_name = (
            type(model).__name__.upper()
            if model is not None
            else (spec.algorithm.algorithm or "PPO").upper()
        )

        # Build a full deploy_contract carrying every numeric the SB3
        # MuJoCo env applied during training. Without this the deploy
        # stack silently defaults action_scale to 1.0 / no clipping / IL
        # heuristic init pose — the canonical sim2sim "twitching" failure
        # mode (see project_release_action_scale_top_level_missing memory).
        # CheckpointBundle.deploy_contract is strict-only: it either
        # accepts a full schema-v1 dict or raises, so a partial dict is
        # worse than none.
        deploy_contract = _build_sb3_deploy_contract(
            spec,
            action_dim=action_dim,
            obs_dim=obs_dim,
            joint_ir_roles=joint_names,
            sim_dt=sim_dt,
            control_dt=control_dt,
        )

        manifest = BundleExporter.build_manifest(
            name=bundle_name,
            version=version,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_type=action_type,
            control_frequency_hz=control_freq_hz,
            decimation=decimation,
            robot_sku=str(getattr(spec.robot, "sku", "") or ""),
            joint_names=joint_names,
            policy_file="policy.onnx",
            policy_format="onnx",
            algorithm=algorithm_name,
            deploy_contract=deploy_contract,
        )

        # build_manifest mirrors deploy_contract.action.scale into
        # action_space.scale. We additionally write a scalar
        # action_space.clip — the per-joint deploy_contract.action.clip
        # is left None (the canvas exposes a single scalar clip range, not
        # per-joint), so without this top-level the bundle would lose
        # action-clip enforcement entirely at ActionApplier-time.
        oa_for_clip = getattr(spec, "obs_action", None)
        scalar_action_clip = (
            float(getattr(oa_for_clip, "action_clip", 0.0) or 0.0) if oa_for_clip else 0.0
        )
        if scalar_action_clip > 0.0 and "clip" not in manifest["action_space"]:
            manifest["action_space"]["clip"] = scalar_action_clip

        # VecNormalize stats round-trip — when the trainer wrapped the env
        # in VecNormalize, the policy observed normalized inputs during
        # training; the deploy stack must apply the same mean/std or the
        # policy is fed out-of-distribution observations and drifts /
        # twitches. The trainer hands its wrapped vec_env in; we extract
        # the running stats and ship them alongside the bundle as
        # ``normalization.json``, then point the manifest at it so
        # PolicyRunner._load_bundle_normalizer picks them up at load time.
        extra_files: Dict[str, Union[bytes, str]] = {}
        norm_payload = _extract_vec_normalize_stats(vec_env, obs_dim=obs_dim)
        if norm_payload is not None:
            extra_files["normalization.json"] = json.dumps(
                norm_payload, indent=2, sort_keys=False
            )
            manifest["normalization"] = {"file": "normalization.json"}

        source_meta = {
            "type": "training",
            "name": bundle_name,
            "version": version,
            "run_id": str(run_id),
            "algorithm": algorithm_name,
            "robot_sku": spec.robot.sku,
            "robot_name": spec.robot.name,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }

        return BundleExporter.export_from_artifacts(
            name=bundle_name,
            version=version,
            manifest=manifest,
            onnx_source=onnx_bytes,
            source_meta=source_meta,
            overwrite=overwrite,
            project=project,
            backend_id=backend_id,
            extra_files=extra_files or None,
        )


__all__ = [
    "BundleExporter",
    "ExportResult",
    "_identity_norm_stats",
]
