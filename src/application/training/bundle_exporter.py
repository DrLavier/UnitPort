# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
# Per-item obs tail (SB3 per-item-reward canvases)
# ---------------------------------------------------------------------------


def _resolve_per_item_obs(spec: Any) -> Optional[Dict[str, Any]]:
    """Per-item observation-tail spec for the deploy contract, or ``None``.

    When the canvas wires per-item rewards (``rewards.terms_by_item``), the SB3
    env exposes a tail ``[cmd_norm, soft_weight per item]`` on the policy obs
    (``GenericMujocoEnv._expose_active_item_obs``) — so the trained network's
    input is ``base_obs + (1 + n_items)``. The deployed actor therefore needs
    the same tail, which the deploy stack reconstructs from the command via the
    SAME ItemResolver. This returns the data the contract must carry for that
    reconstruction: item order, per-item command_ranges, and the blend width.

    Gating mirrors :func:`application.training.envs.generic_mujoco_env.make_env`
    exactly (same item_id order = ``terms_by_item`` insertion order). Raises
    (§8) if per-item rewards are wired but ``training_items`` is missing — the
    policy was trained with the tail, so a self-contained bundle (§9) MUST be
    able to rebuild it; silently dropping it would ship a wrong-dim contract.
    """
    rewards_cfg = getattr(spec, "rewards", None)
    motion_cfg = getattr(spec, "motion", None)
    raw_per_item = (getattr(rewards_cfg, "terms_by_item", None) or {}) if rewards_cfg else {}
    if not raw_per_item:
        return None
    training_items = (getattr(motion_cfg, "training_items", None) or {}) if motion_cfg else {}
    if not training_items:
        raise ValueError(
            "[bundle_exporter] spec.rewards.terms_by_item is set but "
            "spec.motion.training_items is missing — the trained policy's obs "
            "carries a per-item tail (cmd_norm + item weights) that the deploy "
            "contract cannot reconstruct without each item's command_ranges. "
            "Wire the TrainingMotion node's command envelopes, or remove the "
            "per-item reward edges. (CLAUDE.md §8/§9 — the bundle must be "
            "self-contained.)"
        )
    item_ids = [str(k) for k in raw_per_item.keys()]
    command_ranges: Dict[str, Dict[str, List[float]]] = {}
    for iid in item_ids:
        payload = training_items.get(iid)
        ranges = payload.get("command_ranges") if isinstance(payload, dict) else None
        command_ranges[iid] = {
            str(ch): [float(span[0]), float(span[1])]
            for ch, span in (ranges or {}).items()
            if isinstance(span, (list, tuple)) and len(span) == 2
        }
    blend_width = float(getattr(motion_cfg, "deadzone", 0.10) or 0.10)
    return {
        "item_ids": item_ids,
        "command_ranges": command_ranges,
        "blend_width": blend_width,
    }


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
    frame_stack: int = 1,
) -> Dict[str, Any]:
    """Compose a schema-v2 deploy_contract for an SB3-trained bundle.

    Phase 5 IR-only joint contract: ``joint_sdk_names`` carries IR role
    names; the deploy stack resolves to physical names via ``robot_sku``.
    PD gains are mass-weighted from the RobotNode's canonical (omega_n, zeta)
    via ``mujoco_gain_solver`` (CLAUDE.md §10) — the contract carries
    ``pd_param`` (source) plus per-joint ``mujoco_pd_gains`` / ``mujoco_pd_damping``
    (and mirrors them into ``stiffness`` / ``damping`` for v1 readers).
    ``effort_limit`` is the RobotNode scalar broadcast per joint.
    ``default_joint_pos`` is translated from ``spec.actor.joint_init``
    (IR-keyed → IR-order list) so the deploy spawn pose matches the policy's
    training distribution.

    The observation block mirrors :class:`GenericMujocoEnv._build_obs`
    layout: ``base_ang_vel(3) + projected_gravity(3) + joint_pos(N) +
    joint_vel(N) + last_action(N) + commands(3)``. Per-component clip
    is taken from ``spec.obs_action.obs_clip_range`` when non-zero.
    """
    n = int(action_dim)
    actor = getattr(spec, "actor", None)
    obs_action = getattr(spec, "obs_action", None)

    # CLAUDE.md §1.8 + §10: PD gains are derived from the canonical
    # (omega_n, zeta) on the RobotNode via the mass-weighted MuJoCo solver
    # — never from fabricated Go2-class defaults and never from raw scalar
    # kp/kd. SB3 deploys on MuJoCo, so it uses the SAME effective inertia
    # the IL finalizer / env_cfg compiler use (mjcf nominal stance), giving
    # per-joint kp = m_eff·omega_n² that match what training applied.
    if actor is None:
        raise ValueError(
            "[bundle_exporter] spec.actor is missing — cannot build "
            "deploy_contract for an SB3 bundle without the canvas "
            "ActorSetting node. Wire ActorSetting before exporting. "
            "(CLAUDE.md §1.8 forbids fabricating Go2-class PD defaults.)"
        )

    pd_param = getattr(actor, "pd_param", None)
    if pd_param is None:
        raise ValueError(
            "[bundle_exporter] spec.actor.pd_param is missing — cannot derive "
            "per-joint PD gains. This means the RobotNode has no resolvable "
            "(omega_n, zeta) (typically the bound robot has no declared "
            "families). Complete the Robot Asset card's family selection and "
            "ensure the RobotNode is wired. PD is parameterized only by "
            "(omega_n, zeta) (CLAUDE.md §10) — refusing to substitute scalar "
            "Go2-class defaults that would silently corrupt deploy dynamics."
        )

    # effort_limit / velocity_limit come from the RobotNode (spec.actor.*,
    # populated by spec_compiler._compile_pd_param). velocity 0.0 = no limit.
    eff_raw = getattr(actor, "effort_limit", None)
    if eff_raw is None:
        raise ValueError(
            "[bundle_exporter] spec.actor.effort_limit is missing — set it on "
            "the RobotNode (pd_param_table panel). Refusing a hardcoded "
            "default (CLAUDE.md §1.8)."
        )
    effort_val = float(eff_raw)
    vel_raw = getattr(actor, "velocity_limit", None)
    velocity_val = float(vel_raw) if vel_raw is not None else 0.0

    # Resolve MJCF + physical joint order, then mass-weight the gains.
    sku = str(getattr(spec.robot, "sku", "") or "")
    if not sku:
        raise ValueError(
            "[bundle_exporter] spec.robot.sku is empty — cannot resolve the "
            "MJCF needed for the mass-weighted PD solver (CLAUDE.md §7/§10)."
        )
    from application.service.robot_assets.service import get_robot_asset_service
    from application.training.joint_ir import JointIRResolver
    from application.physics.mujoco_gain_solver import solve as solve_mujoco

    asset = get_robot_asset_service().resolve(sku)
    mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
    if mjcf_path is None or not Path(mjcf_path).is_file():
        raise ValueError(
            f"[bundle_exporter] robot sku={sku!r} has no on-disk MJCF "
            f"(mjcf_path={str(mjcf_path) if mjcf_path else None!r}). The "
            f"(omega_n, zeta) PD parameterization derives gains as "
            f"kp = m_eff·omega_n² from the MJCF mass matrix. Open the Robot "
            f"Asset card and run 'Dump MJCF' before exporting."
        )
    resolver = JointIRResolver(spec.robot)
    physical = [resolver.to_physical(r) for r in joint_ir_roles]
    gains = solve_mujoco(
        mjcf_path=Path(mjcf_path),
        joint_order_physical=physical,
        joint_ir_roles=list(joint_ir_roles),
        nominal_qpos=None,  # MJCF keyframe-0 / qpos0 — same m_eff as IL finalizer
        pd_param=pd_param,
    )
    # gains.kp / gains.kd are aligned with joint_ir_roles order by construction.
    stiffness_arr = [float(v) for v in gains.kp]
    damping_arr = [float(v) for v in gains.kd]
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
    # action_scale is read from the ACTOR (single authority = actor_setting,
    # the node IsaacLab reads too). It used to be read from obs_action — a
    # divergent second knob that made SB3 and IsaacLab scale actions
    # differently. The env now also reads actor.action_scale.
    action_scale = (
        float(getattr(actor, "action_scale", 0.25) or 0.25) if actor else 0.25
    )
    use_default_offset = bool(
        getattr(actor, "action_use_default_offset", True)
    ) if actor is not None else True
    action_block: Dict[str, Any] = {
        "scale": action_scale,
        "clip": None,
        "offset_mode": "default_joint_pos" if use_default_offset else "zero",
    }

    # Observations — built from the SAME ``il_terms`` layout the training env
    # assembled, through the shared obs_term_engine. This is the SSOT that
    # makes "train == export == deploy" hold by construction (CLAUDE.md §8/§11):
    # the env's _build_obs and this contract iterate one resolved layout, and
    # the deploy ObsBuilder reads the contract back through the same engine.
    # An empty il_terms (SB3 canvas with no il_observation node) yields the
    # default proprio layout, byte-equal in dim/order to the legacy contract.
    from application.training.envs import obs_term_engine as _obs_eng

    obs_clip = (
        float(getattr(obs_action, "obs_clip_range", 0.0) or 0.0) if obs_action else 0.0
    )
    obs_clip_pair: Optional[List[float]] = (
        [-obs_clip, obs_clip] if obs_clip > 0.0 else None
    )
    il_terms = getattr(obs_action, "il_terms", None) if obs_action is not None else None
    legacy_default = not il_terms
    layout = _obs_eng.normalize_obs_layout(il_terms, num_joints=n, action_dim=n)

    # Per-item obs tail spec — resolved from the TrainingSpec BEFORE the loop
    # below shadows the ``spec`` name with a per-term dict. None when the canvas
    # has no per-item rewards.
    per_item_obs = _resolve_per_item_obs(spec)

    observations: Dict[str, Any] = {}
    for term in layout:
        spec: Dict[str, Any] = {"dim": int(term.dim)}
        spec["scale"] = (
            [float(v) for v in term.scale]
            if isinstance(term.scale, (list, tuple)) else float(term.scale)
        )
        # History: per-term value from il_terms (the IsaacLab-aligned
        # mechanism). For the legacy default layout (no il_terms) we fold the
        # whole-frame VecFrameStack count into each term so existing SB3
        # bundles keep their obs_dim. (Per-term history + frame_stack>1
        # together is a P7 concern; frame_stack defaults to 1.)
        hist = int(term.history_length)
        if legacy_default and int(frame_stack) > 1:
            hist = int(frame_stack)
        spec["history_length"] = hist
        if term.clip is not None:
            spec["clip"] = [float(term.clip[0]), float(term.clip[1])]
        elif obs_clip_pair is not None:
            spec["clip"] = list(obs_clip_pair)
        observations[term.name] = spec

    # ``robot_sku`` is intentionally NOT written here. The bundle's SKU
    # lives in ``manifest.robot.sku`` (the single source of truth);
    # storing a second copy on the contract used to allow silent
    # divergence. Runtime callers (PolicyRunner / CompatibilityChecker)
    # plumb the manifest SKU into ``joint_spaces_from_deploy_contract``
    # explicitly. See ``deploy_contract.DeployContract`` class docstring.
    # Schema v2 (CLAUDE.md §10): carry pd_param (source) + the MuJoCo-derived
    # per-joint gains so loaders prefer pd_param and re-derive (catches stale
    # hand-edited arrays). stiffness/damping mirror mujoco_pd_gains/damping for
    # v1 readers; PDController.from_deploy_contract prefers mujoco_pd_gains.
    contract: Dict[str, Any] = {
        "schema_version": 2,
        "joint_sdk_names": list(joint_ir_roles),
        "joint_ids_map": list(range(n)),
        "pd_param": pd_param.to_dict(),
        "mujoco_pd_gains": list(stiffness_arr),
        "mujoco_pd_damping": list(damping_arr),
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
    # Per-item obs tail: deploy ObsBuilder reconstructs [cmd_norm, item weights]
    # from the command via ItemResolver, so the deployed obs == the trained obs
    # (CLAUDE.md §11). item_ids order == the env's item_id_order (terms_by_item
    # insertion order) so the trained input permutation can't drift.
    if per_item_obs is not None:
        contract["per_item_obs"] = {
            "item_ids": list(per_item_obs["item_ids"]),
            "command_ranges": per_item_obs["command_ranges"],
            "blend_width": float(per_item_obs["blend_width"]),
        }
    if velocity_val > 0.0:
        contract["velocity_limit"] = [velocity_val] * n
        # saturation_effort is REQUIRED alongside velocity_limit for the deploy
        # DCMotor torque-speed envelope to engage (pd_controller.compute gates
        # on BOTH). Without it the deploy stack silently ignored velocity_limit
        # — and SB3 training applied no envelope either. We set it = effort_limit
        # (the zero-speed peak), and GenericMujocoEnv applies the identical
        # envelope during training, so velocity_limit is honored consistently.
        contract["saturation_effort"] = list(effort_arr)

    # init_base_pos: only carry when the canvas set a non-default spawn
    # height. ``actor.init_pos_z`` is None when the user didn't override
    # the spawn height, in which case the deploy stack falls back to the
    # asset's nominal standing height (MJCF keyframe / qpos0 — the model
    # holds the correct height per robot). Omitting the field here mirrors
    # that. We use ``is None`` explicitly rather than ``or 0.0`` because
    # the canvas allows the user to spawn AT z=0 (drop the robot from
    # ground level), and ``or 0.0`` would silently turn that explicit
    # intent into "use asset nominal".
    init_z: Optional[float] = None
    if actor is not None:
        raw_z = getattr(actor, "init_pos_z", None)
        if raw_z is not None:
            init_z = float(raw_z)
    if init_z is not None and init_z != 0.0:
        # Read x/y unconditionally — when the user set a non-default
        # spawn height they implicitly accepted x/y as well (defaulting
        # to 0.0 by canvas convention).
        raw_x = getattr(actor, "init_pos_x", None)
        raw_y = getattr(actor, "init_pos_y", None)
        init_x = float(raw_x) if raw_x is not None else 0.0
        init_y = float(raw_y) if raw_y is not None else 0.0
        contract["init_base_pos"] = [init_x, init_y, init_z]

    # MJCF base spawn-Z offset overlay (USD↔MJCF anchor compensation).
    # Carrying it into the deploy_contract makes bundles self-contained
    # per CLAUDE.md §1.9 — a downstream MuJoCo runtime on a different
    # machine doesn't need the user's robot_assets/state.json overlay.
    # We embed the FULL overlay dict (status, offset_z, calibration
    # metadata) so future deploy paths can choose to re-validate or
    # re-calibrate; existing deploy paths just read offset_z.
    try:
        sku = str(getattr(getattr(spec, "robot", None), "sku", "") or "")
        if sku:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            overlay = get_robot_asset_service().get_mjcf_base_offset(sku)
            if isinstance(overlay, dict) and overlay:
                contract["mjcf_base_height_offset"] = overlay
    except Exception as exc:  # noqa: BLE001
        # WHY KEPT (§1.8 (c)): bundle export must not fail because of
        # an overlay read glitch (service singleton not yet ready in
        # tests, USER_CONFIG_DIR unset). The runtime reader on the
        # deploy side will WARN about uncalibrated spawn and proceed.
        from unitport_sdk import log_warning
        log_warning(
            f"[bundle_exporter] could not attach mjcf_base_height_offset "
            f"to deploy_contract: {exc}"
        )

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
        joint_array_format: Optional[str] = None,
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

        if joint_array_format:
            # Articulation order the bundle's per-joint arrays use (USD for
            # Isaac Lab, MJCF for SB3/MuJoCo). DISTINCT from
            # inference_convention (obs frame) — both backends are
            # "isaac_lab" obs-frame, but only Isaac Lab is USD-ordered.
            # The deploy stack (PolicyRunner USD-reorder, ObsBuilder
            # joint-order check) keys off THIS field, never off
            # inference_convention. See
            # joint_space.resolve_joint_array_format.
            fmt = str(joint_array_format).strip().upper()
            if fmt not in ("MJCF", "USD", "URDF"):
                raise ValueError(
                    f"BundleExporter.build_manifest: joint_array_format "
                    f"must be one of MJCF/USD/URDF, got {joint_array_format!r}"
                )
            manifest["joint_array_format"] = fmt

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
    def wipe_bundle_dir(
        *,
        project: Any,
        backend_id: str,
        bundle_name: str,
    ) -> bool:
        """Delete the on-disk bundle directory for (project, backend, name).

        Called from training tasks at run-start when the canvas Export
        Node has ``overwrite=True``: the contract is *"new bundle
        replaces the old one — and if the new one never lands, the old
        one must not survive masquerading as the new result"*. Without
        this, a failed training (export subprocess crash, cancel, etc.)
        leaves the previous bundle on disk so the user sees a stale
        artifact under the same name → confusion + the SKU-mismatch
        chain we just patched.

        Safe to call when the dir does not exist (returns False).
        Refuses bundle_name=="" or "<NEW>" (sentinel — caller should
        have resolved to a concrete name or skipped the wipe).

        Returns True if a directory was actually removed, False otherwise.
        """
        if project is None:
            return False
        name = (bundle_name or "").strip()
        if not name or name == "<NEW>":
            return False
        bid = (backend_id or "").strip() or "unknown"
        target = Path(project.path) / "training" / "exported" / bid / name
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True

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

            # 2b) manifest.sha256 — integrity sidecar. The loader
            # (ExportedBundleRegistry._parse_entry) verifies the digest
            # against this file before trusting the manifest, so a
            # tampered manifest gets detected at load time. Bundles
            # without a sidecar still load with a one-shot WARN for one
            # release cycle (back-compat with the prior factory).
            # Plan finding P2-1.
            import hashlib as _hashlib
            _digest = _hashlib.sha256(manifest_dst.read_bytes()).hexdigest()
            (staging / "manifest.sha256").write_text(
                _digest + "\n", encoding="utf-8",
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
        # The base obs dim is the SUM of the resolved obs-term layout — the
        # SAME layout the training env (_build_obs) and the deploy contract
        # (_build_sb3_deploy_contract) use, via obs_term_engine. Deriving it
        # from a hardcoded formula instead silently diverged from il_terms
        # (e.g. an added base_lin_vel term: formula said 45, the layout is 48).
        action_dim = int(spec.robot.num_joints if hasattr(spec.robot, "num_joints") else len(spec.robot.joint_order))
        oa = getattr(spec, "obs_action", None)
        from application.training.envs import obs_term_engine as _obs_eng
        _il_terms = getattr(oa, "il_terms", None) if oa is not None else None
        _layout = _obs_eng.normalize_obs_layout(
            _il_terms, num_joints=action_dim, action_dim=action_dim,
        )
        base_obs_dim = int(sum(int(t.dim) * int(t.history_length) for t in _layout))
        frame_stack = int(getattr(oa, "frame_stack", 1) or 1) if oa is not None else 1

        # Per-item obs tail (cmd_norm + soft weight per training item). The env
        # appends it INSIDE _build_obs, so it is part of the per-frame obs the
        # policy network — and VecNormalize — see. The deploy ObsBuilder
        # reconstructs it from the command (same ItemResolver), so the contract
        # only needs the item spec (added in _build_sb3_deploy_contract).
        _per_item = _resolve_per_item_obs(spec)
        tail_dim = (1 + len(_per_item["item_ids"])) if _per_item is not None else 0
        if tail_dim > 0 and frame_stack > 1:
            raise ValueError(
                "[bundle_exporter] per-item rewards (obs tail) combined with "
                f"obs_action.frame_stack={frame_stack} is not supported — the "
                "deploy ObsBuilder appends the per-item tail once per frame, "
                "not stacked. Set frame_stack=1, or remove the per-item reward "
                "edges. (CLAUDE.md §8 — refusing to ship a mismatched contract.)"
            )

        # Per-frame obs = term layout + per-item tail (what VecNormalize sees).
        # The network/manifest obs_dim then frame-stacks the per-frame obs.
        per_frame_obs_dim = base_obs_dim + tail_dim
        obs_dim = per_frame_obs_dim * frame_stack

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
            obs_dim=base_obs_dim,
            joint_ir_roles=joint_names,
            sim_dt=sim_dt,
            control_dt=control_dt,
            frame_stack=frame_stack,
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
            # SB3 training now assembles obs through the shared obs_term_engine
            # in the Isaac-Lab convention (body-frame base velocity,
            # default-relative joint_pos, correct projected_gravity sign), so
            # the deploy ObsBuilder must use the same convention. Set it
            # explicitly rather than relying on the stiffness-inference
            # heuristic in ObsBuilder.__init__.
            inference_convention="isaac_lab",
            # SB3 trains on MuJoCo, so the policy's joint slots — and hence
            # every per-joint array in this bundle — are in MJCF (per-leg)
            # order, NOT Isaac Lab's USD (per-type) order. Declaring this
            # stops the deploy stack from permuting the contract to USD
            # (which would scramble every joint) or rejecting it as
            # "not USD-ordered". (CLAUDE.md §8/§9/§11.)
            joint_array_format="MJCF",
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
        # VecNormalize sits below VecFrameStack so its obs_rms shape is the
        # PER-FRAME dim — which includes the per-item obs tail (the tail is
        # appended inside GenericMujocoEnv._build_obs, below VecNormalize), NOT
        # just the term layout. Check against per_frame_obs_dim (base + tail).
        norm_payload = _extract_vec_normalize_stats(vec_env, obs_dim=per_frame_obs_dim)
        include_norm = bool(getattr(export_cfg, "include_normalization", True))
        if norm_payload is not None:
            if not include_norm:
                # The policy trained on VecNormalize-normalized observations.
                # Shipping it WITHOUT the running mean/std would feed the deploy
                # policy raw (out-of-distribution) obs → train≠deploy drift. We
                # refuse rather than silently honor the flag into a broken
                # bundle (CLAUDE.md §8). Disable env_assembler.obs_normalize if
                # you genuinely want an un-normalized policy.
                raise ValueError(
                    "[bundle_exporter] export.include_normalization=False but "
                    "VecNormalize was active during training — omitting the "
                    "normalization stats would make deploy observations "
                    "mismatch training (train != deploy). Enable "
                    "include_normalization, or turn off "
                    "env_assembler.obs_normalize."
                )
            extra_files["normalization.json"] = json.dumps(
                norm_payload, indent=2, sort_keys=False
            )
            manifest["normalization"] = {"file": "normalization.json"}

        # TorchScript companion artifact (P7) — when the canvas requests it,
        # trace the same actor net to ``policy.pt`` and ship it alongside the
        # ONNX policy. ONNX stays the primary deploy format; the .pt lets a
        # TorchScript-only runtime (deploy JITEngine) consume the bundle. The
        # manifest records the aux file so the deploy side can find it.
        export_cfg2 = getattr(spec, "export", None)
        if export_cfg2 is not None and bool(getattr(export_cfg2, "include_torchscript", False)):
            try:
                from application.training.sb3_onnx_export import (
                    export_to_torchscript_bytes,
                )
                extra_files["policy.pt"] = export_to_torchscript_bytes(model, obs_dim)
                manifest["aux_policy"] = {"file": "policy.pt", "format": "jit"}
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    f"[bundle_exporter] TorchScript export requested but failed "
                    f"({exc}); shipping ONNX only."
                )

        # Custom terrain — embed the heightfield self-contained in the bundle
        # (§9), symmetric with the IsaacLab finalizer. The cross-engine
        # consistency gate runs inside embed_custom_terrain before any bytes
        # are written (§8). SB3 trains on the MuJoCo <hfield> derived from the
        # same source, so the bundle terrain matches what was trained.
        from application.training.terrain.bundle_io import embed_custom_terrain
        _terrain_files = embed_custom_terrain(spec, manifest)
        if _terrain_files:
            extra_files.update(_terrain_files)

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
