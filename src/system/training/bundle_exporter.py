#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D — Checkpoint bundle exporter for trained SB3 policies.

export_bundle(model, spec, run_id, output_root) -> Path
    1. Creates a temporary staging directory.
    2. Exports the SB3 policy actor network to ``policy.onnx``.
    3. Writes ``manifest.yaml`` (all 12 required fields).
    4. Writes ``source.json`` with type="training" and lineage fields.
    5. Validates the manifest via ``validate_manifest()``.
    6. Atomically moves the staged bundle to
       ``<output_root>/custom_mods/training/checkpoints/<policy_id_out>/``.
    7. Returns the final bundle path.

The ONNX model exposes:
    input:  "obs"    shape (batch, obs_dim)  float32
    output: "action" shape (batch, action_dim) float32

ONNX opset 17 is used for broad compatibility with onnxruntime >= 1.14.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from src.system.training.obs_contracts import get_obs_contract

if TYPE_CHECKING:
    from src.system.training.training_spec import TrainingJobSpec


# ---------------------------------------------------------------------------
# Export result contract
# ---------------------------------------------------------------------------

@dataclass
class ExportResult:
    """
    Returned by ``export_bundle()``.

    Callers should use ``primary_path`` for backwards-compatible single-path
    access; ``runtime_bundle_path`` and ``artifact_path`` are set only when
    the respective export target was produced.
    """
    runtime_bundle_path: Optional[Path] = None  # custom_mods/training/checkpoints/<id>/
    artifact_path:       Optional[Path] = None  # training_assets/<id>/

    @property
    def primary_path(self) -> Path:
        """Runtime bundle if present, else training artifact."""
        return self.runtime_bundle_path or self.artifact_path  # type: ignore[return-value]

    def __str__(self) -> str:
        return str(self.primary_path)


# ---------------------------------------------------------------------------
# Fallback defaults — used ONLY when spec.robot_spec lacks runtime metadata
# (i.e. the probe env failed to extract from the MJCF model).
# ---------------------------------------------------------------------------

_GO2_JOINT_NAMES = [
    "FL_hip",  "FL_thigh",  "FL_calf",
    "FR_hip",  "FR_thigh",  "FR_calf",
    "RL_hip",  "RL_thigh",  "RL_calf",
    "RR_hip",  "RR_thigh",  "RR_calf",
]

_NUM_JOINTS_BY_ROBOT = {
    "go2": 12, "go1": 12, "b1": 12, "b2": 12, "h1": 10, "g1": 23,
}


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


def _maybe_collect_norm_stats(model, spec: "TrainingJobSpec", obs_dim: int) -> Optional[dict]:
    """Build exportable normalization stats when ExportConfig requests them.

    The current training stack does not wrap envs with VecNormalize, so in the
    common case this emits explicit identity observation stats. If a compatible
    VecNormalize env is present in the future, its learned statistics are used.
    """
    export_cfg = getattr(spec, "export_config", None)
    if export_cfg is None or not bool(getattr(export_cfg, "include_norm_stats", False)):
        return None

    env_cfg = getattr(spec, "env_config", None)
    clip_obs = float(getattr(env_cfg, "clip_obs", 10.0) or 10.0)

    obs_stats = None
    source = "identity"
    try:
        vec_norm = getattr(model, "get_vec_normalize_env", lambda: None)()
    except Exception:
        vec_norm = None

    if vec_norm is not None:
        obs_rms = getattr(vec_norm, "obs_rms", None)
        if obs_rms is not None and getattr(obs_rms, "mean", None) is not None and getattr(obs_rms, "var", None) is not None:
            try:
                import numpy as np

                mean = np.asarray(obs_rms.mean, dtype=np.float32).reshape(-1)
                std = np.sqrt(np.maximum(np.asarray(obs_rms.var, dtype=np.float32).reshape(-1), 1e-8))
                if mean.size == int(obs_dim) and std.size == int(obs_dim):
                    obs_stats = {
                        "mean": mean.astype(float).tolist(),
                        "std": std.astype(float).tolist(),
                        "clip_min": -clip_obs,
                        "clip_max": clip_obs,
                    }
                    source = "vecnormalize"
            except Exception:
                obs_stats = None

    if obs_stats is None:
        obs_stats = _identity_norm_stats(obs_dim, clip=clip_obs)

    return {
        "version": 1,
        "source": source,
        "observation": obs_stats,
    }


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def _build_ppo_onnx_net(policy):
    """Return a traceable nn.Module for deterministic PPO inference.

    Architecture: obs -> features_extractor -> mlp_extractor.policy_net -> action_net
    """
    import torch
    import torch.nn as nn

    class _PPONet(nn.Module):
        def __init__(self, p):
            super().__init__()
            self._fe   = p.features_extractor
            self._mlp  = p.mlp_extractor.policy_net
            self._anet = p.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features  = self._fe(obs)
            latent_pi = self._mlp(features)
            return self._anet(latent_pi)

    net = _PPONet(policy)
    net.eval()
    return net


def _build_sac_onnx_net(policy):
    """Return a traceable nn.Module for deterministic SAC inference.

    Architecture: obs -> actor.latent_pi -> actor.mu -> tanh
    """
    import torch
    import torch.nn as nn

    class _SACNet(nn.Module):
        def __init__(self, p):
            super().__init__()
            self._latent = p.actor.latent_pi
            self._mu     = p.actor.mu

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self._mu(self._latent(obs)))

    net = _SACNet(policy)
    net.eval()
    return net


def export_to_onnx(model, obs_dim: int, out_path: Path) -> None:
    """
    Export the actor network of a trained SB3 model to ONNX.

    Parameters
    ----------
    model:
        Trained ``stable_baselines3.PPO`` or ``stable_baselines3.SAC`` model.
    obs_dim:
        Observation dimension (input size).
    out_path:
        Destination file path for the ONNX model.

    Raises
    ------
    ValueError
        If the algorithm is not PPO or SAC.
    """
    import torch

    algo_name = type(model).__name__.upper()
    policy    = model.policy

    if algo_name == "PPO":
        net = _build_ppo_onnx_net(policy)
    elif algo_name == "SAC":
        net = _build_sac_onnx_net(policy)
    else:
        raise ValueError(
            f"ONNX export not supported for algorithm '{algo_name}'. "
            "Phase D supports: PPO, SAC"
        )

    try:
        export_device = next(net.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")

    net = net.to(export_device)
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=export_device)

    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy_obs,
            str(out_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={
                "obs":    {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )


def export_to_torchscript(model, obs_dim: int, out_path: Path) -> None:
    """
    Export the actor network of a trained SB3 model to TorchScript (JIT trace).

    Parameters
    ----------
    model:
        Trained ``stable_baselines3.PPO`` or ``stable_baselines3.SAC`` model.
    obs_dim:
        Observation dimension (input size).
    out_path:
        Destination file path for the TorchScript model (``.pt``).

    Raises
    ------
    ValueError
        If the algorithm is not PPO or SAC.
    """
    import torch

    algo_name = type(model).__name__.upper()
    policy    = model.policy

    if algo_name == "PPO":
        net = _build_ppo_onnx_net(policy)
    elif algo_name == "SAC":
        net = _build_sac_onnx_net(policy)
    else:
        raise ValueError(
            f"TorchScript export not supported for algorithm '{algo_name}'. "
            "Phase D supports: PPO, SAC"
        )

    try:
        export_device = next(net.parameters()).device
    except StopIteration:
        export_device = torch.device("cpu")

    net = net.to(export_device)
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=export_device)

    with torch.no_grad():
        traced = torch.jit.trace(net, dummy_obs)
    traced.save(str(out_path))


# ---------------------------------------------------------------------------
# Manifest and source.json builders
# ---------------------------------------------------------------------------

def build_manifest(
    spec: "TrainingJobSpec",
    obs_dim: int,
    action_dim: int,
    policy_file: str = "policy.onnx",
    policy_format: str = "onnx",
    normalization_file: str = "",
) -> dict:
    """
    Build the ``manifest.yaml`` dict for a trained bundle.

    All 12 fields required by ``validate_manifest()`` are included.

    Parameters
    ----------
    spec:
        Compiled TrainingJobSpec supplying robot / task / export metadata.
    obs_dim:
        Observation dimension (used as ``observation_space.dim``).
    action_dim:
        Action dimension (used as ``action_space.dim``).
    policy_file:
        Filename of the policy model (relative to bundle root).
    """
    robot_type = spec.robot_spec.robot_type.lower()
    # Prefer runtime metadata extracted from the MJCF model (populated by
    # SB3Trainer's probe env).  Fall back to static tables only when the
    # spec fields are empty (e.g. probe env failed).
    joint_names = list(spec.robot_spec.joint_names) if spec.robot_spec.joint_names else []
    num_joints = (
        spec.robot_spec.action_dim
        if spec.robot_spec.action_dim > 0
        else _NUM_JOINTS_BY_ROBOT.get(robot_type, action_dim)
    )
    if not joint_names:
        joint_names = _GO2_JOINT_NAMES[:num_joints]  # last-resort fallback

    policy_id_out = (
        getattr(spec.export_config, "bundle_name", "") or
        spec.algorithm_config.policy_id_out or
        f"{spec.policy_id}_trained"
    )
    version = "1.0.0"
    if spec.export_config is not None:
        # Export node doesn't carry version today; keep default.
        pass

    # Determine robot brand
    brand_map = {
        "go2": "unitree", "go1": "unitree", "b1": "unitree",
        "b2": "unitree", "h1": "unitree", "g1": "unitree",
        "spot": "spot", "cyberdog": "cyberdog",
    }
    robot_brand = brand_map.get(robot_type, "unitree")
    observation_space = {
        "dim": obs_dim,
    }
    obs_components = list(getattr(getattr(spec, "obs_action_config", None), "obs_components", []) or [])
    if obs_components:
        observation_space["components"] = obs_components
    frame_stack = int(getattr(getattr(spec, "obs_action_config", None), "frame_stack", 1) or 1)
    if frame_stack > 1:
        observation_space["frame_stack"] = frame_stack
    contract_preset = str(
        getattr(getattr(spec, "obs_action_config", None), "contract_preset", "") or ""
    ).strip()
    try:
        from src.system.training.training_config import resolve_control_timing
        runtime_timing = resolve_control_timing(
            getattr(spec.physics_config, "sim_dt", 0.002),
            getattr(spec.physics_config, "control_dt", 0.02),
        )
    except Exception:
        runtime_timing = {"control_frequency_hz": 50.0, "decimation": 4}
    if contract_preset and contract_preset != "custom":
        contract = get_obs_contract(contract_preset)
        if contract is not None:
            observation_space["contract_preset"] = contract_preset
            observation_space["components"] = list(contract.get("obs_components") or [])

    manifest = {
        "name": policy_id_out,
        "version": version,
        "policy": {
            "file":   policy_file,
            "format": policy_format,
        },
        "observation_space": observation_space,
        "action_space": {
            "dim":  action_dim,
            "type": (
                getattr(spec.obs_action_config, "action_type", None)
                or getattr(spec.physics_config,  "action_type", None)
                or "joint_position"
            ),
            "scale": float(getattr(spec.obs_action_config, "action_scale", 1.0) or 1.0),
            "clip": float(getattr(spec.obs_action_config, "action_clip", 1.0) or 1.0),
        },
        "runtime": {
            "control_frequency_hz": float(runtime_timing["control_frequency_hz"]),
            "decimation": int(runtime_timing["decimation"]),
            "command_defaults": {
                "vx": float(getattr(spec.task_config, "target_vx", 0.5) or 0.0),
                "vy": float(getattr(spec.task_config, "target_vy", 0.0) or 0.0),
                "wz": float(getattr(spec.task_config, "target_wz", 0.0) or 0.0),
            },
        },
        "robot": {
            "brand":        robot_brand,
            "type":         robot_type,
            "num_joints":   num_joints,
            "joint_names":  joint_names,
        },
        "training": {
            "algorithm":        spec.algorithm_config.algorithm,
            "total_timesteps":  spec.algorithm_config.total_timesteps,
            "experiment_id":    spec.experiment_id,
        },
    }
    if normalization_file:
        manifest["normalization"] = {
            "file": normalization_file,
        }

    # ── SkillManifest v2 section ─────────────────────────────────────
    # Derive fields from TrainingJobSpec and fill in runtime-known values.
    try:
        skill_fields = spec.derive_skill_manifest_fields()
        skill_fields["skill_id"] = policy_id_out
        skill_fields["skill_name"] = policy_id_out
        skill_fields["action_dim"] = action_dim
        skill_fields["observation_dim"] = obs_dim
        skill_fields["model_path"] = policy_file
        skill_fields["inference_backend"] = (
            "torchscript" if policy_format == "jit" else "onnx"
        )
        skill_fields["normalize_obs"] = bool(normalization_file)
        if normalization_file:
            skill_fields["normalizer_path"] = normalization_file
        skill_fields["description"] = f"SB3-trained {skill_fields.get('training_source', '')} policy: {policy_id_out}"
        manifest["skill"] = skill_fields
    except Exception:
        pass  # v2 section is best-effort; v1 fields always present

    return manifest


def build_source_json(
    spec: "TrainingJobSpec",
    run_id: str = "",
) -> dict:
    """
    Build the ``source.json`` provenance dict for a training-produced bundle.

    ``type`` is set to ``"training"`` so that CheckpointRegistry correctly
    labels the checkpoint origin and ``source_badge()`` returns "🏋".

    Parameters
    ----------
    spec:
        Compiled TrainingJobSpec.
    run_id:
        Persistent run identifier from TrainingWorkspaceStore.
    """
    return {
        "type":             "training",
        "parent_policy_id": spec.policy_id,
        "experiment_id":    spec.experiment_id,
        "run_id":           run_id,
        "algorithm":        spec.algorithm_config.algorithm,
        "created_at":       time.time(),
    }


# ---------------------------------------------------------------------------
# Training artifact helpers
# ---------------------------------------------------------------------------

def build_artifact_manifest(
    spec: "TrainingJobSpec",
    obs_dim: int,
    action_dim: int,
) -> dict:
    """
    Build the ``asset_manifest.json`` dict written into a training artifact.

    This manifest is read by ``TrainingAssetRegistry`` so the artifact can
    be listed in the Training Assets panel and used as a resume base.
    """
    policy_id_out = (
        getattr(spec.export_config, "bundle_name", "") or
        spec.algorithm_config.policy_id_out or
        f"{spec.policy_id}_trained"
    )
    return {
        "asset_id":           policy_id_out,
        "display_name":       policy_id_out,
        "framework":          "sb3",
        "algorithm":          spec.algorithm_config.algorithm,
        "robot_type":         spec.robot_spec.robot_type,
        "obs_dim":            obs_dim,
        "action_dim":         action_dim,
        "action_type":        (
            getattr(spec.obs_action_config, "action_type", None)
            or getattr(spec.physics_config,  "action_type", None)
            or "joint_position"
        ),
        "primary_checkpoint": "best/best_model.zip",
        "source_type":        "training",
        "experiment_id":      spec.experiment_id,
    }


# ---------------------------------------------------------------------------
# P0: Best-model loader — prefer best checkpoint over final model on export
# ---------------------------------------------------------------------------

def _try_load_best_model(
    model,
    spec: "TrainingJobSpec",
    output_root: Path,
    log_fn=None,
):
    """
    Try to load the best-model checkpoint written by SB3BestModelTracker.

    The tracker saves:
        <output_root>/training_checkpoints/<policy_id_out>/best_model/
            best_model.zip
            best_model_meta.json   {"step": int, "reward_mean": float}

    Returns
    -------
    (model, source_tag, meta)
        ``model``      — loaded best model, or original model if not found.
        ``source_tag`` — ``"best_model"`` or ``"final_model"``.
        ``meta``       — dict from best_model_meta.json, or ``{}``.
    """
    policy_id_out = (
        getattr(getattr(spec, "export_config", None), "bundle_name", "")
        or getattr(getattr(spec, "algorithm_config", None), "policy_id_out", "")
        or "trained_policy"
    )
    best_dir  = output_root / "training_checkpoints" / policy_id_out / "best_model"
    best_zip  = best_dir / "best_model.zip"
    meta_file = best_dir / "best_model_meta.json"

    if not best_zip.exists():
        if log_fn:
            log_fn("[export] No best-model checkpoint found; exporting final model")
        return model, "final_model", {}

    try:
        from stable_baselines3 import PPO, SAC

        algo_name = type(model).__name__.upper()
        algo_cls  = PPO if algo_name == "PPO" else SAC
        # Load without env — we only need the policy network for ONNX/JIT export
        best = algo_cls.load(str(best_zip), device=str(model.device))

        meta: dict = {}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as fh:
                meta = json.load(fh)

        if log_fn:
            step   = meta.get("step", "?")
            reward = meta.get("reward_mean", float("nan"))
            log_fn(
                f"[export] Best-model checkpoint loaded: "
                f"step={step:,}  reward_mean={reward:.3f}"
                if isinstance(step, int)
                else f"[export] Best-model checkpoint loaded: step={step}  reward_mean={reward}"
            )
        return best, "best_model", meta

    except Exception as exc:
        if log_fn:
            log_fn(f"[export] Best-model load failed ({exc}); using final model")
        return model, "final_model", {}


# ---------------------------------------------------------------------------
# Runtime support check
# ---------------------------------------------------------------------------

_RUNTIME_SUPPORTED_ACTION_TYPES = frozenset({"joint_position", "joint_velocity", "torque"})


def _warn_if_runtime_unsupported(action_type: str, log_fn=None) -> None:
    """Emit a warning when the action type is not yet supported by the runtime."""
    if action_type and action_type not in _RUNTIME_SUPPORTED_ACTION_TYPES:
        msg = (
            f"[export] WARNING: action_type='{action_type}' is not supported by "
            f"the runtime ActionApplier (supports: {sorted(_RUNTIME_SUPPORTED_ACTION_TYPES)}). "
            "The runtime bundle is exported but may fail at inference time."
        )
        if log_fn is not None:
            log_fn(msg)
        else:
            import warnings
            warnings.warn(msg, stacklevel=3)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_bundle(
    model,
    spec: "TrainingJobSpec",
    run_id: str = "",
    output_root: Optional[Path] = None,
    log_fn=None,
) -> ExportResult:
    """
    Export a trained SB3 model according to ``spec.export_config.export_target``.

    export_target options
    ---------------------
    ``"runtime_bundle"`` (default)
        ONNX/JIT + manifest.yaml → ``<output_root>/custom_mods/training/checkpoints/<id>/``
        Listed by CheckpointRegistry; usable by the main canvas.
    ``"training_artifact"``
        SB3 zip → ``<output_root>/custom_mods/training/assets/<id>/best/best_model.zip``
        + ``asset_manifest.json``; discoverable by TrainingAssetRegistry as a
        resume base for Phase 3.  NOT listed in main canvas checkpoint panel.
    ``"both"``
        Produces both outputs.

    Returns
    -------
    ExportResult
        ``.runtime_bundle_path`` and/or ``.artifact_path`` set as produced.
        ``.primary_path`` returns the runtime bundle when present.

    Raises
    ------
    ManifestValidationError
        If the runtime bundle manifest fails validation.
    FileExistsError
        If a target directory already exists and ``overwrite=False``.
    """
    from src.system.training.sb3_trainer import get_obs_action_dims
    from src.system.policy.manifest_schema import validate_manifest

    if output_root is None:
        output_root = Path(os.getcwd())
    output_root = Path(output_root)

    # ── P0: Prefer best-model checkpoint over final model ─────────────────
    model, exported_from, best_meta = _try_load_best_model(
        model, spec, output_root, log_fn=log_fn
    )

    algo = spec.algorithm_config
    policy_id_out = (
        getattr(spec.export_config, "bundle_name", "") or
        algo.policy_id_out or
        f"{spec.policy_id}_trained"
    )
    obs_dim, action_dim = get_obs_action_dims(spec)

    export_target = "runtime_bundle"
    overwrite     = False
    do_onnx       = True
    do_jit        = True
    if spec.export_config is not None:
        export_target = spec.export_config.export_target
        overwrite     = spec.export_config.overwrite
        do_onnx       = spec.export_config.export_onnx
        do_jit        = spec.export_config.export_torchscript

    want_runtime  = export_target in ("runtime_bundle", "both")
    want_artifact = export_target in ("training_artifact", "both")
    norm_stats = _maybe_collect_norm_stats(model, spec, obs_dim=obs_dim)

    result = ExportResult()

    # ── Runtime bundle ────────────────────────────────────────────────
    if want_runtime:
        action_type = (
            getattr(spec.obs_action_config, "action_type", None)
            or getattr(spec.physics_config,  "action_type", None)
            or "joint_position"
        )
        _warn_if_runtime_unsupported(action_type)

        if not do_onnx and not do_jit:
            raise ValueError(
                "ExportConfig has both export_onnx=False and export_torchscript=False. "
                "At least one format must be enabled."
            )

        primary_policy_file   = "policy.onnx" if do_onnx else "policy_jit.pt"
        primary_policy_format = "onnx"        if do_onnx else "jit"

        tmp_parent = tempfile.mkdtemp(prefix="unitport_export_")
        stage_dir  = Path(tmp_parent) / policy_id_out
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)

            if do_onnx:
                export_to_onnx(model, obs_dim=obs_dim, out_path=stage_dir / "policy.onnx")

            if do_jit:
                try:
                    export_to_torchscript(model, obs_dim=obs_dim, out_path=stage_dir / "policy_jit.pt")
                except Exception:
                    if not do_onnx:
                        raise

            normalization_file = "normalization_stats.json" if norm_stats is not None else ""

            manifest_dict = build_manifest(
                spec, obs_dim=obs_dim, action_dim=action_dim,
                policy_file=primary_policy_file,
                policy_format=primary_policy_format,
                normalization_file=normalization_file,
            )
            with open(stage_dir / "manifest.yaml", "w", encoding="utf-8") as fh:
                yaml.dump(manifest_dict, fh, default_flow_style=False, allow_unicode=True)

            source_dict = build_source_json(spec, run_id=run_id)
            source_dict["exported_from"] = exported_from
            if best_meta:
                source_dict["best_step"]        = best_meta.get("step")
                source_dict["best_reward_mean"] = best_meta.get("reward_mean")
            with open(stage_dir / "source.json", "w", encoding="utf-8") as fh:
                json.dump(source_dict, fh, indent=2, ensure_ascii=False)
            if norm_stats is not None:
                with open(stage_dir / normalization_file, "w", encoding="utf-8") as fh:
                    json.dump(norm_stats, fh, indent=2, ensure_ascii=False)

            validate_manifest(manifest_dict, bundle_path=stage_dir)

            checkpoints_dir = output_root / "custom_mods/training/checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            final_path = checkpoints_dir / policy_id_out

            if final_path.exists():
                if overwrite:
                    shutil.rmtree(final_path)
                else:
                    raise FileExistsError(
                        f"Runtime bundle already exists at '{final_path}'. "
                        "Set ExportConfig.overwrite=True to replace it."
                    )

            shutil.move(str(stage_dir), str(final_path))
            result.runtime_bundle_path = final_path.resolve()

        finally:
            try:
                shutil.rmtree(tmp_parent, ignore_errors=True)
            except Exception:
                pass

    # ── Training artifact ─────────────────────────────────────────────
    if want_artifact:
        artifacts_dir = output_root / "custom_mods/training/assets"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_dest = artifacts_dir / policy_id_out

        if artifact_dest.exists():
            if overwrite:
                shutil.rmtree(artifact_dest)
            else:
                raise FileExistsError(
                    f"Training artifact already exists at '{artifact_dest}'. "
                    "Set ExportConfig.overwrite=True to replace it."
                )

        tmp_art = tempfile.mkdtemp(prefix="unitport_artifact_")
        stage_art = Path(tmp_art) / policy_id_out
        try:
            best_dir = stage_art / "best"
            best_dir.mkdir(parents=True, exist_ok=True)

            # Save SB3 zip (model.save() appends .zip automatically)
            zip_stem = str(best_dir / "best_model")
            model.save(zip_stem)

            # Write asset_manifest.json (read by TrainingAssetRegistry)
            art_manifest = build_artifact_manifest(spec, obs_dim=obs_dim, action_dim=action_dim)
            with open(stage_art / "asset_manifest.json", "w", encoding="utf-8") as fh:
                json.dump(art_manifest, fh, indent=2, ensure_ascii=False)

            # Write source.json
            source_dict = build_source_json(spec, run_id=run_id)
            with open(stage_art / "source.json", "w", encoding="utf-8") as fh:
                json.dump(source_dict, fh, indent=2, ensure_ascii=False)
            if norm_stats is not None:
                with open(stage_art / "normalization_stats.json", "w", encoding="utf-8") as fh:
                    json.dump(norm_stats, fh, indent=2, ensure_ascii=False)

            shutil.move(str(stage_art), str(artifact_dest))
            result.artifact_path = artifact_dest.resolve()

        finally:
            try:
                shutil.rmtree(tmp_art, ignore_errors=True)
            except Exception:
                pass

    return result
