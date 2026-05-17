"""IL bundle finalizer — turn a finished Isaac Lab training run into a
RELEASE v1 bundle on disk.

Pipeline (mirrors DEMO ``CheckpointRegistry.import_isaac_lab_bundle``
(:py:meth:`~src.system.service.checkpoint_registry.CheckpointRegistry.\
import_isaac_lab_bundle`) but writes the **RELEASE v1 12-field flat
manifest** rather than DEMO's v2 nested SkillManifest):

    1. Locate ``env.yaml`` — directly under ``exported_dir`` or in
       ``params/`` (RSL-RL log-dir layout).
    2. Locate ONNX policy:
         a. ``exported_dir/policy.onnx`` (Isaac Lab play.py wrote it).
         b. ``exported_dir/exported/policy.onnx`` (nested layout).
         c. Fallback — pick the latest ``model_*.pt`` in ``exported_dir``
            and convert in-process via :func:`onnx_export.export_rsl_rl_actor_to_onnx`.
            That requires the matching ``agent.yaml`` to reconstruct the
            actor MLP topology — we look in ``params/agent.yaml`` first,
            then ``agent.yaml`` next to the checkpoint.
    3. Parse ``env.yaml`` via :func:`manifest_parser.parse_isaac_lab_env_yaml`
       to obtain a ``SkillManifest`` (DEMO v2 nested form).
    4. Convert the SkillManifest → RELEASE v1 flat 12-field manifest dict
       via :func:`_skill_manifest_to_v1_dict`, sourcing fields the
       SkillManifest does not expose (decimation, joint_names, brand)
       from the caller-supplied :class:`TrainingSpec`.
    5. (AMP only) Extract ``discriminator_state_dict`` from the source
       ``.pt`` and stage a ``discriminator.pt`` alongside the bundle.
    6. Hand the prepared ONNX bytes + manifest dict to
       :meth:`BundleExporter.export_from_artifacts` for the atomic
       project-scoped write — lands at
       ``<project>/training/exported/isaac_lab/<bundle_name>/``.

Boundary rule (Phase 3 plan §B6): this module is the IL chain's only
caller of :meth:`BundleExporter.export_from_artifacts`; it composes its
own manifest dict end-to-end. It does **not** call ``export_bundle``
(SB3-only entry per rule §B1).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import log_info, log_warning

from application.training.bundle_exporter import (
    BundleExporter,
    ExportResult,
)
from application.training.isaac_lab.manifest_parser import (
    parse_isaac_lab_env_yaml,
)
from application.training.isaac_lab.skill_manifest import (
    ActionSpaceType,
    InferenceBackend,
    SkillManifest,
)


# ---------------------------------------------------------------------------
# ONNX shape validation — fail-fast at export time
# ---------------------------------------------------------------------------

def _validate_onnx_obs_dim(
    *,
    onnx_bytes: bytes,
    manifest_obs_dim: int,
    manifest_action_dim: int,
    obs_keys: List[str],
) -> None:
    """Compare the manifest's obs_dim / action_dim against the trained ONNX.

    Raises ``RuntimeError`` if either width disagrees. The Isaac Lab env.yaml
    + heuristic dim table is the only authority writing ``observation_space.
    dim`` today; if it diverges from the policy's real input width the
    bundle is broken (Launch Review will hit ``InvalidArgument`` at the
    onnxruntime boundary).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "[bundle_finalizer] onnxruntime is required for ONNX shape "
            "validation but is not installed in this venv. Strict-mode "
            "finalize cannot skip this check — install onnxruntime "
            "(see requirements.txt) and retry. Original ImportError: "
            f"{exc}"
        ) from exc

    session = ort.InferenceSession(
        onnx_bytes, providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise RuntimeError(
            "[bundle_finalizer] policy.onnx has no input tensors; cannot "
            "validate observation_space.dim."
        )

    def _last_dim(shape: List[Any]) -> Optional[int]:
        if not shape:
            return None
        tail = shape[-1]
        try:
            return int(tail)
        except (TypeError, ValueError):
            return None

    in_dim = _last_dim(list(inputs[0].shape))
    if in_dim is None:
        log_warning(
            "[bundle_finalizer] policy.onnx input shape "
            f"{inputs[0].shape!r} has a non-numeric last dim; skipping "
            "obs_dim validation."
        )
    elif in_dim != int(manifest_obs_dim):
        raise RuntimeError(
            f"[bundle_finalizer] obs_dim mismatch — manifest.observation_space.dim="
            f"{manifest_obs_dim} but policy.onnx input width is {in_dim}. "
            f"The env.yaml term sum disagrees with the trained policy "
            f"(obs term order = {obs_keys}). Aborting export to avoid a "
            f"corrupt bundle. Each obs term in env.yaml must declare an "
            f"explicit ``dim`` (the strict-canvas migration removed the "
            f"hardcoded default table)."
        )

    if outputs:
        out_dim = _last_dim(list(outputs[0].shape))
        if out_dim is not None and out_dim != int(manifest_action_dim):
            raise RuntimeError(
                f"[bundle_finalizer] action_dim mismatch — manifest.action_space.dim="
                f"{manifest_action_dim} but policy.onnx output width is "
                f"{out_dim}. Aborting export."
            )


def _validate_deploy_contract(
    *,
    contract: Dict[str, Any],
    manifest_obs_dim: int,
    manifest_action_dim: int,
) -> None:
    """Export-time structural validation of the assembled ``deploy_contract``.

    Catches everything that would later make ``DeployContract.from_dict``
    raise at runtime (and a few extras: total obs dim vs ``manifest.obs_dim``,
    per-joint list lengths vs ``action_dim``). Failing here means the bundle
    is **never** written to disk in a state that can't be loaded — the
    runtime no longer needs to be defensive against malformed contracts.
    """
    n = int(manifest_action_dim)

    # joint lists
    joint_sdk_names = contract.get("joint_sdk_names") or []
    if len(joint_sdk_names) != n:
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract.joint_sdk_names has "
            f"{len(joint_sdk_names)} entries but manifest.action_dim={n}."
        )

    joint_ids_map = contract.get("joint_ids_map") or []
    if sorted(joint_ids_map) != list(range(n)):
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract.joint_ids_map must be a "
            f"permutation of 0..{n - 1}; got {joint_ids_map}."
        )

    # per-joint scalar lists
    for key in ("stiffness", "damping", "effort_limit", "default_joint_pos"):
        lst = contract.get(key)
        if not isinstance(lst, list) or len(lst) != n:
            raise RuntimeError(
                f"[bundle_finalizer] deploy_contract.{key} has length "
                f"{len(lst) if isinstance(lst, list) else 'N/A'}, "
                f"expected {n}."
            )

    for key in ("velocity_limit", "saturation_effort"):
        lst = contract.get(key)
        if lst is not None and (not isinstance(lst, list) or len(lst) != n):
            raise RuntimeError(
                f"[bundle_finalizer] deploy_contract.{key} has length "
                f"{len(lst) if isinstance(lst, list) else 'N/A'}, "
                f"expected {n} (or null)."
            )

    # sim_dt / step_dt / decimation consistency (mirrors deploy_contract.py:404)
    try:
        sim_dt = float(contract["sim_dt"])
        step_dt = float(contract["step_dt"])
        decimation = int(contract["decimation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract missing or non-numeric "
            f"sim_dt/step_dt/decimation: {exc}"
        ) from exc
    if sim_dt <= 0 or step_dt <= 0 or decimation < 1:
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract: sim_dt={sim_dt}, "
            f"step_dt={step_dt}, decimation={decimation} (must be positive, "
            f"decimation >= 1)."
        )
    expected = round(step_dt / sim_dt)
    if abs(step_dt / sim_dt - decimation) > 1e-6 or expected != decimation:
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract: decimation ({decimation}) "
            f"does not equal round(step_dt / sim_dt) = round({step_dt} / "
            f"{sim_dt}) = {expected}."
        )

    # observations total dim must match manifest.observation_space.dim
    observations = contract.get("observations") or {}
    if not isinstance(observations, dict) or not observations:
        raise RuntimeError(
            "[bundle_finalizer] deploy_contract.observations is empty."
        )
    total = 0
    for term_name, spec in observations.items():
        if not isinstance(spec, dict):
            raise RuntimeError(
                f"[bundle_finalizer] deploy_contract.observations[{term_name!r}] "
                f"is not a dict."
            )
        try:
            dim = int(spec["dim"])
            history = int(spec.get("history_length", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[bundle_finalizer] deploy_contract.observations[{term_name!r}] "
                f"missing or non-int dim/history_length: {exc}"
            ) from exc
        if dim <= 0 or history < 1:
            raise RuntimeError(
                f"[bundle_finalizer] deploy_contract.observations[{term_name!r}]: "
                f"dim={dim}, history_length={history} (both must be positive)."
            )
        total += dim * history
    if total != int(manifest_obs_dim):
        raise RuntimeError(
            f"[bundle_finalizer] deploy_contract.observations total dim "
            f"({total}) != manifest.observation_space.dim ({manifest_obs_dim}). "
            f"Each term contributes dim * history_length."
        )


# ---------------------------------------------------------------------------
# Locator helpers — file discovery in the exported dir
# ---------------------------------------------------------------------------

def _locate_deploy_meta(exported_dir: Path) -> Optional[Path]:
    """Find the compiler-authored ``deploy_meta.json`` sidecar, if any.

    The sidecar is written by :meth:`IsaacLabConfigCompiler.compile_to_file`
    alongside ``unitport_env_cfg.py`` — which historically lives in the
    run_dir. Different layouts surface it at different relative paths:

      * ``<exported_dir>/deploy_meta.json``         — compiler wrote next to
        play.py outputs (most common when ``exported_dir`` == run_dir).
      * ``<exported_dir>/params/deploy_meta.json``  — raw RSL-RL log_dir
        layout, mirroring how ``params/env.yaml`` is staged.
      * ``<exported_dir>/../deploy_meta.json``      — when play.py writes
        into a sub-folder of run_dir but the sidecar stayed at run_dir.

    Returns the first existing path or ``None`` (back-compat path; older
    bundles produced before the sidecar existed will fall through to the
    env.yaml-only mode of the parser).
    """
    candidates = [
        exported_dir / "deploy_meta.json",
        exported_dir / "params" / "deploy_meta.json",
        exported_dir.parent / "deploy_meta.json",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _locate_env_yaml(exported_dir: Path) -> Path:
    """Find ``env.yaml`` inside the exported directory.

    Two layouts the RSL-RL pipeline produces:
      * ``<exported_dir>/env.yaml``      — Isaac Lab play.py output
      * ``<exported_dir>/params/env.yaml`` — raw RSL-RL log_dir layout
    """
    for cand in (exported_dir / "env.yaml", exported_dir / "params" / "env.yaml"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"env.yaml not found under {exported_dir} (checked root and params/)"
    )


def _locate_agent_yaml(exported_dir: Path, *, near: Path) -> Path:
    """Find ``agent.yaml`` to reconstruct the rsl_rl actor MLP topology.

    Search order matches DEMO's import path: params/ subdir first
    (canonical), then root, then the checkpoint's neighbour params/
    (covers the case where a single run dir held multiple checkpoints).
    """
    for cand in (
        exported_dir / "params" / "agent.yaml",
        exported_dir / "agent.yaml",
        near.parent / "params" / "agent.yaml",
        near.parent / "agent.yaml",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"agent.yaml not found near {exported_dir} or {near.parent}"
    )


@dataclass
class _PolicyArtifact:
    """Resolved policy file plus how we'll write it into the bundle."""
    source_path: Path        # current location on disk
    bundle_filename: str     # name in the bundle (always policy.onnx for v1)
    bundle_format: str       # always "onnx" for v1
    needs_conversion: bool   # True iff source is .pt and we'll run rsl_rl→onnx
    agent_yaml: Optional[Path]   # only set when needs_conversion=True


def _locate_policy(exported_dir: Path) -> _PolicyArtifact:
    """Find a policy file or the best ``model_*.pt`` to convert.

    Preference order (mirrors DEMO):
      1. ``<exported_dir>/policy.onnx`` or ``policy.pt``  — flat layout
      2. ``<exported_dir>/exported/policy.onnx``           — nested layout
      3. Latest ``model_<N>.pt`` in ``<exported_dir>``     — raw rsl_rl
    """
    # Path 1
    for name in ("policy.onnx", "policy.pt"):
        cand = exported_dir / name
        if cand.is_file():
            if name == "policy.onnx":
                return _PolicyArtifact(
                    source_path=cand,
                    bundle_filename="policy.onnx",
                    bundle_format="onnx",
                    needs_conversion=False,
                    agent_yaml=None,
                )
            # .pt at root → convert
            return _PolicyArtifact(
                source_path=cand,
                bundle_filename="policy.onnx",
                bundle_format="onnx",
                needs_conversion=True,
                agent_yaml=_locate_agent_yaml(exported_dir, near=cand),
            )

    # Path 2
    nested = exported_dir / "exported"
    if nested.is_dir():
        cand = nested / "policy.onnx"
        if cand.is_file():
            return _PolicyArtifact(
                source_path=cand,
                bundle_filename="policy.onnx",
                bundle_format="onnx",
                needs_conversion=False,
                agent_yaml=None,
            )

    # Path 3 — latest model_<N>.pt
    def _iter_num(p: Path) -> int:
        try:
            return int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    candidates = sorted(
        exported_dir.glob("model_*.pt"), key=_iter_num, reverse=True,
    )
    candidates = [c for c in candidates if _iter_num(c) >= 0]
    if candidates:
        cp = candidates[0]
        return _PolicyArtifact(
            source_path=cp,
            bundle_filename="policy.onnx",
            bundle_format="onnx",
            needs_conversion=True,
            agent_yaml=_locate_agent_yaml(exported_dir, near=cp),
        )

    raise FileNotFoundError(
        f"No policy.onnx, policy.pt, or model_*.pt found under {exported_dir}"
    )


# ---------------------------------------------------------------------------
# SkillManifest → RELEASE v1 12-field flat dict
# ---------------------------------------------------------------------------

def _skill_manifest_to_v1_dict(
    manifest: SkillManifest,
    *,
    bundle_name: str,
    version: str,
    policy_file: str,
    policy_format: str,
    robot_sku: str,
    joint_names: List[str],
    decimation: int,
    algorithm: str,
    deploy_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the RELEASE v1 manifest dict from the v2 SkillManifest.

    Most fields come from the SkillManifest. Three pieces the v2 schema
    does not surface explicitly are sourced from the caller (which has
    the original :class:`TrainingSpec` in scope):

      * ``robot_sku`` — SkillManifest.target_robot_family is a *family*
        name ("quadruped"), not the bound robot. Caller passes
        ``spec.robot.sku`` (the canonical key from registers.robots).
        BundleExporter.build_manifest looks up the registered
        brand/model/name from this SKU and writes them into the manifest
        as informational fields.
      * ``robot.joint_names`` — SkillManifest.joint_mapping is
        ``{policy_joint: robot_joint}`` and may be partial. Caller
        passes ``spec.robot.joint_ir_roles`` (authoritative).
      * ``runtime.decimation`` — Isaac Lab env.yaml carries it but the
        v2 SkillManifest does not. Caller computes from
        ``IsaacLabConfig.control_dt`` / sim_dt.

    ``deploy_contract`` / ``command_interface`` / ``precondition`` /
    ``postcondition`` from SkillManifest are forwarded as **optional**
    extra top-level fields (not part of the v1 12 required, but
    BundleLoader tolerates them — RELEASE plan §B3 designates them as
    backwards-compatible carry-along).
    """
    if len(joint_names) != int(manifest.action_dim):
        # Manifest is the source of truth for action_dim (it was inferred
        # from env.yaml's action manager). If joint_names disagrees the
        # spec / env.yaml are out of sync — pad/trim to match so
        # ``BundleExporter.build_manifest`` does not raise mid-write.
        # Log so the discrepancy is visible.
        log_warning(
            f"[bundle_finalizer] joint_names len={len(joint_names)} mismatches "
            f"manifest.action_dim={manifest.action_dim}; padding to "
            f"action_dim length so build_manifest accepts the bundle."
        )
        if len(joint_names) < manifest.action_dim:
            joint_names = list(joint_names) + [
                f"joint_{i}" for i in range(len(joint_names), manifest.action_dim)
            ]
        else:
            joint_names = list(joint_names)[: manifest.action_dim]

    extra: Dict[str, Any] = {}
    if manifest.command_interface is not None:
        extra["command_interface"] = {
            "type": manifest.command_interface.type,
            "fields": [
                {
                    "name": f.name,
                    "obs_index": int(f.obs_index),
                    "range": list(f.range),
                    "default": float(f.default),
                }
                for f in manifest.command_interface.fields
            ],
            "input_sources": list(manifest.command_interface.input_sources),
        }
    if manifest.precondition is not None:
        extra["precondition"] = {
            "posture": manifest.precondition.posture,
            "posture_tolerance_rad": float(manifest.precondition.posture_tolerance_rad),
            "velocity_max_mps": float(manifest.precondition.velocity_max_mps),
        }
    if manifest.postcondition is not None:
        extra["postcondition"] = {
            "posture": manifest.postcondition.posture,
            "posture_tolerance_rad": float(manifest.postcondition.posture_tolerance_rad),
            "velocity_max_mps": float(manifest.postcondition.velocity_max_mps),
        }
    if manifest.observation_space_keys:
        extra["observation_space_keys"] = list(manifest.observation_space_keys)
    if manifest.required_sensors:
        extra["required_sensors"] = list(manifest.required_sensors)

    return BundleExporter.build_manifest(
        name=bundle_name,
        version=version,
        obs_dim=int(manifest.observation_dim),
        action_dim=int(manifest.action_dim),
        action_type=str(manifest.action_space_type.value),
        control_frequency_hz=float(manifest.control_frequency_hz),
        decimation=int(decimation),
        robot_sku=str(robot_sku),
        joint_names=list(joint_names),
        policy_file=str(policy_file),
        policy_format=str(policy_format),
        algorithm=str(algorithm),
        deploy_contract=deploy_contract,
        inference_convention="isaac_lab",
        extra=extra or None,
    )


# ---------------------------------------------------------------------------
# AMP discriminator extraction (only invoked when is_amp=True)
# ---------------------------------------------------------------------------

def _extract_normalization_stats(
    source_pt: Path, dst: Path
) -> Tuple[bool, Dict[str, Any]]:
    """Extract EmpiricalNormalization-style obs/value running stats from
    an rsl_rl checkpoint into a standalone ``normalization_stats.json``.

    Looks for keys matching the EmpiricalNormalization layer pattern on
    both the standard rsl_rl OnPolicyRunner save shape ({actor/critic
    state_dict}) and the AMP runner save shape ({model_state_dict}).
    Recognised key suffixes: ``mean`` / ``var`` / ``count`` /
    ``running_mean`` / ``running_var`` / ``running_count``.

    Returns ``(saved, payload)``. saved=False means the checkpoint has
    no normalization layer (the default RSL-RL path is
    ``obs_normalization=False``; the layer is absent unless the launcher
    explicitly turned it on). amp_normalizer is intentionally NOT
    duplicated here — _extract_discriminator_pt already stages it into
    discriminator.pt alongside the discriminator weights.
    """
    import json as _json
    import torch

    src = torch.load(str(source_pt), map_location="cpu", weights_only=False)
    if not isinstance(src, dict):
        return (False, {})

    _NORM_SUFFIXES = (
        "mean", "var", "count",
        "running_mean", "running_var", "running_count",
    )

    def _is_norm_key(k: str) -> bool:
        leaf = k.rsplit(".", 1)[-1]
        return leaf in _NORM_SUFFIXES and ("norm" in k.lower() or leaf in _NORM_SUFFIXES)

    payload: Dict[str, Any] = {}
    for sd_key in ("actor_state_dict", "critic_state_dict", "model_state_dict"):
        sd = src.get(sd_key)
        if not isinstance(sd, dict):
            continue
        norm_subset = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in sd.items()
            if _is_norm_key(k)
        }
        if norm_subset:
            payload[sd_key] = norm_subset

    if not payload:
        return (False, {})

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(dst))
    return (True, payload)


def _extract_discriminator_pt(
    source_pt: Path, dst: Path
) -> Tuple[bool, List[str]]:
    """Extract ``discriminator_state_dict`` (+ ``amp_normalizer`` when present)
    from a raw rsl_rl checkpoint into a standalone ``discriminator.pt``.

    Atomic: writes to ``<dst>.tmp`` then ``os.replace``. Drops optimizer +
    replay buffer (training-only state, would bloat the bundle).

    Returns ``(archived, keys_saved)``. ``archived`` False = source has
    no discriminator key (likely a non-AMP checkpoint that slipped past
    the caller's ``is_amp`` gate, or an older checkpoint format).
    """
    import torch

    src = torch.load(str(source_pt), map_location="cpu", weights_only=False)
    if not isinstance(src, dict) or "discriminator_state_dict" not in src:
        return (False, [])

    bundle: Dict[str, Any] = {
        "discriminator_state_dict": src["discriminator_state_dict"],
    }
    if "amp_normalizer" in src:
        bundle["amp_normalizer"] = src["amp_normalizer"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(bundle, str(tmp))
    os.replace(str(tmp), str(dst))
    return (True, list(bundle.keys()))


# ---------------------------------------------------------------------------
# Public entry — finalize_isaac_lab_bundle
# ---------------------------------------------------------------------------

def finalize_isaac_lab_bundle(
    *,
    exported_dir: Path,
    bundle_name: str,
    version: str,
    overwrite: bool,
    project: Any,                 # ProjectInfo
    spec: Any,                    # TrainingSpec
    run_id: str,
    is_amp: bool = False,
) -> ExportResult:
    """Atomically finalize an Isaac Lab training run into a v1 bundle.

    Args:
        exported_dir: directory containing Isaac Lab's training/play.py
            output — at minimum ``env.yaml`` + a policy file (``policy.onnx``,
            ``policy.pt``, or ``model_*.pt``).
        bundle_name: bundle directory name in
            ``<project>/training/exported/isaac_lab/<bundle_name>/``.
        version: bundle version string (mirrored into source.json).
        overwrite: if True, replace any existing bundle dir; if False,
            ``BundleExporter.export_from_artifacts`` raises FileExistsError.
        project: bound :class:`ProjectInfo` — bundle lands under it.
        spec: :class:`TrainingSpec` — used to fill manifest fields the
            v2 SkillManifest does not surface (robot.brand, joint_names,
            decimation).
        run_id: training run id, mirrored into source.json.
        is_amp: when True, also extract ``discriminator.pt`` into the
            bundle root from the source checkpoint. Caller (IL Task)
            knows this from ``algorithm.training_mode == "AMP_PPO"``.

    Returns:
        :class:`ExportResult` from
        :meth:`BundleExporter.export_from_artifacts`.

    Raises:
        FileNotFoundError: env.yaml or policy file missing under
            ``exported_dir``.
        FileExistsError: bundle dir already exists with overwrite=False.
        Other exceptions from manifest parsing / ONNX conversion are
        surfaced verbatim — caller is responsible for logging.
    """
    exported_dir = Path(exported_dir)
    if not exported_dir.is_dir():
        raise FileNotFoundError(
            f"Isaac Lab exported_dir does not exist or is not a dir: {exported_dir}"
        )

    log_info(
        f"[bundle_finalizer] starting finalize: name={bundle_name!r} "
        f"version={version!r} overwrite={overwrite} amp={is_amp} "
        f"exported_dir={exported_dir}"
    )

    # 1. Locate inputs.
    env_yaml = _locate_env_yaml(exported_dir)
    policy = _locate_policy(exported_dir)
    deploy_meta = _locate_deploy_meta(exported_dir)
    log_info(
        f"[bundle_finalizer] env.yaml={env_yaml}  "
        f"policy={policy.source_path} (convert={policy.needs_conversion}) "
        f"deploy_meta={deploy_meta if deploy_meta else '<none>'}"
    )

    # 2. ONNX bytes — either read directly or convert in-process.
    if policy.needs_conversion:
        from application.training.isaac_lab.onnx_export import (
            export_rsl_rl_actor_to_onnx,
        )
        # Convert into a temp file alongside the source then read bytes.
        # Using a temp file (not BytesIO) because export_rsl_rl_actor_to_onnx
        # mirrors DEMO's API which writes via torch.onnx.export → Path.
        tmp_onnx = exported_dir / f".finalize_{run_id}.onnx.tmp"
        try:
            export_rsl_rl_actor_to_onnx(
                checkpoint_path=policy.source_path,
                agent_yaml_path=policy.agent_yaml,
                onnx_out_path=tmp_onnx,
            )
            onnx_bytes = tmp_onnx.read_bytes()
        finally:
            try:
                if tmp_onnx.exists():
                    tmp_onnx.unlink()
            except Exception:
                pass
        log_info(f"[bundle_finalizer] converted .pt → ONNX ({len(onnx_bytes)} bytes)")
    else:
        onnx_bytes = policy.source_path.read_bytes()

    # 3. Parse env.yaml → v2 SkillManifest. The v2 form's main job here
    # is to authoritatively compute observation_dim from env.yaml's
    # ObsTerm layout (which the spec alone does not always know — Isaac
    # Lab can override obs terms per task).
    #
    # Robot identity (Phase 5): prefer to pass a full RobotSpecRef so
    # parse_isaac_lab_env_yaml can write IR roles into manifest joint_names
    # (the IR-only deploy contract). RobotSpecRef carries IR↔physical
    # parallel arrays + brand/model, so the parser does not need the
    # brand_package model_registry path. Reconstruct from spec.robot
    # whether spec is a TrainingSpec instance or a dict.
    robot_for_parser: Any = None
    if spec is not None and getattr(spec, "robot", None) is not None:
        from application.training.training_spec import RobotSpecRef
        sr = spec.robot
        if isinstance(sr, RobotSpecRef):
            robot_for_parser = sr
        elif isinstance(sr, dict):
            from application.training.training_spec import _spec_from_dict
            robot_for_parser = _spec_from_dict(RobotSpecRef, sr)

    sm = parse_isaac_lab_env_yaml(
        env_yaml,
        skill_id=bundle_name,
        model_path=policy.bundle_filename,
        robot=robot_for_parser,
        deploy_meta_path=deploy_meta,
    )

    # 4. Convert SkillManifest → RELEASE v1 12-field manifest dict.
    # Phase 5 IR-only contract: manifest.robot.joint_names = IR roles
    # (so deploy stack can resolve to physical via robot_sku at run time
    # and the same bundle works across same-family robots).
    robot_sku = ""
    joint_names: List[str] = []
    decimation = 4   # safe default for 50 Hz control over 200 Hz sim
    algorithm = "PPO"
    isaac_lab_order: List[str] = []
    if spec is not None and getattr(spec, "robot", None) is not None:
        # SKU-only contract: the spec MUST carry a registered SKU.
        # ``RobotSpecRef.from_registry`` populates spec.robot.sku from
        # the canvas RobotNode's binding; if it's empty here, lowering
        # produced an unregistered robot and the fix belongs upstream
        # — never paper-over by reading display strings.
        robot_sku = str(getattr(spec.robot, "sku", "") or "")
        # IR roles, NOT physical names — the substrate-side translation
        # happens deploy-time via ir_roles_to_physical_names(roles, sku).
        joint_names = list(getattr(spec.robot, "joint_ir_roles", []) or [])
        if not joint_names:
            # Fallback: legacy spec without joint_ir_roles populated → use
            # joint_order so the manifest is at least self-consistent on
            # length. Should not happen in Phase 5+ flows.
            joint_names = list(getattr(spec.robot, "joint_order", []) or [])
        isaac_lab_order = list(
            getattr(spec.robot, "isaac_lab_joint_order", None) or []
        )

    # Phase 5 IL deploy contract: Isaac Lab's ManagerBasedRLEnv uses the
    # articulation's native USD prim joint order, which for Unitree
    # quadrupeds is grouped by joint TYPE (all hips → all thighs → all
    # calves), NOT the SDK canonical "by leg" order recorded in
    # robots_canonical.json's joint table. Without this reorder, the
    # bundle ships joint_sdk_names / default_joint_pos in the wrong
    # order → ObsBuilder feeds joint_pos/joint_vel/last_action to the
    # policy in wrong slots → "twitching like electric shock" at deploy.
    # See plan: release-demo-sim2sim-policy-...md §1.
    joint_permutation: Optional[List[int]] = None
    if isaac_lab_order and joint_names:
        canonical_index = {role: i for i, role in enumerate(joint_names)}
        try:
            candidate = [canonical_index[r] for r in isaac_lab_order]
        except KeyError as exc:
            log_warning(
                f"[bundle_finalizer] isaac_lab_joint_order role {exc} not "
                f"in spec.robot.joint_ir_roles for SKU {robot_sku!r}; "
                f"falling back to SDK canonical order — sim2sim parity "
                f"NOT guaranteed."
            )
            candidate = None
        if candidate is not None and len(candidate) == len(joint_names):
            joint_permutation = candidate
        elif candidate is not None:
            log_warning(
                f"[bundle_finalizer] isaac_lab_joint_order length "
                f"({len(candidate)}) ≠ joint_ir_roles ({len(joint_names)}) "
                f"for SKU {robot_sku!r}; falling back."
            )
    elif joint_names and robot_sku:
        log_warning(
            f"[bundle_finalizer] robot_sku={robot_sku!r} has no "
            f"isaac_lab_joint_order in robots_canonical.json; bundle "
            f"uses SDK canonical joint order which may not match Isaac "
            f"Lab USD articulation order — sim2sim parity NOT guaranteed. "
            f"Add the field to robots_canonical.json after confirming USD "
            f"joint order for this robot."
        )

    if joint_permutation is not None:
        joint_names = [joint_names[i] for i in joint_permutation]
    # Decimation MUST match what env_cfg_compiler emitted into the trained
    # UnitPortEnvCfg — otherwise the bundle declares a control frequency the
    # policy was never trained for and deploy runs at the wrong rate.
    # env_cfg_compiler reads sim_dt from play_ground_setting (= spec.scene.sim_dt)
    # and hardcodes control_dt = 0.02 (50 Hz). physics_config is a SB3-only
    # node and intentionally NOT consulted on the IL path; reading
    # spec.physics here used to silently fork the two values when the canvas
    # had both nodes with different sim_dt.
    sim_dt = 0.0
    if spec is not None and getattr(spec, "scene", None) is not None:
        sim_dt = float(getattr(spec.scene, "sim_dt", 0.0) or 0.0)
    if sim_dt <= 0 and spec is not None and getattr(spec, "physics", None) is not None:
        sim_dt = float(getattr(spec.physics, "sim_dt", 2e-3) or 2e-3)
    if sim_dt > 0:
        decimation = max(1, int(round(0.02 / sim_dt)))
    if spec is not None and getattr(spec, "algorithm", None) is not None:
        # spec.algorithm.training_mode wins (AMP_PPO / PPO); algorithm field
        # is the underlying SB3 algo (PPO/SAC/TD3) for SB3 chain — for IL
        # we want training_mode if present.
        mode = str(
            getattr(spec.algorithm, "training_mode", "")
            or getattr(spec.algorithm, "algorithm", "PPO")
            or "PPO"
        ).upper()
        algorithm = mode

    # Inject spec.robot.sku into the parser-built contract (the parser
    # doesn't have spec in scope so it leaves robot_sku=""). Phase 5 IR
    # resolution at runtime keys off this field.
    deploy_contract_dict: Optional[Dict[str, Any]] = None
    if sm.deploy_contract is not None:
        deploy_contract_dict = dict(sm.deploy_contract)
        if not deploy_contract_dict.get("robot_sku"):
            deploy_contract_dict["robot_sku"] = robot_sku

        # Persist the Isaac Lab training init root position so MuJoCo
        # deploy spawns the robot at the same base z the policy was
        # trained on. Without this, the deploy stack falls back to a
        # contact-based heuristic over the MJCF keyframe pose, which
        # silently drifts off-distribution for any IL bundle whose
        # default_joint_pos differs from the keyframe leg geometry
        # (which it does for every real IL Go2 / quadruped training).
        # spec_compiler reads this from canvas actor_setting.init_pos_z
        # (default 0.4); env_cfg_compiler already wires the same value
        # into Isaac Lab's ArticulationCfg.InitialStateCfg.pos, so
        # round-trip parity is guaranteed.
        if spec is not None and getattr(spec, "actor", None) is not None:
            try:
                init_pos_x = float(getattr(spec.actor, "init_pos_x", 0.0) or 0.0)
                init_pos_y = float(getattr(spec.actor, "init_pos_y", 0.0) or 0.0)
                init_pos_z = float(getattr(spec.actor, "init_pos_z", 0.0) or 0.0)
            except (TypeError, ValueError):
                init_pos_x = init_pos_y = init_pos_z = 0.0
            if init_pos_z > 0.0:
                deploy_contract_dict["init_base_pos"] = [
                    init_pos_x, init_pos_y, init_pos_z
                ]

        # Apply the same Isaac-Lab-order permutation to every parallel
        # per-joint array in the deploy_contract so they line up with
        # the manifest's reordered joint_names. After this step
        # sdk_order == bundle_order == Isaac Lab training order, so
        # joint_ids_map collapses to identity.
        if joint_permutation is not None:
            n_perm = len(joint_permutation)
            for field_name in (
                "joint_sdk_names",
                "default_joint_pos",
                "stiffness",
                "damping",
                "effort_limit",
                "velocity_limit",
                "saturation_effort",
            ):
                arr = deploy_contract_dict.get(field_name)
                if isinstance(arr, list) and len(arr) == n_perm:
                    deploy_contract_dict[field_name] = [
                        arr[i] for i in joint_permutation
                    ]
            deploy_contract_dict["joint_ids_map"] = list(range(n_perm))

    manifest_dict = _skill_manifest_to_v1_dict(
        sm,
        bundle_name=bundle_name,
        version=version,
        policy_file=policy.bundle_filename,
        policy_format=policy.bundle_format,
        robot_sku=robot_sku,
        joint_names=joint_names,
        decimation=decimation,
        algorithm=algorithm,
        deploy_contract=deploy_contract_dict,
    )

    # 4.5. Validate computed obs_dim against the actual ONNX input shape.
    # ``_compute_obs_layout`` is a heuristic over env.yaml term names; a
    # mismatch with the trained policy's real input width corrupts the
    # bundle silently — Launch Review then fails with InvalidArgument at
    # runtime and the user sees only ``reason=error``. Fail-fast here so
    # the broken bundle is never written to disk.
    _validate_onnx_obs_dim(
        onnx_bytes=onnx_bytes,
        manifest_obs_dim=int(manifest_dict["observation_space"]["dim"]),
        manifest_action_dim=int(manifest_dict["action_space"]["dim"]),
        obs_keys=list(sm.observation_space_keys or ()),
    )

    # 4.6. Validate the deploy_contract block (Module B) — every check
    # DeployContract.from_dict performs at load time, run here so a broken
    # bundle is never written. Skip cleanly for bundles that don't carry a
    # contract (e.g. legacy SB3 path).
    if deploy_contract_dict is not None:
        _validate_deploy_contract(
            contract=deploy_contract_dict,
            manifest_obs_dim=int(manifest_dict["observation_space"]["dim"]),
            manifest_action_dim=int(manifest_dict["action_space"]["dim"]),
        )

    # 5. (AMP only) extract discriminator.pt — staged adjacent to the
    # final bundle dir so we can hand it to BundleExporter via a
    # post-write hook (BundleExporter.export_from_artifacts is atomic
    # on the bundle dir, so we cannot interleave the discriminator copy
    # mid-stage). Strategy: write the bundle first, then drop
    # discriminator.pt in place. If discriminator extraction fails we
    # log + continue — the bundle itself is still valid for inference.
    pending_discriminator: Optional[Path] = None
    if is_amp and policy.source_path.suffix.lower() == ".pt":
        # Stage to a sibling temp file; we'll move it into the bundle
        # dir after export_from_artifacts returns.
        tmp_disc = exported_dir / f".finalize_{run_id}.discriminator.pt.tmp"
        try:
            archived, keys = _extract_discriminator_pt(
                policy.source_path, tmp_disc,
            )
            if archived:
                pending_discriminator = tmp_disc
                manifest_dict["amp_metadata"] = {
                    "source_checkpoint": policy.source_path.name,
                    "discriminator_archived": True,
                    "discriminator_keys": keys,
                }
                log_info(
                    f"[bundle_finalizer] AMP discriminator staged "
                    f"({len(keys)} key(s) in {tmp_disc.name})"
                )
            else:
                log_warning(
                    f"[bundle_finalizer] is_amp=True but {policy.source_path.name} "
                    f"has no discriminator_state_dict — skipping discriminator.pt"
                )
        except Exception as exc:
            log_warning(
                f"[bundle_finalizer] AMP discriminator extraction failed "
                f"({exc}) — bundle proceeds without discriminator.pt"
            )

    # 6. Atomic write through the shared exporter. backend_id is hard-
    # coded "isaac_lab" per rule §B2 — never relies on AppSignals
    # fallback for the IL chain.
    source_meta = {
        "type": "isaac_lab",
        "name": bundle_name,
        "version": version,
        "run_id": str(run_id),
        "algorithm": algorithm,
        "robot_sku": robot_sku,
        "robot_joint_ir_roles": list(joint_names),
        "exported_dir": str(exported_dir),
        "policy_source": str(policy.source_path.name),
        "policy_converted": bool(policy.needs_conversion),
    }
    # Forward Export-node metadata so the UI's Launch Review button +
    # any downstream tooling can read which review backend / scene the
    # user picked + which bundle targets they wanted. These fields are
    # informational on the bundle side (review subprocess wiring is
    # owned by the UI button widget, not this finalizer).
    if spec is not None and getattr(spec, "export", None) is not None:
        export_cfg = spec.export
        review_backend = str(getattr(export_cfg, "review_backend", "") or "")
        review_scene_id = str(getattr(export_cfg, "review_scene_id", "") or "")
        bundle_targets = list(getattr(export_cfg, "bundle_targets", None) or [])
        if review_backend:
            source_meta["review_backend"] = review_backend
        if review_scene_id:
            source_meta["review_scene_id"] = review_scene_id
        if bundle_targets:
            source_meta["bundle_targets"] = bundle_targets
            # Multi-target bundling is reserved for a later iteration; today
            # the finalizer always produces a single runtime_bundle. Warn
            # loudly so the canvas author knows the extra entries were not
            # honoured rather than silently believing N bundles were written.
            extra = [t for t in bundle_targets if t != "runtime_bundle"]
            if extra:
                log_warning(
                    f"[bundle_finalizer] bundle_targets has extra entries "
                    f"{extra!r}; current finalizer only emits a single "
                    f"runtime_bundle. Targets recorded in source.json for "
                    f"future tooling but no additional artifacts written."
                )

    out = BundleExporter.export_from_artifacts(
        name=bundle_name,
        version=version,
        manifest=manifest_dict,
        onnx_source=onnx_bytes,
        source_meta=source_meta,
        overwrite=overwrite,
        project=project,
        backend_id="isaac_lab",
    )

    # 6.4. (Optional) Normalization stats — when the Export node sets
    # include_normalization=True AND the checkpoint actually has obs
    # running stats (default launcher path is obs_normalization=False
    # so this is usually a no-op; AMP-specific amp_normalizer goes via
    # _extract_discriminator_pt into discriminator.pt). Drop the stats
    # as JSON next to the bundle root so the deploy-side Normalizer can
    # populate its running mean/var from the same offline values the
    # policy saw at the last training iter.
    include_norm = False
    if spec is not None and getattr(spec, "export", None) is not None:
        include_norm = bool(getattr(spec.export, "include_normalization", False))
    if include_norm and policy.source_path.suffix.lower() == ".pt":
        try:
            norm_target = out.bundle_path / "normalization_stats.json"
            saved_norm, _norm_payload = _extract_normalization_stats(
                policy.source_path, norm_target,
            )
            if saved_norm:
                log_info(
                    f"[bundle_finalizer] normalization stats → {norm_target}"
                )
            else:
                log_info(
                    "[bundle_finalizer] include_normalization=True but "
                    "checkpoint has no obs-normalization layer "
                    "(rsl_rl was trained with obs_normalization=False); "
                    "no normalization_stats.json written."
                )
        except Exception as exc:
            log_warning(
                f"[bundle_finalizer] normalization extract failed ({exc}); "
                f"bundle still valid without it."
            )

    # 6.5. (Optional) TorchScript companion — when the Export node sets
    # include_torchscript=True, drop ``policy.pt`` next to ``policy.onnx``
    # so deploy stacks that prefer ``torch.jit.load`` (no onnxruntime
    # dependency) have a native option. Only reachable when the source
    # is a .pt checkpoint we can rebuild the actor MLP from; if we landed
    # here via a pre-baked ONNX, the original weights are gone.
    include_ts = False
    if spec is not None and getattr(spec, "export", None) is not None:
        include_ts = bool(getattr(spec.export, "include_torchscript", False))
    if include_ts and policy.source_path.suffix.lower() == ".pt":
        try:
            from application.training.isaac_lab.onnx_export import (
                export_rsl_rl_actor_to_torchscript,
            )
            agent_yaml = policy.agent_yaml or _locate_agent_yaml(
                exported_dir, near=policy.source_path,
            )
            ts_target = out.bundle_path / "policy.pt"
            export_rsl_rl_actor_to_torchscript(
                checkpoint_path=policy.source_path,
                agent_yaml_path=agent_yaml,
                ts_out_path=ts_target,
            )
            log_info(f"[bundle_finalizer] TorchScript policy → {ts_target}")
        except Exception as exc:
            log_warning(
                f"[bundle_finalizer] include_torchscript=True but conversion "
                f"failed ({exc}); bundle still valid with policy.onnx only."
            )

    # 7. Move staged discriminator into the bundle dir (best-effort —
    # bundle is already valid even if this fails).
    if pending_discriminator is not None:
        try:
            target = out.bundle_path / "discriminator.pt"
            os.replace(str(pending_discriminator), str(target))
            log_info(f"[bundle_finalizer] discriminator.pt → {target}")
        except Exception as exc:
            log_warning(
                f"[bundle_finalizer] discriminator move failed ({exc}); "
                f"orphaned at {pending_discriminator}"
            )

    # 8. Drop env.yaml as provenance alongside the bundle (DEMO does
    # this — useful for rsl_rl resume / sim-to-real env reconstruction).
    try:
        shutil.copy2(str(env_yaml), str(out.bundle_path / "env.yaml"))
    except Exception as exc:
        log_warning(
            f"[bundle_finalizer] env.yaml copy failed ({exc}); "
            f"bundle still valid."
        )

    # 8b. Drop the compiler-authored deploy_meta.json into the bundle so
    # downstream loaders / re-export tooling can recover the same
    # compile-time decisions without needing access to the run_dir.
    if deploy_meta is not None:
        try:
            shutil.copy2(str(deploy_meta),
                         str(out.bundle_path / "deploy_meta.json"))
        except Exception as exc:
            log_warning(
                f"[bundle_finalizer] deploy_meta.json copy failed "
                f"({exc}); bundle still valid (parser will fall back to "
                f"env.yaml-only mode if loaded later)."
            )

    log_info(f"[bundle_finalizer] DONE → {out.bundle_path}")
    return out


__all__ = ["finalize_isaac_lab_bundle"]
