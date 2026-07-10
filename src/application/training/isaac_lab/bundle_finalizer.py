# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import log_debug, log_info, log_warning

from application.training.bundle_exporter import (
    BundleExporter,
    ExportResult,
    assert_policy_contract_consistent,
    build_policy_contract_block,
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

    # Stage F: v2 PD provenance fields. When pd_param is present, the
    # per-engine derived arrays MUST be present and length-matched —
    # UNLESS the deploy target is explicitly marked unsupported (e.g.
    # mujoco_deploy_unsupported set when MJCF can't cover the trained
    # joint set; bundle still ships for IsaacSim / cloud targets). The
    # marker carries the reason text so MuJoCo runtime loader can
    # surface it on refused-load.
    if contract.get("pd_param") is not None:
        mujoco_unsupported = bool(contract.get("mujoco_deploy_unsupported"))
        for key in ("mujoco_pd_gains", "mujoco_pd_damping"):
            lst = contract.get(key)
            if mujoco_unsupported:
                # When MuJoCo deploy is explicitly opted out, both arrays
                # MUST be None (sentinel) — present-but-wrong-length is
                # still a bug we should catch.
                if lst is not None:
                    raise RuntimeError(
                        f"[bundle_finalizer] deploy_contract.{key} = "
                        f"{type(lst).__name__} but mujoco_deploy_unsupported "
                        f"is set — expected None when MuJoCo target opted "
                        f"out."
                    )
                continue
            if not isinstance(lst, list) or len(lst) != n:
                raise RuntimeError(
                    f"[bundle_finalizer] deploy_contract.{key} has length "
                    f"{len(lst) if isinstance(lst, list) else 'N/A'}, "
                    f"expected {n} (pd_param is set; per-engine derived "
                    f"arrays must be populated)."
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

    ``deploy_contract`` / ``precondition`` / ``postcondition`` from
    SkillManifest are forwarded as **optional** extra top-level fields
    (not part of the v1 12 required, but BundleLoader tolerates them —
    RELEASE plan §B3 designates them as backwards-compatible
    carry-along). ``command_interface`` was retired 2026-07 in favour of
    ``deploy_contract.commands`` (v1 CommandContract).
    """
    if len(joint_names) != int(manifest.action_dim):
        # CLAUDE.md §1.8: the legacy behaviour padded/trimmed joint_names to
        # manifest.action_dim length (filling with synthetic ``joint_N``
        # placeholders) so ``BundleExporter.build_manifest`` would accept
        # the bundle. That hid a real upstream bug — manifest.action_dim
        # was being silently substituted with 12 (the Go2 default) by
        # ``_extract_action_config`` whenever env.yaml's ``joint_pos``
        # action lacked a static ``dim`` field, corrupting the bundle's
        # joint table with fake names that downstream code (PolicyRunner,
        # deploy_contract validators) couldn't map. Refuse to ship the
        # bundle and surface the real disagreement so the upstream
        # mis-inference can be fixed at its source.
        raise RuntimeError(
            f"[bundle_finalizer] joint_names length ({len(joint_names)}) "
            f"does not match manifest.action_dim ({manifest.action_dim}). "
            f"Both should equal the trained policy's action width — the "
            f"disagreement means either env.yaml's action manager was "
            f"mis-parsed (check _extract_action_config) or "
            f"spec.robot.joint_ir_roles was sourced from the wrong "
            f"format. Refusing to fabricate placeholder joint names "
            f"(CLAUDE.md §1.8). joint_names sample: "
            f"{list(joint_names)[:6]}{'...' if len(joint_names) > 6 else ''}"
        )

    extra: Dict[str, Any] = {}
    # RETIRED (2026-07): the ``command_interface`` carry-along. Superseded by
    # ``deploy_contract.commands`` (v1 CommandContract, single emitter in
    # command_schema.py) — per-channel name/kind/range/default now live
    # there. Old bundles still carrying the manifest key load through the
    # §8(c) legacy branch of review_session._resolve_review_command.
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
        # Isaac Lab / rsl_rl trains in PhysX with joints in USD prim order;
        # this finalizer reorders every per-joint array to that order
        # (see _build_usd_ordered_* below). Declare it so the deploy stack
        # validates against USD (and SB3 bundles, declared MJCF, are not
        # forced into USD). DISTINCT from inference_convention. §11.
        joint_array_format="USD",
        extra=extra or None,
    )


# ---------------------------------------------------------------------------
# AMP discriminator extraction (only invoked when is_amp=True)
# ---------------------------------------------------------------------------

_NORM_KEY_SUFFIXES = (
    "mean", "var", "count", "running_mean", "running_var", "running_count",
)


def _checkpoint_has_normalization(source_pt: Path) -> bool:
    """Build-time probe: does the rsl_rl checkpoint carry an obs-normalization layer?

    Used to set ``policy_contract.normalization_present`` BEFORE export (the
    actual ``normalization_stats.json`` is written later, after the atomic bundle
    write). Mirrors the detection in :func:`_extract_normalization_stats` but
    never writes. Returns False on any read error (the WARN-tier field degrades
    to "absent" rather than blocking finalize — §8(a)-style cold-path guard).
    """
    try:
        import torch

        # WHY KEPT: trusted trainer artifact written by our own runner; same
        # rationale as _extract_normalization_stats (weights_only=False needed
        # for the Normalizer pickle). Cold path only.
        src = torch.load(str(source_pt), map_location="cpu", weights_only=False)
        if not isinstance(src, dict):
            return False
        for sd_key in ("actor_state_dict", "critic_state_dict", "model_state_dict"):
            sd = src.get(sd_key)
            if not isinstance(sd, dict):
                continue
            for k in sd:
                leaf = k.rsplit(".", 1)[-1]
                if leaf in _NORM_KEY_SUFFIXES and ("norm" in k.lower() or leaf in _NORM_KEY_SUFFIXES):
                    return True
        return False
    except Exception:  # noqa: BLE001 — cold-path probe must never block finalize
        return False


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

    # WHY KEPT: source_pt 由本 finalize pipeline 上游的 AmpOnPolicyRunner.save() /
    # OnPolicyRunner.save() 写出，trusted trainer artifact，路径不取自用户输入。
    # weights_only=True 会拒绝合法的 Normalizer / optimizer pickle。Plan P1-1.
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

    # WHY KEPT: source_pt 由本 finalize pipeline 上游的 AmpOnPolicyRunner.save()
    # 写出，trusted trainer artifact。weights_only=True 拒绝合法 pickle。Plan P1-1.
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


def _build_sim2sim_audit_block(
    *,
    matrix: Any,
    scene_contract: Optional[Dict[str, Any]],
    foot_geometry: Optional[Dict[str, Any]],
    calibration_skipped: bool,
) -> Dict[str, Any]:
    """Assemble ``deploy_contract.sim2sim_audit`` from the calibration matrix.

    BOUNDARY (decreed): this block records ONLY the two-engine realized
    CONSISTENCY verdicts ("are the two ends consistent?"). The VALUES and their
    derivation — armature, m_eff, how kp is derived — stay in ``pd_derivation``;
    the audit references those quantities by joint name and never re-stores
    them. Friction (``scene_contract``) and the foot-geometry hash have no
    pd_derivation home (they are not PD inputs), so their reference values DO
    live here — no overlap. ``foot_geometry.aggregate`` is what the runtime
    PolicyRunner re-hashes the deploy MJCF against.
    """
    block: Dict[str, Any] = {
        "schema_version": 1,
        "calibration_skipped": bool(calibration_skipped),
        "scene_contract": scene_contract,
        "foot_geometry": foot_geometry,
    }
    if matrix is None:
        block["overall_verdict"] = (
            "skipped" if calibration_skipped else "unavailable"
        )
        return block
    block["overall_verdict"] = matrix.verdict
    block["dimensions"] = list(matrix.dimensions)
    joint_rows: List[Dict[str, Any]] = []
    engine_rows: Dict[str, Any] = {}
    for row in matrix.rows:
        if row.row_kind == "joint":
            joint_rows.append({
                "ir_role": row.row_id,
                "physical_name": row.joint_physical_name,
                "group": row.group,
                # verdict + relative_diff + thresholds ONLY — never the armature
                # / m_eff values (those live in pd_derivation; match by name).
                "cells": {
                    dim: {
                        "applicable": c.applicable,
                        "verdict": c.verdict,
                        "relative_diff": c.relative_diff,
                        "warn_threshold": c.warn_threshold,
                        "fail_threshold": c.fail_threshold,
                    }
                    for dim, c in row.cells.items()
                },
            })
        else:
            c = next(iter(row.cells.values())) if row.cells else None
            if c is not None:
                engine_rows[row.row_id] = {
                    "applicable": c.applicable,
                    "verdict": c.verdict,
                    "realized": c.realized,
                    "designed": c.designed,
                    "relative_diff": c.relative_diff,
                    "notes": list(c.notes),
                }
    block["joint_rows"] = joint_rows
    block["engine_rows"] = engine_rows
    return block


def _derive_mujoco_pd_gains_for_bundle(
    *,
    deploy_meta_path: Optional[Path],
    robot_sku: str,
    joint_names_ir: List[str],
    default_joint_pos: List[float],
) -> Optional[Dict[str, Any]]:
    """Read pd_param from deploy_meta.json and derive MuJoCo per-joint kp/kd.

    Returns ``{"pd_param": {...}, "mujoco_pd_gains": [...],
    "mujoco_pd_damping": [...]}`` ready to merge into deploy_contract_dict,
    or ``None`` when no ActuatorPDNode was wired (no ``unitport_pd_param``
    block in the sidecar).

    The returned arrays are aligned with ``joint_names_ir`` (the SDK
    canonical order). The bundle finalizer's joint_permutation block
    reorders them into Isaac-Lab order along with the other per-joint
    arrays.

    Raises (fail-loud per RELEASE/CLAUDE.md §1.8) when:
      * pd_param is present but the robot has no MJCF (cannot solve);
      * mujoco_gain_solver fails (NaN inertia, joint name mismatch, etc).
    """
    if deploy_meta_path is None:
        return None
    try:
        meta = json.loads(Path(deploy_meta_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log_warning(
            f"[bundle_finalizer] failed to read deploy_meta.json at "
            f"{deploy_meta_path!r} ({exc}); skipping MuJoCo PD derivation."
        )
        return None
    pd_block = (meta or {}).get("unitport_pd_param")
    if not pd_block:
        return None
    pd_param_dict = pd_block.get("pd_param")
    if not isinstance(pd_param_dict, dict):
        log_warning(
            "[bundle_finalizer] deploy_meta.unitport_pd_param.pd_param is "
            "missing or malformed; skipping MuJoCo PD derivation."
        )
        return None

    # Resolve MJCF path from the SKU. RobotSpec stores the canonical raw
    # string (relative like "menagerie/<dir>/scene.xml"); the real on-disk
    # resolution against _canonical_asset_roots happens in
    # RobotAssetService.resolve() — same code path the canvas / RobotNode
    # uses, so this stays in sync with the UI's view of the asset.
    from registers import robots as _registers_robots
    from application.service.robot_assets.service import get_robot_asset_service

    rs = _registers_robots.get_robot_spec(robot_sku)
    if rs is None:
        raise RuntimeError(
            f"[bundle_finalizer] robot_sku={robot_sku!r} not in registry; "
            f"cannot resolve MJCF for MuJoCo PD derivation."
        )

    asset = get_robot_asset_service().resolve(robot_sku)
    resolved_mjcf = asset.mjcf_path if asset is not None else None
    if resolved_mjcf is None or not resolved_mjcf.is_file():
        # No MJCF on disk → can't run mj_fullM. Surface as a hard error
        # (rule §1.8 fail-loud): the canvas has an ActuatorPDNode wired,
        # so the bundle MUST carry MuJoCo gains for the (omega_n, zeta)
        # contract to hold. Telling the user to run "Dump MJCF" is the
        # cleaner failure mode than silently producing a half-v2 bundle.
        raw_hint = getattr(rs, "mjcf_path", None)
        raise RuntimeError(
            f"[bundle_finalizer] robot_sku={robot_sku!r} has no on-disk "
            f"MJCF (canonical raw={raw_hint!r}, resolved="
            f"{str(resolved_mjcf) if resolved_mjcf else None!r}); cannot "
            f"derive MuJoCo PD gains. Open the Robot Asset card and run "
            f"'Dump MJCF' before exporting the bundle. (ActuatorPDNode is "
            f"wired on the canvas — without an MJCF the canonical "
            f"(omega_n, zeta) contract cannot be honored.)"
        )
    mjcf_path = resolved_mjcf

    # Map IR roles → physical joint names via JointIRResolver, then call
    # the solver. The solver uses mj_fullM at the bundle's default qpos
    # (legs in trained pose), which is what the deploy stack will face.
    try:
        from application.physics.mujoco_gain_solver import solve as _mj_solve
        from application.physics.pd_param import PDParam
        from application.training.joint_ir import JointIRResolver
        from application.training.training_spec import RobotSpecRef
        from application.training.validation.sim2sim_calibration import (
            run_calibration as _run_calibration,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[bundle_finalizer] failed to import physics helpers: {exc}"
        ) from exc

    try:
        pd_param = PDParam.from_dict(pd_param_dict)
    except Exception as exc:
        raise RuntimeError(
            f"[bundle_finalizer] deploy_meta.pd_param is malformed: {exc}"
        ) from exc

    # Build a RobotSpecRef from the registry so JointIRResolver works.
    # MJCF coverage check: when MJCF declares FEWER IR roles than the
    # trained policy uses (e.g. G1 trained on 43-DOF USD but MJCF is
    # the 29-DOF stock variant), we CANNOT derive MuJoCo PD gains for
    # the missing roles. Per user-clarified semantics, this is a
    # deploy-target availability concern, not a bundle-export failure:
    # we ship the bundle with pd_param (source of truth, always
    # populated) + PhysX gains (always derivable for IL) so the
    # IsaacSim deploy target works; mujoco_pd_gains / mujoco_pd_damping
    # are omitted and MuJoCo runtime refuses to load this bundle. The
    # raise pattern previously here blocked bundle export entirely,
    # which the user pushed back on as overly restrictive — the
    # IsaacSim-only bundle is still a valid artifact.
    spec_ref = RobotSpecRef.from_registry(rs, active_format="MJCF")
    resolver = JointIRResolver(spec_ref, active_format="MJCF")
    physical_names: List[str] = []
    missing_in_mjcf: List[str] = []
    for ir_role in joint_names_ir:
        try:
            physical_names.append(resolver.to_physical(ir_role))
        except KeyError:
            missing_in_mjcf.append(str(ir_role))

    if missing_in_mjcf:
        log_warning(
            f"[bundle_finalizer] MJCF table for robot_sku={robot_sku!r} "
            f"does not declare {len(missing_in_mjcf)} of the "
            f"{len(joint_names_ir)} IR roles the trained policy uses: "
            f"{missing_in_mjcf}. Skipping MuJoCo PD gain derivation — "
            f"the bundle will ship with pd_param + PhysX gains only "
            f"(IsaacSim / cloud deploy targets remain available; MuJoCo "
            f"deploy target will be unavailable). If MuJoCo deploy is "
            f"needed, repoint assets.MJCF to a variant that covers the "
            f"trained joint set (for Unitree G1: "
            f"``menagerie/unitree_g1/scene_with_hands.xml``) and re-export."
        )
        return {
            "pd_param": pd_param.to_dict(),
            "mujoco_pd_gains": None,
            "mujoco_pd_damping": None,
            "_mujoco_unsupported_reason": (
                f"MJCF missing {len(missing_in_mjcf)} IR role(s) the "
                f"trained policy uses: {missing_in_mjcf}"
            ),
        }

    # Nominal stance for mj_fullM. MUST be the byte-identical stance the env
    # compiler used when it mass-weighted the PhysX gains, else the two
    # m_eff diverge and the bug is only reshaped. The compiler stamps the
    # exact qpos_ref + its sha256 into deploy_meta.unitport_pd_param
    # .m_eff_source; we PIN to that array (not just nominal_qpos=None) so a
    # later MJCF keyframe edit cannot silently desync the two derivations
    # (CLAUDE.md §10).
    m_eff_src = pd_block.get("m_eff_source") or {}
    stored_qpos_ref = m_eff_src.get("qpos_ref")
    stored_qpos_sha = m_eff_src.get("qpos_sha256")
    stored_m_eff = m_eff_src.get("m_eff")
    # Pass the stored qpos as a plain list — the solver coerces it via
    # np.asarray and validates shape against model.nq.
    nominal_qpos = list(stored_qpos_ref) if stored_qpos_ref else None

    # Resolve the SAME USD→MJCF inertia overlay the env compiler used, so the
    # finalizer derives m_eff on the identical (USD-corrected) link inertia —
    # the m_eff parity guard below (1e-6) then enforces that they match. Absent
    # overlay = None = raw MJCF (unchanged). CLAUDE.md §10/§11.
    inertia_overlay = None
    try:
        from application.service.robot_assets.runtime import (
            read_mjcf_inertia_overlay,
        )
        inertia_overlay = read_mjcf_inertia_overlay(robot_sku) if robot_sku else None
    except Exception as exc:
        log_warning(
            f"[bundle_finalizer] inertia overlay read failed for "
            f"sku={robot_sku!r} ({exc}); deriving m_eff on raw MJCF."
        )
        inertia_overlay = None

    gains = _mj_solve(
        mjcf_path=Path(mjcf_path),
        joint_order_physical=physical_names,
        joint_ir_roles=list(joint_names_ir),
        nominal_qpos=nominal_qpos,
        pd_param=pd_param,
        inertia_overlay=inertia_overlay,
    )

    # Cross-process parity guard: the finalizer's per-joint m_eff MUST equal
    # the compiler's stored m_eff. This is the byte-identical-stance assert
    # the two solvers cannot share in memory (separate processes). A
    # mismatch means the MJCF / its inertia changed between training and
    # export — the PhysX gains baked into the policy no longer correspond to
    # what MuJoCo will apply, so fail loud (§8) rather than ship a desynced
    # bundle.
    finalizer_m_eff = [
        float(e["m_eff"]) for e in gains.derivation.get("per_joint", [])
    ]
    if stored_m_eff is not None:
        if len(stored_m_eff) != len(finalizer_m_eff):
            raise RuntimeError(
                f"[bundle_finalizer] m_eff length mismatch vs compiler "
                f"(stored {len(stored_m_eff)} != finalizer "
                f"{len(finalizer_m_eff)}) — joint set drifted; re-train."
            )
        rels = [
            abs(f - s) / max(abs(s), 1e-9)
            for f, s in zip(finalizer_m_eff, stored_m_eff)
        ]
        worst = max(rels) if rels else 0.0
        if worst > 1e-6:
            raise RuntimeError(
                f"[bundle_finalizer] effective-inertia desync between train "
                f"compile and bundle export: max relative m_eff diff "
                f"{worst:.3e} (>1e-6). The robot's MJCF or its <inertial> "
                f"tags changed since training, so the PhysX gains baked into "
                f"the policy no longer match what MuJoCo would apply. Re-dump "
                f"the MJCF and re-train, or export from the matching commit."
            )

    log_info(
        f"[bundle_finalizer] MuJoCo PD gains derived (mj_fullM @ "
        f"{gains.derivation.get('nominal_qpos_source')}): "
        f"family={pd_block.get('primary_family')} "
        f"kp_range=[{min(gains.kp):.2f}, {max(gains.kp):.2f}] "
        f"kd_range=[{min(gains.kd):.2f}, {max(gains.kd):.2f}] "
        f"(m_eff parity vs compiler OK)"
    )

    # ----- Align the compiler's PhysX gains to the FINALIZER's joint order.
    # The compiler emitted physx_gains in its own resolver order; the MuJoCo
    # `gains` here are in joint_names_ir order. Comparing them element-wise
    # would silently mis-pair joints — exactly the ordering-bug class that
    # caused the original sim2sim regression. Re-key by IR role and rebuild
    # parallel to joint_names_ir so every downstream array lines up by NAME.
    physx_gains = pd_block.get("physx_gains") or {}
    _px_roles = list(physx_gains.get("joint_ir_roles") or [])
    _px_kp_by_role = dict(zip(_px_roles, physx_gains.get("kp") or []))
    _px_kd_by_role = dict(zip(_px_roles, physx_gains.get("kd") or []))
    try:
        physx_kp_aligned = [float(_px_kp_by_role[r]) for r in joint_names_ir]
        physx_kd_aligned = [float(_px_kd_by_role[r]) for r in joint_names_ir]
    except KeyError as exc:
        raise RuntimeError(
            f"[bundle_finalizer] PhysX gains in deploy_meta do not cover IR "
            f"role {exc!s} that the bundle ships. The compiler and finalizer "
            f"disagree on the trained joint set — re-train."
        ) from exc

    # ----- Armature (CLAUDE.md §10), aligned to joint_names_ir by NAME.
    #   * MuJoCo side: read from the finalizer's OWN solve derivation (already in
    #     joint_names_ir order) — the MJCF-authoritative reflected-rotor inertia
    #     folded into m_eff.
    #   * PhysX side: what the compiler actually EMITTED to
    #     ImplicitActuatorCfg.armature, re-keyed by IR role (same alignment as
    #     physx_gains). A bundle trained before this fix has no
    #     ``physx_armature_emitted`` in the stash → PhysX got the USD default
    #     (0) → physx_armature defaults to 0.0 here, which the realized-inertia
    #     gate below will flag as a divergence (that is the intended catch).
    _deriv_pj = {
        e.get("physical_name"): e
        for e in gains.derivation.get("per_joint", [])
    }
    armature_mjcf_aligned = [
        float((_deriv_pj.get(pn) or {}).get("armature", 0.0))
        for pn in physical_names
    ]
    _px_arm_emitted = list(m_eff_src.get("physx_armature_emitted") or [])
    _px_arm_by_role = dict(zip(_px_roles, _px_arm_emitted)) if _px_arm_emitted else {}
    physx_armature_aligned = [
        float(_px_arm_by_role.get(r, 0.0)) for r in joint_names_ir
    ]

    # ----- Export-time STRICT calibration gate (CLAUDE.md §10). Runs BEFORE
    # any bundle bytes are written (the caller invokes this in step F, ahead
    # of BundleExporter.export_from_artifacts), so a fail aborts cleanly with
    # no partial bundle on disk. This is the strict sibling (1e-3 cross-engine
    # torque + (ωn,ζ) round-trip on hardcoded canary joints) of the looser
    # load-time guard in DeployContract.from_dict (1%). The thresholds are
    # intentionally NOT unified — see run_calibration's comment.
    # ----- Stage 0 audit inputs (Layer-1 realized==designed). The compiler
    # stashed the DESIGNED friction (scene_contract) + foot-geometry hash. The
    # gate re-derives the realized foot hash from the SAME MJCF to confirm
    # compiler↔finalizer agreement; foot_roles / ir_to_body_name (registry
    # bodies map, by IR role) let it recompute.
    _scene_contract = pd_block.get("scene_contract")
    _stashed_foot_geometry = pd_block.get("foot_geometry")
    _foot_roles: tuple = ()
    _ir_to_body: Dict[str, str] = {}
    try:
        from application.training.validation.mjcf_base_calibration import (
            resolve_foot_roles_for_sku as _resolve_foot_roles_for_sku,
        )
        _robot_entry = _registers_robots.get_robot(robot_sku) or {}
        _bodies_mjcf = (
            _robot_entry.get("bodies_per_format", {}).get("MJCF", {}) or {}
        )
        for _bspec in _bodies_mjcf.values():
            if isinstance(_bspec, dict):
                _nm = str(_bspec.get("name", ""))
                _ir = str(_bspec.get("ir_role", ""))
                if _nm and _ir:
                    _ir_to_body[_ir] = _nm
        # Same per-side feet fallback as the producer (foot_L → ankle_L) so the
        # gate re-hash resolves H1's ankle feet identically (compiler↔finalizer
        # foot-role parity).
        _foot_roles = _resolve_foot_roles_for_sku(
            str(pd_param.family), _ir_to_body.keys())
    except Exception as exc:  # noqa: BLE001
        log_warning(
            f"[bundle_finalizer] foot-geometry audit inputs unavailable "
            f"({exc}); foot-geom row will be informational only."
        )

    skip_calibration = bool(pd_block.get("skip_calibration", False))
    calibration_blocking = bool(pd_block.get("calibration_blocking", True))
    calibration_verdict = "skipped" if skip_calibration else None
    report = None
    if skip_calibration:
        log_warning(
            "[bundle_finalizer] skip_calibration=True on the ActuatorPDNode "
            "— BYPASSING the export-time PD calibration gate. The skip is "
            "recorded in pd_derivation.calibration for provenance (CLAUDE.md "
            "§10). Gains are shipped UNVERIFIED."
        )
    else:
        import numpy as _np
        # Step 5: hand the calibration gate the remotized lookup tables (role →
        # inline table dict, re-keyed from the physical-name-keyed provenance
        # the compiler stashed) so it can run the multi-point angle-swept
        # cross-engine torque check in addition to the (ωn,ζ) round-trip. None
        # for non-remotized robots (the gate then runs unchanged).
        _remotized_meta = pd_block.get("remotized_joints")
        _remotized_tables = None
        if isinstance(_remotized_meta, dict) and isinstance(
            _remotized_meta.get("joints"), dict
        ):
            _remotized_tables = {}
            for _entry in _remotized_meta["joints"].values():
                _r = (_entry or {}).get("ir_role")
                _t = (_entry or {}).get("table")
                if _r and _t is not None:
                    _remotized_tables[_r] = _t
        report = _run_calibration(
            mjcf_path=Path(mjcf_path),
            pd_param=pd_param,
            joint_order_physical=list(physical_names),
            joint_ir_roles=list(joint_names_ir),
            kp_array=_np.asarray(gains.kp, dtype=float),
            kd_array=_np.asarray(gains.kd, dtype=float),
            family=str(pd_param.family),
            nominal_qpos=nominal_qpos,
            physx_kp_array=_np.asarray(physx_kp_aligned, dtype=float),
            physx_kd_array=_np.asarray(physx_kd_aligned, dtype=float),
            remotized_tables=_remotized_tables,
            # Realized joint-inertia probe (CLAUDE.md §10/§11, Task B): the gate
            # independently re-derives MuJoCo's realized m_eff on the OVERLAID
            # model (so a regression that reverts m_eff to raw link inertia is
            # caught), then decomposes the diagonal with these armatures to check
            # PhysX realizes the SAME inertia. physx_armature=0 (old/unfixed)
            # makes the distal joints fail loud.
            inertia_overlay=inertia_overlay,
            armature_mjcf=_np.asarray(armature_mjcf_aligned, dtype=float),
            physx_armature=_np.asarray(physx_armature_aligned, dtype=float),
            # Stage 0 audit-matrix engine rows: friction single source +
            # foot-geometry realized recompute (gate re-hashes its MJCF).
            scene_contract=_scene_contract,
            foot_geometry=_stashed_foot_geometry,
            foot_roles=_foot_roles,
            ir_to_body_name=_ir_to_body,
        )
        calibration_verdict = report.verdict
        if report.verdict == "fail":
            if calibration_blocking:
                # Name the ACTUAL failing check(s) — the gate folds the PD
                # numeric round-trip + cross-engine torque AND the engine-level
                # realized==designed rows (friction / foot_geom / joint_order)
                # into one verdict, so a generic "PD torque mismatch" message
                # mis-attributes a foot-geometry / ordering break (§8 — the
                # error must report the real cause, not a wrong diagnosis).
                _PD_DIMS = {"omega_zeta_roundtrip", "cross_engine_torque"}
                _fail_cells = []
                _mtx = getattr(report, "matrix", None)
                if _mtx is not None:
                    try:
                        _fail_cells = _mtx.cells_at_or_above("fail")
                    except Exception:  # noqa: BLE001
                        _fail_cells = []
                _fail_dims = sorted({c.dimension for c in _fail_cells})
                _is_pd = any(
                    (d in _PD_DIMS or d.startswith("inertia")) for d in _fail_dims)
                if _fail_dims:
                    _summary = "; ".join(
                        f"{c.dimension}: {' '.join(c.notes)}" for c in _fail_cells)
                else:
                    _summary = "see the report below"
                _pd_hint = (
                    " The PhysX training gains and the MuJoCo deploy gains do "
                    "not produce matching torque, or the (omega_n, zeta) "
                    "round-trip is off (Spot-knee-65x class of bug; the policy "
                    "would not transfer)." if _is_pd else ""
                )
                raise RuntimeError(
                    "[bundle_finalizer] Bundle export ABORTED by the "
                    "realized==designed calibration gate (verdict=fail; no "
                    f"bundle written). Failing check(s): {_summary}.{_pd_hint}"
                    "\n\n" + report.markdown
                )
            log_warning(
                "[bundle_finalizer] PD calibration verdict=fail but "
                "calibration_blocking=False — shipping anyway (NOT "
                "recommended). Report:\n" + report.markdown
            )
        elif report.verdict == "warn":
            log_warning(
                "[bundle_finalizer] PD calibration verdict=warn:\n"
                + report.markdown
            )
        else:
            log_info("[bundle_finalizer] PD calibration verdict=pass.")

    # ----- pd_derivation provenance (CLAUDE.md §10): a single block someone
    # can read in 30s to retrace a gain. m_eff source (mjcf + qpos hash),
    # (ωn, ζ) inputs per group, and final kp/kd outputs for BOTH engines.
    #
    # ARRAY ORDERING — read before trusting any index here. Every per-joint
    # array in this block (m_eff, inputs.joint_*, outputs.*) is in IR-role
    # order: position i is the joint named by ``inputs.joint_ir_roles[i]``
    # (and ``inputs.joint_physical_names[i]``, which is parallel). This is
    # the pre-permutation order. It is DELIBERATELY NOT reordered by the
    # top-level ``joint_permutation`` that re-sorts the deploy_contract's
    # stiffness/damping/mujoco_pd_gains into Isaac-Lab actuator order. So:
    # to line a pd_derivation entry up against the deploy_contract arrays,
    # match by joint NAME, never by raw index. The ``array_ordering`` field
    # below records this contract explicitly — the last sim2sim bug was
    # exactly "two arrays in different orders, nobody remembered the
    # convention"; don't let it recur.
    pd_derivation = {
        "method": "full_mass_matrix",
        "formula_kp": "m_eff * omega_n ** 2",
        "formula_kd": "2 * zeta * sqrt(kp * m_eff)",
        "array_ordering": "ir_roles",  # all arrays parallel to inputs.joint_ir_roles; NOT joint_permutation order
        "calibration": {
            "verdict": calibration_verdict,
            "blocking": calibration_blocking,
            "skipped": skip_calibration,
        },
        "m_eff_source": {
            "mjcf_path": str(mjcf_path),
            "nominal_qpos_source": gains.derivation.get("nominal_qpos_source"),
            "qpos_sha256": stored_qpos_sha,
            "m_eff": finalizer_m_eff,
            # Armature decomposition of the realized joint-inertia diagonal
            # (CLAUDE.md §10), parallel to inputs.joint_ir_roles (match by name,
            # never index). ``armature_mjcf`` is folded into ``m_eff``;
            # ``physx_armature_emitted`` is what training actually applied via
            # ImplicitActuatorCfg.armature. They are EQUAL for a correctly
            # exported bundle (both = the MJCF reflected-rotor inertia); a bundle
            # missing ``physx_armature_emitted`` was trained before this fix
            # (PhysX armature 0) — load-time WARN directs re-export (§8c).
            # ``overlay_applied`` records whether m_eff was derived on the
            # USD-corrected link inertia (Q6: derive == realize).
            "armature_mjcf": list(armature_mjcf_aligned),
            "physx_armature_emitted": list(physx_armature_aligned),
            "overlay_applied": bool(gains.derivation.get("overlay_applied", False)),
        },
        "inputs": {
            "joint_ir_roles": list(joint_names_ir),
            "joint_physical_names": list(physical_names),
            "pd_param": pd_param.to_dict(),
        },
        "outputs": {
            # physx_* re-keyed to joint_names_ir order (see alignment above)
            # so they're parallel to mujoco_* and to inputs.joint_ir_roles.
            "physx_kp": list(physx_kp_aligned),
            "physx_kd": list(physx_kd_aligned),
            "mujoco_kp": [float(v) for v in gains.kp],
            "mujoco_kd": [float(v) for v in gains.kd],
        },
    }

    # Remotized-actuator provenance (Step 3.2): pass through the section the
    # env_cfg compiler stashed (per remotized joint: ir_role, group, table
    # sha256, peak/min torque, and the inline table payload). The embedded
    # table is the SINGLE copy Step 4 reuses for the bundle's
    # ``mujoco_torque_lookups`` — keyed by joint NAME, tagged
    # ``array_ordering: ir_roles`` (the table's own monotonic-axis ordering
    # lives inside each table payload, a separate convention). Absent for
    # non-remotized robots.
    _remotized = pd_block.get("remotized_joints")
    if _remotized is not None:
        pd_derivation["remotized_joints"] = _remotized

    # ----- deploy_contract.sim2sim_audit (Stage 0): the two-engine realized
    # CONSISTENCY verdicts only (values + derivation stay in pd_derivation;
    # friction + foot-geometry reference values live here — no overlap). Built
    # even when calibration was skipped, so deploy still gets the designed
    # friction (PolicyRunner applies it) + foot hash (PolicyRunner self-checks).
    sim2sim_audit = _build_sim2sim_audit_block(
        matrix=(report.matrix if report is not None else None),
        scene_contract=_scene_contract,
        foot_geometry=_stashed_foot_geometry,
        calibration_skipped=skip_calibration,
    )

    return {
        "pd_param": pd_param.to_dict(),
        "mujoco_pd_gains": [float(v) for v in gains.kp],
        "mujoco_pd_damping": [float(v) for v in gains.kd],
        "pd_derivation": pd_derivation,         # persisted into deploy_contract
        "sim2sim_audit": sim2sim_audit,         # persisted into deploy_contract
        "_solver_derivation": gains.derivation,  # dropped before manifest write
    }


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
    # 缺口① — capture the export metadata; its ``recurrent`` block (inferred
    # from the actual exported tensors) is the authoritative source for the
    # deploy_contract recurrent contract injected below.
    export_meta: Optional[Dict[str, Any]] = None
    if policy.needs_conversion:
        from application.training.isaac_lab.onnx_export import (
            export_rsl_rl_actor_to_onnx,
        )
        # Convert into a temp file alongside the source then read bytes.
        # Using a temp file (not BytesIO) because export_rsl_rl_actor_to_onnx
        # mirrors DEMO's API which writes via torch.onnx.export → Path.
        tmp_onnx = exported_dir / f".finalize_{run_id}.onnx.tmp"
        try:
            export_meta = export_rsl_rl_actor_to_onnx(
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
        # — never paper-over by reading display strings. CLAUDE.md §1.7.
        robot_sku = str(getattr(spec.robot, "sku", "") or "")
        if not robot_sku:
            raise ValueError(
                "[bundle_finalizer] spec.robot.sku is empty — lowering "
                "produced an unregistered robot. Fix the canvas Robot Node "
                "binding upstream; bundle finalize refuses to ship a "
                "bundle without a SKU (CLAUDE.md §1.7)."
            )
        # IR roles, NOT physical names — the substrate-side translation
        # happens deploy-time via ir_roles_to_physical_names(roles, sku).
        joint_names = list(getattr(spec.robot, "joint_ir_roles", []) or [])
        if not joint_names:
            # Fallback: legacy spec without joint_ir_roles populated → use
            # joint_order so the manifest is at least self-consistent on
            # length. Should not happen in Phase 5+ flows.
            joint_names = list(getattr(spec.robot, "joint_order", []) or [])

    # sim2sim joint-order contract. The exported ONNX emits its action vector
    # in IsaacLab's LIVE PhysX articulation order: ``JointPositionAction``
    # resolves ``joint_names=[".*"]`` against the running articulation and
    # preserves that order, so action slot ``i`` == articulation DOF ``i``. The
    # bundle's per-joint arrays MUST be ordered by THAT order — captured at
    # train time into ``articulation_joint_order.json`` by the launcher.
    #
    # Ordering by a separately-dumped USD prim order (``joints_per_format
    # ["USD"]`` / ``stage.Traverse()`` authoring order) was the bug that
    # silently scrambled Go2's thigh/calf slots: prim authoring order
    # [hip×4,(thigh,calf)/leg] ≠ PhysX BFS [hip×4,thigh×4,calf×4]. The removed
    # ``isaac_lab_joint_order`` field "drifted from the dumped USD truth" — but
    # the live articulation order, not the dump, is ground truth.
    #
    # With the captured sidecar we build the permutation from it and LOUD-WARN
    # if it disagrees with the registry USD order (stale dump). Without it
    # (pre-capture runs / capture failed) we fall back to the legacy
    # active_format behaviour and WARN that the order is UNVERIFIED.
    active_format = ""
    if spec is not None and getattr(spec, "robot", None) is not None:
        active_format = str(getattr(spec.robot, "active_format", "") or "").upper()

    from application.training.isaac_lab.joint_order_capture import (
        load_joint_order_sidecar,
        resolve_bundle_joint_permutation,
    )
    _captured_sidecar = load_joint_order_sidecar(exported_dir)
    captured_ir_order: Optional[List[str]] = None
    joint_order_source: str = ""

    joint_permutation: Optional[List[int]] = None
    if not joint_names or not robot_sku:
        # No joint topology declared (e.g. minimal smoke spec) — nothing
        # to permute. Allowed to skip silently.
        pass
    elif _captured_sidecar is not None:
        # --- Captured LIVE articulation order = the policy's true action order.
        captured_names = [
            str(n) for n in (_captured_sidecar.get("joint_names") or [])
        ]
        joint_order_source = str(_captured_sidecar.get("source") or "articulation")
        # Captured names are USD-prim joint names → map to IR roles. Prefer the
        # USD table; fall back to MJCF if USD doesn't cover every captured name.
        name_to_role: Dict[str, str] = {}
        for _fmt in ("USD", "MJCF"):
            try:
                _m = spec.robot.joints_role_map_for(_fmt)
            except Exception:
                _m = {}
            if _m and all(n in _m for n in captured_names):
                name_to_role = _m
                break
            if _m and not name_to_role:
                name_to_role = _m
        try:
            captured_ir_order, candidate = resolve_bundle_joint_permutation(
                joint_names, captured_names, name_to_role
            )
        except ValueError as exc:
            raise RuntimeError(
                f"[bundle_finalizer] cannot reconcile the live articulation "
                f"joint order (articulation_joint_order.json) with the registry "
                f"for SKU {robot_sku!r}: {exc}. Re-Dump the asset so the joint "
                f"tables cover the trained joint set (CLAUDE.md §1.8)."
            ) from exc
        # Cross-check vs the order the bundle WOULD have shipped without the
        # capture (the legacy registry/USD order). LOUD WARN on mismatch — the
        # exact diagnostic for the X≠live joint-scramble bug.
        _legacy_order: List[str] = (
            list(joint_names) if active_format == "USD" else []
        )
        if not _legacy_order:
            try:
                _legacy_order = list(spec.robot.joint_ir_roles_for("USD"))
            except Exception:
                _legacy_order = []
        if _legacy_order and list(_legacy_order) != list(captured_ir_order):
            try:
                _li = {r: i for i, r in enumerate(_legacy_order)}
                _perm_dbg = [_li.get(r) for r in captured_ir_order]
            except Exception:
                _perm_dbg = None
            log_debug(
                f"[bundle_finalizer] JOINT-ORDER MISMATCH for SKU {robot_sku!r}: "
                f"the LIVE IsaacLab articulation order (the policy's true action "
                f"order, source={joint_order_source!r}) differs from the registry "
                f"USD order. The bundle is ordered (correctly) by the LIVE order — "
                f"this WARN is INFORMATIONAL, no action needed. The registry USD "
                f"order is the USD PRIM-AUTHORING order from Dump-USD "
                f"(stage.Traverse), which is NOT the PhysX articulation DOF order; "
                f"re-Dumping will NOT change it (the dump reads the USD file, not a "
                f"live articulation), so the train-time capture is the only "
                f"authoritative source.\n  live (used) : {list(captured_ir_order)}\n"
                f"  registry USD: {list(_legacy_order)}\n"
                f"  live->registry permutation: {_perm_dbg}"
            )
        joint_permutation = candidate
    elif active_format == "USD":
        # No captured sidecar → UNVERIFIED. joint_ir_roles is ASSUMED to be in
        # USD-articulation order; ship as-is but WARN loudly that the order was
        # not verified against the trained policy.
        log_warning(
            f"[bundle_finalizer] no articulation_joint_order.json sidecar under "
            f"{exported_dir} for SKU {robot_sku!r}; shipping joint_ir_roles in "
            f"the registry order UNVERIFIED against the trained policy's live "
            f"articulation order (the Go2 sim2sim-twitch class of bug). Re-train "
            f"with the capture-enabled launcher to ship a verified bundle."
        )
    else:
        # No captured sidecar + non-USD active format: derive USD-articulation
        # order from ``joints_per_format["USD"]`` (legacy path), UNVERIFIED.
        log_warning(
            f"[bundle_finalizer] no articulation_joint_order.json sidecar under "
            f"{exported_dir} for SKU {robot_sku!r}; deriving the joint order from "
            f"the registry USD table (UNVERIFIED against the trained policy's "
            f"live articulation order). Re-train with the capture-enabled "
            f"launcher to ship a verified bundle."
        )
        usd_order: List[str] = []
        try:
            usd_order = list(spec.robot.joint_ir_roles_for("USD"))
        except Exception:
            usd_order = []
        if not usd_order:
            raise RuntimeError(
                f"[bundle_finalizer] robot_sku={robot_sku!r} was compiled "
                f"against active_format={active_format!r} (not USD) and "
                f"its registry entry has no ``joints_per_format[\"USD\"]`` "
                f"populated. The bundle would ship joint_sdk_names in "
                f"{active_format!r} order, which does not match the "
                f"USD-articulation order Isaac Lab reports at deploy time "
                f"— sim2sim would then feed obs/action through the wrong "
                f"joint slots and the policy would twitch (CLAUDE.md §1.8 "
                f"forbids shipping known-broken artifacts). Open the "
                f"Robot Asset card for this SKU and run \"Dump USD\" to "
                f"populate the table, then re-train. The USD-table "
                f"insertion order becomes the bundle's joint contract."
            )
        canonical_index = {role: i for i, role in enumerate(joint_names)}
        try:
            candidate = [canonical_index[r] for r in usd_order]
        except KeyError as exc:
            raise RuntimeError(
                f"[bundle_finalizer] joints_per_format[\"USD\"] references "
                f"IR role {exc} that is not in spec.robot.joint_ir_roles "
                f"for SKU {robot_sku!r}. The USD and {active_format!r} "
                f"joint tables disagree on which IR roles exist — re-Dump "
                f"both formats from the Robot Asset card so they cover "
                f"the same joint set."
            ) from exc
        if len(candidate) != len(joint_names):
            raise RuntimeError(
                f"[bundle_finalizer] joints_per_format[\"USD\"] length "
                f"({len(candidate)}) ≠ joint_ir_roles "
                f"({len(joint_names)}) for SKU {robot_sku!r}. Registry "
                f"data is inconsistent — both formats must cover the same "
                f"joint set."
            )
        joint_permutation = candidate

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

    # ``robot_sku`` is NOT stored on the contract anymore — it lives only
    # on manifest.robot.sku (single source of truth). Strip any leftover
    # field that might have come from a legacy parser path so the bundle
    # written to disk is clean.
    deploy_contract_dict: Optional[Dict[str, Any]] = None
    if sm.deploy_contract is not None:
        deploy_contract_dict = dict(sm.deploy_contract)
        deploy_contract_dict.pop("robot_sku", None)

        # Deploy-side command contract — single shared emitter with the SB3
        # exporter (§11 symmetric pair): the full v1 CommandContract from the
        # Training Motion node, so a normalised [-1,1] controller stick maps
        # to the real envelope the policy trained on. Fail-loud (§8): an
        # unresolved condition ABORTS finalize; {} only when the spec carries
        # no canvas dict or the canvas has no training_motion node
        # (presence-gated additive block). Family is spec.robot.families[0]
        # — the gait-registry dispatch key — never a silent "quadruped".
        from application.training.command_schema import deploy_command_block
        _canvas_dict = (getattr(spec, "meta", None) or {}).get("__canvas_dict__")
        _cmd_families = list(
            getattr(getattr(spec, "robot", None), "families", []) or []
        )
        _cmd_block = deploy_command_block(
            _canvas_dict, family=_cmd_families[0] if _cmd_families else ""
        )
        if _cmd_block:
            deploy_contract_dict["commands"] = _cmd_block

        # 缺口① — recurrent policy contract. Source of truth = the export
        # metadata (inferred from the actual exported ONNX tensors); fall back
        # to the canvas spec's policy_net when the bundle shipped a pre-built
        # ONNX (no in-process conversion). Absent ⇒ feed-forward MLP (the
        # block is simply not written — R6, schema_version stays 2). This is
        # what makes a recurrent bundle self-contained (§9): the deploy stack
        # reads ``recurrent`` to thread the hidden state without any
        # local-only reach-back.
        _rec_block = None
        if isinstance(export_meta, dict) and export_meta.get("recurrent"):
            _rec_block = dict(export_meta["recurrent"])
        else:
            _pn = getattr(getattr(spec, "algorithm", None), "policy_net", None)
            _rt = str(getattr(_pn, "rnn_type", "none") or "none").strip().lower()
            if _rt in ("gru", "lstm"):
                _rec_block = {
                    "rnn_type": _rt,
                    "hidden_size": int(getattr(_pn, "rnn_hidden_size", 256)),
                    "num_layers": int(getattr(_pn, "rnn_num_layers", 1)),
                }
        if _rec_block is not None:
            _rec_block.setdefault("reset", "episode")
            deploy_contract_dict["recurrent"] = _rec_block

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

        # MJCF base spawn-Z offset overlay (USD↔MJCF anchor compensation).
        # Embed the per-SKU offset into the bundle so a downstream
        # MuJoCo runtime on a different machine doesn't need the user's
        # robot_assets/state.json — CLAUDE.md §1.9 portable artifacts.
        if robot_sku:
            try:
                from application.service.robot_assets.service import (
                    get_robot_asset_service,
                )
                overlay = get_robot_asset_service().get_mjcf_base_offset(robot_sku)
                if isinstance(overlay, dict) and overlay:
                    deploy_contract_dict["mjcf_base_height_offset"] = overlay
            except Exception as exc:  # noqa: BLE001
                # WHY KEPT (§1.8 (c)): finalize must not fail because
                # of an overlay read glitch; uncalibrated state surfaces
                # via the runtime reader on the deploy machine.
                log_warning(
                    f"[bundle_finalizer] could not attach "
                    f"mjcf_base_height_offset to deploy_contract: {exc}"
                )

        # Stage 2 (收窄版) cross-engine DR coverage provenance. This is the
        # IsaacLab finalizer → the bundle is PhysX-trained → MuJoCo-deployed =
        # the only cross-engine case. Record that the measured friction residual
        # ([0.56,1.02], block-slide 2026-06-04) is COVERED by the existing
        # ground-friction DR (no new channel built); the load-time check in
        # DeployContract.from_dict fail-louds when friction DR was disabled.
        _dr = getattr(spec, "domain_rand", None)
        _il = getattr(_dr, "il", None) if _dr is not None else None
        if _il is not None:
            from application.training.validation.sim2sim_measurement.range_table import (
                build_dr_coverage_provenance,
            )
            _fr = getattr(_il, "static_friction_range", (0.5, 1.25))
            _measured = None
            if robot_sku:
                try:
                    from registers import sim2sim_ranges
                    if sim2sim_ranges.has(robot_sku):
                        _tbl = sim2sim_ranges.get_ranges(robot_sku)
                        for _d in (_tbl.get("dimensions") or []):
                            if _d.get("dimension") == "friction_cone" and \
                                    _d.get("factor_low") is not None:
                                _measured = [_d["factor_low"], _d["factor_high"]]
                                break
                except Exception as exc:  # noqa: BLE001
                    # WHY KEPT (§8(c)): a measured-table read glitch must not
                    # fail finalize; provenance falls back to the campaign
                    # reference constants (recorded measured_for_this_sku=False).
                    log_warning(
                        f"[bundle_finalizer] sim2sim_ranges lookup failed for "
                        f"{robot_sku}: {exc}; using campaign reference residual"
                    )
            deploy_contract_dict["sim2sim_dr_provenance"] = (
                build_dr_coverage_provenance(
                    training_backend="isaac_lab",
                    friction_dr_enabled=bool(getattr(_il, "friction_enabled", True)),
                    friction_dr_range=_fr,
                    sku=robot_sku or "",
                    measured_factor=_measured,
                )
            )

        # Stage F: derive MuJoCo PD gains from deploy_meta.json's
        # unitport_pd_param block (written by env_cfg_compiler when an
        # ActuatorPDNode is wired). Done BEFORE joint_permutation so
        # the new arrays get reordered along with the existing ones.
        # Returns None when no ActuatorPDNode is in the canvas; the
        # bundle is then v2 but with pd_param=None (legacy scalar PD
        # for the MuJoCo runtime, same as v1).
        #
        # Coverage-gap handling: when MJCF doesn't declare every IR role
        # the trained policy uses (e.g. trained against 43-DOF USD G1
        # but MJCF asset is 29-DOF stock variant), the derivation
        # returns a partial payload with mujoco_pd_gains / mujoco_pd_damping
        # = None and _mujoco_unsupported_reason set. We still write
        # pd_param (source of truth, always carried) and mark the bundle
        # as MuJoCo-deploy-unsupported via a sentinel field — MuJoCo
        # runtime loader checks this and refuses to load; IsaacSim /
        # cloud deploy targets remain unaffected.
        pre_perm_joint_names = list(deploy_contract_dict.get("joint_sdk_names") or joint_names)
        pre_perm_default_pos = list(deploy_contract_dict.get("default_joint_pos") or [])
        if robot_sku and pre_perm_joint_names:
            try:
                pd_payload = _derive_mujoco_pd_gains_for_bundle(
                    deploy_meta_path=deploy_meta,
                    robot_sku=robot_sku,
                    joint_names_ir=pre_perm_joint_names,
                    default_joint_pos=pre_perm_default_pos,
                )
            except Exception as exc:
                # Re-raise: this path is for unexpected solver/import
                # failures (genuine bugs). Per-joint MJCF coverage gaps
                # are handled inside the function and return a partial
                # payload, not raised.
                log_warning(
                    f"[bundle_finalizer] MuJoCo PD derivation failed: {exc}"
                )
                raise
            if pd_payload is not None:
                # Drop the diagnostic-only key before injection.
                pd_payload.pop("_solver_derivation", None)
                deploy_contract_dict["pd_param"] = pd_payload["pd_param"]
                deploy_contract_dict["mujoco_pd_gains"] = pd_payload["mujoco_pd_gains"]
                deploy_contract_dict["mujoco_pd_damping"] = pd_payload["mujoco_pd_damping"]
                # pd_derivation provenance (CLAUDE.md §10): retrace any gain
                # in 30s — m_eff source (mjcf + qpos hash), (ωn,ζ) inputs,
                # kp/kd outputs for both engines. Absent on coverage-gap
                # payloads (mujoco arrays None).
                if pd_payload.get("pd_derivation") is not None:
                    deploy_contract_dict["pd_derivation"] = pd_payload["pd_derivation"]
                # Stage 0 realized==designed audit (friction / foot-geometry /
                # joint-order / control-period + realized-consistency verdicts).
                if pd_payload.get("sim2sim_audit") is not None:
                    deploy_contract_dict["sim2sim_audit"] = pd_payload["sim2sim_audit"]
                    # Step 4.2: build the MuJoCo runtime lookup map. Re-key
                    # the provenance block from physical joint name → IR role
                    # (the name space joint_sdk_names / PDController use), and
                    # REUSE the same inline table payload — one copy, so it
                    # cannot drift from pd_derivation (World B lesson). Keyed
                    # by name, NOT index, so the joint_permutation below (which
                    # reorders the gain ARRAYS) leaves this dict untouched.
                    _rj = pd_payload["pd_derivation"].get("remotized_joints")
                    if isinstance(_rj, dict) and isinstance(_rj.get("joints"), dict):
                        _lookups = {}
                        for _phys, _entry in _rj["joints"].items():
                            _role = (_entry or {}).get("ir_role")
                            _tbl = (_entry or {}).get("table")
                            if _role and _tbl is not None:
                                _lookups[_role] = _tbl   # same dict object, no copy
                        if _lookups:
                            deploy_contract_dict["mujoco_torque_lookups"] = _lookups
                # MuJoCo-unsupported marker, when MJCF coverage gap forced
                # the derivation to return None for the engine-specific
                # arrays. Runtime loader (mj_actor / pd_controller) checks
                # this and refuses with the carried reason rather than
                # crashing on AttributeError / None arithmetic.
                unsupported = pd_payload.pop("_mujoco_unsupported_reason", None)
                if unsupported:
                    deploy_contract_dict["mujoco_deploy_unsupported"] = (
                        str(unsupported)
                    )

        # Bump the contract schema_version to v2 unconditionally — every
        # bundle exported by this pipeline carries the v2 shape, with
        # pd_param/mujoco_pd_gains populated only when an ActuatorPDNode
        # was wired. Legacy v1 readers will reject the version; this is
        # intentional (the v1 → v2 transition is irreversible per
        # mjcf-usd-pd plan Stage F).
        from application.service.runtime.policy.deploy_contract import (
            CURRENT_SCHEMA_VERSION as _PD_SCHEMA_VERSION,
        )
        deploy_contract_dict["schema_version"] = int(_PD_SCHEMA_VERSION)

        # Stamp the provenance of the joint order. ``articulation_joint_order``
        # is the captured LIVE PhysX articulation order (= the final bundle
        # order after the permutation below); the deploy stack reads it to
        # confirm/normalize against the policy's true action order, and its
        # ABSENCE on an IsaacLab bundle triggers the load-time re-export WARN.
        if captured_ir_order is not None:
            deploy_contract_dict["articulation_joint_order"] = list(captured_ir_order)
            if joint_order_source:
                deploy_contract_dict["joint_order_source"] = joint_order_source

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
                "mujoco_pd_gains",
                "mujoco_pd_damping",
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

    # B1 — policy contract snapshot. Injected HERE (manifest dims known; before
    # the contract is validated + written). recurrent_shape mirrors the
    # _rec_block set above (its source of truth). normalization presence is
    # probed from the checkpoint (the stats file is written post-export, so the
    # presence must be determined now). IsaacLab trains on PhysX but deploys to
    # MuJoCo — joint arrays are in USD order, obs in isaac_lab convention (same
    # constants build_manifest uses, line ~540/546). The export-strict gate
    # refuses an inconsistent snapshot before bytes are written (§10 two-gate).
    if deploy_contract_dict is not None:
        _il_norm_present = False
        if (
            spec is not None
            and getattr(spec, "export", None) is not None
            and bool(getattr(spec.export, "include_normalization", False))
            and policy.source_path.suffix.lower() == ".pt"
        ):
            _il_norm_present = _checkpoint_has_normalization(policy.source_path)
        _il_disc_dims = None
        if is_amp:
            _il_disc_dims = getattr(
                getattr(
                    getattr(getattr(getattr(spec, "algorithm", None), "il", None), "amp", None),
                    "disc",
                    None,
                ),
                "hidden_dims",
                None,
            )
            _il_disc_dims = list(_il_disc_dims) if _il_disc_dims else None
        _il_rec_block = deploy_contract_dict.get("recurrent")
        _il_policy_contract = build_policy_contract_block(
            spec,
            policy_input_dim=int(manifest_dict["observation_space"]["dim"]),
            policy_output_dim=int(manifest_dict["action_space"]["dim"]),
            recurrent_block=_il_rec_block,
            normalization_present=_il_norm_present,
            normalization_kind="rsl_rl_obs_norm" if _il_norm_present else "none",
            inference_convention="isaac_lab",
            joint_array_format="USD",
            discriminator_hidden_dims=_il_disc_dims,
        )
        assert_policy_contract_consistent(
            _il_policy_contract,
            manifest_obs_dim=int(manifest_dict["observation_space"]["dim"]),
            manifest_action_dim=int(manifest_dict["action_space"]["dim"]),
            recurrent_block=_il_rec_block,
            label="[bundle_finalizer] IsaacLab bundle",
        )
        deploy_contract_dict["policy_contract"] = _il_policy_contract

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

    # 5.9. Custom terrain — embed the heightfield self-contained in the
    # bundle (§9) so the policy is reviewable / re-deployable on the SAME
    # ground on any machine. The cross-engine consistency gate runs inside
    # embed_custom_terrain, BEFORE any bytes are written (§8 / §10
    # discipline): a terrain whose two engines disagree aborts the export.
    from application.training.terrain.bundle_io import embed_custom_terrain
    terrain_extra_files = embed_custom_terrain(spec, manifest_dict)

    out = BundleExporter.export_from_artifacts(
        name=bundle_name,
        version=version,
        manifest=manifest_dict,
        onnx_source=onnx_bytes,
        source_meta=source_meta,
        overwrite=overwrite,
        project=project,
        backend_id="isaac_lab",
        extra_files=terrain_extra_files,
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
