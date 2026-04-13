"""Isaac Lab v2 → v1 manifest upgrade shim.

This module is the **runtime upgrade path** for bundles whose on-disk
``manifest.yaml`` is in the Isaac Lab v2 layout (everything under a
``skill`` block). It rewrites the parsed manifest in-memory so the v1
runtime (``BundleLoader`` / ``ObsBuilder`` / ``ActionApplier``) can
consume it.

History
-------
Prior to the sim2sim centralization (2026-04-11, knowledge_base/
sim2sim_design.yaml S1), this file also contained the entire Isaac Lab
env.yaml parsing pipeline (~900 lines of obs/action/PD/joint-order
extraction). That pipeline has been moved to
:mod:`src.system.policy.sim2sim_compiler` where it is now the single
source of truth for Isaac Lab → DeployContract translation. This module
is a **thin shim** that:

  1. Keeps the ``upgrade_v2_to_v1`` entry point for runtime bundle
     loading (called from ``bundle_loader.BundleLoader``).
  2. Keeps the Go2-specific hardcoded joint order and default pose so
     legacy Go2 bundles without a working MJCF resolver still load.
  3. Re-exports the internal helpers with their old underscore-prefixed
     names so existing tests (``tests/unit/policy/test_il_deploy_contract_assembly.py``)
     continue to import them unchanged.

Future work
-----------
S3 of the sim2sim unification plan writes ``deploy_contract`` into the
manifest at **import time** (``checkpoint_registry.import_isaac_lab_bundle``)
so ``upgrade_v2_to_v1`` becomes a pure lazy-fallback for legacy bundles
that predate the import-time compile step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.system.policy.sim2sim_compiler import (
    assemble_deploy_contract_dict,
    expand_actuator_field_per_joint,
    expand_joint_pos_pattern_dict,
    extract_command_ranges,
    extract_observations_for_contract,
    infer_base_body_name,
    load_mjcf_joint_names,
    normalize_robot_brand_and_type,
    read_env_yaml,
    read_policy_onnx_obs_dim,
    recover_joint_order,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known-good Go2 tables (Isaac Lab convention)
# ---------------------------------------------------------------------------

# Isaac Lab groups Go2's 12 joints by joint *type* (all hips, all thighs, all
# calves) so the policy's action vector index 0 is FL_hip_joint, index 4 is
# FL_thigh_joint, index 8 is FL_calf_joint, etc. This is DIFFERENT from the
# MuJoCo go2 menagerie XML which orders by *leg* (per-leg hip/thigh/calf).
# ActionApplier.remap_to_env handles the remapping when both joint_name lists
# are populated, but it needs the IL-side list — that's this constant.
IL_GO2_JOINT_NAMES: List[str] = [
    "FL_hip_joint",  "FR_hip_joint",  "RL_hip_joint",  "RR_hip_joint",
    "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# Standing pose used by Isaac Lab's stock Go2 velocity-tracking task. This is
# the offset added to the policy output before sending the joint position
# target to the actuator. Order matches IL_GO2_JOINT_NAMES exactly.
IL_GO2_DEFAULT_JOINT_POS: List[float] = [
    # hips (front pair +0.1, rear pair +0.1; left/right sign convention is
    # encoded in the regex pattern .*L/.*R)
     0.1, -0.1,  0.1, -0.1,
    # thighs — front pair +0.8, rear pair +1.0
     0.8,  0.8,  1.0,  1.0,
    # calves — uniform -1.5
    -1.5, -1.5, -1.5, -1.5,
]


# ---------------------------------------------------------------------------
# Backwards-compat underscore aliases (for existing tests and any other
# callers that imported the private helpers directly before S1).
# ---------------------------------------------------------------------------
# These aliases let `tests/unit/policy/test_il_deploy_contract_assembly.py`
# and any other historical callers keep working without touching their
# imports. New code should import from sim2sim_compiler directly.

_load_mjcf_joint_names = load_mjcf_joint_names
_recover_il_joint_order = recover_joint_order
_expand_actuator_field_per_joint = expand_actuator_field_per_joint
_expand_il_joint_pos_pattern_dict = expand_joint_pos_pattern_dict
_extract_il_observations_for_contract = extract_observations_for_contract
_extract_il_command_ranges = extract_command_ranges
_infer_il_base_body_name = infer_base_body_name
_read_policy_onnx_obs_dim = read_policy_onnx_obs_dim
_read_il_env_yaml = read_env_yaml
_normalize_robot_brand_and_type = normalize_robot_brand_and_type
_assemble_il_deploy_contract = assemble_deploy_contract_dict


# ---------------------------------------------------------------------------
# v2 → v1 upgrade
# ---------------------------------------------------------------------------

def upgrade_v2_to_v1(raw: Dict[str, Any], bundle_path: Path) -> Dict[str, Any]:
    """Rewrite a v2-shaped IL manifest dict so the v1 runtime can consume it.

    The v2 layout looks like::

        { "name": ..., "version": ..., "skill": { ... } }

    The runtime stack expects::

        {
          "name": ...,
          "version": ...,
          "policy":           { "file": ..., "format": ... },
          "observation_space":{ "dim": ..., "components": [...] },
          "action_space":     { "dim": ..., "type": ..., "scale": ..., "clip": ... },
          "runtime":          { "control_frequency_hz": ..., "decimation": ...,
                                "command_defaults": {...} },
          "robot":            { "brand": ..., "type": ..., "num_joints": ...,
                                "joint_names": [...] },
          "skill":            { ... }   # kept verbatim for v2 metadata consumers
        }

    Mutating in place is intentional — callers downstream may already hold
    references to ``raw`` (notably ``CheckpointBundle.raw_manifest``).
    """
    skill = dict(raw.get("skill") or {})
    # NOTE: these calls deliberately use the underscore-prefixed aliases
    # (_read_il_env_yaml, _load_mjcf_joint_names, etc.) instead of the
    # imported sim2sim_compiler functions. The aliases are module-level
    # attributes on this module, so ``monkeypatch.setattr(il_manifest_compat,
    # "_read_il_env_yaml", stub)`` in the test suite actually intercepts
    # the call. Python looks up the name at call time in this module's
    # globals — bypass the alias and the test stubs become no-ops.
    env_yaml = _read_il_env_yaml(bundle_path) or {}

    # ── policy ─────────────────────────────────────────────────────────────
    model_path = str(skill.get("model_path") or "policy.onnx")
    inference_backend = str(skill.get("inference_backend") or "onnx").lower()
    raw["policy"] = {
        "file": model_path,
        "format": inference_backend if inference_backend in {"onnx", "jit"} else "onnx",
    }

    # ── observation_space ──────────────────────────────────────────────────
    # policy.onnx is the ONLY authoritative source for obs_dim — Isaac Lab
    # exporters are known to ship stale ``skill.observation_dim`` values
    # (e.g. when an obs term is added after the skill manifest is first
    # generated). Read the ONNX input shape eagerly so ``bundle.obs_dim``
    # matches the policy's actual training shape AND the deploy_contract
    # total computed below. Falls back to the manifest value only when
    # onnx isn't readable.
    onnx_obs_dim = _read_policy_onnx_obs_dim(Path(bundle_path)) if bundle_path else None
    manifest_obs_dim = int(skill.get("observation_dim") or 0)
    effective_obs_dim = onnx_obs_dim if onnx_obs_dim else manifest_obs_dim
    if onnx_obs_dim and manifest_obs_dim and onnx_obs_dim != manifest_obs_dim:
        log.warning(
            "il_manifest_compat: skill.observation_dim=%d disagrees with "
            "policy.onnx input dim=%d for bundle %s — using ONNX value "
            "(the policy's actual training shape).",
            manifest_obs_dim,
            onnx_obs_dim,
            bundle_path,
        )
    raw["observation_space"] = {
        "dim": int(effective_obs_dim),
        "components": list(skill.get("observation_space_keys") or []),
    }

    # ── action_space ───────────────────────────────────────────────────────
    # Pull scale + clip from env.yaml's actions block when present (Isaac Lab
    # Go2 default is scale=0.25). Fall back to 0.25 to match the stock IL
    # preset rather than 1.0 which would dramatically over-drive joints.
    actions_cfg = (env_yaml.get("actions") or {}) if isinstance(env_yaml, dict) else {}
    joint_pos_action = actions_cfg.get("joint_pos") or {}
    try:
        action_scale = float(joint_pos_action.get("scale", 0.25))
    except (TypeError, ValueError):
        action_scale = 0.25
    raw["action_space"] = {
        "dim": int(skill.get("action_dim") or 0),
        "type": str(skill.get("action_space_type") or "joint_position"),
        "scale": action_scale,
        # Isaac Lab clips actions to ±100 in its default RslRlVecEnvWrapper.
        # That's effectively no-op for normal use; surface it explicitly so
        # downstream clip logic doesn't see ``None``.
        "clip": 100.0,
    }

    # ── runtime ────────────────────────────────────────────────────────────
    # decimation = control_dt / sim_dt. env.yaml gives both; manifest only
    # has control_frequency_hz. Compute when we have env.yaml, otherwise
    # fall back to a sensible IL Go2 default.
    sim_cfg = (env_yaml.get("sim") or {}) if isinstance(env_yaml, dict) else {}
    sim_dt: Optional[float]
    try:
        sim_dt = float(sim_cfg.get("dt")) if sim_cfg.get("dt") is not None else None
    except (TypeError, ValueError):
        sim_dt = None
    decimation = env_yaml.get("decimation") if isinstance(env_yaml, dict) else None
    try:
        decimation = int(decimation) if decimation is not None else None
    except (TypeError, ValueError):
        decimation = None
    if decimation is None:
        # Last-ditch fallback: derive from control_frequency_hz assuming a
        # 0.005s sim_dt (Isaac Lab Go2 default).
        try:
            ctrl_hz = float(skill.get("control_frequency_hz") or 50.0)
        except (TypeError, ValueError):
            ctrl_hz = 50.0
        sim_dt_used = sim_dt or 0.005
        decimation = max(1, int(round((1.0 / ctrl_hz) / sim_dt_used)))

    # command_defaults: pull zeros from the v2 command_interface fields when
    # present, otherwise fall back to a forward-walk command (vx=0.5).
    cmd_iface = skill.get("command_interface") or {}
    cmd_defaults: Dict[str, float] = {}
    if isinstance(cmd_iface, dict):
        for field in cmd_iface.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "")
            try:
                cmd_defaults[name] = float(field.get("default", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    cmd_defaults.setdefault("vx", 0.5)
    cmd_defaults.setdefault("vy", 0.0)
    cmd_defaults.setdefault("wz", 0.0)

    raw["runtime"] = {
        "control_frequency_hz": float(skill.get("control_frequency_hz") or 50.0),
        "decimation": int(decimation),
        "command_defaults": cmd_defaults,
    }

    # ── robot ──────────────────────────────────────────────────────────────
    brand, rtype = _normalize_robot_brand_and_type(skill.get("target_robot_models"))
    if not rtype:
        rtype = "go2"
    if not brand:
        brand = "unitree"

    # Joint names: try the generic recovery path FIRST (works for any
    # robot we have an MJCF for — H1, G1, Spot, future bipeds, future
    # arms). If MJCF loading or actuator parsing fails, fall back to
    # the Go2 hardcode for Go2 bundles, otherwise leave empty.
    #
    # The recovery walks scene.robot.actuators[*].joint_names_expr in
    # dict order and applies each regex against the MJCF joint list,
    # reproducing the order Isaac Lab's Articulation iterates joints
    # at training time. See sim2sim_compiler.recover_joint_order for
    # the algorithmic justification.
    joint_names: List[str] = []
    try:
        mjcf_joints = _load_mjcf_joint_names(rtype)
        if mjcf_joints:
            joint_names = _recover_il_joint_order(env_yaml, mjcf_joints)
    except Exception:
        log.exception(
            "IL joint recovery raised for robot_type=%r — falling back",
            rtype,
        )
        joint_names = []
    if not joint_names and rtype == "go2":
        # Last-ditch hardcode so existing Go2 bundles keep working even
        # when the MJCF resolver is unavailable (e.g. unit tests with no
        # mujoco package installed).
        joint_names = list(IL_GO2_JOINT_NAMES)

    raw["robot"] = {
        "brand": brand,
        "type": rtype,
        "num_joints": int(skill.get("action_dim") or len(joint_names) or 12),
        "joint_names": joint_names,
    }

    # ── default joint pos (used by ObsBuilder + PD controller) ─────────────
    # Stash it inside the skill block + at top level so consumers can find
    # it without re-parsing env.yaml on every call.
    init_pos_dict = (
        ((env_yaml.get("scene") or {}).get("robot") or {}).get("init_state", {})
        if isinstance(env_yaml, dict) else {}
    ).get("joint_pos") if isinstance(env_yaml.get("scene"), dict) else None
    if isinstance(init_pos_dict, dict) and joint_names:
        default_joint_pos = _expand_il_joint_pos_pattern_dict(init_pos_dict, joint_names)
    elif rtype == "go2":
        default_joint_pos = list(IL_GO2_DEFAULT_JOINT_POS)
    else:
        default_joint_pos = [0.0] * (raw["robot"]["num_joints"])
    raw["robot"]["default_joint_pos"] = default_joint_pos

    # Convention marker so the obs/action stack can opt into IL-specific
    # math (body-frame velocities, default-pos offset, ...) without having
    # to re-parse anything.
    raw["inference_convention"] = "isaac_lab"

    # ── deploy_contract (sim2sim_design.yaml Stage A1a) ────────────────
    # Assemble a deploy_contract from the env.yaml fields we already
    # parsed. The contract is what makes ObsBuilder/JointSpace/PDController
    # use the strict, validated path instead of the heuristic legacy
    # paths. If env.yaml is too sparse, we leave deploy_contract absent
    # and the runtime falls back to legacy — log loud either way so
    # users know which path the bundle is on.
    try:
        contract_dict = _assemble_il_deploy_contract(
            env_yaml=env_yaml,
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            sim_dt=sim_dt,
            decimation=decimation,
            bundle_path=Path(bundle_path) if bundle_path else None,
        )
    except Exception:
        log.exception(
            "il_manifest_compat: deploy_contract assembly raised; bundle "
            "will load via the legacy IL path."
        )
        contract_dict = None
    if contract_dict is not None:
        raw["deploy_contract"] = contract_dict
        log.info(
            "il_manifest_compat: assembled deploy_contract for bundle %s "
            "(%d joints, %d obs terms)",
            bundle_path,
            len(contract_dict.get("joint_sdk_names", [])),
            len(contract_dict.get("observations", {})),
        )
    else:
        log.warning(
            "il_manifest_compat: could not assemble deploy_contract from "
            "env.yaml for bundle %s — runtime will use legacy IL path "
            "(env.yaml fallback). Sim2sim parity is best-effort.",
            bundle_path,
        )

    return raw
