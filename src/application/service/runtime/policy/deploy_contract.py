"""DeployContract — single source of truth for sim2sim deployment (verbatim port).

A ``DeployContract`` lives inside ``manifest.yaml`` under the top-level
``deploy_contract`` key. It captures every numerical agreement between the
training environment and the deployment runtime so that an IsaacLab- or
SB3-trained policy can be replayed inside MuJoCo with byte-for-byte fidelity.

Design notes:

* Iteration order of ``observations`` is the policy input order. ``yaml.safe_load``
  preserves dict insertion order on Python 3.7+, so the loader pipeline is
  order-correct without extra work — but only as long as nobody re-sorts the
  dict on the way through. Do not call ``dict(sorted(...))`` on this section.
* All length-``n_joints`` lists must agree on length. ``from_dict`` enforces
  this and raises ``ValueError`` listing the offending fields.
* ``decimation == round(step_dt / sim_dt)`` within 1e-9. Mismatch raises.
* This module imports nothing from ``policy_runner`` / ``obs_builder`` /
  ``action_applier`` — it is a leaf to avoid import cycles. Those modules
  import DeployContract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

log = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2
# Legacy schema version that the v2 loader still accepts (with WARN) for
# bundles produced before the (omega_n, zeta) mass-matrix-adaptive PD
# framework landed. See Stage F of the mjcf-usd-pd plan for the migration.
# A v1 bundle has no ``pd_param`` / ``mujoco_pd_gains`` / ``mujoco_pd_damping``
# fields; deploy stacks that consume those (MuJoCo runtime) fall back to
# treating the v1 ``stiffness`` / ``damping`` arrays as both engines' gains
# and emit a one-time WARN directing the user to re-export the bundle.
LEGACY_SCHEMA_VERSION_V1 = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float_list(value: Any, name: str) -> List[float]:
    if value is None:
        raise ValueError(f"DeployContract.{name}: missing required field")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"DeployContract.{name}: expected list, got {type(value).__name__}"
        )
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DeployContract.{name}: contains non-numeric entry ({exc})"
        ) from exc


def _as_int_list(value: Any, name: str) -> List[int]:
    if value is None:
        raise ValueError(f"DeployContract.{name}: missing required field")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"DeployContract.{name}: expected list, got {type(value).__name__}"
        )
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DeployContract.{name}: contains non-integer entry ({exc})"
        ) from exc


def _as_str_list(value: Any, name: str) -> List[str]:
    if value is None:
        raise ValueError(f"DeployContract.{name}: missing required field")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"DeployContract.{name}: expected list, got {type(value).__name__}"
        )
    return [str(v) for v in value]


def _as_optional_clip(value: Any, name: str) -> Optional[List[float]]:
    """Validate a [lo, hi] clip pair, or None."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"DeployContract.{name}: clip must be null or [lo, hi], got {value!r}"
        )
    lo, hi = float(value[0]), float(value[1])
    if not (lo <= hi):
        raise ValueError(
            f"DeployContract.{name}: clip lo ({lo}) must be <= hi ({hi})"
        )
    return [lo, hi]


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObsTermSpec:
    """Single observation term specification.

    The flat observation vector consumed by the policy is built by iterating
    ``DeployContract.observations`` in insertion order; for each term the
    runtime reads the raw component, multiplies by ``scale``, optionally
    clips, and (when ``history_length > 1``) pushes through a per-term ring
    buffer.
    """

    dim: int
    scale: Union[float, List[float]]
    clip: Optional[List[float]] = None  # [lo, hi] or None
    history_length: int = 1
    grid_params: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, raw: Any, name: str = "<unknown>") -> "ObsTermSpec":
        if not isinstance(raw, dict):
            raise ValueError(
                f"observations[{name}]: expected dict, got {type(raw).__name__}"
            )
        if "dim" not in raw:
            raise ValueError(f"observations[{name}]: missing required field 'dim'")
        try:
            dim = int(raw["dim"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"observations[{name}].dim: must be int ({exc})"
            ) from exc
        if dim <= 0:
            raise ValueError(
                f"observations[{name}].dim: must be positive, got {dim}"
            )

        scale_raw = raw.get("scale", 1.0)
        scale: Union[float, List[float]]
        if isinstance(scale_raw, (list, tuple)):
            scale = _as_float_list(scale_raw, f"observations[{name}].scale")
            if len(scale) != dim:
                raise ValueError(
                    f"observations[{name}].scale: length {len(scale)} "
                    f"!= dim {dim}"
                )
        else:
            try:
                scale = float(scale_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"observations[{name}].scale: must be float or list ({exc})"
                ) from exc

        clip = _as_optional_clip(raw.get("clip"), f"observations[{name}].clip")

        history_length = int(raw.get("history_length", 1) or 1)
        if history_length < 1:
            raise ValueError(
                f"observations[{name}].history_length: must be >= 1, "
                f"got {history_length}"
            )

        grid_params_raw = raw.get("grid_params")
        grid_params: Optional[Dict[str, float]] = None
        if isinstance(grid_params_raw, dict):
            try:
                grid_params = {str(k): float(v) for k, v in grid_params_raw.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"observations[{name}].grid_params: all values must be "
                    f"numeric ({exc})"
                ) from exc

        return cls(
            dim=dim,
            scale=scale,
            clip=clip,
            history_length=history_length,
            grid_params=grid_params,
        )

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "dim": int(self.dim),
            "scale": (
                list(self.scale)
                if isinstance(self.scale, (list, tuple))
                else float(self.scale)
            ),
            "history_length": int(self.history_length),
        }
        out["clip"] = list(self.clip) if self.clip is not None else None
        if self.grid_params is not None:
            out["grid_params"] = {str(k): float(v) for k, v in self.grid_params.items()}
        return out


@dataclass
class ActionSpec:
    """Action term specification.

    Pipeline at deploy time: raw network output → clip (BEFORE scale) → scale
    → optional offset (default_joint_pos) → PD compute → effort_limit clip.
    """

    scale: Union[float, List[float]] = 1.0
    clip: Optional[List[List[float]]] = None  # [[lo, hi], ...] per joint, or None
    offset_mode: str = "default_joint_pos"  # "default_joint_pos" | "zero"

    @classmethod
    def from_dict(cls, raw: Any) -> "ActionSpec":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"action: expected dict, got {type(raw).__name__}"
            )

        scale_raw = raw.get("scale", 1.0)
        scale: Union[float, List[float]]
        if isinstance(scale_raw, (list, tuple)):
            scale = _as_float_list(scale_raw, "action.scale")
        else:
            try:
                scale = float(scale_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"action.scale: must be float or list ({exc})"
                ) from exc

        clip_raw = raw.get("clip")
        clip: Optional[List[List[float]]]
        if clip_raw is None:
            clip = None
        else:
            if not isinstance(clip_raw, (list, tuple)):
                raise ValueError(
                    f"action.clip: expected null or list of [lo,hi] pairs, "
                    f"got {type(clip_raw).__name__}"
                )
            clip = []
            for i, pair in enumerate(clip_raw):
                pair_validated = _as_optional_clip(pair, f"action.clip[{i}]")
                if pair_validated is None:
                    raise ValueError(
                        f"action.clip[{i}]: per-joint pair cannot be null"
                    )
                clip.append(pair_validated)

        offset_mode = str(raw.get("offset_mode", "default_joint_pos"))
        if offset_mode not in ("default_joint_pos", "zero"):
            raise ValueError(
                f"action.offset_mode: must be 'default_joint_pos' or 'zero', "
                f"got {offset_mode!r}"
            )

        return cls(scale=scale, clip=clip, offset_mode=offset_mode)

    def to_dict(self) -> dict:
        return {
            "scale": (
                list(self.scale)
                if isinstance(self.scale, (list, tuple))
                else float(self.scale)
            ),
            "clip": (
                [list(pair) for pair in self.clip]
                if self.clip is not None
                else None
            ),
            "offset_mode": str(self.offset_mode),
        }


# ---------------------------------------------------------------------------
# Top-level dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeployContract:
    """Single source of truth for sim2sim deployment numerics.

    Phase 5 — IR-only joint contract:
      * ``joint_sdk_names`` carries **IR role names** (e.g. ``"hip_FL"``,
        ``"thigh_FL"``) in policy execution order — NOT physical joint
        names. The deploy stack resolves each IR role to a physical name
        at MJCF/URDF/SDK binding time via the robot SKU.

    SKU is **not** stored on the contract. ``manifest.robot.sku`` is the
    single source of truth for the bundle's robot identity, and callers
    (PolicyRunner, CompatibilityChecker, joint_space helpers) plumb that
    SKU in explicitly. Storing a second copy here used to allow the two
    to silently diverge, breaking IR→physical translation in subtle ways.
    Legacy bundles that still carry ``robot_sku`` in their on-disk
    contract are accepted at load time — ``from_dict`` discards the field
    silently with a debug log.
    """

    schema_version: int
    joint_sdk_names: List[str]
    joint_ids_map: List[int]
    stiffness: List[float]
    damping: List[float]
    effort_limit: List[float]
    default_joint_pos: List[float]
    step_dt: float
    sim_dt: float
    decimation: int
    observations: Dict[str, ObsTermSpec] = field(default_factory=dict)
    action: ActionSpec = field(default_factory=ActionSpec)
    commands: Dict[str, Any] = field(default_factory=dict)
    base_body_name: str = ""
    velocity_limit: Optional[List[float]] = None
    saturation_effort: Optional[List[float]] = None
    # IL training init root position [x, y, z] in meters. Bundle exporter
    # persists ``spec.actor.init_pos_*`` here so MuJoCo deploy spawns the
    # robot at the same base z the policy was trained on. Without this
    # the deploy stack falls back to a contact-based heuristic that
    # silently drifts off-distribution whenever the bundle's
    # default_joint_pos differs from the MJCF keyframe leg geometry
    # (which it almost always does for IL-trained quadrupeds).
    init_base_pos: Optional[List[float]] = None
    # Stage F: canonical (omega_n, zeta) PD parameterization + per-engine
    # derived gains. ``pd_param`` is the source of truth; the engine
    # arrays are pre-derived so runtime loaders don't need to re-solve.
    # v1 bundles carry all three as None — the MuJoCo runtime then
    # treats the v1 ``stiffness``/``damping`` lists as both engines'
    # gains and WARNs. See LEGACY_SCHEMA_VERSION_V1 above.
    pd_param: Optional[Dict[str, Any]] = None
    mujoco_pd_gains: Optional[List[float]] = None       # per-joint kp for MuJoCo
    mujoco_pd_damping: Optional[List[float]] = None     # per-joint kd for MuJoCo
    # Set when bundle finalize could not derive MuJoCo gains because the
    # registered MJCF asset doesn't cover the trained joint set (e.g. IL
    # G1 trained on 43-DOF USD while assets.MJCF is 29-DOF stock). The
    # bundle still ships for IsaacSim / cloud deploy targets; MuJoCo
    # runtime loader checks this field and refuses to load with the
    # carried reason text. Empty / None = MuJoCo deploy supported.
    mujoco_deploy_unsupported: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Any) -> "DeployContract":
        """Build a DeployContract from a raw manifest section.

        Strict-mode contract: the manifest **must** carry a populated
        ``deploy_contract`` section. The historical legacy-bundle path
        (returning ``None``) was removed in the strict-canvas migration —
        it degraded the deploy stack to a heuristic that silently
        misaligned joint order/limits when the bundle didn't actually
        ship a contract. Re-export old bundles through the current
        spec_compiler / bundle_finalizer.
        """
        if raw is None or (isinstance(raw, dict) and not raw):
            raise ValueError(
                "deploy_contract: manifest section is missing or empty. "
                "Legacy bundles without a deploy_contract are no longer "
                "supported; re-export from the current pipeline."
            )
        if not isinstance(raw, dict):
            raise ValueError(
                f"deploy_contract: expected dict, got {type(raw).__name__}"
            )

        schema_version = int(raw.get("schema_version", 0))
        if schema_version not in (CURRENT_SCHEMA_VERSION, LEGACY_SCHEMA_VERSION_V1):
            raise ValueError(
                f"deploy_contract.schema_version: expected "
                f"{CURRENT_SCHEMA_VERSION} (or legacy "
                f"{LEGACY_SCHEMA_VERSION_V1} with WARN), got {schema_version}"
            )
        if schema_version == LEGACY_SCHEMA_VERSION_V1:
            # WHY KEPT: on-disk legacy compat for bundles produced before
            # the sim2sim PD framework landed (RELEASE/CLAUDE.md §1.8 c).
            # v1 bundles have no pd_param + no mujoco_pd_gains; the MuJoCo
            # deploy stack falls back to treating the v1 stiffness/damping
            # as both engines' gains. WARN directs the user to re-export.
            log.warning(
                "deploy_contract: loading legacy schema v1 bundle — "
                "pd_param + mujoco_pd_gains are absent. The MuJoCo runtime "
                "will fall back to scalar gains; re-export through the "
                "current pipeline to graduate onto the (omega_n, zeta) "
                "mass-matrix-adaptive PD path."
            )

        joint_sdk_names = _as_str_list(raw.get("joint_sdk_names"), "joint_sdk_names")
        joint_ids_map = _as_int_list(raw.get("joint_ids_map"), "joint_ids_map")
        stiffness = _as_float_list(raw.get("stiffness"), "stiffness")
        damping = _as_float_list(raw.get("damping"), "damping")
        effort_limit = _as_float_list(raw.get("effort_limit"), "effort_limit")
        default_joint_pos = _as_float_list(
            raw.get("default_joint_pos"), "default_joint_pos"
        )

        n = len(joint_sdk_names)
        length_check = {
            "joint_ids_map": len(joint_ids_map),
            "stiffness": len(stiffness),
            "damping": len(damping),
            "effort_limit": len(effort_limit),
            "default_joint_pos": len(default_joint_pos),
        }
        mismatched = {k: v for k, v in length_check.items() if v != n}
        if mismatched:
            raise ValueError(
                f"deploy_contract length mismatch — joint_sdk_names has {n} "
                f"entries but {mismatched} differ"
            )

        bad_ids = [i for i in joint_ids_map if not (0 <= i < n)]
        if bad_ids:
            raise ValueError(
                f"deploy_contract.joint_ids_map: out-of-range indices "
                f"{bad_ids} (n_joints={n})"
            )
        if sorted(joint_ids_map) != list(range(n)):
            raise ValueError(
                f"deploy_contract.joint_ids_map: must be a permutation of "
                f"0..{n - 1}, got {joint_ids_map}"
            )

        try:
            sim_dt = float(raw["sim_dt"])
            step_dt = float(raw["step_dt"])
        except KeyError as exc:
            raise ValueError(
                f"deploy_contract: missing required field {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"deploy_contract.sim_dt/step_dt: must be float ({exc})"
            ) from exc

        if sim_dt <= 0 or step_dt <= 0:
            raise ValueError(
                f"deploy_contract: sim_dt and step_dt must be positive, "
                f"got sim_dt={sim_dt}, step_dt={step_dt}"
            )

        try:
            decimation = int(raw["decimation"])
        except KeyError as exc:
            raise ValueError(
                f"deploy_contract: missing required field {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"deploy_contract.decimation: must be int ({exc})"
            ) from exc

        if decimation < 1:
            raise ValueError(
                f"deploy_contract.decimation: must be >= 1, got {decimation}"
            )

        expected = round(step_dt / sim_dt)
        if abs(step_dt / sim_dt - decimation) > 1e-6 or expected != decimation:
            raise ValueError(
                f"deploy_contract: decimation ({decimation}) does not match "
                f"round(step_dt / sim_dt) = round({step_dt} / {sim_dt}) = "
                f"{expected}"
            )

        obs_raw = raw.get("observations") or {}
        if not isinstance(obs_raw, dict):
            raise ValueError(
                f"deploy_contract.observations: expected dict, "
                f"got {type(obs_raw).__name__}"
            )
        observations: Dict[str, ObsTermSpec] = {}
        for term_name, term_raw in obs_raw.items():
            observations[str(term_name)] = ObsTermSpec.from_dict(
                term_raw, name=str(term_name)
            )
        if not observations:
            raise ValueError(
                "deploy_contract.observations: must contain at least one term"
            )
        total_obs_dim = sum(t.dim * t.history_length for t in observations.values())
        if total_obs_dim <= 0:
            raise ValueError(
                "deploy_contract.observations: total dim must be positive"
            )

        action = ActionSpec.from_dict(raw.get("action"))

        commands_raw = raw.get("commands") or {}
        if not isinstance(commands_raw, dict):
            raise ValueError(
                f"deploy_contract.commands: expected dict, "
                f"got {type(commands_raw).__name__}"
            )

        base_body_name = str(raw.get("base_body_name", "") or "")

        velocity_limit: Optional[List[float]] = None
        if raw.get("velocity_limit") is not None:
            velocity_limit = _as_float_list(raw.get("velocity_limit"), "velocity_limit")
            if len(velocity_limit) != n:
                raise ValueError(
                    f"deploy_contract.velocity_limit: length {len(velocity_limit)} "
                    f"!= n_joints {n}"
                )
        saturation_effort: Optional[List[float]] = None
        if raw.get("saturation_effort") is not None:
            saturation_effort = _as_float_list(
                raw.get("saturation_effort"), "saturation_effort"
            )
            if len(saturation_effort) != n:
                raise ValueError(
                    f"deploy_contract.saturation_effort: length "
                    f"{len(saturation_effort)} != n_joints {n}"
                )

        init_base_pos: Optional[List[float]] = None
        ibp_raw = raw.get("init_base_pos")
        if ibp_raw is not None:
            try:
                ibp = [float(x) for x in ibp_raw]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"deploy_contract.init_base_pos: must be a list of "
                    f"three floats ({exc})"
                ) from exc
            if len(ibp) != 3:
                raise ValueError(
                    f"deploy_contract.init_base_pos: expected length 3, "
                    f"got {len(ibp)}"
                )
            init_base_pos = ibp

        # Legacy ``robot_sku`` field — older bundles persisted a copy of
        # the bundle SKU here. Silently drop it; manifest.robot.sku is
        # the single source of truth now (see class docstring).
        if "robot_sku" in raw:
            log.debug(
                "DeployContract.from_dict: legacy 'robot_sku' field "
                "ignored; manifest.robot.sku is the single source of truth"
            )

        # Stage F: v2 fields. Strict-parsed when schema_version == 2;
        # accepted-as-None when v1 (the WARN above directed re-export).
        pd_param_raw: Optional[Dict[str, Any]] = None
        mujoco_pd_gains: Optional[List[float]] = None
        mujoco_pd_damping: Optional[List[float]] = None
        mujoco_deploy_unsupported_raw = raw.get("mujoco_deploy_unsupported")
        mujoco_unsupported = bool(mujoco_deploy_unsupported_raw)
        if schema_version == CURRENT_SCHEMA_VERSION:
            pd_param_obj = raw.get("pd_param")
            if pd_param_obj is not None:
                if not isinstance(pd_param_obj, dict):
                    raise ValueError(
                        f"deploy_contract.pd_param: expected dict, got "
                        f"{type(pd_param_obj).__name__}"
                    )
                pd_param_raw = dict(pd_param_obj)
            mj_gains_raw = raw.get("mujoco_pd_gains")
            mj_damp_raw = raw.get("mujoco_pd_damping")
            if pd_param_raw is not None and not mujoco_unsupported:
                # When pd_param is present AND MuJoCo deploy isn't opted
                # out, the engine arrays MUST be present and length-
                # matched. Missing them = corrupt bundle (the finalizer's
                # job is to derive them).
                if mj_gains_raw is None or mj_damp_raw is None:
                    raise ValueError(
                        "deploy_contract: pd_param is present but "
                        "mujoco_pd_gains / mujoco_pd_damping are missing — "
                        "bundle is incomplete (Stage F finalizer must "
                        "derive these). Re-export the bundle."
                    )
                mujoco_pd_gains = _as_float_list(mj_gains_raw, "mujoco_pd_gains")
                mujoco_pd_damping = _as_float_list(mj_damp_raw, "mujoco_pd_damping")
                if len(mujoco_pd_gains) != n:
                    raise ValueError(
                        f"deploy_contract.mujoco_pd_gains: length "
                        f"{len(mujoco_pd_gains)} != n_joints {n}"
                    )
                if len(mujoco_pd_damping) != n:
                    raise ValueError(
                        f"deploy_contract.mujoco_pd_damping: length "
                        f"{len(mujoco_pd_damping)} != n_joints {n}"
                    )
            elif pd_param_raw is None and (
                mj_gains_raw is not None or mj_damp_raw is not None
            ):
                # pd_param missing but engine arrays present → inconsistent.
                raise ValueError(
                    "deploy_contract: mujoco_pd_gains / mujoco_pd_damping "
                    "are present but pd_param (provenance source) is "
                    "missing. Either populate pd_param or drop the "
                    "derived arrays."
                )
            # When mujoco_unsupported is set, engine arrays MUST be None
            # (or absent). Present-but-non-null arrays + unsupported flag
            # is contradictory.
            if mujoco_unsupported and (
                mj_gains_raw is not None or mj_damp_raw is not None
            ):
                raise ValueError(
                    "deploy_contract: mujoco_deploy_unsupported is set but "
                    "mujoco_pd_gains / mujoco_pd_damping are non-null. "
                    "Drop the arrays when opting out of MuJoCo deploy."
                )

        return cls(
            schema_version=schema_version,
            joint_sdk_names=joint_sdk_names,
            joint_ids_map=joint_ids_map,
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort_limit,
            default_joint_pos=default_joint_pos,
            step_dt=step_dt,
            sim_dt=sim_dt,
            decimation=decimation,
            observations=observations,
            action=action,
            commands=dict(commands_raw),
            base_body_name=base_body_name,
            velocity_limit=velocity_limit,
            saturation_effort=saturation_effort,
            init_base_pos=init_base_pos,
            pd_param=pd_param_raw,
            mujoco_pd_gains=mujoco_pd_gains,
            mujoco_pd_damping=mujoco_pd_damping,
            mujoco_deploy_unsupported=(
                str(mujoco_deploy_unsupported_raw)
                if mujoco_unsupported else None
            ),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize back to a yaml-dump-friendly dict.

        IMPORTANT: callers must use ``yaml.dump(..., sort_keys=False)`` so the
        observations key order is preserved on disk.
        """
        out: Dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "joint_sdk_names": list(self.joint_sdk_names),
            "joint_ids_map": list(self.joint_ids_map),
            "stiffness": list(self.stiffness),
            "damping": list(self.damping),
            "effort_limit": list(self.effort_limit),
            "default_joint_pos": list(self.default_joint_pos),
            "sim_dt": float(self.sim_dt),
            "step_dt": float(self.step_dt),
            "decimation": int(self.decimation),
            "observations": {
                name: spec.to_dict() for name, spec in self.observations.items()
            },
            "action": self.action.to_dict(),
            "commands": dict(self.commands),
            "base_body_name": str(self.base_body_name),
        }
        if self.velocity_limit is not None:
            out["velocity_limit"] = list(self.velocity_limit)
        if self.saturation_effort is not None:
            out["saturation_effort"] = list(self.saturation_effort)
        if self.init_base_pos is not None:
            out["init_base_pos"] = [float(v) for v in self.init_base_pos]
        # Stage F v2 fields. Only emit when populated — keeps v2 bundles
        # whose canvas had no ActuatorPDNode (legacy scalar path) lean.
        if self.pd_param is not None:
            out["pd_param"] = dict(self.pd_param)
        if self.mujoco_pd_gains is not None:
            out["mujoco_pd_gains"] = [float(v) for v in self.mujoco_pd_gains]
        if self.mujoco_pd_damping is not None:
            out["mujoco_pd_damping"] = [float(v) for v in self.mujoco_pd_damping]
        if self.mujoco_deploy_unsupported:
            out["mujoco_deploy_unsupported"] = str(self.mujoco_deploy_unsupported)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joint_sdk_names)

    @property
    def total_obs_dim(self) -> int:
        return sum(t.dim * t.history_length for t in self.observations.values())

    def is_identity_joint_map(self) -> bool:
        """True when joint_ids_map is [0, 1, ..., n-1]."""
        return list(self.joint_ids_map) == list(range(self.n_joints))
