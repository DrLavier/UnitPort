"""DeployContract — single source of truth for sim2sim deployment.

A ``DeployContract`` lives inside ``manifest.yaml`` under the top-level
``deploy_contract`` key. It captures every numerical agreement between the
training environment and the deployment runtime so that an IsaacLab- or
SB3-trained policy can be replayed inside MuJoCo with byte-for-byte fidelity.

Design notes (see knowledge_base/sim2sim_design.yaml for the full plan):

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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

CURRENT_SCHEMA_VERSION = 1


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
    # Term-specific geometric metadata. Currently only populated for
    # `height_scan` / `heightfield_scan`, where the runtime ray-caster
    # needs the training-time grid shape to match ONNX input dims exactly.
    # Keys for height_scan: nx, ny, resolution, offset_z, target_offset.
    # None for terms that don't need it.
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

    Lives inside ``manifest.yaml`` under the top-level ``deploy_contract`` key.
    Loaded lazily via ``CheckpointBundle.deploy_contract`` (cached_property).
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["DeployContract"]:
        """Build a DeployContract from a raw manifest section.

        Returns None when ``raw`` is None or empty (legacy bundle without
        contract). Raises ValueError on any structural problem so callers
        get a crisp load-time error instead of garbage at runtime.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(
                f"deploy_contract: expected dict, got {type(raw).__name__}"
            )
        if not raw:
            return None

        schema_version = int(raw.get("schema_version", 0))
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"deploy_contract.schema_version: expected "
                f"{CURRENT_SCHEMA_VERSION}, got {schema_version}"
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

        # joint_ids_map sanity: every value must be a valid index in [0, n)
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

        # Observations — preserve insertion order. yaml.safe_load on Py3.7+
        # already gives us an ordered dict, but iterate explicitly so the
        # contract is robust to whatever the caller hands us.
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
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize back to a yaml-dump-friendly dict.

        IMPORTANT: callers must use ``yaml.dump(..., sort_keys=False)`` so the
        observations key order is preserved on disk.
        """
        return {
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
