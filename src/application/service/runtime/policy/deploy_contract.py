# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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
class RecurrentSpec:
    """Recurrent-policy contract (缺口①).

    Present only when the exported policy carries a GRU/LSTM memory cell.
    Absent (``DeployContract.recurrent is None``) ⇒ feed-forward MLP policy —
    the loader treats the bundle exactly as before (R6). schema_version stays
    2 (decision R3): this block is additive + orthogonal to the PD numerics,
    so existing v2 MLP bundles load unchanged.
    """

    rnn_type: str            # "gru" | "lstm"
    hidden_size: int         # H
    num_layers: int          # L
    reset: str = "episode"   # hidden-state reset semantics

    @classmethod
    def from_dict(cls, raw: Any) -> "RecurrentSpec":
        if not isinstance(raw, dict):
            raise ValueError(
                f"deploy_contract.recurrent: expected dict, got {type(raw).__name__}"
            )
        rnn_type = str(raw.get("rnn_type", "")).strip().lower()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError(
                f"deploy_contract.recurrent.rnn_type: expected 'gru'|'lstm', "
                f"got {raw.get('rnn_type')!r}"
            )
        hidden_size = int(raw.get("hidden_size", 0))
        num_layers = int(raw.get("num_layers", 0))
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError(
                f"deploy_contract.recurrent: hidden_size ({hidden_size}) and "
                f"num_layers ({num_layers}) must both be positive"
            )
        return cls(
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            reset=str(raw.get("reset", "episode") or "episode"),
        )

    def to_dict(self) -> dict:
        return {
            "rnn_type": self.rnn_type,
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "reset": self.reset,
        }


@dataclass
class PerItemObsSpec:
    """Per-item observation-tail contract (SB3 per-item-reward canvases).

    Present only when the trained policy's obs carries a per-item tail —
    ``[cmd_norm, soft_weight per item]`` — appended by the SB3 env when the
    canvas wires per-item rewards (``GenericMujocoEnv._expose_active_item_obs``).
    The deploy ObsBuilder reconstructs that tail from the current command using
    the SAME ItemResolver, so the deployed obs == the trained obs (CLAUDE.md
    §11). Absent ⇒ no tail (obs is the term layout only).

    The tail layout is ``[cmd_norm, weight(item_ids[0]), ..., weight(item_ids[-1])]``
    so ``tail_dim == 1 + len(item_ids)``. ``item_ids`` order is the training
    item order (terms_by_item insertion order) and MUST NOT be reordered.
    """

    item_ids: List[str]
    command_ranges: Dict[str, Dict[str, List[float]]]
    blend_width: float

    @property
    def tail_dim(self) -> int:
        return 1 + len(self.item_ids)

    @classmethod
    def from_dict(cls, raw: Any) -> "PerItemObsSpec":
        if not isinstance(raw, dict):
            raise ValueError(
                f"deploy_contract.per_item_obs: expected dict, got "
                f"{type(raw).__name__}"
            )
        item_ids = [str(i) for i in (raw.get("item_ids") or [])]
        if not item_ids:
            raise ValueError(
                "deploy_contract.per_item_obs.item_ids: must be a non-empty "
                "list (the policy obs carries one weight per item)."
            )
        ranges_raw = raw.get("command_ranges") or {}
        if not isinstance(ranges_raw, dict):
            raise ValueError(
                "deploy_contract.per_item_obs.command_ranges: expected dict"
            )
        command_ranges: Dict[str, Dict[str, List[float]]] = {}
        for iid in item_ids:
            item_ranges = ranges_raw.get(iid) or {}
            parsed: Dict[str, List[float]] = {}
            for chan, span in (item_ranges.items() if isinstance(item_ranges, dict) else []):
                if isinstance(span, (list, tuple)) and len(span) == 2:
                    parsed[str(chan)] = [float(span[0]), float(span[1])]
            command_ranges[iid] = parsed
        blend_width = float(raw.get("blend_width", 0.10) or 0.10)
        return cls(
            item_ids=item_ids,
            command_ranges=command_ranges,
            blend_width=blend_width,
        )

    def to_dict(self) -> dict:
        return {
            "item_ids": list(self.item_ids),
            "command_ranges": {
                iid: {ch: [float(s[0]), float(s[1])] for ch, s in r.items()}
                for iid, r in self.command_ranges.items()
            },
            "blend_width": float(self.blend_width),
        }


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
    # Step 4 (remotized actuators): per-remotized-joint angle→max-torque
    # lookup tables, keyed by **IR role** (e.g. ``"calf_FL"`` — the same name
    # space as ``joint_sdk_names`` / ``robot.joint_names``, NOT the physical
    # MJCF joint name). Values are ``TorqueLookupTable`` instances. Present
    # only for robots whose brand manifest declared ``remotized_joints``; the
    # MuJoCo ``PDController`` reads these to clamp each remotized joint against
    # its angle-dependent ceiling instead of a static scalar effort_limit.
    # This is the SINGLE runtime copy of the table data the bundle ships — it
    # is the same payload embedded under ``pd_derivation.remotized_joints``
    # (the finalizer reuses it; they cannot drift).
    mujoco_torque_lookups: Optional[Dict[str, Any]] = None
    # 缺口① — recurrent (GRU/LSTM) policy contract. None ⇒ feed-forward MLP
    # (the loader/engine path is byte-identical to pre-recurrent bundles).
    # Additive + orthogonal to the PD numerics; schema_version stays 2 (R3).
    recurrent: Optional["RecurrentSpec"] = None
    # Per-item obs tail (SB3 per-item-reward canvases). None ⇒ obs is the term
    # layout only. When present, the deploy ObsBuilder appends
    # ``[cmd_norm, item weights]`` (tail_dim = 1 + n_items) so the deployed obs
    # matches the trained obs. Additive + orthogonal; schema_version stays 2.
    per_item_obs: Optional["PerItemObsSpec"] = None
    # Stage 0 realized==designed audit (Layer 1). Records the two-engine realized
    # CONSISTENCY verdicts + the friction single-source value (``scene_contract``)
    # and the pose-invariant foot-collision-geometry hash (``foot_geometry``).
    # The deploy side consumes two things from here: PolicyRunner applies
    # ``scene_contract.friction_static`` to the MuJoCo geom_friction (method A —
    # so deploy realizes the canvas μ), and re-hashes the deploy MJCF foot geoms
    # against ``foot_geometry.aggregate`` (fail-loud on mismatch). None ⇒ bundle
    # predates the audit (migration WARN; the runtime self-check is skipped).
    # Additive + orthogonal; schema_version stays 2.
    sim2sim_audit: Optional[Dict[str, Any]] = None
    # Stage 2 (收窄版) cross-engine DR coverage provenance. Records that the
    # measured cross-engine friction residual (block-slide campaign: box factor
    # [0.56,1.02]) is COVERED by the existing ground-friction DR — no new DR
    # channel was built. Present only on IsaacLab/PhysX-trained bundles (the
    # only cross-engine case: PhysX train → MuJoCo deploy; an SB3 bundle trains
    # and deploys in MuJoCo, same engine, no residual). The load-time check
    # below WARNs when ``coverage.friction_dr_enabled`` is False — the
    # "covered" guarantee is switch-dependent, so a bundle trained with
    # friction DR OFF must fail loud, not silently inherit the guarantee (§8).
    # Absent ⇒ no provenance (SB3 bundles, or pre-Stage-2 PhysX bundles) — NOT
    # a defect, so no migration WARN. Additive + orthogonal; schema stays 2.
    sim2sim_dr_provenance: Optional[Dict[str, Any]] = None
    # sim2sim joint-order contract. The LIVE IsaacLab PhysX articulation joint
    # order (in IR-role names) the policy's action vector is emitted in,
    # CAPTURED at train time (``JointPositionAction`` resolves ``.*`` against the
    # live articulation; action slot i == DOF i). This IS the bundle's per-joint
    # array order — for a correctly-finalized bundle it equals
    # ``joint_sdk_names``. The deploy stack treats it as the authoritative joint
    # order. ABSENT on IsaacLab bundles exported before the capture landed: their
    # order was taken from a possibly-stale USD prim dump that can disagree with
    # the live order and silently scramble joints (the Go2 sim2sim-twitch bug) →
    # load-time RE-EXPORT WARN below. ``joint_order_source`` records whether it
    # came from the action term or the raw articulation. Additive + orthogonal;
    # schema_version stays 2 (gated on presence, not version).
    articulation_joint_order: Optional[List[str]] = None
    joint_order_source: Optional[str] = None

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

        # ----- Stage 2 (收窄版) cross-engine DR coverage fail-loud check.
        # The bundle's "cross-engine friction residual is covered by the
        # existing ground-friction DR" guarantee is SWITCH-DEPENDENT: it holds
        # only while friction DR was enabled at training time. A PhysX-trained
        # policy (this provenance is only written by the IsaacLab finalizer)
        # deployed in MuJoCo with friction DR OFF has the [0.56,1.02] residual
        # UNCOVERED. Warn loudly rather than let the guarantee silently lapse
        # when the switch is off (§8 — illusion-of-success defense).
        _dr_prov = raw.get("sim2sim_dr_provenance")
        if isinstance(_dr_prov, dict):
            _cov = _dr_prov.get("coverage")
            if isinstance(_cov, dict) and _cov.get("friction_dr_enabled") is False:
                _res = _dr_prov.get("friction_residual") or {}
                _ext = _res.get("extreme_factor", 1.78)
                log.warning(
                    "deploy_contract: sim2sim_dr_provenance reports friction DR "
                    "was DISABLED at training time (coverage.friction_dr_enabled "
                    "= false). The 'cross-engine friction residual covered by the "
                    "existing ground-friction DR' guarantee therefore does NOT "
                    "hold: this PhysX-trained policy replayed in MuJoCo may show "
                    "a sim2sim friction mismatch (realistic band ~0-12%%, extreme "
                    "up to ~%.2fx at high mu). RE-TRAIN with friction DR enabled, "
                    "or knowingly accept the uncovered residual. See the sim2sim "
                    "Stage 2 (收窄版) provenance.", float(_ext)
                )

        # ----- Joint-order capture migration WARN (§8(c), CLAUDE.md §11).
        # IsaacLab bundles exported before the live-articulation joint-order
        # capture landed carry no ``articulation_joint_order``: their per-joint
        # array order was taken from a (possibly stale) USD prim dump
        # (``stage.Traverse()`` authoring order), which can disagree with the
        # trained policy's true LIVE PhysX articulation order and silently
        # SCRAMBLE joints at deploy — the Go2 sim2sim-convulsion bug. The true
        # order is not recoverable from the bundle, so RE-EXPORT (re-train +
        # re-finalize through the capture-enabled pipeline) is required. WARN,
        # not raise, so existing bundles still load (they may deploy fine if
        # their robot's USD dump order happened to equal the live order).
        # Detected via ``pd_derivation`` — written ONLY by the IsaacLab
        # finalizer, never by the SB3 exporter — so SB3/MuJoCo bundles (which
        # are MJCF-ordered and need no capture) are never falsely flagged.
        if raw.get("articulation_joint_order") is None and isinstance(
            raw.get("pd_derivation"), dict
        ):
            log.warning(
                "deploy_contract: this IsaacLab bundle has no "
                "'articulation_joint_order' — it predates the live joint-order "
                "capture. Its per-joint arrays were ordered from a USD prim "
                "dump, which may disagree with the trained policy's live PhysX "
                "articulation order and SCRAMBLE every joint at deploy (the "
                "sim2sim convulsion bug). RE-EXPORT REQUIRED: re-train + "
                "re-finalize through the current pipeline — the correct joint "
                "order CANNOT be recovered from this bundle."
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
        mujoco_torque_lookups: Optional[Dict[str, Any]] = None
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

                # ----- Armature-parity migration WARN (§8(c), CLAUDE.md §10).
                # Bundles exported before the armature-parity fix have no
                # ``physx_armature_emitted`` in pd_derivation.m_eff_source: their
                # PhysX TRAINING side ran with the USD-default armature (0) while
                # the (ωn,ζ) gains baked in the MJCF armature (mj_fullM folds it
                # in), so PhysX realized a stiffer-than-designed plant at
                # low-inertia joints (ankles/wrists/elbows, where armature is the
                # bulk of m_eff). The policy LEARNED to rely on that hard distal
                # joint — a reliance FROZEN in the network weights.
                #
                # This is a RE-TRAIN directive, NOT re-export. Be explicit (§8 —
                # an "either/or" that lets the user pick the non-working option
                # manufactures an illusion of success):
                #   * Re-export re-derives the deploy gains from the SAME frozen
                #     checkpoint + same past training run. Those gains were
                #     ALREADY armature-0.1-correct (mj_fullM always folded it in),
                #     so re-export is byte-identical w.r.t. armature and cannot
                #     change what PhysX realized while the policy was learning.
                #   * Only re-training reaches the weights, letting the policy
                #     learn on the corrected plant (current pipeline emits
                #     ImplicitActuatorCfg.armature so PhysX realizes the MJCF
                #     armature).
                # WARN (not raise) so existing bundles still load; the export-time
                # realized-inertia gate is the hard gate for new bundles.
                _pd_deriv = raw.get("pd_derivation")
                _m_eff_src = (
                    _pd_deriv.get("m_eff_source", {})
                    if isinstance(_pd_deriv, dict) else {}
                )
                if isinstance(_m_eff_src, dict) and (
                    "physx_armature_emitted" not in _m_eff_src
                ):
                    log.warning(
                        "deploy_contract: this bundle predates armature parity "
                        "(no pd_derivation.m_eff_source.physx_armature_emitted). "
                        "If the robot's MJCF authors a non-zero joint armature, "
                        "its PhysX TRAINING side ran WITHOUT that armature, so "
                        "the policy was trained against a stiffer-than-designed "
                        "plant at low-inertia joints (ankles/wrists/elbows, where "
                        "armature dominates the joint inertia). That reliance is "
                        "FROZEN in the policy weights — the network learned to "
                        "lean on a hard distal joint that exists in NEITHER the "
                        "corrected training physics NOR the MuJoCo deploy "
                        "physics. RE-EXPORT ALONE CANNOT FIX THIS: it only "
                        "re-derives the deploy gains (already armature-correct via "
                        "mj_fullM — byte-identical) from the same frozen "
                        "checkpoint and cannot change what PhysX realized during "
                        "the past training run. You MUST RE-TRAIN with the "
                        "current pipeline (which emits ImplicitActuatorCfg."
                        "armature so PhysX realizes the MJCF armature). See "
                        "CLAUDE.md §10."
                    )

                # ----- Stage 0 sim2sim-audit migration WARN (§8(c)). Bundles
                # exported before the realized==designed audit carry no
                # ``sim2sim_audit`` block: they lack the recorded ground friction
                # (so deploy realizes the MJCF-default μ, NOT the trained canvas
                # value) and the foot-collision-geometry hash (so the deploy-side
                # self-check cannot run). WARN, not raise — the bundle still
                # loads; the export-time audit gate is the hard gate for new
                # bundles. Unlike the armature gap, RE-EXPORT suffices here: the
                # friction value + foot hash are deploy-time alignment metadata,
                # independent of the trained weights (no re-train needed).
                _audit = raw.get("sim2sim_audit")
                if not isinstance(_audit, dict):
                    log.warning(
                        "deploy_contract: this bundle predates the sim2sim "
                        "realized==designed audit (no 'sim2sim_audit' block). "
                        "Deploy will NOT apply the canvas ground friction "
                        "(it realizes the MJCF-default μ, not the trained "
                        "value), and the foot-collision-geometry self-check is "
                        "skipped. RE-EXPORT to attach the audit — the trained "
                        "weights are unaffected (deploy-time alignment + a "
                        "recorded hash), so re-export suffices; re-training is "
                        "NOT required. See CLAUDE.md §10/§11."
                    )
                else:
                    if _audit.get("scene_contract") is None:
                        log.warning(
                            "deploy_contract: sim2sim_audit carries no "
                            "'scene_contract' — no canvas ground friction was "
                            "recorded, so deploy keeps the MJCF-default μ. If "
                            "the policy trained against a non-default ground "
                            "friction, re-export with a play_ground_setting "
                            "node so deploy realizes the same μ."
                        )
                    _fg = _audit.get("foot_geometry")
                    _fg_status = _fg.get("status") if isinstance(_fg, dict) else None
                    if _fg_status != "ok":
                        log.warning(
                            "deploy_contract: sim2sim_audit foot_geometry is "
                            "absent or not 'ok' (status=%r) — the deploy-side "
                            "foot-collision-geometry self-check will be skipped "
                            "for this bundle (it engages only on status='ok').",
                            _fg_status,
                        )
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
                # ----- Load-time cross-engine TORQUE guard (cheap, always-on;
                # runs on the arrays actually parsed into memory).
                #
                # Post-fix, BOTH engines mass-weight off the same m_eff, so the
                # PhysX gains baked into training (stiffness/damping) and the
                # MuJoCo gains the bundle ships (mujoco_pd_gains/_damping)
                # produce identical torque. Old bundles trained under the
                # unit-mass bug carry stiffness = ωn² (no mass) while
                # mujoco_pd_gains = M·ωn²: torque diverges 9–65× (worst at
                # low-inertia joints like the knee). Such a policy was trained
                # against torques MuJoCo can never reproduce, so loading it for
                # local inference yields a silently-broken rollout (robot
                # collapses). Fail loud (§8) — the only fix is to re-train.
                #
                # We probe TORQUE, not raw gains, so a kd-only divergence
                # cannot slip past a matching-kp check. Two sentinel probes
                # (position-error torque kp·Δq, velocity torque kd·q̇) cover
                # kp and kd independently and can't cancel to a false pass.
                # This is the cheaper sibling of the export-time canary-joint
                # check in sim2sim_calibration.run_calibration (1e-3 there;
                # 1% here, looser to tolerate YAML float round-trip noise
                # while still catching the 9–65× bug by a wide margin).
                _DQ, _QD = 0.1, 0.5  # rad position error, rad/s joint velocity
                worst_rel = 0.0
                for sk, sd, mk, md in zip(
                    stiffness, damping, mujoco_pd_gains, mujoco_pd_damping
                ):
                    rel_pos = abs(sk - mk) * _DQ / max(abs(mk) * _DQ, 1e-9)
                    rel_vel = abs(sd - md) * _QD / max(abs(md) * _QD, 1e-9)
                    worst_rel = max(worst_rel, rel_pos, rel_vel)
                if worst_rel > 0.01:
                    raise ValueError(
                        "deploy_contract: PhysX training torque "
                        "(stiffness/damping) diverges from the MuJoCo torque "
                        f"(mujoco_pd_gains/_damping) by up to {worst_rel*100:.0f}% "
                        "— this bundle was trained before the 2026-05 PD fix, "
                        "when PhysX used the unit-mass form (kp=ωn²) while "
                        "MuJoCo mass-weighted (kp=M·ωn²). The policy learned "
                        "against torques MuJoCo cannot reproduce; replaying it "
                        "collapses the robot. RE-TRAIN the policy with the "
                        "current pipeline (both engines now mass-weight "
                        "identically). See CLAUDE.md §10."
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

            # ----- Remotized-actuator lookup tables (Step 4.2 / 4.3).
            # Keyed by IR role (same name space as joint_sdk_names). Parse
            # into TorqueLookupTable objects and run the load-time guard.
            mtl_raw = raw.get("mujoco_torque_lookups")
            if mtl_raw is not None and not mujoco_unsupported:
                if not isinstance(mtl_raw, dict):
                    raise ValueError(
                        "deploy_contract.mujoco_torque_lookups: expected a "
                        f"mapping of ir_role -> table, got "
                        f"{type(mtl_raw).__name__}"
                    )
                from application.physics.actuators.torque_lookup import (
                    TorqueLookupTable,
                )
                effort_by_role = dict(zip(joint_sdk_names, effort_limit))
                parsed: Dict[str, Any] = {}
                for role, tbl_dict in mtl_raw.items():
                    if role not in effort_by_role:
                        raise ValueError(
                            f"deploy_contract.mujoco_torque_lookups: ir_role "
                            f"{role!r} is not in joint_sdk_names — the lookup "
                            f"key space must match the contract joint names "
                            f"(IR roles), not physical MJCF names."
                        )
                    table = TorqueLookupTable.from_dict(tbl_dict)
                    # axis range sanity: a remotized joint that sweeps < 1 rad
                    # is suspicious (knees sweep ~2.5 rad). WARN, don't fail —
                    # a short-range joint is unusual but not provably wrong.
                    lo, hi = table.axis_range
                    if (hi - lo) < 1.0:
                        log.warning(
                            "deploy_contract.mujoco_torque_lookups[%r]: axis "
                            "range %.3f rad is suspiciously small (< 1.0) — "
                            "verify the table covers the joint's operating "
                            "range.", role, hi - lo,
                        )
                    # The static effort_limit for a remotized joint is derived
                    # by manifest_parser as the lookup PEAK (the joint's max
                    # capability), so peak == eff is the NORMAL case. Raise only
                    # when the table peak is MEANINGFULLY below the static cap —
                    # a wrong table (wrong column/robot/units) is orders of
                    # magnitude off, while a 0.1% margin absorbs float round-trip
                    # between the env.yaml lookup and the embedded table without
                    # masking a real defect (§8). A peak below the cap means the
                    # lookup can never reach the box cap — the table is wrong.
                    eff = float(effort_by_role[role])
                    if 0.0 < eff and table.peak_torque < eff * (1.0 - 1e-3):
                        raise ValueError(
                            f"deploy_contract.mujoco_torque_lookups[{role!r}]: "
                            f"table peak {table.peak_torque:.2f} N·m is below "
                            f"the joint's static effort_limit {eff:.2f} N·m. "
                            f"Remotization exists to RAISE the ceiling; a "
                            f"lower peak means the table is wrong (wrong "
                            f"column extracted, wrong robot, or unit error)."
                        )
                    parsed[role] = table

                # Cross-check against the provenance block (Step 3.2): every
                # joint the compiler recorded as remotized MUST have a runtime
                # lookup here, else the MuJoCo side would silently clamp it at
                # the static effort_limit while IsaacLab trained it remotized.
                pd_deriv = raw.get("pd_derivation")
                isaac_tables_by_role: Dict[str, Any] = {}
                if isinstance(pd_deriv, dict):
                    rj = pd_deriv.get("remotized_joints")
                    if isinstance(rj, dict):
                        declared = (rj.get("joints") or {})
                        for _name, entry in declared.items():
                            role = (entry or {}).get("ir_role")
                            if role and role not in parsed:
                                raise ValueError(
                                    f"deploy_contract: pd_derivation marks "
                                    f"joint ir_role={role!r} as remotized but "
                                    f"mujoco_torque_lookups has no table for "
                                    f"it — the MuJoCo runtime would clamp it "
                                    f"at the static effort_limit while it was "
                                    f"trained remotized. Re-export the bundle."
                                )
                            _itbl = (entry or {}).get("table")
                            if role and _itbl is not None:
                                isaac_tables_by_role[role] = (
                                    TorqueLookupTable.from_dict(_itbl)
                                )

                # ----- Step 5: multi-point cross-engine torque consistency
                # (load-time, LOOSER 1% — same threshold layering rationale as
                # the kp/kd guard above; tolerates YAML float round-trip). For
                # each remotized joint, sample the lookup range and confirm the
                # IsaacLab-side ceiling (from pd_derivation, via to_isaaclab_array)
                # and the MuJoCo-side ceiling (from mujoco_torque_lookups) clamp
                # the PD torque to the same value. A divergence is diagnosed as
                # PD-region (gains drifted — World B regression) vs clip-region
                # (lookup data drifted — conversion / embedding bug). Needs the
                # per-engine gains; only runs when they're present (remotization
                # implies an ActuatorPDNode, so they should be).
                if mujoco_pd_gains is not None and mujoco_pd_damping is not None:
                    from application.physics.actuators.torque_consistency import (
                        check_remotized_torque_consistency,
                    )
                    role_to_idx = {nm: i for i, nm in enumerate(joint_sdk_names)}
                    # === D18 one-way call-order contract (top of block) ===
                    # INVARIANT: §10 has enforced 1% PD-gain consistency by this
                    # point. PD-region diagnosis below is unreachable in
                    # production paths under current §10 behavior. If §10 is
                    # relaxed or skipped for any joint, PD-region diagnosis
                    # becomes reachable here and needs integration test coverage.
                    #
                    # We do NOT touch the §10 guard — this is a one-way contract:
                    # the multi-point check self-declares its dependency on §10.
                    # The assert re-states the precondition at runtime (fail-loud,
                    # D2) so a §10 change cannot silently invalidate the
                    # unreachability reasoning. It is always true today (the §10
                    # guard above raised otherwise), so it adds no new production
                    # behavior — only a tripwire. §10's per-joint metric reduces
                    # to |stiffness−mujoco_pd_gains|/|mujoco_pd_gains| ≤ 1%
                    # (the probe Δq cancels), which we recompute here.
                    for _role in parsed:
                        _i = role_to_idx[_role]
                        _mk = float(mujoco_pd_gains[_i])
                        _md = float(mujoco_pd_damping[_i])
                        assert (
                            abs(float(stiffness[_i]) - _mk)
                            <= 0.01 * max(abs(_mk), 1e-9)
                            and abs(float(damping[_i]) - _md)
                            <= 0.01 * max(abs(_md), 1e-9)
                        ), (
                            f"D18 INVARIANT VIOLATED for {_role!r}: §10's 1% "
                            f"PD-gain consistency precondition does not hold "
                            f"here, so the PD-region diagnosis path is now "
                            f"reachable and needs integration-test coverage. "
                            f"Was §10 (the deploy_contract kp/kd guard) relaxed "
                            f"or skipped? See design decision D18."
                        )
                    for role, mtable in parsed.items():
                        idx = role_to_idx[role]
                        # WHY KEPT (§8 (c) defense-in-depth, not a silent
                        # fallback of required input): when provenance carries
                        # the IsaacLab-side table, compare the two contract
                        # LOCATIONS (catches embedding drift between them); when
                        # it doesn't, compare the MuJoCo table against itself
                        # through to_isaaclab_array (still catches a conversion
                        # bug). Either way the check is meaningful.
                        itable = isaac_tables_by_role.get(role, mtable)
                        result = check_remotized_torque_consistency(
                            joint=role,
                            isaac_table=itable,
                            mujoco_table=mtable,
                            kp_isaac=float(stiffness[idx]),
                            kd_isaac=float(damping[idx]),
                            kp_mujoco=float(mujoco_pd_gains[idx]),
                            kd_mujoco=float(mujoco_pd_damping[idx]),
                            threshold=0.01,
                        )
                        if not result.passed:
                            raise ValueError(
                                "deploy_contract: " + result.diagnosis()
                            )
                mujoco_torque_lookups = parsed

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
            mujoco_torque_lookups=mujoco_torque_lookups,
            mujoco_deploy_unsupported=(
                str(mujoco_deploy_unsupported_raw)
                if mujoco_unsupported else None
            ),
            # 缺口① — recurrent block is additive + version-orthogonal (R3):
            # parsed whenever present so both v1 and v2 recurrent bundles load.
            recurrent=(
                RecurrentSpec.from_dict(raw["recurrent"])
                if raw.get("recurrent") is not None else None
            ),
            # Per-item obs tail (additive + version-orthogonal): parsed whenever
            # present so the deploy ObsBuilder can rebuild [cmd_norm, weights].
            per_item_obs=(
                PerItemObsSpec.from_dict(raw["per_item_obs"])
                if raw.get("per_item_obs") is not None else None
            ),
            # Stage 0 audit block (additive + version-orthogonal): parsed
            # whenever present so PolicyRunner can apply the canvas friction +
            # run the foot-geometry self-check. A non-dict value is ignored.
            sim2sim_audit=(
                dict(raw["sim2sim_audit"])
                if isinstance(raw.get("sim2sim_audit"), dict) else None
            ),
            # Stage 2 (收窄版) cross-engine DR coverage provenance (additive +
            # version-orthogonal). The fail-loud check fired above (or below)
            # keys on coverage.friction_dr_enabled.
            sim2sim_dr_provenance=(
                dict(raw["sim2sim_dr_provenance"])
                if isinstance(raw.get("sim2sim_dr_provenance"), dict) else None
            ),
            # sim2sim joint-order contract (additive + version-orthogonal):
            # the captured LIVE articulation order. Parsed whenever present so
            # the deploy stack trusts the policy's true action order.
            articulation_joint_order=(
                [str(x) for x in raw["articulation_joint_order"]]
                if isinstance(raw.get("articulation_joint_order"), list) else None
            ),
            joint_order_source=(
                str(raw["joint_order_source"])
                if raw.get("joint_order_source") is not None else None
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
        if self.mujoco_torque_lookups:
            # Embed each table inline (self-contained bundle). Keyed by IR
            # role; value is the v1 table payload via TorqueLookupTable.to_dict.
            out["mujoco_torque_lookups"] = {
                role: table.to_dict()
                for role, table in self.mujoco_torque_lookups.items()
            }
        if self.mujoco_deploy_unsupported:
            out["mujoco_deploy_unsupported"] = str(self.mujoco_deploy_unsupported)
        if self.recurrent is not None:
            out["recurrent"] = self.recurrent.to_dict()
        if self.per_item_obs is not None:
            out["per_item_obs"] = self.per_item_obs.to_dict()
        if self.sim2sim_audit is not None:
            out["sim2sim_audit"] = dict(self.sim2sim_audit)
        if self.sim2sim_dr_provenance is not None:
            out["sim2sim_dr_provenance"] = dict(self.sim2sim_dr_provenance)
        if self.articulation_joint_order is not None:
            out["articulation_joint_order"] = list(self.articulation_joint_order)
        if self.joint_order_source is not None:
            out["joint_order_source"] = str(self.joint_order_source)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joint_sdk_names)

    @property
    def total_obs_dim(self) -> int:
        base = sum(t.dim * t.history_length for t in self.observations.values())
        tail = self.per_item_obs.tail_dim if self.per_item_obs is not None else 0
        return base + tail

    def is_identity_joint_map(self) -> bool:
        """True when joint_ids_map is [0, 1, ..., n-1]."""
        return list(self.joint_ids_map) == list(range(self.n_joints))
