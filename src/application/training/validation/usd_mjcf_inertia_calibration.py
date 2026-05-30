# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""USD→MJCF mass / inertia calibration.

Closes the sim2sim **base-property gap**: a robot trains in IsaacLab/PhysX on a
USD whose rigid bodies carry authored mass/inertia, then replays in MuJoCo on an
MJCF whose ``<inertial>`` values may be placeholders (the canonical example is
the MuJoCo-Menagerie Boston Dynamics Spot trunk: ``mass=32.86`` with isotropic
``diaginertia=0.13144 0.13144 0.13144`` — pitch/yaw ~10× too low, mass ~2× too
high). The export/load PD-torque calibration gates check per-joint torque only,
**not** base inertia, so this slips through and the policy that stays upright in
PhysX tips over within a few frames in MuJoCo.

This module diffs the USD-authoritative per-body mass/inertia (harvested by the
extended Dump USD path, ``bootstrap/discover_usd_bodies.py`` → ``body_inertials``)
against the live MJCF's ``MjModel.body_*`` arrays and writes a **non-destructive
YAML overlay** under ``USER_CONFIG_DIR/robot_assets/inertia_overlays/<sku>.yaml``.
The original MJCF is never edited; the overlay is applied to the in-memory
``MjModel`` at load (see ``service.runtime.simulation.mujoco.mj_actor`` +
``robot_assets.runtime.read_mjcf_inertia_overlay``).

Design rules (CLAUDE.md §1.8 — no silent fallback):
    * **Unauthored USD inertia → raise.** A body whose USD ``diagonalInertia`` is
      absent/zero would, under PhysX, get an auto-computed tensor at sim load that
      ``pxr`` cannot read. Overriding only its mass and leaving the MJCF inertia
      placeholder is a *half-fix that looks fixed* — more dangerous than no fix.
      We raise unless the caller explicitly passes
      ``unauthored_inertia_policy="ignore"``.
    * **Frame mismatch → raise.** Mass and the inertia principal *moments* (sorted
      eigenvalues) are frame-invariant and transfer cleanly. COM and the principal
      *axes* (quaternion) are expressed in each format's own body-local frame; for
      identity-axes bodies (the trunk) the frames trivially coincide. For
      non-identity-axes bodies we cross-check the invariants: if they disagree we
      cannot confirm the frames align, so we raise unless
      ``frame_mismatch_policy="ignore"``.
    * **Tolerances are explicit + audited.** ``diff_tolerance`` is a parameter, not
      a hardcoded literal; every body evaluated-but-not-overridden is recorded in
      ``ignored_bodies`` with its measured diff, so "which bodies were skipped" is
      reviewable.

The overlay touches **only** four ``MjModel`` arrays — ``body_mass``,
``body_inertia``, ``body_iquat``, ``body_ipos``. Joint dynamics, actuators,
collision geometry, sites, and the kinematic tree are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitport_sdk import log_info, log_warning

from .mjcf_base_calibration import hash_file_sha256, primary_family


OVERLAY_SCHEMA = "unitport.mjcf_inertia_overlay/v1"

# Overlay file lives under USER_CONFIG_DIR, SKU-keyed (CLAUDE.md §7 — SKU is the
# unique key; brand/model are display-only and must never be a filesystem key).
OVERLAY_REL_DIR = "robot_assets/inertia_overlays"


# Default diff tolerances. Explicit + overridable + recorded in the overlay so a
# reviewer can see exactly what threshold produced the ignored_bodies list.
DEFAULT_DIFF_TOLERANCE: Dict[str, float] = {
    "mass_relative": 0.01,      # |Δmass| / mjcf_mass
    "inertia_relative": 0.02,   # max over sorted principal moments
    "com_absolute_m": 0.001,    # ‖Δcom‖ (m)
}

# A quaternion is treated as the identity rotation when |w| ≈ 1 (sign-agnostic:
# q and −q are the same rotation, and MuJoCo may store either).
_IDENTITY_QUAT_TOL = 1e-4

_EPS = 1e-9


class InertiaCalibrationError(ValueError):
    """Raised on a fail-loud calibration condition (unauthored inertia, frame
    mismatch, non-physical tensor, missing registry data). Subclass of
    ValueError so existing ``except ValueError`` callers (mirroring
    mjcf_base_calibration) still catch it."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BodyInertiaOverride:
    """One MJCF body's correction. ``fields_applied`` is the subset of the
    field-group tokens {``mass``, ``inertia``, ``com``} the loader will patch.
    ``inertia`` writes ``body_inertia`` + ``body_iquat`` *together* (they jointly
    define the tensor in the body frame, so they must never be split)."""

    mjcf_body: str
    ir_role: str
    usd_body: str
    fields_applied: List[str]
    # "aligned" | "diverge" | "n/a" — D25 kinematic parent-joint frame check.
    kinematic_frame_check: str = "n/a"
    # from/to pairs (None when that field is not part of this override)
    mass_from: Optional[float] = None
    mass_to: Optional[float] = None
    diag_from: Optional[List[float]] = None
    diag_to: Optional[List[float]] = None
    iquat_from: Optional[List[float]] = None
    iquat_to: Optional[List[float]] = None
    com_from: Optional[List[float]] = None
    com_to: Optional[List[float]] = None
    diff: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ir_role": self.ir_role,
            "usd_body": self.usd_body,
            "fields_applied": list(self.fields_applied),
            "kinematic_frame_check": self.kinematic_frame_check,
            "diff": {k: float(v) for k, v in self.diff.items()},
        }
        if self.mass_to is not None:
            out["mass"] = {"from": _f(self.mass_from), "to": _f(self.mass_to)}
        if self.diag_to is not None:
            out["diaginertia"] = {"from": _fl(self.diag_from), "to": _fl(self.diag_to)}
        if self.iquat_to is not None:
            out["iquat"] = {"from": _fl(self.iquat_from), "to": _fl(self.iquat_to)}
        if self.com_to is not None:
            out["com"] = {"from": _fl(self.com_from), "to": _fl(self.com_to)}
        return out


@dataclass
class IgnoredBody:
    """A body evaluated but not overridden (all diffs below tolerance, or an
    explicitly-acknowledged unauthored/frame-mismatch skip)."""

    mjcf_body: str
    ir_role: str
    reason: str                       # below_tolerance | unauthored_acknowledged | frame_mismatch_acknowledged
    diff: Dict[str, float] = field(default_factory=dict)
    kinematic_frame_check: str = "n/a"   # aligned | diverge | n/a

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ir_role": self.ir_role,
            "reason": self.reason,
            "kinematic_frame_check": self.kinematic_frame_check,
            "diff": {k: float(v) for k, v in self.diff.items()},
        }


@dataclass
class MjcfInertiaOverlayResult:
    sku: str
    status: str                       # "calibrated" | "no_changes_needed"
    brand: str
    model: str
    usd_ref: str
    usd_sha256: str
    usd_stage_units: Dict[str, float]
    mjcf_path: str
    mjcf_sha256: str
    diff_tolerance: Dict[str, float]
    unauthored_inertia_policy: str
    frame_mismatch_policy: str
    overrides: List[BodyInertiaOverride] = field(default_factory=list)
    ignored: List[IgnoredBody] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    computed_at: str = ""

    def to_overlay_dict(self) -> Dict[str, Any]:
        """The full YAML overlay payload (DD-3 schema)."""
        return {
            "schema_version": 1,
            "kind": OVERLAY_SCHEMA,
            "robot_sku": self.sku,
            "robot_display": {"brand": self.brand, "model": self.model},
            "generated_at": self.computed_at,
            "status": self.status,
            "source": {
                "usd_ref": self.usd_ref,
                "usd_sha256": self.usd_sha256,
                "usd_stage_units": dict(self.usd_stage_units),
                "mjcf_path": self.mjcf_path,
                "mjcf_sha256": self.mjcf_sha256,
            },
            "match_by": "ir_role",
            "diff_tolerance": dict(self.diff_tolerance),
            "unauthored_inertia_policy": self.unauthored_inertia_policy,
            "frame_mismatch_policy": self.frame_mismatch_policy,
            "body_overrides": {o.mjcf_body: o.to_dict() for o in self.overrides},
            "ignored_bodies": {i.mjcf_body: i.to_dict() for i in self.ignored},
            "notes": list(self.notes),
        }

    def to_pointer_dict(self, overlay_rel: str) -> Dict[str, Any]:
        """The light summary stored in robot_assets/state.json (mirrors the
        ``mjcf_base_height_offset`` pointer pattern)."""
        return {
            "schema": OVERLAY_SCHEMA,
            "status": self.status,
            "file": overlay_rel,
            "usd_sha256": self.usd_sha256,
            "mjcf_sha256": self.mjcf_sha256,
            "n_overrides": len(self.overrides),
            "generated_at": self.computed_at,
        }


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _f(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def _fl(x: Optional[List[float]]) -> Optional[List[float]]:
    return None if x is None else [float(v) for v in x]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quat_is_identity(q: Optional[List[float]]) -> bool:
    """True iff *q* (w,x,y,z) is the identity rotation (sign-agnostic)."""
    if q is None:
        return True  # absent principalAxes == identity by USD convention
    arr = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < _EPS:
        return True
    arr = arr / n
    return abs(abs(float(arr[0])) - 1.0) < _IDENTITY_QUAT_TOL


def _invariants_agree(d_usd: List[float], d_mjcf: List[float], rel: float) -> bool:
    """Frame-invariant inertia comparison: sorted principal moments within *rel*.

    The eigenvalues of a rigid body's inertia tensor (= the principal moments)
    are intrinsic to the mass distribution and independent of the body-local
    frame. So agreement here means the two assets describe the *same* physical
    inertia and differ at most by a principal-axes rotation.
    """
    su = sorted(float(x) for x in d_usd)
    sm = sorted(float(x) for x in d_mjcf)
    for a, b in zip(su, sm):
        if abs(a - b) / max(abs(b), _EPS) > rel:
            return False
    return True


def _physicality_ok(diag: List[float]) -> Tuple[bool, str]:
    """A valid rigid-body principal-moment triple is positive and satisfies the
    triangle inequalities (each moment ≤ sum of the other two)."""
    d = [float(x) for x in diag]
    if any(x <= 0.0 for x in d):
        return False, f"non-positive principal moment(s): {d}"
    a, b, c = sorted(d)
    # generous numerical slack on the triangle inequality
    if c > a + b + 1e-6 * max(c, 1.0):
        return False, f"triangle inequality violated: sorted={[a, b, c]}"
    return True, ""


# ---------------------------------------------------------------------------
# Kinematic-invariant frame check (D25) — prove USD prim frame == MJCF body frame
# via the parent joint's anchor + axis, independent of the inertia data.
# ---------------------------------------------------------------------------

# Thresholds for "the joint pins parent↔child identically in both assets".
_FRAME_AXIS_TOL_DEG = 1.0
_FRAME_ORIGIN_TOL_M = 1e-3   # 1 mm

_AXIS_UNIT = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _quat_to_R(q: List[float]) -> "np.ndarray":
    """Rotation matrix for quaternion (w,x,y,z). Identity on degenerate input."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < _EPS:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _mjcf_parent_joint_frame(model, child_bid: int):
    """(origin_parent, axis_parent) of ``child_bid``'s first hinge/slide joint,
    expressed in the PARENT body frame — or ``None`` if the body has no such
    joint (root / free / fixed)."""
    import mujoco
    jnum = int(model.body_jntnum[child_bid])
    if jnum < 1:
        return None
    jid = int(model.body_jntadr[child_bid])
    jtype = int(model.jnt_type[jid])
    if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return None
    jnt_pos = np.asarray(model.jnt_pos[jid], dtype=np.float64)      # child-frame anchor
    jnt_axis = np.asarray(model.jnt_axis[jid], dtype=np.float64)    # child-frame axis
    body_pos = np.asarray(model.body_pos[child_bid], dtype=np.float64)   # child→parent
    R_cp = _quat_to_R([float(v) for v in model.body_quat[child_bid]])
    origin_parent = body_pos + R_cp @ jnt_pos
    axis_parent = R_cp @ jnt_axis
    return origin_parent, axis_parent


def _usd_parent_joint_frame(
    usd_joint_frames: Dict[str, Any], child_body: str, parent_body: str,
):
    """(origin_parent, axis_parent) for the USD joint that connects
    ``parent_body`` → ``child_body``, expressed in the parent frame — or
    ``None`` if not found / axis token absent.

    BOTH endpoints must match: a body is the *child* in its own parent joint
    but the *parent* in its children's joints, so matching on the child name
    alone can grab the wrong joint (this caused the spurious 302mm/90° hip
    divergence). Requiring both endpoints disambiguates and is order-agnostic.
    """
    for jf in (usd_joint_frames or {}).values():
        if not isinstance(jf, dict):
            continue
        b0, b1 = jf.get("body0"), jf.get("body1")
        if b1 == child_body and b0 == parent_body:
            anchor, rot = jf.get("local_pos0"), jf.get("local_rot0")   # body0 = parent
        elif b0 == child_body and b1 == parent_body:
            anchor, rot = jf.get("local_pos1"), jf.get("local_rot1")   # body1 = parent
        else:
            continue
        ax = _AXIS_UNIT.get(str(jf.get("axis", "")).upper())
        if anchor is None or ax is None:
            return None
        R = _quat_to_R(rot if rot else [1.0, 0.0, 0.0, 0.0])
        return np.asarray(anchor, dtype=np.float64), R @ np.asarray(ax, dtype=np.float64)
    return None


def _kinematic_frame_check(
    model, mjcf_body: str, usd_body: str, usd_joint_frames: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """D25: prove the USD prim frame == MJCF body frame via the parent joint.

    Returns ``("aligned"|"diverge"|"unknown", {origin_err_m, axis_deg})``.
    ``unknown`` = could not compute either side (missing joint / root body);
    callers treat it like ``diverge`` (cannot prove → respect policy).
    """
    import mujoco
    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjcf_body))
    if bid < 0:
        return "unknown", {}
    # The USD joint connecting parent→child is matched by BOTH body names. USD
    # and MJCF share physical body names for URDF-derived assets, so the MJCF
    # parent name resolves the USD endpoint directly.
    parent_bid = int(model.body_parentid[bid])
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid) or ""
    mj = _mjcf_parent_joint_frame(model, bid)
    usd = _usd_parent_joint_frame(usd_joint_frames, usd_body, parent_name)
    if mj is None or usd is None:
        return "unknown", {}
    (mj_o, mj_a), (usd_o, usd_a) = mj, usd
    origin_err = float(np.linalg.norm(mj_o - usd_o))
    na, nb = np.linalg.norm(mj_a), np.linalg.norm(usd_a)
    if na < _EPS or nb < _EPS:
        return "unknown", {"origin_err_m": origin_err}
    cosang = float(np.clip(np.dot(mj_a, usd_a) / (na * nb), -1.0, 1.0))
    axis_deg = float(np.degrees(np.arccos(cosang)))
    detail = {"origin_err_m": origin_err, "axis_deg": axis_deg}
    if origin_err <= _FRAME_ORIGIN_TOL_M and axis_deg <= _FRAME_AXIS_TOL_DEG:
        return "aligned", detail
    return "diverge", detail


# ---------------------------------------------------------------------------
# MJCF / registry readers
# ---------------------------------------------------------------------------

def _ir_to_body_name(per_format_block: Dict[str, Any]) -> Dict[str, str]:
    """``{uid: {name, ir_role}}`` → ``{ir_role: body_name}`` (skips empty roles)."""
    out: Dict[str, str] = {}
    for spec in (per_format_block or {}).values():
        if isinstance(spec, dict):
            nm = str(spec.get("name", "") or "")
            ir = str(spec.get("ir_role", "") or "")
            if nm and ir:
                out[ir] = nm
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_mjcf_inertia(
    sku: str,
    *,
    diff_tolerance: Optional[Dict[str, float]] = None,
    unauthored_inertia_policy: str = "raise",
    frame_mismatch_policy: str = "raise",
    usd_inertials: Optional[Dict[str, Dict[str, Any]]] = None,
    usd_joint_frames: Optional[Dict[str, Dict[str, Any]]] = None,
    usd_stage_units: Optional[Dict[str, float]] = None,
) -> MjcfInertiaOverlayResult:
    """Compute the USD→MJCF inertia overlay for one SKU.

    Args:
        sku: canonical SKU (from ``registers.robots``).
        diff_tolerance: overrides :data:`DEFAULT_DIFF_TOLERANCE` (any subset).
        unauthored_inertia_policy: ``"raise"`` (default) | ``"ignore"``. See module
            docstring — ``"raise"`` refuses to produce a half-fix.
        frame_mismatch_policy: ``"raise"`` (default) | ``"ignore"`` | ``"transfer"``.
            Only consulted for a non-identity-axes / invariant-disagreement body
            whose **kinematic frame check (D25) could not prove alignment**. When
            the parent-joint anchor+axis match (≤1°, ≤1mm) the transfer happens
            automatically regardless of this policy. ``"transfer"`` forces the
            transfer on a kinematic *diverge*; ``"ignore"`` skips the body.
        usd_joint_frames: optional pre-harvested joint frames (parallel to
            ``usd_inertials``); harvested from the same Dump subprocess when None.
        usd_inertials: optional pre-harvested ``{usd_body_name: {...}}`` (skips the
            Isaac-venv Dump subprocess; used by tests / batch callers). When None,
            this function spawns the Dump USD subprocess to harvest live values.
        usd_stage_units: provenance only (recorded in the overlay) when
            ``usd_inertials`` is injected.

    Raises:
        InertiaCalibrationError: missing registry data (no MJCF / no
            bodies_per_format / no USD ref), unauthored USD inertia under the
            default policy, frame mismatch under the default policy, or a
            non-physical resulting tensor.
    """
    import mujoco

    from application.physics.mujoco_path_compat import safe_mjcf_path
    from application.service.robot_assets.service import get_robot_asset_service
    from registers import robots as _robots_registry

    tol = dict(DEFAULT_DIFF_TOLERANCE)
    if diff_tolerance:
        tol.update({k: float(v) for k, v in diff_tolerance.items()})
    if unauthored_inertia_policy not in ("raise", "ignore"):
        raise InertiaCalibrationError(
            f"unauthored_inertia_policy must be 'raise' or 'ignore', got "
            f"{unauthored_inertia_policy!r}"
        )
    if frame_mismatch_policy not in ("raise", "ignore", "transfer"):
        raise InertiaCalibrationError(
            f"frame_mismatch_policy must be 'raise', 'ignore', or 'transfer', got "
            f"{frame_mismatch_policy!r}"
        )

    entry = _robots_registry.get_robot(sku)
    if not entry:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} is not registered in "
            f"registers.robots. Resolve the SKU via resolve_to_sku() first."
        )
    brand = str(entry.get("brand", "") or "")
    model = str(entry.get("model", "") or "")
    families = list(entry.get("families", []) or [])
    family = primary_family(families) if families else ""

    svc = get_robot_asset_service()
    asset = svc.resolve(sku)
    if asset is None:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} could not be resolved to an "
            f"asset record."
        )

    mjcf_path = getattr(asset, "mjcf_path", None)
    if mjcf_path is None or not Path(mjcf_path).exists():
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} has no resolvable MJCF on disk "
            f"(got {mjcf_path!r}). Register an MJCF on the Robot Asset card."
        )
    mjcf_path = Path(mjcf_path)

    # IR-role → body-name maps from the (already-dumped) registry tables.
    bodies_mjcf = _ir_to_body_name(
        (entry.get("bodies_per_format", {}) or {}).get("MJCF", {}) or {}
    )
    bodies_usd = _ir_to_body_name(
        (entry.get("bodies_per_format", {}) or {}).get("USD", {}) or {}
    )
    if not bodies_mjcf:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} has no bodies_per_format[MJCF]. "
            f"Run Dump MJCF on the Robot Asset card first."
        )
    if not bodies_usd:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} has no bodies_per_format[USD]. "
            f"Run Dump USD on the Robot Asset card first (it also harvests the "
            f"mass/inertia this calibration needs)."
        )

    # ---- USD ref + live inertial harvest (unless injected) ----
    usd_path = getattr(asset, "usd_path", None)
    usd_url = getattr(asset, "usd_url", None)
    usd_ref = str(usd_path) if usd_path else (usd_url or "")
    if not usd_ref:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: sku {sku!r} declares no USD path or Nucleus "
            f"URL — cannot harvest authoritative inertia."
        )
    stage_units = dict(usd_stage_units or {})
    # Persisted-cache path: the regular Dump USD button now writes
    # body_inertials_per_format.USD + joint_frames_per_format.USD (rule §11
    # full-framework persistence — see RobotAssetService.set_discovered_payload).
    # Use the cache when present; only spawn the Isaac subprocess on a cache
    # miss (e.g. legacy registry that pre-dates the new schema). This collapses
    # the historical two-subprocess pattern (Dump button + calibration) into
    # one.
    if usd_inertials is None:
        cached_inertials = (
            (entry.get("body_inertials_per_format", {}) or {}).get("USD") or {}
        )
        if cached_inertials:
            log_info(
                f"[inertia_calibration] sku={sku!r}: using cached USD inertia "
                f"({len(cached_inertials)} bodies) from robots_custom.json — "
                f"no Dump subprocess needed."
            )
            usd_inertials = dict(cached_inertials)
            if usd_joint_frames is None:
                usd_joint_frames = dict(
                    (entry.get("joint_frames_per_format", {}) or {}).get("USD")
                    or {}
                )
    if usd_inertials is None:
        from application.service.robot_assets.discovery_subprocess import (
            DiscoveryError,
            discover_usd,
        )
        log_info(
            f"[inertia_calibration] sku={sku!r}: no cached USD inertia in the "
            f"registry overlay — harvesting via Dump subprocess (ref={usd_ref!r})…"
            f" Re-run Dump USD on the Robot Asset card to populate the cache and "
            f"avoid this subprocess next time."
        )
        try:
            payload = discover_usd(usd_ref, {}, family=family or "generic")
        except DiscoveryError as exc:
            raise InertiaCalibrationError(
                f"calibrate_mjcf_inertia: USD inertia harvest failed for sku "
                f"{sku!r}: {exc}"
            ) from exc
        usd_inertials = dict(payload.get("body_inertials", {}) or {})
        if usd_joint_frames is None:
            usd_joint_frames = dict(payload.get("joint_frames", {}) or {})
    if usd_joint_frames is None:
        usd_joint_frames = {}
    if not usd_inertials:
        raise InertiaCalibrationError(
            f"calibrate_mjcf_inertia: USD dump returned no body_inertials for sku "
            f"{sku!r}. The USD may not author UsdPhysics.MassAPI on its bodies."
        )

    # ---- Load MJCF, read MjModel body_* ----
    model_mj = mujoco.MjModel.from_xml_path(safe_mjcf_path(mjcf_path))

    overrides: List[BodyInertiaOverride] = []
    ignored: List[IgnoredBody] = []
    notes: List[str] = []

    for ir_role, mjcf_body in sorted(bodies_mjcf.items()):
        usd_body = bodies_usd.get(ir_role)
        if not usd_body:
            continue  # role not present in USD table — nothing to calibrate from
        usd = usd_inertials.get(usd_body)
        if not isinstance(usd, dict):
            continue  # USD body has no harvested inertial (e.g. not RigidBodyAPI)

        bid = int(mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, mjcf_body))
        if bid < 0:
            notes.append(
                f"ir_role {ir_role!r} → MJCF body {mjcf_body!r} not found in model; skipped"
            )
            continue

        mjcf_mass = float(model_mj.body_mass[bid])
        mjcf_com = [float(v) for v in model_mj.body_ipos[bid]]
        mjcf_diag = [float(v) for v in model_mj.body_inertia[bid]]
        mjcf_iquat = [float(v) for v in model_mj.body_iquat[bid]]

        usd_mass = usd.get("mass")
        usd_com = usd.get("com")
        usd_diag = usd.get("diagonal_inertia")
        usd_axes = usd.get("principal_axes")
        inertia_source = str(usd.get("inertia_source", "absent"))

        # ---- Unauthored inertia → fail loud (§1.8) ----
        if inertia_source != "authored" or not usd_diag:
            if unauthored_inertia_policy == "ignore":
                ignored.append(IgnoredBody(
                    mjcf_body=mjcf_body, ir_role=ir_role,
                    reason="unauthored_acknowledged",
                ))
                notes.append(
                    f"{mjcf_body}: USD inertia unauthored — skipped by "
                    f"unauthored_inertia_policy=ignore (sim2sim unreliable for this body)"
                )
                continue
            raise InertiaCalibrationError(
                f"USD body {usd_body!r} (ir_role {ir_role!r}, MJCF body "
                f"{mjcf_body!r}) has no authored diagonalInertia. PhysX would "
                f"auto-compute this from geometry+density at simulation time, but "
                f"pxr cannot read the computed value. This creates a silent gap "
                f"between training and sim2sim.\nOptions:\n"
                f"  1. Author the inertia explicitly in the USD source (preferred)\n"
                f"  2. Run a PhysX-boot capture to extract the computed value "
                f"(future feature)\n"
                f"  3. Explicitly acknowledge the gap via "
                f"unauthored_inertia_policy='ignore' (sim2sim will be unreliable "
                f"for this body)."
            )

        # ---- Diffs ----
        mass_rel = (
            abs(float(usd_mass) - mjcf_mass) / max(abs(mjcf_mass), _EPS)
            if usd_mass is not None else 0.0
        )
        su = sorted(float(x) for x in usd_diag)
        sm = sorted(float(x) for x in mjcf_diag)
        inertia_rel = max(
            abs(a - b) / max(abs(b), _EPS) for a, b in zip(su, sm)
        )
        com_abs = (
            float(np.linalg.norm(np.asarray(usd_com, dtype=np.float64)
                                 - np.asarray(mjcf_com, dtype=np.float64)))
            if usd_com is not None else 0.0
        )
        diff = {
            "mass_rel": mass_rel,
            "inertia_rel": inertia_rel,
            "com_abs_m": com_abs,
        }

        # ---- Frame-alignment gate (non-identity-axes + invariant-disagree) ----
        # Identity-axes bodies (trunk) and invariant-agree bodies transfer with
        # no frame question. Otherwise prove frame equality from the kinematic
        # tree (D25); only a true kinematic *diverge* consults the policy.
        kin_status = "n/a"
        axes_identity = _quat_is_identity(usd_axes) and _quat_is_identity(mjcf_iquat)
        if not axes_identity and not _invariants_agree(usd_diag, mjcf_diag,
                                                        tol["inertia_relative"]):
            kin_status, kin_detail = _kinematic_frame_check(
                model_mj, mjcf_body, usd_body, usd_joint_frames,
            )
            diff.update(kin_detail)
            if kin_status == "aligned":
                notes.append(
                    f"{mjcf_body}: parent-joint frames match "
                    f"(origin_err={kin_detail.get('origin_err_m', 0.0)*1000:.3f}mm, "
                    f"axis={kin_detail.get('axis_deg', 0.0):.3f}°) → transfer is safe "
                    f"despite differing inertia magnitude."
                )
            else:
                # Cannot prove alignment (diverge / unknown) → policy decides.
                if frame_mismatch_policy == "transfer":
                    notes.append(
                        f"{mjcf_body}: kinematic frame check = {kin_status} "
                        f"(origin_err={kin_detail.get('origin_err_m', float('nan'))*1000:.3f}mm, "
                        f"axis={kin_detail.get('axis_deg', float('nan')):.3f}°) — "
                        f"forced transfer by frame_mismatch_policy=transfer."
                    )
                elif frame_mismatch_policy == "ignore":
                    ignored.append(IgnoredBody(
                        mjcf_body=mjcf_body, ir_role=ir_role,
                        reason="frame_mismatch_acknowledged", diff=diff,
                        kinematic_frame_check=kin_status,
                    ))
                    notes.append(
                        f"{mjcf_body}: kinematic frame check = {kin_status} — "
                        f"skipped by frame_mismatch_policy=ignore"
                    )
                    continue
                else:  # raise
                    raise InertiaCalibrationError(
                        f"USD body {usd_body!r} (MJCF body {mjcf_body!r}): inertia "
                        f"invariants disagree (sorted USD={su}, sorted MJCF={sm}) "
                        f"AND the kinematic parent-joint frame check = "
                        f"{kin_status!r} (origin_err="
                        f"{kin_detail.get('origin_err_m', float('nan'))*1000:.3f}mm, "
                        f"axis={kin_detail.get('axis_deg', float('nan')):.3f}°). The "
                        f"USD prim frame and MJCF body frame cannot be confirmed "
                        f"aligned, so transferring COM/principal-axes may be wrong. "
                        f"Pass frame_mismatch_policy='transfer' to force it (if you "
                        f"trust the frames) or 'ignore' to skip this body."
                    )

        # ---- Below-tolerance → audit-record + skip ----
        if (mass_rel < tol["mass_relative"]
                and inertia_rel < tol["inertia_relative"]
                and com_abs < tol["com_absolute_m"]):
            ignored.append(IgnoredBody(
                mjcf_body=mjcf_body, ir_role=ir_role,
                reason="below_tolerance", diff=diff,
            ))
            continue

        # ---- Physicality of the value we are about to write ----
        ok, why = _physicality_ok(usd_diag)
        if not ok:
            raise InertiaCalibrationError(
                f"USD body {usd_body!r} (MJCF body {mjcf_body!r}) inertia is "
                f"non-physical and would corrupt the model: {why}."
            )

        # ---- Build override (per-field-group) ----
        fields_applied: List[str] = []
        ov = BodyInertiaOverride(
            mjcf_body=mjcf_body, ir_role=ir_role, usd_body=usd_body,
            fields_applied=fields_applied, diff=diff,
            kinematic_frame_check=kin_status,
        )
        if usd_mass is not None and mass_rel >= tol["mass_relative"]:
            ov.mass_from, ov.mass_to = mjcf_mass, float(usd_mass)
            fields_applied.append("mass")
        if inertia_rel >= tol["inertia_relative"]:
            # diag + iquat are coupled: the loader writes both together.
            ov.diag_from, ov.diag_to = mjcf_diag, [float(x) for x in usd_diag]
            iquat_to = (
                [float(x) for x in usd_axes] if usd_axes is not None
                else [1.0, 0.0, 0.0, 0.0]
            )
            ov.iquat_from, ov.iquat_to = mjcf_iquat, iquat_to
            fields_applied.append("inertia")
        if usd_com is not None and com_abs >= tol["com_absolute_m"]:
            ov.com_from, ov.com_to = mjcf_com, [float(x) for x in usd_com]
            fields_applied.append("com")

        if fields_applied:
            overrides.append(ov)
        else:
            # Some field crossed tolerance in aggregate but each individual
            # group fell below its own threshold — record as ignored, audited.
            ignored.append(IgnoredBody(
                mjcf_body=mjcf_body, ir_role=ir_role,
                reason="below_tolerance", diff=diff,
            ))

    status = "calibrated" if overrides else "no_changes_needed"
    usd_sha = ""
    if usd_path and Path(usd_path).exists():
        usd_sha = hash_file_sha256(Path(usd_path))  # Nucleus refs stay empty

    result = MjcfInertiaOverlayResult(
        sku=sku, status=status, brand=brand, model=model,
        usd_ref=usd_ref, usd_sha256=usd_sha, usd_stage_units=stage_units,
        mjcf_path=str(mjcf_path), mjcf_sha256=hash_file_sha256(mjcf_path),
        diff_tolerance=tol,
        unauthored_inertia_policy=unauthored_inertia_policy,
        frame_mismatch_policy=frame_mismatch_policy,
        overrides=overrides, ignored=ignored, notes=notes,
        computed_at=_now_iso(),
    )
    log_info(
        f"[inertia_calibration] sku={sku!r} status={status} "
        f"overrides={len(overrides)} ignored={len(ignored)} "
        f"(tol mass={tol['mass_relative']} inertia={tol['inertia_relative']} "
        f"com={tol['com_absolute_m']})"
    )
    for n in notes:
        log_warning(f"[inertia_calibration] sku={sku!r}: {n}")
    return result


def overlay_rel_path(sku: str) -> str:
    """USER_CONFIG_DIR-relative path of the active overlay for ``sku``."""
    return f"{OVERLAY_REL_DIR}/{sku}.yaml"


def write_overlay(result: MjcfInertiaOverlayResult) -> str:
    """Persist *result* as the SKU's active YAML overlay + state.json pointer.

    Returns the USER_CONFIG_DIR-relative overlay path. Uses ``push_data`` (atomic
    write under USER_CONFIG_DIR) for the YAML and
    ``RobotAssetService.set_mjcf_inertia_overlay`` for the light pointer.
    """
    from unitport_sdk import push_data
    from application.service.robot_assets.service import get_robot_asset_service

    rel = overlay_rel_path(result.sku)
    if not push_data(rel, result.to_overlay_dict()):
        raise InertiaCalibrationError(
            f"write_overlay: failed to persist inertia overlay to {rel!r}"
        )
    get_robot_asset_service().set_mjcf_inertia_overlay(
        result.sku, result.to_pointer_dict(rel)
    )
    log_info(f"[inertia_calibration] wrote overlay {rel} (status={result.status})")
    return rel


# ---------------------------------------------------------------------------
# Debug export (DD-1 escape hatch) — standalone calibrated XML for mujoco viewer
# ---------------------------------------------------------------------------

_DEBUG_HEADER_TEMPLATE = (
    "<!-- ============================================================\n"
    "     DEBUG-ONLY — GENERATED, DO NOT COMMIT / DO NOT SHIP.\n"
    "     This XML is the original MJCF with the USD->MJCF inertia overlay\n"
    "     baked in, for standalone `mujoco viewer` inspection ONLY.\n"
    "     Robot SKU:   {sku}\n"
    "     Source MJCF: {mjcf}\n"
    "     Overlay:     {overlay}\n"
    "     Generated:   {ts}\n"
    "     The original MJCF + the YAML overlay remain the source of truth;\n"
    "     the runtime applies the overlay in-memory and never reads this file.\n"
    "     Mesh/asset refs resolve relative to the source MJCF directory.\n"
    "     ============================================================ -->\n"
)


def export_calibrated_xml(sku: str, output_path: Optional[str] = None) -> Path:
    """Write a standalone ``*_calibrated.debug.xml`` for ``sku`` (DD-1 escape hatch).

    Builds the ``MjModel`` through the exact runtime path
    (:meth:`MjActor.from_sku` → actuator-neutralize + inertia-overlay apply),
    serialises it with :func:`mujoco.mj_saveLastXML`, and prepends a
    DEBUG-ONLY header. The original MJCF is never touched. Default output is
    ``<source_dir>/<sku>.calibrated.debug.xml`` (so relative mesh refs resolve).

    Returns the written path. Raises :class:`InertiaCalibrationError` on failure.
    """
    import mujoco

    from application.service.runtime.simulation.mujoco.mj_actor import MjActor

    actor = MjActor.from_sku(sku)          # neutralize + _apply_inertia_overlay
    src_mjcf = Path(actor.mjcf_path)

    if output_path:
        out = Path(output_path)
        if not out.is_absolute():
            out = src_mjcf.parent / out
    else:
        out = src_mjcf.parent / f"{sku}.calibrated.debug.xml"

    try:
        mujoco.mj_saveLastXML(str(out), actor.mj_model)
    except Exception as exc:
        raise InertiaCalibrationError(
            f"export_calibrated_xml: mj_saveLastXML failed for sku {sku!r} → "
            f"{out}: {exc!r}"
        ) from exc

    # Prepend the DEBUG-ONLY header. This is a developer throwaway artifact (not
    # managed app state), so a direct text read/rewrite is appropriate here.
    try:
        pointer = None
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            pointer = get_robot_asset_service().get_mjcf_inertia_overlay(sku)
        except Exception:
            pointer = None
        header = _DEBUG_HEADER_TEMPLATE.format(
            sku=sku, mjcf=src_mjcf,
            overlay=(pointer or {}).get("file", "(none — model uncalibrated)"),
            ts=_now_iso(),
        )
        body = out.read_text(encoding="utf-8")
        if body.startswith("<?xml"):
            nl = body.find("\n") + 1
            body = body[:nl] + header + body[nl:]
        else:
            body = header + body
        out.write_text(body, encoding="utf-8")
    except OSError as exc:
        log_warning(
            f"[inertia_calibration] export_calibrated_xml: wrote {out} but could "
            f"not prepend DEBUG header: {exc}"
        )

    log_info(f"[inertia_calibration] exported DEBUG calibrated XML → {out}")
    return out


def _cli(argv: Optional[List[str]] = None) -> int:
    """``python -m application.training.validation.usd_mjcf_inertia_calibration
    <command> ...`` — developer entry point (DD-1 / DD-5).

    Commands:
      * ``calibrate <sku> [--tol-mass F] [--tol-inertia F] [--tol-com F]
        [--allow-unauthored] [--allow-frame-mismatch]`` — harvest USD inertia,
        diff vs MJCF, write the overlay.
      * ``export <sku> [--output PATH]`` — write the DEBUG calibrated XML.
    """
    import argparse
    import sys

    # Standalone-entry responsibility: the app loads the registry at startup,
    # but a bare ``python -m`` invocation must load it itself so the user-overlay
    # (robots_custom.json — where Dump USD writes bodies_per_format[USD]) is merged.
    try:
        from registers import RegistryHub
        RegistryHub.load_all()
    except Exception as exc:  # noqa: BLE001
        print(f"warning: RegistryHub.load_all() failed: {exc!r}", file=sys.stderr)

    p = argparse.ArgumentParser(prog="overlay")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="compute + write the USD->MJCF inertia overlay")
    pc.add_argument("sku")
    pc.add_argument("--tol-mass", type=float, default=None)
    pc.add_argument("--tol-inertia", type=float, default=None)
    pc.add_argument("--tol-com", type=float, default=None)
    pc.add_argument("--allow-unauthored", action="store_true",
                    help="unauthored_inertia_policy=ignore (accept the residual gap)")
    pc.add_argument("--frame-policy", choices=("raise", "ignore", "transfer"),
                    default=None,
                    help="policy for a kinematic-diverge body (default raise). "
                         "'transfer' forces transfer trusting the frames; 'ignore' "
                         "skips the body. Bodies the kinematic check proves aligned "
                         "transfer automatically regardless.")
    pc.add_argument("--allow-frame-mismatch", action="store_true",
                    help="deprecated alias for --frame-policy ignore")

    pe = sub.add_parser("export", help="write a DEBUG calibrated XML for mujoco viewer")
    pe.add_argument("sku")
    pe.add_argument("--output", default=None)

    args = p.parse_args(argv)

    if args.cmd == "calibrate":
        tol = {}
        if args.tol_mass is not None:
            tol["mass_relative"] = args.tol_mass
        if args.tol_inertia is not None:
            tol["inertia_relative"] = args.tol_inertia
        if args.tol_com is not None:
            tol["com_absolute_m"] = args.tol_com
        frame_policy = args.frame_policy or (
            "ignore" if args.allow_frame_mismatch else "raise"
        )
        result = calibrate_mjcf_inertia(
            args.sku,
            diff_tolerance=tol or None,
            unauthored_inertia_policy="ignore" if args.allow_unauthored else "raise",
            frame_mismatch_policy=frame_policy,
        )
        rel = write_overlay(result)
        print(f"status={result.status} overrides={len(result.overrides)} "
              f"ignored={len(result.ignored)} → {rel}")
        for ov in result.overrides:
            print(f"  override {ov.mjcf_body} ({ov.ir_role}): "
                  f"{ov.fields_applied}  frame={ov.kinematic_frame_check}")
        for ig in result.ignored:
            print(f"  ignored  {ig.mjcf_body} ({ig.ir_role}): {ig.reason}  "
                  f"frame={ig.kinematic_frame_check}")
        return 0

    if args.cmd == "export":
        out = export_calibrated_xml(args.sku, args.output)
        print(f"wrote DEBUG calibrated XML → {out}")
        return 0
    return 1


__all__ = [
    "OVERLAY_SCHEMA",
    "OVERLAY_REL_DIR",
    "DEFAULT_DIFF_TOLERANCE",
    "InertiaCalibrationError",
    "BodyInertiaOverride",
    "IgnoredBody",
    "MjcfInertiaOverlayResult",
    "calibrate_mjcf_inertia",
    "overlay_rel_path",
    "write_overlay",
    "export_calibrated_xml",
]


if __name__ == "__main__":
    import sys

    sys.exit(_cli())
