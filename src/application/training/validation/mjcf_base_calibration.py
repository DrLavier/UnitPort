# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MJCF base spawn-height calibration.

Computes a per-SKU vertical offset that compensates for the **USD↔MJCF base
frame anchor mismatch**: the USD articulation root and the MJCF
``<body name="base">`` are anchored at different points on the trunk, so the
same numeric ``init_pos_z`` (authored against USD/IsaacLab) renders the
visible trunk at a different height in MuJoCo and the robot spawns below
ground (穿地).

Approach — multi-landmark IR-role assessment at standing pose:
    1. Load the MJCF, write the standing pose (IR-keyed joint angles from the
       registry blueprint) into qpos, plant the free joint at
       ``(0, 0, reference_init_z)`` with identity orientation, run
       :func:`mujoco.mj_forward`.
    2. For each family-declared landmark IR role (foot/calf/knee/wheel per
       family), find the corresponding body in MJCF via
       ``RobotSpec.bodies_role_map_for("MJCF")`` and enumerate collision-
       eligible geoms attached to it.
    3. For each geom, compute the **world-frame lowest z** using a true
       8-corner AABB transform (geom_aabb × geom_xmat + geom_xpos), so we
       get the actual contact-surface bottom — not just the body frame z.
    4. ``offset_z = max(0.0, -min(all_landmark_z))``. Clamped to ≥ 0: the
       calibration may *lift* but never *sink* the trunk below the user's
       declared ``init_pos_z``.

CLAUDE.md §1.8 (no silent fallback) is enforced:
    - MJCF asset missing → :class:`FileNotFoundError`.
    - ``default_init_joint_angles`` missing in the registry blueprint AND no
      caller-supplied pose → :class:`ValueError` with a directive to fill
      ``registers/data/robots_canonical.json``.
    - Landmark IR roles declared by the family but none present in
      ``bodies_per_format.MJCF`` → :class:`ValueError`. Manipulator family
      (which declares no landmarks by design) returns a
      ``status='not_applicable'`` :class:`MjcfBaseOffsetResult` instead.

Calibration is the **sim-preview carve-out** mentioned in the canvas-IR-only
memory: it does touch the MJCF file directly, but it does so to compute an
overlay value, not to mutate the asset. The overlay lives under
``USER_CONFIG_DIR/robot_assets/state.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitport_sdk import log_info, log_warning


# ---------------------------------------------------------------------------
# Family → landmark IR-role priority set
# ---------------------------------------------------------------------------
# A body counts toward the assessment iff its IR role is declared in this
# SKU's ``bodies_per_format.MJCF``. Un-declared roles are silently skipped;
# if the entire set is empty for a non-manipulator family, the calibration
# raises (§1.8).  Manipulator is the only family that legitimately has
# *zero* landmarks — fixed-base, no spawn-z drift problem — and we
# explicitly emit ``status='not_applicable'`` for it.
LANDMARK_IR_ROLES_PER_FAMILY: Dict[str, Tuple[str, ...]] = {
    "quadruped": (
        "foot_FL", "foot_FR", "foot_RL", "foot_RR",
        "calf_FL", "calf_FR", "calf_RL", "calf_RR",
        "wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR",
    ),
    "biped": (
        "foot_L", "foot_R",
        "knee_L", "knee_R",
        "calf_L", "calf_R",
    ),
    "humanoid": (  # alias of biped per ir_canonical
        "foot_L", "foot_R",
        "knee_L", "knee_R",
        "calf_L", "calf_R",
    ),
    "wheeled": (
        "wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR",
    ),
    "manipulator": (),  # by design: no spawn-z drift, no calibration
}

# Asymmetry tolerance — if the per-role-prefix spread (e.g. all four
# ``foot_*``) exceeds this, a WARN-level note is appended; calibration
# proceeds.
_ASYMMETRY_TOLERANCE_M = 0.02

# Reference base z used during calibration. Any value > 0 works for
# offset computation; the chosen value is recorded in the overlay so
# downstream consumers know what posture the offset was sampled at.
_DEFAULT_REFERENCE_INIT_Z = 0.40


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LandmarkSample:
    """Per-landmark Z reading at standing pose.

    ``min_geom_z`` is the lowest collision-geom point in world frame after
    the trunk has been placed at ``reference_init_z``. ``body_origin_z`` is
    the body frame z for cross-check (handy for spotting MJCF authors that
    anchored a body at the geometric top vs. the bottom).
    """

    ir_role: str
    body_name: str
    min_geom_z: float
    body_origin_z: float


@dataclass
class MjcfBaseOffsetResult:
    """Output of :func:`calibrate_mjcf_base_offset`.

    Two terminal states:
        * ``status='calibrated'``: ``offset_z`` is the lift to add to qpos[base_z].
        * ``status='not_applicable'``: family has no landmark set (e.g. manipulator);
          runtime readers treat this as "no warn, no correction".
    """

    sku: str
    status: str                       # "calibrated" | "not_applicable"
    offset_z: float
    method: str                       # "multi_landmark_ir"
    family: str
    mjcf_sha256: str
    standing_pose_sha256: str
    standing_pose_source: str         # "caller" | "registry_default_init_joint_angles"
    reference_init_z: float
    landmark_z: List[LandmarkSample] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    computed_at: str = ""

    def to_overlay_dict(self) -> Dict[str, Any]:
        """Serialise into the shape stored under ``assets[sku]['mjcf_base_height_offset']``."""
        return {
            "schema": "unitport.mjcf_base_offset/v1",
            "status": self.status,
            "offset_z": float(self.offset_z),
            "calibration": {
                "method": self.method,
                "family": self.family,
                "mjcf_sha256": self.mjcf_sha256,
                "standing_pose_source": self.standing_pose_source,
                "standing_pose_sha256": self.standing_pose_sha256,
                "reference_init_z": float(self.reference_init_z),
                "landmark_z": [
                    {
                        "ir_role": s.ir_role,
                        "body_name": s.body_name,
                        "min_geom_z": float(s.min_geom_z),
                        "body_origin_z": float(s.body_origin_z),
                    }
                    for s in self.landmark_z
                ],
                "notes": list(self.notes),
                "computed_at": self.computed_at,
            },
        }

    @classmethod
    def not_applicable(cls, sku: str, family: str, reason: str) -> "MjcfBaseOffsetResult":
        """Manipulator (or any family with an empty landmark set by design)."""
        return cls(
            sku=sku,
            status="not_applicable",
            offset_z=0.0,
            method="multi_landmark_ir",
            family=family,
            mjcf_sha256="",
            standing_pose_sha256="",
            standing_pose_source="",
            reference_init_z=0.0,
            landmark_z=[],
            notes=[reason],
            computed_at=_now_iso(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_file_sha256(path: Path) -> str:
    """SHA-256 of an on-disk file. Streams in 64 KiB chunks to handle big mesh-embedded MJCFs."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_standing_pose(pose_ir: Dict[str, float]) -> str:
    """SHA-256 of an IR-keyed joint-angle dict. Sorted keys + fixed float repr so
    re-runs with the same logical pose yield the same hash."""
    canonical = {str(k): float(v) for k, v in sorted(pose_ir.items())}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _geom_lowest_world_z(model, data, gid: int) -> float:
    """Lowest world-Z point on geom ``gid`` after current ``mj_forward``.

    Uses a true 8-corner AABB transform via ``geom_aabb`` (axis-aligned bbox
    in the geom's local frame: 6 floats — cx, cy, cz, ex, ey, ez) rotated by
    ``geom_xmat`` and translated by ``geom_xpos``. This is accurate for any
    geom shape mujoco supports (sphere/capsule/cylinder/box/ellipsoid/mesh) —
    mujoco precomputes ``geom_aabb`` even for meshes from their vertex
    extents.
    """
    aabb = model.geom_aabb[gid]  # (6,)
    cx, cy, cz, ex, ey, ez = (float(v) for v in aabb)
    corners_local = np.array([
        [cx + ex, cy + ey, cz + ez],
        [cx + ex, cy + ey, cz - ez],
        [cx + ex, cy - ey, cz + ez],
        [cx + ex, cy - ey, cz - ez],
        [cx - ex, cy + ey, cz + ez],
        [cx - ex, cy + ey, cz - ez],
        [cx - ex, cy - ey, cz + ez],
        [cx - ex, cy - ey, cz - ez],
    ], dtype=np.float64)
    rmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    world = corners_local @ rmat.T + pos
    return float(world[:, 2].min())


def _body_lowest_world_z(model, data, body_id: int) -> Optional[float]:
    """Lowest collision-eligible geom z for a body. Returns None if the body
    has no collision-eligible geoms (the user must declare collision
    geometry for the landmark body — fallback to visual-only geoms would
    mask MJCFs missing real contacts)."""
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    if count <= 0:
        return None
    lowest: Optional[float] = None
    for gid in range(start, start + count):
        # Collision-eligible: contype != 0 OR conaffinity != 0. Visual-only
        # geoms (both flags 0) are intentionally skipped — they don't
        # represent contact surfaces and including them would
        # over-correct.
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        z = _geom_lowest_world_z(model, data, gid)
        if lowest is None or z < lowest:
            lowest = z
    return lowest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def primary_family(families: List[str]) -> str:
    """Pick the family used for landmark lookup. Robots declare multiple
    families (humanoid + biped, etc.); we take the first declared family
    that has a landmark set in :data:`LANDMARK_IR_ROLES_PER_FAMILY`. If
    none match, return the first family verbatim so the error message
    points at concrete catalog data."""
    for f in families:
        if str(f).lower() in LANDMARK_IR_ROLES_PER_FAMILY:
            return str(f).lower()
    return str(families[0]).lower() if families else ""


def calibrate_mjcf_base_offset(
    sku: str,
    *,
    reference_init_z: float = _DEFAULT_REFERENCE_INIT_Z,
    standing_pose_ir: Optional[Dict[str, float]] = None,
) -> MjcfBaseOffsetResult:
    """Compute the MJCF spawn-z offset for one SKU.

    Args:
        sku: canonical SKU (from ``registers.robots``).
        reference_init_z: trunk-z used while sampling landmark heights.
            Any positive value works for offset computation; the value is
            recorded in the overlay so callers know the sampling posture.
        standing_pose_ir: optional caller-supplied IR-role → radians map
            (e.g. the canvas's ActorSetting init_joint_angles). If omitted,
            ``RobotSpec.default_init_joint_angles`` from the registry
            blueprint is used. If neither is available the call raises
            (§1.8 — no silent zero-pose fallback).

    Raises:
        FileNotFoundError: MJCF asset not on disk for ``sku``.
        ValueError: missing IR-keyed standing pose, missing landmark
            bodies for the family, or empty/malformed registry data.
    """
    import mujoco

    from application.service.robot_assets.service import (
        get_robot_asset_service,
    )
    from registers import robots as _robots_registry

    entry = _robots_registry.get_robot(sku)
    if not entry:
        raise ValueError(
            f"calibrate_mjcf_base_offset: sku {sku!r} is not registered in "
            f"registers.robots. The caller is expected to resolve the SKU "
            f"via resolve_to_sku() before invoking calibration."
        )
    families = list(entry.get("families", []) or [])
    family = primary_family(families)

    # ---- Manipulator (or any family with an empty landmark set) ----
    if family in LANDMARK_IR_ROLES_PER_FAMILY:
        landmark_roles = LANDMARK_IR_ROLES_PER_FAMILY[family]
        if not landmark_roles:
            log_info(
                f"[mjcf_base_calibration] sku={sku!r} family={family!r} has "
                f"no landmark set by design; emitting not_applicable overlay."
            )
            return MjcfBaseOffsetResult.not_applicable(
                sku,
                family,
                reason=(
                    f"family={family!r} is fixed-base (no spawn-z drift); "
                    f"no calibration needed."
                ),
            )
    else:
        raise ValueError(
            f"calibrate_mjcf_base_offset: family {family!r} (from sku "
            f"{sku!r}, declared families={families}) is not in the landmark "
            f"table. Update LANDMARK_IR_ROLES_PER_FAMILY in "
            f"application/training/validation/mjcf_base_calibration.py with "
            f"the new family's foot/knee/contact IR roles."
        )

    # ---- Resolve MJCF asset on disk via RobotAssetService.resolve() ----
    svc = get_robot_asset_service()
    asset = svc.resolve(sku)
    mjcf_path: Optional[Path] = getattr(asset, "mjcf_path", None) if asset else None
    if mjcf_path is None or not Path(mjcf_path).exists():
        raise FileNotFoundError(
            f"calibrate_mjcf_base_offset: sku {sku!r} has no resolvable "
            f"MJCF on disk (got {mjcf_path!r}). Calibration cannot run "
            f"without the live MJCF — register an MJCF path on the Robot "
            f"Asset card or wait for menagerie sync to land the file."
        )
    mjcf_path = Path(mjcf_path)

    # ---- Resolve standing pose (caller or registry blueprint) ----
    pose_ir: Dict[str, float] = {}
    pose_source = ""
    if standing_pose_ir:
        pose_ir = {str(k): float(v) for k, v in standing_pose_ir.items()}
        pose_source = "caller"
    else:
        # registers.robots returns dict-shaped entries — pull the blueprint
        # directly from the entry (CLAUDE.md §1.10: this is the canonical
        # standing pose, hand-authored per SKU in robots_canonical.json).
        blueprint = entry.get("default_init_joint_angles") or {}
        if not blueprint:
            raise ValueError(
                f"calibrate_mjcf_base_offset: sku {sku!r} has no "
                f"default_init_joint_angles in the registry blueprint and "
                f"no caller-supplied standing_pose_ir. Add the blueprint to "
                f"registers/data/robots_canonical.json (IR-role → radians) "
                f"so calibration runs against the standing pose instead of "
                f"the zero pose (which would silently produce a wrong "
                f"offset). §1.8 — no silent zero-pose fallback."
            )
        pose_ir = {str(k): float(v) for k, v in blueprint.items()}
        pose_source = "registry_default_init_joint_angles"

    # ---- Load MJCF + plant trunk + write standing pose into qpos ----
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # IR-role → MJCF joint name lookup. Use the per-format roles map from
    # the registry — canvas-IR-only stays intact (we never parse the MJCF
    # for joint names ourselves).
    joints_mjcf = entry.get("joints_per_format", {}).get("MJCF", {}) or {}
    if not joints_mjcf:
        raise ValueError(
            f"calibrate_mjcf_base_offset: sku {sku!r} has no joints_per_format[MJCF] "
            f"in the registry. Cannot map IR-role standing pose to MJCF "
            f"joint names. Fill it via the Robot Asset card's 'Dump MJCF' "
            f"button or by hand-editing the registry."
        )
    ir_to_joint_name: Dict[str, str] = {}
    for jspec in joints_mjcf.values():
        if isinstance(jspec, dict):
            nm = str(jspec.get("name", ""))
            ir = str(jspec.get("ir_role", ""))
            if nm and ir:
                ir_to_joint_name[ir] = nm

    # Locate the free joint so we plant the trunk in world correctly.
    free_qposadr: Optional[int] = None
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_qposadr = int(model.jnt_qposadr[jid])
            break
    # Start from qpos0 (model defaults), then patch.
    data.qpos[:] = model.qpos0
    if free_qposadr is not None:
        data.qpos[free_qposadr + 0] = 0.0
        data.qpos[free_qposadr + 1] = 0.0
        data.qpos[free_qposadr + 2] = float(reference_init_z)
        data.qpos[free_qposadr + 3] = 1.0
        data.qpos[free_qposadr + 4] = 0.0
        data.qpos[free_qposadr + 5] = 0.0
        data.qpos[free_qposadr + 6] = 0.0
    # Apply IR-keyed standing pose to actuated joints.
    missing_ir_in_mjcf: List[str] = []
    for ir, ang in pose_ir.items():
        jname = ir_to_joint_name.get(ir)
        if not jname:
            missing_ir_in_mjcf.append(ir)
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            missing_ir_in_mjcf.append(ir)
            continue
        # Only single-DOF hinge/slide can take a scalar angle.
        jtype = int(model.jnt_type[jid])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(ang)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ---- Sample landmarks ----
    bodies_mjcf = entry.get("bodies_per_format", {}).get("MJCF", {}) or {}
    ir_to_body_name: Dict[str, str] = {}
    for bspec in bodies_mjcf.values():
        if isinstance(bspec, dict):
            nm = str(bspec.get("name", ""))
            ir = str(bspec.get("ir_role", ""))
            if nm and ir:
                ir_to_body_name[ir] = nm

    samples: List[LandmarkSample] = []
    declared_but_no_geom: List[str] = []
    for ir in landmark_roles:
        bname = ir_to_body_name.get(ir)
        if not bname:
            continue  # role not declared for this SKU — silently skip per design
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            declared_but_no_geom.append(f"{ir} ({bname!r} not in MJCF)")
            continue
        min_z = _body_lowest_world_z(model, data, bid)
        if min_z is None:
            declared_but_no_geom.append(f"{ir} ({bname!r} has no collision geoms)")
            continue
        body_origin_z = float(data.xpos[bid, 2])
        samples.append(LandmarkSample(
            ir_role=ir,
            body_name=bname,
            min_geom_z=min_z,
            body_origin_z=body_origin_z,
        ))

    if not samples:
        raise ValueError(
            f"calibrate_mjcf_base_offset: sku {sku!r} family={family!r} has "
            f"no landmark bodies with collision geometry. Expected IR "
            f"roles: {list(landmark_roles)}. Issues: {declared_but_no_geom}. "
            f"Fill bodies_per_format.MJCF in the registry (or via the Robot "
            f"Asset card's 'Dump MJCF' button) and ensure the landmark "
            f"bodies declare collision geoms — without them MuJoCo has "
            f"nothing to contact the ground with. §1.8 — no silent zero "
            f"offset."
        )

    min_landmark_z = min(s.min_geom_z for s in samples)
    offset_z = max(0.0, -min_landmark_z)

    # ---- Asymmetry sanity check ----
    notes: List[str] = []
    if missing_ir_in_mjcf:
        notes.append(
            f"standing pose contains IR roles not mapped in joints_per_format.MJCF: "
            f"{sorted(set(missing_ir_in_mjcf))}"
        )
    if declared_but_no_geom:
        notes.append(
            f"declared landmark roles without collision geoms (skipped): {declared_but_no_geom}"
        )
    # Per-prefix spread (foot_*, calf_*, knee_*, wheel_*) — flag asymmetry.
    by_prefix: Dict[str, List[float]] = {}
    for s in samples:
        prefix = s.ir_role.split("_", 1)[0]
        by_prefix.setdefault(prefix, []).append(s.min_geom_z)
    for prefix, zs in by_prefix.items():
        if len(zs) >= 2:
            spread = max(zs) - min(zs)
            if spread > _ASYMMETRY_TOLERANCE_M:
                notes.append(
                    f"asymmetric {prefix}_* heights: spread={spread:.4f} m (max-min over "
                    f"{len(zs)} landmarks). Check MJCF mesh anchors or the standing "
                    f"pose; calibration proceeds against the lowest landmark."
                )

    result = MjcfBaseOffsetResult(
        sku=sku,
        status="calibrated",
        offset_z=offset_z,
        method="multi_landmark_ir",
        family=family,
        mjcf_sha256=hash_file_sha256(mjcf_path),
        standing_pose_sha256=hash_standing_pose(pose_ir),
        standing_pose_source=pose_source,
        reference_init_z=float(reference_init_z),
        landmark_z=samples,
        notes=notes,
        computed_at=_now_iso(),
    )
    log_info(
        f"[mjcf_base_calibration] sku={sku!r} family={family!r} "
        f"offset_z={offset_z:.4f} m (min landmark z={min_landmark_z:.4f}) "
        f"landmarks={len(samples)} pose_source={pose_source}"
    )
    for note in notes:
        log_warning(f"[mjcf_base_calibration] sku={sku!r}: {note}")
    return result


__all__ = [
    "LANDMARK_IR_ROLES_PER_FAMILY",
    "LandmarkSample",
    "MjcfBaseOffsetResult",
    "calibrate_mjcf_base_offset",
    "hash_file_sha256",
    "hash_standing_pose",
    "primary_family",
]
