# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared pose-invariant hash of a robot's foot collision geometry.

ONE hasher, used by all four sim2sim foot-geometry callers:

  * the **producer** (``env_cfg_compiler`` stash at compile time),
  * the **export gate** (``sim2sim_calibration.run_calibration`` re-hash),
  * the **finalizer** (``bundle_finalizer`` compiler↔finalizer MJCF identity), and
  * the **runtime consumer** (``PolicyRunner.load`` at deploy).

Layer-1 "realized == designed" closes the foot-collision-geometry gap: the
robot the policy is *deployed* on must present byte-identical foot contact
surfaces (shape + contact mask) to the ones training *designed* against. A
silently-different foot geometry (wrong mesh, dropped collision flag, resized
capsule) changes the contact dynamics the policy learned on without any other
signal — exactly the kind of mute divergence the audit matrix exists to catch.

**Pose-invariance (CLAUDE.md §10/§11).** The hash reads ONLY static
``MjModel`` fields — ``geom_type`` / ``geom_size`` / ``geom_aabb`` (the geom's
LOCAL-frame axis-aligned bbox, compile-time, identical in every posture) and
the contact masks ``geom_contype`` / ``geom_conaffinity``. It NEVER touches
``MjData.geom_xpos`` / ``geom_xmat`` (world frame, ``qpos``-dependent), so the
same robot in any pose hashes identically. A resident deploy-side self-check
built on this therefore cannot false-positive on posture — only on a genuine
geometry change.

**Pure** like ``inertia_overlay_apply``: takes an already-parsed ``MjModel``
plus the foot IR roles to hash and an IR-role→MJCF-body-name map (the caller
resolves it from the registry). Imports NO service / training layer, so the
runtime ``policy`` package can depend on it without reaching back into
``training``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence


FOOT_GEOMETRY_SCHEMA = "unitport.foot_geometry/v1"


class FootGeometryError(RuntimeError):
    """A legged robot whose declared foot landmarks resolve to NO
    collision-eligible geometry. That is a real asset gap (missing foot
    contact geoms), never silently defaulted to an empty hash (§8)."""


def _canonical_geom(mj_model: Any, gid: int) -> list:
    """Pose-invariant identity of one geom.

    ``geom_aabb`` is the geom's LOCAL-frame bbox (``MjModel`` static field —
    NOT the world AABB derived from ``MjData.geom_xpos``/``geom_xmat``), so it
    does not change with ``qpos``. For mesh geoms mujoco precomputes it from
    the vertex extents, giving real discriminating power where ``geom_size``
    (often the bare mesh scale) alone would not.
    """
    gt = int(mj_model.geom_type[gid])
    size = [round(float(v), 6) for v in mj_model.geom_size[gid]]
    aabb = [round(float(v), 6) for v in mj_model.geom_aabb[gid]]  # LOCAL, pose-invariant
    ct = int(mj_model.geom_contype[gid])
    ca = int(mj_model.geom_conaffinity[gid])
    return [gt, size, aabb, ct, ca]


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def hash_foot_geometry(
    mj_model: Any,
    *,
    foot_roles: Sequence[str],
    ir_to_body_name: Mapping[str, str],
) -> Dict[str, Any]:
    """Hash the collision geometry of a robot's foot bodies, pose-invariantly.

    Parameters
    ----------
    mj_model:
        An already-parsed ``mujoco.MjModel`` (typed ``Any`` — no hard import).
    foot_roles:
        The ground-contact foot/wheel IR roles to hash. Use
        ``mjcf_base_calibration.canonical_foot_roles(family)`` to derive these
        (calf/knee are spawn-height landmarks, NOT contact feet, and are
        excluded). An empty sequence means the family has no feet by design
        (manipulator) → ``status="not_applicable"``.
    ir_to_body_name:
        IR-role → MJCF body-name map (caller resolves from the registry's
        ``bodies_per_format.MJCF``). Roles absent from this map are skipped
        (the SKU simply doesn't declare that foot).

    Returns
    -------
    dict with ``schema`` and:
      * ``status="not_applicable"`` — no foot roles (manipulator). This is
        DISTINCT from an old bundle that lacks the field entirely: a present
        ``not_applicable`` block means "no feet by design, audited & cleared",
        an ABSENT block means "predates the foot-geometry audit" (migration
        WARN). Never conflate the two.
      * ``status="ok"`` — ``per_role`` (role → sha256 of its sorted collision
        geoms) and ``aggregate`` (sha256 over the sorted ``per_role`` items).

    Raises
    ------
    FootGeometryError:
        Foot roles were declared but NONE resolved to a body with
        collision-eligible geometry — a legged robot must have foot contact
        geoms; refusing to emit an empty hash (§8 — no silent default).
    """
    import mujoco

    if not foot_roles:
        return {
            "schema": FOOT_GEOMETRY_SCHEMA,
            "status": "not_applicable",
            "per_role": {},
            "aggregate": "",
            "notes": [
                "family declares no foot landmarks (fixed-base / by design); "
                "no foot-geometry audit applicable"
            ],
        }

    per_role: Dict[str, str] = {}
    missing: List[str] = []
    for role in foot_roles:
        bname = ir_to_body_name.get(role)
        if not bname:
            continue  # role not declared for this SKU — skip (matches calibration)
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, str(bname))
        if bid < 0:
            missing.append(f"{role} ({bname!r} not in MJCF)")
            continue
        start = int(mj_model.body_geomadr[bid])
        count = int(mj_model.body_geomnum[bid])
        geoms: List[list] = []
        for gid in range(start, start + count):
            # Collision-eligible: contype != 0 OR conaffinity != 0. Visual-only
            # geoms (both flags 0) are not contact surfaces — including them
            # would make the hash sensitive to cosmetic mesh changes that don't
            # affect contact dynamics. Same filter as mjcf_base_calibration.
            ct = int(mj_model.geom_contype[gid])
            ca = int(mj_model.geom_conaffinity[gid])
            if ct == 0 and ca == 0:
                continue
            geoms.append(_canonical_geom(mj_model, gid))
        if not geoms:
            missing.append(f"{role} ({bname!r} has no collision geoms)")
            continue
        # Geom order within a body must not affect the hash.
        geoms.sort(key=lambda g: json.dumps(g, sort_keys=True))
        per_role[role] = _hash_obj(geoms)

    if not per_role:
        raise FootGeometryError(
            "hash_foot_geometry: none of the declared foot roles "
            f"{list(foot_roles)} resolved to a body with collision geometry "
            f"(issues: {missing}). A legged robot must have foot contact geoms "
            "— refusing to emit an empty foot-geometry hash (§8). Fix "
            "bodies_per_format.MJCF and/or the foot collision geoms in the asset."
        )

    out: Dict[str, Any] = {
        "schema": FOOT_GEOMETRY_SCHEMA,
        "status": "ok",
        "per_role": per_role,
        "aggregate": _hash_obj(sorted(per_role.items())),
    }
    if missing:
        out["notes"] = [
            f"declared foot roles skipped (no collision geom): {missing}"
        ]
    return out


__all__ = ["hash_foot_geometry", "FootGeometryError", "FOOT_GEOMETRY_SCHEMA"]
