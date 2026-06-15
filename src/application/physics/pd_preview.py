# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Live PD-preview computation for the RobotNode pd_param_table panel.

Single source for "what per-joint gains will actually be submitted" so the
UI panel and any future caller share one wiring. Given a robot SKU and the
canvas ``pd_groups`` overrides, it builds the canonical :class:`PDParam`
((omega_n, zeta) per group, §10), loads the robot's MJCF, mass-weights the
gains via :func:`application.physics.mujoco_gain_solver.solve`
(``kp = m_eff·omega_n²``), and reports per-joint ``kp``/``kd`` plus the
*effective* effort ceiling (the remotized-table peak for linkage-driven
joints, the scalar cap otherwise).

NO raw kp/kd ever flows in — the panel displays these as read-only derived
values; the editable knobs upstream are (omega_n, zeta) + effort/velocity.
Both engines (PhysX training / MuJoCo deploy) mass-weight off the same
``m_eff`` (MJCF nominal stance), so these MuJoCo-derived numbers equal the
PhysX-side gains by construction (§10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PDPreviewError(RuntimeError):
    """Raised when the preview cannot be computed (fail-loud, §1.8)."""


@dataclass
class PDPreviewGroup:
    """One editable PD group's current (omega_n, zeta)."""

    group_id: str
    omega_n: float
    zeta: float


@dataclass
class PDPreviewJoint:
    """One actuated joint's derived gains + effective effort ceiling."""

    ir_role: str
    physical: str
    group_id: str          # "" if no PD group matched this role
    omega_n: float
    zeta: float
    kp: float              # derived, read-only (kp = m_eff·omega_n²)
    kd: float              # derived, read-only (kd = 2ζ·sqrt(kp·m_eff))
    remotized: bool
    effort_effective: float   # remotized table peak if remotized else scalar cap


@dataclass
class PDPreview:
    """Full preview payload for the panel."""

    family: str
    mjcf_path: str
    effort_limit: float
    velocity_limit: float
    groups: List[PDPreviewGroup] = field(default_factory=list)
    joints: List[PDPreviewJoint] = field(default_factory=list)


def compute_pd_preview(
    sku: str,
    *,
    pd_groups_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    effort_limit: float,
    velocity_limit: float,
) -> PDPreview:
    """Compute the per-joint PD preview for ``sku``.

    Args:
        sku: canonical robot SKU (the unique DB key, §7).
        pd_groups_overrides: the canvas ``pd_groups`` JSON
            (``{group_id: {"omega_n": .., "zeta": ..}}``); empty/None = family
            defaults only.
        effort_limit / velocity_limit: the RobotNode scalar caps.

    Raises:
        PDPreviewError: SKU unknown, no family, no MJCF joint table, MJCF file
            missing, or the solver fails. Every failure is surfaced (the panel
            shows it) rather than substituting a default.
    """
    from registers.robots import get_robot, is_actuated_ir_role
    from registers import brands as _brands_reg
    from registers.pd_groups import find_group_for_role

    entry = get_robot(sku)
    if not entry:
        raise PDPreviewError(
            f"robot sku {sku!r} is not registered — cannot preview PD gains."
        )
    families = list(entry.get("families", []) or [])
    if not families:
        raise PDPreviewError(
            f"robot sku {sku!r} declares no families — cannot resolve "
            f"(omega_n, zeta) group defaults. Complete the Robot Asset card's "
            f"family selection."
        )
    primary_family = str(families[0])

    # MJCF joint table — the solver loads the MJCF and looks up joints by
    # name, so physical names MUST be MJCF names.
    per_format = entry.get("joints_per_format", {}) or {}
    mjcf_block = per_format.get("MJCF") if isinstance(per_format, dict) else None
    if not isinstance(mjcf_block, dict) or not mjcf_block:
        raise PDPreviewError(
            f"robot sku {sku!r} has no joints_per_format['MJCF'] — run "
            f"'Dump MJCF' in the Robot Asset card. The mass-weighted PD "
            f"solver needs the MJCF joint table."
        )
    # Only actuated joints get PD gains. The free-floating base joint
    # (ir_role 'base'), sensor mounts, misc and out-of-scope roles are NOT
    # actuated — feeding them to the mass-weighted solver makes it fail-loud
    # ("does not cover joint role 'base'"). Use the registry's single
    # authoritative predicate (the same one the resolver / RobotSpecRef /
    # deploy-coverage use) rather than a narrower local copy.
    ir_roles: List[str] = []
    physical: List[str] = []
    seen: set = set()
    for jspec in mjcf_block.values():
        if not isinstance(jspec, dict):
            continue
        ir = str(jspec.get("ir_role", "") or "")
        name = str(jspec.get("name", "") or "")
        if not ir or not name or not is_actuated_ir_role(ir) or ir in seen:
            continue
        ir_roles.append(ir)
        physical.append(name)
        seen.add(ir)
    if not ir_roles:
        raise PDPreviewError(
            f"robot sku {sku!r} MJCF joint table has no actuated joints with "
            f"an IR role — registry data is incomplete."
        )

    # Canonical (omega_n, zeta) → mass-weighted gains.
    from application.physics.pd_param import PDParam
    from application.physics.mujoco_gain_solver import solve as solve_mujoco
    from application.service.robot_assets.service import get_robot_asset_service

    try:
        pd_param = PDParam.from_family_defaults(primary_family)
        if pd_groups_overrides:
            pd_param = pd_param.with_overrides(pd_groups_overrides)
    except Exception as exc:  # noqa: BLE001
        raise PDPreviewError(f"PD parameterization failed: {exc}") from exc

    asset = get_robot_asset_service().resolve(sku)
    mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
    if mjcf_path is None or not Path(mjcf_path).is_file():
        raise PDPreviewError(
            f"robot sku {sku!r} has no on-disk MJCF "
            f"(mjcf_path={str(mjcf_path) if mjcf_path else None!r}). Run "
            f"'Dump MJCF' in the Robot Asset card before previewing PD gains."
        )

    try:
        gains = solve_mujoco(
            mjcf_path=Path(mjcf_path),
            joint_order_physical=physical,
            joint_ir_roles=ir_roles,
            nominal_qpos=None,  # MJCF keyframe-0 / qpos0 — matches the compiler
            pd_param=pd_param,
        )
    except Exception as exc:  # noqa: BLE001
        raise PDPreviewError(f"mass-weighted gain solve failed: {exc}") from exc

    # Remotized-joint effort ceilings (per linkage-driven joint, table peak).
    peak_by_physical: Dict[str, float] = {}
    brand = str(entry.get("brand", "") or "")
    model = str(entry.get("model", "") or "")
    manifest = _brands_reg.remotized_manifest(brand, model) if brand and model else None
    if manifest and manifest.get("remotized_joints"):
        from application.training.isaac_lab.remotized_emit import (
            match_remotized_groups,
        )
        from application.physics.actuators.torque_lookup import TorqueLookupTable

        groups_rem = match_remotized_groups(
            physical,
            manifest["remotized_joints"],
            lambda p: TorqueLookupTable.from_yaml(Path(p)),
        )
        for g in groups_rem:
            peak = float(g.table.peak_torque)
            for n in g.joint_names:
                peak_by_physical[n] = peak

    # Assemble rows.
    kp_by_role = dict(zip(gains.joint_ir_roles, gains.kp))
    kd_by_role = dict(zip(gains.joint_ir_roles, gains.kd))
    seen_groups: Dict[str, PDPreviewGroup] = {}
    joints: List[PDPreviewJoint] = []
    for ir, phys in zip(ir_roles, physical):
        match = find_group_for_role(primary_family, ir)
        gid = match[0] if match else ""
        if gid and gid in pd_param.groups:
            grp = pd_param.groups[gid]
            omega_n, zeta = float(grp.omega_n), float(grp.zeta)
            seen_groups.setdefault(
                gid, PDPreviewGroup(group_id=gid, omega_n=omega_n, zeta=zeta)
            )
        else:
            omega_n, zeta = 0.0, 0.0
        remotized = phys in peak_by_physical
        effort_eff = peak_by_physical[phys] if remotized else float(effort_limit)
        joints.append(PDPreviewJoint(
            ir_role=ir,
            physical=phys,
            group_id=gid,
            omega_n=omega_n,
            zeta=zeta,
            kp=float(kp_by_role.get(ir, 0.0)),
            kd=float(kd_by_role.get(ir, 0.0)),
            remotized=remotized,
            effort_effective=effort_eff,
        ))

    return PDPreview(
        family=primary_family,
        mjcf_path=str(mjcf_path),
        effort_limit=float(effort_limit),
        velocity_limit=float(velocity_limit),
        groups=list(seen_groups.values()),
        joints=joints,
    )


# ----------------------------------------------------------------------
# Recommended-actuators resolver (the AUTO button + connect-time reconcile)
# ----------------------------------------------------------------------


@dataclass
class RecommendedActuators:
    """Official recommended PD/actuator preset for a robot.

    ``groups`` maps PD group id → (omega_n, zeta). ``effort_limit`` /
    ``velocity_limit`` are ``None`` when no source could supply them (the AUTO
    button then leaves those spinboxes untouched). ``source_pd`` /
    ``source_caps`` and ``notes`` describe where each part came from, for the
    panel's status line.

    ``status`` is one of:
      * ``"ok"`` — populated by some recommendation source.
      * ``"need_dump_usd"`` — robot has a USD asset declared but no cached
        ``physics_per_format.USD`` block; AUTO should prompt the user to
        re-run Dump USD instead of silently falling through to family.
      * ``"no_usd_family_fallback"`` — robot has no USD asset at all; AUTO
        used the family defaults.
    """

    groups: Dict[str, Tuple[float, float]]
    effort_limit: Optional[float]
    velocity_limit: Optional[float]
    source_pd: str          # "brand" | "usd" | "family"
    source_caps: str        # "brand" | "usd" | "asset" | "none"
    notes: str = ""
    status: str = "ok"


def _usd_drive_recommendation(sku: str) -> Optional[Dict[str, Any]]:
    """Reverse-derive ``(omega_n, zeta)`` per PD group from persisted USD drive
    params, plus per-joint ``max_force`` / ``max_velocity`` peaks.

    The full pipeline this method closes:

      Robot Asset card "Dump USD" button
        → ``RobotAssetService.dump_and_persist`` (writes
          ``physics_per_format.USD`` to the user-overlay registry — see
          rule §11 full-framework persistence).
        → THIS METHOD (reverse-derive at AUTO-button click)
        → :func:`resolve_recommended_actuators` (returns groups + caps).
        → AUTO button fills the canvas RobotNode spinboxes.

    Per-joint derivation uses MJCF mass-matrix diagonal ``M[dof_j]`` at the
    nominal stance (the same ``effective_inertia_diag`` everything else
    consumes — rule §10 keeps both engines locked to one source of m_eff).
    Per-group aggregation is ``median`` across joints of the group, not
    mean — defends against a single outlier joint (e.g. a sensor mount the
    USD authored with placeholder gains) skewing the group.

    Returns ``None`` when:
      * SKU not registered.
      * No ``physics_per_format.USD`` block on the spec (i.e. the user has
        not dumped USD yet, OR is running on a legacy registry).
      * No MJCF on disk (can't compute m_eff to reverse-derive).
      * No USD joint records have an IR role we can route to a PD group.

    The caller (``resolve_recommended_actuators``) treats ``None`` as
    "USD branch not eligible — try the next fallback".

    Returns a dict with keys::

        {
            "groups": {gid: (omega_n, zeta)},   # PD group → reverse-derived
            "effort_limit": Optional[float],     # max(max_force) or None
            "velocity_limit": Optional[float],   # max(max_velocity) or None
            "n_joints_consumed": int,            # joints that contributed
            "n_joints_total": int,               # joints in physics block
            "issues": List[str],                 # human-readable per-joint notes
        }
    """
    from registers.robots import get_robot_spec, is_actuated_ir_role
    from registers.pd_groups import find_group_for_role
    from application.physics.mujoco_gain_solver import effective_inertia_diag
    from application.service.robot_assets.service import get_robot_asset_service

    spec = get_robot_spec(sku)
    if spec is None:
        return None
    physics = spec.physics_for("USD")
    if not physics:
        return None
    families = list(spec.families or [])
    if not families:
        return None
    primary_family = str(families[0])

    asset = get_robot_asset_service().resolve(sku)
    mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
    if mjcf_path is None or not Path(mjcf_path).is_file():
        # Mass matrix path requires MJCF. Without it we'd need a separate
        # PhysX-side inertia source — not implemented today, so we bail.
        return None

    # Build USD joint NAME → IR-role map from the USD joints block.
    usd_joints_block = spec.joints_for("USD")
    usd_name_to_ir: Dict[str, str] = {}
    for jspec in usd_joints_block.values():
        if not isinstance(jspec, dict):
            continue
        nm = str(jspec.get("name", "") or "")
        ir = str(jspec.get("ir_role", "") or "")
        if nm and ir:
            usd_name_to_ir[nm] = ir

    # Build IR-role → MJCF joint name (we mass-weight off MJCF; physics block
    # is keyed by USD name but m_eff is invariant under the IR-role identity
    # so a same-role MJCF joint is the right anchor).
    mjcf_joints_block = spec.joints_for("MJCF")
    ir_to_mjcf: Dict[str, str] = {}
    for jspec in mjcf_joints_block.values():
        if not isinstance(jspec, dict):
            continue
        nm = str(jspec.get("name", "") or "")
        ir = str(jspec.get("ir_role", "") or "")
        if nm and is_actuated_ir_role(ir):
            ir_to_mjcf[ir] = nm

    # Compute m_eff for every MJCF joint we may need (those with an IR role
    # that has a corresponding USD drive entry). Skip if zero overlap.
    needed_mjcf_joints: List[str] = []
    issues: List[str] = []
    eligible_pairs: List[Tuple[str, str, str, Dict[str, Any]]] = []
    # (usd_name, ir_role, mjcf_name, physics_block)
    for usd_name, phys_block in physics.items():
        if not isinstance(phys_block, dict):
            continue
        ir = usd_name_to_ir.get(usd_name, "")
        if not is_actuated_ir_role(ir):
            issues.append(
                f"{usd_name}: USD joint has no actuated IR role mapping "
                f"(ir={ir!r}); skipped for PD reverse-derive but its "
                f"max_force/max_velocity still contribute to caps if present"
            )
        mjcf_name = ir_to_mjcf.get(ir, "")
        if not mjcf_name:
            issues.append(
                f"{usd_name}: ir_role={ir!r} has no matching MJCF joint — "
                f"cannot compute m_eff, skipping PD reverse-derive"
            )
        else:
            needed_mjcf_joints.append(mjcf_name)
            eligible_pairs.append((usd_name, ir, mjcf_name, phys_block))

    m_eff_by_mjcf: Dict[str, float] = {}
    if needed_mjcf_joints:
        try:
            ei = effective_inertia_diag(
                mjcf_path=Path(mjcf_path),
                joint_order_physical=needed_mjcf_joints,
                nominal_qpos=None,
            )
            m_eff_by_mjcf = dict(zip(needed_mjcf_joints, ei.m_eff))
        except Exception as exc:  # noqa: BLE001
            issues.append(
                f"effective_inertia_diag failed ({type(exc).__name__}: {exc}); "
                f"USD-drive PD reverse-derive disabled"
            )

    # Per-joint reverse-derive, group, then median.
    per_group_omega_n: Dict[str, List[float]] = {}
    per_group_zeta: Dict[str, List[float]] = {}
    effort_candidates: List[float] = []
    velocity_candidates: List[float] = []
    n_consumed = 0

    for usd_name, ir, mjcf_name, phys_block in eligible_pairs:
        drive = phys_block.get("drive") or {}
        provenance = (phys_block.get("provenance") or {}).get("attribute_present") or {}
        stiffness = drive.get("stiffness")
        damping = drive.get("damping")
        max_force = drive.get("max_force")
        max_velocity = phys_block.get("max_velocity")

        # max_force / max_velocity contribute to caps even when PD reverse-
        # derive fails — they're independent sources of information.
        if (
            max_force is not None
            and provenance.get("drive_max_force", False)
            and max_force > 0.0
        ):
            try:
                effort_candidates.append(float(max_force))
            except (TypeError, ValueError):
                pass
        if (
            max_velocity is not None
            and provenance.get("max_velocity", False)
            and max_velocity > 0.0
        ):
            try:
                velocity_candidates.append(float(max_velocity))
            except (TypeError, ValueError):
                pass

        if mjcf_name not in m_eff_by_mjcf:
            continue
        m_eff = m_eff_by_mjcf[mjcf_name]
        if not (m_eff > 0.0):
            continue
        if stiffness is None or damping is None:
            continue
        if not provenance.get("drive_stiffness", False):
            continue
        if not provenance.get("drive_damping", False):
            continue
        try:
            k = float(stiffness)
            d = float(damping)
        except (TypeError, ValueError):
            continue
        if k <= 0.0:
            continue
        import math as _math
        omega_n = _math.sqrt(k / m_eff)
        # zeta = d / (2 · sqrt(k · m_eff)) — same form both engines use
        # (rule §10).
        denom = 2.0 * _math.sqrt(k * m_eff)
        if denom <= 0.0:
            continue
        zeta = d / denom

        # USD assets routinely ship PLACEHOLDER drive gains (Go2's USD authors
        # stiffness=1e7, damping=1e5 "make-it-rigid" defaults; many hands ship
        # zero-damping fingers) rather than tuned PD. Reverse-deriving those
        # yields physically-impossible (omega_n, zeta) — Go2 hip → omega_n≈17000
        # rad/s, zeta≈86; G1 fingers → zeta≈0. Returning them is a CLAUDE.md §8
        # illusion: AUTO "succeeds" but writes values that blow up the PD panel
        # (PDGroup.__post_init__ raises) and training. Reject the out-of-band
        # joint here; the caller falls back to family defaults for PD. The caps
        # (max_force / max_velocity) are derived independently above and stay —
        # those USD values are real even when the drive gains are placeholders.
        from application.physics.pd_param import (
            OMEGA_N_MIN, OMEGA_N_MAX, ZETA_MIN, ZETA_MAX,
        )
        if not (OMEGA_N_MIN <= omega_n <= OMEGA_N_MAX) or not (ZETA_MIN <= zeta <= ZETA_MAX):
            issues.append(
                f"{usd_name} (ir={ir}): USD drive (stiffness={k:g}, damping={d:g}) "
                f"reverse-derives to omega_n={omega_n:.1f} rad/s, zeta={zeta:.2f}, "
                f"outside physical [{OMEGA_N_MIN}, {OMEGA_N_MAX}] x "
                f"[{ZETA_MIN}, {ZETA_MAX}] — USD ships placeholder gains, not tuned "
                f"PD. Skipped; PD for this group falls back to family defaults."
            )
            continue

        match = find_group_for_role(primary_family, ir)
        gid = match[0] if match else ""
        if not gid:
            issues.append(
                f"{usd_name} (ir={ir}): no PD group matched for family "
                f"{primary_family!r}; reverse-derived (ωn,ζ) discarded"
            )
            continue
        per_group_omega_n.setdefault(gid, []).append(omega_n)
        per_group_zeta.setdefault(gid, []).append(zeta)
        n_consumed += 1

    groups_out: Dict[str, Tuple[float, float]] = {}
    for gid, ws in per_group_omega_n.items():
        zs = per_group_zeta.get(gid, [])
        if not ws or not zs:
            continue
        # Median is robust against an outlier joint (e.g. sensor mount with
        # placeholder gains in the USD).
        ws_sorted = sorted(ws)
        zs_sorted = sorted(zs)
        mid_w = ws_sorted[len(ws_sorted) // 2]
        mid_z = zs_sorted[len(zs_sorted) // 2]
        groups_out[gid] = (float(mid_w), float(mid_z))

    if not groups_out and not effort_candidates and not velocity_candidates:
        # Block exists but produced nothing usable. Don't pretend it worked.
        return None

    return {
        "groups": groups_out,
        "effort_limit": max(effort_candidates) if effort_candidates else None,
        "velocity_limit": max(velocity_candidates) if velocity_candidates else None,
        "n_joints_consumed": n_consumed,
        "n_joints_total": len(physics),
        "issues": issues,
    }


def _asset_effort_limit(sku: str) -> Optional[float]:
    """Best-effort effort cap from the MJCF (max authored actuator force limit).

    Most Menagerie MJCFs do NOT author force limits (``forcelimited=0``), in
    which case this returns ``None`` — the caller leaves effort unset. ``None``
    is the expected common case, not an error.
    """
    from application.service.robot_assets.service import get_robot_asset_service

    asset = get_robot_asset_service().resolve(sku)
    mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
    if mjcf_path is None or not Path(mjcf_path).is_file():
        return None
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception:  # noqa: BLE001
        return None
    lims = [
        float(m.actuator_forcerange[i][1])
        for i in range(m.nu)
        if bool(m.actuator_forcelimited[i]) and m.actuator_forcerange[i][1] > 0.0
    ]
    return max(lims) if lims else None


def resolve_recommended_actuators(sku: str) -> Optional[RecommendedActuators]:
    """Resolve the official recommended preset for ``sku`` (best-effort).

    **Resolution order (rule §11 full-framework fallback chain):**

    * ``(omega_n, zeta)`` per PD group:
        1. Brand manifest ``recommended_actuators.pd_groups`` (hand-curated,
           highest authority).
        2. USD drive params → :func:`_usd_drive_recommendation` reverse-derive
           with MJCF ``m_eff`` (per-joint, mass-weighted, then per-group
           median). Activates when the user has run Dump USD and the asset
           authors ``UsdPhysics.DriveAPI.stiffness/damping``.
        3. Family ``pd_groups`` defaults (always available; least specific).

    * ``effort_limit``: brand → USD ``max_force`` peak → MJCF
      ``actuator_forcerange`` peak → ``None``.

    * ``velocity_limit``: brand → USD ``max_velocity`` peak → ``None``
      (MJCF has no native velocity-cap field).

    **Status semantics** (read by AUTO button UI):
      * ``"ok"`` — at least one source contributed.
      * ``"need_dump_usd"`` — the robot declares a USD asset (local path or
        Nucleus URL) but no ``physics_per_format.USD`` block is cached. AUTO
        is expected to prompt the user to re-dump rather than silently
        proceeding to family. Brand-only PD still populates ``groups`` so
        the button does fill values; the status flag is the signal that a
        *richer* recommendation is available after a re-dump.
      * ``"no_usd_family_fallback"`` — robot declares no USD at all; the
        family default is the genuine ceiling of what's derivable.

    Returns ``None`` only when SKU is unknown or has no families (so even
    the family fallback is impossible). Never raises for a missing
    recommendation — AUTO's "try" semantics rely on a soft result.
    """
    from registers.robots import get_robot
    from registers import brands as _brands_reg

    entry = get_robot(sku)
    if not entry:
        return None
    families = list(entry.get("families", []) or [])
    if not families:
        return None
    primary_family = str(families[0])
    brand = str(entry.get("brand", "") or "")
    model = str(entry.get("model", "") or "")

    block = (
        _brands_reg.recommended_actuators(brand, model)
        if brand and model else None
    )

    # USD-drive recommendation (computed once, used by BOTH the PD branch
    # and the caps branch when no brand-side value is present).
    usd_rec = _usd_drive_recommendation(sku)

    # USD-availability flag drives the status semantics. "Declared" means
    # the robot has either a local USD path or a Nucleus URL — either way
    # a Dump USD would in principle work.
    assets = entry.get("assets", {}) or {}
    usd_declared = bool(
        (assets.get("USD") or "").strip()
        or (assets.get("USD_URL") or "").strip()
    )

    # ─── (omega_n, zeta) per group ─────────────────────────────────────
    groups: Dict[str, Tuple[float, float]] = {}
    source_pd = "family"
    brand_groups = (block or {}).get("pd_groups") if block else None
    if isinstance(brand_groups, dict) and brand_groups:
        for gid, vals in brand_groups.items():
            if not isinstance(vals, dict):
                continue
            try:
                groups[str(gid)] = (float(vals["omega_n"]), float(vals["zeta"]))
            except (KeyError, TypeError, ValueError):
                continue
        if groups:
            source_pd = "brand"
    if not groups and usd_rec and usd_rec.get("groups"):
        groups = dict(usd_rec["groups"])
        if groups:
            source_pd = "usd"
    if not groups:
        from application.physics.pd_param import PDParam
        try:
            pd = PDParam.from_family_defaults(primary_family)
            groups = {
                gid: (float(g.omega_n), float(g.zeta))
                for gid, g in pd.groups.items()
            }
            source_pd = "family"
        except Exception:  # noqa: BLE001
            return None

    # ─── effort / velocity caps ────────────────────────────────────────
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    source_caps = "none"
    if block:
        try:
            if block.get("effort_limit") is not None:
                effort_limit = float(block["effort_limit"])
            if block.get("velocity_limit") is not None:
                velocity_limit = float(block["velocity_limit"])
        except (TypeError, ValueError):
            pass
        if effort_limit is not None or velocity_limit is not None:
            source_caps = "brand"
    if effort_limit is None and usd_rec:
        ue = usd_rec.get("effort_limit")
        if ue is not None:
            try:
                effort_limit = float(ue)
                source_caps = "usd" if source_caps == "none" else source_caps
            except (TypeError, ValueError):
                pass
    if velocity_limit is None and usd_rec:
        uv = usd_rec.get("velocity_limit")
        if uv is not None:
            try:
                velocity_limit = float(uv)
                if source_caps == "none":
                    source_caps = "usd"
            except (TypeError, ValueError):
                pass
    if effort_limit is None:
        asset_eff = _asset_effort_limit(sku)
        if asset_eff is not None:
            effort_limit = asset_eff
            source_caps = "asset" if source_caps == "none" else source_caps

    # ─── Status flag (for the AUTO button's three-state UI) ────────────
    # The flag is informational: even when the brand manifest fully covers
    # PD + caps, the panel can still show a hint that re-dumping USD would
    # add a second corroboration source. But the actionable cases are
    # the two "user-fixable" ones.
    if usd_rec is None and usd_declared:
        # USD asset exists but no cached drive block — AUTO should offer
        # a re-dump prompt. Brand/family still populated the values, so
        # AUTO is not "broken" — just incomplete.
        status = "need_dump_usd"
    elif not usd_declared:
        status = "no_usd_family_fallback"
    else:
        status = "ok"

    # ─── Human-readable status note ────────────────────────────────────
    pd_sources_human = {
        "brand": (
            str((block or {}).get("source", "brand manifest"))
            if block else "brand manifest"
        ),
        "usd": "USD drive (per-joint, mass-weighted)",
        "family": f"family '{primary_family}' defaults",
    }
    notes = f"(ωn,ζ) from {pd_sources_human.get(source_pd, source_pd)}"
    if source_pd == "usd" and usd_rec:
        notes += (
            f" — {usd_rec.get('n_joints_consumed', 0)}/"
            f"{usd_rec.get('n_joints_total', 0)} joints"
        )
    if source_caps == "none":
        notes += "; no effort/velocity recommendation — set them manually"
    elif source_caps == "asset":
        notes += "; effort from MJCF force limit"
    elif source_caps == "usd":
        notes += "; effort/velocity from USD drive"
    if status == "need_dump_usd":
        notes += "; USD asset declared but no cache — re-dump USD for a richer recommendation"
    elif status == "no_usd_family_fallback":
        notes += "; no USD asset — cannot improve without one"

    return RecommendedActuators(
        groups=groups,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        source_pd=source_pd,
        source_caps=source_caps,
        notes=notes,
        status=status,
    )


__all__ = [
    "PDPreview",
    "PDPreviewGroup",
    "PDPreviewJoint",
    "PDPreviewError",
    "compute_pd_preview",
    "RecommendedActuators",
    "resolve_recommended_actuators",
]
