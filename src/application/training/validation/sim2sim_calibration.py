# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Sim2sim PD calibration (Stage G) — analytical consistency check.

Verifies that the ``mujoco_gain_solver``-derived per-joint ``(kp, kd)``,
when re-substituted into the second-order ODE model alongside the
joint's effective inertia ``M_diag``, recovers the declared
``(omega_n, zeta)`` for that joint's group.

The math: solver produces
    kp_j = M_diag_j * omega_n_j^2
    kd_j = 2 * zeta_j * sqrt(kp_j * M_diag_j)
                       = 2 * zeta_j * omega_n_j * M_diag_j

Round-trip check:
    omega_n_check_j = sqrt(kp_j / M_diag_j)
    zeta_check_j    = kd_j / (2 * sqrt(kp_j * M_diag_j))
                    = kd_j / (2 * M_diag_j * omega_n_check_j)

We expect ``omega_n_check ≈ pd_param.groups[gid].omega_n`` and
``zeta_check ≈ pd_param.groups[gid].zeta`` to within numerical tolerance.

Why analytical-only:
    A MuJoCo single-joint step response on a full articulation is
    dominated by inter-joint coupling artifacts (welded neighbors inject
    constraint forces; floating neighbors absorb reaction torque), and
    neither faithfully reflects PD parameterization correctness. Both
    engines apply gains in real torque units (PhysX's ImplicitActuator
    and MuJoCo's Python PD use the same law, tau = kp·err + kd·err_vel,
    with no inertia normalization), and both derive kp = m_eff·omega_n²
    from the same effective inertia. So the consistency we verify is two
    things: (1) the MuJoCo kp/kd round-trip to the declared (omega_n,
    zeta); and (2) the PhysX gains that training actually used equal the
    MuJoCo gains the bundle ships (passed via ``physx_kp_array`` /
    ``physx_kd_array``). Check (2) is the cross-engine guard that catches
    a re-introduced unit-mass-vs-mass-weighted asymmetry — the bug that
    once made Spot's knee 65x softer in MuJoCo than in PhysX training.

    The genuine cross-engine rollout comparison (yaml lines 192-260) is
    deferred to a later stage — requires Isaac Sim runtime and adds ~10
    min to bundle export.

The bundle finalizer (Stage F) calls :func:`run_calibration` and gates
on the verdict — ``fail`` aborts export, ``warn`` continues with the
report attached to provenance, ``pass`` continues silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np


CalibrationVerdict = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class JointCalibrationResult:
    """Per-joint calibration outcome."""

    joint_ir_role: str
    joint_physical_name: str
    group: str
    # Declared parameterization:
    omega_n_declared: float
    zeta_declared: float
    # Solver output:
    kp: float
    kd: float
    m_eff: float
    # Reconstructed from (kp, kd, m_eff):
    omega_n_reconstructed: float
    zeta_reconstructed: float
    # Relative differences:
    omega_n_relative_diff: float
    zeta_relative_diff: float
    verdict: CalibrationVerdict
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration outcome for the whole bundle."""

    verdict: CalibrationVerdict
    family: str
    canary_groups: List[str]
    per_joint: List[JointCalibrationResult]
    markdown: str

    def write_markdown(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.markdown, encoding="utf-8")
        import os
        os.replace(str(tmp), str(path))


# Canary-group recipe per family is sourced from
# ``registers.pd_calibration_canaries`` — the registry enforces an
# asymmetric overlay rule (user overlays can only EXTEND a family's
# canary list, never SHRINK it), so a user-side configuration error
# cannot silently exempt the joints (knee, hip_y on legged platforms)
# most sensitive to mass-matrix mismatch. The cross-engine TORQUE check
# in run_calibration runs over ALL joints regardless; the canary list
# only scopes the (ωn, ζ) round-trip check.


def _verdict_per_joint(
    *,
    omega_n_rel: float,
    zeta_rel: float,
    warn_omega_n: float = 0.01,    # 1% — solver's mathematical residual
    fail_omega_n: float = 0.10,    # 10% — bug or float overflow
    warn_zeta: float = 0.05,
    fail_zeta: float = 0.20,
) -> tuple[CalibrationVerdict, List[str]]:
    notes: List[str] = []
    verdict: CalibrationVerdict = "pass"
    if abs(omega_n_rel) > fail_omega_n:
        verdict = "fail"
        notes.append(
            f"omega_n reconstruction off by {omega_n_rel*100:.1f}% (fail >{fail_omega_n*100:.0f}%)"
        )
    elif abs(omega_n_rel) > warn_omega_n:
        verdict = "warn"
        notes.append(
            f"omega_n reconstruction off by {omega_n_rel*100:.1f}% (warn >{warn_omega_n*100:.0f}%)"
        )
    if abs(zeta_rel) > fail_zeta:
        verdict = "fail"
        notes.append(
            f"zeta reconstruction off by {zeta_rel*100:.1f}% (fail >{fail_zeta*100:.0f}%)"
        )
    elif abs(zeta_rel) > warn_zeta and verdict == "pass":
        verdict = "warn"
        notes.append(
            f"zeta reconstruction off by {zeta_rel*100:.1f}% (warn >{warn_zeta*100:.0f}%)"
        )
    return verdict, notes


def _select_canary_joint_indices(
    *,
    joint_ir_roles: List[str],
    family: str,
    pd_param: Any,
) -> List[int]:
    from registers import pd_calibration_canaries as _pd_canaries
    if not _pd_canaries.has_canaries(family):
        return []
    canary_groups = list(_pd_canaries.get_canaries(family).canary_groups)
    if not canary_groups:
        return []
    from application.physics.joint_group_resolver import expand_groups_to_joints
    role_to_group = expand_groups_to_joints(pd_param, joint_ir_roles)
    chosen: List[int] = []
    seen_groups: set = set()
    for idx, (_role, group) in enumerate(role_to_group):
        if group.name in canary_groups and group.name not in seen_groups:
            chosen.append(idx)
            seen_groups.add(group.name)
    return chosen


def run_calibration(
    *,
    mjcf_path: Path,
    pd_param: Any,
    joint_order_physical: List[str],
    joint_ir_roles: List[str],
    kp_array: np.ndarray,
    kd_array: np.ndarray,
    family: str,
    nominal_qpos: Optional[np.ndarray] = None,
    report_path: Optional[Path] = None,
    physx_kp_array: Optional[np.ndarray] = None,
    physx_kd_array: Optional[np.ndarray] = None,
    remotized_tables: Optional[Dict[str, Any]] = None,
) -> CalibrationReport:
    """Verify (kp, kd) round-trip to declared (omega_n, zeta).

    For each canary joint:
      1. Re-run ``mujoco_gain_solver`` to get the joint's ``M_diag`` from
         ``derivation["per_joint"][i]["m_eff"]``.
      2. Reconstruct ``omega_n = sqrt(kp / M_diag)`` and
         ``zeta = kd / (2 * sqrt(kp * M_diag))``.
      3. Compare against ``pd_param.groups[<group>].{omega_n, zeta}``.
      4. Emit verdict.

    When ``physx_kp_array`` / ``physx_kd_array`` are supplied (the gains
    training actually used, from ``deploy_meta.physx_gains``), also run a
    cross-engine TORQUE guard: for one representative ``(q_err, q̇)`` the
    torque each engine applies must match within 1e-3 relative, else
    ``fail``. This is the check that catches a re-introduced unit-mass
    asymmetry (the Spot-knee-65× bug) at export time.

    Returns the report; writes markdown if ``report_path`` is given.
    """
    from application.physics.joint_group_resolver import expand_groups_to_joints
    from application.physics.mujoco_gain_solver import solve as _solve_mujoco

    # We need the per-joint m_eff diagnostic; re-solve to grab it
    # (cheap; mj_fullM is one matmul on a sub-100-DoF model).
    gains = _solve_mujoco(
        mjcf_path=mjcf_path,
        joint_order_physical=list(joint_order_physical),
        joint_ir_roles=list(joint_ir_roles),
        nominal_qpos=nominal_qpos,
        pd_param=pd_param,
    )
    per_joint_diag = {
        entry["physical_name"]: entry
        for entry in gains.derivation.get("per_joint", [])
    }

    role_to_group = expand_groups_to_joints(pd_param, joint_ir_roles)
    canary_idx = _select_canary_joint_indices(
        joint_ir_roles=joint_ir_roles,
        family=family,
        pd_param=pd_param,
    )

    per_joint: List[JointCalibrationResult] = []
    worst_verdict: CalibrationVerdict = "pass"
    verdict_rank = {"pass": 0, "warn": 1, "fail": 2}

    for idx in canary_idx:
        role = joint_ir_roles[idx]
        phys = joint_order_physical[idx]
        _, group = role_to_group[idx]
        omega_n_decl = float(group.omega_n)
        zeta_decl = float(group.zeta)
        kp = float(kp_array[idx])
        kd = float(kd_array[idx])
        m_eff = float(per_joint_diag.get(phys, {}).get("m_eff", 0.0))

        notes: List[str] = []
        if m_eff <= 0.0:
            per_joint.append(JointCalibrationResult(
                joint_ir_role=role,
                joint_physical_name=phys,
                group=group.name,
                omega_n_declared=omega_n_decl,
                zeta_declared=zeta_decl,
                kp=kp, kd=kd, m_eff=m_eff,
                omega_n_reconstructed=0.0,
                zeta_reconstructed=0.0,
                omega_n_relative_diff=1.0,
                zeta_relative_diff=1.0,
                verdict="fail",
                notes=["m_eff is non-positive at nominal stance"],
            ))
            worst_verdict = "fail"
            continue

        omega_n_recon = math.sqrt(max(kp, 0.0) / m_eff)
        denom = 2.0 * math.sqrt(max(kp, 0.0) * m_eff)
        zeta_recon = (kd / denom) if denom > 0.0 else 0.0

        omega_rel = (omega_n_recon - omega_n_decl) / max(omega_n_decl, 1e-9)
        zeta_rel = (zeta_recon - zeta_decl) / max(zeta_decl, 1e-9)

        verdict, joint_notes = _verdict_per_joint(
            omega_n_rel=omega_rel,
            zeta_rel=zeta_rel,
        )
        notes.extend(joint_notes)

        per_joint.append(JointCalibrationResult(
            joint_ir_role=role,
            joint_physical_name=phys,
            group=group.name,
            omega_n_declared=omega_n_decl,
            zeta_declared=zeta_decl,
            kp=kp, kd=kd, m_eff=m_eff,
            omega_n_reconstructed=omega_n_recon,
            zeta_reconstructed=zeta_recon,
            omega_n_relative_diff=omega_rel,
            zeta_relative_diff=zeta_rel,
            verdict=verdict,
            notes=notes,
        ))

        if verdict_rank[verdict] > verdict_rank[worst_verdict]:
            worst_verdict = verdict

    # ----- Cross-engine TORQUE check. The gain magnitudes alone aren't the
    # point — what must match is the torque each engine actually applies for
    # the SAME control situation. Both engines use the same PD law in the
    # UnitPort deploy convention (verified against IsaacLab IdealPDActuator /
    # ImplicitActuator.compute and the MuJoCo PDController):
    #     τ = kp·(q* − q) − kd·q̇            (desired joint velocity q̇* = 0,
    #                                         JointPositionAction convention)
    # We feed one representative non-trivial (q_err, q̇) and assert
    # |τ_physx − τ_mujoco| / |τ_mujoco| < 1e-3 per joint. This catches a
    # re-introduced unit-mass-vs-mass-weighted asymmetry (the Spot-knee-65×
    # bug) AND any divergence in the PD law itself — orders of magnitude
    # cheaper than catching it in a sim2sim rollout. (The full PhysX-runtime
    # rollout torque comparison stays deferred — needs Isaac Sim.)
    cross_engine_notes: List[str] = []
    if physx_kp_array is not None and physx_kd_array is not None:
        n = len(kp_array)
        if len(physx_kp_array) != n or len(physx_kd_array) != n:
            cross_engine_notes.append(
                f"physx gain arrays length mismatch "
                f"(kp {len(physx_kp_array)}, kd {len(physx_kd_array)} vs "
                f"mujoco {n}) — cannot cross-check"
            )
            worst_verdict = "fail"
        else:
            # THRESHOLD LAYERING — DO NOT "unify" with the load-time guard.
            # This export-time gate uses a STRICT 1e-3: at export the two
            # gain sets come straight from the solvers (no serialization
            # between them), so any divergence above float-residual is a real
            # bug and must abort the write. The load-time guard in
            # DeployContract.from_dict deliberately uses a LOOSER 1% because
            # it runs on arrays that round-tripped through YAML and must
            # tolerate float-repr noise while still catching the 9–65× bug.
            # The two thresholds guard DIFFERENT failure modes (solver bug at
            # export vs stale/hand-edited bundle at load); collapsing them to
            # one number reintroduces ambiguity. Keep them separate.
            TORQUE_FAIL_REL = 1e-3
            # Representative control situation exercising BOTH kp and kd.
            Q_ERR = 0.1   # rad position tracking error (q* − q)
            QD = 0.5      # rad/s joint velocity

            def _pd_torque(kp: float, kd: float) -> float:
                # UnitPort deploy PD law; q̇* = 0.
                return kp * Q_ERR - kd * QD

            worst_tau_rel = 0.0
            worst_joint = -1
            for i in range(n):
                tau_mj = _pd_torque(float(kp_array[i]), float(kd_array[i]))
                tau_px = _pd_torque(float(physx_kp_array[i]), float(physx_kd_array[i]))
                rel = abs(tau_px - tau_mj) / max(abs(tau_mj), 1e-9)
                if rel > worst_tau_rel:
                    worst_tau_rel = rel
                    worst_joint = i
            if worst_tau_rel > TORQUE_FAIL_REL:
                worst_verdict = "fail"
                jn = (
                    joint_order_physical[worst_joint]
                    if 0 <= worst_joint < len(joint_order_physical) else "?"
                )
                cross_engine_notes.append(
                    f"PhysX vs MuJoCo torque divergence at (q_err={Q_ERR}, "
                    f"q̇={QD}): max |Δτ|/|τ| = {worst_tau_rel:.3e} on joint "
                    f"{jn!r} (fail >{TORQUE_FAIL_REL:.0e}). Both engines must "
                    f"mass-weight off identical inertia — check the env "
                    f"compiler and bundle finalizer use the same MJCF + "
                    f"nominal stance (m_eff parity)."
                )
            else:
                cross_engine_notes.append(
                    f"PhysX ≈ MuJoCo torque OK at (q_err={Q_ERR}, q̇={QD}): "
                    f"max |Δτ|/|τ| = {worst_tau_rel:.3e} (< {TORQUE_FAIL_REL:.0e})"
                )

    # ----- Step 5: multi-point cross-engine check for REMOTIZED joints. The
    # two-probe sentinel above samples a single (q_err, q̇); a remotized joint's
    # ceiling is angle-dependent, so a constant-ceiling bug at one knee angle
    # could hide while another angle works. For each remotized joint we sample
    # its lookup range and confirm the IsaacLab-side ceiling (via
    # to_isaaclab_array) and the MuJoCo-side ceiling (via max_torque_at) clamp
    # the PD torque identically. At export both ceilings trace to the SAME
    # source table, so this primarily guards the to_isaaclab_array conversion
    # and the gain parity across angles. STRICT 1e-3 (export-time), the same
    # threshold the kp/kd cross-check uses — DO NOT loosen to the load-time 1%.
    if remotized_tables and physx_kp_array is not None and physx_kd_array is not None:
        from application.physics.actuators.torque_lookup import TorqueLookupTable
        from application.physics.actuators.torque_consistency import (
            check_remotized_torque_consistency,
        )
        role_to_idx = {r: i for i, r in enumerate(joint_ir_roles)}
        for role, tbl_dict in remotized_tables.items():
            if role not in role_to_idx:
                cross_engine_notes.append(
                    f"remotized joint ir_role={role!r} not in the calibrated "
                    f"joint set — cannot cross-check (skipping)."
                )
                worst_verdict = "fail"
                continue
            ri = role_to_idx[role]
            table = TorqueLookupTable.from_dict(tbl_dict)
            res = check_remotized_torque_consistency(
                joint=role,
                isaac_table=table,
                mujoco_table=table,  # same source at export (one embedded copy)
                kp_isaac=float(physx_kp_array[ri]),
                kd_isaac=float(physx_kd_array[ri]),
                kp_mujoco=float(kp_array[ri]),
                kd_mujoco=float(kd_array[ri]),
                threshold=1e-3,
            )
            if not res.passed:
                worst_verdict = "fail"
                cross_engine_notes.append("REMOTIZED " + res.diagnosis())
            else:
                w = res.worst
                cross_engine_notes.append(
                    f"remotized joint {role!r} cross-engine torque OK across "
                    f"{len(res.points)} probe·angle points "
                    f"(max rel {w.rel_error:.3e} < 1e-3)."
                )

    from registers import pd_calibration_canaries as _pd_canaries
    canary_groups = (
        list(_pd_canaries.get_canaries(family).canary_groups)
        if _pd_canaries.has_canaries(family)
        else []
    )
    md_lines = [
        f"# Sim2Sim PD Calibration Report",
        "",
        f"- **Family**: `{family}`",
        f"- **Canary groups**: {', '.join(canary_groups) or '(none)'}",
        f"- **Joints calibrated**: {len(per_joint)}",
        f"- **Overall verdict**: **`{worst_verdict.upper()}`**",
        "",
        "## Per-joint results",
        "",
        "| Joint | Group | M_diag (kg·m²) | kp | kd | ωn declared / reconstructed | ζ declared / reconstructed | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_joint:
        md_lines.append(
            f"| `{r.joint_ir_role}` | `{r.group}` | {r.m_eff:.6f} | "
            f"{r.kp:.2f} | {r.kd:.3f} | "
            f"{r.omega_n_declared:.2f} / {r.omega_n_reconstructed:.2f} "
            f"({r.omega_n_relative_diff*100:+.1f}%) | "
            f"{r.zeta_declared:.2f} / {r.zeta_reconstructed:.2f} "
            f"({r.zeta_relative_diff*100:+.1f}%) | **{r.verdict}** |"
        )
        if r.notes:
            md_lines.append(f"  - notes: {'; '.join(r.notes)}")
    if not per_joint:
        md_lines.append("| _(no canary joints)_ | | | | | | | |")

    if cross_engine_notes:
        md_lines.extend([
            "",
            "## Cross-engine torque check (PhysX vs MuJoCo)",
            "",
        ])
        md_lines.extend(f"- {note}" for note in cross_engine_notes)

    md_lines.extend([
        "",
        "## Method",
        "",
        "- ``kp_j = M_diag_j * omega_n_j^2`` and "
        "``kd_j = 2 * zeta_j * sqrt(kp_j * M_diag_j)`` — derived for BOTH "
        "engines (``mujoco_gain_solver`` / ``physx_gain_solver``) off the "
        "same effective inertia.",
        "- Reconstructed: ``omega_n = sqrt(kp / M_diag)``, "
        "``zeta = kd / (2 * sqrt(kp * M_diag))``.",
        "- A `fail` verdict aborts bundle export; `warn` continues with "
        "the report attached. `skip_calibration` on ActuatorPDNode bypasses "
        "this gate (logged into provenance).",
        "- Both engines apply gains in real torque units (no inertia "
        "normalization), so matching the declared `(omega_n, zeta)` on the "
        "MuJoCo side AND `physx_kp == mujoco_kp` means a policy trained in "
        "PhysX meets identical gains in MuJoCo verification.",
    ])
    markdown = "\n".join(md_lines) + "\n"

    report = CalibrationReport(
        verdict=worst_verdict,
        family=family,
        canary_groups=list(canary_groups),
        per_joint=per_joint,
        markdown=markdown,
    )

    if report_path is not None:
        report.write_markdown(Path(report_path))

    return report
