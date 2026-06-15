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
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Realized==Designed audit matrix (Stage 0 — Layer 1 unification)
# ---------------------------------------------------------------------------
# The three legacy calibration dimensions (ωn/ζ round-trip, cross-engine
# torque, realized joint-inertia) plus the Stage-0 additions (friction,
# foot-collision-geometry, joint-order identity, control-period) are gathered
# into ONE queryable matrix: one row per joint (cells = the per-joint
# dimensions) + one row per engine-level alignment quantity. The matrix is an
# ADDITIVE wrapper embedded in ``CalibrationReport`` — it never changes the
# legacy ``verdict`` / ``per_joint`` / ``markdown`` surface old readers depend
# on. Each cell carries its OWN warn/fail thresholds: the matrix deliberately
# does NOT define a single shared threshold table (CLAUDE.md §10 forbids
# "unifying" the 1e-3 torque / 1% load / 2%-5% realized-inertia gates — they
# guard different failure modes).

AuditDimension = Literal[
    "omega_zeta_roundtrip",          # Dim 1 — canary joints only
    "cross_engine_torque",           # Dim 2 — all joints
    "realized_inertia_2engine",      # Dim 3a — all joints
    "realized_inertia_vs_designed",  # Dim 3b — all joints
]


@dataclass(frozen=True)
class AuditCell:
    """One (row, dimension) realized==designed verdict.

    ``applicable=False`` renders as "—": the dimension does not apply to this
    row (e.g. the ωn/ζ round-trip on a non-canary joint, or a deferred /
    un-provided engine quantity). ``warn_threshold`` / ``fail_threshold`` are
    per-cell on purpose — they prove, in the rendered matrix, that the gates
    are NOT unified.
    """

    dimension: str
    applicable: bool
    verdict: CalibrationVerdict = "pass"
    realized: Optional[float] = None
    designed: Optional[float] = None
    relative_diff: Optional[float] = None
    warn_threshold: Optional[float] = None
    fail_threshold: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuditRow:
    """One matrix row: a joint (``row_kind="joint"``) or an engine-level
    alignment quantity (``row_kind="engine"``: friction / foot_geom /
    joint_order / control_period)."""

    row_id: str
    row_kind: Literal["joint", "engine"]
    cells: Dict[str, AuditCell]
    joint_physical_name: Optional[str] = None
    group: Optional[str] = None

    @property
    def verdict(self) -> CalibrationVerdict:
        rank = {"pass": 0, "warn": 1, "fail": 2}
        worst = "pass"
        for c in self.cells.values():
            if c.applicable and rank[c.verdict] > rank[worst]:
                worst = c.verdict
        return worst  # type: ignore[return-value]


@dataclass(frozen=True)
class AuditMatrix:
    """The full realized==designed audit, one row per joint + per engine
    quantity. Queryable and self-rendering; embedded in ``CalibrationReport``."""

    verdict: CalibrationVerdict
    family: str
    canary_groups: List[str]
    rows: List[AuditRow]
    dimensions: List[str]

    def query(
        self,
        *,
        row_id: Optional[str] = None,
        dimension: Optional[str] = None,
        min_verdict: CalibrationVerdict = "warn",
    ) -> List[AuditCell]:
        """Return cells at or above ``min_verdict``, optionally filtered by
        row / dimension. Read-only — the resident self-check surface."""
        rank = {"pass": 0, "warn": 1, "fail": 2}
        floor = rank[min_verdict]
        out: List[AuditCell] = []
        for row in self.rows:
            if row_id is not None and row.row_id != row_id:
                continue
            for dim, cell in row.cells.items():
                if dimension is not None and dim != dimension:
                    continue
                if cell.applicable and rank[cell.verdict] >= floor:
                    out.append(cell)
        return out

    def to_markdown(self) -> str:
        rank = {"pass": 0, "warn": 1, "fail": 2}
        joint_rows = [r for r in self.rows if r.row_kind == "joint"]
        engine_rows = [r for r in self.rows if r.row_kind == "engine"]
        lines: List[str] = [
            "## Realized==Designed audit matrix",
            "",
            f"- **Overall**: **`{self.verdict.upper()}`** · "
            f"{len(joint_rows)} joint rows · {len(engine_rows)} engine rows",
            "",
            "_Each cell carries its own warn/fail thresholds — the gates are "
            "deliberately NOT unified (CLAUDE.md §10)._",
            "",
        ]
        # Joint × dimension table.
        dim_short = {
            "omega_zeta_roundtrip": "ωn/ζ",
            "cross_engine_torque": "τ(px/mj)",
            "realized_inertia_2engine": "I(mj/px)",
            "realized_inertia_vs_designed": "I(mj/des)",
        }
        header = ["Joint", "Group"] + [dim_short.get(d, d) for d in self.dimensions]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in joint_rows:
            cells_txt = []
            for d in self.dimensions:
                c = row.cells.get(d)
                if c is None or not c.applicable:
                    cells_txt.append("—")
                else:
                    rd = (
                        f" {c.relative_diff*100:+.1f}%"
                        if c.relative_diff is not None else ""
                    )
                    cells_txt.append(f"{c.verdict}{rd}")
            lines.append(
                f"| `{row.row_id}` | `{row.group or ''}` | "
                + " | ".join(cells_txt) + " |"
            )
        # Engine rows.
        lines.extend(["", "### Engine-level alignment", ""])
        lines.append("| Quantity | Verdict | Realized | Designed | Notes |")
        lines.append("|---|---|---|---|---|")
        for row in engine_rows:
            # An engine row carries a single same-named cell.
            c = next(iter(row.cells.values())) if row.cells else None
            if c is None:
                continue
            rz = "—" if c.realized is None else f"{c.realized:g}"
            dz = "—" if c.designed is None else f"{c.designed:g}"
            applic = c.verdict if c.applicable else "n/a"
            note = "; ".join(c.notes) if c.notes else ""
            lines.append(
                f"| `{row.row_id}` | **{applic}** | {rz} | {dz} | {note} |"
            )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration outcome for the whole bundle."""

    verdict: CalibrationVerdict
    family: str
    canary_groups: List[str]
    per_joint: List[JointCalibrationResult]
    markdown: str
    matrix: Optional[AuditMatrix] = None

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
    inertia_overlay: Optional[Dict[str, Any]] = None,
    armature_mjcf: Optional[np.ndarray] = None,
    physx_armature: Optional[np.ndarray] = None,
    # ----- Stage 0 audit-matrix engine rows (all optional / back-compat) -----
    scene_contract: Optional[Dict[str, Any]] = None,
    foot_geometry: Optional[Dict[str, Any]] = None,
    foot_roles: Optional[Sequence[str]] = None,
    ir_to_body_name: Optional[Mapping[str, str]] = None,
    control_period: Optional[Dict[str, Any]] = None,
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
    #
    # CLAUDE.md §10/§11 (Q6): we pass ``inertia_overlay`` so this re-solve gives
    # the REALIZED (USD-corrected) m_eff — the inertia the robot is actually
    # simulated with. Because ``kp_array`` below is the BUNDLE's gains (fixed
    # bytes, not re-derived here), the (ωn,ζ) round-trip ``sqrt(kp / m_eff)``
    # becomes a genuine check: if a future regression derives ``kp`` on the RAW
    # link inertia while the runtime realizes the overlaid one, ``kp`` and this
    # ``m_eff`` disagree and the round-trip FAILS. (Pre-fix this re-solve used
    # raw too, so the round-trip was self-consistent and blind to the gap.)
    gains = _solve_mujoco(
        mjcf_path=mjcf_path,
        joint_order_physical=list(joint_order_physical),
        joint_ir_roles=list(joint_ir_roles),
        nominal_qpos=nominal_qpos,
        pd_param=pd_param,
        inertia_overlay=inertia_overlay,
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
    # Additive per-joint capture for the audit matrix (Stage 0). Populated
    # INSIDE the existing Dim2/Dim3 loops below — no control-flow change, reuses
    # the exact numbers those loops compute (no formula duplication). Keyed by
    # physical joint name.
    _tau_by_phys: Dict[str, tuple] = {}       # phys -> (tau_mj, tau_px, rel)
    _ri_by_phys: Dict[str, tuple] = {}        # phys -> (wn_mj, wn_px, px_rel, wn_des, des_rel)
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
                if i < len(joint_order_physical):
                    _tau_by_phys[joint_order_physical[i]] = (tau_mj, tau_px, rel)
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

    # ----- Realized joint-inertia diagonal probe (CLAUDE.md §10/§11, Task B).
    # The torque/round-trip checks above verify gain MAGNITUDES and the MuJoCo
    # (ωn,ζ) round-trip; they are BLIND to whether PhysX *realizes* the same
    # joint-space inertia. PhysX's USD authors no armature, so a ``kp`` derived
    # on ``m_eff = link + armature`` is realized in PhysX on ``link`` alone — a
    # stiffer plant at low-inertia joints (H1 ankle: 4.7× over design). This
    # probe decomposes the realized diagonal and checks, over EVERY joint:
    #   (a) two-engine agreement — MuJoCo (link+armature_mjcf) vs PhysX
    #       (link+armature_emitted) realize the same ωn (catches a missing /
    #       wrong PhysX armature), and
    #   (b) MuJoCo realized ωn == DESIGNED ωn — because ``m_eff`` here is the
    #       OVERLAID (realized) inertia while ``kp`` is the bundle's fixed bytes,
    #       this catches a regression that derives m_eff on the RAW link inertia
    #       while the runtime realizes the overlaid one (Q6 "derive ≠ realize").
    #
    # THRESHOLD LAYERING — SEPARATE from the torque probe (1e-3 / 1%); DO NOT
    # unify. This dimension tolerates warn 2% / fail 5%. Rationale: armature and
    # link inertia are physics-solver parameters — once BOTH engines set the
    # authoritative diagonal they agree to solver precision, but 2/5% absorbs the
    # USD↔MJCF link-inertia harvest float + overlay round-trip. The torque probe
    # guards "gain NUMBER identity"; this guards "realized INERTIA identity" —
    # different failure modes (a wrong armature leaves kp/kd byte-identical, so
    # the torque probe cannot see it), so the dimensions must stay distinct.
    if armature_mjcf is not None and physx_armature is not None:
        REAL_WARN_REL = 0.02
        REAL_FAIL_REL = 0.05
        n_ri = len(kp_array)
        if len(armature_mjcf) != n_ri or len(physx_armature) != n_ri:
            cross_engine_notes.append(
                f"realized-inertia probe SKIPPED: armature arrays length "
                f"mismatch (armature_mjcf {len(armature_mjcf)}, physx_armature "
                f"{len(physx_armature)} vs kp {n_ri})."
            )
            worst_verdict = "fail"
        else:
            worst_px_rel = 0.0
            worst_px_j = -1
            worst_des_rel = 0.0
            worst_des_j = -1
            ri_nonphysical: List[str] = []
            for i in range(n_ri):
                phys = joint_order_physical[i]
                m_eff = float(per_joint_diag.get(phys, {}).get("m_eff", 0.0))
                if m_eff <= 0.0:
                    continue
                arm_mj = float(armature_mjcf[i])
                arm_px = float(physx_armature[i])
                i_link = m_eff - arm_mj                  # USD-corrected link inertia
                i_mj = m_eff                             # MuJoCo realizes link + MJCF armature
                i_px = i_link + arm_px                   # PhysX realizes link + emitted armature
                if i_mj <= 0.0 or i_px <= 0.0:
                    ri_nonphysical.append(
                        f"{phys} (I_mj={i_mj:.5f}, I_px={i_px:.5f})"
                    )
                    continue
                kp_i = float(kp_array[i])
                wn_mj = math.sqrt(max(kp_i, 0.0) / i_mj)
                wn_px = math.sqrt(max(kp_i, 0.0) / i_px)
                px_rel = abs(wn_px - wn_mj) / max(wn_mj, 1e-9)
                if px_rel > worst_px_rel:
                    worst_px_rel, worst_px_j = px_rel, i
                _, group_i = role_to_group[i]
                wn_des = float(group_i.omega_n)
                des_rel = abs(wn_mj - wn_des) / max(wn_des, 1e-9)
                _ri_by_phys[phys] = (wn_mj, wn_px, px_rel, wn_des, des_rel)
                if des_rel > worst_des_rel:
                    worst_des_rel, worst_des_j = des_rel, i

            def _jn(j: int) -> str:
                return (
                    joint_order_physical[j]
                    if 0 <= j < len(joint_order_physical) else "?"
                )

            if ri_nonphysical:
                worst_verdict = "fail"
                cross_engine_notes.append(
                    "realized-inertia: non-physical realized inertia (≤0) at "
                    + ", ".join(ri_nonphysical)
                    + " — an emitted armature drove the joint-space inertia "
                    "non-positive; check armature provenance."
                )
            # (a) two-engine realized agreement.
            if worst_px_rel > REAL_FAIL_REL:
                worst_verdict = "fail"
                cross_engine_notes.append(
                    f"realized-inertia FAIL: MuJoCo vs PhysX realized ωn diverge "
                    f"by {worst_px_rel*100:.1f}% at {_jn(worst_px_j)!r} (fail "
                    f">{REAL_FAIL_REL*100:.0f}%). PhysX is NOT realizing the same "
                    f"joint-space inertia — its USD authors no armature; emit "
                    f"ImplicitActuatorCfg.armature = the MJCF dof_armature so "
                    f"both engines realize link + armature (CLAUDE.md §10)."
                )
            elif worst_px_rel > REAL_WARN_REL and verdict_rank[worst_verdict] < 1:
                worst_verdict = "warn"
                cross_engine_notes.append(
                    f"realized-inertia WARN: MuJoCo vs PhysX realized ωn diverge "
                    f"by {worst_px_rel*100:.1f}% at {_jn(worst_px_j)!r} (warn "
                    f">{REAL_WARN_REL*100:.0f}%)."
                )
            # (b) MuJoCo realized ωn vs designed (catches raw-vs-overlaid Q6 regression).
            if worst_des_rel > REAL_FAIL_REL:
                worst_verdict = "fail"
                cross_engine_notes.append(
                    f"realized-inertia FAIL: MuJoCo REALIZED ωn diverges from "
                    f"DESIGNED by {worst_des_rel*100:.1f}% at {_jn(worst_des_j)!r} "
                    f"(fail >{REAL_FAIL_REL*100:.0f}%). kp was derived on a "
                    f"different joint-space inertia than the runtime realizes — "
                    f"effective_inertia_diag must derive m_eff on the SAME "
                    f"(overlay-applied) model the runtime uses (Q6)."
                )
            elif worst_des_rel > REAL_WARN_REL and verdict_rank[worst_verdict] < 1:
                worst_verdict = "warn"
                cross_engine_notes.append(
                    f"realized-inertia WARN: MuJoCo realized vs designed ωn off "
                    f"by {worst_des_rel*100:.1f}% at {_jn(worst_des_j)!r} (warn "
                    f">{REAL_WARN_REL*100:.0f}%)."
                )
            if worst_px_rel <= REAL_WARN_REL and worst_des_rel <= REAL_WARN_REL \
                    and not ri_nonphysical:
                cross_engine_notes.append(
                    f"realized-inertia OK: both engines realize the designed ωn "
                    f"(max two-engine {worst_px_rel*100:.2f}%, max vs-designed "
                    f"{worst_des_rel*100:.2f}%, < {REAL_WARN_REL*100:.0f}%)."
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

    # ===================================================================
    # Stage 0 — realized==designed audit matrix (Layer 1 unification).
    # Engine-level rows (friction / foot_geom / joint_order / control_period)
    # are NEW this stage; each computes its own verdict and ESCALATES
    # ``worst_verdict`` so a friction / foot-geometry / ordering break aborts
    # export exactly like the legacy dimensions. Joint rows are reconstructed
    # from the per-joint data the legacy loops already captured (per_joint,
    # _tau_by_phys, _ri_by_phys) — additive, no recomputation of the gate math.
    # Each cell carries its OWN thresholds (CLAUDE.md §10 — gates not unified).
    # ===================================================================
    audit_engine_notes: List[str] = []
    engine_rows: List[AuditRow] = []

    # Lazy raw-MJCF load for friction + foot-geom realized reads. These
    # quantities are independent of the inertia overlay (which touches only
    # body mass/inertia), so a raw load is correct. ``_solve_mujoco`` already
    # parsed this same MJCF above, so the load cannot fail here.
    _audit_model = None
    _need_model = (scene_contract is not None) or (
        isinstance(foot_geometry, dict)
        and foot_geometry.get("status") == "ok"
        and bool(foot_roles)
        and ir_to_body_name is not None
    )
    if _need_model:
        import mujoco as _mj
        _audit_model = _mj.MjModel.from_xml_path(str(mjcf_path))

    def _escalate(v: CalibrationVerdict) -> None:
        nonlocal worst_verdict
        if verdict_rank[v] > verdict_rank[worst_verdict]:
            worst_verdict = v

    # ----- Row: friction (method A — single source, deploy applies it) -----
    if isinstance(scene_contract, dict):
        mu_s = scene_contract.get("friction_static")
        mu_d = scene_contract.get("friction_dynamic")
        notes_f: List[str] = []
        vf: CalibrationVerdict = "pass"
        realized_f: Optional[float] = None
        designed_f: Optional[float] = None
        if mu_s is None:
            vf = "fail"
            notes_f.append(
                "scene_contract present but friction_static missing — a "
                "play_ground_setting node must stash it (§8)."
            )
        else:
            designed_f = float(mu_s)
            if not math.isfinite(designed_f) or designed_f <= 0.0:
                vf = "fail"
                notes_f.append(
                    f"friction_static={designed_f} is non-finite or ≤0 — invalid "
                    "ground friction; deploy cannot realize the designed contact."
                )
            else:
                # Single source: PolicyRunner writes this exact value into the
                # deploy MuJoCo geom_friction[:,0] (method A), so realized ==
                # designed by construction. Surface the raw MJCF default for
                # context (it is overridden, not a fault).
                realized_f = designed_f
                if _audit_model is not None and int(_audit_model.ngeom) > 0:
                    raw_mu = float(_audit_model.geom_friction[0, 0])
                    if abs(raw_mu - designed_f) > 1e-9:
                        notes_f.append(
                            f"MJCF default μ={raw_mu:g}; deploy overrides to canvas "
                            f"μ={designed_f:g} (PolicyRunner applies friction_static)."
                        )
                if mu_d is not None and abs(float(mu_d) - designed_f) > 1e-9:
                    notes_f.append(
                        f"μ_s={designed_f:g} ≠ μ_d={float(mu_d):g}: MuJoCo's single "
                        "Coulomb coefficient cannot represent the static/dynamic "
                        "split — μ_d recorded, not realized (irreducible engine "
                        "difference, NOTE not fail)."
                    )
                notes_f.append(
                    "cross-engine loaded-state μ (contact-cone behavior) is "
                    "Type-II — deferred to Layer-2 measurement (Kit-gated)."
                )
        cell_f = AuditCell(
            dimension="friction", applicable=True, verdict=vf,
            realized=realized_f, designed=designed_f,
            warn_threshold=None, fail_threshold=0.0, notes=notes_f,
        )
    else:
        cell_f = AuditCell(
            dimension="friction", applicable=False, verdict="pass",
            notes=["no play_ground_setting friction designed (row not audited)"],
        )
    engine_rows.append(AuditRow(
        row_id="friction", row_kind="engine", cells={"friction": cell_f}))
    _escalate(cell_f.verdict)

    # ----- Row: passive joint dissipation (PD must be sole damping) -----
    # Previously invisible to this gate: the torque check is a static
    # gain-identity and the realized-inertia check covers armature, but
    # neither sees the MJCF's velocity-dependent passive dof_damping /
    # dof_frictionloss. Menagerie MJCFs author those on every leg joint;
    # they stack on the PD kd in mj_step, over-damping the joint vs the
    # PhysX-trained policy (PhysX ImplicitActuator has no passive joint
    # damping). The runtime (mj_actor / generic_mujoco_env) now neutralizes
    # them via mjcf_pd_prep so PD kd is the SOLE damping; this row records
    # what was authored-then-stripped so the gate is no longer blind (§11).
    notes_jd: List[str] = []
    vjd: CalibrationVerdict = "pass"
    try:
        import mujoco as _mj_jd
        from application.physics.mjcf_pd_prep import pd_controlled_dofs
        _raw = _mj_jd.MjModel.from_xml_path(str(mjcf_path))
        _dofs = pd_controlled_dofs(_raw)
        _max_damp = max((abs(float(_raw.dof_damping[d])) for d in _dofs),
                        default=0.0)
        _max_fric = max((abs(float(_raw.dof_frictionloss[d])) for d in _dofs),
                        default=0.0)
        if _max_damp == 0.0 and _max_fric == 0.0:
            notes_jd.append(
                "MJCF authors no passive joint dissipation on PD-driven "
                "joints — PD kd is already the sole damping (matches PhysX)."
            )
        else:
            notes_jd.append(
                f"MJCF authors passive dof_damping≤{_max_damp:g} / "
                f"frictionloss≤{_max_fric:g} on {len(_dofs)} PD-driven "
                "joint(s); neutralized at load (mjcf_pd_prep) so PD kd is the "
                "sole damping and MuJoCo realizes the designed ζ — matching "
                "PhysX, which has no passive joint damping. armature kept "
                "(folded into m_eff)."
            )
    except Exception as _jd_exc:                              # pragma: no cover
        vjd = "warn"
        notes_jd.append(
            f"could not inspect MJCF passive dissipation ({_jd_exc}); "
            "runtime still neutralizes it at load."
        )
    cell_jd = AuditCell(
        dimension="joint_dissipation", applicable=True, verdict=vjd,
        notes=notes_jd,
    )
    engine_rows.append(AuditRow(
        row_id="joint_dissipation", row_kind="engine",
        cells={"joint_dissipation": cell_jd}))
    _escalate(cell_jd.verdict)

    # ----- Row: foot collision geometry (pose-invariant hash) -----
    if isinstance(foot_geometry, dict):
        status = foot_geometry.get("status")
        if status == "not_applicable":
            cell_g = AuditCell(
                dimension="foot_geom", applicable=False, verdict="pass",
                notes=["family declares no feet by design (audited, n/a)"],
            )
        elif status == "ok":
            designed_agg = str(foot_geometry.get("aggregate", ""))
            notes_g: List[str] = []
            vg: CalibrationVerdict = "pass"
            if not (foot_roles and ir_to_body_name is not None
                    and _audit_model is not None):
                vg = "warn"
                notes_g.append(
                    "designed foot_geometry present but realized-recompute "
                    "inputs (foot_roles / ir_to_body_name / model) missing — "
                    "cannot verify realized==designed."
                )
            else:
                from application.physics.foot_geometry import hash_foot_geometry
                realized = hash_foot_geometry(
                    _audit_model, foot_roles=foot_roles,
                    ir_to_body_name=ir_to_body_name,
                )
                realized_agg = str(realized.get("aggregate", ""))
                if realized_agg != designed_agg:
                    vg = "fail"
                    notes_g.append(
                        "foot collision geometry MISMATCH — the gate's MJCF foot "
                        f"geoms hash differently than training designed (designed "
                        f"{designed_agg[:12]}… vs realized {realized_agg[:12]}…): "
                        "compiler↔finalizer MJCF disagree or the asset changed."
                    )
                else:
                    notes_g.append(
                        "foot collision geometry matches (aggregate hash equal)."
                    )
            cell_g = AuditCell(
                dimension="foot_geom", applicable=True, verdict=vg, notes=notes_g)
        elif status == "unavailable":
            # The producer (env_cfg_compiler) records 'unavailable' when the
            # foot-geometry hash could not be computed (asset/role gap) and
            # explicitly designs this as NON-FATAL: it WARNs and the deploy-side
            # self-check engages only on status='ok'. The gate MUST match that
            # contract — a non-blocking warn, NOT a fail that aborts export (§11
            # producer↔consumer parity). The foot row is simply not audited.
            cell_g = AuditCell(
                dimension="foot_geom", applicable=False, verdict="warn",
                notes=["foot_geometry status='unavailable' (producer could not "
                       "hash the foot geoms — asset/role gap). Non-fatal by "
                       "design: deploy-side foot self-check is skipped; export "
                       "is NOT aborted."])
        else:
            cell_g = AuditCell(
                dimension="foot_geom", applicable=True, verdict="fail",
                notes=[f"foot_geometry has unexpected status={status!r} (§8)"])
    else:
        cell_g = AuditCell(
            dimension="foot_geom", applicable=False, verdict="pass",
            notes=["no foot_geometry provided (row not audited)"])
    engine_rows.append(AuditRow(
        row_id="foot_geom", row_kind="engine", cells={"foot_geom": cell_g}))
    _escalate(cell_g.verdict)

    # ----- Row: joint-order identity (formalize array_ordering='ir_roles') -----
    notes_jo: List[str] = []
    vjo: CalibrationVerdict = "pass"
    _lengths: Dict[str, int] = {
        "joint_ir_roles": len(joint_ir_roles),
        "joint_order_physical": len(joint_order_physical),
        "kp": len(kp_array), "kd": len(kd_array),
    }
    if physx_kp_array is not None:
        _lengths["physx_kp"] = len(physx_kp_array)
    if physx_kd_array is not None:
        _lengths["physx_kd"] = len(physx_kd_array)
    if armature_mjcf is not None:
        _lengths["armature_mjcf"] = len(armature_mjcf)
    if physx_armature is not None:
        _lengths["physx_armature"] = len(physx_armature)
    if len(set(_lengths.values())) != 1:
        vjo = "fail"
        notes_jo.append(f"per-joint array length mismatch: {_lengths} (§8).")
    if len(set(joint_order_physical)) != len(joint_order_physical):
        vjo = "fail"
        notes_jo.append("duplicate physical joint names — ordering ambiguous (§8).")
    if len(set(joint_ir_roles)) != len(joint_ir_roles):
        vjo = "fail"
        notes_jo.append("duplicate IR roles — name-keyed alignment ambiguous (§8).")
    if vjo == "pass":
        notes_jo.append(
            f"joint-order identity OK: {len(joint_order_physical)} joints, all "
            "per-joint arrays aligned by IR role (array_ordering='ir_roles'). "
            "Cross-producer name alignment enforced upstream in the finalizer."
        )
    cell_jo = AuditCell(
        dimension="joint_order", applicable=True, verdict=vjo, notes=notes_jo)
    engine_rows.append(AuditRow(
        row_id="joint_order", row_kind="engine", cells={"joint_order": cell_jo}))
    _escalate(cell_jo.verdict)

    # ----- Row: control period (decimation round-trip; physics Kit-deferred) -----
    if isinstance(control_period, dict):
        notes_cp: List[str] = []
        vcp: CalibrationVerdict = "pass"
        sim_dt = control_period.get("sim_dt")
        step_dt = control_period.get("step_dt")
        decim = control_period.get("decimation")
        designed_cp: Optional[float] = None
        if sim_dt is None or step_dt is None or decim is None:
            vcp = "fail"
            notes_cp.append(f"control_period incomplete: {control_period} (§8).")
        else:
            sim_dt = float(sim_dt); step_dt = float(step_dt); decim = int(decim)
            designed_cp = float(decim)
            if sim_dt <= 0.0:
                vcp = "fail"
                notes_cp.append(f"sim_dt={sim_dt} ≤ 0 (§8).")
            else:
                expect = round(step_dt / sim_dt)
                if expect != decim:
                    vcp = "fail"
                    notes_cp.append(
                        f"decimation {decim} != round(step_dt/sim_dt) = "
                        f"round({step_dt}/{sim_dt}) = {expect} (§8)."
                    )
                else:
                    notes_cp.append(
                        f"control period OK: decimation {decim} == "
                        f"round({step_dt}/{sim_dt})."
                    )
        notes_cp.append(
            "physics-period realization (does each engine step the declared "
            "dt) is deferred — Kit-gated."
        )
        cell_cp = AuditCell(
            dimension="control_period", applicable=True, verdict=vcp,
            designed=designed_cp, notes=notes_cp)
    else:
        cell_cp = AuditCell(
            dimension="control_period", applicable=False, verdict="pass",
            notes=["control_period not provided (row not audited); "
                   "physics-period Kit-deferred"])
    engine_rows.append(AuditRow(
        row_id="control_period", row_kind="engine",
        cells={"control_period": cell_cp}))
    _escalate(cell_cp.verdict)

    # ----- Joint rows: fold the legacy 3 dimensions into one row per joint.
    # Dim1 (ωn/ζ round-trip) is canary-only — non-canary joints get an
    # applicable=False cell ("—"); per_joint stays canary-only (back-compat).
    _pj_by_phys = {r.joint_physical_name: r for r in per_joint}
    _canary_set = set(canary_idx)
    _DIMS = [
        "omega_zeta_roundtrip", "cross_engine_torque",
        "realized_inertia_2engine", "realized_inertia_vs_designed",
    ]
    joint_rows: List[AuditRow] = []
    for idx in range(len(joint_order_physical)):
        role = joint_ir_roles[idx] if idx < len(joint_ir_roles) else f"?{idx}"
        phys = joint_order_physical[idx]
        grp = role_to_group[idx][1].name if idx < len(role_to_group) else ""
        cells: Dict[str, AuditCell] = {}
        # Dim 1 — canary-gated.
        if idx in _canary_set and phys in _pj_by_phys:
            r = _pj_by_phys[phys]
            cells["omega_zeta_roundtrip"] = AuditCell(
                dimension="omega_zeta_roundtrip", applicable=True,
                verdict=r.verdict, realized=r.omega_n_reconstructed,
                designed=r.omega_n_declared,
                relative_diff=r.omega_n_relative_diff,
                warn_threshold=0.01, fail_threshold=0.10, notes=list(r.notes))
        else:
            cells["omega_zeta_roundtrip"] = AuditCell(
                dimension="omega_zeta_roundtrip", applicable=False)
        # Dim 2 — cross-engine torque (1e-3, fail-only — mirrors inline gate).
        if phys in _tau_by_phys:
            tau_mj, tau_px, rel = _tau_by_phys[phys]
            cells["cross_engine_torque"] = AuditCell(
                dimension="cross_engine_torque", applicable=True,
                verdict=("fail" if rel > 1e-3 else "pass"),
                realized=tau_mj, designed=tau_px, relative_diff=rel,
                warn_threshold=None, fail_threshold=1e-3)
        else:
            cells["cross_engine_torque"] = AuditCell(
                dimension="cross_engine_torque", applicable=False)
        # Dim 3a/3b — realized inertia (2%/5% — mirrors inline gate).
        if phys in _ri_by_phys:
            wn_mj, wn_px, px_rel, wn_des, des_rel = _ri_by_phys[phys]
            cells["realized_inertia_2engine"] = AuditCell(
                dimension="realized_inertia_2engine", applicable=True,
                verdict=("fail" if px_rel > 0.05 else
                         "warn" if px_rel > 0.02 else "pass"),
                realized=wn_mj, designed=wn_px, relative_diff=px_rel,
                warn_threshold=0.02, fail_threshold=0.05)
            cells["realized_inertia_vs_designed"] = AuditCell(
                dimension="realized_inertia_vs_designed", applicable=True,
                verdict=("fail" if des_rel > 0.05 else
                         "warn" if des_rel > 0.02 else "pass"),
                realized=wn_mj, designed=wn_des, relative_diff=des_rel,
                warn_threshold=0.02, fail_threshold=0.05)
        else:
            cells["realized_inertia_2engine"] = AuditCell(
                dimension="realized_inertia_2engine", applicable=False)
            cells["realized_inertia_vs_designed"] = AuditCell(
                dimension="realized_inertia_vs_designed", applicable=False)
        joint_rows.append(AuditRow(
            row_id=role, row_kind="joint", cells=cells,
            joint_physical_name=phys, group=grp))

    matrix = AuditMatrix(
        verdict=worst_verdict,
        family=family,
        canary_groups=list(canary_groups),
        rows=joint_rows + engine_rows,
        dimensions=_DIMS,
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
            "## Cross-engine checks (PhysX vs MuJoCo)",
            "",
            "_Torque (gain-number identity, 1e-3) + realized joint-inertia "
            "(armature + link, 2%/5%) — distinct failure modes, see source._",
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
    # Append the realized==designed audit matrix (joint×dim + engine rows). It
    # is ADDED after the legacy sections, so old readers still find the
    # per-joint table unchanged above (back-compat).
    md_lines.extend(["", matrix.to_markdown()])
    markdown = "\n".join(md_lines) + "\n"

    report = CalibrationReport(
        verdict=worst_verdict,
        family=family,
        canary_groups=list(canary_groups),
        per_joint=per_joint,
        markdown=markdown,
        matrix=matrix,
    )

    if report_path is not None:
        report.write_markdown(Path(report_path))

    return report
