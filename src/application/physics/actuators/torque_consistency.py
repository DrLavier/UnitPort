# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Cross-engine torque-consistency check for remotized actuators (directive Step 5).

A remotized joint's torque ceiling is angle-dependent, served by a
``TorqueLookupTable``. Two engines consume it:

  - IsaacLab (training) reads ``table.to_isaaclab_array()`` (the 3-column
    ``[angle, gear, max_torque]`` form) and clamps the PD torque against it;
  - MuJoCo (deploy) reads ``table.max_torque_at(q)`` inside ``PDController``.

In UnitPort both sides trace to the SAME source table (one embedded copy,
reused — World B lesson), so this check is defense-in-depth: it catches a bug
in ``to_isaaclab_array`` (column swap, dropped row, precision loss) or in the
contract embedding that makes the two engines clamp at different ceilings for
the same angle, BEFORE such a bundle ships and silently mis-transfers.

Why a two-probe, per-engine-gain design (deviates from the directive's literal
single-``τ_pd`` pseudocode — recorded as design decision D17):

  The directive's 5.1 pseudocode computes one ``τ_pd = kp·0.1 − kd·0.5`` and
  clips it with each engine's ceiling. But (a) with a single shared ``τ_pd``,
  ``τ_isaac`` and ``τ_mujoco`` are identical wherever neither clips, so 5.2's
  "PD-region failure means kp/kd diverged" is unreachable; and (b) ``kp·0.1``
  is ~6 N·m, far below a 30–113 N·m knee ceiling, so the clip region — the
  table path this check exists to guard — is never exercised. To make BOTH
  5.2 diagnoses and BOTH 5.3 required tests reachable, each engine uses its
  OWN gains, and two probes are evaluated at each sample angle:

    * sentinel probe (small Δq, finite q̇): stays in the PD region; a
      divergence here means the PhysX/MuJoCo gains disagree — a regression of
      the World B mass-weighting fix (kp/kd drift).
    * saturating probe (Δq large enough that |τ_pd| > the peak ceiling for
      BOTH engines, q̇=0): forces the clip region at every angle; a divergence
      here means the two ceilings disagree — a lookup-data bug
      (``to_isaaclab_array`` conversion or contract-embedding fidelity loss).

Pure Python + numpy. No SDK / app imports, so it loads in both the main app
and the IsaacLab compile venv (same constraint as ``torque_lookup``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from application.physics.actuators.torque_lookup import TorqueLookupTable

# Sentinel probe: the same (position error, joint velocity) the existing
# single-point cross-engine guards use. Small enough to stay in the PD region.
_SENTINEL_DQ = 0.1   # rad position tracking error (q* − q)
_SENTINEL_QD = 0.5   # rad/s joint velocity
# Saturating probe: how far above the peak ceiling to push |τ_pd| so the clip
# region is reached for both engines regardless of gain magnitude. 3× leaves
# generous margin without risking float overflow on realistic kp.
_SATURATION_FACTOR = 3.0
_EPS = 1e-9


@dataclass(frozen=True)
class TorquePointCheck:
    """One (probe, angle) comparison of the two engines' applied torque."""

    probe: str          # "sentinel" | "saturating"
    q: float            # sample joint angle (rad)
    tau_pd_isaac: float
    tau_pd_mujoco: float
    tau_max_isaac: float
    tau_max_mujoco: float
    tau_isaac: float    # clipped torque the IsaacLab side would apply
    tau_mujoco: float   # clipped torque the MuJoCo side would apply
    rel_error: float
    region: str         # "pd" | "clip"


@dataclass(frozen=True)
class RemotizedTorqueCheckResult:
    """Outcome of the multi-point check for ONE remotized joint."""

    joint: str
    threshold: float
    passed: bool
    points: List[TorquePointCheck] = field(default_factory=list)
    worst: Optional[TorquePointCheck] = None

    def diagnosis(self) -> str:
        """Human-readable failure diagnosis distinguishing the two modes.

        Empty string when ``passed``. On failure, names the joint, the probe
        and angle that diverged, both engines' torque, the relative error, and
        — crucially — whether the failure is in the PD-law region (gains
        diverged) or the clip region (lookup data diverged), because the two
        have different root causes and different fixes.
        """
        if self.passed or self.worst is None:
            return ""
        w = self.worst
        if w.region == "pd":
            cause = (
                "PD-LAW region (|τ_pd| < both ceilings): the two engines' "
                "PD gains disagree — τ_pd_isaac=%.4f vs τ_pd_mujoco=%.4f. This "
                "is a regression of the World B mass-weighting fix (kp/kd "
                "drift), NOT a lookup-table problem. Check that PhysX "
                "(stiffness/damping) and MuJoCo (mujoco_pd_gains/_damping) "
                "mass-weight off the same m_eff." % (w.tau_pd_isaac, w.tau_pd_mujoco)
            )
        else:
            cause = (
                "CLIP region (|τ_pd| ≥ a ceiling): the two engines' torque "
                "CEILINGS disagree at this angle — τ_max_isaac=%.4f vs "
                "τ_max_mujoco=%.4f. Both should trace to the same source "
                "table, so this is a lookup-data bug: to_isaaclab_array "
                "conversion (column swap / dropped row) or contract-embedding "
                "fidelity loss between pd_derivation and mujoco_torque_lookups."
                % (w.tau_max_isaac, w.tau_max_mujoco)
            )
        return (
            f"remotized torque mismatch on joint {self.joint!r}: at probe="
            f"{w.probe!r} q={w.q:+.4f} rad, τ_isaac={w.tau_isaac:.4f} vs "
            f"τ_mujoco={w.tau_mujoco:.4f} N·m, rel error {w.rel_error:.3e} "
            f"(> threshold {self.threshold:.0e}). {cause}"
        )


def check_remotized_torque_consistency(
    *,
    joint: str,
    isaac_table: TorqueLookupTable,
    mujoco_table: TorqueLookupTable,
    kp_isaac: float,
    kd_isaac: float,
    kp_mujoco: float,
    kd_mujoco: float,
    threshold: float,
    n_points: int = 5,
) -> RemotizedTorqueCheckResult:
    """Compare the torque the two engines apply for one remotized joint.

    The IsaacLab ceiling is sourced THROUGH ``isaac_table.to_isaaclab_array()``
    (so a conversion bug in that method is caught); the MuJoCo ceiling through
    ``mujoco_table.max_torque_at``. Sample angles are ``n_points`` evenly
    spaced across the MuJoCo table's axis range (inclusive of endpoints). At
    each angle two probes run (see module docstring): a sentinel probe in the
    PD region and a saturating probe in the clip region.

    Returns a :class:`RemotizedTorqueCheckResult`; ``passed`` is False if any
    point's relative torque error exceeds ``threshold``. ``worst`` carries the
    largest-error point for diagnosis.
    """
    # IsaacLab ceiling via the 3-column array: x = angle (col 0), y = max
    # torque (col 2). np.interp clamps to endpoints — identical to IsaacLab's
    # LinearInterpolation zero-order-hold outside range. Routing through
    # to_isaaclab_array (not max_torque_at) is the whole point: it exercises
    # the conversion this check guards.
    arr = isaac_table.to_isaaclab_array()
    isa_x, isa_y = arr[:, 0], arr[:, 2]

    def tau_max_isaac(q: float) -> float:
        return float(np.interp(q, isa_x, isa_y))

    def tau_max_mujoco(q: float) -> float:
        return mujoco_table.max_torque_at(q)

    lo, hi = mujoco_table.axis_range
    q_samples = np.linspace(lo, hi, max(int(n_points), 2))

    # Saturating Δq: push |τ_pd| above the peak ceiling for BOTH engines. Use
    # the smaller |kp| so both saturate; peak from both tables for safety.
    peak = max(isaac_table.peak_torque, mujoco_table.peak_torque)
    kp_min = min(abs(kp_isaac), abs(kp_mujoco))
    dq_sat = _SATURATION_FACTOR * peak / max(kp_min, _EPS)

    def _pd_torque(kp: float, kd: float, dq: float, qd: float) -> float:
        # UnitPort deploy PD law; desired joint velocity q̇* = 0.
        return kp * dq - kd * qd

    points: List[TorquePointCheck] = []
    worst: Optional[TorquePointCheck] = None

    for probe, dq, qd in (
        ("sentinel", _SENTINEL_DQ, _SENTINEL_QD),
        ("saturating", dq_sat, 0.0),
    ):
        tau_pd_i = _pd_torque(kp_isaac, kd_isaac, dq, qd)
        tau_pd_m = _pd_torque(kp_mujoco, kd_mujoco, dq, qd)
        for q in q_samples:
            q = float(q)
            cap_i = tau_max_isaac(q)
            cap_m = tau_max_mujoco(q)
            t_i = float(np.clip(tau_pd_i, -cap_i, cap_i))
            t_m = float(np.clip(tau_pd_m, -cap_m, cap_m))
            rel = abs(t_i - t_m) / max(abs(t_m), _EPS)
            # PD region iff NEITHER engine clips; else clip region.
            region = (
                "pd"
                if (abs(tau_pd_i) < cap_i and abs(tau_pd_m) < cap_m)
                else "clip"
            )
            pt = TorquePointCheck(
                probe=probe, q=q,
                tau_pd_isaac=tau_pd_i, tau_pd_mujoco=tau_pd_m,
                tau_max_isaac=cap_i, tau_max_mujoco=cap_m,
                tau_isaac=t_i, tau_mujoco=t_m,
                rel_error=rel, region=region,
            )
            points.append(pt)
            if worst is None or rel > worst.rel_error:
                worst = pt

    passed = worst is None or worst.rel_error <= threshold
    return RemotizedTorqueCheckResult(
        joint=joint,
        threshold=threshold,
        passed=passed,
        points=points,
        worst=worst,
    )
