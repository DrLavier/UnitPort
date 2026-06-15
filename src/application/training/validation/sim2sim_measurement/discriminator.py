# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Residual spectrum + systematic-vs-chaotic discriminator (Stage-1 v1).

Given the two engines' open-loop rollouts of the SAME probe set (measured AFTER
Layer-1 alignment, so the residual is pure Type-II/III), this:

  1. computes a per-(probe, repeat) cross-engine **residual** on the probe's
     primary channel (contact force for stiffness/transient, slip for the
     friction cone, joint-state drift for trajectory replay);
  2. estimates each probe's **sensitivity** to initial-condition perturbation
     ``S = std(R_r) / (mean(R_r) + ε)`` across its repeats — the §4.3 go/no-go
     metric — and GRADES it on a 3-level scale (NOT a single binary cut, since
     real S clusters in a middle band with no natural bimodal gap):
     **clean** (S<0.1, ~deterministic → no DR), **parameterizable**
     (0.1≤S<1.0, bounded sensitivity → Layer-2 DR), **chaotic** (S≥1.0,
     exponential-ish divergence → Layer-3 fail-loud);
  3. aggregates per **dimension**: a grade + (clean / parameterizable) the
     cross-engine effective-response **factor envelope** ``[low, high]`` that
     Stage-2 randomization must span; chaotic dimensions lose their range.

The continuous S is ALWAYS recorded; the grade is a label over it. Thresholds
are tunable and the report surfaces the full S distribution + sampling-
convergence so a real dataset can recalibrate them (Lyapunov-exponent refinement
is a documented future upgrade, not v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .protocol import EngineResult, ProbeSet, StepRecord

# Sensitivity (= coefficient-of-variation of the cross-engine residual across
# IC-perturbed repeats) GRADES — a 3-level grade, NOT a single binary threshold.
# Real measurement showed S clustering in a 0.17–0.48 middle band with NO
# natural bimodal gap, so a single systematic/chaotic cut is the wrong model:
#   * S <  CLEAN_S          → "clean": residual ~deterministic in IC; no DR needed.
#   * CLEAN_S ≤ S < CHAOS_S → "parameterizable": bounded IC sensitivity →
#                              Layer-2 domain randomization absorbs it.
#   * S >= CHAOS_S          → "chaotic": residual spread exceeds its mean
#                              (exponential-ish divergence) → Layer-3 fail-loud,
#                              NOT coverable by finite randomization.
# The continuous S is always recorded; the grade is a label over it. Thresholds
# are tunable and the report prints the full S distribution + convergence so a
# real dataset can recalibrate them.
CLEAN_S = 0.1
CHAOS_S = 1.0

# Back-compat alias (the old single threshold == the chaos boundary).
CHAOS_SENSITIVITY = CHAOS_S

_EPS = 1e-9


def grade_sensitivity(s: float, *, clean_s: float = CLEAN_S, chaos_s: float = CHAOS_S) -> str:
    """Map a continuous sensitivity S to its 3-level grade."""
    if s < clean_s:
        return "clean"
    if s >= chaos_s:
        return "chaotic"
    return "parameterizable"

# Each probe dimension's primary residual channel.
PRIMARY_CHANNEL: Dict[str, str] = {
    "contact_stiffness": "contact_force",
    "contact_transient": "contact_force",
    "friction_cone": "slip",
    "trajectory": "joint_state",
}

# Scenario overrides the dimension-default channel when the scenario admits a
# physically cleaner measure. The ``block_slide`` single-rigid-body friction
# probe decelerates a box by Coulomb friction alone: its clean signature is the
# STOPPING DISTANCE (net box displacement ∝ v²/2μg), NOT the accumulated slip
# speed. Slip-sum conflates high-μ stick-slip jitter (the box vibrates in place
# at high μ, inflating ∑|v| non-monotonically) — displacement stays monotone in
# μ. So a block_slide friction probe routes to ``box_displacement``, while the
# legged ``lateral_slip`` friction probe stays on ``slip``.
SCENARIO_CHANNEL: Dict[str, str] = {
    "block_slide": "box_displacement",
}


@dataclass
class ProbeResidual:
    probe_id: str
    dimension: str
    scenario: str
    channel: str
    residual_per_repeat: List[float]
    sensitivity: float
    verdict: str                 # "clean" | "parameterizable" | "chaotic"
    mujoco_measure: float        # effective measure (engine-side), repeat-mean
    physx_measure: float
    ratio: float                 # physx_measure / mujoco_measure (effective DR factor)
    # S computed on the first n repeats, n = 2..N (硬化二): if this is still
    # drifting at n=N the sample is too small to trust the verdict; if it has
    # plateaued the verdict is sampling-converged.
    sensitivity_convergence: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "dimension": self.dimension,
            "scenario": self.scenario,
            "channel": self.channel,
            "residual_per_repeat": [float(x) for x in self.residual_per_repeat],
            "sensitivity": float(self.sensitivity),
            "sensitivity_convergence": [float(x) for x in self.sensitivity_convergence],
            "verdict": self.verdict,
            "mujoco_measure": float(self.mujoco_measure),
            "physx_measure": float(self.physx_measure),
            "ratio": float(self.ratio),
        }


@dataclass
class DimensionVerdict:
    dimension: str
    verdict: str                 # "clean" | "parameterizable" | "chaotic"
    n_probes: int
    sensitivities: List[float]
    factor_low: Optional[float]  # cross-engine effective-response envelope
    factor_high: Optional[float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dimension": self.dimension,
            "verdict": self.verdict,
            "n_probes": int(self.n_probes),
            "sensitivities": [float(x) for x in self.sensitivities],
            "factor_low": (None if self.factor_low is None else float(self.factor_low)),
            "factor_high": (None if self.factor_high is None else float(self.factor_high)),
            "notes": list(self.notes),
        }


@dataclass
class MeasurementAnalysis:
    sku: str
    per_probe: List[ProbeResidual]
    per_dimension: List[DimensionVerdict]
    chaos_sensitivity_threshold: float = CHAOS_SENSITIVITY

    def to_dict(self) -> Dict[str, object]:
        return {
            "sku": self.sku,
            "chaos_sensitivity_threshold": float(self.chaos_sensitivity_threshold),
            "per_probe": [p.to_dict() for p in self.per_probe],
            "per_dimension": [d.to_dict() for d in self.per_dimension],
        }


# ---------------------------------------------------------------------------
# channel extraction
# ---------------------------------------------------------------------------

def _channel_series(steps: List[StepRecord], channel: str, n_qpos: int) -> np.ndarray:
    """Per-step scalar/vector series for a channel."""
    if channel == "contact_force":
        return np.asarray([s.contact_force_norm for s in steps], dtype=float)
    if channel == "slip":
        return np.asarray([s.slip for s in steps], dtype=float)
    if channel == "joint_state":
        # Joint-part of qpos (skip the 7-dof free base when present).
        base = 7 if n_qpos >= 7 else 0
        return np.asarray([s.next_qpos[base:] for s in steps], dtype=float)
    if channel == "box_displacement":
        # Box x-position over the rollout (free-base qpos[0]); the effective
        # measure derives the net stopping distance from it.
        return np.asarray([s.next_qpos[0] for s in steps], dtype=float)
    raise ValueError(f"_channel_series: unknown channel {channel!r}")


def _effective_measure(series: np.ndarray, channel: str) -> float:
    """Engine-side effective measure for a channel (the thing whose cross-engine
    ratio becomes the DR factor).

    Contact force uses the IMPULSE (∑F·dt ∝ ∑F over a fixed-dt rollout), NOT the
    peak (硬化一-2). Peak contact force is dominated by the contact-stiffness
    transient and is apples-to-oranges between a stiff (PhysX) and a soft
    (MuJoCo) contact model — it inflated the cross-engine factor ~40×. Impulse is
    momentum-conserving: both engines must absorb the same drop momentum / support
    the same weight, so the impulse-level factor sits near the physical ~1× and
    the residual concentrates in the landing transient (the real object of study)."""
    if series.size == 0:
        return 0.0
    if channel == "contact_force":
        return float(np.sum(series))           # ∝ contact impulse ∫F dt
    if channel == "slip":
        return float(np.sum(np.abs(series)))   # accumulated slip
    if channel == "joint_state":
        return float(np.mean(np.linalg.norm(series, axis=-1)))  # mean joint magnitude
    if channel == "box_displacement":
        # Net stopping distance: |final − initial| box x. Monotone in μ where
        # slip-sum is not (no stick-slip jitter), so the cross-engine ratio is
        # the clean friction-cone factor.
        return float(abs(series[-1] - series[0]))
    return 0.0


def _residual(mj: np.ndarray, px: np.ndarray, channel: str) -> float:
    """Cross-engine divergence of a channel over the rollout, as a single
    global-L2 ratio ``‖a − b‖ / ‖a‖``.

    Global (not per-step-mean) on purpose: a per-step relative residual averaged
    over the rollout saturates in [0, 1] and compresses the cross-repeat
    variance the sensitivity test relies on. This form is unbounded and
    preserves spread — a systematic bias (``b = κ·a``) yields the CONSTANT
    ``|κ − 1|`` every repeat (S → 0), while a chaotic per-repeat offset yields a
    residual that scales with the offset (S large)."""
    n = min(len(mj), len(px))
    if n == 0:
        return 0.0
    a = np.asarray(mj[:n], dtype=float).reshape(-1)
    b = np.asarray(px[:n], dtype=float).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + _EPS))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _index(results: List[EngineResult]) -> Dict[Tuple[str, int], EngineResult]:
    out: Dict[Tuple[str, int], EngineResult] = {}
    for r in results:
        out[(r.probe_id, int(r.repeat))] = r
    return out


def analyze(
    probe_set: ProbeSet,
    mujoco_results: List[EngineResult],
    physx_results: List[EngineResult],
    *,
    clean_s: float = CLEAN_S,
    chaos_s: float = CHAOS_S,
) -> MeasurementAnalysis:
    """Compute the residual spectrum + graded (clean/parameterizable/chaotic)
    verdicts. The continuous sensitivity S is always recorded alongside the
    grade; the grade is just a label over the continuous spectrum."""
    mj_idx = _index(mujoco_results)
    px_idx = _index(physx_results)
    n_qpos = int(probe_set.n_qpos)

    per_probe: List[ProbeResidual] = []
    for probe in probe_set.probes:
        # Scenario override (e.g. block_slide → box_displacement) takes
        # precedence over the dimension default; both are explicit tables (§8).
        channel = (SCENARIO_CHANNEL.get(probe.scenario)
                   or PRIMARY_CHANNEL.get(probe.dimension, "joint_state"))
        residuals: List[float] = []
        mj_measures: List[float] = []
        px_measures: List[float] = []
        for r in range(probe.n_repeats):
            key = (probe.probe_id, r)
            mj = mj_idx.get(key)
            px = px_idx.get(key)
            if mj is None or px is None:
                raise ValueError(
                    f"analyze: missing {('mujoco' if mj is None else 'physx')} "
                    f"result for probe {probe.probe_id!r} repeat {r} — the two "
                    f"engines ran different probe sets (§8)."
                )
            mj_s = _channel_series(mj.steps, channel, n_qpos)
            px_s = _channel_series(px.steps, channel, n_qpos)
            residuals.append(_residual(mj_s, px_s, channel))
            mj_measures.append(_effective_measure(mj_s, channel))
            px_measures.append(_effective_measure(px_s, channel))

        arr = np.asarray(residuals, dtype=float)
        mean_r = float(np.mean(arr))
        sensitivity = float(np.std(arr) / (mean_r + _EPS))
        # Sensitivity-convergence curve: S using the first n repeats (n=2..N).
        conv: List[float] = []
        for n in range(2, len(residuals) + 1):
            sub = arr[:n]
            conv.append(float(np.std(sub) / (float(np.mean(sub)) + _EPS)))
        verdict = grade_sensitivity(sensitivity, clean_s=clean_s, chaos_s=chaos_s)
        mj_m = float(np.mean(mj_measures))
        px_m = float(np.mean(px_measures))
        ratio = px_m / (mj_m + _EPS) if mj_m > _EPS else 1.0
        per_probe.append(ProbeResidual(
            probe_id=probe.probe_id, dimension=probe.dimension,
            scenario=probe.scenario, channel=channel,
            residual_per_repeat=residuals, sensitivity=sensitivity,
            verdict=verdict, mujoco_measure=mj_m, physx_measure=px_m, ratio=ratio,
            sensitivity_convergence=conv,
        ))

    # Per-dimension aggregation.
    per_dimension: List[DimensionVerdict] = []
    dims = sorted({p.dimension for p in per_probe})
    for dim in dims:
        probes = [p for p in per_probe if p.dimension == dim]
        sens = [p.sensitivity for p in probes]
        med = float(np.median(sens)) if sens else 0.0
        verdict = grade_sensitivity(med, clean_s=clean_s, chaos_s=chaos_s)
        notes: List[str] = []
        if verdict == "chaotic":
            # Only a TRUE exponential-divergence dimension (S ≫ 1) loses its
            # range — it cannot be covered by finite randomization → Layer 3.
            factor_low = factor_high = None
            notes.append(
                f"median S={med:.3f} ≥ {chaos_s} (chaotic) — residual not "
                "coverable by finite randomization; routes to Layer 3 fail-loud "
                "(non-convergent zone), NOT Layer 2."
            )
        else:
            ratios = [p.ratio for p in probes if p.ratio > _EPS]
            factor_low = float(min(ratios)) if ratios else 1.0
            factor_high = float(max(ratios)) if ratios else 1.0
            if verdict == "clean":
                notes.append(
                    f"median S={med:.3f} < {clean_s} (clean — residual nearly "
                    f"deterministic in IC); factor span [{factor_low:.3f}, "
                    f"{factor_high:.3f}]. DR optional (factor ≈1 ⇒ unnecessary)."
                )
            else:
                notes.append(
                    f"median S={med:.3f} in [{clean_s}, {chaos_s}) "
                    f"(parameterizable — bounded IC sensitivity); cross-engine "
                    f"factor span [{factor_low:.3f}, {factor_high:.3f}] → "
                    "Layer-2 DR range."
                )
        per_dimension.append(DimensionVerdict(
            dimension=dim, verdict=verdict, n_probes=len(probes),
            sensitivities=sens, factor_low=factor_low, factor_high=factor_high,
            notes=notes,
        ))

    return MeasurementAnalysis(
        sku=probe_set.aligned_plant.sku,
        per_probe=per_probe,
        per_dimension=per_dimension,
        chaos_sensitivity_threshold=chaos_s,
    )


__all__ = [
    "CLEAN_S",
    "CHAOS_S",
    "CHAOS_SENSITIVITY",
    "grade_sensitivity",
    "PRIMARY_CHANNEL",
    "SCENARIO_CHANNEL",
    "ProbeResidual",
    "DimensionVerdict",
    "MeasurementAnalysis",
    "analyze",
]
