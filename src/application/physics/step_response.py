# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Per-joint step-response simulator for sim2sim calibration.

Locks every joint except one at its nominal pose, commands a position
step on the target joint, and records the response. Used by
:mod:`application.training.validation.sim2sim_calibration` (Stage G of
the sim2sim PD plan) to compare engine response curves with their
declared ``(omega_n, zeta)`` parameterization.

The simulator runs in MuJoCo only — the PhysX side runs through Isaac
Lab's own evaluation loop and produces an equivalent metric set (rise
time, overshoot, settling, steady-state error). The two sets of metrics
are compared in :func:`compare_step_responses` to produce the
calibration verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from .pd_param import JointPDGains


@dataclass(frozen=True)
class StepResponseMetrics:
    """Metrics extracted from a single-joint step-response trace.

    All times are in seconds; positions in radians (or meters for
    prismatic joints — currently unused). Negative values indicate the
    metric could not be computed from the trace (e.g. no overshoot
    observed).
    """

    rise_time_s: float          # 10% -> 90% of step
    overshoot_pct: float        # peak excursion above target / step amplitude * 100
    settling_time_s: float      # time to enter and stay within ±2% band
    steady_state_error_rad: float  # |q(t_end) - q_target|
    peak_value: float
    final_value: float
    target_value: float
    step_amplitude: float
    sample_count: int


@dataclass(frozen=True)
class StepResponseTrace:
    """Raw recorded trace from a single-joint step-response run."""

    joint_name: str
    joint_ir_role: str
    sample_rate_hz: float
    t: np.ndarray            # shape (N,)
    q: np.ndarray            # shape (N,) — actual position
    qd: np.ndarray           # shape (N,) — actual velocity
    tau: np.ndarray          # shape (N,) — applied torque
    target: float            # constant step target value
    initial: float           # initial position before step
    kp: float
    kd: float
    metrics: StepResponseMetrics


def _extract_metrics(
    *,
    t: np.ndarray,
    q: np.ndarray,
    target: float,
    initial: float,
    sample_rate_hz: float,
) -> StepResponseMetrics:
    """Curve-fit the trace into rise / overshoot / settling / SS-error metrics."""
    step_amp = float(target - initial)
    if step_amp == 0.0:
        return StepResponseMetrics(
            rise_time_s=-1.0,
            overshoot_pct=-1.0,
            settling_time_s=-1.0,
            steady_state_error_rad=0.0,
            peak_value=float(q[-1]) if len(q) else 0.0,
            final_value=float(q[-1]) if len(q) else 0.0,
            target_value=float(target),
            step_amplitude=0.0,
            sample_count=int(len(q)),
        )

    sign = 1.0 if step_amp > 0 else -1.0
    # Normalize to a 0->1 step in the direction of motion.
    norm = (q - initial) * sign / abs(step_amp)

    # Rise time: 10% -> 90%
    rise_time = -1.0
    try:
        idx10 = int(np.argmax(norm >= 0.1))
        idx90 = int(np.argmax(norm >= 0.9))
        if idx10 < idx90 and norm[idx10] >= 0.1 and norm[idx90] >= 0.9:
            rise_time = float(t[idx90] - t[idx10])
    except (ValueError, IndexError):
        rise_time = -1.0

    # Overshoot: peak above 1.0 (normalized)
    peak_norm = float(np.max(norm))
    overshoot_pct = max(0.0, (peak_norm - 1.0)) * 100.0
    peak_value = float(initial + sign * peak_norm * abs(step_amp))

    # Settling time: last time the response left the ±2% band
    band_lo, band_hi = 0.98, 1.02
    outside_band = (norm < band_lo) | (norm > band_hi)
    if np.any(outside_band):
        last_outside = int(np.where(outside_band)[0][-1])
        if last_outside + 1 < len(t):
            settling_time = float(t[last_outside + 1] - t[0])
        else:
            settling_time = -1.0  # never settled
    else:
        settling_time = 0.0

    final_value = float(q[-1])
    ss_error = float(abs(final_value - target))

    return StepResponseMetrics(
        rise_time_s=rise_time,
        overshoot_pct=overshoot_pct,
        settling_time_s=settling_time,
        steady_state_error_rad=ss_error,
        peak_value=peak_value,
        final_value=final_value,
        target_value=float(target),
        step_amplitude=float(step_amp),
        sample_count=int(len(q)),
    )


def simulate_mujoco_step(
    *,
    mjcf_path: Path,
    joint_name: str,
    joint_ir_role: str,
    nominal_qpos: Optional[np.ndarray],
    target_joint_index: int,
    kp_array: np.ndarray,
    kd_array: np.ndarray,
    step_amplitude_rad: float = 0.1,
    duration_s: float = 1.0,
    sample_rate_hz: float = 500.0,
    sim_dt: Optional[float] = None,
    disable_gravity: bool = True,
) -> StepResponseTrace:
    """Run a single-joint step-response in MuJoCo with the supplied PD gains.

    Locks every joint except the target at the nominal pose, commands a
    position step at t=0, integrates for ``duration_s``, and returns the
    recorded trace plus extracted metrics.

    Parameters
    ----------
    mjcf_path:
        Path to the robot's MJCF on disk.
    joint_name:
        Physical name of the joint to perturb.
    joint_ir_role:
        IR role of ``joint_name`` (stored in the trace for the report).
    nominal_qpos:
        Full-length qpos at which to lock the other joints. ``None``
        falls back to the model's keyframe-0 or qpos0.
    target_joint_index:
        Index of ``joint_name`` inside the env's actuator order — used
        to look up ``kp_array[idx]`` and ``kd_array[idx]``.
    kp_array, kd_array:
        Per-joint PD gains (from :class:`JointPDGains`).
    step_amplitude_rad:
        Magnitude of the position step applied to the target joint at
        t=0.
    duration_s:
        How long to integrate after the step.
    sample_rate_hz:
        Output trace resolution. The sim runs at ``sim_dt`` (default
        0.001 s); samples are decimated to ``sample_rate_hz``.
    sim_dt:
        Integrator step. ``None`` reads ``model.opt.timestep`` from the
        MJCF.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("step_response requires the 'mujoco' package") from exc

    path_obj = Path(mjcf_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"step_response: MJCF not found at {path_obj!r}")

    model = mujoco.MjModel.from_xml_path(str(path_obj))
    data = mujoco.MjData(model)

    if sim_dt is not None:
        model.opt.timestep = float(sim_dt)
    dt = float(model.opt.timestep)
    if dt <= 0.0:
        raise RuntimeError(f"step_response: model.opt.timestep={dt} invalid")
    # Calibration measures PD parameterization correctness. Gravity on a
    # serially-chained articulation injects a torque proportional to the
    # link weight × moment arm into every joint, which then competes with
    # the PD term and produces a steady-state offset = τ_grav / kp. That
    # offset is a function of MJCF mass and pose, NOT of (omega_n, zeta).
    # Disabling gravity isolates the PD response so the calibration
    # actually flags parameterization errors (the user-tunable knob this
    # node exists to gate) instead of gravity errors (which belong on
    # the Robot Asset node, see the cloud spec yaml lines 305-309).
    if disable_gravity:
        model.opt.gravity[:] = 0.0

    # Seed nominal pose.
    if nominal_qpos is None:
        if model.nkey > 0:
            data.qpos[:] = model.key_qpos[0]
        else:
            data.qpos[:] = model.qpos0
    else:
        arr = np.asarray(nominal_qpos, dtype=np.float64)
        if arr.shape != (int(model.nq),):
            raise ValueError(
                f"step_response: nominal_qpos shape {arr.shape} != model.nq=({model.nq},)"
            )
        data.qpos[:] = arr
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Resolve the target joint's qpos index.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"step_response: joint {joint_name!r} not found in MJCF")
    qpos_start = int(model.jnt_qposadr[jid])
    initial_q = float(data.qpos[qpos_start])
    target_q = initial_q + float(step_amplitude_rad)

    # Locate the actuator for this joint (needed to know which ctrl index
    # to write into during the rollout — the step is applied as a PD
    # torque via the actuator's gear=1 motor port).
    actuator_targets: List[Tuple[int, int, float, float]] = []  # (ctrl_idx, qpos_idx, target_q, initial_q)
    for ai in range(int(model.nu)):
        ajid = int(model.actuator_trnid[ai, 0])
        if 0 <= ajid < model.njnt:
            this_qpos = int(model.jnt_qposadr[ajid])
            if ajid == jid:
                actuator_targets.append((ai, this_qpos, target_q, initial_q))
            else:
                # Lock other actuated joints at their nominal q.
                hold_q = float(data.qpos[this_qpos])
                actuator_targets.append((ai, this_qpos, hold_q, hold_q))
    if not any(t == jid for t in (int(model.actuator_trnid[ai, 0]) for ai in range(int(model.nu)))):
        raise RuntimeError(
            f"step_response: joint {joint_name!r} is not driven by any MJCF "
            f"actuator. Step-response calibration requires an actuated joint."
        )

    if target_joint_index < 0 or target_joint_index >= len(kp_array):
        raise IndexError(
            f"step_response: target_joint_index={target_joint_index} out of range "
            f"for kp_array of length {len(kp_array)}"
        )

    # Pre-allocate output arrays.
    n_substeps = max(1, int(round(duration_s / dt)))
    sample_every = max(1, int(round(1.0 / (sample_rate_hz * dt))))
    n_samples = (n_substeps + sample_every - 1) // sample_every
    t_out = np.empty(n_samples, dtype=np.float64)
    q_out = np.empty(n_samples, dtype=np.float64)
    qd_out = np.empty(n_samples, dtype=np.float64)
    tau_out = np.empty(n_samples, dtype=np.float64)

    # Pre-read actuator gears so we can map joint-space torque (what
    # kp/kd produce) into ctrl-space (what MuJoCo applies after the gear
    # multiplier). MJCF <motor gear="K"> gives: joint_torque = gear * ctrl,
    # so ctrl = joint_torque / gear.
    actuator_gear = np.asarray(model.actuator_gear[:, 0], dtype=np.float64)
    # Guard against accidental zero gears in malformed MJCFs.
    actuator_gear = np.where(np.abs(actuator_gear) < 1e-12, 1.0, actuator_gear)

    # "Lock other joints" via inertia: bump non-target bodies' mass by
    # a large factor so they're effectively immovable under the target
    # joint's reaction forces. This is physically valid (a body with
    # huge inertia doesn't move under finite forces) and avoids the
    # numerical artifacts of qpos-resetting after each integrator step.
    # We preserve the target joint's parent body chain at original mass
    # so the target's own m_eff is unchanged.
    target_qpos_idx = qpos_start
    target_dof_idx = int(model.jnt_dofadr[jid])
    # Walk up from the target joint's body to find its parent chain;
    # those bodies must NOT have their inertia scaled (the target's
    # m_eff depends on them via mj_fullM).
    target_body_id = int(model.jnt_bodyid[jid])
    parent_chain: set = set()
    cur = target_body_id
    while cur > 0:
        parent_chain.add(cur)
        cur = int(model.body_parentid[cur])
    parent_chain.add(0)  # world
    # Scale every other body's mass by 1e6 — it acts as a rigid base
    # that the target's reaction forces cannot perturb meaningfully.
    INERTIA_LOCK_FACTOR = 1e6
    for bid in range(int(model.nbody)):
        if bid in parent_chain:
            continue
        # Sibling / downstream link — make immovable. We multiply
        # mass and the diagonal inertia tensor; mj_fullM will see a
        # huge effective inertia for any DoF rooted at this body.
        model.body_mass[bid] *= INERTIA_LOCK_FACTOR
        model.body_inertia[bid] *= INERTIA_LOCK_FACTOR

    sample_idx = 0
    for step_i in range(n_substeps):
        # PD torque on the TARGET joint only.
        q_now = float(data.qpos[target_qpos_idx])
        qd_now = float(data.qvel[target_dof_idx])
        tau = (
            float(kp_array[target_joint_index]) * (target_q - q_now)
            - float(kd_array[target_joint_index]) * qd_now
        )
        data.ctrl[:] = 0.0
        data.ctrl[target_joint_index] = tau / float(actuator_gear[target_joint_index])

        mujoco.mj_step(model, data)

        if step_i % sample_every == 0 and sample_idx < n_samples:
            t_out[sample_idx] = (step_i + 1) * dt
            q_out[sample_idx] = float(data.qpos[qpos_start])
            qd_out[sample_idx] = float(data.qvel[int(model.jnt_dofadr[jid])])
            tau_out[sample_idx] = float(data.ctrl[target_joint_index])
            sample_idx += 1

    # Trim to actual samples written.
    t_out = t_out[:sample_idx]
    q_out = q_out[:sample_idx]
    qd_out = qd_out[:sample_idx]
    tau_out = tau_out[:sample_idx]

    metrics = _extract_metrics(
        t=t_out,
        q=q_out,
        target=target_q,
        initial=initial_q,
        sample_rate_hz=sample_rate_hz,
    )

    return StepResponseTrace(
        joint_name=joint_name,
        joint_ir_role=joint_ir_role,
        sample_rate_hz=sample_rate_hz,
        t=t_out,
        q=q_out,
        qd=qd_out,
        tau=tau_out,
        target=target_q,
        initial=initial_q,
        kp=float(kp_array[target_joint_index]),
        kd=float(kd_array[target_joint_index]),
        metrics=metrics,
    )


@dataclass(frozen=True)
class StepResponseComparison:
    """Side-by-side comparison of one joint between MuJoCo and PhysX traces."""

    joint_name: str
    joint_ir_role: str
    mujoco_metrics: StepResponseMetrics
    physx_metrics: Optional[StepResponseMetrics]   # None if PhysX side unavailable
    rise_time_relative_diff: float
    overshoot_absolute_diff: float
    steady_state_error_rad_abs: float
    verdict: str   # "pass" / "warn" / "fail"


# Default tolerance thresholds.
DEFAULT_RISE_TIME_RELATIVE_DIFF = 0.15
DEFAULT_OVERSHOOT_ABSOLUTE_DIFF = 0.05  # = 5 percentage points
DEFAULT_SS_ERROR_RAD_ABS = 0.005


def compare_step_responses(
    *,
    mujoco_metrics: StepResponseMetrics,
    physx_metrics: Optional[StepResponseMetrics],
    joint_name: str,
    joint_ir_role: str,
    rise_time_relative_diff: float = DEFAULT_RISE_TIME_RELATIVE_DIFF,
    overshoot_absolute_diff: float = DEFAULT_OVERSHOOT_ABSOLUTE_DIFF,
    ss_error_rad_abs: float = DEFAULT_SS_ERROR_RAD_ABS,
) -> StepResponseComparison:
    """Compare two engines' step-response metrics for one joint, produce verdict.

    When ``physx_metrics`` is ``None`` (caller didn't or couldn't run
    the PhysX side), the verdict is based on MuJoCo's standalone
    metrics — settling and steady-state error against the parameterized
    targets.
    """
    if physx_metrics is None:
        # Standalone MuJoCo verdict — accept if rise time, overshoot,
        # and SS error are within reasonable single-engine bounds.
        ok_rise = 0.0 < mujoco_metrics.rise_time_s < 5.0
        ok_overshoot = mujoco_metrics.overshoot_pct < 20.0  # generous
        ok_ss = mujoco_metrics.steady_state_error_rad < ss_error_rad_abs
        if ok_rise and ok_overshoot and ok_ss:
            verdict = "pass"
        elif ok_ss:
            verdict = "warn"
        else:
            verdict = "fail"
        return StepResponseComparison(
            joint_name=joint_name,
            joint_ir_role=joint_ir_role,
            mujoco_metrics=mujoco_metrics,
            physx_metrics=None,
            rise_time_relative_diff=-1.0,
            overshoot_absolute_diff=-1.0,
            steady_state_error_rad_abs=mujoco_metrics.steady_state_error_rad,
            verdict=verdict,
        )

    # Cross-engine comparison.
    if mujoco_metrics.rise_time_s > 0.0 and physx_metrics.rise_time_s > 0.0:
        rel_diff = abs(mujoco_metrics.rise_time_s - physx_metrics.rise_time_s) / max(
            physx_metrics.rise_time_s, 1e-6
        )
    else:
        rel_diff = -1.0

    abs_overshoot = abs(
        mujoco_metrics.overshoot_pct - physx_metrics.overshoot_pct
    ) / 100.0  # convert percentage points to fraction

    abs_ss = max(
        mujoco_metrics.steady_state_error_rad,
        physx_metrics.steady_state_error_rad,
    )

    pass_rise = rel_diff < 0.0 or rel_diff <= rise_time_relative_diff
    pass_over = abs_overshoot <= overshoot_absolute_diff
    pass_ss = abs_ss <= ss_error_rad_abs
    hard_fail_rise = rel_diff > 2.0 * rise_time_relative_diff
    hard_fail_over = abs_overshoot > 2.0 * overshoot_absolute_diff
    hard_fail_ss = abs_ss > 5.0 * ss_error_rad_abs

    if pass_rise and pass_over and pass_ss:
        verdict = "pass"
    elif hard_fail_rise or hard_fail_over or hard_fail_ss:
        verdict = "fail"
    else:
        verdict = "warn"

    return StepResponseComparison(
        joint_name=joint_name,
        joint_ir_role=joint_ir_role,
        mujoco_metrics=mujoco_metrics,
        physx_metrics=physx_metrics,
        rise_time_relative_diff=rel_diff,
        overshoot_absolute_diff=abs_overshoot,
        steady_state_error_rad_abs=abs_ss,
        verdict=verdict,
    )
