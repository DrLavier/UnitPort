# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Probe generators for the Stage-1 measurement (both probe families).

Designed contact scenarios (isolate the Type-II contact dimensions named in the
design §4.2):
  * ``drop_to_contact``   — impact transient + effective contact stiffness
    (swept over drop height × ``solref``/``solimp``);
  * ``lateral_slip``      — friction-cone behaviour (swept over lateral launch
    velocity × sliding μ);
Trajectory replay (the realistic-distribution family):
  * ``trajectory_replay`` — replay a recorded ``(state, action)`` rollout's
    joint torques open-loop; when no rollout is supplied a generic sinusoidal
    excitation stands in (logged as a NOTE — it is an excitation, not a policy
    distribution).

All generators are plant-agnostic: they take the actor's ``qpos0`` + dims and a
free-base detection, and never branch on brand/model.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

from .mujoco_driver import nominal_qpos as _nominal_qpos
from .protocol import AlignedPlant, ContactCfg, Probe, ProbeSet

# Default IC-perturbation scale (rad / rad·s⁻¹ / m) for the sensitivity test.
# Sized to the REAL reset-time pose/velocity jitter a policy faces at episode
# reset (NOT an arbitrarily large kick) — so the systematic/chaotic verdict
# reflects the IC uncertainty that matters in deployment (硬化二). 1e-4 (the old
# value) was far too small for chaos to diverge over a short rollout.
DEFAULT_IC_PERTURB_SCALE = 0.03


# ---------------------------------------------------------------------------
# plant introspection
# ---------------------------------------------------------------------------

def _free_base_qposadr(actor: Any) -> Optional[int]:
    """qpos address of the free (floating-base) joint, or None if the model has
    no free joint (fixed-base — contact scenarios are not applicable)."""
    import mujoco

    model = actor.mj_model
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid])
    return None


def _free_base_dofadr(actor: Any) -> Optional[int]:
    import mujoco

    model = actor.mj_model
    for jid in range(int(model.njnt)):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_dofadr[jid])
    return None


# ---------------------------------------------------------------------------
# designed contact scenarios
# ---------------------------------------------------------------------------

def drop_to_contact_probes(
    actor: Any,
    *,
    heights: Sequence[float] = (0.05, 0.15, 0.30),
    solrefs: Sequence[Optional[List[float]]] = (None, [0.01, 1.0], [0.02, 1.0]),
    n_steps: int = 250,
    n_repeats: int = 5,
    ic_perturb_scale: float = DEFAULT_IC_PERTURB_SCALE,
) -> List[Probe]:
    """Drop the floating base from each height with ZERO joint torque; sweep the
    contact ``solref`` (timeconst, dampratio). The impact transient + settled
    penetration encode the effective contact stiffness — the residual between
    engines is exactly the Type-II quantity Layer 2 must randomize over."""
    qadr = _free_base_qposadr(actor)
    if qadr is None:
        return []
    nq, nv, nu = int(actor.mj_model.nq), int(actor.mj_model.nv), int(actor.mj_model.nu)
    qpos0 = _nominal_qpos(actor)
    z0 = qpos0[qadr + 2]
    zero_tau = [[0.0] * nu for _ in range(n_steps)]
    probes: List[Probe] = []
    for hi, h in enumerate(heights):
        for si, sr in enumerate(solrefs):
            qp = list(qpos0)
            qp[qadr + 2] = z0 + float(h)
            probes.append(Probe(
                probe_id=f"drop_h{hi}_s{si}",
                scenario="drop_to_contact",
                dimension="contact_stiffness",
                init_qpos=qp,
                init_qvel=[0.0] * nv,
                torque_seq=zero_tau,
                contact_cfg=ContactCfg(solref=sr),
                n_repeats=n_repeats,
                ic_perturb_scale=ic_perturb_scale,
            ))
    return probes


def lateral_slip_probes(
    actor: Any,
    *,
    lateral_vels: Sequence[float] = (0.3, 0.8, 1.5),
    frictions: Sequence[float] = (0.4, 0.8, 1.2),
    n_steps: int = 250,
    n_repeats: int = 5,
    ic_perturb_scale: float = DEFAULT_IC_PERTURB_SCALE,
) -> List[Probe]:
    """Launch the settled base with a lateral velocity under ZERO joint torque;
    sweep the sliding μ. The deceleration / slip distance encodes the friction
    cone's effective behaviour (pyramid vs ellipsoid is a Type-II difference)."""
    qadr = _free_base_qposadr(actor)
    dadr = _free_base_dofadr(actor)
    if qadr is None or dadr is None:
        return []
    nq, nv, nu = int(actor.mj_model.nq), int(actor.mj_model.nv), int(actor.mj_model.nu)
    qpos0 = _nominal_qpos(actor)
    zero_tau = [[0.0] * nu for _ in range(n_steps)]
    probes: List[Probe] = []
    for vi, v in enumerate(lateral_vels):
        for fi, mu in enumerate(frictions):
            qv = [0.0] * nv
            qv[dadr + 0] = float(v)   # base linear-x velocity (slide direction)
            probes.append(Probe(
                probe_id=f"slip_v{vi}_f{fi}",
                scenario="lateral_slip",
                dimension="friction_cone",
                init_qpos=list(qpos0),
                init_qvel=qv,
                torque_seq=zero_tau,
                contact_cfg=ContactCfg(friction=float(mu)),
                n_repeats=n_repeats,
                ic_perturb_scale=ic_perturb_scale,
            ))
    return probes


# ---------------------------------------------------------------------------
# trajectory replay
# ---------------------------------------------------------------------------

def trajectory_replay_probes(
    actor: Any,
    *,
    rollout_torques: Optional[Sequence[Sequence[float]]] = None,
    n_steps: int = 250,
    n_repeats: int = 5,
    ic_perturb_scale: float = DEFAULT_IC_PERTURB_SCALE,
    excitation_amp: float = 2.0,
) -> List[Probe]:
    """Replay a recorded policy rollout's joint torques open-loop. When
    ``rollout_torques`` is None, synthesize a generic multi-frequency sinusoidal
    excitation (per-joint phase offset) — flagged as an excitation, NOT a policy
    distribution — so the general residual spectrum is still sampled."""
    nq, nv, nu = int(actor.mj_model.nq), int(actor.mj_model.nv), int(actor.mj_model.nu)
    qpos0 = _nominal_qpos(actor)

    if rollout_torques is not None:
        seq = [[float(x) for x in row] for row in rollout_torques]
        if not seq:
            raise ValueError("trajectory_replay_probes: rollout_torques is empty")
        if any(len(row) != nu for row in seq):
            raise ValueError(
                f"trajectory_replay_probes: rollout torque width != model nu ({nu})"
            )
        note = "policy_rollout"
    else:
        # Synthetic excitation: τ_j(t) = A·sin(ω_j t + φ_j), ω/φ spread per joint.
        seq = []
        for t in range(n_steps):
            row = []
            for j in range(nu):
                omega = 2.0 * math.pi * (0.5 + 0.25 * (j % 4))
                phase = (j * math.pi) / max(nu, 1)
                row.append(excitation_amp * math.sin(omega * t * 0.002 + phase))
            seq.append(row)
        note = "synthetic_excitation"

    return [Probe(
        probe_id=f"traj_{note}",
        scenario="trajectory_replay",
        dimension="trajectory",
        init_qpos=list(qpos0),
        init_qvel=[0.0] * nv,
        torque_seq=seq,
        contact_cfg=ContactCfg(),
        n_repeats=n_repeats,
        ic_perturb_scale=ic_perturb_scale,
    )]


# ---------------------------------------------------------------------------
# default battery
# ---------------------------------------------------------------------------

def build_default_probe_set(
    actor: Any,
    plant: AlignedPlant,
    *,
    n_repeats: int = 5,
    ic_perturb_scale: float = DEFAULT_IC_PERTURB_SCALE,
    rollout_torques: Optional[Sequence[Sequence[float]]] = None,
) -> ProbeSet:
    """Assemble the standard Stage-1 battery: drop + slip + trajectory replay.

    ``plant.joint_names`` is filled from the actor (the canonical order the
    torque/qpos joint-part is in) so the Kit launcher can map by name."""
    from .mujoco_driver import actuator_joint_names, nominal_actuator_joint_pos

    nq, nv, nu = int(actor.mj_model.nq), int(actor.mj_model.nv), int(actor.mj_model.nu)
    # Per-actuator joint names (torque-column order), NOT the full qpos joint
    # list — so the Kit launcher maps torque columns to USD joints by name.
    plant.joint_names = actuator_joint_names(actor)
    # Nominal stance per actuator joint → ArticulationCfg default (limit-valid).
    plant.default_joint_pos = nominal_actuator_joint_pos(actor)

    probes: List[Probe] = []
    probes += drop_to_contact_probes(actor, n_repeats=n_repeats, ic_perturb_scale=ic_perturb_scale)
    probes += lateral_slip_probes(actor, n_repeats=n_repeats, ic_perturb_scale=ic_perturb_scale)
    probes += trajectory_replay_probes(
        actor, rollout_torques=rollout_torques, n_repeats=n_repeats,
        ic_perturb_scale=ic_perturb_scale)
    if not probes:
        raise ValueError(
            "build_default_probe_set: no probes generated — a fixed-base plant "
            "(no free joint) has no contact scenarios; supply rollout_torques "
            "for a trajectory-only measurement, or use a floating-base robot."
        )
    return ProbeSet(
        aligned_plant=plant, probes=probes, n_qpos=nq, n_qvel=nv, nu=nu)


__all__ = [
    "drop_to_contact_probes",
    "lateral_slip_probes",
    "trajectory_replay_probes",
    "build_default_probe_set",
]
