# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MJCF model preparation for UnitPort's mass-weighted PD control.

UnitPort routes ALL PD math through a single authority: the per-joint
``(kp, kd)`` are derived from the canonical ``(omega_n, zeta)`` via
:mod:`application.physics.mujoco_gain_solver` and applied in Python as a
pure-torque ``ctrl`` write (``<motor>`` actuators). For the realized
closed-loop response to actually match the designed ``(omega_n, zeta)``,
**the PD kd must be the ONLY damping acting on each controlled joint.**

Menagerie-style MJCFs ship a passive ``<joint damping=... frictionloss=...>``
on every leg joint (Unitree Go2: ``damping="2"``, ``frictionloss="0.2"``).
MuJoCo applies these passive dissipative forces during ``mj_step`` *in
addition to* the PD torque written to ``ctrl`` — they are NOT mediated by
the actuator and so survive :func:`_neutralize_mjcf_actuators` (which only
rewrites the actuator gain/bias). The PhysX/IsaacLab side has no equivalent
passive joint dissipation: its ``ImplicitActuatorCfg`` damping IS the
designed kd and ``friction`` is null (see a bundle's ``env.yaml``). So a
policy trained at ``zeta=0.8`` in PhysX meets ``zeta≈1.8`` (kd 1.65 + passive
2.0) in MuJoCo — a ~2.2x over-damped, sluggish, "stiff / over-forceful /
won't-walk" sim2sim regression that the PD calibration gate cannot see
(its torque + realized-inertia checks are static gain-identity / armature
checks, blind to a velocity-dependent passive term).

This module neutralizes that gap, mirroring the
``_neutralize_mjcf_actuators`` philosophy: after preparation the Python
PDController is the SOLE source of joint damping, so both engines realize
the same ``(omega_n, zeta)``.

What is and isn't touched
-------------------------
* ``dof_damping`` (viscous) and ``dof_frictionloss`` (Coulomb) on
  **PD-controlled** joints → zeroed. PhysX has neither; PD kd is the only
  damping by design.
* ``dof_armature`` → **kept untouched.** It is the reflected-rotor inertia,
  matched on both engines (Go2: 0.01 on both) and already folded into
  ``m_eff`` by :func:`effective_inertia_diag`. Zeroing it would desync the
  realized inertia from the inertia the gains were solved against.
* Non-actuated dofs (free-joint base, auxiliary passive joints) → left
  alone. We target only dofs reachable from a joint-transmission actuator,
  so we never strip damping the author put on something the PD does not
  drive.

Idempotent: a second call finds nothing to strip and is a silent no-op.
Shared by the deploy runtime (:mod:`...simulation.mujoco.mj_actor`), the
SB3 training env (:class:`...envs.generic_mujoco_env.GenericMujocoEnv`),
and the export-time calibration audit so the three realizations cannot
drift (CLAUDE.md §10/§11).
"""
from __future__ import annotations

import logging
from typing import Any, List, Tuple

log = logging.getLogger(__name__)


def pd_controlled_dofs(mj_model: Any) -> List[int]:
    """Return the dof indices driven by a joint-transmission actuator.

    These are exactly the joints whose damping the Python PDController is
    responsible for; any passive damping on them double-counts against the
    designed kd. Actuators with a non-joint transmission (tendon, site, …)
    contribute no dof here and are ignored.
    """
    import mujoco

    joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
    dofs: List[int] = []
    for i in range(int(mj_model.nu)):
        if int(mj_model.actuator_trntype[i]) != joint_trn:
            continue
        jid = int(mj_model.actuator_trnid[i, 0])
        if jid < 0:
            continue
        dof_adr = int(mj_model.jnt_dofadr[jid])
        # Hinge / slide joints are single-dof; that is the entire UnitPort
        # actuated-joint universe (ball/free joints are never PD-driven).
        if dof_adr >= 0:
            dofs.append(dof_adr)
    return dofs


def neutralize_mjcf_passive_dissipation(
    mj_model: Any, mjcf_label: str = ""
) -> Tuple[List[int], float, float]:
    """Zero passive ``dof_damping`` / ``dof_frictionloss`` on PD-driven dofs.

    Leaves ``dof_armature`` untouched (it belongs to ``m_eff``). Returns
    ``(stripped_dofs, max_damping_stripped, max_frictionloss_stripped)`` for
    callers that want to log or assert. Re-running is a no-op.
    """
    dofs = pd_controlled_dofs(mj_model)
    if not dofs:
        return [], 0.0, 0.0

    stripped: List[int] = []
    max_damp = 0.0
    max_fric = 0.0
    for d in dofs:
        damp = float(mj_model.dof_damping[d])
        fric = float(mj_model.dof_frictionloss[d])
        if damp == 0.0 and fric == 0.0:
            continue
        max_damp = max(max_damp, abs(damp))
        max_fric = max(max_fric, abs(fric))
        mj_model.dof_damping[d] = 0.0
        mj_model.dof_frictionloss[d] = 0.0
        stripped.append(d)

    if stripped:
        log.info(
            "MJCF PD prep: neutralized passive joint dissipation on %d "
            "PD-driven dof(s) (max damping=%.3f, max frictionloss=%.3f "
            "stripped). UnitPort's PDController is now the sole source of "
            "joint damping, so MuJoCo realizes the designed (omega_n, zeta) "
            "and matches the PhysX-trained policy (CLAUDE.md §10). "
            "dof_armature kept (folded into m_eff). Source MJCF: %s",
            len(stripped), max_damp, max_fric, mjcf_label or "?",
        )
    return stripped, max_damp, max_fric
