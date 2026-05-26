# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab / PhysX gain solver.

PhysX's ``ImplicitActuatorCfg`` applies ``stiffness`` and ``damping`` in
**real torque units** (N·m/rad and N·m·s/rad) — the PD law is

    tau = stiffness * (q_des - q) + damping * (qd_des - qd)

with **no inertia normalization**. This is verified against IsaacLab's own
source: ``ImplicitActuator.compute`` and ``IdealPDActuator.compute``
(``isaaclab/actuators/actuator_pd.py``) use exactly that formula, and the
implicit integration only changes the *stability* of the solve, not the
units. It is the same convention MuJoCo's PD and real hardware Kp/Kd use —
which is why legged sim-to-real transfers gains verbatim.

Therefore, to realize a target ``(omega_n, zeta)`` on a joint with
effective inertia ``m_eff``, the gains must be mass-weighted just like the
MuJoCo side:

    kp_j = m_eff_j * omega_n_j ** 2
    kd_j = 2 * zeta_j * sqrt(kp_j * m_eff_j)
                       = 2 * zeta_j * omega_n_j * m_eff_j

``m_eff`` is supplied by the caller (the env compiler), computed once via
:func:`application.physics.mujoco_gain_solver.effective_inertia_diag` from
the robot's MJCF at the nominal stance. Feeding **both** engine solvers the
same ``m_eff`` + the same formula makes ``physx_kp == mujoco_kp`` by
construction, so a policy trained in PhysX meets identical real-unit gains
in MuJoCo verification.

History: this solver previously used the unit-mass form ``kp = omega_n**2``
on the false premise that "PhysX renormalizes by articulation inertia."
IsaacLab does no such thing; that asymmetry (unit-mass PhysX vs mass-weighted
MuJoCo) silently scaled low-inertia joints — Spot's knee 65x softer in
MuJoCo than in training — and collapsed transferred policies. See
CLAUDE.md §10.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .joint_group_resolver import expand_groups_to_joints
from .pd_param import JointPDGains, PDParam


def solve(
    *,
    joint_order_physical: List[str],
    joint_ir_roles: List[str],
    pd_param: PDParam,
    m_eff: List[float],
) -> JointPDGains:
    """Derive PhysX-side ``(kp, kd)`` per joint, mass-weighted by ``m_eff``.

    Parameters
    ----------
    joint_order_physical:
        Physical joint names in the engine's order (e.g. MJCF actuator
        order on the USD asset). Used only for plumbing — the math is
        IR-role-driven.
    joint_ir_roles:
        Parallel list of IR roles for each physical joint.
    pd_param:
        Source of truth for ``(omega_n, zeta)`` per group.
    m_eff:
        Per-joint effective inertia ``M_diag`` at the nominal stance,
        aligned with ``joint_order_physical``. Produced by
        :func:`application.physics.mujoco_gain_solver.effective_inertia_diag`
        and shared with the MuJoCo solver so both engines agree.

    Returns
    -------
    JointPDGains
        ``engine="physx"``, with ``derivation`` carrying per-joint
        ``omega_n``, ``zeta``, ``m_eff``, and the resolved group id.
    """
    n = len(joint_order_physical)
    if len(joint_ir_roles) != n:
        raise ValueError(
            f"physx_gain_solver: joint_order_physical has {n} entries but "
            f"joint_ir_roles has {len(joint_ir_roles)}"
        )
    if len(m_eff) != n:
        raise ValueError(
            f"physx_gain_solver: m_eff has {len(m_eff)} entries but "
            f"joint_order_physical has {n}. The caller must pass the "
            f"effective-inertia diagonal aligned with the joint order "
            f"(see mujoco_gain_solver.effective_inertia_diag)."
        )

    role_to_group = expand_groups_to_joints(pd_param, joint_ir_roles)

    kp: List[float] = []
    kd: List[float] = []
    derivation_per_joint: List[Dict[str, Any]] = []
    for idx, (role, group) in enumerate(role_to_group):
        m_eff_j = float(m_eff[idx])
        if not math.isfinite(m_eff_j) or m_eff_j <= 0.0:
            raise ValueError(
                f"physx_gain_solver: joint index {idx} "
                f"({joint_order_physical[idx]!r}) has non-finite or "
                f"non-positive m_eff={m_eff_j}. Effective inertia must be "
                f"positive — check the nominal stance / MJCF inertia."
            )
        omega_n = float(group.omega_n)
        zeta = float(group.zeta)
        kp_j = m_eff_j * omega_n * omega_n
        kd_j = 2.0 * zeta * math.sqrt(kp_j * m_eff_j)
        kp.append(kp_j)
        kd.append(kd_j)
        derivation_per_joint.append({
            "role": role,
            "group": group.name,
            "omega_n": omega_n,
            "zeta": zeta,
            "m_eff": m_eff_j,
        })

    return JointPDGains(
        joint_ir_roles=list(joint_ir_roles),
        joint_physical_names=list(joint_order_physical),
        kp=kp,
        kd=kd,
        engine="physx",
        derivation={
            "method": "full_mass_matrix",
            "formula_kp": "m_eff * omega_n ** 2",
            "formula_kd": "2 * zeta * sqrt(kp * m_eff)",
            "per_joint": derivation_per_joint,
        },
    )
