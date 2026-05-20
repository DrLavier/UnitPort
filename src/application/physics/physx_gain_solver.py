"""IsaacLab / PhysX gain solver.

PhysX's ``ImplicitActuatorCfg`` internally normalizes by the articulation
inertia, so the user-facing ``stiffness`` and ``damping`` map to a
unit-mass second-order system:

    kp_j = omega_n_j ** 2
    kd_j = 2 * zeta_j * omega_n_j

This means PhysX gains scale only with the parameterization — there's no
mass-matrix call. The solver still exists (rather than inlining the
formula at callsites) because:

1. It gives both engines a typed, identical surface so domain
   randomization and provenance diagnostics flow through one path.
2. The PhysX gain provenance (``derivation`` dict) carries the
   omega_n / zeta per joint, mirroring the MuJoCo solver's ``m_eff``
   diagnostics. Calibration reports compare side-by-side.
3. If PhysX changes its normalization convention in a future Isaac Lab
   release, the fix is one file.
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
) -> JointPDGains:
    """Derive PhysX-side ``(kp, kd)`` per joint.

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

    Returns
    -------
    JointPDGains
        ``engine="physx"``, with ``derivation`` carrying per-joint
        ``omega_n``, ``zeta``, and the resolved group id.
    """
    if len(joint_order_physical) != len(joint_ir_roles):
        raise ValueError(
            f"physx_gain_solver: joint_order_physical has "
            f"{len(joint_order_physical)} entries but joint_ir_roles has "
            f"{len(joint_ir_roles)}"
        )

    role_to_group = expand_groups_to_joints(pd_param, joint_ir_roles)

    kp: List[float] = []
    kd: List[float] = []
    derivation_per_joint: List[Dict[str, Any]] = []
    for role, group in role_to_group:
        omega_n = float(group.omega_n)
        zeta = float(group.zeta)
        kp_j = omega_n * omega_n
        kd_j = 2.0 * zeta * omega_n
        kp.append(kp_j)
        kd.append(kd_j)
        derivation_per_joint.append({
            "role": role,
            "group": group.name,
            "omega_n": omega_n,
            "zeta": zeta,
        })

    return JointPDGains(
        joint_ir_roles=list(joint_ir_roles),
        joint_physical_names=list(joint_order_physical),
        kp=kp,
        kd=kd,
        engine="physx",
        derivation={
            "method": "normalized_inertia",
            "formula_kp": "omega_n ** 2",
            "formula_kd": "2 * zeta * omega_n",
            "per_joint": derivation_per_joint,
        },
    )
