# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo gain solver: mass-matrix-adaptive ``(kp, kd)`` derivation.

MuJoCo's actuators in the UnitPort training stack are pure torque sources
(``<motor>`` with a scalar ``gear``); PD is applied in Python in
:meth:`application.training.envs.generic_mujoco_env.GenericMujocoEnv.step`.
That means the per-joint gain must include the joint's effective mass so
the closed-loop response actually matches the ``(omega_n, zeta)`` target.

Formula:

    m_eff_j = M[dof_j, dof_j]      # diagonal of the FULL inertia matrix at q0
    kp_j   = m_eff_j * omega_n_j ** 2
    kd_j   = 2 * zeta_j * sqrt(kp_j * m_eff_j)

The full (dense) ``M`` carries the off-diagonal coupling between joints,
which is what differentiates this from naive PD on a unit-mass system.
We deliberately read the **diagonal** of the dense matrix (not the
sparse ``qM`` directly) because the dense diagonal includes coupling
contributions that the sparse storage splits across rows.

**Both engines mass-weight.** The same ``m_eff`` (via
:func:`effective_inertia_diag`) feeds the PhysX solver too — PhysX's
``ImplicitActuatorCfg`` applies ``stiffness``/``damping`` in real torque
units (N·m/rad), identical in meaning to this engine's gains and to real
hardware Kp/Kd (verified against IsaacLab's ``IdealPDActuator``/
``ImplicitActuator.compute``: ``tau = kp*(q_des-q) + kd*(qd_des-qd)``,
no inertia term). Deriving both engines' gains from one shared ``m_eff``
+ this one formula makes ``physx_kp == mujoco_kp`` by construction, which
is what lets a policy transfer cleanly from PhysX training to MuJoCo
verification. See :mod:`application.physics.physx_gain_solver`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .joint_group_resolver import expand_groups_to_joints
from .pd_param import JointPDGains, PDParam


# Joint-type semantics — see mujoco/mjmodel.h ``mjtJoint`` enum.
# Public ``mujoco.mjtJoint`` enum values: mjJNT_FREE=0, mjJNT_BALL=1,
# mjJNT_SLIDE=2, mjJNT_HINGE=3. Only single-DOF actuated joints
# (slide/hinge) have an unambiguous "effective mass" — ball joints have
# three coupled rotational DOFs and a free joint has six DOFs, neither
# of which can be reduced to one scalar kp/kd without picking an axis.
_MJTJOINT_HINGE = 3
_MJTJOINT_SLIDE = 2
_SUPPORTED_JOINT_TYPES = {_MJTJOINT_HINGE, _MJTJOINT_SLIDE}


@dataclass(frozen=True)
class EffectiveInertia:
    """Per-joint effective inertia ``M_diag`` evaluated at a nominal stance.

    ``m_eff``, ``dof_index`` are aligned with the requested
    ``joint_order_physical``. ``nominal_qpos_source`` records which
    configuration ``M`` was evaluated at, and ``qpos_ref`` is the FULL
    ``model.nq`` qpos array actually used (so both engine derivations can
    pin to a byte-identical stance — see :func:`qpos_sha256`).
    """

    m_eff: List[float]
    dof_index: List[int]
    nominal_qpos_source: str
    qpos_ref: List[float]

    def qpos_sha256(self) -> str:
        return qpos_sha256(self.qpos_ref)


def qpos_sha256(qpos: Any) -> str:
    """Stable hash of a nominal-stance qpos array, for cross-process parity.

    The compiler stamps this into ``deploy_meta``; the finalizer recomputes
    it and refuses to proceed on mismatch — that is the cross-process form
    of "both solvers got byte-identical qpos_ref" (they run in separate
    processes at different times, so an in-memory ``assert a is b`` is not
    possible). Rounded to 9 d.p. so float-repr noise across numpy/json
    round-trips does not spuriously change the digest.
    """
    import hashlib

    arr = np.asarray(qpos, dtype=np.float64).ravel()
    payload = ",".join(f"{float(v):.9f}" for v in arr).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_inertia_diag(
    *,
    mjcf_path: Path,
    joint_order_physical: List[str],
    nominal_qpos: Optional[np.ndarray],
) -> EffectiveInertia:
    """Compute the dense mass-matrix diagonal ``M[dof_j, dof_j]`` per joint.

    Loads ``mjcf_path``, seeds qpos at the nominal stance (caller-supplied,
    else MJCF keyframe-0, else ``model.qpos0``), runs ``mj_forward`` and
    ``mj_fullM``, and returns the diagonal entry for each joint in
    ``joint_order_physical``.

    This is the single source of effective inertia for BOTH engine gain
    solvers — call it once per engine derivation and pass the result to
    :func:`solve` (MuJoCo) and
    :func:`application.physics.physx_gain_solver.solve` (PhysX) so the two
    engines mass-weight off identical numbers.

    Raises
    ------
    RuntimeError
        ``mujoco`` not importable; joint name not in MJCF; unsupported
        joint type; ``mj_fullM`` produced a non-finite / non-positive
        diagonal entry (singular stance).
    FileNotFoundError
        ``mjcf_path`` does not exist.
    ValueError
        ``nominal_qpos`` has the wrong shape.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "effective_inertia_diag requires the 'mujoco' package — install "
            "via requirements.txt (it's a hard dep of GenericMujocoEnv)."
        ) from exc

    path_obj = Path(mjcf_path)
    if not path_obj.is_file():
        raise FileNotFoundError(
            f"effective_inertia_diag: MJCF not found at {path_obj!r}. The "
            f"caller (env compiler / bundle finalizer) is expected to "
            f"resolve the path before calling."
        )

    model = mujoco.MjModel.from_xml_path(str(path_obj))
    data = mujoco.MjData(model)

    # Seed qpos at the nominal stance.
    if nominal_qpos is None:
        # Prefer the first keyframe if defined (MJCF authors put a
        # "home" pose there); else fall back to model.qpos0.
        if model.nkey > 0:
            data.qpos[:] = model.key_qpos[0]
            source = "model.key_qpos[0]"
        else:
            data.qpos[:] = model.qpos0
            source = "model.qpos0"
    else:
        arr = np.asarray(nominal_qpos, dtype=np.float64)
        if arr.shape != (int(model.nq),):
            raise ValueError(
                f"effective_inertia_diag: nominal_qpos shape {arr.shape} "
                f"does not match model.nq=({model.nq},). The caller must "
                f"build a full qpos including the free root "
                f"(qpos[0:7] = [x, y, z, qw, qx, qy, qz]) and any "
                f"non-actuated joints."
            )
        data.qpos[:] = arr
        source = "caller_supplied"

    # Snapshot the exact stance M is evaluated at (full model.nq), so the
    # caller can pin the other engine's derivation to the same array.
    qpos_ref = [float(v) for v in np.asarray(data.qpos, dtype=np.float64)]

    # Zero velocities — mass matrix is q-only, but mj_forward also
    # populates derived quantities that mj_fullM needs.
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Build the dense mass matrix (nv x nv).
    nv = int(model.nv)
    M_dense = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(model, M_dense, data.qM)

    m_eff_list: List[float] = []
    dof_index_list: List[int] = []
    for physical_name in joint_order_physical:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, physical_name)
        if jid < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(int(model.njnt))
            ]
            raise RuntimeError(
                f"effective_inertia_diag: joint {physical_name!r} not in "
                f"MJCF {path_obj!r}. Available joints: {available}. The "
                f"caller's JointIRResolver and the MJCF appear to disagree "
                f"on joint names — check the registry entry for this robot."
            )
        jtype = int(model.jnt_type[jid])
        if jtype not in _SUPPORTED_JOINT_TYPES:
            raise RuntimeError(
                f"effective_inertia_diag: joint {physical_name!r} has "
                f"mjtJoint type={jtype}, which is not a single-DOF "
                f"slide/hinge actuator. Ball and free joints carry "
                f"multiple DOFs and cannot be reduced to a single "
                f"(kp, kd) — declare per-DOF joints in the MJCF or "
                f"exclude this joint from the PD-controlled action set."
            )
        dof_start = int(model.jnt_dofadr[jid])
        m_eff = float(M_dense[dof_start, dof_start])
        if not math.isfinite(m_eff) or m_eff <= 0.0:
            raise RuntimeError(
                f"effective_inertia_diag: joint {physical_name!r} (dof "
                f"{dof_start}) has non-finite or non-positive effective "
                f"inertia m_eff={m_eff} at the nominal stance. The stance "
                f"is singular or the MJCF inertia values are corrupt — "
                f"check actor.init_pose / MJCF <inertial> tags before "
                f"retrying."
            )
        m_eff_list.append(m_eff)
        dof_index_list.append(dof_start)

    return EffectiveInertia(
        m_eff=m_eff_list,
        dof_index=dof_index_list,
        nominal_qpos_source=source,
        qpos_ref=qpos_ref,
    )


def solve(
    *,
    mjcf_path: Path,
    joint_order_physical: List[str],
    joint_ir_roles: List[str],
    nominal_qpos: Optional[np.ndarray],
    pd_param: PDParam,
) -> JointPDGains:
    """Derive MuJoCo-side ``(kp, kd)`` per joint via ``mj_fullM`` at ``nominal_qpos``.

    Parameters
    ----------
    mjcf_path:
        Path to the robot's MJCF on disk. Loaded with
        ``mujoco.MjModel.from_xml_path``; not modified.
    joint_order_physical:
        Engine-side joint names in the order the policy / env will use
        them. Every name must exist in the MJCF; missing names raise.
    joint_ir_roles:
        Parallel list of IR roles for each physical joint, used to
        look up the matching group via the resolver.
    nominal_qpos:
        Full-length (``model.nq``) qpos at which to evaluate ``M``. If
        ``None``, the model's keyframe-0 (if any) or ``model.qpos0`` is
        used. Passing a non-``None`` array of the wrong length raises.
    pd_param:
        Source of truth for ``(omega_n, zeta)`` per group.

    Raises
    ------
    FileNotFoundError
        ``mjcf_path`` does not exist.
    RuntimeError
        Joint name not in MJCF; unsupported joint type; ``mj_fullM``
        produced a non-positive / non-finite diagonal entry; nominal
        stance is singular.
    """
    if len(joint_order_physical) != len(joint_ir_roles):
        raise ValueError(
            f"mujoco_gain_solver: joint_order_physical has "
            f"{len(joint_order_physical)} entries but joint_ir_roles has "
            f"{len(joint_ir_roles)}"
        )

    # Resolve groups first so a typo / unmapped role fails BEFORE we
    # spin up the MuJoCo loader (cheap fail-fast).
    role_to_group = expand_groups_to_joints(pd_param, joint_ir_roles)

    # Effective inertia at the nominal stance — shared with the PhysX
    # solver so both engines mass-weight off identical numbers.
    inertia = effective_inertia_diag(
        mjcf_path=mjcf_path,
        joint_order_physical=list(joint_order_physical),
        nominal_qpos=nominal_qpos,
    )

    kp: List[float] = []
    kd: List[float] = []
    derivation_per_joint: List[Dict[str, Any]] = []

    for idx, (physical_name, (role, group)) in enumerate(
        zip(joint_order_physical, role_to_group)
    ):
        m_eff = inertia.m_eff[idx]
        omega_n = float(group.omega_n)
        zeta = float(group.zeta)
        kp_j = m_eff * omega_n * omega_n
        kd_j = 2.0 * zeta * math.sqrt(kp_j * m_eff)
        kp.append(kp_j)
        kd.append(kd_j)
        derivation_per_joint.append({
            "role": role,
            "group": group.name,
            "physical_name": physical_name,
            "omega_n": omega_n,
            "zeta": zeta,
            "m_eff": m_eff,
            "dof_index": inertia.dof_index[idx],
        })

    return JointPDGains(
        joint_ir_roles=list(joint_ir_roles),
        joint_physical_names=list(joint_order_physical),
        kp=kp,
        kd=kd,
        engine="mujoco",
        derivation={
            "method": "full_mass_matrix",
            "formula_kp": "m_eff * omega_n ** 2",
            "formula_kd": "2 * zeta * sqrt(kp * m_eff)",
            "mjcf_path": str(Path(mjcf_path)),
            "nominal_qpos_source": inertia.nominal_qpos_source,
            "per_joint": derivation_per_joint,
        },
    )
