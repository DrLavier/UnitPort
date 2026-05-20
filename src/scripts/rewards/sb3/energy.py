"""Energy Penalty — Penalty on mechanical power: Σ|joint_vel|·|torque|."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def energy(qvel_joints, torque_joints):
    """Mechanical power: sum(|joint_vel| * |torque|).

    Uses qfrc_actuator (post-step MuJoCo forces) when available,
    otherwise falls back to PD-computed torques:
        tau = Kp * (q_des - q) - Kd * qdot
    """
    return sum(abs(qvel_joints) * abs(torque_joints))
'''


ENTRY = reward_item(
    key='energy',
    polarity='penalty',
    title='Energy Penalty',
    desc='Penalty on mechanical power: Σ|joint_vel|·|torque|. Consistent with Isaac Lab energy formulation.',
    default=-0.0001,
    min_value=-0.1,
    max_value=0.0,
    step=5e-05,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
