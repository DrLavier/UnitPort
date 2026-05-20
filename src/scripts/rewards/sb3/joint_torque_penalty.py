"""Joint Torque Penalty — L2 penalty on applied joint torques for energy efficiency."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def joint_torque_penalty(torque_joints):
    """L2 penalty on applied joint torques.

    Uses qfrc_actuator (post-step joint-space actuator forces).
    """
    return sum(torque_joints ** 2)
'''


ENTRY = reward_item(
    key='joint_torque_penalty',
    polarity='penalty',
    title='Joint Torque Penalty',
    desc='L2 penalty on applied joint torques for energy efficiency.',
    default=-0.0002,
    min_value=-0.01,
    max_value=0.0,
    step=5e-05,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
