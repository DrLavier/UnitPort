"""Joint Vel Penalty — L2 penalty on joint velocities — discourages overly fast joint motion."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def joint_vel_penalty(qvel_joints):
    """L2 penalty on joint velocities."""
    return sum(qvel_joints ** 2)
'''


ENTRY = reward_item(
    key='joint_vel_penalty',
    polarity='penalty',
    title='Joint Vel Penalty',
    desc='L2 penalty on joint velocities — discourages overly fast joint motion.',
    default=-0.001,
    min_value=-1.0,
    max_value=0.0,
    step=0.0005,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
