"""Joint Accel Penalty — L2 penalty on joint accelerations for smooth motion."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def joint_accel_penalty(qacc_joints):
    """L2 penalty on joint accelerations."""
    return sum(qacc_joints ** 2)
'''


ENTRY = reward_item(
    key='joint_accel_penalty',
    polarity='penalty',
    title='Joint Accel Penalty',
    desc='L2 penalty on joint accelerations for smooth motion.',
    default=-2.5e-07,
    min_value=-0.001,
    max_value=0.0,
    step=1e-07,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
