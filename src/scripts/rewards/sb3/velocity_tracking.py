"""Velocity Tracking — Reward matching commanded forward and lateral velocity."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def velocity_tracking(lin_vel_x, lin_vel_y, vx_tgt, vy_tgt):
    """Exponential tracking reward for commanded XY velocity."""
    r_vx = exp(-((lin_vel_x - vx_tgt) ** 2) / 0.25)
    r_vy = exp(-((lin_vel_y - vy_tgt) ** 2) / 0.25)
    return r_vx + 0.5 * r_vy
'''


ENTRY = reward_item(
    key='velocity_tracking',
    polarity='reward',
    title='Velocity Tracking',
    desc='Reward matching commanded forward and lateral velocity.',
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
