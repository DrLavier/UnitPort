"""Feet Slide — Contact-force-based penalty on foot lateral velocity while in ground contact."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def feet_slide(foot_body_ids, contacts, root_vel):
    """Contact-force-based penalty on foot lateral velocity while grounded.

    contact_indicator * ||foot_vel_xy - root_vel_xy||^2
    Contact detection uses MuJoCo contact pairs (binary, not force magnitude).
    """
    cost = 0.0
    for bid in contacted_foot_bodies:
        rel_vel = body_vel[bid][:2] - root_vel[:2]
        cost += rel_vel[0]**2 + rel_vel[1]**2
    return cost
'''


ENTRY = reward_item(
    key='feet_slide',
    polarity='penalty',
    title='Feet Slide',
    desc='Contact-force-based penalty on foot lateral velocity while in ground contact. Uses contact detection to weight the sliding cost.',
    default=-0.1,
    min_value=-5.0,
    max_value=0.0,
    step=0.01,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
