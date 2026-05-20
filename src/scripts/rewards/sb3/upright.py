"""Upright Bonus — Reward maintaining an upright torso orientation."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def upright(projected_gravity):
    """Reward for keeping the torso upright (gravity z-component)."""
    return clip(-projected_gravity[2], 0.0, 1.0)
'''


ENTRY = reward_item(
    key='upright',
    polarity='reward',
    title='Upright Bonus',
    desc='Reward maintaining an upright torso orientation.',
    default=0.3,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
