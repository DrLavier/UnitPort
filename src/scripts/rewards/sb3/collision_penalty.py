"""Collision Penalty — Penalty for unwanted collisions with the environment or self."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def collision_penalty(base_body_id, cfrc_ext):
    """Penalty proportional to external contact force on the base body.

    Normalised by 150N to [0, 1] range.
    """
    force_mag = norm(cfrc_ext[base_body_id][:3])
    return clip(force_mag / 150.0, 0.0, 1.0)
'''


ENTRY = reward_item(
    key='collision_penalty',
    polarity='penalty',
    title='Collision Penalty',
    desc='Penalty for unwanted collisions with the environment or self.',
    default=-0.2,
    min_value=-10.0,
    max_value=0.0,
    step=0.05,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
