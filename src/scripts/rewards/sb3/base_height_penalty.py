"""Base Height Penalty — Penalty for base height deviating from the nominal standing height (~0.32 m)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def base_height_penalty(base_height, target_height=0.32):
    """Squared deviation from nominal standing height."""
    return (base_height - target_height) ** 2
'''


ENTRY = reward_item(
    key='base_height_penalty',
    polarity='penalty',
    title='Base Height Penalty',
    desc='Penalty for base height deviating from the nominal standing height (~0.32 m). Discourages crouching or over-extension during locomotion.',
    default=-1.0,
    min_value=-10.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
