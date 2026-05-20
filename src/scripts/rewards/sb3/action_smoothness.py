"""Action Smoothness — Penalty for abrupt changes between consecutive actions."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def action_smoothness(action, prev_action):
    """L2 penalty on consecutive action difference (reduces jitter)."""
    return sum((action - prev_action) ** 2)
'''


ENTRY = reward_item(
    key='action_smoothness',
    polarity='penalty',
    title='Action Smoothness',
    desc='Penalty for abrupt changes between consecutive actions.',
    default=-0.02,
    min_value=-10.0,
    max_value=0.0,
    step=0.01,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
