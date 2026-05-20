"""Goal Distance — Reward reducing distance to a target pose or position."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def goal_distance(current_pos, target_pos):
    """Reward for reducing distance to a target position.

    Returns task-specific success score.
    """
    return task_success_score()
'''


ENTRY = reward_item(
    key='goal_distance',
    polarity='reward',
    title='Goal Distance',
    desc='Reward reducing distance to a target pose or position.',
    default=1.0,
    min_value=-5.0,
    max_value=10.0,
    step=0.1,
    applicable_families=frozenset({'manipulator'}),
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
