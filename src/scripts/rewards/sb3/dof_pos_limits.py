"""Joint Pos Limits — Penalty when joint positions approach or exceed soft limits."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def dof_pos_limits(joint_positions, joint_ranges, soft_margin=0.05):
    """Penalty for joints approaching or exceeding soft limits.

    For each limited joint, computes excess beyond 95% of range on both
    sides and sums squared violations.
    """
    cost = 0.0
    for pos, (lo, hi) in zip(joint_positions, joint_ranges):
        span = hi - lo
        margin = span * soft_margin
        below = max(0.0, (lo + margin) - pos)
        above = max(0.0, pos - (hi - margin))
        cost += below ** 2 + above ** 2
    return cost
'''


ENTRY = reward_item(
    key='dof_pos_limits',
    polarity='penalty',
    title='Joint Pos Limits',
    desc='Penalty when joint positions approach or exceed soft limits.',
    default=-5.0,
    min_value=-20.0,
    max_value=0.0,
    step=0.5,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
