# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Foot Position Tracking — Gaussian reward for foot Cartesian positions matching reference FK positions."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def foot_pos_tracking(foot_positions, ref_foot_positions):
    """Gaussian reward for foot Cartesian positions matching reference FK.

    score = exp(-10.0 * mean(||foot_pos - foot_ref||^2))
    Falls back to nominal standing foot positions when no reference is loaded.
    """
    sq_errs = [sum((fp - rp) ** 2) for fp, rp in zip(foot_positions, ref_foot_positions)]
    return exp(-10.0 * mean(sq_errs))
'''


ENTRY = reward_item(
    key='foot_pos_tracking',
    polarity='reward',
    title='Foot Position Tracking',
    desc='Gaussian reward for foot Cartesian positions matching reference FK positions. Falls back to nominal standing foot positions when no reference motion is loaded.',
    default=0.5,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=frozenset({'quadruped', 'biped'}),
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_inline=INLINE_SOURCE,
)
