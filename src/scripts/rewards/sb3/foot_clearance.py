# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Foot Clearance — Reward for lifting feet enough to avoid stumbling during swing."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def foot_clearance(foot_heights):
    """Reward for average foot height above ground.

    Linearly maps [0.02m, 0.20m] -> [0, 1].
    """
    avg_height = mean(foot_heights)
    return clip((avg_height - 0.02) / 0.18, 0.0, 1.0)
'''


ENTRY = reward_item(
    key='foot_clearance',
    polarity='reward',
    title='Foot Clearance',
    desc='Reward for lifting feet enough to avoid stumbling during swing.',
    default=0.2,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=frozenset({'quadruped', 'biped'}),
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
