# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Angular Rate Penalty — Penalty on squared roll and pitch angular rates."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def angular_rate_penalty(roll_rate, pitch_rate):
    """L2 penalty on roll and pitch angular rates."""
    return roll_rate ** 2 + pitch_rate ** 2
'''


ENTRY = reward_item(
    key='angular_rate_penalty',
    polarity='penalty',
    title='Angular Rate Penalty',
    desc='Penalty on squared roll and pitch angular rates. Reduces trunk wobble and promotes smooth, stable locomotion.',
    default=-0.05,
    min_value=-5.0,
    max_value=0.0,
    step=0.01,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
