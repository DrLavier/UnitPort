# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lateral Penalty — Penalty for sideways drift when forward tracking is desired."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def lateral_penalty(lin_vel_y, vy_tgt):
    """Penalty for sideways drift from commanded lateral velocity."""
    return abs(lin_vel_y - vy_tgt)
'''


ENTRY = reward_item(
    key='lateral_penalty',
    polarity='penalty',
    title='Lateral Penalty',
    desc='Penalty for sideways drift when forward tracking is desired.',
    default=-0.1,
    min_value=-10.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
