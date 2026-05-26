# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Slip Penalty — Penalty for excessive foot or wheel slip against the ground."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def slip_penalty(foot_body_ids, xpos, cvel):
    """Penalty for foot lateral velocity when foot is near ground.

    For each foot with height < 0.08m, accumulates lateral speed.
    Returns mean slip speed normalised to [0, 1].
    """
    slip_speeds = []
    for bid in foot_body_ids:
        if xpos[bid][2] > 0.08:
            continue
        lin_vel = cvel[bid][3:6]
        slip_speeds.append(norm(lin_vel[:2]))
    return clip(mean(slip_speeds), 0.0, 1.0) if slip_speeds else 0.0
'''


ENTRY = reward_item(
    key='slip_penalty',
    polarity='penalty',
    title='Slip Penalty',
    desc='Penalty for excessive foot or wheel slip against the ground.',
    default=-0.1,
    min_value=-10.0,
    max_value=0.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
