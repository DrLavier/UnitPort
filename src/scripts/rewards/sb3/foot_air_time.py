# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Foot Air Time — Bonus at each foot touchdown proportional to how long the foot was airborne (capped at 0.5 s)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def foot_air_time(foot_body_ids, dt):
    """Landing bonus proportional to swing-phase air duration.

    At each step, for every tracked foot:
    - Height > 0.05m: foot is in swing, increment air counter.
    - Height <= 0.05m: foot is in stance.
      - On touchdown (swing->stance transition):
        bonus = clip(air_steps * dt, 0, 0.5)
    Returns mean bonus across all feet, in [0, 0.5].
    """
    score = 0.0
    for i, bid in enumerate(foot_body_ids):
        h = xpos[bid][2]
        in_air = h > 0.05
        if not in_air and was_in_air[i]:
            air_duration = air_steps[i] * dt
            score += clip(air_duration, 0.0, 0.5)
    return score / n_feet
'''


ENTRY = reward_item(
    key='foot_air_time',
    polarity='reward',
    title='Foot Air Time',
    desc='Bonus at each foot touchdown proportional to how long the foot was airborne (capped at 0.5 s). Encourages regular, symmetric gait cycles.',
    default=0.5,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=frozenset({'quadruped', 'biped'}),
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
