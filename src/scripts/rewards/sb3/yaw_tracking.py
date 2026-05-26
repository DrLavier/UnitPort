# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Yaw Tracking — Reward matching the target yaw-rate command."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def yaw_tracking(ang_vel_z, yaw_tgt):
    """Exponential tracking reward for commanded yaw rate."""
    return exp(-((ang_vel_z - yaw_tgt) ** 2) / 0.25)
'''


ENTRY = reward_item(
    key='yaw_tracking',
    polarity='reward',
    title='Yaw Tracking',
    desc='Reward matching the target yaw-rate command.',
    default=0.5,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
