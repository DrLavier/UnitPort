# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Velocity Tracking — Gaussian reward for joint velocities matching the reference (finite-diff of reference frames)."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def joint_vel_tracking(cur_vel, ref_vel):
    """Gaussian reward for joint velocities matching reference.

    score = exp(-0.5 * ||qdot_cur - qdot_ref||^2)
    Falls back to zero velocity when no reference is loaded.
    """
    sq_err = sum((cur_vel - ref_vel) ** 2)
    return exp(-0.5 * sq_err)
'''


ENTRY = reward_item(
    key='joint_vel_tracking',
    polarity='reward',
    title='Joint Velocity Tracking',
    desc='Gaussian reward for joint velocities matching the reference (finite-diff of reference frames). Falls back to tracking zero velocity when no reference is loaded.',
    default=0.5,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_inline=INLINE_SOURCE,
)
