# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Reference Tracking — Reward for matching a reference motion trajectory keyframe via Gaussian similarity."""

from __future__ import annotations

from scripts.task_module import (
    ALG_AMP,
    ALG_PPO,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def reference_tracking(cur_joints, ref_joints, sigma=5.0):
    """Gaussian similarity between current and reference joint positions.

    score = weight * exp(-sigma * ||q_cur - q_ref||^2)
    """
    sq_err = sum((cur_joints - ref_joints) ** 2)
    return exp(-sigma * sq_err)
'''


ENTRY = reward_item(
    key='reference_tracking',
    polarity='reward',
    title='Reference Tracking',
    desc='Reward for matching a reference motion trajectory keyframe via Gaussian similarity.',
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_PPO, ALG_AMP}),
    il_inline=INLINE_SOURCE,
)
