# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Grasp Success — Sparse success bonus for stable grasp or task completion."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    reward_item,
)


INLINE_SOURCE = '''\
def grasp_success(success_streak):
    """Sparse binary reward for stable grasp or task completion."""
    return 1.0 if success_streak > 0 else 0.0
'''


ENTRY = reward_item(
    key='grasp_success',
    polarity='reward',
    title='Grasp Success',
    desc='Sparse success bonus for stable grasp or task completion.',
    default=5.0,
    min_value=0.0,
    max_value=20.0,
    step=0.5,
    applicable_families=frozenset({'manipulator'}),
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
