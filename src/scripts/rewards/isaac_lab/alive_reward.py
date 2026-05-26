# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Alive Bonus — Constant per-step survival bonus — encourages the policy to keep the robot alive."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_is_alive(env):
    """Constant per-step survival bonus (1.0 for all alive envs)."""
    import torch
    return (~env.termination_manager.terminated).float()
'''


ENTRY = reward_item(
    key='alive_reward',
    polarity='reward',
    title='Alive Bonus',
    desc='Constant per-step survival bonus — encourages the policy to keep the robot alive.',
    default=0.15,
    min_value=0.0,
    max_value=5.0,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_is_alive',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
