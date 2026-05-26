# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Action Rate Penalty — L2 penalty on action rate of change for smooth control."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    ALL_FAMILIES,
    BACKEND_ISAAC,
    IL_MOD_INLINE,
    reward_item,
)


INLINE_SOURCE = '''
def _unitport_action_rate_l2(env):
    """L2 penalty on action rate of change (consecutive action difference)."""
    import torch
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
'''


ENTRY = reward_item(
    key='action_rate_penalty',
    polarity='penalty',
    title='Action Rate Penalty',
    desc='L2 penalty on action rate of change for smooth control.',
    default=-0.05,
    min_value=-1.0,
    max_value=0.0,
    step=0.005,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
    algorithms=frozenset({ALG_ALL}),
    il_func='_unitport_action_rate_l2',
    il_module=IL_MOD_INLINE,
    il_inline=INLINE_SOURCE,
)
