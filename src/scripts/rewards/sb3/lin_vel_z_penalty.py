# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Lin Vel Z Penalty — L2 penalty on vertical linear velocity to discourage bouncing."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def lin_vel_z_penalty(qvel_z):
    """L2 penalty on vertical linear velocity (discourages bouncing)."""
    return qvel_z ** 2
'''


ENTRY = reward_item(
    key='lin_vel_z_penalty',
    polarity='penalty',
    title='Lin Vel Z Penalty',
    desc='L2 penalty on vertical linear velocity to discourage bouncing.',
    default=-2.0,
    min_value=-20.0,
    max_value=0.0,
    step=0.1,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
