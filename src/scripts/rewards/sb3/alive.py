# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Alive Bonus — Small survival bonus that encourages staying upright."""

from __future__ import annotations

from scripts.task_module import (
    ALG_ALL,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    reward_item,
)


INLINE_SOURCE = '''\
def alive():
    """Constant 1.0 survival bonus per step."""
    return 1.0
'''


ENTRY = reward_item(
    key='alive',
    polarity='reward',
    title='Alive Bonus',
    desc='Small survival bonus that encourages staying upright.',
    default=0.5,
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
    algorithms=frozenset({ALG_ALL}),
    il_inline=INLINE_SOURCE,
)
