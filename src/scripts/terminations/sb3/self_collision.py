# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Self Collision — Terminate when forbidden self-collision is detected."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    termination_item,
)


ENTRY = termination_item(
    key='self_collision',
    title='Self Collision',
    desc='Terminate when forbidden self-collision is detected.',
    default=1.0,
    min_value=0.0,
    max_value=5.0,
    step=0.1,
    applicable_families=frozenset({'manipulator'}),
    backends=frozenset({BACKEND_SB3}),
)
