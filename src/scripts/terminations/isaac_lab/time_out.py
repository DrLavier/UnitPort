# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Episode Timeout — Terminate when episode wall-clock duration (s) exceeds this limit."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    BACKEND_ISAAC,
    termination_item,
)


ENTRY = termination_item(
    key='time_out',
    title='Episode Timeout',
    desc='Terminate when episode wall-clock duration (s) exceeds this limit. Mapped to mdp.time_out in the compiled Isaac Lab task.',
    default=20.0,
    min_value=1.0,
    max_value=300.0,
    step=0.5,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
