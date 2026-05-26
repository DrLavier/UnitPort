# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Base Height — Terminate when the robot base drops below this minimum height (m)."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_ISAAC,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='base_height',
    title='Base Height',
    desc='Terminate when the robot base drops below this minimum height (m).',
    default=0.2,
    min_value=0.05,
    max_value=1.0,
    step=0.01,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
