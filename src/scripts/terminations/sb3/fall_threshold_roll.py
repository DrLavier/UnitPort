# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Roll Threshold — Terminate when roll exceeds the allowed upright margin."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    LEGGED_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='fall_threshold_roll',
    title='Roll Threshold',
    desc='Terminate when roll exceeds the allowed upright margin.',
    default=1.2,
    min_value=0.2,
    max_value=3.14,
    step=0.05,
    applicable_families=LEGGED_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
)
