# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Bad Orientation — Terminate when the base tilts beyond this projected-gravity deviation (rad)."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_ISAAC,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='bad_orientation',
    title='Bad Orientation',
    desc='Terminate when the base tilts beyond this projected-gravity deviation (rad). Mapped to mdp.bad_orientation in the compiled Isaac Lab task.',
    default=0.7,
    min_value=0.1,
    max_value=1.5,
    step=0.05,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_ISAAC}),
)
