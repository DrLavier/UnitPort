"""Minimum Height — Terminate when the robot base drops below this height."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    LOCOMOTION_FAMILIES,
    termination_item,
)


ENTRY = termination_item(
    key='min_height',
    title='Minimum Height',
    desc='Terminate when the robot base drops below this height.',
    default=0.15,
    min_value=0.05,
    max_value=0.5,
    step=0.01,
    applicable_families=LOCOMOTION_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
)
