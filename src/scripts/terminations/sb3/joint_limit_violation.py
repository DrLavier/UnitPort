"""Joint Limit — Terminate when joints exceed safe configured limits."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    BACKEND_SB3,
    termination_item,
)


ENTRY = termination_item(
    key='joint_limit_violation',
    title='Joint Limit',
    desc='Terminate when joints exceed safe configured limits. Threshold is a count: 3 = terminate when 3 or more joints exceed their range.',
    default=3.0,
    min_value=1.0,
    max_value=12.0,
    step=1.0,
    applicable_families=ALL_FAMILIES,
    backends=frozenset({BACKEND_SB3}),
)
