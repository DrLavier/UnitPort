"""Timeout — Terminate when the task exceeds its allowed duration."""

from __future__ import annotations

from scripts.task_module import (
    BACKEND_SB3,
    termination_item,
)


ENTRY = termination_item(
    key='timeout',
    title='Timeout',
    desc='Terminate when the task exceeds its allowed duration.',
    default=1000.0,
    min_value=10.0,
    max_value=100000.0,
    step=10.0,
    backends=frozenset({BACKEND_SB3}),
)
