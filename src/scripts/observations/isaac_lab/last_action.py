"""Last Action — Previous policy output / action applied (N-dim)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='last_action',
    title='Last Action',
    desc='Previous policy output / action applied (N-dim).',
    applicable_families=ALL_FAMILIES,
)
