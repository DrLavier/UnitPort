"""Joint Velocities — Joint velocity readings (N-dim)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='joint_vel',
    title='Joint Velocities',
    desc='Joint velocity readings (N-dim).',
    applicable_families=ALL_FAMILIES,
)
