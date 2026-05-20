"""Velocity Command — Commanded velocity target (3D: vx, vy, wz)."""

from __future__ import annotations

from scripts.task_module import (
    LOCOMOTION_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='velocity_command',
    title='Velocity Command',
    desc='Commanded velocity target (3D: vx, vy, wz).',
    applicable_families=LOCOMOTION_FAMILIES,
)
