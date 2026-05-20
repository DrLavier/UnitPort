"""Base Lin Vel — Base linear velocity in body frame (3D)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='base_lin_vel',
    title='Base Lin Vel',
    desc='Base linear velocity in body frame (3D).',
    applicable_families=ALL_FAMILIES,
)
