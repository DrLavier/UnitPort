"""Base Ang Vel — Base angular velocity in body frame (3D)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='base_ang_vel',
    title='Base Ang Vel',
    desc='Base angular velocity in body frame (3D).',
    applicable_families=ALL_FAMILIES,
)
