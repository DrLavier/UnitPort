# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Projected Gravity — Gravity vector projected into body frame (3D)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='projected_gravity',
    title='Projected Gravity',
    desc='Gravity vector projected into body frame (3D).',
    applicable_families=ALL_FAMILIES,
)
