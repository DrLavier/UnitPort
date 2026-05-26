# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Joint Positions — Joint position readings, relative to default pose (N-dim)."""

from __future__ import annotations

from scripts.task_module import (
    ALL_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='joint_pos',
    title='Joint Positions',
    desc='Joint position readings, relative to default pose (N-dim).',
    applicable_families=ALL_FAMILIES,
)
