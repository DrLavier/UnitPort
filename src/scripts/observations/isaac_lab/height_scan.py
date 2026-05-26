# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Height Scan — Terrain height scanner readings around each foot (M-dim)."""

from __future__ import annotations

from scripts.task_module import (
    LEGGED_FAMILIES,
    observation_item,
)


ENTRY = observation_item(
    key='height_scan',
    title='Height Scan',
    desc='Terrain height scanner readings around each foot (M-dim). Requires a height_scanner sensor on the canvas (Play Ground Setting → height_scan_enabled=true).',
    applicable_families=LEGGED_FAMILIES,
)
