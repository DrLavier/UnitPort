# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Gait preset — pronk (all in phase)."""

from __future__ import annotations

from scripts.gait.presets import GaitPreset


ORDER = 4

ENTRY = GaitPreset(
    name="pronk",
    frequency=3.5,
    phase=(0.0, 0.0, 0.0, 0.0),
    body_height=0.35,
    step_height=0.15,
)
