"""Gait preset — pace (left/right pairs)."""

from __future__ import annotations

from scripts.gait.presets import GaitPreset


ORDER = 3

ENTRY = GaitPreset(
    name="pace",
    frequency=2.5,
    phase=(0.0, 0.5, 0.0, 0.5),
    body_height=0.33,
    step_height=0.07,
)
