"""Gait preset — walk (sequential)."""

from __future__ import annotations

from scripts.gait.presets import GaitPreset


ORDER = 1

ENTRY = GaitPreset(
    name="walk",
    frequency=1.5,
    phase=(0.0, 0.25, 0.5, 0.75),
    body_height=0.30,
    step_height=0.05,
)
