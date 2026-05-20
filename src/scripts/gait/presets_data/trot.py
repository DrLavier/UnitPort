"""Gait preset — trot (diagonal pairs)."""

from __future__ import annotations

from scripts.gait.presets import GaitPreset


ORDER = 0

ENTRY = GaitPreset(
    name="trot",
    frequency=2.5,
    phase=(0.0, 0.5, 0.5, 0.0),
    body_height=0.35,
    step_height=0.08,
)
