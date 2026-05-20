"""Gait preset — bound (front/back pairs)."""

from __future__ import annotations

from scripts.gait.presets import GaitPreset


ORDER = 2

ENTRY = GaitPreset(
    name="bound",
    frequency=3.0,
    phase=(0.0, 0.0, 0.5, 0.5),
    body_height=0.32,
    step_height=0.10,
)
