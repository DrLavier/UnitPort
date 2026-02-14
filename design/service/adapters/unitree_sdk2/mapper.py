"""Mapping between canonical actions and Unitree SDK actions."""

from __future__ import annotations

from typing import Dict


ACTION_MAP: Dict[str, str] = {
    "stand": "stand",
    "sit": "sit",
    "walk": "walk",
    "stop": "stop",
    "lift_right_leg": "lift_right_leg",
}


def map_action(action: str) -> str:
    """Map canonical action name to Unitree adapter action."""
    if not action:
        return ""
    return ACTION_MAP.get(action, action)

