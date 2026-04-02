"""Mapping between canonical actions and Boston Dynamics Spot commands."""

from __future__ import annotations

from typing import Dict


ACTION_MAP: Dict[str, str] = {
    "stand": "stand",
    "sit":   "sit",
    "walk":  "walk",
    "stop":  "stop",
}


def map_action(action: str) -> str:
    """Map canonical action name to Spot adapter command key."""
    if not action:
        return ""
    return ACTION_MAP.get(action, action)
