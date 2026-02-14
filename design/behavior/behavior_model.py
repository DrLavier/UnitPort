"""Behavior model data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BehaviorModel:
    """Node-internal logic specification (state machine, sensor feedback)."""

    state: str = "idle"
    params: Dict[str, Any] = field(default_factory=dict)
    logic_spec: Dict[str, Any] = field(default_factory=dict)
