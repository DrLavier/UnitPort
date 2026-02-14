"""Event protocol for service communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Event:
    """An event emitted by a service adapter or the runtime."""

    event_type: str
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
