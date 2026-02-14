"""Command protocol for service communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Command:
    """A command issued to a service adapter."""

    action: str
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
