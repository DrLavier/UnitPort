"""Behavior state machine runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BehaviorStateMachine:
    """Small state-machine helper for node-internal behavior."""

    state: str = "idle"
    context: Dict[str, Any] = field(default_factory=dict)

    def transition(self, next_state: str) -> None:
        self.state = next_state

    def update(self, **context: Any) -> None:
        self.context.update(context)

