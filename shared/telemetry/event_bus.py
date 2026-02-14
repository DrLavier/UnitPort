"""Lightweight publish/subscribe event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class EventBus:
    """Simple in-process event bus for telemetry and cross-layer signalling."""

    _subscribers: Dict[str, List[Callable]] = field(
        default_factory=dict, init=False, repr=False
    )

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register *callback* for events of *event_type*."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: Any = None) -> None:
        """Publish an event to all subscribers of *event_type*."""
        for callback in list(self._subscribers.get(event_type, [])):
            callback(payload)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Remove *callback* from *event_type* subscribers."""
        if event_type not in self._subscribers:
            return
        self._subscribers[event_type] = [
            fn for fn in self._subscribers[event_type] if fn is not callback
        ]
