"""Safety audit logger."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class SafetyAuditLogger:
    """In-memory audit event store for safety decisions."""

    _events: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def record(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        self._events.append(
            {
                "event_type": event_type,
                "payload": payload or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def list_events(self) -> List[Dict[str, Any]]:
        return list(self._events)

