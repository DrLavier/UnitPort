"""Runtime monitoring and metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class Monitor:
    """Collects and exposes runtime metrics."""

    _active: bool = field(default=False, init=False, repr=False)
    _started_at: float = field(default=0.0, init=False, repr=False)
    _events: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Start collecting metrics."""
        self._active = True
        self._started_at = time.time()
        self._events = 0

    def stop(self) -> None:
        """Stop collecting metrics."""
        self._active = False

    def bump_event(self) -> None:
        """Increment observed event count."""
        if self._active:
            self._events += 1

    def get_metrics(self) -> dict:
        """Return a snapshot of current metrics."""
        uptime = 0.0
        if self._active and self._started_at:
            uptime = time.time() - self._started_at
        return {
            "active": self._active,
            "uptime_sec": round(uptime, 3),
            "events": self._events,
        }
