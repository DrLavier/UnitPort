"""CommandBus — thread-safe pub/sub for real-time command values.

Multiple input sources (gamepad, keyboard, Canvas signals, external API)
publish named command values.  The ReactiveLoop reads them every tick.

Design
------
- Lock-free reads via dict snapshot (GIL-protected single-key writes are
  atomic in CPython, but we use a lock for multi-key batch safety).
- No Qt dependency — pure Python so it can be unit-tested without a GUI.
- Stateless: each ``publish`` overwrites the previous value for that key.
  There is no queue or history.

Usage::

    bus = CommandBus()
    bus.publish("vx", 0.5)
    bus.publish("vy", 0.0)

    val = bus.read("vx")                   # 0.5
    vals = bus.read_all(["vx", "vy"])      # {"vx": 0.5, "vy": 0.0}
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class CommandBus:
    """Thread-safe command value store.

    Publishers call :meth:`publish` from any thread (gamepad poller,
    keyboard handler, network listener).  The consumer (ReactiveLoop)
    calls :meth:`read` / :meth:`read_all` at policy tick rate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def publish(self, key: str, value: float) -> None:
        """Set a single command value (overwrites previous)."""
        with self._lock:
            self._values[key] = float(value)

    def publish_batch(self, values: Dict[str, float]) -> None:
        """Atomically set multiple command values."""
        with self._lock:
            for k, v in values.items():
                self._values[k] = float(v)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, key: str, default: float = 0.0) -> float:
        """Read a single command value. Returns *default* if not published."""
        with self._lock:
            return self._values.get(key, default)

    def read_all(self, keys: List[str], default: float = 0.0) -> Dict[str, float]:
        """Read multiple command values as a dict."""
        with self._lock:
            return {k: self._values.get(k, default) for k in keys}

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def reset(self, keys: Optional[List[str]] = None) -> None:
        """Reset specified keys to 0.0, or all keys if *keys* is None."""
        with self._lock:
            if keys is None:
                for k in self._values:
                    self._values[k] = 0.0
            else:
                for k in keys:
                    self._values[k] = 0.0

    def clear(self) -> None:
        """Remove all stored values."""
        with self._lock:
            self._values.clear()

    def keys(self) -> List[str]:
        """Return currently published key names."""
        with self._lock:
            return list(self._values.keys())
