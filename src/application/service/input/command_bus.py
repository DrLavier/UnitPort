"""Thread-safe key-value scalar store used by all input sources.

The bus carries a flat ``str -> float`` snapshot. Keys map to:
    - axis fields (``vx``, ``vy``, ``vyaw``)
    - reserved internal flags (``__estop``, ``__boost``)
    - per-button gamepad transients (``__btn_<hw>``: 1.0 while held, 0.0 otherwise)

Sources publish via :meth:`set`; UI consumers read via :meth:`read_all` /
:meth:`get`.
"""

from __future__ import annotations

import threading
from typing import Dict, Iterable


class CommandBus:
    """Lock-protected ``str -> float`` store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {}

    def set(self, key: str, value: float) -> None:
        with self._lock:
            self._values[str(key)] = float(value)

    def get(self, key: str, default: float = 0.0) -> float:
        with self._lock:
            return float(self._values.get(str(key), default))

    def read_all(self, keys: Iterable[str], default: float = 0.0) -> Dict[str, float]:
        with self._lock:
            return {k: float(self._values.get(k, default)) for k in keys}

    def reset(self, keys: Iterable[str] = ()) -> None:
        with self._lock:
            if keys:
                for k in keys:
                    self._values[k] = 0.0
            else:
                self._values.clear()


__all__ = ["CommandBus"]
