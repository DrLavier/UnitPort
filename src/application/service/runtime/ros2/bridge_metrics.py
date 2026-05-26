# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Shared per-topic metrics helper.

Tracks the most recent payload and a 1-second sliding-window publish rate.
Thread-safety is the caller's responsibility — NativeDDSBridge wraps every
``Subscription`` access in its own RLock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Subscription:
    """Per-topic cache of the latest payload plus a rate-window."""

    topic: str
    msg_type: str
    qos: Dict[str, Any] = field(default_factory=dict)
    last_payload: Optional[Dict[str, Any]] = None
    last_ts: float = 0.0
    rate_hz: float = 0.0
    window: List[float] = field(default_factory=list)

    def update(self, payload: Dict[str, Any], ts: float, window_seconds: float = 1.0) -> None:
        """Record a new sample: overwrites last_payload/last_ts and recomputes rate."""
        self.last_payload = payload
        self.last_ts = ts
        self.window.append(ts)
        cutoff = ts - window_seconds
        while self.window and self.window[0] < cutoff:
            self.window.pop(0)
        self.rate_hz = float(len(self.window)) / window_seconds

    def stats(self) -> Dict[str, float]:
        """Return a shallow {rate_hz, last_ts} dict for subscription_stats()."""
        return {"rate_hz": self.rate_hz, "last_ts": self.last_ts}
