"""ROS2 bridge diagnostics — topic rate / latency / health summaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass
class TopicHealth:
    topic: str
    rate_hz: float
    last_ts: float
    last_age_s: float
    status: str  # ok | stale | silent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "rate_hz": round(self.rate_hz, 2),
            "last_ts": self.last_ts,
            "last_age_s": round(self.last_age_s, 3),
            "status": self.status,
        }


def evaluate(
    stats: Dict[str, Dict[str, float]],
    *,
    min_rate_hz: float = 1.0,
    stale_age_s: float = 1.0,
    now: Optional[float] = None,
) -> Dict[str, TopicHealth]:
    """Classify each subscription by rate/age."""
    now = now if now is not None else time.time()
    out: Dict[str, TopicHealth] = {}
    for topic, s in stats.items():
        last_ts = float(s.get("last_ts") or 0.0)
        rate = float(s.get("rate_hz") or 0.0)
        age = max(0.0, now - last_ts) if last_ts > 0 else float("inf")
        if last_ts <= 0:
            status = "silent"
        elif age > stale_age_s or rate < min_rate_hz:
            status = "stale"
        else:
            status = "ok"
        out[topic] = TopicHealth(
            topic=topic, rate_hz=rate, last_ts=last_ts, last_age_s=age, status=status
        )
    return out


def summarize(
    stats: Dict[str, Dict[str, float]],
    *,
    required_topics: Optional[Iterable[str]] = None,
    min_rate_hz: float = 1.0,
    stale_age_s: float = 1.0,
) -> Dict[str, Any]:
    """Return a JSON-safe diagnostics summary."""
    health = evaluate(stats, min_rate_hz=min_rate_hz, stale_age_s=stale_age_s)
    req = list(required_topics or [])
    missing = [t for t in req if t not in health]
    unhealthy = [h.to_dict() for h in health.values() if h.status != "ok"]
    return {
        "topics": {t: h.to_dict() for t, h in health.items()},
        "required_missing": missing,
        "unhealthy": unhealthy,
        "all_ok": not missing and not unhealthy,
    }
