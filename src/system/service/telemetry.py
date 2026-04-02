"""Structured telemetry for lifecycle execution — Phase 6 STAGE-03.

Provides:
    TelemetryEvent    — frozen schema for a single lifecycle boundary event.
    TelemetryCollector — accumulates events for one execute_with_lifecycle call.
    emit_event()       — logs a TelemetryEvent at DEBUG level.
    new_trace_id()     — UUID4 string per call.

Privacy rule: credentials (auth_token, hostname) are stripped from diagnostics
before they are stored in any TelemetryEvent.
"""

from __future__ import annotations

import logging as _logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_log = _logging.getLogger("unitport.service.telemetry")

# Keys that must never appear in telemetry diagnostics (privacy rule §10.3)
_CREDENTIAL_KEYS: frozenset = frozenset({"auth_token", "hostname"})


# ── TelemetryEvent ─────────────────────────────────────────────────────────


@dataclass
class TelemetryEvent:
    """Single structured event emitted at a lifecycle boundary.

    Schema (all 13 keys present in ``to_dict()`` output)::

        {
            "trace_id":     str,          # UUID4 per execute_with_lifecycle call
            "adapter_name": str,          # registry key of the target adapter
            "brand":        str,          # brand from session_config, or ""
            "operation":    str,          # RouteOp constant
            "stage":        str,          # lifecycle stage name
            "event":        str,          # "start" | "end"
            "timestamp":    float,        # time.monotonic() at emission
            "duration_ms":  float | None, # elapsed since stage start; None for "start"
            "status":       str,          # "ok"|"error"|"skipped"|"pending"
            "reason":       str,          # LifecycleReason constant or ""
            "retryable":    bool,         # STAGE-04 will wire from REASON_MAP
            "attempt":      int,          # 1-based; 1 when no retry
            "diagnostics":  dict,         # safe subset (no credentials)
        }
    """

    trace_id:     str
    adapter_name: str
    brand:        str
    operation:    str
    stage:        str
    event:        str                  # "start" | "end"
    timestamp:    float
    duration_ms:  Optional[float]      # None for "start" events
    status:       str                  # "ok" | "error" | "skipped" | "pending"
    reason:       str                  # LifecycleReason constant or ""
    retryable:    bool
    attempt:      int                  # 1-based
    diagnostics:  Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict with all 13 schema keys."""
        return {
            "trace_id":     self.trace_id,
            "adapter_name": self.adapter_name,
            "brand":        self.brand,
            "operation":    self.operation,
            "stage":        self.stage,
            "event":        self.event,
            "timestamp":    self.timestamp,
            "duration_ms":  self.duration_ms,
            "status":       self.status,
            "reason":       self.reason,
            "retryable":    self.retryable,
            "attempt":      self.attempt,
            "diagnostics":  self.diagnostics,
        }


# ── TelemetryCollector ─────────────────────────────────────────────────────


class TelemetryCollector:
    """Accumulates :class:`TelemetryEvent` dicts for a single lifecycle call.

    Usage::

        collector = TelemetryCollector(trace_id, adapter_name, brand, op)
        collector.start("open_session")
        # ... do work ...
        collector.end("open_session", "ok")
        events = collector.events   # list of dicts
    """

    def __init__(
        self,
        trace_id:     str,
        adapter_name: str,
        brand:        str,
        operation:    str,
        attempt:      int = 1,
    ) -> None:
        self.trace_id     = trace_id
        self.adapter_name = adapter_name
        self.brand        = brand
        self.operation    = operation
        self.attempt      = attempt
        self._events:      List[Dict[str, Any]] = []
        self._stage_start: Dict[str, float]     = {}

    # ── Emission helpers ───────────────────────────────────────────────────

    def start(self, stage: str) -> None:
        """Emit a ``"start"`` event for *stage* and record the stage timestamp."""
        ts = time.monotonic()
        self._stage_start[stage] = ts
        event = TelemetryEvent(
            trace_id=self.trace_id,
            adapter_name=self.adapter_name,
            brand=self.brand,
            operation=self.operation,
            stage=stage,
            event="start",
            timestamp=ts,
            duration_ms=None,
            status="pending",
            reason="",
            retryable=False,
            attempt=self.attempt,
            diagnostics={},
        )
        emit_event(event)
        self._events.append(event.to_dict())

    def end(
        self,
        stage:       str,
        status:      str,
        reason:      str = "",
        retryable:   bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an ``"end"`` event for *stage* with elapsed duration.

        ``retryable`` is auto-derived from ``REASON_MAP[reason]`` when *reason*
        is non-empty (Phase 6 STAGE-04).  The caller-supplied *retryable* acts
        as a fallback for unknown/future reason codes.
        """
        ts        = time.monotonic()
        start_ts  = self._stage_start.get(stage, ts)
        duration  = (ts - start_ts) * 1000.0
        safe_diag = _scrub_credentials(diagnostics or {})
        # Derive retryable from REASON_MAP; caller value is the fallback.
        actual_retryable = _get_retryable(reason, default=retryable)
        event = TelemetryEvent(
            trace_id=self.trace_id,
            adapter_name=self.adapter_name,
            brand=self.brand,
            operation=self.operation,
            stage=stage,
            event="end",
            timestamp=ts,
            duration_ms=duration,
            status=status,
            reason=reason,
            retryable=actual_retryable,
            attempt=self.attempt,
            diagnostics=safe_diag,
        )
        emit_event(event)
        self._events.append(event.to_dict())

    # ── Result access ──────────────────────────────────────────────────────

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Return a snapshot copy of all accumulated events."""
        return list(self._events)


# ── Public helpers ─────────────────────────────────────────────────────────


def emit_event(event: TelemetryEvent) -> None:
    """Log *event* at DEBUG level via the unitport telemetry logger."""
    _log.debug(
        "telemetry stage=%s event=%s status=%s trace_id=%s adapter=%s",
        event.stage, event.event, event.status,
        event.trace_id, event.adapter_name,
    )


def new_trace_id() -> str:
    """Return a fresh UUID4 string suitable as a ``trace_id``."""
    return str(uuid.uuid4())


# ── Internal helpers ───────────────────────────────────────────────────────


def _get_retryable(reason: str, default: bool = False) -> bool:
    """Return whether *reason* is retryable according to REASON_MAP.

    Uses a lazy import to avoid module-level circular dependency.
    Falls back to *default* for unknown reason codes (forward-compatible
    with Phase 7+ extensions).
    """
    if not reason:
        return default
    try:
        from .reason_codes import REASON_MAP  # lazy — avoids import cycle
        return REASON_MAP.get(reason, {}).get("retryable", default)
    except ImportError:
        return default


def _scrub_credentials(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Return *diag* with credential keys removed (privacy rule §10.3)."""
    return {k: v for k, v in diag.items() if k not in _CREDENTIAL_KEYS}
