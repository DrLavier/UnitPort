"""Lifecycle contract types for Phase 3 service session management.

This module defines the canonical data types used by the service lifecycle
interface:  LifecycleResult, LifecycleReason, and LifecyclePolicy.

All three are imported by BaseAdapter (defaults) and ServiceRouter
(orchestration).  Nothing here imports from adapter or runtime modules,
so there is no circular-import risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


# ── Error reason codes ─────────────────────────────────────────────────────


class LifecycleReason:
    """Machine-readable reason codes for lifecycle failures.

    Values are stable constants — do not rename once published.

    Canonical tier (router + all adapters):
        SESSION_OPEN_FAILED    — open_session returned error
        PREFLIGHT_FAILED       — preflight check failed
        SESSION_CLOSE_FAILED   — close_session returned error
        CAPABILITY_UNAVAILABLE — requested capability not supported
        ADAPTER_NOT_FOUND      — adapter not registered
        EXECUTE_FAILED         — exception raised during operation dispatch

    Adapter-specific tiers (prefixed by adapter name) are defined in
    ``system.service.reason_codes`` and inside each adapter module.
    """

    SESSION_OPEN_FAILED    = "session_open_failed"
    PREFLIGHT_FAILED       = "preflight_failed"
    SESSION_CLOSE_FAILED   = "session_close_failed"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    ADAPTER_NOT_FOUND      = "adapter_not_found"
    # Phase 4: replaces the inline "execute_failed" magic string in ServiceRouter
    EXECUTE_FAILED         = "execute_failed"
    # Phase 6: safety gate explicitly blocked execution before action dispatch
    SAFETY_GATE_BLOCKED    = "safety_gate_blocked"
    # Phase 6: all retry attempts exhausted without success
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"


# ── Result type ────────────────────────────────────────────────────────────


@dataclass
class LifecycleResult:
    """Structured result from any lifecycle operation.

    All lifecycle methods (open_session, preflight, close_session,
    capabilities) return a plain ``dict`` produced by ``to_dict()``.
    Callers that want typed access can reconstruct via
    ``LifecycleResult(**result_dict)``.

    Schema::

        {
            "status":      "ok" | "error",
            "reason":      str,              # LifecycleReason constant; "" on ok
            "stage":       str,              # which lifecycle stage produced this
            "diagnostics": Dict[str, Any],   # optional structured context
        }
    """

    status:      str
    reason:      str
    stage:       str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ── Factories ──────────────────────────────────────────────────────────

    @classmethod
    def ok(cls, stage: str) -> "LifecycleResult":
        """Create a successful lifecycle result."""
        return cls(status="ok", reason="", stage=stage, diagnostics={})

    @classmethod
    def error(
        cls,
        stage: str,
        reason: str,
        diagnostics: Dict[str, Any] | None = None,
    ) -> "LifecycleResult":
        """Create a failed lifecycle result."""
        return cls(
            status="error",
            reason=reason,
            stage=stage,
            diagnostics=diagnostics or {},
        )

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def is_ok(self) -> bool:
        """True iff the lifecycle operation succeeded."""
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe values)."""
        return {
            "status":      self.status,
            "reason":      self.reason,
            "stage":       self.stage,
            "diagnostics": self.diagnostics,
        }


# ── Safety gate policy (Phase 6) ───────────────────────────────────────────


@dataclass
class SafetyPolicy:
    """Safety gate policy for Phase 6 preflight hardening.

    When attached to a :class:`LifecyclePolicy` via ``safety_policy``,
    the gate runs :meth:`BaseAdapter.preflight` before action dispatch
    and returns :attr:`LifecycleReason.SAFETY_GATE_BLOCKED` on failure
    rather than the legacy :attr:`LifecycleReason.PREFLIGHT_FAILED`.

    Attributes:
        require_preflight:               If ``True`` (default), preflight runs
                                         before every execution.
        allow_execute_without_preflight: Explicit escape hatch — set ``True``
                                         to skip preflight even when
                                         ``require_preflight=True`` (e.g. for
                                         trusted STOP emergency commands).
        safety_context:                  Forwarded as ``context`` arg to
                                         :meth:`BaseAdapter.preflight`.
    """

    require_preflight:               bool           = True
    allow_execute_without_preflight: bool           = False
    safety_context:                  Dict[str, Any] = field(default_factory=dict)


# ── Retry policy (Phase 6) ──────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """Bounded retry contract for ``execute_with_lifecycle`` Step 4.

    Default values are conservative and backward-compatible:
    ``max_attempts=1`` means no retry — identical to pre-Phase 6 behaviour.

    Attributes:
        max_attempts:       Total attempts including the first.
                            ``1`` (default) disables retry.
        retryable_reasons:  Reason codes eligible for retry.  Default covers
                            transient execute failures for all three adapters.
        backoff_ms:         Fixed delay between attempts in milliseconds.
                            ``0`` (default) means no delay.
        non_retryable_ops:  RouteOp values that never retry regardless of reason.
                            Default: ``{"stop"}`` — STOP must not be retried.
    """

    max_attempts:      int       = 1
    retryable_reasons: frozenset = field(
        default_factory=lambda: frozenset({
            LifecycleReason.EXECUTE_FAILED,
            "spot_command_timeout",
            "cyberdog_action_failed",
        })
    )
    backoff_ms:        int       = 0
    non_retryable_ops: frozenset = field(
        default_factory=lambda: frozenset({"stop"})
    )


# ── Routing policy ─────────────────────────────────────────────────────────


@dataclass
class LifecyclePolicy:
    """Routing policy for session-aware action execution.

    Default values reproduce the pre-Phase 3 direct-passthrough behaviour,
    so existing call sites need no modification.

    Attributes:
        run_preflight:     If True, call preflight() before executing the action.
                           Superseded by ``safety_policy`` when that is set.
        close_after:       If True, call close_session() after execution completes.
        session_config:    Forwarded as ``config`` arg to open_session().
        preflight_context: Forwarded as ``context`` arg to preflight()
                           (legacy; superseded by SafetyPolicy.safety_context).
        safety_policy:     Phase 6 SafetyPolicy gate.  When ``None`` (default)
                           the legacy ``run_preflight`` flag governs preflight.
        retry_policy:      Phase 6 RetryPolicy.  When ``None`` (default) or
                           ``max_attempts=1``, no retry is performed.
    """

    run_preflight:     bool                  = False
    close_after:       bool                  = False
    session_config:    Dict[str, Any]        = field(default_factory=dict)
    preflight_context: Dict[str, Any]        = field(default_factory=dict)
    safety_policy:     SafetyPolicy | None   = None
    retry_policy:      "RetryPolicy | None"  = None
