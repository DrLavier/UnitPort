"""Service action routing — Phase 3 session-aware orchestration.

Legacy passthrough methods (run_action / stop / get_sensor_data / health)
are preserved unchanged.  Phase 3 adds ``execute_with_lifecycle`` which
orchestrates the full open → preflight → execute → close flow.
"""

from __future__ import annotations

import logging as _logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .service_registry import ServiceRegistry
from .lifecycle import LifecyclePolicy, LifecycleReason, RetryPolicy, SafetyPolicy
from .settings_validator import validate_settings
from .telemetry import TelemetryCollector, new_trace_id

_log = _logging.getLogger("unitport.service.router")


# ── Route operation identifiers ────────────────────────────────────────────


class RouteOp:
    """Operation identifiers for :meth:`ServiceRouter.execute_with_lifecycle`.

    Values are stable constants — do not rename once published.
    """

    RUN_ACTION      = "run_action"
    GET_SENSOR_DATA = "get_sensor_data"
    STOP            = "stop"


# ── Internal result helpers ────────────────────────────────────────────────


def _route_error(
    reason: str,
    stage: str,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a failed lifecycle route result."""
    return {
        "status":      "error",
        "reason":      reason,
        "stage":       stage,
        "payload":     None,
        "diagnostics": diagnostics,
    }


def _route_ok(
    payload: Any,
    diagnostics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a successful lifecycle route result."""
    return {
        "status":      "ok",
        "reason":      "",
        "stage":       "execute",
        "payload":     payload,
        "diagnostics": diagnostics or {},
    }


def _with_adapter(result: Dict[str, Any], adapter_name: str) -> Dict[str, Any]:
    """Inject ``adapter_name`` into a route result for cross-brand diagnostics.

    Called by ``execute_with_lifecycle`` on every return path so callers can
    always identify which adapter produced the result without parsing reason
    code prefixes.
    """
    result["adapter_name"] = adapter_name
    return result


def _inject_telemetry(
    result: Dict[str, Any],
    collector: TelemetryCollector,
) -> Dict[str, Any]:
    """Attach accumulated telemetry events to ``result["diagnostics"]["telemetry"]``.

    Called on every return path so telemetry is always present in the result.
    """
    result["diagnostics"]["telemetry"] = collector.events
    return result


def _dispatch(adapter: Any, op: str, op_args: Dict[str, Any]) -> Any:
    """Dispatch the named operation to the adapter.

    Args:
        adapter:  Any object implementing the BaseAdapter interface.
        op:       One of :class:`RouteOp` constants.
        op_args:  Operation arguments.
                  - RUN_ACTION:      ``{"action": str, "params": dict}``
                  - GET_SENSOR_DATA: ``{}``
                  - STOP:            ``{}``

    Returns:
        Adapter return value (``None`` for STOP).

    Raises:
        ValueError: if ``op`` is not a recognised RouteOp constant.
        Any exception raised by the adapter during execution.
    """
    if op == RouteOp.RUN_ACTION:
        action = op_args.get("action", "")
        params = op_args.get("params", {})
        return adapter.run_action(action, **params)
    elif op == RouteOp.GET_SENSOR_DATA:
        return adapter.get_sensor_data()
    elif op == RouteOp.STOP:
        adapter.stop()
        return None
    else:
        raise ValueError(f"Unknown route operation: {op!r}")


# ── Router ─────────────────────────────────────────────────────────────────


@dataclass
class ServiceRouter:
    """Routes actions and queries to the appropriate service adapter.

    Phase 1 legacy interface (direct passthrough — never removed):
        register_adapter / get_adapter / run_action / stop /
        get_sensor_data / health

    Phase 3 lifecycle interface:
        execute_with_lifecycle
    """

    registry: ServiceRegistry = field(default_factory=ServiceRegistry)

    # ── Adapter management ─────────────────────────────────────────────────

    def register_adapter(self, name: str, adapter: Any) -> None:
        """Register an adapter to the underlying registry."""
        self.registry.register(name, adapter)

    def get_adapter(self, adapter_name: str) -> Any:
        """Return adapter instance or raise KeyError if missing."""
        return self.registry.require(adapter_name)

    # ── Legacy direct-passthrough methods (Phase 1 — never removed) ────────

    def run_action(self, adapter_name: str, action: str, **params: Any) -> Any:
        """Route an action to the named adapter."""
        _log.warning(
            "DEPRECATED: ServiceRouter.run_action() is a Phase 1 legacy passthrough; "
            "use execute_with_lifecycle(op=RouteOp.RUN_ACTION) instead. "
            "This method will be removed in a future release."
        )
        return self.get_adapter(adapter_name).run_action(action, **params)

    def stop(self, adapter_name: str) -> None:
        """Stop the named adapter."""
        _log.warning(
            "DEPRECATED: ServiceRouter.stop() is a Phase 1 legacy passthrough; "
            "use execute_with_lifecycle(op=RouteOp.STOP) instead. "
            "This method will be removed in a future release."
        )
        self.get_adapter(adapter_name).stop()

    def get_sensor_data(self, adapter_name: str) -> Dict[str, Any]:
        """Retrieve sensor data from the named adapter."""
        _log.warning(
            "DEPRECATED: ServiceRouter.get_sensor_data() is a Phase 1 legacy passthrough; "
            "use execute_with_lifecycle(op=RouteOp.GET_SENSOR_DATA) instead. "
            "This method will be removed in a future release."
        )
        return self.get_adapter(adapter_name).get_sensor_data()

    def health(self, adapter_name: str) -> Dict[str, Any]:
        """Retrieve health information from the named adapter."""
        _log.warning(
            "DEPRECATED: ServiceRouter.health() is a Phase 1 legacy passthrough; "
            "use execute_with_lifecycle() instead. "
            "This method will be removed in a future release."
        )
        return self.get_adapter(adapter_name).health()

    # ── Phase 3: session-aware lifecycle routing ───────────────────────────

    def execute_with_lifecycle(
        self,
        adapter_name: str,
        op: str,
        op_args: Dict[str, Any] | None = None,
        policy: LifecyclePolicy | None = None,
    ) -> Dict[str, Any]:
        """Execute an adapter operation through the full lifecycle flow.

        Orchestration (ref §2.4, §8.6 of ``system/service/README.md``)::

            0. settings validation     → PREFLIGHT_FAILED on error  [if brand in session_config]
            1. Acquire adapter         → ADAPTER_NOT_FOUND on miss
            2. open_session(config)    → SESSION_OPEN_FAILED on error
            3. preflight gate          → Phase 6: SAFETY_GATE_BLOCKED on error [if policy.safety_policy set]
                                         Legacy: PREFLIGHT_FAILED on error      [if policy.run_preflight]
                                         (close_session called as best-effort cleanup on any preflight failure)
            4. Execute operation       → EXECUTE_FAILED on exception
                                         (close_session called if policy.close_after)
            5. close_session()         → failure logged in diagnostics, payload NOT masked
                                                                       [if policy.close_after]
            6. Return structured result

        Args:
            adapter_name: Registry key of the target adapter.
            op:           One of :class:`RouteOp` constants.
            op_args:      Operation arguments dict:

                          - ``RUN_ACTION``      → ``{"action": str, "params": dict}``
                          - ``GET_SENSOR_DATA`` → ``{}`` or ``None``
                          - ``STOP``            → ``{}`` or ``None``

            policy:       :class:`LifecyclePolicy`; defaults to
                          ``LifecyclePolicy()`` which reproduces legacy
                          direct-passthrough behaviour (no preflight,
                          session stays open).

        Returns:
            Plain dict with keys::

                {
                    "status":       "ok" | "error",
                    "reason":       str,   # LifecycleReason constant; "" on ok
                    "stage":        str,   # failing stage, or "execute" on success
                    "payload":      Any,   # adapter return value; None on error
                    "diagnostics":  dict,  # structured failure / warning context;
                                   #   always contains "telemetry": List[dict]
                    "adapter_name": str,   # registry key of the target adapter (Phase 4)
                }
        """
        policy  = policy or LifecyclePolicy()
        op_args = op_args or {}

        # ── Telemetry collector — one per execute_with_lifecycle call ──────
        _brand_for_tel = (
            (policy.session_config.get("brand") or "")
            if policy.session_config else ""
        )
        _collector = TelemetryCollector(
            trace_id=new_trace_id(),
            adapter_name=adapter_name,
            brand=_brand_for_tel,
            operation=op,
        )

        # ── Step 0: settings validation (Phase 5 — brand-gated) ───────────
        # Only runs when session_config contains a "brand" key.
        # Callers that pass empty session_config (legacy) are unaffected.
        _brand = policy.session_config.get("brand") if policy.session_config else None
        if _brand:
            _log.debug(
                "lifecycle[settings_validation] adapter=%s brand=%s",
                adapter_name, _brand,
            )
            _collector.start("settings_validation")
            val_result = validate_settings(_brand, policy.session_config)
            if val_result["status"] != "ok":
                _log.warning(
                    "lifecycle[settings_validation] failed adapter=%s brand=%s "
                    "missing=%s invalid=%s",
                    adapter_name, _brand,
                    val_result["missing"], val_result["invalid"],
                )
                _collector.end(
                    "settings_validation", "error",
                    LifecycleReason.PREFLIGHT_FAILED,
                )
                err = _route_error(
                    LifecycleReason.PREFLIGHT_FAILED,
                    "settings_validation",
                    {
                        "brand":   _brand,
                        "message": "Settings validation failed",
                        "context": {
                            "stage":        "settings_validation",
                            "adapter_name": adapter_name,
                            "missing":      val_result["missing"],
                            "invalid":      val_result["invalid"],
                        },
                        "validation": val_result,
                    },
                )
                return _with_adapter(_inject_telemetry(err, _collector), adapter_name)
            _collector.end("settings_validation", "ok")
            _log.debug(
                "lifecycle[settings_validation] ok adapter=%s brand=%s",
                adapter_name, _brand,
            )

        # ── Step 1: acquire adapter ────────────────────────────────────────
        adapter = self.registry.get(adapter_name)
        if adapter is None:
            _log.warning("lifecycle[acquire] adapter not found: %s", adapter_name)
            err = _route_error(
                LifecycleReason.ADAPTER_NOT_FOUND,
                "acquire",
                {"adapter_name": adapter_name},
            )
            return _with_adapter(_inject_telemetry(err, _collector), adapter_name)
        _log.debug("lifecycle[acquire] adapter=%s op=%s", adapter_name, op)

        # ── Step 2: open_session ───────────────────────────────────────────
        _log.debug("lifecycle[open_session] adapter=%s", adapter_name)
        _collector.start("open_session")
        open_res = adapter.open_session(config=policy.session_config)
        if open_res.get("status") != "ok":
            _log.warning(
                "lifecycle[open_session] failed adapter=%s reason=%s",
                adapter_name, open_res.get("reason", ""),
            )
            _collector.end(
                "open_session", "error",
                LifecycleReason.SESSION_OPEN_FAILED,
            )
            err = _route_error(
                LifecycleReason.SESSION_OPEN_FAILED,
                "open_session",
                open_res.get("diagnostics", {}),
            )
            return _with_adapter(_inject_telemetry(err, _collector), adapter_name)
        _collector.end("open_session", "ok")

        # ── Step 3: preflight gate ─────────────────────────────────────────
        # Phase 6 path: SafetyPolicy governs the gate; failure → SAFETY_GATE_BLOCKED.
        # Legacy path:  run_preflight flag governs; failure → PREFLIGHT_FAILED (unchanged).
        _sp = policy.safety_policy
        if _sp is not None:
            _run_preflight    = _sp.require_preflight and not _sp.allow_execute_without_preflight
            _preflight_ctx    = _sp.safety_context
            _preflight_reason = LifecycleReason.SAFETY_GATE_BLOCKED
        else:
            _run_preflight    = policy.run_preflight
            _preflight_ctx    = policy.preflight_context
            _preflight_reason = LifecycleReason.PREFLIGHT_FAILED

        if _run_preflight:
            _log.debug("lifecycle[preflight] adapter=%s", adapter_name)
            _collector.start("preflight")
            pre_res = adapter.preflight(context=_preflight_ctx)
            if pre_res.get("status") != "ok":
                _log.warning(
                    "lifecycle[preflight] failed adapter=%s reason=%s",
                    adapter_name, pre_res.get("reason", ""),
                )
                _collector.end("preflight", "error", _preflight_reason)
                adapter.close_session()   # best-effort cleanup before returning
                err = _route_error(
                    _preflight_reason,
                    "preflight",
                    pre_res.get("diagnostics", {}),
                )
                return _with_adapter(_inject_telemetry(err, _collector), adapter_name)
            _collector.end("preflight", "ok")

        # ── Step 4: execute (with optional bounded retry) ─────────────────
        _rp           = policy.retry_policy
        _max_attempts = _rp.max_attempts if _rp is not None else 1
        _op_excluded  = _rp is not None and op in _rp.non_retryable_ops

        _last_exc: Optional[Exception] = None
        _attempts_made = 0

        for _attempt in range(1, _max_attempts + 1):
            _attempts_made      = _attempt
            _collector.attempt  = _attempt
            _log.debug(
                "lifecycle[execute] op=%s adapter=%s attempt=%d/%d",
                op, adapter_name, _attempt, _max_attempts,
            )
            _collector.start("execute")
            try:
                payload = _dispatch(adapter, op, op_args)
                _last_exc = None
                _collector.end("execute", "ok")
                break  # success — exit retry loop
            except Exception as exc:
                _last_exc = exc
                _log.warning(
                    "lifecycle[execute] failed op=%s adapter=%s attempt=%d: %s",
                    op, adapter_name, _attempt, exc,
                )
                _collector.end(
                    "execute", "error",
                    LifecycleReason.EXECUTE_FAILED,
                    diagnostics={"message": str(exc), "op": op, "attempt": _attempt},
                )
                # Can we retry? (reason retryable, op not excluded, budget remains)
                _can_retry = (
                    _rp is not None
                    and not _op_excluded
                    and LifecycleReason.EXECUTE_FAILED in _rp.retryable_reasons
                    and _attempt < _max_attempts
                )
                if not _can_retry:
                    break  # give up immediately
                # Apply fixed backoff before next attempt
                if _rp.backoff_ms > 0:
                    _time.sleep(_rp.backoff_ms / 1000.0)

        if _last_exc is not None:
            # Determine final failure reason
            _budget_exhausted = (
                _rp is not None
                and not _op_excluded
                and LifecycleReason.EXECUTE_FAILED in _rp.retryable_reasons
                and _max_attempts > 1
                and _attempts_made == _max_attempts
            )
            _final_reason = (
                LifecycleReason.RETRY_BUDGET_EXHAUSTED
                if _budget_exhausted
                else LifecycleReason.EXECUTE_FAILED
            )
            if policy.close_after:
                adapter.close_session()   # best-effort cleanup
            err = _route_error(
                _final_reason,
                "execute",
                {"message": str(_last_exc), "op": op, "attempts": _attempts_made},
            )
            return _with_adapter(_inject_telemetry(err, _collector), adapter_name)

        # ── Step 5: close_session (policy-gated) ──────────────────────────
        close_diagnostics: Dict[str, Any] = {}
        if policy.close_after:
            _log.debug("lifecycle[close_session] adapter=%s", adapter_name)
            _collector.start("close_session")
            close_res = adapter.close_session()
            if close_res.get("status") != "ok":
                _log.warning(
                    "lifecycle[close_session] failed adapter=%s reason=%s",
                    adapter_name, close_res.get("reason", ""),
                )
                _collector.end(
                    "close_session", "error",
                    LifecycleReason.SESSION_CLOSE_FAILED,
                )
                # Surface close failure in diagnostics but do NOT mask payload
                close_diagnostics = {
                    "close_session_failed": True,
                    "close_reason":         close_res.get("reason", ""),
                    "close_diagnostics":    close_res.get("diagnostics", {}),
                }
            else:
                _collector.end("close_session", "ok")

        # ── Step 6: success ────────────────────────────────────────────────
        _log.debug("lifecycle[success] op=%s adapter=%s", op, adapter_name)
        ok = _route_ok(payload, close_diagnostics or None)
        return _with_adapter(_inject_telemetry(ok, _collector), adapter_name)
