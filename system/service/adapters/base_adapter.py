"""Abstract base adapter for robot services.

Phase 3 adds four lifecycle methods (open_session, preflight,
close_session, capabilities) with safe default implementations so that
existing adapters (UnitreeAdapter, etc.) require zero code changes.
The legacy interface (connect, run_action, stop, get_sensor_data, health)
is preserved in full.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from system.service.lifecycle import LifecycleReason, LifecycleResult


class BaseAdapter(ABC):
    """Interface that every robot service adapter must implement.

    Legacy interface (Phase 1 — all abstract, never removed):
        connect / run_action / stop / get_sensor_data / health

    Phase 3 lifecycle interface (non-abstract, safe defaults):
        open_session / preflight / close_session / capabilities
    """

    # ── Legacy interface (Phase 1 contract — never removed) ───────────────

    @abstractmethod
    def connect(self, **kwargs: Any) -> bool:
        """Establish a connection to the robot or simulator."""
        ...

    @abstractmethod
    def run_action(self, action: str, **params: Any) -> Any:
        """Execute a named action with the given parameters."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop all robot activity."""
        ...

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]:
        """Return the latest sensor readings."""
        ...

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return adapter health / connectivity status."""
        ...

    # ── Phase 3 lifecycle interface (non-abstract, safe defaults) ─────────

    def open_session(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Establish a service session.

        Default implementation delegates to ``connect(**config)`` for backward
        compatibility with adapters that use ``connect()`` as their session
        initialiser.  Concrete adapters may override for richer session logic.

        Args:
            config: Optional key-value pairs forwarded to ``connect()``.

        Returns:
            :class:`LifecycleResult` serialised as a plain dict.
        """
        try:
            success = self.connect(**(config or {}))
            if success:
                return LifecycleResult.ok("open_session").to_dict()
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                {"message": "connect() returned False"},
            ).to_dict()
        except Exception as exc:
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                {"message": str(exc)},
            ).to_dict()

    def preflight(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Validate adapter readiness before action execution.

        Default implementation is permissive — always returns ok.
        Concrete adapters override for real checks (liveness, safety limits,
        capability requirements, etc.).

        Args:
            context: Optional execution context from the caller.

        Returns:
            :class:`LifecycleResult` serialised as a plain dict.
        """
        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self) -> Dict[str, Any]:
        """Tear down a service session.

        Default implementation is a no-op — returns ok without disconnecting.
        Legacy adapters do not require explicit teardown, so this preserves
        their behaviour.

        Returns:
            :class:`LifecycleResult` serialised as a plain dict.
        """
        return LifecycleResult.ok("close_session").to_dict()

    def capabilities(self) -> Dict[str, Any]:
        """Return adapter capability descriptor.

        Default implementation returns empty capability sets — safe minimum
        for adapters that have not yet declared their capabilities.

        Returns::

            {
                "actions": List[str],       # supported canonical action names
                "sensors": List[str],       # supported sensor keys
                "flags":   Dict[str, Any],  # feature flags
            }
        """
        return {"actions": [], "sensors": [], "flags": {}}

    # ── Cycle 3 STAGE-04: preemptive cancel contract ───────────────────────

    def cancel_action(self) -> None:
        """Request best-effort cancellation of any currently in-flight action.

        Called by :class:`~system.runtime.runtime_engine.RuntimeEngine` from the
        *caller's thread* when ``request_cancel()`` is invoked while an action
        is executing on the worker thread.

        Contract
        --------
        - Must be thread-safe: it is called from a different thread than the
          thread that is executing ``run_action()``.
        - Must never raise: exceptions are caught by the caller, but raising
          here is considered a contract violation.
        - May be a true no-op: adapters that cannot interrupt an in-flight
          action at the SDK level are permitted to leave this as-is.  The
          cooperative cancel at node boundaries remains the fallback.

        Override guidance
        -----------------
        Concrete adapters should call their SDK's emergency-stop or command-
        cancel API here.  A safe minimal override is ``self.stop()`` if
        ``stop()`` is thread-safe for the underlying SDK.
        """
        # Default: cooperative-cancel at node boundaries is sufficient.
        pass
