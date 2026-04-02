"""Abstract base adapter for robot services.

Phase 3 adds four lifecycle methods (open_session, preflight,
close_session, capabilities) with safe default implementations so that
existing adapters (UnitreeAdapter, etc.) require zero code changes.
The legacy interface (connect, run_action, stop, get_sensor_data, health)
is preserved in full.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from src.system.service.lifecycle import LifecycleReason, LifecycleResult


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

        Called by :class:`~src.system.engine.runtime_engine.RuntimeEngine` from the
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

    # ── Phase 6 C4: SkillManifest-aligned action interface ────────────────
    # These methods enable policy-driven execution through the unified
    # ActionDispatcher.  Semantic actions (run_action) remain for non-policy
    # behaviours (scripted sequences, manual commands).

    def accept_action(
        self,
        action: Any,
        skill_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate a policy action vector into an SDK command.

        Parameters
        ----------
        action:
            Numpy array (or list) of action values from the policy or
            UserInputManager.
        skill_manifest:
            The active SkillManifest as a dict (or SkillManifest instance).
            Contains ``action_space_type``, ``action_dim``, etc.

        Returns
        -------
        dict
            ``{"ok": bool, "reason": str}``

        Default implementation is a no-op that returns ok=True.  Concrete
        adapters override this to translate actions to their SDK's command
        format based on the manifest's ``action_space_type``.
        """
        return {"ok": True, "reason": ""}

    def report_sensors(
        self,
        required_sensors: List[str],
    ) -> Dict[str, Any]:
        """Return sensor data matching the SkillManifest's observation keys.

        Parameters
        ----------
        required_sensors:
            List of sensor type strings from ``SkillManifest.required_sensors``
            (e.g. ``["imu", "joint_encoder"]``).

        Returns
        -------
        dict
            Sensor readings keyed by sensor type.  Missing sensors return None.

        Default implementation delegates to ``get_sensor_data()`` and filters
        by the requested sensor types.
        """
        all_data = self.get_sensor_data()
        return {s: all_data.get(s) for s in required_sensors}

    def check_precondition(
        self,
        precondition: Dict[str, Any],
    ) -> bool:
        """Query the real robot state against a SkillManifest precondition.

        Parameters
        ----------
        precondition:
            Dict with keys ``posture``, ``posture_tolerance_rad``,
            ``velocity_max_mps``, ``custom_check``.

        Returns
        -------
        bool
            True if the robot's current state satisfies the precondition.

        Default implementation is permissive — always returns True.
        Concrete adapters should override to query actual robot state
        (joint positions, IMU, etc.).
        """
        return True
