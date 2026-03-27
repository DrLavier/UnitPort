#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Context - Global Robot State Management

This module provides a centralized way to manage the current robot type and
model instance.  All action nodes should use this context to get the
appropriate robot model based on the user's selection in the UI.

Phase 3 additions (additive - all existing callers unchanged):
  - execute_with_lifecycle wired into run_action / get_sensor_data / stop
  - _lifecycle_route internal helper for structured lifecycle dispatch
  - set_lifecycle_policy / get_lifecycle_policy for policy management
  - BRAND_ADAPTER_MAP extended with Phase 4 placeholder entries

Phase 4 additions (additive - all existing callers unchanged):
  - BRAND_ADAPTER_MAP keys corrected to match canonical brand ids
      "bostiondynamics" -> "spot_sdk"   (was "boston_dynamics")
      "xiaomi"          -> "cyberdog_sdk" (was "cyberdog")
  - _create_adapter_for_brand: factory creates SpotAdapter / CyberDogAdapter
  - _ensure_adapter: uses factory; no silent fallback to unitree_sdk2
  - get_robot_model: safe for adapters that have no get_model() method

Phase 7 STAGE-04 (silent fallback removal - non-additive):
  - _create_model_for_brand: removed (dead code; was silently returning UnitreeModel
    for unknown brands, masking misconfiguration)
  - run_action: removed double-fallback; lifecycle error -> False; exception -> False
  - get_sensor_data: removed double-fallback; lifecycle error -> error dict; exception -> error dict
  - stop: removed double-fallback; lifecycle error -> log_error only; exception -> log_error only

Design Pattern:
    - Singleton-like global state for robot configuration
    - Factory method to create robot model based on brand/type
    - Automatic model routing based on robot_type

Usage:
    from bin.core.robot_context import RobotContext

    # Set robot type (called by UI when user selects robot)
    RobotContext.set_robot_type("go2")

    # Get current robot model (used by action nodes)
    robot = RobotContext.get_robot_model()
    if robot:
        robot.run_action("stand")
"""

import threading
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Tuple
from bin.core.logger import log_info, log_error, log_debug, log_warning
from system.service.service_registry import ServiceRegistry
from system.service.service_router import ServiceRouter, RouteOp
from system.service.lifecycle import LifecyclePolicy
from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
from system.model_registry import (
    canonical_brand_ids,
    get_adapter_key_for_brand,
    get_brand_model_map,
    get_robot_brand_map,
)

if TYPE_CHECKING:
    from models.base import BaseRobotModel


class RobotContext:
    """
    Global Robot Context Manager

    Manages the current robot type and provides the appropriate robot model
    instance based on the selected robot brand and type.

    Supported robot types and their brand mappings:
        - go2, a1, b1, b2, h1       -> unitree         (unitree_sdk2 adapter)
        - spot                       -> bostiondynamics (spot_sdk adapter)
        - cyberdog, cyberdog2        -> xiaomi          (cyberdog_sdk adapter)

    Brand keys match the canonical model registry.
        models/Unitree/       -> "unitree"
        models/BostionDynamics/ -> "bostiondynamics"
        models/XiaoMi/        -> "xiaomi"
    """

    # Global state
    _current_robot_type: str = "go2"
    _current_robot_model: Optional['BaseRobotModel'] = None
    _initialized: bool = False
    _service_registry: ServiceRegistry = ServiceRegistry()
    _service_router: ServiceRouter = ServiceRouter(_service_registry)

    # Phase 3: lifecycle policy (default reproduces pre-Phase 3 passthrough)
    _lifecycle_policy: LifecyclePolicy = LifecyclePolicy()

    # Cycle 3 STAGE-03: thread-local run-scoped policy slot.
    # Each worker thread (e.g. MissionRunThread) can set a per-run policy that
    # is invisible to other threads - no global mutation during mission execution.
    _run_policy_local: threading.local = threading.local()

    @classmethod
    def _get_robot_brand_map(cls) -> Dict[str, str]:
        """Canonical robot->brand map."""
        return dict(get_robot_brand_map())

    @classmethod
    def _get_brand_robots(cls) -> Dict[str, list]:
        """Canonical brand->robots map."""
        return dict(get_brand_model_map(display_names=False))

    # Brand -> adapter name mapping retained as a compatibility view over the
    # canonical model registry.
    BRAND_ADAPTER_MAP: Dict[str, str] = {
        brand_id: get_adapter_key_for_brand(brand_id)
        for brand_id in canonical_brand_ids()
    }

    # -- Policy management (Phase 3) ----------------------------------------

    @classmethod
    def set_lifecycle_policy(cls, policy: LifecyclePolicy) -> None:
        """Set the LifecyclePolicy used by all subsequent lifecycle routes.

        The default ``LifecyclePolicy()`` (run_preflight=False,
        close_after=False) reproduces pre-Phase 3 passthrough behaviour.
        Override for testing or when session-aware routing is desired.
        """
        cls._lifecycle_policy = policy

    @classmethod
    def get_lifecycle_policy(cls) -> LifecyclePolicy:
        """Return the currently active LifecyclePolicy."""
        return cls._lifecycle_policy

    @classmethod
    def make_run_policy(
        cls,
        session_config: Dict[str, Any],
        *,
        base_policy: Optional[LifecyclePolicy] = None,
    ) -> LifecyclePolicy:
        """Create a per-run LifecyclePolicy with *session_config* bound.

        Pure factory - does not mutate any class-level state.  All other
        fields are inherited from *base_policy* (or the current class-level
        policy when *base_policy* is ``None``).

        This is the preferred way to create a run-scoped policy for settings
        injection because it avoids manually constructing ``LifecyclePolicy``
        fields at the call site and keeps class-level state unchanged.

        Args:
            session_config: Dict forwarded to ``LifecyclePolicy.session_config``.
                            Typically ``{**sdk_settings, "brand": brand}``.
            base_policy:    Optional baseline; defaults to
                            ``cls._lifecycle_policy`` when ``None``.

        Returns:
            A fresh :class:`LifecyclePolicy` instance.  Class state is NOT
            mutated.
        """
        base = base_policy if base_policy is not None else cls._lifecycle_policy
        return LifecyclePolicy(
            run_preflight=base.run_preflight,
            close_after=base.close_after,
            session_config=dict(session_config),
            preflight_context=base.preflight_context,
            safety_policy=base.safety_policy,
            retry_policy=base.retry_policy,
        )

    @classmethod
    def set_run_scoped_policy(cls, policy: LifecyclePolicy) -> None:
        """Activate a thread-local run-scoped policy for the calling thread.

        Called by :class:`MissionRunThread` at the start of ``run()`` so that
        all ``RobotContext`` calls from the worker thread use the per-run
        policy without touching the class-level ``_lifecycle_policy``.

        Safe to call from any thread; each thread has its own slot.
        """
        cls._run_policy_local.policy = policy

    @classmethod
    def clear_run_scoped_policy(cls) -> None:
        """Clear the thread-local run-scoped policy for the calling thread.

        Called by :class:`MissionRunThread` in its ``finally`` block so the
        slot is released when the run finishes (even on exception or cancel).
        """
        cls._run_policy_local.policy = None

    @classmethod
    def get_run_scoped_policy(cls) -> Optional[LifecyclePolicy]:
        """Return the active thread-local run-scoped policy, or ``None``.

        Returns ``None`` when the calling thread has no active run-scoped
        policy, in which case ``_lifecycle_route`` falls back to the
        class-level ``_lifecycle_policy``.
        """
        return getattr(cls._run_policy_local, "policy", None)

    # -- Brand/model helpers (unchanged from Phase 2) -----------------------

    @classmethod
    def set_robot_type(cls, robot_type: str) -> bool:
        """
        Set the current robot type and create appropriate model instance.

        Args:
            robot_type: Robot type identifier (e.g., "go2", "a1", "b1")

        Returns:
            True if successfully set, False otherwise
        """
        robot_type = robot_type.lower()

        # Check if robot type is supported
        if robot_type not in cls._get_robot_brand_map():
            log_warning(f"Unknown robot type: {robot_type}, defaulting to 'go2'")
            robot_type = "go2"

        # If same type and already initialized, skip
        if robot_type == cls._current_robot_type and cls._current_robot_model is not None:
            log_debug(f"Robot type already set to: {robot_type}")
            return True

        cls._current_robot_type = robot_type
        cls._current_robot_model = None  # Clear old model
        cls._initialized = False

        log_info(f"Robot type set to: {robot_type} (brand: {cls.get_current_brand()})")
        return True

    @classmethod
    def get_robot_type(cls) -> str:
        """Get the current robot type."""
        return cls._current_robot_type

    @classmethod
    def get_current_brand(cls) -> str:
        """Get the brand of the current robot type."""
        return cls._get_robot_brand_map().get(cls._current_robot_type, "unitree")

    @classmethod
    def get_robot_model(cls, force_reinit: bool = False) -> Optional['BaseRobotModel']:
        """
        Get the robot model instance for the current robot type.
        Creates the model lazily on first access.

        Args:
            force_reinit: If True, recreate the model even if already initialized

        Returns:
            Robot model instance, or None if creation fails
        """
        if cls._current_robot_model is not None and cls._initialized and not force_reinit:
            return cls._current_robot_model

        # Create model based on brand
        brand = cls.get_current_brand()
        robot_type = cls._current_robot_type

        try:
            adapter = cls._ensure_adapter(brand, robot_type, force_reinit=force_reinit)
            if adapter is None:
                return None
            # Only UnitreeAdapter exposes get_model(); other adapters route directly
            model = adapter.get_model() if hasattr(adapter, "get_model") else None
            if model:
                cls._current_robot_model = model
                cls._initialized = True
                log_info(f"Robot model created: {brand}/{robot_type}")
            else:
                # Non-Unitree adapter: mark initialized even without a model object
                cls._initialized = True
            return model
        except Exception as e:
            log_error(f"Failed to create robot model: {e}")
            return None

    @classmethod
    def _create_adapter_for_brand(cls, brand: str, robot_type: str):
        """Factory: instantiate the right adapter for *brand*.

        Returns None for unknown brands - no silent Unitree fallback for
        Phase 4+ brands so callers can surface the error properly.
        """
        if brand == "unitree":
            return UnitreeAdapter(robot_type)
        elif brand == "bostiondynamics":
            from system.service.adapters.spot_sdk.adapter import SpotAdapter
            return SpotAdapter()
        elif brand == "xiaomi":
            from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter
            return CyberDogAdapter()
        else:
            log_warning(f"Unknown brand '{brand}': no adapter factory - routing unavailable")
            return None

    @classmethod
    def _ensure_adapter(cls, brand: str, robot_type: str, force_reinit: bool = False):
        """Ensure adapter is registered and bound to current robot type."""
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand)
        if adapter_name is None:
            log_error(f"No adapter registered for brand '{brand}' - routing unavailable")
            return None
        adapter = cls._service_registry.get(adapter_name)

        if adapter is None:
            adapter = cls._create_adapter_for_brand(brand, robot_type)
            if adapter is None:
                log_error(f"No adapter available for brand '{brand}'")
                return None
            cls._service_registry.register(adapter_name, adapter)

        try:
            adapter.connect(robot_type=robot_type, force_reinit=force_reinit)
        except Exception as e:
            log_error(f"Adapter connect failed ({adapter_name}): {e}")
            return None

        return adapter

    # -- Phase 3: lifecycle-aware routing ----------------------------------

    @classmethod
    def _lifecycle_route(
        cls,
        op: str,
        adapter_name: str,
        op_args: Dict[str, Any] | None = None,
        policy: LifecyclePolicy | None = None,
    ) -> Tuple[bool, Any]:
        """Dispatch an operation through ServiceRouter.execute_with_lifecycle.

        Args:
            op:           One of :class:`RouteOp` constants.
            adapter_name: Registry key of the target adapter.
            op_args:      Operation arguments forwarded to execute_with_lifecycle.
            policy:       LifecyclePolicy to use; defaults to class-level policy.

        Returns:
            ``(success: bool, payload: Any)``
        """
        # Resolution order (Cycle 3 STAGE-03):
        #   1) explicit `policy` argument - highest priority
        #   2) thread-local run-scoped policy (set by MissionRunThread)
        #   3) class-level _lifecycle_policy - fallback / legacy default
        resolved_policy = (
            policy
            or getattr(cls._run_policy_local, "policy", None)
            or cls._lifecycle_policy
        )
        result = cls._service_router.execute_with_lifecycle(
            adapter_name,
            op,
            op_args,
            resolved_policy,
        )
        return result["status"] == "ok", result["payload"]

    # -- Public entry points (Phase 1 signatures preserved exactly) --------

    @classmethod
    def run_action(cls, action_name: str, **kwargs) -> bool:
        """
        Execute an action on the current robot model.

        This is a convenience method that action nodes can use directly.

        Args:
            action_name: Name of the action to execute
            **kwargs: Action parameters

        Returns:
            True if action executed successfully, False otherwise
        """
        brand        = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand)
        if adapter_name is None:
            log_error(f"Cannot execute action '{action_name}': unknown brand '{brand}'")
            return False
        adapter      = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            log_error(f"Cannot execute action '{action_name}': Adapter unavailable")
            return False
        try:
            ok, payload = cls._lifecycle_route(
                RouteOp.RUN_ACTION,
                adapter_name,
                {"action": action_name, "params": kwargs},
            )
            if not ok:
                log_error(f"Action '{action_name}' failed via lifecycle route")
                return False
            return bool(payload)
        except Exception as e:
            log_error(f"Action routing failed ({action_name}): {e}")
            return False

    @classmethod
    def get_sensor_data(cls) -> Dict[str, Any]:
        """
        Get sensor data from the current robot model.

        Returns:
            Sensor data dictionary
        """
        brand        = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand)
        if adapter_name is None:
            log_error(f"get_sensor_data: unknown brand '{brand}' - routing unavailable")
            return {'error': f"Unknown brand '{brand}'"}
        adapter      = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            return {'error': 'No robot model available'}
        try:
            ok, payload = cls._lifecycle_route(RouteOp.GET_SENSOR_DATA, adapter_name)
            if not ok:
                log_error("Sensor data retrieval failed via lifecycle route")
                return {'error': 'Sensor data unavailable'}
            return payload
        except Exception as e:
            log_error(f"Sensor routing failed: {e}")
            return {'error': str(e)}

    @classmethod
    def stop(cls) -> None:
        """Stop the current robot."""
        brand        = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand)
        if adapter_name is None:
            log_warning(f"stop: unknown brand '{brand}' - no-op")
            return
        adapter      = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            return
        try:
            ok, _ = cls._lifecycle_route(RouteOp.STOP, adapter_name)
            if not ok:
                log_error("Stop command failed via lifecycle route")
        except Exception as e:
            log_error(f"Stop routing failed: {e}")

    @classmethod
    def _get_active_model(cls) -> Optional['BaseRobotModel']:
        """Resolve the active adapter-backed model instance, if any."""
        brand = cls.get_current_brand()
        adapter = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            return cls._current_robot_model
        if hasattr(adapter, "get_model"):
            try:
                model = adapter.get_model()
                if model is not None:
                    cls._current_robot_model = model
                    cls._initialized = True
                return model
            except Exception as e:
                log_error(f"Failed to resolve active model: {e}")
                return cls._current_robot_model
        return cls._current_robot_model

    @classmethod
    def pause(cls) -> bool:
        """Best-effort pause for the active robot model."""
        model = cls._get_active_model()
        if model is None:
            return False
        pause_fn = getattr(model, "pause", None)
        if not callable(pause_fn):
            return False
        try:
            pause_fn()
            return True
        except Exception as e:
            log_error(f"Pause failed: {e}")
            return False

    @classmethod
    def resume(cls) -> bool:
        """Best-effort resume for the active robot model."""
        model = cls._get_active_model()
        if model is None:
            return False
        resume_fn = getattr(model, "resume", None)
        if not callable(resume_fn):
            return False
        try:
            resume_fn()
            return True
        except Exception as e:
            log_error(f"Resume failed: {e}")
            return False

    @classmethod
    def cancel_action(cls) -> bool:
        """Request best-effort cancellation of the active in-flight action."""
        brand = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand)
        if adapter_name is not None:
            adapter = cls._service_registry.get(adapter_name)
            if adapter is not None:
                cancel_fn = getattr(adapter, "cancel_action", None)
                if callable(cancel_fn):
                    try:
                        cancel_fn()
                        return True
                    except Exception as e:
                        log_error(f"Adapter cancel_action failed: {e}")
        model = cls._get_active_model()
        if model is None:
            return False
        stop_fn = getattr(model, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
                return True
            except Exception as e:
                log_error(f"Model stop fallback for cancel failed: {e}")
        return False

    @classmethod
    def reset_simulation(cls) -> bool:
        """Best-effort simulation reset on the active robot model."""
        model = cls._get_active_model()
        if model is None:
            return False
        reset_fn = getattr(model, "reset_simulation", None)
        if not callable(reset_fn):
            return False
        try:
            return bool(reset_fn())
        except Exception as e:
            log_error(f"Simulation reset failed: {e}")
            return False

    # -- Additional helpers (unchanged from Phase 2) ------------------------

    @classmethod
    def get_available_actions(cls) -> list:
        """Get list of available actions for current robot."""
        robot = cls.get_robot_model()
        if robot:
            return robot.get_available_actions()
        return []

    @classmethod
    def is_available(cls) -> bool:
        """Check if robot model is available and ready."""
        robot = cls.get_robot_model()
        return robot is not None and robot.is_available

    @classmethod
    def get_supported_robots(cls) -> Dict[str, list]:
        """Get all supported robots grouped by brand."""
        return dict(cls._get_brand_robots())

    @classmethod
    def reset(cls) -> None:
        """Reset the context to initial state."""
        cls._current_robot_type = "go2"
        cls._current_robot_model = None
        cls._initialized = False
        cls._service_registry = ServiceRegistry()
        cls._service_router = ServiceRouter(cls._service_registry)
        cls._lifecycle_policy = LifecyclePolicy()       # Phase 3: reset policy too
        cls._run_policy_local = threading.local()       # Cycle 3: reset thread-local slot
        log_debug("Robot context reset")


# Convenience functions for direct import
def get_robot() -> Optional['BaseRobotModel']:
    """Convenience function to get current robot model."""
    return RobotContext.get_robot_model()


def run_action(action_name: str, **kwargs) -> bool:
    """Convenience function to run action on current robot."""
    return RobotContext.run_action(action_name, **kwargs)


def get_sensor_data() -> Dict[str, Any]:
    """Convenience function to get sensor data from current robot."""
    return RobotContext.get_sensor_data()
