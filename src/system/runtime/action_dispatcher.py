"""Unified ActionDispatcher — routes actions to sim/real/hybrid execution targets.

Normalises heterogeneous action outputs into robot-executable commands by:

1. Receiving a :class:`BindingOutput` (from the Binding layer) or raw policy
   action (via :meth:`dispatch_policy_action`).
2. Resolving the correct executor based on ``backend`` (sdk / mujoco) and
   ``brand`` (unitree / spot / cyberdog).
3. Delegating to the executor's ``execute()`` method.
4. Returning an :class:`ExecutorResult` uniformly across all backend targets.

For SkillManifest-aware dispatch (Phase 5+), the dispatcher also handles:
- ``action_space_type`` routing (joint_position, joint_torque, cartesian_delta, etc.)
- ``action_range`` clamping
- Control frequency adaptation (future)

Public API
----------
ActionDispatcher.dispatch(binding_output, context) -> ExecutorResult
ActionDispatcher.dispatch_policy_action(action, skill_manifest, context) -> ExecutorResult
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.system.binding.output import BACKEND_MUJOCO, BACKEND_SDK, BindingOutput
from src.system.binding.executors.base import ExecutorResult, OperationExecutor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executor registry — brand+backend → concrete executor class
# ---------------------------------------------------------------------------

_EXECUTOR_REGISTRY: Dict[str, type] = {}


def _ensure_registry() -> None:
    """Populate the executor registry on first use (lazy imports)."""
    if _EXECUTOR_REGISTRY:
        return

    from src.system.binding.executors.unitree_sdk_executor import UnitreeSDKExecutor
    from src.system.binding.executors.go2_mujoco_executor import Go2MujocoExecutor
    from src.system.binding.executors.spot_sdk_executor import SpotSDKExecutor
    from src.system.binding.executors.spot_mujoco_executor import SpotMujocoExecutor
    from src.system.binding.executors.sdk_executor import SDKOperationExecutor
    from src.system.binding.executors.mujoco_executor import MujocoOperationExecutor

    _EXECUTOR_REGISTRY.update({
        # Brand-specific
        "unitree:sdk": UnitreeSDKExecutor,
        "unitree:mujoco": Go2MujocoExecutor,
        "spot:sdk": SpotSDKExecutor,
        "spot:mujoco": SpotMujocoExecutor,
        # Fallbacks (generic base classes)
        ":sdk": SDKOperationExecutor,
        ":mujoco": MujocoOperationExecutor,
    })


def _resolve_executor(brand: str, backend: str) -> OperationExecutor:
    """Resolve the correct executor instance for a brand+backend pair."""
    _ensure_registry()
    brand_lower = (brand or "").strip().lower()
    key = f"{brand_lower}:{backend}"
    cls = _EXECUTOR_REGISTRY.get(key)
    if cls is None:
        # Fallback to generic executor for this backend
        cls = _EXECUTOR_REGISTRY.get(f":{backend}")
    if cls is None:
        raise ValueError(f"No executor registered for brand={brand!r}, backend={backend!r}")
    return cls()


# ---------------------------------------------------------------------------
# ActionDispatcher
# ---------------------------------------------------------------------------

class ActionDispatcher:
    """Unified dispatcher for sim/real/hybrid action execution.

    Routes :class:`BindingOutput` to the appropriate executor (SDK or MuJoCo)
    based on the ``backend`` and ``brand`` fields.  Returns
    :class:`ExecutorResult` uniformly.

    Usage::

        dispatcher = ActionDispatcher()
        result = dispatcher.dispatch(binding_output, context)
        if not result.success:
            log.warning("Dispatch failed: %s", result.reason)
    """

    def __init__(self) -> None:
        # Cache executor instances per brand:backend to avoid repeated construction
        self._executor_cache: Dict[str, OperationExecutor] = {}

    def _get_executor(self, brand: str, backend: str) -> OperationExecutor:
        """Get or create an executor for the given brand+backend."""
        key = f"{(brand or '').lower()}:{backend}"
        if key not in self._executor_cache:
            self._executor_cache[key] = _resolve_executor(brand, backend)
        return self._executor_cache[key]

    def dispatch(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        """Dispatch a backend operation to the appropriate executor.

        Parameters
        ----------
        binding_output:
            Fully resolved backend operation from the Binding layer.
        context:
            Runtime context dict (typically includes ``"robot_model"``).

        Returns
        -------
        ExecutorResult
            Always returns a result; never raises.
        """
        try:
            executor = self._get_executor(binding_output.brand, binding_output.backend)
            return executor.execute(binding_output, context)
        except Exception as exc:
            return ExecutorResult.failed(
                binding_output.operation,
                f"Dispatch error: {exc}",
            )

    def dispatch_policy_action(
        self,
        action: np.ndarray,
        *,
        skill_manifest: Any,
        context: Dict[str, Any],
        backend: str = BACKEND_MUJOCO,
    ) -> ExecutorResult:
        """Dispatch a raw policy action array through the unified path.

        This is the SkillManifest-aware entry point for policy inference
        output.  It applies action_range clamping from the manifest, then
        passes the action to the appropriate executor via the context dict.

        Parameters
        ----------
        action:
            Raw action vector from policy inference (numpy array).
        skill_manifest:
            The active SkillManifest instance (provides action_space_type,
            action_range, target_robot_family, etc.).
        context:
            Runtime context dict (must include ``"robot_model"``).
        backend:
            Execution backend target (``"sdk"`` or ``"mujoco"``).

        Returns
        -------
        ExecutorResult
        """
        try:
            # Apply action_range clamping from SkillManifest
            clamped = self._clamp_action(action, skill_manifest)

            # Inject clamped action into context for the executor
            ctx = dict(context)
            ctx["policy_action"] = clamped
            ctx["skill_manifest"] = skill_manifest

            # Determine brand from manifest or context
            brand = ""
            if hasattr(skill_manifest, "target_robot_models") and skill_manifest.target_robot_models:
                model_str = skill_manifest.target_robot_models[0]
                brand = model_str.split("_")[0] if "_" in model_str else model_str
            if not brand:
                robot_model = context.get("robot_model")
                brand = getattr(robot_model, "brand", "") if robot_model else ""

            # Build a synthetic BindingOutput for policy action dispatch
            action_type = "joint_position"
            if hasattr(skill_manifest, "action_space_type"):
                action_type = (
                    skill_manifest.action_space_type.value
                    if hasattr(skill_manifest.action_space_type, "value")
                    else str(skill_manifest.action_space_type)
                )

            binding_output = BindingOutput(
                operation=f"policy_action_{action_type}",
                params={
                    "action": clamped.tolist(),
                    "action_space_type": action_type,
                    "action_dim": int(clamped.shape[0]),
                },
                intent_id=f"policy.{getattr(skill_manifest, 'skill_id', 'unknown')}",
                brand=brand,
                robot_type=self._extract_robot_type(skill_manifest),
                backend=backend,
            )

            return self.dispatch(binding_output, ctx)
        except Exception as exc:
            return ExecutorResult.failed("policy_action", f"Policy dispatch error: {exc}")

    @staticmethod
    def _clamp_action(action: np.ndarray, skill_manifest: Any) -> np.ndarray:
        """Apply action_range clamping from the SkillManifest."""
        action = np.asarray(action, dtype=np.float32).flatten()
        action_range = getattr(skill_manifest, "action_range", None)
        if action_range and isinstance(action_range, dict):
            # Per-joint clamping (joint_name → [min, max])
            # For now, extract global min/max from all ranges
            all_mins = []
            all_maxs = []
            for bounds in action_range.values():
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    all_mins.append(bounds[0])
                    all_maxs.append(bounds[1])
            if all_mins and all_maxs:
                action = np.clip(action, min(all_mins), max(all_maxs))
        return action

    @staticmethod
    def _extract_robot_type(skill_manifest: Any) -> str:
        """Extract robot_type string from manifest target_robot_models."""
        models = getattr(skill_manifest, "target_robot_models", [])
        if models:
            model_str = models[0]
            # "unitree_go2" → "go2"
            parts = model_str.split("_")
            return parts[-1] if len(parts) > 1 else model_str
        return ""
