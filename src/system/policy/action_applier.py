from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .joint_name_utils import canonicalize_joint_name, canonicalize_joint_names
from .manifest_schema import CheckpointBundle
from .sim_env_context import SimEnvContext

_SUPPORTED_ACTION_TYPES = {"joint_position", "joint_velocity", "torque"}

log = logging.getLogger(__name__)


@dataclass
class ActionSpec:
    action_type: str
    dim: int
    joint_names: List[str]
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    scale: float = 1.0


class ActionApplier:
    """Converts raw policy outputs into control-space action vectors."""

    def __init__(self, bundle: CheckpointBundle):
        if bundle.action_type not in _SUPPORTED_ACTION_TYPES:
            raise ValueError(
                f"Unsupported action type '{bundle.action_type}'. "
                f"Runtime supports: {sorted(_SUPPORTED_ACTION_TYPES)}"
            )
        raw_action_space = dict((bundle.raw_manifest or {}).get("action_space") or {})
        clip = raw_action_space.get("clip")
        clip_val = None
        try:
            clip_val = float(clip) if clip is not None else None
        except (TypeError, ValueError):
            clip_val = None
        self._spec = ActionSpec(
            action_type=bundle.action_type,
            dim=bundle.action_dim,
            joint_names=list(bundle.joint_names),
            clip_min=(-clip_val if clip_val is not None else None),
            clip_max=clip_val,
            scale=float(raw_action_space.get("scale", 1.0) or 1.0),
        )

    def expected_dim(self) -> int:
        return self._spec.dim

    def remap_to_env(self, action: np.ndarray, env: SimEnvContext) -> np.ndarray:
        """Remap action from bundle joint order to env joint order."""
        bundle_joints = self._spec.joint_names
        env_joints = list(env.joint_names)

        if bundle_joints == env_joints:
            return action.astype(np.float32)

        if canonicalize_joint_names(bundle_joints) == canonicalize_joint_names(env_joints):
            return action.astype(np.float32)

        bundle_index_by_canonical = {
            canonicalize_joint_name(jname): idx
            for idx, jname in enumerate(bundle_joints)
        }

        env_to_bundle: list[int] = []
        missing: list[str] = []
        for jname in env_joints:
            idx = bundle_index_by_canonical.get(canonicalize_joint_name(jname))
            if idx is None:
                missing.append(jname)
            else:
                env_to_bundle.append(idx)

        if missing:
            raise ValueError(
                f"Cannot remap action: env joints not in bundle: {missing}. "
                f"Bundle joints: {bundle_joints}"
            )

        return action.astype(np.float32)[env_to_bundle]

    def apply(self, raw_action: np.ndarray, env: SimEnvContext) -> np.ndarray:
        """Apply scale, clip, remap, then adapter-specific ctrl conversion."""
        if raw_action.shape[0] != self._spec.dim:
            raise ValueError(
                f"ActionApplier.apply(): input has dimension {raw_action.shape[0]}, "
                f"expected {self._spec.dim}."
            )

        out = raw_action.flatten().astype(np.float32)

        if self._spec.scale != 1.0:
            out = out * np.float32(self._spec.scale)

        if self._spec.clip_min is not None or self._spec.clip_max is not None:
            out = np.clip(out, self._spec.clip_min, self._spec.clip_max).astype(np.float32)

        remapped = self.remap_to_env(out, env)
        adapter = getattr(env, "adapter", None)
        action_to_ctrl = getattr(adapter, "_action_to_ctrl", None)
        if callable(action_to_ctrl):
            try:
                converted = np.asarray(action_to_ctrl(remapped), dtype=np.float32)
                if converted.shape == remapped.shape:
                    return converted
                if converted.ndim == 1 and converted.size == remapped.size:
                    return converted.reshape(remapped.shape)
            except Exception:
                pass
        return remapped

    def dispatch_to_backend(
        self,
        raw_action: np.ndarray,
        env: SimEnvContext,
        *,
        skill_manifest: Any = None,
        context: Optional[Dict[str, Any]] = None,
        backend: str = "mujoco",
    ) -> Dict[str, Any]:
        """Apply action then dispatch through the unified ActionDispatcher.

        This method extends the standard :meth:`apply` pipeline with a
        final dispatch step through the unified ActionDispatcher, routing
        the processed action to the correct sim/real executor.

        Parameters
        ----------
        raw_action:
            Raw policy output action vector.
        env:
            Simulation environment context.
        skill_manifest:
            Active SkillManifest (if available).  Used for action_range
            clamping and backend routing in the dispatcher.
        context:
            Runtime context dict (must include ``"robot_model"`` for SDK path).
        backend:
            Target backend (``"mujoco"`` or ``"sdk"``).

        Returns
        -------
        dict
            ``{"action": np.ndarray, "dispatch_result": ExecutorResult.to_dict()}``
            where ``action`` is the processed action vector and
            ``dispatch_result`` is the executor outcome.
        """
        action = self.apply(raw_action, env)

        dispatch_result = None
        if skill_manifest is not None:
            try:
                from src.system.runtime.action_dispatcher import ActionDispatcher

                dispatcher = ActionDispatcher()
                ctx = dict(context or {})
                ctx.setdefault("robot_model", getattr(env, "adapter", None))

                result = dispatcher.dispatch_policy_action(
                    action,
                    skill_manifest=skill_manifest,
                    context=ctx,
                    backend=backend,
                )
                dispatch_result = result.to_dict()
            except Exception as exc:
                log.debug("ActionDispatcher delegation failed: %s", exc)
                dispatch_result = {
                    "success": False,
                    "operation": "policy_action",
                    "reason": str(exc),
                    "diag_code": "dispatcher.unavailable",
                }

        return {
            "action": action,
            "dispatch_result": dispatch_result,
        }
