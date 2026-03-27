#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concrete ActionBinding for locomotion intents.

ForwardBinding — handles ``locomotion.forward`` → ``velocity_move`` for
both ``backend="sdk"`` and ``backend="mujoco"``.

Parameter conversion (Step 5)
------------------------------
Uses ``LocomotionProfile`` from ``system/binding/profiles.py``.  No numeric
velocity constants are hardcoded here; all model-specific values come from
the profile object.

    vx   = speed * profile.max_forward_vx
    vy   = 0.0   (always; no lateral IR param defined yet)
    vyaw = 0.0   (always; no yaw IR param defined yet)
    duration = ir_params["duration"]

Missing profile fallback
------------------------
When no profile exists for a brand/model, a safe conservative default
(``_SAFE_FALLBACK_MAX_VX = 0.3 m/s``) is used so the binding never raises.
A future diagnostic hook can be added here when needed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from system.binding.base import ActionBinding
from system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE
from system.binding.profiles import get_locomotion_profile

# Safe fallback velocity for models with no registered profile.
# Kept deliberately low so unverified models move cautiously.
_SAFE_FALLBACK_MAX_VX: float = 0.3


class ForwardBinding(ActionBinding):
    """Binding for ``locomotion.forward`` → ``velocity_move``.

    Accepts both ``backend="sdk"`` and ``backend="mujoco"``; the executor
    layer decides what to do with the resulting ``BindingOutput``.
    """

    @property
    def intent_id(self) -> str:
        return "locomotion.forward"

    def bind(
        self,
        intent_id: str,
        ir_params: Dict[str, Any],
        brand: str,
        robot_type: str,
        backend: str,
    ) -> Optional[BindingOutput]:
        """Bind Forward intent to velocity_move operation.

        Returns None when the target model does not support Forward locomotion
        (i.e., no LocomotionProfile exists for the brand/model combination).
        This enables proper capability-based filtering at the binding layer.
        """
        # Check if the model has a registered LocomotionProfile
        # If no profile exists, the model does not support Forward locomotion
        profile = get_locomotion_profile(brand, robot_type)
        if profile is None:
            # Model does not support Forward locomotion - return None
            # This allows the runtime to fall back to legacy path or emit diagnostic
            return None

        duration = float(ir_params.get("duration", 1.0))
        max_vx   = profile.max_forward_vx

        if "vx" in ir_params and ir_params.get("vx") is not None:
            vx = float(ir_params.get("vx", 0.0))
        else:
            speed = float(ir_params.get("speed", 0.5))
            vx = speed * max_vx

        if vx < 0.0:
            vx = 0.0
        if vx > max_vx:
            vx = max_vx

        return BindingOutput.velocity_move(
            vx=vx,
            vy=0.0,
            vyaw=0.0,
            duration=duration,
            intent_id=intent_id,
            brand=brand,
            robot_type=robot_type,
            backend=backend,
        )
