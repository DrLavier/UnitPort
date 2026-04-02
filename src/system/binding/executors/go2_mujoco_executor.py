#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Go2MujocoExecutor — concrete MuJoCo executor for velocity_move.

Architectural Decisions (from DEVELOP_TODO.txt)
------------------------------------------------
- Decision 1: Humanoid robots (g1, h1, h1_2) use independent walking semantics,
  NOT quadruped trot gait. Implemented via _walk_humanoid() in UnitreeModel.
- Decision 2: go2w maintains MuJoCo unsupported status (wheeled quadruped).
- Decision 4: SDK-only path is NOT the acceptance criteria for this round.
  Minimum closed loop is MuJoCo runtime; SDK path is secondary.

Supported Models (Circle 3 Step 3.3)
------------------------------------
go2   — quadruped trot gait (adapter.velocity_move_mujoco() -> _walk_trot_gait_quadruped())
b2    — quadruped trot gait with adjusted parameters
g1    — humanoid independent walking (_walk_humanoid())
h1    — humanoid independent walking (_walk_humanoid())
h1_2  — humanoid independent walking (_walk_humanoid())
go2w  — returns ExecutorResult.unsupported() explicitly at executor level

Context key
-----------
``context["robot_model"]`` — the active adapter instance.  Must expose a
``velocity_move_mujoco(vx, vy, vyaw, duration) -> bool`` method.
"""

from __future__ import annotations

from typing import Any, Dict

from src.system.binding.executors.base import ExecutorResult
from src.system.binding.executors.mujoco_executor import MujocoOperationExecutor
from src.system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE
from src.system.models.model_registry import mujoco_supported_models

_UNSUPPORTED_REASON = {
    "b1": "b1 MuJoCo model not available (SDK-only)",
}

# Supported robot types for MuJoCo velocity_move
_SUPPORTED_ROBOT_TYPES = frozenset(mujoco_supported_models("unitree"))


class Go2MujocoExecutor(MujocoOperationExecutor):
    """MuJoCo executor for Unitree velocity_move operations.

    Overrides ``_execute_velocity_move()`` to:
    - Return ``unsupported`` for unsupported models (go2w, a1, b1, b2w).
    - Call ``adapter.velocity_move_mujoco()`` for supported models (go2, b2, g1, h1, h1_2).
    """

    def _execute_velocity_move(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        robot_type = (binding_output.robot_type or "").strip().lower()

        # Check if robot type is supported for MuJoCo velocity_move
        if robot_type not in _SUPPORTED_ROBOT_TYPES:
            reason = _UNSUPPORTED_REASON.get(
                robot_type,
                f"{robot_type} MuJoCo velocity_move not implemented"
            )
            return ExecutorResult(
                success=False,
                operation=OPERATION_VELOCITY_MOVE,
                reason=reason,
                diag_code=f"executor.unsupported.{OPERATION_VELOCITY_MOVE}",
            )

        adapter = context.get("robot_model")
        if adapter is None:
            return ExecutorResult.failed(
                OPERATION_VELOCITY_MOVE,
                "No robot_model in execution context",
            )

        fn = getattr(adapter, "velocity_move_mujoco", None)
        if not callable(fn):
            return ExecutorResult.failed(
                OPERATION_VELOCITY_MOVE,
                f"adapter {type(adapter).__name__!r} has no velocity_move_mujoco() method",
            )

        params = binding_output.params
        try:
            success = fn(
                params["vx"],
                params["vy"],
                params["vyaw"],
                params["duration"],
            )
        except Exception as exc:
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, str(exc))

        if success:
            return ExecutorResult.ok(OPERATION_VELOCITY_MOVE)
        return ExecutorResult.failed(
            OPERATION_VELOCITY_MOVE,
            "adapter.velocity_move_mujoco() returned False",
        )
