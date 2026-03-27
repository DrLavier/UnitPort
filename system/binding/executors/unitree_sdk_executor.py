#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UnitreeSDKExecutor — concrete SDK executor for velocity_move (Step 7).

Handles backend="sdk" for both unitree/go2 and unitree/go2w through the same
SportClient velocity channel.  No model-specific branching inside this class;
the executor is robot-type agnostic on the SDK side.

Context key
-----------
``context["robot_model"]`` — the active adapter instance.  Must expose a
``velocity_move(vx, vy, vyaw, duration) -> bool`` method.
"""

from __future__ import annotations

from typing import Any, Dict

from system.binding.executors.base import ExecutorResult
from system.binding.executors.sdk_executor import SDKOperationExecutor
from system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE


class UnitreeSDKExecutor(SDKOperationExecutor):
    """SDK executor for Unitree robots (go2 + go2w).

    Overrides ``_execute_velocity_move()`` to call the adapter's
    first-class ``velocity_move(vx, vy, vyaw, duration)`` API.
    All other operations fall through to the base ``unsupported`` response.
    """

    def _execute_velocity_move(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        adapter = context.get("robot_model")
        if adapter is None:
            return ExecutorResult.failed(
                OPERATION_VELOCITY_MOVE,
                "No robot_model in execution context",
            )

        fn = getattr(adapter, "velocity_move", None)
        if not callable(fn):
            return ExecutorResult.failed(
                OPERATION_VELOCITY_MOVE,
                f"adapter {type(adapter).__name__!r} has no velocity_move() method",
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
            "adapter.velocity_move() returned False",
        )
