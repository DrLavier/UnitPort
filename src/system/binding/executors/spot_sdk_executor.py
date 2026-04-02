#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spot SDK executor for velocity_move."""

from __future__ import annotations

from typing import Any, Dict

from src.system.binding.executors.base import ExecutorResult
from src.system.binding.executors.sdk_executor import SDKOperationExecutor
from src.system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE


class SpotSDKExecutor(SDKOperationExecutor):
    def _execute_velocity_move(self, binding_output: BindingOutput, context: Dict[str, Any]) -> ExecutorResult:
        adapter = context.get("robot_model")
        if adapter is None:
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, "No robot_model in execution context")
        fn = getattr(adapter, "velocity_move", None)
        if not callable(fn):
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, f"adapter {type(adapter).__name__!r} has no velocity_move() method")
        params = binding_output.params
        try:
            success = fn(params["vx"], params["vy"], params["vyaw"], params["duration"])
        except Exception as exc:
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, str(exc))
        if success:
            return ExecutorResult.ok(OPERATION_VELOCITY_MOVE)
        return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, "adapter.velocity_move() returned False")
