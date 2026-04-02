#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spot MuJoCo executor for velocity_move."""

from __future__ import annotations

from typing import Any, Dict

from src.system.binding.executors.base import ExecutorResult
from src.system.binding.executors.mujoco_executor import MujocoOperationExecutor
from src.system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE

_SUPPORTED_ROBOT_TYPES = frozenset({"spot"})


class SpotMujocoExecutor(MujocoOperationExecutor):
    def _execute_velocity_move(self, binding_output: BindingOutput, context: Dict[str, Any]) -> ExecutorResult:
        robot_type = (binding_output.robot_type or "").strip().lower()
        if robot_type not in _SUPPORTED_ROBOT_TYPES:
            return ExecutorResult(success=False, operation=OPERATION_VELOCITY_MOVE, reason=f"{robot_type} MuJoCo velocity_move not implemented", diag_code=f"executor.unsupported.{OPERATION_VELOCITY_MOVE}")
        adapter = context.get("robot_model")
        if adapter is None:
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, "No robot_model in execution context")
        fn = getattr(adapter, "velocity_move_mujoco", None)
        if not callable(fn):
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, f"adapter {type(adapter).__name__!r} has no velocity_move_mujoco() method")
        params = binding_output.params
        try:
            success = fn(params["vx"], params["vy"], params["vyaw"], params["duration"])
        except Exception as exc:
            return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, str(exc))
        if success:
            return ExecutorResult.ok(OPERATION_VELOCITY_MOVE)
        return ExecutorResult.failed(OPERATION_VELOCITY_MOVE, "adapter.velocity_move_mujoco() returned False")
