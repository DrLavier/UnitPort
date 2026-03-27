#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MujocoOperationExecutor — base executor for backend="mujoco" operations (Step 4).

At Step 4 this class returns ``ExecutorResult.unsupported(...)`` for every
operation.  Step 7 introduces ``Go2MujocoExecutor(MujocoOperationExecutor)``
which overrides ``_execute_velocity_move()`` for go2 simulation, and declares
go2w as explicitly unsupported at the executor level.

Subclassing pattern (Step 7)
-----------------------------
    class Go2MujocoExecutor(MujocoOperationExecutor):
        def _execute_velocity_move(self, binding_output, context):
            ...   # go2: call MuJoCo controller
            # go2w: return ExecutorResult.unsupported(OPERATION_VELOCITY_MOVE)
"""

from __future__ import annotations

from typing import Any, Dict

from system.binding.executors.base import ExecutorResult, OperationExecutor
from system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE


class MujocoOperationExecutor(OperationExecutor):
    """Base executor for MuJoCo simulation backend operations.

    Dispatches by operation name to dedicated handler methods.  Unknown
    operations return ``ExecutorResult.unsupported(...)`` immediately;
    known operations whose handler raises will return
    ``ExecutorResult.failed(...)`` — the exception is never propagated.

    Supported operations (Step 4 default)
    --------------------------------------
    - ``velocity_move`` → ``_execute_velocity_move()`` — returns unsupported
      until Step 7 provides a concrete subclass implementation.
    """

    def execute(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        op = binding_output.operation
        try:
            if op == OPERATION_VELOCITY_MOVE:
                return self._execute_velocity_move(binding_output, context)
            return ExecutorResult.unsupported(op)
        except Exception as exc:  # pragma: no cover — safety net only
            return ExecutorResult.failed(op, str(exc))

    # ------------------------------------------------------------------
    # Operation handlers — override in subclasses (Step 7)
    # ------------------------------------------------------------------

    def _execute_velocity_move(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        """MuJoCo velocity_move handler.

        Returns ``unsupported`` at Step 4 baseline; Step 7 overrides this in
        ``Go2MujocoExecutor`` for go2 (implements simulation), and leaves
        go2w returning ``unsupported`` explicitly at the executor level.
        """
        return ExecutorResult.unsupported(OPERATION_VELOCITY_MOVE)
