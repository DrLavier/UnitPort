#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDKOperationExecutor — base executor for backend="sdk" operations (Step 4).

At Step 4 this class returns ``ExecutorResult.unsupported(...)`` for every
operation.  Step 7 introduces ``UnitreeSDKExecutor(SDKOperationExecutor)``
which overrides ``_execute_velocity_move()`` with the real SDK call.

Subclassing pattern (Step 7)
-----------------------------
    class UnitreeSDKExecutor(SDKOperationExecutor):
        def _execute_velocity_move(self, binding_output, context):
            ...   # call UnitreeModel.velocity_move(...)
"""

from __future__ import annotations

from typing import Any, Dict

from system.binding.executors.base import ExecutorResult, OperationExecutor
from system.binding.output import BindingOutput, OPERATION_VELOCITY_MOVE


class SDKOperationExecutor(OperationExecutor):
    """Base executor for SDK (real-hardware) backend operations.

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
        """SDK velocity_move handler.

        Returns ``unsupported`` at Step 4 baseline; Step 7 overrides this in
        ``UnitreeSDKExecutor`` with a real ``UnitreeModel.velocity_move()`` call.
        """
        return ExecutorResult.unsupported(OPERATION_VELOCITY_MOVE)
