#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executor interfaces for backend operation dispatch (Step 4).

ExecutorResult
--------------
Plain dataclass returned by every executor.  Fields mirror the three
diagnostic states defined in Acceptance Benchmark 10:

    executor.unsupported.<operation>  -- operation not handled by this executor
    executor.failed.<operation>       -- execution was attempted but failed
    (empty diag_code)                 -- success

OperationExecutor
-----------------
Abstract base class.  All concrete executors implement exactly one method:

    execute(binding_output, context) -> ExecutorResult

Contract:
- Must not resolve semantic intents (BindingOutput already carries resolved data).
- Must not raise unhandled exceptions; always return ExecutorResult.
- Return ExecutorResult.unsupported(...) for unrecognised operations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

from system.binding.output import BindingOutput


# ---------------------------------------------------------------------------
# ExecutorResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutorResult:
    """Result returned by every OperationExecutor.execute() call.

    Attributes
    ----------
    success:
        True when the operation completed without error.
    operation:
        The operation name from the originating BindingOutput.  Included so
        callers do not need to track which operation was attempted.
    reason:
        Human-readable explanation.  Empty string on success.
    diag_code:
        Structured diagnostic code for programmatic handling.
        Conventions:
          - ``""``                              — success (no code needed)
          - ``"executor.unsupported.<op>"``     — executor cannot handle op
          - ``"executor.failed.<op>"``          — execution attempted, failed
    """

    success: bool
    operation: str
    reason: str = field(default="")
    diag_code: str = field(default="")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def ok(cls, operation: str) -> "ExecutorResult":
        """Successful execution result."""
        return cls(success=True, operation=operation, reason="", diag_code="")

    @classmethod
    def unsupported(cls, operation: str) -> "ExecutorResult":
        """The executor does not support this operation at all."""
        return cls(
            success=False,
            operation=operation,
            reason=f"Unsupported operation: {operation!r}",
            diag_code=f"executor.unsupported.{operation}",
        )

    @classmethod
    def failed(cls, operation: str, reason: str) -> "ExecutorResult":
        """Execution was attempted but failed at runtime."""
        return cls(
            success=False,
            operation=operation,
            reason=reason,
            diag_code=f"executor.failed.{operation}",
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "reason": self.reason,
            "diag_code": self.diag_code,
        }


# ---------------------------------------------------------------------------
# OperationExecutor — abstract base
# ---------------------------------------------------------------------------

class OperationExecutor(abc.ABC):
    """Abstract base for backend operation executors.

    Subclass contract
    -----------------
    1. Override ``execute()`` to perform the operation described by
       ``binding_output``.
    2. Never raise unhandled exceptions from ``execute()``.  Catch all
       errors internally and return ``ExecutorResult.failed(...)``.
    3. Return ``ExecutorResult.unsupported(binding_output.operation)`` for
       any operation this executor does not handle.
    4. Do not resolve semantic intents; ``binding_output`` already contains
       the resolved backend operation and parameters.
    """

    @abc.abstractmethod
    def execute(
        self,
        binding_output: BindingOutput,
        context: Dict[str, Any],
    ) -> ExecutorResult:
        """Execute a backend operation.

        Parameters
        ----------
        binding_output:
            The fully resolved backend operation produced by a Binding.
            Contains operation name, backend-ready params, brand/model, and
            backend target.
        context:
            Runtime context dict.  Typically includes ``"robot_model"``
            (the live adapter instance) and scenario settings.

        Returns
        -------
        ExecutorResult
            Always returns an ExecutorResult; never raises.
        """
