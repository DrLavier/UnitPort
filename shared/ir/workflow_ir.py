"""Shared IR re-export.

The canonical IR definitions live in ``compiler.ir.workflow_ir``.
This module re-exports them so that the new layered packages can
import from ``shared.ir`` without duplicating dataclasses.
"""

from compiler.ir.workflow_ir import (  # noqa: F401
    EdgeType,
    IREdge,
    IRNode,
    IRNodeUI,
    IRParam,
    IRVariable,
    NodeKind,
    SourceSpan,
    WorkflowIR,
)
