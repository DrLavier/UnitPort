"""Mission planning and composition utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.system.compiler.lowering.canvas_to_ir import CanvasToIR
from src.system.compiler.lowering.ast_to_ir import ASTToIR
from src.system.compiler.parser.parser import Parser
from src.system.compiler.semantic.diagnostics import Diagnostic
from src.system.ir.workflow_ir import WorkflowIR


class MissionPlanner:
    """Build mission IR from canvas payloads or source code."""

    def from_canvas(self, graph_data: Dict[str, Any], robot_type: str = "go2") -> Tuple[WorkflowIR, List[Diagnostic]]:
        return CanvasToIR().convert(graph_data, robot_type=robot_type)

    def from_source(self, source: str, robot_type: str = "go2") -> Tuple[WorkflowIR, List[Diagnostic]]:
        module, parse_diags = Parser(source or "").parse()
        ir, lower_diags = ASTToIR().lower(module, robot_type=robot_type)
        return ir, [*parse_diags, *lower_diags]


