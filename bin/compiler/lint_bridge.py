"""Validation bridge from source code to semantic diagnostics."""

from typing import List

from compiler.lowering.ast_to_ir import ASTToIR
from compiler.semantic.diagnostics import Diagnostic
from compiler.semantic.validator import SemanticValidator

from .dsl_support import parse_source


def lint_source(source: str, robot_type: str = "go2") -> List[Diagnostic]:
    """Run parse/lower/semantic validation for compiler source."""
    ast, parse_diags = parse_source(source)
    ir, lower_diags = ASTToIR().lower(ast, robot_type=robot_type)
    semantic_diags = SemanticValidator().validate(ir)
    return [*parse_diags, *lower_diags, *semantic_diags]

