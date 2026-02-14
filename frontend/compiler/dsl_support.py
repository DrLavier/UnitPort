"""DSL support bridge for the frontend compiler layer."""

from typing import Tuple, List

from compiler.parser.parser import Parser
from compiler.parser.ast_nodes import Module
from compiler.semantic.diagnostics import Diagnostic


def parse_source(source: str) -> Tuple[Module, List[Diagnostic]]:
    """Parse DSL source into AST and diagnostics."""
    parser = Parser(source or "")
    return parser.parse()

