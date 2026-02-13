#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST node definitions for the UnitPort DSL.

The DSL is a restricted Python subset supporting:
- Assignments
- Whitelisted function calls (RobotContext.*, time.sleep, range, abs, min, max, sum)
- if / elif / else
- while loops
- for i in range(...) loops
- Literals (int, float, bool, string)
- Binary/unary operations
- Comments
- Opaque blocks (unparseable code preserved verbatim)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASTNode:
    """Base AST node."""
    line: int = 0
    col: int = 0


# ---------- Literals ----------

@dataclass
class NumberLiteral(ASTNode):
    """Integer or float literal."""
    value: float = 0

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int) or (isinstance(self.value, float) and self.value == int(self.value))


@dataclass
class StringLiteral(ASTNode):
    """String literal."""
    value: str = ""


@dataclass
class BoolLiteral(ASTNode):
    """Boolean literal (True / False)."""
    value: bool = False


@dataclass
class Identifier(ASTNode):
    """Variable or name reference."""
    name: str = ""


@dataclass
class AttributeAccess(ASTNode):
    """Dotted name access, e.g. RobotContext.run_action."""
    object: ASTNode = field(default_factory=ASTNode)
    attribute: str = ""


# ---------- Expressions ----------

@dataclass
class BinaryOp(ASTNode):
    """Binary operation: left op right."""
    left: ASTNode = field(default_factory=ASTNode)
    op: str = ""
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class UnaryOp(ASTNode):
    """Unary operation: op operand."""
    op: str = ""
    operand: ASTNode = field(default_factory=ASTNode)


@dataclass
class FunctionCall(ASTNode):
    """Function call expression."""
    func: ASTNode = field(default_factory=ASTNode)
    args: List[ASTNode] = field(default_factory=list)


@dataclass
class CompareOp(ASTNode):
    """Comparison: left op right (==, !=, <, >, <=, >=)."""
    left: ASTNode = field(default_factory=ASTNode)
    op: str = ""
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class BooleanOp(ASTNode):
    """Boolean operation: left op right (and, or)."""
    left: ASTNode = field(default_factory=ASTNode)
    op: str = ""  # "and" or "or"
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class NotOp(ASTNode):
    """Boolean not: not operand."""
    operand: ASTNode = field(default_factory=ASTNode)


# ---------- Statements ----------

@dataclass
class Assignment(ASTNode):
    """Variable assignment: name = value."""
    target: str = ""
    value: ASTNode = field(default_factory=ASTNode)


@dataclass
class ExpressionStatement(ASTNode):
    """Standalone expression (typically a function call)."""
    expression: ASTNode = field(default_factory=ASTNode)


@dataclass
class IfStatement(ASTNode):
    """if / elif / else statement."""
    condition: ASTNode = field(default_factory=ASTNode)
    body: List[ASTNode] = field(default_factory=list)
    elifs: List[ElifClause] = field(default_factory=list)
    else_body: List[ASTNode] = field(default_factory=list)


@dataclass
class ElifClause(ASTNode):
    """elif clause."""
    condition: ASTNode = field(default_factory=ASTNode)
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class WhileStatement(ASTNode):
    """while loop."""
    condition: ASTNode = field(default_factory=ASTNode)
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class ForRangeStatement(ASTNode):
    """for i in range(...) loop."""
    variable: str = "i"
    start: ASTNode = field(default_factory=lambda: NumberLiteral(value=0))
    end: ASTNode = field(default_factory=lambda: NumberLiteral(value=10))
    step: ASTNode = field(default_factory=lambda: NumberLiteral(value=1))
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class PassStatement(ASTNode):
    """pass statement."""
    pass


@dataclass
class ReturnStatement(ASTNode):
    """return statement."""
    value: Optional[ASTNode] = None


@dataclass
class BreakStatement(ASTNode):
    """break statement."""
    pass


@dataclass
class ContinueStatement(ASTNode):
    """continue statement."""
    pass


@dataclass
class ImportStatement(ASTNode):
    """import or from ... import statement (preserved but not executed)."""
    module: str = ""
    names: List[str] = field(default_factory=list)
    is_from: bool = False


@dataclass
class CommentNode(ASTNode):
    """Comment line."""
    text: str = ""


@dataclass
class OpaqueBlock(ASTNode):
    """Unparseable code block preserved verbatim."""
    code: str = ""


@dataclass
class FunctionDef(ASTNode):
    """Function definition (treated as opaque/skipped for IR)."""
    name: str = ""
    body: List[ASTNode] = field(default_factory=list)


# ---------- Top-level ----------

@dataclass
class Module(ASTNode):
    """Top-level module (entire file)."""
    body: List[ASTNode] = field(default_factory=list)
