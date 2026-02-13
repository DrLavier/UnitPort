#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST to IR conversion.
Converts a parsed DSL AST into a WorkflowIR.

Maps recognized patterns:
- RobotContext.run_action('name') -> action_execution node
- RobotContext.stop() -> stop node
- RobotContext.get_sensor_data() -> sensor_input node
- time.sleep(n) -> timer node
- if/elif/else -> if node + branches
- while cond: -> while_loop node
- for i in range(start, end, step): -> while_loop node (loop_type='for')
- comparison expressions -> comparison node
- assignments -> variable node
- unrecognized code -> opaque node
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, IRNodeUI, NodeKind, EdgeType,
)
from compiler.parser.ast_nodes import (
    ASTNode, Module, Assignment, ExpressionStatement,
    NumberLiteral, StringLiteral, BoolLiteral, Identifier, AttributeAccess,
    BinaryOp, UnaryOp, CompareOp, BooleanOp, NotOp,
    FunctionCall, IfStatement, ElifClause, WhileStatement, ForRangeStatement,
    PassStatement, ReturnStatement, BreakStatement, ContinueStatement,
    ImportStatement, CommentNode, OpaqueBlock, FunctionDef,
)
from compiler.semantic.diagnostics import Diagnostic, make_warning, make_info


# Operator symbols to display names for comparison nodes
_OP_TO_UI = {
    "==": "Equal", "!=": "Not Equal", ">": "Greater Than",
    "<": "Less Than", ">=": "Greater Equal", "<=": "Less Equal",
}


class ASTToIR:
    """Convert a DSL AST into a WorkflowIR."""

    def lower(self, ast: Module, robot_type: str = "go2"
              ) -> Tuple[WorkflowIR, List[Diagnostic]]:
        """
        Lower an AST to WorkflowIR.

        Args:
            ast: Parsed Module AST
            robot_type: Target robot type

        Returns:
            (WorkflowIR, diagnostics)
        """
        self._diags: List[Diagnostic] = []
        self._ir = WorkflowIR(robot_type=robot_type,
                               brand=self._brand_for(robot_type))
        self._node_counter = 0

        # Find the workflow body:
        # 1. If there's a function def named 'execute_workflow', use its body
        # 2. Otherwise, use top-level statements (skip imports, comments, etc.)
        body = self._find_workflow_body(ast)

        # Convert statements sequentially, linking them with flow edges
        prev_node_id = None
        prev_port = "flow_out"

        for stmt in body:
            node_ids = self._convert_statement(stmt)
            if node_ids and prev_node_id:
                first_id = node_ids[0]
                self._ir.add_edge(IREdge(
                    from_node=prev_node_id,
                    from_port=prev_port,
                    to_node=first_id,
                    to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                ))
            if node_ids:
                prev_node_id = node_ids[-1]
                prev_port = "flow_out"

        self._diags.append(make_info(
            "I4002",
            f"AST lowered: {len(self._ir.nodes)} nodes, {len(self._ir.edges)} edges",
        ))
        return self._ir, self._diags

    def _find_workflow_body(self, ast: Module) -> List[ASTNode]:
        """Extract the workflow body from the AST."""
        # Look for execute_workflow function
        for stmt in ast.body:
            if isinstance(stmt, FunctionDef) and stmt.name == "execute_workflow":
                return stmt.body

        # Filter out imports, comments, and the __main__ block
        body = []
        skip_main = False
        for stmt in ast.body:
            if isinstance(stmt, (ImportStatement, CommentNode)):
                continue
            if isinstance(stmt, IfStatement):
                # Check if this is __name__ == '__main__' block
                cond = stmt.condition
                if isinstance(cond, CompareOp):
                    if (isinstance(cond.left, Identifier) and
                            cond.left.name == "__name__"):
                        continue
                body.append(stmt)
            elif isinstance(stmt, FunctionDef):
                # Skip other function defs
                continue
            elif isinstance(stmt, PassStatement):
                continue
            else:
                body.append(stmt)
        return body

    def _next_id(self) -> str:
        """Generate a sequential node ID."""
        nid = str(self._node_counter)
        self._node_counter += 1
        return nid

    def _convert_statement(self, stmt: ASTNode) -> List[str]:
        """
        Convert a statement to IR node(s).
        Returns list of node IDs that were created (first = entry, last = exit).
        """
        if isinstance(stmt, ExpressionStatement):
            return self._convert_expr_stmt(stmt)
        elif isinstance(stmt, IfStatement):
            return self._convert_if(stmt)
        elif isinstance(stmt, WhileStatement):
            return self._convert_while(stmt)
        elif isinstance(stmt, ForRangeStatement):
            return self._convert_for(stmt)
        elif isinstance(stmt, Assignment):
            return self._convert_assignment(stmt)
        elif isinstance(stmt, OpaqueBlock):
            return self._convert_opaque(stmt)
        elif isinstance(stmt, (PassStatement, CommentNode, ImportStatement,
                                ReturnStatement, BreakStatement, ContinueStatement)):
            return []
        else:
            # Unknown statement - make opaque
            return self._convert_opaque_from_stmt(stmt)

    def _convert_expr_stmt(self, stmt: ExpressionStatement) -> List[str]:
        """Convert an expression statement (typically a function call)."""
        expr = stmt.expression
        if isinstance(expr, FunctionCall):
            return self._convert_function_call(expr)
        return []

    def _convert_function_call(self, call: FunctionCall) -> List[str]:
        """Convert a function call to IR node."""
        func_name = self._get_func_name(call.func)

        if func_name == "RobotContext.run_action":
            action = self._extract_string_arg(call.args, 0, "stand")
            nid = self._next_id()
            node = IRNode(
                id=nid,
                schema_id="builtin.action_execution",
                kind=NodeKind.ACTION,
                params={"action": IRParam("action", action, "string")},
            )
            self._ir.add_node(node)
            return [nid]

        elif func_name == "RobotContext.stop":
            nid = self._next_id()
            node = IRNode(
                id=nid,
                schema_id="builtin.stop",
                kind=NodeKind.STOP,
            )
            self._ir.add_node(node)
            return [nid]

        elif func_name == "RobotContext.get_sensor_data":
            nid = self._next_id()
            node = IRNode(
                id=nid,
                schema_id="builtin.sensor_input",
                kind=NodeKind.SENSOR,
                params={"sensor_type": IRParam("sensor_type", "imu", "string")},
            )
            self._ir.add_node(node)
            return [nid]

        elif func_name == "time.sleep":
            duration = self._extract_number_arg(call.args, 0, 1.0)
            nid = self._next_id()
            node = IRNode(
                id=nid,
                schema_id="builtin.timer",
                kind=NodeKind.TIMER,
                params={
                    "duration": IRParam("duration", duration, "float"),
                    "unit": IRParam("unit", "seconds", "string"),
                },
            )
            self._ir.add_node(node)
            return [nid]

        else:
            # Unknown function call - opaque
            code = self._reconstruct_call(call)
            nid = self._next_id()
            node = IRNode(
                id=nid,
                schema_id="builtin.opaque",
                kind=NodeKind.OPAQUE,
                opaque_code=code,
            )
            self._ir.add_node(node)
            self._diags.append(make_warning(
                "W2002",
                f"Unknown function call '{func_name}' wrapped as opaque block",
            ))
            return [nid]

    def _convert_if(self, stmt: IfStatement) -> List[str]:
        """Convert an if/elif/else statement to IR nodes."""
        nid = self._next_id()
        condition_text = self._expr_to_string(stmt.condition)

        elif_conditions = []
        for ec in stmt.elifs:
            elif_conditions.append(self._expr_to_string(ec.condition))

        params = {
            "condition_expr": IRParam("condition_expr", condition_text, "string"),
        }
        if elif_conditions:
            params["elif_conditions"] = IRParam("elif_conditions", elif_conditions, "string")

        node = IRNode(
            id=nid,
            schema_id="builtin.if",
            kind=NodeKind.LOGIC,
            params=params,
        )
        self._ir.add_node(node)

        # True branch
        self._convert_branch(stmt.body, nid, "out_if")

        # Elif branches
        for i, ec in enumerate(stmt.elifs):
            self._convert_branch(ec.body, nid, f"out_elif_{i}")

        # Else branch
        if stmt.else_body:
            self._convert_branch(stmt.else_body, nid, "out_else")

        return [nid]

    def _convert_while(self, stmt: WhileStatement) -> List[str]:
        """Convert a while loop to IR node."""
        nid = self._next_id()
        condition_text = self._expr_to_string(stmt.condition)

        node = IRNode(
            id=nid,
            schema_id="builtin.while_loop",
            kind=NodeKind.LOGIC,
            params={
                "loop_type": IRParam("loop_type", "while", "string"),
                "condition_expr": IRParam("condition_expr", condition_text, "string"),
                "for_start": IRParam("for_start", 0, "int"),
                "for_end": IRParam("for_end", 10, "int"),
                "for_step": IRParam("for_step", 1, "int"),
            },
        )
        self._ir.add_node(node)

        # Loop body
        self._convert_branch(stmt.body, nid, "loop_body")

        return [nid]

    def _convert_for(self, stmt: ForRangeStatement) -> List[str]:
        """Convert a for-range loop to IR node."""
        nid = self._next_id()
        start = self._extract_literal_value(stmt.start, 0)
        end = self._extract_literal_value(stmt.end, 10)
        step = self._extract_literal_value(stmt.step, 1)

        node = IRNode(
            id=nid,
            schema_id="builtin.while_loop",
            kind=NodeKind.LOGIC,
            params={
                "loop_type": IRParam("loop_type", "for", "string"),
                "condition_expr": IRParam("condition_expr", "", "string"),
                "for_start": IRParam("for_start", int(start), "int"),
                "for_end": IRParam("for_end", int(end), "int"),
                "for_step": IRParam("for_step", int(step), "int"),
            },
        )
        self._ir.add_node(node)

        # Loop body
        self._convert_branch(stmt.body, nid, "loop_body")

        return [nid]

    def _convert_assignment(self, stmt: Assignment) -> List[str]:
        """Convert an assignment to a variable node or function call node."""
        # Check if RHS is a recognized function call (e.g. sensor_data = RobotContext.get_sensor_data())
        if isinstance(stmt.value, FunctionCall):
            func_name = self._get_func_name(stmt.value.func)
            if func_name in ("RobotContext.get_sensor_data",
                             "RobotContext.run_action",
                             "RobotContext.stop",
                             "time.sleep"):
                return self._convert_function_call(stmt.value)

        nid = self._next_id()
        value = self._extract_literal_value(stmt.value, 0)

        node = IRNode(
            id=nid,
            schema_id="builtin.variable",
            kind=NodeKind.VARIABLE,
            params={
                "name": IRParam("name", stmt.target, "string"),
                "initial_value": IRParam("initial_value", value, "any"),
            },
        )
        self._ir.add_node(node)
        return [nid]

    def _convert_opaque(self, block: OpaqueBlock) -> List[str]:
        """Convert an opaque block to an opaque IR node."""
        nid = self._next_id()
        node = IRNode(
            id=nid,
            schema_id="builtin.opaque",
            kind=NodeKind.OPAQUE,
            opaque_code=block.code,
        )
        self._ir.add_node(node)
        return [nid]

    def _convert_opaque_from_stmt(self, stmt: ASTNode) -> List[str]:
        """Convert any unknown statement to an opaque node."""
        nid = self._next_id()
        node = IRNode(
            id=nid,
            schema_id="builtin.opaque",
            kind=NodeKind.OPAQUE,
            opaque_code=f"# unsupported: {type(stmt).__name__}",
        )
        self._ir.add_node(node)
        return [nid]

    def _convert_branch(self, stmts: List[ASTNode], parent_id: str, port: str):
        """Convert a list of statements and connect the first to parent_id:port."""
        prev_id = parent_id
        prev_port = port
        for stmt in stmts:
            node_ids = self._convert_statement(stmt)
            if node_ids:
                self._ir.add_edge(IREdge(
                    from_node=prev_id,
                    from_port=prev_port,
                    to_node=node_ids[0],
                    to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                ))
                prev_id = node_ids[-1]
                prev_port = "flow_out"

    # ---------- Helper methods ----------

    def _get_func_name(self, node: ASTNode) -> str:
        """Get dotted function name from an AST node."""
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, AttributeAccess):
            obj = self._get_func_name(node.object)
            return f"{obj}.{node.attribute}"
        return "unknown"

    def _extract_string_arg(self, args: List[ASTNode], idx: int,
                            default: str) -> str:
        """Extract string argument from call args."""
        if idx < len(args):
            arg = args[idx]
            if isinstance(arg, StringLiteral):
                return arg.value
            if isinstance(arg, Identifier):
                return arg.name
        return default

    def _extract_number_arg(self, args: List[ASTNode], idx: int,
                            default: float) -> float:
        """Extract numeric argument from call args."""
        if idx < len(args):
            arg = args[idx]
            if isinstance(arg, NumberLiteral):
                return arg.value
            if isinstance(arg, BinaryOp):
                # e.g. duration / 1000
                return default
        return default

    def _extract_literal_value(self, node: ASTNode, default=0):
        """Extract a literal value from an AST node."""
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, StringLiteral):
            return node.value
        if isinstance(node, BoolLiteral):
            return node.value
        if isinstance(node, Identifier):
            return node.name
        return default

    def _expr_to_string(self, node: ASTNode) -> str:
        """Convert an expression AST node back to a string representation."""
        if isinstance(node, NumberLiteral):
            if isinstance(node.value, int):
                return str(node.value)
            return str(node.value)
        if isinstance(node, StringLiteral):
            return repr(node.value)
        if isinstance(node, BoolLiteral):
            return str(node.value)
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, AttributeAccess):
            obj = self._expr_to_string(node.object)
            return f"{obj}.{node.attribute}"
        if isinstance(node, BinaryOp):
            left = self._expr_to_string(node.left)
            right = self._expr_to_string(node.right)
            return f"{left} {node.op} {right}"
        if isinstance(node, UnaryOp):
            operand = self._expr_to_string(node.operand)
            return f"{node.op}{operand}"
        if isinstance(node, CompareOp):
            left = self._expr_to_string(node.left)
            right = self._expr_to_string(node.right)
            return f"{left} {node.op} {right}"
        if isinstance(node, BooleanOp):
            left = self._expr_to_string(node.left)
            right = self._expr_to_string(node.right)
            return f"{left} {node.op} {right}"
        if isinstance(node, NotOp):
            operand = self._expr_to_string(node.operand)
            return f"not {operand}"
        if isinstance(node, FunctionCall):
            return self._reconstruct_call(node)
        return "???"

    def _reconstruct_call(self, call: FunctionCall) -> str:
        """Reconstruct a function call as a string."""
        func = self._get_func_name(call.func)
        args = ", ".join(self._expr_to_string(a) for a in call.args)
        return f"{func}({args})"

    @staticmethod
    def _brand_for(robot_type: str) -> str:
        brands = {
            "go2": "unitree", "a1": "unitree", "b1": "unitree",
            "b2": "unitree", "h1": "unitree",
        }
        return brands.get(robot_type, "unknown")
