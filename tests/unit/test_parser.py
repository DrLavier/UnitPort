#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the DSL parser."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.parser.parser import Parser
from compiler.parser.ast_nodes import (
    Module, Assignment, ExpressionStatement,
    NumberLiteral, StringLiteral, BoolLiteral, Identifier, AttributeAccess,
    BinaryOp, UnaryOp, CompareOp, BooleanOp, NotOp,
    FunctionCall, IfStatement, ElifClause, WhileStatement, ForRangeStatement,
    PassStatement, ReturnStatement, BreakStatement, ContinueStatement,
    ImportStatement, CommentNode, OpaqueBlock, FunctionDef,
)


class TestParserExpressions(unittest.TestCase):
    def _parse_expr(self, source):
        """Parse a single expression statement and return the expression."""
        parser = Parser(source)
        module, diags = parser.parse()
        stmts = [s for s in module.body if not isinstance(s, CommentNode)]
        self.assertGreater(len(stmts), 0,
                           f"No statements parsed from: {source!r}")
        stmt = stmts[0]
        if isinstance(stmt, ExpressionStatement):
            return stmt.expression
        return stmt

    def test_integer_literal(self):
        expr = self._parse_expr("42")
        self.assertIsInstance(expr, NumberLiteral)
        self.assertEqual(expr.value, 42)

    def test_float_literal(self):
        expr = self._parse_expr("3.14")
        self.assertIsInstance(expr, NumberLiteral)
        self.assertAlmostEqual(expr.value, 3.14)

    def test_string_literal(self):
        expr = self._parse_expr("'hello'")
        self.assertIsInstance(expr, StringLiteral)
        self.assertEqual(expr.value, "hello")

    def test_bool_literal(self):
        expr = self._parse_expr("True")
        self.assertIsInstance(expr, BoolLiteral)
        self.assertEqual(expr.value, True)

    def test_identifier(self):
        expr = self._parse_expr("my_var")
        self.assertIsInstance(expr, Identifier)
        self.assertEqual(expr.name, "my_var")

    def test_binary_add(self):
        expr = self._parse_expr("1 + 2")
        self.assertIsInstance(expr, BinaryOp)
        self.assertEqual(expr.op, "+")

    def test_binary_precedence(self):
        expr = self._parse_expr("1 + 2 * 3")
        self.assertIsInstance(expr, BinaryOp)
        self.assertEqual(expr.op, "+")
        self.assertIsInstance(expr.right, BinaryOp)
        self.assertEqual(expr.right.op, "*")

    def test_comparison(self):
        expr = self._parse_expr("x > 5")
        self.assertIsInstance(expr, CompareOp)
        self.assertEqual(expr.op, ">")

    def test_boolean_and(self):
        expr = self._parse_expr("x > 5 and y < 10")
        self.assertIsInstance(expr, BooleanOp)
        self.assertEqual(expr.op, "and")

    def test_not_expr(self):
        expr = self._parse_expr("not True")
        self.assertIsInstance(expr, NotOp)

    def test_unary_minus(self):
        expr = self._parse_expr("-5")
        self.assertIsInstance(expr, UnaryOp)
        self.assertEqual(expr.op, "-")

    def test_function_call(self):
        expr = self._parse_expr("abs(5)")
        self.assertIsInstance(expr, FunctionCall)
        self.assertIsInstance(expr.func, Identifier)
        self.assertEqual(expr.func.name, "abs")
        self.assertEqual(len(expr.args), 1)

    def test_method_call(self):
        expr = self._parse_expr("RobotContext.run_action('stand')")
        self.assertIsInstance(expr, FunctionCall)
        self.assertIsInstance(expr.func, AttributeAccess)
        self.assertEqual(expr.func.attribute, "run_action")
        self.assertEqual(len(expr.args), 1)
        self.assertIsInstance(expr.args[0], StringLiteral)
        self.assertEqual(expr.args[0].value, "stand")

    def test_parenthesized(self):
        expr = self._parse_expr("(1 + 2) * 3")
        self.assertIsInstance(expr, BinaryOp)
        self.assertEqual(expr.op, "*")
        self.assertIsInstance(expr.left, BinaryOp)


class TestParserStatements(unittest.TestCase):
    def _parse(self, source):
        parser = Parser(source)
        module, diags = parser.parse()
        return module, diags

    def test_assignment(self):
        module, _ = self._parse("x = 42")
        stmts = [s for s in module.body if isinstance(s, Assignment)]
        self.assertEqual(len(stmts), 1)
        self.assertEqual(stmts[0].target, "x")
        self.assertIsInstance(stmts[0].value, NumberLiteral)
        self.assertEqual(stmts[0].value.value, 42)

    def test_augmented_assignment(self):
        module, _ = self._parse("x += 1")
        stmts = [s for s in module.body if isinstance(s, Assignment)]
        self.assertEqual(len(stmts), 1)
        self.assertEqual(stmts[0].target, "x")
        self.assertIsInstance(stmts[0].value, BinaryOp)
        self.assertEqual(stmts[0].value.op, "+")

    def test_pass(self):
        module, _ = self._parse("pass")
        stmts = [s for s in module.body if isinstance(s, PassStatement)]
        self.assertEqual(len(stmts), 1)

    def test_comment(self):
        module, _ = self._parse("# hello world")
        comments = [s for s in module.body if isinstance(s, CommentNode)]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].text, "hello world")

    def test_import(self):
        module, _ = self._parse("import time")
        imports = [s for s in module.body if isinstance(s, ImportStatement)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "time")

    def test_from_import(self):
        module, _ = self._parse("from bin.core.robot_context import RobotContext")
        imports = [s for s in module.body if isinstance(s, ImportStatement)]
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_from)
        self.assertIn("RobotContext", imports[0].names)


class TestParserControlFlow(unittest.TestCase):
    def _parse(self, source):
        parser = Parser(source)
        module, diags = parser.parse()
        return module, diags

    def test_if_simple(self):
        source = "if True:\n    pass"
        module, _ = self._parse(source)
        ifs = [s for s in module.body if isinstance(s, IfStatement)]
        self.assertEqual(len(ifs), 1)
        self.assertIsInstance(ifs[0].condition, BoolLiteral)
        self.assertEqual(len(ifs[0].body), 1)
        self.assertIsInstance(ifs[0].body[0], PassStatement)

    def test_if_else(self):
        source = "if True:\n    pass\nelse:\n    pass"
        module, _ = self._parse(source)
        ifs = [s for s in module.body if isinstance(s, IfStatement)]
        self.assertEqual(len(ifs), 1)
        self.assertEqual(len(ifs[0].else_body), 1)

    def test_if_elif_else(self):
        source = "if x > 5:\n    pass\nelif x > 3:\n    pass\nelse:\n    pass"
        module, _ = self._parse(source)
        ifs = [s for s in module.body if isinstance(s, IfStatement)]
        self.assertEqual(len(ifs), 1)
        self.assertEqual(len(ifs[0].elifs), 1)
        self.assertEqual(len(ifs[0].else_body), 1)

    def test_while(self):
        source = "while True:\n    pass"
        module, _ = self._parse(source)
        whiles = [s for s in module.body if isinstance(s, WhileStatement)]
        self.assertEqual(len(whiles), 1)
        self.assertIsInstance(whiles[0].condition, BoolLiteral)

    def test_for_range_1arg(self):
        source = "for i in range(5):\n    pass"
        module, _ = self._parse(source)
        fors = [s for s in module.body if isinstance(s, ForRangeStatement)]
        self.assertEqual(len(fors), 1)
        self.assertEqual(fors[0].variable, "i")
        self.assertEqual(fors[0].end.value, 5)

    def test_for_range_3args(self):
        source = "for i in range(0, 10, 2):\n    pass"
        module, _ = self._parse(source)
        fors = [s for s in module.body if isinstance(s, ForRangeStatement)]
        self.assertEqual(len(fors), 1)
        self.assertEqual(fors[0].start.value, 0)
        self.assertEqual(fors[0].end.value, 10)
        self.assertEqual(fors[0].step.value, 2)

    def test_nested_if_in_while(self):
        source = "while True:\n    if x > 5:\n        pass"
        module, _ = self._parse(source)
        whiles = [s for s in module.body if isinstance(s, WhileStatement)]
        self.assertEqual(len(whiles), 1)
        ifs = [s for s in whiles[0].body if isinstance(s, IfStatement)]
        self.assertEqual(len(ifs), 1)

    def test_function_def(self):
        source = "def execute_workflow(robot=None):\n    pass"
        module, _ = self._parse(source)
        funcs = [s for s in module.body if isinstance(s, FunctionDef)]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "execute_workflow")


class TestParserWorkflow(unittest.TestCase):
    """Test parsing of realistic workflow code."""

    def _parse(self, source):
        parser = Parser(source)
        module, diags = parser.parse()
        return module, diags

    def test_single_action(self):
        source = "RobotContext.run_action('stand')"
        module, _ = self._parse(source)
        stmts = [s for s in module.body
                 if isinstance(s, ExpressionStatement)]
        self.assertEqual(len(stmts), 1)
        call = stmts[0].expression
        self.assertIsInstance(call, FunctionCall)

    def test_timer(self):
        source = "time.sleep(2.0)"
        module, _ = self._parse(source)
        stmts = [s for s in module.body
                 if isinstance(s, ExpressionStatement)]
        self.assertEqual(len(stmts), 1)
        call = stmts[0].expression
        self.assertIsInstance(call, FunctionCall)
        self.assertEqual(call.args[0].value, 2.0)

    def test_full_workflow(self):
        source = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Auto-generated workflow code\"\"\"

import time
from bin.core.robot_context import RobotContext


def execute_workflow(robot=None):
    '''Execute the visual workflow'''
    RobotContext.run_action('stand')
    time.sleep(2.0)
    RobotContext.run_action('walk')


if __name__ == '__main__':
    robot = None
    execute_workflow(robot)
"""
        module, diags = self._parse(source)
        # Should parse without fatal errors
        errors = [d for d in diags if d.level.value == "error"]
        self.assertEqual(len(errors), 0,
                         f"Unexpected errors: {[d.message for d in errors]}")

    def test_if_else_workflow(self):
        source = """if True:
    RobotContext.run_action('stand')
else:
    RobotContext.run_action('sit')
"""
        module, _ = self._parse(source)
        ifs = [s for s in module.body if isinstance(s, IfStatement)]
        self.assertEqual(len(ifs), 1)
        # True branch should have a function call
        self.assertEqual(len(ifs[0].body), 1)
        self.assertEqual(len(ifs[0].else_body), 1)

    def test_for_loop_workflow(self):
        source = """for i in range(0, 5, 1):
    RobotContext.run_action('stand')
"""
        module, _ = self._parse(source)
        fors = [s for s in module.body if isinstance(s, ForRangeStatement)]
        self.assertEqual(len(fors), 1)
        self.assertEqual(len(fors[0].body), 1)

    def test_unexpected_top_level_indent_does_not_hang(self):
        source = "    x = 1\nx = 2\n"
        module, diags = self._parse(source)
        assigns = [s for s in module.body if isinstance(s, Assignment)]
        self.assertGreaterEqual(len(assigns), 1)
        self.assertTrue(any(d.code == "E1002" for d in diags))


if __name__ == "__main__":
    unittest.main()
