#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for AST to IR lowering."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.parser.parser import Parser
from compiler.lowering.ast_to_ir import ASTToIR
from compiler.ir.workflow_ir import NodeKind
from compiler.schema.registry import SchemaRegistry


class TestASTToIR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def _lower(self, source, robot_type="go2"):
        parser = Parser(source)
        ast, _ = parser.parse()
        lowerer = ASTToIR()
        ir, diags = lowerer.lower(ast, robot_type)
        return ir, diags

    def test_single_action(self):
        ir, _ = self._lower("RobotContext.run_action('stand')")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.ACTION)
        self.assertEqual(ir.nodes[0].get_param_value("action"), "stand")

    def test_stop(self):
        ir, _ = self._lower("RobotContext.stop()")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.STOP)

    def test_sensor(self):
        ir, _ = self._lower("sensor_data = RobotContext.get_sensor_data()")
        sensor_nodes = [n for n in ir.nodes if n.kind == NodeKind.SENSOR]
        self.assertGreater(len(sensor_nodes), 0)

    def test_timer(self):
        ir, _ = self._lower("time.sleep(2.5)")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.TIMER)
        self.assertAlmostEqual(ir.nodes[0].get_param_value("duration"), 2.5)

    def test_chained_actions(self):
        source = "RobotContext.run_action('stand')\nRobotContext.run_action('walk')"
        ir, _ = self._lower(source)
        self.assertEqual(len(ir.nodes), 2)
        self.assertEqual(len(ir.edges), 1)
        # First node should connect to second
        self.assertEqual(ir.edges[0].from_node, "0")
        self.assertEqual(ir.edges[0].to_node, "1")

    def test_if_else(self):
        source = """if True:
    RobotContext.run_action('stand')
else:
    RobotContext.run_action('sit')
"""
        ir, _ = self._lower(source)
        logic_nodes = [n for n in ir.nodes if n.kind == NodeKind.LOGIC]
        self.assertEqual(len(logic_nodes), 1)
        action_nodes = [n for n in ir.nodes if n.kind == NodeKind.ACTION]
        self.assertEqual(len(action_nodes), 2)
        # Should have edges from if to both branches
        self.assertGreater(len(ir.edges), 0)

    def test_while_loop(self):
        source = """while True:
    RobotContext.run_action('stand')
"""
        ir, _ = self._lower(source)
        logic_nodes = [n for n in ir.nodes if n.kind == NodeKind.LOGIC]
        self.assertEqual(len(logic_nodes), 1)
        self.assertEqual(logic_nodes[0].get_param_value("loop_type"), "while")

    def test_for_range(self):
        source = """for i in range(0, 5, 1):
    RobotContext.run_action('stand')
"""
        ir, _ = self._lower(source)
        logic_nodes = [n for n in ir.nodes if n.kind == NodeKind.LOGIC]
        self.assertEqual(len(logic_nodes), 1)
        self.assertEqual(logic_nodes[0].get_param_value("loop_type"), "for")
        self.assertEqual(logic_nodes[0].get_param_value("for_start"), 0)
        self.assertEqual(logic_nodes[0].get_param_value("for_end"), 5)
        self.assertEqual(logic_nodes[0].get_param_value("for_step"), 1)

    def test_assignment_creates_variable(self):
        ir, _ = self._lower("x = 42")
        var_nodes = [n for n in ir.nodes if n.kind == NodeKind.VARIABLE]
        self.assertEqual(len(var_nodes), 1)
        self.assertEqual(var_nodes[0].get_param_value("name"), "x")

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
        ir, diags = self._lower(source)
        # Should extract body from execute_workflow
        action_nodes = [n for n in ir.nodes if n.kind == NodeKind.ACTION]
        self.assertEqual(len(action_nodes), 2)
        timer_nodes = [n for n in ir.nodes if n.kind == NodeKind.TIMER]
        self.assertEqual(len(timer_nodes), 1)
        # Should have 2 flow edges connecting them
        self.assertEqual(len(ir.edges), 2)

    def test_if_elif_else(self):
        source = """if x > 5:
    RobotContext.run_action('stand')
elif x > 3:
    RobotContext.run_action('walk')
else:
    RobotContext.run_action('sit')
"""
        ir, _ = self._lower(source)
        logic_nodes = [n for n in ir.nodes if n.kind == NodeKind.LOGIC]
        self.assertEqual(len(logic_nodes), 1)
        elif_conds = logic_nodes[0].get_param_value("elif_conditions")
        self.assertIsNotNone(elif_conds)
        self.assertEqual(len(elif_conds), 1)


if __name__ == "__main__":
    unittest.main()
