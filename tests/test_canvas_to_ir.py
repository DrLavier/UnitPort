#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for canvas_to_ir and ir_to_code pipeline."""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.lowering.canvas_to_ir import CanvasToIR
from compiler.codegen.ir_to_code import IRToCode
from compiler.semantic.validator import SemanticValidator
from compiler.semantic.diagnostics import DiagnosticLevel
from compiler.schema.registry import SchemaRegistry
from compiler.ir.workflow_ir import NodeKind


class TestCanvasToIR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def test_single_action_node(self):
        data = {
            "nodes": [{
                "id": 0,
                "display_name": "Action Execution",
                "position": {"x": 100, "y": 100},
                "node_type": "action_execution",
                "ui_selection": "Stand",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.ACTION)
        self.assertEqual(ir.nodes[0].get_param_value("action"), "stand")

    def test_two_chained_actions(self):
        data = {
            "nodes": [
                {"id": 0, "display_name": "Action Execution",
                 "position": {"x": 100, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
                {"id": 1, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Walk"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "flow_out",
                 "to_node": 1, "to_port": "flow_in"},
            ],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(len(ir.nodes), 2)
        self.assertEqual(len(ir.edges), 1)

    def test_if_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Logic Control",
                "position": {"x": 100, "y": 100},
                "node_type": "if", "ui_selection": "If",
                "condition_expr": "x > 5",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].kind, NodeKind.LOGIC)
        self.assertEqual(ir.nodes[0].schema_id, "builtin.if")
        self.assertEqual(ir.nodes[0].get_param_value("condition_expr"), "x > 5")

    def test_while_loop_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Logic Control",
                "position": {"x": 100, "y": 100},
                "node_type": "while_loop", "ui_selection": "While Loop",
                "loop_type": "While", "condition_expr": "running",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].schema_id, "builtin.while_loop")
        self.assertEqual(ir.nodes[0].get_param_value("loop_type"), "while")

    def test_for_loop_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Logic Control",
                "position": {"x": 100, "y": 100},
                "node_type": "while_loop", "ui_selection": "While Loop",
                "loop_type": "For", "for_start": "0", "for_end": "5", "for_step": "1",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].get_param_value("loop_type"), "for")
        self.assertEqual(ir.nodes[0].get_param_value("for_start"), 0)
        self.assertEqual(ir.nodes[0].get_param_value("for_end"), 5)

    def test_comparison_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Condition",
                "position": {"x": 100, "y": 100},
                "node_type": "comparison", "ui_selection": "Greater Than",
                "left_value": "x", "right_value": "10",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].kind, NodeKind.COMPARISON)
        self.assertEqual(ir.nodes[0].get_param_value("operator"), ">")

    def test_timer_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Timer",
                "position": {"x": 100, "y": 100},
                "node_type": "timer", "duration": "2.5",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].kind, NodeKind.TIMER)
        self.assertEqual(ir.nodes[0].get_param_value("duration"), 2.5)

    def test_sensor_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Sensor Input",
                "position": {"x": 100, "y": 100},
                "node_type": "sensor_input", "ui_selection": "Read IMU",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].kind, NodeKind.SENSOR)
        self.assertEqual(ir.nodes[0].get_param_value("sensor_type"), "imu")

    def test_stop_node(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Action Execution",
                "position": {"x": 100, "y": 100},
                "node_type": "action_execution", "ui_selection": "Stop",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, diags = converter.convert(data, "go2")
        self.assertEqual(ir.nodes[0].kind, NodeKind.STOP)


class TestIRToCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def _pipeline(self, data, robot_type="go2"):
        converter = CanvasToIR()
        ir, _ = converter.convert(data, robot_type)
        generator = IRToCode()
        code, diags, sm = generator.generate(ir)
        return code

    def test_single_action_code(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Action Execution",
                "position": {"x": 100, "y": 100},
                "node_type": "action_execution", "ui_selection": "Stand",
            }],
            "connections": [],
        }
        code = self._pipeline(data)
        self.assertIn("RobotContext.run_action('stand')", code)
        self.assertIn("def execute_workflow", code)

    def test_chained_actions_code(self):
        data = {
            "nodes": [
                {"id": 0, "display_name": "Action Execution",
                 "position": {"x": 100, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
                {"id": 1, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Walk"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "flow_out",
                 "to_node": 1, "to_port": "flow_in"},
            ],
        }
        code = self._pipeline(data)
        self.assertIn("RobotContext.run_action('stand')", code)
        self.assertIn("RobotContext.run_action('walk')", code)
        # stand should come before walk
        stand_idx = code.index("stand")
        walk_idx = code.index("walk")
        self.assertLess(stand_idx, walk_idx)

    def test_if_else_code(self):
        data = {
            "nodes": [
                {"id": 0, "display_name": "Logic Control",
                 "position": {"x": 100, "y": 100},
                 "node_type": "if", "ui_selection": "If",
                 "condition_expr": "True"},
                {"id": 1, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 50},
                 "node_type": "action_execution", "ui_selection": "Stand"},
                {"id": 2, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 200},
                 "node_type": "action_execution", "ui_selection": "Sit"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "out_if",
                 "to_node": 1, "to_port": "flow_in"},
                {"from_node": 0, "from_port": "out_else",
                 "to_node": 2, "to_port": "flow_in"},
            ],
        }
        code = self._pipeline(data)
        self.assertIn("if True:", code)
        self.assertIn("RobotContext.run_action('stand')", code)
        self.assertIn("else:", code)
        self.assertIn("RobotContext.run_action('sit')", code)

    def test_for_loop_code(self):
        data = {
            "nodes": [
                {"id": 0, "display_name": "Logic Control",
                 "position": {"x": 100, "y": 100},
                 "node_type": "while_loop", "ui_selection": "While Loop",
                 "loop_type": "For", "for_start": "0", "for_end": "5", "for_step": "1"},
                {"id": 1, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "loop_body",
                 "to_node": 1, "to_port": "flow_in"},
            ],
        }
        code = self._pipeline(data)
        self.assertIn("for i in range(0, 5, 1):", code)
        self.assertIn("RobotContext.run_action('stand')", code)

    def test_timer_code(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Timer",
                "position": {"x": 100, "y": 100},
                "node_type": "timer", "duration": "2.0",
            }],
            "connections": [],
        }
        code = self._pipeline(data)
        self.assertIn("time.sleep(2.0)", code)

    def test_empty_workflow(self):
        data = {"nodes": [], "connections": []}
        code = self._pipeline(data)
        self.assertIn("pass", code)


class TestSemanticValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def test_valid_workflow(self):
        data = {
            "nodes": [{
                "id": 0, "display_name": "Action Execution",
                "position": {"x": 100, "y": 100},
                "node_type": "action_execution", "ui_selection": "Stand",
            }],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, _ = converter.convert(data, "go2")
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_dangling_edge(self):
        from compiler.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind, EdgeType
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_edge(IREdge("n1", "flow_out", "nonexistent", "flow_in"))
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
