#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for IR to Canvas conversion."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType,
)
from compiler.lowering.ir_to_canvas import IRToCanvas
from compiler.lowering.layout import LayoutEngine
from compiler.schema.registry import SchemaRegistry


class TestIRToCanvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def test_single_action(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION,
                           params={"action": IRParam("action", "stand", "string")}))
        converter = IRToCanvas()
        data, diags = converter.convert(ir)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["display_name"], "Action Execution")
        self.assertEqual(data["nodes"][0]["ui_selection"], "Stand")

    def test_stop_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.stop",
                           kind=NodeKind.STOP))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["ui_selection"], "Stop")

    def test_timer_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.timer",
                           kind=NodeKind.TIMER,
                           params={"duration": IRParam("duration", 2.5, "float")}))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["display_name"], "Timer")
        self.assertEqual(data["nodes"][0]["duration"], "2.5")

    def test_if_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.if",
                           kind=NodeKind.LOGIC,
                           params={"condition_expr": IRParam("condition_expr", "x > 5", "string")}))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["display_name"], "Logic Control")
        self.assertEqual(data["nodes"][0]["condition_expr"], "x > 5")

    def test_for_loop_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.while_loop",
                           kind=NodeKind.LOGIC,
                           params={
                               "loop_type": IRParam("loop_type", "for", "string"),
                               "for_start": IRParam("for_start", 0, "int"),
                               "for_end": IRParam("for_end", 5, "int"),
                               "for_step": IRParam("for_step", 1, "int"),
                           }))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["loop_type"], "For")
        self.assertEqual(data["nodes"][0]["for_start"], "0")
        self.assertEqual(data["nodes"][0]["for_end"], "5")

    def test_sensor_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.sensor_input",
                           kind=NodeKind.SENSOR,
                           params={"sensor_type": IRParam("sensor_type", "imu", "string")}))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["display_name"], "Sensor Input")
        self.assertEqual(data["nodes"][0]["ui_selection"], "Read IMU")

    def test_connections_preserved(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION,
                           params={"action": IRParam("action", "stand", "string")}))
        ir.add_node(IRNode(id="1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION,
                           params={"action": IRParam("action", "walk", "string")}))
        ir.add_edge(IREdge("0", "flow_out", "1", "flow_in"))
        converter = IRToCanvas()
        data, _ = converter.convert(ir)
        self.assertEqual(len(data["connections"]), 1)
        self.assertEqual(data["connections"][0]["from_node"], 0)
        self.assertEqual(data["connections"][0]["to_node"], 1)

    def test_opaque_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.opaque",
                           kind=NodeKind.OPAQUE,
                           opaque_code="print('hello')"))
        converter = IRToCanvas()
        data, diags = converter.convert(ir)
        self.assertEqual(data["nodes"][0]["node_type"], "opaque")
        self.assertEqual(data["nodes"][0]["code"], "print('hello')")


class TestLayoutEngine(unittest.TestCase):
    def test_single_node(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        engine = LayoutEngine()
        engine.layout(ir)
        self.assertIsNotNone(ir.nodes[0].ui)
        self.assertGreater(ir.nodes[0].ui.x, 0)

    def test_chained_nodes_ordered(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_node(IRNode(id="1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_edge(IREdge("0", "flow_out", "1", "flow_in"))
        engine = LayoutEngine()
        engine.layout(ir)
        # Second node should be to the right
        self.assertGreater(ir.nodes[1].ui.x, ir.nodes[0].ui.x)

    def test_branching_layout(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="0", schema_id="builtin.if",
                           kind=NodeKind.LOGIC))
        ir.add_node(IRNode(id="1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_node(IRNode(id="2", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_edge(IREdge("0", "out_if", "1", "flow_in"))
        ir.add_edge(IREdge("0", "out_else", "2", "flow_in"))
        engine = LayoutEngine()
        engine.layout(ir)
        # Both branch nodes should be to the right of the if node
        self.assertGreater(ir.nodes[1].ui.x, ir.nodes[0].ui.x)
        self.assertGreater(ir.nodes[2].ui.x, ir.nodes[0].ui.x)
        # Branch nodes should have different Y positions
        self.assertNotEqual(ir.nodes[1].ui.y, ir.nodes[2].ui.y)

    def test_empty_ir(self):
        ir = WorkflowIR()
        engine = LayoutEngine()
        engine.layout(ir)  # Should not crash


if __name__ == "__main__":
    unittest.main()
