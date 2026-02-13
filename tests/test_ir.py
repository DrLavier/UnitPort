#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the Workflow IR data structures."""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, IRVariable,
    IRNodeUI, SourceSpan, NodeKind, EdgeType,
)
from compiler.ir.types import IRType, PortDirection


class TestIRTypes(unittest.TestCase):
    def test_ir_type_from_string(self):
        self.assertEqual(IRType.from_string("int"), IRType.INT)
        self.assertEqual(IRType.from_string("FLOAT"), IRType.FLOAT)
        self.assertEqual(IRType.from_string("unknown"), IRType.ANY)

    def test_port_direction(self):
        self.assertEqual(PortDirection.INPUT.value, "input")
        self.assertEqual(PortDirection.OUTPUT.value, "output")


class TestIRParam(unittest.TestCase):
    def test_round_trip(self):
        p = IRParam(name="action", value="stand", param_type="string")
        d = p.to_dict()
        p2 = IRParam.from_dict(d)
        self.assertEqual(p.name, p2.name)
        self.assertEqual(p.value, p2.value)
        self.assertEqual(p.param_type, p2.param_type)


class TestIRNode(unittest.TestCase):
    def test_new_id(self):
        id1 = IRNode.new_id()
        id2 = IRNode.new_id()
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(id1), 8)

    def test_round_trip(self):
        node = IRNode(
            id="abc123",
            schema_id="builtin.action_execution",
            kind=NodeKind.ACTION,
            params={"action": IRParam("action", "stand", "string")},
            ui=IRNodeUI(x=100, y=200, width=180, height=110),
        )
        d = node.to_dict()
        node2 = IRNode.from_dict(d)
        self.assertEqual(node.id, node2.id)
        self.assertEqual(node.schema_id, node2.schema_id)
        self.assertEqual(node.kind, node2.kind)
        self.assertEqual(node2.get_param_value("action"), "stand")
        self.assertIsNotNone(node2.ui)
        self.assertEqual(node2.ui.x, 100)

    def test_opaque_node(self):
        node = IRNode(
            id="opq1",
            schema_id="builtin.opaque_code",
            kind=NodeKind.OPAQUE,
            opaque_code="print('hello')",
        )
        d = node.to_dict()
        node2 = IRNode.from_dict(d)
        self.assertEqual(node2.opaque_code, "print('hello')")
        self.assertEqual(node2.kind, NodeKind.OPAQUE)

    def test_set_param(self):
        node = IRNode(id="n1", schema_id="test", kind=NodeKind.ACTION)
        node.set_param("speed", 1.5, "float")
        self.assertEqual(node.get_param_value("speed"), 1.5)
        self.assertEqual(node.get_param_value("missing", "default"), "default")


class TestIREdge(unittest.TestCase):
    def test_round_trip(self):
        e = IREdge(from_node="n1", from_port="flow_out",
                   to_node="n2", to_port="flow_in", edge_type=EdgeType.FLOW)
        d = e.to_dict()
        e2 = IREdge.from_dict(d)
        self.assertEqual(e.from_node, e2.from_node)
        self.assertEqual(e.edge_type, e2.edge_type)

    def test_data_edge(self):
        e = IREdge(from_node="n1", from_port="result",
                   to_node="n2", to_port="condition", edge_type=EdgeType.DATA)
        d = e.to_dict()
        e2 = IREdge.from_dict(d)
        self.assertEqual(e2.edge_type, EdgeType.DATA)


class TestWorkflowIR(unittest.TestCase):
    def _make_simple_ir(self) -> WorkflowIR:
        ir = WorkflowIR(name="test_workflow", robot_type="go2")
        n1 = IRNode(id="n1", schema_id="builtin.action_execution",
                    kind=NodeKind.ACTION)
        n1.set_param("action", "stand")
        n2 = IRNode(id="n2", schema_id="builtin.action_execution",
                    kind=NodeKind.ACTION)
        n2.set_param("action", "walk")
        ir.add_node(n1)
        ir.add_node(n2)
        ir.add_edge(IREdge("n1", "flow_out", "n2", "flow_in", EdgeType.FLOW))
        return ir

    def test_json_round_trip(self):
        ir = self._make_simple_ir()
        json_str = ir.to_json()
        ir2 = WorkflowIR.from_json(json_str)
        self.assertEqual(ir.name, ir2.name)
        self.assertEqual(ir.robot_type, ir2.robot_type)
        self.assertEqual(len(ir2.nodes), 2)
        self.assertEqual(len(ir2.edges), 1)
        self.assertEqual(ir2.nodes[0].get_param_value("action"), "stand")

    def test_dict_round_trip(self):
        ir = self._make_simple_ir()
        d = ir.to_dict()
        ir2 = WorkflowIR.from_dict(d)
        self.assertEqual(len(ir2.nodes), len(ir.nodes))
        self.assertEqual(len(ir2.edges), len(ir.edges))

    def test_get_node(self):
        ir = self._make_simple_ir()
        self.assertIsNotNone(ir.get_node("n1"))
        self.assertIsNone(ir.get_node("nonexistent"))

    def test_get_entry_nodes(self):
        ir = self._make_simple_ir()
        entries = ir.get_entry_nodes()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "n1")

    def test_get_edges(self):
        ir = self._make_simple_ir()
        self.assertEqual(len(ir.get_outgoing_edges("n1")), 1)
        self.assertEqual(len(ir.get_incoming_edges("n2")), 1)
        self.assertEqual(len(ir.get_outgoing_edges("n2")), 0)

    def test_get_nodes_by_kind(self):
        ir = self._make_simple_ir()
        actions = ir.get_nodes_by_kind(NodeKind.ACTION)
        self.assertEqual(len(actions), 2)

    def test_variables(self):
        ir = WorkflowIR()
        ir.variables.append(IRVariable("counter", 0, "number"))
        d = ir.to_dict()
        ir2 = WorkflowIR.from_dict(d)
        self.assertEqual(len(ir2.variables), 1)
        self.assertEqual(ir2.variables[0].name, "counter")

    def test_empty_ir(self):
        ir = WorkflowIR()
        json_str = ir.to_json()
        ir2 = WorkflowIR.from_json(json_str)
        self.assertEqual(len(ir2.nodes), 0)
        self.assertEqual(len(ir2.edges), 0)
        self.assertEqual(ir2.ir_version, "1.0")


class TestSourceSpan(unittest.TestCase):
    def test_round_trip(self):
        span = SourceSpan(line_start=1, line_end=3, col_start=0, col_end=20)
        d = span.to_dict()
        span2 = SourceSpan.from_dict(d)
        self.assertEqual(span.line_start, span2.line_start)
        self.assertEqual(span.col_end, span2.col_end)


if __name__ == "__main__":
    unittest.main()
