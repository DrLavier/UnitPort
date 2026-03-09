#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 4 fix: schema loading and validator coverage for protocol_emit."""

import unittest


class TestProtocolEmitSchemaLoading(unittest.TestCase):
    def test_registry_loads_builtin_protocol_emit_schema(self):
        from compiler.schema.registry import SchemaRegistry

        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()
        schema = SchemaRegistry.get("builtin.protocol_emit")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.node_type, "protocol_emit")


class TestProtocolEmitSemanticValidation(unittest.TestCase):
    def test_semantic_validator_accepts_builtin_protocol_emit_schema_id(self):
        from compiler.semantic.validator import SemanticValidator
        from compiler.ir.workflow_ir import WorkflowIR, IRNode, NodeKind

        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="builtin.protocol_emit", kind=NodeKind.PROTOCOL_EMIT))
        diags = SemanticValidator().validate(ir)
        codes = [d.code for d in diags]
        self.assertNotIn("E2001", codes, f"unexpected schema error: {codes}")

    def test_canvas_to_ir_uses_builtin_protocol_emit_schema_id(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR

        graph_data = {
            "nodes": [
                {"id": 1, "display_name": "Protocol Emit", "node_type": "protocol_emit"},
            ],
            "connections": [],
        }
        ir, diags = CanvasToIR().convert(graph_data, "go2")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].schema_id, "builtin.protocol_emit")
        self.assertEqual(ir.nodes[0].kind.value, "protocol_emit")
        # Ensure converter did not emit schema-not-found warning for this node.
        msgs = [d.message for d in diags]
        self.assertFalse(any("No schema found for node type 'protocol_emit'" in m for m in msgs))


if __name__ == "__main__":
    unittest.main()

