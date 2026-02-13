#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the diagnostic system and error codes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.semantic.diagnostics import (
    Diagnostic, DiagnosticLevel, DiagnosticLocation,
    make_error, make_warning, make_info,
)
from compiler.semantic.error_codes import get_error_code, get_all_codes, ErrorCodeEntry
from compiler.semantic.validator import SemanticValidator
from compiler.ir.workflow_ir import WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType
from compiler.schema.registry import SchemaRegistry


class TestDiagnosticCreation(unittest.TestCase):
    def test_make_error(self):
        diag = make_error("E2001", "Schema not found", node_id="n1")
        self.assertEqual(diag.level, DiagnosticLevel.ERROR)
        self.assertEqual(diag.code, "E2001")
        self.assertIn("Schema not found", diag.message)
        self.assertEqual(diag.location.node_id, "n1")

    def test_make_warning(self):
        diag = make_warning("W3001", "Edge skipped")
        self.assertEqual(diag.level, DiagnosticLevel.WARNING)

    def test_make_info(self):
        diag = make_info("I4001", "Code generated")
        self.assertEqual(diag.level, DiagnosticLevel.INFO)

    def test_diagnostic_level_values(self):
        self.assertEqual(DiagnosticLevel.ERROR.value, "error")
        self.assertEqual(DiagnosticLevel.WARNING.value, "warn")
        self.assertEqual(DiagnosticLevel.INFO.value, "info")


class TestErrorCodes(unittest.TestCase):
    def test_error_code_lookup(self):
        entry = get_error_code("E1001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.code, "E1001")
        self.assertEqual(entry.category, "syntax")
        self.assertEqual(entry.severity, "error")

    def test_warning_code_lookup(self):
        entry = get_error_code("W3001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.severity, "warn")

    def test_info_code_lookup(self):
        entry = get_error_code("I4001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.severity, "info")

    def test_nonexistent_code(self):
        entry = get_error_code("X9999")
        self.assertIsNone(entry)

    def test_all_codes_exist(self):
        codes = get_all_codes()
        self.assertGreater(len(codes), 10)
        for code, entry in codes.items():
            self.assertEqual(code, entry.code)
            self.assertIn(entry.severity, ("error", "warn", "info"))

    def test_code_prefix_convention(self):
        """E codes should be errors/warnings, W should be warnings, I should be info."""
        codes = get_all_codes()
        for code, entry in codes.items():
            if code.startswith("E") and int(code[1:]) < 2000:
                self.assertIn(entry.severity, ("error", "warn"))
            elif code.startswith("W"):
                self.assertEqual(entry.severity, "warn")
            elif code.startswith("I"):
                self.assertEqual(entry.severity, "info")


class TestValidatorDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def test_valid_workflow_no_errors(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION,
                           params={"action": IRParam("action", "stand", "string")}))
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_dangling_edge_error(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="builtin.action_execution",
                           kind=NodeKind.ACTION))
        ir.add_edge(IREdge("n1", "flow_out", "nonexistent", "flow_in"))
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertGreater(len(errors), 0)

    def test_unknown_schema_error(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="unknown.foo",
                           kind=NodeKind.CUSTOM))
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertGreater(len(errors), 0)

    def test_param_out_of_range(self):
        ir = WorkflowIR()
        ir.add_node(IRNode(id="n1", schema_id="builtin.timer",
                           kind=NodeKind.TIMER,
                           params={"duration": IRParam("duration", -5.0, "float")}))
        validator = SemanticValidator()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.level == DiagnosticLevel.ERROR]
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
