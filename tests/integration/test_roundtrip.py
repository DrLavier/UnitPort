#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip tests: Canvas -> IR -> Code -> IR -> Canvas
Verifies that the bidirectional compilation pipeline preserves workflow semantics.
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.lowering.canvas_to_ir import CanvasToIR
from compiler.codegen.ir_to_code import IRToCode
from compiler.parser.parser import Parser
from compiler.lowering.ast_to_ir import ASTToIR
from compiler.lowering.ir_to_canvas import IRToCanvas
from compiler.roundtrip.normalizer import IRNormalizer
from compiler.schema.registry import SchemaRegistry
from compiler.ir.workflow_ir import NodeKind


SAMPLES_DIR = Path(__file__).parent.parent / "regression" / "samples"


class TestNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def test_identical_ir(self):
        """Two identical IRs should have score 1.0."""
        data = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 100, "y": 100},
                        "node_type": "action_execution", "ui_selection": "Stand"}],
            "connections": [],
        }
        converter = CanvasToIR()
        ir1, _ = converter.convert(data, "go2")
        ir2, _ = converter.convert(data, "go2")

        normalizer = IRNormalizer()
        score = normalizer.compare(ir1, ir2)
        self.assertEqual(score, 1.0)

    def test_different_positions_same_score(self):
        """Position differences should not affect comparison."""
        data1 = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 100, "y": 100},
                        "node_type": "action_execution", "ui_selection": "Stand"}],
            "connections": [],
        }
        data2 = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 999, "y": 999},
                        "node_type": "action_execution", "ui_selection": "Stand"}],
            "connections": [],
        }
        converter = CanvasToIR()
        ir1, _ = converter.convert(data1, "go2")
        ir2, _ = converter.convert(data2, "go2")

        normalizer = IRNormalizer()
        score = normalizer.compare(ir1, ir2)
        self.assertEqual(score, 1.0)

    def test_different_actions_low_score(self):
        """Different actions should produce a lower score."""
        data1 = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 100, "y": 100},
                        "node_type": "action_execution", "ui_selection": "Stand"}],
            "connections": [],
        }
        data2 = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 100, "y": 100},
                        "node_type": "action_execution", "ui_selection": "Walk"}],
            "connections": [],
        }
        converter = CanvasToIR()
        ir1, _ = converter.convert(data1, "go2")
        ir2, _ = converter.convert(data2, "go2")

        normalizer = IRNormalizer()
        score = normalizer.compare(ir1, ir2)
        # Same kind/schema but different param -> should be positive but not 1.0
        self.assertGreaterEqual(score, 0.3)
        self.assertLess(score, 1.0)

    def test_empty_irs(self):
        """Two empty IRs should be equivalent."""
        from compiler.ir.workflow_ir import WorkflowIR
        normalizer = IRNormalizer()
        score = normalizer.compare(WorkflowIR(), WorkflowIR())
        self.assertEqual(score, 1.0)

    def test_normalize_strips_ui(self):
        """Normalization should strip UI metadata."""
        data = {
            "nodes": [{"id": 0, "display_name": "Action Execution",
                        "position": {"x": 100, "y": 200},
                        "node_type": "action_execution", "ui_selection": "Stand"}],
            "connections": [],
        }
        converter = CanvasToIR()
        ir, _ = converter.convert(data, "go2")
        normalizer = IRNormalizer()
        normalized = normalizer.normalize(ir)
        self.assertIsNone(normalized.nodes[0].ui)

    def test_normalize_sequential_ids(self):
        """Normalization should assign sequential IDs."""
        data = {
            "nodes": [
                {"id": 5, "display_name": "Action Execution",
                 "position": {"x": 100, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
                {"id": 10, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Walk"},
            ],
            "connections": [
                {"from_node": 5, "from_port": "flow_out",
                 "to_node": 10, "to_port": "flow_in"},
            ],
        }
        converter = CanvasToIR()
        ir, _ = converter.convert(data, "go2")
        normalizer = IRNormalizer()
        normalized = normalizer.normalize(ir)
        self.assertEqual(normalized.nodes[0].id, "0")
        self.assertEqual(normalized.nodes[1].id, "1")
        self.assertEqual(normalized.edges[0].from_node, "0")
        self.assertEqual(normalized.edges[0].to_node, "1")


class TestRoundTrip(unittest.TestCase):
    """
    Test full round-trip: Canvas -> IR -> Code -> IR

    For each sample:
    1. Load canvas graph_data
    2. Convert to IR (canvas_to_ir)
    3. Generate code (ir_to_code)
    4. Parse code back (parser)
    5. Lower to IR (ast_to_ir)
    6. Compare normalized IRs
    """

    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()
        cls.normalizer = IRNormalizer()
        cls.canvas_converter = CanvasToIR()
        cls.code_generator = IRToCode()

    def _roundtrip(self, graph_data: dict) -> float:
        """Run a full round-trip and return equivalence score."""
        # Forward: Canvas -> IR -> Code
        ir_forward, _ = self.canvas_converter.convert(graph_data, "go2")
        code, _, _ = self.code_generator.generate(ir_forward)

        # Reverse: Code -> AST -> IR
        parser = Parser(code)
        ast, _ = parser.parse()
        lowerer = ASTToIR()
        ir_reverse, _ = lowerer.lower(ast, "go2")

        # Compare
        return self.normalizer.compare(ir_forward, ir_reverse)

    def test_sample_01_single_action(self):
        data = self._load_sample("sample_01_single_action.json")
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.95, f"Round-trip score too low: {score}")

    def test_sample_02_chained_actions(self):
        sample_path = self._find_sample("sample_02")
        data = self._load_sample_path(sample_path)
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.95, f"Round-trip score too low: {score}")

    def test_sample_03_if_else(self):
        sample_path = self._find_sample("sample_03")
        data = self._load_sample_path(sample_path)
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.80, f"Round-trip score too low: {score}")

    def test_sample_05_while_loop(self):
        data = self._load_sample("sample_05_while_loop.json")
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.80, f"Round-trip score too low: {score}")

    def test_sample_06_for_loop(self):
        data = self._load_sample("sample_06_for_loop.json")
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.80, f"Round-trip score too low: {score}")

    def test_sample_08_timer(self):
        sample_path = self._find_sample("sample_08")
        data = self._load_sample_path(sample_path)
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.80, f"Round-trip score too low: {score}")

    def test_sample_09_sensor(self):
        sample_path = self._find_sample("sample_09")
        data = self._load_sample_path(sample_path)
        score = self._roundtrip(data)
        self.assertGreaterEqual(score, 0.70, f"Round-trip score too low: {score}")

    def test_canvas_to_code_to_canvas(self):
        """Test Canvas -> IR -> Code -> IR -> Canvas data round-trip."""
        data = {
            "nodes": [
                {"id": 0, "display_name": "Action Execution",
                 "position": {"x": 100, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
                {"id": 1, "display_name": "Timer",
                 "position": {"x": 400, "y": 100},
                 "node_type": "timer", "duration": "2.0"},
                {"id": 2, "display_name": "Action Execution",
                 "position": {"x": 700, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Walk"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "flow_out",
                 "to_node": 1, "to_port": "flow_in"},
                {"from_node": 1, "from_port": "flow_out",
                 "to_node": 2, "to_port": "flow_in"},
            ],
        }

        # Forward: Canvas -> IR -> Code
        ir1, _ = self.canvas_converter.convert(data, "go2")
        code, _, _ = self.code_generator.generate(ir1)

        # Reverse: Code -> IR -> Canvas
        parser = Parser(code)
        ast, _ = parser.parse()
        lowerer = ASTToIR()
        ir2, _ = lowerer.lower(ast, "go2")
        ir_to_canvas = IRToCanvas()
        canvas_data, _ = ir_to_canvas.convert(ir2)

        # Re-forward: Canvas -> IR
        ir3, _ = self.canvas_converter.convert(canvas_data, "go2")

        # Compare ir1 with ir3
        score = self.normalizer.compare(ir1, ir3)
        self.assertGreaterEqual(score, 0.80,
                                f"Full round-trip score too low: {score}")

    # ---------- Helpers ----------

    @staticmethod
    def _load_sample(filename: str) -> dict:
        path = SAMPLES_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_sample_path(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_sample(prefix: str) -> Path:
        """Find sample file by prefix (handles naming variations)."""
        for p in SAMPLES_DIR.iterdir():
            if p.name.startswith(prefix) and p.suffix == ".json":
                return p
        raise FileNotFoundError(f"No sample found with prefix: {prefix}")


if __name__ == "__main__":
    unittest.main()
