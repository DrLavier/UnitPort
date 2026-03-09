#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests — Circle 1 Step 1.3/1.4: Explicit behavior node dispatch.

Coverage
--------
- NodeKind.BEHAVIOR_CALL existence in compiler IR
- _DISPLAY_NAME_TO_NODE_TYPE["Behavior"] == "behavior_call"
- CanvasToIR lowering: "Behavior" canvas node → NodeKind.BEHAVIOR_CALL + behavior_ref param
- CanvasToIR lowering: behavior_ref falls back to display_name when ui_selection blank
- NodeExecutor _is_behavior detection fires for type "behavior_call"
- NodeExecutor _is_behavior fires for schema_id "behavior_call"
- WorkflowRunner: behavior_call node → status="skipped", reason="behavior_call_requires_workflowir"
- WorkflowRunner: behavior node (legacy type) → same skipped result
- WorkflowRunner: behavior_call flow continues to downstream nodes
- WorkflowRunner: behavior_call result does NOT set failed flag (run ok=True)
- WorkflowRunner: behavior_ref is captured in result dict
- node_status_callback receives "success" (not "failed") for skipped behavior nodes
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# compiler.ir — NodeKind.BEHAVIOR_CALL
# ---------------------------------------------------------------------------
from compiler.ir.workflow_ir import NodeKind   # noqa: E402


class TestNodeKindBehaviorCall(unittest.TestCase):
    """NodeKind.BEHAVIOR_CALL must exist and have the correct value."""

    def test_behavior_call_exists(self):
        self.assertTrue(hasattr(NodeKind, "BEHAVIOR_CALL"),
                        "NodeKind.BEHAVIOR_CALL not found in compiler.ir.workflow_ir")

    def test_behavior_call_value(self):
        self.assertEqual(NodeKind.BEHAVIOR_CALL.value, "behavior_call")

    def test_from_string_behavior_call(self):
        result = NodeKind.from_string("behavior_call")
        self.assertEqual(result, NodeKind.BEHAVIOR_CALL)

    def test_behavior_call_is_not_custom(self):
        self.assertNotEqual(NodeKind.BEHAVIOR_CALL, NodeKind.CUSTOM)


# ---------------------------------------------------------------------------
# compiler.lowering.canvas_to_ir — display name mapping + lowering
# ---------------------------------------------------------------------------
from compiler.lowering.canvas_to_ir import CanvasToIR, _DISPLAY_NAME_TO_NODE_TYPE  # noqa: E402
from compiler.schema.registry import SchemaRegistry                                  # noqa: E402


class TestDisplayNameMapping(unittest.TestCase):
    def test_behavior_maps_to_behavior_call(self):
        self.assertEqual(_DISPLAY_NAME_TO_NODE_TYPE.get("Behavior"), "behavior_call",
                         '"Behavior" display name must map to "behavior_call" in canvas_to_ir')


class TestCanvasToIRBehaviorNode(unittest.TestCase):
    """CanvasToIR correctly lowers a 'Behavior' canvas node."""

    @classmethod
    def setUpClass(cls):
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    def _make_behavior_canvas_node(self, node_id=0, ui_selection="walk_patrol",
                                   behavior_ref=None, display_name="Behavior"):
        node = {
            "id": node_id,
            "display_name": display_name,
            "position": {"x": 100, "y": 100},
            "node_type": "behavior_call",
            "ui_selection": ui_selection,
        }
        if behavior_ref is not None:
            node["behavior_ref"] = behavior_ref
        return node

    def test_behavior_node_kind(self):
        data = {
            "nodes": [self._make_behavior_canvas_node(ui_selection="patrol")],
            "connections": [],
        }
        ir, diags = CanvasToIR().convert(data, "go2")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.BEHAVIOR_CALL)

    def test_behavior_node_schema_id(self):
        data = {
            "nodes": [self._make_behavior_canvas_node(ui_selection="patrol")],
            "connections": [],
        }
        ir, _ = CanvasToIR().convert(data, "go2")
        self.assertEqual(ir.nodes[0].schema_id, "behavior_call")

    def test_behavior_ref_from_ui_selection(self):
        data = {
            "nodes": [self._make_behavior_canvas_node(ui_selection="walk_patrol")],
            "connections": [],
        }
        ir, _ = CanvasToIR().convert(data, "go2")
        ref = ir.nodes[0].get_param_value("behavior_ref")
        self.assertEqual(ref, "walk_patrol")

    def test_explicit_behavior_ref_overrides_ui_selection(self):
        data = {
            "nodes": [self._make_behavior_canvas_node(
                ui_selection="something", behavior_ref="explicit_ref")],
            "connections": [],
        }
        ir, _ = CanvasToIR().convert(data, "go2")
        ref = ir.nodes[0].get_param_value("behavior_ref")
        self.assertEqual(ref, "explicit_ref")

    def test_behavior_ref_fallback_to_display_name(self):
        """When ui_selection is blank, behavior_ref falls back to display_name."""
        data = {
            "nodes": [{
                "id": 0,
                "display_name": "MyBehavior",
                "position": {"x": 0, "y": 0},
                "node_type": "behavior_call",
                "ui_selection": "",
            }],
            "connections": [],
        }
        ir, _ = CanvasToIR().convert(data, "go2")
        ref = ir.nodes[0].get_param_value("behavior_ref")
        self.assertEqual(ref, "MyBehavior")

    def test_no_compile_errors_for_behavior_node(self):
        """Lowering a behavior_call node must not produce error-level diagnostics."""
        from compiler.semantic.diagnostics import DiagnosticLevel
        data = {
            "nodes": [self._make_behavior_canvas_node(ui_selection="survey_area")],
            "connections": [],
        }
        _, diags = CanvasToIR().convert(data, "go2")
        errors = [d for d in diags
                  if getattr(d.level, "value", str(d.level)) == "error"]
        self.assertEqual(errors, [], f"Unexpected compile errors: {errors}")

    def test_behavior_node_chained_to_action(self):
        """behavior_call followed by an action: edges preserved after lowering."""
        data = {
            "nodes": [
                self._make_behavior_canvas_node(node_id=0, ui_selection="patrol"),
                {"id": 1, "display_name": "Action Execution",
                 "position": {"x": 400, "y": 100},
                 "node_type": "action_execution", "ui_selection": "Stand"},
            ],
            "connections": [
                {"from_node": 0, "from_port": "flow_out",
                 "to_node": 1, "to_port": "flow_in"},
            ],
        }
        ir, _ = CanvasToIR().convert(data, "go2")
        self.assertEqual(len(ir.nodes), 2)
        self.assertEqual(len(ir.edges), 1)
        behavior_node = next(n for n in ir.nodes if n.kind == NodeKind.BEHAVIOR_CALL)
        self.assertIsNotNone(behavior_node)


# ---------------------------------------------------------------------------
# system.runtime.node_executor — _is_behavior detection
# ---------------------------------------------------------------------------
from system.runtime.node_executor import NodeExecutor   # noqa: E402

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "node": self.nid}


class TestNodeExecutorBehaviorCallDetection(unittest.TestCase):
    """NodeExecutor routes behavior_call typed nodes to _execute_behavior_node."""

    @classmethod
    def setUpClass(cls):
        cls._orig = sys.modules.get("nodes")
        _registry.clear()
        _registry["simple"] = SimpleNode
        sys.modules["nodes"] = _nodes_mod

    @classmethod
    def tearDownClass(cls):
        if cls._orig is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig

    def _run_single_node(self, node_type: str, node_data: dict | None = None) -> dict:
        ex = NodeExecutor()
        ex.add_node("b0", node_type, node_data or {})
        return ex.execute()

    def test_behavior_call_type_is_detected_as_behavior(self):
        """Node with type='behavior_call' gets skipped (no bridge) — not 'unknown_type'."""
        r = self._run_single_node("behavior_call")
        node_r = r.get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")
        # The reason must be the behavior-skipped code, NOT the unknown_type code
        self.assertNotIn("unknown_type", node_r.get("reason", ""))

    def test_behavior_schema_id_is_detected(self):
        """Node with schema_id='behavior_call' is treated as behavior."""
        r = self._run_single_node("some_type", {"schema_id": "behavior_call"})
        node_r = r.get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")

    def test_legacy_behavior_type_still_detected(self):
        """Node with type='behavior' (legacy) is still detected as behavior."""
        r = self._run_single_node("behavior")
        node_r = r.get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")

    def test_external_kind_behavior_detected(self):
        """Node with external_kind='behavior' is detected as behavior."""
        r = self._run_single_node("action_execution", {"external_kind": "behavior"})
        node_r = r.get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")

    def test_non_behavior_action_not_skipped(self):
        """A plain action_execution node without behavior markers is NOT treated as behavior."""
        r = self._run_single_node("simple")
        node_r = r.get("b0", {})
        # simple node executes normally
        self.assertEqual(node_r.get("status"), "ok")


# ---------------------------------------------------------------------------
# system.runtime.workflow_runner — behavior_call explicit fallback
# ---------------------------------------------------------------------------
from system.runtime.workflow_runner import WorkflowRunner   # noqa: E402


def _minimal_exec_graph(node_type: str = "behavior_call",
                         behavior_ref: str = "my_behavior",
                         external_kind: str | None = None) -> dict:
    """Build a minimal exec_graph with one behavior-type node."""
    node_data = {
        "id": "b0",
        "name": "Behavior",
        "type": node_type,
        "behavior_ref": behavior_ref,
    }
    if external_kind is not None:
        node_data["external_kind"] = external_kind
    return {
        "nodes": {"b0": node_data},
        "outgoing": {"b0": {}},
        "incoming": {"b0": {}},
        "entry_nodes": ["b0"],
    }


class TestWorkflowRunnerBehaviorCallFallback(unittest.TestCase):
    """WorkflowRunner produces deterministic skipped result for behavior nodes."""

    def _run(self, graph: dict, **kw) -> dict:
        return WorkflowRunner().run(graph, robot_model=None, **kw)

    def test_behavior_call_produces_skipped_status(self):
        r = self._run(_minimal_exec_graph("behavior_call"))
        node_r = r["results"].get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")

    def test_behavior_call_reason_code(self):
        r = self._run(_minimal_exec_graph("behavior_call"))
        node_r = r["results"].get("b0", {})
        self.assertEqual(node_r.get("reason"), "behavior_call_requires_workflowir")

    def test_legacy_behavior_type_also_skipped(self):
        r = self._run(_minimal_exec_graph("behavior"))
        node_r = r["results"].get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")
        self.assertEqual(node_r.get("reason"), "behavior_call_requires_workflowir")

    def test_external_kind_behavior_skipped(self):
        g = _minimal_exec_graph("action_execution", external_kind="behavior")
        r = self._run(g)
        node_r = r["results"].get("b0", {})
        self.assertEqual(node_r.get("status"), "skipped")
        self.assertEqual(node_r.get("reason"), "behavior_call_requires_workflowir")

    def test_behavior_ref_captured_in_result(self):
        r = self._run(_minimal_exec_graph("behavior_call", behavior_ref="survey_area"))
        node_r = r["results"].get("b0", {})
        self.assertEqual(node_r.get("behavior_ref"), "survey_area")

    def test_run_ok_true_despite_behavior_skip(self):
        """Skipping a behavior node is NOT a failure — run.ok must be True."""
        r = self._run(_minimal_exec_graph("behavior_call"))
        self.assertTrue(r.get("ok"), f"Expected ok=True, got: {r}")

    def test_node_status_callback_receives_success(self):
        """node_status_callback must be called with 'success' for skipped behavior node."""
        events = []
        self._run(
            _minimal_exec_graph("behavior_call"),
            node_status_callback=lambda nid, st: events.append((nid, st)),
        )
        # Expect running then success, NOT failed
        statuses = [st for nid, st in events if nid == "b0"]
        self.assertIn("running", statuses)
        self.assertIn("success", statuses)
        self.assertNotIn("failed", statuses)

    def test_flow_continues_after_behavior_skip(self):
        """Downstream nodes after a behavior_call node still execute.

        WorkflowRunner.run() returns executed_count; verify it equals 2 to
        confirm that n1 (the downstream behavior_call) was also visited.
        Nodes without logic_node may not write to results, so we use
        executed_count as the authoritative traversal counter.
        """
        graph = {
            "nodes": {
                "b0": {"id": "b0", "name": "Behavior", "type": "behavior_call",
                       "behavior_ref": "x"},
                # Use another behavior_call so it ALSO writes a result entry.
                "n1": {"id": "n1", "name": "Behavior", "type": "behavior_call",
                       "behavior_ref": "y"},
            },
            "outgoing": {
                "b0": {"flow_out": [("n1", "flow_in")]},
                "n1": {},
            },
            "incoming": {
                "b0": {},
                "n1": {"flow_in": [("b0", "flow_out")]},
            },
            "entry_nodes": ["b0"],
        }
        r = WorkflowRunner().run(graph, robot_model=None)
        # b0 must be skipped with the correct reason code
        self.assertEqual(r["results"].get("b0", {}).get("status"), "skipped")
        # n1 must also appear in results (it's a behavior_call node → skipped result)
        self.assertIn("n1", r["results"])
        # executed_count must be 2 (both nodes visited)
        self.assertEqual(r["executed_count"], 2)


if __name__ == "__main__":
    unittest.main()
