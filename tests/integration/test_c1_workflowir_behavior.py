#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests — Circle 1 exit gate: WorkflowIR behavior dispatch.

Coverage
--------
- exec_graph with behavior_call node: WorkflowRunner skips gracefully,
  run ok=True, reason code surface in results
- exec_graph with non-behavior nodes: unchanged (regression)
- WorkflowIR path (NodeExecutor) with behavior_call node: skipped (no bridge),
  RuntimeResult.status == "success" (behavior skip is non-fatal)
- RuntimeResult.diagnostics surfaces BEHAVIOR_ENABLED_RUN and
  EXECUTION_PATH_REASON correctly for both paths
- CanvasToIR → RuntimeEngine integration: behavior canvas node compiles to
  WorkflowIR with BEHAVIOR_CALL kind and executes without error
- Legacy exec_graph mission with no behavior nodes: unaffected (regression)

Isolation
---------
- No Qt, no PySide6, no SDK.
- nodes registry stub installed/removed in setUpClass/tearDownClass.
- SchemaRegistry reset between test classes.
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.runtime_engine import RuntimeEngine              # noqa: E402
from system.runtime.contracts import DiagnosticsKey, RuntimeStatus   # noqa: E402
from system.ir.workflow_ir import (                                   # noqa: E402
    EdgeType, IREdge, IRNode, IRParam, NodeKind, WorkflowIR,
)


# ---------------------------------------------------------------------------
# Shared node stubs
# ---------------------------------------------------------------------------
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "node": self.nid}


def _install_registry():
    _registry.clear()
    _registry["simple"] = SimpleNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_exec_graph(n_nodes: int = 1) -> dict:
    """Build a minimal exec_graph with plain nodes (no behavior)."""
    nodes = {}
    outgoing = {}
    incoming = {}
    for i in range(n_nodes):
        nid = f"n{i}"
        nodes[nid] = {"id": nid, "name": f"Node{i}", "type": "simple", "logic_node": None}
        outgoing[nid] = {}
        incoming[nid] = {}
    if n_nodes > 1:
        # Chain: n0 → n1 → …
        for i in range(n_nodes - 1):
            outgoing[f"n{i}"]["flow_out"] = [(f"n{i+1}", "flow_in")]
            incoming[f"n{i+1}"]["flow_in"] = [(f"n{i}", "flow_out")]
    return {
        "nodes": nodes,
        "outgoing": outgoing,
        "incoming": incoming,
        "entry_nodes": ["n0"],
    }


def _behavior_exec_graph(behavior_ref: str = "patrol") -> dict:
    """Build an exec_graph whose first node is a behavior_call node."""
    return {
        "nodes": {
            "b0": {"id": "b0", "name": "Behavior",
                   "type": "behavior_call", "behavior_ref": behavior_ref},
        },
        "outgoing": {"b0": {}},
        "incoming": {"b0": {}},
        "entry_nodes": ["b0"],
    }


def _behavior_then_behavior_exec_graph() -> dict:
    """behavior_call followed by another behavior_call (both write skipped results)."""
    return {
        "nodes": {
            "b0": {"id": "b0", "name": "Behavior",
                   "type": "behavior_call", "behavior_ref": "survey"},
            "b1": {"id": "b1", "name": "Behavior",
                   "type": "behavior_call", "behavior_ref": "return_home"},
        },
        "outgoing": {
            "b0": {"flow_out": [("b1", "flow_in")]},
            "b1": {},
        },
        "incoming": {
            "b0": {},
            "b1": {"flow_in": [("b0", "flow_out")]},
        },
        "entry_nodes": ["b0"],
    }


def _workflowir_with_behavior(behavior_ref: str = "patrol") -> WorkflowIR:
    """Build a WorkflowIR with a single BEHAVIOR_CALL node."""
    node = IRNode(
        id="bc0",
        schema_id="behavior_call",
        kind=NodeKind.BEHAVIOR_CALL,
        params={"behavior_ref": IRParam("behavior_ref", behavior_ref, "string")},
    )
    return WorkflowIR(nodes=[node], edges=[])


def _workflowir_simple(n_nodes: int = 2) -> WorkflowIR:
    """Build a WorkflowIR with N simple ACTION nodes (no behavior)."""
    nodes = [
        IRNode(id=f"n{i}", schema_id="simple", kind=NodeKind.ACTION)
        for i in range(n_nodes)
    ]
    edges = [
        IREdge(from_node=f"n{i}", from_port="flow_out",
               to_node=f"n{i+1}", to_port="flow_in",
               edge_type=EdgeType.FLOW)
        for i in range(n_nodes - 1)
    ]
    return WorkflowIR(nodes=nodes, edges=edges)


# ══════════════════════════════════════════════════════════════════════════════
# 1. exec_graph path — WorkflowRunner handles behavior_call gracefully
# ══════════════════════════════════════════════════════════════════════════════

class TestExecGraphBehaviorCallFallback(unittest.TestCase):
    """RuntimeEngine exec_graph path: behavior_call node produces skipped result."""

    @classmethod
    def setUpClass(cls):
        cls._orig = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig

    def _run(self, graph: dict) -> dict:
        engine = RuntimeEngine()
        return engine.execute(graph, scenario={})

    def test_behavior_call_run_status_success(self):
        """Mission with only a behavior_call node returns status=success (skipped, not failed)."""
        r = self._run(_behavior_exec_graph("patrol"))
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS,
                         f"Expected success, got: {r['status']} — {r}")

    def test_behavior_call_node_in_results(self):
        r = self._run(_behavior_exec_graph("patrol"))
        self.assertIn("b0", r["results"])

    def test_behavior_call_result_skipped(self):
        r = self._run(_behavior_exec_graph("patrol"))
        node_r = r["results"]["b0"]
        self.assertEqual(node_r.get("status"), "skipped")
        self.assertEqual(node_r.get("reason"), "behavior_call_requires_workflowir")

    def test_behavior_call_ref_in_result(self):
        r = self._run(_behavior_exec_graph("survey_area"))
        node_r = r["results"]["b0"]
        self.assertEqual(node_r.get("behavior_ref"), "survey_area")

    def test_behavior_enabled_run_false_on_exec_graph_path(self):
        """exec_graph path → BEHAVIOR_ENABLED_RUN must be False."""
        r = self._run(_behavior_exec_graph())
        diag = r.get("diagnostics", {})
        self.assertFalse(diag.get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN),
                         "exec_graph path must set behavior_enabled_run=False")

    def test_execution_path_reason_exec_graph(self):
        r = self._run(_behavior_exec_graph())
        diag = r.get("diagnostics", {})
        reason = diag.get(DiagnosticsKey.EXECUTION_PATH_REASON, "")
        self.assertEqual(reason, "exec_graph_compat")

    def test_behavior_then_behavior_both_in_results(self):
        """Downstream behavior_call after behavior_call must also be in results."""
        r = self._run(_behavior_then_behavior_exec_graph())
        self.assertIn("b0", r["results"])
        self.assertIn("b1", r["results"])

    def test_behavior_then_behavior_overall_success(self):
        r = self._run(_behavior_then_behavior_exec_graph())
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)


# ══════════════════════════════════════════════════════════════════════════════
# 2. WorkflowIR path — NodeExecutor handles BEHAVIOR_CALL kind
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowIRBehaviorCallNode(unittest.TestCase):
    """RuntimeEngine WorkflowIR path: behavior_call node skipped (no bridge)."""

    @classmethod
    def setUpClass(cls):
        cls._orig = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig

    def _run(self, ir: WorkflowIR) -> dict:
        engine = RuntimeEngine()
        return engine.execute(ir, scenario={})

    def test_behavior_call_ir_node_status_success(self):
        """Single BEHAVIOR_CALL IR node → RuntimeResult.status == success."""
        r = self._run(_workflowir_with_behavior("patrol"))
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS,
                         f"Expected success, got: {r}")

    def test_behavior_call_ir_node_skipped(self):
        r = self._run(_workflowir_with_behavior("patrol"))
        node_r = r["results"].get("bc0", {})
        self.assertEqual(node_r.get("status"), "skipped")

    def test_behavior_enabled_run_true_on_workflowir_path(self):
        """WorkflowIR path → BEHAVIOR_ENABLED_RUN must be True."""
        r = self._run(_workflowir_with_behavior())
        diag = r.get("diagnostics", {})
        self.assertTrue(diag.get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN),
                        "WorkflowIR path must set behavior_enabled_run=True")

    def test_execution_path_reason_workflowir_direct(self):
        r = self._run(_workflowir_with_behavior())
        diag = r.get("diagnostics", {})
        reason = diag.get(DiagnosticsKey.EXECUTION_PATH_REASON, "")
        self.assertEqual(reason, "workflowir_direct")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CanvasToIR → RuntimeEngine: end-to-end lowering integration
# ══════════════════════════════════════════════════════════════════════════════

class TestCanvasToIRBehaviorIntegration(unittest.TestCase):
    """Canvas behavior node lowers to WorkflowIR and executes via RuntimeEngine."""

    @classmethod
    def setUpClass(cls):
        cls._orig = sys.modules.get("nodes")
        _install_registry()
        from compiler.schema.registry import SchemaRegistry
        SchemaRegistry.reset()
        SchemaRegistry.load_builtins()

    @classmethod
    def tearDownClass(cls):
        if cls._orig is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig

    def _canvas_data_with_behavior(self, ui_selection: str = "walk_patrol") -> dict:
        return {
            "nodes": [{
                "id": 0,
                "display_name": "Behavior",
                "position": {"x": 100, "y": 100},
                "node_type": "behavior_call",
                "ui_selection": ui_selection,
            }],
            "connections": [],
        }

    def test_canvas_behavior_compiles_to_behavior_call_kind(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        data = self._canvas_data_with_behavior("patrol")
        ir, diags = CanvasToIR().convert(data, "go2")
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0].kind, NodeKind.BEHAVIOR_CALL)

    def test_canvas_behavior_compiled_ir_runs_without_error(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        data = self._canvas_data_with_behavior("survey_area")
        ir, _ = CanvasToIR().convert(data, "go2")
        engine = RuntimeEngine()
        r = engine.execute(ir, scenario={})
        # Must not crash and must succeed (behavior skipped but not failed)
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS,
                         f"Expected success, got: {r}")

    def test_canvas_behavior_ir_behavior_ref_param(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        data = self._canvas_data_with_behavior("custom_patrol")
        ir, _ = CanvasToIR().convert(data, "go2")
        ref = ir.nodes[0].get_param_value("behavior_ref")
        self.assertEqual(ref, "custom_patrol")

    def test_canvas_behavior_ir_path_diagnostics(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        data = self._canvas_data_with_behavior()
        ir, _ = CanvasToIR().convert(data, "go2")
        engine = RuntimeEngine()
        r = engine.execute(ir, scenario={})
        diag = r.get("diagnostics", {})
        self.assertTrue(diag.get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN))
        self.assertEqual(diag.get(DiagnosticsKey.EXECUTION_PATH_REASON), "workflowir_direct")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Regression: non-behavior missions unaffected
# ══════════════════════════════════════════════════════════════════════════════

class TestNonBehaviorMissionsRegression(unittest.TestCase):
    """Non-behavior missions must be fully unaffected by Circle 1 changes."""

    @classmethod
    def setUpClass(cls):
        cls._orig = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig

    def _engine(self) -> RuntimeEngine:
        return RuntimeEngine()

    def test_exec_graph_no_behavior_status_success(self):
        r = self._engine().execute(_simple_exec_graph(1), scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_exec_graph_two_nodes_both_executed(self):
        """Both nodes must be visited; exec_graph nodes without logic_node
        don't write to results — verify via diagnostics.executed_count."""
        r = self._engine().execute(_simple_exec_graph(2), scenario={})
        executed = r.get("diagnostics", {}).get(DiagnosticsKey.EXECUTED_COUNT, 0)
        self.assertEqual(executed, 2,
                         f"Expected executed_count=2, got {executed}")

    def test_exec_graph_behavior_enabled_run_false(self):
        r = self._engine().execute(_simple_exec_graph(1), scenario={})
        self.assertFalse(r["diagnostics"].get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN))

    def test_exec_graph_execution_path_reason(self):
        r = self._engine().execute(_simple_exec_graph(1), scenario={})
        self.assertEqual(r["diagnostics"].get(DiagnosticsKey.EXECUTION_PATH_REASON),
                         "exec_graph_compat")

    def test_workflowir_simple_status_success(self):
        r = self._engine().execute(_workflowir_simple(2), scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_workflowir_behavior_enabled_run_true(self):
        r = self._engine().execute(_workflowir_simple(2), scenario={})
        self.assertTrue(r["diagnostics"].get(DiagnosticsKey.BEHAVIOR_ENABLED_RUN))

    def test_workflowir_execution_path_reason(self):
        r = self._engine().execute(_workflowir_simple(2), scenario={})
        self.assertEqual(r["diagnostics"].get(DiagnosticsKey.EXECUTION_PATH_REASON),
                         "workflowir_direct")

    def test_exec_graph_ui_behavior_path_reason_forwarded(self):
        """scenario._behavior_path_reason tag is forwarded to diagnostics."""
        scenario = {"_behavior_path_reason": "workflowir_behavior_enabled"}
        r = self._engine().execute(_simple_exec_graph(1), scenario=scenario)
        self.assertEqual(r["diagnostics"].get(DiagnosticsKey.EXECUTION_PATH_REASON),
                         "workflowir_behavior_enabled")


if __name__ == "__main__":
    unittest.main()
