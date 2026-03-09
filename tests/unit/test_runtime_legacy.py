#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests: WorkflowIR compat (topological) path vs new (flow-aware) path.

STEP-06: proves that switching use_flow_aware_execution=False (legacy compat)
does not break the unified output contract and produces equivalent results
for simple linear graphs where topological order == flow order.

Isolation strategy
------------------
No module-level sys.modules injection.  The ``nodes`` registry stub is
installed/removed in setUpClass / tearDownClass only.
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.node_executor import NodeExecutor           # noqa: E402
from system.runtime.contracts import RuntimeStatus              # noqa: E402
from system.runtime.runtime_engine import RuntimeEngine         # noqa: E402
from system.ir.workflow_ir import IRNode, NodeKind, WorkflowIR  # noqa: E402


# ── Fake node registry ─────────────────────────────────────────────────────
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleNode:
    """Returns a status output for the node."""
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "node": self.nid}


def _install_nodes_stub() -> None:
    _registry.clear()
    _registry["simple"] = SimpleNode
    sys.modules["nodes"] = _nodes_mod


# ── Helpers ────────────────────────────────────────────────────────────────

def _linear_executor(n_nodes: int, flow_aware: bool) -> NodeExecutor:
    ex = NodeExecutor()
    ex.use_flow_aware_execution = flow_aware
    prev = None
    for i in range(n_nodes):
        nid = f"n{i}"
        ex.add_node(nid, "simple", {})
        if prev is not None:
            ex.add_connection(prev, "out", nid, "in", "flow")
        prev = nid
    return ex


def _linear_ir(n_nodes: int) -> WorkflowIR:
    nodes = [
        IRNode(id=f"n{i}", kind=NodeKind.ACTION, schema_id="simple", params={})
        for i in range(n_nodes)
    ]
    return WorkflowIR(nodes=nodes, edges=[])


# ══════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyCompatPath(unittest.TestCase):
    """Compat (topological) path produces same result shape as new (flow-aware) path."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_nodes_stub()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def test_single_node_compat(self):
        ex = _linear_executor(1, flow_aware=False)
        r = ex.execute()
        self.assertIn("n0", r)
        self.assertNotIn("error", r.get("n0", {}))
        self.assertEqual(r["n0"].get("status"), "ok")

    def test_linear_two_nodes_compat_results(self):
        ex = _linear_executor(2, flow_aware=False)
        r = ex.execute()
        self.assertIn("n0", r)
        self.assertIn("n1", r)
        self.assertNotIn("error", r.get("n0", {}))
        self.assertNotIn("error", r.get("n1", {}))

    def test_linear_two_nodes_new_path_results(self):
        ex = _linear_executor(2, flow_aware=True)
        r = ex.execute()
        self.assertIn("n0", r)
        self.assertIn("n1", r)
        self.assertNotIn("error", r.get("n0", {}))
        self.assertNotIn("error", r.get("n1", {}))

    def test_compat_and_new_both_set_executed_count(self):
        for flow_aware in (False, True):
            with self.subTest(flow_aware=flow_aware):
                ex = _linear_executor(3, flow_aware=flow_aware)
                ex.execute()
                self.assertEqual(ex._last_executed_count, 3,
                                 f"flow_aware={flow_aware}: expected 3 nodes executed")

    def test_compat_path_no_loop_limit_exceeded(self):
        ex = _linear_executor(2, flow_aware=False)
        ex.execute()
        self.assertFalse(ex._loop_limit_exceeded)

    def test_clear_resets_state_for_both_paths(self):
        for flow_aware in (False, True):
            with self.subTest(flow_aware=flow_aware):
                ex = _linear_executor(2, flow_aware=flow_aware)
                ex.execute()
                self.assertGreater(ex._last_executed_count, 0)
                ex.clear()
                self.assertEqual(ex._last_executed_count, 0)
                self.assertFalse(ex._abort)
                self.assertFalse(ex._loop_limit_exceeded)


class TestRuntimeEngineLegacyIntegration(unittest.TestCase):
    """RuntimeEngine WorkflowIR path produces valid contract dict."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_nodes_stub()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def test_engine_workflow_ir_produces_all_contract_fields(self):
        ir = _linear_ir(2)
        r = RuntimeEngine().execute(ir, scenario={})
        required = {"status", "reason", "task_id", "node_count",
                    "results", "metrics", "diagnostics"}
        missing = required - r.keys()
        self.assertFalse(missing, f"missing fields: {missing}")

    def test_engine_workflow_ir_success_status(self):
        ir = _linear_ir(2)
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_engine_workflow_ir_executed_count_matches_node_count(self):
        n = 3
        ir = _linear_ir(n)
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertEqual(r["node_count"], n)
        self.assertEqual(r["diagnostics"]["executed_count"], n)

    def test_engine_results_keyed_by_node_id(self):
        ir = _linear_ir(2)
        r = RuntimeEngine().execute(ir, scenario={})
        for nid in ("n0", "n1"):
            self.assertIn(nid, r["results"])


if __name__ == "__main__":
    unittest.main()
