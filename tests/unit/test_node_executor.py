#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/runtime/node_executor.py.

STEP-06: covers node failure propagation, loop limit, abort/stop semantics,
         flow-aware branching, migration switch (use_flow_aware_execution).

Isolation strategy
------------------
All sys.modules mutations are confined to setUpClass / tearDownClass pairs.
No module-level sys.modules injection is performed, so this file can be
discovered alongside other test modules without polluting their imports.
The ``nodes`` registry stub is the only sys.modules key modified here;
it does NOT affect bin or any Qt-dependent module.
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.node_executor import NodeExecutor  # noqa: E402


# ── Fake node registry helpers ─────────────────────────────────────────────
# Prepare the stub module and registry dict at module level (no injection yet).
# Injection into sys.modules happens only inside setUpClass / tearDownClass.

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


def _install_nodes_stub(extra: dict | None = None) -> None:
    """Inject the stub registry into sys.modules["nodes"].

    Restores the stub with the node classes for this test file.
    Call inside setUpClass only.
    """
    _registry.clear()
    _registry.update({
        "echo": EchoNode,
        "pass": PassthroughNode,
        "boom": BoomNode,
        "abort": AbortNode,
        "if": IfNode,
        "while_inf": InfiniteWhileNode,
        "for_node": ForNode,
    })
    if extra:
        _registry.update(extra)
    sys.modules["nodes"] = _nodes_mod


# ── Shared node stubs ──────────────────────────────────────────────────────

class EchoNode:
    """Returns its 'value' param as output."""
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"value": self._params.get("value", "echo")}


class PassthroughNode:
    """Passes through; records execution."""
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): return {"done": True}


class BoomNode:
    """Always raises a generic RuntimeError."""
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): raise RuntimeError("bang")


class AbortNode:
    """Raises AbortWorkflow RuntimeError."""
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): raise RuntimeError("AbortWorkflow: test_abort")


class IfNode:
    """Minimal if-branch node."""
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs):
        cond = inputs.get("condition", False)
        if cond:
            return {"out_if": True, "out_else": None}
        return {"out_if": None, "out_else": True}


class InfiniteWhileNode:
    """Always signals loop_body (never ends on its own)."""
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): return {"loop_body": True, "loop_end": None}


class ForNode:
    """Signals for-loop body (engine drives iterations via params)."""
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): return {"loop_body": True, "loop_end": None}


# ── Helper ─────────────────────────────────────────────────────────────────

def _ex(**kwargs) -> NodeExecutor:
    ex = NodeExecutor()
    ex.max_loop_iterations = kwargs.get("max_loop_iterations", 100)
    return ex


# ══════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════

class TestNodeFailurePropagation(unittest.TestCase):
    """Single-node failure is recorded; execution continues for other nodes."""

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

    def test_unknown_type_skipped_not_raised(self):
        ex = _ex()
        ex.add_node("n1", "totally_unknown_xyz", {})
        r = ex.execute()
        self.assertIn("n1", r)
        self.assertEqual(r["n1"]["status"], "skipped")
        self.assertIn("unknown_type", r["n1"]["reason"])

    def test_runtime_error_recorded_in_results(self):
        ex = _ex()
        ex.add_node("boom", "boom", {})
        r = ex.execute()
        self.assertIn("boom", r)
        self.assertIn("error", r["boom"])

    def test_failure_does_not_prevent_independent_node_execution(self):
        ex = _ex()
        ex.add_node("boom", "boom", {})
        ex.add_node("ok", "pass", {})
        r = ex.execute()
        self.assertIn("error", r.get("boom", {}))
        self.assertIn("ok", r)
        self.assertNotIn("error", r.get("ok", {}))

    def test_downstream_of_boom_not_executed(self):
        ex = _ex()
        ex.add_node("boom", "boom", {})
        ex.add_node("after", "pass", {})
        ex.add_connection("boom", "flow_out", "after", "flow_in", "flow")
        r = ex.execute()
        self.assertIn("error", r.get("boom", {}))
        self.assertNotIn("after", r)


class TestLoopLimit(unittest.TestCase):
    """max_loop_iterations caps while loops; _loop_limit_exceeded is set."""

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

    def test_while_loop_capped(self):
        counter = []

        class CountBody:
            def __init__(self, nid): self.nid = nid
            def set_parameter(self, k, v): pass
            def execute(self, inputs): counter.append(1); return {"done": True}

        _registry["_count"] = CountBody
        ex = _ex(max_loop_iterations=3)
        ex.add_node("wh", "while_inf", {"params": {"loop_type": {"value": "while"}}})
        ex.add_node("body", "_count", {})
        ex.add_connection("wh", "loop_body", "body", "flow_in", "flow")
        ex.execute()
        self.assertTrue(ex._loop_limit_exceeded)
        self.assertEqual(len(counter), 3)

    def test_loop_limit_zero_means_no_body_iterations(self):
        counter = []

        class CountBody:
            def __init__(self, nid): self.nid = nid
            def set_parameter(self, k, v): pass
            def execute(self, inputs): counter.append(1); return {"done": True}

        _registry["_count0"] = CountBody
        ex = _ex(max_loop_iterations=0)
        ex.add_node("wh", "while_inf", {"params": {"loop_type": {"value": "while"}}})
        ex.add_node("body", "_count0", {})
        ex.add_connection("wh", "loop_body", "body", "flow_in", "flow")
        ex.execute()
        self.assertEqual(len(counter), 0)

    def test_for_loop_exact_iterations(self):
        counter = []

        class CountBody:
            def __init__(self, nid): self.nid = nid
            def set_parameter(self, k, v): pass
            def execute(self, inputs): counter.append(1); return {"done": True}

        _registry["_for_count"] = CountBody
        ex = _ex()
        ex.add_node("fl", "for_node", {"params": {
            "loop_type": {"value": "for"},
            "for_start": {"value": 0},
            "for_end":   {"value": 5},
            "for_step":  {"value": 1},
        }})
        ex.add_node("body", "_for_count", {})
        ex.add_connection("fl", "loop_body", "body", "flow_in", "flow")
        ex.execute()
        self.assertFalse(ex._loop_limit_exceeded)
        self.assertEqual(len(counter), 5)


class TestAbortSemantics(unittest.TestCase):
    """AbortWorkflow stops traversal immediately."""

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

    def test_abort_sets_flag(self):
        ex = _ex()
        ex.add_node("a", "abort", {})
        ex.execute()
        self.assertTrue(ex._abort)
        self.assertIn("AbortWorkflow", ex._abort_reason)

    def test_downstream_not_executed_after_abort(self):
        ex = _ex()
        ex.add_node("a", "abort", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        r = ex.execute()
        self.assertTrue(ex._abort)
        self.assertNotIn("b", r)

    def test_abort_node_result_has_error(self):
        ex = _ex()
        ex.add_node("a", "abort", {})
        r = ex.execute()
        self.assertIn("error", r.get("a", {}))

    def test_clear_resets_abort_state(self):
        ex = _ex()
        ex.add_node("a", "abort", {})
        ex.execute()
        self.assertTrue(ex._abort)
        ex.clear()
        self.assertFalse(ex._abort)
        self.assertEqual(ex._abort_reason, "")


class TestConditionalBranching(unittest.TestCase):
    """If-node only visits the active branch."""

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

    def _run_if(self, condition_value: bool):
        ex = _ex()
        ex.add_node("src", "echo", {"params": {"value": {"value": condition_value}}})
        ex.add_node("if1", "if", {"params": {}})
        ex.add_node("t_branch", "pass", {})
        ex.add_node("f_branch", "pass", {})
        ex.add_connection("src", "value", "if1", "condition", "data")
        ex.add_connection("if1", "out_if",   "t_branch", "flow_in", "flow")
        ex.add_connection("if1", "out_else",  "f_branch", "flow_in", "flow")
        return ex.execute()

    def test_if_true_visits_true_branch_only(self):
        r = self._run_if(True)
        self.assertIn("t_branch", r)
        self.assertNotIn("f_branch", r)

    def test_if_false_visits_false_branch_only(self):
        r = self._run_if(False)
        self.assertNotIn("t_branch", r)
        self.assertIn("f_branch", r)


class TestMigrationSwitch(unittest.TestCase):
    """use_flow_aware_execution=False falls back to topological sort."""

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

    def test_compat_path_executes_all_connected_nodes(self):
        ex = _ex()
        ex.use_flow_aware_execution = False
        ex.add_node("a", "pass", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        r = ex.execute()
        self.assertIn("a", r)
        self.assertIn("b", r)

    def test_new_path_executes_all_connected_nodes(self):
        ex = _ex()
        ex.use_flow_aware_execution = True
        ex.add_node("a", "pass", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        r = ex.execute()
        self.assertIn("a", r)
        self.assertIn("b", r)

    def test_switch_preserved_across_clear(self):
        ex = _ex()
        ex.use_flow_aware_execution = False
        ex.add_node("a", "pass", {})
        ex.execute()
        ex.clear()
        self.assertFalse(ex.use_flow_aware_execution)

    def test_last_executed_count_reset_on_execute(self):
        ex = _ex()
        ex.add_node("a", "pass", {})
        ex.execute()
        count1 = ex._last_executed_count
        ex.clear()
        ex.add_node("a", "pass", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        ex.execute()
        count2 = ex._last_executed_count
        self.assertGreater(count2, count1)


class TestExecutedCount(unittest.TestCase):
    """_last_executed_count tracks actual nodes visited."""

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

    def test_single_node(self):
        ex = _ex()
        ex.add_node("a", "pass", {})
        ex.execute()
        self.assertEqual(ex._last_executed_count, 1)

    def test_two_chained_nodes(self):
        ex = _ex()
        ex.add_node("a", "pass", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        ex.execute()
        self.assertEqual(ex._last_executed_count, 2)

    def test_abort_stops_count(self):
        ex = _ex()
        ex.add_node("a", "abort", {})
        ex.add_node("b", "pass", {})
        ex.add_connection("a", "out", "b", "in", "flow")
        ex.execute()
        self.assertEqual(ex._last_executed_count, 1)


if __name__ == "__main__":
    unittest.main()
