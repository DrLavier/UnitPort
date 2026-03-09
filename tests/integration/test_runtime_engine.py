#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for RuntimeEngine — unified output contract.

STEP-06: verifies that both execution paths (exec_graph / WorkflowIR) produce
homogeneous RuntimeResult dicts with the same field set and consistent status
semantics; also covers blocked paths and the migration switch.

Isolation strategy
------------------
No module-level sys.modules injection.  The ``nodes`` registry stub is
installed/removed in setUpClass / tearDownClass only, so this file is safe
for unittest discover alongside Qt-dependent test modules.
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.runtime_engine import RuntimeEngine       # noqa: E402
from system.runtime.contracts import (                        # noqa: E402
    DiagnosticsKey, ExecutionPath, RuntimeStatus,
)
from system.ir.workflow_ir import (                           # noqa: E402
    IRNode, NodeKind, WorkflowIR,
)


# ── Fake node registry ─────────────────────────────────────────────────────
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleActionNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "output": f"{self.nid}_done"}


class BoomNode:
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): raise RuntimeError("intentional_failure")


def _install_nodes_stub() -> None:
    _registry.clear()
    _registry["simple"] = SimpleActionNode
    _registry["boom"] = BoomNode
    sys.modules["nodes"] = _nodes_mod


# ── Shared helpers ─────────────────────────────────────────────────────────

_EXPECTED_TOPLEVEL = frozenset({
    "status", "reason", "task_id", "node_count", "results", "metrics", "diagnostics",
})
# Phase 1 keys + Phase 2 STAGE-05 behavior tracing keys.
# All five are present on success/failed results for both execution paths.
_EXPECTED_DIAG = frozenset({
    DiagnosticsKey.PATH,
    DiagnosticsKey.EXECUTED_COUNT,
    DiagnosticsKey.FAILED_NODES,
    DiagnosticsKey.HAS_ACTION,
    DiagnosticsKey.MISSION_TRACE_ID,
    DiagnosticsKey.BEHAVIOR_TRACE_IDS,
    DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
})


def _workflow_ir(*node_defs, edges=None) -> WorkflowIR:
    nodes = [
        IRNode(id=nid, kind=kind, schema_id=sid, params={})
        for nid, sid, kind in node_defs
    ]
    return WorkflowIR(nodes=nodes, edges=edges or [])


def _exec_graph(node_ids, entry=None, outgoing=None) -> dict:
    return {
        "nodes": {nid: {"type": "action_execution"} for nid in node_ids},
        "outgoing": outgoing or {nid: {"flow_out": []} for nid in node_ids},
        "incoming": {},
        "entry_nodes": entry or list(node_ids)[:1],
    }


# ══════════════════════════════════════════════════════════════════════════
# Test cases
# ══════════════════════════════════════════════════════════════════════════

class TestUnifiedOutputShape(unittest.TestCase):
    """Both paths return the same top-level field set."""

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

    def _assert_shape(self, result: dict, expected_path: str):
        missing = _EXPECTED_TOPLEVEL - result.keys()
        self.assertFalse(missing, f"missing top-level fields: {missing}")
        diag = result["diagnostics"]
        missing_diag = _EXPECTED_DIAG - diag.keys()
        self.assertFalse(missing_diag, f"missing diagnostics fields: {missing_diag}")
        self.assertEqual(diag[DiagnosticsKey.PATH], expected_path)

    def test_workflow_ir_shape(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        result = RuntimeEngine().execute(ir, scenario={})
        self._assert_shape(result, ExecutionPath.WORKFLOW_IR)

    def test_exec_graph_shape(self):
        eg = _exec_graph(["n1"])
        result = RuntimeEngine().execute(eg, scenario={})
        self._assert_shape(result, ExecutionPath.EXECUTION_GRAPH)


class TestSuccessSemantics(unittest.TestCase):

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

    def test_workflow_ir_success(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)
        self.assertEqual(r["reason"], "")
        self.assertNotEqual(r["task_id"], "")
        self.assertEqual(r["node_count"], 1)

    def test_exec_graph_success(self):
        eg = _exec_graph(["n1"])
        r = RuntimeEngine().execute(eg, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)
        self.assertNotEqual(r["task_id"], "")


class TestFailureSemantics(unittest.TestCase):

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

    def test_workflow_ir_node_failure(self):
        ir = _workflow_ir(("boom1", "boom", NodeKind.ACTION))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.FAILED)
        self.assertEqual(r["reason"], "node_execution_failed")
        self.assertIn("boom1", r["diagnostics"][DiagnosticsKey.FAILED_NODES])

    def test_failed_nodes_in_results(self):
        ir = _workflow_ir(("boom1", "boom", NodeKind.ACTION))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertIn("error", r["results"].get("boom1", {}))


class TestBlockedPaths(unittest.TestCase):
    """CompileGuard / ExecuteGuard blocks return phase + reason (legacy compat)."""

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

    def _blocked_compile_engine(self):
        eng = RuntimeEngine()
        eng._compile_guard.check = lambda ir: {"ok": False, "reason": "forbidden_ir"}
        return eng

    def _blocked_execute_engine(self):
        eng = RuntimeEngine()
        eng._execute_guard.check = lambda sc: {"ok": False, "reason": "no_sim"}
        return eng

    def test_compile_blocked_fields(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        r = self._blocked_compile_engine().execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.BLOCKED)
        self.assertEqual(r["phase"], "compile")
        self.assertEqual(r["reason"], "forbidden_ir")
        self.assertEqual(r["task_id"], "")
        self.assertEqual(r["node_count"], 0)

    def test_execute_blocked_fields(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        r = self._blocked_execute_engine().execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.BLOCKED)
        self.assertEqual(r["phase"], "execute")
        self.assertEqual(r["reason"], "no_sim")

    def test_no_phase_in_success(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertNotIn("phase", r)


class TestExecutedCountAlignment(unittest.TestCase):

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

    def test_workflow_ir_executed_count_equals_node_count(self):
        ir = _workflow_ir(
            ("n1", "simple", NodeKind.ACTION),
            ("n2", "simple", NodeKind.ACTION),
        )
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertEqual(r["diagnostics"][DiagnosticsKey.EXECUTED_COUNT], 2)
        self.assertEqual(r["node_count"], 2)

    def test_exec_graph_executed_count_is_int(self):
        eg = _exec_graph(["n1"])
        r = RuntimeEngine().execute(eg, scenario={})
        self.assertIsInstance(r["diagnostics"][DiagnosticsKey.EXECUTED_COUNT], int)


class TestHasActionAlignment(unittest.TestCase):

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

    def test_workflow_ir_has_action_true(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertTrue(r["diagnostics"][DiagnosticsKey.HAS_ACTION])

    def test_workflow_ir_has_action_false_for_start_node(self):
        ir = _workflow_ir(("n1", "simple", NodeKind.START))
        r = RuntimeEngine().execute(ir, scenario={})
        self.assertFalse(r["diagnostics"][DiagnosticsKey.HAS_ACTION])


class TestMigrationFlagAuditLog(unittest.TestCase):

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

    def test_migration_flags_recorded(self):
        logged = []
        ir = _workflow_ir(("n1", "simple", NodeKind.ACTION))
        eng = RuntimeEngine()
        eng._audit_logger.record = lambda event, data: logged.append((event, data))
        eng.execute(ir, scenario={})
        events = [e for e, _ in logged]
        self.assertIn("migration_flags", events)
        flag_data = next(d for e, d in logged if e == "migration_flags")
        self.assertIn("use_flow_aware_execution", flag_data)


if __name__ == "__main__":
    unittest.main()
