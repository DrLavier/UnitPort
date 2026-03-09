#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 Test Matrix — STAGE-06.

This file is the single-document evidence of Phase 2 completeness.
It is organised in three explicit sections mirroring the STAGE-06 task list:

  SECTION A — Unit coverage
    BehaviorArtifact contract, BehaviorErrorCode constants,
    BehaviorResolveResult factories, BehaviorCompilerBridge core operations.

  SECTION B — Integration coverage
    Full compile → register → mission-execute chain;
    runtime result propagation with behavior failures;
    trace_id end-to-end correlation.

  SECTION C — Regression coverage
    Legacy non-behavior workflows unchanged;
    Phase 1 runtime contract fields stable;
    new Phase 2 DiagnosticsKey fields present on success/failed results;
    new fields absent on blocked results (as designed).

Isolation
---------
Node registry stub installed in setUpClass / tearDownClass.
No Qt, no SDK.
"""

import sys
import types
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.runtime_engine import RuntimeEngine                   # noqa: E402
from system.runtime.contracts import DiagnosticsKey, ExecutionPath, RuntimeStatus  # noqa: E402
from system.behavior.behavior_artifact import (                            # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
    BehaviorResolveResult,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge  # noqa: E402
from system.ir.workflow_ir import (                                           # noqa: E402
    EdgeType, IREdge, IRNode, IRParam, NodeKind, WorkflowIR,
)


# ---------------------------------------------------------------------------
# Shared node registry stub
# ---------------------------------------------------------------------------
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "out": self.nid}


class SubNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "sub_ok", "out": self.nid}


def _install_registry():
    _registry.clear()
    _registry["simple"] = SimpleNode
    _registry["sub_action"] = SubNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Shared IR builders
# ---------------------------------------------------------------------------

def _subgraph_ir(node_id: str = "s0") -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id=node_id, kind=NodeKind.ACTION, schema_id="sub_action", params={})],
        edges=[],
    )


def _behavior_node(behavior_ref: str, node_id: str = "b0") -> IRNode:
    return IRNode(
        id=node_id,
        kind=NodeKind.ACTION,
        schema_id="behavior",
        params={
            "behavior_ref": IRParam(
                name="behavior_ref", value=behavior_ref, param_type="string"
            )
        },
    )


def _simple_node(node_id: str = "a0") -> IRNode:
    return IRNode(id=node_id, kind=NodeKind.ACTION, schema_id="simple", params={})


def _valid_artifact(ref: str) -> BehaviorArtifact:
    return BehaviorArtifact.create(behavior_ref=ref, behavior_ir=_subgraph_ir())


def _invalid_artifact(ref: str) -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=ref,
        behavior_ir=WorkflowIR(nodes=[], edges=[]),
        diagnostics=[BehaviorDiagnostic.error("E1", "compile failure")],
    )


# ===========================================================================
# SECTION A — Unit coverage
# ===========================================================================

class TestBehaviorArtifactContract(unittest.TestCase):
    """A.1 — BehaviorArtifact data contract."""

    def test_create_generates_unique_artifact_id(self):
        a1 = _valid_artifact("ref1")
        a2 = _valid_artifact("ref1")
        self.assertNotEqual(a1.artifact_id, a2.artifact_id)

    def test_create_preserves_behavior_ref(self):
        a = _valid_artifact("my_ref")
        self.assertEqual(a.behavior_ref, "my_ref")

    def test_create_uses_supplied_trace_id(self):
        a = BehaviorArtifact.create(
            behavior_ref="r", behavior_ir=_subgraph_ir(), trace_id="custom-tid"
        )
        self.assertEqual(a.trace_id, "custom-tid")

    def test_create_uses_supplied_source_hash(self):
        a = BehaviorArtifact.create(
            behavior_ref="r", behavior_ir=_subgraph_ir(), source_hash="abc123"
        )
        self.assertEqual(a.source_hash, "abc123")

    def test_is_valid_true_when_no_error_diagnostics(self):
        a = _valid_artifact("ok_ref")
        self.assertTrue(a.is_valid)

    def test_is_valid_false_when_error_present(self):
        a = _invalid_artifact("bad_ref")
        self.assertFalse(a.is_valid)

    def test_error_count(self):
        a = _invalid_artifact("bad_ref2")
        self.assertEqual(a.error_count, 1)

    def test_warning_count(self):
        a = BehaviorArtifact.create(
            behavior_ref="w",
            behavior_ir=_subgraph_ir(),
            diagnostics=[BehaviorDiagnostic.warning("W1", "warn")],
        )
        self.assertEqual(a.warning_count, 1)
        self.assertTrue(a.is_valid)   # warnings don't invalidate artifact

    def test_to_dict_required_keys(self):
        d = _valid_artifact("ref3").to_dict()
        for key in ("artifact_id", "behavior_ref", "behavior_ir_summary",
                    "diagnostics", "trace_id", "compiled_at", "source_hash", "is_valid"):
            self.assertIn(key, d)

    def test_to_dict_ir_summary_node_count(self):
        a = _valid_artifact("ref4")
        self.assertEqual(a.to_dict()["behavior_ir_summary"]["node_count"], 1)


class TestBehaviorErrorCodeContract(unittest.TestCase):
    """A.2 — BehaviorErrorCode constants are non-empty, distinct strings."""

    _CODES = [
        BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
        BehaviorErrorCode.ARTIFACT_INVALID,
        BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
        BehaviorErrorCode.SUBGRAPH_ABORTED,
        BehaviorErrorCode.BEHAVIOR_COMPILE_ERROR,
    ]

    def test_all_codes_are_strings(self):
        for code in self._CODES:
            self.assertIsInstance(code, str)

    def test_all_codes_are_nonempty(self):
        for code in self._CODES:
            self.assertTrue(code, f"empty code: {code!r}")

    def test_all_codes_are_distinct(self):
        self.assertEqual(len(set(self._CODES)), len(self._CODES))

    def test_ref_not_found_value(self):
        self.assertEqual(
            BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND, "behavior_ref_not_found"
        )

    def test_artifact_invalid_value(self):
        self.assertEqual(BehaviorErrorCode.ARTIFACT_INVALID, "artifact_invalid")


class TestBehaviorResolveResultContract(unittest.TestCase):
    """A.3 — BehaviorResolveResult factory invariants."""

    def test_success_ok_is_true(self):
        a = _valid_artifact("ref")
        r = BehaviorResolveResult.success(a, "tid")
        self.assertTrue(r.ok)

    def test_success_reason_empty(self):
        a = _valid_artifact("ref")
        r = BehaviorResolveResult.success(a, "tid")
        self.assertEqual(r.reason, "")

    def test_success_diagnostics_empty(self):
        a = _valid_artifact("ref")
        r = BehaviorResolveResult.success(a, "tid")
        self.assertEqual(r.diagnostics, [])

    def test_missing_ok_is_false(self):
        r = BehaviorResolveResult.missing("ghost", "tid")
        self.assertFalse(r.ok)

    def test_missing_reason_code(self):
        r = BehaviorResolveResult.missing("ghost", "tid")
        self.assertEqual(r.reason, BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_missing_artifact_is_none(self):
        r = BehaviorResolveResult.missing("ghost", "tid")
        self.assertIsNone(r.artifact)

    def test_invalid_ok_is_false(self):
        a = _invalid_artifact("bad")
        r = BehaviorResolveResult.invalid(a, "tid")
        self.assertFalse(r.ok)

    def test_invalid_reason_code(self):
        a = _invalid_artifact("bad2")
        r = BehaviorResolveResult.invalid(a, "tid")
        self.assertEqual(r.reason, BehaviorErrorCode.ARTIFACT_INVALID)

    def test_invalid_includes_compile_diagnostics(self):
        a = _invalid_artifact("bad3")
        r = BehaviorResolveResult.invalid(a, "tid")
        # Resolution-level error + artifact's own compile error
        self.assertGreaterEqual(len(r.diagnostics), 2)


# ===========================================================================
# SECTION B — Integration coverage
# ===========================================================================

class TestPhase2IntegrationChain(unittest.TestCase):
    """B.1 — Compile → register → execute → RuntimeResult chain."""

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

    def _run(self, ref: str, bridge=None) -> dict:
        if bridge is None:
            bridge = BehaviorCompilerBridge()
            bridge.register(_valid_artifact(ref))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node(ref)], edges=[])
        return engine.execute(ir, scenario={})

    def test_behavior_node_success_status(self):
        self.assertEqual(self._run("stand")["status"], RuntimeStatus.SUCCESS)

    def test_behavior_node_in_results(self):
        r = self._run("stand2")
        self.assertIn("b0", r["results"])

    def test_behavior_node_result_status_success(self):
        r = self._run("walk")
        self.assertEqual(r["results"]["b0"].get("status"), "success")

    def test_behavior_node_result_has_trace_id(self):
        r = self._run("walk2")
        self.assertIn("trace_id", r["results"]["b0"])

    def test_behavior_node_result_trace_id_equals_mission_trace_id(self):
        r = self._run("run")
        mission_tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        node_tid = r["results"]["b0"]["trace_id"]
        self.assertEqual(mission_tid, node_tid)

    def test_behavior_trace_ids_in_diagnostics(self):
        r = self._run("jump")
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        self.assertIn("b0", tids)

    def test_behavior_diagnostics_empty_on_success(self):
        r = self._run("sit")
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])

    def test_failed_nodes_empty_on_success(self):
        r = self._run("crouch")
        self.assertEqual(r["diagnostics"][DiagnosticsKey.FAILED_NODES], [])

    def test_missing_ref_status_failed(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.FAILED)

    def test_missing_ref_error_code_in_node_result(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost2")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(
            r["results"]["b0"]["error"],
            BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
        )

    def test_missing_ref_diagnostics_aggregated(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost3")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertGreater(
            len(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]), 0
        )

    def test_invalid_artifact_error_code(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_invalid_artifact("broken"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node("broken")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(
            r["results"]["b0"]["error"],
            BehaviorErrorCode.ARTIFACT_INVALID,
        )

    def test_no_bridge_behavior_node_skipped_not_failed(self):
        engine = RuntimeEngine()  # behavior_bridge=None
        ir = WorkflowIR(nodes=[_behavior_node("stand")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)
        self.assertEqual(r["results"]["b0"].get("status"), "skipped")

    def test_mixed_mission_both_nodes_executed(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("wave"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(
            nodes=[_simple_node("a0"), _behavior_node("wave", "b0")],
            edges=[
                IREdge(
                    from_node="a0", from_port="flow_out",
                    to_node="b0", to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                )
            ],
        )
        r = engine.execute(ir, scenario={})
        self.assertIn("a0", r["results"])
        self.assertIn("b0", r["results"])
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_compile_source_roundtrip(self):
        """BehaviorCompilerBridge.compile() → register → execute → success."""
        bridge = BehaviorCompilerBridge()
        # Empty source compiles to valid artifact (no semantic errors).
        artifact = bridge.compile("", "auto_ref")
        self.assertTrue(artifact.is_valid)
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node("auto_ref")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)


class TestPhase2TraceCorrelation(unittest.TestCase):
    """B.2 — Trace ID end-to-end correlation."""

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

    def test_mission_trace_id_is_valid_uuid(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("tid_ref"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        r = engine.execute(WorkflowIR(nodes=[_behavior_node("tid_ref")], edges=[]), scenario={})
        tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        uuid.UUID(tid)   # raises ValueError if not a valid UUID

    def test_behavior_trace_id_matches_mission_trace_id(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("corr_ref"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        r = engine.execute(WorkflowIR(nodes=[_behavior_node("corr_ref")], edges=[]), scenario={})
        mission_tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        node_tid = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS].get("b0")
        self.assertEqual(mission_tid, node_tid)

    def test_compile_trace_id_propagates_to_all_diagnostics(self):
        """All diagnostics in a compiled artifact share the artifact trace_id."""
        bridge = BehaviorCompilerBridge()
        tid = "fixed-trace-id-for-test"
        artifact = bridge.compile("", "trace_ref", trace_id=tid)
        self.assertEqual(artifact.trace_id, tid)
        for d in artifact.diagnostics:
            self.assertEqual(d.trace_id, tid)

    def test_two_executions_have_different_mission_trace_ids(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("two_ref"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node("two_ref")], edges=[])
        r1 = engine.execute(ir, scenario={})
        r2 = engine.execute(ir, scenario={})
        t1 = r1["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        t2 = r2["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        self.assertNotEqual(t1, t2)

    def test_failure_result_trace_id_in_behavior_trace_ids(self):
        """Even on failure, behavior_trace_ids captures the trace for the node."""
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost_trace")], edges=[])
        r = engine.execute(ir, scenario={})
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        self.assertIn("b0", tids)

    def test_failure_diagnostic_trace_id_matches_mission(self):
        """Diagnostics aggregated on failure carry the mission trace_id."""
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost_diag")], edges=[])
        r = engine.execute(ir, scenario={})
        mission_tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        diags = r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]
        self.assertTrue(diags)
        for d in diags:
            self.assertEqual(
                d["trace_id"], mission_tid,
                "diagnostic trace_id must match mission_trace_id"
            )


# ===========================================================================
# SECTION C — Regression coverage
# ===========================================================================

class TestPhase1NonBehaviorStability(unittest.TestCase):
    """C.1 — Non-behavior WorkflowIR missions unchanged by Phase 2."""

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

    def _run_simple(self) -> dict:
        engine = RuntimeEngine()
        ir = WorkflowIR(nodes=[_simple_node("a0")], edges=[])
        return engine.execute(ir, scenario={})

    def test_non_behavior_mission_succeeds(self):
        self.assertEqual(self._run_simple()["status"], RuntimeStatus.SUCCESS)

    def test_non_behavior_node_in_results(self):
        r = self._run_simple()
        self.assertIn("a0", r["results"])

    def test_non_behavior_result_has_no_error(self):
        r = self._run_simple()
        self.assertNotIn("error", r["results"]["a0"])

    def test_phase1_top_level_fields_present(self):
        r = self._run_simple()
        for key in ("status", "reason", "task_id", "node_count", "results",
                    "metrics", "diagnostics"):
            self.assertIn(key, r)

    def test_phase1_diagnostics_keys_present(self):
        r = self._run_simple()
        for key in (DiagnosticsKey.PATH, DiagnosticsKey.EXECUTED_COUNT,
                    DiagnosticsKey.FAILED_NODES, DiagnosticsKey.HAS_ACTION):
            self.assertIn(key, r["diagnostics"])

    def test_failed_nodes_empty_for_successful_non_behavior_mission(self):
        r = self._run_simple()
        self.assertEqual(r["diagnostics"][DiagnosticsKey.FAILED_NODES], [])

    def test_task_id_nonempty(self):
        r = self._run_simple()
        self.assertNotEqual(r["task_id"], "")

    def test_node_count_correct(self):
        r = self._run_simple()
        self.assertEqual(r["node_count"], 1)


class TestPhase2FieldsOnAllNonBlockedResults(unittest.TestCase):
    """C.2 — New Phase 2 DiagnosticsKey fields present on success and failed results."""

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

    _NEW_KEYS = (
        DiagnosticsKey.MISSION_TRACE_ID,
        DiagnosticsKey.BEHAVIOR_TRACE_IDS,
        DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
    )

    def test_new_keys_present_on_success_result(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        for key in self._NEW_KEYS:
            self.assertIn(key, r["diagnostics"], f"missing key: {key}")

    def test_new_keys_present_on_failed_result(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost_x")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["status"], RuntimeStatus.FAILED)
        for key in self._NEW_KEYS:
            self.assertIn(key, r["diagnostics"], f"missing key on failed: {key}")

    def test_mission_trace_id_is_str(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        self.assertIsInstance(r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID], str)

    def test_behavior_trace_ids_is_dict(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        self.assertIsInstance(r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS], dict)

    def test_behavior_diagnostics_is_list(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        self.assertIsInstance(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], list)

    def test_behavior_trace_ids_empty_for_non_behavior_mission(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS], {})

    def test_behavior_diagnostics_empty_for_non_behavior_mission(self):
        engine = RuntimeEngine()
        r = engine.execute(WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={})
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])

    def test_mission_trace_id_nonempty_on_failed_result(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node("ghost_y")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertTrue(r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID])


class TestBlockedPathRegression(unittest.TestCase):
    """C.3 — Blocked results keep Phase 1 contract; no behavior tracing keys."""

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

    def _blocked_engine(self):
        eng = RuntimeEngine()
        eng._compile_guard.check = lambda ir: {"ok": False, "reason": "forbidden"}
        return eng

    def test_blocked_result_status(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertEqual(r["status"], RuntimeStatus.BLOCKED)

    def test_blocked_result_has_phase_field(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertEqual(r.get("phase"), "compile")

    def test_blocked_result_task_id_empty(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertEqual(r["task_id"], "")

    def test_blocked_result_node_count_zero(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertEqual(r["node_count"], 0)

    def test_blocked_result_no_mission_trace_id(self):
        """blocked() factory intentionally omits behavior tracing keys."""
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertNotIn(DiagnosticsKey.MISSION_TRACE_ID, r["diagnostics"])

    def test_blocked_result_no_behavior_trace_ids(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertNotIn(DiagnosticsKey.BEHAVIOR_TRACE_IDS, r["diagnostics"])

    def test_blocked_result_no_behavior_diagnostics(self):
        r = self._blocked_engine().execute(
            WorkflowIR(nodes=[_simple_node()], edges=[]), scenario={}
        )
        self.assertNotIn(DiagnosticsKey.BEHAVIOR_DIAGNOSTICS, r["diagnostics"])


if __name__ == "__main__":
    unittest.main()
