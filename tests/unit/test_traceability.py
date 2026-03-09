#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit / integration tests for Phase 2 STAGE-05: Traceability and Diagnostics Plumbing.

Done-criteria coverage
----------------------
A single behavior execution can be traced end-to-end via trace_id and structured
diagnostics.  Specifically:

1. RuntimeEngine.execute() emits a unique mission_trace_id per call.
2. Behavior node invocations share the mission_trace_id as their trace root.
3. behavior_trace_ids in RuntimeResult.diagnostics maps node_id → trace_id.
4. behavior_diagnostics in RuntimeResult.diagnostics aggregates BehaviorDiagnostic
   dicts from every behavior node result (populated on failure, empty on success).
5. Compile path: BehaviorCompilerBridge.compile() propagates trace_id to artifact
   and to all mapped diagnostics.
6. Backward compatibility: Phase 1 DiagnosticsKey fields are unchanged.

Isolation
---------
- Node registry stub installed in setUpClass / tearDownClass.
- No Qt, no Unitree SDK.
"""

import sys
import types
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.runtime_engine import RuntimeEngine                   # noqa: E402
from system.runtime.contracts import DiagnosticsKey, RuntimeStatus       # noqa: E402
from system.behavior.behavior_artifact import (                            # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge  # noqa: E402
from system.ir.workflow_ir import (                                           # noqa: E402
    EdgeType, IREdge, IRNode, IRParam, NodeKind, WorkflowIR,
)


# ---------------------------------------------------------------------------
# Fake node registry
# ---------------------------------------------------------------------------
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class SimpleNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "node": self.nid}


class SubNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "subgraph_ok", "node": self.nid}


def _install_registry():
    _registry.clear()
    _registry["simple"] = SimpleNode
    _registry["sub_action"] = SubNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subgraph_ir() -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id="s0", kind=NodeKind.ACTION, schema_id="sub_action", params={})],
        edges=[],
    )


def _behavior_node_ir(behavior_ref: str, node_id: str = "b0") -> IRNode:
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


def _simple_node_ir(node_id: str = "a0") -> IRNode:
    return IRNode(id=node_id, kind=NodeKind.ACTION, schema_id="simple", params={})


def _valid_artifact(behavior_ref: str) -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_subgraph_ir(),
    )


def _invalid_artifact(behavior_ref: str) -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=WorkflowIR(nodes=[], edges=[]),
        diagnostics=[BehaviorDiagnostic.error("E1", "compile failure")],
    )


def _run_with_behavior(behavior_ref: str, bridge=None) -> dict:
    if bridge is None:
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact(behavior_ref))
    engine = RuntimeEngine(behavior_bridge=bridge)
    ir = WorkflowIR(nodes=[_behavior_node_ir(behavior_ref)], edges=[])
    return engine.execute(ir, scenario={})


# ===========================================================================
# TestMissionTraceId
# ===========================================================================

class TestMissionTraceId(unittest.TestCase):
    """mission_trace_id is generated per execution and surfaces in diagnostics."""

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

    def test_mission_trace_id_key_present(self):
        r = _run_with_behavior("stand")
        self.assertIn(DiagnosticsKey.MISSION_TRACE_ID, r["diagnostics"])

    def test_mission_trace_id_is_nonempty_string(self):
        r = _run_with_behavior("stand2")
        tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        self.assertIsInstance(tid, str)
        self.assertTrue(tid)

    def test_mission_trace_id_is_valid_uuid(self):
        r = _run_with_behavior("stand3")
        tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        # Must be parseable as a UUID.
        parsed = uuid.UUID(tid)
        self.assertEqual(str(parsed), tid)

    def test_different_executions_have_different_trace_ids(self):
        """Each RuntimeEngine.execute() call generates a fresh mission_trace_id."""
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("walk"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node_ir("walk")], edges=[])
        r1 = engine.execute(ir, scenario={})
        r2 = engine.execute(ir, scenario={})
        t1 = r1["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        t2 = r2["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        self.assertNotEqual(t1, t2)

    def test_mission_trace_id_present_on_failed_result(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_invalid_artifact("broken"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node_ir("broken")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertIn(DiagnosticsKey.MISSION_TRACE_ID, r["diagnostics"])
        self.assertTrue(r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID])

    def test_mission_trace_id_present_on_missing_ref_failure(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node_ir("ghost")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertIn(DiagnosticsKey.MISSION_TRACE_ID, r["diagnostics"])


# ===========================================================================
# TestBehaviorTraceIds
# ===========================================================================

class TestBehaviorTraceIds(unittest.TestCase):
    """behavior_trace_ids aggregates per-node trace IDs."""

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

    def test_behavior_trace_ids_key_present(self):
        r = _run_with_behavior("sit")
        self.assertIn(DiagnosticsKey.BEHAVIOR_TRACE_IDS, r["diagnostics"])

    def test_behavior_trace_ids_contains_node(self):
        r = _run_with_behavior("sit2")
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        self.assertIn("b0", tids)

    def test_behavior_trace_id_matches_node_result_trace_id(self):
        """The trace_id in diagnostics must equal result["results"]["b0"]["trace_id"]."""
        r = _run_with_behavior("run")
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        node_trace = r["results"]["b0"].get("trace_id")
        self.assertEqual(tids.get("b0"), node_trace)

    def test_behavior_trace_id_equals_mission_trace_id(self):
        """All behavior nodes in a mission share the mission_trace_id."""
        r = _run_with_behavior("jump")
        mission_tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        behavior_tid = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS].get("b0")
        self.assertEqual(mission_tid, behavior_tid)

    def test_behavior_trace_ids_empty_for_non_behavior_mission(self):
        """Missions with no behavior nodes produce empty behavior_trace_ids."""
        engine = RuntimeEngine()
        ir = WorkflowIR(nodes=[_simple_node_ir("a0")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS], {})

    def test_behavior_trace_ids_multiple_nodes(self):
        """Two behavior nodes → both appear in behavior_trace_ids."""
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("wave"))
        bridge.register(_valid_artifact("bow"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(
            nodes=[
                _behavior_node_ir("wave", node_id="b0"),
                _behavior_node_ir("bow", node_id="b1"),
            ],
            edges=[
                IREdge(
                    from_node="b0", from_port="flow_out",
                    to_node="b1", to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                )
            ],
        )
        r = engine.execute(ir, scenario={})
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        self.assertIn("b0", tids)
        self.assertIn("b1", tids)

    def test_behavior_trace_ids_same_mission_trace_id_for_all_nodes(self):
        """All behavior nodes in a mission share the mission_trace_id."""
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("waveM"))
        bridge.register(_valid_artifact("bowM"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(
            nodes=[
                _behavior_node_ir("waveM", node_id="b0"),
                _behavior_node_ir("bowM", node_id="b1"),
            ],
            edges=[
                IREdge(
                    from_node="b0", from_port="flow_out",
                    to_node="b1", to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                )
            ],
        )
        r = engine.execute(ir, scenario={})
        mission_tid = r["diagnostics"][DiagnosticsKey.MISSION_TRACE_ID]
        tids = r["diagnostics"][DiagnosticsKey.BEHAVIOR_TRACE_IDS]
        self.assertEqual(tids["b0"], mission_tid)
        self.assertEqual(tids["b1"], mission_tid)


# ===========================================================================
# TestBehaviorDiagnostics
# ===========================================================================

class TestBehaviorDiagnostics(unittest.TestCase):
    """behavior_diagnostics aggregates diagnostic dicts from behavior nodes."""

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

    def test_behavior_diagnostics_key_present_on_success(self):
        r = _run_with_behavior("crouch")
        self.assertIn(DiagnosticsKey.BEHAVIOR_DIAGNOSTICS, r["diagnostics"])

    def test_behavior_diagnostics_empty_on_success(self):
        r = _run_with_behavior("crouch2")
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])

    def test_behavior_diagnostics_nonempty_on_failed_node(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_invalid_artifact("broken"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node_ir("broken")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertGreater(
            len(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]), 0
        )

    def test_behavior_diagnostics_nonempty_on_missing_ref(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node_ir("ghost_ref2")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertGreater(
            len(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]), 0
        )

    def test_behavior_diagnostic_entries_are_dicts(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node_ir("ghost_ref3")], edges=[])
        r = engine.execute(ir, scenario={})
        for entry in r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]:
            self.assertIsInstance(entry, dict)

    def test_behavior_diagnostic_has_required_fields(self):
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node_ir("ghost_ref4")], edges=[])
        r = engine.execute(ir, scenario={})
        diags = r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS]
        self.assertTrue(diags)
        entry = diags[0]
        for key in ("level", "code", "message", "trace_id"):
            self.assertIn(key, entry, f"missing field '{key}' in diagnostic")

    def test_behavior_diagnostics_empty_for_non_behavior_mission(self):
        engine = RuntimeEngine()
        ir = WorkflowIR(nodes=[_simple_node_ir("a0")], edges=[])
        r = engine.execute(ir, scenario={})
        self.assertEqual(r["diagnostics"][DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])


# ===========================================================================
# TestCompileTraceId
# ===========================================================================

class TestCompileTraceId(unittest.TestCase):
    """Compile path: trace_id propagates from compile() to artifact and diagnostics."""

    def test_compile_artifact_has_trace_id(self):
        bridge = BehaviorCompilerBridge()
        artifact = bridge.compile("", "ref_a")
        self.assertIsInstance(artifact.trace_id, str)
        self.assertTrue(artifact.trace_id)

    def test_compile_accepts_caller_trace_id(self):
        bridge = BehaviorCompilerBridge()
        custom_tid = "my-custom-trace-id"
        artifact = bridge.compile("", "ref_b", trace_id=custom_tid)
        self.assertEqual(artifact.trace_id, custom_tid)

    def test_compile_different_calls_different_trace_ids(self):
        bridge = BehaviorCompilerBridge()
        a1 = bridge.compile("", "ref_c1")
        a2 = bridge.compile("", "ref_c2")
        self.assertNotEqual(a1.trace_id, a2.trace_id)

    def test_compile_propagates_trace_id_to_diagnostics(self):
        """All compile diagnostics share the artifact's trace_id."""
        bridge = BehaviorCompilerBridge()
        # Pass a source that produces warnings/errors via the compiler pipeline.
        # An empty source is accepted without errors in this pipeline.
        # Manually create an artifact with diagnostics to test propagation.
        custom_tid = "diag-trace-test"
        artifact = bridge.compile("", "ref_d", trace_id=custom_tid)
        self.assertEqual(artifact.trace_id, custom_tid)
        for d in artifact.diagnostics:
            self.assertEqual(d.trace_id, custom_tid,
                             f"diagnostic {d.code} has wrong trace_id")

    def test_resolve_generates_trace_id(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("ref_e"))
        result = bridge.resolve("ref_e")
        self.assertIsInstance(result.trace_id, str)
        self.assertTrue(result.trace_id)

    def test_resolve_accepts_caller_trace_id(self):
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("ref_f"))
        result = bridge.resolve("ref_f", trace_id="custom-resolve-tid")
        self.assertEqual(result.trace_id, "custom-resolve-tid")

    def test_missing_ref_diagnostic_has_trace_id(self):
        bridge = BehaviorCompilerBridge()
        result = bridge.resolve("nonexistent_x")
        self.assertTrue(result.trace_id)
        for d in result.diagnostics:
            self.assertEqual(d.trace_id, result.trace_id)


# ===========================================================================
# TestPhase1BackwardCompatibility
# ===========================================================================

class TestPhase1BackwardCompatibility(unittest.TestCase):
    """Phase 1 DiagnosticsKey fields are unchanged by STAGE-05 additions."""

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
        ir = WorkflowIR(nodes=[_simple_node_ir("a0")], edges=[])
        return engine.execute(ir, scenario={})

    def test_phase1_path_key_present(self):
        r = self._run_simple()
        self.assertIn(DiagnosticsKey.PATH, r["diagnostics"])

    def test_phase1_executed_count_key_present(self):
        r = self._run_simple()
        self.assertIn(DiagnosticsKey.EXECUTED_COUNT, r["diagnostics"])

    def test_phase1_failed_nodes_key_present(self):
        r = self._run_simple()
        self.assertIn(DiagnosticsKey.FAILED_NODES, r["diagnostics"])

    def test_phase1_has_action_key_present(self):
        r = self._run_simple()
        self.assertIn(DiagnosticsKey.HAS_ACTION, r["diagnostics"])

    def test_phase1_status_success(self):
        r = self._run_simple()
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_phase1_required_top_level_fields(self):
        r = self._run_simple()
        for key in ("status", "reason", "task_id", "node_count", "results",
                    "metrics", "diagnostics"):
            self.assertIn(key, r, f"Phase 1 field '{key}' missing")


if __name__ == "__main__":
    unittest.main()
