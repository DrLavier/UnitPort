#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests: Mission → Behavior Subgraph Invocation — Phase 2 STAGE-03.

Done criteria
-------------
A mission WorkflowIR containing a "behavior" schema_id node executes through
real BehaviorSubgraphInvoker → BehaviorArtifact → NodeExecutor subgraph
invocation, and the result is propagated back to the Mission RuntimeResult
in the unified Phase 1 contract format.

Coverage
--------
- Mission with behavior node + valid artifact → success, result in "results"
- Mission with behavior node + missing ref → failed, node in failed_nodes
- Mission with behavior node + invalid artifact → failed, node in failed_nodes
- Mission with behavior node + no bridge → behavior node skipped (backward compat)
- Mixed mission: one normal node + one behavior node → both executed
- Behavior node failure does NOT break the RuntimeResult contract schema
- trace_id propagated into behavior node result

Isolation
---------
- nodes registry stub installed in setUpClass / tearDownClass.
- No Qt, no SDK.
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.runtime_engine import RuntimeEngine              # noqa: E402
from system.runtime.contracts import DiagnosticsKey, RuntimeStatus   # noqa: E402
from system.behavior.behavior_artifact import (                       # noqa: E402
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


class SubgraphNode:
    """Node used inside behavior subgraphs."""
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "subgraph_ok", "node": self.nid}


def _install_registry():
    _registry.clear()
    _registry["simple"] = SimpleNode
    _registry["subgraph_action"] = SubgraphNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _behavior_node_ir(behavior_ref: str, node_id: str = "b0") -> IRNode:
    """Create a WorkflowIR IRNode with schema_id='behavior'."""
    return IRNode(
        id=node_id,
        kind=NodeKind.ACTION,
        schema_id="behavior",
        params={
            "behavior_ref": IRParam(
                name="behavior_ref",
                value=behavior_ref,
                param_type="string",
            )
        },
    )


def _simple_node_ir(node_id: str = "a0") -> IRNode:
    return IRNode(id=node_id, kind=NodeKind.ACTION, schema_id="simple", params={})


def _subgraph_ir(node_id: str = "s0") -> WorkflowIR:
    """A one-node subgraph WorkflowIR for behavior artifacts."""
    return WorkflowIR(
        nodes=[IRNode(id=node_id, kind=NodeKind.ACTION, schema_id="subgraph_action", params={})],
        edges=[],
    )


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


_REQUIRED_FIELDS = frozenset({
    "status", "reason", "task_id", "node_count", "results", "metrics", "diagnostics"
})


# ===========================================================================
# TestBehaviorNodeSuccess
# ===========================================================================

class TestBehaviorNodeSuccess(unittest.TestCase):
    """Mission with a behavior node that resolves and executes successfully."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def _run(self, behavior_ref: str) -> dict:
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact(behavior_ref))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node_ir(behavior_ref)], edges=[])
        return engine.execute(ir, scenario={})

    def test_status_success(self):
        r = self._run("stand")
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_contract_fields_present(self):
        r = self._run("stand2")
        missing = _REQUIRED_FIELDS - r.keys()
        self.assertFalse(missing, f"missing fields: {missing}")

    def test_behavior_node_in_results(self):
        r = self._run("walk")
        self.assertIn("b0", r["results"])

    def test_behavior_node_result_has_no_error(self):
        r = self._run("walk2")
        self.assertNotIn("error", r["results"].get("b0", {}))

    def test_behavior_node_result_status_success(self):
        r = self._run("sit")
        self.assertEqual(r["results"]["b0"].get("status"), "success")

    def test_behavior_node_result_has_outputs(self):
        r = self._run("run")
        self.assertIn("outputs", r["results"]["b0"])

    def test_behavior_node_result_has_node_results(self):
        r = self._run("jump")
        self.assertIn("node_results", r["results"]["b0"])
        # Subgraph node 's0' should be present
        self.assertIn("s0", r["results"]["b0"]["node_results"])

    def test_failed_nodes_empty(self):
        r = self._run("crouch")
        self.assertEqual(r["diagnostics"][DiagnosticsKey.FAILED_NODES], [])

    def test_node_count_reflects_mission_nodes(self):
        r = self._run("spin")
        self.assertEqual(r["node_count"], 1)

    def test_trace_id_in_behavior_result(self):
        r = self._run("wave")
        b_result = r["results"].get("b0", {})
        self.assertIn("trace_id", b_result)


# ===========================================================================
# TestBehaviorNodeMissingRef
# ===========================================================================

class TestBehaviorNodeMissingRef(unittest.TestCase):
    """Mission with a behavior node whose ref is not registered."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def _run(self) -> dict:
        engine = RuntimeEngine(behavior_bridge=BehaviorCompilerBridge())
        ir = WorkflowIR(nodes=[_behavior_node_ir("ghost_ref")], edges=[])
        return engine.execute(ir, scenario={})

    def test_status_failed(self):
        r = self._run()
        self.assertEqual(r["status"], RuntimeStatus.FAILED)

    def test_reason_node_execution_failed(self):
        r = self._run()
        self.assertEqual(r["reason"], "node_execution_failed")

    def test_behavior_node_in_failed_nodes(self):
        r = self._run()
        self.assertIn("b0", r["diagnostics"][DiagnosticsKey.FAILED_NODES])

    def test_behavior_node_result_has_error_key(self):
        r = self._run()
        self.assertIn("error", r["results"].get("b0", {}))

    def test_error_value_is_ref_not_found(self):
        r = self._run()
        self.assertEqual(
            r["results"]["b0"]["error"],
            BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
        )

    def test_contract_fields_present(self):
        r = self._run()
        missing = _REQUIRED_FIELDS - r.keys()
        self.assertFalse(missing, f"missing fields: {missing}")


# ===========================================================================
# TestBehaviorNodeInvalidArtifact
# ===========================================================================

class TestBehaviorNodeInvalidArtifact(unittest.TestCase):
    """Mission with a behavior node whose artifact has compile errors."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def _run(self) -> dict:
        bridge = BehaviorCompilerBridge()
        bridge.register(_invalid_artifact("broken"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(nodes=[_behavior_node_ir("broken")], edges=[])
        return engine.execute(ir, scenario={})

    def test_status_failed(self):
        self.assertEqual(self._run()["status"], RuntimeStatus.FAILED)

    def test_error_is_artifact_invalid(self):
        r = self._run()
        self.assertEqual(
            r["results"]["b0"]["error"],
            BehaviorErrorCode.ARTIFACT_INVALID,
        )

    def test_behavior_node_in_failed_nodes(self):
        r = self._run()
        self.assertIn("b0", r["diagnostics"][DiagnosticsKey.FAILED_NODES])


# ===========================================================================
# TestBehaviorNodeNoBridge (backward compatibility)
# ===========================================================================

class TestBehaviorNodeNoBridge(unittest.TestCase):
    """No behavior_bridge configured → behavior node returns skipped (not an error)."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def _run(self) -> dict:
        engine = RuntimeEngine()   # behavior_bridge=None (default)
        ir = WorkflowIR(nodes=[_behavior_node_ir("stand")], edges=[])
        return engine.execute(ir, scenario={})

    def test_status_success(self):
        # "skipped" result has no "error" key → not a failed_node → success
        r = self._run()
        self.assertEqual(r["status"], RuntimeStatus.SUCCESS)

    def test_behavior_node_not_in_failed_nodes(self):
        r = self._run()
        self.assertNotIn("b0", r["diagnostics"][DiagnosticsKey.FAILED_NODES])

    def test_behavior_node_result_is_skipped(self):
        r = self._run()
        b = r["results"].get("b0", {})
        self.assertEqual(b.get("status"), "skipped")
        self.assertIn("no_behavior_invoker", b.get("reason", ""))


# ===========================================================================
# TestMixedMission
# ===========================================================================

class TestMixedMission(unittest.TestCase):
    """Mission with one normal node + one behavior node — both must execute."""

    @classmethod
    def setUpClass(cls):
        cls._orig_nodes = sys.modules.get("nodes")
        _install_registry()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_nodes is None:
            sys.modules.pop("nodes", None)
        else:
            sys.modules["nodes"] = cls._orig_nodes

    def _run(self) -> dict:
        bridge = BehaviorCompilerBridge()
        bridge.register(_valid_artifact("wave"))
        engine = RuntimeEngine(behavior_bridge=bridge)
        ir = WorkflowIR(
            nodes=[
                _simple_node_ir("a0"),
                _behavior_node_ir("wave", node_id="b0"),
            ],
            edges=[
                IREdge(
                    from_node="a0", from_port="flow_out",
                    to_node="b0", to_port="flow_in",
                    edge_type=EdgeType.FLOW,
                )
            ],
        )
        return engine.execute(ir, scenario={})

    def test_status_success(self):
        self.assertEqual(self._run()["status"], RuntimeStatus.SUCCESS)

    def test_normal_node_executed(self):
        r = self._run()
        self.assertIn("a0", r["results"])
        self.assertEqual(r["results"]["a0"].get("status"), "ok")

    def test_behavior_node_executed(self):
        r = self._run()
        self.assertIn("b0", r["results"])
        self.assertEqual(r["results"]["b0"].get("status"), "success")

    def test_node_count(self):
        r = self._run()
        self.assertEqual(r["node_count"], 2)

    def test_failed_nodes_empty(self):
        r = self._run()
        self.assertEqual(r["diagnostics"][DiagnosticsKey.FAILED_NODES], [])


if __name__ == "__main__":
    unittest.main()
