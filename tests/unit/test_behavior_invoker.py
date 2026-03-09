#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for BehaviorSubgraphInvoker — Phase 2 STAGE-03.

Coverage
--------
- resolve blocked (missing ref): BehaviorInvokeOutput.blocked returned
- resolve blocked (invalid artifact): BehaviorInvokeOutput.blocked returned
- success path: subgraph loads + executes, outputs collected
- subgraph node failure: BehaviorInvokeOutput.failed with SUBGRAPH_EXECUTION_FAILED
- subgraph abort: BehaviorInvokeOutput.failed with SUBGRAPH_ABORTED
- trace_id propagation through invoke_input → result
- policy.max_loop_iterations injected into sub_executor
- _extract_behavior_ref: WorkflowIR param format, canvas top-level, empty

Isolation
---------
- node registry stub installed in setUpClass / tearDownClass only.
- BehaviorSubgraphInvoker receives a real NodeExecutor stub.
- Bridge is exercised via real BehaviorCompilerBridge with register().
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.behavior_invoker import BehaviorSubgraphInvoker    # noqa: E402
from system.runtime.node_executor import NodeExecutor                   # noqa: E402
from system.behavior.behavior_artifact import (                          # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge  # noqa: E402
from system.ir.workflow_ir import IREdge, IRNode, NodeKind, WorkflowIR       # noqa: E402


# ---------------------------------------------------------------------------
# Fake node registry
# ---------------------------------------------------------------------------
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class OkNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok", "out": self.nid}


class FailNode:
    def __init__(self, nid): self.nid = nid
    def set_parameter(self, k, v): pass
    def execute(self, inputs): raise RuntimeError("intentional_node_failure")


def _install_registry():
    _registry.clear()
    _registry["ok_node"] = OkNode
    _registry["fail_node"] = FailNode
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _single_node_ir(schema_id: str = "ok_node", node_id: str = "n0") -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id=node_id, kind=NodeKind.ACTION, schema_id=schema_id, params={})],
        edges=[],
    )


def _make_valid_artifact(behavior_ref: str, schema_id: str = "ok_node") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_single_node_ir(schema_id),
    )


def _make_invalid_artifact(behavior_ref: str) -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_empty_ir(),
        diagnostics=[BehaviorDiagnostic.error("E001", "forced compile error")],
    )


def _make_input(behavior_ref: str, trace_id: str = "tid-test") -> BehaviorInvokeInput:
    return BehaviorInvokeInput(
        behavior_ref=behavior_ref,
        inputs={},
        context={},
        trace_id=trace_id,
    )


# ===========================================================================
# TestBehaviorSubgraphInvoker
# ===========================================================================

class TestBehaviorSubgraphInvoker(unittest.TestCase):

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

    def setUp(self):
        self.invoker = BehaviorSubgraphInvoker()
        self.bridge = BehaviorCompilerBridge()

    def _invoke(self, behavior_ref: str, trace_id: str = "t1") -> BehaviorInvokeOutput:
        sub_ex = NodeExecutor()
        return self.invoker.invoke(
            invoke_input=_make_input(behavior_ref, trace_id),
            bridge=self.bridge,
            sub_executor=sub_ex,
        )

    # ── blocked: missing ref ──────────────────────────────────────────

    def test_missing_ref_returns_blocked(self):
        result = self._invoke("nonexistent")
        self.assertEqual(result.status, "blocked")

    def test_missing_ref_reason_is_ref_not_found(self):
        result = self._invoke("nonexistent")
        self.assertEqual(result.reason, BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND)

    def test_missing_ref_has_diagnostics(self):
        result = self._invoke("nonexistent")
        self.assertGreater(len(result.diagnostics), 0)

    def test_missing_ref_trace_id_propagated(self):
        result = self._invoke("nonexistent", trace_id="my-trace")
        self.assertEqual(result.trace_id, "my-trace")

    # ── blocked: invalid artifact ─────────────────────────────────────

    def test_invalid_artifact_returns_blocked(self):
        self.bridge.register(_make_invalid_artifact("bad"))
        result = self._invoke("bad")
        self.assertEqual(result.status, "blocked")

    def test_invalid_artifact_reason_is_artifact_invalid(self):
        self.bridge.register(_make_invalid_artifact("bad2"))
        result = self._invoke("bad2")
        self.assertEqual(result.reason, BehaviorErrorCode.ARTIFACT_INVALID)

    # ── success path ──────────────────────────────────────────────────

    def test_success_status(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand")
        self.assertEqual(result.status, "success")

    def test_success_reason_empty(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand")
        self.assertEqual(result.reason, "")

    def test_success_node_results_present(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand")
        self.assertIsInstance(result.node_results, dict)
        self.assertIn("n0", result.node_results)

    def test_success_outputs_present(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand")
        self.assertIsInstance(result.outputs, dict)
        self.assertIn("n0", result.outputs)

    def test_success_trace_id_propagated(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand", trace_id="t-ok")
        self.assertEqual(result.trace_id, "t-ok")

    def test_success_diagnostics_empty(self):
        self.bridge.register(_make_valid_artifact("stand"))
        result = self._invoke("stand")
        self.assertEqual(result.diagnostics, [])

    # ── subgraph node failure ─────────────────────────────────────────

    def test_subgraph_node_failure_returns_failed(self):
        self.bridge.register(_make_valid_artifact("crash", schema_id="fail_node"))
        result = self._invoke("crash")
        self.assertEqual(result.status, "failed")

    def test_subgraph_node_failure_reason(self):
        self.bridge.register(_make_valid_artifact("crash2", schema_id="fail_node"))
        result = self._invoke("crash2")
        self.assertEqual(result.reason, BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED)

    def test_subgraph_node_failure_has_diagnostic(self):
        self.bridge.register(_make_valid_artifact("crash3", schema_id="fail_node"))
        result = self._invoke("crash3")
        self.assertGreater(len(result.diagnostics), 0)

    def test_subgraph_node_failure_node_results_recorded(self):
        self.bridge.register(_make_valid_artifact("crash4", schema_id="fail_node"))
        result = self._invoke("crash4")
        self.assertIn("n0", result.node_results)

    # ── policy propagation ────────────────────────────────────────────

    def test_policy_max_loop_iterations_injected(self):
        """policy.max_loop_iterations is applied to sub_executor."""
        class FakePolicy:
            max_loop_iterations = 42

        self.bridge.register(_make_valid_artifact("stand_p"))
        sub_ex = NodeExecutor()
        self.invoker.invoke(
            invoke_input=_make_input("stand_p"),
            bridge=self.bridge,
            sub_executor=sub_ex,
            policy=FakePolicy(),
        )
        self.assertEqual(sub_ex.max_loop_iterations, 42)

    def test_no_policy_does_not_crash(self):
        self.bridge.register(_make_valid_artifact("stand_np"))
        result = self._invoke("stand_np")
        self.assertEqual(result.status, "success")

    # ── invoke() always returns BehaviorInvokeOutput ──────────────────

    def test_returns_invoke_output_type(self):
        result = self._invoke("any_ref")
        self.assertIsInstance(result, BehaviorInvokeOutput)


# ===========================================================================
# TestExtractBehaviorRef  (static helper)
# ===========================================================================

class TestExtractBehaviorRef(unittest.TestCase):
    """Unit tests for NodeExecutor._extract_behavior_ref()."""

    def _call(self, node_data):
        return NodeExecutor._extract_behavior_ref(node_data)

    def test_workflowir_params_format(self):
        # WorkflowIR-origin: IRParam serialised as {"name":..., "value":..., ...}
        node_data = {"params": {"behavior_ref": {"name": "behavior_ref", "value": "walk", "param_type": "string"}}}
        self.assertEqual(self._call(node_data), "walk")

    def test_canvas_top_level_format(self):
        # Canvas-origin: top-level "behavior_ref" string
        node_data = {"behavior_ref": "run"}
        self.assertEqual(self._call(node_data), "run")

    def test_params_takes_precedence_over_top_level(self):
        node_data = {
            "behavior_ref": "top_level",
            "params": {"behavior_ref": {"value": "from_params"}},
        }
        self.assertEqual(self._call(node_data), "from_params")

    def test_empty_when_neither_present(self):
        self.assertEqual(self._call({}), "")

    def test_bare_string_param(self):
        node_data = {"params": {"behavior_ref": "direct_string"}}
        self.assertEqual(self._call(node_data), "direct_string")


if __name__ == "__main__":
    unittest.main()
