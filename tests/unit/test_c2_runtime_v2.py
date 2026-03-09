#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests — Circle 2 Step 2.3: Runtime v2 wiring.

Coverage
--------
BehaviorInvokeInput v2 fields:
- sdk_settings and heartbeat_policy fields default to None
- to_dict() includes sdk_settings_present and heartbeat_policy keys

BehaviorInvokeOutput v2 fields:
- heartbeat_policy field defaults to None
- success() / failed() / blocked() accept heartbeat_policy and heartbeat_diagnostics kwargs
- to_dict() includes heartbeat_policy key

DiagnosticsKey new constants:
- BEHAVIOR_CORE_DIAGNOSTICS, HEARTBEAT_DIAGNOSTICS, HEARTBEAT_STATUS defined

RuntimeResult factories include new keys:
- BEHAVIOR_CORE_DIAGNOSTICS equals BEHAVIOR_DIAGNOSTICS (same data)
- HEARTBEAT_DIAGNOSTICS populated from heartbeat_diagnostics param
- HEARTBEAT_STATUS present (empty string by default in Circle 2)

NodeExecutor v2 wiring:
- sdk_settings extracted from context["scenario"]["sdk_settings"]
- sdk_settings extracted from context["sdk_settings"] directly
- _cancel_fn propagated to sub_executor

Behavior result dict includes heartbeat section keys:
- heartbeat_status, heartbeat_tick_count, heartbeat_diagnostics, heartbeat_policy
"""

import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.behavior_artifact import (  # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
)
from system.runtime.contracts import DiagnosticsKey, RuntimeResult  # noqa: E402
from system.ir.workflow_ir import WorkflowIR                         # noqa: E402


# ---------------------------------------------------------------------------
# Minimal fake registry so NodeExecutor can import nodes
# ---------------------------------------------------------------------------
_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


def _install_registry():
    _registry.clear()
    sys.modules["nodes"] = _nodes_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _make_artifact(policy=None) -> BehaviorArtifact:
    from system.behavior.heartbeat_policy import HeartbeatPolicy
    return BehaviorArtifact.create(
        behavior_ref="test_ref",
        behavior_ir=_empty_ir(),
        heartbeat_policy=policy if policy is not None else HeartbeatPolicy.default().to_dict(),
    )


def _default_metrics() -> dict:
    return {"events": 0, "elapsed_s": 0.0}


# ---------------------------------------------------------------------------
# BehaviorInvokeInput v2 fields
# ---------------------------------------------------------------------------

class TestBehaviorInvokeInputV2Fields(unittest.TestCase):
    def test_sdk_settings_defaults_to_none(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        self.assertIsNone(inp.sdk_settings)

    def test_heartbeat_policy_defaults_to_none(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        self.assertIsNone(inp.heartbeat_policy)

    def test_sdk_settings_can_be_set(self):
        sdk = {"host": "192.168.1.1", "port": 8080}
        inp = BehaviorInvokeInput(behavior_ref="ref", sdk_settings=sdk)
        self.assertEqual(inp.sdk_settings, sdk)

    def test_heartbeat_policy_can_be_set(self):
        policy = {"tick_ms": 200, "timeout_ms": 5000}
        inp = BehaviorInvokeInput(behavior_ref="ref", heartbeat_policy=policy)
        self.assertEqual(inp.heartbeat_policy, policy)

    def test_to_dict_has_sdk_settings_present_key(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        d = inp.to_dict()
        self.assertIn("sdk_settings_present", d)

    def test_to_dict_sdk_settings_present_false_when_none(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        self.assertFalse(inp.to_dict()["sdk_settings_present"])

    def test_to_dict_sdk_settings_present_true_when_set(self):
        inp = BehaviorInvokeInput(behavior_ref="ref", sdk_settings={"k": "v"})
        self.assertTrue(inp.to_dict()["sdk_settings_present"])

    def test_to_dict_has_heartbeat_policy_key(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        self.assertIn("heartbeat_policy", inp.to_dict())

    def test_to_dict_heartbeat_policy_none_when_not_set(self):
        inp = BehaviorInvokeInput(behavior_ref="ref")
        self.assertIsNone(inp.to_dict()["heartbeat_policy"])

    def test_to_dict_heartbeat_policy_present_when_set(self):
        policy = {"tick_ms": 100}
        inp = BehaviorInvokeInput(behavior_ref="ref", heartbeat_policy=policy)
        self.assertEqual(inp.to_dict()["heartbeat_policy"], policy)


# ---------------------------------------------------------------------------
# BehaviorInvokeOutput v2 fields
# ---------------------------------------------------------------------------

class TestBehaviorInvokeOutputV2Fields(unittest.TestCase):
    def test_heartbeat_policy_defaults_to_none(self):
        out = BehaviorInvokeOutput(status="success")
        self.assertIsNone(out.heartbeat_policy)

    def test_success_factory_accepts_heartbeat_policy(self):
        policy = {"tick_ms": 100, "timeout_ms": 5000, "fail_policy": "stop_behavior",
                  "max_override": 0.2, "safety_mode": "strict"}
        out = BehaviorInvokeOutput.success(trace_id="t1", heartbeat_policy=policy)
        self.assertEqual(out.heartbeat_policy, policy)

    def test_failed_factory_accepts_heartbeat_policy(self):
        policy = {"tick_ms": 100}
        out = BehaviorInvokeOutput.failed(
            trace_id="t1", reason="r", heartbeat_policy=policy
        )
        self.assertEqual(out.heartbeat_policy, policy)

    def test_success_factory_accepts_heartbeat_diagnostics(self):
        diag = BehaviorDiagnostic.error(code="test", message="msg", trace_id="t1")
        out = BehaviorInvokeOutput.success(trace_id="t1", heartbeat_diagnostics=[diag])
        self.assertEqual(len(out.heartbeat_diagnostics), 1)

    def test_failed_factory_accepts_heartbeat_diagnostics(self):
        diag = BehaviorDiagnostic.warning(code="test", message="msg", trace_id="t1")
        out = BehaviorInvokeOutput.failed(
            trace_id="t1", reason="r", heartbeat_diagnostics=[diag]
        )
        self.assertEqual(len(out.heartbeat_diagnostics), 1)

    def test_blocked_factory_heartbeat_policy_none(self):
        out = BehaviorInvokeOutput.blocked(trace_id="t1", reason="r")
        self.assertIsNone(out.heartbeat_policy)

    def test_to_dict_has_heartbeat_policy(self):
        out = BehaviorInvokeOutput.success(trace_id="t1")
        self.assertIn("heartbeat_policy", out.to_dict())

    def test_to_dict_heartbeat_policy_matches(self):
        policy = {"tick_ms": 150}
        out = BehaviorInvokeOutput.success(trace_id="t1", heartbeat_policy=policy)
        self.assertEqual(out.to_dict()["heartbeat_policy"], policy)

    def test_to_dict_heartbeat_policy_none_by_default(self):
        out = BehaviorInvokeOutput.success(trace_id="t1")
        self.assertIsNone(out.to_dict()["heartbeat_policy"])


# ---------------------------------------------------------------------------
# DiagnosticsKey new constants
# ---------------------------------------------------------------------------

class TestDiagnosticsKeyV2Constants(unittest.TestCase):
    def test_behavior_core_diagnostics_key_defined(self):
        self.assertTrue(hasattr(DiagnosticsKey, "BEHAVIOR_CORE_DIAGNOSTICS"))

    def test_heartbeat_diagnostics_key_defined(self):
        self.assertTrue(hasattr(DiagnosticsKey, "HEARTBEAT_DIAGNOSTICS"))

    def test_heartbeat_status_key_defined(self):
        self.assertTrue(hasattr(DiagnosticsKey, "HEARTBEAT_STATUS"))

    def test_behavior_core_diagnostics_value_is_string(self):
        self.assertIsInstance(DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS, str)

    def test_heartbeat_diagnostics_value_is_string(self):
        self.assertIsInstance(DiagnosticsKey.HEARTBEAT_DIAGNOSTICS, str)

    def test_heartbeat_status_value_is_string(self):
        self.assertIsInstance(DiagnosticsKey.HEARTBEAT_STATUS, str)

    def test_behavior_core_diagnostics_key_not_same_as_behavior_diagnostics(self):
        # The keys must be distinct strings so they occupy separate dict slots.
        self.assertNotEqual(
            DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS,
            DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
        )

    def test_heartbeat_diagnostics_key_distinct(self):
        self.assertNotEqual(
            DiagnosticsKey.HEARTBEAT_DIAGNOSTICS,
            DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
        )


# ---------------------------------------------------------------------------
# RuntimeResult factories include new keys
# ---------------------------------------------------------------------------

class TestRuntimeResultV2Keys(unittest.TestCase):
    def _make_success(self, behavior_diags=None, hb_diags=None) -> dict:
        from system.runtime.contracts import ExecutionPath
        result = RuntimeResult.success(
            task_id="t1",
            node_count=1,
            results={},
            metrics=_default_metrics(),
            path=ExecutionPath.WORKFLOW_IR,
            behavior_diagnostics=behavior_diags,
            heartbeat_diagnostics=hb_diags,
        )
        return result.diagnostics

    def _make_failed(self, hb_diags=None) -> dict:
        from system.runtime.contracts import ExecutionPath
        result = RuntimeResult.failed(
            task_id="t1",
            node_count=1,
            results={},
            metrics=_default_metrics(),
            reason="test_reason",
            path=ExecutionPath.WORKFLOW_IR,
            heartbeat_diagnostics=hb_diags,
        )
        return result.diagnostics

    def test_success_has_behavior_core_diagnostics_key(self):
        d = self._make_success()
        self.assertIn(DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS, d)

    def test_success_has_heartbeat_diagnostics_key(self):
        d = self._make_success()
        self.assertIn(DiagnosticsKey.HEARTBEAT_DIAGNOSTICS, d)

    def test_success_has_heartbeat_status_key(self):
        d = self._make_success()
        self.assertIn(DiagnosticsKey.HEARTBEAT_STATUS, d)

    def test_success_behavior_core_diagnostics_empty_by_default(self):
        d = self._make_success()
        self.assertEqual(d[DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS], [])

    def test_success_heartbeat_diagnostics_empty_by_default(self):
        d = self._make_success()
        self.assertEqual(d[DiagnosticsKey.HEARTBEAT_DIAGNOSTICS], [])

    def test_success_heartbeat_status_empty_string_by_default(self):
        d = self._make_success()
        self.assertEqual(d[DiagnosticsKey.HEARTBEAT_STATUS], "")

    def test_success_behavior_core_diagnostics_matches_behavior_diagnostics(self):
        """BEHAVIOR_CORE_DIAGNOSTICS and BEHAVIOR_DIAGNOSTICS must hold same data."""
        bdiag = [{"level": "info", "code": "c1"}]
        d = self._make_success(behavior_diags=bdiag)
        self.assertEqual(
            d[DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS],
            d[DiagnosticsKey.BEHAVIOR_DIAGNOSTICS],
        )

    def test_success_heartbeat_diagnostics_populated(self):
        hb = [{"level": "warning", "code": "heartbeat_policy.tick_ms_out_of_range"}]
        d = self._make_success(hb_diags=hb)
        self.assertEqual(d[DiagnosticsKey.HEARTBEAT_DIAGNOSTICS], hb)

    def test_failed_has_behavior_core_diagnostics_key(self):
        d = self._make_failed()
        self.assertIn(DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS, d)

    def test_failed_has_heartbeat_diagnostics_key(self):
        d = self._make_failed()
        self.assertIn(DiagnosticsKey.HEARTBEAT_DIAGNOSTICS, d)

    def test_failed_heartbeat_diagnostics_populated(self):
        hb = [{"level": "warning", "code": "heartbeat_policy.timeout_ms_out_of_range"}]
        d = self._make_failed(hb_diags=hb)
        self.assertEqual(d[DiagnosticsKey.HEARTBEAT_DIAGNOSTICS], hb)


# ---------------------------------------------------------------------------
# NodeExecutor v2 wiring: sdk_settings extraction and cancel_fn propagation
# ---------------------------------------------------------------------------

class TestNodeExecutorV2Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_registry()

    def setUp(self):
        from system.runtime.node_executor import NodeExecutor
        self.executor = NodeExecutor()
        self.executor.use_flow_aware_execution = True

    def _make_bridge_with_artifact(self, behavior_ref: str, policy=None):
        """Return a bridge with one registered empty artifact."""
        from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
        bridge = BehaviorCompilerBridge.__new__(BehaviorCompilerBridge)
        bridge._registry = {}
        bridge._allow_disk_lookup = False
        from pathlib import Path
        bridge._cache_dir = Path("/nonexistent")
        artifact = _make_artifact(policy=policy)
        artifact.behavior_ref = behavior_ref
        bridge._registry[behavior_ref] = artifact
        return bridge, artifact

    def test_sdk_settings_extracted_from_scenario_dict(self):
        """sdk_settings from context["scenario"]["sdk_settings"] → invoke_input.sdk_settings."""
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge, artifact = self._make_bridge_with_artifact("test_ref")
        self.executor._behavior_invoker = BehaviorSubgraphInvoker()
        self.executor._behavior_bridge = bridge

        captured_sdk = []

        # Patch invoker to capture invoke_input
        original_invoke = self.executor._behavior_invoker.invoke

        def capturing_invoke(invoke_input, **kw):
            captured_sdk.append(invoke_input.sdk_settings)
            return original_invoke(invoke_input, **kw)

        self.executor._behavior_invoker.invoke = capturing_invoke

        from system.ir.workflow_ir import IRNode, NodeKind
        n = IRNode(id="b1", schema_id="behavior_call", kind=NodeKind.BEHAVIOR_CALL, params={})
        self.executor.add_node("b1", "behavior_call", n.to_dict())

        sdk = {"brand": "unitree", "host": "192.168.1.100"}
        self.executor.execute(context={"scenario": {"sdk_settings": sdk}, "mission_trace_id": "tid"})
        self.assertEqual(captured_sdk[0], sdk)

    def test_sdk_settings_extracted_from_direct_context(self):
        """sdk_settings from context["sdk_settings"] → invoke_input.sdk_settings."""
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge, _ = self._make_bridge_with_artifact("ref2")
        self.executor._behavior_invoker = BehaviorSubgraphInvoker()
        self.executor._behavior_bridge = bridge

        captured_sdk = []
        original_invoke = self.executor._behavior_invoker.invoke

        def capturing_invoke(invoke_input, **kw):
            captured_sdk.append(invoke_input.sdk_settings)
            return original_invoke(invoke_input, **kw)

        self.executor._behavior_invoker.invoke = capturing_invoke

        from system.ir.workflow_ir import IRNode, NodeKind
        n = IRNode(id="b2", schema_id="behavior_call", kind=NodeKind.BEHAVIOR_CALL, params={})
        n_dict = n.to_dict()
        n_dict["params"] = {"behavior_ref": {"value": "ref2", "name": "behavior_ref", "param_type": "string"}}
        self.executor.add_node("b2", "behavior_call", n_dict)

        sdk = {"direct_key": True}
        self.executor.execute(context={"sdk_settings": sdk, "mission_trace_id": "tid2"})
        self.assertEqual(captured_sdk[0], sdk)

    def test_cancel_fn_propagated_to_sub_executor(self):
        """_cancel_fn on parent executor is propagated to the sub_executor."""
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge, _ = self._make_bridge_with_artifact("ref3")
        self.executor._behavior_invoker = BehaviorSubgraphInvoker()
        self.executor._behavior_bridge = bridge

        cancel_fn = lambda: False
        self.executor._cancel_fn = cancel_fn

        captured_sub_cancel = []
        original_invoke = self.executor._behavior_invoker.invoke

        def capturing_invoke(invoke_input, bridge, sub_executor, **kw):
            captured_sub_cancel.append(sub_executor._cancel_fn)
            return original_invoke(invoke_input, bridge=bridge, sub_executor=sub_executor, **kw)

        self.executor._behavior_invoker.invoke = capturing_invoke

        from system.ir.workflow_ir import IRNode, NodeKind
        n = IRNode(id="b3", schema_id="behavior_call", kind=NodeKind.BEHAVIOR_CALL, params={})
        n_dict = n.to_dict()
        n_dict["params"] = {"behavior_ref": {"value": "ref3", "name": "behavior_ref", "param_type": "string"}}
        self.executor.add_node("b3", "behavior_call", n_dict)

        self.executor.execute(context={"mission_trace_id": "tid3"})
        self.assertIsNotNone(captured_sub_cancel[0])
        self.assertIs(captured_sub_cancel[0], cancel_fn)


# ---------------------------------------------------------------------------
# Behavior node result dict includes heartbeat section keys
# ---------------------------------------------------------------------------

class TestBehaviorResultHeartbeatKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_registry()

    def setUp(self):
        from system.runtime.node_executor import NodeExecutor
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge

        self.executor = NodeExecutor()
        self.executor.use_flow_aware_execution = True

        bridge = BehaviorCompilerBridge.__new__(BehaviorCompilerBridge)
        bridge._registry = {}
        bridge._allow_disk_lookup = False
        from pathlib import Path
        bridge._cache_dir = Path("/nonexistent")
        artifact = _make_artifact()
        artifact.behavior_ref = "stand"
        bridge._registry["stand"] = artifact

        self.executor._behavior_invoker = BehaviorSubgraphInvoker()
        self.executor._behavior_bridge = bridge

        from system.ir.workflow_ir import IRNode, NodeKind
        n = IRNode(id="b_stand", schema_id="behavior_call", kind=NodeKind.BEHAVIOR_CALL, params={})
        n_dict = n.to_dict()
        n_dict["params"] = {"behavior_ref": {"value": "stand", "name": "behavior_ref", "param_type": "string"}}
        self.executor.add_node("b_stand", "behavior_call", n_dict)

    def test_result_has_heartbeat_status_key(self):
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertIn("heartbeat_status", out)

    def test_result_has_heartbeat_tick_count_key(self):
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertIn("heartbeat_tick_count", out)

    def test_result_has_heartbeat_diagnostics_key(self):
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertIn("heartbeat_diagnostics", out)

    def test_result_has_heartbeat_policy_key(self):
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertIn("heartbeat_policy", out)

    def test_result_heartbeat_policy_matches_artifact(self):
        """heartbeat_policy in result matches the artifact's policy."""
        from system.behavior.heartbeat_policy import HeartbeatPolicy
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertEqual(out["heartbeat_policy"], HeartbeatPolicy.default().to_dict())

    def test_result_heartbeat_diagnostics_is_list(self):
        results = self.executor.execute(context={"mission_trace_id": "t1"})
        out = results.get("b_stand", {})
        self.assertIsInstance(out["heartbeat_diagnostics"], list)


if __name__ == "__main__":
    unittest.main()
