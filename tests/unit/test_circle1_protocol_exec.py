#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Circle 1 — Protocol Execution Semantics Completion.

Coverage
--------
motor_weight_protocol — strict mode (unknown top-level field rejection):
  - strict mode (default): unknown top-level field → (False, [error UNKNOWN_FIELD])
  - compat mode: unknown top-level field → (True, [warning UNKNOWN_FIELD])
  - strict mode: unknown target field → (False, [error UNKNOWN_FIELD])
  - compat mode: unknown target field → (True, [warning UNKNOWN_FIELD])
  - known fields only: passes in both modes
  - multiple unknown top-level fields all reported

ProtocolApplyEngine — apply call point:
  - no adapter: all targets → context-inject success, success_count correct
  - adapter with apply_motor_param returning True → success path
  - adapter with apply_motor_param returning False → skipped path
  - adapter raising exception → APPLY_FAILED diagnostic, skipped count
  - mixed adapter (some pass, some fail) → counts correct
  - empty target list → zero counts, no diagnostics
  - apply result to_dict() has all expected keys

BehaviorInvokeOutput — apply fields:
  - default dataclass: apply_success_count=0, apply_skipped_count=0
  - success() factory accepts apply kwargs, propagates them
  - failed() factory accepts apply kwargs, propagates them
  - to_dict() includes apply_success_count, apply_skipped_count, apply_failure_reasons

BehaviorSubgraphInvoker integration (via mock bridge):
  - valid protocol + no adapter → apply_success_count equals len(targets)
  - valid protocol + adapter (success) → apply_success_count set, no apply_failure_reasons
  - valid protocol + adapter (skips one) → apply_skipped_count=1
  - compat_mode context key enables compat validation path
  - invalid protocol → blocked, apply counts zero
  - stale protocol → blocked, apply counts zero
  - absent protocol → absent status, apply counts zero

NodeExecutor integration:
  - behavior node result dict includes apply_success_count, apply_skipped_count,
    apply_failure_reasons on success path
  - behavior node result dict includes apply counts on failure/blocked path (zeroed)

Diagnostics completeness — Circle 1 DoD:
  - success path: protocol_diagnostics contains VALID + APPLY_SUCCESS codes
  - compat path: protocol_diagnostics contains UNKNOWN_FIELD warning codes
  - clamped value path: CLAMPED_VALUE diag present alongside APPLY_SUCCESS
  - unknown-address path: UNKNOWN_ADDRESS diag present, target excluded from apply
"""

import sys
import time
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.motor_weight_protocol import (
    ProtocolDiagCode,
    ParsedTarget,
    validate_protocol_payload,
    parse_protocol_targets,
)
from system.behavior.protocol_apply_engine import ProtocolApplyEngine, ProtocolApplyResult
from system.behavior.behavior_artifact import (
    BehaviorDiagnostic,
    BehaviorInvokeOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_payload(**overrides):
    """Return a minimal valid protocol payload dict."""
    now = int(time.time() * 1000)
    p = {
        "protocol_version": "1.0",
        "trace_id": "tid-test",
        "timestamp_ms": now,
        "mode": "constant",
        "max_age_ms": 60_000,
        "targets": [
            {
                "motor_key": "leg_fl_hip",
                "param_key": "gain",
                "address": 1,
                "dtype": "float32",
                "shape": [1],
                "source": {"kind": "constant", "constant": 0.5},
            }
        ],
    }
    p.update(overrides)
    return p


def _parsed_target(motor_key="leg_fl_hip", param_key="gain", address=1, value=0.5):
    return ParsedTarget(
        motor_key=motor_key,
        param_key=param_key,
        address=address,
        dtype="float32",
        shape=[1],
        final_value=value,
    )


class _OkAdapter:
    """Mock adapter whose apply_motor_param always returns True."""
    def __init__(self):
        self.calls = []

    def apply_motor_param(self, target):
        self.calls.append(target)
        return True


class _SkipAdapter:
    """Mock adapter whose apply_motor_param always returns False."""
    def apply_motor_param(self, target):
        return False


class _ErrorAdapter:
    """Mock adapter whose apply_motor_param always raises."""
    def apply_motor_param(self, target):
        raise RuntimeError("hardware_write_failed")


class _MixedAdapter:
    """Mock adapter: returns True for first call, False for subsequent."""
    def __init__(self):
        self._call_count = 0

    def apply_motor_param(self, target):
        self._call_count += 1
        return self._call_count == 1


# ---------------------------------------------------------------------------
# Tests: strict mode (motor_weight_protocol)
# ---------------------------------------------------------------------------

class TestStrictMode(unittest.TestCase):
    """validate_protocol_payload strict_mode=True (default) behaviour."""

    def test_unknown_top_level_field_strict_blocks(self):
        p = _base_payload()
        p["extra_field"] = "surprise"
        ok, diags = validate_protocol_payload(p, strict_mode=True)
        self.assertFalse(ok)
        codes = [d.code for d in diags]
        self.assertIn(ProtocolDiagCode.UNKNOWN_FIELD, codes)
        levels = [d.level for d in diags if d.code == ProtocolDiagCode.UNKNOWN_FIELD]
        self.assertTrue(all(l == "error" for l in levels))

    def test_unknown_top_level_default_is_strict(self):
        """No explicit strict_mode arg — default must be strict."""
        p = _base_payload()
        p["unknown_key"] = 42
        ok, diags = validate_protocol_payload(p)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == ProtocolDiagCode.UNKNOWN_FIELD for d in diags))

    def test_multiple_unknown_top_level_all_reported(self):
        p = _base_payload()
        p["extra_a"] = 1
        p["extra_b"] = 2
        ok, diags = validate_protocol_payload(p, strict_mode=True)
        self.assertFalse(ok)
        uf_codes = [d for d in diags if d.code == ProtocolDiagCode.UNKNOWN_FIELD]
        unknown_names = {d.message for d in uf_codes}
        self.assertEqual(len(uf_codes), 2)
        self.assertTrue(any("extra_a" in m for m in unknown_names))
        self.assertTrue(any("extra_b" in m for m in unknown_names))

    def test_compat_mode_unknown_top_level_warns_not_blocks(self):
        p = _base_payload()
        p["extra_field"] = "ok_in_compat"
        ok, diags = validate_protocol_payload(p, strict_mode=False)
        self.assertTrue(ok)
        uf_diags = [d for d in diags if d.code == ProtocolDiagCode.UNKNOWN_FIELD]
        self.assertTrue(len(uf_diags) >= 1)
        self.assertEqual(uf_diags[0].level, "warning")

    def test_unknown_target_field_strict_blocks(self):
        p = _base_payload()
        p["targets"][0]["mystery_param"] = "value"
        ok, diags = validate_protocol_payload(p, strict_mode=True)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == ProtocolDiagCode.UNKNOWN_FIELD for d in diags))

    def test_unknown_target_field_compat_warns(self):
        p = _base_payload()
        p["targets"][0]["mystery_param"] = "value"
        ok, diags = validate_protocol_payload(p, strict_mode=False)
        self.assertTrue(ok)
        uf = [d for d in diags if d.code == ProtocolDiagCode.UNKNOWN_FIELD]
        self.assertTrue(len(uf) >= 1)
        self.assertEqual(uf[0].level, "warning")

    def test_known_fields_pass_strict(self):
        p = _base_payload()
        ok, diags = validate_protocol_payload(p, strict_mode=True)
        self.assertTrue(ok)
        self.assertFalse(any(d.code == ProtocolDiagCode.UNKNOWN_FIELD for d in diags))

    def test_known_fields_pass_compat(self):
        p = _base_payload()
        ok, diags = validate_protocol_payload(p, strict_mode=False)
        self.assertTrue(ok)
        self.assertFalse(any(d.code == ProtocolDiagCode.UNKNOWN_FIELD for d in diags))


# ---------------------------------------------------------------------------
# Tests: ProtocolApplyEngine
# ---------------------------------------------------------------------------

class TestProtocolApplyEngine(unittest.TestCase):
    """ProtocolApplyEngine.apply() result counts and diagnostics."""

    def _engine(self):
        return ProtocolApplyEngine()

    def test_no_adapter_all_context_inject_success(self):
        targets = [_parsed_target("m1", "p1"), _parsed_target("m2", "p2")]
        result = self._engine().apply(targets, adapter=None, trace_id="t1")
        self.assertEqual(result.apply_success_count, 2)
        self.assertEqual(result.apply_skipped_count, 0)
        self.assertEqual(result.apply_failure_reasons, [])
        codes = [d.code for d in result.diagnostics]
        self.assertEqual(codes.count(ProtocolDiagCode.APPLY_SUCCESS), 2)

    def test_ok_adapter_counts_success(self):
        adapter = _OkAdapter()
        targets = [_parsed_target("m1", "p1"), _parsed_target("m2", "p2")]
        result = self._engine().apply(targets, adapter=adapter, trace_id="t1")
        self.assertEqual(result.apply_success_count, 2)
        self.assertEqual(result.apply_skipped_count, 0)
        self.assertEqual(len(adapter.calls), 2)

    def test_skip_adapter_counts_skipped(self):
        result = self._engine().apply(
            [_parsed_target()], adapter=_SkipAdapter(), trace_id="t"
        )
        self.assertEqual(result.apply_success_count, 0)
        self.assertEqual(result.apply_skipped_count, 1)
        self.assertTrue(len(result.apply_failure_reasons) == 1)
        self.assertEqual(result.diagnostics[0].code, ProtocolDiagCode.APPLY_SKIPPED)

    def test_error_adapter_records_apply_failed(self):
        result = self._engine().apply(
            [_parsed_target()], adapter=_ErrorAdapter(), trace_id="t"
        )
        self.assertEqual(result.apply_success_count, 0)
        self.assertEqual(result.apply_skipped_count, 1)
        self.assertIn("hardware_write_failed", result.apply_failure_reasons[0])
        self.assertEqual(result.diagnostics[0].code, ProtocolDiagCode.APPLY_FAILED)

    def test_mixed_adapter_counts_both(self):
        targets = [_parsed_target("m1", "p1"), _parsed_target("m2", "p2")]
        result = self._engine().apply(targets, adapter=_MixedAdapter(), trace_id="t")
        self.assertEqual(result.apply_success_count, 1)
        self.assertEqual(result.apply_skipped_count, 1)

    def test_empty_targets_zero_counts(self):
        result = self._engine().apply([], adapter=None, trace_id="t")
        self.assertEqual(result.apply_success_count, 0)
        self.assertEqual(result.apply_skipped_count, 0)
        self.assertEqual(result.diagnostics, [])

    def test_to_dict_has_expected_keys(self):
        result = ProtocolApplyResult(
            apply_success_count=3,
            apply_skipped_count=1,
            apply_failure_reasons=["some reason"],
        )
        d = result.to_dict()
        self.assertIn("apply_success_count", d)
        self.assertIn("apply_skipped_count", d)
        self.assertIn("apply_failure_reasons", d)
        self.assertIn("diagnostics", d)
        self.assertEqual(d["apply_success_count"], 3)
        self.assertEqual(d["apply_skipped_count"], 1)

    def test_no_adapter_with_multiple_targets_all_success(self):
        targets = [_parsed_target(f"m{i}", "p", i) for i in range(5)]
        result = self._engine().apply(targets, adapter=None, trace_id="t")
        self.assertEqual(result.apply_success_count, 5)
        self.assertEqual(result.apply_skipped_count, 0)

    def test_diag_trace_id_propagated(self):
        result = self._engine().apply(
            [_parsed_target()], adapter=None, trace_id="my-trace-id"
        )
        for d in result.diagnostics:
            self.assertEqual(d.trace_id, "my-trace-id")


# ---------------------------------------------------------------------------
# Tests: BehaviorInvokeOutput apply fields
# ---------------------------------------------------------------------------

class TestBehaviorInvokeOutputApplyFields(unittest.TestCase):
    """apply_success_count / apply_skipped_count / apply_failure_reasons fields."""

    def test_defaults_are_zero(self):
        from system.ir.workflow_ir import WorkflowIR
        out = BehaviorInvokeOutput(status="success")
        self.assertEqual(out.apply_success_count, 0)
        self.assertEqual(out.apply_skipped_count, 0)
        self.assertEqual(out.apply_failure_reasons, [])

    def test_success_factory_propagates_apply_fields(self):
        out = BehaviorInvokeOutput.success(
            trace_id="t",
            apply_success_count=3,
            apply_skipped_count=1,
            apply_failure_reasons=["reason_a"],
        )
        self.assertEqual(out.apply_success_count, 3)
        self.assertEqual(out.apply_skipped_count, 1)
        self.assertEqual(out.apply_failure_reasons, ["reason_a"])

    def test_failed_factory_propagates_apply_fields(self):
        out = BehaviorInvokeOutput.failed(
            trace_id="t",
            reason="err",
            apply_success_count=0,
            apply_skipped_count=2,
            apply_failure_reasons=["r1", "r2"],
        )
        self.assertEqual(out.apply_skipped_count, 2)
        self.assertEqual(len(out.apply_failure_reasons), 2)

    def test_blocked_factory_has_zero_apply_counts(self):
        out = BehaviorInvokeOutput.blocked(trace_id="t", reason="INVALID_PROTOCOL")
        self.assertEqual(out.apply_success_count, 0)
        self.assertEqual(out.apply_skipped_count, 0)

    def test_blocked_factory_propagates_protocol_fields(self):
        diag = BehaviorDiagnostic.warning(
            code=ProtocolDiagCode.STALE,
            message="stale",
            trace_id="t",
        )
        out = BehaviorInvokeOutput.blocked(
            trace_id="t",
            reason="INVALID_PROTOCOL",
            protocol_status="stale",
            protocol_diagnostics=[diag],
        )
        self.assertEqual(out.protocol_status, "stale")
        self.assertEqual(len(out.protocol_diagnostics), 1)
        self.assertEqual(out.protocol_diagnostics[0].code, ProtocolDiagCode.STALE)

    def test_to_dict_includes_apply_fields(self):
        out = BehaviorInvokeOutput.success(
            trace_id="t",
            apply_success_count=2,
            apply_skipped_count=0,
        )
        d = out.to_dict()
        self.assertIn("apply_success_count", d)
        self.assertIn("apply_skipped_count", d)
        self.assertIn("apply_failure_reasons", d)
        self.assertEqual(d["apply_success_count"], 2)

    def test_success_factory_defaults_none_failure_reasons_to_empty_list(self):
        out = BehaviorInvokeOutput.success(trace_id="t")
        self.assertEqual(out.apply_failure_reasons, [])


# ---------------------------------------------------------------------------
# Tests: BehaviorSubgraphInvoker integration
# ---------------------------------------------------------------------------

def _build_minimal_ir():
    """Build a minimal WorkflowIR with one start node."""
    from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind
    ir = WorkflowIR(nodes=[], edges=[])
    return ir


def _make_stub_bridge(ref="test_behavior", valid=True):
    """Return a BehaviorCompilerBridge stub with a pre-loaded artifact."""
    from system.behavior.behavior_artifact import BehaviorArtifact
    from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
    from system.behavior.heartbeat_policy import HeartbeatPolicy
    bridge = BehaviorCompilerBridge()
    if valid:
        artifact = BehaviorArtifact.create(
            behavior_ref=ref,
            behavior_ir=_build_minimal_ir(),
            heartbeat_policy=HeartbeatPolicy.default().to_dict(),
        )
        bridge._registry[ref] = artifact
    return bridge


class TestInvokerApplyIntegration(unittest.TestCase):
    """BehaviorSubgraphInvoker end-to-end apply path."""

    def _make_invoke_input(self, ref="test_behavior", payload=None, context=None):
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        return BehaviorInvokeInput(
            behavior_ref=ref,
            inputs={},
            context=context or {},
            trace_id="tid-invoke",
            protocol_payload=payload,
        )

    def _make_sub_executor(self):
        from system.runtime.node_executor import NodeExecutor
        return NodeExecutor()

    def test_absent_protocol_apply_counts_zero(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=None),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.apply_success_count, 0)
        self.assertEqual(result.apply_skipped_count, 0)
        self.assertEqual(result.protocol_status, "absent")

    def test_valid_protocol_no_adapter_context_inject_success(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=payload),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.protocol_status, "valid")
        # 1 target → 1 context-inject success
        self.assertEqual(result.apply_success_count, 1)
        self.assertEqual(result.apply_skipped_count, 0)

    def test_valid_protocol_ok_adapter_apply_success(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        adapter = _OkAdapter()
        payload = _base_payload()
        ctx = {"robot_model": adapter}
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=payload, context=ctx),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.apply_success_count, 1)
        self.assertEqual(len(adapter.calls), 1)

    def test_valid_protocol_skip_adapter_apply_skipped(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        ctx = {"robot_model": _SkipAdapter()}
        payload = _base_payload()
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=payload, context=ctx),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.apply_skipped_count, 1)
        self.assertEqual(result.apply_success_count, 0)

    def test_invalid_protocol_blocked_apply_zero(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        bad_payload = {"protocol_version": "1.0", "trace_id": "x"}  # missing required
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=bad_payload),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.protocol_status, "invalid")
        self.assertTrue(len(result.protocol_diagnostics) > 0)
        self.assertEqual(result.apply_success_count, 0)
        self.assertEqual(result.apply_skipped_count, 0)

    def test_stale_protocol_blocked_apply_zero(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        stale_payload = _base_payload(timestamp_ms=0, max_age_ms=1)
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=stale_payload),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.protocol_status, "stale")
        self.assertTrue(any(d.code == ProtocolDiagCode.STALE for d in result.protocol_diagnostics))
        self.assertEqual(result.apply_success_count, 0)

    def test_compat_mode_unknown_field_does_not_block(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        payload["extra_key"] = "should_warn_not_block"
        ctx = {"protocol_compat_mode": True}
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=payload, context=ctx),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "success")
        # compat path: still applies
        self.assertEqual(result.apply_success_count, 1)

    def test_strict_mode_unknown_field_blocks(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        payload["rogue_key"] = "should_block"
        # default strict_mode=True (no compat_mode in context)
        result = inv.invoke(
            invoke_input=self._make_invoke_input(payload=payload),
            bridge=bridge,
            sub_executor=self._make_sub_executor(),
        )
        self.assertEqual(result.status, "blocked")


# ---------------------------------------------------------------------------
# Tests: NodeExecutor result dict apply fields
# ---------------------------------------------------------------------------

class TestNodeExecutorApplyFields(unittest.TestCase):
    """node_executor._execute_behavior_node result dict includes apply counts."""

    def _setup_executor_with_behavior_node(self, payload=None, adapter=None):
        """Build a NodeExecutor wired with a stub invoker/bridge, return executor + context."""
        from system.runtime.node_executor import NodeExecutor
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        executor = NodeExecutor()
        bridge = _make_stub_bridge("my_behavior")
        executor._behavior_invoker = BehaviorSubgraphInvoker()
        executor._behavior_bridge = bridge
        executor.add_node("b1", "behavior_call", {
            "params": {"behavior_ref": {"name": "behavior_ref", "value": "my_behavior",
                                        "param_type": "string"}},
        })
        ctx: dict = {}
        if adapter is not None:
            ctx["robot_model"] = adapter
        if payload is not None:
            ctx["_test_protocol_payload"] = payload
        return executor, ctx, payload

    def _run_behavior_node_direct(self, payload=None, adapter=None):
        """Directly call _execute_behavior_node and return its result dict."""
        from system.runtime.node_executor import NodeExecutor
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        executor = NodeExecutor()
        bridge = _make_stub_bridge("my_behavior")
        executor._behavior_invoker = BehaviorSubgraphInvoker()
        executor._behavior_bridge = bridge
        node = {
            "id": "b1",
            "type": "behavior_call",
            "data": {
                "params": {"behavior_ref": {"name": "behavior_ref",
                                            "value": "my_behavior",
                                            "param_type": "string"}},
            },
        }
        inputs = {"condition": payload} if payload else {}
        context: dict = {}
        if adapter is not None:
            context["robot_model"] = adapter
        return executor._execute_behavior_node(node, inputs, context)

    def test_success_result_has_apply_fields(self):
        result = self._run_behavior_node_direct(payload=None)
        self.assertIn("apply_success_count", result)
        self.assertIn("apply_skipped_count", result)
        self.assertIn("apply_failure_reasons", result)

    def test_valid_protocol_result_apply_count_nonzero(self):
        payload = _base_payload()
        result = self._run_behavior_node_direct(payload=payload)
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result["apply_success_count"], 1)

    def test_failure_result_has_apply_fields_zeroed(self):
        """blocked path: apply counts should be 0."""
        bad_payload = {"not": "a valid protocol"}
        result = self._run_behavior_node_direct(payload=bad_payload)
        self.assertIn("apply_success_count", result)
        self.assertEqual(result["apply_success_count"], 0)
        self.assertEqual(result["apply_skipped_count"], 0)

    def test_adapter_ok_reflected_in_result(self):
        payload = _base_payload()
        adapter = _OkAdapter()
        result = self._run_behavior_node_direct(payload=payload, adapter=adapter)
        self.assertEqual(result.get("status"), "success")
        self.assertEqual(result["apply_success_count"], 1)
        self.assertEqual(len(adapter.calls), 1)

    def test_adapter_skip_reflected_in_result(self):
        payload = _base_payload()
        result = self._run_behavior_node_direct(
            payload=payload, adapter=_SkipAdapter()
        )
        self.assertEqual(result["apply_skipped_count"], 1)
        self.assertEqual(result["apply_success_count"], 0)


# ---------------------------------------------------------------------------
# Tests: Diagnostics completeness (Circle 1 DoD)
# ---------------------------------------------------------------------------

class TestDiagnosticsCompleteness(unittest.TestCase):
    """Verify that unified diagnostics cover all Circle 1 required codes."""

    def test_valid_path_includes_valid_and_apply_success_codes(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        from system.runtime.node_executor import NodeExecutor
        result = inv.invoke(
            invoke_input=BehaviorInvokeInput(
                behavior_ref="test_behavior",
                inputs={},
                context={},
                trace_id="t",
                protocol_payload=payload,
            ),
            bridge=bridge,
            sub_executor=NodeExecutor(),
        )
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.VALID, proto_codes)
        self.assertIn(ProtocolDiagCode.APPLY_SUCCESS, proto_codes)

    def test_clamped_path_includes_clamped_and_apply_success(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        from system.runtime.node_executor import NodeExecutor
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        payload["targets"][0]["limits"] = {"min": 0.0, "max": 0.3, "clamp": True}
        # source value 0.5 will be clamped to 0.3
        result = inv.invoke(
            invoke_input=BehaviorInvokeInput(
                behavior_ref="test_behavior",
                inputs={},
                context={},
                trace_id="t",
                protocol_payload=payload,
            ),
            bridge=bridge,
            sub_executor=NodeExecutor(),
        )
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.CLAMPED_VALUE, proto_codes)
        self.assertIn(ProtocolDiagCode.APPLY_SUCCESS, proto_codes)

    def test_unknown_address_target_excluded_from_apply(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        from system.runtime.node_executor import NodeExecutor
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        # Add a second target with unknown address (known_addresses = {1})
        payload["targets"].append({
            "motor_key": "leg_rr_knee",
            "param_key": "gain",
            "address": 999,
            "dtype": "float32",
            "shape": [1],
            "source": {"kind": "constant", "constant": 0.8},
        })
        # Provide known_addresses so the parser excludes address 999
        ctx = {"motor_address_registry": {1}}
        result = inv.invoke(
            invoke_input=BehaviorInvokeInput(
                behavior_ref="test_behavior",
                inputs={},
                context=ctx,
                trace_id="t",
                protocol_payload=payload,
            ),
            bridge=bridge,
            sub_executor=NodeExecutor(),
        )
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.UNKNOWN_ADDRESS, proto_codes)
        # Only 1 target resolved (address 999 excluded) → 1 apply success
        self.assertEqual(result.apply_success_count, 1)

    def test_compat_mode_includes_unknown_field_warning_in_proto_diags(self):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        from system.runtime.node_executor import NodeExecutor
        bridge = _make_stub_bridge()
        inv = BehaviorSubgraphInvoker()
        payload = _base_payload()
        payload["extra_compat_field"] = "x"
        ctx = {"protocol_compat_mode": True}
        result = inv.invoke(
            invoke_input=BehaviorInvokeInput(
                behavior_ref="test_behavior",
                inputs={},
                context=ctx,
                trace_id="t",
                protocol_payload=payload,
            ),
            bridge=bridge,
            sub_executor=NodeExecutor(),
        )
        self.assertEqual(result.status, "success")
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.UNKNOWN_FIELD, proto_codes)
        # Warning level (not error)
        uf_levels = [d.level for d in result.protocol_diagnostics
                     if d.code == ProtocolDiagCode.UNKNOWN_FIELD]
        self.assertTrue(all(l == "warning" for l in uf_levels))


# ---------------------------------------------------------------------------
# ProtocolDiagCode constants
# ---------------------------------------------------------------------------

class TestProtocolDiagCodeConstants(unittest.TestCase):
    def test_circle1_constants_exist(self):
        self.assertEqual(ProtocolDiagCode.UNKNOWN_FIELD, "protocol.unknown_field")
        self.assertEqual(ProtocolDiagCode.APPLY_SUCCESS, "protocol.apply_success")
        self.assertEqual(ProtocolDiagCode.APPLY_SKIPPED, "protocol.apply_skipped")
        self.assertEqual(ProtocolDiagCode.APPLY_FAILED, "protocol.apply_failed")


# ---------------------------------------------------------------------------
# Gap fix: blocked() carries consistent protocol_status / protocol_diagnostics
# ---------------------------------------------------------------------------

class TestBlockedProtocolFieldConsistency(unittest.TestCase):
    """Circle 1 gap fix: blocked path must expose protocol_status/protocol_diagnostics.

    Previously blocked() defaulted these to "absent"/[], making the unified
    protocol diagnostic fields inconsistent across success/failed/blocked paths.
    """

    def _invoke_with_payload(self, payload):
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        from system.behavior.behavior_artifact import BehaviorInvokeInput
        from system.runtime.node_executor import NodeExecutor
        bridge = _make_stub_bridge("test_behavior")
        inv = BehaviorSubgraphInvoker()
        return inv.invoke(
            invoke_input=BehaviorInvokeInput(
                behavior_ref="test_behavior",
                inputs={},
                context={},
                trace_id="gap-test",
                protocol_payload=payload,
            ),
            bridge=bridge,
            sub_executor=NodeExecutor(),
        )

    def test_invalid_protocol_blocked_has_protocol_status_invalid(self):
        bad = {"not": "a valid protocol"}
        result = self._invoke_with_payload(bad)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.protocol_status, "invalid")

    def test_invalid_protocol_blocked_protocol_diagnostics_non_empty(self):
        bad = {"not": "a valid protocol"}
        result = self._invoke_with_payload(bad)
        self.assertTrue(len(result.protocol_diagnostics) > 0)

    def test_stale_protocol_blocked_has_protocol_status_stale(self):
        stale = _base_payload(timestamp_ms=0, max_age_ms=1)
        result = self._invoke_with_payload(stale)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.protocol_status, "stale")

    def test_stale_protocol_blocked_protocol_diagnostics_contains_stale_code(self):
        stale = _base_payload(timestamp_ms=0, max_age_ms=1)
        result = self._invoke_with_payload(stale)
        codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.STALE, codes)

    def test_strict_unknown_field_blocked_status_is_invalid(self):
        p = _base_payload()
        p["rogue"] = "x"
        result = self._invoke_with_payload(p)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.protocol_status, "invalid")
        codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.UNKNOWN_FIELD, codes)

    def test_blocked_factory_default_still_absent(self):
        """blocked() without protocol args still defaults to absent (backward compat)."""
        out = BehaviorInvokeOutput.blocked(trace_id="t", reason="some_reason")
        self.assertEqual(out.protocol_status, "absent")
        self.assertEqual(out.protocol_diagnostics, [])

    def test_blocked_factory_accepts_protocol_fields(self):
        diag = BehaviorDiagnostic.warning(
            code=ProtocolDiagCode.STALE, message="stale", trace_id="t"
        )
        out = BehaviorInvokeOutput.blocked(
            trace_id="t",
            reason="INVALID_PROTOCOL",
            protocol_status="stale",
            protocol_diagnostics=[diag],
        )
        self.assertEqual(out.protocol_status, "stale")
        self.assertEqual(len(out.protocol_diagnostics), 1)
        self.assertEqual(out.protocol_diagnostics[0].code, ProtocolDiagCode.STALE)

    def test_to_dict_on_blocked_includes_protocol_fields(self):
        diag = BehaviorDiagnostic.error(
            code=ProtocolDiagCode.UNKNOWN_FIELD, message="bad field", trace_id="t"
        )
        out = BehaviorInvokeOutput.blocked(
            trace_id="t",
            reason="INVALID_PROTOCOL",
            protocol_status="invalid",
            protocol_diagnostics=[diag],
        )
        d = out.to_dict()
        self.assertEqual(d["protocol_status"], "invalid")
        self.assertEqual(len(d["protocol_diagnostics"]), 1)


if __name__ == "__main__":
    unittest.main()
