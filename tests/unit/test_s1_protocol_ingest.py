#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 1 — Behavior External Interface: Optional Protocol Ingest.

Coverage
--------
motor_weight_protocol.validate_protocol_payload():
  - valid payload passes → (True, [info diag])
  - missing required fields → (False, [error diag])
  - wrong protocol_version → (False, [error diag])
  - invalid mode → (False, [error diag])
  - stale payload → (False, [warning with STALE code])
  - non-dict payload → (False, [error diag])
  - invalid target (missing field) → (False, [error diag])
  - constant source missing value → (False, [error diag])
  - external source missing external_value → (False, [error diag])
  - clamped value emits CLAMPED_VALUE diagnostic

motor_weight_protocol.parse_protocol_targets():
  - constant source resolves final_value correctly
  - external source uses external_value
  - clamped scalar emits CLAMPED_VALUE diagnostic
  - unclamped value (clamp=False) not changed
  - vector value clamped correctly
  - no limits → final_value unchanged

BehaviorInvokeInput:
  - backward compat: protocol_payload defaults to None
  - to_dict includes protocol_payload_present flag

BehaviorInvokeOutput:
  - protocol_status defaults to "absent"
  - protocol_diagnostics defaults to empty list
  - success/failed factories accept protocol_status/diagnostics kwargs
  - to_dict includes protocol_status and protocol_diagnostics

BehaviorSubgraphInvoker.invoke():
  - absent protocol → status "absent", no protocol diags, sub_context has protocol_targets=None
  - valid protocol → status "valid", clamped diag, sub_context has protocol_targets list
  - invalid protocol → returns blocked with INVALID_PROTOCOL reason
  - stale protocol → returns blocked with INVALID_PROTOCOL reason

DiagnosticsKey:
  - PROTOCOL_STATUS and PROTOCOL_DIAGNOSTICS constants defined
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
from system.behavior.behavior_artifact import (
    BehaviorArtifact,
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
from system.runtime.contracts import DiagnosticsKey
from system.ir.workflow_ir import IREdge, IRNode, NodeKind, WorkflowIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_payload(*, now_ms: int = 0, age_offset: int = 0) -> dict:
    """Return a minimal valid protocol payload."""
    return {
        "protocol_version": "1.0",
        "trace_id": "test-trace-001",
        "timestamp_ms": now_ms,
        "mode": "constant",
        "max_age_ms": 5000,
        "targets": [
            {
                "motor_key": "leg_fl_hip",
                "param_key": "gain",
                "address": "motors.0.gain",
                "dtype": "float32",
                "shape": [1],
                "source": {"kind": "constant", "constant": 1.0},
            }
        ],
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Fake node registry for invoker tests
# ---------------------------------------------------------------------------

_registry: dict = {}
_nodes_mod = types.ModuleType("nodes")
_nodes_mod.get_node_class = lambda name: _registry.get(name)


class OkNode:
    def __init__(self, nid): self.nid = nid; self._params = {}
    def set_parameter(self, k, v): self._params[k] = v
    def execute(self, inputs): return {"status": "ok"}


def _install_registry():
    _registry.clear()
    _registry["ok_node"] = OkNode
    sys.modules["nodes"] = _nodes_mod


def _make_bridge_with_artifact() -> BehaviorCompilerBridge:
    bridge = BehaviorCompilerBridge()
    ir = WorkflowIR(
        nodes=[IRNode(id="n0", schema_id="ok_node", kind=NodeKind.ACTION, params={})],
        edges=[],
    )
    artifact = BehaviorArtifact.create(
        behavior_ref="test_ref",
        behavior_ir=ir,
        diagnostics=[],
    )
    bridge.register(artifact)
    return bridge


# ---------------------------------------------------------------------------
# Tests: validate_protocol_payload
# ---------------------------------------------------------------------------

class TestValidateProtocolPayload(unittest.TestCase):

    def test_valid_payload_returns_true(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertTrue(ok)
        codes = [d.code for d in diags]
        self.assertIn(ProtocolDiagCode.VALID, codes)

    def test_non_dict_returns_false(self):
        ok, diags = validate_protocol_payload("not a dict")
        self.assertFalse(ok)
        self.assertTrue(any(d.code == ProtocolDiagCode.INVALID_SCHEMA for d in diags))

    def test_missing_required_field(self):
        payload = _minimal_payload(now_ms=_now_ms())
        del payload["targets"]
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == ProtocolDiagCode.INVALID_SCHEMA for d in diags))

    def test_wrong_protocol_version(self):
        payload = _minimal_payload(now_ms=_now_ms())
        payload["protocol_version"] = "2.0"
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("unsupported protocol_version" in d.message for d in diags))

    def test_invalid_mode(self):
        payload = _minimal_payload(now_ms=_now_ms())
        payload["mode"] = "unknown_mode"
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any("mode" in d.message for d in diags))

    def test_stale_payload_returns_false_with_stale_code(self):
        # timestamp far in the past
        old_ts = _now_ms() - 100_000  # 100 seconds ago
        payload = _minimal_payload(now_ms=old_ts)
        payload["max_age_ms"] = 50  # only 50ms tolerance
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)
        self.assertTrue(any(d.code == ProtocolDiagCode.STALE for d in diags))

    def test_fresh_payload_not_stale(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertTrue(ok)
        self.assertFalse(any(d.code == ProtocolDiagCode.STALE for d in diags))

    def test_invalid_target_missing_field(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        del payload["targets"][0]["motor_key"]
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)
        self.assertTrue(any("motor_key" in d.message for d in diags))

    def test_constant_source_missing_constant(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"][0]["source"] = {"kind": "constant"}  # missing constant
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)
        self.assertTrue(any("constant" in d.message for d in diags))

    def test_external_source_missing_external_value(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"][0]["source"] = {"kind": "external", "external_key": "foo"}
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)
        self.assertTrue(any("external_value" in d.message for d in diags))

    def test_empty_targets_invalid(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"] = []
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)

    def test_negative_timestamp_invalid(self):
        payload = _minimal_payload(now_ms=0)
        payload["timestamp_ms"] = -1
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)

    def test_invalid_dtype(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"][0]["dtype"] = "int32"
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)

    def test_invalid_shape(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"][0]["shape"] = [0]  # must be >= 1
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)

    def test_trace_id_empty_string_invalid(self):
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["trace_id"] = ""
        ok, diags = validate_protocol_payload(payload, now_ms=now)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# Tests: parse_protocol_targets
# ---------------------------------------------------------------------------

class TestParseProtocolTargets(unittest.TestCase):

    def _valid_payload(self, **overrides) -> dict:
        now = _now_ms()
        p = _minimal_payload(now_ms=now)
        p.update(overrides)
        return p

    def test_constant_source_resolves_value(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": 0.75}
        targets, diags = parse_protocol_targets(payload)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].final_value, 0.75)
        self.assertFalse(targets[0].was_clamped)
        self.assertEqual(targets[0].source_kind, "constant")

    def test_external_source_uses_external_value(self):
        payload = self._valid_payload()
        payload["mode"] = "external"
        payload["targets"][0]["source"] = {
            "kind": "external",
            "external_key": "balance.gain",
            "external_value": 1.2,
        }
        targets, diags = parse_protocol_targets(payload)
        self.assertEqual(targets[0].final_value, 1.2)
        self.assertEqual(targets[0].source_kind, "external")

    def test_clamp_scalar_above_max(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": 2.5}
        payload["targets"][0]["limits"] = {"min": 0.5, "max": 1.5, "clamp": True}
        targets, diags = parse_protocol_targets(payload)
        self.assertAlmostEqual(targets[0].final_value, 1.5)
        self.assertTrue(targets[0].was_clamped)
        self.assertTrue(any(d.code == ProtocolDiagCode.CLAMPED_VALUE for d in diags))

    def test_clamp_scalar_below_min(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": 0.1}
        payload["targets"][0]["limits"] = {"min": 0.5, "max": 1.5, "clamp": True}
        targets, diags = parse_protocol_targets(payload)
        self.assertAlmostEqual(targets[0].final_value, 0.5)
        self.assertTrue(targets[0].was_clamped)

    def test_no_clamp_when_clamp_false(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": 99.0}
        payload["targets"][0]["limits"] = {"min": 0.5, "max": 1.5, "clamp": False}
        targets, diags = parse_protocol_targets(payload)
        self.assertEqual(targets[0].final_value, 99.0)
        self.assertFalse(targets[0].was_clamped)
        self.assertFalse(any(d.code == ProtocolDiagCode.CLAMPED_VALUE for d in diags))

    def test_no_limits_value_unchanged(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": 42.0}
        targets, diags = parse_protocol_targets(payload)
        self.assertEqual(targets[0].final_value, 42.0)
        self.assertFalse(targets[0].was_clamped)

    def test_vector_value_clamped(self):
        payload = self._valid_payload()
        payload["targets"][0]["source"] = {"kind": "constant", "constant": [0.1, 2.0, 1.0]}
        payload["targets"][0]["shape"] = [3]
        payload["targets"][0]["limits"] = {"min": 0.5, "max": 1.5, "clamp": True}
        targets, diags = parse_protocol_targets(payload)
        result = targets[0].final_value
        self.assertEqual(result[0], 0.5)  # clamped up
        self.assertEqual(result[1], 1.5)  # clamped down
        self.assertEqual(result[2], 1.0)  # unchanged
        self.assertTrue(targets[0].was_clamped)

    def test_motor_and_param_key_preserved(self):
        payload = self._valid_payload()
        targets, _ = parse_protocol_targets(payload)
        self.assertEqual(targets[0].motor_key, "leg_fl_hip")
        self.assertEqual(targets[0].param_key, "gain")
        self.assertEqual(targets[0].address, "motors.0.gain")


# ---------------------------------------------------------------------------
# Tests: BehaviorInvokeInput backward compat
# ---------------------------------------------------------------------------

class TestBehaviorInvokeInputCompat(unittest.TestCase):

    def test_protocol_payload_defaults_to_none(self):
        inp = BehaviorInvokeInput(behavior_ref="foo")
        self.assertIsNone(inp.protocol_payload)

    def test_to_dict_includes_protocol_payload_present_false(self):
        inp = BehaviorInvokeInput(behavior_ref="foo")
        d = inp.to_dict()
        self.assertIn("protocol_payload_present", d)
        self.assertFalse(d["protocol_payload_present"])

    def test_to_dict_includes_protocol_payload_present_true(self):
        inp = BehaviorInvokeInput(behavior_ref="foo", protocol_payload={"some": "data"})
        d = inp.to_dict()
        self.assertTrue(d["protocol_payload_present"])


# ---------------------------------------------------------------------------
# Tests: BehaviorInvokeOutput
# ---------------------------------------------------------------------------

class TestBehaviorInvokeOutputProtocol(unittest.TestCase):

    def test_default_protocol_status_absent(self):
        out = BehaviorInvokeOutput(status="success")
        self.assertEqual(out.protocol_status, "absent")
        self.assertEqual(out.protocol_diagnostics, [])

    def test_success_factory_accepts_protocol_kwargs(self):
        diag = BehaviorDiagnostic.info(code="protocol.valid", message="ok")
        out = BehaviorInvokeOutput.success(
            trace_id="t1",
            protocol_status="valid",
            protocol_diagnostics=[diag],
        )
        self.assertEqual(out.protocol_status, "valid")
        self.assertEqual(len(out.protocol_diagnostics), 1)

    def test_failed_factory_accepts_protocol_kwargs(self):
        out = BehaviorInvokeOutput.failed(
            trace_id="t1",
            reason="INVALID_PROTOCOL",
            protocol_status="invalid",
        )
        self.assertEqual(out.protocol_status, "invalid")

    def test_to_dict_includes_protocol_fields(self):
        out = BehaviorInvokeOutput.success(trace_id="t1", protocol_status="valid")
        d = out.to_dict()
        self.assertIn("protocol_status", d)
        self.assertIn("protocol_diagnostics", d)
        self.assertEqual(d["protocol_status"], "valid")

    def test_blocked_factory_protocol_status_absent(self):
        out = BehaviorInvokeOutput.blocked(trace_id="t1", reason="ref_not_found")
        # blocked factory does not accept protocol_status — defaults to "absent"
        self.assertEqual(out.protocol_status, "absent")


# ---------------------------------------------------------------------------
# Tests: DiagnosticsKey
# ---------------------------------------------------------------------------

class TestDiagnosticsKeyProtocol(unittest.TestCase):

    def test_protocol_status_constant_defined(self):
        self.assertEqual(DiagnosticsKey.PROTOCOL_STATUS, "protocol_status")

    def test_protocol_diagnostics_constant_defined(self):
        self.assertEqual(DiagnosticsKey.PROTOCOL_DIAGNOSTICS, "protocol_diagnostics")


# ---------------------------------------------------------------------------
# Tests: BehaviorSubgraphInvoker protocol paths
# ---------------------------------------------------------------------------

class TestInvokerProtocolPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _install_registry()

    def _invoker_and_bridge(self):
        invoker = BehaviorSubgraphInvoker()
        bridge = _make_bridge_with_artifact()
        return invoker, bridge

    def _make_sub_executor(self):
        from system.runtime.node_executor import NodeExecutor
        return NodeExecutor()

    # ------------------------------------------------------------------
    # Case 1: no protocol payload → legacy path
    # ------------------------------------------------------------------
    def test_absent_protocol_success(self):
        invoker, bridge = self._invoker_and_bridge()
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=None,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.protocol_status, "absent")
        self.assertEqual(result.protocol_diagnostics, [])

    # ------------------------------------------------------------------
    # Case 2: valid protocol payload → status "valid"
    # ------------------------------------------------------------------
    def test_valid_protocol_success(self):
        invoker, bridge = self._invoker_and_bridge()
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=payload,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.protocol_status, "valid")
        # Should have at least the VALID info diagnostic
        codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.VALID, codes)

    # ------------------------------------------------------------------
    # Case 3: invalid protocol → blocked with INVALID_PROTOCOL
    # ------------------------------------------------------------------
    def test_invalid_protocol_blocked(self):
        invoker, bridge = self._invoker_and_bridge()
        invalid_payload = {"protocol_version": "1.0"}  # missing required fields
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=invalid_payload,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.reason, "INVALID_PROTOCOL")
        self.assertTrue(len(result.diagnostics) > 0)

    # ------------------------------------------------------------------
    # Case 4: stale protocol → blocked
    # ------------------------------------------------------------------
    def test_stale_protocol_blocked(self):
        invoker, bridge = self._invoker_and_bridge()
        old_ts = _now_ms() - 100_000  # 100 seconds ago
        payload = _minimal_payload(now_ms=old_ts)
        payload["max_age_ms"] = 50  # 50ms tolerance
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=payload,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.reason, "INVALID_PROTOCOL")
        stale_codes = [d.code for d in result.diagnostics]
        self.assertIn(ProtocolDiagCode.STALE, stale_codes)

    # ------------------------------------------------------------------
    # Case 5: valid protocol with clamped value → clamped diag
    # ------------------------------------------------------------------
    def test_valid_protocol_with_clamped_value(self):
        invoker, bridge = self._invoker_and_bridge()
        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        payload["targets"][0]["source"]["constant"] = 3.0
        payload["targets"][0]["limits"] = {"min": 0.5, "max": 1.5, "clamp": True}
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=payload,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.protocol_status, "valid")
        codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.CLAMPED_VALUE, codes)

    # ------------------------------------------------------------------
    # Case 6: protocol with non-dict value → treated as absent (no crash)
    # ------------------------------------------------------------------
    def test_non_dict_protocol_in_node_inputs_treated_as_absent(self):
        # node_executor normalises non-dict to None before building BehaviorInvokeInput;
        # test that supplying None directly still works.
        invoker, bridge = self._invoker_and_bridge()
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=None,
        )
        result = invoker.invoke(inp, bridge, self._make_sub_executor())
        self.assertEqual(result.status, "success")
        self.assertEqual(result.protocol_status, "absent")


# ---------------------------------------------------------------------------
# Tests: unknown_address via known_addresses parameter
# ---------------------------------------------------------------------------

class TestUnknownAddress(unittest.TestCase):
    """ProtocolDiagCode.UNKNOWN_ADDRESS is emitted by parse_protocol_targets()
    when an address is not in the caller-supplied known_addresses set."""

    def _valid_payload_with_address(self, address) -> dict:
        now = _now_ms()
        p = _minimal_payload(now_ms=now)
        p["targets"][0]["address"] = address
        return p

    def test_known_address_no_warning(self):
        payload = self._valid_payload_with_address("motors.0.gain")
        targets, diags = parse_protocol_targets(
            payload,
            known_addresses={"motors.0.gain", "motors.1.gain"},
        )
        self.assertEqual(len(targets), 1)
        self.assertFalse(any(d.code == ProtocolDiagCode.UNKNOWN_ADDRESS for d in diags))

    def test_unknown_address_emits_warning(self):
        payload = self._valid_payload_with_address("motors.99.gain")
        targets, diags = parse_protocol_targets(
            payload,
            known_addresses={"motors.0.gain"},
        )
        # Target with unknown address is NOT added to ParsedTarget list.
        self.assertEqual(len(targets), 0)
        self.assertTrue(any(d.code == ProtocolDiagCode.UNKNOWN_ADDRESS for d in diags))

    def test_unknown_address_warning_level(self):
        payload = self._valid_payload_with_address("bad.address")
        _, diags = parse_protocol_targets(
            payload,
            known_addresses={"motors.0.gain"},
        )
        unknown_diags = [d for d in diags if d.code == ProtocolDiagCode.UNKNOWN_ADDRESS]
        self.assertTrue(len(unknown_diags) > 0)
        self.assertEqual(unknown_diags[0].level, "warning")

    def test_no_known_addresses_no_check(self):
        """When known_addresses=None (default), no UNKNOWN_ADDRESS emitted."""
        payload = self._valid_payload_with_address("any.arbitrary.address")
        targets, diags = parse_protocol_targets(payload, known_addresses=None)
        self.assertEqual(len(targets), 1)
        self.assertFalse(any(d.code == ProtocolDiagCode.UNKNOWN_ADDRESS for d in diags))

    def test_numeric_address_unknown(self):
        payload = self._valid_payload_with_address(999)
        targets, diags = parse_protocol_targets(
            payload,
            known_addresses={0, 1, 2},
        )
        self.assertEqual(len(targets), 0)
        self.assertTrue(any(d.code == ProtocolDiagCode.UNKNOWN_ADDRESS for d in diags))

    def test_numeric_address_known(self):
        payload = self._valid_payload_with_address(1)
        targets, diags = parse_protocol_targets(
            payload,
            known_addresses={0, 1, 2},
        )
        self.assertEqual(len(targets), 1)
        self.assertFalse(any(d.code == ProtocolDiagCode.UNKNOWN_ADDRESS for d in diags))

    def test_invoker_passes_known_addresses_from_context(self):
        """When context has motor_address_registry with correct addresses,
        no UNKNOWN_ADDRESS is emitted."""
        _install_registry()
        invoker = BehaviorSubgraphInvoker()
        bridge = _make_bridge_with_artifact()
        from system.runtime.node_executor import NodeExecutor

        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        # address in payload is "motors.0.gain" — known
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=payload,
            context={"motor_address_registry": {"motors.0.gain", "motors.1.gain"}},
        )
        result = invoker.invoke(inp, bridge, NodeExecutor())
        self.assertEqual(result.status, "success")
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertNotIn(ProtocolDiagCode.UNKNOWN_ADDRESS, proto_codes)

    def test_invoker_unknown_address_from_context(self):
        """When context motor_address_registry excludes the target address,
        UNKNOWN_ADDRESS is emitted in protocol_diagnostics."""
        _install_registry()
        invoker = BehaviorSubgraphInvoker()
        bridge = _make_bridge_with_artifact()
        from system.runtime.node_executor import NodeExecutor

        now = _now_ms()
        payload = _minimal_payload(now_ms=now)
        # address is "motors.0.gain" but registry only has "motors.1.gain"
        inp = BehaviorInvokeInput(
            behavior_ref="test_ref",
            protocol_payload=payload,
            context={"motor_address_registry": {"motors.1.gain"}},
        )
        result = invoker.invoke(inp, bridge, NodeExecutor())
        self.assertEqual(result.status, "success")
        proto_codes = [d.code for d in result.protocol_diagnostics]
        self.assertIn(ProtocolDiagCode.UNKNOWN_ADDRESS, proto_codes)


# ---------------------------------------------------------------------------
# Tests: RuntimeResult.success/failed include PROTOCOL_DIAGNOSTICS key
# ---------------------------------------------------------------------------

class TestRuntimeResultProtocolDiagnostics(unittest.TestCase):
    """RuntimeResult factories must populate DiagnosticsKey.PROTOCOL_DIAGNOSTICS."""

    def _make_success(self, protocol_diagnostics=None):
        from system.runtime.contracts import RuntimeResult, ExecutionPath
        return RuntimeResult.success(
            task_id="t1",
            node_count=1,
            results={},
            metrics={},
            path=ExecutionPath.WORKFLOW_IR,
            protocol_diagnostics=protocol_diagnostics,
        )

    def _make_failed(self, protocol_diagnostics=None):
        from system.runtime.contracts import RuntimeResult, ExecutionPath
        return RuntimeResult.failed(
            task_id="t1",
            node_count=1,
            results={},
            metrics={},
            reason="node_execution_failed",
            path=ExecutionPath.WORKFLOW_IR,
            protocol_diagnostics=protocol_diagnostics,
        )

    def test_success_has_protocol_diagnostics_key(self):
        result = self._make_success()
        self.assertIn(DiagnosticsKey.PROTOCOL_DIAGNOSTICS, result.diagnostics)

    def test_success_protocol_diagnostics_empty_by_default(self):
        result = self._make_success()
        self.assertEqual(result.diagnostics[DiagnosticsKey.PROTOCOL_DIAGNOSTICS], [])

    def test_success_protocol_diagnostics_populated(self):
        diag = {"code": "protocol.clamped_value", "level": "info", "message": "clamped"}
        result = self._make_success(protocol_diagnostics=[diag])
        self.assertEqual(
            result.diagnostics[DiagnosticsKey.PROTOCOL_DIAGNOSTICS],
            [diag],
        )

    def test_failed_has_protocol_diagnostics_key(self):
        result = self._make_failed()
        self.assertIn(DiagnosticsKey.PROTOCOL_DIAGNOSTICS, result.diagnostics)

    def test_failed_protocol_diagnostics_populated(self):
        diag = {"code": "protocol.stale", "level": "warning", "message": "stale"}
        result = self._make_failed(protocol_diagnostics=[diag])
        self.assertEqual(
            result.diagnostics[DiagnosticsKey.PROTOCOL_DIAGNOSTICS],
            [diag],
        )


if __name__ == "__main__":
    unittest.main()
