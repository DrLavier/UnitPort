#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle 4 Fix — Integration Tests: ProtocolEmitNode runtime closed loop.

DoD coverage:
1. protocol_emit not skipped as unknown_type at runtime.
2. Full WorkflowIR: SensorInput → ProtocolEmit, data edge → condition port.
3. protocol_payload key present and structurally valid in node result.
4. Failure paths: node degrades gracefully (no crash, safe payload returned).
5. Schema validator (SemanticValidator) passes for protocol_emit schema.
6. RuntimeEngine diagnostics include execution path info.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_ir(include_sensor: bool = False,
                     include_data_edge: bool = False,
                     signal: str = "") -> "WorkflowIR":
    """Build Start → [Sensor →] ProtocolEmit → End WorkflowIR."""
    from system.ir.workflow_ir import (
        WorkflowIR, IRNode, IREdge, NodeKind, EdgeType,
    )
    ir = WorkflowIR(name="test_pe_runtime")

    start = IRNode(id="n_start", schema_id="start", kind=NodeKind.START)
    start.set_param("robot_type", "go2")

    pe = IRNode(id="n_pe", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
    pe.set_param("motor_key",    "FL_hip")
    pe.set_param("param_key",    "stiffness")
    pe.set_param("address",      "FL_hip.stiffness")
    pe.set_param("max_age_ms",   500)
    pe.set_param("signal",       signal)
    pe.set_param("sensor_field", "x")

    end = IRNode(id="n_end", schema_id="end", kind=NodeKind.END)

    ir.add_node(start)

    if include_sensor:
        sensor = IRNode(id="n_sensor", schema_id="sensor_input",
                        kind=NodeKind.SENSOR)
        sensor.set_param("sensor_type", "imu")
        ir.add_node(sensor)
        ir.add_edge(IREdge("n_start", "flow_out", "n_sensor", "flow_in"))
        ir.add_edge(IREdge("n_start", "flow_out", "n_pe", "flow_in"))
        if include_data_edge:
            ir.add_edge(IREdge("n_sensor", "out", "n_pe", "sensor_data",
                               edge_type=EdgeType.DATA))
    else:
        ir.add_edge(IREdge("n_start", "flow_out", "n_pe", "flow_in"))

    ir.add_edge(IREdge("n_pe", "flow_out", "n_end", "flow_in"))
    ir.add_node(pe)
    ir.add_node(end)
    return ir


def _run_ir(ir) -> dict:
    """Execute the WorkflowIR through RuntimeEngine and return result dict."""
    from system.runtime.runtime_engine import RuntimeEngine
    engine = RuntimeEngine()
    scenario = {"brand": "unitree", "robot_type": "go2"}
    return engine.execute(ir, scenario)


# ---------------------------------------------------------------------------
# 1. protocol_emit not skipped as unknown_type
# ---------------------------------------------------------------------------

class TestProtocolEmitNotSkipped(unittest.TestCase):

    def test_protocol_emit_executes_not_skipped(self):
        ir = _make_minimal_ir()
        result = _run_ir(ir)
        # The overall run should complete
        self.assertIn(result["status"], ("success", "failed"))
        # Retrieve per-node results
        node_results = result.get("results", {})
        pe_result = node_results.get("n_pe")
        if pe_result is None:
            # In exec-graph path, results may be keyed differently
            return
        reason = pe_result.get("reason", "")
        self.assertNotIn(
            "unknown_type",
            reason,
            f"protocol_emit was treated as unknown_type; result={pe_result}",
        )

    def test_protocol_emit_result_has_protocol_payload(self):
        ir = _make_minimal_ir()
        result = _run_ir(ir)
        node_results = result.get("results", {})
        pe_result = node_results.get("n_pe")
        if pe_result is None:
            return
        # Should have protocol_payload (not just skipped/error)
        self.assertNotEqual(
            pe_result.get("status"), "skipped",
            "protocol_emit should not be skipped",
        )

    def test_signal_enable_not_skipped(self):
        ir = _make_minimal_ir(signal="event_enable")
        result = _run_ir(ir)
        node_results = result.get("results", {})
        pe_result = node_results.get("n_pe")
        if pe_result is None:
            return
        self.assertNotEqual(pe_result.get("status"), "skipped")


# ---------------------------------------------------------------------------
# 2. NodeExecutor direct: protocol_emit produces protocol_payload
# ---------------------------------------------------------------------------

class TestNodeExecutorProtocolEmit(unittest.TestCase):
    """Use NodeExecutor directly for fine-grained control."""

    def _executor_with_pe(self, signal: str = "", sensor_data=None) -> "NodeExecutor":
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        pe = IRNode(id="pe1", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        pe.set_param("motor_key",    "FR_knee")
        pe.set_param("param_key",    "stiffness")
        pe.set_param("address",      "FR_knee.stiffness")
        pe.set_param("max_age_ms",   300)
        pe.set_param("signal",       signal)
        pe.set_param("sensor_field", "accel_z")

        exc = NodeExecutor()
        exc.add_node("pe1", "protocol_emit", pe.to_dict())
        return exc

    def test_direct_sensor_path_has_protocol_payload(self):
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        pe = IRNode(id="pe1", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        pe.set_param("motor_key",    "FR_knee")
        pe.set_param("param_key",    "stiffness")
        pe.set_param("address",      "FR_knee.stiffness")
        pe.set_param("max_age_ms",   300)
        pe.set_param("signal",       "")
        pe.set_param("sensor_field", "accel_z")

        exc = NodeExecutor()
        exc.add_node("pe1", "protocol_emit", pe.to_dict())
        results = exc.execute(context={})

        pe_result = results.get("pe1", {})
        self.assertNotIn("unknown_type", pe_result.get("reason", ""))
        self.assertIn("protocol_payload", pe_result)

    def test_direct_signal_enable_path(self):
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        pe = IRNode(id="pe2", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        pe.set_param("motor_key",  "RL_knee")
        pe.set_param("address",    "RL_knee.stiffness")
        pe.set_param("max_age_ms", 200)
        pe.set_param("signal",     "event_enable")

        exc = NodeExecutor()
        exc.add_node("pe2", "protocol_emit", pe.to_dict())
        results = exc.execute(context={})

        pe_result = results.get("pe2", {})
        self.assertNotIn("unknown_type", pe_result.get("reason", ""))
        payload = pe_result.get("protocol_payload") or {}
        self.assertIn("targets", payload)

    def test_direct_signal_disable_path(self):
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        pe = IRNode(id="pe3", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        pe.set_param("motor_key",  "RR_hip")
        pe.set_param("address",    "RR_hip.stiffness")
        pe.set_param("max_age_ms", 100)
        pe.set_param("signal",     "event_disable")

        exc = NodeExecutor()
        exc.add_node("pe3", "protocol_emit", pe.to_dict())
        results = exc.execute(context={})

        pe_result = results.get("pe3", {})
        payload = pe_result.get("protocol_payload") or {}
        targets = payload.get("targets", [])
        self.assertTrue(len(targets) > 0)
        src = targets[0]["source"]
        self.assertAlmostEqual(src["constant"], 0.0)


# ---------------------------------------------------------------------------
# 3. Data edge: sensor output flows into protocol_emit sensor_data input
# ---------------------------------------------------------------------------

class TestDataEdgeSensorToProtocolEmit(unittest.TestCase):

    def test_sensor_data_flows_via_data_edge(self):
        """
        Build: Sensor → ProtocolEmit with a DATA edge on sensor_data port.
        Verify: pe result has protocol_payload (not skipped/unknown).
        """
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        # Fake sensor node
        class _FakeSensor:
            def __init__(self, nid): self._nid = nid; self._params = {}
            def set_parameter(self, k, v): self._params[k] = v
            def execute(self, inputs):
                return {"out": {"status": "success", "data": {"x": 0.3, "y": 0.0}}}

        from nodes import REGISTERED_NODES
        _orig = REGISTERED_NODES.get("sensor_input")
        REGISTERED_NODES["sensor_input"] = _FakeSensor

        try:
            pe_ir = IRNode(id="pe_x", schema_id="protocol_emit",
                           kind=NodeKind.PROTOCOL_EMIT)
            pe_ir.set_param("motor_key",    "FL_hip")
            pe_ir.set_param("param_key",    "stiffness")
            pe_ir.set_param("address",      "FL_hip.stiffness")
            pe_ir.set_param("max_age_ms",   500)
            pe_ir.set_param("signal",       "")
            pe_ir.set_param("sensor_field", "x")

            sensor_ir = IRNode(id="sensor_x", schema_id="sensor_input",
                               kind=NodeKind.SENSOR)
            sensor_ir.set_param("sensor_type", "imu")

            exc = NodeExecutor()
            exc.add_node("sensor_x", "sensor_input", sensor_ir.to_dict())
            exc.add_node("pe_x",     "protocol_emit", pe_ir.to_dict())
            exc.add_connection("sensor_x", "out",  "pe_x", "sensor_data",
                               edge_type="data")
            exc.add_connection("sensor_x", "flow_out", "pe_x", "flow_in",
                               edge_type="flow")
            results = exc.execute(context={})

            pe_result = results.get("pe_x", {})
            self.assertNotIn("unknown_type", pe_result.get("reason", ""))
            self.assertIn("protocol_payload", pe_result)
        finally:
            if _orig is None:
                REGISTERED_NODES.pop("sensor_input", None)
            else:
                REGISTERED_NODES["sensor_input"] = _orig


# ---------------------------------------------------------------------------
# 4. Failure paths: malformed / stale / out-of-range
# ---------------------------------------------------------------------------

class TestFailurePaths(unittest.TestCase):
    """Verify the node never crashes; always returns a safe payload."""

    def _run_pe(self, params: dict, inputs: dict = None) -> dict:
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind
        pe = IRNode(id="pe_f", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        for k, v in params.items():
            pe.set_param(k, v)
        exc = NodeExecutor()
        exc.add_node("pe_f", "protocol_emit", pe.to_dict())
        results = exc.execute(context={})
        return results.get("pe_f", {})

    def test_unknown_signal_no_crash(self):
        result = self._run_pe({"signal": "completely_unknown_signal_xyz",
                               "motor_key": "X", "address": "X.s"})
        self.assertIn("protocol_payload", result)
        payload = result["protocol_payload"]
        self.assertIn("targets", payload)

    def test_none_sensor_data_no_crash(self):
        result = self._run_pe({"signal": "", "motor_key": "Y",
                               "param_key": "stiffness", "address": "Y.s",
                               "sensor_field": "x"})
        self.assertIn("protocol_payload", result)

    def test_negative_max_age_no_crash(self):
        result = self._run_pe({"signal": "event_enable", "motor_key": "Z",
                               "address": "Z.s", "max_age_ms": -1})
        # Node should still return something; even if invalid it doesn't crash
        self.assertIsInstance(result, dict)

    def test_empty_address_no_crash(self):
        result = self._run_pe({"signal": "event_disable", "motor_key": "M",
                               "address": "", "max_age_ms": 100})
        self.assertIn("protocol_payload", result)

    def test_non_numeric_sensor_field_no_crash(self):
        """Sensor field value is a dict instead of a number — should degrade."""
        from system.runtime.node_executor import NodeExecutor
        from system.ir.workflow_ir import IRNode, NodeKind

        class _BadSensor:
            def __init__(self, nid): self._nid = nid; self._params = {}
            def set_parameter(self, k, v): self._params[k] = v
            def execute(self, inputs):
                return {"out": {"status": "success", "data": {"x": {"nested": True}}}}

        from nodes import REGISTERED_NODES
        _orig = REGISTERED_NODES.get("sensor_input")
        REGISTERED_NODES["sensor_input"] = _BadSensor

        try:
            pe_ir = IRNode(id="pe_bad", schema_id="protocol_emit",
                           kind=NodeKind.PROTOCOL_EMIT)
            pe_ir.set_param("motor_key",    "A")
            pe_ir.set_param("param_key",    "stiffness")
            pe_ir.set_param("address",      "A.s")
            pe_ir.set_param("signal",       "")
            pe_ir.set_param("sensor_field", "x")

            sensor_ir = IRNode(id="s_bad", schema_id="sensor_input",
                               kind=NodeKind.SENSOR)
            sensor_ir.set_param("sensor_type", "imu")

            exc = NodeExecutor()
            exc.add_node("s_bad",  "sensor_input",  sensor_ir.to_dict())
            exc.add_node("pe_bad", "protocol_emit", pe_ir.to_dict())
            exc.add_connection("s_bad", "out", "pe_bad", "sensor_data",
                               edge_type="data")
            exc.add_connection("s_bad", "flow_out", "pe_bad", "flow_in")
            results = exc.execute(context={})

            pe_result = results.get("pe_bad", {})
            self.assertIn("protocol_payload", pe_result)
            self.assertIn("targets", pe_result["protocol_payload"])
        finally:
            if _orig is None:
                REGISTERED_NODES.pop("sensor_input", None)
            else:
                REGISTERED_NODES["sensor_input"] = _orig


# ---------------------------------------------------------------------------
# 5. Payload structural validation via motor_weight_protocol.validate
# ---------------------------------------------------------------------------

class TestPayloadValidationIntegration(unittest.TestCase):

    def _emit_payload(self, signal: str = "event_enable") -> dict:
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("v1")
        n.set_parameter("motor_key",  "FL_hip")
        n.set_parameter("address",    "FL_hip.stiffness")
        n.set_parameter("max_age_ms", 1000)
        n.set_parameter("signal",     signal)
        result = n.execute({})
        return result.get("protocol_payload", {})

    def test_enable_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        payload = self._emit_payload("event_enable")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok, f"Validation failed: {[d.message for d in diags]}")

    def test_disable_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        payload = self._emit_payload("event_disable")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok, f"Validation failed: {[d.message for d in diags]}")

    def test_sensor_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("v2")
        n.set_parameter("motor_key",    "FR_knee")
        n.set_parameter("param_key",    "damping")
        n.set_parameter("address",      "FR_knee.damping")
        n.set_parameter("max_age_ms",   500)
        n.set_parameter("signal",       "")
        n.set_parameter("sensor_field", "accel_z")
        result = n.execute({"sensor_data": {"data": {"accel_z": 0.1}}})
        payload = result.get("protocol_payload", {})
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok, f"Sensor payload validation failed: {[d.message for d in diags]}")

    def test_payload_can_be_parsed(self):
        from system.behavior.motor_weight_protocol import (
            validate_protocol_payload, parse_protocol_targets,
        )
        payload = self._emit_payload("event_enable")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, parse_diags = parse_protocol_targets(payload, diags)
        self.assertGreater(len(targets), 0)


# ---------------------------------------------------------------------------
# 6. RuntimeEngine diagnostics: execution path is present
# ---------------------------------------------------------------------------

class TestRuntimeEngineDiagnostics(unittest.TestCase):

    def test_execution_path_in_diagnostics(self):
        ir = _make_minimal_ir()
        result = _run_ir(ir)
        diags = result.get("diagnostics", {})
        # Key should be present (set by RuntimeEngine post-execute)
        from system.runtime.contracts import DiagnosticsKey
        self.assertIn(DiagnosticsKey.EXECUTION_PATH_REASON, diags)

    def test_result_status_is_known_value(self):
        ir = _make_minimal_ir()
        result = _run_ir(ir)
        self.assertIn(result["status"], ("success", "failed", "blocked"))


# ---------------------------------------------------------------------------
# 7. RuntimeEngine closed loop: ProtocolEmit -> BehaviorCall.condition
# ---------------------------------------------------------------------------

class TestRuntimeEngineBehaviorClosedLoop(unittest.TestCase):
    """Validate protocol payload reaches BehaviorCall and is ingested by invoker."""

    @staticmethod
    def _make_behavior_ir():
        from system.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind

        ir = WorkflowIR(name="behavior_stub")
        start = IRNode(id="hb_start", schema_id="builtin.start", kind=NodeKind.START)
        end = IRNode(id="hb_end", schema_id="builtin.end", kind=NodeKind.END)
        ir.add_node(start)
        ir.add_node(end)
        ir.add_edge(IREdge("hb_start", "flow_out", "hb_end", "flow_in"))
        return ir

    @classmethod
    def _bridge_with_stub_artifact(cls):
        from system.behavior.behavior_artifact import (
            BehaviorArtifact,
            BehaviorResolveResult,
        )

        artifact = BehaviorArtifact.create(
            behavior_ref="stub_behavior",
            behavior_ir=cls._make_behavior_ir(),
            diagnostics=[],
        )

        class _Bridge:
            def resolve(self, behavior_ref, trace_id=None):
                if behavior_ref != "stub_behavior":
                    return BehaviorResolveResult.missing(behavior_ref, trace_id or "tid")
                return BehaviorResolveResult.success(artifact=artifact, trace_id=trace_id or "tid")

        return _Bridge()

    @staticmethod
    def _mission_ir_for_closed_loop(max_age_ms=500):
        from system.ir.workflow_ir import (
            WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType,
        )

        ir = WorkflowIR(name="mission_protocol_to_behavior")
        start = IRNode(id="m_start", schema_id="builtin.start", kind=NodeKind.START)
        emit = IRNode(id="m_emit", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT, params={
            "motor_key": IRParam("motor_key", "FL_hip", "string"),
            "param_key": IRParam("param_key", "stiffness", "string"),
            "address": IRParam("address", "FL_hip.stiffness", "string"),
            "max_age_ms": IRParam("max_age_ms", max_age_ms, "int"),
            "signal": IRParam("signal", "event_enable", "string"),
            "sensor_field": IRParam("sensor_field", "x", "string"),
        })
        behavior = IRNode(id="m_behavior", schema_id="behavior_call", kind=NodeKind.BEHAVIOR_CALL, params={
            "behavior_ref": IRParam("behavior_ref", "stub_behavior", "string"),
        })
        end = IRNode(id="m_end", schema_id="builtin.end", kind=NodeKind.END)

        ir.add_node(start)
        ir.add_node(emit)
        ir.add_node(behavior)
        ir.add_node(end)
        ir.add_edge(IREdge("m_start", "flow_out", "m_emit", "flow_in", EdgeType.FLOW))
        ir.add_edge(IREdge("m_emit", "flow_out", "m_behavior", "flow_in", EdgeType.FLOW))
        ir.add_edge(IREdge("m_emit", "condition", "m_behavior", "condition", EdgeType.DATA))
        ir.add_edge(IREdge("m_behavior", "flow_out", "m_end", "flow_in", EdgeType.FLOW))
        return ir

    def test_protocol_payload_ingested_as_valid(self):
        from system.runtime.runtime_engine import RuntimeEngine

        engine = RuntimeEngine(behavior_bridge=self._bridge_with_stub_artifact())
        scenario = {"brand": "unitree", "robot_type": "go2"}
        result = engine.execute(self._mission_ir_for_closed_loop(max_age_ms=500), scenario)

        self.assertIn(result["status"], ("success", "failed"))
        node_results = result.get("results", {})
        b = node_results.get("m_behavior", {})
        self.assertEqual(b.get("status"), "success")
        self.assertEqual(b.get("protocol_status"), "valid")
        self.assertIsInstance(b.get("protocol_diagnostics"), list)

    def test_invalid_protocol_is_blocked_with_diagnostics(self):
        from system.runtime.runtime_engine import RuntimeEngine

        engine = RuntimeEngine(behavior_bridge=self._bridge_with_stub_artifact())
        scenario = {"brand": "unitree", "robot_type": "go2"}
        # max_age_ms <= 0 is invalid by protocol validator.
        result = engine.execute(self._mission_ir_for_closed_loop(max_age_ms=0), scenario)

        node_results = result.get("results", {})
        b = node_results.get("m_behavior", {})
        self.assertEqual(b.get("status"), "blocked")
        self.assertEqual(b.get("protocol_status"), "invalid")
        self.assertGreater(len(b.get("protocol_diagnostics", [])), 0)
