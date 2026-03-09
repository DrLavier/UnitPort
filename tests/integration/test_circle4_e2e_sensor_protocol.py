"""Integration tests: Circle 4 — sensor → protocol → behavior apply pipeline.

DoD coverage:
1. End-to-end sample: sensor(imu/gyro) → build payload → validate → parse → apply.
2. Failure paths: malformed, stale, out-of-range covered.
3. Signal-convention payloads validate and apply correctly.
4. WorkflowIR with PROTOCOL_EMIT node compiles to runnable Python code.
5. Codegen output is syntactically valid and contains expected calls.
"""

from __future__ import annotations

import time
import unittest

from compiler.lowering.protocol_compiler import (
    SIGNAL_DISABLE,
    SIGNAL_ENABLE,
    SIGNAL_SWITCH,
    ProtocolPayloadCompiler,
    ProtocolTargetSpec,
)
from system.behavior.motor_weight_protocol import (
    validate_protocol_payload,
    parse_protocol_targets,
)


# ---------------------------------------------------------------------------
# 1. IMU sensor → protocol → validate → parse  (happy path)
# ---------------------------------------------------------------------------

class TestSensorImuToProtocol(unittest.TestCase):
    """Full sensor(imu) → ProtocolPayloadCompiler → validate → parse pipeline."""

    def _imu_data(self, accel_z: float = 0.15) -> dict:
        return {"accel_z": accel_z, "accel_x": 0.0, "accel_y": 0.0,
                "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0}

    def test_imu_accel_z_to_gain(self):
        sensor_data = self._imu_data(accel_z=0.42)
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_fl_hip", param_key="gain", sensor_field="accel_z"
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok, f"Validation errors: {[d.message for d in diags]}")

    def test_imu_gyro_z_to_stiffness(self):
        sensor_data = {"gyro_z": -0.05}
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_rr_knee", param_key="stiffness",
            sensor_field="gyro_z"
        )
        ok, _ = validate_protocol_payload(payload)
        self.assertTrue(ok)

    def test_parse_targets_after_validation(self):
        sensor_data = self._imu_data(accel_z=0.3)
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_fl_hip", param_key="gain"
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, parse_diags = parse_protocol_targets(payload, diags)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].motor_key, "leg_fl_hip")
        self.assertEqual(targets[0].param_key, "gain")
        self.assertAlmostEqual(targets[0].final_value, 0.3, places=5)

    def test_imu_value_preserved_in_parsed_target(self):
        sensor_data = {"accel_z": 0.77}
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="m1", param_key="gain", sensor_field="accel_z"
        )
        _, diags = validate_protocol_payload(payload)
        targets, _ = parse_protocol_targets(payload, diags)
        self.assertAlmostEqual(targets[0].final_value, 0.77, places=5)

    def test_limits_clamp_applied(self):
        sensor_data = {"accel_z": 5.0}  # Out of limits
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="m1", param_key="gain", sensor_field="accel_z",
            limits={"min": 0.0, "max": 1.0, "clamp": True}
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)
        self.assertEqual(len(targets), 1)
        self.assertTrue(targets[0].was_clamped)
        self.assertLessEqual(targets[0].final_value, 1.0)


# ---------------------------------------------------------------------------
# 2. Signal convention → validate → parse
# ---------------------------------------------------------------------------

class TestSignalConventionPipeline(unittest.TestCase):

    def test_enable_validates_and_parses(self):
        payload = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="leg_fl")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)
        self.assertEqual(targets[0].param_key, "stiffness")
        self.assertEqual(targets[0].final_value, 1.0)

    def test_disable_validates_and_parses(self):
        payload = ProtocolPayloadCompiler.for_signal(SIGNAL_DISABLE, motor_key="leg_rr")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)
        self.assertEqual(targets[0].param_key, "stiffness")
        self.assertEqual(targets[0].final_value, 0.0)

    def test_switch_validates_and_parses(self):
        payload = ProtocolPayloadCompiler.for_signal(
            SIGNAL_SWITCH, motor_key="leg_fl", switch_value=2.0
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)
        self.assertEqual(targets[0].param_key, "mode")
        self.assertAlmostEqual(targets[0].final_value, 2.0, places=5)

    def test_all_signals_produce_valid_payloads(self):
        for signal in (SIGNAL_ENABLE, SIGNAL_DISABLE):
            with self.subTest(signal=signal):
                p = ProtocolPayloadCompiler.for_signal(signal, motor_key="m")
                ok, _ = validate_protocol_payload(p)
                self.assertTrue(ok, f"{signal} payload failed validation")
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_SWITCH, motor_key="m", switch_value=0.5)
        ok, _ = validate_protocol_payload(p)
        self.assertTrue(ok, "SIGNAL_SWITCH payload failed validation")


# ---------------------------------------------------------------------------
# 3. Failure paths
# ---------------------------------------------------------------------------

class TestFailurePaths(unittest.TestCase):

    def test_malformed_missing_required_field(self):
        """Manually crafted payload missing 'targets' must fail validation."""
        bad = {
            "protocol_version": "1.0",
            "trace_id": "t1",
            "timestamp_ms": int(time.time() * 1000),
            "mode": "constant",
            "max_age_ms": 1000,
            # 'targets' intentionally omitted
        }
        ok, _ = validate_protocol_payload(bad)
        self.assertFalse(ok)

    def test_malformed_wrong_protocol_version(self):
        spec = ProtocolTargetSpec("m1", "gain", 0.5)
        payload = ProtocolPayloadCompiler().build_payload([spec])
        payload["protocol_version"] = "99.0"
        ok, _ = validate_protocol_payload(payload)
        self.assertFalse(ok)

    def test_stale_payload_fails_validation(self):
        spec = ProtocolTargetSpec("m1", "gain", 0.5)
        payload = ProtocolPayloadCompiler().build_payload(
            [spec],
            max_age_ms=1,
            timestamp_ms=int(time.time() * 1000) - 5000,  # 5 seconds old
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertFalse(ok)
        stale_codes = [d.code for d in diags if "stale" in str(d.code)]
        self.assertTrue(len(stale_codes) > 0, "Expected stale diagnostic")

    def test_empty_targets_list_fails(self):
        payload = ProtocolPayloadCompiler().build_payload([])
        ok, _ = validate_protocol_payload(payload)
        self.assertFalse(ok)

    def test_non_dict_target_fails(self):
        spec = ProtocolTargetSpec("m1", "gain", 0.5)
        payload = ProtocolPayloadCompiler().build_payload([spec])
        payload["targets"] = ["not_a_dict"]
        ok, _ = validate_protocol_payload(payload)
        self.assertFalse(ok)

    def test_out_of_range_clamped_not_error(self):
        """Out-of-range value WITH clamp=True is not a validation error — just clamped."""
        spec = ProtocolTargetSpec(
            "m1", "gain", 999.0,
            limits={"min": 0.0, "max": 1.0, "clamp": True}
        )
        payload = ProtocolPayloadCompiler().build_payload([spec])
        ok, _ = validate_protocol_payload(payload)
        self.assertTrue(ok, "Clamped out-of-range should still pass validation")

    def test_unknown_signal_raises_value_error(self):
        with self.assertRaises(ValueError):
            ProtocolPayloadCompiler.for_signal("bad_signal", motor_key="m")

    def test_switch_without_value_raises(self):
        with self.assertRaises(ValueError):
            ProtocolPayloadCompiler.for_signal(SIGNAL_SWITCH, motor_key="m")

    def test_missing_motor_key_in_target_fails(self):
        spec = ProtocolTargetSpec("m1", "gain", 0.5)
        payload = ProtocolPayloadCompiler().build_payload([spec])
        payload["targets"][0]["motor_key"] = ""
        ok, _ = validate_protocol_payload(payload)
        self.assertFalse(ok)

    def test_max_age_ms_zero_fails(self):
        spec = ProtocolTargetSpec("m1", "gain", 0.5)
        payload = ProtocolPayloadCompiler().build_payload([spec], max_age_ms=0)
        ok, _ = validate_protocol_payload(payload)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 4. Full pipeline: sensor → compile → apply (ProtocolApplyEngine)
# ---------------------------------------------------------------------------

class TestSensorToApplyPipeline(unittest.TestCase):
    """Full DoD path: sensor(imu) → protocol → validate → parse → apply."""

    def test_imu_to_apply_no_adapter(self):
        """Without an adapter, apply succeeds silently (hardware not connected)."""
        from system.behavior.protocol_apply_engine import ProtocolApplyEngine

        sensor_data = {"accel_z": 0.25}
        payload = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_fl_hip", param_key="gain"
        )
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)

        engine = ProtocolApplyEngine()
        result = engine.apply(targets, adapter=None, trace_id="test-trace")
        self.assertEqual(result.apply_success_count + result.apply_skipped_count, len(targets))

    def test_signal_enable_full_pipeline(self):
        from system.behavior.protocol_apply_engine import ProtocolApplyEngine

        payload = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="leg_rr")
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)

        engine = ProtocolApplyEngine()
        result = engine.apply(targets, adapter=None)
        self.assertEqual(result.apply_success_count + result.apply_skipped_count, 1)

    def test_multi_motor_apply(self):
        from system.behavior.protocol_apply_engine import ProtocolApplyEngine

        specs = [
            ProtocolTargetSpec("leg_fl_hip", "gain", 0.3),
            ProtocolTargetSpec("leg_rr_knee", "stiffness", 0.7),
        ]
        payload = ProtocolPayloadCompiler().build_payload(specs)
        ok, diags = validate_protocol_payload(payload)
        self.assertTrue(ok)
        targets, _ = parse_protocol_targets(payload, diags)

        result = ProtocolApplyEngine().apply(targets, adapter=None)
        self.assertEqual(result.apply_success_count + result.apply_skipped_count, 2)


# ---------------------------------------------------------------------------
# 5. WorkflowIR PROTOCOL_EMIT codegen → syntactically valid Python
# ---------------------------------------------------------------------------

class TestWorkflowIRProtocolEmitCodegen(unittest.TestCase):

    def _make_sensor_protocol_ir(self):
        """Build: Start → Sensor(imu) → Protocol Emit → End."""
        from compiler.ir.workflow_ir import (
            WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType
        )
        start  = IRNode("n0", "start",         NodeKind.START,   {})
        sensor = IRNode("n1", "sensor_input",  NodeKind.SENSOR,  {
            "sensor_type": IRParam("sensor_type", "imu", "string"),
        })
        emit   = IRNode("n2", "protocol_emit", NodeKind.PROTOCOL_EMIT, {
            "motor_key":    IRParam("motor_key",    "leg_fl_hip", "string"),
            "param_key":    IRParam("param_key",    "gain",       "string"),
            "address":      IRParam("address",      0,            "int"),
            "max_age_ms":   IRParam("max_age_ms",   1000,         "int"),
            "signal":       IRParam("signal",       "",           "string"),
            "sensor_field": IRParam("sensor_field", "accel_z",   "string"),
        })
        end    = IRNode("n3", "end",            NodeKind.END,     {})
        edges  = [
            IREdge("n0", "flow_out", "n1", "flow_in", EdgeType.FLOW),
            IREdge("n1", "flow_out", "n2", "flow_in", EdgeType.FLOW),
            IREdge("n2", "flow_out", "n3", "flow_in", EdgeType.FLOW),
        ]
        return WorkflowIR(nodes=[start, sensor, emit, end], edges=edges)

    def _make_signal_protocol_ir(self, signal: str = "event_enable"):
        from compiler.ir.workflow_ir import (
            WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType
        )
        start = IRNode("n0", "start",         NodeKind.START,        {})
        emit  = IRNode("n1", "protocol_emit", NodeKind.PROTOCOL_EMIT, {
            "motor_key":    IRParam("motor_key",    "leg_fl_hip", "string"),
            "param_key":    IRParam("param_key",    "stiffness",  "string"),
            "address":      IRParam("address",      0,            "int"),
            "max_age_ms":   IRParam("max_age_ms",   500,          "int"),
            "signal":       IRParam("signal",       signal,       "string"),
            "sensor_field": IRParam("sensor_field", "accel_z",   "string"),
        })
        end   = IRNode("n2", "end", NodeKind.END, {})
        edges = [
            IREdge("n0", "flow_out", "n1", "flow_in", EdgeType.FLOW),
            IREdge("n1", "flow_out", "n2", "flow_in", EdgeType.FLOW),
        ]
        return WorkflowIR(nodes=[start, emit, end], edges=edges)

    def test_sensor_protocol_ir_codegen_valid_syntax(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_sensor_protocol_ir())
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in generated code: {e}\n---\n{code}")

    def test_sensor_protocol_ir_contains_from_sensor_imu(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_sensor_protocol_ir())
        self.assertIn("from_sensor_imu", code)
        self.assertIn("protocol_payload", code)

    def test_sensor_protocol_ir_contains_sensor_read(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_sensor_protocol_ir())
        self.assertIn("sensor_data", code)

    def test_signal_protocol_ir_codegen_valid_syntax(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_signal_protocol_ir("event_enable"))
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error: {e}\n---\n{code}")

    def test_signal_protocol_ir_contains_for_signal(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_signal_protocol_ir("event_disable"))
        self.assertIn("for_signal", code)
        self.assertIn("event_disable", code)

    def test_codegen_from_canvas_via_full_pipeline(self):
        """Canvas-lowering → IR → codegen pipeline produces valid code."""
        from compiler.lowering.canvas_to_ir import CanvasToIR
        from compiler.codegen.ir_to_code import IRToCode

        canvas_data = {
            "nodes": [
                {"id": 1, "type": "Start", "display_name": "Start"},
                {
                    "id": 2, "type": "Sensor Input", "display_name": "Sensor Input",
                    "sensor_type": "imu",
                },
                {
                    "id": 3, "type": "Protocol Emit", "display_name": "Protocol Emit",
                    "motor_key": "leg_fl_hip", "param_key": "gain",
                    "address": 0, "dtype": "float32", "max_age_ms": 1000,
                    "signal": "", "sensor_field": "accel_z",
                },
                {"id": 4, "type": "End", "display_name": "End"},
            ],
            "connections": [
                {"from_node": 1, "to_node": 2, "from_port": "flow_out", "to_port": "flow_in"},
                {"from_node": 2, "to_node": 3, "from_port": "flow_out", "to_port": "flow_in"},
                {"from_node": 3, "to_node": 4, "from_port": "flow_out", "to_port": "flow_in"},
            ],
        }
        ir, _ = CanvasToIR().convert(canvas_data, "go2")
        code, _, _ = IRToCode().generate(ir)
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            self.fail(f"Full pipeline codegen syntax error: {e}\n---\n{code}")
        self.assertIn("ProtocolPayloadCompiler", code)


if __name__ == "__main__":
    unittest.main()
