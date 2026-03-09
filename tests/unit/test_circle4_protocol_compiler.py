"""Unit tests for Circle 4 ProtocolPayloadCompiler and ProtocolTargetSpec.

Covers:
- ProtocolTargetSpec.to_target_dict(): constant and external source forms
- ProtocolPayloadCompiler.build_payload(): envelope fields, content, edge cases
- ProtocolPayloadCompiler.from_sensor_imu(): field extraction, fallback
- ProtocolPayloadCompiler.for_signal(): SIGNAL_ENABLE/DISABLE/SWITCH, errors
- Signal convention constants
- NodeKind.PROTOCOL_EMIT existence
- Canvas lowering: protocol_emit node params
- Codegen: PROTOCOL_EMIT generates valid Python containing compiler call
"""

from __future__ import annotations

import unittest

from compiler.lowering.protocol_compiler import (
    DEFAULT_MAX_AGE_MS,
    DEFAULT_MODE,
    PROTOCOL_VERSION,
    SIGNAL_DISABLE,
    SIGNAL_ENABLE,
    SIGNAL_SWITCH,
    VALID_SIGNALS,
    ProtocolPayloadCompiler,
    ProtocolTargetSpec,
)


# ---------------------------------------------------------------------------
# ProtocolTargetSpec
# ---------------------------------------------------------------------------

class TestProtocolTargetSpec(unittest.TestCase):

    def _make(self, **kwargs) -> ProtocolTargetSpec:
        defaults = dict(motor_key="leg_fl_hip", param_key="gain", value=0.5)
        defaults.update(kwargs)
        return ProtocolTargetSpec(**defaults)

    def test_to_target_dict_constant_source(self):
        d = self._make().to_target_dict()
        self.assertEqual(d["source"]["kind"], "constant")
        self.assertEqual(d["source"]["constant"], 0.5)

    def test_to_target_dict_external_source(self):
        spec = self._make(source_key="imu.accel_z")
        d = spec.to_target_dict()
        self.assertEqual(d["source"]["kind"], "external")
        self.assertEqual(d["source"]["external_key"], "imu.accel_z")
        self.assertEqual(d["source"]["external_value"], 0.5)

    def test_to_target_dict_required_fields_present(self):
        d = self._make().to_target_dict()
        for key in ("motor_key", "param_key", "address", "dtype", "shape", "source"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_to_target_dict_motor_and_param_key(self):
        d = self._make(motor_key="leg_rr_knee", param_key="stiffness").to_target_dict()
        self.assertEqual(d["motor_key"], "leg_rr_knee")
        self.assertEqual(d["param_key"], "stiffness")

    def test_to_target_dict_limits_included_when_set(self):
        spec = self._make(limits={"min": 0.0, "max": 1.0, "clamp": True})
        d = spec.to_target_dict()
        self.assertIn("limits", d)
        self.assertEqual(d["limits"]["min"], 0.0)

    def test_to_target_dict_no_limits_key_when_none(self):
        d = self._make(limits=None).to_target_dict()
        self.assertNotIn("limits", d)

    def test_to_target_dict_note_included_when_set(self):
        d = self._make(note="test note").to_target_dict()
        self.assertIn("note", d)

    def test_to_target_dict_no_note_key_when_empty(self):
        d = self._make(note="").to_target_dict()
        self.assertNotIn("note", d)

    def test_default_dtype_float32(self):
        d = self._make().to_target_dict()
        self.assertEqual(d["dtype"], "float32")

    def test_default_shape_scalar(self):
        d = self._make().to_target_dict()
        self.assertEqual(d["shape"], [1])

    def test_default_address_zero(self):
        d = self._make().to_target_dict()
        self.assertEqual(d["address"], 0)

    def test_array_value_accepted(self):
        spec = self._make(value=[1.0, 2.0, 3.0], shape=[3])
        d = spec.to_target_dict()
        self.assertEqual(d["source"]["constant"], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# ProtocolPayloadCompiler.build_payload
# ---------------------------------------------------------------------------

class TestBuildPayload(unittest.TestCase):

    def _compiler(self):
        return ProtocolPayloadCompiler()

    def _one_target(self):
        return [ProtocolTargetSpec(motor_key="m1", param_key="gain", value=0.3)]

    def test_protocol_version_correct(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertEqual(p["protocol_version"], PROTOCOL_VERSION)

    def test_has_trace_id(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertIsInstance(p["trace_id"], str)
        self.assertGreater(len(p["trace_id"]), 0)

    def test_custom_trace_id_preserved(self):
        p = self._compiler().build_payload(self._one_target(), trace_id="my-trace")
        self.assertEqual(p["trace_id"], "my-trace")

    def test_has_timestamp_ms(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertIsInstance(p["timestamp_ms"], int)
        self.assertGreater(p["timestamp_ms"], 0)

    def test_custom_timestamp_preserved(self):
        p = self._compiler().build_payload(self._one_target(), timestamp_ms=12345)
        self.assertEqual(p["timestamp_ms"], 12345)

    def test_default_mode(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertEqual(p["mode"], DEFAULT_MODE)

    def test_custom_mode(self):
        p = self._compiler().build_payload(self._one_target(), mode="external")
        self.assertEqual(p["mode"], "external")

    def test_default_max_age_ms(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertEqual(p["max_age_ms"], DEFAULT_MAX_AGE_MS)

    def test_custom_max_age_ms(self):
        p = self._compiler().build_payload(self._one_target(), max_age_ms=500)
        self.assertEqual(p["max_age_ms"], 500)

    def test_targets_list_present(self):
        p = self._compiler().build_payload(self._one_target())
        self.assertIsInstance(p["targets"], list)
        self.assertEqual(len(p["targets"]), 1)

    def test_empty_targets_accepted(self):
        p = self._compiler().build_payload([])
        self.assertEqual(p["targets"], [])

    def test_multiple_targets(self):
        specs = [
            ProtocolTargetSpec("m1", "gain", 0.1),
            ProtocolTargetSpec("m2", "stiffness", 0.9),
        ]
        p = self._compiler().build_payload(specs)
        self.assertEqual(len(p["targets"]), 2)

    def test_required_envelope_keys(self):
        p = self._compiler().build_payload(self._one_target())
        for key in ("protocol_version", "trace_id", "timestamp_ms", "mode", "max_age_ms", "targets"):
            self.assertIn(key, p)


# ---------------------------------------------------------------------------
# Integration: build_payload → validate_protocol_payload
# ---------------------------------------------------------------------------

class TestBuildPayloadValidates(unittest.TestCase):
    """Payload built by ProtocolPayloadCompiler must pass the validator."""

    def _validate(self, payload):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        ok, diags = validate_protocol_payload(payload)
        return ok, diags

    def test_single_target_validates(self):
        spec = ProtocolTargetSpec("leg_fl_hip", "gain", 0.5)
        p = ProtocolPayloadCompiler().build_payload([spec])
        ok, diags = self._validate(p)
        errors = [d for d in diags if getattr(d, "level", None) is not None
                  and str(d.level) in ("DiagLevel.error", "error")]
        self.assertTrue(ok, f"Validation failed: {[d.message for d in errors]}")

    def test_multi_target_validates(self):
        specs = [
            ProtocolTargetSpec("m1", "gain", 0.1),
            ProtocolTargetSpec("m2", "stiffness", 1.0),
        ]
        p = ProtocolPayloadCompiler().build_payload(specs)
        ok, _ = self._validate(p)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# ProtocolPayloadCompiler.from_sensor_imu
# ---------------------------------------------------------------------------

class TestFromSensorImu(unittest.TestCase):

    def test_extracts_correct_sensor_field(self):
        sensor_data = {"accel_z": 0.42, "accel_x": 1.0}
        p = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_fl", sensor_field="accel_z"
        )
        self.assertEqual(p["targets"][0]["source"]["constant"], 0.42)

    def test_different_sensor_field(self):
        sensor_data = {"gyro_y": -0.1}
        p = ProtocolPayloadCompiler.from_sensor_imu(
            sensor_data, motor_key="leg_rr", sensor_field="gyro_y"
        )
        self.assertAlmostEqual(p["targets"][0]["source"]["constant"], -0.1, places=5)

    def test_missing_field_falls_back_to_zero(self):
        p = ProtocolPayloadCompiler.from_sensor_imu({}, motor_key="m1")
        self.assertEqual(p["targets"][0]["source"]["constant"], 0.0)

    def test_non_numeric_falls_back_to_zero(self):
        p = ProtocolPayloadCompiler.from_sensor_imu(
            {"accel_z": "bad"}, motor_key="m1", sensor_field="accel_z"
        )
        self.assertEqual(p["targets"][0]["source"]["constant"], 0.0)

    def test_none_sensor_data_falls_back_to_zero(self):
        p = ProtocolPayloadCompiler.from_sensor_imu(None, motor_key="m1")  # type: ignore
        self.assertEqual(p["targets"][0]["source"]["constant"], 0.0)

    def test_motor_key_and_param_key_propagated(self):
        p = ProtocolPayloadCompiler.from_sensor_imu(
            {"accel_z": 0.1}, motor_key="leg_rr_knee", param_key="stiffness"
        )
        t = p["targets"][0]
        self.assertEqual(t["motor_key"], "leg_rr_knee")
        self.assertEqual(t["param_key"], "stiffness")

    def test_custom_max_age_ms(self):
        p = ProtocolPayloadCompiler.from_sensor_imu(
            {"accel_z": 0.1}, motor_key="m", max_age_ms=500
        )
        self.assertEqual(p["max_age_ms"], 500)

    def test_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        sensor_data = {"accel_z": 0.15}
        p = ProtocolPayloadCompiler.from_sensor_imu(sensor_data, motor_key="leg_fl")
        ok, _ = validate_protocol_payload(p)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# ProtocolPayloadCompiler.for_signal
# ---------------------------------------------------------------------------

class TestForSignal(unittest.TestCase):

    def test_enable_sets_stiffness_one(self):
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="leg_fl")
        t = p["targets"][0]
        self.assertEqual(t["param_key"], "stiffness")
        self.assertEqual(t["source"]["constant"], 1.0)

    def test_disable_sets_stiffness_zero(self):
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_DISABLE, motor_key="leg_fl")
        t = p["targets"][0]
        self.assertEqual(t["param_key"], "stiffness")
        self.assertEqual(t["source"]["constant"], 0.0)

    def test_switch_sets_mode_with_caller_value(self):
        p = ProtocolPayloadCompiler.for_signal(
            SIGNAL_SWITCH, motor_key="leg_fl", switch_value=2.0
        )
        t = p["targets"][0]
        self.assertEqual(t["param_key"], "mode")
        self.assertEqual(t["source"]["constant"], 2.0)

    def test_motor_key_propagated(self):
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="leg_rr_hip")
        self.assertEqual(p["targets"][0]["motor_key"], "leg_rr_hip")

    def test_unknown_signal_raises(self):
        with self.assertRaises(ValueError):
            ProtocolPayloadCompiler.for_signal("event_unknown", motor_key="m")

    def test_switch_without_value_raises(self):
        with self.assertRaises(ValueError):
            ProtocolPayloadCompiler.for_signal(SIGNAL_SWITCH, motor_key="m")

    def test_max_age_ms_propagated(self):
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="m", max_age_ms=250)
        self.assertEqual(p["max_age_ms"], 250)

    def test_enable_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_ENABLE, motor_key="leg_fl")
        ok, _ = validate_protocol_payload(p)
        self.assertTrue(ok)

    def test_disable_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_DISABLE, motor_key="leg_fl")
        ok, _ = validate_protocol_payload(p)
        self.assertTrue(ok)

    def test_switch_payload_validates(self):
        from system.behavior.motor_weight_protocol import validate_protocol_payload
        p = ProtocolPayloadCompiler.for_signal(SIGNAL_SWITCH, motor_key="m", switch_value=1.0)
        ok, _ = validate_protocol_payload(p)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------

class TestSignalConstants(unittest.TestCase):

    def test_signal_enable_string(self):
        self.assertEqual(SIGNAL_ENABLE, "event_enable")

    def test_signal_disable_string(self):
        self.assertEqual(SIGNAL_DISABLE, "event_disable")

    def test_signal_switch_string(self):
        self.assertEqual(SIGNAL_SWITCH, "event_switch")

    def test_valid_signals_contains_all_three(self):
        self.assertIn(SIGNAL_ENABLE, VALID_SIGNALS)
        self.assertIn(SIGNAL_DISABLE, VALID_SIGNALS)
        self.assertIn(SIGNAL_SWITCH, VALID_SIGNALS)


# ---------------------------------------------------------------------------
# NodeKind.PROTOCOL_EMIT
# ---------------------------------------------------------------------------

class TestNodeKindProtocolEmit(unittest.TestCase):

    def test_protocol_emit_exists(self):
        from compiler.ir.workflow_ir import NodeKind
        self.assertEqual(NodeKind.PROTOCOL_EMIT.value, "protocol_emit")

    def test_from_string_protocol_emit(self):
        from compiler.ir.workflow_ir import NodeKind
        self.assertEqual(NodeKind.from_string("protocol_emit"), NodeKind.PROTOCOL_EMIT)


# ---------------------------------------------------------------------------
# Canvas lowering: "Protocol Emit" → protocol_emit params
# ---------------------------------------------------------------------------

class TestCanvasToIRProtocolEmit(unittest.TestCase):

    def _make_canvas_data(self, **overrides):
        node = {
            "id": 1,
            "type": "Protocol Emit",
            "display_name": "Protocol Emit",
            "motor_key": "leg_fl_hip",
            "param_key": "gain",
            "address": 0,
            "dtype": "float32",
            "max_age_ms": 1000,
            "signal": "",
            "sensor_field": "accel_z",
        }
        node.update(overrides)
        return {
            "nodes": [node],
            "connections": [],
        }

    def test_protocol_emit_lowered_to_correct_kind(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        ir, diags = CanvasToIR().convert(self._make_canvas_data(), "go2")
        errors = [d for d in diags if getattr(d, "level", "").name == "error"
                  if hasattr(getattr(d, "level", None), "name")]
        self.assertTrue(any(
            n.kind.value == "protocol_emit" for n in ir.nodes
        ), "No protocol_emit node found in IR")

    def test_protocol_emit_motor_key_param(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        ir, _ = CanvasToIR().convert(self._make_canvas_data(motor_key="test_motor"), "go2")
        pe_nodes = [n for n in ir.nodes if n.kind.value == "protocol_emit"]
        self.assertTrue(pe_nodes)
        self.assertEqual(pe_nodes[0].get_param_value("motor_key"), "test_motor")

    def test_protocol_emit_sensor_field_param(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        ir, _ = CanvasToIR().convert(
            self._make_canvas_data(sensor_field="gyro_z"), "go2"
        )
        pe = [n for n in ir.nodes if n.kind.value == "protocol_emit"][0]
        self.assertEqual(pe.get_param_value("sensor_field"), "gyro_z")

    def test_protocol_emit_signal_param(self):
        from compiler.lowering.canvas_to_ir import CanvasToIR
        ir, _ = CanvasToIR().convert(
            self._make_canvas_data(signal="event_enable"), "go2"
        )
        pe = [n for n in ir.nodes if n.kind.value == "protocol_emit"][0]
        self.assertEqual(pe.get_param_value("signal"), "event_enable")


# ---------------------------------------------------------------------------
# Codegen: PROTOCOL_EMIT generates valid Python containing expected calls
# ---------------------------------------------------------------------------

class TestIRToCodeProtocolEmit(unittest.TestCase):

    def _make_ir(self, signal: str = "", sensor_field: str = "accel_z"):
        from compiler.ir.workflow_ir import WorkflowIR, IRNode, IREdge, IRParam, NodeKind, EdgeType
        start = IRNode("n0", "start", NodeKind.START, {})
        emit  = IRNode("n1", "protocol_emit", NodeKind.PROTOCOL_EMIT, {
            "motor_key":    IRParam("motor_key",    "leg_fl_hip", "string"),
            "param_key":    IRParam("param_key",    "gain",       "string"),
            "address":      IRParam("address",      0,            "int"),
            "max_age_ms":   IRParam("max_age_ms",   1000,         "int"),
            "signal":       IRParam("signal",       signal,       "string"),
            "sensor_field": IRParam("sensor_field", sensor_field, "string"),
        })
        end   = IRNode("n2", "end", NodeKind.END, {})
        edges = [
            IREdge("n0", "flow_out", "n1", "flow_in", EdgeType.FLOW),
            IREdge("n1", "flow_out", "n2", "flow_in", EdgeType.FLOW),
        ]
        return WorkflowIR(nodes=[start, emit, end], edges=edges)

    def test_codegen_produces_protocol_payload_assignment(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_ir())
        self.assertIn("protocol_payload", code)
        self.assertIn("ProtocolPayloadCompiler", code)

    def test_codegen_sensor_path_uses_from_sensor_imu(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_ir(signal=""))
        self.assertIn("from_sensor_imu", code)

    def test_codegen_signal_path_uses_for_signal(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_ir(signal="event_enable"))
        self.assertIn("for_signal", code)

    def test_codegen_is_syntactically_valid(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_ir())
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")

    def test_codegen_motor_key_in_output(self):
        from compiler.codegen.ir_to_code import IRToCode
        code, _, _ = IRToCode().generate(self._make_ir())
        self.assertIn("leg_fl_hip", code)


if __name__ == "__main__":
    unittest.main()
