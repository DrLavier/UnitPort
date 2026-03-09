#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle 4 Fix — Unit Tests: ProtocolEmitNode registry, execution contract,
schema existence, and ir_to_canvas roundtrip.
"""

import os
import json
import pytest
import sys

# Ensure project root on path
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:
    def test_protocol_emit_in_registry(self):
        from nodes import get_node_class
        cls = get_node_class("protocol_emit")
        assert cls is not None

    def test_protocol_emit_is_base_node_subclass(self):
        from nodes import get_node_class
        from nodes.sys_nodes.base_node import BaseNode
        cls = get_node_class("protocol_emit")
        assert issubclass(cls, BaseNode)

    def test_protocol_emit_in_list_node_types(self):
        from nodes import list_node_types
        assert "protocol_emit" in list_node_types()

    def test_protocol_emit_in_list_system_nodes(self):
        from nodes import list_system_nodes
        assert "protocol_emit" in list_system_nodes()

    def test_protocol_emit_importable_directly(self):
        from nodes.sys_nodes import ProtocolEmitNode
        assert ProtocolEmitNode is not None

    def test_create_node_factory(self):
        from nodes import create_node
        node = create_node("protocol_emit", "test-pe-01")
        assert node.node_type == "protocol_emit"
        assert node.node_id == "test-pe-01"


# ===========================================================================
# Node execution contract
# ===========================================================================

class TestProtocolEmitNodeSensorPath:
    def _node(self):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("pe1")
        n.set_parameter("motor_key",    "FL_hip")
        n.set_parameter("param_key",    "stiffness")
        n.set_parameter("address",      "FL_hip.stiffness")
        n.set_parameter("max_age_ms",   500)
        n.set_parameter("signal",       "")
        n.set_parameter("sensor_field", "x")
        return n

    def _sensor_input(self, x=0.5):
        return {"data": {"x": x, "y": 0.0, "z": 0.0}}

    def test_sensor_path_returns_success(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input(0.5)})
        assert result["status"] == "success"

    def test_sensor_path_payload_keys(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input(0.5)})
        payload = result["protocol_payload"]
        for key in ("protocol_version", "trace_id", "timestamp_ms", "mode",
                    "max_age_ms", "targets"):
            assert key in payload, f"Missing key: {key}"

    def test_sensor_path_condition_alias(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input(0.5)})
        assert result["condition"] is result["protocol_payload"]

    def test_sensor_path_targets_nonempty(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input(0.5)})
        assert len(result["protocol_payload"]["targets"]) > 0

    def test_sensor_path_target_motor_key(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input(0.5)})
        target = result["protocol_payload"]["targets"][0]
        assert target.get("motor_key") == "FL_hip"

    def test_sensor_path_max_age_forwarded(self):
        n = self._node()
        result = n.execute({"sensor_data": self._sensor_input()})
        assert result["protocol_payload"]["max_age_ms"] == 500

    def test_sensor_path_no_sensor_data_safe(self):
        """Missing sensor_data should not crash — degrade gracefully."""
        n = self._node()
        result = n.execute({})
        # Either success with zero value or error with safe payload
        assert "protocol_payload" in result
        assert "targets" in result["protocol_payload"]

    def test_sensor_path_numeric_string_field(self):
        """If sensor field value is not numeric, falls back to 0.0."""
        n = self._node()
        result = n.execute({"sensor_data": {"data": {"x": "not_a_number"}}})
        assert "protocol_payload" in result


class TestProtocolEmitNodeSignalPath:
    def _node(self, signal):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("pe2")
        n.set_parameter("motor_key",  "FR_hip")
        n.set_parameter("address",    "FR_hip.stiffness")
        n.set_parameter("max_age_ms", 1000)
        n.set_parameter("signal",     signal)
        return n

    def test_signal_enable_success(self):
        n = self._node("event_enable")
        result = n.execute({})
        assert result["status"] == "success"

    def test_signal_enable_stiffness_1(self):
        n = self._node("event_enable")
        result = n.execute({})
        targets = result["protocol_payload"]["targets"]
        assert len(targets) == 1
        src = targets[0]["source"]
        assert src["kind"] == "constant"
        assert src["constant"] == pytest.approx(1.0)

    def test_signal_disable_stiffness_0(self):
        n = self._node("event_disable")
        result = n.execute({})
        targets = result["protocol_payload"]["targets"]
        src = targets[0]["source"]
        assert src["constant"] == pytest.approx(0.0)

    def test_signal_switch_uses_switch_value(self):
        n = self._node("event_switch")
        result = n.execute({"switch_value": 42.0})
        targets = result["protocol_payload"]["targets"]
        src = targets[0]["source"]
        assert src["constant"] == pytest.approx(42.0)

    def test_signal_switch_default_zero_when_missing(self):
        """switch_value missing → falls back to 0.0, no crash."""
        n = self._node("event_switch")
        result = n.execute({})
        assert "protocol_payload" in result

    def test_unknown_signal_degrades_gracefully(self):
        """Unknown signal string → error branch, safe payload returned."""
        n = self._node("event_unknown_xyz")
        result = n.execute({})
        assert "protocol_payload" in result
        assert "targets" in result["protocol_payload"]


class TestProtocolEmitNodeContract:
    def test_get_display_name(self):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("x")
        assert n.get_display_name() == "Protocol Emit"

    def test_get_description_nonempty(self):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("x")
        assert len(n.get_description()) > 0

    def test_inputs_has_sensor_data_port(self):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("x")
        assert "sensor_data" in n.inputs

    def test_outputs_has_protocol_payload_and_condition(self):
        from nodes.sys_nodes.protocol_emit_node import ProtocolEmitNode
        n = ProtocolEmitNode("x")
        assert "protocol_payload" in n.outputs
        assert "condition" in n.outputs


# ===========================================================================
# Schema existence
# ===========================================================================

class TestSchemaFile:
    def _schema_path(self):
        return os.path.join(_ROOT, "compiler", "schema", "builtin", "protocol_emit.json")

    def test_file_exists(self):
        assert os.path.isfile(self._schema_path())

    def test_valid_json(self):
        with open(self._schema_path(), encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_schema_id(self):
        with open(self._schema_path(), encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_id"] == "builtin.protocol_emit"

    def test_node_type(self):
        with open(self._schema_path(), encoding="utf-8") as f:
            data = json.load(f)
        assert data["node_type"] == "protocol_emit"

    def test_has_ports(self):
        with open(self._schema_path(), encoding="utf-8") as f:
            data = json.load(f)
        port_names = [p["name"] for p in data["ports"]]
        for p in ("flow_in", "flow_out", "protocol_payload", "condition"):
            assert p in port_names

    def test_has_parameters(self):
        with open(self._schema_path(), encoding="utf-8") as f:
            data = json.load(f)
        param_names = [p["name"] for p in data["parameters"]]
        for name in ("motor_key", "param_key", "address", "max_age_ms",
                     "signal", "sensor_field"):
            assert name in param_names


# ===========================================================================
# ir_to_canvas roundtrip
# ===========================================================================

class TestIRToCanvasRoundtrip:
    def _make_ir(self):
        from compiler.ir.workflow_ir import (
            WorkflowIR, IRNode, IREdge, NodeKind, EdgeType,
        )
        ir = WorkflowIR(name="test_pe_roundtrip")
        start = IRNode(id="s1", schema_id="builtin.start", kind=NodeKind.START)
        pe = IRNode(id="pe1", schema_id="protocol_emit", kind=NodeKind.PROTOCOL_EMIT)
        pe.set_param("motor_key",    "RL_knee")
        pe.set_param("param_key",    "damping")
        pe.set_param("address",      "RL_knee.damping")
        pe.set_param("dtype",        "float")
        pe.set_param("max_age_ms",   750)
        pe.set_param("signal",       "")
        pe.set_param("sensor_field", "z")
        end = IRNode(id="e1", schema_id="builtin.end", kind=NodeKind.END)
        ir.add_node(start)
        ir.add_node(pe)
        ir.add_node(end)
        ir.add_edge(IREdge("s1", "flow_out", "pe1", "flow_in"))
        ir.add_edge(IREdge("pe1", "flow_out", "e1", "flow_in"))
        return ir

    def test_convert_does_not_produce_unknown(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, diags = IRToCanvas().convert(ir)
        pe_node = next((n for n in graph_data["nodes"]
                        if n.get("node_type") == "protocol_emit"), None)
        assert pe_node is not None, "protocol_emit node missing from canvas output"

    def test_roundtrip_motor_key(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["motor_key"] == "RL_knee"

    def test_roundtrip_param_key(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["param_key"] == "damping"

    def test_roundtrip_address(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["address"] == "RL_knee.damping"

    def test_roundtrip_max_age_ms(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["max_age_ms"] == "750"

    def test_roundtrip_sensor_field(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["sensor_field"] == "z"

    def test_roundtrip_display_name(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        graph_data, _ = IRToCanvas().convert(ir)
        pe_node = next(n for n in graph_data["nodes"]
                       if n.get("node_type") == "protocol_emit")
        assert pe_node["display_name"] == "Protocol Emit"

    def test_no_w3003_unknown_diag(self):
        from compiler.lowering.ir_to_canvas import IRToCanvas
        ir = self._make_ir()
        _, diags = IRToCanvas().convert(ir)
        codes = [d.code for d in diags]
        assert "W3003" not in codes, "W3003 (unknown kind) fired for PROTOCOL_EMIT"
