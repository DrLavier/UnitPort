#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the 6 new Base Control Nodes:
Wait, Gate, Timeout, Retry, Cancel, Abort

Tests cover:
1. Canvas → IR param extraction
2. IR → Code generation
3. IR → Canvas roundtrip
"""

import pytest

from compiler.ir.workflow_ir import (
    WorkflowIR, IRNode, IREdge, IRParam, IRNodeUI, NodeKind, EdgeType,
)
from compiler.lowering.canvas_to_ir import CanvasToIR
from compiler.lowering.ir_to_canvas import IRToCanvas
from compiler.codegen.ir_to_code import IRToCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canvas_node(display_name, node_type, node_id=0, **extra):
    """Create a minimal canvas node dict for CanvasToIR."""
    node = {
        "id": node_id,
        "display_name": display_name,
        "node_type": node_type,
        "position": {"x": 100, "y": 100},
        "width": 180,
        "height": 110,
    }
    node.update(extra)
    return node


def _make_graph_data(nodes, connections=None):
    return {"nodes": nodes, "connections": connections or []}


def _ir_with_single_node(kind: NodeKind, schema_id: str, params: dict) -> WorkflowIR:
    """Build a minimal IR with one node."""
    ir = WorkflowIR()
    ir_params = {k: IRParam(k, v) for k, v in params.items()}
    node = IRNode(id="0", schema_id=schema_id, kind=kind, params=ir_params,
                  ui=IRNodeUI(x=100, y=100))
    ir.add_node(node)
    return ir


# ===========================================================================
# 1. Canvas → IR param extraction
# ===========================================================================

class TestCanvasToIR:
    """Test CanvasToIR._extract_params for each new node type."""

    def setup_method(self):
        self.converter = CanvasToIR()

    def test_wait_duration_mode(self):
        node = _make_canvas_node("Wait", "wait", duration="2.5", wait_mode="duration",
                                 event_name="", timeout="0")
        ir, diags = self.converter.convert(_make_graph_data([node]))
        assert len(ir.nodes) == 1
        n = ir.nodes[0]
        assert n.kind == NodeKind.WAIT
        assert n.get_param_value("mode") == "duration"
        assert n.get_param_value("duration") == 2.5

    def test_wait_event_mode(self):
        node = _make_canvas_node("Wait", "wait", duration="1.0", wait_mode="event",
                                 event_name="sensor_ready", timeout="10")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.get_param_value("mode") == "event"
        assert n.get_param_value("event_name") == "sensor_ready"
        assert n.get_param_value("timeout") == 10.0

    def test_gate_condition(self):
        node = _make_canvas_node("Gate", "gate", gate_mode="condition",
                                 condition_expr="x > 5")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.kind == NodeKind.GATE
        assert n.get_param_value("mode") == "condition"
        assert n.get_param_value("condition_expr") == "x > 5"

    def test_gate_event(self):
        node = _make_canvas_node("Gate", "gate", gate_mode="event",
                                 event_name="door_open")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.get_param_value("mode") == "event"
        assert n.get_param_value("event_name") == "door_open"

    def test_timeout(self):
        node = _make_canvas_node("Timeout", "timeout", duration="3.0",
                                 timeout_on="next_step")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.kind == NodeKind.TIMEOUT
        assert n.get_param_value("duration") == 3.0

    def test_retry(self):
        node = _make_canvas_node("Retry", "retry", max_attempts="5",
                                 backoff_sec="2.0", strategy="exp")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.kind == NodeKind.RETRY
        assert n.get_param_value("max_attempts") == 5
        assert n.get_param_value("backoff_sec") == 2.0
        assert n.get_param_value("strategy") == "exp"

    def test_cancel(self):
        node = _make_canvas_node("Cancel", "cancel", cancel_target="tag",
                                 cancel_tag="navigation")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.kind == NodeKind.CANCEL
        assert n.get_param_value("target") == "tag"
        assert n.get_param_value("tag") == "navigation"

    def test_abort(self):
        node = _make_canvas_node("Abort", "abort", abort_reason="emergency stop",
                                 abort_emergency="true")
        ir, _ = self.converter.convert(_make_graph_data([node]))
        n = ir.nodes[0]
        assert n.kind == NodeKind.ABORT
        assert n.get_param_value("reason") == "emergency stop"
        assert n.get_param_value("emergency") is True


# ===========================================================================
# 2. IR → Code generation
# ===========================================================================

class TestIRToCode:
    """Test IRToCode._generate_node_code for each new node type."""

    def setup_method(self):
        self.gen = IRToCode()

    def test_wait_duration_code(self):
        ir = _ir_with_single_node(NodeKind.WAIT, "builtin.wait",
                                  {"mode": "duration", "duration": 2.5})
        code, _, _ = self.gen.generate(ir)
        assert "time.sleep(2.5)" in code

    def test_wait_event_code(self):
        ir = _ir_with_single_node(NodeKind.WAIT, "builtin.wait",
                                  {"mode": "event", "event_name": "ready", "timeout": 10})
        code, _, _ = self.gen.generate(ir)
        assert "event_bus.wait('ready', timeout=10)" in code

    def test_gate_condition_code(self):
        ir = _ir_with_single_node(NodeKind.GATE, "builtin.gate",
                                  {"mode": "condition", "condition_expr": "x > 5"})
        code, _, _ = self.gen.generate(ir)
        assert "if x > 5:" in code

    def test_gate_event_code(self):
        ir = _ir_with_single_node(NodeKind.GATE, "builtin.gate",
                                  {"mode": "event", "event_name": "door_open"})
        code, _, _ = self.gen.generate(ir)
        assert "event_bus.check('door_open')" in code

    def test_timeout_code(self):
        ir = _ir_with_single_node(NodeKind.TIMEOUT, "builtin.timeout",
                                  {"duration": 5.0})
        code, _, _ = self.gen.generate(ir)
        assert "_timeout_start = time.time()" in code
        assert "5.0" in code

    def test_retry_fixed_code(self):
        ir = _ir_with_single_node(NodeKind.RETRY, "builtin.retry",
                                  {"max_attempts": 3, "backoff_sec": 1.0, "strategy": "fixed"})
        code, _, _ = self.gen.generate(ir)
        assert "for _attempt in range(1, 3 + 1):" in code
        assert "time.sleep(1.0)" in code

    def test_retry_exp_code(self):
        ir = _ir_with_single_node(NodeKind.RETRY, "builtin.retry",
                                  {"max_attempts": 5, "backoff_sec": 0.5, "strategy": "exp"})
        code, _, _ = self.gen.generate(ir)
        assert "2 ** _attempt" in code

    def test_cancel_current_code(self):
        ir = _ir_with_single_node(NodeKind.CANCEL, "builtin.cancel",
                                  {"target": "current", "tag": ""})
        code, _, _ = self.gen.generate(ir)
        assert "scheduler.cancel('current')" in code

    def test_cancel_tag_code(self):
        ir = _ir_with_single_node(NodeKind.CANCEL, "builtin.cancel",
                                  {"target": "tag", "tag": "nav"})
        code, _, _ = self.gen.generate(ir)
        assert "scheduler.cancel('tag', 'nav')" in code

    def test_abort_code(self):
        ir = _ir_with_single_node(NodeKind.ABORT, "builtin.abort",
                                  {"reason": "fail", "emergency": True})
        code, _, _ = self.gen.generate(ir)
        assert "raise AbortWorkflow('fail', emergency=True)" in code


# ===========================================================================
# 3. IR → Canvas → IR roundtrip
# ===========================================================================

class TestRoundtrip:
    """Test IR → Canvas → IR preserves params for all 6 new nodes."""

    def _roundtrip(self, kind, schema_id, params):
        ir_orig = _ir_with_single_node(kind, schema_id, params)
        canvas_data, _ = IRToCanvas().convert(ir_orig)
        ir_back, _ = CanvasToIR().convert(canvas_data)
        assert len(ir_back.nodes) == 1
        return ir_back.nodes[0]

    def test_wait_roundtrip(self):
        n = self._roundtrip(NodeKind.WAIT, "builtin.wait",
                            {"mode": "event", "duration": 2.0,
                             "event_name": "sensor", "timeout": 10.0})
        assert n.get_param_value("mode") == "event"
        assert n.get_param_value("duration") == 2.0
        assert n.get_param_value("event_name") == "sensor"
        assert n.get_param_value("timeout") == 10.0

    def test_gate_roundtrip(self):
        n = self._roundtrip(NodeKind.GATE, "builtin.gate",
                            {"mode": "condition", "condition_expr": "a > b",
                             "event_name": ""})
        assert n.get_param_value("mode") == "condition"
        assert n.get_param_value("condition_expr") == "a > b"

    def test_timeout_roundtrip(self):
        n = self._roundtrip(NodeKind.TIMEOUT, "builtin.timeout",
                            {"duration": 7.5, "on": "next_step"})
        assert n.get_param_value("duration") == 7.5

    def test_retry_roundtrip(self):
        n = self._roundtrip(NodeKind.RETRY, "builtin.retry",
                            {"max_attempts": 5, "backoff_sec": 2.0,
                             "strategy": "exp"})
        assert n.get_param_value("max_attempts") == 5
        assert n.get_param_value("backoff_sec") == 2.0
        assert n.get_param_value("strategy") == "exp"

    def test_cancel_roundtrip(self):
        n = self._roundtrip(NodeKind.CANCEL, "builtin.cancel",
                            {"target": "tag", "tag": "nav"})
        assert n.get_param_value("target") == "tag"
        assert n.get_param_value("tag") == "nav"

    def test_abort_roundtrip(self):
        n = self._roundtrip(NodeKind.ABORT, "builtin.abort",
                            {"reason": "halt", "emergency": True})
        assert n.get_param_value("reason") == "halt"
        assert n.get_param_value("emergency") is True
