#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for heartbeat panel data model — Circle 1 Step 1.1.

Coverage
--------
- HBNodeKind constants
- HBGraphNode: to_dict / from_dict / roundtrip
- HBGraphEdge: to_dict / from_dict / roundtrip
- HBGraphData: to_dict / from_dict / roundtrip
- HBGraphData.default_template(): shape and connectivity
- Missing/malformed key tolerance in from_dict()
- HBEventStateEntry: to_dict / from_dict / defaults
- HBIOMapping: to_dict / from_dict / defaults

Isolation
---------
- No QApplication. No Qt imports. Pure-Python data model only.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bin.layout.behavior_panel import (  # noqa: E402
    HBDraftPayload,
    HBEventStateEntry,
    HBGraphData,
    HBGraphEdge,
    HBGraphNode,
    HBIOMapping,
    HBNodeKind,
    HBPolicyDraft,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_node(kind: str = HBNodeKind.SOURCE) -> HBGraphNode:
    return HBGraphNode(
        node_id="n1",
        kind=kind,
        label="Test Node",
        x=10.0,
        y=20.0,
        params={"sensor": "imu"},
    )


def _make_edge() -> HBGraphEdge:
    return HBGraphEdge(edge_id="e1", source_id="n1", target_id="n2")


# ===========================================================================
# HBNodeKind
# ===========================================================================

class TestHBNodeKind(unittest.TestCase):

    def test_all_list_contains_five_kinds(self):
        self.assertEqual(len(HBNodeKind.ALL), 5)

    def test_all_list_contains_expected_kinds(self):
        expected = {
            HBNodeKind.SOURCE,
            HBNodeKind.SENSOR_READ,
            HBNodeKind.SENSOR_TRANSFORM,
            HBNodeKind.MOVEMENT_BLEND,
            HBNodeKind.SAFETY_GATE,
        }
        self.assertEqual(set(HBNodeKind.ALL), expected)

    def test_kind_strings_are_snake_case(self):
        for kind in HBNodeKind.ALL:
            self.assertRegex(kind, r"^[a-z][a-z0-9_]*$",
                             msg=f"Kind '{kind}' is not snake_case")

    def test_source_constant(self):
        self.assertEqual(HBNodeKind.SOURCE, "heartbeat_source")

    def test_movement_blend_constant(self):
        self.assertEqual(HBNodeKind.MOVEMENT_BLEND, "movement_blend")


# ===========================================================================
# HBGraphNode
# ===========================================================================

class TestHBGraphNode(unittest.TestCase):

    def test_to_dict_has_all_keys(self):
        node = _make_node()
        d = node.to_dict()
        for key in ("node_id", "kind", "label", "x", "y", "params"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        node = _make_node(HBNodeKind.SENSOR_READ)
        d = node.to_dict()
        self.assertEqual(d["node_id"], "n1")
        self.assertEqual(d["kind"], HBNodeKind.SENSOR_READ)
        self.assertEqual(d["label"], "Test Node")
        self.assertAlmostEqual(d["x"], 10.0)
        self.assertAlmostEqual(d["y"], 20.0)
        self.assertEqual(d["params"], {"sensor": "imu"})

    def test_from_dict_roundtrip(self):
        node = _make_node()
        restored = HBGraphNode.from_dict(node.to_dict())
        self.assertEqual(restored.node_id, node.node_id)
        self.assertEqual(restored.kind, node.kind)
        self.assertEqual(restored.label, node.label)
        self.assertAlmostEqual(restored.x, node.x)
        self.assertAlmostEqual(restored.y, node.y)
        self.assertEqual(restored.params, node.params)

    def test_from_dict_missing_keys_uses_defaults(self):
        node = HBGraphNode.from_dict({})
        self.assertEqual(node.node_id, "")
        self.assertEqual(node.kind, HBNodeKind.SOURCE)
        self.assertEqual(node.label, "")
        self.assertAlmostEqual(node.x, 0.0)
        self.assertAlmostEqual(node.y, 0.0)
        self.assertEqual(node.params, {})

    def test_to_dict_params_is_copy(self):
        node = _make_node()
        d = node.to_dict()
        d["params"]["extra"] = "modified"
        # Original params unchanged
        self.assertNotIn("extra", node.params)

    def test_from_dict_x_y_coercion(self):
        node = HBGraphNode.from_dict({"x": "3", "y": "7.5"})
        self.assertAlmostEqual(node.x, 3.0)
        self.assertAlmostEqual(node.y, 7.5)


# ===========================================================================
# HBGraphEdge
# ===========================================================================

class TestHBGraphEdge(unittest.TestCase):

    def test_to_dict_has_all_keys(self):
        edge = _make_edge()
        d = edge.to_dict()
        for key in ("edge_id", "source_id", "target_id"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        edge = _make_edge()
        d = edge.to_dict()
        self.assertEqual(d["edge_id"], "e1")
        self.assertEqual(d["source_id"], "n1")
        self.assertEqual(d["target_id"], "n2")

    def test_from_dict_roundtrip(self):
        edge = _make_edge()
        restored = HBGraphEdge.from_dict(edge.to_dict())
        self.assertEqual(restored.edge_id, edge.edge_id)
        self.assertEqual(restored.source_id, edge.source_id)
        self.assertEqual(restored.target_id, edge.target_id)

    def test_from_dict_missing_keys_empty_strings(self):
        edge = HBGraphEdge.from_dict({})
        self.assertEqual(edge.edge_id, "")
        self.assertEqual(edge.source_id, "")
        self.assertEqual(edge.target_id, "")


# ===========================================================================
# HBGraphData
# ===========================================================================

class TestHBGraphData(unittest.TestCase):

    def test_empty_graph_to_dict(self):
        g = HBGraphData()
        d = g.to_dict()
        self.assertEqual(d["nodes"], [])
        self.assertEqual(d["edges"], [])

    def test_from_dict_empty(self):
        g = HBGraphData.from_dict({})
        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, [])

    def test_roundtrip_preserves_nodes_and_edges(self):
        g = HBGraphData(
            nodes=[
                HBGraphNode("n1", HBNodeKind.SOURCE, "Source", 0.0, 0.0),
                HBGraphNode("n2", HBNodeKind.SENSOR_READ, "IMU", 200.0, 0.0),
            ],
            edges=[HBGraphEdge("e0", "n1", "n2")],
        )
        d = g.to_dict()
        restored = HBGraphData.from_dict(d)
        self.assertEqual(len(restored.nodes), 2)
        self.assertEqual(len(restored.edges), 1)
        self.assertEqual(restored.nodes[0].node_id, "n1")
        self.assertEqual(restored.edges[0].source_id, "n1")
        self.assertEqual(restored.edges[0].target_id, "n2")

    def test_roundtrip_node_kinds_preserved(self):
        g = HBGraphData(
            nodes=[HBGraphNode(f"n{i}", kind, kind, float(i * 100), 0.0)
                   for i, kind in enumerate(HBNodeKind.ALL)],
            edges=[],
        )
        restored = HBGraphData.from_dict(g.to_dict())
        kinds = [n.kind for n in restored.nodes]
        self.assertEqual(kinds, HBNodeKind.ALL)

    def test_roundtrip_params_preserved(self):
        g = HBGraphData(
            nodes=[HBGraphNode("n1", HBNodeKind.SENSOR_READ, "IMU",
                               0.0, 0.0, params={"sensor": "imu", "interval": 0.1})],
            edges=[],
        )
        restored = HBGraphData.from_dict(g.to_dict())
        self.assertEqual(restored.nodes[0].params["sensor"], "imu")
        self.assertAlmostEqual(restored.nodes[0].params["interval"], 0.1)


# ===========================================================================
# HBGraphData.default_template()
# ===========================================================================

class TestHBGraphDataDefaultTemplate(unittest.TestCase):

    def setUp(self):
        self.template = HBGraphData.default_template()

    def test_has_three_nodes(self):
        self.assertEqual(len(self.template.nodes), 3)

    def test_has_two_edges(self):
        self.assertEqual(len(self.template.edges), 2)

    def test_first_node_is_source(self):
        self.assertEqual(self.template.nodes[0].kind, HBNodeKind.SOURCE)

    def test_second_node_is_sensor_read(self):
        self.assertEqual(self.template.nodes[1].kind, HBNodeKind.SENSOR_READ)

    def test_third_node_is_movement_blend(self):
        self.assertEqual(self.template.nodes[2].kind, HBNodeKind.MOVEMENT_BLEND)

    def test_edges_connect_linearly(self):
        src_id = self.template.nodes[0].node_id
        mid_id = self.template.nodes[1].node_id
        end_id = self.template.nodes[2].node_id

        edge_pairs = {(e.source_id, e.target_id) for e in self.template.edges}
        self.assertIn((src_id, mid_id), edge_pairs)
        self.assertIn((mid_id, end_id), edge_pairs)

    def test_sensor_read_has_sensor_param(self):
        sensor_node = self.template.nodes[1]
        self.assertIn("sensor", sensor_node.params)
        self.assertEqual(sensor_node.params["sensor"], "imu")

    def test_node_ids_are_unique(self):
        ids = [n.node_id for n in self.template.nodes]
        self.assertEqual(len(ids), len(set(ids)))

    def test_edge_ids_are_unique(self):
        ids = [e.edge_id for e in self.template.edges]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_nodes_have_positive_x(self):
        for node in self.template.nodes:
            self.assertGreaterEqual(node.x, 0.0, msg=f"node {node.node_id} has x={node.x}")

    def test_template_roundtrip(self):
        template = HBGraphData.default_template()
        restored = HBGraphData.from_dict(template.to_dict())
        self.assertEqual(len(restored.nodes), len(template.nodes))
        self.assertEqual(len(restored.edges), len(template.edges))
        for orig, res in zip(template.nodes, restored.nodes):
            self.assertEqual(orig.node_id, res.node_id)
            self.assertEqual(orig.kind, res.kind)

    def test_multiple_calls_return_independent_objects(self):
        t1 = HBGraphData.default_template()
        t2 = HBGraphData.default_template()
        t1.nodes[0].label = "MUTATED"
        self.assertNotEqual(t2.nodes[0].label, "MUTATED")


# ===========================================================================
# HBEventStateEntry
# ===========================================================================

class TestHBEventStateEntry(unittest.TestCase):

    def test_to_dict_has_all_keys(self):
        e = HBEventStateEntry("tick", "active", "heartbeat_loop", "2026-01-01", "ok")
        d = e.to_dict()
        for key in ("event_name", "state", "source", "timestamp", "reason"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        e = HBEventStateEntry("tick", "active", "heartbeat_loop", "T0", "test")
        d = e.to_dict()
        self.assertEqual(d["event_name"], "tick")
        self.assertEqual(d["state"], "active")
        self.assertEqual(d["source"], "heartbeat_loop")
        self.assertEqual(d["timestamp"], "T0")
        self.assertEqual(d["reason"], "test")

    def test_from_dict_roundtrip(self):
        e = HBEventStateEntry("imu_drift", "error", "sensor_read", "T1", "exceeded threshold")
        restored = HBEventStateEntry.from_dict(e.to_dict())
        self.assertEqual(restored.event_name, e.event_name)
        self.assertEqual(restored.state, e.state)
        self.assertEqual(restored.source, e.source)
        self.assertEqual(restored.timestamp, e.timestamp)
        self.assertEqual(restored.reason, e.reason)

    def test_from_dict_missing_keys_defaults(self):
        e = HBEventStateEntry.from_dict({})
        self.assertEqual(e.event_name, "")
        self.assertEqual(e.state, "inactive")
        self.assertEqual(e.source, "")
        self.assertEqual(e.timestamp, "")
        self.assertEqual(e.reason, "")

    def test_reason_default_empty_string(self):
        e = HBEventStateEntry("tick", "active", "src", "T0")
        self.assertEqual(e.reason, "")

    def test_state_values_accepted(self):
        for state in ("active", "inactive", "pending", "error"):
            e = HBEventStateEntry.from_dict({"state": state})
            self.assertEqual(e.state, state)


# ===========================================================================
# HBIOMapping
# ===========================================================================

class TestHBIOMapping(unittest.TestCase):

    def test_to_dict_has_all_keys(self):
        m = HBIOMapping("imu.pitch", "posture.pitch_comp", "inbound", "active")
        d = m.to_dict()
        for key in ("signal", "target_param", "direction", "status"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        m = HBIOMapping("imu.roll", "posture.roll_comp", "inbound", "unsupported")
        d = m.to_dict()
        self.assertEqual(d["signal"], "imu.roll")
        self.assertEqual(d["target_param"], "posture.roll_comp")
        self.assertEqual(d["direction"], "inbound")
        self.assertEqual(d["status"], "unsupported")

    def test_from_dict_roundtrip(self):
        m = HBIOMapping("imu.pitch", "walk.speed_scale", "outbound", "unmapped")
        restored = HBIOMapping.from_dict(m.to_dict())
        self.assertEqual(restored.signal, m.signal)
        self.assertEqual(restored.target_param, m.target_param)
        self.assertEqual(restored.direction, m.direction)
        self.assertEqual(restored.status, m.status)

    def test_from_dict_missing_keys_defaults(self):
        m = HBIOMapping.from_dict({})
        self.assertEqual(m.signal, "")
        self.assertEqual(m.target_param, "")
        self.assertEqual(m.direction, "inbound")
        self.assertEqual(m.status, "active")

    def test_status_default_active(self):
        m = HBIOMapping("imu.pitch", "posture.pitch_comp", "inbound")
        self.assertEqual(m.status, "active")

    def test_direction_outbound_accepted(self):
        m = HBIOMapping.from_dict({"direction": "outbound"})
        self.assertEqual(m.direction, "outbound")

    def test_roundtrip_all_status_values(self):
        for status in ("active", "unmapped", "unsupported"):
            m = HBIOMapping("s", "t", "inbound", status)
            restored = HBIOMapping.from_dict(m.to_dict())
            self.assertEqual(restored.status, status)


# ===========================================================================
# HBPolicyDraft
# ===========================================================================

class TestHBPolicyDraft(unittest.TestCase):

    def test_default_values(self):
        p = HBPolicyDraft.default()
        self.assertEqual(p.tick_ms, 100)
        self.assertEqual(p.timeout_ms, 5000)
        self.assertEqual(p.fail_policy, "stop_behavior")
        self.assertAlmostEqual(p.max_override, 0.2)
        self.assertEqual(p.safety_mode, "strict")

    def test_to_dict_has_all_keys(self):
        p = HBPolicyDraft.default()
        d = p.to_dict()
        for key in ("tick_ms", "timeout_ms", "fail_policy", "max_override", "safety_mode"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        p = HBPolicyDraft(tick_ms=50, timeout_ms=2000, fail_policy="continue",
                          max_override=0.5, safety_mode="relaxed")
        d = p.to_dict()
        self.assertEqual(d["tick_ms"], 50)
        self.assertEqual(d["timeout_ms"], 2000)
        self.assertEqual(d["fail_policy"], "continue")
        self.assertAlmostEqual(d["max_override"], 0.5)
        self.assertEqual(d["safety_mode"], "relaxed")

    def test_from_dict_roundtrip(self):
        p = HBPolicyDraft(tick_ms=200, timeout_ms=8000, fail_policy="warn_only",
                          max_override=0.1, safety_mode="disabled")
        restored = HBPolicyDraft.from_dict(p.to_dict())
        self.assertEqual(restored.tick_ms, p.tick_ms)
        self.assertEqual(restored.timeout_ms, p.timeout_ms)
        self.assertEqual(restored.fail_policy, p.fail_policy)
        self.assertAlmostEqual(restored.max_override, p.max_override)
        self.assertEqual(restored.safety_mode, p.safety_mode)

    def test_from_dict_missing_keys_uses_defaults(self):
        p = HBPolicyDraft.from_dict({})
        self.assertEqual(p.tick_ms, 100)
        self.assertEqual(p.timeout_ms, 5000)
        self.assertEqual(p.fail_policy, "stop_behavior")
        self.assertAlmostEqual(p.max_override, 0.2)
        self.assertEqual(p.safety_mode, "strict")

    def test_fail_policies_class_var(self):
        self.assertIn("stop_behavior", HBPolicyDraft.FAIL_POLICIES)
        self.assertIn("continue", HBPolicyDraft.FAIL_POLICIES)
        self.assertIn("warn_only", HBPolicyDraft.FAIL_POLICIES)

    def test_safety_modes_class_var(self):
        self.assertIn("strict", HBPolicyDraft.SAFETY_MODES)
        self.assertIn("relaxed", HBPolicyDraft.SAFETY_MODES)
        self.assertIn("disabled", HBPolicyDraft.SAFETY_MODES)


# ===========================================================================
# HBDraftPayload
# ===========================================================================

class TestHBDraftPayload(unittest.TestCase):

    def test_default_has_expected_authoring_mode(self):
        # Default authoring mode updated to "navigator" (product simplification)
        d = HBDraftPayload.default()
        self.assertEqual(d.authoring_mode, "navigator")

    def test_default_has_canvas_graph(self):
        d = HBDraftPayload.default()
        self.assertIsInstance(d.canvas_graph, dict)
        self.assertIn("nodes", d.canvas_graph)
        self.assertIn("edges", d.canvas_graph)

    def test_default_has_script(self):
        d = HBDraftPayload.default()
        self.assertIsInstance(d.script, str)
        self.assertGreater(len(d.script), 0)

    def test_default_policy_is_hb_policy_draft(self):
        d = HBDraftPayload.default()
        self.assertIsInstance(d.policy, HBPolicyDraft)
        self.assertEqual(d.policy.tick_ms, 100)

    def test_default_io_bindings_and_event_schema_empty(self):
        d = HBDraftPayload.default()
        self.assertEqual(d.io_bindings, [])
        self.assertEqual(d.event_schema, [])

    def test_to_dict_has_all_keys(self):
        d = HBDraftPayload.default().to_dict()
        for key in ("script", "canvas_graph", "policy", "io_bindings",
                    "event_schema", "authoring_mode"):
            self.assertIn(key, d)

    def test_to_dict_policy_is_dict(self):
        d = HBDraftPayload.default().to_dict()
        self.assertIsInstance(d["policy"], dict)
        self.assertIn("tick_ms", d["policy"])

    def test_from_dict_roundtrip(self):
        payload = HBDraftPayload(
            script="# test script",
            canvas_graph=HBGraphData.default_template().to_dict(),
            policy=HBPolicyDraft(tick_ms=50, timeout_ms=3000, fail_policy="continue",
                                 max_override=0.3, safety_mode="relaxed"),
            io_bindings=[{"signal": "imu.pitch", "target_param": "p", "direction": "inbound", "status": "active"}],
            event_schema=[{"event_name": "tick", "state": "active", "source": "hb", "timestamp": "T0", "reason": ""}],
            authoring_mode="script",
        )
        d = payload.to_dict()
        restored = HBDraftPayload.from_dict(d)
        self.assertEqual(restored.script, payload.script)
        self.assertEqual(restored.authoring_mode, "script")
        self.assertEqual(restored.policy.tick_ms, 50)
        self.assertEqual(restored.policy.fail_policy, "continue")
        self.assertEqual(len(restored.io_bindings), 1)
        self.assertEqual(restored.io_bindings[0]["signal"], "imu.pitch")
        self.assertEqual(len(restored.event_schema), 1)
        self.assertEqual(restored.event_schema[0]["event_name"], "tick")

    def test_from_dict_empty_uses_defaults(self):
        p = HBDraftPayload.from_dict({})
        self.assertEqual(p.script, "")
        self.assertEqual(p.canvas_graph, {})
        self.assertEqual(p.authoring_mode, "navigator")  # updated default
        self.assertEqual(p.io_bindings, [])
        self.assertEqual(p.event_schema, [])
        self.assertEqual(p.policy.tick_ms, 100)

    def test_canvas_graph_roundtrip_preserves_nodes(self):
        template = HBGraphData.default_template()
        payload = HBDraftPayload(canvas_graph=template.to_dict())
        d = payload.to_dict()
        restored = HBDraftPayload.from_dict(d)
        graph = HBGraphData.from_dict(restored.canvas_graph)
        self.assertEqual(len(graph.nodes), len(template.nodes))
        self.assertEqual(len(graph.edges), len(template.edges))

    def test_authoring_mode_values(self):
        for mode in ("canvas", "script"):
            p = HBDraftPayload.from_dict({"authoring_mode": mode})
            self.assertEqual(p.authoring_mode, mode)

    def test_io_bindings_independence(self):
        p = HBDraftPayload.default()
        d = p.to_dict()
        d["io_bindings"].append({"signal": "x"})
        # original unchanged
        self.assertEqual(p.io_bindings, [])

    def test_multiple_defaults_are_independent(self):
        p1 = HBDraftPayload.default()
        p2 = HBDraftPayload.default()
        p1.script = "MUTATED"
        self.assertNotEqual(p2.script, "MUTATED")


if __name__ == "__main__":
    unittest.main()
