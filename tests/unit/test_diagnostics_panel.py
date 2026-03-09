"""Tests for Cycle 1 STAGE-04: Diagnostics Drill-Down UX.

Covers:
  - error_ux module: get_operator_text, get_stage_category, format_node_diagnostics,
    extract_failed_nodes_info (pure Python, no Qt)
  - DiagnosticsPanel widget: visibility, node selector, friendly/raw toggle,
    navigate signal, close/clear behaviour
  - graph_scene.get_node_item: node lookup by ID
"""

import os
import sys
import unittest

# Headless Qt for PySide6 tests
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure project root is on path
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_run_result(
    failed_nodes=None,
    results=None,
    status="failed",
    compat_path=False,
    compat_reason="",
):
    """Build a minimal run_result dict for testing."""
    return {
        "status": status,
        "reason": "execute_failed" if status == "failed" else "",
        "node_count": 3,
        "results": results or {},
        "diagnostics": {
            "failed_nodes": failed_nodes or [],
            "executed_count": len(results or {}),
            "compat_path": compat_path,
            "compat_reason": compat_reason,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: Pure-Python error_ux logic (no Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorUxModule(unittest.TestCase):
    """Tests for bin/core/error_ux.py — no Qt required."""

    def setUp(self):
        from bin.core.error_ux import (
            get_operator_text,
            get_stage_category,
            format_node_diagnostics,
            extract_failed_nodes_info,
            DISPLAY_KEY_ORDER,
            REASON_OPERATOR_TEXT,
        )
        self.get_operator_text      = get_operator_text
        self.get_stage_category     = get_stage_category
        self.format_node_diagnostics = format_node_diagnostics
        self.extract_failed_nodes_info = extract_failed_nodes_info
        self.DISPLAY_KEY_ORDER      = DISPLAY_KEY_ORDER
        self.REASON_OPERATOR_TEXT   = REASON_OPERATOR_TEXT

    # ── get_operator_text ────────────────────────────────────────────────

    def test_known_canonical_reason(self):
        text = self.get_operator_text("execute_failed")
        self.assertIn("execution failed", text.lower())

    def test_known_spot_reason(self):
        text = self.get_operator_text("spot_estop_active")
        self.assertIn("e-stop", text.lower())

    def test_known_cyberdog_reason(self):
        text = self.get_operator_text("cyberdog_ros2_unavailable")
        self.assertIn("ros2", text.lower())

    def test_unknown_reason_fallback(self):
        text = self.get_operator_text("totally_unknown_code")
        self.assertIn("totally_unknown_code", text)

    def test_all_registered_reasons_have_text(self):
        for code in self.REASON_OPERATOR_TEXT:
            self.assertIsInstance(self.REASON_OPERATOR_TEXT[code], str)
            self.assertTrue(len(self.REASON_OPERATOR_TEXT[code]) > 0)

    # ── get_stage_category ───────────────────────────────────────────────

    def test_known_stages(self):
        self.assertEqual(self.get_stage_category("open_session"), "Connection")
        self.assertEqual(self.get_stage_category("preflight"),    "Preflight / Safety")
        self.assertEqual(self.get_stage_category("execute"),      "Execution")
        self.assertEqual(self.get_stage_category("close_session"), "Cleanup")

    def test_unknown_stage_title_case(self):
        label = self.get_stage_category("custom_stage")
        self.assertEqual(label, "Custom Stage")

    # ── format_node_diagnostics ──────────────────────────────────────────

    def test_basic_fields_present(self):
        result = self.format_node_diagnostics(
            1,
            {"error": "execute_failed", "status": "failed"},
        )
        self.assertIn("node_id",       result)
        self.assertIn("status",        result)
        self.assertIn("reason",        result)
        self.assertIn("operator_text", result)
        self.assertIn("message",       result)
        self.assertIn("retryable",     result)
        self.assertIn("context",       result)

    def test_node_id_normalised_to_str(self):
        result = self.format_node_diagnostics(42, {"error": "execute_failed"})
        self.assertEqual(result["node_id"], "42")

    def test_node_name_used_when_provided(self):
        result = self.format_node_diagnostics(7, {"error": "x"}, node_name="MyNode")
        self.assertEqual(result["node_name"], "MyNode")

    def test_node_name_defaults_to_str_id(self):
        result = self.format_node_diagnostics(7, {"error": "x"})
        self.assertEqual(result["node_name"], "7")

    def test_reason_extracted_from_error_key(self):
        result = self.format_node_diagnostics(1, {"error": "spot_estop_active"})
        self.assertEqual(result["reason"], "spot_estop_active")

    def test_reason_takes_reason_key_over_error(self):
        result = self.format_node_diagnostics(
            1, {"error": "fallback", "reason": "explicit_reason"}
        )
        self.assertEqual(result["reason"], "explicit_reason")

    def test_operator_text_populated_for_known_reason(self):
        result = self.format_node_diagnostics(1, {"error": "execute_failed"})
        self.assertIn("execution", result["operator_text"].lower())

    def test_context_from_diagnostics_key(self):
        result = self.format_node_diagnostics(
            1, {"error": "x", "diagnostics": {"extra": "val"}}
        )
        self.assertEqual(result["context"], {"extra": "val"})

    def test_compat_path_included_from_run_result(self):
        run_result = _make_run_result(compat_path=True, compat_reason="env_var_set")
        result = self.format_node_diagnostics(
            1, {"error": "execute_failed"}, run_result=run_result
        )
        self.assertTrue(result.get("compat_path"))
        self.assertEqual(result.get("compat_reason"), "env_var_set")

    def test_compat_path_absent_when_false(self):
        run_result = _make_run_result(compat_path=False)
        result = self.format_node_diagnostics(1, {"error": "x"}, run_result=run_result)
        self.assertNotIn("compat_path", result)

    def test_deterministic_key_order_leading_keys(self):
        """First keys in result must follow DISPLAY_KEY_ORDER."""
        result = self.format_node_diagnostics(5, {"error": "execute_failed"})
        keys = list(result.keys())
        ordered = self.DISPLAY_KEY_ORDER
        # The leading keys in result should be the DISPLAY_KEY_ORDER entries that exist
        expected_leading = [k for k in ordered if k in result]
        actual_leading   = keys[: len(expected_leading)]
        self.assertEqual(actual_leading, expected_leading)

    # ── extract_failed_nodes_info ────────────────────────────────────────

    def test_returns_empty_when_no_failures(self):
        run_result = _make_run_result(failed_nodes=[], status="success")
        infos = self.extract_failed_nodes_info(run_result)
        self.assertEqual(infos, [])

    def test_returns_one_info_per_failed_node(self):
        run_result = _make_run_result(
            failed_nodes=[1, 2],
            results={"1": {"error": "execute_failed"}, "2": {"error": "spot_no_session"}},
        )
        infos = self.extract_failed_nodes_info(run_result)
        self.assertEqual(len(infos), 2)

    def test_node_names_injected(self):
        run_result = _make_run_result(
            failed_nodes=[3],
            results={"3": {"error": "execute_failed"}},
        )
        infos = self.extract_failed_nodes_info(run_result, node_names={3: "ActionNode"})
        self.assertEqual(infos[0]["node_name"], "ActionNode")

    def test_fallback_result_for_missing_node(self):
        """If a failed node has no entry in results, a fallback dict is used."""
        run_result = _make_run_result(failed_nodes=[99], results={})
        infos = self.extract_failed_nodes_info(run_result)
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0]["node_id"], "99")

    def test_order_preserves_failed_nodes_list(self):
        run_result = _make_run_result(
            failed_nodes=[5, 3, 7],
            results={
                "5": {"error": "a"},
                "3": {"error": "b"},
                "7": {"error": "c"},
            },
        )
        infos = self.extract_failed_nodes_info(run_result)
        ids = [info["node_id"] for info in infos]
        self.assertEqual(ids, ["5", "3", "7"])


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: DiagnosticsPanel widget (PySide6)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticsPanel(unittest.TestCase):
    """Tests for bin/components/diagnostics_panel.DiagnosticsPanel."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _make_panel(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        return DiagnosticsPanel()

    def _sample_infos(self, count=1):
        from bin.core.error_ux import format_node_diagnostics
        return [
            format_node_diagnostics(i, {"error": "execute_failed"}, node_name=f"Node{i}")
            for i in range(1, count + 1)
        ]

    # ── Visibility ───────────────────────────────────────────────────────

    def test_initially_hidden(self):
        p = self._make_panel()
        self.assertTrue(p.isHidden())

    def test_show_diagnostics_makes_visible(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        self.assertFalse(p.isHidden())

    def test_clear_hides_panel(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        p.clear()
        self.assertTrue(p.isHidden())

    # ── Node selector combo ──────────────────────────────────────────────

    def test_combo_hidden_for_single_node(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        self.assertTrue(p._node_combo.isHidden())

    def test_combo_visible_for_multiple_nodes(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(3))
        self.assertFalse(p._node_combo.isHidden())

    def test_combo_populated_with_node_names(self):
        p = self._make_panel()
        infos = self._sample_infos(2)
        p.show_diagnostics(infos)
        self.assertEqual(p._node_combo.count(), 2)
        self.assertEqual(p._node_combo.itemText(0), "Node1")
        self.assertEqual(p._node_combo.itemText(1), "Node2")

    def test_start_index_sets_combo_index(self):
        p = self._make_panel()
        infos = self._sample_infos(3)
        p.show_diagnostics(infos, start_index=2)
        self.assertEqual(p._node_combo.currentIndex(), 2)
        self.assertEqual(p._current_index, 2)

    # ── Raw toggle ───────────────────────────────────────────────────────

    def test_friendly_view_default(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        self.assertFalse(p._raw_toggle.isChecked())
        self.assertFalse(p._scroll_area.isHidden())
        self.assertTrue(p._raw_text.isHidden())

    def test_raw_toggle_switches_views(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        p._raw_toggle.setChecked(True)
        self.assertTrue(p._scroll_area.isHidden())
        self.assertFalse(p._raw_text.isHidden())

    def test_raw_text_contains_json(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        p._raw_toggle.setChecked(True)
        text = p._raw_text.toPlainText()
        self.assertIn('"node_id"', text)
        self.assertIn('"execute_failed"', text)

    # ── Navigate signal ──────────────────────────────────────────────────

    def test_goto_emits_navigate_requested(self):
        p = self._make_panel()
        received = []
        p.navigate_requested.connect(lambda nid: received.append(nid))
        from bin.core.error_ux import format_node_diagnostics
        info = format_node_diagnostics(42, {"error": "execute_failed"})
        p.show_diagnostics([info])
        p._goto_btn.click()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], 42)

    def test_goto_emits_int_node_id(self):
        p = self._make_panel()
        received = []
        p.navigate_requested.connect(lambda nid: received.append(nid))
        from bin.core.error_ux import format_node_diagnostics
        info = format_node_diagnostics(7, {"error": "x"})
        p.show_diagnostics([info])
        p._goto_btn.click()
        self.assertIsInstance(received[0], int)

    def test_goto_noop_when_no_infos(self):
        p = self._make_panel()
        received = []
        p.navigate_requested.connect(lambda nid: received.append(nid))
        p._goto_btn.click()
        self.assertEqual(received, [])

    # ── Close ────────────────────────────────────────────────────────────

    def test_close_emits_closed_signal(self):
        p = self._make_panel()
        fired = []
        p.closed.connect(lambda: fired.append(1))
        p.show_diagnostics(self._sample_infos(1))
        # Click the close button
        close_btn = p.findChild(type(p._goto_btn), "diagCloseBtn")
        self.assertIsNotNone(close_btn)
        close_btn.click()
        self.assertEqual(len(fired), 1)

    def test_close_hides_panel(self):
        p = self._make_panel()
        p.show_diagnostics(self._sample_infos(1))
        close_btn = p.findChild(type(p._goto_btn), "diagCloseBtn")
        close_btn.click()
        self.assertTrue(p.isHidden())

    # ── Format value helpers ─────────────────────────────────────────────

    def test_format_value_stage(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertEqual(DiagnosticsPanel._format_value("stage", "open_session"), "Connection")
        self.assertEqual(DiagnosticsPanel._format_value("stage", "execute"), "Execution")

    def test_format_value_retryable(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertEqual(DiagnosticsPanel._format_value("retryable", True),  "Yes")
        self.assertEqual(DiagnosticsPanel._format_value("retryable", False), "No")

    def test_format_value_empty_context(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertEqual(DiagnosticsPanel._format_value("context", {}), "(empty)")

    def test_format_value_none(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertEqual(DiagnosticsPanel._format_value("message", None), "(none)")
        self.assertEqual(DiagnosticsPanel._format_value("message", ""),   "(none)")


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: graph_scene.get_node_item (PySide6)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNodeItem(unittest.TestCase):
    """Tests for GraphScene.get_node_item(node_id)."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _make_scene(self):
        from PySide6.QtCore import QPointF
        from bin.components.graph_scene import GraphScene
        scene = GraphScene()
        scene.create_node("Timer", QPointF(0, 0))
        scene.create_node("Timer", QPointF(100, 0))
        return scene

    def test_returns_item_for_existing_id(self):
        from PySide6.QtCore import QPointF
        from bin.components.graph_scene import GraphScene
        scene = GraphScene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)
        found = scene.get_node_item(nid)
        self.assertIsNotNone(found)
        self.assertIs(found, item)

    def test_returns_none_for_nonexistent_id(self):
        scene = self._make_scene()
        result = scene.get_node_item(99999)
        self.assertIsNone(result)

    def test_returns_correct_item_among_multiple(self):
        from PySide6.QtCore import QPointF
        from bin.components.graph_scene import GraphScene
        scene = GraphScene()
        item_a = scene.create_node("Timer", QPointF(0, 0))
        item_b = scene.create_node("Timer", QPointF(200, 0))
        nid_a = item_a.data(12)
        nid_b = item_b.data(12)
        self.assertIs(scene.get_node_item(nid_a), item_a)
        self.assertIs(scene.get_node_item(nid_b), item_b)

    def test_returns_none_after_node_deleted(self):
        from PySide6.QtCore import QPointF
        from bin.components.graph_scene import GraphScene
        scene = GraphScene()
        item = scene.create_node("Timer", QPointF(0, 0))
        nid = item.data(12)
        # Select and delete
        item.setSelected(True)
        scene._delete_items([item])
        self.assertIsNone(scene.get_node_item(nid))


# ─────────────────────────────────────────────────────────────────────────────
# Suite 4: Telemetry pointer extraction — pure-Python
# ─────────────────────────────────────────────────────────────────────────────

class TestTelemetryPointers(unittest.TestCase):
    """Verify that format_node_diagnostics surfaces telemetry pointer keys."""

    def setUp(self):
        from bin.core.error_ux import format_node_diagnostics, DISPLAY_KEY_ORDER
        self.fmt = format_node_diagnostics
        self.KEY_ORDER = DISPLAY_KEY_ORDER

    # ── trace_id ─────────────────────────────────────────────────────────

    def test_trace_id_from_node_result_included(self):
        result = self.fmt(1, {"error": "execute_failed", "trace_id": "abc-123"})
        self.assertEqual(result.get("trace_id"), "abc-123")

    def test_trace_id_absent_when_not_in_node_result_and_no_run_result(self):
        result = self.fmt(1, {"error": "execute_failed"})
        self.assertNotIn("trace_id", result)

    def test_trace_id_absent_when_node_result_has_empty_string(self):
        result = self.fmt(1, {"error": "x", "trace_id": ""})
        self.assertNotIn("trace_id", result)

    def test_trace_id_from_behavior_trace_ids_map(self):
        """Run_result.diagnostics.behavior_trace_ids[node_id] used as fallback."""
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [5],
                "behavior_trace_ids": {"5": "trace-from-map"},
                "mission_trace_id": "",
            },
        }
        result = self.fmt(5, {"error": "execute_failed"}, run_result=run_result)
        self.assertEqual(result.get("trace_id"), "trace-from-map")

    def test_trace_id_node_result_wins_over_map(self):
        """node_result.trace_id takes precedence over behavior_trace_ids map."""
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [5],
                "behavior_trace_ids": {"5": "trace-from-map"},
                "mission_trace_id": "",
            },
        }
        result = self.fmt(
            5, {"error": "execute_failed", "trace_id": "direct-trace"}, run_result=run_result
        )
        self.assertEqual(result.get("trace_id"), "direct-trace")

    def test_trace_id_absent_when_map_entry_missing(self):
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [7],
                "behavior_trace_ids": {},
                "mission_trace_id": "",
            },
        }
        result = self.fmt(7, {"error": "execute_failed"}, run_result=run_result)
        self.assertNotIn("trace_id", result)

    # ── mission_trace_id ──────────────────────────────────────────────────

    def test_mission_trace_id_from_run_result_diagnostics(self):
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [1],
                "behavior_trace_ids": {},
                "mission_trace_id": "mission-uuid-xyz",
            },
        }
        result = self.fmt(1, {"error": "execute_failed"}, run_result=run_result)
        self.assertEqual(result.get("mission_trace_id"), "mission-uuid-xyz")

    def test_mission_trace_id_absent_when_empty_string(self):
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [1],
                "behavior_trace_ids": {},
                "mission_trace_id": "",
            },
        }
        result = self.fmt(1, {"error": "execute_failed"}, run_result=run_result)
        self.assertNotIn("mission_trace_id", result)

    def test_mission_trace_id_absent_when_no_run_result(self):
        result = self.fmt(1, {"error": "execute_failed"})
        self.assertNotIn("mission_trace_id", result)

    # ── Deterministic display order ───────────────────────────────────────

    def test_trace_id_in_display_key_order(self):
        self.assertIn("trace_id", self.KEY_ORDER)

    def test_mission_trace_id_in_display_key_order(self):
        self.assertIn("mission_trace_id", self.KEY_ORDER)

    def test_trace_id_after_context_in_display_order(self):
        idx_context = self.KEY_ORDER.index("context")
        idx_trace   = self.KEY_ORDER.index("trace_id")
        self.assertGreater(idx_trace, idx_context)

    def test_mission_trace_after_trace_in_display_order(self):
        idx_trace    = self.KEY_ORDER.index("trace_id")
        idx_mission  = self.KEY_ORDER.index("mission_trace_id")
        self.assertGreater(idx_mission, idx_trace)

    def test_telemetry_keys_appear_after_context_in_result(self):
        """When trace_id is present, it must appear after context in result keys."""
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [1],
                "behavior_trace_ids": {},
                "mission_trace_id": "m-trace",
            },
        }
        node_result = {"error": "execute_failed", "trace_id": "t-123"}
        result = self.fmt(1, node_result, run_result=run_result)
        keys = list(result.keys())
        if "context" in keys and "trace_id" in keys:
            self.assertLess(keys.index("context"), keys.index("trace_id"))

    def test_mission_trace_appears_after_trace_id_in_result(self):
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [1],
                "behavior_trace_ids": {},
                "mission_trace_id": "m-trace",
            },
        }
        node_result = {"error": "execute_failed", "trace_id": "t-123"}
        result = self.fmt(1, node_result, run_result=run_result)
        keys = list(result.keys())
        if "trace_id" in keys and "mission_trace_id" in keys:
            self.assertLess(keys.index("trace_id"), keys.index("mission_trace_id"))


# ─────────────────────────────────────────────────────────────────────────────
# Suite 5: Telemetry pointer friendly rendering (PySide6)
# ─────────────────────────────────────────────────────────────────────────────

class TestTelemetryPanelRendering(unittest.TestCase):
    """Verify that telemetry pointer fields are rendered in the panel."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def _make_panel(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        return DiagnosticsPanel()

    def _info_with_telemetry(self, trace_id="t-abc", mission_trace_id="m-xyz"):
        """Build a diagnostics info dict that includes telemetry pointers."""
        from bin.core.error_ux import format_node_diagnostics
        run_result = {
            "status": "failed",
            "results": {},
            "diagnostics": {
                "failed_nodes": [1],
                "behavior_trace_ids": {},
                "mission_trace_id": mission_trace_id,
            },
        }
        return format_node_diagnostics(
            1,
            {"error": "execute_failed", "trace_id": trace_id},
            run_result=run_result,
            node_name="TestNode",
        )

    # ── _format_value ──────────────────────────────────────────────────

    def test_format_trace_id_returns_string(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        result = DiagnosticsPanel._format_value("trace_id", "abc-123-uuid")
        self.assertEqual(result, "abc-123-uuid")

    def test_format_mission_trace_returns_string(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        result = DiagnosticsPanel._format_value("mission_trace_id", "mission-abc")
        self.assertEqual(result, "mission-abc")

    def test_format_empty_trace_returns_none_sentinel(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        result = DiagnosticsPanel._format_value("trace_id", "")
        self.assertEqual(result, "(none)")

    # ── _KEY_LABELS ────────────────────────────────────────────────────

    def test_trace_id_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("trace_id", DiagnosticsPanel._KEY_LABELS)

    def test_mission_trace_id_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("mission_trace_id", DiagnosticsPanel._KEY_LABELS)

    def test_trace_id_label_is_human_readable(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._KEY_LABELS["trace_id"]
        # Must be human-readable — no underscores
        self.assertNotIn("_", label)

    # ── _FRIENDLY_KEYS ─────────────────────────────────────────────────

    def test_trace_id_in_friendly_keys(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("trace_id", DiagnosticsPanel._FRIENDLY_KEYS)

    def test_mission_trace_id_in_friendly_keys(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("mission_trace_id", DiagnosticsPanel._FRIENDLY_KEYS)

    # ── Friendly view rendering ────────────────────────────────────────

    def test_panel_shows_trace_id_in_friendly_view(self):
        """Panel friendly view renders trace_id when present in info."""
        p = self._make_panel()
        info = self._info_with_telemetry(trace_id="uuid-1234")
        p.show_diagnostics([info])
        # Collect all label text from the fields widget
        from PySide6.QtWidgets import QLabel
        labels = p._fields_widget.findChildren(QLabel)
        all_text = " ".join(lbl.text() for lbl in labels)
        self.assertIn("uuid-1234", all_text)

    def test_panel_shows_mission_trace_in_friendly_view(self):
        p = self._make_panel()
        info = self._info_with_telemetry(mission_trace_id="mission-uuid-9999")
        p.show_diagnostics([info])
        from PySide6.QtWidgets import QLabel
        labels = p._fields_widget.findChildren(QLabel)
        all_text = " ".join(lbl.text() for lbl in labels)
        self.assertIn("mission-uuid-9999", all_text)

    def test_panel_no_trace_row_when_trace_absent(self):
        """When trace_id is not in info, no Trace ID label appears."""
        from bin.core.error_ux import format_node_diagnostics
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from PySide6.QtWidgets import QLabel
        info = format_node_diagnostics(2, {"error": "execute_failed"}, node_name="N2")
        # trace_id must not be in info (no run_result, no trace_id in node_result)
        self.assertNotIn("trace_id", info)
        p = DiagnosticsPanel()
        p.show_diagnostics([info])
        labels = p._fields_widget.findChildren(QLabel)
        all_text = " ".join(lbl.text() for lbl in labels)
        # "Trace ID" label must not appear since the field is absent
        self.assertNotIn("Trace ID", all_text)


if __name__ == "__main__":
    unittest.main()
