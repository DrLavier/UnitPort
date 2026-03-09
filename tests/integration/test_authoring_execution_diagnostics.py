"""Integration test: authoring → execution → diagnostics display pipeline.

STAGE-07 Cycle 1 — Integration tests: authoring → execution → diagnostics.

Verifies the complete pipeline from canvas authoring through RuntimeEngine
execution to diagnostic data extraction and display formatting:

    GraphScene (authoring)
        → get_execution_graph()
        → RuntimeEngine.execute()
        → error_ux.extract_failed_nodes_info()
        → DiagnosticsPanel.show_diagnostics() (Suites C/D; PySide6)

Done criteria:
  "End-to-end authoring → execution → diagnostics pipeline produces
   correctly structured display data for all failed nodes."

All tests use QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PYSIDE6_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF
    _PYSIDE6_AVAILABLE = True
except ImportError:
    pass

_app = None


def _get_app():
    global _app
    if not _PYSIDE6_AVAILABLE:
        return None
    if QApplication.instance() is None:
        _app = QApplication(sys.argv[:1])
    else:
        _app = QApplication.instance()
    return _app


def _make_scene():
    from bin.components.graph_scene import GraphScene
    return GraphScene()


def _run_engine(exec_graph):
    from system.runtime import RuntimeEngine
    engine = RuntimeEngine()
    return engine.execute(exec_graph, {})


# ── Suite A: Canvas → execute pipeline structure ───────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestAuthoringExecutionPipeline(unittest.TestCase):
    """Verify canvas authoring → execution chain produces the expected run_result."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _pipeline(self, *node_names):
        """Create scene with named nodes, execute, return run_result."""
        scene = _make_scene()
        for i, name in enumerate(node_names):
            scene.create_node(name, QPointF(i * 100, 100))
        exec_graph = scene.get_execution_graph()
        return _run_engine(exec_graph)

    def test_default_canvas_produces_dict(self):
        """Default (empty) canvas → execute returns a dict."""
        result = self._pipeline()
        self.assertIsInstance(result, dict)

    def test_default_canvas_has_status(self):
        """Pipeline result has 'status' key."""
        result = self._pipeline()
        self.assertIn("status", result)

    def test_default_canvas_has_diagnostics(self):
        """Pipeline result has 'diagnostics' key."""
        result = self._pipeline()
        self.assertIn("diagnostics", result)

    def test_default_canvas_has_results(self):
        """Pipeline result has 'results' key."""
        result = self._pipeline()
        self.assertIn("results", result)

    def test_status_is_valid_string(self):
        """Pipeline result status is one of success / failed / blocked."""
        result = self._pipeline()
        self.assertIn(result.get("status"), {"success", "failed", "blocked"})

    def test_not_blocked_for_default_scene(self):
        """Default scene must not be blocked by compile/safety checks."""
        result = self._pipeline()
        self.assertNotEqual(result.get("status"), "blocked")

    def test_timer_node_canvas_executes_without_exception(self):
        """Canvas with a Timer node executes without raising."""
        try:
            result = self._pipeline("Timer")
        except Exception as exc:
            self.fail(f"Pipeline raised unexpectedly: {exc}")
        self.assertIsInstance(result, dict)

    def test_wait_node_canvas_executes_without_exception(self):
        """Canvas with a Wait node executes without raising."""
        try:
            result = self._pipeline("Wait")
        except Exception as exc:
            self.fail(f"Pipeline raised unexpectedly: {exc}")
        self.assertIsInstance(result, dict)

    def test_failed_nodes_list_is_list(self):
        """diagnostics.failed_nodes is always a list."""
        result = self._pipeline()
        failed = result.get("diagnostics", {}).get("failed_nodes", [])
        self.assertIsInstance(failed, list)

    def test_results_dict_maps_node_ids(self):
        """'results' is a dict (node_id → node result dict)."""
        result = self._pipeline()
        self.assertIsInstance(result.get("results", {}), dict)


# ── Suite B: error_ux diagnostics extraction (pure Python) ────────────────────

class TestErrorUxExtraction(unittest.TestCase):
    """Tests for error_ux.extract_failed_nodes_info with synthetic run_results."""

    def _failed_result(self, node_id, reason="execute_failed", adapter="test"):
        """Build a minimal run_result with one failed node."""
        return {
            "status": "failed",
            "results": {
                str(node_id): {
                    "status": "failed",
                    "reason": reason,
                    "stage": "execute",
                    "message": f"Node {node_id} failed",
                    "adapter_name": adapter,
                }
            },
            "diagnostics": {
                "failed_nodes": [node_id],
                "behavior_diagnostics": {},
            },
        }

    def test_extract_returns_list(self):
        """extract_failed_nodes_info returns a list."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(1))
        self.assertIsInstance(infos, list)

    def test_extract_one_failed_node_returns_one_entry(self):
        """One failed node → one entry in the returned list."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(1))
        self.assertEqual(len(infos), 1)

    def test_extracted_info_has_required_keys(self):
        """Extracted info dict contains required diagnostic keys."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(42))
        info = infos[0]
        for key in ("node_id", "status", "stage", "error_category", "reason",
                    "operator_text", "retryable"):
            self.assertIn(key, info, f"Missing required key: '{key}'")

    def test_extracted_node_id_is_string(self):
        """node_id in extracted info is always a string."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(99))
        self.assertEqual(infos[0]["node_id"], "99")

    def test_extracted_error_category_is_valid(self):
        """error_category is one of the four canonical category constants."""
        from bin.core.error_ux import (
            extract_failed_nodes_info,
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
            ERROR_CATEGORY_COMPAT,
        )
        valid = {ERROR_CATEGORY_VALIDATION, ERROR_CATEGORY_PREFLIGHT,
                 ERROR_CATEGORY_RUNTIME, ERROR_CATEGORY_COMPAT}
        infos = extract_failed_nodes_info(self._failed_result(1))
        self.assertIn(infos[0]["error_category"], valid)

    def test_extracted_operator_text_is_nonempty_string(self):
        """operator_text in extracted info is a non-empty string."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(1))
        op_text = infos[0].get("operator_text", "")
        self.assertIsInstance(op_text, str)
        self.assertTrue(op_text, "operator_text must be non-empty")

    def test_extracted_retryable_is_bool(self):
        """retryable in extracted info is a boolean."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(self._failed_result(1))
        self.assertIsInstance(infos[0]["retryable"], bool)

    def test_extract_multiple_nodes_preserves_order(self):
        """Multiple failed nodes appear in the order given in failed_nodes list."""
        from bin.core.error_ux import extract_failed_nodes_info
        result = {
            "status": "failed",
            "results": {
                "1": {"status": "failed", "reason": "execute_failed"},
                "2": {"status": "failed", "reason": "execute_failed"},
                "3": {"status": "failed", "reason": "execute_failed"},
            },
            "diagnostics": {"failed_nodes": [1, 2, 3]},
        }
        infos = extract_failed_nodes_info(result)
        self.assertEqual(len(infos), 3)
        self.assertEqual(infos[0]["node_id"], "1")
        self.assertEqual(infos[1]["node_id"], "2")
        self.assertEqual(infos[2]["node_id"], "3")

    def test_extract_with_node_names_mapping(self):
        """node_name in extracted info uses the provided node_names mapping."""
        from bin.core.error_ux import extract_failed_nodes_info
        infos = extract_failed_nodes_info(
            self._failed_result(5),
            node_names={5: "TimerNode"},
        )
        self.assertEqual(infos[0]["node_name"], "TimerNode")

    def test_extract_empty_failed_nodes_returns_empty_list(self):
        """No failed nodes → empty list returned."""
        from bin.core.error_ux import extract_failed_nodes_info
        result = {
            "status": "success",
            "results": {},
            "diagnostics": {"failed_nodes": []},
        }
        self.assertEqual(extract_failed_nodes_info(result), [])

    def test_unknown_node_falls_back_to_unknown_status(self):
        """Node ID in failed_nodes but absent from results gets a fallback entry."""
        from bin.core.error_ux import extract_failed_nodes_info
        result = {
            "status": "failed",
            "results": {},
            "diagnostics": {"failed_nodes": [999]},
        }
        infos = extract_failed_nodes_info(result)
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0]["node_id"], "999")


# ── Suite C: classify_run_result and key ordering ─────────────────────────────

class TestClassifyRunResult(unittest.TestCase):
    """Tests for classify_run_result error category decision tree."""

    def test_classify_execution_failure_is_runtime(self):
        """execute_failed reason → runtime category."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_RUNTIME
        result = {
            "status": "failed",
            "reason": "execute_failed",
            "results": {},
            "diagnostics": {},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_RUNTIME)

    def test_classify_adapter_not_found_is_validation(self):
        """adapter_not_found reason → validation category."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_VALIDATION
        result = {
            "status": "failed",
            "reason": "adapter_not_found",
            "results": {},
            "diagnostics": {},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_VALIDATION)

    def test_classify_blocked_compile_is_validation(self):
        """blocked status with blocked_compile path → validation."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_VALIDATION
        result = {
            "status": "blocked",
            "results": {},
            "diagnostics": {"path": "blocked_compile"},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_VALIDATION)

    def test_classify_blocked_safety_is_preflight(self):
        """blocked status with blocked_safety path → preflight_safety."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_PREFLIGHT
        result = {
            "status": "blocked",
            "results": {},
            "diagnostics": {"path": "blocked_safety"},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_PREFLIGHT)

    def test_classify_compat_path_is_compat_warning(self):
        """compat_path True → compat_warning category."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_COMPAT
        result = {
            "status": "success",
            "results": {},
            "diagnostics": {"compat_path": True, "failed_nodes": []},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_COMPAT)

    def test_classify_no_reason_with_failed_nodes_is_runtime(self):
        """No reason code + failed_nodes present → runtime fallback."""
        from bin.core.error_ux import classify_run_result, ERROR_CATEGORY_RUNTIME
        result = {
            "status": "failed",
            "results": {},
            "diagnostics": {"failed_nodes": [1]},
        }
        self.assertEqual(classify_run_result(result), ERROR_CATEGORY_RUNTIME)


class TestDiagnosticsDisplayKeyOrder(unittest.TestCase):
    """Tests for format_node_diagnostics key ordering contract."""

    def test_keys_follow_display_key_order(self):
        """format_node_diagnostics output starts with DISPLAY_KEY_ORDER keys."""
        from bin.core.error_ux import format_node_diagnostics, DISPLAY_KEY_ORDER
        node_result = {
            "status": "failed",
            "reason": "execute_failed",
            "stage": "execute",
            "message": "test failure",
            "adapter_name": "test",
        }
        info = format_node_diagnostics(1, node_result)
        keys = list(info.keys())
        ordered_present = [k for k in DISPLAY_KEY_ORDER if k in info]
        for i, expected_key in enumerate(ordered_present):
            self.assertEqual(
                keys[i], expected_key,
                f"Position {i}: expected {expected_key!r}, got {keys[i]!r}",
            )

    def test_error_category_precedes_reason(self):
        """error_category appears before reason in the output dict."""
        from bin.core.error_ux import format_node_diagnostics
        node_result = {"status": "failed", "reason": "execute_failed"}
        info = format_node_diagnostics(1, node_result)
        keys = list(info.keys())
        ec_idx = keys.index("error_category") if "error_category" in keys else -1
        r_idx = keys.index("reason") if "reason" in keys else -1
        if ec_idx >= 0 and r_idx >= 0:
            self.assertLess(ec_idx, r_idx, "error_category must precede reason")

    def test_telemetry_keys_absent_when_no_run_result(self):
        """trace_id and mission_trace_id absent when run_result not provided."""
        from bin.core.error_ux import format_node_diagnostics
        node_result = {"status": "failed", "reason": "execute_failed"}
        info = format_node_diagnostics(1, node_result)
        self.assertNotIn("trace_id", info)
        self.assertNotIn("mission_trace_id", info)

    def test_trace_id_present_when_in_behavior_trace_ids(self):
        """trace_id appears when provided via behavior_trace_ids in run_result."""
        from bin.core.error_ux import format_node_diagnostics
        node_result = {"status": "failed", "reason": "execute_failed"}
        run_result = {
            "diagnostics": {"behavior_trace_ids": {"1": "abc-trace-001"}}
        }
        info = format_node_diagnostics(1, node_result, run_result=run_result)
        self.assertIn("trace_id", info)
        self.assertEqual(info["trace_id"], "abc-trace-001")

    def test_node_name_defaults_to_node_id_string(self):
        """format_node_diagnostics uses str(node_id) as fallback for node_name."""
        from bin.core.error_ux import format_node_diagnostics
        node_result = {"status": "failed", "reason": "execute_failed"}
        info = format_node_diagnostics(42, node_result)
        self.assertEqual(info["node_name"], "42")

    def test_node_name_uses_provided_value(self):
        """format_node_diagnostics uses the provided node_name argument."""
        from bin.core.error_ux import format_node_diagnostics
        node_result = {"status": "failed", "reason": "execute_failed"}
        info = format_node_diagnostics(42, node_result, node_name="MyTimer")
        self.assertEqual(info["node_name"], "MyTimer")


# ── Suite D: DiagnosticsPanel integration (PySide6 required) ──────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestDiagnosticsPanelIntegration(unittest.TestCase):
    """Full pipeline from synthetic run_result to DiagnosticsPanel display."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _failed_result(self, node_id=1, reason="execute_failed"):
        return {
            "status": "failed",
            "results": {
                str(node_id): {
                    "status": "failed",
                    "reason": reason,
                    "stage": "execute",
                    "message": "Test failure",
                    "adapter_name": "test_adapter",
                }
            },
            "diagnostics": {"failed_nodes": [node_id]},
        }

    def test_show_diagnostics_does_not_raise(self):
        """DiagnosticsPanel.show_diagnostics() does not raise for valid info."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(self._failed_result(1))
        try:
            panel.show_diagnostics(infos)
        except Exception as exc:
            self.fail(f"show_diagnostics raised: {exc}")

    def test_panel_visible_after_show_diagnostics(self):
        """Panel becomes visible after show_diagnostics() with a non-empty list."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(self._failed_result(1))
        panel.show_diagnostics(infos)
        self.assertTrue(panel.isVisible())

    def test_panel_hidden_after_clear(self):
        """Panel becomes hidden after clear()."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(self._failed_result(1))
        panel.show_diagnostics(infos)
        panel.clear()
        self.assertFalse(panel.isVisible())

    def test_navigate_signal_emits_node_id(self):
        """navigate_requested emits the integer node_id when Go to Node is clicked."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(self._failed_result(99))
        panel.show_diagnostics(infos)
        received = []
        panel.navigate_requested.connect(received.append)
        panel._on_goto_clicked()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], 99)

    def test_combo_hidden_for_single_node(self):
        """Node selector combo is hidden when only one failed node."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(self._failed_result(1))
        panel.show_diagnostics(infos)
        self.assertFalse(panel._node_combo.isVisible())

    def test_combo_visible_for_multiple_nodes(self):
        """Node selector combo is visible when multiple failed nodes are shown."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        panel = DiagnosticsPanel()
        result = {
            "status": "failed",
            "results": {
                "1": {"status": "failed", "reason": "execute_failed"},
                "2": {"status": "failed", "reason": "execute_failed"},
            },
            "diagnostics": {"failed_nodes": [1, 2]},
        }
        infos = extract_failed_nodes_info(result)
        panel.show_diagnostics(infos)
        self.assertTrue(panel._node_combo.isVisible())

    def test_end_to_end_canvas_execute_to_diagnostics(self):
        """Complete pipeline: canvas authoring → execute → extract → show panel."""
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info
        # Authoring
        scene = _make_scene()
        exec_graph = scene.get_execution_graph()
        # Execution
        run_result = _run_engine(exec_graph)
        self.assertIsInstance(run_result, dict)
        # Diagnostics extraction (may be empty for healthy scene)
        failed = run_result.get("diagnostics", {}).get("failed_nodes", [])
        infos = extract_failed_nodes_info(run_result)
        self.assertEqual(len(infos), len(failed))
        # Panel display — must not raise regardless of node count
        panel = DiagnosticsPanel()
        if infos:
            try:
                panel.show_diagnostics(infos)
            except Exception as exc:
                self.fail(f"show_diagnostics raised for live pipeline: {exc}")


if __name__ == "__main__":
    unittest.main()
