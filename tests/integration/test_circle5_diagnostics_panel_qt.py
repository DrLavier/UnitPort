#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 5 close-out: Qt DiagnosticsPanel click-through integration tests."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PYSIDE6_AVAILABLE = False
try:
    from PySide6.QtCore import Qt  # noqa: F401
    from PySide6.QtTest import QTest  # noqa: F401
    from PySide6.QtWidgets import QApplication, QLabel  # noqa: F401
    _PYSIDE6_AVAILABLE = True
except ImportError:
    pass


def _failed_run_result_simple() -> dict:
    return {
        "status": "failed",
        "reason": "execute_failed",
        "results": {
            "11": {
                "status": "failed",
                "reason": "execute_failed",
                "stage": "execute",
                "message": "Simple path failed",
                "adapter_name": "unitree_sdk2",
                "trace_id": "simple-node-trace-11",
            }
        },
        "diagnostics": {
            "failed_nodes": [11],
            "mission_trace_id": "simple-mission-trace-001",
            "behavior_trace_ids": {"11": "simple-node-trace-11"},
        },
    }


def _failed_run_result_advanced() -> dict:
    return {
        "status": "failed",
        "reason": "execute_failed",
        "results": {
            "b1": {
                "status": "failed",
                "reason": "execute_failed",
                "stage": "execute",
                "message": "Advanced package-expanded path failed",
                "adapter_name": "unitree_sdk2",
                "trace_id": "adv-node-trace-b1",
            }
        },
        "diagnostics": {
            "failed_nodes": ["b1"],
            "mission_trace_id": "adv-mission-trace-009",
            "behavior_trace_ids": {"b1": "adv-node-trace-b1"},
            "package_metadata_trace": {
                "package_id": "pkg.walk.advanced",
                "package_version": "2.4.1",
                "schema_version": "3",
            },
        },
    }


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestCircle5DiagnosticsPanelQtClickThrough(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _friendly_text(self, panel) -> str:
        labels = panel._fields_widget.findChildren(QLabel)
        return " ".join(lbl.text() for lbl in labels)

    def test_simple_path_failure_render_and_clickthrough(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info

        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(_failed_run_result_simple(), {"11": "Simple Fail Node"})
        self.assertEqual(len(infos), 1)
        info = infos[0]

        panel.show_diagnostics(infos)
        self.assertTrue(panel.isVisible())

        # Required contract keys must exist in structured diagnostics.
        self.assertEqual(info.get("reason"), "execute_failed")
        self.assertTrue(info.get("operator_text"))
        self.assertEqual(info.get("error_category"), "runtime")
        self.assertEqual(info.get("trace_id"), "simple-node-trace-11")
        self.assertEqual(info.get("mission_trace_id"), "simple-mission-trace-001")

        friendly = self._friendly_text(panel)
        self.assertIn("execute_failed", friendly)
        self.assertIn("Action execution failed", friendly)
        self.assertIn("Runtime", friendly)
        self.assertIn("simple-node-trace-11", friendly)
        self.assertIn("simple-mission-trace-001", friendly)

        # Click-through: Raw toggle and Go to Node button.
        panel._raw_toggle.click()
        raw = panel._raw_text.toPlainText()
        self.assertIn('"reason": "execute_failed"', raw)
        self.assertIn('"trace_id": "simple-node-trace-11"', raw)
        self.assertIn('"mission_trace_id": "simple-mission-trace-001"', raw)

        received = []
        panel.navigate_requested.connect(received.append)
        panel._goto_btn.click()
        self.assertEqual(received, [11])

    def test_advanced_path_failure_render_includes_package_metadata_trace(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info

        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(_failed_run_result_advanced(), {"b1": "Advanced Behavior Node"})
        self.assertEqual(len(infos), 1)
        info = infos[0]

        panel.show_diagnostics(infos)
        self.assertTrue(panel.isVisible())

        self.assertEqual(info.get("reason"), "execute_failed")
        self.assertTrue(info.get("operator_text"))
        self.assertEqual(info.get("error_category"), "runtime")
        self.assertEqual(info.get("trace_id"), "adv-node-trace-b1")
        self.assertEqual(info.get("mission_trace_id"), "adv-mission-trace-009")
        self.assertIn("package_metadata_trace", info)

        pkg = info["package_metadata_trace"]
        self.assertEqual(pkg.get("package_id"), "pkg.walk.advanced")
        self.assertEqual(pkg.get("package_version"), "2.4.1")
        self.assertEqual(pkg.get("schema_version"), "3")

        friendly = self._friendly_text(panel)
        self.assertIn("adv-node-trace-b1", friendly)
        self.assertIn("adv-mission-trace-009", friendly)
        self.assertIn("pkg.walk.advanced", friendly)
        self.assertIn("2.4.1", friendly)
        self.assertIn("schema_version", friendly)

        panel._raw_toggle.click()
        raw = panel._raw_text.toPlainText()
        self.assertIn('"package_metadata_trace"', raw)
        self.assertIn('"package_id": "pkg.walk.advanced"', raw)
        self.assertIn('"package_version": "2.4.1"', raw)
        self.assertIn('"schema_version": "3"', raw)

    def test_qt_clickthrough_toggle_and_close(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info

        panel = DiagnosticsPanel()
        infos = extract_failed_nodes_info(_failed_run_result_simple(), {"11": "Simple Fail Node"})
        panel.show_diagnostics(infos)
        self.assertTrue(panel.isVisible())

        # Real Qt click-through path using QTest
        QTest.mouseClick(panel._raw_toggle, Qt.LeftButton)
        self.assertTrue(panel._raw_toggle.isChecked())
        self.assertTrue(panel._raw_text.isVisible())

        QTest.mouseClick(panel._raw_toggle, Qt.LeftButton)
        self.assertFalse(panel._raw_toggle.isChecked())
        self.assertTrue(panel._scroll_area.isVisible())

        QTest.mouseClick(panel._close_btn, Qt.LeftButton)
        self.assertFalse(panel.isVisible())


if __name__ == "__main__":
    unittest.main()
