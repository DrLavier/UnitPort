#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circle 5 close-out: localization consistency tests for diagnostics UX."""

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
    from PySide6.QtWidgets import QApplication, QLabel  # noqa: F401
    _PYSIDE6_AVAILABLE = True
except ImportError:
    pass


def _run_result_with_advanced_trace() -> dict:
    return {
        "status": "failed",
        "reason": "execute_failed",
        "results": {
            "21": {
                "status": "failed",
                "reason": "execute_failed",
                "stage": "execute",
                "message": "Localization consistency sample",
                "adapter_name": "unitree_sdk2",
                "trace_id": "loc-node-trace-21",
            }
        },
        "diagnostics": {
            "failed_nodes": [21],
            "mission_trace_id": "loc-mission-trace-777",
            "behavior_trace_ids": {"21": "loc-node-trace-21"},
            "package_metadata_trace": {
                "package_id": "pkg.localization.sample",
                "package_version": "9.9.9",
                "schema_version": "3",
            },
        },
    }


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestCircle5LocalizationConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        from bin.core.localisation import get_localisation

        self._loc = get_localisation()
        self._previous_lang = self._loc.current_language
        self._loc.load_language("en")

    def tearDown(self):
        self._loc.load_language(self._previous_lang or "en")

    def _friendly_text(self, panel) -> str:
        labels = panel._fields_widget.findChildren(QLabel)
        return " ".join(lbl.text() for lbl in labels)

    def _friendly_rows_snapshot(self, panel) -> list:
        rows = []
        for i in range(panel._fields_layout.count()):
            item = panel._fields_layout.itemAt(i)
            row_layout = item.layout()
            if row_layout is None or row_layout.count() < 2:
                continue
            key_widget = row_layout.itemAt(0).widget()
            val_widget = row_layout.itemAt(1).widget()
            if not key_widget or not val_widget:
                continue
            key_text = key_widget.text().replace("<b>", "").replace("</b>", "").strip()
            val_text = " ".join(val_widget.text().split())
            rows.append(f"{key_text} {val_text}")
        return rows

    def _build_panel_and_info(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info

        infos = extract_failed_nodes_info(_run_result_with_advanced_trace(), {"21": "Localization Node"})
        panel = DiagnosticsPanel()
        panel.show_diagnostics(infos)
        return panel, infos[0]

    def test_friendly_view_text_localizes_between_en_and_zh(self):
        # English snapshot
        self.assertTrue(self._loc.load_language("en"))
        panel_en, info_en = self._build_panel_and_info()
        text_en = self._friendly_text(panel_en)

        # Chinese snapshot
        self.assertTrue(self._loc.load_language("zh"))
        panel_zh, info_zh = self._build_panel_and_info()
        text_zh = self._friendly_text(panel_zh)

        self.assertIn("Category", text_en)
        self.assertIn("Runtime Failure", text_en)
        self.assertIn("类别", text_zh)
        self.assertIn("运行时失败", text_zh)

        # Machine-code stability across locale.
        self.assertEqual(info_en["reason"], info_zh["reason"])
        self.assertEqual(info_en["error_category"], info_zh["error_category"])
        self.assertEqual(info_en["trace_id"], info_zh["trace_id"])
        self.assertEqual(info_en["mission_trace_id"], info_zh["mission_trace_id"])

    def test_friendly_view_locale_snapshots_en_and_zh(self):
        # EN snapshot
        self.assertTrue(self._loc.load_language("en"))
        panel_en, _ = self._build_panel_and_info()
        snapshot_en = self._friendly_rows_snapshot(panel_en)
        self.assertIn("Stage: Execution", snapshot_en)
        self.assertIn("Category: Runtime Failure", snapshot_en)
        self.assertIn("Reason Code: execute_failed", snapshot_en)
        self.assertIn("Trace ID: loc-node-trace-21", snapshot_en)

        # ZH snapshot
        self.assertTrue(self._loc.load_language("zh"))
        panel_zh, _ = self._build_panel_and_info()
        snapshot_zh = self._friendly_rows_snapshot(panel_zh)
        self.assertIn("阶段: 执行", snapshot_zh)
        self.assertIn("类别: 运行时失败", snapshot_zh)
        self.assertIn("原因码: execute_failed", snapshot_zh)
        self.assertIn("追踪 ID: loc-node-trace-21", snapshot_zh)

    def test_raw_json_not_translated_and_not_polluted(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import extract_failed_nodes_info

        self.assertTrue(self._loc.load_language("zh"))
        info = extract_failed_nodes_info(_run_result_with_advanced_trace(), {"21": "Localization Node"})[0]

        panel = DiagnosticsPanel()
        panel.show_diagnostics([info])
        panel._raw_toggle.click()
        raw_zh = panel._raw_text.toPlainText()

        self.assertIn('"reason": "execute_failed"', raw_zh)
        self.assertIn('"error_category": "runtime"', raw_zh)
        self.assertIn('"trace_id": "loc-node-trace-21"', raw_zh)
        self.assertIn('"mission_trace_id": "loc-mission-trace-777"', raw_zh)
        self.assertIn('"package_metadata_trace"', raw_zh)
        self.assertNotIn("运行时失败", raw_zh)
        self.assertNotIn("类别", raw_zh)

        self.assertTrue(self._loc.load_language("en"))
        panel_en = DiagnosticsPanel()
        panel_en.show_diagnostics([info])
        panel_en._raw_toggle.click()
        raw_en = panel_en._raw_text.toPlainText()
        self.assertEqual(raw_en, raw_zh)


if __name__ == "__main__":
    unittest.main()
