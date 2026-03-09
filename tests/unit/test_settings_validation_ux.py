"""Tests for Cycle 2 STAGE-05: Validation and Error UX Integration.

Coverage:

    A. format_settings_validation_error — output shape, all expected keys,
       ordered dict, raw payload preservation
    B. Reason code registration — settings_validation_failed in
       REASON_OPERATOR_TEXT and REASON_CATEGORY (→ validation)
    C. _validate_settings_pre_run — passes on valid config, blocks on invalid,
       resilient to validator exceptions
    D. ExecutionSummaryBar surface — summary bar shown on failure, hidden on pass
    E. DiagnosticsPanel surface — diagnostics populated on failure
    F. Settings tab navigation — Settings tab activated on failure
    G. Capability inspector refresh — missing fields highlighted on failure
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from bin.core.error_ux import (
    DISPLAY_KEY_ORDER,
    ERROR_CATEGORY_VALIDATION,
    REASON_CATEGORY,
    REASON_OPERATOR_TEXT,
    format_settings_validation_error,
    get_error_category,
    get_operator_text,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

_VALIDATION_OK   = {"status": "ok", "missing": [], "invalid": []}
_VALIDATION_MISS = {"status": "error", "missing": ["brand", "robot_type"], "invalid": []}
_VALIDATION_INV  = {"status": "error", "missing": [], "invalid": ["timeout"]}
_VALIDATION_BOTH = {"status": "error", "missing": ["brand"], "invalid": ["timeout"]}


def _make_window():
    from bin.core.config_manager import ConfigManager
    from bin.ui import MainWindow
    return MainWindow(ConfigManager())


# ── A. format_settings_validation_error ──────────────────────────────────────

class TestFormatSettingsValidationError(unittest.TestCase):

    def test_returns_dict(self):
        result = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertIsInstance(result, dict)

    def test_node_id_is_settings_marker(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["node_id"], "settings")

    def test_node_name_contains_brand(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertIn("unitree", r["node_name"])

    def test_status_is_failed(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["status"], "failed")

    def test_error_category_is_validation(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["error_category"], ERROR_CATEGORY_VALIDATION)

    def test_reason_is_settings_validation_failed(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["reason"], "settings_validation_failed")

    def test_stage_is_preflight(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["stage"], "preflight")

    def test_operator_text_is_non_empty_string(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertIsInstance(r["operator_text"], str)
        self.assertTrue(len(r["operator_text"]) > 0)

    def test_message_lists_missing_fields(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertIn("brand", r["message"])
        self.assertIn("robot_type", r["message"])

    def test_message_lists_invalid_fields(self):
        r = format_settings_validation_error("unitree", _VALIDATION_INV)
        self.assertIn("timeout", r["message"])

    def test_message_lists_both_missing_and_invalid(self):
        r = format_settings_validation_error("unitree", _VALIDATION_BOTH)
        self.assertIn("brand", r["message"])
        self.assertIn("timeout", r["message"])

    def test_retryable_is_false(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertFalse(r["retryable"])

    def test_context_preserves_raw_validation_result(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        ctx = r["context"]
        self.assertIn("validation_result", ctx)
        self.assertEqual(ctx["validation_result"], _VALIDATION_MISS)

    def test_context_has_missing_fields_list(self):
        r = format_settings_validation_error("unitree", _VALIDATION_MISS)
        self.assertEqual(r["context"]["missing_fields"], ["brand", "robot_type"])

    def test_context_has_invalid_fields_list(self):
        r = format_settings_validation_error("unitree", _VALIDATION_INV)
        self.assertEqual(r["context"]["invalid_fields"], ["timeout"])

    def test_adapter_name_is_brand(self):
        r = format_settings_validation_error("bostiondynamics", _VALIDATION_MISS)
        self.assertEqual(r["adapter_name"], "bostiondynamics")

    def test_standard_display_keys_come_first(self):
        r = format_settings_validation_error("unitree", _VALIDATION_BOTH)
        keys = list(r.keys())
        for i, k in enumerate(DISPLAY_KEY_ORDER):
            if k in r:
                # Find position of k in keys; it must come before any non-DISPLAY_KEY_ORDER key
                pos = keys.index(k)
                non_display_after = [
                    kk for kk in keys[:pos] if kk not in DISPLAY_KEY_ORDER
                ]
                self.assertEqual(
                    non_display_after, [],
                    f"Key {k!r} appeared after a non-DISPLAY_KEY_ORDER key",
                )

    def test_empty_validation_result_no_crash(self):
        r = format_settings_validation_error("unitree", {})
        self.assertIsInstance(r, dict)

    def test_brand_variants_no_crash(self):
        for brand in ("unitree", "bostiondynamics", "xiaomi"):
            r = format_settings_validation_error(brand, _VALIDATION_MISS)
            self.assertIn(brand, r["node_name"])


# ── B. Reason code registration ───────────────────────────────────────────────

class TestReasonCodeRegistration(unittest.TestCase):

    def test_settings_validation_failed_in_reason_operator_text(self):
        self.assertIn("settings_validation_failed", REASON_OPERATOR_TEXT)

    def test_settings_validation_failed_operator_text_is_string(self):
        text = REASON_OPERATOR_TEXT["settings_validation_failed"]
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_settings_validation_failed_in_reason_category(self):
        self.assertIn("settings_validation_failed", REASON_CATEGORY)

    def test_settings_validation_failed_maps_to_validation_category(self):
        self.assertEqual(
            REASON_CATEGORY["settings_validation_failed"],
            ERROR_CATEGORY_VALIDATION,
        )

    def test_get_error_category_returns_validation(self):
        self.assertEqual(
            get_error_category("settings_validation_failed"),
            ERROR_CATEGORY_VALIDATION,
        )

    def test_get_operator_text_returns_registered_text(self):
        text = get_operator_text("settings_validation_failed")
        # Must be the registered text, not the generic fallback
        self.assertNotIn("An error occurred", text)

    def test_no_tr_calls_in_operator_text(self):
        # Localization contract: no tr() in the raw text string
        text = REASON_OPERATOR_TEXT["settings_validation_failed"]
        self.assertNotIn("tr(", text)


# ── C. _validate_settings_pre_run ─────────────────────────────────────────────

class TestValidateSettingsPreRun(unittest.TestCase):

    def test_returns_true_when_validation_ok(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value={"status": "ok"},
        ):
            result = win._validate_settings_pre_run()
        self.assertTrue(result)

    def test_returns_false_when_validation_fails(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            result = win._validate_settings_pre_run()
        self.assertFalse(result)

    def test_returns_true_when_validator_raises(self):
        """Resilience: unexpected validator failure must not block run."""
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            side_effect=RuntimeError("validator crash"),
        ):
            result = win._validate_settings_pre_run()
        self.assertTrue(result)

    def test_does_not_raise_on_missing_brand(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            try:
                win._validate_settings_pre_run()
            except Exception as exc:
                self.fail(f"Should not raise: {exc}")

    def test_does_not_raise_on_invalid_fields(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_INV,
        ):
            try:
                win._validate_settings_pre_run()
            except Exception as exc:
                self.fail(f"Should not raise: {exc}")


# ── D. ExecutionSummaryBar surface ────────────────────────────────────────────

class TestExecutionSummaryBarSurface(unittest.TestCase):

    def test_summary_bar_shown_on_validation_failure(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        # Summary bar should be visible after failure
        bar = win.main_zone._exec_summary_bar
        self.assertTrue(bar.isVisible() or not bar.isHidden())

    def test_summary_bar_icon_is_failure_on_validation_failure(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        icon_text = win.main_zone._exec_status_icon.text()
        self.assertEqual(icon_text, "✖")

    def test_summary_text_contains_reason_on_failure(self):
        win = _make_window()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        summary = win.main_zone._exec_summary_label.text()
        self.assertIn("settings_validation_failed", summary)

    def test_summary_bar_not_shown_when_validation_passes(self):
        win = _make_window()
        # Clear summary first
        win.main_zone.clear_execution_summary()
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value={"status": "ok"},
        ):
            win._validate_settings_pre_run()
        # Bar should still be hidden (was cleared and not re-shown)
        bar = win.main_zone._exec_summary_bar
        self.assertTrue(bar.isHidden())


# ── E. DiagnosticsPanel surface ───────────────────────────────────────────────

class TestDiagnosticsPanelSurface(unittest.TestCase):

    def test_diagnostics_panel_populated_on_validation_failure(self):
        win = _make_window()
        show_calls = []
        original = win.main_zone.show_diagnostics_panel
        win.main_zone.show_diagnostics_panel = (
            lambda infos, *a, **kw: show_calls.append(infos) or original(infos, *a, **kw)
        )
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        self.assertGreater(len(show_calls), 0, "show_diagnostics_panel should be called")
        infos = show_calls[0]
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0]["reason"], "settings_validation_failed")

    def test_diagnostics_diag_info_has_correct_error_category(self):
        win = _make_window()
        captured = []
        original = win.main_zone.show_diagnostics_panel
        win.main_zone.show_diagnostics_panel = (
            lambda infos, *a, **kw: captured.append(infos) or original(infos, *a, **kw)
        )
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_BOTH,
        ):
            win._validate_settings_pre_run()
        info = captured[0][0]
        self.assertEqual(info["error_category"], ERROR_CATEGORY_VALIDATION)

    def test_diagnostics_not_populated_when_validation_passes(self):
        win = _make_window()
        show_calls = []
        win.main_zone.show_diagnostics_panel = (
            lambda infos, *a, **kw: show_calls.append(infos)
        )
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value={"status": "ok"},
        ):
            win._validate_settings_pre_run()
        self.assertEqual(len(show_calls), 0, "DiagnosticsPanel must not be shown on pass")


# ── F. Settings tab navigation ────────────────────────────────────────────────

class TestSettingsTabNavigation(unittest.TestCase):

    def _settings_tab_index(self, win) -> int:
        tabs = win.main_zone.workspace_tabs
        for i in range(tabs.count()):
            if tabs.tabText(i) == "Settings":
                return i
        return -1

    def test_settings_tab_activated_on_validation_failure(self):
        win = _make_window()
        tabs = win.main_zone.workspace_tabs
        # Start on a non-settings tab
        tabs.setCurrentIndex(0)
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        settings_idx = self._settings_tab_index(win)
        self.assertEqual(tabs.currentIndex(), settings_idx)

    def test_settings_tab_not_forcibly_activated_on_pass(self):
        win = _make_window()
        tabs = win.main_zone.workspace_tabs
        initial_idx = 0
        tabs.setCurrentIndex(initial_idx)
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value={"status": "ok"},
        ):
            win._validate_settings_pre_run()
        self.assertEqual(tabs.currentIndex(), initial_idx)


# ── G. Capability inspector refresh on failure ────────────────────────────────

class TestCapabilityInspectorRefreshOnFailure(unittest.TestCase):

    def test_capability_inspector_refreshed_on_validation_failure(self):
        win = _make_window()
        refresh_calls = []
        original = win._refresh_capability_inspector
        win._refresh_capability_inspector = (
            lambda: refresh_calls.append(True) or original()
        )
        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._validate_settings_pre_run()
        self.assertGreater(
            len(refresh_calls), 0,
            "_refresh_capability_inspector should be called on validation failure",
        )

    def test_on_run_blocked_by_invalid_settings(self):
        """_on_run() must return early when settings are invalid."""
        win = _make_window()
        runtime_calls = []
        original_execute = win.runtime_engine.execute
        win.runtime_engine.execute = lambda *a, **kw: runtime_calls.append(True) or original_execute(*a, **kw)

        with patch(
            "system.service.settings_validator.validate_settings",
            return_value=_VALIDATION_MISS,
        ):
            win._on_run()

        self.assertEqual(
            len(runtime_calls), 0,
            "RuntimeEngine.execute must not be called when settings validation fails",
        )

    def test_on_run_proceeds_when_settings_valid(self):
        """_on_run() proceeds past the validation gate when settings are ok."""
        win = _make_window()
        # Patch validate_settings to always pass
        # Also need an execution graph or it'll be blocked by "no nodes" check
        with patch("system.service.settings_validator.validate_settings", return_value={"status": "ok"}):
            # No nodes in graph — run will be blocked by the node-count check,
            # not the settings gate.  We just verify that we pass the settings gate.
            passed_gate = []
            original_activate = win.main_zone.activate_runtime_fullscreen
            win.main_zone.activate_runtime_fullscreen = (
                lambda: passed_gate.append(True) or original_activate()
            )
            win._on_run()
        # If we reached activate_runtime_fullscreen, we passed the settings gate
        self.assertGreater(
            len(passed_gate), 0,
            "Run should proceed past settings gate when validation passes",
        )


if __name__ == "__main__":
    unittest.main()
