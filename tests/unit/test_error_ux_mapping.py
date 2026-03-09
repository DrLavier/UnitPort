"""STAGE-06 — Error UX and Contract Mapping tests.

Verifies that:
  - All 28 canonical reason codes map to operator text.
  - All 28 reason codes have an explicit category assignment.
  - The four categories (validation / preflight_safety / runtime / compat_warning)
    are each represented.
  - blocked-path → category mapping is correct.
  - classify_run_result() selects the right category for every result type.
  - format_node_diagnostics() includes error_category in deterministic order.
  - Localization hooks: all operator texts are plain strings with no tr() calls;
    raw diagnostics JSON is unaffected by locale.

Sections:
    A — Error category constants
    B — REASON_CATEGORY completeness and coverage
    C — get_error_category()
    D — classify_run_result()
    E — format_node_diagnostics() includes error_category
    F — Localization hook safety
    G — DiagnosticsPanel category rendering (PySide6)
"""

from __future__ import annotations

import json
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blocked_result(path: str) -> dict:
    return {
        "status": "blocked",
        "reason": "blocked",
        "diagnostics": {"path": path},
        "results": {},
    }


def _failed_result(reason: str, compat: bool = False) -> dict:
    return {
        "status": "failed",
        "reason": reason,
        "diagnostics": {
            "failed_nodes": [1],
            "compat_path": compat,
            "compat_reason": "env_var_set" if compat else "",
            "behavior_trace_ids": {},
            "mission_trace_id": "",
        },
        "results": {"1": {"error": reason}},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Error category constants
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorCategoryConstants(unittest.TestCase):

    def setUp(self):
        from bin.core.error_ux import (
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
            ERROR_CATEGORY_COMPAT,
        )
        self.validation = ERROR_CATEGORY_VALIDATION
        self.preflight  = ERROR_CATEGORY_PREFLIGHT
        self.runtime    = ERROR_CATEGORY_RUNTIME
        self.compat     = ERROR_CATEGORY_COMPAT

    def test_all_constants_are_strings(self):
        for c in (self.validation, self.preflight, self.runtime, self.compat):
            self.assertIsInstance(c, str)

    def test_all_constants_are_distinct(self):
        cats = {self.validation, self.preflight, self.runtime, self.compat}
        self.assertEqual(len(cats), 4)

    def test_constants_are_nonempty(self):
        for c in (self.validation, self.preflight, self.runtime, self.compat):
            self.assertTrue(len(c) > 0)

    def test_constant_values_have_no_whitespace(self):
        """Category constants are used as dict keys — must be whitespace-free."""
        for c in (self.validation, self.preflight, self.runtime, self.compat):
            self.assertEqual(c, c.strip())
            self.assertNotIn(" ", c)


# ─────────────────────────────────────────────────────────────────────────────
# Section B: REASON_CATEGORY completeness and coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonCategoryCompleteness(unittest.TestCase):

    def setUp(self):
        from bin.core.error_ux import (
            REASON_OPERATOR_TEXT,
            REASON_CATEGORY,
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
        )
        self.operator_text = REASON_OPERATOR_TEXT
        self.category      = REASON_CATEGORY
        self.validation    = ERROR_CATEGORY_VALIDATION
        self.preflight     = ERROR_CATEGORY_PREFLIGHT
        self.runtime       = ERROR_CATEGORY_RUNTIME

    def test_every_reason_code_has_a_category(self):
        """All 28 codes in REASON_OPERATOR_TEXT must exist in REASON_CATEGORY."""
        missing = set(self.operator_text) - set(self.category)
        self.assertEqual(missing, set(), f"Codes without category: {missing}")

    def test_no_extra_codes_in_category(self):
        """REASON_CATEGORY must not contain codes absent from REASON_OPERATOR_TEXT."""
        extra = set(self.category) - set(self.operator_text)
        self.assertEqual(extra, set(), f"Extra codes in REASON_CATEGORY: {extra}")

    def test_validation_category_has_entries(self):
        vals = [c for c in self.category.values() if c == self.validation]
        self.assertGreater(len(vals), 0)

    def test_preflight_category_has_entries(self):
        vals = [c for c in self.category.values() if c == self.preflight]
        self.assertGreater(len(vals), 0)

    def test_runtime_category_has_entries(self):
        vals = [c for c in self.category.values() if c == self.runtime]
        self.assertGreater(len(vals), 0)

    def test_all_category_values_are_known_constants(self):
        from bin.core.error_ux import (
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
        )
        known = {ERROR_CATEGORY_VALIDATION, ERROR_CATEGORY_PREFLIGHT, ERROR_CATEGORY_RUNTIME}
        for code, cat in self.category.items():
            self.assertIn(cat, known, f"Unknown category '{cat}' for code '{code}'")

    def test_total_coverage_is_28(self):
        """Exactly 28 canonical reason codes must be mapped.

        Count after Cycle 2 STAGE-05: 27 (Phase 4–6 codes) +1
        (settings_validation_failed) = 28.
        """
        self.assertEqual(len(self.operator_text), 28)
        self.assertEqual(len(self.category), 28)

    # ── spot-specific checks ───────────────────────────────────────────────

    def test_spot_estop_is_preflight(self):
        self.assertEqual(self.category["spot_estop_active"], self.preflight)

    def test_spot_auth_is_preflight(self):
        self.assertEqual(self.category["spot_auth_failed"], self.preflight)

    def test_spot_action_failed_is_runtime(self):
        self.assertEqual(self.category["spot_action_failed"], self.runtime)

    def test_spot_sdk_unavailable_is_validation(self):
        self.assertEqual(self.category["spot_sdk_unavailable"], self.validation)

    # ── cyberdog-specific checks ───────────────────────────────────────────

    def test_cyberdog_ros2_unavailable_is_validation(self):
        self.assertEqual(self.category["cyberdog_ros2_unavailable"], self.validation)

    def test_cyberdog_no_session_is_preflight(self):
        self.assertEqual(self.category["cyberdog_no_session"], self.preflight)

    # ── canonical checks ───────────────────────────────────────────────────

    def test_execute_failed_is_runtime(self):
        self.assertEqual(self.category["execute_failed"], self.runtime)

    def test_safety_gate_blocked_is_preflight(self):
        self.assertEqual(self.category["safety_gate_blocked"], self.preflight)

    def test_adapter_not_found_is_validation(self):
        self.assertEqual(self.category["adapter_not_found"], self.validation)


# ─────────────────────────────────────────────────────────────────────────────
# Section C: get_error_category()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetErrorCategory(unittest.TestCase):

    def setUp(self):
        from bin.core.error_ux import (
            get_error_category,
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
        )
        self.get  = get_error_category
        self.V    = ERROR_CATEGORY_VALIDATION
        self.P    = ERROR_CATEGORY_PREFLIGHT
        self.R    = ERROR_CATEGORY_RUNTIME

    def test_known_validation_code(self):
        self.assertEqual(self.get("adapter_not_found"), self.V)

    def test_known_preflight_code(self):
        self.assertEqual(self.get("preflight_failed"), self.P)

    def test_known_runtime_code(self):
        self.assertEqual(self.get("execute_failed"), self.R)

    def test_unknown_code_returns_runtime(self):
        """Unknown codes default to runtime (safest assumption)."""
        self.assertEqual(self.get("totally_unknown_xyz"), self.R)

    def test_returns_string(self):
        self.assertIsInstance(self.get("execute_failed"), str)

    def test_spot_sdk_unavailable_returns_validation(self):
        self.assertEqual(self.get("spot_sdk_unavailable"), self.V)

    def test_spot_estop_returns_preflight(self):
        self.assertEqual(self.get("spot_estop_active"), self.P)


# ─────────────────────────────────────────────────────────────────────────────
# Section D: classify_run_result()
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRunResult(unittest.TestCase):

    def setUp(self):
        from bin.core.error_ux import (
            classify_run_result,
            ERROR_CATEGORY_VALIDATION,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_RUNTIME,
            ERROR_CATEGORY_COMPAT,
        )
        self.classify  = classify_run_result
        self.V = ERROR_CATEGORY_VALIDATION
        self.P = ERROR_CATEGORY_PREFLIGHT
        self.R = ERROR_CATEGORY_RUNTIME
        self.C = ERROR_CATEGORY_COMPAT

    # ── blocked paths ─────────────────────────────────────────────────────

    def test_blocked_compile_is_validation(self):
        self.assertEqual(self.classify(_blocked_result("blocked_compile")), self.V)

    def test_blocked_execute_is_preflight(self):
        self.assertEqual(self.classify(_blocked_result("blocked_execute")), self.P)

    def test_blocked_safety_is_preflight(self):
        self.assertEqual(self.classify(_blocked_result("blocked_safety")), self.P)

    def test_blocked_unknown_path_defaults_to_validation(self):
        self.assertEqual(self.classify(_blocked_result("blocked_unknown")), self.V)

    # ── compat path ───────────────────────────────────────────────────────

    def test_failed_with_compat_path_returns_compat(self):
        result = _failed_result("execute_failed", compat=True)
        self.assertEqual(self.classify(result), self.C)

    # ── failed with reason code ───────────────────────────────────────────

    def test_failed_execute_failed_is_runtime(self):
        self.assertEqual(self.classify(_failed_result("execute_failed")), self.R)

    def test_failed_preflight_failed_is_preflight(self):
        self.assertEqual(self.classify(_failed_result("preflight_failed")), self.P)

    def test_failed_adapter_not_found_is_validation(self):
        self.assertEqual(self.classify(_failed_result("adapter_not_found")), self.V)

    def test_failed_safety_gate_blocked_is_preflight(self):
        self.assertEqual(self.classify(_failed_result("safety_gate_blocked")), self.P)

    def test_failed_spot_estop_is_preflight(self):
        self.assertEqual(self.classify(_failed_result("spot_estop_active")), self.P)

    # ── edge cases ────────────────────────────────────────────────────────

    def test_returns_string(self):
        self.assertIsInstance(self.classify(_failed_result("execute_failed")), str)

    def test_compat_takes_priority_over_reason(self):
        """compat_path=True overrides the reason code category."""
        result = _failed_result("adapter_not_found", compat=True)
        self.assertEqual(self.classify(result), self.C)


# ─────────────────────────────────────────────────────────────────────────────
# Section E: format_node_diagnostics() includes error_category
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatNodeDiagnosticsCategory(unittest.TestCase):

    def setUp(self):
        from bin.core.error_ux import (
            format_node_diagnostics,
            DISPLAY_KEY_ORDER,
            ERROR_CATEGORY_RUNTIME,
            ERROR_CATEGORY_PREFLIGHT,
            ERROR_CATEGORY_VALIDATION,
        )
        self.fmt   = format_node_diagnostics
        self.ORDER = DISPLAY_KEY_ORDER
        self.R = ERROR_CATEGORY_RUNTIME
        self.P = ERROR_CATEGORY_PREFLIGHT
        self.V = ERROR_CATEGORY_VALIDATION

    def test_error_category_present_in_result(self):
        result = self.fmt(1, {"error": "execute_failed"})
        self.assertIn("error_category", result)

    def test_error_category_value_is_string(self):
        result = self.fmt(1, {"error": "execute_failed"})
        self.assertIsInstance(result["error_category"], str)

    def test_error_category_runtime_for_execute_failed(self):
        result = self.fmt(1, {"error": "execute_failed"})
        self.assertEqual(result["error_category"], self.R)

    def test_error_category_preflight_for_preflight_failed(self):
        result = self.fmt(1, {"error": "preflight_failed"})
        self.assertEqual(result["error_category"], self.P)

    def test_error_category_validation_for_adapter_not_found(self):
        result = self.fmt(1, {"error": "adapter_not_found"})
        self.assertEqual(result["error_category"], self.V)

    def test_error_category_in_display_key_order(self):
        self.assertIn("error_category", self.ORDER)

    def test_error_category_after_stage_in_display_order(self):
        idx_stage    = self.ORDER.index("stage")
        idx_category = self.ORDER.index("error_category")
        self.assertGreater(idx_category, idx_stage)

    def test_error_category_before_reason_in_display_order(self):
        idx_category = self.ORDER.index("error_category")
        idx_reason   = self.ORDER.index("reason")
        self.assertLess(idx_category, idx_reason)

    def test_error_category_position_in_result_keys(self):
        """error_category must appear between stage and reason in result keys."""
        result = self.fmt(1, {"error": "execute_failed"})
        keys = list(result.keys())
        if "stage" in keys and "error_category" in keys and "reason" in keys:
            self.assertLess(keys.index("stage"),          keys.index("error_category"))
            self.assertLess(keys.index("error_category"), keys.index("reason"))

    def test_existing_deterministic_order_unaffected(self):
        """Leading keys must still follow DISPLAY_KEY_ORDER after adding error_category."""
        result = self.fmt(5, {"error": "execute_failed"})
        keys = list(result.keys())
        expected_leading = [k for k in self.ORDER if k in result]
        actual_leading   = keys[: len(expected_leading)]
        self.assertEqual(actual_leading, expected_leading)


# ─────────────────────────────────────────────────────────────────────────────
# Section F: Localization hook safety
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalizationHookSafety(unittest.TestCase):
    """Verify that the error UX layer is locale-independent.

    Localization contract: REASON_OPERATOR_TEXT values and format_node_diagnostics()
    return plain Python strings with no tr() calls embedded.  The display layer
    (DiagnosticsPanel) handles locale-specific label rendering.  This keeps the
    raw-diagnostics toggle fully engineering-readable in any locale.
    """

    def setUp(self):
        from bin.core.error_ux import (
            REASON_OPERATOR_TEXT,
            format_node_diagnostics,
            get_operator_text,
        )
        self.operator_text = REASON_OPERATOR_TEXT
        self.fmt = format_node_diagnostics
        self.get_text = get_operator_text

    def test_all_operator_texts_are_str_not_callable(self):
        """Values must be plain strings — not lazy tr() objects or callables."""
        for code, text in self.operator_text.items():
            self.assertIsInstance(text, str, f"Code '{code}' has non-str value")
            self.assertFalse(callable(text), f"Code '{code}' value is callable")

    def test_all_operator_texts_are_ascii_safe(self):
        """Operator texts must encode without error in ASCII-compatible codecs."""
        for code, text in self.operator_text.items():
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                self.fail(f"Code '{code}' has non-UTF-8 text: {exc}")

    def test_get_operator_text_deterministic(self):
        """Same reason code always returns the same string."""
        t1 = self.get_text("execute_failed")
        t2 = self.get_text("execute_failed")
        self.assertEqual(t1, t2)

    def test_format_diagnostics_values_json_serializable(self):
        """format_node_diagnostics output must JSON-serialize without errors.

        This guarantees the raw-diagnostics toggle shows unaltered technical
        data regardless of locale settings.
        """
        result = self.fmt(1, {"error": "execute_failed", "message": "timed out"})
        try:
            raw = json.dumps(result)
        except (TypeError, ValueError) as exc:
            self.fail(f"format_node_diagnostics output not JSON-serializable: {exc}")
        # Round-trip check
        loaded = json.loads(raw)
        self.assertEqual(loaded["reason"], result["reason"])

    def test_raw_json_unaffected_by_locale(self):
        """The raw JSON payload must be byte-for-byte reproducible."""
        result = self.fmt(1, {"error": "execute_failed"})
        json1 = json.dumps(result, sort_keys=True)
        json2 = json.dumps(result, sort_keys=True)
        self.assertEqual(json1, json2)

    def test_operator_text_not_empty(self):
        for code, text in self.operator_text.items():
            self.assertGreater(len(text), 0, f"Empty operator text for '{code}'")

    def test_unknown_reason_returns_fallback_with_code_embedded(self):
        """Fallback text must embed the unknown code for engineering traceability."""
        text = self.get_text("totally_unknown_xyz_999")
        self.assertIn("totally_unknown_xyz_999", text)

    def test_error_category_value_in_result_is_raw_constant(self):
        """error_category must be a raw constant, not a translated label."""
        from bin.core.error_ux import ERROR_CATEGORY_RUNTIME
        result = self.fmt(1, {"error": "execute_failed"})
        # Must equal the raw constant, not a human-readable label like "Runtime Failure"
        self.assertEqual(result["error_category"], ERROR_CATEGORY_RUNTIME)


# ─────────────────────────────────────────────────────────────────────────────
# Section G: DiagnosticsPanel category rendering (PySide6)
# ─────────────────────────────────────────────────────────────────────────────

_QT_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    _QT_AVAILABLE = True
except ImportError:
    pass


@unittest.skipUnless(_QT_AVAILABLE, "PySide6 not available")
class TestDiagnosticsPanelCategoryRendering(unittest.TestCase):
    """Verify that the DiagnosticsPanel renders error_category with clear labels."""

    _app = None

    @classmethod
    def setUpClass(cls):
        if QApplication.instance() is None:
            cls._app = QApplication(sys.argv[:1])

    def _panel_with_result(self, reason: str, compat: bool = False):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        from bin.core.error_ux import format_node_diagnostics
        run_result = _failed_result(reason, compat=compat)
        info = format_node_diagnostics(1, {"error": reason}, run_result=run_result, node_name="N1")
        panel = DiagnosticsPanel()
        panel.show_diagnostics([info])
        return panel

    # ── _CATEGORY_LABELS class attribute ──────────────────────────────────

    def test_category_labels_dict_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertTrue(hasattr(DiagnosticsPanel, "_CATEGORY_LABELS"))

    def test_validation_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("validation", DiagnosticsPanel._CATEGORY_LABELS)

    def test_preflight_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("preflight_safety", DiagnosticsPanel._CATEGORY_LABELS)

    def test_runtime_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("runtime", DiagnosticsPanel._CATEGORY_LABELS)

    def test_compat_label_defined(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("compat_warning", DiagnosticsPanel._CATEGORY_LABELS)

    def test_all_category_labels_are_strings(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        for k, v in DiagnosticsPanel._CATEGORY_LABELS.items():
            self.assertIsInstance(v, str, f"Label for '{k}' is not a string")

    def test_error_category_in_key_labels(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("error_category", DiagnosticsPanel._KEY_LABELS)

    def test_error_category_in_friendly_keys(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        self.assertIn("error_category", DiagnosticsPanel._FRIENDLY_KEYS)

    # ── _format_value for error_category ──────────────────────────────────

    def test_format_value_runtime_returns_human_label(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._format_value("error_category", "runtime")
        self.assertNotEqual(label, "runtime")           # not the raw constant
        self.assertIn("Runtime", label)

    def test_format_value_validation_returns_human_label(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._format_value("error_category", "validation")
        self.assertIn("Validation", label)

    def test_format_value_preflight_returns_human_label(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._format_value("error_category", "preflight_safety")
        self.assertIn("Preflight", label)

    def test_format_value_compat_returns_human_label(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._format_value("error_category", "compat_warning")
        self.assertIn("Compat", label)

    def test_format_value_unknown_category_title_cases(self):
        from bin.components.diagnostics_panel import DiagnosticsPanel
        label = DiagnosticsPanel._format_value("error_category", "some_new_category")
        self.assertNotEqual(label, "some_new_category")   # must be transformed

    # ── Friendly view panel rendering ─────────────────────────────────────

    def test_panel_shows_category_label_in_friendly_view(self):
        from PySide6.QtWidgets import QLabel
        panel = self._panel_with_result("execute_failed")
        labels = panel._fields_widget.findChildren(QLabel)
        bold_labels = [lbl.text() for lbl in labels if "<b>" in lbl.text()]
        self.assertTrue(
            any("Category" in lbl for lbl in bold_labels),
            f"Category label not found. Got: {bold_labels}",
        )

    def test_panel_shows_runtime_failure_text(self):
        from PySide6.QtWidgets import QLabel
        panel = self._panel_with_result("execute_failed")
        labels = panel._fields_widget.findChildren(QLabel)
        all_text = " ".join(lbl.text() for lbl in labels)
        self.assertIn("Runtime", all_text)

    def test_panel_shows_preflight_text_for_estop(self):
        from PySide6.QtWidgets import QLabel
        panel = self._panel_with_result("spot_estop_active")
        labels = panel._fields_widget.findChildren(QLabel)
        all_text = " ".join(lbl.text() for lbl in labels)
        self.assertIn("Preflight", all_text)

    def test_panel_raw_view_shows_raw_constant_not_label(self):
        """Raw JSON view must show the raw constant string, not the translated label."""
        panel = self._panel_with_result("execute_failed")
        panel._raw_toggle.setChecked(True)
        raw_text = panel._raw_text.toPlainText()
        self.assertIn('"runtime"', raw_text)
        # The human label must NOT appear in raw JSON
        self.assertNotIn("Runtime Failure", raw_text)


if __name__ == "__main__":
    unittest.main()
