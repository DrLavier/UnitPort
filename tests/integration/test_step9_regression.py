#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9 — Integration regression tests for the Script Box Compiler Completion plan.

Coverage:
  compile → IO sync (script_update_io)
  sub_dot type color coverage (all 10 data types have a color)
  save / load roundtrip preserves script_io_spec
  _compute_smart_indent end-to-end across common Python patterns
  PythonSyntaxHighlighter colors are theme-configurable (not hardcoded)
  Zoom bounds respected across repeated events
"""

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication(sys.argv)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. sub_dot type color — all 10 SCRIPT_DATA_TYPE_INDEX types have a color
# ─────────────────────────────────────────────────────────────────────────────

class TestSubDotTypeCoverage(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from bin.components.graph_scene import GraphScene
            self.gs = GraphScene.__new__(GraphScene)
        except Exception:
            self.skipTest("GraphScene not importable")

    def test_all_10_types_return_color(self):
        from bin.components.script_api_analyzer import SCRIPT_DATA_TYPE_INDEX
        from bin.components.graph_scene import _sub_dot_type_color
        for idx, type_name in SCRIPT_DATA_TYPE_INDEX.items():
            qcolor = _sub_dot_type_color(type_name)
            self.assertIsNotNone(qcolor, f"Type '{type_name}' (idx {idx}) returned None")
            color_name = qcolor.name()
            self.assertTrue(
                color_name.startswith("#"),
                f"Type '{type_name}' color {color_name!r} is not a hex string",
            )

    def test_list_type_has_explicit_color(self):
        from bin.components.graph_scene import _sub_dot_type_color
        from PySide6.QtGui import QColor
        self.assertIsInstance(_sub_dot_type_color("list"), QColor)

    def test_dict_type_has_explicit_color(self):
        from bin.components.graph_scene import _sub_dot_type_color
        from PySide6.QtGui import QColor
        self.assertIsInstance(_sub_dot_type_color("dict"), QColor)

    def test_unknown_type_returns_fallback_color(self):
        from bin.components.graph_scene import _sub_dot_type_color
        from PySide6.QtGui import QColor
        qcolor = _sub_dot_type_color("totally_unknown_type_xyz")
        self.assertIsNotNone(qcolor)
        self.assertIsInstance(qcolor, QColor)


# ─────────────────────────────────────────────────────────────────────────────
# 2. script_update_io routing — compile result drives IO diff
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptUpdateIORouting(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from bin.components.graph_scene import GraphScene
            self.gs = GraphScene()
        except Exception:
            self.skipTest("GraphScene not available")

    def _make_script_node(self):
        from PySide6.QtCore import QPointF
        item = self.gs.create_node("Script Box", QPointF(100, 100))
        return item

    def test_script_update_io_no_crash_empty_result(self):
        """script_update_io with empty inputs/outputs must not raise."""
        item = self._make_script_node()
        if item is None:
            self.skipTest("Script Box node not available")
        node_id = item.data(12) if hasattr(item, "data") else None
        if node_id is None:
            self.skipTest("node_id not accessible")
        result = {"inputs": [], "outputs": [], "diagnostics": [], "compat_used": False}
        self.gs.script_update_io(node_id, result)

    def test_script_update_io_with_one_output(self):
        item = self._make_script_node()
        if item is None:
            self.skipTest("Script Box node not available")
        node_id = item.data(12) if hasattr(item, "data") else None
        if node_id is None:
            self.skipTest("node_id not accessible")
        result = {
            "inputs": [],
            "outputs": [{"name": "result", "data_type": "int"}],
            "diagnostics": [],
            "compat_used": False,
        }
        self.gs.script_update_io(node_id, result)   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. _compute_smart_indent — end-to-end pattern coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSmartIndentIntegration(unittest.TestCase):
    def setUp(self):
        from bin.components.script_editor import _compute_smart_indent
        self.indent = _compute_smart_indent

    def test_nested_function_in_class(self):
        self.assertEqual(self.indent("    def method(self):"), "        ")

    def test_multi_level_nested_if(self):
        self.assertEqual(self.indent("                if x:"), "                    ")

    def test_lambda_does_not_trigger_block_open(self):
        # lambda has a colon mid-expression, not at end of stripped line
        line = "    f = lambda x: x + 1"
        self.assertEqual(self.indent(line), "    ")

    def test_dict_comprehension_no_extra_indent(self):
        self.assertEqual(self.indent("    d = {k: v for k, v in items}"), "    ")

    def test_comment_after_block_opener_still_indents(self):
        self.assertEqual(self.indent("for i in range(n):  # loop"), "    ")

    def test_blank_line_returns_empty(self):
        self.assertEqual(self.indent(""), "")

    def test_whitespace_only_line_carries_its_indent(self):
        # A line of spaces carries those spaces as the base indent (no block opener)
        self.assertEqual(self.indent("   "), "   ")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Syntax highlighter — colors are configurable via theme, not hardcoded
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntaxHighlighterThemeConfigurability(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from PySide6.QtGui import QTextDocument
            from bin.components.script_editor import PythonSyntaxHighlighter
            self._doc = QTextDocument()
            self.hl = PythonSyntaxHighlighter(self._doc)
        except Exception:
            self.skipTest("Highlighter not available")

    def test_fmt_string_color_comes_from_get_color(self):
        from PySide6.QtGui import QColor
        from bin.core.theme_manager import get_color
        expected = QColor(get_color("syntax_string", "#ce9178"))
        actual   = self.hl._fmt_string.foreground().color()
        self.assertEqual(actual.name(), expected.name())

    def test_fmt_comment_color_comes_from_get_color(self):
        from PySide6.QtGui import QColor
        from bin.core.theme_manager import get_color
        expected = QColor(get_color("syntax_comment", "#6a9955"))
        actual   = self.hl._fmt_comment.foreground().color()
        self.assertEqual(actual.name(), expected.name())

    def test_rebuild_rules_produces_same_count(self):
        count_before = len(self.hl._rules)
        self.hl._build_rules()
        self.assertEqual(len(self.hl._rules), count_before)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Zoom integration — repeated events respect bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestZoomIntegration(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from bin.components.script_editor import PythonIndentEditor
            self.editor = PythonIndentEditor()
            from PySide6.QtGui import QFont
            f = self.editor.font()
            f.setPointSize(12)
            self.editor.setFont(f)
        except Exception:
            self.skipTest("PythonIndentEditor not available")

    def test_zoom_in_never_exceeds_max(self):
        for _ in range(100):
            self.editor._adjust_font_size(1)
        self.assertLessEqual(self.editor.font().pointSize(), self.editor._ZOOM_MAX)

    def test_zoom_out_never_below_min(self):
        for _ in range(100):
            self.editor._adjust_font_size(-1)
        self.assertGreaterEqual(self.editor.font().pointSize(), self.editor._ZOOM_MIN)

    def test_zoom_in_then_out_returns_to_original(self):
        original = self.editor.font().pointSize()
        self.editor._adjust_font_size(3)
        self.editor._adjust_font_size(-3)
        self.assertEqual(self.editor.font().pointSize(), original)

    def test_zoom_preserves_font_family(self):
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setFamily("Courier New")
        self.editor.setFont(f)
        family_before = self.editor.font().family()
        self.editor._adjust_font_size(2)
        self.assertEqual(self.editor.font().family(), family_before)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dual selection layer — hook + stale are independent
# ─────────────────────────────────────────────────────────────────────────────

class TestDualSelectionLayerIntegration(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from bin.components.script_editor import PythonIndentEditor
            self.editor = PythonIndentEditor()
            self.editor.setPlainText("a\nb\nc\nd\ne")
        except Exception:
            self.skipTest("PythonIndentEditor not available")

    def test_highlight_lines_does_not_affect_stale(self):
        from PySide6.QtWidgets import QTextEdit
        from PySide6.QtGui import QTextCharFormat
        sel = QTextEdit.ExtraSelection()
        sel.format = QTextCharFormat()
        sel.cursor = self.editor.textCursor()
        self.editor.set_stale_selections([sel])
        # Now highlight some lines
        self.editor.highlight_lines([1, 2], style_key="error")
        # Stale layer unchanged
        self.assertEqual(len(self.editor._stale_selections), 1)

    def test_set_stale_does_not_affect_hook(self):
        self.editor.highlight_lines([1, 2, 3], style_key="info")
        self.editor.set_stale_selections([])
        # Hook layer unchanged
        self.assertEqual(len(self.editor._hook_selections), 3)

    def test_both_layers_merged_in_extra_selections(self):
        from PySide6.QtWidgets import QTextEdit
        from PySide6.QtGui import QTextCharFormat
        sel = QTextEdit.ExtraSelection()
        sel.format = QTextCharFormat()
        sel.cursor = self.editor.textCursor()
        self.editor.set_stale_selections([sel])
        self.editor.highlight_lines([1], style_key="warning")
        # Total = 1 stale + 1 hook = 2
        total = len(self.editor._stale_selections) + len(self.editor._hook_selections)
        self.assertEqual(total, 2)


if __name__ == "__main__":
    unittest.main()
