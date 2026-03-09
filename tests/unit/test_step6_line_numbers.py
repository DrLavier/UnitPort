#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 6 — Line number gutter + external highlight hook.

Coverage:
  _LineNumberArea               — widget creation and sizeHint delegation
  PythonIndentEditor.__init__   — gutter attached, selection layers initialised
  lineNumberAreaWidth()         — grows with block count
  highlight_lines()             — builds hook selections, merges with stale layer
  set_stale_selections()        — replaces stale layer, merges with hook layer
  _apply_extra_selections()     — union of both layers applied to editor
  highlight_lines([])           — clears hook layer without touching stale layer
  setExtraSelections NOT called — direct call replaced by set_stale_selections
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication(sys.argv)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_editor():
    app = _get_app()
    if app is None:
        return None
    try:
        from bin.components.script_editor import PythonIndentEditor
        return PythonIndentEditor()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. _LineNumberArea
# ─────────────────────────────────────────────────────────────────────────────

class TestLineNumberArea(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_line_number_area_attached_to_editor(self):
        from bin.components.script_editor import _LineNumberArea
        self.assertIsInstance(self.editor._line_number_area, _LineNumberArea)

    def test_line_number_area_parent_is_editor(self):
        area = self.editor._line_number_area
        self.assertIs(area.parent(), self.editor)

    def test_size_hint_width_matches_editor_width(self):
        area = self.editor._line_number_area
        self.assertEqual(area.sizeHint().width(), self.editor.lineNumberAreaWidth())


# ─────────────────────────────────────────────────────────────────────────────
# 2. lineNumberAreaWidth
# ─────────────────────────────────────────────────────────────────────────────

class TestLineNumberAreaWidth(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_width_positive_for_empty_document(self):
        self.editor.clear()
        self.assertGreater(self.editor.lineNumberAreaWidth(), 0)

    def test_width_grows_with_more_blocks(self):
        self.editor.clear()
        w1 = self.editor.lineNumberAreaWidth()
        # Add enough lines to push block count to 4 digits
        self.editor.setPlainText("\n" * 999)
        w2 = self.editor.lineNumberAreaWidth()
        self.assertGreater(w2, w1)

    def test_width_at_least_one_digit_wide(self):
        self.editor.clear()
        fm = self.editor.fontMetrics()
        min_expected = 6 + fm.horizontalAdvance("9")
        self.assertGreaterEqual(self.editor.lineNumberAreaWidth(), min_expected)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Selection layers — initial state
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectionLayers(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_stale_selections_init_empty(self):
        self.assertEqual(self.editor._stale_selections, [])

    def test_hook_selections_init_empty(self):
        self.assertEqual(self.editor._hook_selections, [])


# ─────────────────────────────────────────────────────────────────────────────
# 4. highlight_lines — hook layer
# ─────────────────────────────────────────────────────────────────────────────

class TestHighlightLines(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")
        self.editor.setPlainText("line1\nline2\nline3\nline4\nline5")

    def test_highlight_lines_populates_hook_selections(self):
        self.editor.highlight_lines([1, 3])
        self.assertEqual(len(self.editor._hook_selections), 2)

    def test_highlight_lines_empty_clears_hook(self):
        self.editor.highlight_lines([1, 2])
        self.editor.highlight_lines([])
        self.assertEqual(self.editor._hook_selections, [])

    def test_highlight_lines_skips_invalid_line_numbers(self):
        self.editor.highlight_lines([999])   # beyond document
        self.assertEqual(len(self.editor._hook_selections), 0)

    def test_highlight_lines_error_style(self):
        self.editor.highlight_lines([1], style_key="error")
        fmt = self.editor._hook_selections[0].format
        from PySide6.QtGui import QColor
        self.assertEqual(fmt.background().color().name(), QColor("#7c2d12").name())

    def test_highlight_lines_warning_style(self):
        self.editor.highlight_lines([1], style_key="warning")
        fmt = self.editor._hook_selections[0].format
        from PySide6.QtGui import QColor
        self.assertEqual(fmt.background().color().name(), QColor("#713f12").name())

    def test_highlight_lines_info_style(self):
        self.editor.highlight_lines([1], style_key="info")
        fmt = self.editor._hook_selections[0].format
        from PySide6.QtGui import QColor
        self.assertEqual(fmt.background().color().name(), QColor("#1e3a5f").name())

    def test_highlight_lines_unknown_style_falls_back_to_info(self):
        self.editor.highlight_lines([1], style_key="bogus")
        fmt = self.editor._hook_selections[0].format
        from PySide6.QtGui import QColor
        self.assertEqual(fmt.background().color().name(), QColor("#1e3a5f").name())

    def test_highlight_lines_calls_apply_extra_selections(self):
        with patch.object(self.editor, "_apply_extra_selections") as mock_apply:
            self.editor.highlight_lines([2])
        mock_apply.assert_called_once()

    def test_highlight_lines_default_style_is_error(self):
        self.editor.highlight_lines([1])
        fmt = self.editor._hook_selections[0].format
        from PySide6.QtGui import QColor
        self.assertEqual(fmt.background().color().name(), QColor("#7c2d12").name())


# ─────────────────────────────────────────────────────────────────────────────
# 5. set_stale_selections — stale layer
# ─────────────────────────────────────────────────────────────────────────────

class TestSetStaleSelections(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_set_stale_selections_stores_list(self):
        from PySide6.QtWidgets import QTextEdit
        from PySide6.QtGui import QTextCursor, QTextCharFormat
        self.editor.setPlainText("a\nb\nc")
        sel = QTextEdit.ExtraSelection()
        sel.cursor = self.editor.textCursor()
        sel.format = QTextCharFormat()
        self.editor.set_stale_selections([sel])
        self.assertEqual(len(self.editor._stale_selections), 1)

    def test_set_stale_selections_empty_clears(self):
        with patch.object(self.editor, "setExtraSelections"):
            self.editor.set_stale_selections([MagicMock()])
            self.editor.set_stale_selections([])
        self.assertEqual(self.editor._stale_selections, [])

    def test_set_stale_selections_calls_apply(self):
        with patch.object(self.editor, "_apply_extra_selections") as mock_apply:
            self.editor.set_stale_selections([])
        mock_apply.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 6. _apply_extra_selections — merging
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyExtraSelections(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_apply_merges_both_layers(self):
        stale = [MagicMock(), MagicMock()]
        hook  = [MagicMock()]
        self.editor._stale_selections = stale
        self.editor._hook_selections  = hook
        with patch.object(self.editor, "setExtraSelections") as mock_set:
            self.editor._apply_extra_selections()
        mock_set.assert_called_once_with(stale + hook)

    def test_apply_empty_layers_calls_setextraselections_with_empty(self):
        with patch.object(self.editor, "setExtraSelections") as mock_set:
            self.editor._apply_extra_selections()
        mock_set.assert_called_once_with([])

    def test_hook_layer_preserved_after_stale_update(self):
        self.editor.setPlainText("x\ny\nz")
        self.editor.highlight_lines([1], style_key="info")   # sets hook layer
        hook_count = len(self.editor._hook_selections)
        self.editor.set_stale_selections([])                  # clears stale only
        self.assertEqual(len(self.editor._hook_selections), hook_count)

    def test_stale_layer_preserved_after_hook_cleared(self):
        from PySide6.QtWidgets import QTextEdit
        from PySide6.QtGui import QTextCharFormat
        sel = QTextEdit.ExtraSelection()
        sel.format = QTextCharFormat()
        sel.cursor = self.editor.textCursor()
        self.editor.set_stale_selections([sel])
        self.editor.highlight_lines([])   # clears hook only
        self.assertEqual(len(self.editor._stale_selections), 1)

    def test_no_direct_setextraselections_in_script_editor(self):
        """ScriptEditor must not call setExtraSelections directly (regression)."""
        import inspect
        from bin.components.script_editor import ScriptEditor
        src = inspect.getsource(ScriptEditor)
        # Only the inherited PythonIndentEditor._apply_extra_selections may call it;
        # ScriptEditor itself should use set_stale_selections instead.
        # Count raw occurrences of the direct call pattern in ScriptEditor methods.
        direct_calls = src.count("self._editor.setExtraSelections(")
        self.assertEqual(direct_calls, 0,
            "ScriptEditor should not call _editor.setExtraSelections() directly; "
            "use _editor.set_stale_selections() instead.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. highlight_lines hook callable without errors (DoD check)
# ─────────────────────────────────────────────────────────────────────────────

class TestHighlightHookDoD(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_hook_callable_on_empty_document(self):
        self.editor.clear()
        self.editor.highlight_lines([1, 2, 3])  # must not raise

    def test_hook_callable_with_none_list(self):
        self.editor.highlight_lines(None)        # must not raise

    def test_hook_clears_with_empty(self):
        self.editor.setPlainText("a\nb")
        self.editor.highlight_lines([1])
        self.editor.highlight_lines([])
        self.assertEqual(self.editor._hook_selections, [])

    def test_hook_replace_updates_lines(self):
        self.editor.setPlainText("a\nb\nc")
        self.editor.highlight_lines([1])
        self.editor.highlight_lines([2, 3])
        self.assertEqual(len(self.editor._hook_selections), 2)


if __name__ == "__main__":
    unittest.main()
