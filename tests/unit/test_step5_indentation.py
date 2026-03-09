#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 5 — Python indentation engine.

Coverage:
  _compute_smart_indent()      — pure-Python, no Qt required
  PythonIndentEditor behaviour — Tab / Shift+Tab / Enter via Qt key simulation
"""

import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bin.components.script_editor import _compute_smart_indent


# ─────────────────────────────────────────────────────────────────────────────
# 1. _compute_smart_indent — pure logic (no Qt)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSmartIndent(unittest.TestCase):

    # ── blank / flat lines ────────────────────────────────────────────────────

    def test_empty_line_gives_empty_indent(self):
        self.assertEqual(_compute_smart_indent(""), "")

    def test_flat_line_carries_zero_indent(self):
        self.assertEqual(_compute_smart_indent("x = 1"), "")

    def test_indented_line_carries_indent(self):
        self.assertEqual(_compute_smart_indent("    x = 1"), "    ")

    def test_double_indented_line_carries_indent(self):
        self.assertEqual(_compute_smart_indent("        x = 1"), "        ")

    def test_trailing_spaces_do_not_affect_indent_count(self):
        self.assertEqual(_compute_smart_indent("    x = 1   "), "    ")

    # ── block openers — should add 4 spaces ──────────────────────────────────

    def test_if_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("if x > 0:"), "    ")

    def test_for_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("for i in range(10):"), "    ")

    def test_while_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("while True:"), "    ")

    def test_def_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("def run():"), "    ")

    def test_class_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("class Foo:"), "    ")

    def test_else_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("    else:"), "        ")

    def test_elif_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("    elif x > 0:"), "        ")

    def test_except_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("    except Exception:"), "        ")

    def test_finally_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("    finally:"), "        ")

    def test_try_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("try:"), "    ")

    def test_with_colon_adds_indent(self):
        self.assertEqual(_compute_smart_indent("    with open(f) as fh:"), "        ")

    # ── colon mid-line or in strings should not trigger ───────────────────────

    def test_dict_literal_does_not_add_indent(self):
        # "d = {'key': 'val'}" — colon is NOT at end of stripped line
        self.assertEqual(_compute_smart_indent("    d = {'key': 'val'}"), "    ")

    def test_comment_after_colon_still_adds_indent(self):
        self.assertEqual(_compute_smart_indent("if x:  # check"), "    ")

    def test_colon_with_trailing_comment_no_space(self):
        self.assertEqual(_compute_smart_indent("if x:#check"), "    ")

    # ── indented block openers preserve existing indent ───────────────────────

    def test_nested_if_carries_and_adds(self):
        self.assertEqual(_compute_smart_indent("        if y:"), "            ")

    def test_nested_for_inside_def(self):
        self.assertEqual(_compute_smart_indent("    for i in lst:"), "        ")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PythonIndentEditor widget tests (require Qt)
# ─────────────────────────────────────────────────────────────────────────────

def _make_editor():
    """Return a PythonIndentEditor instance; skip if Qt unavailable."""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from bin.components.script_editor import PythonIndentEditor
        return PythonIndentEditor()
    except Exception:
        return None


def _send_key(editor, key, modifier=None):
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QKeyEvent
    mod = modifier if modifier is not None else Qt.NoModifier
    press = QKeyEvent(QEvent.KeyPress, key, mod)
    editor.keyPressEvent(press)


def _send_tab(editor):
    from PySide6.QtCore import Qt
    _send_key(editor, Qt.Key_Tab)


def _send_backtab(editor):
    from PySide6.QtCore import Qt
    _send_key(editor, Qt.Key_Backtab)


def _send_return(editor):
    from PySide6.QtCore import Qt
    _send_key(editor, Qt.Key_Return)


class TestPythonIndentEditorTab(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")
        self.editor.clear()

    def test_tab_inserts_4_spaces(self):
        _send_tab(self.editor)
        self.assertEqual(self.editor.toPlainText(), "    ")

    def test_two_tabs_inserts_8_spaces(self):
        _send_tab(self.editor)
        _send_tab(self.editor)
        self.assertEqual(self.editor.toPlainText(), "        ")

    def test_tab_does_not_insert_tab_character(self):
        _send_tab(self.editor)
        self.assertNotIn("\t", self.editor.toPlainText())

    def test_tab_with_selection_indents_all_lines(self):
        from PySide6.QtGui import QTextCursor
        self.editor.setPlainText("line1\nline2\nline3")
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        self.editor.setTextCursor(cursor)
        _send_tab(self.editor)
        lines = self.editor.toPlainText().splitlines()
        self.assertTrue(all(ln.startswith("    ") for ln in lines))

    def test_tab_with_partial_selection_indents_covered_lines(self):
        from PySide6.QtGui import QTextCursor
        self.editor.setPlainText("line1\nline2\nline3")
        # Select only first two lines
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 1)
        cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
        self.editor.setTextCursor(cursor)
        _send_tab(self.editor)
        lines = self.editor.toPlainText().splitlines()
        self.assertTrue(lines[0].startswith("    "))
        self.assertTrue(lines[1].startswith("    "))
        self.assertFalse(lines[2].startswith("    "))


class TestPythonIndentEditorBacktab(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def test_backtab_removes_4_spaces(self):
        self.editor.setPlainText("    line")
        from PySide6.QtGui import QTextCursor
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.editor.setTextCursor(cursor)
        _send_backtab(self.editor)
        self.assertEqual(self.editor.toPlainText(), "line")

    def test_backtab_removes_at_most_4_spaces(self):
        self.editor.setPlainText("  line")
        from PySide6.QtGui import QTextCursor
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.editor.setTextCursor(cursor)
        _send_backtab(self.editor)
        self.assertEqual(self.editor.toPlainText(), "line")

    def test_backtab_on_unindented_line_is_noop(self):
        self.editor.setPlainText("line")
        from PySide6.QtGui import QTextCursor
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.editor.setTextCursor(cursor)
        _send_backtab(self.editor)
        self.assertEqual(self.editor.toPlainText(), "line")

    def test_backtab_with_selection_dedents_all_lines(self):
        from PySide6.QtGui import QTextCursor
        self.editor.setPlainText("    a\n    b\n    c")
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        self.editor.setTextCursor(cursor)
        _send_backtab(self.editor)
        lines = self.editor.toPlainText().splitlines()
        self.assertEqual(lines, ["a", "b", "c"])

    def test_backtab_with_mixed_indent_removes_up_to_4_each(self):
        from PySide6.QtGui import QTextCursor
        self.editor.setPlainText("      deep\n  shallow")
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        self.editor.setTextCursor(cursor)
        _send_backtab(self.editor)
        lines = self.editor.toPlainText().splitlines()
        self.assertEqual(lines[0], "  deep")    # 6-4 = 2 spaces remaining
        self.assertEqual(lines[1], "shallow")   # 2-2 = 0 spaces remaining


class TestPythonIndentEditorReturn(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")

    def _type_line_and_enter(self, line: str) -> str:
        """Set editor to *line*, put cursor at end, press Enter, return new text."""
        self.editor.setPlainText(line)
        from PySide6.QtGui import QTextCursor
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.editor.setTextCursor(cursor)
        _send_return(self.editor)
        return self.editor.toPlainText()

    def test_enter_on_flat_line_gives_no_indent(self):
        text = self._type_line_and_enter("x = 1")
        lines = text.split("\n")   # split preserves trailing empty string
        self.assertEqual(lines[1], "")

    def test_enter_carries_current_indent(self):
        text = self._type_line_and_enter("    x = 1")
        lines = text.splitlines()
        self.assertEqual(lines[1], "    ")

    def test_enter_after_if_colon_adds_level(self):
        text = self._type_line_and_enter("if x > 0:")
        lines = text.splitlines()
        self.assertEqual(lines[1], "    ")

    def test_enter_after_for_colon_adds_level(self):
        text = self._type_line_and_enter("for i in range(n):")
        lines = text.splitlines()
        self.assertEqual(lines[1], "    ")

    def test_enter_after_def_colon_adds_level(self):
        text = self._type_line_and_enter("def run():")
        lines = text.splitlines()
        self.assertEqual(lines[1], "    ")

    def test_enter_after_indented_if_carries_and_adds(self):
        text = self._type_line_and_enter("    if cond:")
        lines = text.splitlines()
        self.assertEqual(lines[1], "        ")

    def test_enter_after_else_colon_adds_level(self):
        text = self._type_line_and_enter("    else:")
        lines = text.splitlines()
        self.assertEqual(lines[1], "        ")

    def test_enter_after_dict_does_not_add_level(self):
        text = self._type_line_and_enter("    d = {'a': 1}")
        lines = text.splitlines()
        self.assertEqual(lines[1], "    ")

    def test_enter_with_selection_deletes_then_indents(self):
        from PySide6.QtGui import QTextCursor
        self.editor.setPlainText("if x:")
        cursor = self.editor.textCursor()
        # Select "x" inside the if condition
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.Right, QTextCursor.MoveAnchor, 3)
        cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 1)
        self.editor.setTextCursor(cursor)
        _send_return(self.editor)
        # Should NOT crash; content should have a newline
        text = self.editor.toPlainText()
        self.assertIn("\n", text)

    def test_enter_after_comment_colon_does_not_add_level(self):
        # "x = 1  # colon: here" — the regex requires the colon to be the
        # last non-comment, non-whitespace character. "here" follows the
        # colon, so the regex does NOT match → indent stays 0.
        text = self._type_line_and_enter("x = 1  # colon: here")
        lines = text.split("\n")   # split preserves trailing empty string
        self.assertEqual(lines[1], "")


if __name__ == "__main__":
    unittest.main()
