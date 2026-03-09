#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 8 — Ctrl + Mouse Wheel zoom in PythonIndentEditor.

Coverage:
  PythonIndentEditor._ZOOM_MIN / _ZOOM_MAX  — class constants
  _adjust_font_size()                        — increases/decreases font pt size
  _adjust_font_size() clamp                 — never exceeds [MIN, MAX]
  _adjust_font_size() no-op at boundary     — font unchanged at limits
  wheelEvent() + Ctrl                       — increases/decreases font
  wheelEvent() without Ctrl                 — does NOT change font size
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


def _make_editor():
    app = _get_app()
    if app is None:
        return None
    try:
        from bin.components.script_editor import PythonIndentEditor
        return PythonIndentEditor()
    except Exception:
        return None


def _ctrl_wheel_event(delta_y: int):
    """Build a Ctrl+wheel QWheelEvent with the given vertical angleDelta."""
    from PySide6.QtCore import Qt, QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    return QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.NoButton,
        Qt.ControlModifier,
        Qt.NoScrollPhase,
        False,
    )


def _plain_wheel_event(delta_y: int):
    """Build a plain (no modifier) wheel QWheelEvent."""
    from PySide6.QtCore import Qt, QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    return QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class constants
# ─────────────────────────────────────────────────────────────────────────────

class TestZoomConstants(unittest.TestCase):
    def test_zoom_min_less_than_zoom_max(self):
        from bin.components.script_editor import PythonIndentEditor
        self.assertLess(PythonIndentEditor._ZOOM_MIN, PythonIndentEditor._ZOOM_MAX)

    def test_zoom_min_positive(self):
        from bin.components.script_editor import PythonIndentEditor
        self.assertGreater(PythonIndentEditor._ZOOM_MIN, 0)

    def test_zoom_max_reasonable(self):
        from bin.components.script_editor import PythonIndentEditor
        self.assertGreaterEqual(PythonIndentEditor._ZOOM_MAX, 20)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _adjust_font_size
# ─────────────────────────────────────────────────────────────────────────────

class TestAdjustFontSize(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(12)
        self.editor.setFont(f)

    def test_positive_step_increases_size(self):
        self.editor._adjust_font_size(1)
        self.assertEqual(self.editor.font().pointSize(), 13)

    def test_negative_step_decreases_size(self):
        self.editor._adjust_font_size(-1)
        self.assertEqual(self.editor.font().pointSize(), 11)

    def test_large_positive_step_clamped_at_max(self):
        self.editor._adjust_font_size(9999)
        self.assertEqual(self.editor.font().pointSize(),
                         self.editor._ZOOM_MAX)

    def test_large_negative_step_clamped_at_min(self):
        self.editor._adjust_font_size(-9999)
        self.assertEqual(self.editor.font().pointSize(),
                         self.editor._ZOOM_MIN)

    def test_at_max_positive_step_is_noop(self):
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(self.editor._ZOOM_MAX)
        self.editor.setFont(f)
        self.editor._adjust_font_size(1)
        self.assertEqual(self.editor.font().pointSize(), self.editor._ZOOM_MAX)

    def test_at_min_negative_step_is_noop(self):
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(self.editor._ZOOM_MIN)
        self.editor.setFont(f)
        self.editor._adjust_font_size(-1)
        self.assertEqual(self.editor.font().pointSize(), self.editor._ZOOM_MIN)

    def test_zero_step_is_noop(self):
        before = self.editor.font().pointSize()
        self.editor._adjust_font_size(0)
        self.assertEqual(self.editor.font().pointSize(), before)

    def test_multi_step_accumulates(self):
        self.editor._adjust_font_size(2)
        self.editor._adjust_font_size(3)
        self.assertEqual(self.editor.font().pointSize(), 17)


# ─────────────────────────────────────────────────────────────────────────────
# 3. wheelEvent — Ctrl held
# ─────────────────────────────────────────────────────────────────────────────

class TestWheelEventCtrl(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(14)
        self.editor.setFont(f)

    def test_ctrl_wheel_up_increases_font(self):
        self.editor.wheelEvent(_ctrl_wheel_event(120))
        self.assertEqual(self.editor.font().pointSize(), 15)

    def test_ctrl_wheel_down_decreases_font(self):
        self.editor.wheelEvent(_ctrl_wheel_event(-120))
        self.assertEqual(self.editor.font().pointSize(), 13)

    def test_ctrl_wheel_zero_delta_noop(self):
        before = self.editor.font().pointSize()
        self.editor.wheelEvent(_ctrl_wheel_event(0))
        self.assertEqual(self.editor.font().pointSize(), before)

    def test_ctrl_wheel_repeated_up_clamps(self):
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(self.editor._ZOOM_MAX - 1)
        self.editor.setFont(f)
        self.editor.wheelEvent(_ctrl_wheel_event(120))
        self.editor.wheelEvent(_ctrl_wheel_event(120))   # second one clamped
        self.assertEqual(self.editor.font().pointSize(), self.editor._ZOOM_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# 4. wheelEvent — plain scroll (no Ctrl)
# ─────────────────────────────────────────────────────────────────────────────

class TestWheelEventNoCtrl(unittest.TestCase):
    def setUp(self):
        self.editor = _make_editor()
        if self.editor is None:
            self.skipTest("Qt not available")
        from PySide6.QtGui import QFont
        f = self.editor.font()
        f.setPointSize(14)
        self.editor.setFont(f)

    def test_plain_wheel_up_does_not_change_font(self):
        self.editor.wheelEvent(_plain_wheel_event(120))
        self.assertEqual(self.editor.font().pointSize(), 14)

    def test_plain_wheel_down_does_not_change_font(self):
        self.editor.wheelEvent(_plain_wheel_event(-120))
        self.assertEqual(self.editor.font().pointSize(), 14)


if __name__ == "__main__":
    unittest.main()
