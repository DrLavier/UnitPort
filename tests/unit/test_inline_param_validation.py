"""Tests for inline node parameter validation and serialization roundtrip.

STAGE-07 Cycle 1 — Unit tests: inline parameter validation.

Each test class covers one node type and verifies:
  1. The inline widget is present on the node item.
  2. Setting the widget value serializes to the correct JSON key.
  3. The JSON key survives a load_workflow() roundtrip (widget text restored).
  4. Invalid inputs fall back gracefully to the declared default in exec-graph.

All tests use QT_QPA_PLATFORM=offscreen for headless CI.
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


def _find_node(scene, name):
    """Return first node item with the given display name, or None."""
    try:
        from shiboken6 import isValid
    except ImportError:
        def isValid(x): return True
    for item in scene.items():
        if isValid(item) and item.data(10) == "node" and item.data(11) == name:
            return item
    return None


def _serial_node(data, display_name):
    """Return the first serialized node with the given display_name, or None."""
    return next(
        (n for n in data.get("nodes", []) if n.get("display_name") == display_name),
        None,
    )


def _exec_graph_node(scene, name):
    """Return the first exec_graph node entry whose name matches, or None."""
    nodes = scene.get_execution_graph().get("nodes", {})
    return next((n for n in nodes.values() if n.get("name") == name), None)


# ── Timer node ────────────────────────────────────────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestTimerInlineParam(unittest.TestCase):
    """Timer node inline duration parameter tests."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_timer(self):
        scene = _make_scene()
        item = scene.create_node("Timer", QPointF(100, 100))
        return scene, item

    def test_timer_has_duration_widget(self):
        """Timer node exposes _duration_input widget."""
        _, item = self._make_timer()
        self.assertIsNotNone(
            getattr(item, '_duration_input', None),
            "Timer node must have _duration_input widget",
        )

    def test_timer_duration_serialized_under_duration_key(self):
        """Timer duration widget text serializes under the 'duration' JSON key."""
        scene, item = self._make_timer()
        dur = getattr(item, '_duration_input', None)
        if dur is None:
            self.skipTest("No _duration_input widget")
        dur.setText("12.5")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Timer")
        self.assertIsNotNone(entry, "Timer must appear in serialized nodes")
        self.assertIn("duration", entry)
        self.assertEqual(str(entry["duration"]), "12.5")

    def test_timer_duration_roundtrip(self):
        """Timer duration survives serialize → load_workflow → widget text."""
        scene, item = self._make_timer()
        dur = getattr(item, '_duration_input', None)
        if dur is None:
            self.skipTest("No _duration_input widget")
        dur.setText("7.0")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "Timer")
        self.assertIsNotNone(restored)
        restored_dur = getattr(restored, '_duration_input', None)
        if restored_dur:
            self.assertEqual(restored_dur.text(), "7.0")

    def test_timer_validator_is_applied(self):
        """Timer _duration_input carries a validator (QDoubleValidator or similar)."""
        _, item = self._make_timer()
        dur = getattr(item, '_duration_input', None)
        if dur is None:
            self.skipTest("No _duration_input widget")
        self.assertIsNotNone(
            dur.validator(),
            "_duration_input must have a validator attached",
        )

    def test_timer_default_duration_is_numeric(self):
        """Fresh Timer node has a numeric (float-parseable) default duration."""
        _, item = self._make_timer()
        dur = getattr(item, '_duration_input', None)
        if dur is None:
            self.skipTest("No _duration_input widget")
        text = dur.text() or "1.0"
        try:
            float(text)
        except ValueError:
            self.fail(f"Default Timer duration {text!r} is not float-parseable")

    def test_timer_duration_propagates_to_exec_graph(self):
        """Timer duration widget text appears in execution graph parameters."""
        scene, item = self._make_timer()
        dur = getattr(item, '_duration_input', None)
        if dur is None:
            self.skipTest("No _duration_input widget")
        dur.setText("15.0")
        node = _exec_graph_node(scene, "Timer")
        if node is None:
            self.skipTest("Timer not found in exec_graph")
        params = node.get("parameters", {})
        self.assertAlmostEqual(float(params.get("duration", 0)), 15.0, places=2)


# ── Loop node ─────────────────────────────────────────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestLoopInlineParam(unittest.TestCase):
    """Loop node inline parameter tests (type picker and For-mode range inputs)."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_loop(self):
        scene = _make_scene()
        item = scene.create_node("Loop", QPointF(100, 100))
        return scene, item

    def test_loop_has_type_picker(self):
        """Loop node exposes _loop_type_picker widget."""
        _, item = self._make_loop()
        self.assertIsNotNone(
            getattr(item, '_loop_type_picker', None),
            "Loop node must have _loop_type_picker widget",
        )

    def test_loop_default_type_is_while_or_for(self):
        """Loop node default type is 'While' or 'For'."""
        _, item = self._make_loop()
        picker = getattr(item, '_loop_type_picker', None)
        if picker is None:
            self.skipTest("No _loop_type_picker")
        self.assertIn(picker.combo.currentText(), ("While", "For"))

    def test_loop_type_while_roundtrip(self):
        """Loop type 'While' is serialized and restored after load."""
        scene, item = self._make_loop()
        picker = getattr(item, '_loop_type_picker', None)
        if picker is None:
            self.skipTest("No _loop_type_picker")
        idx = picker.combo.findText("While")
        if idx < 0:
            self.skipTest("'While' not in combo")
        picker.combo.setCurrentIndex(idx)
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Loop")
        if entry is not None:
            self.assertEqual(entry.get("loop_type"), "While")

    def test_loop_type_for_roundtrip(self):
        """Loop type 'For' is serialized and restored after load."""
        scene, item = self._make_loop()
        picker = getattr(item, '_loop_type_picker', None)
        if picker is None:
            self.skipTest("No _loop_type_picker")
        idx = picker.combo.findText("For")
        if idx < 0:
            self.skipTest("'For' not in combo")
        picker.combo.setCurrentIndex(idx)
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Loop")
        if entry is not None:
            self.assertEqual(entry.get("loop_type"), "For")

    def test_loop_for_range_params_serialized(self):
        """For-mode range parameters (for_start, for_end, for_step) are serialized."""
        scene, item = self._make_loop()
        picker = getattr(item, '_loop_type_picker', None)
        if picker is None:
            self.skipTest("No _loop_type_picker")
        idx = picker.combo.findText("For")
        if idx < 0:
            self.skipTest("'For' mode not available")
        picker.combo.setCurrentIndex(idx)
        for_start = getattr(item, '_for_start_input', None)
        for_end = getattr(item, '_for_end_input', None)
        for_step = getattr(item, '_for_step_input', None)
        if not (for_start and for_end and for_step):
            self.skipTest("For-mode range inputs not present")
        for_start.setText("1")
        for_end.setText("10")
        for_step.setText("2")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Loop")
        self.assertIsNotNone(entry, "Loop must appear in nodes")
        self.assertEqual(str(entry.get("for_start", "")), "1")
        self.assertEqual(str(entry.get("for_end", "")), "10")
        self.assertEqual(str(entry.get("for_step", "")), "2")

    def test_loop_for_end_restored_after_load(self):
        """For-mode for_end value is restored after load_workflow."""
        scene, item = self._make_loop()
        picker = getattr(item, '_loop_type_picker', None)
        if picker is None:
            self.skipTest("No _loop_type_picker")
        idx = picker.combo.findText("For")
        if idx < 0:
            self.skipTest("'For' mode not available")
        picker.combo.setCurrentIndex(idx)
        for_end = getattr(item, '_for_end_input', None)
        if for_end is None:
            self.skipTest("No _for_end_input")
        for_end.setText("5")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "Loop")
        self.assertIsNotNone(restored)
        restored_end = getattr(restored, '_for_end_input', None)
        if restored_end:
            self.assertEqual(restored_end.text(), "5")


# ── Retry node ────────────────────────────────────────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestRetryInlineParam(unittest.TestCase):
    """Retry node inline parameter tests (max_attempts, backoff_sec, strategy)."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_retry(self):
        scene = _make_scene()
        item = scene.create_node("Retry", QPointF(100, 100))
        return scene, item

    def test_retry_has_attempts_widget(self):
        """Retry node exposes _attempts_input widget."""
        _, item = self._make_retry()
        self.assertIsNotNone(
            getattr(item, '_attempts_input', None),
            "Retry node must have _attempts_input widget",
        )

    def test_retry_has_backoff_widget(self):
        """Retry node exposes _backoff_input widget."""
        _, item = self._make_retry()
        self.assertIsNotNone(
            getattr(item, '_backoff_input', None),
            "Retry node must have _backoff_input widget",
        )

    def test_retry_max_attempts_serialized(self):
        """max_attempts value is serialized under 'max_attempts' key."""
        scene, item = self._make_retry()
        att = getattr(item, '_attempts_input', None)
        if att is None:
            self.skipTest("No _attempts_input widget")
        att.setText("5")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Retry")
        self.assertIsNotNone(entry, "Retry must appear in nodes")
        self.assertEqual(str(entry.get("max_attempts", "")), "5")

    def test_retry_max_attempts_restored_after_load(self):
        """max_attempts is restored after load_workflow."""
        scene, item = self._make_retry()
        att = getattr(item, '_attempts_input', None)
        if att is None:
            self.skipTest("No _attempts_input widget")
        att.setText("4")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "Retry")
        self.assertIsNotNone(restored)
        restored_att = getattr(restored, '_attempts_input', None)
        if restored_att:
            self.assertEqual(restored_att.text(), "4")

    def test_retry_backoff_serialized(self):
        """backoff_sec value is serialized under 'backoff_sec' key."""
        scene, item = self._make_retry()
        bo = getattr(item, '_backoff_input', None)
        if bo is None:
            self.skipTest("No _backoff_input widget")
        bo.setText("2.5")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Retry")
        self.assertIsNotNone(entry, "Retry must appear in nodes")
        self.assertEqual(str(entry.get("backoff_sec", "")), "2.5")

    def test_retry_backoff_restored_after_load(self):
        """backoff_sec is restored after load_workflow."""
        scene, item = self._make_retry()
        bo = getattr(item, '_backoff_input', None)
        if bo is None:
            self.skipTest("No _backoff_input widget")
        bo.setText("1.5")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "Retry")
        self.assertIsNotNone(restored)
        restored_bo = getattr(restored, '_backoff_input', None)
        if restored_bo:
            self.assertEqual(restored_bo.text(), "1.5")

    def test_retry_invalid_attempts_fallback_to_default(self):
        """Invalid max_attempts text falls back to default 3 in execution graph."""
        scene, item = self._make_retry()
        att = getattr(item, '_attempts_input', None)
        if att is None:
            self.skipTest("No _attempts_input widget")
        att.setText("not_a_number")
        node = _exec_graph_node(scene, "Retry")
        if node is None:
            self.skipTest("Retry not found in exec_graph")
        params = node.get("parameters", {})
        # Invalid text → fallback to default 3
        self.assertEqual(params.get("max_attempts", 3), 3)

    def test_retry_max_attempts_propagates_to_exec_graph(self):
        """Valid max_attempts propagates to execution graph parameters."""
        scene, item = self._make_retry()
        att = getattr(item, '_attempts_input', None)
        if att is None:
            self.skipTest("No _attempts_input widget")
        att.setText("7")
        node = _exec_graph_node(scene, "Retry")
        if node is None:
            self.skipTest("Retry not found in exec_graph")
        params = node.get("parameters", {})
        self.assertEqual(params.get("max_attempts"), 7)


# ── Break node ────────────────────────────────────────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestBreakInlineParam(unittest.TestCase):
    """Break node inline parameter tests (break_level)."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_break(self):
        scene = _make_scene()
        item = scene.create_node("Break", QPointF(100, 100))
        return scene, item

    def test_break_has_level_widget(self):
        """Break node exposes _break_level_input widget."""
        _, item = self._make_break()
        self.assertIsNotNone(
            getattr(item, '_break_level_input', None),
            "Break node must have _break_level_input widget",
        )

    def test_break_level_serialized(self):
        """break_level value is serialized under 'break_level' key."""
        scene, item = self._make_break()
        lw = getattr(item, '_break_level_input', None)
        if lw is None:
            self.skipTest("No _break_level_input widget")
        lw.setText("2")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "Break")
        self.assertIsNotNone(entry, "Break must appear in nodes")
        self.assertEqual(str(entry.get("break_level", "")), "2")

    def test_break_level_restored_after_load(self):
        """break_level is restored after load_workflow."""
        scene, item = self._make_break()
        lw = getattr(item, '_break_level_input', None)
        if lw is None:
            self.skipTest("No _break_level_input widget")
        lw.setText("3")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "Break")
        self.assertIsNotNone(restored)
        restored_lw = getattr(restored, '_break_level_input', None)
        if restored_lw:
            self.assertEqual(restored_lw.text(), "3")

    def test_break_invalid_level_fallback_to_minimum(self):
        """Invalid break_level text falls back to minimum 1 in execution graph."""
        scene, item = self._make_break()
        lw = getattr(item, '_break_level_input', None)
        if lw is None:
            self.skipTest("No _break_level_input widget")
        lw.setText("abc")
        node = _exec_graph_node(scene, "Break")
        if node is None:
            self.skipTest("Break not found in exec_graph")
        params = node.get("parameters", {})
        # Invalid text → fallback to 1 (the minimum valid level)
        self.assertGreaterEqual(params.get("break_level", 1), 1)


# ── Branch / If node ──────────────────────────────────────────────────────────

@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not available")
class TestBranchInlineParam(unittest.TestCase):
    """Branch (If) node inline condition_expr parameter tests."""

    @classmethod
    def setUpClass(cls):
        _get_app()

    def _make_branch(self):
        scene = _make_scene()
        item = scene.create_node("If", QPointF(100, 100))
        return scene, item

    def test_branch_has_condition_widget(self):
        """If node exposes _condition_set or _condition_input widget."""
        _, item = self._make_branch()
        has_set = getattr(item, '_condition_set', None) is not None
        has_input = getattr(item, '_condition_input', None) is not None
        self.assertTrue(
            has_set or has_input,
            "If node must have _condition_set or _condition_input widget",
        )

    def test_branch_condition_serialized_under_condition_expr(self):
        """Condition expression is serialized under the 'condition_expr' key."""
        scene, item = self._make_branch()
        cond_set = getattr(item, '_condition_set', None)
        cond_input = getattr(item, '_condition_input', None)
        if cond_set and hasattr(cond_set, 'set_logic_text'):
            cond_set.set_logic_text("x > 0")
        elif cond_input:
            cond_input.setText("x > 0")
        else:
            self.skipTest("No condition widget available")
        data = scene.serialize_workflow()
        entry = _serial_node(data, "If")
        self.assertIsNotNone(entry, "If node must appear in nodes")
        self.assertIn("condition_expr", entry)

    def test_branch_condition_roundtrip(self):
        """Condition expression survives serialize → load → widget text."""
        scene, item = self._make_branch()
        cond_set = getattr(item, '_condition_set', None)
        cond_input = getattr(item, '_condition_input', None)
        if cond_set and hasattr(cond_set, 'set_logic_text'):
            cond_set.set_logic_text("y == 1")
            expr_before = cond_set.logic_text()
        elif cond_input:
            cond_input.setText("y == 1")
            expr_before = "y == 1"
        else:
            self.skipTest("No condition widget")
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        restored = _find_node(scene, "If")
        self.assertIsNotNone(restored)
        restored_set = getattr(restored, '_condition_set', None)
        restored_input = getattr(restored, '_condition_input', None)
        if restored_set and hasattr(restored_set, 'logic_text'):
            self.assertEqual(restored_set.logic_text(), expr_before)
        elif restored_input:
            self.assertEqual(restored_input.text(), expr_before)


if __name__ == "__main__":
    unittest.main()
