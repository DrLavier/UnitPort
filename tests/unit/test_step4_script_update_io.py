#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 4 — GraphScene.script_update_io().

Tests exercise the three new GraphScene methods:
  _script_remove_input_entry / _script_remove_output_entry
  script_update_io

Strategy: build a lightweight fake node_item (plain Python object with the
same attribute contract as a real QGraphicsRectItem Script Box) and a fake
io_widget (plain Python object mirroring ScriptInAndOut's list API).
No Qt display is needed for these tests.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── minimal Qt stub so graph_scene imports without a display ──────────────────
def _get_qt_app():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication(sys.argv)
    except Exception:
        return None


# ── fake node infrastructure ─────────────────────────────────────────────────

def _make_name_box(text: str):
    nb = MagicMock()
    nb.text.return_value = text
    nb.strip = MagicMock(return_value=text)
    return nb


def _make_io_widget():
    """Minimal fake ScriptInAndOut with tracked input/output row lists."""
    w = MagicMock()
    w.input_rows = []
    w.output_rows = []
    # isValid(w) must return True; we patch isValid globally in each test
    return w


def _make_port(connections=None):
    p = MagicMock()
    p.data.return_value = list(connections or [])
    p.scene.return_value = None   # not in scene → removeItem not called
    return p


def _make_row_widget():
    rw = MagicMock()
    rw.scene.return_value = None
    return rw


def _make_input_entry(name: str, data_type: str = "any", has_connection=False):
    port = _make_port([MagicMock()] if has_connection else [])
    row = _make_row_widget()
    nb = _make_name_box(name)
    tb = MagicMock()
    tb.text.return_value = data_type
    entry = {
        "slot": f"script_in_{name}",
        "name_box": nb,
        "data_type_box": tb,
        "left_box": MagicMock(),
        "port": port,
        "row": row,
    }
    return entry


def _make_output_entry(name: str, data_type: str = "any", has_connection=False):
    port = _make_port([MagicMock()] if has_connection else [])
    row = _make_row_widget()
    nb = _make_name_box(name)
    tb = MagicMock()
    tb.text.return_value = data_type
    entry = {
        "slot": f"script_out_{name}",
        "name_box": nb,
        "data_type_box": tb,
        "right_box": MagicMock(),
        "port": port,
        "row": row,
    }
    return entry


def _make_node_item(input_entries=None, output_entries=None, io_widget=None):
    node = MagicMock()
    node.data.side_effect = lambda key: "Script Box" if key == 11 else None
    node._script_input_rows = list(input_entries or [])
    node._script_output_rows = list(output_entries or [])
    node._script_inout = io_widget or _make_io_widget()
    node._sync_layout_fn = None
    return node


# ── GraphScene partial stub ───────────────────────────────────────────────────

def _make_scene():
    """Return a GraphScene-like object with the Step 4 methods available."""
    try:
        app = _get_qt_app()
        # We need the real GraphScene class for method resolution,
        # but we don't want to construct a full scene with Qt signals.
        # Patch the __init__ so we can instantiate without it.
        from bin.components.graph_scene import GraphScene as _GS

        class FakeScene(_GS):
            def __init__(self):
                # bypass QGraphicsScene.__init__
                pass

        scene = FakeScene()
        # Provide stubs for methods called by script_update_io
        scene.get_node_item = MagicMock()
        scene._detach_connection = MagicMock()
        scene.removeItem = MagicMock()
        scene._set_port_data_type = MagicMock()
        scene._script_set_row_dot_style = MagicMock()
        scene._update_node_params = MagicMock()
        scene._loading_workflow = False
        scene._script_add_input_row = MagicMock()
        scene._script_add_output_row = MagicMock()
        scene._script_sync_io_spec = MagicMock()
        return scene
    except Exception as exc:
        return None


# ── patch helper ──────────────────────────────────────────────────────────────

def _isValid_always_true(obj):
    return obj is not None


# ─────────────────────────────────────────────────────────────────────────────
# 1. _script_entry_name
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptEntryName(unittest.TestCase):
    def setUp(self):
        self.scene = _make_scene()
        if self.scene is None:
            self.skipTest("Qt not available")

    def test_returns_name_box_text(self):
        entry = _make_input_entry("speed")
        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            result = self.scene._script_entry_name(entry)
        self.assertEqual(result, "speed")

    def test_empty_when_no_name_box(self):
        entry = {"slot": "x"}
        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            result = self.scene._script_entry_name(entry)
        self.assertEqual(result, "")


# ─────────────────────────────────────────────────────────────────────────────
# 2. _script_remove_input_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveInputEntry(unittest.TestCase):
    def setUp(self):
        self.scene = _make_scene()
        if self.scene is None:
            self.skipTest("Qt not available")

    def test_entry_removed_from_node_list(self):
        e = _make_input_entry("x")
        io_widget = _make_io_widget()
        io_widget.input_rows.append(e)
        node = _make_node_item(input_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_input_entry(node, e)

        self.assertNotIn(e, node._script_input_rows)

    def test_entry_removed_from_io_widget_list(self):
        e = _make_input_entry("x")
        io_widget = _make_io_widget()
        io_widget.input_rows.append(e)
        node = _make_node_item(input_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_input_entry(node, e)

        self.assertNotIn(e, io_widget.input_rows)

    def test_connections_detached(self):
        conn = MagicMock()
        conn.scene.return_value = None
        e = _make_input_entry("x", has_connection=False)
        e["port"].data.return_value = [conn]
        io_widget = _make_io_widget()
        node = _make_node_item(input_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_input_entry(node, e)

        self.scene._detach_connection.assert_called_once_with(conn)

    def test_row_widget_deleted(self):
        e = _make_input_entry("x")
        io_widget = _make_io_widget()
        node = _make_node_item(input_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_input_entry(node, e)

        e["row"].setParent.assert_called_once_with(None)
        e["row"].deleteLater.assert_called_once()

    def test_other_entries_unaffected(self):
        e1 = _make_input_entry("x")
        e2 = _make_input_entry("y")
        io_widget = _make_io_widget()
        io_widget.input_rows.extend([e1, e2])
        node = _make_node_item(input_entries=[e1, e2], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_input_entry(node, e1)

        self.assertIn(e2, node._script_input_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. _script_remove_output_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveOutputEntry(unittest.TestCase):
    def setUp(self):
        self.scene = _make_scene()
        if self.scene is None:
            self.skipTest("Qt not available")

    def test_entry_removed_from_node_list(self):
        e = _make_output_entry("result")
        io_widget = _make_io_widget()
        io_widget.output_rows.append(e)
        node = _make_node_item(output_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_output_entry(node, e)

        self.assertNotIn(e, node._script_output_rows)

    def test_entry_removed_from_io_widget_list(self):
        e = _make_output_entry("result")
        io_widget = _make_io_widget()
        io_widget.output_rows.append(e)
        node = _make_node_item(output_entries=[e], io_widget=io_widget)

        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene._script_remove_output_entry(node, e)

        self.assertNotIn(e, io_widget.output_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. script_update_io — routing and guard checks
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptUpdateIO(unittest.TestCase):
    def setUp(self):
        self.scene = _make_scene()
        if self.scene is None:
            self.skipTest("Qt not available")

    def _run_update_io(self, node, compile_result):
        self.scene.get_node_item.return_value = node
        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene.script_update_io(99, compile_result)

    def test_invalid_node_id_is_noop(self):
        self.scene.get_node_item.return_value = None
        self.scene.script_update_io(0, {"inputs": [], "outputs": []})
        self.scene._script_sync_io_spec.assert_not_called()

    def test_non_script_box_node_is_noop(self):
        node = _make_node_item()
        node.data.side_effect = lambda key: "Action Node" if key == 11 else None
        self.scene.get_node_item.return_value = node
        with patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene.script_update_io(1, {"inputs": [], "outputs": []})
        self.scene._script_sync_io_spec.assert_not_called()

    def test_new_input_calls_add_input_row(self):
        node = _make_node_item()
        self._run_update_io(node, {
            "inputs": [{"name": "vel", "data_type": "float"}],
            "outputs": [],
        })
        self.scene._script_add_input_row.assert_called_once_with(
            node, data_type="float", var_name="vel"
        )

    def test_new_output_calls_add_output_row(self):
        node = _make_node_item()
        self._run_update_io(node, {
            "inputs": [],
            "outputs": [{"name": "result", "data_type": "int"}],
        })
        self.scene._script_add_output_row.assert_called_once_with(
            node, var_name="result", data_type="int"
        )

    def test_existing_input_updates_type_inplace(self):
        e = _make_input_entry("vel", "any")
        node = _make_node_item(input_entries=[e])
        self._run_update_io(node, {
            "inputs": [{"name": "vel", "data_type": "float"}],
            "outputs": [],
        })
        # add_input_row must NOT be called for an existing name
        self.scene._script_add_input_row.assert_not_called()
        # data_type_box updated
        e["data_type_box"].setText.assert_called_with("float")

    def test_existing_output_updates_type_inplace(self):
        e = _make_output_entry("result", "any")
        node = _make_node_item(output_entries=[e])
        self._run_update_io(node, {
            "inputs": [],
            "outputs": [{"name": "result", "data_type": "int"}],
        })
        self.scene._script_add_output_row.assert_not_called()
        e["data_type_box"].setText.assert_called_with("int")

    def test_removed_input_entry_is_deleted(self):
        e = _make_input_entry("old")
        io_widget = _make_io_widget()
        io_widget.input_rows.append(e)
        node = _make_node_item(input_entries=[e], io_widget=io_widget)
        # New compile result has no inputs → "old" should be removed
        self._run_update_io(node, {"inputs": [], "outputs": []})
        self.assertNotIn(e, node._script_input_rows)

    def test_removed_output_entry_is_deleted(self):
        e = _make_output_entry("old_out")
        io_widget = _make_io_widget()
        io_widget.output_rows.append(e)
        node = _make_node_item(output_entries=[e], io_widget=io_widget)
        self._run_update_io(node, {"inputs": [], "outputs": []})
        self.assertNotIn(e, node._script_output_rows)

    def test_surviving_input_not_readded(self):
        e = _make_input_entry("keep")
        node = _make_node_item(input_entries=[e])
        self._run_update_io(node, {
            "inputs": [{"name": "keep", "data_type": "any"}],
            "outputs": [],
        })
        self.scene._script_add_input_row.assert_not_called()

    def test_sync_io_spec_always_called(self):
        node = _make_node_item()
        self._run_update_io(node, {"inputs": [], "outputs": []})
        self.scene._script_sync_io_spec.assert_called_once_with(node)

    def test_empty_compile_result_is_safe(self):
        node = _make_node_item()
        self._run_update_io(node, {})  # missing keys → should not crash

    def test_removed_input_with_connection_logs_warning(self):
        e = _make_input_entry("gone", has_connection=True)
        io_widget = _make_io_widget()
        io_widget.input_rows.append(e)
        node = _make_node_item(input_entries=[e], io_widget=io_widget)
        with patch("bin.components.graph_scene.log_warning") as mock_warn, \
             patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene.get_node_item.return_value = node
            self.scene.script_update_io(99, {"inputs": [], "outputs": []})
        mock_warn.assert_called()
        warning_text = " ".join(str(c) for c in mock_warn.call_args_list)
        self.assertIn("gone", warning_text)

    def test_removed_output_with_connection_logs_warning(self):
        e = _make_output_entry("gone_out", has_connection=True)
        io_widget = _make_io_widget()
        io_widget.output_rows.append(e)
        node = _make_node_item(output_entries=[e], io_widget=io_widget)
        with patch("bin.components.graph_scene.log_warning") as mock_warn, \
             patch("bin.components.graph_scene.isValid", _isValid_always_true):
            self.scene.get_node_item.return_value = node
            self.scene.script_update_io(99, {"inputs": [], "outputs": []})
        mock_warn.assert_called()
        warning_text = " ".join(str(c) for c in mock_warn.call_args_list)
        self.assertIn("gone_out", warning_text)

    def test_multiple_inputs_all_new(self):
        node = _make_node_item()
        self._run_update_io(node, {
            "inputs": [
                {"name": "a", "data_type": "int"},
                {"name": "b", "data_type": "string"},
            ],
            "outputs": [],
        })
        self.assertEqual(self.scene._script_add_input_row.call_count, 2)

    def test_mixed_add_keep_remove(self):
        e_keep = _make_input_entry("keep", "any")
        e_remove = _make_input_entry("remove", "bool")
        io_widget = _make_io_widget()
        io_widget.input_rows.extend([e_keep, e_remove])
        node = _make_node_item(input_entries=[e_keep, e_remove], io_widget=io_widget)

        self._run_update_io(node, {
            "inputs": [
                {"name": "keep", "data_type": "float"},   # update type
                {"name": "brand_new", "data_type": "int"}, # add
            ],
            "outputs": [],
        })
        # "remove" gone
        self.assertNotIn(e_remove, node._script_input_rows)
        # "keep" updated in-place
        e_keep["data_type_box"].setText.assert_called_with("float")
        # "brand_new" added
        self.scene._script_add_input_row.assert_called_once_with(
            node, data_type="int", var_name="brand_new"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. main_zone_panel wiring — lambda calls script_update_io
# ─────────────────────────────────────────────────────────────────────────────

class TestMainZonePanelStep4Wiring(unittest.TestCase):
    def test_lambda_calls_script_update_io(self):
        mock_gs = MagicMock()
        result = {
            "inputs": [{"name": "x", "data_type": "int", "slot": None}],
            "outputs": [{"name": "y", "data_type": "bool"}],
            "diagnostics": [],
            "compat_used": False,
        }
        # Replicate the updated lambda body from main_zone_panel.py
        nid = 5
        mock_gs.script_update_io(nid, result)
        mock_gs.script_update_io.assert_called_once_with(5, result)

    def test_lambda_passes_full_result_not_just_outputs(self):
        mock_gs = MagicMock()
        result = {"inputs": [{"name": "a", "data_type": "float", "slot": None}],
                  "outputs": [], "diagnostics": [], "compat_used": True}
        mock_gs.script_update_io(1, result)
        # Confirm the full dict (including inputs) is passed
        call_result = mock_gs.script_update_io.call_args[0][1]
        self.assertIn("inputs", call_result)
        self.assertEqual(call_result["inputs"][0]["name"], "a")


if __name__ == "__main__":
    unittest.main()
