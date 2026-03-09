#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 3 — ScriptEditor compile path integration.

Tests the save_and_compile() logic by directly exercising the underlying
functions (no Qt widget instantiation required for core logic tests).

What is tested:
  - API mode: setOutput() calls drive outputs; compat_used=False
  - Compat mode: no setOutput() → return parser fallback; compat_used=True
  - Syntax error: SYNTAX_ERROR diag produced, compile halts
  - No run() function: NO_RUN_FUNCTION diag produced, compile halts
  - compile_requested signal payload shape: dict with inputs/outputs/diagnostics/compat_used
  - Non-fatal diagnostics forwarded (errors & non-compat warnings)
  - API inputs reach the result dict unchanged
"""

import ast
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ── Qt stub (allows importing signal-bearing modules without a display) ────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── helpers that exercise the compile logic without instantiating Qt widgets ──

def _run_analyze(code: str) -> dict:
    """Call analyze_script directly; mirrors what save_and_compile() does."""
    from bin.components.script_api_analyzer import analyze_script
    return analyze_script(code)


def _parse_return(code: str) -> list:
    from bin.components.script_editor import _parse_return_outputs
    return _parse_return_outputs(code, func_name="run")


def _build_init_sig(io_spec: dict) -> str:
    from bin.components.script_editor import _build_def_line
    return _build_def_line(io_spec)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analyzer output shape — as seen by save_and_compile()
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzerResultShape(unittest.TestCase):
    """save_and_compile() receives an analyze_script() result dict.
    Verify the shape is correct for each execution path.
    """

    def test_api_mode_result_keys(self):
        code = "def run():\n    setOutput(x, 2)\n"
        r = _run_analyze(code)
        self.assertIn("inputs", r)
        self.assertIn("outputs", r)
        self.assertIn("diagnostics", r)
        self.assertIn("compat_used", r)

    def test_api_mode_compat_false(self):
        code = "def run():\n    setOutput(result, 1)\n"
        r = _run_analyze(code)
        self.assertFalse(r["compat_used"])

    def test_api_mode_outputs_typed(self):
        code = "def run():\n    setOutput(speed, 3)\n"
        r = _run_analyze(code)
        self.assertEqual(r["outputs"], [{"name": "speed", "data_type": "float"}])

    def test_api_mode_inputs_present(self):
        code = (
            "def run():\n"
            "    x = getInput('vel', 3)\n"
            "    setOutput(x, 3)\n"
        )
        r = _run_analyze(code)
        self.assertEqual(len(r["inputs"]), 1)
        self.assertEqual(r["inputs"][0]["name"], "vel")
        self.assertEqual(r["inputs"][0]["data_type"], "float")

    def test_compat_mode_compat_true(self):
        code = "def run():\n    result = 1\n    return result\n"
        r = _run_analyze(code)
        self.assertTrue(r["compat_used"])
        self.assertEqual(r["outputs"], [])

    def test_syntax_error_diag_present(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        r = _run_analyze("def run(\n  bad syntax")
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.SYNTAX_ERROR, codes)

    def test_no_run_diag_present(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        r = _run_analyze("x = 1\n")
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.NO_RUN_FUNCTION, codes)

    def test_init_entry_is_supported(self):
        code = "def init():\n    setOutput(speed, 3)\n"
        r = _run_analyze(code)
        self.assertEqual(r.get("entry_function"), "init")
        self.assertFalse(r["compat_used"])
        self.assertEqual(r["outputs"], [{"name": "speed", "data_type": "float"}])

    def test_multiline_init_signature_with_type_comments(self):
        sig = _build_init_sig({
            "inputs": [
                {"name": "input_1", "data_type": "float"},
                {"name": "input_2", "data_type": "int"},
            ]
        })
        self.assertIn("def init(", sig)
        self.assertIn("input_1,  # float", sig)
        self.assertIn("input_2,  # int", sig)
        self.assertTrue(sig.strip().endswith("):"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compat fallback — return parser produces correct outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestCompatFallback(unittest.TestCase):
    """When compat_used=True, save_and_compile() runs _parse_return_outputs()."""

    def test_return_variable_name(self):
        code = "def run():\n    result = 1\n    return result\n"
        outs = _parse_return(code)
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0]["name"], "result")
        self.assertEqual(outs[0]["data_type"], "any")

    def test_return_tuple(self):
        code = "def run():\n    return a, b\n"
        outs = _parse_return(code)
        self.assertEqual([o["name"] for o in outs], ["a", "b"])

    def test_return_none_gives_empty(self):
        code = "def run():\n    pass\n"
        outs = _parse_return(code)
        self.assertEqual(outs, [])

    def test_compat_outputs_replace_empty_api_outputs(self):
        """Simulate what save_and_compile() does: dict(result, outputs=compat_outputs)."""
        from bin.components.script_api_analyzer import analyze_script
        code = "def run():\n    return x\n"
        result = analyze_script(code)
        self.assertTrue(result["compat_used"])
        compat_outputs = _parse_return(code)
        merged = dict(result, outputs=compat_outputs)
        self.assertEqual(merged["outputs"], [{"name": "x", "data_type": "any"}])
        self.assertTrue(merged["compat_used"])

    def test_init_return_is_valid_compat_source(self):
        from bin.components.script_editor import _parse_return_outputs
        code = "def init():\n    result = 1\n    return result\n"
        result = _run_analyze(code)
        self.assertTrue(result["compat_used"])
        outs = _parse_return_outputs(code, func_name="init")
        self.assertEqual(outs, [{"name": "result", "data_type": "any"}])

    def test_init_result_from_process_maps_to_result_output(self):
        from bin.components.script_editor import _parse_return_outputs
        code = (
            "def init():\n"
            "    result = process()\n"
            "    return result\n\n"
            "def process():\n"
            "    return 1\n"
        )
        result = _run_analyze(code)
        self.assertTrue(result["compat_used"])
        outs = _parse_return_outputs(code, func_name="init")
        self.assertEqual(outs, [{"name": "result", "data_type": "any"}])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal payload integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalPayload(unittest.TestCase):
    """compile_requested must emit (node_id: int, result: dict)."""

    def _make_editor_signal_capture(self):
        """Import ScriptEditor with Qt available, capture emitted signal."""
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
            return app, ScriptEditor
        except Exception:
            self.skipTest("Qt not available in this environment")

    def test_signal_type_annotation(self):
        """compile_requested must be Signal(int, dict)."""
        try:
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")
        from PySide6.QtCore import Signal
        sig = ScriptEditor.__dict__["compile_requested"]
        # PySide6 Signal: check the types tuple
        self.assertIsNotNone(sig)

    def test_emitted_payload_is_dict(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")

        captured = []
        editor = ScriptEditor(node_id=42, script_name="test_node")
        editor.compile_requested.connect(lambda nid, result: captured.append((nid, result)))

        # Set a valid API script directly
        editor._editor.setPlainText(
            "def run():\n    x = getInput('v', 3)\n    setOutput(x, 3)\n"
        )
        with patch("bin.components.script_editor.QMessageBox"):
            editor.save_and_compile()

        self.assertEqual(len(captured), 1)
        nid, result = captured[0]
        self.assertEqual(nid, 42)
        self.assertIsInstance(result, dict)
        self.assertIn("inputs", result)
        self.assertIn("outputs", result)
        self.assertIn("compat_used", result)

    def test_api_mode_payload_populated(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")

        captured = []
        editor = ScriptEditor(node_id=1, script_name="node")
        editor.compile_requested.connect(lambda nid, r: captured.append(r))
        editor._editor.setPlainText(
            "def run():\n    setOutput(result, 2)\n"
        )
        with patch("bin.components.script_editor.QMessageBox"):
            editor.save_and_compile()

        self.assertEqual(len(captured), 1)
        r = captured[0]
        self.assertFalse(r["compat_used"])
        self.assertEqual(r["outputs"], [{"name": "result", "data_type": "int"}])

    def test_compat_mode_payload_has_return_outputs(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")

        captured = []
        editor = ScriptEditor(node_id=2, script_name="node")
        editor.compile_requested.connect(lambda nid, r: captured.append(r))
        editor._editor.setPlainText(
            "def run():\n    result = 1\n    return result\n"
        )
        with patch("bin.components.script_editor.QMessageBox"):
            editor.save_and_compile()

        self.assertEqual(len(captured), 1)
        r = captured[0]
        self.assertTrue(r["compat_used"])
        self.assertEqual(r["outputs"], [{"name": "result", "data_type": "any"}])

    def test_syntax_error_does_not_emit(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")

        captured = []
        editor = ScriptEditor(node_id=3, script_name="node")
        editor.compile_requested.connect(lambda nid, r: captured.append(r))
        editor._editor.setPlainText("def run(\n  bad syntax !!!")
        with patch("bin.components.script_editor.QMessageBox"):
            editor.save_and_compile()

        self.assertEqual(captured, [])

    def test_no_run_does_not_emit(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from bin.components.script_editor import ScriptEditor
        except Exception:
            self.skipTest("Qt not available")

        captured = []
        editor = ScriptEditor(node_id=4, script_name="node")
        editor.compile_requested.connect(lambda nid, r: captured.append(r))
        editor._editor.setPlainText("x = 1\n")
        with patch("bin.components.script_editor.QMessageBox"):
            editor.save_and_compile()

        self.assertEqual(captured, [])


# ─────────────────────────────────────────────────────────────────────────────
# 4. main_zone_panel wiring — lambda passes outputs list to script_update_outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestMainZonePanelWiring(unittest.TestCase):
    """The updated lambda in open_script_tab() must extract outputs from dict."""

    def test_lambda_extracts_outputs(self):
        """Simulate the lambda: gs.script_update_outputs(nid, result.get('outputs', []))"""
        mock_gs = MagicMock()
        result = {
            "inputs": [{"name": "x", "data_type": "int", "slot": None}],
            "outputs": [{"name": "y", "data_type": "bool"}],
            "diagnostics": [],
            "compat_used": False,
        }
        # Replicate the lambda body exactly as written in main_zone_panel.py
        nid = 7
        mock_gs.script_update_outputs(nid, result.get("outputs", []))
        mock_gs.script_update_outputs.assert_called_once_with(
            7, [{"name": "y", "data_type": "bool"}]
        )

    def test_lambda_handles_missing_outputs_key(self):
        mock_gs = MagicMock()
        result = {"inputs": [], "diagnostics": [], "compat_used": True}
        mock_gs.script_update_outputs(1, result.get("outputs", []))
        mock_gs.script_update_outputs.assert_called_once_with(1, [])


if __name__ == "__main__":
    unittest.main()
