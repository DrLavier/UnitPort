#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9 — Cross-cutting acceptance checks.

Verifies that the full Script Box Compiler Completion plan is fulfilled:
  - SCRIPT_DATA_TYPE_INDEX completeness (10 types, 0-9)
  - ScriptDiagCode uniqueness
  - PythonIndentEditor API surface (all required methods present)
  - PythonSyntaxHighlighter API surface
  - _PY_KEYWORDS and _PY_BUILTINS coverage
  - Zoom constants sanity
  - analyze_script() shape for valid / invalid / compat scripts
  - compile_requested signal carries dict payload
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
# 1. SCRIPT_DATA_TYPE_INDEX
# ─────────────────────────────────────────────────────────────────────────────

class TestDataTypeIndex(unittest.TestCase):
    def setUp(self):
        from bin.components.script_api_analyzer import SCRIPT_DATA_TYPE_INDEX
        self.mapping = SCRIPT_DATA_TYPE_INDEX

    def test_has_10_entries(self):
        self.assertEqual(len(self.mapping), 10)

    def test_indices_are_0_to_9(self):
        self.assertEqual(set(self.mapping.keys()), set(range(10)))

    def test_required_types_present(self):
        values = set(self.mapping.values())
        for t in ("any", "bool", "int", "float", "string",
                  "list", "dict", "number", "function", "protocol"):
            self.assertIn(t, values, f"Type '{t}' missing from mapping")

    def test_all_values_are_strings(self):
        for k, v in self.mapping.items():
            self.assertIsInstance(v, str, f"Index {k} maps to non-string {v!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ScriptDiagCode uniqueness
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptDiagCode(unittest.TestCase):
    def setUp(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        self.cls = ScriptDiagCode

    def _all_codes(self):
        return [
            v for k, v in vars(self.cls).items()
            if not k.startswith("_") and isinstance(v, str)
        ]

    def test_all_codes_unique(self):
        codes = self._all_codes()
        self.assertEqual(len(codes), len(set(codes)), "Duplicate diag codes found")

    def test_required_codes_exist(self):
        required = (
            "SYNTAX_ERROR", "NO_RUN_FUNCTION", "INVALID_SETOUTPUT_ARGS",
            "INVALID_GETINPUT_ARGS", "INVALID_DATA_TYPE_INDEX",
            "GETINPUT_MISSING_NAME", "GETINPUT_EMPTY_NAME",
            "DUPLICATE_INPUT_NAME", "COMPAT_RETURN_USED",
        )
        for attr in required:
            self.assertTrue(hasattr(self.cls, attr), f"ScriptDiagCode.{attr} missing")

    def test_codes_follow_dot_namespace(self):
        for code in self._all_codes():
            self.assertIn(".", code, f"Code {code!r} has no namespace dot")


# ─────────────────────────────────────────────────────────────────────────────
# 3. analyze_script() result shape
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeScriptShape(unittest.TestCase):

    _VALID_API_SCRIPT = (
        "def run():\n"
        "    val = getInput('x', 2)\n"
        "    setOutput(val, 2)\n"
    )
    _COMPAT_SCRIPT = (
        "def run():\n"
        "    result = 42\n"
        "    return result\n"
    )
    _SYNTAX_ERROR_SCRIPT = "def run(\n    pass\n"
    _NO_RUN_SCRIPT = "def helper(): pass\n"

    def setUp(self):
        from bin.components.script_api_analyzer import analyze_script
        self.analyze = analyze_script

    def test_valid_api_result_keys(self):
        r = self.analyze(self._VALID_API_SCRIPT)
        for key in ("inputs", "outputs", "diagnostics", "compat_used"):
            self.assertIn(key, r, f"Key '{key}' missing from analyze_script result")

    def test_valid_api_compat_false(self):
        r = self.analyze(self._VALID_API_SCRIPT)
        self.assertFalse(r["compat_used"])

    def test_valid_api_has_one_output(self):
        r = self.analyze(self._VALID_API_SCRIPT)
        self.assertEqual(len(r["outputs"]), 1)

    def test_valid_api_has_one_input(self):
        r = self.analyze(self._VALID_API_SCRIPT)
        self.assertEqual(len(r["inputs"]), 1)

    def test_compat_script_sets_compat_used(self):
        r = self.analyze(self._COMPAT_SCRIPT)
        self.assertTrue(r["compat_used"])

    def test_compat_script_no_outputs_from_analyzer(self):
        # compat mode: analyzer found no setOutput → outputs == []
        r = self.analyze(self._COMPAT_SCRIPT)
        self.assertEqual(r["outputs"], [])

    def test_syntax_error_has_diag(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        r = self.analyze(self._SYNTAX_ERROR_SCRIPT)
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.SYNTAX_ERROR, codes)

    def test_no_run_function_has_diag(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        r = self.analyze(self._NO_RUN_SCRIPT)
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.NO_RUN_FUNCTION, codes)

    def test_diagnostics_are_dicts_with_required_keys(self):
        r = self.analyze(self._SYNTAX_ERROR_SCRIPT)
        for d in r["diagnostics"]:
            for k in ("code", "level", "message", "lineno"):
                self.assertIn(k, d, f"Diagnostic missing key '{k}'")

    def test_invalid_type_index_emits_diag(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        script = "def run():\n    setOutput(x, 999)\n"
        r = self.analyze(script)
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.INVALID_DATA_TYPE_INDEX, codes)

    def test_empty_string_returns_no_run_diag(self):
        from bin.components.script_api_analyzer import ScriptDiagCode
        r = self.analyze("")
        codes = [d["code"] for d in r["diagnostics"]]
        self.assertIn(ScriptDiagCode.NO_RUN_FUNCTION, codes)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PythonIndentEditor API surface
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonIndentEditorAPISurface(unittest.TestCase):
    def setUp(self):
        from bin.components.script_editor import PythonIndentEditor
        self.cls = PythonIndentEditor

    def test_has_highlight_lines(self):
        self.assertTrue(callable(getattr(self.cls, "highlight_lines", None)))

    def test_has_set_stale_selections(self):
        self.assertTrue(callable(getattr(self.cls, "set_stale_selections", None)))

    def test_has_apply_extra_selections(self):
        self.assertTrue(callable(getattr(self.cls, "_apply_extra_selections", None)))

    def test_has_line_number_area_width(self):
        self.assertTrue(callable(getattr(self.cls, "lineNumberAreaWidth", None)))

    def test_has_wheel_event(self):
        self.assertTrue(callable(getattr(self.cls, "wheelEvent", None)))

    def test_has_adjust_font_size(self):
        self.assertTrue(callable(getattr(self.cls, "_adjust_font_size", None)))

    def test_has_zoom_constants(self):
        self.assertIsInstance(self.cls._ZOOM_MIN, int)
        self.assertIsInstance(self.cls._ZOOM_MAX, int)

    def test_has_highlighter_in_instance(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        from bin.components.script_editor import PythonIndentEditor
        e = PythonIndentEditor()
        self.assertTrue(hasattr(e, "_highlighter"))

    def test_has_line_number_area_in_instance(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        from bin.components.script_editor import PythonIndentEditor
        e = PythonIndentEditor()
        self.assertTrue(hasattr(e, "_line_number_area"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. PythonSyntaxHighlighter API surface
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntaxHighlighterAPISurface(unittest.TestCase):
    def setUp(self):
        from bin.components.script_editor import PythonSyntaxHighlighter
        self.cls = PythonSyntaxHighlighter

    def test_has_build_rules(self):
        self.assertTrue(callable(getattr(self.cls, "_build_rules", None)))

    def test_has_highlight_block(self):
        self.assertTrue(callable(getattr(self.cls, "highlightBlock", None)))

    def test_has_make_fmt(self):
        self.assertTrue(callable(getattr(self.cls, "_make_fmt", None)))

    def test_state_constants_defined(self):
        self.assertIn("_STATE_NORMAL",   vars(self.cls))
        self.assertIn("_STATE_TRIPLE_D", vars(self.cls))
        self.assertIn("_STATE_TRIPLE_S", vars(self.cls))


# ─────────────────────────────────────────────────────────────────────────────
# 6. _PY_KEYWORDS / _PY_BUILTINS coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestPyTokenSets(unittest.TestCase):
    def setUp(self):
        from bin.components.script_editor import _PY_KEYWORDS, _PY_BUILTINS
        self._kw = set(_PY_KEYWORDS.split("|"))
        self._bi = set(_PY_BUILTINS.split("|"))

    def test_keywords_include_control_flow(self):
        for kw in ("if", "else", "elif", "for", "while", "try", "except",
                   "finally", "with", "return", "import", "from", "class", "def"):
            self.assertIn(kw, self._kw, f"keyword '{kw}' missing")

    def test_keywords_include_literals(self):
        for kw in ("True", "False", "None"):
            self.assertIn(kw, self._kw, f"literal '{kw}' missing")

    def test_builtins_include_common(self):
        for bi in ("print", "len", "range", "int", "str", "list", "dict",
                   "isinstance", "type", "super", "property"):
            self.assertIn(bi, self._bi, f"builtin '{bi}' missing")

    def test_no_overlap_between_keywords_and_builtins(self):
        # Keywords and builtins should be disjoint sets
        overlap = self._kw & self._bi
        self.assertEqual(overlap, set(), f"Overlap: {overlap}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ScriptEditor.compile_requested signal carries dict
# ─────────────────────────────────────────────────────────────────────────────

class TestCompileRequestedSignal(unittest.TestCase):
    def setUp(self):
        app = _get_app()
        if app is None:
            self.skipTest("Qt not available")
        try:
            from bin.components.script_editor import ScriptEditor
            self.editor = ScriptEditor(node_id=99, script_name="AcceptanceTest")
        except Exception:
            self.skipTest("ScriptEditor not available")

    def test_signal_second_arg_is_dict(self):
        """compile_requested must emit (int, dict) — not (int, list)."""
        received = []
        self.editor.compile_requested.connect(
            lambda nid, result: received.append((nid, result))
        )
        # Inject a minimal valid script and trigger compile
        self.editor._editor.setPlainText(
            "def run():\n"
            "    setOutput(42, 0)\n"
        )
        self.editor.save_and_compile()
        self.assertEqual(len(received), 1)
        nid, result = received[0]
        self.assertIsInstance(result, dict, "compile_requested second arg must be dict")

    def test_signal_result_has_required_keys(self):
        received = []
        self.editor.compile_requested.connect(
            lambda nid, result: received.append(result)
        )
        self.editor._editor.setPlainText(
            "def run():\n"
            "    setOutput(42, 0)\n"
        )
        self.editor.save_and_compile()
        self.assertEqual(len(received), 1)
        for key in ("inputs", "outputs", "diagnostics", "compat_used"):
            self.assertIn(key, received[0])

    def test_compat_mode_signal_still_dict(self):
        received = []
        self.editor.compile_requested.connect(
            lambda nid, result: received.append(result)
        )
        self.editor._editor.setPlainText(
            "def run():\n"
            "    return 42\n"
        )
        self.editor.save_and_compile()
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], dict)
        self.assertTrue(received[0]["compat_used"])


if __name__ == "__main__":
    unittest.main()
