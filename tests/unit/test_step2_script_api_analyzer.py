#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for bin/components/script_api_analyzer.py — Step 2.

Coverage:
  - Valid setOutput / getInput patterns
  - Invalid signatures (wrong arg count, non-literal index, out-of-range index)
  - getInput name validation (missing, non-literal, empty, duplicate)
  - Compat mode (no setOutput → compat_used=True + COMPAT_RETURN_USED diag)
  - Syntax error / no run() function
  - Nested def / lambda isolation (calls inside are NOT collected)
  - Name extraction from setOutput data expression
  - Mixed inputs + outputs in same script
  - Type index boundary values
"""

import pytest

from bin.components.script_api_analyzer import (
    SCRIPT_DATA_TYPE_INDEX,
    ScriptDiagCode,
    analyze_script,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _codes(result):
    return [d["code"] for d in result["diagnostics"]]


def _errors(result):
    return [d for d in result["diagnostics"] if d["level"] == "error"]


def _warnings(result):
    return [d for d in result["diagnostics"] if d["level"] == "warning"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Syntax error
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntaxError:
    def test_syntax_error_returns_early(self):
        result = analyze_script("def run(\n  broken syntax !!!")
        assert result["inputs"] == []
        assert result["outputs"] == []
        assert result["compat_used"] is False
        assert ScriptDiagCode.SYNTAX_ERROR in _codes(result)

    def test_syntax_error_has_lineno(self):
        result = analyze_script("x = (")
        diag = next(d for d in result["diagnostics"] if d["code"] == ScriptDiagCode.SYNTAX_ERROR)
        assert diag["lineno"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. No run() function
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRunFunction:
    def test_no_run_returns_error(self):
        result = analyze_script("def process():\n    pass\n")
        assert ScriptDiagCode.NO_RUN_FUNCTION not in _codes(result)
        assert result["entry_function"] == "process"
        assert result["compat_used"] is True

    def test_empty_script(self):
        result = analyze_script("")
        assert ScriptDiagCode.NO_RUN_FUNCTION in _codes(result)

    def test_run_inside_class_not_found(self):
        code = "class Foo:\n    def run(self):\n        setOutput(x, 0)\n"
        # Only top-level defs are considered entry functions.
        result = analyze_script(code)
        assert ScriptDiagCode.NO_RUN_FUNCTION in _codes(result)


class TestEntrySelection:
    def test_init_is_preferred_over_run(self):
        code = (
            "def run():\n"
            "    setOutput(from_run, 2)\n\n"
            "def init():\n"
            "    setOutput(from_init, 1)\n"
        )
        result = analyze_script(code)
        assert result["entry_function"] == "init"
        assert result["outputs"] == [{"name": "from_init", "data_type": "bool"}]

    def test_single_custom_function_is_entry(self):
        code = "def process_data():\n    setOutput(x, 2)\n"
        result = analyze_script(code)
        assert result["entry_function"] == "process_data"
        assert result["compat_used"] is False
        assert result["outputs"] == [{"name": "x", "data_type": "int"}]

    def test_multiple_custom_functions_without_init_or_run_is_error(self):
        code = (
            "def alpha():\n"
            "    return 1\n\n"
            "def beta():\n"
            "    return 2\n"
        )
        result = analyze_script(code)
        assert ScriptDiagCode.NO_RUN_FUNCTION in _codes(result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compat mode
# ─────────────────────────────────────────────────────────────────────────────

class TestCompatMode:
    def test_no_setoutput_triggers_compat(self):
        code = "def run():\n    result = 42\n    return result\n"
        result = analyze_script(code)
        assert result["compat_used"] is True
        assert ScriptDiagCode.COMPAT_RETURN_USED in _codes(result)

    def test_compat_outputs_empty(self):
        code = "def run():\n    return 1\n"
        result = analyze_script(code)
        assert result["outputs"] == []

    def test_setoutput_suppresses_compat(self):
        code = "def run():\n    setOutput(x, 0)\n    return x\n"
        result = analyze_script(code)
        assert result["compat_used"] is False
        assert ScriptDiagCode.COMPAT_RETURN_USED not in _codes(result)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Valid setOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestSetOutputValid:
    def test_single_setoutput_variable(self):
        code = "def run():\n    result = 1\n    setOutput(result, 2)\n"
        r = analyze_script(code)
        assert len(r["outputs"]) == 1
        assert r["outputs"][0]["name"] == "result"
        assert r["outputs"][0]["data_type"] == "int"
        assert _errors(r) == []

    def test_multiple_setoutput_preserves_order(self):
        code = (
            "def run():\n"
            "    a = 1\n"
            "    setOutput(a, 1)\n"   # bool
            "    b = 'x'\n"
            "    setOutput(b, 4)\n"   # string
            "    setOutput(c, 3)\n"   # float
        )
        r = analyze_script(code)
        assert len(r["outputs"]) == 3
        assert r["outputs"][0] == {"name": "a", "data_type": "bool"}
        assert r["outputs"][1] == {"name": "b", "data_type": "string"}
        assert r["outputs"][2] == {"name": "c", "data_type": "float"}

    def test_all_valid_type_indices(self):
        for idx, type_name in SCRIPT_DATA_TYPE_INDEX.items():
            code = f"def run():\n    setOutput(x, {idx})\n"
            r = analyze_script(code)
            assert len(r["outputs"]) == 1, f"index {idx} failed"
            assert r["outputs"][0]["data_type"] == type_name

    def test_attribute_expr_uses_attr_name(self):
        code = "def run():\n    setOutput(self.speed, 3)\n"
        r = analyze_script(code)
        assert r["outputs"][0]["name"] == "speed"

    def test_complex_expr_uses_fallback_name(self):
        code = "def run():\n    setOutput(a + b, 2)\n"
        r = analyze_script(code)
        assert r["outputs"][0]["name"] == "out_0"

    def test_second_complex_expr_uses_index_1(self):
        code = "def run():\n    setOutput(x, 0)\n    setOutput(a + b, 2)\n"
        r = analyze_script(code)
        assert r["outputs"][1]["name"] == "out_1"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Invalid setOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestSetOutputInvalid:
    def test_too_few_args(self):
        code = "def run():\n    setOutput(x)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_SETOUTPUT_ARGS in _codes(r)
        assert r["outputs"] == []

    def test_too_many_args(self):
        code = "def run():\n    setOutput(x, 1, 2)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_SETOUTPUT_ARGS in _codes(r)

    def test_keyword_arg_rejected(self):
        code = "def run():\n    setOutput(x, data_type_index=1)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_SETOUTPUT_ARGS in _codes(r)

    def test_non_literal_index(self):
        code = "def run():\n    setOutput(x, idx)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)
        assert r["outputs"] == []

    def test_string_literal_as_index(self):
        code = "def run():\n    setOutput(x, 'int')\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_out_of_range_index_low(self):
        code = "def run():\n    setOutput(x, -1)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_out_of_range_index_high(self):
        code = "def run():\n    setOutput(x, 99)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_invalid_call_does_not_block_subsequent_valid(self):
        code = (
            "def run():\n"
            "    setOutput(x)\n"       # invalid
            "    setOutput(y, 1)\n"    # valid
        )
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_SETOUTPUT_ARGS in _codes(r)
        assert len(r["outputs"]) == 1
        assert r["outputs"][0]["name"] == "y"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Valid getInput
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInputValid:
    def test_positional_name_only(self):
        code = "def run():\n    x = getInput('speed')\n"
        r = analyze_script(code)
        assert len(r["inputs"]) == 1
        assert r["inputs"][0]["name"] == "speed"
        assert r["inputs"][0]["data_type"] == "any"
        assert r["inputs"][0]["slot"] is None

    def test_positional_name_and_index(self):
        code = "def run():\n    x = getInput('count', 2)\n"
        r = analyze_script(code)
        assert r["inputs"][0]["data_type"] == "int"

    def test_keyword_name(self):
        code = "def run():\n    x = getInput(name='label', data_type_index=4)\n"
        r = analyze_script(code)
        assert r["inputs"][0]["name"] == "label"
        assert r["inputs"][0]["data_type"] == "string"

    def test_multiple_inputs_order_preserved(self):
        code = (
            "def run():\n"
            "    a = getInput('alpha', 1)\n"
            "    b = getInput('beta', 3)\n"
            "    c = getInput('gamma')\n"
        )
        r = analyze_script(code)
        assert [i["name"] for i in r["inputs"]] == ["alpha", "beta", "gamma"]
        assert [i["data_type"] for i in r["inputs"]] == ["bool", "float", "any"]

    def test_slot_is_none(self):
        code = "def run():\n    getInput('x', 0)\n"
        r = analyze_script(code)
        assert r["inputs"][0]["slot"] is None

    def test_name_whitespace_stripped(self):
        code = "def run():\n    getInput('  my_input  ', 1)\n"
        r = analyze_script(code)
        assert r["inputs"][0]["name"] == "my_input"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Invalid getInput
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInputInvalid:
    def test_no_args(self):
        code = "def run():\n    getInput()\n"
        r = analyze_script(code)
        assert ScriptDiagCode.GETINPUT_MISSING_NAME in _codes(r)
        assert r["inputs"] == []

    def test_variable_as_name(self):
        code = "def run():\n    getInput(my_var)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.GETINPUT_MISSING_NAME in _codes(r)

    def test_expression_as_name(self):
        code = "def run():\n    getInput('a' + 'b')\n"
        r = analyze_script(code)
        assert ScriptDiagCode.GETINPUT_MISSING_NAME in _codes(r)

    def test_empty_name(self):
        code = "def run():\n    getInput('')\n"
        r = analyze_script(code)
        assert ScriptDiagCode.GETINPUT_EMPTY_NAME in _codes(r)

    def test_whitespace_only_name(self):
        code = "def run():\n    getInput('   ')\n"
        r = analyze_script(code)
        assert ScriptDiagCode.GETINPUT_EMPTY_NAME in _codes(r)

    def test_duplicate_name_warning(self):
        code = (
            "def run():\n"
            "    getInput('speed', 3)\n"
            "    getInput('speed', 2)\n"   # duplicate
        )
        r = analyze_script(code)
        assert ScriptDiagCode.DUPLICATE_INPUT_NAME in _codes(r)
        # Only first occurrence added
        assert len(r["inputs"]) == 1
        assert r["inputs"][0]["data_type"] == "float"

    def test_duplicate_is_warning_not_error(self):
        code = (
            "def run():\n"
            "    getInput('x')\n"
            "    getInput('x')\n"
        )
        r = analyze_script(code)
        dup = next(d for d in r["diagnostics"] if d["code"] == ScriptDiagCode.DUPLICATE_INPUT_NAME)
        assert dup["level"] == "warning"

    def test_non_literal_type_index(self):
        code = "def run():\n    getInput('foo', idx)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_out_of_range_type_index(self):
        code = "def run():\n    getInput('foo', 100)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_invalid_getinput_does_not_block_subsequent(self):
        code = (
            "def run():\n"
            "    getInput()\n"          # invalid
            "    getInput('ok', 1)\n"   # valid
        )
        r = analyze_script(code)
        assert len(r["inputs"]) == 1
        assert r["inputs"][0]["name"] == "ok"

    def test_too_many_positional_args(self):
        code = "def run():\n    getInput('x', 1, 2)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_GETINPUT_ARGS in _codes(r)
        assert r["inputs"] == []

    def test_unknown_kwarg_rejected(self):
        code = "def run():\n    getInput('x', extra=99)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_GETINPUT_ARGS in _codes(r)
        assert r["inputs"] == []

    def test_star_kwargs_rejected(self):
        code = "def run():\n    opts = {}\n    getInput('x', **opts)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_GETINPUT_ARGS in _codes(r)

    def test_invalid_getinput_args_is_error_not_warning(self):
        code = "def run():\n    getInput('x', 1, 2)\n"
        r = analyze_script(code)
        err = next(d for d in r["diagnostics"] if d["code"] == ScriptDiagCode.INVALID_GETINPUT_ARGS)
        assert err["level"] == "error"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Mixed inputs and outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestMixed:
    def test_inputs_and_outputs_together(self):
        code = (
            "def run():\n"
            "    speed = getInput('speed', 3)\n"
            "    mode  = getInput('mode', 4)\n"
            "    result = speed * 2\n"
            "    setOutput(result, 3)\n"
            "    setOutput(mode, 4)\n"
        )
        r = analyze_script(code)
        assert len(r["inputs"]) == 2
        assert len(r["outputs"]) == 2
        assert r["inputs"][0]["name"] == "speed"
        assert r["inputs"][1]["name"] == "mode"
        assert r["outputs"][0]["data_type"] == "float"
        assert r["outputs"][1]["data_type"] == "string"
        assert _errors(r) == []
        assert r["compat_used"] is False

    def test_no_diagnostics_on_clean_script(self):
        code = (
            "def run():\n"
            "    x = getInput('x', 2)\n"
            "    setOutput(x, 2)\n"
        )
        r = analyze_script(code)
        assert r["diagnostics"] == []
        assert r["compat_used"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. Nested def / lambda isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestNestedIsolation:
    def test_setoutput_in_nested_def_ignored(self):
        code = (
            "def run():\n"
            "    def helper():\n"
            "        setOutput(x, 1)\n"  # must NOT be collected
            "    setOutput(y, 2)\n"      # must be collected
        )
        r = analyze_script(code)
        assert len(r["outputs"]) == 1
        assert r["outputs"][0]["name"] == "y"

    def test_getinput_in_nested_def_ignored(self):
        code = (
            "def run():\n"
            "    def inner():\n"
            "        getInput('ignored', 1)\n"  # NOT collected
            "    getInput('real', 2)\n"          # collected
        )
        r = analyze_script(code)
        assert len(r["inputs"]) == 1
        assert r["inputs"][0]["name"] == "real"

    def test_setoutput_in_lambda_ignored(self):
        code = (
            "def run():\n"
            "    f = lambda: setOutput(x, 0)\n"  # NOT collected
            "    setOutput(y, 1)\n"               # collected
        )
        r = analyze_script(code)
        assert len(r["outputs"]) == 1
        assert r["outputs"][0]["name"] == "y"

    def test_nested_async_def_ignored(self):
        code = (
            "def run():\n"
            "    async def worker():\n"
            "        setOutput(z, 9)\n"  # NOT collected
            "    setOutput(a, 0)\n"      # collected
        )
        r = analyze_script(code)
        assert len(r["outputs"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 10. Type index boundary / mapping completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeIndexMapping:
    def test_index_0_is_any(self):
        assert SCRIPT_DATA_TYPE_INDEX[0] == "any"

    def test_index_9_is_protocol(self):
        assert SCRIPT_DATA_TYPE_INDEX[9] == "protocol"

    def test_all_indices_unique_names(self):
        names = list(SCRIPT_DATA_TYPE_INDEX.values())
        assert len(names) == len(set(names))

    def test_boundary_0_valid(self):
        code = "def run():\n    setOutput(x, 0)\n"
        r = analyze_script(code)
        assert r["outputs"][0]["data_type"] == "any"

    def test_boundary_9_valid(self):
        code = "def run():\n    setOutput(x, 9)\n"
        r = analyze_script(code)
        assert r["outputs"][0]["data_type"] == "protocol"

    def test_boundary_10_invalid(self):
        code = "def run():\n    setOutput(x, 10)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)

    def test_negative_invalid(self):
        code = "def run():\n    setOutput(x, -1)\n"
        r = analyze_script(code)
        assert ScriptDiagCode.INVALID_DATA_TYPE_INDEX in _codes(r)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Diagnostics structure
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticsStructure:
    def test_each_diag_has_required_keys(self):
        code = "def run():\n    setOutput(x)\n"
        r = analyze_script(code)
        for d in r["diagnostics"]:
            assert "code" in d
            assert "level" in d
            assert "message" in d
            assert "lineno" in d

    def test_error_diag_has_lineno(self):
        code = "def run():\n    setOutput(x)\n"
        r = analyze_script(code)
        err = _errors(r)[0]
        assert err["lineno"] == 2

    def test_compat_warning_has_no_lineno(self):
        code = "def run():\n    return 1\n"
        r = analyze_script(code)
        compat = next(d for d in r["diagnostics"] if d["code"] == ScriptDiagCode.COMPAT_RETURN_USED)
        assert compat["lineno"] is None
