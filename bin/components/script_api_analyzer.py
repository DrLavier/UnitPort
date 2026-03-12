#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script API Analyzer — AST-based IO declaration extractor for Script Box nodes.

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DESIGN CONTRACT (Frozen 2026-03-07)
# ═══════════════════════════════════════════════════════════════════════════════

## 1. Current Call Chain (Baseline)

    ScriptEditor.save_and_compile()           [bin/components/script_editor.py:261]
      │  ast.parse(code)                       — syntax gate
      │  _parse_return_outputs(code, "run")    — infers outputs from last return
      │  emit compile_requested(node_id, outputs: list[{"name","data_type"}])
      ▼
    main_zone_panel.open_script_tab()         [bin/layout/main_zone_panel.py:575]
      │  ScriptEditor created with (node_id, script_name, io_spec)
      │  editor.compile_requested → graph_scene.script_update_outputs(nid, outputs)
      ▼
    GraphScene.script_update_outputs()        [bin/components/graph_scene.py:5262]
      │  Removes all existing output ports + active connections (with log warning)
      │  _script_add_output_row() for each new output entry
      ▼
    GraphScene._script_sync_io_spec()         [bin/components/graph_scene.py:5047]
      │  Reads _script_input_rows / _script_output_rows from node_item
      │  Builds node_item._script_io_spec = {"inputs": [...], "outputs": [...]}
      │  Calls _update_node_params() → persists to node_item.data(20)["script_io_spec"]
      │  Calls node_item._on_io_spec_updated(spec) if registered (syncs editor header)
      ▼
    IR Persistence                            [compiler/lowering/canvas_to_ir.py:412]
      │  Reads node_data["script_ref"] + node_data["script_io_spec"]
      │  Stored as IRParam in ir_node.params["script_ref"] / ["script_io_spec"]
      │  Restored by ir_to_canvas.py:349 on roundtrip

Current limitation: _parse_return_outputs() always returns data_type="any" for every
output slot; inputs are never modified by the compiler. This step is being replaced.


## 2. New API Contract

### 2.1  setOutput(data, NAME=None, TYPE=None)
    - Purpose: declare an output slot from inside run().
    - `data`            — the value or variable being output (any expression).
    - `NAME` — optional string literal override; default inferred from `data`.
    - `TYPE` — optional string literal type token, e.g. TYPE='str'.
    - May appear multiple times; each call adds one output slot in call order.
    - Call order determines slot assignment (deterministic / stable).
    - Validation rules:
        * Exactly 1 positional argument (data) required.
        * NAME/TYPE must be keyword string literals when provided.
        * TYPE must be one of: any/bool/int/float/str/list/dict/number/function/protocol.
        * data may be any AST expression (not validated beyond presence).

### 2.2  getInput(NAME=None, TYPE=None)
    - Purpose: declare an input slot and receive its runtime value.
    - `NAME` — optional string literal input name.
               If omitted, analyzer auto-infers name from assignment target:
               `speed = getInput()` -> NAME='speed'.
    - `TYPE` — optional string literal type token, e.g. TYPE='float'.
    - Validation rules:
        * NAME must be a non-empty string literal when provided.
        * If NAME omitted, assignment inference is required.
        * TYPE, if present, must be a supported TYPE token.
        * Duplicate names result in a DUPLICATE_INPUT_NAME diagnostic (second call ignored).

### 2.3  Compat mode (return-based, legacy)
    - If no setOutput() calls are found in run(), the analyzer sets compat_used=True.
    - A COMPAT_RETURN_USED diagnostic (level="warning") is emitted.
    - The caller (ScriptEditor Step 3) runs _parse_return_outputs() as fallback.
    - If setOutput() calls exist, return-based inference is skipped entirely.


## 3. Canonical data_type_index → data_type Mapping

    Index  Type name   Sub-dot color (fallback)   Theme key
    ─────  ─────────   ────────────────────────   ─────────────────
      0    "any"       #9ca3af  (gray)             sub_dot_any
      1    "bool"      #22c55e  (green)            sub_dot_bool
      2    "int"       #3b82f6  (blue)             sub_dot_int
      3    "float"     #06b6d4  (cyan)             sub_dot_float
      4    "string"    #f59e0b  (amber)            sub_dot_string
      5    "list"      #f97316  (orange)           sub_dot_list
      6    "dict"      #ec4899  (pink)             sub_dot_dict
      7    "number"    #0ea5a4  (teal)             sub_dot_number
      8    "function"  #a855f7  (purple)           sub_dot_function
      9    "protocol"  #a78bfa  (violet)           sub_dot_protocol

    SCRIPT_DATA_TYPE_INDEX is the authoritative mapping; do not duplicate it elsewhere.
    _sub_dot_type_color() in graph_scene.py already reads the same theme keys.


## 4. Diagnostic Codes

    Code                              Level    Trigger
    ────────────────────────────────  ───────  ──────────────────────────────────────────
    script.syntax_error               error    ast.parse() raised SyntaxError
    script.no_run_function            error    No def run(...) found in the script
    script.invalid_setoutput_args     error    setOutput() called with wrong arg count
    script.invalid_getinput_args      error    getInput() called with wrong arg count
    script.invalid_data_type_index    error    data_type_index not in SCRIPT_DATA_TYPE_INDEX
    script.getinput_missing_name      error    getInput() name arg absent or not a string literal
    script.getinput_empty_name        error    getInput() name resolves to empty string
    script.duplicate_input_name       warning  getInput() called twice with same name
    script.compat_return_used         warning  No setOutput() found; fell back to return parser

    Each diagnostic is a dict: {"code": str, "level": str, "message": str, "lineno": int|None}


## 5. Analyzer Result Shape

    {
        "inputs":      [{"name": str, "data_type": str, "slot": str|None}, ...],
        "outputs":     [{"name": str, "data_type": str}, ...],
        "diagnostics": [{"code": str, "level": str, "message": str, "lineno": int|None}, ...],
        "compat_used": bool,   # True when return-based fallback was used
    }

    - inputs  : ordered by first getInput() call; duplicates dropped after first.
    - outputs : ordered by setOutput() call order.
    - slot    : None in Step 2 (assigned by GraphScene on sync).

# ═══════════════════════════════════════════════════════════════════════════════
# END DESIGN CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

# ── Canonical type index table ────────────────────────────────────────────────

SCRIPT_DATA_TYPE_INDEX: Dict[int, str] = {
    0: "any",
    1: "bool",
    2: "int",
    3: "float",
    4: "string",
    5: "list",
    6: "dict",
    7: "number",
    8: "function",
    9: "protocol",
}

# Canonical type names accepted by Script Box IO declarations.
_CANONICAL_TYPES = frozenset(SCRIPT_DATA_TYPE_INDEX.values())

# Human-facing aliases accepted in TYPE='...'.
# "str" is the primary spelling shown in UI docs; maps to canonical "string".
_TYPE_ALIASES: Dict[str, str] = {
    "any": "any",
    "bool": "bool",
    "int": "int",
    "float": "float",
    "str": "string",
    "string": "string",
    "list": "list",
    "dict": "dict",
    "number": "number",
    "function": "function",
    "protocol": "protocol",
}

_VALID_TYPE_HINT = "any, bool, int, float, str, list, dict, number, function, protocol"
_VALID_GETINPUT_SIGNATURE = (
    "getInput() / getInput('input_name', 3) / getInput(NAME='input_name', TYPE='str')"
)
_VALID_SETOUTPUT_SIGNATURE = (
    "setOutput(data) / setOutput(data, 3) / setOutput(data, NAME='output_name', TYPE='str')"
)


# ── Diagnostic code constants ─────────────────────────────────────────────────

class ScriptDiagCode:
    """Diagnostic code strings for Script API validation."""

    SYNTAX_ERROR              = "script.syntax_error"
    NO_RUN_FUNCTION           = "script.no_run_function"
    INVALID_SETOUTPUT_ARGS    = "script.invalid_setoutput_args"
    INVALID_GETINPUT_ARGS     = "script.invalid_getinput_args"
    INVALID_DATA_TYPE_INDEX   = "script.invalid_data_type_index"
    GETINPUT_MISSING_NAME     = "script.getinput_missing_name"
    GETINPUT_EMPTY_NAME       = "script.getinput_empty_name"
    DUPLICATE_INPUT_NAME      = "script.duplicate_input_name"
    COMPAT_RETURN_USED        = "script.compat_return_used"


def _make_diag(
    code: str,
    level: str,
    message: str,
    lineno: Optional[int] = None,
) -> Dict:
    return {"code": code, "level": level, "message": message, "lineno": lineno}


# ── AST helpers ───────────────────────────────────────────────────────────────

def _resolve_str_literal(node) -> Tuple[Optional[str], bool]:
    """Extract a string constant from an AST node.

    Returns (value, ok). ok=False when the node is not a string literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    return None, False


def _resolve_int_type_literal(node) -> Tuple[Optional[str], bool]:
    """Resolve a positional type argument — accepts integer indices only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        canonical = SCRIPT_DATA_TYPE_INDEX.get(node.value)
        return (canonical, canonical is not None)
    return None, False


def _resolve_type_literal(node) -> Tuple[Optional[str], bool]:
    """Resolve TYPE literal to canonical data_type string.

    Accepts either the legacy integer index form or the newer string form.
    Returns (canonical_type, ok). ok=False when not a supported type literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        canonical = SCRIPT_DATA_TYPE_INDEX.get(node.value)
        return (canonical, canonical is not None)
    raw, ok = _resolve_str_literal(node)
    if not ok:
        return None, False
    normalized = (raw or "").strip().lower()
    canonical = _TYPE_ALIASES.get(normalized)
    if canonical is None or canonical not in _CANONICAL_TYPES:
        return None, False
    return canonical, True


def _extract_name_from_expr(expr_node, fallback_idx: int) -> str:
    """Best-effort output name from a setOutput data expression.

    - ast.Name  → variable name (most common: setOutput(result, 2))
    - ast.Attribute → attribute name (setOutput(self.value, 1))
    - anything else → out_<fallback_idx>
    """
    if isinstance(expr_node, ast.Name):
        return expr_node.id
    if isinstance(expr_node, ast.Attribute):
        return expr_node.attr
    return f"out_{fallback_idx}"


def _extract_assignment_name(target_node) -> Optional[str]:
    """Best-effort variable name from assignment target node."""
    if isinstance(target_node, ast.Name):
        return target_node.id
    if isinstance(target_node, ast.Attribute):
        return target_node.attr
    if isinstance(target_node, (ast.Tuple, ast.List)):
        for elt in target_node.elts:
            inferred = _extract_assignment_name(elt)
            if inferred:
                return inferred
    return None


# ── API call visitor ──────────────────────────────────────────────────────────

class _APICallVisitor(ast.NodeVisitor):
    """Collects setOutput/getInput Call nodes in source order.

    Skips nested FunctionDef and AsyncFunctionDef bodies so that calls inside
    inner helper functions are not incorrectly attributed to the outer run().
    Also skips Lambda bodies for the same reason.

    Usage: instantiate, then call visitor.visit(stmt) for each statement in
    run_func.body.  Collected calls are in self.calls as (func_name, ast.Call).
    """

    def __init__(self):
        self._skip_depth = 0
        self._name_hint_stack: List[Optional[str]] = []
        self.calls: List[Tuple[str, ast.Call, Optional[str]]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._skip_depth += 1
        self.generic_visit(node)
        self._skip_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda):
        self._skip_depth += 1
        self.generic_visit(node)
        self._skip_depth -= 1

    def visit_Call(self, node: ast.Call):
        if self._skip_depth == 0 and isinstance(node.func, ast.Name):
            if node.func.id in ("setOutput", "getInput"):
                name_hint = self._name_hint_stack[-1] if self._name_hint_stack else None
                self.calls.append((node.func.id, node, name_hint))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        name_hint = _extract_assignment_name(node.targets[0]) if node.targets else None
        self._name_hint_stack.append(name_hint)
        self.visit(node.value)
        self._name_hint_stack.pop()
        for t in node.targets:
            self.visit(t)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        name_hint = _extract_assignment_name(node.target)
        self._name_hint_stack.append(name_hint)
        if node.value is not None:
            self.visit(node.value)
        self._name_hint_stack.pop()
        self.visit(node.target)
        if node.annotation is not None:
            self.visit(node.annotation)

    def visit_AugAssign(self, node: ast.AugAssign):
        name_hint = _extract_assignment_name(node.target)
        self._name_hint_stack.append(name_hint)
        self.visit(node.value)
        self._name_hint_stack.pop()
        self.visit(node.target)


# ── Per-call processors ───────────────────────────────────────────────────────

def _process_setoutput(
    call_node: ast.Call,
    outputs: List[Dict],
    diagnostics: List[Dict],
    inferred_name: Optional[str],
) -> bool:
    """Validate and append one setOutput() call to *outputs*.

    Returns True on success, False if a diagnostic error was emitted.
    """
    lineno = getattr(call_node, "lineno", None)
    args = list(call_node.args or [])
    kw_pairs = [kw for kw in call_node.keywords if kw.arg is not None]
    kwargs_raw = {kw.arg: kw.value for kw in kw_pairs}
    kwargs_up = {k.upper(): v for k, v in kwargs_raw.items()}
    star_kwargs = [kw for kw in call_node.keywords if kw.arg is None]
    allowed = frozenset({"NAME", "TYPE"})
    unknown = sorted(k for k in kwargs_up if k not in allowed)

    has_explicit_type = "TYPE" in kwargs_up
    if not ((len(args) == 2) or (len(args) == 1 and has_explicit_type)):
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_SETOUTPUT_ARGS,
            "error",
            "setOutput() requires 2 positional arguments (data, TYPE_INDEX) or "
            "1 positional argument with TYPE= keyword. "
            f"Got {len(args)} positional arg(s). "
            f"Valid usage: {_VALID_SETOUTPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    if len(kwargs_up) != len(kwargs_raw) or star_kwargs or unknown:
        offenders = (["**<expr>"] * len(star_kwargs)) + [f"'{k}'" for k in unknown]
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_SETOUTPUT_ARGS,
            "error",
            "setOutput() received invalid keyword argument(s): "
            f"{', '.join(offenders) if offenders else 'duplicate keyword(s)'}. "
            f"Allowed kwargs: NAME, TYPE. Valid usage: {_VALID_SETOUTPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    data_node = args[0]
    positional_type_node = args[1] if len(args) > 1 else None
    explicit_name_node = kwargs_up.get("NAME")
    explicit_type_node = kwargs_up.get("TYPE")

    if positional_type_node is not None and explicit_type_node is not None:
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_SETOUTPUT_ARGS,
            "error",
            "setOutput() TYPE was provided both positionally and via TYPE=...; "
            "use only one form. "
            f"Valid usage: {_VALID_SETOUTPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    if explicit_name_node is not None:
        output_name, ok = _resolve_str_literal(explicit_name_node)
        output_name = (output_name or "").strip() if ok else ""
        if not ok or not output_name:
            diagnostics.append(_make_diag(
                ScriptDiagCode.INVALID_SETOUTPUT_ARGS,
                "error",
                "setOutput(NAME=...) must be a non-empty string literal. "
                f"Valid usage: {_VALID_SETOUTPUT_SIGNATURE}.",
                lineno,
            ))
            return False
    else:
        output_name = _extract_name_from_expr(data_node, len(outputs))

    data_type = "any"
    if positional_type_node is not None:
        resolved, ok = _resolve_int_type_literal(positional_type_node)
        if not ok:
            diagnostics.append(_make_diag(
                ScriptDiagCode.INVALID_DATA_TYPE_INDEX,
                "error",
                "setOutput() positional TYPE must be an integer index (0-9). "
                f"Supported indices: {list(SCRIPT_DATA_TYPE_INDEX.keys())}.",
                lineno,
            ))
            return False
        data_type = resolved
    elif explicit_type_node is not None:
        resolved, ok = _resolve_type_literal(explicit_type_node)
        if not ok:
            diagnostics.append(_make_diag(
                ScriptDiagCode.INVALID_DATA_TYPE_INDEX,
                "error",
                "setOutput(TYPE=...) is invalid. "
                f"Supported TYPE values: {_VALID_TYPE_HINT}.",
                lineno,
            ))
            return False
        data_type = resolved

    # If NAME was inferred from assignment context, keep explicit NAME/data inference.
    # setOutput usually appears as a standalone statement, so inferred_name is unused.
    if not output_name:
        output_name = inferred_name or _extract_name_from_expr(data_node, len(outputs))
    outputs.append({"name": output_name, "data_type": data_type})
    return True


def _process_getinput(
    call_node: ast.Call,
    inputs: List[Dict],
    diagnostics: List[Dict],
    seen_names: Dict[str, Optional[int]],
    inferred_name: Optional[str],
) -> bool:
    """Validate and append one getInput() call to *inputs*.

    Returns True on success, False if a diagnostic error was emitted.
    """
    lineno = getattr(call_node, "lineno", None)
    args = list(call_node.args or [])
    kw_pairs = [kw for kw in call_node.keywords if kw.arg is not None]
    kwargs_raw = {kw.arg: kw.value for kw in kw_pairs}
    kwargs_up = {k.upper(): v for k, v in kwargs_raw.items()}
    star_kwargs = [kw for kw in call_node.keywords if kw.arg is None]
    allowed = frozenset({"NAME", "TYPE", "DATA_TYPE_INDEX"})
    unknown = sorted(k for k in kwargs_up if k not in allowed)

    # Positional forms: getInput(), getInput('name'), getInput('name', TYPE)
    if len(args) > 2:
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_GETINPUT_ARGS, "error",
            "getInput() takes at most 2 positional arguments (NAME[, TYPE]). "
            f"Got {len(args)} positional arg(s). "
            f"Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    if len(kwargs_up) != len(kwargs_raw) or star_kwargs or unknown:
        offenders = (["**<expr>"] * len(star_kwargs)) + [f"'{k}'" for k in unknown]
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_GETINPUT_ARGS, "error",
            "getInput() received invalid keyword argument(s): "
            f"{', '.join(offenders) if offenders else 'duplicate keyword(s)'}. "
            f"Allowed kwargs: NAME, TYPE. Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    # Resolve NAME priority: kw NAME > positional > assignment inference.
    explicit_name_node = kwargs_up.get("NAME")
    pos_name_node = args[0] if args else None
    positional_type_node = args[1] if len(args) > 1 else None
    if explicit_name_node is not None and pos_name_node is not None:
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_GETINPUT_ARGS,
            "error",
            "getInput() NAME was provided both positionally and via NAME=...; "
            "use only one form. "
            f"Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
            lineno,
        ))
        return False
    if positional_type_node is not None and kwargs_up.get("TYPE") is not None:
        diagnostics.append(_make_diag(
            ScriptDiagCode.INVALID_GETINPUT_ARGS,
            "error",
            "getInput() TYPE was provided both positionally and via TYPE=...; "
            "use only one form. "
            f"Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
            lineno,
        ))
        return False

    name_node = explicit_name_node if explicit_name_node is not None else pos_name_node
    if name_node is not None:
        name_val, name_ok = _resolve_str_literal(name_node)
        if not name_ok:
            diagnostics.append(_make_diag(
                ScriptDiagCode.GETINPUT_MISSING_NAME,
                "error",
                "getInput(NAME=...) must be a string literal. "
                f"Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
                lineno,
            ))
            return False
        name_val = name_val.strip()
        if not name_val:
            diagnostics.append(_make_diag(
                ScriptDiagCode.GETINPUT_EMPTY_NAME, "error",
                "getInput() NAME must not be empty or whitespace-only.",
                lineno,
            ))
            return False
    else:
        name_val = (inferred_name or "").strip()
        if not name_val:
            diagnostics.append(_make_diag(
                ScriptDiagCode.GETINPUT_MISSING_NAME, "error",
                "getInput() could not infer NAME automatically. "
                "Assign it to a variable (e.g. speed = getInput()) or pass NAME='speed'. "
                f"Valid usage: {_VALID_GETINPUT_SIGNATURE}.",
                lineno,
            ))
            return False

    if name_val in seen_names:
        diagnostics.append(_make_diag(
            ScriptDiagCode.DUPLICATE_INPUT_NAME, "warning",
            f"getInput() name '{name_val}' was already declared "
            f"(first at line {seen_names[name_val]}); "
            f"this call at line {lineno} is ignored.",
            lineno,
        ))
        return False

    data_type = "any"
    type_node = (
        kwargs_up.get("TYPE")
        or kwargs_up.get("DATA_TYPE_INDEX")
        or positional_type_node
    )
    if type_node is not None:
        resolved, ok = _resolve_type_literal(type_node)
        if not ok:
            diagnostics.append(_make_diag(
                ScriptDiagCode.INVALID_DATA_TYPE_INDEX, "error",
                f"getInput('{name_val}') TYPE is invalid. "
                f"Supported TYPE values: {_VALID_TYPE_HINT}.",
                lineno,
            ))
            return False
        data_type = resolved

    seen_names[name_val] = lineno
    inputs.append({"name": name_val, "data_type": data_type, "slot": None})
    return True


# ── Public API ────────────────────────────────────────────────────────────────

_NON_ENTRY_NAMES: frozenset = frozenset({
    "helper", "helpers", "util", "utils", "utility",
    "internal", "private", "setup", "teardown",
})


def _select_entry_function(tree: ast.AST) -> Tuple[Optional[ast.FunctionDef], Optional[str]]:
    """Select script entry function.

    Priority:
      1) top-level ``def init(...)`` (new preferred contract)
      2) top-level ``def run(...)``  (legacy compatibility)
      3) exactly one top-level function whose name is not in the known
         non-entry set (custom entry)
    """
    top_level_funcs = [
        n for n in getattr(tree, "body", [])
        if isinstance(n, ast.FunctionDef)
    ]
    for fn in top_level_funcs:
        if fn.name == "init":
            return fn, "init"
    for fn in top_level_funcs:
        if fn.name == "run":
            return fn, "run"
    if len(top_level_funcs) == 1 and top_level_funcs[0].name not in _NON_ENTRY_NAMES:
        return top_level_funcs[0], "custom"
    return None, None


def _extract_signature_inputs(code: str, entry_func: ast.FunctionDef) -> List[Dict]:
    """Extract input declarations from the selected entry function signature.

    Supports the editor-generated multiline form::

        def init(
            speed,   # float
            mode,    # string
        ):

    If no type comment is present, the input defaults to ``any``.
    """
    if entry_func is None:
        return []

    positional_args = list(getattr(entry_func.args, "posonlyargs", []) or [])
    positional_args.extend(list(getattr(entry_func.args, "args", []) or []))
    if positional_args and getattr(positional_args[0], "arg", None) == "self":
        positional_args = positional_args[1:]

    if not positional_args:
        return []

    lines = code.splitlines()
    sig_start = max((getattr(entry_func, "lineno", 1) or 1) - 1, 0)
    body_start = max((entry_func.body[0].lineno - 1), sig_start) if getattr(entry_func, "body", None) else sig_start
    sig_lines = lines[sig_start:body_start]
    type_by_name: Dict[str, str] = {}
    for raw_line in sig_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("def "):
            continue
        name_part, sep, comment_part = raw_line.partition("#")
        name_token = name_part.strip().rstrip(",")
        if not name_token or name_token == "):":
            continue
        data_type = "any"
        if sep:
            comment_type = comment_part.strip().rstrip(",:")
            if comment_type:
                data_type = comment_type
        type_by_name[name_token] = data_type

    inputs: List[Dict] = []
    seen_names = set()
    for arg in positional_args:
        name = str(getattr(arg, "arg", "") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        inputs.append({
            "name": name,
            "data_type": type_by_name.get(name, "any"),
            "slot": None,
        })
    return inputs


def analyze_script(code: str) -> Dict:
    """Analyze *code* for setOutput/getInput API calls inside entry function.

    Pure-Python, no Qt dependency.  Safe to call from tests or worker threads.

    Returns a dict with keys:
        inputs      : list of {"name", "data_type", "slot"} — from getInput() calls
        outputs     : list of {"name", "data_type"}         — from setOutput() calls
        diagnostics : list of {"code", "level", "message", "lineno"}
        compat_used : True when no setOutput() found and compat fallback is needed
        entry_function : selected entry function name (init/run/custom name)

    When compat_used is True the caller (ScriptEditor) should run
    _parse_return_outputs() and merge its result as the outputs list.
    """
    # ── 1. Syntax check ───────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "inputs": [],
            "outputs": [],
            "diagnostics": [_make_diag(
                ScriptDiagCode.SYNTAX_ERROR, "error",
                f"Syntax error on line {exc.lineno}: {exc.msg}",
                exc.lineno,
            )],
            "compat_used": False,
        }

    # ── 2. Locate entry function ──────────────────────────────────────────────
    entry_func, entry_kind = _select_entry_function(tree)
    if entry_func is None:
        return {
            "inputs": [],
            "outputs": [],
            "diagnostics": [_make_diag(
                ScriptDiagCode.NO_RUN_FUNCTION, "error",
                "No entry function found. Define 'def init(...)' (preferred), "
                "'def run(...)' (legacy), or keep exactly one top-level function.",
            )],
            "compat_used": False,
            "entry_function": None,
        }

    # ── 3. Collect input declarations from entry signature ────────────────────
    inputs: List[Dict] = _extract_signature_inputs(code, entry_func)
    seen_input_names: Dict[str, Optional[int]] = {
        inp["name"]: None for inp in inputs if inp.get("name")
    }

    # ── 4. Collect API calls in source order ──────────────────────────────────
    # Walk only the entry function body; nested defs/lambdas excluded by visitor.
    visitor = _APICallVisitor()
    for stmt in entry_func.body:
        visitor.visit(stmt)

    # ── 5. Process calls ──────────────────────────────────────────────────────
    outputs: List[Dict] = []
    diagnostics: List[Dict] = []

    for func_name, call_node, inferred_name in visitor.calls:
        if func_name == "setOutput":
            _process_setoutput(call_node, outputs, diagnostics, inferred_name)
        else:
            _process_getinput(
                call_node, inputs, diagnostics, seen_input_names, inferred_name
            )

    # ── 6. Compat check ───────────────────────────────────────────────────────
    compat_used = len(outputs) == 0
    if compat_used:
        diagnostics.append(_make_diag(
            ScriptDiagCode.COMPAT_RETURN_USED, "warning",
            "No setOutput() calls found in script. "
            "Falling back to return-based output discovery (legacy mode). "
            "Use setOutput(data, NAME='...', TYPE='...') to silence this warning.",
        ))

    return {
        "inputs": inputs,
        "outputs": outputs,
        "diagnostics": diagnostics,
        "compat_used": compat_used,
        "entry_function": entry_func.name,
    }
