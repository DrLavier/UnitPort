"""Syntax highlighters for CodeEditorWidget.

Provides:

* :class:`PythonSyntaxHighlighter` — full Python token rules with a
  triple-quoted string state machine (carries DEMO behaviour 1:1).
* :class:`JsonSyntaxHighlighter` — lightweight JSON token rules
  (key / string / number / bool / null / punctuation).

Colors are read via :func:`unitport_sdk.Config.get_color` so end users can
override the syntax palette in ``user.ini[Theme]`` without restarting.
"""

from __future__ import annotations

import re
from typing import List, Optional

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat

from ..sys import Config

# ---------------------------------------------------------------------------
# Python token sets
# ---------------------------------------------------------------------------

_PY_KEYWORDS = (
    "False|None|True|and|as|assert|async|await|break|class|continue|def|del|"
    "elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|"
    "not|or|pass|raise|return|try|while|with|yield"
)

_PY_BUILTINS = (
    "abs|all|any|bin|bool|breakpoint|bytearray|bytes|callable|chr|classmethod|"
    "compile|complex|delattr|dict|dir|divmod|enumerate|eval|exec|filter|float|"
    "format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|"
    "isinstance|issubclass|iter|len|list|locals|map|max|memoryview|min|next|"
    "object|oct|open|ord|pow|print|property|range|repr|reversed|round|set|"
    "setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|vars|zip|"
    "__import__|__name__|__file__|__doc__"
)


def _make_fmt(slot: str, fallback: str, bold: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(Config.get_color(slot, fallback)))
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)
    return fmt


# ---------------------------------------------------------------------------
# Python highlighter
# ---------------------------------------------------------------------------

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Theme-aware Python syntax highlighter.

    Token slots (read from ``[Theme]``):

    * ``syntax_keyword``        — control flow keywords
    * ``syntax_builtin``        — built-in names (print, len, ...)
    * ``syntax_system_builtin`` — extra API built-ins (optional)
    * ``syntax_comment``        — line comments
    * ``syntax_string``         — string literals (single + triple)
    * ``syntax_number``         — numeric literals
    * ``syntax_defname``        — function/class name after def/class
    * ``syntax_decorator``      — @decorator names
    """

    _STATE_NORMAL = -1
    _STATE_TRIPLE_D = 1   # inside """
    _STATE_TRIPLE_S = 2   # inside '''

    def __init__(self, document, *, extra_builtins: Optional[List[str]] = None):
        super().__init__(document)
        self._extra_builtins = list(extra_builtins or [])
        self._rules: list = []
        self._fmt_string = QTextCharFormat()
        self._fmt_comment = QTextCharFormat()
        self._fmt_docstring = QTextCharFormat()
        self._build_rules()

    def set_extra_builtins(self, names: List[str]) -> None:
        """Update the extra builtin list and rehighlight."""
        self._extra_builtins = list(names)
        self._build_rules()
        self.rehighlight()

    def _build_rules(self) -> None:
        self._rules.clear()

        def _add(pattern: str, slot: str, fallback: str,
                 bold: bool = False, cap: int = 0) -> None:
            self._rules.append(
                (QRegularExpression(pattern), _make_fmt(slot, fallback, bold), cap)
            )

        # Generic identifiers (variables) — registered FIRST so any later,
        # more-specific rule (keyword/builtin/defname/self) overpaints it.
        _add(r"\b[A-Za-z_]\w*\b", "syntax_variable", "#60CCCA")

        # Decorators
        _add(r"@[\w.]+", "syntax_decorator", "#c586c0")

        # Function / class name (capture group 1)
        _add(r"\b(?:def|class)\s+(\w+)", "misc_4", "#dcdcaa", bold=True, cap=1)

        # Keywords
        _add(rf"\b(?:{_PY_KEYWORDS})\b", "misc_2", "#569cd6", bold=True)

        # Built-in names
        _add(rf"\b(?:{_PY_BUILTINS})\b", "syntax_builtin", "#4ec9b0")

        # Extra / system builtins
        if self._extra_builtins:
            pattern = "|".join(re.escape(n) for n in self._extra_builtins)
            _add(rf"\b(?:{pattern})\b", "syntax_system_builtin", "#f59e0b", bold=True)

        # Conventional instance/class refs (override the generic-identifier rule)
        _add(r"\b(?:self|cls|mcs)\b", "syntax_property", "#D4B181")

        # Numbers
        _add(
            r"\b0[xX][0-9a-fA-F]+\b"
            r"|\b0[bB][01]+\b"
            r"|\b0[oO][0-7]+\b"
            r"|\b\d+\.\d*(?:[eE][+-]?\d+)?\b"
            r"|\b\d+[eE][+-]?\d+\b"
            r"|\b\d+\b",
            "misc_1", "#b5cea8",
        )

        # Single-line strings
        _add(r'"[^"\\\n]*(?:\\.[^"\\\n]*)*"', "syntax_string", "#ce9178")
        _add(r"'[^'\\\n]*(?:\\.[^'\\\n]*)*'", "syntax_string", "#ce9178")

        # Stored for multi-block constructs
        self._fmt_string = _make_fmt("syntax_string", "#ce9178")
        self._fmt_comment = _make_fmt("syntax_comment", "#6a9955")
        # Triple-quoted text reads as a structural docstring/long comment,
        # not as a value-carrying string literal.
        self._fmt_docstring = _make_fmt("syntax_long_comment", "#6B6B6B")

    def highlightBlock(self, text: str) -> None:  # type: ignore[override]
        self.setCurrentBlockState(self._STATE_NORMAL)
        prev = self.previousBlockState()
        start = 0

        # Resume triple-quoted string from previous block
        if prev in (self._STATE_TRIPLE_D, self._STATE_TRIPLE_S):
            delim = '"""' if prev == self._STATE_TRIPLE_D else "'''"
            end = text.find(delim)
            if end == -1:
                self.setFormat(0, len(text), self._fmt_docstring)
                self.setCurrentBlockState(prev)
                return
            self.setFormat(0, end + 3, self._fmt_docstring)
            start = end + 3

        # Inline rules
        for regex, fmt, cap_group in self._rules:
            it = regex.globalMatch(text, start)
            while it.hasNext():
                m = it.next()
                s = m.capturedStart(cap_group) if cap_group else m.capturedStart()
                ln = m.capturedLength(cap_group) if cap_group else m.capturedLength()
                if s >= start:
                    self.setFormat(s, ln, fmt)

        # Triple-quoted strings — coloured as docstring/long comment, not as
        # a regular string literal.
        i = start
        while i < len(text):
            matched = False
            for delim, state in (('"""', self._STATE_TRIPLE_D), ("'''", self._STATE_TRIPLE_S)):
                if text[i:i + 3] == delim:
                    end = text.find(delim, i + 3)
                    if end == -1:
                        self.setFormat(i, len(text) - i, self._fmt_docstring)
                        self.setCurrentBlockState(state)
                        i = len(text)
                    else:
                        self.setFormat(i, end + 3 - i, self._fmt_docstring)
                        i = end + 3
                    matched = True
                    break
            if not matched:
                i += 1

        # Comments override (only outside triple strings)
        if self.currentBlockState() == self._STATE_NORMAL:
            c = text.find("#", start)
            if c != -1:
                self.setFormat(c, len(text) - c, self._fmt_comment)


# ---------------------------------------------------------------------------
# JSON highlighter
# ---------------------------------------------------------------------------

class JsonSyntaxHighlighter(QSyntaxHighlighter):
    """Lightweight JSON syntax highlighter.

    Token slots (read from ``[Theme]``):

    * ``syntax_json_key``    — keys (the string before a ``:``)
    * ``syntax_json_string`` — string values
    * ``syntax_json_number`` — numeric values
    * ``syntax_json_bool``   — ``true`` / ``false``
    * ``syntax_json_null``   — ``null``
    * ``syntax_json_punct``  — ``{ } [ ] , :``

    Strict JSON has no comments and no multi-line strings, so we run
    purely with single-block regex rules — no state machine needed.
    """

    def __init__(self, document):
        super().__init__(document)
        self._rules: list = []
        self._fmt_key = QTextCharFormat()
        self._build_rules()

    def _build_rules(self) -> None:
        self._rules.clear()

        def _add(pattern: str, slot: str, fallback: str, cap: int = 0) -> None:
            self._rules.append(
                (QRegularExpression(pattern), _make_fmt(slot, fallback), cap)
            )

        # Punctuation
        _add(r"[{}\[\],:]", "main_t1", "#d6d3c7")

        # Numbers (int / float / scientific, optional leading sign)
        _add(
            r"-?\b\d+\.\d*(?:[eE][+-]?\d+)?\b"
            r"|-?\b\d+[eE][+-]?\d+\b"
            r"|-?\b\d+\b",
            "misc_1", "#b5cea8",
        )

        # true / false / null
        _add(r"\btrue\b|\bfalse\b", "misc_2", "#569cd6")
        _add(r"\bnull\b", "misc_2", "#569cd6")

        # All double-quoted strings — coloured as values; key rule below
        # re-paints over the key occurrences.
        _add(r'"(?:[^"\\\n]|\\.)*"', "syntax_json_string", "#ce9178")

        # Keys: a double-quoted string immediately followed by ``:``
        # (whitespace allowed). Capture group 1 is the quoted key itself.
        _add(
            r'("(?:[^"\\\n]|\\.)*")\s*:',
            "syntax_json_key", "#9cdcfe", cap=1,
        )

    def highlightBlock(self, text: str) -> None:  # type: ignore[override]
        for regex, fmt, cap_group in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                s = m.capturedStart(cap_group) if cap_group else m.capturedStart()
                ln = m.capturedLength(cap_group) if cap_group else m.capturedLength()
                if s >= 0 and ln > 0:
                    self.setFormat(s, ln, fmt)
