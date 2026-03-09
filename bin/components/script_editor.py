#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScriptEditor — per-Script-Box editable Python code tab."""

import ast
import re
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal, QRect, QSize, QRegularExpression, QTimer, QEvent
from PySide6.QtGui import (
    QFont,
    QTextCharFormat,
    QColor,
    QTextCursor,
    QPainter,
    QSyntaxHighlighter,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QTextEdit,
    QLabel,
    QPushButton,
    QMessageBox,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QSplitter,
    QListWidget,
    QListWidgetItem,
)

from bin.core.theme_manager import get_color, get_font_size
from bin.core.logger import log_info, log_warning, log_error
from bin.components.script_api_analyzer import analyze_script, ScriptDiagCode
from bin.components.script_builtin_registry import SCRIPT_BUILTIN_FUNCTIONS

_ENTRY_DEF_PREFIXES = ("def init(", "def run(")

# ── Indentation engine ────────────────────────────────────────────────────────

# Matches a line that ends with ':' (possibly followed by a comment/whitespace),
# signalling that the next line should be indented one more level.
_BLOCK_OPENER_RE = re.compile(r":\s*(#.*)?$")


def _compute_smart_indent(line_text: str) -> str:
    """Return the indent string to insert on the new line after *line_text*.

    Rules:
      - Carry the current indentation of *line_text*.
      - If the stripped line ends with ':' (block opener), add 4 more spaces.

    Pure Python — no Qt dependency; safe for unit tests without a display.
    """
    stripped = line_text.rstrip()
    n_spaces = len(line_text) - len(line_text.lstrip(" "))
    indent = " " * n_spaces
    if _BLOCK_OPENER_RE.search(stripped):
        indent += "    "
    return indent


# ── Syntax highlighter token sets ────────────────────────────────────────────

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

_SCRIPT_API_BUILTINS = tuple(spec.name for spec in SCRIPT_BUILTIN_FUNCTIONS)
_SCRIPT_API_PATTERN = "|".join(re.escape(name) for name in _SCRIPT_API_BUILTINS)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Python token syntax highlighter for PythonIndentEditor.

    Colors are read from ui.ini via get_color() with safe defaults so the
    editor works correctly even when config keys are absent.

    Token groups and their ui.ini keys ([Dark] / [Light] sections):
        syntax_keyword   — keywords / control flow
        syntax_builtin   — built-in names
        syntax_system_builtin — Script Box built-ins (getInput/setOutput)
        syntax_comment   — # line comments
        syntax_string    — string literals (single- and triple-quoted)
        syntax_number    — numeric literals
        syntax_defname   — function / class name after def / class
        syntax_decorator — @decorator names
    """

    _STATE_NORMAL   = -1   # default between triple-quoted strings
    _STATE_TRIPLE_D = 1    # inside """
    _STATE_TRIPLE_S = 2    # inside '''

    def __init__(self, document):
        super().__init__(document)
        self._rules: list = []          # (QRegularExpression, QTextCharFormat, cap_group)
        self._fmt_string  = QTextCharFormat()
        self._fmt_comment = QTextCharFormat()
        self._build_rules()

    # ── Format helpers ────────────────────────────────────────────────────────

    def _make_fmt(self, color_key: str, fallback: str, bold: bool = False) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(get_color(color_key, fallback)))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        return fmt

    def _build_rules(self):
        """Compile regex rules from current theme config. Re-call after theme change."""
        self._rules.clear()

        def _add(pattern: str, key: str, fallback: str, bold: bool = False, cap: int = 0):
            self._rules.append(
                (QRegularExpression(pattern), self._make_fmt(key, fallback, bold), cap)
            )

        # Decorators
        _add(r"@[\w.]+", "syntax_decorator", "#c586c0")

        # Function / class name — capture group 1 is the name only
        _add(r"\b(?:def|class)\s+(\w+)", "syntax_defname", "#dcdcaa", bold=True, cap=1)

        # Keywords / control flow
        _add(rf"\b(?:{_PY_KEYWORDS})\b", "syntax_keyword", "#569cd6", bold=True)

        # Built-in names
        _add(rf"\b(?:{_PY_BUILTINS})\b", "syntax_builtin", "#4ec9b0")
        # Script system built-ins (higher priority than generic built-ins)
        if _SCRIPT_API_PATTERN:
            _add(
                rf"\b(?:{_SCRIPT_API_PATTERN})\b",
                "syntax_system_builtin",
                "#f59e0b",
                bold=True,
            )

        # Numbers: hex, binary, octal, float, int
        _add(
            r"\b0[xX][0-9a-fA-F]+\b"
            r"|\b0[bB][01]+\b"
            r"|\b0[oO][0-7]+\b"
            r"|\b\d+\.\d*(?:[eE][+-]?\d+)?\b"
            r"|\b\d+[eE][+-]?\d+\b"
            r"|\b\d+\b",
            "syntax_number", "#b5cea8",
        )

        # Single-line strings (with backslash-escape support)
        _add(r'"[^"\\\n]*(?:\\.[^"\\\n]*)*"', "syntax_string", "#ce9178")
        _add(r"'[^'\\\n]*(?:\\.[^'\\\n]*)*'", "syntax_string", "#ce9178")

        # Store string/comment formats separately for multi-block constructs
        self._fmt_string  = self._make_fmt("syntax_string",  "#ce9178")
        self._fmt_comment = self._make_fmt("syntax_comment", "#6a9955")

    # ── Core highlighting ─────────────────────────────────────────────────────

    def highlightBlock(self, text: str):
        self.setCurrentBlockState(self._STATE_NORMAL)
        prev  = self.previousBlockState()
        start = 0

        # ── Resume triple-quoted string from previous block ─────────────────
        if prev in (self._STATE_TRIPLE_D, self._STATE_TRIPLE_S):
            delim = '"""' if prev == self._STATE_TRIPLE_D else "'''"
            end   = text.find(delim)
            if end == -1:
                # Entire line is inside the triple-quoted string
                self.setFormat(0, len(text), self._fmt_string)
                self.setCurrentBlockState(prev)
                return
            # Closing delimiter found — colour up to and including it
            self.setFormat(0, end + 3, self._fmt_string)
            start = end + 3

        # ── Apply inline rules ───────────────────────────────────────────────
        for regex, fmt, cap_group in self._rules:
            it = regex.globalMatch(text, start)
            while it.hasNext():
                m  = it.next()
                s  = m.capturedStart(cap_group) if cap_group else m.capturedStart()
                ln = m.capturedLength(cap_group) if cap_group else m.capturedLength()
                if s >= start:
                    self.setFormat(s, ln, fmt)

        # ── Scan for triple-quoted strings opening on this line ─────────────
        i = start
        while i < len(text):
            matched = False
            for delim, state in (('"""', self._STATE_TRIPLE_D), ("'''", self._STATE_TRIPLE_S)):
                if text[i:i + 3] == delim:
                    end = text.find(delim, i + 3)
                    if end == -1:
                        # Opens but does not close on this line
                        self.setFormat(i, len(text) - i, self._fmt_string)
                        self.setCurrentBlockState(state)
                        i = len(text)
                    else:
                        # Opens and closes on the same line
                        self.setFormat(i, end + 3 - i, self._fmt_string)
                        i = end + 3
                    matched = True
                    break
            if not matched:
                i += 1

        # ── Comments: override from '#' to end of line ──────────────────────
        if self.currentBlockState() == self._STATE_NORMAL:
            c = text.find("#", start)
            if c != -1:
                self.setFormat(c, len(text) - c, self._fmt_comment)


class _LineNumberArea(QWidget):
    """Left gutter that paints line numbers for PythonIndentEditor."""

    def __init__(self, editor: "PythonIndentEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self._editor.lineNumberAreaPaintEvent(event)


class PythonIndentEditor(QPlainTextEdit):
    """QPlainTextEdit subclass with Python-friendly indentation behaviour.

    Key bindings added:
      Tab          — insert 4 spaces (or indent every selected line by 4)
      Shift+Tab    — remove up to 4 leading spaces from line(s) (dedent)
      Enter/Return — carry current indent; add one level after block openers (':')
      Ctrl+Wheel   — zoom font size in/out (clamped to _ZOOM_MIN.._ZOOM_MAX pt)
    """

    _ZOOM_MIN = 6    # minimum font size in points
    _ZOOM_MAX = 40   # maximum font size in points

    def keyPressEvent(self, event):
        key = event.key()

        if self._completion_popup.isVisible():
            if key in (Qt.Key_Up, Qt.Key_Down):
                self._move_completion_selection(-1 if key == Qt.Key_Up else 1)
                return
            if key in (Qt.Key_Tab, Qt.Key_Return, Qt.Key_Enter):
                self._accept_completion()
                return
            if key == Qt.Key_Escape:
                self._completion_popup.hide()
                return

        if key == Qt.Key_Tab:
            self._handle_tab()
            return

        if key == Qt.Key_Backtab:          # Shift+Tab on all platforms
            self._handle_backtab()
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._handle_return()
            return

        super().keyPressEvent(event)
        if key in (
            Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Left, Qt.Key_Right,
            Qt.Key_Home, Qt.Key_End,
        ) or (32 <= key <= 126):
            self._update_completion_popup()

    # ── Ctrl+Wheel zoom ───────────────────────────────────────────────────────

    def wheelEvent(self, event):
        """Ctrl+wheel zooms font size; plain wheel scrolls normally."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                self._adjust_font_size(1 if delta > 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    def _adjust_font_size(self, step: int):
        """Change font point size by *step*, clamped to [_ZOOM_MIN, _ZOOM_MAX]."""
        font = self.font()
        new_size = max(self._ZOOM_MIN, min(self._ZOOM_MAX, font.pointSize() + step))
        if new_size != font.pointSize():
            font.setPointSize(new_size)
            self.setFont(font)
            self._on_block_count_changed(0)   # refresh gutter margin width

    # ── Row style ───────────────────────────────────────────────────────────────

    def set_row_colors(self, base_bg: str, alt_bg: str, row_border: str = ""):
        """Set editor background, gutter background and optional row separator."""
        self._base_bg = base_bg
        self._alt_bg = alt_bg
        self._row_border = row_border
        self.viewport().update()
        self._line_number_area.update()

    def eventFilter(self, obj, event):
        """Paint editor row background on the viewport before Qt draws text.

        Because we install this filter *after* QAbstractScrollArea installs its
        own (which forwards the Paint event to QPlainTextEdit::paintEvent), our
        filter is called first.  We fill the background, return False, then Qt
        draws text on top without refilling the background (autoFillBackground
        is disabled on the viewport).
        """
        if obj is self.viewport() and event.type() == QEvent.Type.Paint:
            painter = QPainter(self.viewport())
            vp = self.viewport().rect()
            painter.fillRect(vp, QColor(self._base_bg))

            painter.end()
        return False

    # ── Tab ───────────────────────────────────────────────────────────────────

    def _handle_tab(self):
        if self._completion_popup.isVisible():
            self._accept_completion()
            return
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._indent_selection(cursor, 4)
        else:
            cursor.insertText("    ")
        self._completion_popup.hide()

    def _indent_selection(self, cursor, spaces: int):
        """Prepend *spaces* spaces to every line in the current selection."""
        start = cursor.selectionStart()
        end   = cursor.selectionEnd()
        doc   = self.document()

        start_block = doc.findBlock(start).blockNumber()
        end_block   = doc.findBlock(end).blockNumber()

        cursor.beginEditBlock()
        # Walk blocks from last to first so character offsets stay valid
        for block_no in range(end_block, start_block - 1, -1):
            block = doc.findBlockByNumber(block_no)
            if not block.isValid():
                continue
            c = QTextCursor(block)
            c.movePosition(QTextCursor.StartOfBlock)
            c.insertText(" " * spaces)
        cursor.endEditBlock()

    # ── Shift+Tab ─────────────────────────────────────────────────────────────

    def _handle_backtab(self):
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._dedent_selection(cursor, 4)
        else:
            self._dedent_block(cursor.block(), 4)

    def _dedent_selection(self, cursor, max_spaces: int):
        """Remove up to *max_spaces* leading spaces from every line in selection."""
        start = cursor.selectionStart()
        end   = cursor.selectionEnd()
        doc   = self.document()

        start_block = doc.findBlock(start).blockNumber()
        end_block   = doc.findBlock(end).blockNumber()

        cursor.beginEditBlock()
        for block_no in range(end_block, start_block - 1, -1):
            block = doc.findBlockByNumber(block_no)
            if block.isValid():
                self._dedent_block_in_edit(block, max_spaces)
        cursor.endEditBlock()

    def _dedent_block(self, block, max_spaces: int):
        cursor = QTextCursor(block)
        cursor.beginEditBlock()
        self._dedent_block_in_edit(block, max_spaces)
        cursor.endEditBlock()

    @staticmethod
    def _dedent_block_in_edit(block, max_spaces: int):
        """Remove up to *max_spaces* leading spaces from *block* (no editBlock guard)."""
        text = block.text()
        remove = min(len(text) - len(text.lstrip(" ")), max_spaces)
        if remove:
            c = QTextCursor(block)
            c.movePosition(QTextCursor.StartOfBlock)
            c.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, remove)
            c.removeSelectedText()

    # ── constructor ───────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._line_number_area = _LineNumberArea(self)
        self._base_bg: str = "#111827"  # even-row background (filled by ScriptEditor)
        self._alt_bg: str = "#1a1f2e"   # gutter background (set by ScriptEditor)
        self._row_border: str = ""      # optional per-row separator color
        self._stale_selections: list = []   # managed by ScriptEditor stale-var logic
        self._hook_selections:  list = []   # managed by highlight_lines()
        self._highlighter = PythonSyntaxHighlighter(self.document())
        self._completion_words = list(_SCRIPT_API_BUILTINS)
        self._completion_popup = QListWidget(self)
        self._completion_popup.setObjectName("scriptAutoComplete")
        self._completion_popup.hide()
        self._completion_popup.setFocusPolicy(Qt.NoFocus)
        self._completion_popup.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._completion_popup.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._completion_popup.setSelectionMode(QAbstractItemView.SingleSelection)
        self._completion_popup.itemClicked.connect(lambda *_: self._accept_completion())

        # Disable Qt's auto-fill so our event-filter paint is not overwritten
        self.viewport().setAutoFillBackground(False)
        # Last-installed filter runs first → we paint before Qt renders text
        self.viewport().installEventFilter(self)

        self.blockCountChanged.connect(self._on_block_count_changed)
        self.updateRequest.connect(self._update_line_number_area)
        self._on_block_count_changed(0)
        self.refresh_style()

    def refresh_style(self):
        """Refresh theme-driven editor chrome that is owned by PythonIndentEditor."""
        popup_bg = get_color("code_bg", get_color("cmd_bg", "#111827"))
        popup_text = get_color("code_editor_text", "#e5e7eb")
        popup_border = get_color("border", "#374151")
        popup_selected_bg = get_color("code_selection", get_color("hover_bg", "#1f2937"))
        popup_selected_text = get_color("syntax_system_builtin", "#f59e0b")

        self._completion_popup.setStyleSheet(
            "QListWidget#scriptAutoComplete {"
            f" background: {popup_bg}; color: {popup_text};"
            f" border: 1px solid {popup_border}; padding: 2px; }}"
            "QListWidget#scriptAutoComplete::item { padding: 4px 8px; }"
            "QListWidget#scriptAutoComplete::item:selected {"
            f" background: {popup_selected_bg}; color: {popup_selected_text}; }}"
        )

        if self._highlighter is not None:
            self._highlighter._build_rules()
            self._highlighter.rehighlight()
        self.viewport().update()
        self._line_number_area.update()

    def _current_completion_prefix(self) -> str:
        cur = self.textCursor()
        block_text = cur.block().text()
        col = cur.positionInBlock()
        if col <= 0:
            return ""
        left = block_text[:col]
        m = re.search(r"([A-Za-z_]\w*)$", left)
        return m.group(1) if m else ""

    def _update_completion_popup(self):
        prefix = self._current_completion_prefix()
        if len(prefix) < 1:
            self._completion_popup.hide()
            return
        matches = [w for w in self._completion_words if w.startswith(prefix)]
        if not matches:
            self._completion_popup.hide()
            return
        self._completion_popup.clear()
        for name in matches:
            self._completion_popup.addItem(QListWidgetItem(name))
        self._completion_popup.setCurrentRow(0)
        row_h = self._completion_popup.sizeHintForRow(0) if self._completion_popup.count() else 20
        row_h = max(20, row_h)
        popup_h = min(180, row_h * min(6, self._completion_popup.count()) + 6)
        popup_w = max(180, self._completion_popup.sizeHintForColumn(0) + 20)
        cursor_rect = self.cursorRect()
        self._completion_popup.setGeometry(
            cursor_rect.left(),
            cursor_rect.bottom() + 2,
            popup_w,
            popup_h,
        )
        self._completion_popup.show()
        self._completion_popup.raise_()

    def _move_completion_selection(self, delta: int):
        if not self._completion_popup.isVisible() or self._completion_popup.count() <= 0:
            return
        row = self._completion_popup.currentRow()
        if row < 0:
            row = 0
        row = (row + delta) % self._completion_popup.count()
        self._completion_popup.setCurrentRow(row)

    def _accept_completion(self):
        if not self._completion_popup.isVisible():
            return
        item = self._completion_popup.currentItem()
        if item is None:
            self._completion_popup.hide()
            return
        completion = item.text()
        prefix = self._current_completion_prefix()
        cur = self.textCursor()
        if prefix:
            cur.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(prefix))
            cur.removeSelectedText()
        cur.insertText(f"{completion}()")
        cur.movePosition(QTextCursor.Left)
        self.setTextCursor(cur)
        self._completion_popup.hide()

    # ── Enter ─────────────────────────────────────────────────────────────────

    def _handle_return(self):
        self._completion_popup.hide()
        cursor = self.textCursor()
        if cursor.hasSelection():
            cursor.removeSelectedText()
        line_text = cursor.block().text()
        new_indent = _compute_smart_indent(line_text)
        cursor.beginEditBlock()
        cursor.insertText("\n" + new_indent)
        cursor.endEditBlock()
        self.ensureCursorVisible()

    # ── Line number area ──────────────────────────────────────────────────────

    def lineNumberAreaWidth(self) -> int:
        """Gutter width in pixels, wide enough for the largest line number."""
        digits = max(1, len(str(max(1, self.blockCount()))))
        return 6 + self.fontMetrics().horizontalAdvance("9") * digits

    def _on_block_count_changed(self, _new_count: int = 0):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy: int):
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(), self._line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self._on_block_count_changed(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height())
        )
        if self._completion_popup.isVisible():
            self._update_completion_popup()

    def lineNumberAreaPaintEvent(self, event):
        """Paint line numbers inside the gutter (called by _LineNumberArea)."""
        gutter_bg = self._alt_bg or get_color("code_alt_bg", "#1a1f2e")
        gutter_text = get_color("gutter_text", "#6b7280")

        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor(gutter_bg))

        block        = self.firstVisibleBlock()
        block_number = block.blockNumber()
        offset       = self.contentOffset()
        top          = round(self.blockBoundingGeometry(block).translated(offset).top())
        bottom       = top + round(self.blockBoundingRect(block).height())
        line_h       = self.fontMetrics().height()
        gutter_width = self._line_number_area.width()

        painter.setPen(QColor(gutter_text))
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(
                    4, top,
                    max(0, gutter_width - 8), line_h,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    str(block_number + 1),
                )
                if self._row_border:
                    painter.setPen(QColor(self._row_border))
                    y = bottom - 1
                    painter.drawLine(0, y, gutter_width, y)
                    painter.setPen(QColor(gutter_text))
            block        = block.next()
            top          = bottom
            bottom       = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def focusOutEvent(self, event):
        self._completion_popup.hide()
        super().focusOutEvent(event)

    # ── External line highlight hook ──────────────────────────────────────────

    def highlight_lines(self, line_numbers: list, style_key: str = "error"):
        """Highlight specific lines from outside (diagnostics / runtime tracing).

        Args:
            line_numbers: 1-based line numbers to highlight.
                          Pass an empty list to clear all hook highlights.
            style_key:    One of "error", "warning", "info" — controls colours.
                          Unknown keys fall back to "info".
        """
        fmt        = self._get_highlight_format(style_key)
        selections = []
        doc        = self.document()
        for lineno in (line_numbers or []):
            block = doc.findBlockByLineNumber(lineno - 1)
            if not block.isValid():
                continue
            sel            = QTextEdit.ExtraSelection()
            sel.format     = fmt
            sel.cursor     = QTextCursor(block)
            sel.cursor.select(QTextCursor.LineUnderCursor)
            selections.append(sel)
        self._hook_selections = selections
        self._apply_extra_selections()

    def _get_highlight_format(self, style_key: str) -> QTextCharFormat:
        _STYLES = {
            "error": (
                get_color("editor_error_bg", "#7c2d12"),
                get_color("editor_error_underline", "#ef4444"),
            ),
            "warning": (
                get_color("editor_warning_bg", "#713f12"),
                get_color("editor_warning_underline", "#eab308"),
            ),
            "info": (
                get_color("editor_info_bg", "#1e3a5f"),
                get_color("editor_info_underline", "#60a5fa"),
            ),
        }
        bg_hex, ul_hex = _STYLES.get(style_key, _STYLES["info"])
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(bg_hex))
        fmt.setUnderlineColor(QColor(ul_hex))
        fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)
        return fmt

    def set_stale_selections(self, selections: list):
        """Replace stale-variable highlight layer (called by ScriptEditor)."""
        self._stale_selections = list(selections)
        self._apply_extra_selections()

    def _apply_extra_selections(self):
        """Merge stale-var and hook layers, then commit to the editor."""
        self.setExtraSelections(self._stale_selections + self._hook_selections)


# ── Template helpers ─────────────────────────────────────────────────────────

def _build_header(script_name: str, io_spec: dict) -> str:
    inputs = (io_spec or {}).get("inputs", []) or []
    bar = "─" * max(0, 48 - len(script_name))
    lines = [
        f"# ─── Script: {script_name} {bar}",
        "# Inputs (from Canvas):",
    ]
    if inputs:
        for inp in inputs:
            name = (inp.get("name") or "").strip() or "?"
            dtype = (inp.get("data_type") or "any").strip()
            lines.append(f"#   {name:<16}: {dtype}")
    else:
        lines.append("#   (none)")
    lines.append("# " + "─" * 52)
    return "\n".join(lines)


def _build_def_line(io_spec: dict) -> str:
    inputs = (io_spec or {}).get("inputs", []) or []
    if not inputs:
        return "def init():"
    lines = ["def init("]
    for i, inp in enumerate(inputs):
        name = (inp.get("name") or "").strip() or f"in_{i}"
        dtype = (inp.get("data_type") or "any").strip() or "any"
        lines.append(f"    {name},  # {dtype}")
    lines.append("):")
    return "\n".join(lines)


def _generate_template(script_name: str, io_spec: dict) -> str:
    header = _build_header(script_name, io_spec)
    init_line = _build_def_line(io_spec)
    input_names = [
        (inp.get("name") or "").strip() or f"in_{i}"
        for i, inp in enumerate((io_spec or {}).get("inputs", []) or [])
    ]
    process_call = ", ".join(input_names)
    process_sig = process_call
    body = (
        f"{init_line}\n"
        f'    """Entry point for \'{script_name}\'. You can call any custom method here."""\n'
        f"    result = process({process_call})\n"
        "    return result  # return in init() can declare outputs\n\n"
        f"def process({process_sig}):\n"
        "    # ── implement logic here ──────────────────────────\n"
        "    result = None\n"
        "    return result\n"
    )
    return f"{header}\n{body}"


def _find_entry_def_range(lines: List[str]) -> tuple:
    """Return (start_idx, end_idx) for the first entry function signature block."""
    start_idx = next(
        (
            i for i, ln in enumerate(lines)
            if ln.strip().startswith(_ENTRY_DEF_PREFIXES)
            or ln.strip().startswith("def ")
        ),
        None,
    )
    if start_idx is None:
        return None, None
    start_line = lines[start_idx].strip()
    if start_line.endswith("):") or start_line.endswith(":"):
        return start_idx, start_idx
    for j in range(start_idx + 1, len(lines)):
        ln = lines[j].strip()
        if ln == "):" or ln.endswith("):"):
            return start_idx, j
    return start_idx, start_idx


# ── Return-value parser ───────────────────────────────────────────────────────

def _parse_return_outputs(code: str, func_name: str = "init") -> List[Dict]:
    """Parse the last return statement inside *func_name* → [{name, data_type}]."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if not returns:
                return []
            return _extract_return_names(returns[-1].value)
    return []


def _extract_return_names(value_node) -> List[Dict]:
    if value_node is None:
        return []
    if isinstance(value_node, ast.Name):
        return [{"name": value_node.id, "data_type": "any"}]
    if isinstance(value_node, ast.Tuple):
        result = []
        for i, elt in enumerate(value_node.elts):
            if isinstance(elt, ast.Name):
                result.append({"name": elt.id, "data_type": "any"})
            else:
                result.append({"name": f"out_{i}", "data_type": "any"})
        return result
    if isinstance(value_node, ast.Dict):
        result = []
        for key in value_node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                result.append({"name": key.value, "data_type": "any"})
        return result or [{"name": "result", "data_type": "any"}]
    return [{"name": "result", "data_type": "any"}]


# ── ScriptEditor widget ───────────────────────────────────────────────────────

class ScriptEditor(QWidget):
    """Editable Python script editor attached to one Script Box node."""

    # Emitted after successful Save & Compile: (node_id, compile_result)
    # compile_result keys: inputs, outputs, diagnostics, compat_used
    compile_requested = Signal(int, dict)

    def __init__(
        self,
        node_id: int,
        script_name: str,
        io_spec: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._node_id = node_id
        self._script_name = script_name
        self._io_spec: dict = dict(io_spec) if io_spec else {"inputs": [], "outputs": []}
        self._init_ui()
        self._apply_style()
        self._editor.setPlainText(_generate_template(self._script_name, self._io_spec))

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── header bar ───────────────────────────────────────────────────────
        self._header = QWidget()
        self._header.setObjectName("scriptEditorHeader")
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.setSpacing(8)

        self._title_label = QLabel(self._script_name)
        self._title_label.setObjectName("scriptEditorTitle")
        hl.addWidget(self._title_label)
        hl.addStretch(1)

        self._save_btn = QPushButton("Save && Compile")
        self._save_btn.setObjectName("scriptSaveBtn")
        self._save_btn.setFixedHeight(28)
        self._save_btn.setCursor(Qt.PointingHandCursor)
        self._save_btn.setToolTip("Check syntax and update Script Box outputs")
        self._save_btn.clicked.connect(self.save_and_compile)
        hl.addWidget(self._save_btn)
        layout.addWidget(self._header)

        # ── stale-variable warning bar (hidden by default) ────────────────
        self._warn_bar = QLabel("")
        self._warn_bar.setObjectName("scriptWarnBar")
        self._warn_bar.setWordWrap(True)
        self._warn_bar.setVisible(False)
        layout.addWidget(self._warn_bar)

        # ── resizable center: editor + custom function panel ─────────────
        self._center_splitter = QSplitter(Qt.Vertical)
        self._center_splitter.setChildrenCollapsible(False)

        # editor body
        self._editor = PythonIndentEditor()
        self._editor.setObjectName("scriptEditorBody")
        font = QFont("Courier New", get_font_size("size_small", 11))
        font.setStyleHint(QFont.Monospace)
        self._editor.setFont(font)
        self._center_splitter.addWidget(self._editor)

        # custom function panel
        self._builtin_panel = QFrame()
        self._builtin_panel.setObjectName("scriptBuiltinPanel")
        bl = QVBoxLayout(self._builtin_panel)
        bl.setContentsMargins(10, 8, 10, 10)
        bl.setSpacing(6)
        self._builtin_title = QLabel("Custom Functions")
        self._builtin_title.setObjectName("scriptBuiltinTitle")
        bl.addWidget(self._builtin_title)
        self._builtin_table = QTableWidget(0, 3, self._builtin_panel)
        self._builtin_table.setObjectName("scriptBuiltinTable")
        self._builtin_table.setHorizontalHeaderLabels(["Functions", "Ex.", "Desc"])
        self._builtin_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._builtin_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._builtin_table.setFocusPolicy(Qt.NoFocus)
        self._builtin_table.verticalHeader().setVisible(False)
        hdr = self._builtin_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.Interactive)
        self._builtin_table.setWordWrap(False)
        self._builtin_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._builtin_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._builtin_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self._builtin_table.verticalHeader().setDefaultSectionSize(26)
        self._builtin_table.setAlternatingRowColors(False)
        bl.addWidget(self._builtin_table)
        self._center_splitter.addWidget(self._builtin_panel)
        self._center_splitter.setStretchFactor(0, 4)
        self._center_splitter.setStretchFactor(1, 1)
        self._center_splitter.splitterMoved.connect(lambda *_: self._fit_and_stretch_builtin_columns())
        layout.addWidget(self._center_splitter, 1)
        self._populate_builtin_table()
        # First paint/layout pass may report stale viewport width; re-fit after settle.
        QTimer.singleShot(0, self._fit_and_stretch_builtin_columns)
        QTimer.singleShot(30, self._fit_and_stretch_builtin_columns)

    def _populate_builtin_table(self):
        self._builtin_table.setRowCount(len(SCRIPT_BUILTIN_FUNCTIONS))
        for row, spec in enumerate(SCRIPT_BUILTIN_FUNCTIONS):
            name_item = QTableWidgetItem(spec.name)
            example_item = QTableWidgetItem(spec.example)
            desc_item = QTableWidgetItem(spec.description)
            self._builtin_table.setItem(row, 0, name_item)
            self._builtin_table.setItem(row, 1, example_item)
            self._builtin_table.setItem(row, 2, desc_item)
        self._fit_and_stretch_builtin_columns()

    def _fit_and_stretch_builtin_columns(self):
        """Fit to content first, then stretch all columns to fill available width."""
        table = self._builtin_table
        if table.columnCount() <= 0:
            return
        table.resizeColumnsToContents()
        widths = [table.columnWidth(i) for i in range(table.columnCount())]
        base_total = sum(widths)
        if base_total <= 0:
            return
        avail = max(0, table.viewport().width())
        if avail <= 0 or base_total >= avail:
            return
        extra = avail - base_total
        new_widths = []
        used = 0
        for w in widths:
            add = int(extra * (w / base_total))
            new_w = w + add
            new_widths.append(new_w)
            used += new_w
        if new_widths:
            new_widths[-1] += max(0, avail - used)
        for i, w in enumerate(new_widths):
            table.setColumnWidth(i, w)

    def _apply_style(self):
        header_bg = get_color("code_header_bg", get_color("card_bg", "#1f2937"))
        header_border = get_color("border", "#374151")
        title_text = get_color("code_title_text", "#e5e7eb")
        editor_bg = get_color("code_bg", get_color("cmd_bg", "#111827"))
        alt_bg = get_color("code_alt_bg", editor_bg)
        editor_text = get_color("code_editor_text", "#e5e7eb")
        sel_bg = get_color("code_selection", get_color("hover_bg", "#374151"))
        accent = get_color("accent", "#3b82f6")
        accent_hover = get_color("accent_hover", "#2563eb")
        accent_text = get_color("accent_text", get_color("text_on_accent", "#ffffff"))
        warning_bg = get_color("editor_warning_bg", "#7c2d12")
        warning_text = get_color("editor_warning_text", "#fde68a")
        size_normal = get_font_size("size_normal", 12)
        size_small = get_font_size("size_small", 11)

        self._header.setStyleSheet(
            f"QWidget#scriptEditorHeader {{"
            f"  background: {alt_bg};"
            f"  border-bottom: 1px solid {header_border};"
            f"}}"
        )
        self._title_label.setStyleSheet(
            f"QLabel {{ color: {title_text}; font-size: {size_normal}px;"
            f" font-weight: bold; background: transparent; }}"
        )
        self._save_btn.setStyleSheet(
            f"QPushButton {{ background: {accent}; color: {accent_text}; border: none;"
            f" border-radius: 4px; padding: 4px 12px; font-size: {size_small}px; }}"
            f"QPushButton:hover {{ background: {accent_hover}; }}"
        )
        self._warn_bar.setStyleSheet(
            f"QLabel {{ background: {warning_bg}; color: {warning_text};"
            f" padding: 4px 10px; font-size: {size_small}px; }}"
        )
        # background is painted by PythonIndentEditor.eventFilter — not via stylesheet
        self._editor.setStyleSheet(
            f"QPlainTextEdit {{ color: {editor_text};"
            f" border: none; font-family: 'Courier New', Consolas, monospace;"
            f" padding: 12px;"
            f" selection-background-color: {sel_bg}; }}"
        )
        self._builtin_panel.setStyleSheet(
            f"QFrame#scriptBuiltinPanel {{ "
            f"background: {editor_bg}; border-top: 1px solid {header_border}; }}"
            f"QLabel#scriptBuiltinTitle {{ color: {title_text}; font-size: {size_small}px; "
            f"font-weight: bold; background: transparent; }}"
        )
        self._builtin_table.setStyleSheet(
            f"QTableWidget#scriptBuiltinTable {{ "
            f"background: {editor_bg}; color: {editor_text}; "
            f"gridline-color: {header_border}; border: 1px solid {header_border}; "
            f"font-size: {size_small}px; }}"
            f"QHeaderView::section {{ background: {alt_bg}; color: {title_text}; "
            f"border: 1px solid {header_border}; padding: 4px; }}"
        )
        self._editor.set_row_colors(editor_bg, alt_bg, header_border)

    def refresh_style(self):
        self._editor.refresh_style()
        self._apply_style()
        self._fit_and_stretch_builtin_columns()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_and_stretch_builtin_columns()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_signature(self, new_io_spec: dict):
        """Called when Canvas ports change.

        Replaces only the header comment block and entry function signature block.
        The function body is preserved unchanged.
        Highlights lines that still reference variables removed from Canvas.
        """
        old_names = {
            (inp.get("name") or "").strip()
            for inp in (self._io_spec.get("inputs") or [])
        }
        new_names = {
            (inp.get("name") or "").strip()
            for inp in (new_io_spec.get("inputs") or [])
        }
        removed = {n for n in (old_names - new_names) if n}
        self._io_spec = dict(new_io_spec)

        current_text = self._editor.toPlainText()
        lines = current_text.splitlines()

        def_start, def_end = _find_entry_def_range(lines)
        body_lines = lines[def_end + 1:] if def_end is not None else lines

        new_header = _build_header(self._script_name, new_io_spec)
        new_def = _build_def_line(new_io_spec)
        new_text = new_header + "\n" + new_def + "\n" + "\n".join(body_lines)

        # Preserve cursor position (clamped)
        cursor_pos = self._editor.textCursor().position()
        self._editor.setPlainText(new_text)
        cur = self._editor.textCursor()
        cur.setPosition(min(cursor_pos, len(new_text)))
        self._editor.setTextCursor(cur)

        if removed:
            self._highlight_stale_vars(removed, new_text)
        else:
            self._editor.set_stale_selections([])
            self._warn_bar.setVisible(False)

    def set_script_name(self, script_name: str) -> None:
        """Update editor-local script name display without touching code content."""
        self._script_name = str(script_name or "").strip() or self._script_name
        self._title_label.setText(self._script_name)

    def save_and_compile(self, show_dialogs: bool = True):
        """Analyze the script via API analyzer, emit compile_requested with full IO spec.

        Primary path: setOutput/getInput calls in selected entry function → API mode.
        Compat path: no setOutput found → entry-return fallback with warning.
        """
        code = self._editor.toPlainText()

        # 1. Run API analyzer (syntax check is included)
        result = analyze_script(code)

        # 2. Hard stop: syntax error
        syntax_diag = next(
            (d for d in result["diagnostics"] if d["code"] == ScriptDiagCode.SYNTAX_ERROR),
            None,
        )
        if syntax_diag is not None:
            if show_dialogs:
                QMessageBox.warning(
                    self,
                    "Syntax Error",
                    f"Cannot compile — syntax error:\n\nLine {syntax_diag['lineno']}: "
                    f"{syntax_diag['message']}",
                )
            self._jump_to_line(syntax_diag["lineno"] or 1)
            return

        # 3. Hard stop: no recognized entry function
        if any(d["code"] == ScriptDiagCode.NO_RUN_FUNCTION for d in result["diagnostics"]):
            if show_dialogs:
                QMessageBox.warning(
                    self,
                    "Compile Error",
                    "Cannot compile — no entry function found.\n"
                    "Define 'def init(...)' (preferred), 'def run(...)' (legacy), "
                    "or keep exactly one top-level function.",
                )
            return

        # 4. Compat fallback: no setOutput() → use entry function return parser
        if result["compat_used"]:
            entry_func = str(result.get("entry_function") or "init")
            compat_outputs = _parse_return_outputs(code, func_name=entry_func)
            result = dict(result, outputs=compat_outputs)
            log_warning(
                f"[ScriptEditor] '{self._script_name}': compat mode — "
                f"no setOutput() found, return parser yielded "
                f"{len(compat_outputs)} output(s): "
                f"{[o['name'] for o in compat_outputs]}"
            )
        else:
            log_info(
                f"[ScriptEditor] '{self._script_name}': API mode — "
                f"{len(result['outputs'])} output(s), "
                f"{len(result['inputs'])} input(s)"
            )

        # Preserve existing canvas-declared inputs when script does not declare
        # getInput() calls; this prevents accidental input edge drops on save.
        if not (result.get("inputs") or []):
            preserved_inputs = []
            for inp in list(self._io_spec.get("inputs") or []):
                if not isinstance(inp, dict):
                    continue
                name = str(inp.get("name") or "").strip()
                if not name:
                    continue
                preserved_inputs.append(
                    {
                        "name": name,
                        "data_type": str(inp.get("data_type") or "any"),
                        "slot": inp.get("slot"),
                    }
                )
            result = dict(result, inputs=preserved_inputs)

        # 5. Intercept API argument errors before mutating UI ports.
        error_diags = [
            d for d in result["diagnostics"]
            if d.get("level") == "error"
            and d.get("code") not in (ScriptDiagCode.SYNTAX_ERROR, ScriptDiagCode.NO_RUN_FUNCTION)
        ]
        if error_diags:
            for diag in error_diags:
                lineno_str = f" (line {diag['lineno']})" if diag["lineno"] else ""
                log_error(f"[ScriptEditor] {diag['code']}{lineno_str}: {diag['message']}")
            first = error_diags[0]
            if show_dialogs:
                QMessageBox.warning(
                    self,
                    "Compile Error",
                    f"Invalid Script API arguments at line {first.get('lineno') or '?'}:\n\n"
                    f"{first.get('message')}",
                )
            if first.get("lineno"):
                self._jump_to_line(first["lineno"])
            return

        # 6. Log non-fatal diagnostics (warnings, excluding compat notice)
        for diag in result["diagnostics"]:
            if diag["code"] == ScriptDiagCode.COMPAT_RETURN_USED:
                continue
            lineno_str = f" (line {diag['lineno']})" if diag["lineno"] else ""
            if diag.get("level") == "error":
                log_error(f"[ScriptEditor] {diag['code']}{lineno_str}: {diag['message']}")
            else:
                log_warning(f"[ScriptEditor] {diag['code']}{lineno_str}: {diag['message']}")

        self.compile_requested.emit(self._node_id, result)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _highlight_stale_vars(self, removed_names: set, full_text: str):
        """Mark function-body lines that still reference removed input names."""
        lines = full_text.splitlines()
        _, def_end = _find_entry_def_range(lines)
        body_start = (def_end + 1) if def_end is not None else 0
        patterns = {
            name: re.compile(r"\b" + re.escape(name) + r"\b")
            for name in removed_names
        }
        stale_lines: Dict[int, List[str]] = {}
        for i, ln in enumerate(lines[body_start:], start=body_start):
            for name, pat in patterns.items():
                if pat.search(ln):
                    stale_lines.setdefault(i, []).append(name)

        if not stale_lines:
            self._editor.set_stale_selections([])
            self._warn_bar.setVisible(False)
            return

        warn_fmt = QTextCharFormat()
        warn_fmt.setBackground(QColor(get_color("editor_warning_bg", "#7c2d12")))
        warn_fmt.setUnderlineColor(QColor(get_color("editor_warning_underline", "#eab308")))
        warn_fmt.setUnderlineStyle(QTextCharFormat.WaveUnderline)

        selections = []
        doc = self._editor.document()
        for line_idx in stale_lines:
            block = doc.findBlockByLineNumber(line_idx)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format = warn_fmt
            sel.cursor = QTextCursor(block)
            sel.cursor.select(QTextCursor.LineUnderCursor)
            selections.append(sel)

        self._editor.set_stale_selections(selections)

        names_str = ", ".join(f"'{n}'" for n in sorted(removed_names))
        line_nums = ", ".join(str(ln + 1) for ln in sorted(stale_lines))
        self._warn_bar.setText(
            f"\u26a0  Variable(s) {names_str} were removed from Canvas "
            f"but are still referenced in the script (line {line_nums}). "
            f"Please update or remove these references."
        )
        self._warn_bar.setVisible(True)

        if stale_lines:
            self._jump_to_line(min(stale_lines) + 1)

    def _jump_to_line(self, line_no: int):
        """Move editor cursor to *line_no* (1-based) and centre the view."""
        doc = self._editor.document()
        block = doc.findBlockByLineNumber(max(0, line_no - 1))
        if block.isValid():
            cur = QTextCursor(block)
            self._editor.setTextCursor(cur)
            self._editor.centerCursor()
