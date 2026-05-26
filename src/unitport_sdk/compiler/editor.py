# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CodeEditorWidget — reusable code editor with line numbers, smart indent,
syntax highlighting, auto-completion popup, and DataManager-backed file I/O.

Migrated from DEMO ``bin/components/code_editor_widget.py`` (PySide6) to
PyQt6 + ``unitport_sdk``. Public API is a strict superset of the DEMO version
so existing call sites only need to swap the import path.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Union

from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygon,
    QTextBlock,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

from ..logger import log_error, log_warning
from ..sys import Config, DataManager
from .fold_area import _FoldArea
from .highlighters import JsonSyntaxHighlighter, PythonSyntaxHighlighter
from .indent import _compute_smart_indent
from .line_number_area import _LineNumberArea

PathLike = Union[str, Path]

_PY_EXTS = {".py", ".pyi", ".pyw"}
_JSON_EXTS = {".json", ".jsonc"}


def _detect_mode(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _PY_EXTS:
        return "python"
    if ext in _JSON_EXTS:
        return "json"
    return "plain"


class CodeEditorWidget(QPlainTextEdit):
    """Reusable code editor with line numbers and smart indentation.

    Features
    --------
    * Line number gutter (theme-aware)
    * Python or JSON syntax highlighting (mode-selectable)
    * Smart indentation — ``Tab``/``Shift+Tab``, auto-indent after ``:``
    * ``Ctrl+Wheel`` font zoom (6 – 40 pt)
    * Optional auto-completion popup
    * External line-highlight hooks (error / warning / info)
    * DataManager-backed file I/O (``load_file`` / ``save_file``)

    Parameters
    ----------
    parent : QWidget | None
    completion_words : list[str] | None
        Words for the auto-completion popup. Pass ``None`` to disable.
    extra_builtins : list[str] | None
        Additional built-in names for Python syntax highlighting.
    read_only : bool
        Start in read-only mode.
    mode : str
        ``"python"``, ``"json"`` or ``"plain"`` — selects the highlighter.

    Public methods (DEMO-compatible plus new file I/O)
    --------------------------------------------------
    ``set_completion_words``, ``set_row_colors``, ``set_extra_builtins``,
    ``highlight_lines``, ``set_extra_selections_layer1``, ``jump_to_line``,
    ``refresh_style``, ``line_number_area_width``,
    ``line_number_area_paint_event``,
    ``load_file``, ``save_file``, ``set_text``, ``text``,
    ``current_path``, ``is_modified``, ``set_mode``,
    ``undo``, ``redo`` (inherited), ``toggle_line_comment``.

    Signals
    -------
    ``dirtyChanged(bool)`` — emitted whenever the dirty state flips. Dirty
    state is computed from a SHA-1 baseline taken at ``__init__`` /
    ``set_text`` / ``load_file`` / ``save_file``, so undoing back to the
    saved content clears the flag.
    """

    dirtyChanged = pyqtSignal(bool)

    _ZOOM_MIN = 6
    _ZOOM_MAX = 40
    _TEXT_PADDING = 8  # px — internal text margin via documentMargin
    _FOLD_AREA_W = 14  # px — fold-button gutter width (font-independent)
    _INDENT_UNIT = 4   # spaces per indent level (matches _handle_tab insertion)
    _COMMENT_MARKERS = {"python": "# ", "json": "// ", "plain": "# "}

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        completion_words: Optional[List[str]] = None,
        extra_builtins: Optional[List[str]] = None,
        read_only: bool = False,
        mode: str = "python",
    ):
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setReadOnly(read_only)

        # Line number gutter + fold-button gutter
        self._line_number_area = _LineNumberArea(self)
        self._fold_area = _FoldArea(self)

        # Colors — initialised from theme, overridable via set_row_colors()
        self._base_bg: str = Config.get_color("canvas_bg", "#111827")
        self._alt_bg: str = Config.get_color("hover_2", "#1A1F2E")
        self._row_border: str = Config.get_color("border_1", "#374151")
        # Fold/indent visuals — refreshed in _apply_theme_style()
        self._indent_guide_color: str = Config.get_color("editor_indent_guide", "#B0B0B0")
        self._fold_marker_color: str = Config.get_color("editor_fold_marker", "#9CA3AF")
        self._fold_marker_hover_color: str = Config.get_color("editor_fold_marker_hover", "#E5E7EB")
        self._fold_scope_line_color: str = Config.get_color("checked_1", "#67CCF5")
        self._fold_button_bg_color: str = Config.get_color("bg_2", "#1A1A1A")
        self._fold_button_border_color: str = Config.get_color("border_2", "#3D3D3D")

        # Fold state (block-number set; remapped on contentsChange)
        self._folded_headers: set[int] = set()
        self._fold_hover_block: Optional[int] = None
        self._prev_block_count: int = 1

        # Dirty tracking — hash-based, so undoing back to baseline clears it.
        self._baseline_hash: str = self._hash_text("")
        self._baseline_len: int = 0
        self._dirty: bool = False
        self.textChanged.connect(self._recompute_dirty)

        # Extra selection layers
        self._current_line_selection: list = []
        self._extra_layer_1: list = []   # external (e.g. stale-var)
        self._extra_layer_2: list = []   # error/warning/info hooks

        # Syntax highlighter — created here, swapped by set_mode()
        self._mode: str = "python"
        self._extra_builtins_init = list(extra_builtins or [])
        self._highlighter = None  # type: ignore[assignment]
        self._install_highlighter(mode)

        # Auto-completion popup
        self._completion_words: list = list(completion_words or [])
        self._completion_popup = QListWidget(self)
        self._completion_popup.setObjectName("codeEditorAutoComplete")
        self._completion_popup.hide()
        self._completion_popup.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._completion_popup.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._completion_popup.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._completion_popup.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._completion_popup.itemClicked.connect(lambda *_: self._accept_completion())

        # File I/O state
        self._current_path: Optional[Path] = None

        # Viewport painting
        self.viewport().setAutoFillBackground(False)
        self.viewport().installEventFilter(self)

        self.blockCountChanged.connect(self._on_block_count_changed)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._update_current_line_highlight)
        self.document().contentsChange.connect(self._on_contents_change)
        self._prev_block_count = self.blockCount()
        self._on_block_count_changed(0)

        # Default font
        font = QFont("Courier New", Config.get_font_size("size_small", 11))
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

        # Use documentMargin for internal text padding so the line-number
        # gutter (set via setViewportMargins) doesn't overlap the text area.
        self.document().setDocumentMargin(self._TEXT_PADDING)

        self._apply_theme_style()
        self.refresh_style()

        # Some setup paths (highlighter attach, stylesheet pass) can leave the
        # fresh document flagged modified — align baseline with current text
        # so is_modified() / dirtyChanged reflect only user edits.
        self.document().setModified(False)
        self._refresh_baseline()

    # == Highlighter mode =======================================================

    def _install_highlighter(self, mode: str) -> None:
        """Attach the highlighter for *mode*. Replaces any existing one."""
        # Detach previous highlighter from the document
        if self._highlighter is not None:
            self._highlighter.setDocument(None)
            self._highlighter = None

        m = (mode or "plain").lower()
        if m == "python":
            self._highlighter = PythonSyntaxHighlighter(
                self.document(), extra_builtins=self._extra_builtins_init,
            )
            self._mode = "python"
        elif m == "json":
            self._highlighter = JsonSyntaxHighlighter(self.document())
            self._mode = "json"
        else:
            self._highlighter = None
            self._mode = "plain"

    def set_mode(self, mode: str) -> None:
        """Switch the active syntax highlighter (``python`` / ``json`` / ``plain``)."""
        if mode == self._mode:
            return
        self._install_highlighter(mode)
        self.viewport().update()

    # == Theme ==================================================================

    def _apply_theme_style(self) -> None:
        """Apply the full theme colour scheme from Config."""
        editor_text = Config.get_color("code_editor_text", "#e5e7eb")
        editor_bg = Config.get_color("canvas_bg", "#111827")
        alt_bg = Config.get_color("hover_2", editor_bg)
        border = Config.get_color("border_1", "#374151")
        sel_bg = Config.get_color("code_selection", "#374151")

        # Fold/indent visuals — cache per-frame for paint hot-paths
        self._indent_guide_color = Config.get_color("editor_indent_guide", "#B0B0B0")
        self._fold_marker_color = Config.get_color("editor_fold_marker", "#9CA3AF")
        self._fold_marker_hover_color = Config.get_color("editor_fold_marker_hover", "#E5E7EB")
        self._fold_scope_line_color = Config.get_color("checked_1", "#67CCF5")
        self._fold_button_bg_color = Config.get_color("bg_2", "#1A1A1A")
        self._fold_button_border_color = Config.get_color("border_2", "#3D3D3D")

        # CSS padding is intentionally omitted — we use documentMargin so the
        # line-number gutter and the text content area never overlap.
        self.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  color: {editor_text};"
            f"  border: none;"
            f"  font-family: 'Courier New', Consolas, monospace;"
            f"  selection-background-color: {sel_bg};"
            f"}}"
        )
        self.set_row_colors(editor_bg, alt_bg, border)

    # == Public API =============================================================

    def set_completion_words(self, words: List[str]) -> None:
        """Replace the auto-completion word list."""
        self._completion_words = list(words)

    def set_row_colors(self, base_bg: str, alt_bg: str, row_border: str = "") -> None:
        """Set editor background, gutter background and optional row separator."""
        self._base_bg = base_bg
        self._alt_bg = alt_bg
        self._row_border = row_border
        self.viewport().update()
        self._line_number_area.update()

    def set_extra_builtins(self, names: List[str]) -> None:
        """Forward to the Python highlighter (no-op in JSON/plain modes)."""
        self._extra_builtins_init = list(names)
        if isinstance(self._highlighter, PythonSyntaxHighlighter):
            self._highlighter.set_extra_builtins(names)

    def highlight_lines(self, line_numbers: list, style_key: str = "error") -> None:
        """Highlight specific lines (1-based). Pass an empty list to clear.

        ``style_key``: ``"error"`` | ``"warning"`` | ``"info"``.
        """
        fmt = self._get_highlight_format(style_key)
        selections = []
        doc = self.document()
        for lineno in (line_numbers or []):
            block = doc.findBlockByLineNumber(lineno - 1)
            if not block.isValid():
                continue
            sel = QTextEdit.ExtraSelection()
            sel.format = fmt
            sel.cursor = QTextCursor(block)
            sel.cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            selections.append(sel)
        self._extra_layer_2 = selections
        self._apply_extra_selections()

    def set_extra_selections_layer1(self, selections: list) -> None:
        """Replace extra-selection layer 1 (e.g. stale-variable highlights)."""
        self._extra_layer_1 = list(selections)
        self._apply_extra_selections()

    def refresh_style(self) -> None:
        """Refresh all theme-driven visuals (editor + popup + highlighter)."""
        self._apply_theme_style()

        popup_bg = Config.get_color("canvas_bg", "#111827")
        popup_text = Config.get_color("code_editor_text", "#e5e7eb")
        popup_border = Config.get_color("border_1", "#374151")
        popup_sel_bg = Config.get_color("code_selection", "#1f2937")
        popup_sel_text = Config.get_color("syntax_system_builtin", "#f59e0b")

        self._completion_popup.setStyleSheet(
            "QListWidget#codeEditorAutoComplete {"
            f" background: {popup_bg}; color: {popup_text};"
            f" border: 1px solid {popup_border}; padding: 2px; }}"
            "QListWidget#codeEditorAutoComplete::item { padding: 4px 8px; }"
            "QListWidget#codeEditorAutoComplete::item:selected {"
            f" background: {popup_sel_bg}; color: {popup_sel_text}; }}"
        )

        if self._highlighter is not None:
            if hasattr(self._highlighter, "_build_rules"):
                self._highlighter._build_rules()
            self._highlighter.rehighlight()
        self.viewport().update()
        self._line_number_area.update()
        self._fold_area.update()

    def jump_to_line(self, line_no: int) -> None:
        """Move cursor to *line_no* (1-based) and centre the view.

        If the target line sits inside one or more folded regions, those
        regions are unfolded first so the cursor lands on a visible row.
        Resolution is by absolute block number (line N = block N-1) so the
        result is fold-state-independent — unlike ``findBlockByLineNumber``,
        which counts only visible lines.
        """
        block = self.document().findBlockByNumber(max(0, line_no - 1))
        if not block.isValid():
            return
        if not block.isVisible():
            self._unfold_to_make_visible(block.blockNumber())
        cur = QTextCursor(block)
        self.setTextCursor(cur)
        self.centerCursor()

    # == Public API: file I/O via DataManager ===================================

    def load_file(self, path: PathLike, *, mode: Optional[str] = None) -> bool:
        """Read *path* into the editor via ``DataManager``.

        ``mode`` chooses the highlighter; if ``None``, it is inferred from
        the file extension (``.py`` → python, ``.json`` → json, else plain).

        Returns ``True`` on success. On failure, logs via ``log_error`` and
        returns ``False`` — the editor contents are left unchanged.
        """
        p = Path(path)
        try:
            text = DataManager.read(p, format=".txt")
        except Exception as exc:
            log_error(f"[code_editor] load_file failed: {p} ({exc})")
            return False
        if not isinstance(text, str):
            log_warning(f"[code_editor] load_file: unexpected type from DataManager: {type(text).__name__}")
            text = str(text)
        self.set_mode(mode or _detect_mode(p))
        # Block textChanged so the load doesn't briefly flip dirty=True
        # before _refresh_baseline reasserts the new baseline.
        with QSignalBlocker(self):
            self.setPlainText(text)
        self.document().setModified(False)
        self._refresh_baseline()
        self._current_path = p
        return True

    def save_file(self, path: Optional[PathLike] = None) -> bool:
        """Write current text to *path* via ``DataManager`` (atomic).

        If *path* is ``None``, reuses the path most recently passed to
        ``load_file`` / ``save_file``. JSON files (by extension or current
        mode) are validated with ``json.loads`` first; on parse failure
        the call returns ``False`` without touching the file.
        """
        target: Optional[Path] = Path(path) if path is not None else self._current_path
        if target is None:
            log_error("[code_editor] save_file called with no path and no last-loaded path")
            return False

        text = self.text()

        # JSON validation — guard before atomic write
        if target.suffix.lower() in _JSON_EXTS or self._mode == "json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                log_error(f"[code_editor] JSON parse failed, save aborted: {exc}")
                return False

        try:
            ok = DataManager.write(target, text, format=".txt")
        except Exception as exc:
            log_error(f"[code_editor] save_file failed: {target} ({exc})")
            return False
        if not ok:
            log_error(f"[code_editor] save_file: DataManager.write returned False for {target}")
            return False

        self._current_path = target
        self.document().setModified(False)
        self._refresh_baseline()
        return True

    def set_text(self, text: str) -> None:
        """Replace editor content and reset the modified flag."""
        with QSignalBlocker(self):
            self.setPlainText(text)
        self.document().setModified(False)
        self._refresh_baseline()

    def text(self) -> str:
        """Return current editor content (alias for ``toPlainText()``)."""
        return self.toPlainText()

    def current_path(self) -> Optional[Path]:
        """Return the path most recently loaded/saved, or ``None``."""
        return self._current_path

    def is_modified(self) -> bool:
        """Return whether the document has unsaved changes.

        Backed by a SHA-1 baseline (refreshed on load/save/set_text), not
        the Qt ``QTextDocument.isModified`` flag — so undoing back to the
        last saved content clears this.
        """
        return self._dirty

    # == Private: dirty-state baseline ==========================================

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()

    def _refresh_baseline(self) -> None:
        """Snapshot current text as the clean baseline; emit dirtyChanged(False)."""
        text = self.toPlainText()
        self._baseline_hash = self._hash_text(text)
        self._baseline_len = len(text)
        if self._dirty:
            self._dirty = False
            self.dirtyChanged.emit(False)

    def _recompute_dirty(self) -> None:
        """Compare current text against baseline; emit dirtyChanged on flip."""
        text = self.toPlainText()
        if len(text) != self._baseline_len:
            new_dirty = True
        else:
            new_dirty = self._hash_text(text) != self._baseline_hash
        if new_dirty != self._dirty:
            self._dirty = new_dirty
            self.dirtyChanged.emit(new_dirty)

    # == Line number area =======================================================

    def line_number_area_width(self) -> int:
        digits = max(1, len(str(max(1, self.blockCount()))))
        return 8 + self.fontMetrics().horizontalAdvance("9") * (digits + 1)

    def line_number_area_paint_event(self, event) -> None:
        gutter_bg = self._alt_bg or Config.get_color("hover_2", "#1a1f2e")
        gutter_text = Config.get_color("gutter_text", "#6b7280")

        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor(gutter_bg))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        offset = self.contentOffset()
        top = round(self.blockBoundingGeometry(block).translated(offset).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        line_h = self.fontMetrics().height()
        gutter_w = self._line_number_area.width()

        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.setPen(QColor(gutter_text))
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(
                    4, top,
                    max(0, gutter_w - 8), line_h,
                    align,
                    str(block_number + 1),
                )
                if self._row_border:
                    painter.setPen(QColor(self._row_border))
                    y = bottom - 1
                    painter.drawLine(0, y, gutter_w, y)
                    painter.setPen(QColor(gutter_text))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    # == Key handling ===========================================================

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()

        # Completion popup navigation
        if self._completion_popup.isVisible():
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                self._move_completion_selection(-1 if key == Qt.Key.Key_Up else 1)
                return
            if key in (Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._accept_completion()
                return
            if key == Qt.Key.Key_Escape:
                self._completion_popup.hide()
                return

        # Cross-platform Ctrl shortcuts: undo / redo / toggle-comment.
        # Done explicitly (rather than relying on Qt defaults) so Ctrl+Y
        # works on every platform, not just Windows.
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                if mods & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                return
            if key == Qt.Key.Key_Y:
                self.redo()
                return
            if key == Qt.Key.Key_K:
                self.toggle_line_comment()
                return

        if key == Qt.Key.Key_Tab:
            self._handle_tab()
            return
        if key == Qt.Key.Key_Backtab:
            self._handle_backtab()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._handle_return()
            return

        super().keyPressEvent(event)

        if self._completion_words and (
            key in (
                Qt.Key.Key_Backspace, Qt.Key.Key_Delete,
                Qt.Key.Key_Left, Qt.Key.Key_Right,
                Qt.Key.Key_Home, Qt.Key.Key_End,
            )
            or (32 <= key <= 126)
        ):
            self._update_completion_popup()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                self._adjust_font_size(1 if delta > 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        self._completion_popup.hide()
        super().focusOutEvent(event)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self.viewport() and event.type() == QEvent.Type.Paint:
            painter = QPainter(self.viewport())
            painter.fillRect(self.viewport().rect(), QColor(self._base_bg))
            self._paint_indent_guides(painter)
            painter.end()
        return False

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        cr = self.contentsRect()
        ln_w = self.line_number_area_width()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), ln_w, cr.height())
        )
        self._fold_area.setGeometry(
            QRect(cr.left() + ln_w, cr.top(), self._FOLD_AREA_W, cr.height())
        )
        if self._completion_popup.isVisible():
            self._update_completion_popup()

    # == Private: indentation ===================================================

    def _handle_tab(self) -> None:
        if self._completion_popup.isVisible():
            self._accept_completion()
            return
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._indent_selection(cursor, 4)
        else:
            cursor.insertText("    ")
        self._completion_popup.hide()

    def _handle_backtab(self) -> None:
        cursor = self.textCursor()
        if cursor.hasSelection():
            self._dedent_selection(cursor, 4)
        else:
            self._dedent_block(cursor.block(), 4)

    def _handle_return(self) -> None:
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

    def _indent_selection(self, cursor, spaces: int) -> None:
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        doc = self.document()
        start_block = doc.findBlock(start).blockNumber()
        end_block = doc.findBlock(end).blockNumber()
        cursor.beginEditBlock()
        for block_no in range(end_block, start_block - 1, -1):
            block = doc.findBlockByNumber(block_no)
            if not block.isValid():
                continue
            c = QTextCursor(block)
            c.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            c.insertText(" " * spaces)
        cursor.endEditBlock()

    def _dedent_selection(self, cursor, max_spaces: int) -> None:
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        doc = self.document()
        start_block = doc.findBlock(start).blockNumber()
        end_block = doc.findBlock(end).blockNumber()
        cursor.beginEditBlock()
        for block_no in range(end_block, start_block - 1, -1):
            block = doc.findBlockByNumber(block_no)
            if block.isValid():
                self._dedent_block_in_edit(block, max_spaces)
        cursor.endEditBlock()

    def _dedent_block(self, block, max_spaces: int) -> None:
        cursor = QTextCursor(block)
        cursor.beginEditBlock()
        self._dedent_block_in_edit(block, max_spaces)
        cursor.endEditBlock()

    def toggle_line_comment(self) -> None:
        """Comment / uncomment every non-empty line in the current selection.

        Behaviour mirrors VS Code's ``Ctrl+/``:

        * Marker is mode-driven (``# `` for python/plain, ``// `` for json).
        * If every non-empty line in the range already starts with the
          marker (after its own leading whitespace), all are uncommented;
          otherwise all are commented at the column of the shallowest
          non-empty indent so the markers line up.
        * Empty / whitespace-only lines are skipped.
        * Selection that ends exactly at column 0 of the next line excludes
          that next line (matches VS Code).
        * The whole toggle is one undo unit (``beginEditBlock``).
        """
        marker = self._COMMENT_MARKERS.get(self._mode)
        if marker is None:
            return
        cursor = self.textCursor()
        doc = self.document()

        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            first_no = doc.findBlock(start).blockNumber()
            last_no = doc.findBlock(end).blockNumber()
            if end > start and doc.findBlock(end).position() == end:
                last_no = max(first_no, last_no - 1)
        else:
            first_no = last_no = cursor.block().blockNumber()

        blocks = []
        for n in range(first_no, last_no + 1):
            b = doc.findBlockByNumber(n)
            if b.isValid():
                blocks.append(b)

        nonempty = [b for b in blocks if b.text().strip()]
        if not nonempty:
            return

        stripped = marker.rstrip()

        def _has_marker(b) -> bool:
            text = b.text()
            i = len(text) - len(text.lstrip(" "))
            return text[i:i + len(stripped)] == stripped

        all_commented = all(_has_marker(b) for b in nonempty)

        cursor.beginEditBlock()
        if all_commented:
            for b in nonempty:
                text = b.text()
                i = len(text) - len(text.lstrip(" "))
                j = i + len(stripped)
                if j < len(text) and text[j] == " ":
                    j += 1
                c = QTextCursor(b)
                c.setPosition(b.position() + i)
                c.setPosition(b.position() + j, QTextCursor.MoveMode.KeepAnchor)
                c.removeSelectedText()
        else:
            col = min(
                len(b.text()) - len(b.text().lstrip(" ")) for b in nonempty
            )
            for b in nonempty:
                c = QTextCursor(b)
                c.setPosition(b.position() + col)
                c.insertText(marker)
        cursor.endEditBlock()

    @staticmethod
    def _dedent_block_in_edit(block, max_spaces: int) -> None:
        text = block.text()
        remove = min(len(text) - len(text.lstrip(" ")), max_spaces)
        if remove:
            c = QTextCursor(block)
            c.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            c.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.KeepAnchor,
                remove,
            )
            c.removeSelectedText()

    # == Private: zoom ==========================================================

    def _adjust_font_size(self, step: int) -> None:
        font = self.font()
        new_size = max(self._ZOOM_MIN, min(self._ZOOM_MAX, font.pointSize() + step))
        if new_size != font.pointSize():
            font.setPointSize(new_size)
            self.setFont(font)
            self._on_block_count_changed(0)

    # == Private: line number area ==============================================

    def _on_block_count_changed(self, _new_count: int = 0) -> None:
        self.setViewportMargins(
            self.line_number_area_width() + self._FOLD_AREA_W, 0, 0, 0,
        )

    def _update_line_number_area(self, rect, dy: int) -> None:
        if dy:
            self._line_number_area.scroll(0, dy)
            self._fold_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(), self._line_number_area.width(), rect.height()
            )
            self._fold_area.update(
                0, rect.y(), self._fold_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self._on_block_count_changed(0)

    # == Private: auto-completion ===============================================

    def _current_completion_prefix(self) -> str:
        cur = self.textCursor()
        block_text = cur.block().text()
        col = cur.positionInBlock()
        if col <= 0:
            return ""
        left = block_text[:col]
        m = re.search(r"([A-Za-z_]\w*)$", left)
        return m.group(1) if m else ""

    def _update_completion_popup(self) -> None:
        if not self._completion_words:
            return
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

    def _move_completion_selection(self, delta: int) -> None:
        if not self._completion_popup.isVisible() or self._completion_popup.count() <= 0:
            return
        row = self._completion_popup.currentRow()
        if row < 0:
            row = 0
        row = (row + delta) % self._completion_popup.count()
        self._completion_popup.setCurrentRow(row)

    def _accept_completion(self) -> None:
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
            cur.movePosition(
                QTextCursor.MoveOperation.Left,
                QTextCursor.MoveMode.KeepAnchor,
                len(prefix),
            )
            cur.removeSelectedText()
        cur.insertText(f"{completion}()")
        cur.movePosition(QTextCursor.MoveOperation.Left)
        self.setTextCursor(cur)
        self._completion_popup.hide()

    # == Private: highlight helpers =============================================

    def _get_highlight_format(self, style_key: str) -> QTextCharFormat:
        styles = {
            "error": (
                Config.get_color("editor_error_bg", "#7c2d12"),
                Config.get_color("editor_error_underline", "#ef4444"),
            ),
            "warning": (
                Config.get_color("editor_warning_bg", "#713f12"),
                Config.get_color("editor_warning_underline", "#eab308"),
            ),
            "info": (
                Config.get_color("editor_info_bg", "#1e3a5f"),
                Config.get_color("editor_info_underline", "#60a5fa"),
            ),
        }
        bg_hex, ul_hex = styles.get(style_key, styles["info"])
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(bg_hex))
        fmt.setUnderlineColor(QColor(ul_hex))
        fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
        return fmt

    def _update_current_line_highlight(self) -> None:
        """Highlight the line where the cursor sits with a subtle background."""
        sel = QTextEdit.ExtraSelection()
        # Subtle white overlay — QColor(str) doesn't parse #RRGGBBAA,
        # so we construct with explicit alpha.
        bg = QColor(255, 255, 255, 18)
        cfg = Config.get_color("editor_current_line_bg", "")
        if cfg:
            bg = QColor(cfg)
        sel.format.setBackground(bg)
        sel.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self._current_line_selection = [sel]
        self._apply_extra_selections()

    def _apply_extra_selections(self) -> None:
        self.setExtraSelections(
            self._current_line_selection
            + self._extra_layer_1
            + self._extra_layer_2
        )

    # == Private: indent guides =================================================

    def _indent_level_of(self, block: QTextBlock) -> int:
        """Indent level of *block* in `_INDENT_UNIT`-spaced columns.

        Returns -1 for empty/whitespace-only lines so callers can route them
        through inheritance.
        """
        text = block.text()
        if not text.strip():
            return -1
        leading = len(text) - len(text.lstrip(" "))
        return leading // self._INDENT_UNIT

    def _block_indent_with_inheritance(self, block: QTextBlock) -> int:
        """Indent level for guide drawing — empty lines inherit from neighbours.

        Following VS Code: an empty line draws guides up to ``min(prev, next)``
        of its surrounding non-empty blocks, so trailing blanks below a function
        body do not extend the body's guides.
        """
        own = self._indent_level_of(block)
        if own >= 0:
            return own
        prev = block.previous()
        while prev.isValid() and not prev.text().strip():
            prev = prev.previous()
        nxt = block.next()
        while nxt.isValid() and not nxt.text().strip():
            nxt = nxt.next()
        prev_lvl = max(0, self._indent_level_of(prev)) if prev.isValid() else 0
        nxt_lvl = max(0, self._indent_level_of(nxt)) if nxt.isValid() else 0
        if prev.isValid() and nxt.isValid():
            return min(prev_lvl, nxt_lvl)
        return prev_lvl if prev.isValid() else nxt_lvl

    def _paint_indent_guides(self, painter: QPainter) -> None:
        """Draw 1px solid vertical guides at every indent column on each row.

        Uses ``editor_indent_guide`` (defaults aligned with ``syntax_criterion``
        for a neutral "structural-rule" tone).
        """
        indent_px = self.fontMetrics().horizontalAdvance(" ") * self._INDENT_UNIT
        if indent_px <= 0:
            return
        doc_margin = int(self.document().documentMargin())
        pen = QPen(QColor(self._indent_guide_color), 1, Qt.PenStyle.SolidLine)
        painter.setPen(pen)

        viewport_rect = self.viewport().rect()
        block = self.firstVisibleBlock()
        offset = self.contentOffset()
        top = self.blockBoundingGeometry(block).translated(offset).top()

        while block.isValid() and top <= viewport_rect.bottom():
            if block.isVisible():
                height = self.blockBoundingRect(block).height()
                bottom = top + height
                if bottom >= viewport_rect.top():
                    level = self._block_indent_with_inheritance(block)
                    y0 = int(top)
                    y1 = int(bottom) - 1
                    for i in range(level):
                        x = doc_margin + i * indent_px
                        painter.drawLine(x, y0, x, y1)
                top = bottom
            else:
                top = top + self.blockBoundingRect(block).height()
            block = block.next()

    # == Private: fold detection ================================================

    def _is_fold_header(self, block: QTextBlock) -> bool:
        """A block is a fold header iff its next non-empty block has a strictly
        greater indent level (covers ``def/class/if`` and multi-line literals).
        """
        own = self._indent_level_of(block)
        if own < 0:
            return False
        nxt = block.next()
        while nxt.isValid() and not nxt.text().strip():
            nxt = nxt.next()
        if not nxt.isValid():
            return False
        return self._indent_level_of(nxt) > own

    def _is_fold_header_by_no(self, block_no: int) -> bool:
        block = self.document().findBlockByNumber(block_no)
        return block.isValid() and self._is_fold_header(block)

    def _fold_range(self, header: QTextBlock) -> tuple[int, int]:
        """Return ``(first_child_no, last_child_no)`` for *header* (inclusive).

        Includes trailing empty lines that are followed by deeper-indented
        content; stops before the first non-empty block whose indent is
        ``<= header_indent``.
        """
        own = self._indent_level_of(header)
        first = header.blockNumber() + 1
        last = first - 1
        block = header.next()
        last_nonempty = first - 1
        while block.isValid():
            if not block.text().strip():
                # Defer commit until we see the next non-empty block.
                block = block.next()
                continue
            if self._indent_level_of(block) <= own:
                break
            last_nonempty = block.blockNumber()
            block = block.next()
        # Don't swallow blank trailing rows beyond the last indented line —
        # keep range tight to indented content.
        last = last_nonempty
        return first, last

    # == Fold area: paint / mouse ===============================================

    def fold_area_width(self) -> int:
        return self._FOLD_AREA_W

    def fold_area_paint_event(self, event) -> None:
        gutter_bg = self._alt_bg or Config.get_color("hover_2", "#1A1F2E")
        marker_color = QColor(self._fold_marker_color)
        hover_color = QColor(self._fold_marker_hover_color)
        scope_color = QColor(self._fold_scope_line_color)
        btn_bg = QColor(self._fold_button_bg_color)
        btn_border = QColor(self._fold_button_border_color)

        painter = QPainter(self._fold_area)
        painter.fillRect(event.rect(), QColor(gutter_bg))

        block = self.firstVisibleBlock()
        offset = self.contentOffset()
        top = round(self.blockBoundingGeometry(block).translated(offset).top())
        x_center = self._FOLD_AREA_W // 2
        line_h = self.fontMetrics().height()

        hover_rect: Optional[QRect] = None
        hover_scope_y_end: Optional[int] = None

        while block.isValid() and top <= event.rect().bottom():
            block_no = block.blockNumber()
            if block.isVisible():
                height = round(self.blockBoundingRect(block).height())
                bottom = top + height
                if bottom >= event.rect().top() and self._is_fold_header(block):
                    folded = block_no in self._folded_headers
                    is_hover = (self._fold_hover_block == block_no)
                    glyph_color = hover_color if is_hover else marker_color
                    border_color = hover_color if is_hover else btn_border
                    y_center = top + line_h // 2
                    self._draw_fold_button(
                        painter, x_center, y_center, folded,
                        glyph_color, btn_bg, border_color,
                    )
                    if is_hover and not folded:
                        # Compute fold scope end y for the connector line.
                        _first, last_no = self._fold_range(block)
                        last_block = self.document().findBlockByNumber(last_no)
                        if last_block.isValid():
                            last_top = round(
                                self.blockBoundingGeometry(last_block)
                                .translated(offset).top()
                            )
                            last_h = round(self.blockBoundingRect(last_block).height())
                            hover_scope_y_end = last_top + last_h
                            scope_y_start = y_center + (self._FOLD_BTN_SIZE // 2) + 1
                            hover_rect = QRect(
                                x_center, scope_y_start, 1,
                                max(0, hover_scope_y_end - scope_y_start),
                            )
                top = bottom
            else:
                top = top + round(self.blockBoundingRect(block).height())
            block = block.next()

        # Draw the hover scope line on top so it sits above the gutter bg.
        if hover_rect is not None and hover_rect.height() > 0:
            painter.fillRect(hover_rect, scope_color)

    # 10×10 px square fold button (border + 1px-rounded corners + glyph)
    _FOLD_BTN_SIZE = 10

    def _draw_fold_button(self, painter: QPainter, cx: int, cy: int,
                           folded: bool, glyph: QColor,
                           bg: QColor, border: QColor) -> None:
        size = self._FOLD_BTN_SIZE
        # Pixel-aligned 1px stroke: integer top-left + 0.5 offset on the rect.
        x = cx - size // 2
        y = cy - size // 2
        rect = QRectF(x + 0.5, y + 0.5, size - 1, size - 1)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(bg)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(rect, 1.0, 1.0)

        # Glyph: ▶ when folded, ▼ when expanded — drawn with antialiasing for
        # crisp diagonals while the box stays pixel-aligned.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glyph)
        if folded:
            pts = [QPoint(cx - 2, cy - 3), QPoint(cx + 2, cy), QPoint(cx - 2, cy + 3)]
        else:
            pts = [QPoint(cx - 3, cy - 1), QPoint(cx + 3, cy - 1), QPoint(cx, cy + 2)]
        painter.drawPolygon(QPolygon(pts))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def fold_area_mouse_press(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        block = self._block_at_fold_area_y(event.position().toPoint().y())
        if block is None or not self._is_fold_header(block):
            return
        self._toggle_fold(block.blockNumber())

    def fold_area_mouse_move(self, event) -> None:
        block = self._block_at_fold_area_y(event.position().toPoint().y())
        is_header = block is not None and self._is_fold_header(block)
        new_hover = block.blockNumber() if is_header else None
        # Switch to a click cursor only when the pointer is over a fold button.
        self._fold_area.setCursor(
            Qt.CursorShape.PointingHandCursor if is_header
            else Qt.CursorShape.ArrowCursor
        )
        if new_hover != self._fold_hover_block:
            self._fold_hover_block = new_hover
            self._fold_area.update()

    def fold_area_leave(self, _event) -> None:
        self._fold_area.unsetCursor()
        if self._fold_hover_block is not None:
            self._fold_hover_block = None
            self._fold_area.update()

    def _block_at_fold_area_y(self, y: int) -> Optional[QTextBlock]:
        block = self.firstVisibleBlock()
        offset = self.contentOffset()
        top = round(self.blockBoundingGeometry(block).translated(offset).top())
        while block.isValid():
            if block.isVisible():
                bottom = top + round(self.blockBoundingRect(block).height())
                if top <= y < bottom:
                    return block
                top = bottom
            else:
                top = top + round(self.blockBoundingRect(block).height())
            block = block.next()
        return None

    # == Fold state: toggle / apply / remap =====================================

    def _toggle_fold(self, header_no: int) -> None:
        header = self.document().findBlockByNumber(header_no)
        if not header.isValid() or not self._is_fold_header(header):
            return
        if header_no in self._folded_headers:
            self._set_fold_state(header_no, fold=False)
            self._folded_headers.discard(header_no)
        else:
            self._set_fold_state(header_no, fold=True)
            self._folded_headers.add(header_no)
        self._post_fold_refresh()

    def _set_fold_state(self, header_no: int, *, fold: bool) -> None:
        header = self.document().findBlockByNumber(header_no)
        if not header.isValid():
            return
        first, last = self._fold_range(header)
        if last < first:
            return
        block = self.document().findBlockByNumber(first)
        while block.isValid() and block.blockNumber() <= last:
            if fold:
                if block.isVisible():
                    block.setVisible(False)
            else:
                # Only un-hide blocks that aren't kept hidden by a nested fold.
                if not self._block_hidden_by_other_fold(block.blockNumber(), header_no):
                    block.setVisible(True)
            block = block.next()

    def _block_hidden_by_other_fold(self, block_no: int, exclude_header: int) -> bool:
        for h in self._folded_headers:
            if h == exclude_header:
                continue
            header = self.document().findBlockByNumber(h)
            if not header.isValid() or not self._is_fold_header(header):
                continue
            first, last = self._fold_range(header)
            if first <= block_no <= last:
                return True
        return False

    def _post_fold_refresh(self) -> None:
        doc = self.document()
        doc.markContentsDirty(0, doc.characterCount())
        layout = doc.documentLayout()
        if hasattr(layout, "requestUpdate"):
            layout.requestUpdate()
        self.viewport().update()
        self._line_number_area.update()
        self._fold_area.update()

    def _unfold_to_make_visible(self, target_no: int) -> None:
        """Unfold every folded header whose range covers *target_no*."""
        # Iterate over a snapshot so we can mutate the set safely.
        for h in list(self._folded_headers):
            header = self.document().findBlockByNumber(h)
            if not header.isValid():
                self._folded_headers.discard(h)
                continue
            first, last = self._fold_range(header)
            if first <= target_no <= last:
                self._set_fold_state(h, fold=False)
                self._folded_headers.discard(h)
        self._post_fold_refresh()

    def _on_contents_change(self, position: int, chars_removed: int,
                             chars_added: int) -> None:
        """Remap fold-header block numbers across line-count changes."""
        new_count = self.blockCount()
        prev_count = self._prev_block_count
        self._prev_block_count = new_count
        delta = new_count - prev_count
        if delta == 0:
            return
        if not self._folded_headers:
            return
        start_block_no = self.document().findBlock(position).blockNumber()
        new_set: set[int] = set()
        for h in self._folded_headers:
            if h < start_block_no:
                new_set.add(h)
            elif delta < 0 and h < start_block_no + (-delta):
                # Header sat in the deleted region — drop it.
                continue
            else:
                shifted = h + delta
                if shifted >= 0:
                    new_set.add(shifted)
        # Drop entries that no longer point to a valid fold header (e.g. the
        # edit broke the indent shape).
        self._folded_headers = {h for h in new_set if self._is_fold_header_by_no(h)}
        # Re-apply visibility for surviving folds, as Qt resets newly inserted
        # blocks to visible by default.
        for h in self._folded_headers:
            self._set_fold_state(h, fold=True)
        self._post_fold_refresh()
