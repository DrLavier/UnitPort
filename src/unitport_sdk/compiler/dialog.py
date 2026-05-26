# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CodeEditorDialog — modal popup that wraps :class:`CodeEditorWidget`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..sys import Config, I18n, tr
from .editor import CodeEditorWidget

PathLike = Union[str, Path]


class CodeEditorDialog(QDialog):
    """Ready-made dialog for quick code viewing / editing.

    Parameters
    ----------
    title : str
        Window title. Empty string falls back to ``tr("code_editor.title", "Code Editor")``.
    code : str
        Initial code content.
    read_only : bool
        If ``True`` the editor is not editable; the OK button is hidden.
    completion_words : list[str] | None
        Auto-completion words.
    extra_builtins : list[str] | None
        Extra builtin names for Python syntax highlighting.
    mode : str
        ``"python"`` / ``"json"`` / ``"plain"`` — propagated to the editor.
    width, height : int
        Initial dialog size.
    parent : QWidget | None
    """

    def __init__(
        self,
        title: str = "",
        code: str = "",
        *,
        read_only: bool = False,
        completion_words: Optional[List[str]] = None,
        extra_builtins: Optional[List[str]] = None,
        mode: str = "python",
        width: int = 800,
        height: int = 600,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._title_key = "code_editor.title"
        self._title_default = title or tr(self._title_key, "Code Editor")
        self.setWindowTitle(self._title_default)
        self.setModal(True)
        self.resize(width, height)

        self._read_only = read_only
        self._init_ui(code, read_only, completion_words, extra_builtins, mode)

        # Auto-retranslate on language switch
        I18n.instance().language_changed.connect(self._retranslate)

    @classmethod
    def from_file(cls, path: PathLike, **kw) -> "CodeEditorDialog":
        """Construct a dialog and pre-populate it from *path* via DataManager."""
        dlg = cls(**kw)
        dlg._editor.load_file(path)
        return dlg

    def _init_ui(self, code, read_only, completion_words, extra_builtins, mode) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._editor = CodeEditorWidget(
            self,
            completion_words=completion_words,
            extra_builtins=extra_builtins,
            read_only=read_only,
            mode=mode,
        )
        if code:
            self._editor.setPlainText(code)

        layout.addWidget(self._editor, stretch=1)

        # Status label
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self._status)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._ok_btn: Optional[QPushButton] = None
        if not read_only:
            self._ok_btn = QPushButton(tr("code_editor.btn.ok", "OK"))
            self._ok_btn.setFixedWidth(80)
            self._ok_btn.clicked.connect(self.accept)
            btn_row.addWidget(self._ok_btn)

        cancel_default = "Close" if read_only else "Cancel"
        cancel_key = "code_editor.btn.close" if read_only else "code_editor.btn.cancel"
        self._cancel_btn = QPushButton(tr(cancel_key, cancel_default))
        self._cancel_key = cancel_key
        self._cancel_default = cancel_default
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        layout.addLayout(btn_row)

        # Dialog chrome — themed via Config slots
        dialog_bg = Config.get_color("card_bg", "#1f2937")
        accent = Config.get_color("accent", "#3b82f6")
        accent_text = Config.get_color("main_t2", "#ffffff")
        accent_hover = Config.get_color("accent_hover", "#2563eb")
        self.setStyleSheet(
            f"CodeEditorDialog {{ background: {dialog_bg}; }}"
            f"QPushButton {{ background: {accent};"
            f" color: {accent_text};"
            f" border: none; border-radius: 4px;"
            f" padding: 6px 16px; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {accent_hover}; }}"
        )

    def _retranslate(self, *_) -> None:
        self.setWindowTitle(tr(self._title_key, self._title_default))
        if self._ok_btn is not None:
            self._ok_btn.setText(tr("code_editor.btn.ok", "OK"))
        self._cancel_btn.setText(tr(self._cancel_key, self._cancel_default))

    # -- Public API ------------------------------------------------------------

    def get_code(self) -> str:
        """Return the current editor text."""
        return self._editor.toPlainText()

    def set_code(self, code: str) -> None:
        """Replace the editor text."""
        self._editor.setPlainText(code)

    def set_status(self, text: str, color: str = "#888") -> None:
        """Show a status message below the editor."""
        self._status.setText(text)
        self._status.setStyleSheet(f"font-size: 10px; color: {color};")

    @property
    def editor(self) -> CodeEditorWidget:
        """Direct access to the underlying ``CodeEditorWidget``."""
        return self._editor
