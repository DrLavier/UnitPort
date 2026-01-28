#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Editor Component
Displays auto-generated Python code.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLabel
from PySide6.QtGui import QFont

from bin.core.theme_manager import get_color, get_font_size
from bin.core.localisation import tr


class CodeEditor(QWidget):
    """Code editor component."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.title = QLabel(tr("code_gen.title", "Generated Code"))
        layout.addWidget(self.title)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(False)

        font = QFont("Courier New", get_font_size("size_small", 11))
        font.setStyleHint(QFont.Monospace)
        self.text_edit.setFont(font)

        self.set_code(
            tr(
                "code_gen.placeholder",
                "# Code will appear here\n# Drag nodes to the canvas and connect them\n",
            )
        )

        layout.addWidget(self.text_edit)
        self._apply_style()

    def _apply_style(self):
        """Apply theme styles"""
        header_bg = get_color("code_header_bg", get_color("card_bg", "#1f2937"))
        header_border = get_color("border", "#374151")
        text_primary = get_color("text_primary", "#e5e7eb")
        editor_bg = get_color("code_bg", get_color("cmd_bg", "#111827"))
        selection_bg = get_color("code_selection", get_color("hover_bg", "#374151"))

        size_normal = get_font_size("size_normal", 12)
        size_small = get_font_size("size_small", 11)

        self.title.setStyleSheet(
            f"""
            QLabel {{
                background: {header_bg};
                color: {text_primary};
                padding: 12px;
                font-size: {size_normal}px;
                font-weight: bold;
                border-bottom: 1px solid {header_border};
            }}
            """
        )

        self.text_edit.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background: {editor_bg};
                color: {text_primary};
                border: none;
                font-family: 'Courier New', Consolas, monospace;
                font-size: {size_small}px;
                padding: 12px;
                selection-background-color: {selection_bg};
            }}
            """
        )

    def refresh_style(self):
        """Refresh theme styles"""
        self._apply_style()

    def set_code(self, code: str):
        """Set code content"""
        self.text_edit.setPlainText(code)

    def get_code(self) -> str:
        """Get code content"""
        return self.text_edit.toPlainText()

    def append_code(self, code: str):
        """Append code content"""
        current = self.get_code()
        self.set_code(current + "\n" + code)

    def clear(self):
        """Clear code content"""
        self.set_code("")
