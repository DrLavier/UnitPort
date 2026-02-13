#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Editor Component
Displays auto-generated Python code.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QLabel, QPushButton,
)
from PySide6.QtGui import QFont

from bin.core.theme_manager import get_color, get_font_size
from bin.core.localisation import tr


class CodeEditor(QWidget):
    """Code editor component with bidirectional compilation support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph_scene = None
        self._compilation_source = None  # 'canvas' or 'code' to prevent loops
        self._init_ui()

    def set_graph_scene(self, scene):
        """Set the graph scene reference for Code->Canvas compilation."""
        self._graph_scene = scene

    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header row with title and compile button
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        self.title = QLabel(tr("code_gen.title", "Generated Code"))
        header_layout.addWidget(self.title)
        header_layout.addStretch(1)

        self.compile_btn = QPushButton(tr("code_gen.compile_btn", "Compile to Canvas"))
        self.compile_btn.setFixedHeight(28)
        self.compile_btn.clicked.connect(self.compile_code)
        header_layout.addWidget(self.compile_btn)

        layout.addWidget(header)

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
        accent = get_color("accent", "#3b82f6")

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

        self.compile_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {accent};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: {size_small}px;
                margin: 4px 8px;
            }}
            QPushButton:hover {{
                background: {get_color("accent_hover", "#2563eb")};
            }}
            QPushButton:pressed {{
                background: {get_color("accent_pressed", "#1d4ed8")};
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

    def show_diagnostics(self, diagnostics):
        """Show diagnostics from the compiler pipeline in the title bar."""
        errors = [d for d in diagnostics if d.level.value == "error"]
        warnings = [d for d in diagnostics if d.level.value == "warn"]

        if errors:
            self.title.setText(
                tr("code_gen.title_errors",
                   "Generated Code ({count} errors)",
                   count=len(errors))
            )
        elif warnings:
            self.title.setText(
                tr("code_gen.title_warnings",
                   "Generated Code ({count} warnings)",
                   count=len(warnings))
            )
        else:
            self.title.setText(tr("code_gen.title", "Generated Code"))

    def compile_code(self):
        """Compile code editor content to canvas via Code -> AST -> IR -> Canvas."""
        if self._graph_scene is None:
            return

        if self._compilation_source == "canvas":
            return  # Prevent circular trigger

        self._compilation_source = "code"
        try:
            from compiler.parser.parser import Parser
            from compiler.lowering.ast_to_ir import ASTToIR
            from compiler.lowering.ir_to_canvas import IRToCanvas
            from compiler.semantic.validator import SemanticValidator
            from bin.core.logger import log_info, log_warning

            code = self.get_code()
            if not code.strip():
                return

            # Parse
            parser = Parser(code)
            ast, parse_diags = parser.parse()

            # Lower to IR
            lowerer = ASTToIR()
            robot_type = getattr(self._graph_scene, '_robot_type', 'go2')
            ir, lower_diags = lowerer.lower(ast, robot_type)

            # Validate
            validator = SemanticValidator()
            validate_diags = validator.validate(ir)

            # Convert to canvas data
            converter = IRToCanvas()
            graph_data, convert_diags = converter.convert(ir)

            all_diags = parse_diags + lower_diags + validate_diags + convert_diags
            self.show_diagnostics(all_diags)

            # Apply to canvas
            self._graph_scene.load_workflow(graph_data)

            log_info(f"Code compiled to canvas: {len(ir.nodes)} nodes")

        except Exception as e:
            from bin.core.logger import log_warning
            log_warning(f"Code compilation failed: {e}")
            self.title.setText(
                tr("code_gen.title_compile_error",
                   "Generated Code (compile error)")
            )
        finally:
            self._compilation_source = None
