#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Editor Component
Displays auto-generated Python code.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from PySide6.QtGui import QFont

from src.system.core.theme_manager import get_color, get_font_size
from src.system.core.localisation import tr
from bin.pages.canvas.script_editor import PythonIndentEditor


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

        self.text_edit = PythonIndentEditor()
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
        code_editor_text = get_color("code_editor_text", "#e5e7eb")
        editor_bg = get_color("code_bg", get_color("cmd_bg", "#111827"))
        index_bg = get_color("code_alt_bg", editor_bg)
        border = get_color("border", "#374151")
        selection_bg = get_color("code_selection", get_color("hover_bg", "#374151"))

        self.text_edit.setStyleSheet(
            f"""
            QPlainTextEdit {{
                color: {code_editor_text};
                border: none;
                font-family: 'Courier New', Consolas, monospace;
                padding: 12px;
                selection-background-color: {selection_bg};
            }}
            """
        )
        self.text_edit.set_row_colors(editor_bg, index_bg, border)

    def refresh_style(self):
        """Refresh theme styles"""
        if hasattr(self.text_edit, "refresh_style"):
            self.text_edit.refresh_style()
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
        """Diagnostics are currently not shown in editor header UI."""
        _ = diagnostics

    def compile_code(self):
        """Compile code editor content to canvas via Code -> AST -> IR -> Canvas."""
        if self._graph_scene is None:
            return

        if self._compilation_source == "canvas":
            return  # Prevent circular trigger

        self._compilation_source = "code"
        try:
            from src.system.compiler.parser.parser import Parser
            from src.system.compiler.lowering.ast_to_ir import ASTToIR
            from src.system.compiler.lowering.ir_to_canvas import IRToCanvas
            from src.system.compiler.semantic.validator import SemanticValidator
            from src.system.core.logger import log_info, log_warning

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
            from src.system.core.logger import log_warning
            log_warning(f"Code compilation failed: {e}")
        finally:
            self._compilation_source = None
