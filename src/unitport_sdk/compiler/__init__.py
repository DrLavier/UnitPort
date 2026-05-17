"""``unitport_sdk.compiler`` — reusable code-editor widget primitives.

This subpackage hosts a full-featured code editor (line-number gutter,
smart indent, Python + JSON syntax highlighting, auto-completion popup,
DataManager-backed file I/O) that any RELEASE application can embed via
``from unitport_sdk import CodeEditorWidget, CodeEditorDialog``.

It is a cross-project infrastructure widget — sibling in spirit to the
``Lavi*`` themed widgets in ``unitport_sdk.canvas``.
"""

from .dialog import CodeEditorDialog
from .editor import CodeEditorWidget
from .highlighters import JsonSyntaxHighlighter, PythonSyntaxHighlighter

__all__ = [
    "CodeEditorWidget",
    "CodeEditorDialog",
    "PythonSyntaxHighlighter",
    "JsonSyntaxHighlighter",
]
