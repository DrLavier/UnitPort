"""Indentation engine for CodeEditorWidget.

Pure helpers: no Qt, no SDK imports. Reused by editor.py.
"""

from __future__ import annotations

import re

# Lines ending with ":" (optionally followed by a comment) open a new
# indentation block — Python `def`/`class`/`if`/`for`/etc.
_BLOCK_OPENER_RE = re.compile(r":\s*(#.*)?$")


def _compute_smart_indent(line_text: str) -> str:
    """Return the indent string to insert on the new line after *line_text*.

    Rules:
      - Carry the current indentation of *line_text*.
      - If the stripped line ends with ':' (block opener), add 4 more spaces.
    """
    stripped = line_text.rstrip()
    n_spaces = len(line_text) - len(line_text.lstrip(" "))
    indent = " " * n_spaces
    if _BLOCK_OPENER_RE.search(stripped):
        indent += "    "
    return indent
