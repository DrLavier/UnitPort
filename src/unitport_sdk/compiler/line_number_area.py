# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Line-number gutter widget — internal helper for CodeEditorWidget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .editor import CodeEditorWidget


class _LineNumberArea(QWidget):
    """Left gutter that paints line numbers for CodeEditorWidget.

    Pure pass-through: width and paint are delegated to the parent editor
    via ``line_number_area_width()`` / ``line_number_area_paint_event()``.
    """

    def __init__(self, editor: "CodeEditorWidget"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):  # type: ignore[override]
        self._editor.line_number_area_paint_event(event)
