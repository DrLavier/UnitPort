# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Fold-button gutter widget — internal helper for CodeEditorWidget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .editor import CodeEditorWidget


class _FoldArea(QWidget):
    """Narrow column between line-number gutter and code canvas.

    Hosts fold/unfold triangles for indent-block headers. All paint and
    mouse logic is delegated to the parent editor so the same QPainter
    block-iteration that drives the line-number gutter can be reused.
    """

    def __init__(self, editor: "CodeEditorWidget"):
        super().__init__(editor)
        self._editor = editor
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self._editor.fold_area_width(), 0)

    def paintEvent(self, event):  # type: ignore[override]
        self._editor.fold_area_paint_event(event)

    def mousePressEvent(self, event):  # type: ignore[override]
        self._editor.fold_area_mouse_press(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        self._editor.fold_area_mouse_move(event)

    def leaveEvent(self, event):  # type: ignore[override]
        self._editor.fold_area_leave(event)
