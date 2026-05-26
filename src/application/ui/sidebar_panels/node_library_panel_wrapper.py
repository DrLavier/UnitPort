# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Sidebar Node Library panel wrapper.

Hosts :class:`application.ui.canvas.node_library_panel.NodeLibraryPanel` inside
the Sidebar pop-out panel. The panel adapts to the host's available height /
width (TitleGroupBox body), and re-emits ``node_requested`` for the host
(MainWindow) to route into the live canvas.

The underlying tree font size is already pinned to ``size_small`` in
``NodeLibraryPanel._NodeTree.refresh_style``.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from application.ui.canvas.node_library_panel import NodeLibraryPanel


class SidebarNodeLibraryPanel(QWidget):
    """Sidebar-hosted Node Library panel — fills available space, no fixed size."""

    node_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._panel = NodeLibraryPanel(self)
        # 让面板填满 Sidebar 弹出面板的可用宽高（去掉 NodeLibraryPanel.sizeHint
        # 给出的 240×360 软约束）。
        self._panel.setMinimumSize(0, 0)
        self._panel.node_requested.connect(self.node_requested)
        layout.addWidget(self._panel, 1)

    def refresh_style(self) -> None:
        self._panel.refresh_style()


__all__ = ["SidebarNodeLibraryPanel"]
