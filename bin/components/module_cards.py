#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module palette component for the node library.
"""

import json
from typing import Dict, Any, Optional

from PySide6.QtCore import Qt, QMimeData, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QTreeWidget, QTreeWidgetItem
)

from bin.core.logger import log_debug
from bin.core.localisation import tr
from bin.core.theme_manager import get_color, get_font_size
class NodeTree(QTreeWidget):
    """Draggable node tree"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.setIndentation(12)
        self.setFocusPolicy(Qt.NoFocus)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.refresh_style()

    def refresh_style(self):
        """Refresh theme styles"""
        text_primary = get_color("text_primary", "#e5e7eb")
        hover_bg = get_color("hover_bg", "#111827")
        selected_bg = get_color("card_bg", "#1f2937")
        font_size = get_font_size("size_small", 12)

        self.setStyleSheet(
            f"""
            QTreeWidget {{
                background: transparent;
                border: none;
                color: {text_primary};
                font-size: {font_size}px;
            }}
            QTreeWidget::item {{
                padding: 6px 8px;
                margin: 2px 0px;
                border-radius: 10px;
                border: 1px solid transparent;
            }}
            QTreeWidget::item:selected {{
                background: {selected_bg};
                color: {text_primary};
                border: 1px solid {selected_bg};
            }}
            QTreeWidget::item:hover {{
                background: {hover_bg};
                border: 1px solid {hover_bg};
            }}
            """
        )

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
        payload = item.data(0, Qt.UserRole)
        if not payload or not payload.get("draggable"):
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-module-card", json.dumps(payload).encode("utf-8"))
        mime.setText(f"Node: {payload.get('title', '')}")
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)
        log_debug(tr("log.module_dragged", "Module dragged: {title}", title=payload.get("title", "")))


class ModulePalette(QWidget):
    """Module palette widget"""

    node_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.panel = QFrame()
        self.panel.setObjectName("panel")

        v = QVBoxLayout(self.panel)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self.title = QLabel(tr("modules.panel_title", "Node Library"))
        self.title.setObjectName("panelTitle")
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("panelSubtitle")
        self.subtitle.setVisible(False)

        title_row.addWidget(self.title)
        title_row.addStretch(1)
        title_row.addWidget(self.subtitle)
        v.addLayout(title_row)

        self.tree = NodeTree()
        self._populate_tree()
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        v.addWidget(self.tree, 1)

        self.status_label = QLabel(tr("modules.status_ready", "Graph editor ready"))
        self.status_label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.status_label)

        root.addWidget(self.panel)
        self._apply_style()

    def _apply_style(self):
        """Apply theme styles"""
        panel_bg = get_color("panel_bg", get_color("card_bg", "#2a2c33"))
        panel_border = get_color("panel_border", get_color("border", "#3f4147"))
        title_color = get_color("text_primary", "#e5e7eb")
        subtitle_color = get_color("text_secondary", "#9ca3af")
        status_text = get_color("text_secondary", "#9ca3af")
        status_bg = get_color("hover_bg", "rgba(255, 255, 255, 0.03)")
        status_border = get_color("border", "rgba(255, 255, 255, 0.1)")
        title_size = get_font_size("size_large", 16)
        subtitle_size = get_font_size("size_small", 12)

        self.panel.setStyleSheet(
            f"""
            #panel {{
                background: {panel_bg};
                border-radius: 14px;
                border: 1px solid {panel_border};
            }}
            QLabel#panelTitle {{
                color: {title_color};
                font-weight: 700;
                font-size: {title_size}px;
            }}
            QLabel#panelSubtitle {{
                color: {subtitle_color};
                font-size: {subtitle_size}px;
            }}
            """
        )

        self.status_label.setStyleSheet(
            f"""
            QLabel {{
                color: {status_text};
                font-size: {subtitle_size}px;
                padding: 6px;
                background: {status_bg};
                border-radius: 6px;
                border: 1px solid {status_border};
            }}
            """
        )

    def refresh_style(self):
        """Refresh theme styles"""
        self.title.setText(tr("modules.panel_title", "Node Library"))
        self.subtitle.setText("")
        self.subtitle.setVisible(False)
        self.status_label.setText(tr("modules.status_ready", "Graph editor ready"))
        self._apply_style()
        self.tree.refresh_style()
        self._populate_tree()

    def _on_item_double_clicked(self, item: QTreeWidgetItem):
        payload = item.data(0, Qt.UserRole)
        if not payload or not payload.get("draggable"):
            return
        self.node_requested.emit(payload)

    def _populate_tree(self):
        self.tree.clear()

        root = QTreeWidgetItem([tr("modules.system_nodes", "System Nodes")])
        root.setExpanded(True)
        self.tree.addTopLevelItem(root)

        groups = [
            (tr("modules.group_base", "Base"), [
                "Start",
                "End",
                "Wait(duration/event)",
                "Gate(condition/event)",
                "Timeout(duration)",
                "Retry(max_attempts, backoff)",
                "Cancel",
                "Abort",
            ]),
            (tr("modules.group_logic", "Logic"), [
                "If / Elif / Else",
                "Loop",
                "Break",
                "Switch / Case",
                "Try / Catch / Finally",
                "Parallel Branch",
                "Join",
            ]),
            (tr("modules.group_action", "Action"), [
                "Behavior",
                "Sensor",
                "Trigger",
            ]),
            (tr("modules.group_script", "Script"), [
                "Script Box",
            ]),
            (tr("modules.group_state_event", "State/Event"), [
                "OnEvent(event_name)",
                "PublishEvent(event_name, payload)",
                "SetState(key, value)",
                "CheckState(key, expected)",
                "HumanConfirm(prompt, timeout)",
            ]),
        ]

        for group_name, node_titles in groups:
            group = QTreeWidgetItem([group_name])
            root.addChild(group)
            for title in node_titles:
                self._add_node_item(group, title, {
                    "title": title,
                    "features": [],
                })

        self.tree.expandAll()

    def _add_node_item(self, parent: QTreeWidgetItem, label: str, payload: Optional[Dict[str, Any]] = None):
        item = QTreeWidgetItem([label])
        item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        data = payload or {}
        if "draggable" not in data:
            data["draggable"] = True
        item.setData(0, Qt.UserRole, data)
        parent.addChild(item)


NodePalette = ModulePalette

__all__ = ["NodePalette", "ModulePalette"]
