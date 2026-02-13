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
from nodes.sys_nodes import (
    ActionExecutionNode,
    StopNode,
    IfNode,
    WhileLoopNode,
    ComparisonNode,
    SensorInputNode,
    BaseNode
)
from custom_nodes import get_custom_nodes


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
                padding: 4px 6px;
                border-radius: 6px;
            }}
            QTreeWidget::item:selected {{
                background: {selected_bg};
                color: {text_primary};
            }}
            QTreeWidget::item:hover {{
                background: {hover_bg};
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
        self.subtitle = QLabel(tr("modules.panel_subtitle", "Drag to canvas"))
        self.subtitle.setObjectName("panelSubtitle")

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
        self.subtitle.setText(tr("modules.panel_subtitle", "Drag to canvas"))
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

        system_root = QTreeWidgetItem([tr("modules.system_nodes", "System Nodes")])
        custom_root = QTreeWidgetItem([tr("modules.custom_nodes", "Custom Nodes")])
        system_root.setExpanded(True)
        custom_root.setExpanded(True)
        self.tree.addTopLevelItem(system_root)
        self.tree.addTopLevelItem(custom_root)

        action_group = QTreeWidgetItem([tr("modules.action_nodes", "Action Nodes")])
        logic_group = QTreeWidgetItem([tr("modules.logic_nodes", "Logic Nodes")])
        sensor_group = QTreeWidgetItem([tr("modules.sensor_nodes", "Sensor Nodes")])
        utility_group = QTreeWidgetItem([tr("modules.utility_nodes", "Utility Nodes")])

        system_root.addChild(action_group)
        system_root.addChild(logic_group)
        system_root.addChild(sensor_group)
        system_root.addChild(utility_group)

        # Action nodes
        self._add_node_item(action_group, "ActionExecutionNode", {
            "title": "Action Execution",
            "features": ["Lift Right Leg", "Stand", "Sit", "Walk", "Stop"],
            "preset": "Stand"
        })
        self._add_node_item(action_group, "StopNode", {
            "title": "Action Execution",
            "features": ["Lift Right Leg", "Stand", "Sit", "Walk", "Stop"],
            "preset": "Stop"
        })

        # Logic nodes
        self._add_node_item(logic_group, "IfNode", {
            "title": "Logic Control",
            "features": ["If", "While Loop"],
            "preset": "If"
        })
        self._add_node_item(logic_group, "WhileLoopNode", {
            "title": "Logic Control",
            "features": ["If", "While Loop"],
            "preset": "While Loop"
        })
        self._add_node_item(logic_group, "ComparisonNode", {
            "title": "Condition",
            "features": ["Equal", "Not Equal", "Greater Than", "Less Than", "Greater Equal", "Less Equal"],
            "preset": "Equal"
        })

        # Sensor nodes
        self._add_node_item(sensor_group, "SensorInputNode", {
            "title": "Sensor Input",
            "features": ["Read Ultrasonic", "Read Infrared", "Read Camera", "Read IMU", "Read Odometry"],
            "preset": "Read IMU"
        })

        # Utility nodes
        self._add_node_item(utility_group, "MathNode", {
            "title": "Math",
            "features": ["Add", "Subtract", "Multiply", "Divide", "Power", "Modulo", "Min", "Max", "Abs", "Sum", "Average"],
            "preset": "Add"
        })
        self._add_node_item(utility_group, "TimerNode", {
            "title": "Timer",
            "features": []
        })

        custom_nodes = get_custom_nodes()
        if not custom_nodes:
            empty = QTreeWidgetItem([tr("modules.no_custom_nodes", "(no custom nodes)")])
            custom_root.addChild(empty)
        else:
            for node_type in sorted(custom_nodes.keys()):
                self._add_node_item(custom_root, node_type, {
                    "title": node_type,
                    "features": [node_type],
                    "preset": node_type
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
