#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块卡片组件
左侧面板的可拖拽节点库（折叠列表）
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
    """可拖拽节点树"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.setIndentation(12)
        self.setFocusPolicy(Qt.NoFocus)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.setStyleSheet("""
            QTreeWidget {
                background: transparent;
                border: none;
                color: #e5e7eb;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 4px 6px;
                border-radius: 6px;
            }
            QTreeWidget::item:selected {
                background: #1f2937;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background: #111827;
            }
        """)

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
        log_debug(f"拖拽节点: {payload.get('title')}")


class ModulePalette(QWidget):
    """模块面板"""

    node_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._init_ui()

    def _init_ui(self):
        """初始化UI"""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setStyleSheet("""
            #panel {
                background: #2a2c33;
                border-radius: 14px;
                border: 1px solid #3f4147;
            }
            QLabel#panelTitle {
                color: #e5e7eb;
                font-weight: 700;
                font-size: 16px;
            }
            QLabel#panelSubtitle {
                color: #9ca3af;
                font-size: 12px;
            }
        """)

        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        title = QLabel("🧩 Node Library")
        title.setObjectName("panelTitle")
        subtitle = QLabel("Drag or double-click")
        subtitle.setObjectName("panelSubtitle")

        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(subtitle)
        v.addLayout(title_row)

        self.tree = NodeTree()
        self._populate_tree()
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        v.addWidget(self.tree, 1)

        status_label = QLabel("✅ 图形编辑器就绪")
        status_label.setStyleSheet("""
            QLabel {
                color: #10b981;
                font-size: 11px;
                padding: 6px;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        status_label.setAlignment(Qt.AlignCenter)
        v.addWidget(status_label)

        root.addWidget(panel)

    def _on_item_double_clicked(self, item: QTreeWidgetItem):
        payload = item.data(0, Qt.UserRole)
        if not payload or not payload.get("draggable"):
            return
        self.node_requested.emit(payload)

    def _populate_tree(self):
        self.tree.clear()

        system_root = QTreeWidgetItem(["System Nodes"])
        custom_root = QTreeWidgetItem(["Custom Nodes"])
        system_root.setExpanded(True)
        custom_root.setExpanded(True)
        self.tree.addTopLevelItem(system_root)
        self.tree.addTopLevelItem(custom_root)

        action_group = QTreeWidgetItem(["Action Nodes"])
        base_group = QTreeWidgetItem(["Base Nodes"])
        logic_group = QTreeWidgetItem(["Logic Nodes"])
        sensor_group = QTreeWidgetItem(["Sensor Nodes"])

        system_root.addChild(action_group)
        system_root.addChild(base_group)
        system_root.addChild(logic_group)
        system_root.addChild(sensor_group)

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

        self._add_node_item(base_group, "BaseNode", {
            "title": "Base Node",
            "features": ["BaseNode"],
            "preset": "BaseNode"
        })

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
            "features": ["Equal", "Not Equal", "Greater Than", "Less Than"],
            "preset": "Equal"
        })

        self._add_node_item(sensor_group, "SensorInputNode", {
            "title": "Sensor Input",
            "features": ["Read Ultrasonic", "Read Infrared", "Read Camera", "Read IMU", "Read Odometry"],
            "preset": "Read IMU"
        })

        custom_nodes = get_custom_nodes()
        if not custom_nodes:
            empty = QTreeWidgetItem(["(no custom nodes)"])
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
