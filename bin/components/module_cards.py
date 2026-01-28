#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Cards Component
Draggable module cards for the left panel
"""

import json
from typing import List, Dict

from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QEnterEvent, QColor
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QApplication, QGraphicsDropShadowEffect
)
from bin.core.theme_manager import get_color, get_font_size, set_theme
from bin.core.logger import log_debug


class ModuleCard(QFrame):
    """Module Card"""

    def __init__(self, title: str, subtitle: str, grad: tuple, features: List[str], parent=None):
        super().__init__(parent)

        self.title_text = title
        self.subtitle_text = subtitle
        self.grad = grad
        self.features = features

        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        # Title
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        self.title = QLabel(self.title_text)
        self.title.setStyleSheet(f"""
            QLabel {{
                color: {get_color('text_primary')};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        title_row.addWidget(self.title)
        title_row.addStretch(1)

        # Features label
        features_text = f"Features: {', '.join(self.features[:3])}{'...' if len(self.features) > 3 else ''}"
        features_label = QLabel(features_text)
        features_label.setStyleSheet(f"""
            QLabel {{
                color: {get_color('text_primary')};
                font-size: 11px;
                background: rgba(255, 255, 255, 0.08);
                padding: 2px 6px;
                border-radius: 4px;
                margin-top: 2px;
            }}
        """)

        # Subtitle
        self.subtitle = QLabel(self.subtitle_text)
        self.subtitle.setStyleSheet(f"""
            QLabel {{
                color: {get_color('text_primary')};
                opacity: 0.8;
                font-size: 12px;
                line-height: 1.4;
            }}
        """)
        self.subtitle.setWordWrap(True)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        layout.addLayout(title_row)
        layout.addWidget(features_label)
        # layout.addWidget(self.subtitle)

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(130 if len(self.features) > 3 else 120)

        # Gradient background
        c1, c2 = self.grad
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: 12px;
                background: qlineargradient(
                    x1:0, y1:0,
                    x2:1, y2:1,
                    stop:0 {c1},
                    stop:1 {c2}
                );
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        """)

        # Shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        self.setGraphicsEffect(self.shadow)

        # Drag hint
        drag_hint = QLabel("Drag to use")
        drag_hint.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.6);
                font-size: 10px;
                background: rgba(0, 0, 0, 0.2);
                padding: 1px 6px;
                border-radius: 4px;
                margin-top: 4px;
            }
        """)
        drag_hint.setAlignment(Qt.AlignRight)
        layout.addWidget(drag_hint)

    def mousePressEvent(self, event):
        """Mouse press"""
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move - start drag"""
        if not (event.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(event)

        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if (pos - getattr(self, "_drag_start_pos", pos)).manhattanLength() < QApplication.startDragDistance():
            return

        # Create drag
        drag = QDrag(self)
        mime = QMimeData()

        # Build drag data
        payload = {
            "title": self.title_text,
            "grad": list(self.grad),
            "features": list(self.features)
        }
        mime.setData("application/x-module-card", json.dumps(payload).encode("utf-8"))
        mime.setText(f"Module: {self.title_text}")

        drag.setMimeData(mime)

        # Create drag preview
        pm = self.grab()
        drag.setPixmap(pm)
        drag.setHotSpot(pos)

        # Execute drag
        drag.exec(Qt.CopyAction)

        log_debug(f"Module dragged: {self.title_text}")

    def enterEvent(self, event: QEnterEvent):
        """Mouse enter"""
        self.shadow.setBlurRadius(28)
        self.shadow.setOffset(0, 8)
        c1, c2 = self.grad
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: 12px;
                background: qlineargradient(
                    x1:0, y1:0,
                    x2:1, y2:1,
                    stop:0 {c1},
                    stop:1 {c2}
                );
                border: 1px solid rgba(255, 255, 255, 0.3);
            }}
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leave"""
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        c1, c2 = self.grad
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: 12px;
                background: qlineargradient(
                    x1:0, y1:0,
                    x2:1, y2:1,
                    stop:0 {c1},
                    stop:1 {c2}
                );
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        """)
        super().leaveEvent(event)


class ModulePalette(QWidget):
    """Module Palette"""

    DEFAULT_MODULES: List[Dict] = [
        {
            "title": "Logic Control",
            "subtitle": "Flow control structures\nIncluding conditional branches and loops",
            "grad": ("#7dd3fc", "#38bdf8"),
            "features": ["If", "While Loop", "Until Loop"]
        },
        {
            "title": "Action Execution",
            "subtitle": "Control robot to execute actions",
            "grad": ("#fb923c", "#f97316"),
            "features": ["Lift Right Leg", "Stand", "Sit", "Walk", "Stop"]
        },
        {
            "title": "Sensor Input",
            "subtitle": "Read environment sensor data",
            "grad": ("#58d26b", "#87e36a"),
            "features": ["Read Ultrasonic", "Read Infrared", "Read Camera Frame", "Read IMU", "Read Odometry"]
        },
        {
            "title": "Condition",
            "subtitle": "Branch execution based on conditions",
            "grad": ("#d946ef", "#fb7185"),
            "features": ["Equal", "Not Equal", "Greater Than", "Less Than"]
        },
        {
            "title": "Compute",
            "subtitle": "Numerical/logical operations",
            "grad": ("#3b82f6", "#22d3ee"),
            "features": ["Add", "Subtract", "Multiply", "Divide"]
        }
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Panel container
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

        # Panel title
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        title = QLabel("Modules")
        title.setObjectName("panelTitle")

        subtitle = QLabel("Drag to canvas")
        subtitle.setObjectName("panelSubtitle")

        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(subtitle)
        v.addLayout(title_row)

        # Module cards scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #1f2025;
                border: none;
                width: 10px;
                margin: 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #4b4f57;
                border-radius: 5px;
                min-height: 40px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5d626c;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent;
            }
        """)

        list_container = QWidget()
        self.cards_layout = QVBoxLayout(list_container)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)
        self.cards_layout.setSpacing(12)

        # Add module info
        info_label = QLabel("Drag modules to canvas on the right, connect modules to build control flow")
        info_label.setStyleSheet("""
            QLabel {
                color: #9ca3af;
                font-size: 11px;
                padding: 4px 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                border-left: 3px solid #60a5fa;
            }
        """)
        info_label.setWordWrap(True)
        self.cards_layout.addWidget(info_label)

        # Populate module cards
        self.populate(self.DEFAULT_MODULES)

        scroll.setWidget(list_container)
        v.addWidget(scroll)

        # Status hint
        status_label = QLabel("Graph editor ready")
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

    def populate(self, modules: List[Dict]):
        """Populate module cards"""
        # Clear existing cards (keep first info_label)
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(1)
            w = item.widget()
            if w:
                w.setParent(None)

        # Add module cards
        for m in modules:
            card = ModuleCard(
                m["title"],
                m["subtitle"],
                m["grad"],
                m.get("features", [])
            )
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch(1)
