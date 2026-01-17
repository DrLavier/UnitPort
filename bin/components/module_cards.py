#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块卡片组件
左侧面板的可拖拽模块卡片
"""

import json
from typing import List, Dict

from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QEnterEvent, QColor
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QApplication, QGraphicsDropShadowEffect
)

from bin.core.logger import log_debug


class ModuleCard(QFrame):
    """模块卡片"""
    
    def __init__(self, title: str, subtitle: str, grad: tuple, features: List[str], parent=None):
        super().__init__(parent)
        
        self.title_text = title
        self.subtitle_text = subtitle
        self.grad = grad
        self.features = features
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        # 标题
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        
        self.title = QLabel(self.title_text)
        self.title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        title_row.addWidget(self.title)
        title_row.addStretch(1)
        
        # 功能标签
        features_text = f"功能: {', '.join(self.features[:3])}{'...' if len(self.features) > 3 else ''}"
        features_label = QLabel(features_text)
        features_label.setStyleSheet("""
            QLabel {
                color: #d1d5db;
                font-size: 11px;
                background: rgba(255, 255, 255, 0.08);
                padding: 2px 6px;
                border-radius: 4px;
                margin-top: 2px;
            }
        """)
        
        # 副标题
        self.subtitle = QLabel(self.subtitle_text)
        self.subtitle.setStyleSheet("""
            QLabel {
                color: #f3f4f6; 
                opacity: 0.8; 
                font-size: 12px; 
                line-height: 1.4;
            }
        """)
        self.subtitle.setWordWrap(True)
        
        # 布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        layout.addLayout(title_row)
        layout.addWidget(features_label)
        layout.addWidget(self.subtitle)
        
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(130 if len(self.features) > 3 else 120)
        
        # 渐变背景
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
        
        # 阴影效果
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        self.setGraphicsEffect(self.shadow)
        
        # 拖拽提示
        drag_hint = QLabel("👆 拖拽使用")
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
        """鼠标按下"""
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动 - 启动拖拽"""
        if not (event.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(event)
        
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if (pos - getattr(self, "_drag_start_pos", pos)).manhattanLength() < QApplication.startDragDistance():
            return
        
        # 创建拖拽
        drag = QDrag(self)
        mime = QMimeData()
        
        # 构建拖拽数据
        payload = {
            "title": self.title_text,
            "grad": list(self.grad),
            "features": list(self.features)
        }
        mime.setData("application/x-module-card", json.dumps(payload).encode("utf-8"))
        mime.setText(f"模块: {self.title_text}")
        
        drag.setMimeData(mime)
        
        # 创建拖拽预览图
        pm = self.grab()
        drag.setPixmap(pm)
        drag.setHotSpot(pos)
        
        # 执行拖拽
        drag.exec(Qt.CopyAction)
        
        log_debug(f"拖拽模块: {self.title_text}")
    
    def enterEvent(self, event: QEnterEvent):
        """鼠标进入"""
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
        """鼠标离开"""
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
    """模块面板"""
    
    DEFAULT_MODULES: List[Dict] = [
        {
            "title": "逻辑控制",
            "subtitle": "流程控制结构\n包括条件分支和循环",
            "grad": ("#7dd3fc", "#38bdf8"),
            "features": ["如果", "当循环", "直到循环"]
        },
        {
            "title": "动作执行",
            "subtitle": "控制机器人执行具体动作",
            "grad": ("#fb923c", "#f97316"),
            "features": ["抬右腿", "站立", "坐下", "行走", "停止"]
        },
        {
            "title": "传感输入",
            "subtitle": "读取环境传感器数据",
            "grad": ("#58d26b", "#87e36a"),
            "features": ["读取超声波", "读取红外", "读取摄像头帧", "读取IMU", "读取里程计"]
        },
        {
            "title": "条件判断",
            "subtitle": "根据条件分支执行",
            "grad": ("#d946ef", "#fb7185"),
            "features": ["等于", "不等于", "大于", "小于"]
        },
        {
            "title": "功能运算",
            "subtitle": "数值/逻辑运算模块",
            "grad": ("#3b82f6", "#22d3ee"),
            "features": ["加", "减", "乘", "除"]
        }
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        
        # 面板容器
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
        
        # 面板标题
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        
        title = QLabel("🧩 功能模块")
        title.setObjectName("panelTitle")
        
        subtitle = QLabel("拖拽到右侧画布")
        subtitle.setObjectName("panelSubtitle")
        
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(subtitle)
        v.addLayout(title_row)
        
        # 模块卡片滚动区域
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
        
        # 添加模块说明
        info_label = QLabel("拖拽模块到右侧画布，连接模块构建控制流程")
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
        
        # 填充模块卡片
        self.populate(self.DEFAULT_MODULES)
        
        scroll.setWidget(list_container)
        v.addWidget(scroll)
        
        # 状态提示
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
    
    def populate(self, modules: List[Dict]):
        """填充模块卡片"""
        # 清空现有卡片（保留第一个info_label）
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(1)
            w = item.widget()
            if w:
                w.setParent(None)
        
        # 添加模块卡片
        for m in modules:
            card = ModuleCard(
                m["title"],
                m["subtitle"],
                m["grad"],
                m.get("features", [])
            )
            self.cards_layout.addWidget(card)
        
        self.cards_layout.addStretch(1)