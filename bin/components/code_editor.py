#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码编辑器组件
显示生成的代码
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPlainTextEdit, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


class CodeEditor(QWidget):
    """代码编辑器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 标题
        title = QLabel("生成的代码")
        title.setStyleSheet("""
            QLabel {
                background: #1f2937;
                color: #e5e7eb;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
                border-bottom: 1px solid #374151;
            }
        """)
        layout.addWidget(title)
        
        # 代码编辑器
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(False)
        self.text_edit.setStyleSheet("""
            QPlainTextEdit {
                background: #111827;
                color: #e5e7eb;
                border: none;
                font-family: 'Courier New', Consolas, monospace;
                font-size: 11px;
                padding: 12px;
                selection-background-color: #374151;
            }
        """)
        
        # 设置字体
        font = QFont("Courier New", 10)
        font.setStyleHint(QFont.Monospace)
        self.text_edit.setFont(font)
        
        # 设置初始内容
        self.set_code("# 代码将在这里生成\n# 拖拽节点到画布并连接它们\n")
        
        layout.addWidget(self.text_edit)
    
    def set_code(self, code: str):
        """设置代码内容"""
        self.text_edit.setPlainText(code)
    
    def get_code(self) -> str:
        """获取代码内容"""
        return self.text_edit.toPlainText()
    
    def append_code(self, code: str):
        """追加代码"""
        current = self.get_code()
        self.set_code(current + "\n" + code)
    
    def clear(self):
        """清空代码"""
        self.set_code("")