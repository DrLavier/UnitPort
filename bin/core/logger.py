#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger Module
Thread-safe log signal transmission and UI display
"""

import threading
from typing import Optional
from datetime import datetime

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QColor, QTextCursor, QTextCharFormat

from bin.core.theme_manager import get_color, get_font_size
from bin.core.localisation import tr


# ============================================================================
# LogSignal
# ============================================================================

class LogSignal(QObject):
    """Log signal (thread-safe)"""
    log_message = Signal(str, str, bool, bool)  # text, log_type, wrap, typer

    _instance: Optional["LogSignal"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LogSignal":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def emit_log(self, text: str, log_type="info", wrap=True, typer=False):
        """Emit log signal"""
        self.log_message.emit(text, log_type, wrap, typer)

    def debug(self, text, wrap=True, typer=False):
        self.emit_log(text, "debug", wrap, typer)

    def info(self, text, wrap=True, typer=False):
        self.emit_log(text, "info", wrap, typer)

    def warning(self, text, wrap=True, typer=False):
        self.emit_log(text, "warning", wrap, typer)

    def error(self, text, wrap=True, typer=False):
        self.emit_log(text, "error", wrap, typer)

    def success(self, text, wrap=True, typer=False):
        self.emit_log(text, "success", wrap, typer)


def get_log_signal() -> LogSignal:
    """Get global log signal instance"""
    return LogSignal.instance()


def log(text, log_type="info", wrap=True, typer=False):
    """Send log"""
    text = str(text)
    get_log_signal().emit_log(text, log_type, wrap, typer)


def log_info(text, wrap=True, typer=False):
    """Info log"""
    text = str(text)
    get_log_signal().info(text, wrap, typer)


def log_debug(text, wrap=True, typer=False):
    """Debug log"""
    text = str(text)
    get_log_signal().debug(text, wrap, typer)


def log_warning(text, wrap=True, typer=False):
    """Warning log"""
    text = str(text)
    get_log_signal().warning(text, wrap, typer)


def log_error(text, wrap=True, typer=False):
    """Error log"""
    text = str(text)
    get_log_signal().error(text, wrap, typer)


def log_success(text, wrap=True, typer=False):
    """Success log"""
    text = str(text)
    get_log_signal().success(text, wrap, typer)


# ============================================================================
# Typer Thread
# ============================================================================

class TyperThread(QThread):
    """Typewriter effect thread"""
    char_ready = Signal(str)
    finished = Signal()

    def __init__(self, text: str, interval=30):
        super().__init__()
        self._text = text
        self._interval = interval
        self._stop = False

    def run(self):
        for ch in self._text:
            if self._stop:
                break
            self.char_ready.emit(ch)
            self.msleep(self._interval)
        self.finished.emit()

    def stop(self):
        self._stop = True


# ============================================================================
# CmdLogWidget
# ============================================================================

class CmdLogWidget(QWidget):
    """Command line log display widget"""

    LOG_COLORS = {
        "debug": "#8B8B8B",
        "info": "#FFFFFF",
        "success": "#00D26A",
        "warning": "#FFC107",
        "error": "#FF6B6B",
    }

    LOG_PREFIX = {
        "debug": "DEBUG",
        "info": "INFO",
        "success": "SUCCESS",
        "warning": "WARNING",
        "error": "ERROR",
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self._status_start_pos: Optional[int] = None
        self._typer_thread: Optional[TyperThread] = None

        self._init_ui()
        get_log_signal().log_message.connect(self._on_log)

    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(10, 5, 10, 5)

        tt = QLabel(tr("console.title", "Console"))
        tt.setMaximumHeight(35)
        hl.addWidget(tt)
        hl.addStretch()

        btn = QPushButton(tr("console.clear", "Clear"))
        btn.clicked.connect(self.clear)
        hl.addWidget(btn)

        layout.addWidget(header)

        # Text editor
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self._apply_style()

    def _on_log(self, text, log_type, wrap, typer):
        """Handle log message"""
        if typer:
            self._start_typer(text, log_type, wrap)
        else:
            self._append_log(text, log_type, wrap)

    def _clear_status_line(self):
        """Clear status line"""
        if self._status_start_pos is None:
            return

        cursor = self.text_edit.textCursor()
        cursor.setPosition(self._status_start_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End,
                            QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()

        self._status_start_pos = None

    def _append_log(self, text, log_type, wrap):
        """Append log"""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self.LOG_COLORS.get(log_type, "#FFFFFF")))

        ts = datetime.now().strftime("%H:%M:%S")
        prefix = self.LOG_PREFIX.get(log_type, "INFO")
        content = f"[{ts}] [{prefix}] {text}"

        if not wrap:
            self._clear_status_line()
            self._status_start_pos = cursor.position()
            cursor.setCharFormat(fmt)
            cursor.insertText(content)
        else:
            self._clear_status_line()
            cursor.setCharFormat(fmt)
            cursor.insertText(content + "\n")

        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _start_typer(self, text, log_type, wrap):
        """Start typewriter effect"""
        if self._typer_thread and self._typer_thread.isRunning():
            self._typer_thread.stop()
            self._typer_thread.wait()

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self.LOG_COLORS.get(log_type, "#FFFFFF")))

        ts = datetime.now().strftime("%H:%M:%S")
        prefix = self.LOG_PREFIX.get(log_type, "INFO")
        header = f"[{ts}] [{prefix}] "

        self._clear_status_line()
        self._status_start_pos = cursor.position() if not wrap else None

        cursor.setCharFormat(fmt)
        cursor.insertText(header)
        self.text_edit.setTextCursor(cursor)

        self._typer_thread = TyperThread(text)
        self._typer_thread.char_ready.connect(
            lambda ch: self._append_char(ch, fmt)
        )
        self._typer_thread.finished.connect(
            lambda: self._on_typer_finished(wrap)
        )
        self._typer_thread.start()

    def _append_char(self, ch, fmt):
        """Append character"""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.setCharFormat(fmt)
        cursor.insertText(ch)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _on_typer_finished(self, wrap):
        """Typewriter effect finished"""
        if wrap:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n")
            self.text_edit.setTextCursor(cursor)
            self._status_start_pos = None

    def clear(self):
        """Clear log"""
        self.text_edit.clear()
        self._status_start_pos = None

    def _apply_style(self):
        """Apply style"""
        btn_r = 10
        cmd_bg = get_color('cmd_bg', '#1e1e1e')
        card_bg = get_color('card_bg', '#2d2d2d')
        hover_bg = get_color('hover_bg', '#3d3d3d')
        border = get_color('border', '#444444')
        text_primary = get_color('text_primary', '#ffffff')

        size_small = get_font_size('size_small', 11)
        size_normal = get_font_size('size_normal', 12)

        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {cmd_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                font-family: Consolas, Monaco, monospace;
                font-size: {size_small}px;
            }}
            QLabel {{
                color: {text_primary};
                font-family: Consolas, Monaco, monospace;
                font-size: {size_normal}px;
            }}
            QPushButton {{
                background-color: {card_bg};
                font-size: {size_small}px;
                color: {text_primary};
                border: none;
                border-radius: {btn_r}px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
        """)

    def refresh_style(self):
        """Refresh style (called when theme changes)"""
        self._apply_style()
