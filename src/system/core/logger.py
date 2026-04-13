#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger Module
Thread-safe log signal transmission and UI display
"""

import sys
import threading
from typing import Optional
from datetime import datetime

from loguru import logger as _loguru

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QColor, QTextCursor, QTextCharFormat

from src.system.core.config_manager import ConfigManager
from src.system.core.theme_manager import get_font_size, get_color
from src.system.core.localisation import tr
from bin.pages.layout.misc import SwitchButton

# ---------------------------------------------------------------------------
# Loguru → external CMD (stderr) setup
# ---------------------------------------------------------------------------
_loguru.remove()  # remove default handler to avoid duplicate output
_loguru.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<level>{message}</level>",
    level="DEBUG",
    colorize=True,
)


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
    _loguru.info(text)
    get_log_signal().info(text, wrap, typer)


def log_debug(text, wrap=True, typer=False):
    """Debug log"""
    text = str(text)
    _loguru.debug(text)
    get_log_signal().debug(text, wrap, typer)


def log_warning(text, wrap=True, typer=False):
    """Warning log"""
    text = str(text)
    _loguru.warning(text)
    get_log_signal().warning(text, wrap, typer)


def log_error(text, wrap=True, typer=False):
    """Error log"""
    text = str(text)
    _loguru.error(text)
    get_log_signal().error(text, wrap, typer)


def log_success(text, wrap=True, typer=False):
    """Success log"""
    text = str(text)
    _loguru.success(text)
    get_log_signal().success(text, wrap, typer)


# Backward-compatible aliases for legacy/dynamic call sites.
def logger_debug(text, wrap=True, typer=False):
    log_debug(text, wrap=wrap, typer=typer)


def logger_info(text, wrap=True, typer=False):
    log_info(text, wrap=wrap, typer=typer)


def logger_warning(text, wrap=True, typer=False):
    log_warning(text, wrap=wrap, typer=typer)


def logger_error(text, wrap=True, typer=False):
    log_error(text, wrap=wrap, typer=typer)


def logger_success(text, wrap=True, typer=False):
    log_success(text, wrap=wrap, typer=typer)


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

    _DEBUG_CONFIG_SECTION = "DEBUG"
    _DEBUG_CONFIG_KEY = "show_debug_log"

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
        self.setObjectName("cmdLogPanel")

        self._config = ConfigManager()
        self._status_start_pos: Optional[int] = None
        self._typer_thread: Optional[TyperThread] = None
        self._show_debug_logs = self._load_debug_log_preference()
        self._log_entries = []

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

        debug_label = QLabel("Debug log:")
        debug_label.setObjectName("cmdDebugLabel")
        debug_label.setMaximumHeight(35)
        hl.addWidget(debug_label)

        self.debug_switch = SwitchButton(header, checked=self._show_debug_logs)
        self.debug_switch.toggled.connect(self._on_debug_switch_toggled)
        hl.addWidget(self.debug_switch)
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
        text_str = str(text)
        entry = {
            "text": text_str,
            "log_type": str(log_type),
            "wrap": bool(wrap),
            "typer": bool(typer),
        }
        self._log_entries.append(entry)
        if not self._should_show_entry(entry):
            return
        if typer:
            self._start_typer(text_str, log_type, wrap)
        else:
            self._append_log(text_str, log_type, wrap)

    def _on_debug_switch_toggled(self, checked: bool):
        self._show_debug_logs = bool(checked)
        self._save_debug_log_preference(self._show_debug_logs)
        self._rerender_logs()

    def _load_debug_log_preference(self) -> bool:
        try:
            return bool(
                self._config.get_bool(
                    self._DEBUG_CONFIG_SECTION,
                    self._DEBUG_CONFIG_KEY,
                    fallback=False,
                    config_type="user",
                )
            )
        except Exception:
            return False

    def _save_debug_log_preference(self, checked: bool) -> None:
        try:
            self._config.set(
                self._DEBUG_CONFIG_SECTION,
                self._DEBUG_CONFIG_KEY,
                "true" if checked else "false",
                config_type="user",
            )
            self._config.save_user_config()
        except Exception:
            pass

    def _should_show_entry(self, entry) -> bool:
        text_str = str(entry.get("text", ""))
        log_type = str(entry.get("log_type", "")).strip().lower()
        if not self._show_debug_logs and (log_type == "debug" or "[DEBUG]" in text_str):
            return False
        return True

    def _rerender_logs(self):
        if self._typer_thread and self._typer_thread.isRunning():
            self._typer_thread.stop()
            self._typer_thread.wait()
        self.text_edit.clear()
        self._status_start_pos = None
        for entry in self._log_entries:
            if self._should_show_entry(entry):
                self._append_log(entry["text"], entry["log_type"], entry["wrap"])

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
        self._log_entries.clear()

    def _apply_style(self):
        """Apply style"""
        btn_r = 10
        # Keep QTextEdit as blackboard, but panel chrome follows app theme.
        cmd_bg = "#0B0F13"
        panel_bg = get_color("bg", "#1e1e1e")
        header_bg = get_color("card_bg", "#2d2d2d")
        hover_bg = get_color("hover_bg", "#3d3d3d")
        border = get_color("border", "#444444")
        text_primary = get_color("text_primary", "#ffffff")

        size_small = get_font_size('size_small', 11)
        size_normal = get_font_size('size_normal', 12)

        self.setStyleSheet(f"""
            #cmdLogPanel {{
                background-color: {panel_bg};
                border: none;
            }}
            #cmdDebugLabel {{
                background-color: transparent;
                color: {text_primary};
                border: none;
                border-radius: 0px;
                padding: 0px;
                font-family: Consolas, Monaco, monospace;
                font-size: {size_normal}px;
            }}
            QTextEdit {{
                background-color: {cmd_bg};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                font-family: Consolas, Monaco, monospace;
                font-size: {size_small}px;
            }}
            QLabel {{
                background-color: {header_bg};
                color: {text_primary};
                font-family: Consolas, Monaco, monospace;
                font-size: {size_normal}px;
            }}
            QPushButton {{
                background-color: {header_bg};
                font-size: {size_small}px;
                color: {text_primary};
                border: 1px solid {border};
                border-radius: {btn_r}px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
        """)
        if hasattr(self, "debug_switch"):
            self.debug_switch.refresh_style()

    def refresh_style(self):
        """Refresh style (called when theme changes)"""
        self._apply_style()
