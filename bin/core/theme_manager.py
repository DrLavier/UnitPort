#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theme Manager
Supports dynamic loading of color and font configurations
"""

import configparser
from pathlib import Path
from typing import Optional
from PySide6.QtGui import QFont, QColor


# Global config path - set by main.py via init_theme_manager()
_ui_config_path: Optional[str] = None


def init_theme_manager(config_path: str):
    """
    Initialize theme manager with config path from main.py

    Args:
        config_path: Path to ui.ini file
    """
    global _ui_config_path
    _ui_config_path = config_path

    # Reinitialize slots if they exist
    global _color_slot, _font_slot
    if _color_slot is not None:
        _color_slot._config_path = config_path
        _color_slot.reload()
    if _font_slot is not None:
        _font_slot._config_path = config_path
        _font_slot.reload()


def _get_default_config_path() -> str:
    """Get default config path (fallback)"""
    if _ui_config_path:
        return _ui_config_path
    # Fallback to relative path calculation
    return str(Path(__file__).parent.parent.parent / "config" / "ui.ini")


class ColorSlot:
    """
    Color slot manager (singleton)
    Reads color config from [Light] and [Dark] sections of ui.ini
    """

    _instance: Optional['ColorSlot'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config_path = _get_default_config_path()
        self._current_theme = "dark"
        self._colors = {}
        self._loaded = False
        self._initialized = True

    def _ensure_loaded(self):
        """Ensure config is loaded"""
        if not self._loaded:
            self._load_colors()

    def _load_colors(self):
        """Load colors from config file"""
        try:
            config = configparser.ConfigParser()
            config.read(self._config_path, encoding='utf-8')

            self._colors.clear()

            # Load Light theme
            if 'Light' in config:
                self._colors['light'] = dict(config['Light'])
            else:
                self._colors['light'] = {}

            # Load Dark theme
            if 'Dark' in config:
                self._colors['dark'] = dict(config['Dark'])
            else:
                self._colors['dark'] = {}

            self._loaded = True
        except Exception as e:
            # Silent fallback - avoid print in production
            self._colors = {'light': {}, 'dark': {}}
            self._loaded = True

    def set_theme(self, theme: str):
        """Set current theme"""
        self._ensure_loaded()
        if theme in ['light', 'dark']:
            self._current_theme = theme

    def get_color(self, color_key: str, fallback: str = "#FFFFFF") -> str:
        """Get color value for current theme (returns string)"""
        self._ensure_loaded()
        theme_colors = self._colors.get(self._current_theme, {})
        return theme_colors.get(color_key, fallback)

    def get_qcolor(self, color_key: str, fallback: str = "#FFFFFF") -> QColor:
        """Get QColor object"""
        color_str = self.get_color(color_key, fallback)
        return QColor(color_str)

    def get_color_int(self, color_key: str, fallback: str = "#FFFFFF") -> tuple:
        """Get RGB integer tuple"""
        color = self.get_qcolor(color_key, fallback)
        return (color.red(), color.green(), color.blue())

    def reload(self):
        """Reload config"""
        self._loaded = False
        self._load_colors()


class FontSlot:
    """
    Font slot manager (singleton)
    Reads font config from [Font] section of ui.ini
    """

    _instance: Optional['FontSlot'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config_path = _get_default_config_path()
        self._family = "Arial"
        self._sizes = {}
        self._loaded = False
        self._initialized = True

    def _ensure_loaded(self):
        """Ensure config is loaded"""
        if not self._loaded:
            self._load_fonts()

    def _load_fonts(self):
        """Load fonts from config file"""
        try:
            config = configparser.ConfigParser()
            config.read(self._config_path, encoding='utf-8')

            if 'Font' not in config:
                self._family = "Arial"
                self._sizes = {}
                self._loaded = True
                return

            font_section = config['Font']
            self._family = font_section.get('family', 'Arial')

            self._sizes.clear()
            for key in font_section.keys():
                if key == "family":
                    continue
                try:
                    self._sizes[key] = font_section.getint(key)
                except ValueError:
                    continue

            self._loaded = True
        except Exception as e:
            # Silent fallback
            self._loaded = True

    def get_qfont(self, size_slot: str, fallback_size: int = 12) -> QFont:
        """Get QFont object"""
        self._ensure_loaded()
        size = self._sizes.get(size_slot, fallback_size)
        font = QFont(self._family)
        font.setPointSize(size)
        return font

    def get_size(self, size_slot: str, fallback_size: int = 12) -> int:
        """Get font size (int only)"""
        self._ensure_loaded()
        return self._sizes.get(size_slot, fallback_size)

    def family(self) -> str:
        """Get current font family"""
        self._ensure_loaded()
        return self._family

    def reload(self):
        """Reload config"""
        self._loaded = False
        self._load_fonts()


# ============================================================================
# Global instances and convenience functions
# ============================================================================

_color_slot: Optional[ColorSlot] = None
_font_slot: Optional[FontSlot] = None


def get_color_slot() -> ColorSlot:
    """Get ColorSlot singleton"""
    global _color_slot
    if _color_slot is None:
        _color_slot = ColorSlot()
    return _color_slot


def get_font_slot() -> FontSlot:
    """Get FontSlot singleton"""
    global _font_slot
    if _font_slot is None:
        _font_slot = FontSlot()
    return _font_slot


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """Get color value (string)"""
    return get_color_slot().get_color(color_key, fallback)


def get_qcolor(color_key: str, fallback: str = "#FFFFFF") -> QColor:
    """Get QColor object"""
    return get_color_slot().get_qcolor(color_key, fallback)


def get_color_int(color_key: str, fallback: str = "#FFFFFF") -> tuple:
    """Get RGB integer tuple"""
    return get_color_slot().get_color_int(color_key, fallback)


def get_font(size_slot: str, fallback_size: int = 12) -> QFont:
    """Get font object"""
    return get_font_slot().get_qfont(size_slot, fallback_size)


def get_font_size(size_slot: str, fallback_size: int = 12) -> int:
    """Get font size (int only)"""
    return get_font_slot().get_size(size_slot, fallback_size)


def set_theme(theme: str):
    """Set global theme"""
    get_color_slot().set_theme(theme)
