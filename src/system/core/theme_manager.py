#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theme Manager
Supports dynamic loading of color and font configurations
"""

import configparser
from pathlib import Path
from typing import Optional, Union
from PySide6.QtGui import QFont, QColor, QIcon


# Global config path - set by main.py via init_theme_manager()
_ui_config_path: Optional[str] = None

# Icon asset root — set once by init_theme_manager() or lazily resolved
_icon_dir: Optional[Path] = None


def init_theme_manager(config_path: str):
    """
    Initialize theme manager with config path from main.py

    Args:
        config_path: Path to ui.ini file
    """
    global _ui_config_path
    _ui_config_path = config_path

    # Reinitialize slots if they exist
    global _color_slot, _node_color_slot, _font_slot
    if _color_slot is not None:
        _color_slot._config_path = config_path
        _color_slot.reload()
    if _node_color_slot is not None:
        _node_color_slot._config_path = config_path
        _node_color_slot.reload()
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


class NodeColorSlot:
    """
    Node color slot manager (singleton)
    Reads gradient colors from theme-aware sections in ui.ini.
    Sections:
    - [LightNodeColors]
    - [DarkNodeColors]
    """

    _instance: Optional['NodeColorSlot'] = None

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
        self._colors = {"light": {}, "dark": {}}
        self._loaded = False
        self._initialized = True

    def _ensure_loaded(self):
        """Ensure config is loaded"""
        if not self._loaded:
            self._load_colors()

    def _load_colors(self):
        """Load node colors from config file"""
        try:
            config = configparser.ConfigParser()
            config.read(self._config_path, encoding='utf-8')

            light_colors = {}
            dark_colors = {}
            if 'LightNodeColors' in config:
                light_colors.update(dict(config['LightNodeColors']))
            if 'DarkNodeColors' in config:
                dark_colors.update(dict(config['DarkNodeColors']))

            self._colors = {
                'light': light_colors,
                'dark': dark_colors,
            }

            self._loaded = True
        except Exception:
            self._colors = {'light': {}, 'dark': {}}
            self._loaded = True

    def get_color(self, color_key: str, fallback: str = "#2d2d2d") -> str:
        """Get node color value (string)"""
        self._ensure_loaded()
        theme_colors = self._colors.get(self._current_theme, {})
        return theme_colors.get(color_key, fallback)

    def get_pair(self, prefix: str, fallback_start: str = "#2d2d2d", fallback_end: str = "#2d2d2d") -> tuple:
        """Get gradient pair for a node type prefix"""
        start_key = f"{prefix}_start"
        end_key = f"{prefix}_end"
        return (
            self.get_color(start_key, fallback_start),
            self.get_color(end_key, fallback_end)
        )

    def reload(self):
        """Reload config"""
        self._loaded = False
        self._load_colors()

    def set_theme(self, theme: str):
        """Set current theme for node palette lookup."""
        self._ensure_loaded()
        if theme in ['light', 'dark']:
            self._current_theme = theme


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
_node_color_slot: Optional[NodeColorSlot] = None
_font_slot: Optional[FontSlot] = None


def get_color_slot() -> ColorSlot:
    """Get ColorSlot singleton"""
    global _color_slot
    if _color_slot is None:
        _color_slot = ColorSlot()
    return _color_slot


def get_node_color_slot() -> NodeColorSlot:
    """Get NodeColorSlot singleton"""
    global _node_color_slot
    if _node_color_slot is None:
        _node_color_slot = NodeColorSlot()
    return _node_color_slot


def get_font_slot() -> FontSlot:
    """Get FontSlot singleton"""
    global _font_slot
    if _font_slot is None:
        _font_slot = FontSlot()
    return _font_slot


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """Get color value (string)"""
    return get_color_slot().get_color(color_key, fallback)


def get_color_for_theme(color_key: str, theme: str, fallback: str = "#FFFFFF") -> str:
    """Get color from a specific theme section without mutating global theme state."""
    slot = get_color_slot()
    slot._ensure_loaded()
    theme_key = (theme or "dark").lower()
    if theme_key not in ("light", "dark"):
        theme_key = "dark"
    return slot._colors.get(theme_key, {}).get(color_key, fallback)


def get_node_color(color_key: str, fallback: str = "#2d2d2d") -> str:
    """Get node color value (string)"""
    return get_node_color_slot().get_color(color_key, fallback)


def get_node_color_pair(prefix: str, fallback_start: str = "#2d2d2d", fallback_end: str = "#2d2d2d") -> tuple:
    """Get node gradient pair for a category"""
    return get_node_color_slot().get_pair(prefix, fallback_start, fallback_end)


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
    get_node_color_slot().set_theme(theme)


# ============================================================================
# Icon helper
# ============================================================================

def _get_icon_dir() -> Path:
    """Return the icon asset directory (bin/assets/icon inside project root)."""
    global _icon_dir
    if _icon_dir is None:
        # Resolve from theme_manager.py location:
        # src/system/core/theme_manager.py -> src/system/core -> src/system -> src -> NEW/
        _icon_dir = Path(__file__).parent.parent.parent.parent / "bin" / "assets" / "icon"
    return _icon_dir


def get_icon(name: str) -> QIcon:
    """Return a QIcon for the given icon name, auto-selecting dark/light variant.

    Naming convention in ``bin/assets/icon/``:
    - Light-theme file: ``<name>.svg``
    - Dark-theme file:  ``<name>_w.svg``

    The function checks the current global theme and picks the appropriate
    variant.  If the preferred file does not exist it falls back to the other
    variant, then to an empty ``QIcon()``.

    *name* can be passed with or without the ``icon_`` prefix and with or
    without the ``.svg`` extension — they are normalised away before lookup.

    Examples::

        get_icon("play")          # icon_play.svg / icon_play_w.svg
        get_icon("icon_play")     # same
        get_icon("ICON_CL")      # ICON_CL.svg / ICON_CL_W.svg
        get_icon("START_LOGO")    # START_LOGO.svg / START_LOGO_W.svg
        get_icon("logo_w")        # logo_w.svg (returned as-is)
    """
    icon_dir = _get_icon_dir()

    # Strip .svg suffix if provided
    if name.lower().endswith(".svg"):
        name = name[:-4]

    # Determine if this is an UPPER_CASE icon (window-control style: ICON_CL, START_LOGO)
    is_upper = name.startswith("ICON_") or name.startswith("START_")

    # Build dark/light filenames
    if is_upper:
        dark_name = f"{name}_W.svg"
        light_name = f"{name}.svg"
    else:
        # Ensure icon_ prefix for lowercase convention icons
        base = name if name.startswith("icon_") else f"icon_{name}"
        dark_name = f"{base}_w.svg"
        light_name = f"{base}.svg"

    # Pick based on current theme
    try:
        is_dark = get_color_slot()._current_theme == "dark"
    except Exception:
        is_dark = True

    preferred = dark_name if is_dark else light_name
    fallback = light_name if is_dark else dark_name

    preferred_path = icon_dir / preferred
    if preferred_path.exists():
        icon = QIcon(str(preferred_path))
        if not icon.isNull():
            return icon

    fallback_path = icon_dir / fallback
    if fallback_path.exists():
        icon = QIcon(str(fallback_path))
        if not icon.isNull():
            return icon

    return QIcon()


def get_icon_path(name: str) -> str:
    """Return the resolved filesystem path string for an icon (theme-aware).

    Same lookup logic as :func:`get_icon` but returns a ``str`` path instead
    of a ``QIcon``.  Useful for ``QSvgWidget`` which takes a path string.
    Returns ``""`` when no matching file exists.
    """
    icon_dir = _get_icon_dir()

    if name.lower().endswith(".svg"):
        name = name[:-4]

    is_upper = name.startswith("ICON_") or name.startswith("START_")

    if is_upper:
        dark_name = f"{name}_W.svg"
        light_name = f"{name}.svg"
    else:
        base = name if name.startswith("icon_") else f"icon_{name}"
        dark_name = f"{base}_w.svg"
        light_name = f"{base}.svg"

    try:
        is_dark = get_color_slot()._current_theme == "dark"
    except Exception:
        is_dark = True

    preferred = dark_name if is_dark else light_name
    fallback = light_name if is_dark else dark_name

    preferred_path = icon_dir / preferred
    if preferred_path.exists():
        return str(preferred_path)

    fallback_path = icon_dir / fallback
    if fallback_path.exists():
        return str(fallback_path)

    return ""
