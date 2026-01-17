#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题管理器
支持颜色和字体配置的动态加载
"""

import configparser
from pathlib import Path
from typing import Optional
from PySide6.QtGui import QFont, QColor


class ColorSlot:
    """
    颜色槽位管理器（单例）
    从 UI.ini 的 [Light] 和 [Dark] 读取颜色配置
    """
    
    _instance: Optional['ColorSlot'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
        
        self._config_path = config_path or str(Path(__file__).parent.parent.parent / "config" / "ui.ini")
        self._current_theme = "light"
        self._colors = {}
        self._loaded = False
        self._initialized = True
    
    def _ensure_loaded(self):
        """确保配置已加载"""
        if not self._loaded:
            self._load_colors()
    
    def _load_colors(self):
        """从配置文件加载颜色"""
        try:
            config = configparser.ConfigParser()
            config.read(self._config_path, encoding='utf-8')
            
            self._colors.clear()
            
            # 加载 Light 主题
            if 'Light' in config:
                self._colors['light'] = dict(config['Light'])
            else:
                self._colors['light'] = {}
            
            # 加载 Dark 主题
            if 'Dark' in config:
                self._colors['dark'] = dict(config['Dark'])
            else:
                self._colors['dark'] = {}
            
            self._loaded = True
        except Exception as e:
            print(f"[ColorSlot] Error loading color config: {e}")
            self._colors = {'light': {}, 'dark': {}}
            self._loaded = True
    
    def set_theme(self, theme: str):
        """设置当前主题"""
        self._ensure_loaded()
        if theme in ['light', 'dark']:
            self._current_theme = theme
    
    def get_color(self, color_key: str, fallback: str = "#FFFFFF") -> str:
        """获取当前主题的颜色值（返回字符串）"""
        self._ensure_loaded()
        theme_colors = self._colors.get(self._current_theme, {})
        return theme_colors.get(color_key, fallback)
    
    def get_qcolor(self, color_key: str, fallback: str = "#FFFFFF") -> QColor:
        """获取 QColor 对象"""
        color_str = self.get_color(color_key, fallback)
        return QColor(color_str)
    
    def get_color_int(self, color_key: str, fallback: str = "#FFFFFF") -> tuple:
        """获取 RGB 整数元组"""
        color = self.get_qcolor(color_key, fallback)
        return (color.red(), color.green(), color.blue())
    
    def reload(self):
        """重新加载配置"""
        self._loaded = False
        self._load_colors()


class FontSlot:
    """
    字体槽位管理器（单例）
    从 UI.ini 的 [Font] 读取字体配置
    """
    
    _instance: Optional['FontSlot'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
        
        self._config_path = config_path or str(Path(__file__).parent.parent.parent / "config" / "ui.ini")
        self._family = "Arial"
        self._sizes = {}
        self._loaded = False
        self._initialized = True
    
    def _ensure_loaded(self):
        """确保配置已加载"""
        if not self._loaded:
            self._load_fonts()
    
    def _load_fonts(self):
        """从配置文件加载字体"""
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
            print(f"[FontSlot] Error loading font config: {e}")
            self._loaded = True
    
    def get_qfont(self, size_slot: str, fallback_size: int = 12) -> QFont:
        """获取 QFont 对象"""
        self._ensure_loaded()
        size = self._sizes.get(size_slot, fallback_size)
        font = QFont(self._family)
        font.setPointSize(size)
        return font
    
    def get_size(self, size_slot: str, fallback_size: int = 12) -> int:
        """只取字号（整数）"""
        self._ensure_loaded()
        return self._sizes.get(size_slot, fallback_size)
    
    def family(self) -> str:
        """获取当前字体 family"""
        self._ensure_loaded()
        return self._family
    
    def reload(self):
        """重新加载配置"""
        self._loaded = False
        self._load_fonts()


# ============================================================================
# 全局实例和便捷函数
# ============================================================================

_color_slot: Optional[ColorSlot] = None
_font_slot: Optional[FontSlot] = None


def get_color_slot() -> ColorSlot:
    """获取 ColorSlot 单例"""
    global _color_slot
    if _color_slot is None:
        _color_slot = ColorSlot()
    return _color_slot


def get_font_slot() -> FontSlot:
    """获取 FontSlot 单例"""
    global _font_slot
    if _font_slot is None:
        _font_slot = FontSlot()
    return _font_slot


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """获取颜色值（字符串）"""
    return get_color_slot().get_color(color_key, fallback)


def get_qcolor(color_key: str, fallback: str = "#FFFFFF") -> QColor:
    """获取 QColor 对象"""
    return get_color_slot().get_qcolor(color_key, fallback)


def get_color_int(color_key: str, fallback: str = "#FFFFFF") -> tuple:
    """获取 RGB 整数元组"""
    return get_color_slot().get_color_int(color_key, fallback)


def get_font(size_slot: str, fallback_size: int = 12) -> QFont:
    """获取字体对象"""
    return get_font_slot().get_qfont(size_slot, fallback_size)


def get_font_size(size_slot: str, fallback_size: int = 12) -> int:
    """获取字体大小（仅返回 int）"""
    return get_font_slot().get_size(size_slot, fallback_size)


def set_theme(theme: str):
    """设置全局主题"""
    get_color_slot().set_theme(theme)
