# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.widgets — 主题化输入控件 / Themed input widgets.

补 ``unitport_sdk/ui.py`` 缺的三类 QWidget 包装：

- ``LaviLineEdit``       —— 单行输入（替换 ``QLineEdit``）
- ``LaviSpinBox``        —— int 数字输入（替换 ``QSpinBox``）
- ``LaviDoubleSpinBox``  —— float 数字输入（替换 ``QDoubleSpinBox``）
- ``LaviCodeEditor``     —— 多行代码 / JSON 编辑（替换 ``QPlainTextEdit``）

这些控件：
- 全部通过 ``Config.get_color`` / ``Config.get_font_size`` 取颜色与字号
- 主题切换后调 ``refresh_style()`` 重新刷新 stylesheet
- 不嵌套 ``QGraphicsProxyWidget`` —— 仅在 ``QDialog`` 内使用，不进画布

设计跟 ``LaviButton`` / ``LaviComboBox`` 一致：构造时 apply stylesheet，
对外暴露 ``refresh_style()`` 接口。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

from unitport_sdk.sys import Config


# =============================================================================
# 主题 stylesheet 工具
# =============================================================================

def _input_stylesheet(target_class: str) -> str:
    """返回输入类控件的 QSS（QLineEdit / QSpinBox / QPlainTextEdit 通用）.

    所有颜色取自 ``system.ini[Theme]``，禁止 inline hex literals。
    """
    bg            = Config.get_color("bg_2")
    bg_focus      = Config.get_color("bg_3")
    text          = Config.get_color("main_t1")
    border        = Config.get_color("border_1")
    accent        = Config.get_color("highlight")
    disabled_text = Config.get_color("sub_t2")
    disabled_bg   = Config.get_color("bg_3")
    disabled_brd  = Config.get_color("canvas_node_sep")
    return f"""
        {target_class} {{
            background: {bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 4px 8px;
            selection-background-color: {accent};
            selection-color: {bg};
        }}
        {target_class}:focus {{
            background: {bg_focus};
            border: 1px solid {accent};
        }}
        {target_class}:disabled {{
            color: {disabled_text};
            background: {disabled_bg};
            border: 1px solid {disabled_brd};
        }}
    """


def _spin_stylesheet(target_class: str) -> str:
    """SpinBox 在普通输入 QSS 上加箭头按钮配色."""
    base = _input_stylesheet(target_class)
    btn_bg     = Config.get_color("btn_1")
    btn_hover  = Config.get_color("hover_1")
    arrow      = Config.get_color("main_t2")
    return base + f"""
        {target_class}::up-button, {target_class}::down-button {{
            background: {btn_bg};
            border: 0px;
            width: 16px;
            subcontrol-origin: border;
        }}
        {target_class}::up-button {{ subcontrol-position: top right; }}
        {target_class}::down-button {{ subcontrol-position: bottom right; }}
        {target_class}::up-button:hover, {target_class}::down-button:hover {{
            background: {btn_hover};
        }}
        {target_class}::up-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid {arrow};
            width: 0px;
            height: 0px;
        }}
        {target_class}::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {arrow};
            width: 0px;
            height: 0px;
        }}
    """


# =============================================================================
# LaviLineEdit
# =============================================================================

class LaviLineEdit(QLineEdit):
    """主题化单行输入 / Themed single-line input."""

    def __init__(
        self,
        text: str = "",
        *,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if text:
            self.setText(text)
        if placeholder:
            self.setPlaceholderText(placeholder)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_normal", 14)))
        self.setFont(font)
        self.setMinimumHeight(28)
        self.refresh_style()

    def refresh_style(self) -> None:
        self.setStyleSheet(_input_stylesheet("QLineEdit"))


# =============================================================================
# LaviSpinBox / LaviDoubleSpinBox
# =============================================================================

class LaviSpinBox(QSpinBox):
    """主题化整数输入 / Themed integer input."""

    def __init__(
        self,
        value: int = 0,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
        step: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setRange(
            int(minimum) if minimum is not None else -10**9,
            int(maximum) if maximum is not None else 10**9,
        )
        if step is not None and int(step) > 0:
            self.setSingleStep(int(step))
        self.setValue(int(value))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_normal", 14)))
        self.setFont(font)
        self.setMinimumHeight(28)
        self.refresh_style()

    def refresh_style(self) -> None:
        self.setStyleSheet(_spin_stylesheet("QSpinBox"))


class LaviDoubleSpinBox(QDoubleSpinBox):
    """主题化浮点输入 / Themed float input."""

    def __init__(
        self,
        value: float = 0.0,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        step: Optional[float] = None,
        decimals: int = 6,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setDecimals(int(decimals))
        self.setRange(
            float(minimum) if minimum is not None else -1e12,
            float(maximum) if maximum is not None else 1e12,
        )
        if step is not None:
            self.setSingleStep(float(step))
        self.setValue(float(value))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_normal", 14)))
        self.setFont(font)
        self.setMinimumHeight(28)
        self.refresh_style()

    def refresh_style(self) -> None:
        self.setStyleSheet(_spin_stylesheet("QDoubleSpinBox"))


# =============================================================================
# LaviCodeEditor
# =============================================================================

class JsonHighlighter(QSyntaxHighlighter):
    """轻量 JSON 高亮：字符串 / 数字 / 关键字 / 标点."""

    def __init__(self, doc) -> None:
        super().__init__(doc)
        # 走专门的 syntax_json_* 槽位（system.ini[Theme]），禁止 inline hex
        self._fmt_string = QTextCharFormat()
        self._fmt_string.setForeground(QColor(Config.get_color("syntax_json_string")))
        self._fmt_number = QTextCharFormat()
        self._fmt_number.setForeground(QColor(Config.get_color("misc_1")))
        self._fmt_kw = QTextCharFormat()
        self._fmt_kw.setForeground(QColor(Config.get_color("misc_2")))
        self._fmt_punct = QTextCharFormat()
        self._fmt_punct.setForeground(QColor(Config.get_color("main_t1")))

    def highlightBlock(self, text: str) -> None:
        import re
        # strings (含转义；不严格但够用)
        for m in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_string)
        # numbers
        for m in re.finditer(r"\b-?\d+(?:\.\d+)?\b", text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_number)
        # keywords
        for m in re.finditer(r"\b(true|false|null)\b", text):
            self.setFormat(m.start(), m.end() - m.start(), self._fmt_kw)
        # punctuation
        for ch_idx, ch in enumerate(text):
            if ch in "{}[]:,":
                self.setFormat(ch_idx, 1, self._fmt_punct)


class LaviCodeEditor(QPlainTextEdit):
    """主题化代码 / JSON 编辑器 / Themed code editor.

    - 等宽字体 + 行号留白（行号本身留给 future enhancement）
    - 不换行（``LineWrapMode.NoWrap``）
    - ``language="json"`` 时打开轻量语法高亮
    """

    def __init__(
        self,
        text: str = "",
        *,
        language: str = "python",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if text:
            self.setPlainText(text)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # 等宽字体
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(int(Config.get_font_size("size_normal", 14)))
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance("    "))
        self._language = language
        self._highlighter: Optional[QSyntaxHighlighter] = None
        if language == "json":
            self._highlighter = JsonHighlighter(self.document())
        self.refresh_style()

    def refresh_style(self) -> None:
        self.setStyleSheet(_input_stylesheet("QPlainTextEdit"))


# =============================================================================
# 工厂便捷函数（与 setButton / setComboBox 风格统一）
# =============================================================================

def setLineEdit(
    text: str = "",
    *,
    placeholder: str = "",
    parent: Optional[QWidget] = None,
) -> LaviLineEdit:
    return LaviLineEdit(text=text, placeholder=placeholder, parent=parent)


def setSpinBox(
    value: int = 0,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    step: Optional[int] = None,
    parent: Optional[QWidget] = None,
) -> LaviSpinBox:
    return LaviSpinBox(value, minimum=minimum, maximum=maximum, step=step, parent=parent)


def setDoubleSpinBox(
    value: float = 0.0,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,
    decimals: int = 6,
    parent: Optional[QWidget] = None,
) -> LaviDoubleSpinBox:
    return LaviDoubleSpinBox(
        value,
        minimum=minimum,
        maximum=maximum,
        step=step,
        decimals=decimals,
        parent=parent,
    )


def setCodeEditor(
    text: str = "",
    *,
    language: str = "python",
    parent: Optional[QWidget] = None,
) -> LaviCodeEditor:
    return LaviCodeEditor(text=text, language=language, parent=parent)


__all__ = [
    "LaviLineEdit",
    "LaviSpinBox",
    "LaviDoubleSpinBox",
    "LaviCodeEditor",
    "setLineEdit",
    "setSpinBox",
    "setDoubleSpinBox",
    "setCodeEditor",
]
