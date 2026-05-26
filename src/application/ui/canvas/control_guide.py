# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasControlGuide — 画布左下角悬浮操控提示图.

实现形式参考 ``CanvasMiniMap``：作为 ``QGraphicsView.viewport()`` 的子 widget，
固定像素位置（不随 scene pan / zoom 移动），透明背景、无边框、不接收任何交互——
纯展示。内部委托给 ``QSvgWidget`` 渲染 SVG，保证可见性（与 loading_screen 同样的
渲染路径）。
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import QGraphicsView, QWidget


class CanvasControlGuide(QWidget):
    """Bottom-left floating control-guide overlay (fixed viewport position)."""

    _HEIGHT = 100

    def __init__(self, graph_view: QGraphicsView, svg_path: Path) -> None:
        super().__init__(graph_view.viewport())
        self.setObjectName("canvasControlGuide")
        # 不接收任何鼠标 / 焦点；纯展示
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # 内嵌 QSvgWidget 负责实际渲染 —— 与 loading_screen.LogoPulse 同路径
        self._svg = QSvgWidget(str(svg_path), self)
        self._svg.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._svg.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # 按 SVG 默认尺寸算等比宽度（拿不到时退化为正方形）
        renderer = self._svg.renderer()
        default_size = renderer.defaultSize() if renderer is not None else None
        if (
            default_size is not None
            and default_size.width() > 0
            and default_size.height() > 0
        ):
            ratio = default_size.width() / default_size.height()
        else:
            ratio = 1.0
        h = self._HEIGHT
        w = max(1, int(round(h * ratio)))
        self.setFixedSize(w, h)
        self._svg.setFixedSize(w, h)
        self._svg.move(0, 0)
        # 父 widget 已经 show 时（CanvasPage __init__ 后期构造），子 widget 需显式 show
        self._svg.show()
        self.show()
