# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.choice_box — 下拉选择框 paint helper.

替换 DEMO ``MainPicker`` 的视觉契约（DEMO node_ui_rows.py:1352-1436）：
    [文本                          ▾]
左侧文字（截断带省略号），右侧三角箭头；hover 时背景加亮、边框加深；
``selected`` 态用 ``highlight`` 底色 + ``alt_t1`` 文字（强调当前选中）。

颜色 **全部** 走 ``Config.get_color`` 取自 ``system.ini``，本模块不允许定义
自己的颜色字段。新增视觉 slot 必须先加进 ``system.ini[Theme]``。

调用方负责：
- 几何（rect 是 item-local QRectF）
- 当前显示文本
- 鼠标命中 + 弹 ``QMenu``（SDK 不弹菜单——属于交互层职责）

LOD 建议：
    T0/T1 → 跳过本函数，画一根静态文字即可
    T2+   → 走本函数完整视觉
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF

from unitport_sdk.sys import Config


def draw_choice_dropdown(
    painter: QPainter,
    rect: QRectF,
    text: str,
    *,
    hover: bool = False,
    open_state: bool = False,
    selected: bool = False,
    font: Optional[QFont] = None,
) -> None:
    """画下拉选择框 / Paint a choice dropdown.

    Args:
        painter: ``QPainter``
        rect: 选择框矩形（item-local）
        text: 当前选项文字（自动右侧截断省略号）
        hover: 鼠标悬停态（背景加亮）
        open_state: 弹出菜单展开中（箭头反向）
        selected: 选中/激活态——底色 ``highlight``、文字 ``alt_t1``
        font: 文字字体（None 用 painter 当前字体）
    """
    # ---- 颜色全部走 system.ini，禁止本地常量 ----
    # 与 CodeRow / IndexRow / RangeRow 端点按钮统一：
    # bg=canvas_node_title, border=canvas_node_border, text=canvas_node_title_text；
    # hover 走 .lighter(115) 与 CodeRow 行内按钮完全一致；
    # selected 维持主题黄底 + alt_t1 字体（popup 卡片同一规则）。
    if selected:
        bg_color = QColor(Config.get_color("highlight"))
        text_color = QColor(Config.get_color("bg_1"))
        arrow_color = QColor(Config.get_color("bg_1"))
    else:
        base_bg = QColor(Config.get_color("btn_1"))
        bg_color = base_bg.lighter(115) if hover else base_bg
        text_color = QColor(Config.get_color("main_t1"))
        arrow_color = QColor(Config.get_color("main_t1"))
    border_color = QColor(Config.get_color("border_1"))

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 1. 背景 + 边框
    painter.setBrush(QBrush(bg_color))
    painter.setPen(QPen(border_color, 1.0))
    painter.drawRoundedRect(rect, 3.0, 3.0)

    # 2. 文字（右边留 18px 给箭头）
    if font is not None:
        painter.setFont(font)
    painter.setPen(QPen(text_color))
    pad_left = 8.0
    text_rect = QRectF(
        rect.left() + pad_left,
        rect.top(),
        max(0.0, rect.width() - pad_left - 18.0),
        rect.height(),
    )
    metrics = painter.fontMetrics()
    elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(text_rect.width()))
    painter.drawText(
        text_rect,
        int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
        elided,
    )

    # 3. 箭头 ▾ / ▴
    arrow_cx = rect.right() - 10.0
    arrow_cy = rect.top() + rect.height() * 0.5
    arrow_w = 4.5
    arrow_h = 3.5
    if open_state:
        # ▴
        tri = QPolygonF([
            QPointF(arrow_cx, arrow_cy - arrow_h * 0.5),
            QPointF(arrow_cx - arrow_w, arrow_cy + arrow_h * 0.5),
            QPointF(arrow_cx + arrow_w, arrow_cy + arrow_h * 0.5),
        ])
    else:
        # ▾
        tri = QPolygonF([
            QPointF(arrow_cx, arrow_cy + arrow_h * 0.5),
            QPointF(arrow_cx - arrow_w, arrow_cy - arrow_h * 0.5),
            QPointF(arrow_cx + arrow_w, arrow_cy - arrow_h * 0.5),
        ])
    painter.setBrush(QBrush(arrow_color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPolygon(tri)

    painter.restore()


__all__ = ["draw_choice_dropdown"]
