"""unitport_sdk.canvas.edit_button — 编辑按钮 paint helper.

替换 DEMO ``CodeEdit`` 按钮（training_node_items.py:1341-1470）+ Path 浏览
按钮 + Pencil 编辑按钮 三种视觉，统一一个函数 ``draw_edit_button(kind=...)``。

颜色 **全部** 走 ``Config.get_color`` 取自 ``system.ini``，本模块不允许定义
自己的颜色字段。新增视觉 slot 必须先加进 ``system.ini[Theme]``。

调用方负责：
- 几何（rect 是 item-local QRectF）
- hover / pressed 状态判断
- 单击触发（弹 ``QFileDialog`` / ``CodeEditorDialog`` 等——SDK 不弹）

LOD 建议：T0/T1 跳过；T2+ 完整视觉。
"""

from __future__ import annotations

from typing import Literal

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

from unitport_sdk.sys import Config


ButtonKind = Literal["pencil", "code", "browse", "play", "save"]


def draw_edit_button(
    painter: QPainter,
    rect: QRectF,
    *,
    kind: ButtonKind = "pencil",
    hover: bool = False,
    pressed: bool = False,
) -> None:
    """画编辑按钮 / Paint an edit-launcher button.

    Args:
        painter: ``QPainter``
        rect: 按钮矩形（item-local）
        kind: 图标类型——pencil（铅笔）/ code（⌘ 大括号）/ browse（···）/
            play（▶）/ save（💾 简化）
        hover / pressed: 视觉状态
    """
    # ---- 颜色全部走 system.ini，禁止本地常量 ----
    if pressed:
        bg_color = QColor(Config.get_color("bg_3"))
    elif hover:
        bg_color = QColor(Config.get_color("hover_1"))
    else:
        bg_color = QColor(Config.get_color("btn_1"))
    border_color = QColor(Config.get_color("border_1"))
    icon_color = QColor(
        Config.get_color("checked_1") if kind == "code"
        else Config.get_color("main_t1")
    )

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 1. 背景 + 边框
    painter.setBrush(QBrush(bg_color))
    painter.setPen(QPen(border_color, 1.0))
    painter.drawRoundedRect(rect, 3.0, 3.0)

    # 2. 图标（按 kind 走不同几何）
    cx = rect.left() + rect.width() * 0.5
    cy = rect.top() + rect.height() * 0.5
    size = min(rect.width(), rect.height()) * 0.5

    if kind == "pencil":
        _draw_pencil(painter, cx, cy, size, icon_color)
    elif kind == "code":
        _draw_code(painter, cx, cy, size, icon_color)
    elif kind == "browse":
        _draw_dots(painter, cx, cy, size, icon_color)
    elif kind == "play":
        _draw_play(painter, cx, cy, size, icon_color)
    elif kind == "save":
        _draw_save(painter, cx, cy, size, icon_color)

    painter.restore()


# =============================================================================
# 图标几何（小工具）
# =============================================================================

def _draw_pencil(p: QPainter, cx: float, cy: float, s: float, color: QColor) -> None:
    """斜置铅笔 / pencil tilted 45°."""
    pen = QPen(color, max(1.4, s * 0.18))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(cx - s * 0.55, cy + s * 0.55), QPointF(cx + s * 0.55, cy - s * 0.55))
    # 笔头小三角
    p.setBrush(QBrush(color))
    p.setPen(Qt.PenStyle.NoPen)
    tip = QPolygonF([
        QPointF(cx - s * 0.65, cy + s * 0.65),
        QPointF(cx - s * 0.35, cy + s * 0.55),
        QPointF(cx - s * 0.55, cy + s * 0.35),
    ])
    p.drawPolygon(tip)


def _draw_code(p: QPainter, cx: float, cy: float, s: float, color: QColor) -> None:
    """{·} 代码大括号."""
    pen = QPen(color, max(1.4, s * 0.16))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    # 左大括号 {
    p.drawLine(QPointF(cx - s * 0.45, cy - s * 0.6), QPointF(cx - s * 0.6, cy - s * 0.6))
    p.drawLine(QPointF(cx - s * 0.6, cy - s * 0.6), QPointF(cx - s * 0.6, cy - s * 0.1))
    p.drawLine(QPointF(cx - s * 0.6, cy - s * 0.1), QPointF(cx - s * 0.75, cy))
    p.drawLine(QPointF(cx - s * 0.75, cy), QPointF(cx - s * 0.6, cy + s * 0.1))
    p.drawLine(QPointF(cx - s * 0.6, cy + s * 0.1), QPointF(cx - s * 0.6, cy + s * 0.6))
    p.drawLine(QPointF(cx - s * 0.6, cy + s * 0.6), QPointF(cx - s * 0.45, cy + s * 0.6))
    # 右大括号 }
    p.drawLine(QPointF(cx + s * 0.45, cy - s * 0.6), QPointF(cx + s * 0.6, cy - s * 0.6))
    p.drawLine(QPointF(cx + s * 0.6, cy - s * 0.6), QPointF(cx + s * 0.6, cy - s * 0.1))
    p.drawLine(QPointF(cx + s * 0.6, cy - s * 0.1), QPointF(cx + s * 0.75, cy))
    p.drawLine(QPointF(cx + s * 0.75, cy), QPointF(cx + s * 0.6, cy + s * 0.1))
    p.drawLine(QPointF(cx + s * 0.6, cy + s * 0.1), QPointF(cx + s * 0.6, cy + s * 0.6))
    p.drawLine(QPointF(cx + s * 0.6, cy + s * 0.6), QPointF(cx + s * 0.45, cy + s * 0.6))
    # 中间一个小点
    p.setBrush(QBrush(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(cx, cy), s * 0.1, s * 0.1)


def _draw_dots(p: QPainter, cx: float, cy: float, s: float, color: QColor) -> None:
    """··· 三个圆点（browse / more）."""
    p.setBrush(QBrush(color))
    p.setPen(Qt.PenStyle.NoPen)
    r = max(1.5, s * 0.13)
    p.drawEllipse(QPointF(cx - s * 0.5, cy), r, r)
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.drawEllipse(QPointF(cx + s * 0.5, cy), r, r)


def _draw_play(p: QPainter, cx: float, cy: float, s: float, color: QColor) -> None:
    """▶ 三角形播放按钮."""
    p.setBrush(QBrush(color))
    p.setPen(Qt.PenStyle.NoPen)
    tri = QPolygonF([
        QPointF(cx - s * 0.4, cy - s * 0.55),
        QPointF(cx - s * 0.4, cy + s * 0.55),
        QPointF(cx + s * 0.55, cy),
    ])
    p.drawPolygon(tri)


def _draw_save(p: QPainter, cx: float, cy: float, s: float, color: QColor) -> None:
    """💾 简化软盘图标（外框 + 顶部缺角）."""
    pen = QPen(color, max(1.4, s * 0.14))
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    rect = QRectF(cx - s * 0.55, cy - s * 0.55, s * 1.1, s * 1.1)
    p.drawRect(rect)
    # 顶部缺角
    p.drawLine(
        QPointF(cx + s * 0.25, cy - s * 0.55),
        QPointF(cx + s * 0.55, cy - s * 0.25),
    )


__all__ = ["draw_edit_button", "ButtonKind"]
