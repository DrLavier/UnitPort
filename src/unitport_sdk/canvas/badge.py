"""unitport_sdk.canvas.badge — 极性徽章 paint helper.

替换 DEMO ``_ModulePolarityBadge``（training_node_items.py:888-930）的视觉
契约：18×18 状态徽章——奖励（圆）/ 惩罚（方）/ 双向（菱形）。

调用方负责：
- 几何（rect 是 item-local QRectF）
- 当前 kind 状态
- 单击循环切换（SDK 不处理鼠标）

LOD 建议：T0/T1 跳过；T2+ 完整视觉。
"""

from __future__ import annotations

from typing import Literal, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

from unitport_sdk.sys import Config


PolarityKind = Literal["reward", "penalty", "bidirectional", "neutral"]


# 极性 → system.ini[Theme] slot 映射（禁止本地常量）
_KIND_TO_SLOT: dict = {
    "reward":         "safe_zone",       # 绿
    "penalty":        "danger_zone",     # 红
    "bidirectional":  "highlight",      # 黄
    "neutral":        "sub_t1",          # 灰
}


def draw_polarity_badge(
    painter: QPainter,
    rect: QRectF,
    *,
    kind: PolarityKind = "reward",
    color: Optional[QColor] = None,
    border_color: Optional[QColor] = None,
) -> None:
    """画极性徽章 / Paint a polarity badge.

    Args:
        painter: ``QPainter``
        rect: 徽章矩形（item-local，建议 18×18）
        kind: ``"reward"`` 圆 ``/ "penalty"`` 方 ``/ "bidirectional"`` 菱形 /
            ``"neutral"`` 灰圆
        color: 填充色（None → 按 kind 取 ``system.ini[Theme]`` 槽位）
        border_color: 边框色（None → 比 fill 暗 25%）
    """
    if color is None:
        slot = _KIND_TO_SLOT.get(kind, "sub_t1")
        color = QColor(Config.get_color(slot))
    if border_color is None:
        border_color = color.darker(140)

    cx = rect.left() + rect.width() * 0.5
    cy = rect.top() + rect.height() * 0.5
    s = min(rect.width(), rect.height()) * 0.5 - 1.0  # 1px 内缩留边

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(border_color, 1.0))

    if kind == "penalty":
        # 方
        painter.drawRect(QRectF(cx - s, cy - s, s * 2, s * 2))
    elif kind == "bidirectional":
        # 菱形
        diamond = QPolygonF([
            QPointF(cx, cy - s),
            QPointF(cx + s, cy),
            QPointF(cx, cy + s),
            QPointF(cx - s, cy),
        ])
        painter.drawPolygon(diamond)
    else:
        # reward / neutral → 圆
        painter.drawEllipse(QPointF(cx, cy), s, s)

    painter.restore()


__all__ = ["draw_polarity_badge", "PolarityKind"]
