# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.port_dot — 端口圆点 paint helper.

替换 DEMO ``TrainingNodePort.paint``（graph_scene.py:3429）的视觉契约：
六态可视——normal / hover / connected / valid (snap target) / invalid /
disabled——全部以纯 paint 表达，调用方负责状态判断。

坐标系：center 是 *item-local* 坐标，调用方在 ``QGraphicsItem.paint`` 内传当前
item 的 (0, 0) 即可（PortItem 的 boundingRect 已以中心对齐）。

LOD 行为约定（推荐）：
    tier == 0 (Overview)  → 完全跳过（调用方自己判断不调本函数）
    tier == 1 (Minimap)   → radius *= 0.6，hover 不加亮
    tier >= 2 (Working+)  → 完整视觉
"""

from __future__ import annotations

from typing import Literal, Optional

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen

from unitport_sdk.sys import Config


PortStatus = Literal["normal", "hover", "snap_valid", "snap_invalid", "disabled"]


def draw_dot_port(
    painter: QPainter,
    center: QPointF,
    color: QColor,
    *,
    radius: float = 6.0,
    status: PortStatus = "normal",
    border_color: Optional[QColor] = None,
) -> None:
    """画一个端口圆点 / Paint a port dot.

    所有视觉颜色 **全部** 取自 ``system.ini[Theme]``（``canvas_node_border``、
    ``btn_1``、``main_t2``、``canvas_safety_error``）。本模块禁止硬编码 hex。

    Args:
        painter: 当前 ``QPainter``（调用方负责 begin/end）
        center: 圆心 item-local 坐标
        color: 端口主色（一般来自 ``get_port_color(spec.type)``）
        radius: 视觉半径，默认 6px。LOD 远缩放时调用方自行减半
        status: 端口状态，决定填充 / 边框 / 描边宽度变化
        border_color: 边框颜色（None → ``canvas_node_border`` slot；若调用方
            有定制主题应显式传入）
    """
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    if border_color is None:
        border_color = QColor(Config.get_color("border_1"))

    if status == "disabled":
        # 灰显，无填充
        painter.setBrush(QBrush(QColor(Config.get_color("btn_1"))))
        painter.setPen(QPen(border_color, 1.0))
        painter.drawEllipse(center, radius, radius)
        painter.restore()
        return

    if status == "snap_valid":
        # 合法 snap 目标：填充 + 1.5px 描边 + 外圈光晕
        glow = QColor(color)
        glow.setAlpha(110)
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius * 1.7, radius * 1.7)
        painter.setBrush(QBrush(color.lighter(130)))
        painter.setPen(QPen(QColor(Config.get_color("main_t2")), 1.5))
        painter.drawEllipse(center, radius, radius)
        painter.restore()
        return

    if status == "snap_invalid":
        # 不兼容目标：填充原色 + 红描边
        painter.setBrush(QBrush(color.darker(140)))
        painter.setPen(QPen(QColor(Config.get_color("canvas_safety_error")), 1.6))
        painter.drawEllipse(center, radius, radius)
        painter.restore()
        return

    if status == "hover":
        # hover：填充加亮 +20% + 1.6px 描边
        painter.setBrush(QBrush(color.lighter(125)))
        painter.setPen(QPen(border_color, 1.6))
        painter.drawEllipse(center, radius + 0.5, radius + 0.5)
        painter.restore()
        return

    # normal
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(border_color, 1.2))
    painter.drawEllipse(center, radius, radius)
    painter.restore()


__all__ = ["draw_dot_port", "PortStatus"]
