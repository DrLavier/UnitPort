"""unitport_sdk.canvas.port_item — Training Node 端口图元.

DEMO 对应（``bin/pages/canvas/graph_scene.py``）：
    TrainingNodePort (3429-3549) → Node_TrainingNodePort

将 ``draw_dot_port`` paint helper 包装成 ``QGraphicsObject`` —— 端口可拖拽、
可命中、可发信号。``application/ui/canvas/items.py`` 的 ``PortItem`` 在
Phase 2.6 切换时会用本类替换。

约束：
- 不嵌套 ``QGraphicsProxyWidget``（RELEASE 硬规）
- paint() 完全走 SDK ``draw_dot_port``，不在本文件硬编码视觉
- 命中半径 ``PORT_HIT_R`` 走画布常量；hover / snap 状态机外部驱动
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsObject

from unitport_sdk.sys import Config

from .port_dot import PortStatus, draw_dot_port


# =============================================================================
# 几何常量
# =============================================================================

PORT_R = 6.0          # 视觉半径
PORT_HIT_R = 11.0     # 命中半径（>= 视觉以保证可点）


# =============================================================================
# Node_TrainingNodePort
# =============================================================================

class Node_TrainingNodePort(QGraphicsObject):
    """画布端口图元 / Canvas port glyph.

    属性：
        port_id: 节点内端口 id（caller-defined）
        direction: ``"in"`` / ``"out"``
        port_type: 端口类型字符串（用于 palette 取色）
        color: 当前端口色 QColor

    信号：
        hoverChanged(bool)         —— 鼠标进入/离开
        clicked(QPointF)           —— LeftButton 单击（场景坐标）
    """

    hoverChanged = pyqtSignal(bool)
    clicked = pyqtSignal(QPointF)

    def __init__(
        self,
        port_id: str,
        *,
        direction: str = "in",
        port_type: str = "any",
        color: Optional[QColor] = None,
        radius: float = PORT_R,
        hit_radius: float = PORT_HIT_R,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self._port_id = port_id
        self._direction = direction
        self._port_type = port_type
        self._color = color if color is not None else QColor(Config.get_color("canvas_port_default"))
        self._radius = float(radius)
        self._hit_radius = float(hit_radius)
        self._hover = False
        self._snap_hover: Optional[bool] = None  # None / True (valid) / False (invalid)
        self._disabled = False

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(5.0)

    # ----- public properties -----------------------------------------
    @property
    def port_id(self) -> str:
        return self._port_id

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def port_type(self) -> str:
        return self._port_type

    def setColor(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    def setSnapHover(self, valid: Optional[bool]) -> None:
        if valid != self._snap_hover:
            self._snap_hover = valid
            self.update()

    def setDisabled(self, disabled: bool) -> None:
        if disabled != self._disabled:
            self._disabled = bool(disabled)
            self.update()

    # ----- QGraphicsObject overrides ---------------------------------
    def boundingRect(self) -> QRectF:
        r = self._hit_radius
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        status: PortStatus = "normal"
        if self._disabled:
            status = "disabled"
        elif self._snap_hover is True:
            status = "snap_valid"
        elif self._snap_hover is False:
            status = "snap_invalid"
        elif self._hover:
            status = "hover"
        draw_dot_port(
            painter,
            QPointF(0.0, 0.0),
            self._color,
            radius=self._radius,
            status=status,
        )

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()
        self.hoverChanged.emit(True)
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()
        self.hoverChanged.emit(False)
        super().hoverLeaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(e.scenePos())
            e.accept()
            return
        super().mousePressEvent(e)


__all__ = ["Node_TrainingNodePort", "PORT_R", "PORT_HIT_R"]
