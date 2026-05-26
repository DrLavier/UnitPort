# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasMiniMap — 画布右下角浮层小地图（DEMO 复刻 / verbatim port）.

复刻自 ``DEMO/bin/pages/canvas/graph_view.py:17-195``。尺寸、布局、绘制
颜色、交互行为完全等价；仅做必需的接口翻译：
    PySide6      → PyQt6
    src.system.core.logger → unitport_sdk

视觉签名（panel 边框 / 节点 silhouette / 连线 / viewport frame 颜色）原样
保留——这是该浮层的设计意图，不属于业务主题 slot。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsView, QWidget


class CanvasMiniMap(QWidget):
    """Bottom-right miniature overview of the full canvas with a camera-view frame."""

    def __init__(self, graph_view: QGraphicsView):
        super().__init__(graph_view.viewport())
        self._graph_view = graph_view
        self.setObjectName("canvasMiniMap")
        self.setFixedSize(220, 160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def _full_scene_rect(self) -> QRectF:
        """Bounding rect of all scene items with padding — the minimap world space."""
        _MIN_W, _MIN_H = 8000.0, 6000.0
        scene = self._graph_view.scene()
        if scene is None:
            return QRectF(-_MIN_W / 2, -_MIN_H / 2, _MIN_W, _MIN_H)
        items_rect = scene.itemsBoundingRect()
        if items_rect.isNull() or items_rect.isEmpty():
            return QRectF(-_MIN_W / 2, -_MIN_H / 2, _MIN_W, _MIN_H)
        pad_x = max(items_rect.width() * 0.15, 400)
        pad_y = max(items_rect.height() * 0.15, 400)
        padded = items_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        if padded.width() < _MIN_W:
            extra = (_MIN_W - padded.width()) / 2
            padded = padded.adjusted(-extra, 0, extra, 0)
        if padded.height() < _MIN_H:
            extra = (_MIN_H - padded.height()) / 2
            padded = padded.adjusted(0, -extra, 0, extra)
        return padded

    def _viewport_scene_rect(self) -> QRectF:
        view = self._graph_view
        vp = view.viewport().rect()
        if vp.isNull():
            return QRectF()
        return QRectF(view.mapToScene(vp.topLeft()), view.mapToScene(vp.bottomRight()))

    def _content_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)

    def _scene_to_map(self, point: QPointF, full_rect: QRectF, content_rect: QRectF) -> QPointF:
        if full_rect.width() <= 0 or full_rect.height() <= 0:
            return QPointF(content_rect.center())
        scale = min(content_rect.width() / full_rect.width(),
                    content_rect.height() / full_rect.height())
        ox = content_rect.left() + (content_rect.width() - full_rect.width() * scale) / 2.0
        oy = content_rect.top() + (content_rect.height() - full_rect.height() * scale) / 2.0
        return QPointF(ox + (point.x() - full_rect.left()) * scale,
                       oy + (point.y() - full_rect.top()) * scale)

    def _map_to_scene(self, point: QPointF, full_rect: QRectF, content_rect: QRectF) -> QPointF:
        if full_rect.width() <= 0 or full_rect.height() <= 0:
            return full_rect.center()
        scale = min(content_rect.width() / full_rect.width(),
                    content_rect.height() / full_rect.height())
        ox = content_rect.left() + (content_rect.width() - full_rect.width() * scale) / 2.0
        oy = content_rect.top() + (content_rect.height() - full_rect.height() * scale) / 2.0
        x = full_rect.left() + (point.x() - ox) / scale
        y = full_rect.top() + (point.y() - oy) / scale
        return QPointF(x, y)

    def _draw_connection_silhouettes(self, painter: QPainter, scene, full_rect: QRectF, content_rect: QRectF) -> None:
        pen = QPen(QColor(129, 140, 153, 170), 1.25)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for item in scene.items():
            if item.data(0) != "connection":
                continue
            # ConnectionItem 实际属性名为 src_port / dst_port（out → in 方向）
            src_port = getattr(item, "src_port", None) or getattr(item, "out_port", None)
            dst_port = getattr(item, "dst_port", None) or getattr(item, "in_port", None)
            if src_port is None or dst_port is None:
                continue
            try:
                start = src_port.mapToScene(src_port.boundingRect().center())
                end = dst_port.mapToScene(dst_port.boundingRect().center())
            except Exception:
                continue

            dx = end.x() - start.x()
            c1 = QPointF(start.x() + dx * 0.5, start.y())
            c2 = QPointF(end.x() - dx * 0.5, end.y())
            path = QPainterPath()
            path.moveTo(self._scene_to_map(start, full_rect, content_rect))
            path.cubicTo(
                self._scene_to_map(c1, full_rect, content_rect),
                self._scene_to_map(c2, full_rect, content_rect),
                self._scene_to_map(end, full_rect, content_rect),
            )
            painter.drawPath(path)

    def _draw_node_silhouettes(self, painter: QPainter, scene, full_rect: QRectF, content_rect: QRectF) -> None:
        painter.setPen(QPen(QColor(230, 238, 248, 150), 1.0))
        painter.setBrush(QBrush(QColor(226, 232, 240, 110)))

        for item in reversed(scene.items()):
            # 节点的 ROLE_KIND（data role 0）== "node"；DEMO 旧协议曾用 role 10，
            # 这里同时兼容以避免回归。
            if item.data(0) != "node" and item.data(10) != "node":
                continue
            scene_rect = item.sceneBoundingRect()
            rect = QRectF(
                self._scene_to_map(scene_rect.topLeft(), full_rect, content_rect),
                self._scene_to_map(scene_rect.bottomRight(), full_rect, content_rect),
            ).normalized()
            if rect.isNull() or rect.width() < 1.0 or rect.height() < 1.0:
                continue
            painter.drawRoundedRect(rect, 2, 2)

    def _draw_viewport_frame(self, painter: QPainter, full_rect: QRectF, content_rect: QRectF) -> None:
        """White frame tracking the current camera viewport."""
        vp_rect = self._viewport_scene_rect()
        if not vp_rect.isValid():
            return
        tl = self._scene_to_map(vp_rect.topLeft(), full_rect, content_rect)
        br = self._scene_to_map(vp_rect.bottomRight(), full_rect, content_rect)
        frame = QRectF(tl, br).normalized()
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 18)))
        painter.drawRect(frame)

    def _jump_to_map_position(self, pos: QPointF) -> None:
        full_rect = self._full_scene_rect()
        content_rect = self._content_rect()
        scene_target = self._map_to_scene(pos, full_rect, content_rect)
        self._graph_view.centerOn(scene_target)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._jump_to_map_position(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._jump_to_map_position(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        scene = self._graph_view.scene()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        panel_rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(120, 120, 120, 180), 1.0))
        painter.setBrush(QBrush(QColor(18, 22, 28, 220)))
        painter.drawRoundedRect(panel_rect, 10, 10)

        if scene is None:
            return

        full_rect = self._full_scene_rect()
        content_rect = self._content_rect()

        painter.save()
        painter.setClipRect(QRectF(self.rect()).adjusted(4, 4, -4, -4))
        self._draw_connection_silhouettes(painter, scene, full_rect, content_rect)
        self._draw_node_silhouettes(painter, scene, full_rect, content_rect)
        self._draw_viewport_frame(painter, full_rect, content_rect)
        painter.restore()


__all__ = ["CanvasMiniMap"]
