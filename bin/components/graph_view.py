#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph editor view.
Supports drag-drop node creation, zooming, panning, and box selection.
"""

import json
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QWheelEvent, QColor, QPen, QBrush, QPainterPath
from PySide6.QtWidgets import QGraphicsView, QWidget

from bin.core.logger import log_info, log_debug, log_error
from bin.components.graph_scene import GraphScene


class CanvasMiniMap(QWidget):
    """Bottom-right miniature overview of the full canvas with a camera-view frame."""

    def __init__(self, graph_view: "GraphView"):
        super().__init__(graph_view.viewport())
        self._graph_view = graph_view
        self.setObjectName("canvasMiniMap")
        self.setFixedSize(220, 160)
        self.setCursor(Qt.PointingHandCursor)
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
        # Enforce a minimum canvas size so the minimap never over-zooms
        if padded.width() < _MIN_W:
            extra = (_MIN_W - padded.width()) / 2
            padded = padded.adjusted(-extra, 0, extra, 0)
        if padded.height() < _MIN_H:
            extra = (_MIN_H - padded.height()) / 2
            padded = padded.adjusted(0, -extra, 0, extra)
        return padded

    def _viewport_scene_rect(self) -> QRectF:
        """Current viewport expressed in scene coordinates."""
        view = self._graph_view
        vp = view.viewport().rect()
        if vp.isNull():
            return QRectF()
        return QRectF(view.mapToScene(vp.topLeft()), view.mapToScene(vp.bottomRight()))

    def _content_rect(self) -> QRectF:
        """Drawable area inside the panel (inner padding)."""
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
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        for item in scene.items():
            if item.data(0) != "connection":
                continue
            out_port = getattr(item, "out_port", None)
            in_port = getattr(item, "in_port", None)
            if out_port is None or in_port is None:
                continue
            try:
                start = out_port.mapToScene(out_port.boundingRect().center())
                end = in_port.mapToScene(in_port.boundingRect().center())
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
            if item.data(10) != "node":
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
        if event.button() == Qt.LeftButton:
            self._jump_to_map_position(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self._jump_to_map_position(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        scene = self._graph_view.scene()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Single panel box
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


class GraphView(QGraphicsView):
    """Graph editor view."""

    def __init__(self, scene: GraphScene, parent=None):
        super().__init__(scene, parent)

        # View config
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Performance hints
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)

        # Default to rubber-band selection
        self.setDragMode(QGraphicsView.RubberBandDrag)

        # Accept drops from module palette
        self.setAcceptDrops(True)

        # Zoom constraints
        self._zoom_factor = 1.0
        self._zoom_min = 0.3
        self._zoom_max = 3.0
        self._drag_zoom_sensitivity = 1.004

        # Interaction state
        self._is_panning = False
        self._pan_start_pos = None
        self._pan_button = Qt.NoButton
        self._is_middle_zooming = False
        self._zoom_drag_pos = None

        # Connection state used to disable rubber-band while wiring
        self._is_connecting = False
        self._set_canvas_cursor(Qt.ArrowCursor)
        self._did_initial_center = False

        self._minimap = CanvasMiniMap(self)
        self._minimap.raise_()
        self._position_minimap()

        self.horizontalScrollBar().valueChanged.connect(self._refresh_minimap)
        self.verticalScrollBar().valueChanged.connect(self._refresh_minimap)
        if scene is not None:
            scene.changed.connect(lambda *_: self._refresh_minimap())

        # Run once after widgets/layout are ready so startup camera sees initial nodes.
        QTimer.singleShot(0, self._center_on_initial_origin)

        log_debug("GraphView initialized")

    def _set_canvas_cursor(self, cursor_shape) -> None:
        """Keep the view and its viewport cursor in sync."""
        self.setCursor(cursor_shape)
        self.viewport().setCursor(cursor_shape)

    def _position_minimap(self) -> None:
        margin = 12
        size = self._minimap.size()
        self._minimap.move(
            max(0, self.viewport().width() - size.width() - margin),
            max(0, self.viewport().height() - size.height() - margin),
        )
        self._minimap.raise_()

    def _refresh_minimap(self) -> None:
        self._position_minimap()
        self._minimap.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._center_on_initial_origin()
        self._refresh_minimap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_minimap()

    def _center_on_initial_origin(self):
        if self._did_initial_center:
            return
        self.recenter_to_origin()
        self._did_initial_center = True

    def recenter_to_origin(self):
        """Recenter the camera to scene initial origin without changing zoom."""
        scene = self.scene()
        if not scene:
            return
        origin = getattr(scene, "_initial_origin", None)
        if origin is not None:
            self.centerOn(origin)
        else:
            self.centerOn(0, 0)
        self._refresh_minimap()

    def wheelEvent(self, event: QWheelEvent):
        """Mouse wheel zoom."""
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.85

        new_zoom = self._zoom_factor * factor
        if new_zoom < self._zoom_min or new_zoom > self._zoom_max:
            return

        self._zoom_factor = new_zoom
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self._refresh_minimap()

    def keyPressEvent(self, event):
        """Keyboard shortcuts for view controls."""
        if event.key() == Qt.Key_Home:
            self.recenter_to_origin()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Mouse press."""
        scene = self.scene()
        if isinstance(scene, GraphScene):
            scene_pos = self.mapToScene(event.position().toPoint())
            item = scene.itemAt(scene_pos, self.transform())
            near_port = item if scene._is_port(item) else scene._find_port_near(
                scene_pos,
                radius=scene.get_port_interaction_radius(),
            )

            # Starting a connection from a port disables rubber-band selection.
            if scene._is_port(near_port):
                self._is_connecting = True
                self.setDragMode(QGraphicsView.NoDrag)
                log_debug("Connection start detected, rubber-band selection disabled")

        # Right button pans the canvas.
        if event.button() == Qt.RightButton:
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self._pan_button = event.button()
            self._set_canvas_cursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Middle button drag zooms: +X/+Y zoom in, reverse zooms out.
        if event.button() == Qt.MiddleButton:
            self._is_middle_zooming = True
            self._zoom_drag_pos = event.pos()
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self._set_canvas_cursor(Qt.SizeVerCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move."""
        if self._is_panning and not (event.buttons() & self._pan_button):
            self._is_panning = False
            self._pan_button = Qt.NoButton
            self._pan_start_pos = None
            self._set_canvas_cursor(Qt.ArrowCursor)

        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._refresh_minimap()

            event.accept()
            return

        if self._is_middle_zooming:
            delta = event.pos() - self._zoom_drag_pos
            self._zoom_drag_pos = event.pos()
            zoom_delta = delta.x() + delta.y()

            if zoom_delta:
                factor = self._drag_zoom_sensitivity ** zoom_delta
                target_zoom = max(self._zoom_min, min(self._zoom_max, self._zoom_factor * factor))
                applied_factor = target_zoom / self._zoom_factor if self._zoom_factor else 1.0
                if applied_factor != 1.0:
                    self._zoom_factor = target_zoom
                    self.scale(applied_factor, applied_factor)
                    self._refresh_minimap()

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Mouse release."""
        if self._is_connecting:
            self._is_connecting = False
            self.setDragMode(QGraphicsView.RubberBandDrag)
            log_debug("Connection finished, rubber-band selection restored")

        if self._is_panning and event.button() == self._pan_button:
            self._is_panning = False
            self._pan_button = Qt.NoButton
            self._pan_start_pos = None
            self._set_canvas_cursor(Qt.ArrowCursor)
            self._refresh_minimap()
            event.accept()
            return

        if self._is_middle_zooming and event.button() == Qt.MiddleButton:
            self._is_middle_zooming = False
            self._zoom_drag_pos = None
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self._set_canvas_cursor(Qt.ArrowCursor)
            self._refresh_minimap()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Reset cursor/hover state when mouse leaves canvas."""
        if not self._is_panning and not self._is_middle_zooming:
            self._set_canvas_cursor(Qt.ArrowCursor)
        scene = self.scene()
        if isinstance(scene, GraphScene):
            scene._clear_port_hover()
        super().leaveEvent(event)

    def dragEnterEvent(self, event):
        """Drag enter."""
        if event.mimeData().hasFormat("application/x-module-card"):
            event.acceptProposedAction()
            log_debug("Drag entered graph view")
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Drag move."""
        if event.mimeData().hasFormat("application/x-module-card"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Drop event."""
        if not event.mimeData().hasFormat("application/x-module-card"):
            event.ignore()
            return

        try:
            data = event.mimeData().data("application/x-module-card")
            payload = json.loads(bytes(data).decode("utf-8"))

            title = payload.get("title", "Unknown Module")
            grad = tuple(payload.get("grad", ["#45a049", "#4CAF50"]))
            features = payload.get("features", [])
            preset = payload.get("preset")

            scene_pos = self.mapToScene(event.position().toPoint())

            scene = self.scene()
            if isinstance(scene, GraphScene):
                # Prevent duplicate Start/End nodes
                if title in ("Start", "End"):
                    for item in scene.items():
                        if item.data(10) == "node" and item.data(11) == title:
                            log_info(f"'{title}' node already exists, cannot create duplicate")
                            event.acceptProposedAction()
                            return

                node_item = scene.create_node(title, scene_pos, features, grad)
                if preset and hasattr(node_item, "_condition_set") and node_item._condition_set:
                    node_item._condition_set.set_logic_text(preset)

            log_info(f"Node created: {title} at ({scene_pos.x():.0f}, {scene_pos.y():.0f})")
            self._refresh_minimap()
            event.acceptProposedAction()

        except Exception as e:
            log_error(f"Drop failed: {e}")
            event.ignore()

    def reset_view(self):
        """Reset zoom/pan."""
        self.resetTransform()
        self._zoom_factor = 1.0
        self._refresh_minimap()
        log_info("View reset")

    def fit_to_contents(self):
        """Fit all items in viewport."""
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        transform = self.transform()
        self._zoom_factor = transform.m11() if transform is not None else 1.0
        self._refresh_minimap()
        log_info("View fitted to contents")
