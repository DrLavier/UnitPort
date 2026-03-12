#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph editor view.
Supports drag-drop node creation, zooming, panning, and box selection.
"""

import json
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from bin.core.logger import log_info, log_debug, log_error
from bin.components.graph_scene import GraphScene


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

        # Panning state
        self._is_panning = False
        self._pan_start_pos = None
        self._pan_button = Qt.NoButton

        # Connection state used to disable rubber-band while wiring
        self._is_connecting = False
        self.setCursor(Qt.ArrowCursor)
        self._did_initial_center = False

        # Run once after widgets/layout are ready so startup camera sees initial nodes.
        QTimer.singleShot(0, self._center_on_initial_origin)

        log_debug("GraphView initialized")

    def showEvent(self, event):
        super().showEvent(event)
        self._center_on_initial_origin()

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

        # Middle/right button panning
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._pan_start_pos = event.pos()
            self._pan_button = event.button()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move."""
        if self._is_panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

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
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Reset cursor/hover state when mouse leaves canvas."""
        if not self._is_panning:
            self.setCursor(Qt.ArrowCursor)
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
            event.acceptProposedAction()

        except Exception as e:
            log_error(f"Drop failed: {e}")
            event.ignore()

    def reset_view(self):
        """Reset zoom/pan."""
        self.resetTransform()
        self._zoom_factor = 1.0
        log_info("View reset")

    def fit_to_contents(self):
        """Fit all items in viewport."""
        self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        log_info("View fitted to contents")
