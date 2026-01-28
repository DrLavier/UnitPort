#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Scene
Contains nodes, connections, grid and other elements
"""

import json
from typing import Optional, List, Dict, Any
from shiboken6 import isValid

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QLinearGradient, QGradient, QPainterPath
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPathItem, QGraphicsProxyWidget, QComboBox, QLineEdit, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QLabel
)

from bin.core.logger import log_info, log_error, log_debug, log_warning, log_success


class ConnectionItem(QGraphicsPathItem):
    """Connection Line Item - Supports auto-update and endpoint editing"""

    def __init__(self, out_port, in_port, parent=None):
        super().__init__(parent)

        self.out_port = out_port
        self.in_port = in_port

        # Set style
        self.setPen(QPen(QColor("#60a5fa"), 2.5))
        self.setZValue(-1)
        self.setData(0, "connection")

        # Selectable and clickable
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        # Endpoint markers
        self._start_marker = None
        self._end_marker = None
        self._create_markers()

        # Update path
        self.update_path()

    def _create_markers(self):
        """Create endpoint markers (for reconnection)"""
        # Start marker
        self._start_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._start_marker.setBrush(QBrush(QColor("#60a5fa")))
        self._start_marker.setPen(QPen(QColor("#ffffff"), 1))
        self._start_marker.setZValue(10)
        self._start_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._start_marker.setData(0, "connection_marker")
        self._start_marker.setData(1, "start")
        self._start_marker.setData(2, self)  # Associated connection
        self._start_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._start_marker.setVisible(False)

        # End marker
        self._end_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._end_marker.setBrush(QBrush(QColor("#60a5fa")))
        self._end_marker.setPen(QPen(QColor("#ffffff"), 1))
        self._end_marker.setZValue(10)
        self._end_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._end_marker.setData(0, "connection_marker")
        self._end_marker.setData(1, "end")
        self._end_marker.setData(2, self)  # Associated connection
        self._end_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._end_marker.setVisible(False)

    def update_path(self):
        """Update connection path"""
        if not self.out_port or not self.in_port:
            return

        # Check if ports are valid
        if not isValid(self.out_port) or not isValid(self.in_port):
            return

        # Get port center positions
        try:
            start = self.out_port.mapToScene(self.out_port.boundingRect().center())
            end = self.in_port.mapToScene(self.in_port.boundingRect().center())
        except RuntimeError:
            return

        # Create bezier curve path
        path = QPainterPath()
        path.moveTo(start)

        dx = end.x() - start.x()
        path.cubicTo(
            start.x() + dx * 0.5, start.y(),
            end.x() - dx * 0.5, end.y(),
            end.x(), end.y()
        )

        self.setPath(path)

        # Update endpoint marker positions
        if self._start_marker:
            self._start_marker.setPos(start)
        if self._end_marker:
            self._end_marker.setPos(end)

    def hoverEnterEvent(self, event):
        """Mouse hover - show endpoint markers"""
        self.setPen(QPen(QColor("#3b82f6"), 3.5))
        if self._start_marker:
            self._start_marker.setVisible(True)
        if self._end_marker:
            self._end_marker.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Mouse leave - hide endpoint markers"""
        if not self.isSelected():
            self.setPen(QPen(QColor("#60a5fa"), 2.5))
        if self._start_marker:
            self._start_marker.setVisible(False)
        if self._end_marker:
            self._end_marker.setVisible(False)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """Item change event"""
        if change == QGraphicsItem.ItemSelectedHasChanged:
            if value:  # Selected
                self.setPen(QPen(QColor("#3b82f6"), 3.5))
                if self._start_marker:
                    self._start_marker.setVisible(True)
                if self._end_marker:
                    self._end_marker.setVisible(True)
            else:  # Not selected
                self.setPen(QPen(QColor("#60a5fa"), 2.5))
                if self._start_marker:
                    self._start_marker.setVisible(False)
                if self._end_marker:
                    self._end_marker.setVisible(False)

        return super().itemChange(change, value)


class GraphScene(QGraphicsScene):
    """Graph Editor Scene"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene configuration
        self.setSceneRect(-2500, -2500, 5000, 5000)
        self.grid_small = 20
        self.grid_big = self.grid_small * 5

        # Color configuration
        self.color_bg = QColor(30, 31, 34)
        self.color_grid_small = QColor(50, 51, 55)
        self.color_grid_big = QColor(60, 62, 67)

        # Nodes and connections
        self._node_seq = 0
        self._temp_connection = None
        self._temp_start_port = None

        # Reconnection state
        self._reconnecting = False
        self._reconnect_connection = None
        self._reconnect_end = None  # "start" or "end"

        # Action mapping
        self._action_mapping = {
            "Lift Right Leg": "lift_right_leg",
            "Stand": "stand",
            "Sit": "sit",
            "Walk": "walk",
            "Stop": "stop"
        }

        # References
        self._code_editor = None
        self._simulation_thread = None
        self._robot_type = "go2"

        # Timer - for updating connections
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_all_connections)
        self._update_timer.start(16)  # 60fps

        log_debug("GraphScene initialized")

    def set_code_editor(self, editor):
        """Set code editor reference"""
        self._code_editor = editor

    def set_robot_type(self, robot_type: str):
        """Set robot type"""
        self._robot_type = robot_type
        log_info(f"Robot type set to: {robot_type}")

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw grid background"""
        painter.fillRect(rect, self.color_bg)

        # Draw small grid
        left = int(rect.left()) - (int(rect.left()) % self.grid_small)
        top = int(rect.top()) - (int(rect.top()) % self.grid_small)

        lines = []

        # Vertical lines
        x = left
        while x < rect.right():
            lines.append((x, rect.top(), x, rect.bottom()))
            x += self.grid_small

        # Horizontal lines
        y = top
        while y < rect.bottom():
            lines.append((rect.left(), y, rect.right(), y))
            y += self.grid_small

        # Draw small grid
        painter.setPen(QPen(self.color_grid_small, 1))
        for x1, y1, x2, y2 in lines:
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw big grid
        left = int(rect.left()) - (int(rect.left()) % self.grid_big)
        top = int(rect.top()) - (int(rect.top()) % self.grid_big)

        big_lines = []
        x = left
        while x < rect.right():
            big_lines.append((x, rect.top(), x, rect.bottom()))
            x += self.grid_big

        y = top
        while y < rect.bottom():
            big_lines.append((rect.left(), y, rect.right(), y))
            y += self.grid_big

        painter.setPen(QPen(self.color_grid_big, 2))
        for x1, y1, x2, y2 in big_lines:
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def mousePressEvent(self, event):
        """Mouse press event"""
        pos = event.scenePos()
        item = self.itemAt(pos, self.views()[0].transform() if self.views() else None)

        # Check if clicked on connection marker (endpoint)
        if item and item.data(0) == "connection_marker":
            connection = item.data(2)
            end_type = item.data(1)
            self._start_reconnection(connection, end_type, pos)
            return

        # Check if clicked on port
        if self._is_port(item):
            self._start_connection(item, pos)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move event"""
        if self._temp_connection:
            # Update temporary connection
            self._update_temp_connection(event.scenePos())
            return

        if self._reconnecting and self._temp_connection:
            # Update reconnection temporary line
            self._update_temp_connection(event.scenePos())
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Mouse release event"""
        if self._reconnecting:
            pos = event.scenePos()
            target_port = self._find_port_near(pos)

            if target_port:
                self._finish_reconnection(target_port)
            else:
                self._cancel_reconnection()
            return

        if self._temp_connection:
            pos = event.scenePos()
            target_port = self._find_port_near(pos)

            if target_port and target_port != self._temp_start_port:
                self._finish_connection(target_port)
            else:
                self._cancel_connection()

            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Keyboard press event"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # Delete selected items
            selected_items = self.selectedItems()
            if selected_items:
                self._delete_items(selected_items)
                event.accept()
                return

        super().keyPressEvent(event)

    def _delete_items(self, items):
        """Delete selected items"""
        deleted_nodes = []
        deleted_connections = []

        for item in items:
            # Delete node
            if item.data(10) == "node":
                node_name = item.data(11)
                node_id = item.data(12)

                # Delete all connections related to the node
                self._delete_node_connections(item)

                # Delete the node itself
                self.removeItem(item)
                deleted_nodes.append(f"{node_name} (ID: {node_id})")
                log_info(f"Node deleted: {node_name} (ID: {node_id})")

            # Delete connection
            elif item.data(0) == "connection" or isinstance(item, ConnectionItem):
                # Remove connection reference from ports
                if isinstance(item, ConnectionItem):
                    self._detach_connection(item)
                self.removeItem(item)
                deleted_connections.append("Connection")

        if deleted_nodes:
            log_success(f"{len(deleted_nodes)} node(s) deleted")
        if deleted_connections:
            log_info(f"{len(deleted_connections)} connection(s) deleted")

        # Regenerate code
        self.regenerate_code()

    def _delete_node_connections(self, node_item):
        """Delete all connections related to a node"""
        # Find all ports
        ports = []
        for child in node_item.childItems():
            if self._is_port(child):
                ports.append(child)

        # Delete connections for each port
        for port in ports:
            connections = port.data(2) or []
            for conn in list(connections):  # Use list() to create copy to avoid modification during iteration
                if conn and isValid(conn) and conn.scene() is not None:
                    self.removeItem(conn)

    def _detach_connection(self, connection):
        """Remove connection reference from ports"""
        if isinstance(connection, ConnectionItem):
            # Remove from output port
            if connection.out_port and isValid(connection.out_port):
                conns = connection.out_port.data(2) or []
                if connection in conns:
                    conns.remove(connection)
                    connection.out_port.setData(2, conns)

            # Remove from input port
            if connection.in_port and isValid(connection.in_port):
                conns = connection.in_port.data(2) or []
                if connection in conns:
                    conns.remove(connection)
                    connection.in_port.setData(2, conns)
                self._clear_input_for_port(connection.in_port)

    def _start_reconnection(self, connection, end_type, pos):
        """Start reconnection"""
        if not isinstance(connection, ConnectionItem):
            return

        self._reconnecting = True
        self._reconnect_connection = connection
        self._reconnect_end = end_type

        # Determine start port
        if end_type == "start":
            self._temp_start_port = connection.out_port
            # Temporarily remove output end
            connection.out_port = None
        else:
            self._temp_start_port = connection.in_port
            # Temporarily remove input end
            connection.in_port = None

        # Create temporary connection
        path = QPainterPath()
        center = self._port_center(self._temp_start_port)
        path.moveTo(center)
        path.lineTo(pos)

        self._temp_connection = QGraphicsPathItem(path)
        self._temp_connection.setPen(QPen(QColor("#f59e0b"), 3))
        self.addItem(self._temp_connection)

        log_debug(f"Starting reconnection: {end_type} end")

    def _finish_reconnection(self, target_port):
        """Finish reconnection"""
        if not self._reconnect_connection or not target_port:
            self._cancel_reconnection()
            return

        # Check connection direction
        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)

        if start_io == target_io:
            log_warning("Cannot connect ports of the same type")
            self._cancel_reconnection()
            return

        # Update connection ports
        if self._reconnect_end == "start":
            # Reconnect start
            if start_io == "out":
                self._reconnect_connection.out_port = target_port
            else:
                self._reconnect_connection.in_port = target_port
        else:
            # Reconnect end
            if start_io == "out":
                self._reconnect_connection.in_port = target_port
            else:
                self._reconnect_connection.out_port = target_port

        # Attach to new port
        self._attach_connection_safe(target_port, self._reconnect_connection)

        # Update path
        self._reconnect_connection.update_path()
        if self._reconnect_connection.in_port and self._reconnect_connection.out_port:
            self._apply_connection_to_input(
                self._reconnect_connection.in_port,
                self._reconnect_connection.out_port
            )

        # Clean up temporary state
        self._cancel_reconnection()

        log_info("Reconnection successful")
        self.regenerate_code()

    def _cancel_reconnection(self):
        """Cancel reconnection"""
        if self._reconnect_connection:
            # Restore original connection
            if self._reconnect_end == "start":
                # Restore removed port
                pass  # Connection has been deleted or kept as is

            self._reconnect_connection = None

        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None

        self._temp_start_port = None
        self._reconnecting = False

    def _start_connection(self, port_item, pos):
        """Start creating connection"""
        self._temp_start_port = port_item

        # Create temporary connection
        path = QPainterPath()
        center = self._port_center(port_item)
        path.moveTo(center)
        path.lineTo(pos)

        self._temp_connection = QGraphicsPathItem(path)
        self._temp_connection.setPen(QPen(QColor("#60a5fa"), 3))
        self.addItem(self._temp_connection)

    def _update_temp_connection(self, pos):
        """Update temporary connection"""
        if not self._temp_connection or not self._temp_start_port:
            return

        start = self._port_center(self._temp_start_port)
        path = QPainterPath()
        path.moveTo(start)

        # Bezier curve
        dx = pos.x() - start.x()
        path.cubicTo(
            start.x() + dx * 0.5, start.y(),
            pos.x() - dx * 0.5, pos.y(),
            pos.x(), pos.y()
        )

        self._temp_connection.setPath(path)

    def _finish_connection(self, target_port):
        """Finish connection"""
        if not self._temp_start_port or not target_port:
            self._cancel_connection()
            return

        # Check connection direction
        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)

        if start_io == target_io:
            log_warning("Cannot connect ports of the same type")
            self._cancel_connection()
            return

        # Determine output and input ports
        out_port = self._temp_start_port if start_io == "out" else target_port
        in_port = target_port if start_io == "out" else self._temp_start_port

        # Create connection
        self._create_connection(out_port, in_port)
        self._cancel_connection()

        # Update code
        self.regenerate_code()

    def _cancel_connection(self):
        """Cancel connection"""
        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None
        self._temp_start_port = None

    def _create_connection(self, out_port, in_port):
        """Create connection - using ConnectionItem"""
        conn = ConnectionItem(out_port, in_port)
        self.addItem(conn)

        # Attach to ports
        self._attach_connection_safe(out_port, conn)
        self._attach_connection_safe(in_port, conn)

        log_debug(f"Connection created: {out_port.data(3)} -> {in_port.data(3)}")
        self._apply_connection_to_input(in_port, out_port)

    def _update_all_connections(self):
        """Update all connection paths"""
        for item in self.items():
            if isinstance(item, ConnectionItem):
                item.update_path()

    def create_node(self, name: str, scene_pos: QPointF,
                    features: List[str] = None, grad: tuple = None):
        """
        Create node

        Args:
            name: Node name
            scene_pos: Scene position
            features: Feature list
            grad: Gradient colors (color1, color2)
        """
        # Adjust width based on node type
        if "Logic Control" in name or "逻辑控制" in name:
            w, h = 220, 170
        elif "Condition" in name or "条件判断" in name:
            w, h = 210, 140
        else:
            w, h = 180, 110

        # Create node rectangle
        rect = QGraphicsRectItem(0, 0, w, h)

        # Gradient background
        if grad and len(grad) == 2:
            g = QLinearGradient(0, 0, 1, 1)
            g.setCoordinateMode(QGradient.ObjectBoundingMode)
            g.setColorAt(0.0, QColor(grad[0]))
            g.setColorAt(1.0, QColor(grad[1]))
            rect.setBrush(QBrush(g))
        else:
            rect.setBrush(QBrush(QColor(45, 50, 60)))

        rect.setPen(QPen(QColor(120, 130, 140), 2))
        rect.setFlag(QGraphicsItem.ItemIsMovable, True)
        rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
        rect.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)  # Important: send geometry change signals
        rect.setPos(scene_pos - QPointF(w / 2, h / 2))
        self.addItem(rect)

        # Title - use fixed width to ensure display
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        label = self.addText(str(name), f)
        label.setDefaultTextColor(QColor("#ffffff"))
        label.setParentItem(rect)
        label.setZValue(2)
        label.setPos(8, 6)

        # If title is too long, crop display
        label_width = label.boundingRect().width()
        if label_width > w - 16:
            # Adjust font size
            f.setPointSize(8)
            label.setFont(f)

        # Create port function
        port_r = 6

        def _mk_port(x, y, io, slot):
            p = QGraphicsEllipseItem(-port_r, -port_r, port_r * 2, port_r * 2, rect)
            p.setPos(x, y)
            p.setBrush(QBrush(QColor("#1f2937")))
            p.setPen(QPen(QColor("#60a5fa"), 2))
            p.setData(0, "port")
            p.setData(1, io)
            p.setData(2, [])
            p.setData(3, slot)
            p.setZValue(3)
            p.setAcceptedMouseButtons(Qt.LeftButton)
            p.setAcceptHoverEvents(True)
            return p

        # Create different UI and ports based on node type
        combo = None

        if "Logic Control" in name or "逻辑控制" in name:
            features = features or ["If", "While Loop"]
            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(int(w * 0.8))
            combo.setMaximumWidth(int(w - 16))
            combo.setStyleSheet("""
                QComboBox {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
                QComboBox QAbstractItemView {
                    background: #111827;
                    color: #e5e7eb;
                    selection-background-color: #334155;
                }
            """)

            condition_input = QLineEdit()
            condition_input.setPlaceholderText("condition / connect")
            condition_input.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)

            add_elif_btn = QPushButton("+elif")
            add_elif_btn.setFixedWidth(48)
            add_elif_btn.setStyleSheet("""
                QPushButton {
                    background: #111827;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #1f2937;
                }
            """)

            loop_type_combo = QComboBox()
            loop_type_combo.addItems(["While", "For"])
            loop_type_combo.setStyleSheet("""
                QComboBox {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)

            loop_label = QLabel("Loop:")
            loop_label.setStyleSheet("QLabel { color: #cbd5e1; font-size: 11px; }")

            for_start_input = QLineEdit()
            for_start_input.setPlaceholderText("start")
            for_start_input.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            for_start_input.setMaximumWidth(52)

            for_end_input = QLineEdit()
            for_end_input.setPlaceholderText("end")
            for_end_input.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            for_end_input.setMaximumWidth(52)

            for_step_input = QLineEdit()
            for_step_input.setPlaceholderText("step")
            for_step_input.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            for_step_input.setMaximumWidth(52)

            widget_container = QWidget()
            vbox = QVBoxLayout(widget_container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)
            vbox.addWidget(combo)

            cond_row = QHBoxLayout()
            cond_row.setContentsMargins(0, 0, 0, 0)
            cond_row.setSpacing(4)
            cond_row.addWidget(condition_input)
            cond_row.addWidget(add_elif_btn)
            vbox.addLayout(cond_row)

            loop_row = QHBoxLayout()
            loop_row.setContentsMargins(0, 0, 0, 0)
            loop_row.setSpacing(4)
            loop_row.addWidget(loop_label)
            loop_row.addWidget(loop_type_combo)
            loop_row.addStretch(1)
            vbox.addLayout(loop_row)

            for_row = QHBoxLayout()
            for_row.setContentsMargins(0, 0, 0, 0)
            for_row.setSpacing(4)
            for_row.addWidget(for_start_input)
            for_row.addWidget(for_end_input)
            for_row.addWidget(for_step_input)
            for_row.addStretch(1)
            vbox.addLayout(for_row)

            # Ports
            condition_port = _mk_port(0, h * 0.50, "in", "condition")
            for_start_port = _mk_port(0, h * 0.62, "in", "for_start")
            for_end_port = _mk_port(0, h * 0.72, "in", "for_end")
            for_step_port = _mk_port(0, h * 0.82, "in", "for_step")

            out_true = _mk_port(w, h * 0.28, "out", "out_true")
            out_false = _mk_port(w, h * 0.88, "out", "out_false")
            loop_body = _mk_port(w, h * 0.28, "out", "loop_body")
            loop_end = _mk_port(w, h * 0.88, "out", "loop_end")

            rect._elif_input_ports = []
            rect._elif_output_ports = []
            rect._elif_inputs = []

            def _layout_elif_ports():
                total = len(rect._elif_output_ports)
                if total == 0:
                    return
                start, end = 0.45, 0.72
                for i, (pin, pout) in enumerate(zip(rect._elif_input_ports, rect._elif_output_ports)):
                    y = h * (start + (end - start) * (i + 1) / (total + 1))
                    pin.setPos(0, y)
                    pout.setPos(w, y)

            def _add_elif():
                idx = len(rect._elif_output_ports)
                inp = _mk_port(0, h * 0.60, "in", f"elif_{idx}")
                outp = _mk_port(w, h * 0.60, "out", f"out_elif_{idx}")
                rect._elif_input_ports.append(inp)
                rect._elif_output_ports.append(outp)

                elif_input = QLineEdit()
                elif_input.setPlaceholderText(f"elif {idx}")
                elif_input.setStyleSheet("""
                    QLineEdit {
                        background: #0f1115;
                        color: #e5e7eb;
                        border: 1px solid #4b5563;
                        border-radius: 4px;
                        padding: 2px 4px;
                        font-size: 11px;
                    }
                """)
                rect._elif_inputs.append(elif_input)
                elif_input.textChanged.connect(lambda _t: self._update_node_params(rect))
                vbox.insertWidget(vbox.count() - 2, elif_input)
                _layout_elif_ports()
                self.regenerate_code()

            add_elif_btn.clicked.connect(_add_elif)

            def _set_if_mode(enabled: bool):
                condition_input.setVisible(enabled)
                add_elif_btn.setVisible(enabled)
                for inp in rect._elif_inputs:
                    inp.setVisible(enabled)
                for p in rect._elif_input_ports + rect._elif_output_ports:
                    p.setVisible(enabled)
                out_true.setVisible(enabled)
                out_false.setVisible(enabled)
                condition_port.setVisible(enabled)

                loop_body.setVisible(not enabled)
                loop_end.setVisible(not enabled)

            def _set_for_ports_visible(enabled: bool):
                for_start_input.setVisible(enabled)
                for_end_input.setVisible(enabled)
                for_step_input.setVisible(enabled)
                for_start_port.setVisible(enabled)
                for_end_port.setVisible(enabled)
                for_step_port.setVisible(enabled)

            def _set_loop_mode():
                _set_if_mode(False)
                loop_label.setVisible(True)
                loop_type_combo.setVisible(True)
                loop_body.setVisible(True)
                loop_end.setVisible(True)
                is_for = loop_type_combo.currentText() == "For"
                condition_port.setVisible(not is_for)
                condition_input.setVisible(not is_for)
                _set_for_ports_visible(is_for)
                self.regenerate_code()

            def _set_if_only():
                loop_label.setVisible(False)
                loop_type_combo.setVisible(False)
                _set_for_ports_visible(False)
                _set_if_mode(True)
                self.regenerate_code()

            def _on_mode_change():
                if combo.currentText().lower().startswith("while"):
                    _set_loop_mode()
                else:
                    _set_if_only()

            combo.currentTextChanged.connect(_on_mode_change)
            loop_type_combo.currentTextChanged.connect(lambda _t: _set_loop_mode())

            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(widget_container)
            proxy.setPos(8, 38)
            proxy.setZValue(2)

            rect._condition_input = condition_input
            rect._loop_type_combo = loop_type_combo
            rect._for_start_input = for_start_input
            rect._for_end_input = for_end_input
            rect._for_step_input = for_step_input

            _on_mode_change()
            condition_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            loop_type_combo.currentTextChanged.connect(lambda _t: self._update_node_params(rect))
            for_start_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            for_end_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            for_step_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            combo.currentTextChanged.connect(lambda _t: self._update_node_params(rect))

        elif "Condition" in name or "条件判断" in name:
            features = features or ["Equal", "Not Equal", "Greater Than", "Less Than"]
            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(60)
            combo.setMaximumWidth(70)
            combo.setStyleSheet("""
                QComboBox {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)

            input_box = QLineEdit()
            input_box.setPlaceholderText("inputs")
            input_box.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            input_box.setMaximumWidth(80)

            output_box = QLineEdit()
            output_box.setPlaceholderText("outputs")
            output_box.setStyleSheet("""
                QLineEdit {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)
            output_box.setMaximumWidth(80)

            widget_container = QWidget()
            hbox = QHBoxLayout(widget_container)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(4)
            hbox.addWidget(combo)
            hbox.addWidget(input_box)
            hbox.addWidget(output_box)

            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(widget_container)
            proxy.setPos(8, 38)
            proxy.setZValue(2)

            _mk_port(0, h * 0.45, "in", "left")
            _mk_port(0, h * 0.70, "in", "right")
            _mk_port(w, h / 2, "out", "result")

            rect._input_box = input_box
            rect._output_box = output_box
            rect._combo = combo
            input_box.textChanged.connect(lambda _t: self._update_node_params(rect))
            output_box.textChanged.connect(lambda _t: self._update_node_params(rect))
            combo.currentTextChanged.connect(lambda _t: self._update_node_params(rect))

        else:
            # Other node types
            if "Action Execution" in name:
                features = features or ["Lift Right Leg", "Stand", "Sit", "Walk", "Stop"]
            elif "Sensor Input" in name:
                features = features or ["Read Ultrasonic", "Read Infrared", "Read Camera", "Read IMU",
                                        "Read Odometry"]
            elif "Compute" in name:
                features = features or ["Add", "Subtract", "Multiply", "Divide"]

            combo = QComboBox()
            combo.addItems(features)
            combo.setMinimumWidth(int(w * 0.85))
            combo.setMaximumWidth(int(w - 16))
            combo.setStyleSheet("""
                QComboBox {
                    background: #0f1115;
                    color: #e5e7eb;
                    border: 1px solid #4b5563;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-size: 11px;
                }
            """)

            _mk_port(0, h / 2, "in", "in")
            _mk_port(w, h / 2, "out", "out")

            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(combo)
            proxy.setPos(8, 38)
            proxy.setZValue(2)

        # Node metadata
        node_id = self._node_seq
        self._node_seq += 1
        rect.setData(10, "node")
        rect.setData(11, name)
        rect.setData(12, node_id)

        # Save combo reference
        if combo:
            rect._combo = combo
            combo.currentTextChanged.connect(lambda _t: self.regenerate_code())

        self.regenerate_code()
        log_info(f"Node created: {name} (ID: {node_id})")

        return rect

    def _find_port_near(self, pos, radius=14):
        """Find port near position"""
        search_rect = QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        candidates = []

        for it in self.items(search_rect):
            if self._is_port(it):
                c = self._port_center(it)
                dist2 = (c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2
                candidates.append((dist2, it))

        return min(candidates, key=lambda t: t[0])[1] if candidates else None

    def _is_port(self, item):
        """Check if item is a port"""
        return bool(item) and item.data(0) == "port"

    def _port_center(self, port_item):
        """Get port center position"""
        return port_item.mapToScene(port_item.boundingRect().center())

    def _attach_connection_safe(self, port_item, conn_item):
        """Safely attach connection to port"""
        try:
            conns = port_item.data(2) or []
            cleaned = []
            for c in conns:
                if c and isValid(c) and (c.scene() is not None):
                    cleaned.append(c)
            cleaned.append(conn_item)
            port_item.setData(2, cleaned)
        except Exception:
            pass

    def _apply_connection_to_input(self, in_port, out_port):
        """Apply incoming connection to input widgets"""
        if not in_port or not out_port:
            return
        node_item = in_port.parentItem()
        if not node_item or node_item.data(10) != "node":
            return

        in_slot = in_port.data(3)
        label = self._format_connection_label(out_port)

        if in_slot == "condition":
            inp = getattr(node_item, "_condition_input", None)
            if inp:
                inp.setText(label)
        elif isinstance(in_slot, str) and in_slot.startswith("elif_"):
            idx = int(in_slot.split("_")[1])
            elif_inputs = getattr(node_item, "_elif_inputs", [])
            if idx < len(elif_inputs):
                elif_inputs[idx].setText(label)
        elif in_slot in ("for_start", "for_end", "for_step"):
            field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
            if field:
                field.setText(label)
        elif in_slot in ("left", "right"):
            input_box = getattr(node_item, "_input_box", None)
            if input_box:
                cmp_inputs = getattr(node_item, "_cmp_inputs", {"left": "", "right": ""})
                cmp_inputs[in_slot] = label
                node_item._cmp_inputs = cmp_inputs
                left = cmp_inputs.get("left", "")
                right = cmp_inputs.get("right", "")
                parts = []
                if left:
                    parts.append(f"left={left}")
                if right:
                    parts.append(f"right={right}")
                input_box.setText(", ".join(parts))

        self._update_node_params(node_item)

    def _clear_input_for_port(self, in_port):
        """Clear input widgets when a connection is removed"""
        if not in_port:
            return
        node_item = in_port.parentItem()
        if not node_item or node_item.data(10) != "node":
            return
        in_slot = in_port.data(3)

        if in_slot == "condition":
            inp = getattr(node_item, "_condition_input", None)
            if inp:
                inp.setText("")
        elif isinstance(in_slot, str) and in_slot.startswith("elif_"):
            idx = int(in_slot.split("_")[1])
            elif_inputs = getattr(node_item, "_elif_inputs", [])
            if idx < len(elif_inputs):
                elif_inputs[idx].setText("")
        elif in_slot in ("for_start", "for_end", "for_step"):
            field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
            if field:
                field.setText("")
        elif in_slot in ("left", "right"):
            input_box = getattr(node_item, "_input_box", None)
            if input_box:
                cmp_inputs = getattr(node_item, "_cmp_inputs", {"left": "", "right": ""})
                cmp_inputs[in_slot] = ""
                node_item._cmp_inputs = cmp_inputs
                left = cmp_inputs.get("left", "")
                right = cmp_inputs.get("right", "")
                parts = []
                if left:
                    parts.append(f"left={left}")
                if right:
                    parts.append(f"right={right}")
                input_box.setText(", ".join(parts))

        self._update_node_params(node_item)

    def _format_connection_label(self, out_port):
        """Format a readable label for a connected output"""
        if not out_port:
            return ""
        node_item = out_port.parentItem()
        if node_item and node_item.data(10) == "node":
            name = node_item.data(11)
            slot = out_port.data(3)
            return f"{name}.{slot}"
        return str(out_port.data(3))

    def _update_node_params(self, node_item):
        """Collect UI values into node metadata for later execution"""
        if not node_item or node_item.data(10) != "node":
            return

        params = node_item.data(20) or {}

        if hasattr(node_item, "_condition_input"):
            params["condition_expr"] = node_item._condition_input.text()
        if hasattr(node_item, "_elif_inputs"):
            params["elif_conditions"] = [w.text() for w in node_item._elif_inputs]
        if hasattr(node_item, "_loop_type_combo"):
            params["loop_type"] = node_item._loop_type_combo.currentText().lower()
        if hasattr(node_item, "_for_start_input"):
            params["for_start"] = node_item._for_start_input.text()
        if hasattr(node_item, "_for_end_input"):
            params["for_end"] = node_item._for_end_input.text()
        if hasattr(node_item, "_for_step_input"):
            params["for_step"] = node_item._for_step_input.text()
        if hasattr(node_item, "_combo"):
            params["ui_selection"] = node_item._combo.currentText()

        if hasattr(node_item, "_input_box"):
            params["input_expr"] = node_item._input_box.text()
        if hasattr(node_item, "_output_box"):
            params["output_name"] = node_item._output_box.text()

        node_item.setData(20, params)

    def regenerate_code(self):
        """Regenerate code"""
        if not self._code_editor:
            return

        code_lines = [
            "# Auto-generated code",
            "# Generated by Celebrimbor",
            "",
            "def execute_workflow():",
        ]

        # Get all nodes
        nodes = []
        for item in self.items():
            if item.data(10) == "node":
                node_info = {
                    'id': item.data(12),
                    'name': item.data(11),
                    'combo': getattr(item, '_combo', None)
                }
                nodes.append(node_info)

        # Generate code
        for node in nodes:
            code_lines.append(f"    # {node['name']} (ID: {node['id']})")
            if node['combo']:
                action = node['combo'].currentText()
                code_lines.append(f"    # Action: {action}")

            # Logic control details
            if node['name'] and ("Logic Control" in node['name'] or "逻辑控制" in node['name']):
                item = None
                for it in self.items():
                    if it.data(10) == "node" and it.data(12) == node['id']:
                        item = it
                        break
                if item is not None:
                    is_if = node['combo'] and node['combo'].currentText().lower().startswith("if")
                    cond_input = getattr(item, "_condition_input", None)
                    if cond_input:
                        code_lines.append(f"    # condition: {cond_input.text()}")
                    if is_if:
                        for idx, inp in enumerate(getattr(item, "_elif_inputs", [])):
                            code_lines.append(f"    # elif_{idx}: {inp.text()}")
                    else:
                        loop_type_combo = getattr(item, "_loop_type_combo", None)
                        loop_type = loop_type_combo.currentText() if loop_type_combo else "While"
                        code_lines.append(f"    # loop_type: {loop_type}")
                        if loop_type == "For":
                            fs = getattr(item, "_for_start_input", None)
                            fe = getattr(item, "_for_end_input", None)
                            fp = getattr(item, "_for_step_input", None)
                            if fs and fe and fp:
                                code_lines.append(
                                    f"    # for: start={fs.text()}, end={fe.text()}, step={fp.text()}"
                                )

            # Condition/comparison details
            if node['name'] and ("Condition" in node['name'] or "条件判断" in node['name']):
                item = None
                for it in self.items():
                    if it.data(10) == "node" and it.data(12) == node['id']:
                        item = it
                        break
                if item is not None:
                    inp = getattr(item, "_input_box", None)
                    outp = getattr(item, "_output_box", None)
                    if inp:
                        code_lines.append(f"    # inputs: {inp.text()}")
                    if outp:
                        code_lines.append(f"    # outputs: {outp.text()}")

            code_lines.append("")

        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_workflow()")

        # Use set_code method
        self._code_editor.set_code("\n".join(code_lines))
