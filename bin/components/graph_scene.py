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
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QLinearGradient, QGradient, QPainterPath, QDoubleValidator
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPathItem, QGraphicsProxyWidget, QComboBox, QLineEdit, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QLabel
)

from bin.core.logger import log_info, log_error, log_debug, log_warning, log_success
from bin.core.theme_manager import get_color, get_node_color_pair

# Import node system
from nodes import create_node as create_logic_node, get_node_class, REGISTERED_NODES


class ConnectionItem(QGraphicsPathItem):
    """Connection Line Item - Supports auto-update and endpoint editing"""

    def __init__(self, out_port, in_port, parent=None):
        super().__init__(parent)

        self.out_port = out_port
        self.in_port = in_port

        # Set style
        self.setPen(QPen(QColor(self._base_color()), 2.5))
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
        self._start_marker.setBrush(QBrush(QColor(self._base_color())))
        self._start_marker.setPen(QPen(QColor(self._marker_border_color()), 1))
        self._start_marker.setZValue(10)
        self._start_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._start_marker.setData(0, "connection_marker")
        self._start_marker.setData(1, "start")
        self._start_marker.setData(2, self)  # Associated connection
        self._start_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._start_marker.setVisible(False)

        # End marker
        self._end_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._end_marker.setBrush(QBrush(QColor(self._base_color())))
        self._end_marker.setPen(QPen(QColor(self._marker_border_color()), 1))
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

        # Check if parent items exist (ports must belong to nodes)
        if not self.out_port.parentItem() or not self.in_port.parentItem():
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
        self.setPen(QPen(QColor(self._hover_color()), 3.5))
        if self._start_marker:
            self._start_marker.setVisible(True)
        if self._end_marker:
            self._end_marker.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Mouse leave - hide endpoint markers"""
        if not self.isSelected():
            self.setPen(QPen(QColor(self._base_color()), 2.5))
        if self._start_marker:
            self._start_marker.setVisible(False)
        if self._end_marker:
            self._end_marker.setVisible(False)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """Item change event"""
        if change == QGraphicsItem.ItemSelectedHasChanged:
            if value:  # Selected
                self.setPen(QPen(QColor(self._hover_color()), 3.5))
                if self._start_marker:
                    self._start_marker.setVisible(True)
                if self._end_marker:
                    self._end_marker.setVisible(True)
            else:  # Not selected
                self.setPen(QPen(QColor(self._base_color()), 2.5))
                if self._start_marker:
                    self._start_marker.setVisible(False)
                if self._end_marker:
                    self._end_marker.setVisible(False)

        return super().itemChange(change, value)

    def _base_color(self) -> str:
        return get_color("connection", "#60a5fa")

    def _hover_color(self) -> str:
        return get_color("connection_hover", "#3b82f6")

    def _marker_border_color(self) -> str:
        return get_color("connection_marker_border", "#ffffff")

    def refresh_style(self):
        """Refresh connection colors"""
        if self.isSelected():
            self.setPen(QPen(QColor(self._hover_color()), 3.5))
        else:
            self.setPen(QPen(QColor(self._base_color()), 2.5))
        if self._start_marker:
            self._start_marker.setBrush(QBrush(QColor(self._base_color())))
            self._start_marker.setPen(QPen(QColor(self._marker_border_color()), 1))
        if self._end_marker:
            self._end_marker.setBrush(QBrush(QColor(self._base_color())))
            self._end_marker.setPen(QPen(QColor(self._marker_border_color()), 1))


class PortInputRow(QWidget):
    """Input row widget that keeps a port aligned to its geometry."""

    def __init__(self, placeholder: str, style: str, trailing: Optional[QWidget] = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.setStyleSheet(style)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.line_edit)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing)

    def center_y(self, proxy: QGraphicsProxyWidget) -> float:
        geo = self.geometry()
        return proxy.pos().y() + geo.y() + geo.height() / 2


def _block_recursive(widget, block: bool):
    """Recursively block/unblock signals on a widget and all its children."""
    if widget is None:
        return
    widget.blockSignals(block)
    for child in widget.findChildren(QWidget):
        child.blockSignals(block)


class GraphScene(QGraphicsScene):
    """Graph Editor Scene"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene configuration
        self.setSceneRect(-2500, -2500, 5000, 5000)
        self.grid_small = 20
        self.grid_big = self.grid_small * 5

        # Color configuration
        self._apply_theme()

        # Nodes and connections
        self._node_seq = 0
        self._temp_connection = None
        self._temp_start_port = None

        # Reconnection state
        self._reconnecting = False
        self._reconnect_connection = None
        self._reconnect_end = None  # "start" or "end"
        self._reconnect_original_port = None  # Save original port for cancel

        # Action mapping (UI action name -> robot action)
        self._action_mapping = {
            "Lift Right Leg": "lift_right_leg",
            "Stand": "stand",
            "Sit": "sit",
            "Walk": "walk",
            "Stop": "stop"
        }

        # Node display name -> logic node type mapping
        self._node_type_mapping = {
            "Action Execution": "action_execution",
            "Sensor Input": "sensor_input",
            "Logic Control": "if",  # Default to if, will be updated based on combo
            "Condition": "comparison",
            "Math": "math",
            "Timer": "timer",
        }

        # Store logic node instances (node_id -> BaseNode instance)
        self._logic_nodes: Dict[int, Any] = {}

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

    def _apply_theme(self):
        """Apply theme colors and widget styles"""
        self.color_bg = QColor(get_color("graph_bg", get_color("bg", "#1e1e1e")))
        self.color_grid_small = QColor(get_color("graph_grid_small", "#323337"))
        self.color_grid_big = QColor(get_color("graph_grid_big", "#3c3e43"))
        self._node_title_color = QColor(get_color("node_title", "#ffffff"))

        input_bg = get_color("input_bg", get_color("cmd_bg", "#0f1115"))
        input_text = get_color("input_text", get_color("text_primary", "#e5e7eb"))
        input_border = get_color("input_border", get_color("border", "#4b5563"))
        popup_bg = get_color("input_popup_bg", get_color("card_bg", "#111827"))
        popup_sel = get_color("input_popup_selected_bg", get_color("hover_bg", "#334155"))
        hover_bg = get_color("hover_bg", "#1f2937")
        button_bg = get_color("button_bg", get_color("card_bg", "#111827"))
        button_text = get_color("button_text", get_color("text_primary", "#e5e7eb"))
        button_border = get_color("button_border", input_border)

        self._combo_style = f"""
            QComboBox {{
                background: {input_bg};
                color: {input_text};
                border: 1px solid {input_border};
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }}
        """
        self._combo_view_style = f"""
            QComboBox QAbstractItemView {{
                background: {popup_bg};
                color: {input_text};
                selection-background-color: {popup_sel};
            }}
        """
        self._input_style = f"""
            QLineEdit {{
                background: {input_bg};
                color: {input_text};
                border: 1px solid {input_border};
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }}
        """
        self._button_style = f"""
            QPushButton {{
                background: {button_bg};
                color: {button_text};
                border: 1px solid {button_border};
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {hover_bg};
            }}
        """
        tag_bg = get_color("tag_bg", get_color("hover_bg", "#3d3d3d"))
        tag_text = get_color("tag_text", get_color("text_primary", "#ffffff"))
        self._tag_style = f"""
            QLabel#nodeTag {{
                color: {tag_text};
                background: {tag_bg};
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }}
        """

    def _resolve_node_gradient(self, name: str, grad: Optional[tuple]) -> tuple:
        """Resolve node gradient from ui.ini NodeColors"""
        card_bg = get_color("card_bg", "#2d2d2d")
        fallback_start = grad[0] if grad and len(grad) == 2 else card_bg
        fallback_end = grad[1] if grad and len(grad) == 2 else card_bg

        if "Logic Control" in name or "閫昏緫鎺у埗" in name:
            return get_node_color_pair("logic", fallback_start, fallback_end)
        if "Condition" in name or "鏉′欢鍒ゆ柇" in name:
            return get_node_color_pair("condition", fallback_start, fallback_end)
        if "Action Execution" in name:
            return get_node_color_pair("action", fallback_start, fallback_end)
        if "Sensor Input" in name:
            return get_node_color_pair("sensor", fallback_start, fallback_end)
        if "Compute" in name:
            return get_node_color_pair("compute", fallback_start, fallback_end)

        if grad and len(grad) == 2:
            return tuple(grad)
        return (card_bg, card_bg)

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
            try:
                if not isValid(item):
                    continue

                # Delete node
                if item.data(10) == "node":
                    node_name = item.data(11)
                    node_id = item.data(12)

                    # Delete all connections related to the node
                    self._delete_node_connections(item)

                    # Delete corresponding logic node
                    if node_id is not None and node_id in self._logic_nodes:
                        del self._logic_nodes[node_id]

                    # Delete the node itself
                    if item.scene() is not None:
                        self.removeItem(item)
                    deleted_nodes.append(f"{node_name} (ID: {node_id})")
                    log_info(f"Node deleted: {node_name} (ID: {node_id})")

                # Delete connection
                elif item.data(0) == "connection" or isinstance(item, ConnectionItem):
                    # Remove connection reference from ports
                    if isinstance(item, ConnectionItem):
                        self._detach_connection(item)
                    if item.scene() is not None:
                        self.removeItem(item)
                    deleted_connections.append("Connection")
            except Exception as e:
                log_debug(f"Error deleting item: {e}")

        if deleted_nodes:
            log_success(f"{len(deleted_nodes)} node(s) deleted")
        if deleted_connections:
            log_info(f"{len(deleted_connections)} connection(s) deleted")

        # Regenerate code
        self.regenerate_code()

    def _delete_node_connections(self, node_item):
        """Delete all connections related to a node"""
        try:
            if not node_item or not isValid(node_item):
                return

            # Find all ports
            ports = []
            for child in node_item.childItems():
                if child and isValid(child) and self._is_port(child):
                    ports.append(child)

            # Delete connections for each port
            for port in ports:
                connections = port.data(2) or []
                for conn in list(connections):  # Use list() to create copy to avoid modification during iteration
                    if conn and isValid(conn) and conn.scene() is not None:
                        self._detach_connection(conn)
                        self.removeItem(conn)
        except Exception as e:
            log_debug(f"Error deleting node connections: {e}")

    def _detach_connection(self, connection):
        """Remove connection reference from ports"""
        try:
            if not isinstance(connection, ConnectionItem):
                return

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
        except Exception as e:
            log_debug(f"Error detaching connection: {e}")

    def _start_reconnection(self, connection, end_type, pos):
        """Start reconnection"""
        if not isinstance(connection, ConnectionItem):
            return

        self._reconnecting = True
        self._reconnect_connection = connection
        self._reconnect_end = end_type

        # Determine start port and save original port for potential cancel
        if end_type == "start":
            self._reconnect_original_port = connection.out_port
            self._temp_start_port = connection.in_port  # Keep the other end as anchor
            # Temporarily remove output end
            connection.out_port = None
        else:
            self._reconnect_original_port = connection.in_port
            self._temp_start_port = connection.out_port  # Keep the other end as anchor
            # Temporarily remove input end
            connection.in_port = None

        # Create temporary connection
        path = QPainterPath()
        center = self._port_center(self._temp_start_port)
        path.moveTo(center)
        path.lineTo(pos)

        self._temp_connection = QGraphicsPathItem(path)
        temp_color = get_color("connection_temp", "#f59e0b")
        self._temp_connection.setPen(QPen(QColor(temp_color), 3))
        self.addItem(self._temp_connection)

        log_debug(f"Starting reconnection: {end_type} end")

    def _finish_reconnection(self, target_port):
        """Finish reconnection"""
        if not self._reconnect_connection or not target_port:
            self._cancel_reconnection()
            return

        # Check if temp_start_port is valid
        if not self._temp_start_port or not isValid(self._temp_start_port):
            self._cancel_reconnection()
            return

        # Check connection direction
        start_io = self._temp_start_port.data(1)
        target_io = target_port.data(1)

        # For reconnection, we need to connect to the opposite type
        # start_io is the anchor port type, target should be same as original disconnected end
        if self._reconnect_end == "start":
            # Reconnecting the start (out) end, target should be "out"
            if target_io != "out":
                log_warning("Cannot connect ports of the same type")
                self._cancel_reconnection()
                return
            self._reconnect_connection.out_port = target_port
        else:
            # Reconnecting the end (in) end, target should be "in"
            if target_io != "in":
                log_warning("Cannot connect ports of the same type")
                self._cancel_reconnection()
                return
            self._reconnect_connection.in_port = target_port

        # Remove from original port's connection list if different
        if self._reconnect_original_port and self._reconnect_original_port != target_port:
            conns = self._reconnect_original_port.data(2) or []
            if self._reconnect_connection in conns:
                conns.remove(self._reconnect_connection)
                self._reconnect_original_port.setData(2, conns)
            # Clear input widget if it was an input port
            if self._reconnect_original_port.data(1) == "in":
                self._clear_input_for_port(self._reconnect_original_port)

        # Attach to new port
        self._attach_connection_safe(target_port, self._reconnect_connection)

        # Update path
        self._reconnect_connection.update_path()
        if self._reconnect_connection.in_port and self._reconnect_connection.out_port:
            self._apply_connection_to_input(
                self._reconnect_connection.in_port,
                self._reconnect_connection.out_port
            )

        # Clean up temporary state (but don't restore original port since we succeeded)
        conn = self._reconnect_connection
        self._reconnect_connection = None
        self._reconnect_original_port = None

        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None

        self._temp_start_port = None
        self._reconnecting = False

        log_info("Reconnection successful")
        self.regenerate_code()

    def _cancel_reconnection(self):
        """Cancel reconnection"""
        if self._reconnect_connection and self._reconnect_original_port:
            # Restore original port reference
            if self._reconnect_end == "start":
                self._reconnect_connection.out_port = self._reconnect_original_port
            else:
                self._reconnect_connection.in_port = self._reconnect_original_port
            # Update the connection path
            self._reconnect_connection.update_path()

        self._reconnect_connection = None
        self._reconnect_original_port = None

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
        temp_color = get_color("connection", "#60a5fa")
        self._temp_connection.setPen(QPen(QColor(temp_color), 3))
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

    def refresh_style(self):
        """Refresh theme styles across the scene"""
        self._apply_theme()
        port_bg = get_color("port_bg", "#1f2937")
        port_border = get_color("port_border", get_color("connection", "#60a5fa"))
        node_border = get_color("node_border", get_color("border", "#78828c"))

        for item in self.items():
            if isinstance(item, ConnectionItem):
                item.refresh_style()
                continue

            if item.data(0) == "port":
                item.setBrush(QBrush(QColor(port_bg)))
                item.setPen(QPen(QColor(port_border), 2))
                continue

            if item.data(10) == "node":
                if isinstance(item, QGraphicsRectItem):
                    item.setPen(QPen(QColor(node_border), 2))
                for child in item.childItems():
                    if isinstance(child, QGraphicsProxyWidget):
                        self._apply_proxy_widget_theme(child)
                    elif hasattr(child, "setDefaultTextColor"):
                        child.setDefaultTextColor(self._node_title_color)

        self.update()

    def _apply_proxy_widget_theme(self, proxy: QGraphicsProxyWidget):
        widget = proxy.widget()
        if not widget:
            return
        if isinstance(widget, QComboBox):
            widget.setStyleSheet(self._combo_style + self._combo_view_style)
        if isinstance(widget, QLineEdit):
            widget.setStyleSheet(self._input_style)
        if isinstance(widget, QPushButton):
            widget.setStyleSheet(self._button_style)
        if isinstance(widget, QLabel) and widget.objectName() == "nodeTag":
            widget.setStyleSheet(self._tag_style)

        for combo in widget.findChildren(QComboBox):
            combo.setStyleSheet(self._combo_style + self._combo_view_style)
        for line_edit in widget.findChildren(QLineEdit):
            line_edit.setStyleSheet(self._input_style)
        for btn in widget.findChildren(QPushButton):
            btn.setStyleSheet(self._button_style)
        for lbl in widget.findChildren(QLabel):
            if lbl.objectName() == "nodeTag":
                lbl.setStyleSheet(self._tag_style)

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
            w, h = 240, 200
        elif "Condition" in name or "条件判断" in name:
            w, h = 260, 170
        else:
            w, h = 180, 110

        # Create node rectangle
        rect = QGraphicsRectItem(0, 0, w, h)

        # Gradient background
        resolved_grad = self._resolve_node_gradient(name, grad)
        if resolved_grad and len(resolved_grad) == 2:
            g = QLinearGradient(0, 0, 1, 1)
            g.setCoordinateMode(QGradient.ObjectBoundingMode)
            g.setColorAt(0.0, QColor(resolved_grad[0]))
            g.setColorAt(1.0, QColor(resolved_grad[1]))
            rect.setBrush(QBrush(g))
        else:
            rect.setBrush(QBrush(QColor(45, 50, 60)))

        rect.setPen(QPen(QColor(get_color("node_border", get_color("border", "#78828c"))), 2))
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
        label.setDefaultTextColor(self._node_title_color)
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

        def _mk_port(x, y, io, slot, radius=port_r):
            p = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2, rect)
            p.setPos(x, y)
            port_bg = get_color("port_bg", "#1f2937")
            port_border = get_color("port_border", get_color("connection", "#60a5fa"))
            p.setBrush(QBrush(QColor(port_bg)))
            p.setPen(QPen(QColor(port_border), 2))
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
            combo.setStyleSheet(self._combo_style + self._combo_view_style)

            condition_input = QLineEdit()
            condition_input.setPlaceholderText("condition / connect")
            condition_input.setStyleSheet(self._input_style)

            add_elif_btn = QPushButton("+elif")
            add_elif_btn.setFixedWidth(48)
            add_elif_btn.setStyleSheet(self._button_style)

            loop_type_combo = QComboBox()
            loop_type_combo.addItems(["While", "For"])
            loop_type_combo.setStyleSheet(self._combo_style + self._combo_view_style)

            def _make_tag(text: str) -> QLabel:
                lbl = QLabel(text)
                lbl.setObjectName("nodeTag")
                lbl.setStyleSheet(self._tag_style)
                return lbl

            loop_label = _make_tag("Loop")
            condition_label = _make_tag("Condition")
            out_if_label = _make_tag("If")
            out_else_label = _make_tag("Else")
            loop_body_label = _make_tag("Body")
            loop_end_label = _make_tag("End")
            for_end_label = _make_tag("End")

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
            for_start_input.setMaximumWidth(int(w - 16))

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
            for_end_input.setMaximumWidth(int(w - 16))

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
            for_step_input.setMaximumWidth(int(w - 16))

            widget_container = QWidget()
            widget_container.setStyleSheet("background: transparent;")
            vbox = QVBoxLayout(widget_container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)
            vbox.addWidget(combo)

            cond_row = QHBoxLayout()
            cond_row.setContentsMargins(0, 0, 0, 0)
            cond_row.setSpacing(4)
            cond_row.addWidget(condition_input)
            cond_row.addWidget(condition_label)
            cond_row.addWidget(add_elif_btn)
            cond_row.addStretch(1)
            cond_row.addWidget(out_if_label)
            vbox.addLayout(cond_row)

            else_row_widget = QWidget()
            else_row_widget.setStyleSheet("background: transparent;")
            else_row = QHBoxLayout(else_row_widget)
            else_row.setContentsMargins(0, 0, 0, 0)
            else_row.setSpacing(4)
            else_label = _make_tag("Else")
            else_row.addWidget(else_label)
            else_row.addStretch(1)
            else_row.addWidget(out_else_label)
            vbox.addWidget(else_row_widget)

            loop_row_widget = QWidget()
            loop_row_widget.setStyleSheet("background: transparent;")
            loop_row = QHBoxLayout(loop_row_widget)
            loop_row.setContentsMargins(0, 0, 0, 0)
            loop_row.setSpacing(4)
            loop_row.addWidget(loop_label)
            loop_row.addWidget(loop_type_combo)
            loop_row.addStretch(1)
            loop_row.addWidget(loop_body_label)
            vbox.addWidget(loop_row_widget)

            loop_end_row_widget = QWidget()
            loop_end_row_widget.setStyleSheet("background: transparent;")
            loop_end_row = QHBoxLayout(loop_end_row_widget)
            loop_end_row.setContentsMargins(0, 0, 0, 0)
            loop_end_row.setSpacing(4)
            loop_end_row.addStretch(1)
            loop_end_row.addWidget(loop_end_label)
            vbox.addWidget(loop_end_row_widget)

            for_start_row_widget = QWidget()
            for_start_row_widget.setStyleSheet("background: transparent;")
            for_start_row = QHBoxLayout(for_start_row_widget)
            for_start_row.setContentsMargins(0, 0, 0, 0)
            for_start_row.setSpacing(4)
            for_start_row.addWidget(for_start_input)
            for_start_row.addStretch(1)
            vbox.addWidget(for_start_row_widget)

            for_end_row_widget = QWidget()
            for_end_row_widget.setStyleSheet("background: transparent;")
            for_end_row = QHBoxLayout(for_end_row_widget)
            for_end_row.setContentsMargins(0, 0, 0, 0)
            for_end_row.setSpacing(4)
            for_end_row.addWidget(for_end_input)
            for_end_row.addStretch(1)
            vbox.addWidget(for_end_row_widget)

            for_step_row_widget = QWidget()
            for_step_row_widget.setStyleSheet("background: transparent;")
            for_step_row = QHBoxLayout(for_step_row_widget)
            for_step_row.setContentsMargins(0, 0, 0, 0)
            for_step_row.setSpacing(4)
            for_step_row.addWidget(for_step_input)
            for_step_row.addStretch(1)
            for_step_row.addWidget(for_end_label)
            vbox.addWidget(for_step_row_widget)

            # Ports
            data_port_x = 12
            flow_in_port = _mk_port(0, h * 0.22, "in", "flow_in", radius=6)
            condition_port = _mk_port(data_port_x, h * 0.50, "in", "condition", radius=4)
            for_start_port = _mk_port(data_port_x, h * 0.62, "in", "for_start", radius=4)
            for_end_port = _mk_port(data_port_x, h * 0.72, "in", "for_end", radius=4)
            for_step_port = _mk_port(data_port_x, h * 0.82, "in", "for_step", radius=4)

            out_if = _mk_port(w, h * 0.28, "out", "out_if", radius=6)
            out_else = _mk_port(w, h * 0.88, "out", "out_else", radius=6)
            loop_body = _mk_port(w, h * 0.28, "out", "loop_body", radius=6)
            loop_end = _mk_port(w, h * 0.88, "out", "loop_end", radius=6)

            rect._elif_input_ports = []
            rect._elif_output_ports = []
            rect._elif_inputs = []
            rect._elif_rows = []
            rect._elif_remove_btns = []

            def _remove_port_connections(port):
                conns = port.data(2) or []
                for conn in list(conns):
                    if conn and isValid(conn) and conn.scene() is not None:
                        self.removeItem(conn)

            def _reindex_elifs():
                for i, inp in enumerate(rect._elif_inputs):
                    inp.setPlaceholderText(f"elif {i}")
                for i, port in enumerate(rect._elif_input_ports):
                    port.setData(3, f"elif_{i}")
                for i, port in enumerate(rect._elif_output_ports):
                    port.setData(3, f"out_elif_{i}")

            def _add_elif():
                idx = len(rect._elif_output_ports)
                inp = _mk_port(data_port_x, h * 0.60, "in", f"elif_{idx}", radius=4)
                outp = _mk_port(w, h * 0.60, "out", f"out_elif_{idx}", radius=6)
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
                remove_btn = QPushButton("X")
                remove_btn.setFixedWidth(20)
                remove_btn.setStyleSheet("""
                    QPushButton {
                        background: #111827;
                        color: #e5e7eb;
                        border: 1px solid #4b5563;
                        border-radius: 4px;
                        padding: 0px;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: #1f2937;
                    }
                """)

                row_widget = QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                row_layout.addWidget(elif_input)
                row_layout.addWidget(remove_btn)
                row_layout.addStretch(1)
                row_layout.addWidget(_make_tag("Elif"))

                rect._elif_inputs.append(elif_input)
                rect._elif_rows.append(row_widget)
                rect._elif_remove_btns.append(remove_btn)

                def _remove_this():
                    idx_local = rect._elif_rows.index(row_widget) if row_widget in rect._elif_rows else -1
                    if idx_local < 0:
                        return
                    row_widget.setParent(None)
                    row_widget.deleteLater()
                    for port in (rect._elif_input_ports[idx_local], rect._elif_output_ports[idx_local]):
                        _remove_port_connections(port)
                        if port.scene() is not None:
                            self.removeItem(port)
                    rect._elif_inputs.pop(idx_local)
                    rect._elif_rows.pop(idx_local)
                    rect._elif_remove_btns.pop(idx_local)
                    rect._elif_input_ports.pop(idx_local)
                    rect._elif_output_ports.pop(idx_local)
                    _reindex_elifs()
                    QTimer.singleShot(0, _sync_layout)
                    self.regenerate_code()

                remove_btn.clicked.connect(_remove_this)
                elif_input.textChanged.connect(lambda _t: self._update_node_params(rect))

                insert_index = vbox.indexOf(else_row_widget)
                vbox.insertWidget(insert_index, row_widget)
                QTimer.singleShot(0, _sync_layout)
                self.regenerate_code()

            rect._add_elif = _add_elif
            add_elif_btn.clicked.connect(_add_elif)

            def _set_if_mode(enabled: bool):
                condition_input.setVisible(enabled)
                add_elif_btn.setVisible(enabled)
                else_row_widget.setVisible(enabled)
                condition_label.setVisible(not enabled)
                out_if_label.setVisible(enabled)
                out_else_label.setVisible(enabled)
                # no loop condition label in IF mode
                loop_body_label.setVisible(not enabled)
                loop_end_label.setVisible(False)
                for_end_label.setVisible(False)
                for row in rect._elif_rows:
                    row.setVisible(enabled)
                for p in rect._elif_input_ports + rect._elif_output_ports:
                    p.setVisible(enabled)
                out_if.setVisible(enabled)
                out_else.setVisible(enabled)
                condition_port.setVisible(enabled)
                # no loop condition port in IF mode

                loop_body.setVisible(not enabled)
                loop_end.setVisible(not enabled)
                loop_row_widget.setVisible(not enabled)
                loop_end_row_widget.setVisible(False)
                for_start_row_widget.setVisible(False)
                for_end_row_widget.setVisible(False)
                for_step_row_widget.setVisible(False)

            def _set_for_ports_visible(enabled: bool):
                for_start_input.setVisible(enabled)
                for_end_input.setVisible(enabled)
                for_step_input.setVisible(enabled)
                for_start_port.setVisible(enabled)
                for_end_port.setVisible(enabled)
                for_step_port.setVisible(enabled)
                for_start_row_widget.setVisible(enabled)
                for_end_row_widget.setVisible(enabled)
                for_step_row_widget.setVisible(enabled)
                for_end_label.setVisible(enabled)

            def _set_loop_mode():
                _set_if_mode(False)
                loop_label.setVisible(True)
                loop_type_combo.setVisible(True)
                loop_body.setVisible(True)
                loop_end.setVisible(True)
                is_for = loop_type_combo.currentText() == "For"
                condition_port.setVisible(not is_for)
                condition_input.setVisible(False)
                condition_label.setVisible(not is_for)
                _set_for_ports_visible(is_for)
                loop_end_row_widget.setVisible(not is_for)
                loop_end_label.setVisible(not is_for)
                QTimer.singleShot(0, _sync_layout)
                self.regenerate_code()

            def _set_if_only():
                loop_label.setVisible(False)
                loop_type_combo.setVisible(False)
                _set_for_ports_visible(False)
                loop_row_widget.setVisible(False)
                loop_end_row_widget.setVisible(False)
                _set_if_mode(True)
                QTimer.singleShot(0, _sync_layout)
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

            def _port_y(widget):
                if not widget:
                    return None
                if hasattr(widget, "center_y"):
                    return widget.center_y(proxy)
                geo = widget.geometry()
                return proxy.pos().y() + geo.y() + geo.height() / 2

            def _resize_to_fit(min_height: Optional[int] = None):
                layout = widget_container.layout()
                if layout:
                    layout.activate()
                widget_container.adjustSize()
                size_hint = widget_container.sizeHint()
                try:
                    proxy.setMinimumSize(size_hint)
                except Exception:
                    pass
                content_h = size_hint.height()
                target_h = int(proxy.pos().y() + content_h + 10)
                if min_height is not None:
                    target_h = max(target_h, min_height)
                rect_h = rect.rect().height()
                if target_h != rect_h:
                    rect.setRect(0, 0, rect.rect().width(), target_h)

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                flow_in_port.setPos(0, rect.rect().height() / 2)
                y = _port_y(condition_input if condition_input.isVisible() else condition_label)
                if y is not None:
                    condition_port.setPos(data_port_x, y)
                    out_if.setPos(w_now, y)
                y = _port_y(else_row_widget)
                if y is not None:
                    out_else.setPos(w_now, y)
                y = _port_y(for_start_input)
                if y is not None:
                    for_start_port.setPos(data_port_x, y)
                y = _port_y(for_end_input)
                if y is not None:
                    for_end_port.setPos(data_port_x, y)
                y = _port_y(for_step_input)
                if y is not None:
                    for_step_port.setPos(data_port_x, y)
                y = _port_y(loop_type_combo)
                if y is not None:
                    loop_body.setPos(w_now, y)
                y = _port_y(for_step_row_widget if for_step_row_widget.isVisible() else loop_end_row_widget if loop_end_row_widget.isVisible() else loop_row_widget if loop_row_widget.isVisible() else condition_input)
                if y is not None:
                    loop_end.setPos(w_now, y)

                for idx, inp in enumerate(rect._elif_inputs):
                    if idx >= len(rect._elif_input_ports):
                        continue
                    y = _port_y(inp)
                    if y is not None:
                        rect._elif_input_ports[idx].setPos(data_port_x, y)
                        rect._elif_output_ports[idx].setPos(w_now, y)

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)

            rect._condition_input = condition_input
            rect._condition_label = condition_label
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
            combo.setStyleSheet(self._combo_style + self._combo_view_style)

            input_style = self._input_style

            left_row = PortInputRow("left", input_style)
            left_input = left_row.line_edit
            left_input.setMaximumWidth(int(w - 16))

            def _make_tag(text: str) -> QLabel:
                lbl = QLabel(text)
                lbl.setObjectName("nodeTag")
                lbl.setStyleSheet(self._tag_style)
                return lbl

            out_label = _make_tag("Result")

            widget_container = QWidget()
            widget_container.setStyleSheet("background: transparent;")
            vbox = QVBoxLayout(widget_container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)
            vbox.addWidget(combo)
            vbox.addWidget(left_row)
            right_row = PortInputRow("right", input_style, trailing=out_label)
            right_input = right_row.line_edit
            right_input.setMaximumWidth(int(w - 16))
            vbox.addWidget(right_row)

            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(widget_container)
            proxy.setPos(8, 38)
            proxy.setZValue(2)

            data_port_x = 12
            left_port = _mk_port(data_port_x, h * 0.45, "in", "left", radius=4)
            right_port = _mk_port(data_port_x, h * 0.70, "in", "right", radius=4)
            result_port = _mk_port(w - data_port_x, h / 2, "out", "result", radius=4)

            def _port_y(widget):
                if not widget:
                    return None
                geo = widget.geometry()
                return proxy.pos().y() + geo.y() + geo.height() / 2

            def _resize_to_fit(min_height: Optional[int] = None):
                layout = widget_container.layout()
                if layout:
                    layout.activate()
                widget_container.adjustSize()
                size_hint = widget_container.sizeHint()
                try:
                    proxy.setMinimumSize(size_hint)
                except Exception:
                    pass
                content_h = size_hint.height()
                target_h = int(proxy.pos().y() + content_h + 10)
                if min_height is not None:
                    target_h = max(target_h, min_height)
                rect_h = rect.rect().height()
                if target_h != rect_h:
                    rect.setRect(0, 0, rect.rect().width(), target_h)

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(left_row)
                if y is not None:
                    left_port.setPos(data_port_x, y)
                y = _port_y(right_row)
                if y is not None:
                    right_port.setPos(data_port_x, y)
                y = _port_y(right_row)
                if y is not None:
                    result_port.setPos(w_now - data_port_x, y)

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)

            rect._left_input = left_input
            rect._right_input = right_input
            rect._combo = combo
            left_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            right_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            combo.currentTextChanged.connect(lambda _t: self._update_node_params(rect))

        elif "Timer" in name:
            def _make_tag(text: str) -> QLabel:
                lbl = QLabel(text)
                lbl.setObjectName("nodeTag")
                lbl.setStyleSheet(self._tag_style)
                return lbl

            unit_label = _make_tag("s")
            duration_row = PortInputRow("duration (s)", self._input_style, trailing=unit_label)
            duration_input = duration_row.line_edit
            duration_input.setMaximumWidth(int(w - 16))
            duration_input.setValidator(QDoubleValidator(0.0, 60.0, 3, duration_input))

            widget_container = QWidget()
            widget_container.setStyleSheet("background: transparent;")
            vbox = QVBoxLayout(widget_container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)
            vbox.addWidget(duration_row)

            proxy = QGraphicsProxyWidget(rect)
            proxy.setWidget(widget_container)
            proxy.setPos(8, 38)
            proxy.setZValue(2)

            data_port_x = 12
            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6)
            flow_out_port = _mk_port(w, h / 2, "out", "flow_out", radius=6)
            duration_port = _mk_port(data_port_x, h * 0.65, "in", "duration", radius=4)

            def _resize_to_fit(min_height: Optional[int] = None):
                layout = widget_container.layout()
                if layout:
                    layout.activate()
                widget_container.adjustSize()
                size_hint = widget_container.sizeHint()
                try:
                    proxy.setMinimumSize(size_hint)
                except Exception:
                    pass
                content_h = size_hint.height()
                target_h = int(proxy.pos().y() + content_h + 10)
                if min_height is not None:
                    target_h = max(target_h, min_height)
                rect_h = rect.rect().height()
                if target_h != rect_h:
                    rect.setRect(0, 0, rect.rect().width(), target_h)

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                h_now = rect.rect().height()
                flow_in_port.setPos(0, h_now / 2)
                flow_out_port.setPos(w_now, h_now / 2)
                y = duration_row.center_y(proxy)
                duration_port.setPos(data_port_x, y)

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)

            rect._duration_input = duration_input

            def _on_duration_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            duration_input.textChanged.connect(lambda _t: _on_duration_change())

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
            combo.setStyleSheet(self._combo_style + self._combo_view_style)

            _mk_port(0, h / 2, "in", "flow_in", radius=6)
            _mk_port(w, h / 2, "out", "flow_out", radius=6)

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

        # Create corresponding logic node instance
        logic_node = self._create_logic_node(name, node_id, rect)
        if logic_node:
            self._logic_nodes[node_id] = logic_node
            rect.setData(13, logic_node)  # Store reference in graphics item

        # Save combo reference
        if combo:
            rect._combo = combo
            combo.currentTextChanged.connect(lambda _t: self._on_combo_changed(rect))

        self.regenerate_code()
        log_info(f"Node created: {name} (ID: {node_id})")

        return rect

    def _create_logic_node(self, name: str, node_id: int, rect_item) -> Optional[Any]:
        """Create corresponding logic node instance for a graphics node"""
        # Determine node type from display name
        node_type = None
        for display_name, ntype in self._node_type_mapping.items():
            if display_name in name:
                node_type = ntype
                break

        if not node_type:
            log_debug(f"No logic node mapping for: {name}")
            return None

        # Handle Logic Control special case
        if "Logic Control" in name:
            combo = getattr(rect_item, '_combo', None)
            if combo:
                selection = combo.currentText().lower()
                if selection.startswith("while") or selection.startswith("for"):
                    node_type = "while_loop"
                else:
                    node_type = "if"

        try:
            logic_node = create_logic_node(node_type, str(node_id))
            log_debug(f"Created logic node: {node_type} (ID: {node_id})")
            return logic_node
        except Exception as e:
            log_error(f"Failed to create logic node: {e}")
            return None

    def _on_combo_changed(self, rect_item):
        """Handle combo box selection change"""
        node_id = rect_item.data(12)
        name = rect_item.data(11)

        # Update logic node if needed (especially for Logic Control)
        if "Logic Control" in name:
            combo = getattr(rect_item, '_combo', None)
            if combo:
                selection = combo.currentText().lower()
                new_type = "while_loop" if (selection.startswith("while") or selection.startswith("for")) else "if"

                # Check if type changed
                old_node = self._logic_nodes.get(node_id)
                if old_node and old_node.node_type != new_type:
                    # Create new logic node with correct type
                    try:
                        new_node = create_logic_node(new_type, str(node_id))
                        self._logic_nodes[node_id] = new_node
                        rect_item.setData(13, new_node)
                        log_debug(f"Updated logic node type: {new_type}")
                    except Exception as e:
                        log_error(f"Failed to update logic node: {e}")

        # Sync parameters
        self._sync_node_parameters(rect_item)
        self.regenerate_code()

    def _sync_node_parameters(self, rect_item):
        """Sync UI values to logic node parameters"""
        node_id = rect_item.data(12)
        logic_node = self._logic_nodes.get(node_id)
        if not logic_node:
            return

        name = rect_item.data(11)
        combo = getattr(rect_item, '_combo', None)

        # Sync based on node type
        if "Action Execution" in name and combo:
            action = combo.currentText()
            # Map UI action to robot action
            robot_action = self._action_mapping.get(action, action.lower().replace(" ", "_"))
            logic_node.set_parameter('action', robot_action)

        elif "Sensor Input" in name and combo:
            sensor_map = {
                "Read Ultrasonic": "ultrasonic",
                "Read Infrared": "infrared",
                "Read Camera": "camera",
                "Read IMU": "imu",
                "Read Odometry": "odometry"
            }
            sensor_type = sensor_map.get(combo.currentText(), "imu")
            logic_node.set_parameter('sensor_type', sensor_type)

        elif "Logic Control" in name:
            cond_input = getattr(rect_item, '_condition_input', None)
            if cond_input:
                logic_node.set_parameter('condition_expr', cond_input.text())

            loop_type_combo = getattr(rect_item, '_loop_type_combo', None)
            if loop_type_combo:
                logic_node.set_parameter('loop_type', loop_type_combo.currentText().lower())

            # For loop parameters
            for_start = getattr(rect_item, '_for_start_input', None)
            for_end = getattr(rect_item, '_for_end_input', None)
            for_step = getattr(rect_item, '_for_step_input', None)
            if for_start:
                try:
                    logic_node.set_parameter('for_start', int(for_start.text() or 0))
                except ValueError:
                    logic_node.set_parameter('for_start', 0)
            if for_end:
                try:
                    logic_node.set_parameter('for_end', int(for_end.text() or 1))
                except ValueError:
                    logic_node.set_parameter('for_end', 1)
            if for_step:
                try:
                    logic_node.set_parameter('for_step', int(for_step.text() or 1))
                except ValueError:
                    logic_node.set_parameter('for_step', 1)

            # Elif conditions
            elif_inputs = getattr(rect_item, '_elif_inputs', [])
            logic_node.set_parameter('elif_conditions', [inp.text() for inp in elif_inputs])

        elif "Condition" in name:
            left_input = getattr(rect_item, '_left_input', None)
            right_input = getattr(rect_item, '_right_input', None)
            node_id = rect_item.data(12)
            if left_input:
                logic_node.set_parameter('input_expr', left_input.text())
            if right_input:
                # Keep the original text to preserve user input format
                logic_node.set_parameter('compare_value', right_input.text() or '0')
            # Set a unique output variable name based on node id
            logic_node.set_parameter('output_name', f'condition_{node_id}')
            if combo:
                op_map = {
                    "Equal": "==",
                    "Not Equal": "!=",
                    "Greater Than": ">",
                    "Less Than": "<",
                    "Greater Equal": ">=",
                    "Less Equal": "<="
                }
                logic_node.set_parameter('operator', op_map.get(combo.currentText(), "=="))

        elif "Math" in name and combo:
            # Map UI selection to operation
            op_map = {
                "Add": "add",
                "Subtract": "subtract",
                "Multiply": "multiply",
                "Divide": "divide",
                "Power": "power",
                "Modulo": "modulo",
                "Min": "min",
                "Max": "max",
                "Abs": "abs",
                "Sum": "sum",
                "Average": "average"
            }
            logic_node.set_parameter('operation', op_map.get(combo.currentText(), "add"))
            # Get input values if available
            left_input = getattr(rect_item, '_left_input', None)
            right_input = getattr(rect_item, '_right_input', None)
            if left_input:
                try:
                    logic_node.set_parameter('value_a', float(left_input.text() or 0))
                except ValueError:
                    logic_node.set_parameter('value_a', 0)
            if right_input:
                try:
                    logic_node.set_parameter('value_b', float(right_input.text() or 0))
                except ValueError:
                    logic_node.set_parameter('value_b', 0)

        elif "Timer" in name:
            duration_input = getattr(rect_item, "_duration_input", None)
            duration_text = duration_input.text().strip() if duration_input else ""
            try:
                duration_value = float(duration_text) if duration_text else 1.0
            except ValueError:
                duration_value = 1.0
            logic_node.set_parameter('duration', duration_value)
            logic_node.set_parameter('unit', 'seconds')

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
        try:
            if not in_port or not out_port:
                return
            if not isValid(in_port) or not isValid(out_port):
                return
            node_item = in_port.parentItem()
            if not node_item or not isValid(node_item) or node_item.data(10) != "node":
                return

            in_slot = in_port.data(3)
            label = self._format_connection_label(out_port)

            if in_slot == "condition":
                inp = getattr(node_item, "_condition_input", None)
                if inp and inp.isVisible():
                    inp.setText(label)
                else:
                    lbl = getattr(node_item, "_condition_label", None)
                    if lbl:
                        lbl.setText(label)
            elif isinstance(in_slot, str) and in_slot.startswith("elif_"):
                idx = int(in_slot.split("_")[1])
                elif_inputs = getattr(node_item, "_elif_inputs", [])
                if idx < len(elif_inputs):
                    elif_inputs[idx].setText(label)
            elif in_slot in ("for_start", "for_end", "for_step"):
                field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
                if field:
                    field.setText(label)
            elif in_slot == "duration":
                duration_input = getattr(node_item, "_duration_input", None)
                if duration_input:
                    duration_input.setText(label)
            elif in_slot in ("left", "right"):
                left_input = getattr(node_item, "_left_input", None)
                right_input = getattr(node_item, "_right_input", None)
                if left_input or right_input:
                    if in_slot == "left" and left_input:
                        left_input.setText(label)
                    if in_slot == "right" and right_input:
                        right_input.setText(label)
                else:
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
        except Exception as e:
            log_debug(f"Error applying connection to input: {e}")

    def _clear_input_for_port(self, in_port):
        """Clear input widgets when a connection is removed"""
        try:
            if not in_port or not isValid(in_port):
                return
            node_item = in_port.parentItem()
            if not node_item or not isValid(node_item) or node_item.data(10) != "node":
                return
            in_slot = in_port.data(3)

            if in_slot == "condition":
                inp = getattr(node_item, "_condition_input", None)
                if inp and inp.isVisible():
                    inp.setText("")
                else:
                    lbl = getattr(node_item, "_condition_label", None)
                    if lbl:
                        lbl.setText("Condition")
            elif isinstance(in_slot, str) and in_slot.startswith("elif_"):
                idx = int(in_slot.split("_")[1])
                elif_inputs = getattr(node_item, "_elif_inputs", [])
                if idx < len(elif_inputs):
                    elif_inputs[idx].setText("")
            elif in_slot in ("for_start", "for_end", "for_step"):
                field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
                if field:
                    field.setText("")
            elif in_slot == "duration":
                duration_input = getattr(node_item, "_duration_input", None)
                if duration_input:
                    duration_input.setText("")
            elif in_slot in ("left", "right"):
                left_input = getattr(node_item, "_left_input", None)
                right_input = getattr(node_item, "_right_input", None)
                if left_input or right_input:
                    if in_slot == "left" and left_input:
                        left_input.setText("")
                    if in_slot == "right" and right_input:
                        right_input.setText("")
                else:
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
        except Exception as e:
            log_debug(f"Error clearing input for port: {e}")

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

        if hasattr(node_item, "_left_input") or hasattr(node_item, "_right_input"):
            left_input = getattr(node_item, "_left_input", None)
            right_input = getattr(node_item, "_right_input", None)
            left = left_input.text() if left_input else ""
            right = right_input.text() if right_input else ""
            parts = []
            if left:
                parts.append(f"left={left}")
            if right:
                parts.append(f"right={right}")
            params["input_expr"] = ", ".join(parts)
        elif hasattr(node_item, "_input_box"):
            params["input_expr"] = node_item._input_box.text()
        if hasattr(node_item, "_output_box"):
            params["output_name"] = node_item._output_box.text()
        if hasattr(node_item, "_duration_input"):
            params["duration"] = node_item._duration_input.text()

        node_item.setData(20, params)

        # Trigger code regeneration after parameter update
        self.regenerate_code()

    def _build_workflow_order(self) -> List[Any]:
        """
        Build workflow execution order based on connections (left to right).
        Only includes connected nodes, using topological sort.

        Returns:
            List of node items in execution order
        """
        # Create snapshot to avoid modification during iteration
        items_snapshot = list(self.items())

        # Collect all connections and connected nodes
        connections = []
        connected_node_ids = set()
        node_map = {}  # id -> node item

        for item in items_snapshot:
            # Skip invalid items
            if not isValid(item):
                continue

            if isinstance(item, ConnectionItem):
                # Skip incomplete connections
                if not item.out_port or not item.in_port:
                    continue
                if not isValid(item.out_port) or not isValid(item.in_port):
                    continue

                out_node = item.out_port.parentItem()
                in_node = item.in_port.parentItem()

                if not out_node or not in_node:
                    continue
                if out_node.data(10) != "node" or in_node.data(10) != "node":
                    continue

                out_id = out_node.data(12)
                in_id = in_node.data(12)

                if out_id is None or in_id is None:
                    continue

                connections.append((out_id, in_id))
                connected_node_ids.add(out_id)
                connected_node_ids.add(in_id)
                node_map[out_id] = out_node
                node_map[in_id] = in_node

            elif item.data(10) == "node":
                node_id = item.data(12)
                if node_id is not None:
                    node_map[node_id] = item

        # If no connections, return empty (no connected workflow)
        if not connected_node_ids:
            return []

        # Build adjacency list and in-degree for topological sort
        graph = {nid: [] for nid in connected_node_ids}
        in_degree = {nid: 0 for nid in connected_node_ids}

        for out_id, in_id in connections:
            if out_id in graph and in_id in graph:
                graph[out_id].append(in_id)
                in_degree[in_id] += 1

        # Topological sort with position-based tie-breaking (left to right)
        # When multiple nodes have in_degree=0, prefer the leftmost one
        result = []
        candidates = [nid for nid, deg in in_degree.items() if deg == 0]

        while candidates:
            # Sort candidates by x position (leftmost first)
            candidates.sort(key=lambda nid: node_map[nid].pos().x() if nid in node_map else 0)
            current = candidates.pop(0)
            result.append(node_map[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidates.append(neighbor)

        # Check for cycles (not all nodes processed)
        if len(result) != len(connected_node_ids):
            log_warning("Cycle detected in node connections, using partial order")

        return result

    def _build_connection_graph(self) -> Dict[str, Any]:
        """Build a graph of all connections for code generation"""
        graph = {
            'nodes': {},        # node_id -> node_item
            'outgoing': {},     # node_id -> {port_name -> [(target_node_id, target_port)]}
            'incoming': {},     # node_id -> {port_name -> [(source_node_id, source_port)]}
        }

        # Create snapshot to avoid modification during iteration
        items_snapshot = list(self.items())

        # Collect all nodes
        for item in items_snapshot:
            if not isValid(item):
                continue
            if item.data(10) == "node":
                node_id = item.data(12)
                if node_id is not None:
                    graph['nodes'][node_id] = item
                    graph['outgoing'][node_id] = {}
                    graph['incoming'][node_id] = {}

        # Collect all connections
        for item in items_snapshot:
            if not isValid(item):
                continue
            if isinstance(item, ConnectionItem):
                # Skip incomplete connections
                if not item.out_port or not item.in_port:
                    continue
                if not isValid(item.out_port) or not isValid(item.in_port):
                    continue

                out_node = item.out_port.parentItem()
                in_node = item.in_port.parentItem()

                if not out_node or not in_node:
                    continue
                if out_node.data(10) != "node" or in_node.data(10) != "node":
                    continue

                out_id = out_node.data(12)
                in_id = in_node.data(12)

                if out_id is None or in_id is None:
                    continue
                if out_id not in graph['nodes'] or in_id not in graph['nodes']:
                    continue

                out_port = item.out_port.data(3)
                in_port = item.in_port.data(3)

                # Add to outgoing
                if out_port not in graph['outgoing'][out_id]:
                    graph['outgoing'][out_id][out_port] = []
                graph['outgoing'][out_id][out_port].append((in_id, in_port))

                # Add to incoming
                if in_port not in graph['incoming'][in_id]:
                    graph['incoming'][in_id][in_port] = []
                graph['incoming'][in_id][in_port].append((out_id, out_port))

        return graph

    def _find_entry_nodes(self, graph: Dict[str, Any]) -> List[int]:
        """Find nodes with no flow_in connections (entry points)"""
        entry_nodes = []
        for node_id, item in graph['nodes'].items():
            if not item or not isValid(item):
                continue
            incoming = graph['incoming'].get(node_id, {})
            # Check if has flow_in connection
            has_flow_in = 'flow_in' in incoming and len(incoming['flow_in']) > 0
            if not has_flow_in:
                entry_nodes.append(node_id)

        # Sort by x position (left to right) - with safe access
        def get_x_pos(nid):
            item = graph['nodes'].get(nid)
            if item and isValid(item):
                try:
                    return item.pos().x()
                except RuntimeError:
                    return 0
            return 0

        entry_nodes.sort(key=get_x_pos)
        return entry_nodes

    def _get_condition_from_connection(self, node_id: int, graph: Dict[str, Any]) -> str:
        """Get condition expression from connected Condition node or input field"""
        incoming = graph['incoming'].get(node_id, {})
        condition_sources = incoming.get('condition', [])

        if condition_sources:
            source_id, source_port = condition_sources[0]
            source_item = graph['nodes'].get(source_id)
            if source_item and isValid(source_item):
                source_name = source_item.data(11)
                # If connected to a Condition node, use its output variable
                if source_name and "Condition" in source_name:
                    logic_node = self._logic_nodes.get(source_id)
                    if logic_node:
                        output_name = logic_node.get_parameter('output_name', '') or 'result'
                        return output_name

        # Fallback to condition input text - always read current value from widget
        item = graph['nodes'].get(node_id)
        if item and isValid(item):
            cond_input = getattr(item, '_condition_input', None)
            if cond_input:
                text = cond_input.text()
                if text:
                    return text

        return 'condition'

    def _generate_node_code(self, node_id: int, graph: Dict[str, Any],
                            indent: int, generated: set) -> List[str]:
        """Recursively generate code for a node and its downstream nodes"""
        if node_id in generated:
            return []
        if node_id is None:
            return []

        generated.add(node_id)
        code_lines = []
        indent_str = "    " * indent

        item = graph['nodes'].get(node_id)
        if not item or not isValid(item):
            return []

        node_name = item.data(11)
        if not node_name:
            return []
        logic_node = self._logic_nodes.get(node_id)
        outgoing = graph['outgoing'].get(node_id, {})

        # Handle Logic Control nodes (if/while/for)
        if "Logic Control" in node_name:
            combo = getattr(item, '_combo', None)
            selection = combo.currentText().lower() if combo else "if"

            if selection.startswith("if"):
                # Generate if statement
                condition_expr = self._get_condition_from_connection(node_id, graph)
                code_lines.append(f"{indent_str}if {condition_expr}:")

                # Generate true branch
                true_targets = outgoing.get('out_if', [])
                if true_targets:
                    for target_id, _ in true_targets:
                        code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
                else:
                    code_lines.append(f"{indent_str}    pass")

                # Generate elif branches
                elif_inputs = getattr(item, '_elif_inputs', [])
                elif_output_ports = getattr(item, '_elif_output_ports', [])
                for idx, elif_input in enumerate(elif_inputs):
                    elif_cond = elif_input.text() or f"elif_condition_{idx}"
                    code_lines.append(f"{indent_str}elif {elif_cond}:")
                    elif_port = f'out_elif_{idx}'
                    elif_targets = outgoing.get(elif_port, [])
                    if elif_targets:
                        for target_id, _ in elif_targets:
                            code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
                    else:
                        code_lines.append(f"{indent_str}    pass")

                # Generate else branch
                code_lines.append(f"{indent_str}else:")
                false_targets = outgoing.get('out_else', [])
                if false_targets:
                    for target_id, _ in false_targets:
                        code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
                else:
                    code_lines.append(f"{indent_str}    pass")

            else:
                # While or For loop
                loop_type_combo = getattr(item, '_loop_type_combo', None)
                loop_type = loop_type_combo.currentText().lower() if loop_type_combo else "while"

                if loop_type == "for":
                    fs = getattr(item, '_for_start_input', None)
                    fe = getattr(item, '_for_end_input', None)
                    fp = getattr(item, '_for_step_input', None)
                    start = fs.text() if fs and fs.text() else "0"
                    end = fe.text() if fe and fe.text() else "10"
                    step = fp.text() if fp and fp.text() else "1"
                    code_lines.append(f"{indent_str}for i in range({start}, {end}, {step}):")
                else:
                    condition_expr = self._get_condition_from_connection(node_id, graph)
                    code_lines.append(f"{indent_str}while {condition_expr}:")

                # Generate loop body
                body_targets = outgoing.get('loop_body', [])
                if body_targets:
                    for target_id, _ in body_targets:
                        code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
                else:
                    code_lines.append(f"{indent_str}    pass")

                # Continue with loop_end (code after loop)
                end_targets = outgoing.get('loop_end', [])
                for target_id, _ in end_targets:
                    code_lines.extend(self._generate_node_code(target_id, graph, indent, generated))

        # Handle Condition nodes
        elif "Condition" in node_name:
            if logic_node:
                # Sync parameters
                self._sync_node_parameters(item)
                node_code = logic_node.to_code()
                for line in node_code.strip().split('\n'):
                    code_lines.append(f"{indent_str}{line}")

            # Continue with result output (data flow, not control flow)
            # Don't follow result connections as they are data, not flow

        # Handle Action/Sensor nodes
        else:
            if logic_node:
                self._sync_node_parameters(item)
                node_code = logic_node.to_code()
                for line in node_code.strip().split('\n'):
                    code_lines.append(f"{indent_str}{line}")
            else:
                # Fallback
                combo = getattr(item, '_combo', None)
                if combo and "Action Execution" in node_name:
                    action = combo.currentText()
                    robot_action = self._action_mapping.get(action, action.lower().replace(" ", "_"))
                    code_lines.append(f"{indent_str}# Action: {action}")
                    code_lines.append(f"{indent_str}robot.run_action('{robot_action}')")
                elif combo and "Sensor Input" in node_name:
                    code_lines.append(f"{indent_str}# Sensor read")
                    code_lines.append(f"{indent_str}sensor_data = robot.get_sensor_data()")

            # Continue with flow_out
            flow_targets = outgoing.get('flow_out', [])
            for target_id, _ in flow_targets:
                code_lines.extend(self._generate_node_code(target_id, graph, indent, generated))

        return code_lines

    def regenerate_code(self):
        """Regenerate code with proper control flow nesting"""
        # Suppress during batch load
        if getattr(self, '_loading_workflow', False):
            return
        # Prevent recursive calls
        if getattr(self, '_regenerating', False):
            return
        if not self._code_editor:
            return

        self._regenerating = True
        try:
            self._regenerate_code_impl()
        except Exception as e:
            # On any error, show a safe fallback code
            log_warning(f"Code generation error: {e}")
            fallback_code = [
                "#!/usr/bin/env python3",
                "# -*- coding: utf-8 -*-",
                '"""',
                "Auto-generated workflow code",
                "Generated by UnitPort - Celebrimbor",
                '"""',
                "",
                "def execute_workflow(robot=None):",
                "    '''Execute the visual workflow'''",
                "    pass  # Workflow incomplete or error during generation",
                "",
                "if __name__ == '__main__':",
                "    robot = None",
                "    execute_workflow(robot)",
            ]
            try:
                self._code_editor.set_code("\n".join(fallback_code))
            except Exception:
                pass
        finally:
            self._regenerating = False

    def export_graph_data(self) -> Dict[str, Any]:
        """
        Export graph data as a Qt-independent dict for the compiler pipeline.
        Syncs all node parameters before export.

        Returns:
            Dict with nodes and connections suitable for CanvasToIR.
        """
        items_snapshot = list(self.items())
        for item in items_snapshot:
            if not isValid(item):
                continue
            if item.data(10) == "node":
                self._sync_node_parameters(item)

        return self.serialize_workflow()

    def _show_diagnostics(self, diags):
        """Show diagnostics from the compiler pipeline."""
        for diag in diags:
            msg = str(diag)
            if diag.level.value == "error":
                log_error(msg)
            elif diag.level.value == "warn":
                log_warning(msg)
            else:
                log_debug(msg)

    def _regenerate_code_impl(self):
        """Internal implementation of code regeneration using the compiler IR pipeline."""
        try:
            from compiler.lowering.canvas_to_ir import CanvasToIR
            from compiler.codegen.ir_to_code import IRToCode
            from compiler.semantic.validator import SemanticValidator

            graph_data = self.export_graph_data()
            converter = CanvasToIR()
            ir, convert_diags = converter.convert(graph_data, self._robot_type)

            validator = SemanticValidator()
            validate_diags = validator.validate(ir)

            generator = IRToCode()
            code, gen_diags, source_map = generator.generate(ir)

            all_diags = convert_diags + validate_diags + gen_diags
            self._show_diagnostics(all_diags)

            self._code_editor.set_code(code)
            return
        except Exception as e:
            log_warning(f"IR pipeline error, falling back to legacy codegen: {e}")

        # Legacy fallback
        self._regenerate_code_impl_legacy()

    def _regenerate_code_impl_legacy(self):
        """Legacy code regeneration (pre-IR pipeline)."""
        # Create snapshot and sync all node parameters before generating code
        items_snapshot = list(self.items())
        for item in items_snapshot:
            if not isValid(item):
                continue
            if item.data(10) == "node":
                self._sync_node_parameters(item)

        code_lines = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            '"""',
            "Auto-generated workflow code",
            "Generated by UnitPort - Celebrimbor",
            '"""',
            "",
        ]

        # Build connection graph
        graph = self._build_connection_graph()

        if not graph['nodes']:
            code_lines.extend([
                "def execute_workflow(robot=None):",
                "    '''Execute the visual workflow'''",
                "    pass  # No nodes in workflow",
                "",
            ])
        else:
            code_lines.extend([
                "def execute_workflow(robot=None):",
                "    '''Execute the visual workflow'''",
                "",
            ])

            # Find entry points and generate code
            entry_nodes = self._find_entry_nodes(graph)
            generated = set()

            # First generate Condition nodes that provide data (not in control flow)
            for node_id, item in graph['nodes'].items():
                node_name = item.data(11)
                if "Condition" in node_name:
                    # Check if this feeds into a Logic Control node
                    outgoing = graph['outgoing'].get(node_id, {})
                    result_targets = outgoing.get('result', [])
                    for target_id, target_port in result_targets:
                        if target_port == 'condition':
                            # This is a data provider, generate it first
                            code_lines.extend(self._generate_node_code(node_id, graph, 1, generated))
                            code_lines.append("")
                            break

            # Generate code starting from entry nodes
            for entry_id in entry_nodes:
                if entry_id not in generated:
                    node_code = self._generate_node_code(entry_id, graph, 1, generated)
                    code_lines.extend(node_code)
                    if node_code:
                        code_lines.append("")

            # Check if any code was generated
            if len(code_lines) <= 8:  # Only header
                code_lines.append("    pass  # No connected workflow")

        code_lines.extend([
            "",
            "if __name__ == '__main__':",
            "    # Initialize robot (simulation or real)",
            "    # from models import get_robot_model",
            "    # robot = get_robot_model('go2')",
            "    robot = None  # Replace with actual robot instance",
            "    execute_workflow(robot)",
        ])

        # Use set_code method
        self._code_editor.set_code("\n".join(code_lines))

    def serialize_workflow(self) -> Dict[str, Any]:
        """
        Serialize the current workflow state to a JSON-compatible dict.
        Used for saving, regression baselines, and round-trip testing.

        Returns:
            Dict with nodes, connections, and metadata.
        """
        items_snapshot = list(self.items())

        nodes = []
        for item in items_snapshot:
            if not isValid(item):
                continue
            if item.data(10) != "node":
                continue

            node_id = item.data(12)
            if node_id is None:
                continue

            name = item.data(11) or ""
            logic_node = self._logic_nodes.get(node_id)

            # Position
            pos = item.pos()
            node_entry = {
                "id": node_id,
                "display_name": name,
                "position": {"x": round(pos.x(), 1), "y": round(pos.y(), 1)},
                "width": round(item.rect().width(), 1),
                "height": round(item.rect().height(), 1),
                "node_type": logic_node.node_type if logic_node else "unknown",
            }

            # Combo selection
            combo = getattr(item, '_combo', None)
            if combo:
                node_entry["ui_selection"] = combo.currentText()

            # Condition input
            cond_input = getattr(item, '_condition_input', None)
            if cond_input:
                node_entry["condition_expr"] = cond_input.text()

            # Loop type
            loop_type_combo = getattr(item, '_loop_type_combo', None)
            if loop_type_combo:
                node_entry["loop_type"] = loop_type_combo.currentText()

            # For loop params
            for_start = getattr(item, '_for_start_input', None)
            for_end = getattr(item, '_for_end_input', None)
            for_step = getattr(item, '_for_step_input', None)
            if for_start:
                node_entry["for_start"] = for_start.text() or "0"
            if for_end:
                node_entry["for_end"] = for_end.text() or "10"
            if for_step:
                node_entry["for_step"] = for_step.text() or "1"

            # Condition node inputs
            left_input = getattr(item, '_left_input', None)
            right_input = getattr(item, '_right_input', None)
            if left_input:
                node_entry["left_value"] = left_input.text()
            if right_input:
                node_entry["right_value"] = right_input.text()

            # Timer duration
            duration_input = getattr(item, '_duration_input', None)
            if duration_input:
                node_entry["duration"] = duration_input.text()

            # Elif conditions
            elif_inputs = getattr(item, '_elif_inputs', None)
            if elif_inputs:
                node_entry["elif_conditions"] = [inp.text() for inp in elif_inputs]

            # Features (available combo items)
            if combo:
                node_entry["features"] = [combo.itemText(i) for i in range(combo.count())]

            nodes.append(node_entry)

        # Connections
        connections = []
        for item in items_snapshot:
            if not isValid(item):
                continue
            if not isinstance(item, ConnectionItem):
                continue
            if not item.out_port or not item.in_port:
                continue
            if not isValid(item.out_port) or not isValid(item.in_port):
                continue

            out_node = item.out_port.parentItem()
            in_node = item.in_port.parentItem()
            if not out_node or not in_node:
                continue

            from_id = out_node.data(12)
            to_id = in_node.data(12)
            if from_id is None or to_id is None:
                continue

            connections.append({
                "from_node": from_id,
                "from_port": item.out_port.data(3),
                "to_node": to_id,
                "to_port": item.in_port.data(3),
            })

        return {
            "version": "1.0",
            "robot_type": self._robot_type,
            "nodes": nodes,
            "connections": connections,
        }

    def load_workflow(self, data: Dict[str, Any]):
        """
        Load a workflow from a serialized dict (as produced by serialize_workflow).
        Clears the current scene and recreates all nodes and connections.

        Args:
            data: Workflow dict with nodes and connections.
        """
        # Suppress regenerate_code during batch loading
        self._loading_workflow = True
        try:
            self._load_workflow_impl(data)
        finally:
            self._loading_workflow = False

        # Single regenerate at the end
        self.regenerate_code()
        log_info(f"Workflow loaded: {len(data.get('nodes', []))} nodes, "
                 f"{len(data.get('connections', []))} connections")

    def _load_workflow_impl(self, data: Dict[str, Any]):
        """Internal implementation of workflow loading (signals suppressed)."""
        # Clear existing nodes
        self.clear_all_nodes()

        # Set robot type
        if "robot_type" in data:
            self._robot_type = data["robot_type"]

        # Map old node IDs to new graphics items for connection wiring
        id_to_item: Dict[int, Any] = {}

        for node_data in data.get("nodes", []):
            old_id = node_data["id"]
            node_type = str(node_data.get("node_type", "") or "").lower()
            name = node_data.get("display_name", "")
            pos_data = node_data.get("position", {})
            x = pos_data.get("x", 100)
            y = pos_data.get("y", 100)
            w = node_data.get("width", 180)
            h = node_data.get("height", 110)
            features = node_data.get("features", [])
            ui_selection = node_data.get("ui_selection")

            # Prefer explicit node_type when restoring from compiler output.
            # This avoids ambiguity between logic nodes that share display names.
            if node_type == "if":
                name = "Logic Control"
                features = features or ["If", "While Loop"]
                ui_selection = ui_selection or "If"
            elif node_type == "while_loop":
                name = "Logic Control"
                features = features or ["If", "While Loop"]
                ui_selection = ui_selection or "While Loop"

            pos = QPointF(x + w / 2, y + h / 2)
            rect = self.create_node(name, pos, features)

            # Block signals on all child widgets to prevent triggering
            # regenerate_code during batch restoration
            self._set_node_widgets_silent(rect, True)

            try:
                # Set combo selection
                combo = getattr(rect, '_combo', None)
                if ui_selection and combo:
                    combo.setCurrentText(ui_selection)

                # Set condition expression
                cond_expr = node_data.get("condition_expr")
                cond_input = getattr(rect, '_condition_input', None)
                if cond_expr and cond_input:
                    cond_input.setText(cond_expr)

                # Restore elif branches
                elif_conditions = node_data.get("elif_conditions", [])
                add_elif_fn = getattr(rect, '_add_elif', None)
                if elif_conditions and add_elif_fn:
                    for elif_cond in elif_conditions:
                        add_elif_fn()
                    # Set the text for each elif input
                    elif_inputs = getattr(rect, '_elif_inputs', [])
                    for i, cond_text in enumerate(elif_conditions):
                        if i < len(elif_inputs):
                            elif_inputs[i].setText(cond_text)

                # Set loop type
                loop_type = node_data.get("loop_type")
                loop_combo = getattr(rect, '_loop_type_combo', None)
                if loop_type and loop_combo:
                    loop_combo.setCurrentText(loop_type)

                # Set for loop params
                for field_name, attr_name in [
                    ("for_start", "_for_start_input"),
                    ("for_end", "_for_end_input"),
                    ("for_step", "_for_step_input"),
                ]:
                    val = node_data.get(field_name)
                    widget = getattr(rect, attr_name, None)
                    if val is not None and widget:
                        widget.setText(str(val))

                # Set condition node inputs
                for field_name, attr_name in [
                    ("left_value", "_left_input"),
                    ("right_value", "_right_input"),
                ]:
                    val = node_data.get(field_name)
                    widget = getattr(rect, attr_name, None)
                    if val is not None and widget:
                        widget.setText(str(val))

                # Set timer duration
                duration = node_data.get("duration")
                duration_widget = getattr(rect, '_duration_input', None)
                if duration is not None and duration_widget:
                    duration_widget.setText(str(duration))
            finally:
                # Restore signals
                self._set_node_widgets_silent(rect, False)

            # Sync params once (without regenerating code)
            self._update_node_params(rect)

            id_to_item[old_id] = rect

        # Build port lookup: (node_item, port_name) -> port_item
        def _find_port(node_item, port_name, io_direction):
            if not node_item or not isValid(node_item):
                return None
            for child in node_item.childItems():
                if (child.data(0) == "port"
                        and child.data(1) == io_direction
                        and child.data(3) == port_name):
                    return child
            return None

        # Create connections
        for conn_data in data.get("connections", []):
            from_id = conn_data.get("from_node")
            to_id = conn_data.get("to_node")
            from_port_name = conn_data.get("from_port", "flow_out")
            to_port_name = conn_data.get("to_port", "flow_in")

            from_item = id_to_item.get(from_id)
            to_item = id_to_item.get(to_id)
            if not from_item or not to_item:
                continue

            out_port = _find_port(from_item, from_port_name, "out")
            in_port = _find_port(to_item, to_port_name, "in")
            if not out_port or not in_port:
                continue

            self._create_connection(out_port, in_port)

    @staticmethod
    def _set_node_widgets_silent(node_item, block: bool):
        """Block or unblock signals on all QWidget children of a node item."""
        from PySide6.QtWidgets import QGraphicsProxyWidget
        if not node_item or not isValid(node_item):
            return
        for child in node_item.childItems():
            if not isValid(child):
                continue
            if isinstance(child, QGraphicsProxyWidget):
                w = child.widget()
                if w:
                    _block_recursive(w, block)

    def _center_view_on_content(self):
        """Center the graph view on the content bounding rect."""
        items = [item for item in self.items()
                 if isValid(item) and item.data(10) == "node"]
        if not items:
            return
        # Compute bounding rect of all nodes
        from PySide6.QtCore import QRectF
        rect = QRectF()
        for item in items:
            rect = rect.united(item.sceneBoundingRect())
        # Add padding
        rect.adjust(-50, -50, 50, 50)
        # Find the view and fit
        for view in self.views():
            view.fitInView(rect, Qt.KeepAspectRatio)
            break

    def clear_all_nodes(self):
        """Clear all nodes and connections from the scene."""
        # Remove all items
        items_to_remove = []
        for item in self.items():
            if not isValid(item):
                continue
            if isinstance(item, ConnectionItem) or item.data(10) == "node":
                items_to_remove.append(item)

        for item in items_to_remove:
            if isValid(item) and item.scene() is not None:
                self.removeItem(item)

        self._logic_nodes.clear()
        self._node_seq = 0
        log_info("All nodes cleared")

    def get_workflow_data(self) -> Dict[str, Any]:
        """Get workflow data for execution"""
        ordered_nodes = self._build_workflow_order()
        workflow = {
            'nodes': [],
            'connections': [],
            'execution_order': []
        }

        for item in ordered_nodes:
            # Skip invalid items
            if not isValid(item):
                continue

            node_id = item.data(12)
            if node_id is None:
                continue

            node_name = item.data(11)
            logic_node = self._logic_nodes.get(node_id)

            node_data = {
                'id': node_id,
                'name': node_name,
                'type': logic_node.node_type if logic_node else 'unknown',
                'logic_node': logic_node,
                'parameters': logic_node.parameters.copy() if logic_node else {}
            }

            # Add UI-specific parameters
            combo = getattr(item, '_combo', None)
            if combo:
                node_data['ui_selection'] = combo.currentText()

            workflow['nodes'].append(node_data)
            workflow['execution_order'].append(node_id)

        # Create snapshot for connections to avoid modification during iteration
        items_snapshot = list(self.items())

        # Collect connections
        for item in items_snapshot:
            if not isValid(item):
                continue
            if isinstance(item, ConnectionItem):
                # Skip incomplete connections
                if not item.out_port or not item.in_port:
                    continue
                if not isValid(item.out_port) or not isValid(item.in_port):
                    continue

                out_node = item.out_port.parentItem()
                in_node = item.in_port.parentItem()

                if not out_node or not in_node:
                    continue

                from_node_id = out_node.data(12)
                to_node_id = in_node.data(12)

                if from_node_id is None or to_node_id is None:
                    continue

                workflow['connections'].append({
                    'from_node': from_node_id,
                    'from_port': item.out_port.data(3),
                    'to_node': to_node_id,
                    'to_port': item.in_port.data(3)
                })

        return workflow

    def get_execution_graph(self) -> Dict[str, Any]:
        """
        Get execution graph for workflow execution with control flow support.

        Returns a graph structure that supports conditional branching and loops.
        """
        graph = self._build_connection_graph()

        # Build node data map
        node_data_map = {}
        for node_id, item in graph['nodes'].items():
            if not item or not isValid(item):
                continue

            node_name = item.data(11)
            logic_node = self._logic_nodes.get(node_id)

            node_data = {
                'id': node_id,
                'name': node_name or '',
                'item': item,
                'type': logic_node.node_type if logic_node else 'unknown',
                'logic_node': logic_node,
                'parameters': logic_node.parameters.copy() if logic_node else {}
            }

            # Add UI-specific parameters
            combo = getattr(item, '_combo', None)
            if combo:
                node_data['ui_selection'] = combo.currentText()

            # Add condition input for Logic Control nodes
            cond_input = getattr(item, '_condition_input', None)
            if cond_input:
                node_data['condition_expr'] = cond_input.text()

            # Add loop parameters
            loop_type_combo = getattr(item, '_loop_type_combo', None)
            if loop_type_combo:
                node_data['loop_type'] = loop_type_combo.currentText().lower()

            for_start = getattr(item, '_for_start_input', None)
            for_end = getattr(item, '_for_end_input', None)
            for_step = getattr(item, '_for_step_input', None)
            if for_start:
                node_data['for_start'] = for_start.text() or '0'
            if for_end:
                node_data['for_end'] = for_end.text() or '10'
            if for_step:
                node_data['for_step'] = for_step.text() or '1'

            # Add Condition node inputs
            left_input = getattr(item, '_left_input', None)
            right_input = getattr(item, '_right_input', None)
            if left_input:
                node_data['left_value'] = left_input.text()
            if right_input:
                node_data['right_value'] = right_input.text()

            node_data_map[node_id] = node_data

        # Find entry nodes
        entry_nodes = self._find_entry_nodes(graph)

        return {
            'nodes': node_data_map,
            'outgoing': graph['outgoing'],
            'incoming': graph['incoming'],
            'entry_nodes': entry_nodes
        }
