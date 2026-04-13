#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Scene
Contains nodes, connections, grid and other elements
"""

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from shiboken6 import isValid

from PySide6.QtCore import Qt, QRectF, QPoint, QPointF, QTimer, QStringListModel, QEvent, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QLinearGradient, QGradient, QPainterPath, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPathItem, QGraphicsProxyWidget, QComboBox, QLineEdit, QWidget, QSlider,
    QHBoxLayout, QVBoxLayout, QPushButton, QMessageBox, QLabel, QToolTip,
    QApplication, QTextEdit, QPlainTextEdit
)

from src.system.core.logger import log_info, log_error, log_debug, log_warning, log_success
from src.system.core.theme_manager import get_color, get_node_color_pair, get_color_slot
from src.system.core.localisation import tr
from src.system.behavior.action_registry import get_semantic_actions
from src.system.models.model_registry import get_model_ids, list_brand_items, normalize_brand_id
from bin.nodes.node_ui_rows import (
    make_tag,
    NodeRowStack,
    NodeWidgetFactory,
    ConditionSet,
    OutputSet,
    InputSetting,
    ComboSetting,
    BehaviorSetting,
    ScriptSetting,
    ScriptInAndOut,
)

# Import node system
from src.system.nodes import create_node as create_logic_node, get_node_class, REGISTERED_NODES

_FLOW_SLOTS = {
    "flow_in", "flow_out", "out_if", "out_else", "loop_body", "loop_end",
    "loop_break_in", "break_out",
    "out_pass", "out_block", "out_timeout", "retry_body", "out_success", "out_failed",
    "try_body", "except_body", "finally_body",
    "case_default", "join_out",
}

def _normalize_data_type(data_type: str) -> str:
    t = str(data_type or "any").strip().lower()
    alias = {
        "boolean": "bool",
        "str": "string",
        "integer": "int",
        "double": "float",
        "callable": "function",
        "fn": "function",
        "void": "flow",
    }
    return alias.get(t, t or "any")


def _sub_dot_type_color(data_type: str) -> QColor:
    t = _normalize_data_type(data_type)
    fallback = {
        "bool": "#22c55e",
        "string": "#f59e0b",
        "int": "#3b82f6",
        "float": "#06b6d4",
        "number": "#0ea5a4",
        "function": "#a855f7",
        "protocol": "#a78bfa",  # violet 鈥?motor-weight protocol payload
        "list": "#f97316",      # orange 鈥?ordered collection
        "dict": "#ec4899",      # pink 鈥?key-value mapping
        "any": "#9ca3af",
    }
    theme_key = f"sub_dot_{t}"
    return QColor(get_color(theme_key, fallback.get(t, fallback["any"])))


class _FanInPortItem(QGraphicsEllipseItem):
    """Port dot with an 'F' label drawn inside, indicating Fan-In capability."""

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#ffffff"), 0))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, "F")


class ConnectionItem(QGraphicsPathItem):
    """Connection Line Item - Supports auto-update and endpoint editing"""

    def __init__(self, out_port, in_port, parent=None):
        super().__init__(parent)

        self.out_port = out_port
        self.in_port = in_port

        # Set style
        self.setPen(ConnectionStyleDelegate.make_pen(self, "normal"))
        self.setZValue(-1)
        self.setData(0, "connection")

        # Keep connection non-selectable so rubber-band box selection does not
        # show selection handles on curved edges between nodes.
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
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
        self._start_marker.setBrush(QBrush(ConnectionStyleDelegate.make_marker_color(self)))
        self._start_marker.setPen(ConnectionStyleDelegate.make_marker_border_pen(self))
        self._start_marker.setZValue(10)
        self._start_marker.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._start_marker.setData(0, "connection_marker")
        self._start_marker.setData(1, "start")
        self._start_marker.setData(2, self)  # Associated connection
        self._start_marker.setAcceptedMouseButtons(Qt.LeftButton)
        self._start_marker.setVisible(False)

        # End marker
        self._end_marker = QGraphicsEllipseItem(-4, -4, 8, 8, self)
        self._end_marker.setBrush(QBrush(ConnectionStyleDelegate.make_marker_color(self)))
        self._end_marker.setPen(ConnectionStyleDelegate.make_marker_border_pen(self))
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
        self.setPen(ConnectionStyleDelegate.make_pen(self, "hover"))
        if self._start_marker:
            self._start_marker.setVisible(True)
        if self._end_marker:
            self._end_marker.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Mouse leave - hide endpoint markers"""
        if not self.isSelected():
            self.setPen(ConnectionStyleDelegate.make_pen(self, "normal"))
        if self._start_marker:
            self._start_marker.setVisible(False)
        if self._end_marker:
            self._end_marker.setVisible(False)
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """Item change event"""
        if change == QGraphicsItem.ItemSelectedHasChanged:
            if value:  # Selected
                self.setPen(ConnectionStyleDelegate.make_pen(self, "hover"))
                if self._start_marker:
                    self._start_marker.setVisible(True)
                if self._end_marker:
                    self._end_marker.setVisible(True)
            else:  # Not selected
                self.setPen(ConnectionStyleDelegate.make_pen(self, "normal"))
                if self._start_marker:
                    self._start_marker.setVisible(False)
                if self._end_marker:
                    self._end_marker.setVisible(False)

        return super().itemChange(change, value)

    def refresh_style(self):
        """Refresh connection colors"""
        if self.isSelected():
            self.setPen(ConnectionStyleDelegate.make_pen(self, "hover"))
        else:
            self.setPen(ConnectionStyleDelegate.make_pen(self, "normal"))
        if self._start_marker:
            self._start_marker.setBrush(QBrush(ConnectionStyleDelegate.make_marker_color(self)))
            self._start_marker.setPen(ConnectionStyleDelegate.make_marker_border_pen(self))
        if self._end_marker:
            self._end_marker.setBrush(QBrush(ConnectionStyleDelegate.make_marker_color(self)))
            self._end_marker.setPen(ConnectionStyleDelegate.make_marker_border_pen(self))


class ConnectionStyleDelegate:
    """Reusable style delegate for connection lines and endpoint markers."""

    _SUB_DOT_ALPHA_NORMAL = 130
    _SUB_DOT_ALPHA_HOVER = 190
    _SUB_DOT_WIDTH_NORMAL = 1.4
    _SUB_DOT_WIDTH_HOVER = 2.0

    @staticmethod
    def _port_dot_kind(port_item) -> str:
        if not port_item or not isValid(port_item):
            return ""
        meta = port_item.data(20) or {}
        return str(meta.get("dot_kind", "")).strip().lower()

    @staticmethod
    def _port_data_type(port_item) -> str:
        if not port_item or not isValid(port_item):
            return "any"
        meta = port_item.data(20) or {}
        return _normalize_data_type(meta.get("data_type", "any"))

    @staticmethod
    def _port_type_color(port_item) -> QColor:
        """Return the color for a port's data type.

        Checks ``data(20)["type_color"]`` first (set by TrainingNodePort for
        training-specific types). Falls back to ``_sub_dot_type_color`` for
        standard Mission Canvas types.
        """
        if not port_item or not isValid(port_item):
            return _sub_dot_type_color("any")
        meta = port_item.data(20) or {}
        hex_c = meta.get("type_color", "")
        if hex_c:
            return QColor(hex_c)
        return _sub_dot_type_color(_normalize_data_type(meta.get("data_type", "any")))

    @classmethod
    def is_sub_dot_link(cls, conn: ConnectionItem) -> bool:
        return (
            cls._port_dot_kind(conn.out_port) == "sub_dot"
            and cls._port_dot_kind(conn.in_port) == "sub_dot"
        )

    @classmethod
    def make_pen(cls, conn: ConnectionItem, state: str = "normal") -> QPen:
        state = (state or "normal").strip().lower()
        is_hover = state == "hover"
        if cls.is_sub_dot_link(conn):
            color = cls._port_type_color(conn.out_port)
            color.setAlpha(cls._SUB_DOT_ALPHA_HOVER if is_hover else cls._SUB_DOT_ALPHA_NORMAL)
            width = cls._SUB_DOT_WIDTH_HOVER if is_hover else cls._SUB_DOT_WIDTH_NORMAL
            return QPen(color, width)
        color_name = get_color("connection_hover", "#3b82f6") if is_hover else get_color("connection", "#60a5fa")
        width = 3.5 if is_hover else 2.5
        return QPen(QColor(color_name), width)

    @classmethod
    def make_marker_color(cls, conn: ConnectionItem) -> QColor:
        if cls.is_sub_dot_link(conn):
            c = cls._port_type_color(conn.out_port)
            c.setAlpha(170)
            return c
        return QColor(get_color("connection", "#60a5fa"))

    @classmethod
    def make_marker_border_pen(cls, conn: ConnectionItem) -> QPen:
        if cls.is_sub_dot_link(conn):
            border = cls._port_type_color(conn.out_port)
            border.setAlpha(210)
            return QPen(border, 1)
        return QPen(QColor(get_color("connection_marker_border", "#ffffff")), 1)


class ResizableNodeItem(QGraphicsRectItem):
    """Node item with border-resize support and content-aware min size constraints."""

    def __init__(self, x, y, w, h, min_size_fn=None, resized_cb=None, parent=None):
        super().__init__(x, y, w, h, parent)
        # Use one shared edge hit zone for both cursor preview and resize start.
        # The band is intentionally wider than before, but corner areas are
        # excluded so resize only starts when the cursor is the horizontal or
        # vertical resize cursor.
        self._edge_margin = 12
        self._corner_exclusion = 18
        self._min_size_fn = min_size_fn
        self._resized_cb = resized_cb
        self._hover_passthrough_items = []
        self._hover_resize_mode = None
        self._resize_mode = None
        self._resize_start_scene = None
        self._resize_start_rect = None
        self._resize_start_pos = None
        self.setAcceptHoverEvents(True)

    def set_resize_callbacks(self, min_size_fn=None, resized_cb=None):
        self._min_size_fn = min_size_fn
        self._resized_cb = resized_cb

    def register_hover_passthrough(self, item):
        if item is None:
            return
        if item not in self._hover_passthrough_items:
            self._hover_passthrough_items.append(item)
            item.setAcceptHoverEvents(True)
            item.installSceneEventFilter(self)

    def _edge_hit(self, pos):
        r = self.rect()
        if r.isNull():
            return None
        x = pos.x()
        y = pos.y()
        left = abs(x - r.left()) <= self._edge_margin
        right = abs(x - r.right()) <= self._edge_margin
        top = abs(y - r.top()) <= self._edge_margin
        bottom = abs(y - r.bottom()) <= self._edge_margin
        within_vertical_span = (r.top() + self._corner_exclusion) <= y <= (r.bottom() - self._corner_exclusion)
        within_horizontal_span = (r.left() + self._corner_exclusion) <= x <= (r.right() - self._corner_exclusion)
        if left and within_vertical_span:
            return "left"
        if right and within_vertical_span:
            return "right"
        if top and within_horizontal_span:
            return "top"
        if bottom and within_horizontal_span:
            return "bottom"
        if not (left or right or top or bottom):
            return None
        return None

    @staticmethod
    def _cursor_for_mode(mode):
        if mode in ("left", "right"):
            return Qt.SizeHorCursor
        if mode in ("top", "bottom"):
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def _set_hover_resize_mode(self, mode):
        self._hover_resize_mode = mode
        cursor = self._cursor_for_mode(mode)
        self.setCursor(cursor)
        for item in list(self._hover_passthrough_items):
            if item is not None and isValid(item):
                item.setCursor(cursor)

    def _begin_resize(self, mode, scene_pos):
        self._resize_mode = mode
        self._resize_start_scene = scene_pos
        self._resize_start_rect = QRectF(self.rect())
        self._resize_start_pos = QPointF(self.pos())
        self.setCursor(self._cursor_for_mode(mode))

    def _event_pos_in_self(self, watched, event):
        pos = getattr(event, "pos", None)
        if callable(pos):
            try:
                return self.mapFromItem(watched, event.pos())
            except Exception:
                return None
        scene_pos = getattr(event, "scenePos", None)
        if callable(scene_pos):
            try:
                return self.mapFromScene(event.scenePos())
            except Exception:
                return None
        return None

    def _event_cursor_matches_mode(self, event, mode):
        widget = event.widget() if hasattr(event, "widget") else None
        if widget is None:
            return False
        return widget.cursor().shape() == self._cursor_for_mode(mode)

    def sceneEventFilter(self, watched, event):
        event_type = event.type()
        if event_type in (QEvent.GraphicsSceneHoverMove, QEvent.GraphicsSceneHoverEnter):
            local_pos = self._event_pos_in_self(watched, event)
            self._set_hover_resize_mode(self._edge_hit(local_pos) if local_pos is not None else None)
        elif event_type == QEvent.GraphicsSceneHoverLeave and not self._resize_mode:
            self._set_hover_resize_mode(None)
        elif event_type == QEvent.GraphicsSceneMousePress and event.button() == Qt.LeftButton:
            local_pos = self._event_pos_in_self(watched, event)
            mode = self._edge_hit(local_pos) if local_pos is not None else None
            if mode:
                self._set_hover_resize_mode(mode)
        return super().sceneEventFilter(watched, event)

    def hoverMoveEvent(self, event):
        mode = self._edge_hit(event.pos())
        self._set_hover_resize_mode(mode)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self._set_hover_resize_mode(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            mode = self._edge_hit(event.pos())
            if mode and self._hover_resize_mode == mode and self._event_cursor_matches_mode(event, mode):
                self._begin_resize(mode, event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._resize_mode:
            super().mouseMoveEvent(event)
            return
        if not self._resize_start_scene or not self._resize_start_rect or self._resize_start_pos is None:
            super().mouseMoveEvent(event)
            return

        min_w, min_h = 100.0, 80.0
        if callable(self._min_size_fn):
            try:
                mw, mh = self._min_size_fn()
                min_w = float(max(min_w, mw))
                min_h = float(max(min_h, mh))
            except Exception:
                pass

        delta = event.scenePos() - self._resize_start_scene
        dx = float(delta.x())
        dy = float(delta.y())
        start_rect = self._resize_start_rect
        start_pos = self._resize_start_pos

        new_w = start_rect.width()
        new_h = start_rect.height()
        new_x = start_pos.x()
        new_y = start_pos.y()

        mode = self._resize_mode
        self.setCursor(self._cursor_for_mode(mode))
        if "right" in mode:
            new_w = max(min_w, start_rect.width() + dx)
        if "left" in mode:
            new_w = max(min_w, start_rect.width() - dx)
            new_x = start_pos.x() + (start_rect.width() - new_w)
        if "bottom" in mode:
            new_h = max(min_h, start_rect.height() + dy)
        if "top" in mode:
            new_h = max(min_h, start_rect.height() - dy)
            new_y = start_pos.y() + (start_rect.height() - new_h)

        self.setPos(QPointF(new_x, new_y))
        self.setRect(0, 0, new_w, new_h)
        if callable(self._resized_cb):
            self._resized_cb()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._resize_mode:
            self._resize_mode = None
            self._resize_start_scene = None
            self._resize_start_rect = None
            self._resize_start_pos = None
            self._set_hover_resize_mode(None)
            if callable(self._resized_cb):
                self._resized_cb()
            event.accept()
            return
        super().mouseReleaseEvent(event)


def _block_recursive(widget, block: bool):
    """Recursively block/unblock signals on a widget and all its children."""
    if widget is None:
        return
    widget.blockSignals(block)
    for child in widget.findChildren(QWidget):
        child.blockSignals(block)


class GroupContainerItem(QGraphicsRectItem):
    """Visual group container for a set of mission nodes.

    Design notes
    ------------
    - Members are kept as independent scene items (no Qt parent鈥揷hild nesting)
      so that port connections are never invalidated.
    - Movement is propagated to members via itemChange(ItemPositionHasChanged).
    - A ``_prev_pos`` guard avoids re-entrant delta-moves during the FIRST
      setPos call (container creation), where members should NOT be displaced.
    - ``data(10) == "group"`` marks the item type; ``data(12)`` carries
      group_id for quick lookup.
    """

    # Padding around member nodes when auto-fitting the container rect.
    _FIT_PADDING = 20

    def __init__(self, group_id: int, title: str = "", parent=None):
        super().__init__(parent)
        self._group_id: int = group_id
        self._title: str = title or f"Group {group_id}"
        self._member_ids: set = set()   # node_id ints
        self._prev_pos = None           # used for delta-move propagation
        self._syncing: bool = False     # re-entrancy guard

        self.setData(10, "group")
        self.setData(12, group_id)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(-2)              # behind nodes, above canvas bg
        self.setAcceptHoverEvents(True)

    # ------------------------------------------------------------------
    # Visual
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None):
        r = self.rect()
        if r.isNull():
            return

        # Semi-transparent fill
        painter.setBrush(QBrush(QColor(60, 80, 140, 45)))
        border_pen = QPen(QColor(130, 160, 230, 180), 1.5, Qt.DashLine)
        painter.setPen(border_pen)
        painter.drawRect(r)

        # Title text
        title_pen = QPen(QColor(180, 200, 255, 200))
        painter.setPen(title_pen)
        font = QFont("Arial", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(r.adjusted(6, 4, -6, 0).toRect(), Qt.AlignTop | Qt.AlignLeft,
                         self._title)

    def boundingRect(self) -> QRectF:
        return self.rect()

    # ------------------------------------------------------------------
    # Movement propagation
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and not self._syncing:
            if self._prev_pos is not None:
                delta = self.pos() - self._prev_pos
                if delta.manhattanLength() > 0:
                    self._syncing = True
                    try:
                        scene = self.scene()
                        if scene is not None:
                            for item in scene.items():
                                if (isValid(item)
                                        and item.data(10) == "node"
                                        and getattr(item, '_group_id', None) == self._group_id):
                                    item.setPos(item.pos() + delta)
                    finally:
                        self._syncing = False
            self._prev_pos = QPointF(self.pos())
        return super().itemChange(change, value)


class GraphScene(QGraphicsScene):
    robot_type_changed = Signal(str)
    workspace_open_requested = Signal(str)  # policy_id — emitted by Checkpoint node "Open Training Ground"

    # Emitted whenever the dirty state flips. bool = True when the canvas has
    # unsaved changes relative to the last mark_clean() / load. Wired by the
    # main window into ``files_row.mark_dirty(file_id, ...)``.
    content_changed = Signal(bool)

    """Graph Editor Scene"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene configuration
        self.setSceneRect(-2500, -2500, 5000, 5000)
        self.grid_small = 20
        self.grid_big = self.grid_small * 5
        self._initial_origin = QPointF(0, 0)

        # Color configuration
        self._apply_theme()

        # Nodes and connections
        self._node_seq = 0
        self._temp_connection = None
        self._temp_start_port = None
        self._preview_port = None
        self._snap_target_port = None
        self._hover_port = None
        self._sub_dot_tooltip_port = None
        self._sub_dot_tooltip_scene_pos = QPointF()
        self._sub_dot_tooltip_visible = False
        self._sub_dot_tooltip_delay_ms = 500
        self._sub_dot_tooltip_timer = QTimer(self)
        self._sub_dot_tooltip_timer.setSingleShot(True)
        self._sub_dot_tooltip_timer.timeout.connect(self._show_sub_dot_tooltip)

        # Reconnection state
        self._reconnecting = False
        self._reconnect_connection = None
        self._reconnect_end = None  # "start" or "end"
        self._reconnect_original_port = None  # Save original port for cancel
        self._autowiring_break = False
        # Shared interaction radius for hover/snap/cursor near-port checks.
        self._port_interaction_radius = 11

        # Action mapping (UI action name -> robot action)
        self._action_mapping = {
            "Lift Right Leg": "lift_right_leg",
            "Lift Left Leg": "lift_left_leg",
            "Stand": "stand",
            "Sit": "sit",
            "Walk": "walk",
            "Stop": "stop"
        }

        # Node display name -> logic node type mapping
        self._node_type_mapping = {
            "Start": "start",
            "End": "end",
            "Sensor": "sensor_input",
            "Trigger": "wait",
            "Action Execution": "action_execution",
            "Sensor Input": "sensor_input",
            "Logic Control": "if",
            "Loop": "while_loop",
            "Condition": "comparison",
            "Math": "math",
            "Timer": "timer",
            "Wait": "wait",
            "Gate": "gate",
            "Timeout": "timeout",
            "Retry": "retry",
            "Cancel": "cancel",
            "Abort": "abort",
            "Break": "break",
            "Switch": "switch",
            "Try": "try_catch",
            "Parallel": "parallel",
            "Join": "join",
            "Script Box": "opaque_code",
            "OnEvent": "opaque_code",
            "PublishEvent": "opaque_code",
            "SetState": "opaque_code",
            "CheckState": "opaque_code",
            "HumanConfirm": "opaque_code",
            # Control pipeline nodes
            "Checkpoint": "checkpoint",
            "Behavior":   "behavior",
            "Reactive Loco": "reactive_loco",
            # Training nodes (Env Config / Train Config / Train / Task Config /
            # Eval Config) are intentionally absent from this map.
            # The Mission Canvas must never create training nodes.
            # The Training Ground canvas manages its own separate type map.
        }
        self._port_meta_key = 20

        # Store logic node instances (node_id -> BaseNode instance)
        self._logic_nodes: Dict[int, Any] = {}

        # Node groups (group_id -> GroupContainerItem)
        self._groups: Dict[int, Any] = {}
        self._group_seq: int = 0

        # References
        self._code_editor = None
        self._simulation_thread = None
        self._subgraph_opener: Optional[Callable[..., None]] = None
        self._script_tab_closer: Optional[Callable[[int], None]] = None
        self._script_tab_renamer: Optional[Callable[[int, str], None]] = None
        self._behavior_param_provider: Optional[Callable[..., List[dict]]] = None
        self._behavior_param_writer: Optional[Callable[..., bool]] = None
        self._robot_brand = "unitree"
        self._robot_type = "go2"
        self._start_field_id = "flat_ground"   # v1.4: StartNode field selector
        self._start_render_enabled = True       # v1.5: StartNode viewer toggle
        self._event_symbol_set: Set[str] = set()
        self._script_symbol_set: Set[str] = set()
        self._policy_symbol_set: Set[str] = set()   # policy_id values from CheckpointRegistry
        self._event_symbol_model = QStringListModel(self)
        self._script_symbol_model = QStringListModel(self)
        self._policy_symbol_model = QStringListModel(self)
        self._completer_popup_style = ""
        self._symbol_combos: List[QComboBox] = []
        self._symbol_line_to_combo: Dict[QLineEdit, QComboBox] = {}
        self._policy_select_combos: List[QComboBox] = []  # non-editable Checkpoint pickers

        # Timer - for updating connections
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_all_connections)
        self._update_timer.start(16)  # 60fps

        # Drag-end detection for the move → dirty pipeline. Populated on
        # mousePressEvent with {id(item): (item, QPointF(pre_drag_pos))}.
        self._drag_start_positions: Dict[int, Any] = {}

        # ── History / dirty-state machinery ─────────────────────────────
        # Snapshot-based undo stack. ``self._history`` is a list of snapshot
        # dicts; ``self._history_index`` points at the state the scene
        # currently matches. On mutation we truncate forward history, append
        # the new snapshot, and compare the index against ``_clean_index``
        # to decide the dirty flag.
        #
        # Mutations are routed through ``self._touch_content()``; programmatic
        # bulk loads must wrap themselves in ``with self.suppress_history():``
        # and then call ``self.reset_history(...)`` so the post-load state
        # becomes the new clean baseline.
        self._history: List[Dict[str, Any]] = []
        self._history_index: int = -1
        self._clean_index: int = -1
        self._history_max: int = 100
        self._history_suppress_depth: int = 0
        self._last_emitted_dirty: bool = False

        # Place default Start/End nodes (must happen BEFORE first snapshot
        # so the initial history entry already contains the template nodes).
        self._place_initial_nodes()

        # Seed history with the empty template; subclasses that override
        # _snapshot_for_history() must be ready to serialize at this point.
        try:
            initial = self._snapshot_for_history()
        except Exception:
            initial = {}
        self._history = [initial] if initial else []
        self._history_index = 0 if self._history else -1
        self._clean_index = self._history_index

        log_debug("GraphScene initialized")

    # ------------------------------------------------------------------
    # History / dirty-state — public + protected API
    # ------------------------------------------------------------------

    def _snapshot_for_history(self) -> Dict[str, Any]:
        """Return a full serialized snapshot of the canvas for the undo stack.

        Subclasses should override this to delegate to the most complete
        snapshot method they have (e.g. the Mission canvas uses
        ``serialize_workflow`` and the Training canvas overrides this to
        return ``serialize_training_graph(for_compiler=False)``).

        The default implementation prefers ``serialize_workflow`` when
        available so plain Mission scenes work without additional wiring.
        """
        fn = getattr(self, "serialize_workflow", None)
        if callable(fn):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"_snapshot_for_history: serialize_workflow failed — {exc}")
                return {}
        return {}

    def _restore_from_history(self, snapshot: Dict[str, Any]) -> None:
        """Restore the canvas from a snapshot produced by ``_snapshot_for_history``.

        Subclasses override when they use a different serializer. The load
        path is always wrapped in ``suppress_history`` by ``undo()`` /
        ``redo()`` so the restore itself cannot re-enter the history stack.
        """
        fn = getattr(self, "load_workflow", None)
        if callable(fn):
            try:
                fn(snapshot)
            except Exception as exc:  # noqa: BLE001
                log_error(f"_restore_from_history: load_workflow failed — {exc}")

    @contextmanager
    def suppress_history(self):
        """Context manager that disables history pushes for its body.

        Re-entrant via a depth counter. Use this around programmatic loads,
        preset applications, backend layer restores, and any other bulk
        mutation that must not generate undo entries.
        """
        self._history_suppress_depth += 1
        try:
            yield
        finally:
            self._history_suppress_depth = max(0, self._history_suppress_depth - 1)

    def _touch_content(self) -> None:
        """Record a history entry for a fresh mutation and flip dirty.

        Call after a user-initiated change (node create/delete/move/param,
        connection create/delete, paste, group/ungroup). A no-op while
        ``suppress_history`` is active. Consecutive identical snapshots are
        deduplicated so that a single logical edit that fans out into
        multiple internal hooks (e.g. create_node → regenerate_code → touch,
        then explicit touch at the callsite → would otherwise push twice)
        only produces one undo entry.
        """
        if self._history_suppress_depth > 0:
            return
        try:
            snap = self._snapshot_for_history()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"_touch_content: snapshot failed — {exc}")
            return
        if not snap:
            return

        # Dedup: if the fresh snapshot matches the one at the current index,
        # this touch is a no-op (same logical state as before). Cheap equality
        # check on dict is O(n) in the graph size but avoids an undo-stack
        # polluted with duplicate states.
        if 0 <= self._history_index < len(self._history):
            try:
                if self._history[self._history_index] == snap:
                    return
            except Exception:
                pass

        # Truncate any forward (redo) history — new branching work discards
        # the old redo path.
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
            # If the clean index was in the discarded tail, the file can
            # never return to a non-dirty state by undo alone.
            if self._clean_index > self._history_index:
                self._clean_index = -1

        self._history.append(snap)
        self._history_index = len(self._history) - 1

        # Cap depth to avoid unbounded memory growth on long editing sessions.
        if len(self._history) > self._history_max:
            drop = len(self._history) - self._history_max
            self._history = self._history[drop:]
            self._history_index -= drop
            if self._clean_index >= 0:
                self._clean_index -= drop
                if self._clean_index < 0:
                    # The clean baseline fell off the back of the stack; from
                    # here we can no longer tell when we'd be "clean again".
                    self._clean_index = -1

        self._emit_dirty_if_changed()

    def can_undo(self) -> bool:
        return self._history_index > 0

    def can_redo(self) -> bool:
        return 0 <= self._history_index < len(self._history) - 1

    def undo(self) -> bool:
        """Step one state backwards. Returns True on success."""
        if not self.can_undo():
            return False
        self._history_index -= 1
        snap = self._history[self._history_index]
        with self.suppress_history():
            self._restore_from_history(snap)
        self._emit_dirty_if_changed()
        return True

    def redo(self) -> bool:
        """Step one state forwards. Returns True on success."""
        if not self.can_redo():
            return False
        self._history_index += 1
        snap = self._history[self._history_index]
        with self.suppress_history():
            self._restore_from_history(snap)
        self._emit_dirty_if_changed()
        return True

    def is_dirty(self) -> bool:
        if self._clean_index < 0:
            return True  # baseline lost (e.g. stack overflowed past it)
        return self._history_index != self._clean_index

    def mark_clean(self) -> None:
        """Pin the current state as the new clean baseline (post-save)."""
        self._clean_index = self._history_index
        self._emit_dirty_if_changed()

    def reset_history(self, initial_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Wipe the undo stack and seed it with a fresh clean baseline.

        Called after programmatic loads (open file, reset template, backend
        layer restore) so the history does not carry state from whatever
        canvas content was showing before the load.
        """
        if initial_snapshot is None:
            try:
                initial_snapshot = self._snapshot_for_history()
            except Exception:
                initial_snapshot = {}
        self._history = [initial_snapshot] if initial_snapshot else []
        self._history_index = 0 if self._history else -1
        self._clean_index = self._history_index
        self._emit_dirty_if_changed()

    def export_history_state(self) -> Dict[str, Any]:
        """Serialize the undo stack so it can be stashed per-file/per-layer."""
        return {
            "history": list(self._history),
            "history_index": self._history_index,
            "clean_index": self._clean_index,
        }

    def import_history_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore a previously exported history state."""
        if not state:
            self.reset_history()
            return
        self._history = list(state.get("history") or [])
        self._history_index = int(state.get("history_index", -1))
        self._clean_index = int(state.get("clean_index", -1))
        # Clamp in case the serialized state is stale.
        if self._history_index >= len(self._history):
            self._history_index = len(self._history) - 1
        if self._clean_index >= len(self._history):
            self._clean_index = -1
        self._emit_dirty_if_changed()

    def _emit_dirty_if_changed(self) -> None:
        dirty = self.is_dirty()
        if dirty != self._last_emitted_dirty:
            self._last_emitted_dirty = dirty
            try:
                self.content_changed.emit(dirty)
            except Exception:
                pass

    def set_code_editor(self, editor):
        """Set code editor reference"""
        self._code_editor = editor

    def set_subgraph_opener(self, opener: Optional[Callable[..., None]]):
        """Set callback for opening nested editors (Behavior / Script / Settings)."""
        self._subgraph_opener = opener

    def set_script_tab_closer(self, closer: Optional[Callable[[int], None]]):
        """Set callback invoked with node_id when a Script Box node is deleted."""
        self._script_tab_closer = closer

    def set_script_tab_renamer(self, renamer: Optional[Callable[[int, str], None]]):
        """Set callback invoked when a Script Box script_ref is renamed/switched."""
        self._script_tab_renamer = renamer

    def set_behavior_param_provider(self, provider: Optional[Callable[..., List[dict]]]):
        """Set callback returning exposed Behavior action parameters for a node."""
        self._behavior_param_provider = provider

    def set_behavior_param_writer(self, writer: Optional[Callable[..., bool]]):
        """Set callback that persists one exposed Behavior action parameter edit."""
        self._behavior_param_writer = writer

    def eventFilter(self, watched, event):
        if isinstance(watched, QLineEdit) and watched.property("symbol_combo_kind"):
            kind = str(watched.property("symbol_combo_kind") or "")
            ev_type = event.type()
            if ev_type in (QEvent.FocusIn, QEvent.MouseButtonPress):
                if kind != "script":
                    QTimer.singleShot(0, lambda le=watched: self._show_symbol_combo_popup(le))
            elif ev_type == QEvent.KeyPress:
                key = event.key()
                if key in (Qt.Key_Tab, Qt.Key_Backtab) and kind != "script":
                    if self._apply_symbol_combo_completion(watched):
                        event.accept()
                        return True
                elif key in (Qt.Key_Up, Qt.Key_Down):
                    self._show_symbol_combo_popup(watched)
        return super().eventFilter(watched, event)

    def _refresh_symbol_models(self):
        self._event_symbol_model.setStringList(sorted(self._event_symbol_set, key=lambda s: s.lower()))
        self._script_symbol_model.setStringList(sorted(self._script_symbol_set, key=lambda s: s.lower()))
        self._policy_symbol_model.setStringList(sorted(self._policy_symbol_set, key=lambda s: s.lower()))
        alive: List[QComboBox] = []
        for combo in self._symbol_combos:
            if combo is None or not isValid(combo):
                continue
            alive.append(combo)
            self._sync_symbol_combo_items(combo)
        self._symbol_combos = alive

    def refresh_policy_registry(self) -> None:
        """
        Populate the policy_id dropdown items from CheckpointRegistry.

        Called by the main window (or any controller) after CheckpointRegistry is
        available (Circle 3+).  Safe to call before Circle 3 — silently no-ops
        if CheckpointRegistry cannot be imported.

        Usage (Circle 3+):
            scene.refresh_policy_registry()
        """
        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            registry = CheckpointRegistry()
            policies = registry.discover()
            self._policy_symbol_set = {p.policy_id for p in policies}
        except Exception:
            pass
        self._refresh_symbol_models()
        self._refresh_policy_select_combos()

    def _refresh_policy_select_combos(self) -> None:
        """Repopulate all non-editable Checkpoint node policy_id pickers."""
        items = sorted(self._policy_symbol_set, key=lambda s: s.lower())
        alive = []
        for combo in self._policy_select_combos:
            if combo is None or not isValid(combo):
                continue
            alive.append(combo)
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if current:
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                elif items:
                    combo.setCurrentIndex(0)
            elif items:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
            # The Checkpoint node's "Open Training Ground" button state depends on
            # currentTextChanged. Because repopulation is signal-blocked above,
            # emit one sync notification after refresh.
            try:
                combo.currentTextChanged.emit(combo.currentText())
            except Exception:
                pass
        self._policy_select_combos = alive

    def retarget_checkpoint_policy_references(
        self,
        source_policy_id: str,
        target_policy_id: str,
    ) -> int:
        """
        Replace Mission-side Checkpoint picker selections from source -> target.

        Returns the number of Checkpoint nodes updated. Only nodes whose current
        picker selection exactly matches ``source_policy_id`` are modified.
        """
        source = str(source_policy_id or "").strip()
        target = str(target_policy_id or "").strip()
        if not source or not target or source == target:
            return 0

        updated = 0
        for item in self.items():
            if not item or not isValid(item):
                continue
            picker = getattr(item, "_ckpt_policy_id_picker", None)
            if picker is None or getattr(picker, "combo", None) is None:
                continue
            combo = picker.combo
            if not isValid(combo):
                continue
            if combo.currentText().strip() != source:
                continue

            idx = combo.findText(target)
            if idx < 0:
                continue

            combo.setCurrentIndex(idx)
            updated += 1

        return updated

    @staticmethod
    def _checkpoint_has_trainable_asset(bundle_path: Path) -> bool:
        """Return True when *bundle_path* looks like a recoverable training package."""
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            return False
        candidate_patterns = (
            "best/best_model.zip",
            "best_model.zip",
            "checkpoints/*.zip",
            "*.zip",
        )
        for pattern in candidate_patterns:
            for candidate in bundle_path.glob(pattern):
                if candidate.is_file():
                    return True
        return False

    def _bundle_sim_compat_check(self, bundle) -> str:
        """Return an error string if *bundle* cannot run in the current sim env, else ''.

        Two static checks (no live MuJoCo simulation required):

        1. ONNX dry-run — load the inference engine and run one forward pass
           with a zero observation. Catches corrupt or missing model files and
           non-finite outputs that would blow up physics.

        2. Joint name compatibility — resolve the MJCF for this bundle's robot
           via mujoco_asset_registry, load the MuJoCo model (read-only, no
           simulation), extract actuator names, and compare against the bundle's
           joint_names list.  Mismatches mean CompatibilityChecker would raise
           IncompatibleWeightError at PolicyRunner.load() time.
        """
        import numpy as np

        # ── 1. ONNX inference dry-run ──────────────────────────────────────
        try:
            from src.system.policy.inference_engine import ONNXEngine
            engine = ONNXEngine()
            engine.load(bundle.policy_file)
            dummy_obs = np.zeros(bundle.obs_dim, dtype=np.float32)
            action = engine.predict(dummy_obs)
            if not np.all(np.isfinite(action)):
                return (
                    f"ONNX 推理产生非有限输出（NaN/Inf），该模型无法安全驱动物理仿真。"
                )
        except Exception as exc:  # noqa: BLE001
            return f"ONNX 推理干跑失败：{exc}"

        # ── 2. Joint name compatibility vs registered MJCF ─────────────────
        try:
            robot_type = (bundle.raw_manifest.get("robot") or {}).get("type", "")
            robot_brand = str(bundle.robot_brand or "").lower()
            if robot_type and robot_brand:
                from src.system.models.mujoco_asset_registry import resolve_mujoco_asset
                loc = resolve_mujoco_asset(robot_brand, robot_type)
                if loc is not None and loc.scene_path.exists():
                    import mujoco as _mj
                    m = _mj.MjModel.from_xml_path(str(loc.scene_path))
                    mjcf_actuators = set()
                    for i in range(int(getattr(m, "nu", 0))):
                        name = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_ACTUATOR, i)
                        if name:
                            mjcf_actuators.add(name)
                    if mjcf_actuators:
                        from src.system.policy.joint_name_utils import canonicalize_joint_names
                        bundle_joints = set(canonicalize_joint_names(bundle.joint_names))
                        mjcf_actuator_names = set(canonicalize_joint_names(mjcf_actuators))
                        extra_in_bundle = bundle_joints - mjcf_actuator_names
                        if extra_in_bundle:
                            return (
                                f"关节名称与 MJCF 不兼容，运行时将触发 IncompatibleWeightError。"
                                f"\nBundle 中存在 MJCF 不认识的关节：{sorted(extra_in_bundle)}"
                                f"\n请在 Training Ground 中重新导出以匹配当前机型配置。"
                            )
        except Exception:  # noqa: BLE001
            pass  # Registry or MuJoCo unavailable; skip this check

        return ""

    def _checkpoint_validation_state(self, policy_id: str) -> Dict[str, str]:
        """Classify a checkpoint for runtime use and Training Ground recovery."""
        policy_id = str(policy_id or "").strip()
        if not policy_id:
            return {
                "symbol": "❌",
                "level": "invalid",
                "tooltip": "未选择 Checkpoint。",
            }

        try:
            from src.system.service.checkpoint_registry import CheckpointRegistry
            bundle_path = CheckpointRegistry().get_bundle_path(policy_id)
        except Exception as exc:  # noqa: BLE001
            return {
                "symbol": "❌",
                "level": "invalid",
                "tooltip": f"Checkpoint 未注册或无法定位：{exc}",
            }

        try:
            from src.system.policy.bundle_loader import BundleLoader
            bundle = BundleLoader().load(bundle_path)
            compat_error = self._bundle_sim_compat_check(bundle)
            if compat_error:
                raise RuntimeError(compat_error)
            return {
                "symbol": "✔",
                "level": "runnable",
                "tooltip": (
                    "该 Checkpoint 已是合法运行产物，可直接进入 MuJoCo 模拟。"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            runtime_error = str(exc)

        if self._checkpoint_has_trainable_asset(Path(bundle_path)):
            return {
                "symbol": "⚠",
                "level": "trainable",
                "tooltip": (
                    "该 Checkpoint 不是当前可直接运行的合法产物，但检测到可恢复的训练资产。"
                    f"\n需要先进入 Training Ground 做校准/微调并重新导出。"
                    f"\n运行拦截原因：{runtime_error}"
                ),
            }

        return {
            "symbol": "❌",
            "level": "invalid",
            "tooltip": (
                "该 Checkpoint 既不是合法运行产物，也未检测到可恢复的训练资产。"
                f"\n校验失败：{runtime_error}"
            ),
        }

    def _record_symbol(self, kind: str, raw_text: str):
        token = str(raw_text or "").strip()
        if not token or token.startswith("<"):
            return
        changed = False
        if kind == "event":
            if token not in self._event_symbol_set:
                self._event_symbol_set.add(token)
                changed = True
        elif kind == "script":
            if token not in self._script_symbol_set:
                self._script_symbol_set.add(token)
                changed = True
        elif kind == "policy":
            if token not in self._policy_symbol_set:
                self._policy_symbol_set.add(token)
                changed = True
        if changed:
            self._refresh_symbol_models()

    @staticmethod
    def _combo_no_arrow_style() -> str:
        return """
            QComboBox::drop-down {
                border: 0px;
                width: 0px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
        """

    def _sync_symbol_combo_items(self, combo: QComboBox):
        kind = str(combo.property("symbol_combo_kind") or "")
        if kind == "event":
            values = self._event_symbol_model.stringList()
        elif kind == "policy":
            values = self._policy_symbol_model.stringList()
        else:
            values = self._script_symbol_model.stringList()
        current = combo.currentText().strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        if current:
            combo.setEditText(current)
        combo.blockSignals(False)

    def _wire_symbol_combo(self, combo: Optional[QComboBox], kind: str):
        if combo is None or not isValid(combo):
            return
        if combo.property("symbol_combo_kind"):
            return
        combo.setProperty("symbol_combo_kind", kind)
        combo.setProperty("hide_native_arrow", True)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setStyleSheet((combo.styleSheet() or "") + self._combo_no_arrow_style())
        line_edit = combo.lineEdit()
        if line_edit is None:
            return
        line_edit.setProperty("symbol_combo_kind", kind)
        line_edit.installEventFilter(self)
        self._symbol_combos.append(combo)
        self._symbol_line_to_combo[line_edit] = combo
        if kind != "script":
            combo.editTextChanged.connect(
                lambda _t, cb=combo: self._show_symbol_combo_popup(cb.lineEdit())
            )
        line_edit.editingFinished.connect(lambda cb=combo, k=kind: self._record_symbol(k, cb.currentText()))
        self._sync_symbol_combo_items(combo)
        self._record_symbol(kind, combo.currentText())

    def _show_symbol_combo_popup(self, line_edit: Optional[QLineEdit]):
        if line_edit is None or not isValid(line_edit):
            return
        combo = self._symbol_line_to_combo.get(line_edit)
        if combo is None or not isValid(combo):
            return
        kind = str(combo.property("symbol_combo_kind") or "")
        if kind == "event":
            source = self._event_symbol_model.stringList()
        elif kind == "policy":
            source = self._policy_symbol_model.stringList()
        else:
            source = self._script_symbol_model.stringList()
        token = line_edit.text().strip().lower()
        filtered = source if not token else [v for v in source if token in v.lower()]
        current = line_edit.text()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(filtered)
        combo.setEditText(current)
        combo.blockSignals(False)
        if not filtered:
            combo.hidePopup()
            return
        combo.showPopup()

    def _apply_symbol_combo_completion(self, line_edit: Optional[QLineEdit]) -> bool:
        if line_edit is None or not isValid(line_edit):
            return False
        combo = self._symbol_line_to_combo.get(line_edit)
        if combo is None or not isValid(combo):
            return False
        value = ""
        if combo.view() is not None and combo.view().isVisible():
            cur = combo.view().currentIndex()
            if cur.isValid():
                value = combo.itemText(cur.row())
        if not value:
            value = combo.currentText()
        value = value.strip()
        if not value:
            return False
        combo.setEditText(value)
        line_edit.setCursorPosition(len(value))
        self._record_symbol(str(line_edit.property("symbol_combo_kind")), value)
        self._sync_symbol_combo_items(combo)
        combo.hidePopup()
        return True

    def set_robot_type(self, robot_type: str):
        """Set robot type"""
        self._robot_type = robot_type
        # Sync Start node model picker if present
        for item in self.items():
            if isValid(item) and item.data(10) == "node" and item.data(11) == "Start":
                model_combo = getattr(item, '_model_picker', None)
                if model_combo:
                    idx = model_combo.findText(robot_type)
                    if idx >= 0:
                        model_combo.setCurrentIndex(idx)
                break
        log_info(f"Robot type set to: {robot_type}")

    def get_start_node_selection(self) -> tuple[str, str]:
        """Return the live Start-node brand/model selection when present."""
        for item in self.items():
            if not isValid(item) or item.data(10) != "node" or item.data(11) != "Start":
                continue
            brand_combo = getattr(item, "_brand_picker", None)
            model_combo = getattr(item, "_model_picker", None)
            brand = str(
                brand_combo.currentData(Qt.ItemDataRole.UserRole)
                if brand_combo else self._robot_brand or ""
            ).strip().lower()
            robot_type = str(model_combo.currentText() if model_combo else self._robot_type or "").strip().lower()
            return brand, robot_type
        return (
            normalize_brand_id(str(getattr(self, "_robot_brand", "") or "")),
            str(getattr(self, "_robot_type", "") or "").strip().lower(),
        )

    @staticmethod
    def _behavior_action_descriptors(brand: str = "", robot_type: str = "") -> List[dict]:
        """Return all canonical IR action descriptors for Behavior-node dropdowns.

        ALL registered IR intents are always returned regardless of brand/model.
        Behavior actions are IR-level abstractions, so their labels must remain
        stable across model switches. Brand/robot_type is kept only for future
        parameter-default overlays, not for label changes.
        """
        from src.system.behavior.ir_action_registry import list_ir_actions_by_intent

        base_by_intent = list_ir_actions_by_intent()

        result = []
        for intent_id in sorted(base_by_intent.keys()):
            desc = base_by_intent[intent_id]
            label = desc.display_name or desc.raw_action or desc.intent_id
            result.append({
                "label": label,
                "intent_id": desc.intent_id,
                "raw_action": desc.raw_action,
                "availability": desc.availability,
                "is_unsupported": False,  # IR actions are never blocked in the authoring UI
            })

        return result

    @staticmethod
    def _set_behavior_combo_items(combo, descriptors: List[dict], current_label: str = "", brand: str = "", robot_type: str = "") -> None:
        """Populate a Behavior-node combo with IR action labels + metadata.

        All IR actions are always selectable — no item is ever disabled.
        """
        combo.clear()
        labels = []
        for entry in descriptors:
            if isinstance(entry, dict):
                label = str(entry.get("label") or "").strip()
                intent_id = str(entry.get("intent_id") or "").strip()
                raw_action = str(entry.get("raw_action") or "").strip()
            else:
                # SemanticActionDescriptor object
                label = str(
                    getattr(entry, "display_name", "")
                    or getattr(entry, "raw_action", "")
                    or getattr(entry, "intent_id", "")
                ).strip()
                intent_id = str(getattr(entry, "intent_id", "") or "").strip()
                raw_action = str(getattr(entry, "raw_action", "") or "").strip()

            if not label:
                continue

            combo.addItem(label, {
                "intent_id": intent_id,
                "raw_action": raw_action,
                "label": label,
            })
            labels.append(label)

        if current_label and current_label in labels:
            combo.setCurrentText(current_label)
        elif labels:
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(-1)


    @staticmethod
    def _default_behavior_ref(node_id: int) -> str:
        return f"behavior_node_{int(node_id)}"

    def _ensure_behavior_ref(self, rect_item) -> str:
        node_id = int(rect_item.data(12) or 0)
        existing = str(getattr(rect_item, "_behavior_ref", "") or "").strip()
        if existing:
            return existing
        ref = self._default_behavior_ref(node_id)
        rect_item._behavior_ref = ref
        return ref

    def refresh_behavior_node_actions(self, descriptors: list = None) -> None:
        """Update every Behavior node's action combo when the robot selection changes.

        Always rebuilds from the full canonical IR catalog — all intents are
        shown regardless of brand/model.  The ``descriptors`` argument is kept
        for call-site compatibility but is no longer used; brand/model context
        is read live from the Start node via ``get_start_node_selection()``.
        """
        brand, robot_type = self.get_start_node_selection()
        current_descriptors = self._behavior_action_descriptors(brand, robot_type)
        for item in self.items():
            if not isValid(item):
                continue
            if item.data(10) != "node" or item.data(11) != "Behavior":
                continue
            condition_set = getattr(item, "_condition_set", None)
            if condition_set is None:
                continue
            combo = getattr(condition_set, "combo", None)
            if combo is None:
                continue
            current = combo.currentText()
            combo.blockSignals(True)
            self._set_behavior_combo_items(combo, current_descriptors, current, brand, robot_type)
            combo.blockSignals(False)
            self._refresh_behavior_node_parameter_rows(item)
            self._sync_node_parameters(item)

    def refresh_behavior_node_parameter_rows(self, node_id: Optional[int] = None) -> None:
        """Refresh dynamic parameter rows for one Behavior node or all of them."""
        if node_id is not None:
            item = self.get_node_item(node_id)
            if item is not None and item.data(11) == "Behavior":
                self._refresh_behavior_node_parameter_rows(item)
            return
        for item in self.items():
            if not isValid(item):
                continue
            if item.data(10) != "node" or item.data(11) != "Behavior":
                continue
            self._refresh_behavior_node_parameter_rows(item)

    def _refresh_behavior_node_parameter_rows(self, rect_item) -> None:
        row_stack = getattr(rect_item, "_row_stack", None)
        if row_stack is None:
            return
        for row_id in list(getattr(rect_item, "_behavior_param_row_ids", [])):
            row_stack.remove_row(row_id)
        rect_item._behavior_param_row_ids = []
        rect_item._behavior_param_inputs = {}

        provider = self._behavior_param_provider
        if not callable(provider):
            sync_layout = getattr(rect_item, "_sync_layout_fn", None)
            if callable(sync_layout):
                sync_layout()
            return

        cond_set = getattr(rect_item, "_condition_set", None)
        combo = getattr(cond_set, "combo", None) if cond_set is not None else None
        combo_data = combo.currentData(Qt.ItemDataRole.UserRole) if combo is not None else {}
        if not isinstance(combo_data, dict):
            combo_data = {}
        display_name = cond_set.logic_text() if cond_set is not None else ""
        behavior_ref = self._ensure_behavior_ref(rect_item)
        _brand, robot_type = self.get_start_node_selection()
        rows = provider(
            node_id=int(rect_item.data(12) or 0),
            behavior_ref=behavior_ref,
            intent_id=str(combo_data.get("intent_id") or ""),
            display_name=display_name,
            robot_type=robot_type or self._robot_type,
        ) or []

        if not rows:
            sync_layout = getattr(rect_item, "_sync_layout_fn", None)
            if callable(sync_layout):
                sync_layout()
            self._update_node_params(rect_item)
            return

        rect_item._behavior_param_suspend = True
        try:
            for idx, entry in enumerate(rows):
                key = str(entry.get("key") or "").strip()
                if not key:
                    continue
                row_id = f"behavior_param_row_{idx}"
                setting = InputSetting(
                    self._condition_set_left_bg,
                    self._condition_set_right_bg,
                    get_color("input_bg", "#0f1115"),
                    self._button_border,
                    self._condition_set_text,
                    key_text=str(entry.get("label") or key),
                    placeholder="",
                    show_left_dot=False,
                    show_right_dot=False,
                )
                setting.set_value(str(entry.get("value") or ""))
                setting.input_edit.editingFinished.connect(
                    lambda _k=key, _r=rect_item, _w=setting.input_edit: self._on_behavior_param_input_committed(_r, _k, _w)
                )
                row_widget = QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(8, 2, 8, 2)
                row_layout.setSpacing(6)
                row_layout.addWidget(setting, 1)
                row_stack.add_row(row_id, row_widget)
                rect_item._behavior_param_row_ids.append(row_id)
                rect_item._behavior_param_inputs[key] = setting.input_edit
        finally:
            rect_item._behavior_param_suspend = False

        sync_layout = getattr(rect_item, "_sync_layout_fn", None)
        if callable(sync_layout):
            sync_layout()
        self._update_node_params(rect_item)

    def _on_behavior_param_input_committed(self, rect_item, key: str, line_edit: QLineEdit) -> None:
        if getattr(rect_item, "_behavior_param_suspend", False):
            return
        writer = self._behavior_param_writer
        if not callable(writer):
            return
        cond_set = getattr(rect_item, "_condition_set", None)
        combo = getattr(cond_set, "combo", None) if cond_set is not None else None
        combo_data = combo.currentData(Qt.ItemDataRole.UserRole) if combo is not None else {}
        if not isinstance(combo_data, dict):
            combo_data = {}
        display_name = cond_set.logic_text() if cond_set is not None else ""
        behavior_ref = self._ensure_behavior_ref(rect_item)
        _brand, robot_type = self.get_start_node_selection()
        writer(
            node_id=int(rect_item.data(12) or 0),
            behavior_ref=behavior_ref,
            intent_id=str(combo_data.get("intent_id") or ""),
            display_name=display_name,
            robot_type=robot_type or self._robot_type,
            key=str(key),
            value=line_edit.text(),
        )
        self._update_node_params(rect_item)
        self._sync_node_parameters(rect_item)
        self.regenerate_code()

    def _place_initial_nodes(self):
        """
        Place the default Mission Canvas template.

        Workflow lane:
            START >> BEHAVIOR >> END

        Runtime lane:
            CHECKPOINT >> BEHAVIOR
        """
        x0 = self._initial_origin.x()
        y0 = self._initial_origin.y()

        start = self.create_node("Start", QPointF(x0, y0))
        behavior = self.create_node("Behavior", QPointF(x0 + 300, y0 - 30))
        end = self.create_node("End", QPointF(x0 + 620, y0))

        checkpoint = self.create_node("Checkpoint", QPointF(x0, y0 + 220))

        def _wire(src_item, src_slot: str, dst_item, dst_slot: str):
            out_p = self._get_node_port(src_item, src_slot, "out")
            in_p = self._get_node_port(dst_item, dst_slot, "in")
            if out_p and in_p:
                self._create_connection(out_p, in_p)

        # Workflow path
        _wire(start, "flow_out", behavior, "flow_in")
        _wire(behavior, "flow_out", end, "flow_in")

        # Runtime checkpoint input
        _wire(checkpoint, "checkpoint", behavior, "checkpoint")

    def _apply_theme(self):
        """Apply theme colors and widget styles"""
        # Always reload theme values from ui.ini before applying styles.
        get_color_slot().reload()
        self.color_bg = QColor(get_color("canvas_bg", get_color("bg", "#1e1e1e")))
        self.color_grid_small = QColor(get_color("graph_grid_small", "#323337"))
        self.color_grid_big = QColor(get_color("graph_grid_big", "#3c3e43"))
        self._node_title_color = QColor(get_color("node_title", "#ffffff"))
        node_content_text = get_color("node_content_text", "#000000")

        input_bg = get_color("input_bg", get_color("cmd_bg", "#0f1115"))
        input_text = get_color("input_text", "#000000")
        input_border = get_color("input_border", get_color("border", "#4b5563"))
        popup_bg = get_color("input_popup_bg", get_color("card_bg", "#111827"))
        popup_sel = get_color("input_popup_selected_bg", get_color("hover_bg", "#334155"))
        hover_bg = get_color("hover_bg", "#1f2937")
        button_bg = get_color("node_control_bg", get_color("button_bg", "#334155"))
        button_text = get_color("node_control_text", "#000000")
        button_border = get_color("node_control_border", get_color("button_border", input_border))
        button_hover_bg = get_color("node_control_hover_bg", get_color("hover_bg", "#475569"))
        condition_set_left_bg = get_color("part_dot_zone_bg", button_bg)
        condition_set_right_bg = get_color("part_main_zone_bg", button_bg)
        condition_set_text = get_color("condition_set_text", button_text)
        main_picker_text = get_color("main_picker_text", input_text)

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
            QComboBox::item {{
                color: {input_text};
                background: {popup_bg};
            }}
            QComboBox::item:selected {{
                background: {popup_sel};
                color: {input_text};
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
                background: {button_hover_bg};
            }}
        """
        self._completer_popup_style = f"""
            QListView {{
                background: {popup_bg};
                color: {input_text};
                selection-background-color: {popup_sel};
                border: 1px solid {input_border};
            }}
        """
        tag_text = get_color("tag_text", "#000000")
        self._tag_style = f"""
            QLabel#nodeTag {{
                color: {tag_text};
                background: transparent;
                border: none;
                padding: 0px;
                font-size: 11px;
            }}
        """
        # Store individual colours for ConditionSet and other direct-style widgets
        self._button_bg = button_bg
        self._button_border = button_border
        self._button_text = button_text
        self._hover_bg = button_hover_bg
        self._input_bg = input_bg
        self._input_text = input_text
        self._node_content_text = node_content_text
        self._condition_set_left_bg = condition_set_left_bg
        self._condition_set_right_bg = condition_set_right_bg
        self._condition_set_text = condition_set_text
        self._main_picker_text = main_picker_text
        self._node_base_color = get_color("node_base_color", get_color("card_bg", "#2d2d2d"))

        self._main_dot_color = get_color("main_dot_border", get_color("connection", "#60a5fa"))
        self._sub_dot_color = get_color("sub_dot_border", "#22c55e")
        self._port_invalid_color = get_color("port_invalid", "#9ca3af")

    def _resolve_node_gradient(self, name: str, grad: Optional[tuple]) -> tuple:
        """Resolve node gradient from ui.ini NodeColors"""
        card_bg = get_color("card_bg", "#2d2d2d")
        fallback_start = grad[0] if grad and len(grad) == 2 else card_bg
        fallback_end = grad[1] if grad and len(grad) == 2 else card_bg

        # Map palette labels and canonical names to a stable NodeColors category.
        name_l = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
        if any(k in name_l for k in ("logic control", "while loop", "for loop", "loop", "if", "elif", "else", "switch", "parallel", "join", "try catch", "gate")):
            return get_node_color_pair("logic", fallback_start, fallback_end)
        if any(k in name_l for k in ("condition", "compare", "checkstate", "check state")):
            return get_node_color_pair("condition", fallback_start, fallback_end)
        if any(k in name_l for k in ("reactive loco", "reactive_loco", "reactive locomotion")):
            return get_node_color_pair("transition", fallback_start, fallback_end)
        if any(k in name_l for k in ("action execution", "behavior", "executebehavior", "callservice", "sendcommand", "movetopose", "gripper", "triggerplugin", "abort", "cancel", "end", "start")):
            return get_node_color_pair("action", fallback_start, fallback_end)
        if any(k in name_l for k in ("sensor input", "readsensor", "sensor", "trigger", "event", "on event", "publishevent", "humanconfirm", "alert")):
            return get_node_color_pair("sensor", fallback_start, fallback_end)
        if any(k in name_l for k in ("compute", "math", "setvariable", "expression", "timer", "timeout", "retry", "wait", "formatdata", "converttype")):
            return get_node_color_pair("compute", fallback_start, fallback_end)
        if any(k in name_l for k in ("script box", "script")):
            return get_node_color_pair("script", fallback_start, fallback_end)
        if any(k in name_l for k in ("checkpoint", "policy", "conductor")):
            return get_node_color_pair("compute", fallback_start, fallback_end)
        # Phase 7 A5: transition nodes get a distinct color (teal/cyan)
        if any(k in name_l for k in ("transition", "transition behavior", "skill transition")):
            return get_node_color_pair("transition", fallback_start, fallback_end)
        # Phase 6 C2: manual control node
        if any(k in name_l for k in ("manual control", "manual_control", "manualcontrol")):
            return get_node_color_pair("action", fallback_start, fallback_end)

        if grad and len(grad) == 2:
            return tuple(grad)
        return (card_bg, card_bg)

    @staticmethod
    def _is_unique_color_node(name: str, unique_color: bool = False) -> bool:
        if unique_color:
            return True
        return str(name or "").strip().upper() in {"START", "END", "BEHAVIOR", "SCRIPT BOX"}

    @staticmethod
    def _title_row_gradient_style(grad: Optional[tuple]) -> str:
        if grad and len(grad) == 2:
            return (
                "QWidget {"
                " border: 0px;"
                f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {grad[0]}, stop:1 {grad[1]});"
                " }"
            )
        return "QWidget { border: 0px; background: transparent; }"

    def _apply_node_fill_and_header_style(self, rect_item, node_name: str, grad_seed: Optional[tuple] = None, unique_color: bool = False):
        """Apply node body fill + title-row style according to uniqueColor rule."""
        resolved_grad = self._resolve_node_gradient(node_name, grad_seed)
        is_unique = self._is_unique_color_node(node_name, unique_color)

        if is_unique and resolved_grad and len(resolved_grad) == 2:
            g = QLinearGradient(0, 0, 1, 1)
            g.setCoordinateMode(QGradient.ObjectBoundingMode)
            g.setColorAt(0.0, QColor(resolved_grad[0]))
            g.setColorAt(1.0, QColor(resolved_grad[1]))
            rect_item.setBrush(QBrush(g))
        else:
            rect_item.setBrush(QBrush(QColor(getattr(self, "_node_base_color", get_color("card_bg", "#2d2d2d")))))

        header_row_widget = getattr(rect_item, "_header_row_widget", None)
        if header_row_widget and isValid(header_row_widget):
            if is_unique:
                header_row_widget.setStyleSheet("QWidget { border: 0px; background: transparent; }")
            else:
                header_row_widget.setStyleSheet(self._title_row_gradient_style(resolved_grad))

        rect_item._unique_color = is_unique
        rect_item._node_grad_seed = tuple(grad_seed) if grad_seed and len(grad_seed) == 2 else None
        rect_item._resolved_grad = tuple(resolved_grad) if resolved_grad and len(resolved_grad) == 2 else None

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
        self._cancel_sub_dot_tooltip()
        pos = event.scenePos()
        item = self.itemAt(pos, self.views()[0].transform() if self.views() else None)

        # Capture pre-drag positions of selected nodes so mouseReleaseEvent
        # can decide whether to push an undo entry for the move. Only record
        # positions on left-button presses that land on something selectable;
        # other cases (e.g. right-click panning, port drags) ignore this.
        self._drag_start_positions = {}
        try:
            if event.button() == Qt.LeftButton:
                for sel in self.selectedItems():
                    if isValid(sel) and sel.data(10) in ("node", "group"):
                        self._drag_start_positions[id(sel)] = (sel, QPointF(sel.pos()))
        except Exception:
            self._drag_start_positions = {}

        # Check if clicked on connection marker (endpoint)
        if item and item.data(0) == "connection_marker":
            connection = item.data(2)
            end_type = item.data(1)
            self._start_reconnection(connection, end_type, pos)
            return

        # Use the same radius-based hit test as hover/cursor preview so
        # "highlighted == clickable" for starting a connection.
        clicked_port = item if self._is_port(item) else self._find_port_near(
            pos,
            radius=self.get_port_interaction_radius(),
        )

        # Check if clicked on port
        if self._is_port(clicked_port):
            # If port already has a connection, detach and reconnect instead
            # of creating a new line.
            existing = self._get_reconnectable_connection(clicked_port)
            if existing:
                conn, end_type = existing
                self._start_reconnection(conn, end_type, pos)
                return
            self._start_connection(clicked_port, pos)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move event"""
        if self._temp_connection:
            # Update temporary connection
            self._clear_port_hover()
            self._cancel_sub_dot_tooltip()
            self._update_temp_connection(event.scenePos())
            return

        if self._reconnecting and self._temp_connection:
            # Update reconnection temporary line
            self._clear_port_hover()
            self._cancel_sub_dot_tooltip()
            self._update_temp_connection(event.scenePos())
            return

        self._update_port_hover(event.scenePos())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Mouse release event"""
        if self._reconnecting:
            r = self.get_port_interaction_radius()
            target_port = self._snap_target_port or self._find_port_near(event.scenePos(), radius=r)

            if target_port:
                self._finish_reconnection(target_port)
            else:
                self._cancel_reconnection()
            self._drag_start_positions = {}
            return

        if self._temp_connection:
            r = self.get_port_interaction_radius()
            target_port = self._snap_target_port or self._find_port_near(event.scenePos(), radius=r)

            if target_port and target_port != self._temp_start_port:
                self._finish_connection(target_port)
            else:
                self._cancel_connection()

            self._drag_start_positions = {}
            return

        super().mouseReleaseEvent(event)

        # Detect node/group drag: compare recorded pre-drag positions with
        # current positions. If any item moved by more than a pixel, count it
        # as a single undoable step. This catches both plain node drags and
        # multi-selection moves in one entry.
        moved = False
        try:
            for _key, (item, old_pos) in (self._drag_start_positions or {}).items():
                if not isValid(item):
                    continue
                delta = item.pos() - old_pos
                if abs(delta.x()) > 0.5 or abs(delta.y()) > 0.5:
                    moved = True
                    break
        except Exception:
            moved = False
        finally:
            self._drag_start_positions = {}
        if moved:
            self._touch_content()

    def _cancel_sub_dot_tooltip(self):
        """Stop pending sub_dot tooltip and hide active tooltip."""
        if self._sub_dot_tooltip_timer.isActive():
            self._sub_dot_tooltip_timer.stop()
        self._sub_dot_tooltip_port = None
        self._sub_dot_tooltip_scene_pos = QPointF()
        if self._sub_dot_tooltip_visible:
            QToolTip.hideText()
        self._sub_dot_tooltip_visible = False

    def _schedule_sub_dot_tooltip(self, port_item, scene_pos: QPointF):
        """Schedule delayed tooltip for a sub_dot input port."""
        if not port_item or not isValid(port_item):
            self._cancel_sub_dot_tooltip()
            return
        if self._sub_dot_tooltip_port == port_item and self._sub_dot_tooltip_timer.isActive():
            self._sub_dot_tooltip_scene_pos = scene_pos
            return
        self._sub_dot_tooltip_port = port_item
        self._sub_dot_tooltip_scene_pos = scene_pos
        self._sub_dot_tooltip_visible = False
        self._sub_dot_tooltip_timer.start(self._sub_dot_tooltip_delay_ms)

    def _update_sub_dot_tooltip_candidate(self, port_item, scene_pos: QPointF):
        """Keep tooltip candidate in sync with current hover target."""
        if not self._is_port(port_item):
            self._cancel_sub_dot_tooltip()
            return
        self._schedule_sub_dot_tooltip(port_item, scene_pos)

    @staticmethod
    def _port_tooltip_text(meta: Dict[str, Any]) -> str:
        """Build human-readable tooltip text for a port."""
        data_type = _normalize_data_type(meta.get("data_type", "any"))
        max_conn = meta.get("max_connections")
        if max_conn is not None and max_conn < 0:
            return f"Fan-In  [{data_type}]"
        return data_type

    def _show_sub_dot_tooltip(self):
        """Render tooltip for a port after delayed hover."""
        port_item = self._sub_dot_tooltip_port
        if not port_item or not isValid(port_item):
            self._cancel_sub_dot_tooltip()
            return
        meta = self._port_meta(port_item)
        if not self.views():
            self._cancel_sub_dot_tooltip()
            return
        view = self.views()[0]
        global_pos = view.mapToGlobal(view.mapFromScene(self._sub_dot_tooltip_scene_pos))
        QToolTip.showText(global_pos, self._port_tooltip_text(meta), view)
        self._sub_dot_tooltip_visible = True

    def keyPressEvent(self, event):
        """Keyboard press event"""
        if event.key() == Qt.Key_Delete:
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            if isinstance(focused, QComboBox) and focused.isEditable():
                le = focused.lineEdit()
                if le is not None and le.hasFocus():
                    super().keyPressEvent(event)
                    return
            # Delete selected items
            selected_items = self.selectedItems()
            if selected_items:
                self._delete_items(selected_items)
                event.accept()
                return

        elif event.key() == Qt.Key_D and event.modifiers() & Qt.ControlModifier:
            self.duplicate_selected_nodes()
            event.accept()
            return

        elif event.key() == Qt.Key_G and event.modifiers() & Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                self.ungroup_selected_groups()
            else:
                self.group_selected_nodes()
            event.accept()
            return

        # ── Undo (Ctrl+Z) / Redo (Ctrl+Y or Ctrl+Shift+Z) ───────────────
        # Only act on the canvas when the focus is NOT inside a text-editing
        # widget — those widgets already have their own native undo stacks
        # and stealing the shortcut would break editing inside node params.
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            if isinstance(focused, QComboBox) and focused.isEditable():
                le = focused.lineEdit()
                if le is not None and le.hasFocus():
                    super().keyPressEvent(event)
                    return
            if event.modifiers() & Qt.ShiftModifier:
                self.redo()
            else:
                self.undo()
            event.accept()
            return

        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            focused = QApplication.focusWidget()
            if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            if isinstance(focused, QComboBox) and focused.isEditable():
                le = focused.lineEdit()
                if le is not None and le.hasFocus():
                    super().keyPressEvent(event)
                    return
            self.redo()
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
                    # Skip protected nodes (Start/End)
                    if getattr(item, '_is_protected', False):
                        log_info(f"Cannot delete protected node: {item.data(11)}")
                        continue

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
                    # Close any open script editor tab for this node
                    if node_name == "Script Box" and node_id is not None:
                        if callable(self._script_tab_closer):
                            try:
                                self._script_tab_closer(node_id)
                            except Exception:
                                pass

                # Delete group container (members are NOT deleted 鈥?just detached)
                elif item.data(10) == "group":
                    gid = item._group_id
                    for scene_item in self.items():
                        if (isValid(scene_item) and scene_item.data(10) == "node"
                                and getattr(scene_item, '_group_id', None) == gid):
                            scene_item._group_id = None
                    if item.scene() is not None:
                        self.removeItem(item)
                    self._groups.pop(gid, None)
                    log_info(f"Group container {gid!r} deleted (members retained)")

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

        # Dirty + undo history
        if deleted_nodes or deleted_connections:
            self._touch_content()

    def duplicate_selected_nodes(self):
        """Duplicate currently selected (non-protected) nodes, placing copies offset by (30, 30)."""
        selected = [item for item in self.selectedItems()
                    if isValid(item) and item.data(10) == "node"
                    and not getattr(item, '_is_protected', False)]
        if not selected:
            return

        # Snapshot serialized data for all selected nodes
        wf_data = self.serialize_workflow()
        selected_ids = {item.data(12) for item in selected}
        nodes_to_dup = [n for n in wf_data.get("nodes", []) if n["id"] in selected_ids]
        if not nodes_to_dup:
            return

        _DUP_OFFSET_X = 30.0
        _DUP_OFFSET_Y = 30.0

        self.clearSelection()

        for node_data in nodes_to_dup:
            pos_data = node_data.get("position", {})
            w = node_data.get("width", 180)
            h = node_data.get("height", 110)
            # create_node expects center position
            cx = pos_data.get("x", 0) + w / 2 + _DUP_OFFSET_X
            cy = pos_data.get("y", 0) + h / 2 + _DUP_OFFSET_Y

            name = node_data.get("display_name", "")
            new_item = self.create_node(name, QPointF(cx, cy))

            if not new_item or not isValid(new_item):
                continue

            self._set_node_widgets_silent(new_item, True)
            try:
                self._apply_node_data(new_item, node_data)
            finally:
                self._set_node_widgets_silent(new_item, False)

            self._update_node_params(new_item)
            new_item.setSelected(True)

        self.regenerate_code()
        log_info(f"Duplicated {len(nodes_to_dup)} node(s)")
        self._touch_content()

    # ------------------------------------------------------------------
    # Group / Ungroup
    # ------------------------------------------------------------------

    def group_selected_nodes(self, title: str = "") -> Optional[Any]:
        """Group currently selected non-protected nodes into a GroupContainerItem.

        Args:
            title: Optional label for the group container.
        Returns:
            The created GroupContainerItem, or None if grouping was skipped.
        """
        candidates = [
            item for item in self.selectedItems()
            if isValid(item) and item.data(10) == "node"
            and not getattr(item, '_is_protected', False)
        ]
        if len(candidates) < 1:
            log_info("group_selected_nodes: no eligible nodes selected")
            return None

        self._group_seq += 1
        gid = self._group_seq
        group_item = GroupContainerItem(gid, title)
        self.addItem(group_item)

        # Assign group membership to nodes
        for node_item in candidates:
            node_item._group_id = gid
            group_item._member_ids.add(node_item.data(12))

        # Fit container rect around members, then position it
        self._fit_group_to_members(group_item)

        # Store registry entry
        self._groups[gid] = group_item

        # Select the group container; deselect individual nodes
        self.clearSelection()
        group_item.setSelected(True)

        log_info(f"Grouped {len(candidates)} node(s) into group {gid!r} ({group_item._title!r})")
        self._touch_content()
        return group_item

    def ungroup_selected_groups(self):
        """Ungroup all currently selected group containers.

        Removes the container and detaches group membership from member nodes.
        Member nodes, their connections, and their parameters are untouched.
        """
        selected_groups = [
            item for item in self.selectedItems()
            if isValid(item) and item.data(10) == "group"
        ]
        if not selected_groups:
            log_info("ungroup_selected_groups: no group containers selected")
            return

        for g_item in selected_groups:
            gid = g_item._group_id

            # Detach _group_id from all member nodes
            for item in self.items():
                if (isValid(item) and item.data(10) == "node"
                        and getattr(item, '_group_id', None) == gid):
                    item._group_id = None

            # Remove container from scene and registry
            if g_item.scene() is not None:
                self.removeItem(g_item)
            self._groups.pop(gid, None)
            log_info(f"Ungrouped group {gid!r}")

        self._touch_content()

    def _fit_group_to_members(self, group_item) -> None:
        """Resize and reposition group_item to tightly enclose its member nodes.

        The container is placed at the bounding rect of all members with
        GroupContainerItem._FIT_PADDING padding on every side.
        """
        gid = group_item._group_id
        pad = GroupContainerItem._FIT_PADDING

        member_items = [
            item for item in self.items()
            if isValid(item) and item.data(10) == "node"
            and getattr(item, '_group_id', None) == gid
        ]
        if not member_items:
            group_item.setRect(QRectF(0, 0, 120, 80))
            return

        # Compute bounding rect in scene coords
        bounding = QRectF()
        for node_item in member_items:
            bounding = bounding.united(node_item.sceneBoundingRect())

        # Position container at top-left of bounding (minus padding)
        container_pos = QPointF(bounding.x() - pad, bounding.y() - pad)
        w = bounding.width() + 2 * pad
        h = bounding.height() + 2 * pad

        # setPos BEFORE setRect so that _prev_pos is set on the first change
        group_item.setPos(container_pos)
        group_item.setRect(0, 0, w, h)

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
            out_port_ref = connection.out_port if connection.out_port and isValid(connection.out_port) else None
            in_port_ref = connection.in_port if connection.in_port and isValid(connection.in_port) else None
            in_node_ref = in_port_ref.parentItem() if in_port_ref else None

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
                self._refresh_behavior_protocol_state(in_port_ref)
            if (not self._autowiring_break) and in_node_ref and isValid(in_node_ref) and in_node_ref.data(10) == "node":
                if in_node_ref.data(11) == "Break" and in_port_ref and in_port_ref.data(3) == "flow_in":
                    self._auto_wire_break_node(in_node_ref)
            if (not self._autowiring_break) and out_port_ref and isValid(out_port_ref):
                out_node_ref = out_port_ref.parentItem()
                if out_node_ref and isValid(out_node_ref) and out_node_ref.data(11) == "Break":
                    if out_port_ref.data(3) == "break_out":
                        self._auto_wire_break_node(out_node_ref)
        except Exception as e:
            log_debug(f"Error detaching connection: {e}")

    def _get_reconnectable_connection(self, port_item):
        """If port has exactly one connection and is capacity-limited, return
        (connection, end_type) so the caller can start a reconnection.
        For output ports with unlimited capacity, return None (allow fan-out).
        """
        conns = port_item.data(2) or []
        conns = [c for c in conns if c and isValid(c) and c.scene() is not None]
        if not conns:
            return None

        io = port_item.data(1)  # "in" or "out"
        meta = port_item.data(self._port_meta_key) or {}
        max_conn = meta.get("max_connections")

        # Input ports (max=1 by default): always reconnect the existing wire.
        # Output ports: only reconnect when max_connections is set and reached.
        if io == "in":
            conn = conns[0]
            return (conn, "end")  # detach the "end" (input) side
        elif max_conn is not None and len(conns) >= max_conn:
            conn = conns[-1]
            return (conn, "start")  # detach the "start" (output) side

        return None

    def _start_reconnection(self, connection, end_type, pos):
        """Start reconnection"""
        if not isinstance(connection, ConnectionItem):
            return

        self._reset_connection_preview()
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

        # Dim the old connection to semi-transparent gray while reconnecting
        ghost_color = QColor("#888888")
        ghost_color.setAlpha(90)
        connection.setPen(QPen(ghost_color, 2, Qt.DashLine))
        connection.setOpacity(0.5)

        # Create temporary drag line in normal blue
        path = QPainterPath()
        center = self._port_center(self._temp_start_port)
        path.moveTo(center)
        path.lineTo(pos)

        self._temp_connection = QGraphicsPathItem(path)
        drag_color = QColor(get_color("connection", "#60a5fa"))
        self._temp_connection.setPen(QPen(drag_color, 2.5))
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

        # For reconnection, target must match the disconnected end direction.
        if self._reconnect_end == "start":
            # Reconnecting the start (out) end.
            if target_port.data(1) != "out":
                log_warning("Reconnect target must be an output port")
                self._cancel_reconnection()
                return
            if not self._can_connect_ports(
                target_port,
                self._temp_start_port,
                ignore_connection=self._reconnect_connection,
            ):
                self._cancel_reconnection()
                return
            self._reconnect_connection.out_port = target_port
        else:
            # Reconnecting the end (in) end.
            target_port = self._resolve_script_proxy_input_target(self._temp_start_port, target_port)
            if not target_port:
                self._cancel_reconnection()
                return
            if target_port.data(1) != "in":
                log_warning("Reconnect target must be an input port")
                self._cancel_reconnection()
                return
            if not self._can_connect_ports(
                self._temp_start_port,
                target_port,
                ignore_connection=self._reconnect_connection,
            ):
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

        # Restore normal visual on the reconnected line
        self._reconnect_connection.setOpacity(1.0)
        self._reconnect_connection.setPen(
            ConnectionStyleDelegate.make_pen(self._reconnect_connection, "normal"))

        # Update path
        self._reconnect_connection.update_path()
        if self._reconnect_connection.in_port and self._reconnect_connection.out_port:
            self._apply_connection_to_input(
                self._reconnect_connection.in_port,
                self._reconnect_connection.out_port
            )
            self._maybe_autowire_break_for_connection(
                self._reconnect_connection.out_port,
                self._reconnect_connection.in_port,
            )

        # Clean up temporary state (but don't restore original port since we succeeded)
        self._reconnect_connection = None
        self._reconnect_original_port = None

        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None

        self._temp_start_port = None
        self._reconnecting = False
        self._reset_connection_preview()

        log_info("Reconnection successful")
        self.regenerate_code()
        # Dirty + undo history
        self._touch_content()

    def _cancel_reconnection(self):
        """Cancel reconnection 鈥?dropping without a target deletes the connection."""
        conn = self._reconnect_connection
        original_port = self._reconnect_original_port
        removed_conn = False

        if conn:
            # Restore port ref temporarily so _detach_connection can clean both ends
            if self._reconnect_end == "start" and original_port:
                conn.out_port = original_port
            elif self._reconnect_end == "end" and original_port:
                conn.in_port = original_port

            # Fully detach and remove the old connection from the scene
            self._detach_connection(conn)
            if conn.scene() is not None:
                self.removeItem(conn)
            log_debug("Reconnection cancelled 鈥?connection removed")
            removed_conn = True

        self._reconnect_connection = None
        self._reconnect_original_port = None

        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None

        self._temp_start_port = None
        self._reconnecting = False
        self._reset_connection_preview()
        self.regenerate_code()
        # Drop-to-empty is a destructive edit (the old connection is gone).
        if removed_conn:
            self._touch_content()

    def _start_connection(self, port_item, pos):
        """Start creating connection"""
        self._temp_start_port = port_item
        self._clear_port_hover()
        self._reset_connection_preview()

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

        end_pos = self._preview_connection_target(
            pos,
            ignore_connection=self._reconnect_connection if self._reconnecting else None,
        )
        start = self._port_center(self._temp_start_port)
        path = QPainterPath()
        path.moveTo(start)

        # Bezier curve
        dx = end_pos.x() - start.x()
        path.cubicTo(
            start.x() + dx * 0.5, start.y(),
            end_pos.x() - dx * 0.5, end_pos.y(),
            end_pos.x(), end_pos.y()
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

        if not self._can_connect_ports(out_port, in_port):
            self._cancel_connection()
            return

        # Create connection
        self._create_connection(out_port, in_port)
        self._cancel_connection()

        # Update code
        self.regenerate_code()

        # Dirty + undo history
        self._touch_content()

    def _cancel_connection(self):
        """Cancel connection"""
        if self._temp_connection:
            self.removeItem(self._temp_connection)
            self._temp_connection = None
        self._temp_start_port = None
        self._reset_connection_preview()

    def _create_connection(self, out_port, in_port):
        """Create connection - using ConnectionItem"""
        in_port = self._resolve_script_proxy_input_target(out_port, in_port)
        if not in_port:
            return
        try:
            if not out_port or not in_port or not isValid(out_port) or not isValid(in_port):
                return
        except Exception:
            return
        if not self._can_connect_ports(out_port, in_port):
            return
        try:
            existing = out_port.data(2) or []
        except Exception:
            existing = []
        for conn in existing:
            if not conn or not isValid(conn):
                continue
            if conn.out_port == out_port and conn.in_port == in_port:
                return
        conn = ConnectionItem(out_port, in_port)
        self.addItem(conn)

        # Attach to ports
        self._attach_connection_safe(out_port, conn)
        self._attach_connection_safe(in_port, conn)

        out_meta = self._port_meta(out_port)
        in_meta = self._port_meta(in_port)
        log_debug(f"Connection created: {out_meta['slot']} -> {in_meta['slot']}")
        self._apply_connection_to_input(in_port, out_port)
        self._refresh_behavior_protocol_state(in_port)
        self._maybe_autowire_break_for_connection(out_port, in_port)
        self._maybe_refresh_behavior_on_flow_connect(in_port, out_port)

    def _maybe_refresh_behavior_on_flow_connect(self, in_port, out_port) -> None:
        """Refresh a Behavior node's action combo when it is flow-connected.

        When a Behavior node's flow_in port is connected to any other node, pull
        the current Start-node brand/model selection and rebuild the combo so it
        shows the right actions for the active robot.  This covers the case where
        the Behavior node was created before a Start node existed, or before the
        user had selected a model.
        """
        meta = self._port_meta(in_port) if in_port and isValid(in_port) else {}
        if meta.get("channel") != "flow":
            return
        node_item = in_port.parentItem() if in_port and isValid(in_port) else None
        if node_item is None or not isValid(node_item):
            return
        if node_item.data(11) != "Behavior":
            return
        condition_set = getattr(node_item, "_condition_set", None)
        if condition_set is None:
            return
        combo = getattr(condition_set, "combo", None)
        if combo is None:
            return
        brand, robot_type = self.get_start_node_selection()
        current = combo.currentText()
        descriptors = self._behavior_action_descriptors(brand, robot_type)
        combo.blockSignals(True)
        self._set_behavior_combo_items(combo, descriptors, current, brand, robot_type)
        combo.blockSignals(False)
        self._refresh_behavior_node_parameter_rows(node_item)
        self._sync_node_parameters(node_item)

    def _find_node_port(self, node_item, slot: str, io: str):
        if not node_item or not isValid(node_item):
            return None
        for child in node_item.childItems():
            if not child or not isValid(child):
                continue
            if child.data(0) == "port" and child.data(1) == io and child.data(3) == slot:
                return child
        return None

    def _iter_upstream_flow_nodes(self, node_item):
        if not node_item or not isValid(node_item):
            return []
        upstream = []
        for child in node_item.childItems():
            if not self._is_port(child):
                continue
            if child.data(1) != "in":
                continue
            meta = self._port_meta(child)
            if meta.get("channel") != "flow":
                continue
            conns = child.data(2) or []
            for conn in conns:
                if not conn or not isValid(conn):
                    continue
                if conn.in_port != child:
                    continue
                out_port = conn.out_port
                if not out_port or not isValid(out_port):
                    continue
                src = out_port.parentItem()
                if src and isValid(src) and src.data(10) == "node":
                    upstream.append(src)
        return upstream

    def _find_enclosing_loop_for_break(self, break_node, break_level: int = 1):
        if not break_node or not isValid(break_node):
            return None
        level = max(1, int(break_level or 1))
        frontier = [break_node]
        visited = set()
        found = 0
        while frontier:
            current = frontier.pop(0)
            for upstream in self._iter_upstream_flow_nodes(current):
                node_id = upstream.data(12)
                if node_id in visited:
                    continue
                visited.add(node_id)
                if upstream.data(11) == "Loop":
                    found += 1
                    if found >= level:
                        return upstream
                frontier.append(upstream)
        return None

    def _remove_break_exit_links(self, break_out_port):
        if not break_out_port or not isValid(break_out_port):
            return
        conns = list(break_out_port.data(2) or [])
        for conn in conns:
            if not conn or not isValid(conn):
                continue
            in_port = conn.in_port
            if not in_port or not isValid(in_port):
                continue
            if in_port.data(3) != "loop_break_in":
                continue
            self._detach_connection(conn)
            if conn.scene() is not None:
                self.removeItem(conn)

    def _auto_wire_break_node(self, break_node):
        if self._autowiring_break:
            return
        if not break_node or not isValid(break_node):
            return
        if break_node.data(11) != "Break":
            return
        self._autowiring_break = True
        try:
            break_in = self._find_node_port(break_node, "flow_in", "in")
            break_out = self._find_node_port(break_node, "break_out", "out")
            if not break_in or not break_out:
                return

            self._remove_break_exit_links(break_out)
            in_conns = [c for c in (break_in.data(2) or []) if c and isValid(c) and c.in_port == break_in]
            if not in_conns:
                return

            break_level_input = getattr(break_node, "_break_level_input", None)
            try:
                break_level = int(break_level_input.text()) if break_level_input and break_level_input.text() else 1
            except ValueError:
                break_level = 1
            target_loop = self._find_enclosing_loop_for_break(break_node, break_level)
            if not target_loop:
                return

            loop_break_in = self._find_node_port(target_loop, "loop_break_in", "in")
            if not loop_break_in:
                return
            if not self._can_connect_ports(break_out, loop_break_in, log_fail=False):
                return
            self._create_connection(break_out, loop_break_in)
        finally:
            self._autowiring_break = False

    def _maybe_autowire_break_for_connection(self, out_port, in_port):
        if not out_port or not in_port:
            return
        in_node = in_port.parentItem()
        if in_node and isValid(in_node) and in_node.data(10) == "node":
            if in_node.data(11) == "Break" and in_port.data(3) == "flow_in":
                self._auto_wire_break_node(in_node)

    def _update_all_connections(self):
        """Update all connection paths"""
        for item in self.items():
            if isinstance(item, ConnectionItem):
                item.update_path()

    def refresh_style(self):
        """Refresh theme styles across the scene"""
        self._apply_theme()
        node_border = get_color("node_border", get_color("border", "#78828c"))

        for item in self.items():
            if isinstance(item, ConnectionItem):
                item.refresh_style()
                continue

            if item.data(0) == "port":
                self._apply_port_visual(item, "normal")
                continue

            if item.data(10) == "node":
                if isinstance(item, QGraphicsRectItem):
                    self._apply_node_fill_and_header_style(
                        item,
                        str(item.data(11) or ""),
                        getattr(item, "_node_grad_seed", None),
                        bool(getattr(item, "_unique_color", False)),
                    )
                    # Reapply protocol border if this node has an active protocol state;
                    # otherwise restore the default theme border.
                    proto_state = getattr(item, "_protocol_state", None)
                    exec_status = getattr(item, "_exec_status", None)
                    if proto_state and proto_state in self._PROTOCOL_BORDER_COLORS:
                        color_hex = self._PROTOCOL_BORDER_COLORS[proto_state]
                        item.setPen(QPen(QColor(color_hex), 2.5))
                    elif exec_status and exec_status in self._STATUS_BORDER_COLORS:
                        item.setPen(QPen(QColor(self._STATUS_BORDER_COLORS[exec_status]), 2.5))
                    else:
                        item.setPen(QPen(QColor(node_border), 2))
                for child in item.childItems():
                    if isinstance(child, QGraphicsProxyWidget):
                        self._apply_proxy_widget_theme(child)
                    elif hasattr(child, "setDefaultTextColor"):
                        child.setDefaultTextColor(self._node_title_color)
                sync_layout = getattr(item, "_sync_layout_fn", None)
                if callable(sync_layout):
                    sync_layout()
                align_main = getattr(item, "_align_main_ports_fn", None)
                if callable(align_main):
                    align_main()

        self.update()

    # ------------------------------------------------------------------
    # Execution Status Badges (Cycle 1 STAGE-03)
    # ------------------------------------------------------------------

    _STATUS_BORDER_COLORS: Dict[str, str] = {
        "success": "#22c55e",
        "failed":  "#ef4444",
        "skipped": "#94a3b8",
        "running": "#3b82f6",
        "pending": "#6b7280",
    }

    def set_node_execution_status(self, node_id, status: str) -> None:
        """Apply a coloured border to the node matching *node_id*.

        Args:
            node_id: The integer node ID stored in item.data(12).
            status:  One of "success" | "failed" | "skipped" | "running" | "pending".
        """
        color_hex = self._STATUS_BORDER_COLORS.get(status)
        if not color_hex:
            return
        pen = QPen(QColor(color_hex), 2.5)
        for item in self.items():
            if isValid(item) and item.data(10) == "node" and item.data(12) == node_id:
                item._exec_status = status
                if isinstance(item, QGraphicsRectItem):
                    item.setPen(pen)
                break

    def reset_execution_statuses(self) -> None:
        """Clear all execution status badges; restore default node borders."""
        node_border = get_color("node_border", get_color("border", "#78828c"))
        default_pen = QPen(QColor(node_border), 2)
        for item in self.items():
            if not isValid(item):
                continue
            if item.data(10) == "node":
                item._exec_status = None
                if isinstance(item, QGraphicsRectItem):
                    item.setPen(default_pen)

    def get_node_item(self, node_id):
        """Return the QGraphicsRectItem for *node_id*, or None if not found.

        Args:
            node_id: The integer node ID stored in item.data(12).
        """
        for item in self.items():
            if isValid(item) and item.data(10) == "node" and item.data(12) == node_id:
                return item
        return None

    # ------------------------------------------------------------------
    # P6-T5: region warning badges
    # ------------------------------------------------------------------

    # Marker text for region warning badges. A unicode triangle is more
    # legible at small sizes than a full ⚠ glyph.
    _REGION_BADGE_TEXT = "▲"

    def set_region_warnings(self, warnings_by_node_id: dict) -> None:
        """Display a yellow ▲ badge on every node whose id appears in
        ``warnings_by_node_id``. Removes badges from any node that has
        no entry. Tooltip on the node carries the joined warning
        messages.

        Args:
            warnings_by_node_id: ``{node_id: [str, str, ...]}``. The
                node id type matches whatever item.data(12) holds (int
                or str). The validator emits string node ids; the
                canvas stores int ids — callers should normalize via
                ``int(...)`` when feeding the dict in.
        """
        # Sweep + apply. We do BOTH passes in one loop because the same
        # node item satisfies "needs badge" or "needs cleanup" exactly
        # once, and the items list is not huge.
        for item in list(self.items()):
            if not isValid(item):
                continue
            if item.data(10) != "node":
                continue
            node_id = item.data(12)
            warnings = warnings_by_node_id.get(node_id) or warnings_by_node_id.get(str(node_id))
            self._apply_region_badge(item, warnings)

    def clear_region_warnings(self) -> None:
        """Remove every region warning badge from the canvas."""
        self.set_region_warnings({})

    def _apply_region_badge(self, node_item, warnings) -> None:
        """Attach or remove a region warning badge child on *node_item*."""
        try:
            existing = getattr(node_item, "_region_badge_item", None)
        except Exception:
            existing = None

        if not warnings:
            if existing is not None:
                try:
                    if isValid(existing):
                        self.removeItem(existing)
                except Exception:
                    pass
                try:
                    node_item._region_badge_item = None
                except Exception:
                    pass
                try:
                    node_item.setToolTip("")
                except Exception:
                    pass
            return

        # Reuse the existing badge child if we already created one.
        if existing is not None and isValid(existing):
            badge = existing
        else:
            from PySide6.QtWidgets import QGraphicsSimpleTextItem
            badge = QGraphicsSimpleTextItem(self._REGION_BADGE_TEXT, parent=node_item)
            try:
                badge.setBrush(QColor(250, 200, 60))
                f = QFont()
                f.setPointSize(13)
                f.setBold(True)
                badge.setFont(f)
            except Exception:
                pass
            try:
                node_item._region_badge_item = badge
            except Exception:
                pass

        # Anchor the badge to the top-right of the node rect.
        try:
            br = node_item.boundingRect()
            badge.setPos(br.right() - 18, br.top() + 2)
        except Exception:
            pass

        # Tooltip carries the joined warning messages so a hover surfaces
        # the actual problem text.
        try:
            tooltip = "Region warnings:\n" + "\n".join(f"• {w}" for w in warnings)
            node_item.setToolTip(tooltip)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Protocol sub_dot border states (Step 4)
    # ------------------------------------------------------------------

    _PROTOCOL_BORDER_COLORS: Dict[str, str] = {
        "protocol_valid":   "#22c55e",  # green  鈥?valid protocol type connected
        "protocol_invalid": "#f59e0b",  # amber  鈥?wrong type connected to protocol port
        "protocol_none":    "#6b7280",  # grey   鈥?no connection / legacy mode
    }

    def set_behavior_protocol_state(self, node_id, state: str) -> None:
        """Apply a protocol-state border to the Behavior node matching *node_id*.

        Args:
            node_id: The integer node ID stored in item.data(12).
            state:   One of "protocol_valid" | "protocol_invalid" | "protocol_none".
        """
        color_hex = self._PROTOCOL_BORDER_COLORS.get(state)
        if not color_hex:
            return
        pen = QPen(QColor(color_hex), 2.5)
        for item in self.items():
            if isValid(item) and item.data(10) == "node" and item.data(12) == node_id:
                item._protocol_state = state
                if isinstance(item, QGraphicsRectItem):
                    item.setPen(pen)
                break

    def apply_behavior_protocol_states_from_run_result(self, run_result: dict) -> None:
        """Apply runtime-authoritative protocol border states to Behavior nodes.

        Called after mission execution completes.  Runtime result takes priority
        over design-time type-check state.

        Mapping (protocol_status from per-node results):
          "invalid" | "stale" 鈫?protocol_invalid  (amber)
          "valid"              鈫?protocol_valid    (green)
          "absent" / missing   鈫?protocol_none     (grey)

        Args:
            run_result: The dict returned by RuntimeEngine.execute() / MissionRunThread.
        """
        results = run_result.get("results") or {}
        for node_id, node_result in results.items():
            if not isinstance(node_result, dict):
                continue
            proto_status = node_result.get("protocol_status", "absent")
            reason = node_result.get("reason", "")

            if proto_status in ("invalid", "stale") or reason == "INVALID_PROTOCOL":
                state = "protocol_invalid"
            elif proto_status == "valid":
                state = "protocol_valid"
            else:
                state = "protocol_none"

            # Apply only to Behavior nodes (data(11) == "Behavior").
            for item in self.items():
                if (
                    isValid(item)
                    and item.data(10) == "node"
                    and str(item.data(12)) == str(node_id)
                    and item.data(11) == "Behavior"
                ):
                    self.set_behavior_protocol_state(item.data(12), state)
                    break

    def _refresh_behavior_protocol_state(self, in_port) -> None:
        """Recompute and apply the protocol border for the Behavior node owning *in_port*.

        Called after any connection add/remove on the "control_pipe" slot of a Behavior node.
        State rules:
          - No connection         鈫?protocol_none
          - Connected, type match 鈫?protocol_valid   (out_port data_type == "protocol")
          - Connected, type mismatch 鈫?protocol_invalid
        """
        try:
            if not in_port or not isValid(in_port):
                return
            if in_port.data(3) != "control_pipe":
                return
            node_item = in_port.parentItem()
            if not node_item or not isValid(node_item):
                return
            if node_item.data(11) != "Behavior":
                return
            node_id = node_item.data(12)

            conns = [c for c in (in_port.data(2) or []) if c and isValid(c)]
            if not conns:
                self.set_behavior_protocol_state(node_id, "protocol_none")
                return

            # Check if all connected output ports are protocol-typed.
            all_protocol = all(
                _normalize_data_type(
                    (c.out_port.data(self._port_meta_key) or {}).get("data_type", "any")
                ) == "protocol"
                for c in conns
                if c.out_port and isValid(c.out_port)
            )
            state = "protocol_valid" if all_protocol else "protocol_invalid"
            self.set_behavior_protocol_state(node_id, state)
        except Exception as e:
            log_debug(f"_refresh_behavior_protocol_state error: {e}")

    def _apply_proxy_widget_theme(self, proxy: QGraphicsProxyWidget):
        widget = proxy.widget()
        if not widget:
            return

        combo_no_arrow = """
            QComboBox::drop-down {
                border: 0px;
                width: 0px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
        """

        if isinstance(widget, QComboBox):
            if not bool(widget.property("keep_own_style")):
                style = self._combo_style + self._combo_view_style
                if bool(widget.property("main_picker_combo")):
                    mp_text = getattr(self, "_main_picker_text", "#000000")
                    style += (
                        "QComboBox {"
                        f" color: {mp_text};"
                        "} "
                        "QComboBox QAbstractItemView {"
                        f" color: {mp_text};"
                        "}"
                    )
                if bool(widget.property("hide_native_arrow")):
                    style += combo_no_arrow
                widget.setStyleSheet(style)
        if isinstance(widget, QLineEdit):
            if not bool(widget.property("keep_own_style")):
                widget.setStyleSheet(self._input_style)
                comp = widget.completer()
                if comp is not None and comp.popup() is not None:
                    comp.popup().setStyleSheet(self._completer_popup_style)
        if isinstance(widget, QPushButton):
            if not bool(widget.property("keep_own_style")):
                widget.setStyleSheet(self._button_style)
        if isinstance(widget, QLabel) and widget.objectName() == "nodeTag":
            widget.setStyleSheet(self._tag_style)
        if isinstance(widget, ScriptSetting):
            widget.set_theme_colors(
                self._condition_set_left_bg,
                self._input_bg,
                self._button_border,
                self._condition_set_text,
                self._hover_bg,
                hint_bg_color=self._condition_set_right_bg,
            )
        elif isinstance(widget, BehaviorSetting):
            widget.set_theme_colors(
                self._condition_set_left_bg,
                self._input_bg,
                self._button_border,
                self._condition_set_text,
                self._hover_bg,
            )
        elif isinstance(widget, ScriptInAndOut):
            widget.set_theme_colors(
                self._condition_set_left_bg,
                self._input_bg,
                self._button_border,
                self._condition_set_text,
                getattr(self, "_node_base_color", get_color("card_bg", "#2d2d2d")),
            )

        for combo in widget.findChildren(QComboBox):
            if bool(combo.property("keep_own_style")):
                continue
            style = self._combo_style + self._combo_view_style
            if bool(combo.property("main_picker_combo")):
                mp_text = getattr(self, "_main_picker_text", "#000000")
                style += (
                    "QComboBox {"
                    f" color: {mp_text};"
                    "} "
                    "QComboBox QAbstractItemView {"
                    f" color: {mp_text};"
                    "}"
                )
            if bool(combo.property("hide_native_arrow")):
                style += combo_no_arrow
            combo.setStyleSheet(style)
        for line_edit in widget.findChildren(QLineEdit):
            if bool(line_edit.property("keep_own_style")):
                continue
            line_edit.setStyleSheet(self._input_style)
            comp = line_edit.completer()
            if comp is not None and comp.popup() is not None:
                comp.popup().setStyleSheet(self._completer_popup_style)
        for btn in widget.findChildren(QPushButton):
            if bool(btn.property("keep_own_style")):
                continue
            btn.setStyleSheet(self._button_style)
        for lbl in widget.findChildren(QLabel):
            if lbl.objectName() == "nodeTag":
                lbl.setStyleSheet(self._tag_style)

    def create_node(self, name: str, scene_pos: QPointF,
                    features: List[str] = None, grad: tuple = None, uniqueColor: bool = False):
        """
        Create node

        Args:
            name: Node name
            scene_pos: Scene position
            features: Feature list
            grad: Gradient colors (color1, color2)
            uniqueColor: Whether to keep legacy full-node unique coloring.
        """
        # Normalise palette titles to canonical internal names so that every
        # downstream check (branch selection, gradient, logic-node mapping)
        # works regardless of how the node was created.
        _PALETTE_ALIASES = {
            "If / Elif / Else": "Logic Control",
            "Compare(op, left, right)": "Condition",
            "MathOp(operation, a, b)": "Math",
            "Timer(duration, unit)": "Timer",
            "ExecuteBehavior(subgraph_id)": "Behavior",
            "ReadSensor(sensor_type)": "Sensor",
            "TriggerPlugin(api_name, args)": "Trigger",
            "Wait(duration/event)": "Wait",
            "Gate(condition/event)": "Gate",
            "Timeout(duration)": "Timeout",
            "Retry(max_attempts, backoff)": "Retry",
            "Break(loop)": "Break",
            "Script Box": "Script Box",
            "OnEvent(event_name)": "OnEvent",
            "PublishEvent(event_name, payload)": "PublishEvent",
            "SetState(key, value)": "SetState",
            "CheckState(key, expected)": "CheckState",
            "HumanConfirm(prompt, timeout)": "HumanConfirm",
            "Switch / Case": "Switch",
            "Try / Catch / Finally": "Try",
            "Parallel Branch": "Parallel Branch",
            # Legacy names redirect to unified Loop node
            "While Loop": "Loop",
            "For Loop": "Loop",
        }
        for alias, canonical in _PALETTE_ALIASES.items():
            if alias in name:
                name = canonical
                break

        # Deprecated node types must not be creatable in the Mission Canvas.
        _DEPRECATED_NAMES = {"Policy", "Conductor"}
        if name in _DEPRECATED_NAMES:
            log_debug(f"create_node: '{name}' is deprecated and cannot be created in Mission Canvas")
            return None

        # --- Sizing ---
        # Initial w/h are placeholders; _resize_to_fit() derives the real
        # size from measured content after all rows have been added.
        w = 100
        h = 80

        # Create node rectangle (border-resizable)
        rect = ResizableNodeItem(0, 0, w, h)

        resolved_grad = self._resolve_node_gradient(name, grad)

        rect.setPen(QPen(QColor(get_color("node_border", get_color("border", "#78828c"))), 2))
        rect.setFlag(QGraphicsItem.ItemIsMovable, True)
        rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
        rect.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        rect.setPos(scene_pos - QPointF(w / 2, h / 2))
        self.addItem(rect)

        _ROW_HEIGHT = 24
        _ROW_SPACING = 6
        _NODE_PADDING_X = 0
        _NODE_PADDING_Y = 0
        _NODE_PADDING_RIGHT = 0
        _NODE_PADDING_BOTTOM = 0

        row_stack = NodeRowStack(
            spacing=_ROW_SPACING,
            row_height=_ROW_HEIGHT,
            show_row_border=False,
            row_border_color="#ff2bd6",
        )
        # Keep a reference for robust load-time row collapse/expand restoration.
        rect._row_stack = row_stack
        widget_container = row_stack.container
        widget_factory = NodeWidgetFactory(
            self._input_style,
            self._button_style,
            self._combo_style,
            self._combo_view_style,
        )
        node_padding_x = _NODE_PADDING_X
        node_padding_y = _NODE_PADDING_Y
        node_padding_right = _NODE_PADDING_RIGHT
        node_padding_bottom = _NODE_PADDING_BOTTOM

        header_row_widget = QWidget()
        header_row_widget.setObjectName("nodeHeaderRow")
        header_row_widget.setStyleSheet("QWidget { border: 0px; background: transparent; }")
        header_row_layout = QHBoxLayout(header_row_widget)
        header_row_layout.setContentsMargins(10, 3, 8, 0)
        header_row_layout.setSpacing(6)

        title_text = re.sub(r"\s*\([^)]*\)", "", str(name)).strip()
        if not title_text:
            title_text = str(name)
        if "Logic Control" in name:
            title_text = "IF"
        elif name == "Loop":
            title_text = "LOOP"
        elif "Switch" in name:
            title_text = "SWITCH"
        elif "Try" in name and "/" in name:
            title_text = "TRY"
        elif "Parallel" in name:
            title_text = "PARALLEL"
        elif name == "Join" or name == "Join":
            title_text = "JOIN"
        header_title = QLabel(title_text)
        header_title.setStyleSheet(
            f"QLabel {{ color: {get_color('text_primary', '#e5e7eb')}; "
            "font-size: 12px; font-weight: 700; background: transparent; border: none; }"
        )
        header_row_layout.addWidget(header_title)
        header_row_layout.addStretch(1)

        open_subgraph_controls = []

        delete_btn = widget_factory.button("x", size=20)
        header_row_layout.addWidget(delete_btn)
        row_stack.add_row("header_row", header_row_widget)

        proxy = QGraphicsProxyWidget(rect)
        proxy.setWidget(widget_container)
        proxy.setPos(node_padding_x, node_padding_y)
        proxy.setZValue(2)
        rect.register_hover_passthrough(proxy)
        rect._header_row_widget = header_row_widget
        self._apply_node_fill_and_header_style(rect, name, grad_seed=grad, unique_color=uniqueColor)

        def _port_y(widget):
            if not widget:
                return None
            if hasattr(widget, "center_y"):
                return widget.center_y(proxy)
            geo = widget.geometry()
            return proxy.pos().y() + geo.y() + geo.height() / 2

        def _apply_content_width():
            content_w = max(96, int(rect.rect().width() - node_padding_x - node_padding_right))
            widget_container.setFixedWidth(content_w)

        def _measure_rows():
            """Measure minimum content size from actual row sizeHints.

            Uses sizeHint only 鈥?never calls adjustSize() which would force
            rows to shrink during interactive resize.
            """
            layout = row_stack.layout
            spacing = max(0, int(layout.spacing()))
            rows = list(getattr(row_stack, "_rows", {}).values())
            if not rows:
                return 0, 0

            max_row_w = 0
            total_h = 0
            visible_rows = 0
            for row in rows:
                if not row or row.isHidden():
                    continue
                row_layout = row.layout()
                if row_layout is not None:
                    margins = row_layout.contentsMargins()
                    lay_min = row_layout.minimumSize()
                    row_min_hint = row.minimumSizeHint()
                    row_w = int(max(
                        lay_min.width() + margins.left() + margins.right(),
                        row_min_hint.width(),
                        int(row.minimumWidth()),
                    ))
                    # Use the row widget's own constraints as source of truth.
                    # Layout hint can under-report fixed-height rows, which leads
                    # to node boxes being too short and visual overflow.
                    fixed_h = int(row.minimumHeight())
                    max_h = int(row.maximumHeight())
                    if max_h > 0 and max_h < 16777215:
                        fixed_h = max(fixed_h, max_h)
                    lay_hint = row_layout.sizeHint()
                    row_hint_h = int(lay_hint.height() + margins.top() + margins.bottom())
                    row_h = max(row_hint_h, fixed_h, int(row.sizeHint().height()), 24)
                else:
                    min_hint = row.minimumSizeHint()
                    hint = row.sizeHint()
                    row_w = max(int(min_hint.width()), int(row.minimumWidth()))
                    row_h = max(int(hint.height()), int(min_hint.height()), int(row.minimumHeight()))
                max_row_w = max(max_row_w, row_w)
                total_h += row_h
                visible_rows += 1

            if visible_rows > 1:
                total_h += spacing * (visible_rows - 1)
            return max_row_w, total_h

        def _widget_center_in_node(widget: QWidget) -> Optional[QPointF]:
            """Map nested QWidget center to node-local coordinates."""
            if not widget or widget_container is None:
                return None
            center_local = QPoint(widget.width() // 2, widget.height() // 2)
            try:
                center_in_container = widget.mapTo(widget_container, center_local)
            except Exception:
                return None
            return QPointF(
                float(proxy.pos().x() + center_in_container.x()),
                float(proxy.pos().y() + center_in_container.y()),
            )

        def _min_size_for_content():
            row_w, row_h = _measure_rows()
            min_w = int(row_w + node_padding_x + node_padding_right)
            min_h = int(row_h + node_padding_y + node_padding_bottom)
            return min_w, min_h

        _NODE_MIN_W = 100
        _NODE_MIN_H = 80

        def _resize_to_fit(min_height: Optional[int] = None, min_width: Optional[int] = None):
            min_content_w, min_content_h = _min_size_for_content()
            target_w = max(int(min_content_w), _NODE_MIN_W, int(min_width or 0))
            target_h = max(int(min_content_h), _NODE_MIN_H, int(min_height or 0))
            rect.setRect(0, 0, target_w, target_h)
            _apply_content_width()

        port_r = 6

        def _mk_port(
            x,
            y,
            io,
            slot,
            radius=None,
            *,
            channel: Optional[str] = None,
            data_type: str = "any",
            max_connections: Optional[int] = None,
            dot_kind: Optional[str] = None,
        ):
            resolved_channel = channel or ("flow" if slot in _FLOW_SLOTS or slot.startswith("out_elif") or slot.startswith("branch_") or slot.startswith("case_") else "data")
            resolved_dot = dot_kind or ("main_dot" if resolved_channel == "flow" else "sub_dot")
            resolved_max = max_connections
            if resolved_max is None and io == "in":
                resolved_max = 1
            is_fan_in = (resolved_max is not None and resolved_max < 0)
            resolved_radius = radius if radius is not None else (port_r if resolved_dot == "main_dot" else 4)
            if is_fan_in:
                resolved_radius += 1
            port_cls = _FanInPortItem if is_fan_in else QGraphicsEllipseItem
            p = port_cls(-resolved_radius, -resolved_radius, resolved_radius * 2, resolved_radius * 2, rect)
            p.setPos(x, y)
            p.setData(0, "port")
            p.setData(1, io)
            p.setData(2, [])
            p.setData(3, slot)
            p.setData(self._port_meta_key, {
                "channel": resolved_channel,
                "data_type": data_type,
                "max_connections": resolved_max,
                "dot_kind": resolved_dot,
            })
            p.setZValue(3)
            p.setAcceptedMouseButtons(Qt.LeftButton)
            p.setAcceptHoverEvents(True)
            p.setCursor(Qt.PointingHandCursor)
            self._apply_port_visual(p, "normal")
            return p

        def _make_row(port_mode: str = "none"):
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 2, 8, 2)
            row_layout.setSpacing(6)
            if port_mode in ("output", "both"):
                row_layout.addStretch(1)
            return row_widget, row_layout

        def _finish_row(row_layout: QHBoxLayout, port_mode: str = "none"):
            if port_mode in ("input", "both"):
                row_layout.addStretch(1)

        def _set_row_visible_compact(row_widget: QWidget, visible: bool):
            """Show/hide a row without leaving spacing residue in QVBoxLayout.

            IMPORTANT:
            For dynamic node UIs (Wait/Gate/Cancel...), do not rely on
            QWidget.setVisible(False) alone. The row must be collapsed via
            NodeRowStack so hidden rows do not keep phantom layout gaps.
            """
            if not row_widget:
                return
            row_id = row_widget.property("_row_id")
            if row_id and hasattr(row_stack, "set_row_collapsed"):
                row_stack.set_row_collapsed(str(row_id), not visible)
                return
            row_widget.setVisible(visible)
            if visible:
                row_widget.setMinimumHeight(_ROW_HEIGHT)
                row_widget.setMaximumHeight(_ROW_HEIGHT)
            else:
                row_widget.setMinimumHeight(0)
                row_widget.setMaximumHeight(0)

        def _make_input_setting_row(
            row_id: str,
            key_text: str,
            placeholder: str,
            *,
            show_left_dot: bool = False,
            show_right_dot: bool = False,
            input_slot: Optional[str] = None,
        ):
            setting = InputSetting(
                self._condition_set_left_bg,
                self._condition_set_right_bg,
                get_color("input_bg", "#0f1115"),
                self._button_border,
                self._condition_set_text,
                key_text=key_text,
                placeholder=placeholder,
                show_left_dot=show_left_dot,
                show_right_dot=show_right_dot,
            )
            row_widget, row_layout = _make_row("none")
            row_layout.addWidget(setting, 1)
            _finish_row(row_layout, "none")
            row_stack.add_row(row_id, row_widget)
            if input_slot:
                self._register_bound_input_widget(rect, input_slot, setting)
            return row_widget, setting

        def _make_combo_setting_row(
            row_id: str,
            key_text: str,
            items,
            *,
            show_left_dot: bool = False,
            show_right_dot: bool = False,
        ):
            setting = ComboSetting(
                self._condition_set_left_bg,
                self._condition_set_right_bg,
                get_color("input_bg", "#0f1115"),
                self._button_border,
                self._condition_set_text,
                key_text=key_text,
                items=items,
                show_left_dot=show_left_dot,
                show_right_dot=show_right_dot,
            )
            row_widget, row_layout = _make_row("none")
            row_layout.addWidget(setting, 1)
            _finish_row(row_layout, "none")
            row_stack.add_row(row_id, row_widget)
            return row_widget, setting

        def _make_behavior_setting_row(
            row_id: str,
            items,
            *,
            show_left_dot: bool = False,
            show_advanced_button: bool = True,
            combo_editable: bool = False,
        ):
            setting = BehaviorSetting(
                self._condition_set_left_bg,
                get_color("input_bg", "#0f1115"),
                self._button_border,
                self._condition_set_text,
                self._hover_bg,
                items=items,
                show_left_dot=show_left_dot,
                show_advanced_button=show_advanced_button,
                combo_editable=combo_editable,
            )
            row_widget, row_layout = _make_row("none")
            row_layout.addWidget(setting, 1)
            _finish_row(row_layout, "none")
            row_stack.add_row(row_id, row_widget)
            return row_widget, setting

        def _make_script_setting_row(
            row_id: str,
            items,
            *,
            show_left_dot: bool = False,
            show_advanced_button: bool = True,
            combo_editable: bool = False,
        ):
            setting = ScriptSetting(
                self._condition_set_left_bg,
                get_color("input_bg", "#0f1115"),
                self._button_border,
                self._condition_set_text,
                self._hover_bg,
                hint_bg_color=self._condition_set_right_bg,
                items=items,
                show_left_dot=show_left_dot,
                show_advanced_button=show_advanced_button,
                combo_editable=combo_editable,
            )
            row_widget, row_layout = _make_row("none")
            row_layout.addWidget(setting, 1)
            _finish_row(row_layout, "none")
            row_stack.add_row(row_id, row_widget)
            return row_widget, setting

        def _make_script_io_row(row_id: str):
            io_widget = ScriptInAndOut(
                self._condition_set_left_bg,
                get_color("input_bg", "#0f1115"),
                self._button_border,
                self._condition_set_text,
                getattr(self, "_node_base_color", get_color("card_bg", "#2d2d2d")),
            )
            row_widget, row_layout = _make_row("none")
            row_layout.addWidget(io_widget, 1)
            _finish_row(row_layout, "none")
            row_stack.add_row(row_id, row_widget)
            # Script IO region must be vertically elastic (no fixed single-row clamp).
            row_widget.setMinimumHeight(56)
            row_widget.setMaximumHeight(16777215)
            return row_widget, io_widget

        resize_ports_cb = None
        align_main_ports_cb = None

        def _align_main_flow_ports_to_title_row():
            """Default rule: keep main flow in/out on title row to free content rows.

            Applies only to canonical flow ports: `flow_in` and `flow_out`.
            Branching nodes keep their existing row-aligned flow layout.
            """
            if not isValid(rect):
                return
            header_y = _port_y(header_row_widget)
            if header_y is None:
                return
            w_now = rect.rect().width()
            for child in rect.childItems():
                if not self._is_port(child):
                    continue
                slot = str(child.data(3) or "")
                if slot not in ("flow_in", "flow_out"):
                    continue
                meta = self._port_meta(child)
                if meta.get("channel") != "flow" or meta.get("dot_kind") != "main_dot":
                    continue
                io = child.data(1)
                if io == "in":
                    child.setPos(0, header_y)
                elif io == "out":
                    child.setPos(w_now, header_y)

        if "Logic Control" in name:
            condition_set = ConditionSet(
                self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                self._condition_set_text, self._hover_bg, "< logic >",
            )

            add_elif_btn = widget_factory.button("+", size=20)
            header_row_layout.insertWidget(header_row_layout.count() - 1, add_elif_btn)

            if_row_widget, if_row = _make_row("none")
            if_row.addWidget(condition_set)
            _finish_row(if_row, "input")
            row_stack.add_row("if_row", if_row_widget)

            else_row_widget, else_row_layout = _make_row("output")
            else_row_layout.addWidget(make_tag("else", self._tag_style))
            row_stack.add_row("else_row", else_row_widget)

            flow_in_port = _mk_port(0, h * 0.42, "in", "flow_in", radius=6, channel="flow")
            condition_port = _mk_port(12, h * 0.42, "in", "condition", channel="data", data_type="bool", dot_kind="sub_dot")
            out_if = _mk_port(w, h * 0.28, "out", "out_if", radius=6, channel="flow")
            out_else = _mk_port(w, h * 0.78, "out", "out_else", radius=6, channel="flow")

            rect._elif_output_ports = []
            rect._elif_condition_ports = []
            rect._elif_rows = []
            rect._elif_inputs = []

            def _remove_port_connections(port):
                conns = port.data(2) or []
                for conn in list(conns):
                    if conn and isValid(conn) and conn.scene() is not None:
                        self.removeItem(conn)

            def _reindex_elifs():
                for i, row in enumerate(rect._elif_rows):
                    input_local = getattr(row, "_elif_condition_input", None)
                    if input_local and hasattr(input_local, "set_logic_text"):
                        input_local.set_logic_text(f"< elif {i} >")
                for i, port in enumerate(rect._elif_output_ports):
                    port.setData(3, f"out_elif_{i}")

            def _add_elif():
                idx = len(rect._elif_output_ports)
                outp = _mk_port(w, h * 0.60, "out", f"out_elif_{idx}", radius=6, channel="flow")
                rect._elif_output_ports.append(outp)
                cond_p = _mk_port(12, h * 0.60, "in", f"elif_cond_{idx}", channel="data", data_type="bool", dot_kind="sub_dot")
                rect._elif_condition_ports.append(cond_p)

                remove_btn = widget_factory.button("x", size=20)

                elif_cond = ConditionSet(
                    self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                    self._condition_set_text, self._hover_bg,
                    f"< elif {idx} >",
                )
                row_widget, row_layout = _make_row("output")
                row_layout.addWidget(elif_cond)
                row_layout.addWidget(remove_btn)
                row_widget._elif_title_label = None
                row_widget._elif_condition_input = elif_cond

                rect._elif_rows.append(row_widget)
                rect._elif_inputs.append(elif_cond)

                def _remove_this():
                    idx_local = rect._elif_rows.index(row_widget) if row_widget in rect._elif_rows else -1
                    if idx_local < 0:
                        return
                    row_stack.remove_row(f"elif_row_{idx_local}")
                    port = rect._elif_output_ports[idx_local]
                    _remove_port_connections(port)
                    if port.scene() is not None:
                        self.removeItem(port)
                    cp = rect._elif_condition_ports[idx_local]
                    _remove_port_connections(cp)
                    if cp.scene() is not None:
                        self.removeItem(cp)
                    rect._elif_rows.pop(idx_local)
                    rect._elif_output_ports.pop(idx_local)
                    rect._elif_condition_ports.pop(idx_local)
                    rect._elif_inputs.pop(idx_local)
                    _reindex_elifs()
                    QTimer.singleShot(0, _sync_layout)
                    self.regenerate_code()

                remove_btn.clicked.connect(_remove_this)
                elif_cond.button.clicked.connect(lambda: self._update_node_params(rect))

                row_stack.add_row(f"elif_row_{idx}", row_widget, before_row_id="else_row")
                QTimer.singleShot(0, _sync_layout)
                self.regenerate_code()

            rect._add_elif = _add_elif
            add_elif_btn.clicked.connect(_add_elif)

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_if = _port_y(if_row_widget)
                if y_if is not None:
                    flow_in_port.setPos(0, y_if)
                    cond_center = _widget_center_in_node(condition_set.left_box)
                    if cond_center is not None:
                        condition_port.setPos(cond_center.x(), cond_center.y())
                    out_if.setPos(w_now, y_if)
                y_else = _port_y(else_row_widget)
                if y_else is not None:
                    out_else.setPos(w_now, y_else)
                for idx, row in enumerate(rect._elif_rows):
                    if idx >= len(rect._elif_output_ports):
                        continue
                    y = _port_y(row)
                    if y is not None:
                        rect._elif_output_ports[idx].setPos(w_now, y)
                    if idx < len(rect._elif_condition_ports) and idx < len(rect._elif_inputs):
                        elif_cs = rect._elif_inputs[idx]
                        if hasattr(elif_cs, 'left_box'):
                            cc = _widget_center_in_node(elif_cs.left_box)
                            if cc is not None:
                                rect._elif_condition_ports[idx].setPos(cc.x(), cc.y())

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)
            rect._sync_layout_fn = _sync_layout
            resize_ports_cb = _sync_ports
            rect._condition_set = condition_set
            rect._condition_input = condition_set.button
            condition_set.button.clicked.connect(lambda: self._update_node_params(rect))

        elif name == "Loop":
            # --- Type picker (While / For) ---
            type_row, type_picker = _make_combo_setting_row(
                "loop_type_row",
                "type",
                items=["While", "For"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # --- While-mode rows ---
            condition_set = ConditionSet(
                self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                self._condition_set_text, self._hover_bg, "< logic >",
            )
            while_row1, while_row1_layout = _make_row("none")
            while_row1_layout.addWidget(condition_set)
            _finish_row(while_row1_layout, "input")
            row_stack.add_row("loop_while_cond", while_row1)

            # --- For-mode rows ---
            for_start_row, for_start_setting = _make_input_setting_row(
                "loop_for_start",
                "start",
                "0",
                show_left_dot=True,
                show_right_dot=True,
                input_slot="for_start",
            )
            start_input = for_start_setting.input_edit
            start_input.setValidator(QIntValidator(-9999, 9999, start_input))

            for_end_row, for_end_setting = _make_input_setting_row(
                "loop_for_end",
                "to",
                "10",
                show_left_dot=True,
                show_right_dot=True,
                input_slot="for_end",
            )
            end_input = for_end_setting.input_edit
            end_input.setValidator(QIntValidator(-9999, 9999, end_input))

            for_step_row, for_step_setting = _make_input_setting_row(
                "loop_for_step",
                "step",
                "1",
                show_left_dot=True,
                show_right_dot=True,
                input_slot="for_step",
            )
            step_input = for_step_setting.input_edit
            step_input.setValidator(QIntValidator(-99, 99, step_input))

            # Keep break row at the bottom for both While/For modes.
            break_set = OutputSet(
                self._condition_set_left_bg,
                self._condition_set_right_bg,
                self._button_border,
                self._condition_set_text,
                "if Break",
            )
            break_row, break_row_layout = _make_row("none")
            break_row_layout.addWidget(break_set)
            _finish_row(break_row_layout, "none")
            row_stack.add_row("loop_break_row", break_row)

            # --- Ports ---
            flow_in_port = _mk_port(0, h * 0.35, "in", "flow_in", radius=6, channel="flow")
            condition_port = _mk_port(12, h * 0.35, "in", "condition", channel="data",
                                      data_type="bool", dot_kind="sub_dot")
            loop_body = _mk_port(w, h * 0.35, "out", "loop_body", radius=6, channel="flow")
            for_start_in = _mk_port(12, h * 0.45, "in", "for_start_in", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            for_start_out = _mk_port(w - 12, h * 0.45, "out", "for_start_out", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            for_end_in = _mk_port(12, h * 0.55, "in", "for_end_in", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            for_end_out = _mk_port(w - 12, h * 0.55, "out", "for_end_out", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            for_step_in = _mk_port(12, h * 0.65, "in", "for_step_in", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            for_step_out = _mk_port(w - 12, h * 0.65, "out", "for_step_out", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
            loop_break_in = _mk_port(w - 12, h * 0.75, "in", "loop_break_in", radius=4, channel="flow",
                                     max_connections=-1, dot_kind="sub_dot")
            loop_end = _mk_port(w, h * 0.75, "out", "loop_end", radius=6, channel="flow")

            def _sync_ports_loop():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                is_for = type_picker.combo.currentText() == "For"
                y1 = _port_y(for_start_row if is_for else while_row1)
                if y1 is not None:
                    flow_in_port.setPos(0, y1)
                    if not is_for:
                        cc = _widget_center_in_node(condition_set.left_box)
                        if cc is not None:
                            condition_port.setPos(cc.x(), cc.y())
                y_body = _port_y(type_row) if is_for else _port_y(while_row1)
                if y_body is not None:
                    loop_body.setPos(w_now, y_body)
                y2 = _port_y(break_row)
                if y2 is not None:
                    loop_end.setPos(w_now, y2)
                if is_for:
                    y_start = _port_y(for_start_row)
                    if y_start is not None:
                        in_center = _widget_center_in_node(for_start_setting.left_box) if for_start_setting.left_box else None
                        out_center = _widget_center_in_node(for_start_setting.right_box) if for_start_setting.right_box else None
                        for_start_in.setPos(in_center.x(), in_center.y()) if in_center is not None else for_start_in.setPos(12, y_start)
                        for_start_out.setPos(out_center.x(), out_center.y()) if out_center is not None else for_start_out.setPos(w_now - 12, y_start)
                    y_end = _port_y(for_end_row)
                    if y_end is not None:
                        in_center = _widget_center_in_node(for_end_setting.left_box) if for_end_setting.left_box else None
                        out_center = _widget_center_in_node(for_end_setting.right_box) if for_end_setting.right_box else None
                        for_end_in.setPos(in_center.x(), in_center.y()) if in_center is not None else for_end_in.setPos(12, y_end)
                        for_end_out.setPos(out_center.x(), out_center.y()) if out_center is not None else for_end_out.setPos(w_now - 12, y_end)
                    y_step = _port_y(for_step_row)
                    if y_step is not None:
                        in_center = _widget_center_in_node(for_step_setting.left_box) if for_step_setting.left_box else None
                        out_center = _widget_center_in_node(for_step_setting.right_box) if for_step_setting.right_box else None
                        for_step_in.setPos(in_center.x(), in_center.y()) if in_center is not None else for_step_in.setPos(12, y_step)
                        for_step_out.setPos(out_center.x(), out_center.y()) if out_center is not None else for_step_out.setPos(w_now - 12, y_step)
                y_break = _port_y(break_row)
                if y_break is not None:
                    break_center = _widget_center_in_node(break_set.left_box)
                    if break_center is not None:
                        loop_break_in.setPos(break_center.x(), break_center.y())
                    else:
                        loop_break_in.setPos(12, y_break)
                if isValid(condition_port):
                    condition_port.setVisible(not is_for)
                for p in (for_start_in, for_start_out, for_end_in, for_end_out, for_step_in, for_step_out):
                    if isValid(p):
                        p.setVisible(is_for)

            def _apply_loop_mode(mode: str):
                is_for = (mode == "For")
                _set_row_visible_compact(while_row1, not is_for)
                _set_row_visible_compact(for_start_row, is_for)
                _set_row_visible_compact(for_end_row, is_for)
                _set_row_visible_compact(for_step_row, is_for)

            def _sync_layout_loop():
                _resize_to_fit()
                _sync_ports_loop()

            def _on_loop_type_change(_idx):
                mode = type_picker.combo.currentText()
                _apply_loop_mode(mode)
                QTimer.singleShot(0, _sync_layout_loop)
                self._update_node_params(rect)

            type_picker.combo.currentIndexChanged.connect(_on_loop_type_change)

            QTimer.singleShot(0, _sync_layout_loop)
            rect._sync_layout_fn = _sync_layout_loop
            resize_ports_cb = _sync_ports_loop

            rect._loop_type_picker = type_picker
            rect._loop_while_rows = [while_row1]
            rect._loop_for_rows = [for_start_row, for_end_row, for_step_row]
            rect._condition_set = condition_set
            rect._condition_input = condition_set.button
            rect._for_start_input = start_input
            rect._for_end_input = end_input
            rect._for_step_input = step_input

            condition_set.button.clicked.connect(lambda: self._update_node_params(rect))
            start_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            end_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            step_input.textChanged.connect(lambda _t: self._update_node_params(rect))

            # Initialise to While mode
            _apply_loop_mode("While")

        elif name == "Break":
            level_row, level_setting = _make_input_setting_row(
                "break_level_row",
                "level",
                "1",
                show_left_dot=False,
                show_right_dot=True,
            )
            level_input = level_setting.input_edit
            level_input.setValidator(QIntValidator(1, 99, level_input))

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            break_out_port = _mk_port(w, h / 2, "out", "break_out", radius=4, channel="flow", dot_kind="sub_dot")

            def _sync_ports_break():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(level_row)
                if y is not None:
                    flow_in_port.setPos(0, y)
                    out_center = _widget_center_in_node(level_setting.right_box) if level_setting.right_box else None
                    break_out_port.setPos(out_center.x(), out_center.y()) if out_center is not None else break_out_port.setPos(w_now, y)

            def _sync_layout_break():
                _resize_to_fit()
                _sync_ports_break()

            QTimer.singleShot(0, _sync_layout_break)
            rect._sync_layout_fn = _sync_layout_break
            resize_ports_cb = _sync_ports_break
            rect._break_level_input = level_input

            def _on_break_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self._auto_wire_break_node(rect)
                self.regenerate_code()

            level_input.textChanged.connect(lambda _t: _on_break_change())

        elif "Condition" in name:
            condition_set = ConditionSet(
                self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                self._condition_set_text, self._hover_bg, "< logic >",
            )

            left_row, left_setting = _make_input_setting_row(
                "left_row",
                "left",
                "left",
                show_left_dot=True,
                show_right_dot=False,
                input_slot="left",
            )
            left_input = left_setting.input_edit
            right_row_widget, right_setting = _make_input_setting_row(
                "right_row",
                "right",
                "right",
                show_left_dot=True,
                show_right_dot=True,
                input_slot="right",
            )
            right_input_widget = right_setting.input_edit

            cond_row_widget, cond_row_layout = _make_row("input")
            cond_row_layout.addWidget(condition_set)
            _finish_row(cond_row_layout, "input")

            row_stack.add_row("op_row", cond_row_widget)

            left_port = _mk_port(12, h * 0.45, "in", "left", radius=4, channel="data")
            right_port = _mk_port(12, h * 0.70, "in", "right", radius=4, channel="data")
            result_port = _mk_port(w - 12, h / 2, "out", "result", radius=4, channel="data", data_type="bool")
            condition_port = _mk_port(12, h * 0.25, "in", "condition", channel="data", data_type="bool", dot_kind="sub_dot")

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_cond = _port_y(cond_row_widget)
                if y_cond is not None:
                    cond_center = _widget_center_in_node(condition_set.left_box)
                    if cond_center is not None:
                        condition_port.setPos(cond_center.x(), cond_center.y())
                y = _port_y(left_row)
                if y is not None:
                    left_center = _widget_center_in_node(left_setting.left_box) if left_setting.left_box else None
                    left_port.setPos(left_center.x(), left_center.y()) if left_center is not None else left_port.setPos(12, y)
                y = _port_y(right_row_widget)
                if y is not None:
                    right_in_center = _widget_center_in_node(right_setting.left_box) if right_setting.left_box else None
                    right_out_center = _widget_center_in_node(right_setting.right_box) if right_setting.right_box else None
                    right_port.setPos(right_in_center.x(), right_in_center.y()) if right_in_center is not None else right_port.setPos(12, y)
                    result_port.setPos(right_out_center.x(), right_out_center.y()) if right_out_center is not None else result_port.setPos(w_now - 12, y)

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)
            rect._sync_layout_fn = _sync_layout
            resize_ports_cb = _sync_ports
            rect._left_input = left_input
            rect._right_input = right_input_widget
            rect._condition_set = condition_set
            rect._condition_input = condition_set.button
            left_input.textChanged.connect(lambda _t: self._update_node_params(rect))
            right_input_widget.textChanged.connect(lambda _t: self._update_node_params(rect))
            condition_set.button.clicked.connect(lambda: self._update_node_params(rect))

        elif name == "Start":
            # Start node: brand picker (row 1) + model picker (row 2)
            #             + field picker (row 3, v1.4) + flow_out
            brand_items = list(list_brand_items())
            if not brand_items:
                brand_items = [("unitree", "Unitree")]
            brand_list = [display_name for _, display_name in brand_items]
            brand_id_by_display = {display_name: brand_id for brand_id, display_name in brand_items}
            initial_brand_id = normalize_brand_id(self._robot_brand) or brand_items[0][0]
            if initial_brand_id not in {brand_id for brand_id, _ in brand_items}:
                initial_brand_id = brand_items[0][0]
            model_list = get_model_ids(initial_brand_id) or ["go2"]

            # Row 1 — Brand
            brand_row, brand_picker_widget = _make_combo_setting_row(
                "brand_row",
                tr("node_content.start_brand_label", "Brand"),
                items=brand_list,
                show_left_dot=False,
                show_right_dot=False,
            )
            brand_combo = brand_picker_widget.combo
            for idx, (brand_id, _display_name) in enumerate(brand_items):
                brand_combo.setItemData(idx, brand_id, Qt.ItemDataRole.UserRole)
            brand_idx = brand_combo.findData(initial_brand_id, Qt.ItemDataRole.UserRole)
            brand_combo.setCurrentIndex(brand_idx if brand_idx >= 0 else 0)
            self._robot_brand = initial_brand_id

            # Row 2 — Model
            model_row, model_picker_widget = _make_combo_setting_row(
                "model_row",
                tr("node_content.start_model_label", "Model"),
                items=model_list,
                show_left_dot=False,
                show_right_dot=False,
            )
            model_combo = model_picker_widget.combo
            if self._robot_type in model_list:
                model_combo.setCurrentText(self._robot_type)
            else:
                model_combo.setCurrentIndex(0)
                self._robot_type = model_combo.currentText()

            # Row 3 — Field (v1.4: mission_design.yaml — field selection
            # lives on the StartNode itself, not as a separate node).
            # Lists project-local + shared fields via field_loader; the
            # active project (when set via project_session) wins on
            # name collisions. Falls back to ["flat_ground"] when no
            # field library is reachable so the row never disappears.
            try:
                from src.system.runtime.simulation.mujoco import (
                    list_all_fields,
                )
                from src.system.core import project_session
                _proj_root = project_session.get_active_project_root()
                _field_choices = list_all_fields(_proj_root) or ["flat_ground"]
            except Exception:
                _field_choices = ["flat_ground"]
            if "flat_ground" not in _field_choices:
                _field_choices = ["flat_ground"] + _field_choices
            field_row, field_picker_widget = _make_combo_setting_row(
                "field_row",
                tr("node_content.start_field_label", "Field"),
                items=_field_choices,
                show_left_dot=False,
                show_right_dot=False,
            )
            field_combo = field_picker_widget.combo
            # Default selection: "flat_ground" unless the StartNode
            # already has a different field_id stamped on it.
            _initial_field = getattr(self, "_start_field_id", "") or "flat_ground"
            if _initial_field in _field_choices:
                field_combo.setCurrentText(_initial_field)
            else:
                field_combo.setCurrentIndex(0)

            # Row 4 — Render (v1.5: viewer toggle)
            # 2-item combo doubles as a checkbox without a new widget
            # type. Default "On" — interactive Mission canvases get
            # the MuJoCo passive viewer; pick "Off" to run headless.
            _render_choices = ["On", "Off"]
            render_row, render_picker_widget = _make_combo_setting_row(
                "render_row",
                tr("node_content.start_render_label", "Render"),
                items=_render_choices,
                show_left_dot=False,
                show_right_dot=False,
            )
            render_combo = render_picker_widget.combo
            _initial_render = getattr(self, "_start_render_enabled", True)
            render_combo.setCurrentText("On" if _initial_render else "Off")

            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")

            def _sync_ports_start():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                # Anchor flow_out to the lowest content row so the port
                # stays at the visual edge of the node body regardless
                # of how tall the editor grew.
                y = _port_y(render_row) or _port_y(field_row) or _port_y(model_row)
                if y is not None:
                    flow_out.setPos(w_now, y)

            def _sync_layout_start():
                _resize_to_fit()
                _sync_ports_start()

            QTimer.singleShot(0, _sync_layout_start)
            rect._sync_layout_fn = _sync_layout_start
            resize_ports_cb = _sync_ports_start
            rect._brand_picker = brand_combo
            rect._model_picker = model_combo
            rect._field_picker = field_combo   # v1.4: field_id selector
            rect._render_picker = render_combo   # v1.5: viewer toggle
            # Keep legacy alias for backward compat in serialization helpers
            rect._is_protected = True
            header_title.setText("START")
            delete_btn.setVisible(False)
            start_settings_btn = widget_factory.button("⚙", size=20)
            start_settings_btn.setToolTip("Open Start Settings")
            header_row_layout.addWidget(start_settings_btn)
            open_subgraph_controls.append(start_settings_btn)

            def _on_brand_changed(brand_text):
                brand_id = str(
                    brand_combo.currentData(Qt.ItemDataRole.UserRole)
                    or brand_id_by_display.get(brand_text)
                    or normalize_brand_id(brand_text)
                ).strip().lower()
                self._robot_brand = brand_id
                new_models = get_model_ids(brand_id) or ["(no models)"]
                previous_model = model_combo.currentText().strip()
                # Replace model list atomically and choose a model valid for the new brand.
                model_combo.blockSignals(True)
                model_combo.clear()
                model_combo.addItems(new_models)
                if previous_model and previous_model in new_models:
                    model_combo.setCurrentText(previous_model)
                else:
                    model_combo.setCurrentIndex(0)
                model_combo.blockSignals(False)
                # Manually sync state and regenerate since model signal was blocked.
                self._robot_type = model_combo.currentText()
                self.robot_type_changed.emit(self._robot_type)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            def _on_model_changed(text):
                self._robot_type = text
                self.robot_type_changed.emit(self._robot_type)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            def _on_field_changed(text):
                # v1.4: track the selected field_id on the scene so the
                # canvas-wide CanvasToIR conversion picks it up via
                # _sync_node_parameters / serialize_workflow.
                self._start_field_id = str(text or "flat_ground")
                self._sync_node_parameters(rect)
                self.regenerate_code()

            def _on_render_changed(text):
                # v1.5: viewer toggle. Stored as a bool on the scene so
                # serialize_workflow can write it into the canvas json.
                self._start_render_enabled = (str(text or "On").strip().lower() == "on")
                self._sync_node_parameters(rect)
                self.regenerate_code()

            brand_combo.currentTextChanged.connect(_on_brand_changed)
            model_combo.currentTextChanged.connect(_on_model_changed)
            field_combo.currentTextChanged.connect(_on_field_changed)
            render_combo.currentTextChanged.connect(_on_render_changed)

        elif name == "End":
            # End node: only flow_in, no internal components
            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")

            def _sync_ports_end():
                if not isValid(rect):
                    return
                cur_h = rect.rect().height()
                flow_in.setPos(0, cur_h / 2)

            def _sync_layout_end():
                _resize_to_fit(min_height=60)
                _sync_ports_end()

            QTimer.singleShot(0, _sync_layout_end)
            rect._sync_layout_fn = _sync_layout_end
            resize_ports_cb = _sync_ports_end
            rect._is_protected = True
            header_title.setText("END")
            delete_btn.setVisible(False)

        elif "Timer" in name:
            duration_row, duration_setting = _make_input_setting_row(
                "duration_row",
                "duration",
                "duration (s)",
                show_left_dot=True,
                show_right_dot=False,
                input_slot="duration",
            )
            duration_input = duration_setting.input_edit
            duration_input.setValidator(QDoubleValidator(0.0, 60.0, 3, duration_input))

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out_port = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            duration_port = _mk_port(12, h * 0.65, "in", "duration", radius=4, channel="data", data_type="number")

            def _sync_ports():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(duration_row)
                if y is not None:
                    flow_in_port.setPos(0, y)
                    flow_out_port.setPos(w_now, y)
                    dur_center = _widget_center_in_node(duration_setting.left_box) if duration_setting.left_box else None
                    duration_port.setPos(dur_center.x(), dur_center.y()) if dur_center is not None else duration_port.setPos(12, y)

            def _sync_layout():
                _resize_to_fit()
                _sync_ports()

            QTimer.singleShot(0, _sync_layout)
            rect._sync_layout_fn = _sync_layout
            resize_ports_cb = _sync_ports
            rect._duration_input = duration_input

            def _on_duration_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            duration_input.textChanged.connect(lambda _t: _on_duration_change())

        elif name == "Wait":
            # Row 1 鈥?Mode picker
            mode_row, mode_picker = _make_combo_setting_row(
                "mode_row",
                tr("node_content.wait_mode", "Mode"),
                items=["duration", "event"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # Row 2 鈥?Duration (visible in duration mode)
            dur_row, duration_setting = _make_input_setting_row(
                "duration_row",
                "duration",
                "duration (s)",
                show_left_dot=False,
                show_right_dot=False,
            )
            duration_input = duration_setting.input_edit
            duration_input.setValidator(QDoubleValidator(0.0, 60.0, 3, duration_input))

            # Row 3 鈥?Event name (visible in event mode)
            event_row, event_setting = _make_combo_setting_row(
                "event_row",
                "event",
                items=[],
                show_left_dot=False,
                show_right_dot=False,
            )
            event_combo = event_setting.combo

            # Row 4 鈥?Timeout (visible in event mode)
            to_row, timeout_setting = _make_input_setting_row(
                "timeout_row",
                "timeout",
                "timeout (s)",
                show_left_dot=False,
                show_right_dot=False,
            )
            timeout_input = timeout_setting.input_edit
            timeout_input.setValidator(QDoubleValidator(0.0, 300.0, 3, timeout_input))

            # Toggle row visibility based on mode
            def _wait_mode_changed(idx, dr=dur_row, er=event_row, tr_=to_row):
                is_duration = (idx == 0)
                _set_row_visible_compact(dr, is_duration)
                _set_row_visible_compact(er, not is_duration)
                _set_row_visible_compact(tr_, not is_duration)
                sync = getattr(rect, '_sync_layout_fn', None)
                if callable(sync):
                    sync()
            mode_picker.combo.currentIndexChanged.connect(_wait_mode_changed)
            _set_row_visible_compact(dur_row, True)
            _set_row_visible_compact(event_row, False)
            _set_row_visible_compact(to_row, False)

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out_port = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")

            def _sync_ports_wait():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(dur_row) if dur_row.isVisible() else _port_y(event_row)
                if y is not None:
                    flow_in_port.setPos(0, y)
                    flow_out_port.setPos(w_now, y)

            def _sync_layout_wait():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_wait()

            QTimer.singleShot(0, _sync_layout_wait)
            rect._sync_layout_fn = _sync_layout_wait
            resize_ports_cb = _sync_ports_wait
            rect._wait_mode_picker = mode_picker
            rect._duration_input = duration_input
            rect._event_combo = event_combo
            self._wire_symbol_combo(event_combo, "event")
            rect._event_input = event_combo.lineEdit()
            rect._timeout_input = timeout_input

            def _on_wait_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            duration_input.textChanged.connect(lambda _t: _on_wait_change())
            event_combo.currentTextChanged.connect(lambda _t: _on_wait_change())
            timeout_input.textChanged.connect(lambda _t: _on_wait_change())
            mode_picker.combo.currentIndexChanged.connect(lambda _i: _on_wait_change())

        elif name == "Gate":
            # Row 1 鈥?Mode picker
            mode_row, mode_picker = _make_combo_setting_row(
                "mode_row",
                tr("node_content.gate_mode", "Mode"),
                items=["condition", "event"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # Row 2 鈥?ConditionSet (visible in condition mode)
            cond_input = ConditionSet(
                self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                self._condition_set_text, self._hover_bg, "< logic >",
            )
            cond_row, cond_layout = _make_row("none")
            cond_layout.addWidget(cond_input)
            _finish_row(cond_layout, "input")
            row_stack.add_row("cond_row", cond_row)

            # Row 3 鈥?Event name (visible in event mode)
            event_row, event_setting = _make_combo_setting_row(
                "event_row",
                "event",
                items=[],
                show_left_dot=False,
                show_right_dot=False,
            )
            event_combo = event_setting.combo

            def _gate_mode_changed(idx, cr=cond_row, er=event_row):
                _set_row_visible_compact(cr, idx == 0)
                _set_row_visible_compact(er, idx == 1)
                condition_port.setVisible(idx == 0)
                sync = getattr(rect, '_sync_layout_fn', None)
                if callable(sync):
                    sync()
            mode_picker.combo.currentIndexChanged.connect(_gate_mode_changed)
            _set_row_visible_compact(cond_row, True)
            _set_row_visible_compact(event_row, False)

            pass_row, pass_layout = _make_row("output")
            pass_layout.addWidget(make_tag("pass", self._tag_style))
            row_stack.add_row("pass_row", pass_row)

            block_row, block_layout = _make_row("output")
            block_layout.addWidget(make_tag("block", self._tag_style))
            row_stack.add_row("block_row", block_row)

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            condition_port = _mk_port(12, h / 2, "in", "condition", channel="data", data_type="bool", dot_kind="sub_dot")
            out_pass_port = _mk_port(w, h / 2, "out", "out_pass", radius=6, channel="flow")
            out_block_port = _mk_port(w, h / 2, "out", "out_block", radius=6, channel="flow")

            def _sync_ports_gate():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_mode = _port_y(mode_row)
                if y_mode is not None:
                    flow_in_port.setPos(0, y_mode)
                if cond_input.isVisible():
                    cond_center = _widget_center_in_node(cond_input.left_box)
                    if cond_center is not None:
                        condition_port.setPos(cond_center.x(), cond_center.y())
                y_pass = _port_y(pass_row)
                if y_pass is not None:
                    out_pass_port.setPos(w_now, y_pass)
                y_block = _port_y(block_row)
                if y_block is not None:
                    out_block_port.setPos(w_now, y_block)

            def _sync_layout_gate():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_gate()

            QTimer.singleShot(0, _sync_layout_gate)
            rect._sync_layout_fn = _sync_layout_gate
            resize_ports_cb = _sync_ports_gate
            rect._gate_mode_picker = mode_picker
            rect._condition_set = cond_input
            rect._condition_input = cond_input.button
            rect._event_combo = event_combo
            self._wire_symbol_combo(event_combo, "event")
            rect._event_input = event_combo.lineEdit()

            def _on_gate_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            cond_input.button.clicked.connect(lambda: _on_gate_change())
            event_combo.currentTextChanged.connect(lambda _t: _on_gate_change())
            mode_picker.combo.currentIndexChanged.connect(lambda _i: _on_gate_change())

        elif name == "Timeout":
            # Single row 鈥?duration input (same pattern as Timer)
            dur_row, duration_setting = _make_input_setting_row(
                "duration_row",
                "duration",
                "duration (s)",
                show_left_dot=False,
                show_right_dot=False,
            )
            duration_input = duration_setting.input_edit
            duration_input.setValidator(QDoubleValidator(0.0, 300.0, 3, duration_input))

            out_row, out_layout = _make_row("output")
            out_layout.addWidget(make_tag("next", self._tag_style))
            row_stack.add_row("out_row", out_row)

            timeout_row, timeout_layout = _make_row("output")
            timeout_layout.addWidget(make_tag("timeout", self._tag_style))
            row_stack.add_row("timeout_row", timeout_row)

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out_port = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            out_timeout_port = _mk_port(w, h / 2, "out", "out_timeout", radius=6, channel="flow")

            def _sync_ports_timeout():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_dur = _port_y(dur_row)
                if y_dur is not None:
                    flow_in_port.setPos(0, y_dur)
                y_out = _port_y(out_row)
                if y_out is not None:
                    flow_out_port.setPos(w_now, y_out)
                y_to = _port_y(timeout_row)
                if y_to is not None:
                    out_timeout_port.setPos(w_now, y_to)

            def _sync_layout_timeout():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_timeout()

            QTimer.singleShot(0, _sync_layout_timeout)
            rect._sync_layout_fn = _sync_layout_timeout
            resize_ports_cb = _sync_ports_timeout
            rect._duration_input = duration_input

            def _on_timeout_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            duration_input.textChanged.connect(lambda _t: _on_timeout_change())

        elif name == "Retry":
            # Row 1 鈥?Max attempts
            att_row, attempts_setting = _make_input_setting_row(
                "attempts_row",
                tr("node_content.retry_attempts", "Max"),
                "3",
                show_left_dot=False,
                show_right_dot=False,
            )
            attempts_input = attempts_setting.input_edit
            attempts_input.setValidator(QIntValidator(1, 100, attempts_input))

            # Row 2 鈥?Backoff
            bo_row, backoff_setting = _make_input_setting_row(
                "backoff_row",
                tr("node_content.retry_backoff", "Wait"),
                "1.0",
                show_left_dot=False,
                show_right_dot=False,
            )
            backoff_input = backoff_setting.input_edit
            backoff_input.setValidator(QDoubleValidator(0.0, 300.0, 3, backoff_input))

            # Row 3 鈥?Strategy picker
            strat_row, strategy_picker = _make_combo_setting_row(
                "strategy_row",
                tr("node_content.retry_strategy", "Backoff"),
                items=["fixed", "exp"],
                show_left_dot=False,
                show_right_dot=False,
            )

            body_row, body_layout = _make_row("output")
            body_layout.addWidget(make_tag("body", self._tag_style))
            row_stack.add_row("body_row", body_row)

            success_row, success_layout = _make_row("output")
            success_layout.addWidget(make_tag("success", self._tag_style))
            row_stack.add_row("success_row", success_row)

            failed_row, failed_layout = _make_row("output")
            failed_layout.addWidget(make_tag("failed", self._tag_style))
            row_stack.add_row("failed_row", failed_row)

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            retry_body_port = _mk_port(w, h / 2, "out", "retry_body", radius=6, channel="flow")
            out_success_port = _mk_port(w, h / 2, "out", "out_success", radius=6, channel="flow")
            out_failed_port = _mk_port(w, h / 2, "out", "out_failed", radius=6, channel="flow")

            def _sync_ports_retry():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_att = _port_y(att_row)
                if y_att is not None:
                    flow_in_port.setPos(0, y_att)
                y_body = _port_y(body_row)
                if y_body is not None:
                    retry_body_port.setPos(w_now, y_body)
                y_success = _port_y(success_row)
                if y_success is not None:
                    out_success_port.setPos(w_now, y_success)
                y_failed = _port_y(failed_row)
                if y_failed is not None:
                    out_failed_port.setPos(w_now, y_failed)

            def _sync_layout_retry():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_retry()

            QTimer.singleShot(0, _sync_layout_retry)
            rect._sync_layout_fn = _sync_layout_retry
            resize_ports_cb = _sync_ports_retry
            rect._attempts_input = attempts_input
            rect._backoff_input = backoff_input
            rect._strategy_picker = strategy_picker

            def _on_retry_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            attempts_input.textChanged.connect(lambda _t: _on_retry_change())
            backoff_input.textChanged.connect(lambda _t: _on_retry_change())
            strategy_picker.combo.currentIndexChanged.connect(lambda _i: _on_retry_change())

        elif name == "Cancel":
            # Row 1 鈥?Target picker
            target_row, target_picker = _make_combo_setting_row(
                "target_row",
                tr("node_content.cancel_target", "Target"),
                items=["current", "tag", "all"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # Row 2 鈥?Tag name (visible only when target=tag)
            tag_row, tag_setting = _make_input_setting_row(
                "tag_row",
                "tag",
                "tag name",
                show_left_dot=False,
                show_right_dot=False,
            )
            tag_input = tag_setting.input_edit
            _set_row_visible_compact(tag_row, False)

            def _cancel_target_changed(idx, tr_=tag_row):
                _set_row_visible_compact(tr_, idx == 1)  # show tag input only for "tag" target
                sync = getattr(rect, '_sync_layout_fn', None)
                if callable(sync):
                    sync()
            target_picker.combo.currentIndexChanged.connect(_cancel_target_changed)

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out_port = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")

            def _sync_ports_cancel():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(target_row)
                if y is not None:
                    flow_in_port.setPos(0, y)
                    flow_out_port.setPos(w_now, y)

            def _sync_layout_cancel():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_cancel()

            QTimer.singleShot(0, _sync_layout_cancel)
            rect._sync_layout_fn = _sync_layout_cancel
            resize_ports_cb = _sync_ports_cancel
            rect._cancel_target_picker = target_picker
            rect._cancel_tag_input = tag_input

            def _on_cancel_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            target_picker.combo.currentIndexChanged.connect(lambda _i: _on_cancel_change())
            tag_input.textChanged.connect(lambda _t: _on_cancel_change())

        elif name == "Abort":
            # Row 1 鈥?Reason input
            reason_row, reason_setting = _make_input_setting_row(
                "reason_row",
                "reason",
                "reason",
                show_left_dot=False,
                show_right_dot=False,
            )
            reason_input = reason_setting.input_edit

            # Row 2 鈥?Emergency picker
            em_row, emergency_picker = _make_combo_setting_row(
                "emergency_row",
                tr("node_content.abort_emergency", "Emergency"),
                items=["false", "true"],
                show_left_dot=False,
                show_right_dot=False,
            )

            flow_in_port = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            # Abort is terminal 鈥?no output ports

            def _sync_ports_abort():
                if not isValid(rect):
                    return
                y = _port_y(reason_row)
                if y is not None:
                    flow_in_port.setPos(0, y)

            def _sync_layout_abort():
                # Dynamic height from current visible rows.
                _resize_to_fit()
                _sync_ports_abort()

            QTimer.singleShot(0, _sync_layout_abort)
            rect._sync_layout_fn = _sync_layout_abort
            resize_ports_cb = _sync_ports_abort
            rect._abort_reason_input = reason_input
            rect._abort_emergency_picker = emergency_picker

            def _on_abort_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()
            reason_input.textChanged.connect(lambda _t: _on_abort_change())
            emergency_picker.combo.currentIndexChanged.connect(lambda _i: _on_abort_change())

        elif "Switch" in name:
            # Expression input row
            expr_row, expr_setting = _make_input_setting_row(
                "switch_expr_row",
                "value",
                "value",
                show_left_dot=True,
                show_right_dot=False,
                input_slot="value",
            )
            expr_input = expr_setting.input_edit

            # Default output row
            default_row, default_layout = _make_row("output")
            default_layout.addWidget(make_tag("default", self._tag_style))
            row_stack.add_row("switch_default_row", default_row)

            # Add case button in header
            add_case_btn = widget_factory.button("+", size=20)
            header_row_layout.insertWidget(header_row_layout.count() - 1, add_case_btn)

            flow_in_port = _mk_port(0, h * 0.35, "in", "flow_in", radius=6, channel="flow")
            value_port = _mk_port(12, h * 0.35, "in", "value_in", radius=4, channel="data",
                                  data_type="any", dot_kind="sub_dot")
            default_port = _mk_port(w, h * 0.80, "out", "case_default", radius=6, channel="flow")

            rect._switch_case_ports = []
            rect._switch_case_inputs = []
            rect._switch_case_rows = []

            def _reindex_switch_cases():
                for _i, _port in enumerate(rect._switch_case_ports):
                    _port.setData(3, f"case_{_i}")

            def _add_switch_case():
                _idx = len(rect._switch_case_ports)
                case_port = _mk_port(w, h * 0.50, "out", f"case_{_idx}",
                                     radius=6, channel="flow")
                rect._switch_case_ports.append(case_port)
                remove_case_btn = widget_factory.button("x", size=20)
                case_input = widget_factory.line_edit(f"case {_idx}")
                case_row, case_layout = _make_row("output")
                case_layout.addWidget(make_tag(f"={_idx}:", self._tag_style))
                case_layout.addWidget(case_input)
                case_layout.addWidget(remove_case_btn)
                row_stack.add_row(f"switch_case_row_{_idx}", case_row,
                                  before_row_id="switch_default_row")
                rect._switch_case_inputs.append(case_input)
                rect._switch_case_rows.append(case_row)

                def _remove_this_case(_row=case_row):
                    _local_idx = rect._switch_case_rows.index(_row) if _row in rect._switch_case_rows else -1
                    if _local_idx < 0:
                        return
                    row_stack.remove_row(f"switch_case_row_{_local_idx}")
                    _port = rect._switch_case_ports[_local_idx]
                    _conns = _port.data(2) or []
                    for _conn in list(_conns):
                        if _conn and isValid(_conn) and _conn.scene() is not None:
                            self.removeItem(_conn)
                    if _port.scene() is not None:
                        self.removeItem(_port)
                    rect._switch_case_ports.pop(_local_idx)
                    rect._switch_case_inputs.pop(_local_idx)
                    rect._switch_case_rows.pop(_local_idx)
                    _reindex_switch_cases()
                    QTimer.singleShot(0, _sync_layout_switch)
                    self.regenerate_code()

                remove_case_btn.clicked.connect(_remove_this_case)
                case_input.textChanged.connect(lambda _t: self._update_node_params(rect))
                QTimer.singleShot(0, _sync_layout_switch)
                self.regenerate_code()

            rect._add_switch_case = _add_switch_case
            add_case_btn.clicked.connect(_add_switch_case)

            def _sync_ports_switch():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_expr = _port_y(expr_row)
                if y_expr is not None:
                    flow_in_port.setPos(0, y_expr)
                    expr_center = _widget_center_in_node(expr_setting.left_box) if expr_setting.left_box else None
                    value_port.setPos(expr_center.x(), expr_center.y()) if expr_center is not None else value_port.setPos(12, y_expr)
                y_default = _port_y(default_row)
                if y_default is not None:
                    default_port.setPos(w_now, y_default)
                for _row, _port in zip(rect._switch_case_rows, rect._switch_case_ports):
                    _y = _port_y(_row)
                    if _y is not None:
                        _port.setPos(w_now, _y)

            def _sync_layout_switch():
                _resize_to_fit()
                _sync_ports_switch()

            QTimer.singleShot(0, _sync_layout_switch)
            rect._sync_layout_fn = _sync_layout_switch
            resize_ports_cb = _sync_ports_switch
            rect._switch_expr_input = expr_input
            header_title.setText("SWITCH")
            expr_input.textChanged.connect(lambda _t: self._update_node_params(rect))

        elif "Try" in name:
            # Exception type input
            exc_row, exc_setting = _make_input_setting_row(
                "try_exc_row",
                "except",
                "Exception",
                show_left_dot=False,
                show_right_dot=False,
            )
            exc_input = exc_setting.input_edit

            # Branch output rows
            try_label_row, try_label_layout = _make_row("output")
            try_label_layout.addWidget(make_tag("try \u2192", self._tag_style))
            row_stack.add_row("try_body_row", try_label_row)

            except_label_row, except_label_layout = _make_row("output")
            except_label_layout.addWidget(make_tag("except \u2192", self._tag_style))
            row_stack.add_row("except_body_row", except_label_row)

            finally_label_row, finally_label_layout = _make_row("output")
            finally_label_layout.addWidget(make_tag("finally \u2192", self._tag_style))
            row_stack.add_row("finally_body_row", finally_label_row)

            flow_in_port = _mk_port(0, h * 0.30, "in", "flow_in", radius=6, channel="flow")
            try_body_port = _mk_port(w, h * 0.45, "out", "try_body", radius=6, channel="flow")
            except_body_port = _mk_port(w, h * 0.62, "out", "except_body", radius=6, channel="flow")
            finally_body_port = _mk_port(w, h * 0.80, "out", "finally_body", radius=6, channel="flow")

            def _sync_ports_try():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_exc = _port_y(exc_row)
                if y_exc is not None:
                    flow_in_port.setPos(0, y_exc)
                y_try = _port_y(try_label_row)
                if y_try is not None:
                    try_body_port.setPos(w_now, y_try)
                y_except = _port_y(except_label_row)
                if y_except is not None:
                    except_body_port.setPos(w_now, y_except)
                y_finally = _port_y(finally_label_row)
                if y_finally is not None:
                    finally_body_port.setPos(w_now, y_finally)

            def _sync_layout_try():
                _resize_to_fit()
                _sync_ports_try()

            QTimer.singleShot(0, _sync_layout_try)
            rect._sync_layout_fn = _sync_layout_try
            resize_ports_cb = _sync_ports_try
            rect._exc_type_input = exc_input
            header_title.setText("TRY")
            exc_input.textChanged.connect(lambda _t: self._update_node_params(rect))

        elif "Parallel" in name:
            count_row_par, count_picker_par = _make_combo_setting_row(
                "parallel_count_row",
                tr("node_content.parallel_branches", "Branches"),
                items=["2", "3", "4"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # Branch label rows (up to 4)
            branch_label_rows_par = []
            for _i in range(4):
                _br_row, _br_layout = _make_row("output")
                _br_layout.addWidget(make_tag(f"br {_i} \u2192", self._tag_style))
                row_stack.add_row(f"parallel_branch_row_{_i}", _br_row)
                branch_label_rows_par.append(_br_row)

            join_label_row_par, join_label_layout_par = _make_row("output")
            join_label_layout_par.addWidget(make_tag("merged \u2192", self._tag_style))
            row_stack.add_row("parallel_join_row", join_label_row_par)

            flow_in_port_par = _mk_port(0, h * 0.35, "in", "flow_in", radius=6, channel="flow")
            branch_ports_par = [
                _mk_port(w, h * 0.35, "out", "branch_0", radius=6, channel="flow"),
                _mk_port(w, h * 0.50, "out", "branch_1", radius=6, channel="flow"),
                _mk_port(w, h * 0.65, "out", "branch_2", radius=6, channel="flow"),
                _mk_port(w, h * 0.80, "out", "branch_3", radius=6, channel="flow"),
            ]
            join_out_port_par = _mk_port(w, h * 0.95, "out", "join_out", radius=6, channel="flow")

            # Initially hide branches 2 and 3
            _set_row_visible_compact(branch_label_rows_par[2], False)
            _set_row_visible_compact(branch_label_rows_par[3], False)
            branch_ports_par[2].setVisible(False)
            branch_ports_par[3].setVisible(False)

            def _update_parallel_branches(count_str):
                _count = int(count_str)
                for _i in range(4):
                    _visible = _i < _count
                    _set_row_visible_compact(branch_label_rows_par[_i], _visible)
                    branch_ports_par[_i].setVisible(_visible)
                QTimer.singleShot(0, _sync_layout_par)
                self._update_node_params(rect)
                self.regenerate_code()

            count_picker_par.combo.currentTextChanged.connect(_update_parallel_branches)

            def _sync_ports_par():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_count = _port_y(count_row_par)
                if y_count is not None:
                    flow_in_port_par.setPos(0, y_count)
                for _i, (_br_row, _port) in enumerate(zip(branch_label_rows_par, branch_ports_par)):
                    _y = _port_y(_br_row)
                    if _y is not None:
                        _port.setPos(w_now, _y)
                y_join = _port_y(join_label_row_par)
                if y_join is not None:
                    join_out_port_par.setPos(w_now, y_join)

            def _sync_layout_par():
                _resize_to_fit()
                _sync_ports_par()

            QTimer.singleShot(0, _sync_layout_par)
            rect._sync_layout_fn = _sync_layout_par
            resize_ports_cb = _sync_ports_par
            rect._branch_count_picker = count_picker_par
            header_title.setText("PARALLEL")
            count_picker_par.combo.currentTextChanged.connect(
                lambda _t: self._update_node_params(rect))

        elif "Join" in name or name == "Join":
            count_row_join, count_picker_join = _make_combo_setting_row(
                "join_count_row",
                tr("node_content.join_inputs", "Inputs"),
                items=["2", "3", "4"],
                show_left_dot=False,
                show_right_dot=False,
            )

            # Branch input label rows (up to 4)
            branch_in_rows_join = []
            for _i in range(4):
                _bi_row, _bi_layout = _make_row("input")
                _bi_layout.addWidget(make_tag(f"\u2190 br {_i}", self._tag_style))
                _finish_row(_bi_layout, "input")
                row_stack.add_row(f"join_branch_row_{_i}", _bi_row)
                branch_in_rows_join.append(_bi_row)

            out_row_join, out_layout_join = _make_row("output")
            out_layout_join.addWidget(make_tag("continue \u2192", self._tag_style))
            row_stack.add_row("join_out_row", out_row_join)

            branch_in_ports_join = [
                _mk_port(0, h * 0.35, "in", "branch_0", radius=6, channel="flow"),
                _mk_port(0, h * 0.50, "in", "branch_1", radius=6, channel="flow"),
                _mk_port(0, h * 0.65, "in", "branch_2", radius=6, channel="flow"),
                _mk_port(0, h * 0.80, "in", "branch_3", radius=6, channel="flow"),
            ]
            flow_out_port_join = _mk_port(w, h * 0.55, "out", "flow_out", radius=6, channel="flow")

            # Initially hide branches 2 and 3
            _set_row_visible_compact(branch_in_rows_join[2], False)
            _set_row_visible_compact(branch_in_rows_join[3], False)
            branch_in_ports_join[2].setVisible(False)
            branch_in_ports_join[3].setVisible(False)

            def _update_join_inputs(count_str):
                _count = int(count_str)
                for _i in range(4):
                    _visible = _i < _count
                    _set_row_visible_compact(branch_in_rows_join[_i], _visible)
                    branch_in_ports_join[_i].setVisible(_visible)
                QTimer.singleShot(0, _sync_layout_join)
                self._update_node_params(rect)
                self.regenerate_code()

            count_picker_join.combo.currentTextChanged.connect(_update_join_inputs)

            def _sync_ports_join():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                for _i, (_bi_row, _port) in enumerate(zip(branch_in_rows_join, branch_in_ports_join)):
                    _y = _port_y(_bi_row)
                    if _y is not None:
                        _port.setPos(0, _y)
                y_out = _port_y(out_row_join)
                if y_out is not None:
                    flow_out_port_join.setPos(w_now, y_out)

            def _sync_layout_join():
                _resize_to_fit()
                _sync_ports_join()

            QTimer.singleShot(0, _sync_layout_join)
            rect._sync_layout_fn = _sync_layout_join
            resize_ports_cb = _sync_ports_join
            rect._join_count_picker = count_picker_join
            header_title.setText("JOIN")
            count_picker_join.combo.currentTextChanged.connect(
                lambda _t: self._update_node_params(rect))

        elif name in ("OnEvent", "PublishEvent", "SetState", "CheckState", "HumanConfirm"):
            # Mission-layer simplified State/Event shells.
            # Behavior-layer rollout should consume `_behavior_extension_notes`
            # and optionally replace these with richer nested editors.
            flow_in = None
            flow_out = None
            field_widgets: Dict[str, Any] = {}
            note_row, note_layout = _make_row("none")
            note_layout.addWidget(make_tag("ext: behavior hook", self._tag_style))
            _finish_row(note_layout, "none")
            row_stack.add_row("state_event_note", note_row)
            setting_input_bg = get_color("input_bg", "#0f1115")

            def _make_input_setting_row(
                row_id: str,
                key_text: str,
                placeholder: str,
                *,
                show_left_dot: bool,
                show_right_dot: bool,
                input_slot: Optional[str] = None,
            ):
                setting = InputSetting(
                    self._condition_set_left_bg,
                    self._condition_set_right_bg,
                    setting_input_bg,
                    self._button_border,
                    self._condition_set_text,
                    key_text=key_text,
                    placeholder=placeholder,
                    show_left_dot=show_left_dot,
                    show_right_dot=show_right_dot,
                )
                row_widget, row_layout = _make_row("none")
                row_layout.addWidget(setting)
                _finish_row(row_layout, "none")
                row_stack.add_row(row_id, row_widget)
                if input_slot:
                    self._register_bound_input_widget(rect, input_slot, setting)
                return row_widget, setting

            def _make_symbol_combo_row(
                row_id: str,
                key_text: str,
                *,
                show_left_dot: bool,
                show_right_dot: bool,
                input_slot: Optional[str] = None,
            ):
                setting = ComboSetting(
                    self._condition_set_left_bg,
                    self._condition_set_right_bg,
                    setting_input_bg,
                    self._button_border,
                    self._condition_set_text,
                    key_text=key_text,
                    items=[],
                    show_left_dot=show_left_dot,
                    show_right_dot=show_right_dot,
                )
                row_widget, row_layout = _make_row("none")
                row_layout.addWidget(setting)
                _finish_row(row_layout, "none")
                row_stack.add_row(row_id, row_widget)
                if input_slot:
                    self._register_bound_input_widget(rect, input_slot, setting)
                return row_widget, setting

            if name == "OnEvent":
                row, event_setting = _make_symbol_combo_row(
                    "on_event_row",
                    "event",
                    show_left_dot=True,
                    show_right_dot=True,
                    input_slot="se_event_name",
                )
                event_combo = event_setting.combo
                self._wire_symbol_combo(event_combo, "event")
                event_input = event_combo.lineEdit()
                event_name_in = _mk_port(12, h / 2, "in", "se_event_name", radius=4, channel="data", data_type="string", dot_kind="sub_dot")
                event_state_out = _mk_port(w - 12, h / 2, "out", "result", radius=4, channel="data", data_type="bool", dot_kind="sub_dot")
                field_widgets["event_name"] = event_input

                def _sync_ports_state_event():
                    if not isValid(rect):
                        return
                    w_now = rect.rect().width()
                    y = _port_y(row)
                    if y is not None:
                        in_center = _widget_center_in_node(event_setting.left_box) if event_setting.left_box else None
                        out_center = _widget_center_in_node(event_setting.right_box) if event_setting.right_box else None
                        event_name_in.setPos(in_center.x(), in_center.y()) if in_center is not None else event_name_in.setPos(12, y)
                        event_state_out.setPos(out_center.x(), out_center.y()) if out_center is not None else event_state_out.setPos(w_now - 12, y)

            elif name == "PublishEvent":
                row1, event_setting = _make_symbol_combo_row(
                    "pub_event_row",
                    "event",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_event_name",
                )
                row2, payload_setting = _make_input_setting_row(
                    "pub_payload_row",
                    "payload",
                    "payload",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_payload",
                )
                event_combo = event_setting.combo
                self._wire_symbol_combo(event_combo, "event")
                event_input = event_combo.lineEdit()
                payload_input = payload_setting.input_edit
                flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
                flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
                event_name_in = _mk_port(12, h / 2, "in", "se_event_name", radius=4, channel="data", data_type="string", dot_kind="sub_dot")
                payload_in = _mk_port(12, h / 2, "in", "se_payload", radius=4, channel="data", data_type="any", dot_kind="sub_dot")
                field_widgets["event_name"] = event_input
                field_widgets["payload"] = payload_input

                def _sync_ports_state_event():
                    if not isValid(rect):
                        return
                    w_now = rect.rect().width()
                    y = _port_y(row1)
                    if y is not None:
                        flow_in.setPos(0, y)
                        flow_out.setPos(w_now, y)
                        event_center = _widget_center_in_node(event_setting.left_box) if event_setting.left_box else None
                        event_name_in.setPos(event_center.x(), event_center.y()) if event_center is not None else event_name_in.setPos(12, y)
                    y_payload = _port_y(row2)
                    if y_payload is not None:
                        payload_center = _widget_center_in_node(payload_setting.left_box) if payload_setting.left_box else None
                        payload_in.setPos(payload_center.x(), payload_center.y()) if payload_center is not None else payload_in.setPos(12, y_payload)

            elif name == "SetState":
                row1, key_setting = _make_input_setting_row(
                    "set_state_key_row",
                    "key",
                    "state_key",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_key",
                )
                row2, value_setting = _make_input_setting_row(
                    "set_state_value_row",
                    "value",
                    "value",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_value",
                )
                key_input = key_setting.input_edit
                value_input = value_setting.input_edit
                flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
                flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
                key_in = _mk_port(12, h / 2, "in", "se_key", radius=4, channel="data", data_type="string", dot_kind="sub_dot")
                value_in = _mk_port(12, h / 2, "in", "se_value", radius=4, channel="data", data_type="any", dot_kind="sub_dot")
                field_widgets["state_key"] = key_input
                field_widgets["state_value"] = value_input

                def _sync_ports_state_event():
                    if not isValid(rect):
                        return
                    w_now = rect.rect().width()
                    y = _port_y(row1)
                    if y is not None:
                        flow_in.setPos(0, y)
                        flow_out.setPos(w_now, y)
                        key_center = _widget_center_in_node(key_setting.left_box) if key_setting.left_box else None
                        key_in.setPos(key_center.x(), key_center.y()) if key_center is not None else key_in.setPos(12, y)
                    y_value = _port_y(row2)
                    if y_value is not None:
                        value_center = _widget_center_in_node(value_setting.left_box) if value_setting.left_box else None
                        value_in.setPos(value_center.x(), value_center.y()) if value_center is not None else value_in.setPos(12, y_value)

            elif name == "CheckState":
                row1, key_setting = _make_input_setting_row(
                    "check_state_key_row",
                    "key",
                    "state_key",
                    show_left_dot=False,
                    show_right_dot=True,
                )
                row2, expected_setting = _make_input_setting_row(
                    "check_state_expected_row",
                    "expected",
                    "expected",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_expected",
                )
                key_input = key_setting.input_edit
                expected_input = expected_setting.input_edit
                expected_in = _mk_port(12, h / 2, "in", "se_expected", radius=4, channel="data", data_type="any", dot_kind="sub_dot")
                result_out = _mk_port(w - 12, h / 2, "out", "result", radius=4, channel="data", data_type="bool", dot_kind="sub_dot")
                field_widgets["state_key"] = key_input
                field_widgets["state_expected"] = expected_input

                def _sync_ports_state_event():
                    if not isValid(rect):
                        return
                    w_now = rect.rect().width()
                    y = _port_y(row1)
                    if y is not None:
                        out_center = _widget_center_in_node(key_setting.right_box) if key_setting.right_box else None
                        result_out.setPos(out_center.x(), out_center.y()) if out_center is not None else result_out.setPos(w_now - 12, y)
                    y_expected = _port_y(row2)
                    if y_expected is not None:
                        expected_center = _widget_center_in_node(expected_setting.left_box) if expected_setting.left_box else None
                        expected_in.setPos(expected_center.x(), expected_center.y()) if expected_center is not None else expected_in.setPos(12, y_expected)

            else:
                row1, prompt_setting = _make_input_setting_row(
                    "confirm_prompt_row",
                    "prompt",
                    "confirm prompt",
                    show_left_dot=True,
                    show_right_dot=True,
                    input_slot="se_prompt",
                )
                row2, timeout_setting = _make_input_setting_row(
                    "confirm_timeout_row",
                    "timeout",
                    "timeout",
                    show_left_dot=True,
                    show_right_dot=False,
                    input_slot="se_timeout",
                )
                prompt_input = prompt_setting.input_edit
                timeout_input = timeout_setting.input_edit
                timeout_input.setValidator(QDoubleValidator(0.0, 3600.0, 3, timeout_input))
                flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
                flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
                prompt_in = _mk_port(12, h / 2, "in", "se_prompt", radius=4, channel="data", data_type="string", dot_kind="sub_dot")
                timeout_in = _mk_port(12, h / 2, "in", "se_timeout", radius=4, channel="data", data_type="number", dot_kind="sub_dot")
                result_out = _mk_port(w - 12, h / 2, "out", "result", radius=4, channel="data", data_type="bool", dot_kind="sub_dot")
                field_widgets["confirm_prompt"] = prompt_input
                field_widgets["confirm_timeout"] = timeout_input

                def _sync_ports_state_event():
                    if not isValid(rect):
                        return
                    w_now = rect.rect().width()
                    y = _port_y(row1)
                    if y is not None:
                        flow_in.setPos(0, y)
                        flow_out.setPos(w_now, y)
                        prompt_center = _widget_center_in_node(prompt_setting.left_box) if prompt_setting.left_box else None
                        result_center = _widget_center_in_node(prompt_setting.right_box) if prompt_setting.right_box else None
                        prompt_in.setPos(prompt_center.x(), prompt_center.y()) if prompt_center is not None else prompt_in.setPos(12, y)
                        result_out.setPos(result_center.x(), result_center.y()) if result_center is not None else result_out.setPos(w_now - 12, y)
                    y_timeout = _port_y(row2)
                    if y_timeout is not None:
                        timeout_center = _widget_center_in_node(timeout_setting.left_box) if timeout_setting.left_box else None
                        timeout_in.setPos(timeout_center.x(), timeout_center.y()) if timeout_center is not None else timeout_in.setPos(12, y_timeout)

            def _sync_layout_state_event():
                _resize_to_fit(min_width=180)
                _sync_ports_state_event()

            QTimer.singleShot(0, _sync_layout_state_event)
            rect._sync_layout_fn = _sync_layout_state_event
            resize_ports_cb = _sync_ports_state_event
            rect._state_event_fields = field_widgets
            rect._behavior_extension_notes = {
                "owner": "behavior-layer",
                "todo": [
                    "replace shell logic with behavior graph binding",
                    "allow typed input/output variable slots",
                    "support async event correlation and timeout policies",
                ],
            }

            for _w in field_widgets.values():
                if hasattr(_w, "textChanged"):
                    _w.textChanged.connect(lambda _t: self._update_node_params(rect))
        elif name == "Script Box":
            script_row, script_setting = _make_script_setting_row(
                "script_row",
                items=[],
                show_left_dot=True,
                show_advanced_button=True,
                combo_editable=True,
            )
            script_io_row, script_io_widget = _make_script_io_row("script_io_row")
            script_ref_combo = script_setting.combo
            self._wire_symbol_combo(script_ref_combo, "script")
            script_ref_input = script_ref_combo.lineEdit()

            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            script_in_port = _mk_port(
                12,
                h / 2,
                "in",
                "condition",
                channel="data",
                data_type="any",
                dot_kind="sub_dot",
                max_connections=-1,
            )

            def _sync_ports_script_box():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(script_row)
                if y is not None:
                    flow_in.setPos(0, y)
                    flow_out.setPos(w_now, y)
                    in_center = _widget_center_in_node(script_setting.left_box) if script_setting.left_box else None
                    script_in_port.setPos(in_center.x(), in_center.y()) if in_center is not None else script_in_port.setPos(12, y)
                for input_entry in list(getattr(rect, "_script_input_rows", []) or []):
                    port = input_entry.get("port")
                    left_box = input_entry.get("left_box")
                    if not port or not isValid(port):
                        continue
                    center = _widget_center_in_node(left_box) if left_box is not None else None
                    if center is None:
                        row_widget = input_entry.get("row")
                        row_y = _port_y(row_widget) if row_widget is not None else None
                        if row_y is not None:
                            port.setPos(12, row_y)
                        continue
                    port.setPos(center.x(), center.y())
                for output_entry in list(getattr(rect, "_script_output_rows", []) or []):
                    port = output_entry.get("port")
                    right_box = output_entry.get("right_box")
                    if not port or not isValid(port):
                        continue
                    center = _widget_center_in_node(right_box) if right_box is not None else None
                    if center is None:
                        row_widget = output_entry.get("row")
                        row_y = _port_y(row_widget) if row_widget is not None else None
                        if row_y is not None:
                            port.setPos(w_now - 12, row_y)
                        continue
                    port.setPos(center.x(), center.y())

            def _sync_layout_script_box():
                io_pref_h = 56
                if script_io_widget is not None and isValid(script_io_widget):
                    try:
                        io_pref_h = max(
                            56,
                            int(script_io_widget.preferred_height()),
                            int(script_io_widget.sizeHint().height()),
                        )
                    except Exception:
                        io_pref_h = max(56, int(script_io_widget.sizeHint().height()))
                # Keep script IO row height content-driven; this node is intentionally dynamic.
                script_io_row.setMinimumHeight(io_pref_h)
                script_io_row.setMaximumHeight(io_pref_h)
                if script_io_widget is not None and isValid(script_io_widget):
                    script_io_widget.updateGeometry()
                _resize_to_fit()
                _sync_ports_script_box()

            QTimer.singleShot(0, _sync_layout_script_box)
            rect._sync_layout_fn = _sync_layout_script_box
            resize_ports_cb = _sync_ports_script_box
            rect._script_ref = ""
            rect._script_ref_combo = script_ref_combo
            rect._script_ref_input = script_ref_input
            rect._script_io_spec = {"inputs": [], "outputs": []}
            rect._script_inout = script_io_widget
            rect._script_io_row = script_io_row
            rect._script_proxy_in_port = script_in_port
            rect._script_input_rows = []
            rect._script_output_rows = []
            rect._add_script_input_row = (
                lambda data_type="any", var_name=None, slot=None, _r=rect:
                self._script_add_input_row(_r, data_type=data_type, var_name=var_name, slot=slot)
            )
            rect._add_script_output_row = (
                lambda var_name=None, data_type="any", slot=None, _r=rect:
                self._script_add_output_row(_r, var_name=var_name, data_type=data_type, slot=slot)
            )
            rect._condition_set = script_setting
            if script_setting.advanced_btn is not None:
                script_setting.advanced_btn.setToolTip("Open Script editor")
                open_subgraph_controls.append(script_setting.advanced_btn)

            def _on_script_ref_change(record_symbol: bool = False):
                rect._script_ref = script_ref_combo.currentText().strip()
                if record_symbol:
                    self._record_symbol("script", rect._script_ref)
                node_id_local = int(rect.data(12) or 0)
                if callable(self._script_tab_renamer) and node_id_local > 0:
                    try:
                        self._script_tab_renamer(node_id_local, rect._script_ref)
                    except Exception:
                        pass
                self._update_node_params(rect)

            # Typing in the script-name field is treated as "rename current script".
            # Only explicit dropdown activation is treated as switching/loading.
            if script_ref_input is not None:
                script_ref_input.textEdited.connect(
                    lambda _t: _on_script_ref_change(record_symbol=False)
                )
            script_ref_combo.activated.connect(
                lambda _idx: _on_script_ref_change(record_symbol=True)
            )

        elif name == "Behavior":
            # Executes the trained policy of the bound checkpoint.

            # Row 1: checkpoint selector — left dot = checkpoint in-port
            beh_pid_row, beh_pid_setting = _make_combo_setting_row(
                "beh_policy_id_row",
                "Checkpoint",
                items=[],
                show_left_dot=True,
                show_right_dot=False,
            )

            speed_row_widget, speed_row_layout = _make_row("none")
            speed_wrap = QWidget()
            speed_wrap.setStyleSheet("background: transparent;")
            speed_wrap_layout = QHBoxLayout(speed_wrap)
            speed_wrap_layout.setContentsMargins(0, 0, 0, 0)
            speed_wrap_layout.setSpacing(6)

            speed_key = QLabel("Speed")
            speed_key.setFixedHeight(20)
            speed_key.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            speed_key.setStyleSheet(
                f"QLabel {{ background: {self._condition_set_right_bg}; color: {self._condition_set_text}; "
                f"border: 1px solid {self._button_border}; padding: 2px 6px; font-size: 10px; }}"
            )
            speed_key.setFixedWidth(46)

            speed_slider = QSlider(Qt.Orientation.Horizontal)
            speed_slider.setRange(0, 200)
            speed_slider.setValue(100)
            speed_slider.setFixedHeight(20)
            speed_slider.setStyleSheet("background: transparent;")

            speed_value = QLabel("100%")
            speed_value.setFixedHeight(20)
            speed_value.setFixedWidth(34)
            speed_value.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            speed_value.setStyleSheet(
                f"QLabel {{ color: {self._condition_set_text}; background: transparent; border: none; font-size: 10px; }}"
            )

            speed_wrap_layout.addWidget(speed_key, 0)
            speed_wrap_layout.addWidget(speed_slider, 1)
            speed_wrap_layout.addWidget(speed_value, 0)
            speed_row_layout.addWidget(speed_wrap, 1)
            row_stack.add_row("beh_speed_scale_row", speed_row_widget)

            # Row 3: SkillManifest badge (Phase 7 A5)
            manifest_row_w = QWidget()
            manifest_row_w.setObjectName("beh_manifest_badge_row")
            manifest_row_w.setStyleSheet("QWidget { border: 0px; background: transparent; }")
            manifest_row_layout = QHBoxLayout(manifest_row_w)
            manifest_row_layout.setContentsMargins(8, 2, 8, 2)
            manifest_row_layout.setSpacing(4)

            beh_action_badge = QLabel("—")
            beh_action_badge.setObjectName("behActionBadge")
            beh_action_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            beh_action_badge.setFixedSize(56, 18)
            beh_action_badge.setStyleSheet(
                "QLabel#behActionBadge {"
                f"background: {self._button_border}; color: #ffffff; "
                "font-size: 9px; font-weight: 600; border-radius: 3px; }"
            )
            beh_action_badge.setToolTip("Action space type")

            beh_freq_label = QLabel("")
            beh_freq_label.setObjectName("behFreqLabel")
            beh_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            beh_freq_label.setFixedHeight(18)
            beh_freq_label.setStyleSheet(
                f"QLabel {{ color: {self._condition_set_text}; background: transparent; "
                "border: none; font-size: 9px; }}"
            )

            beh_posture_badge = QLabel("")
            beh_posture_badge.setObjectName("behPostureBadge")
            beh_posture_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            beh_posture_badge.setFixedHeight(18)
            beh_posture_badge.setStyleSheet(
                f"QLabel {{ color: {self._condition_set_text}; background: transparent; "
                "border: none; font-size: 9px; }}"
            )

            beh_compat_badge = QLabel("")
            beh_compat_badge.setObjectName("behCompatBadge")
            beh_compat_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            beh_compat_badge.setFixedSize(18, 18)
            beh_compat_badge.setStyleSheet(
                "QLabel#behCompatBadge {"
                "background: #9CA3AF; color: #ffffff; "
                "font-size: 10px; font-weight: 700; border-radius: 9px; }"
            )
            beh_compat_badge.setToolTip("Compatibility status")

            manifest_row_layout.addWidget(beh_action_badge)
            manifest_row_layout.addWidget(beh_freq_label, 1)
            manifest_row_layout.addWidget(beh_posture_badge, 1)
            manifest_row_layout.addWidget(beh_compat_badge)
            row_stack.add_row("beh_manifest_badge_row", manifest_row_w)
            rect._beh_action_badge = beh_action_badge
            rect._beh_freq_label = beh_freq_label
            rect._beh_posture_badge = beh_posture_badge
            rect._beh_compat_badge = beh_compat_badge

            # Ports
            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            checkpoint_in_beh = _mk_port(12, h / 2, "in", "checkpoint", radius=4,
                                         channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_behavior():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_pid = _port_y(beh_pid_row)
                if y_pid is not None:
                    flow_in.setPos(0, y_pid)
                    flow_out.setPos(w_now, y_pid)
                    in_center = _widget_center_in_node(beh_pid_setting.left_box) if beh_pid_setting.left_box else None
                    checkpoint_in_beh.setPos(in_center.x(), in_center.y()) if in_center is not None else checkpoint_in_beh.setPos(12, y_pid)

            def _sync_layout_behavior():
                _resize_to_fit()
                _sync_ports_behavior()

            QTimer.singleShot(0, _sync_layout_behavior)
            rect._sync_layout_fn = _sync_layout_behavior
            resize_ports_cb = _sync_ports_behavior
            rect._beh_policy_id_picker   = beh_pid_setting
            rect._beh_command_scale_slider = speed_slider
            rect._beh_command_scale_label = speed_value

            def _update_behavior_manifest_badge():
                """Update SkillManifest badge display from bound checkpoint."""
                _ACTION_ABBREV = {
                    "joint_position": "JP",
                    "joint_torque": "JT",
                    "cartesian_delta": "CD",
                    "end_effector_6d": "EE",
                    "hybrid": "HY",
                }
                _ACTION_COLOR = {
                    "joint_position": "#3B82F6",
                    "joint_torque": "#EF4444",
                    "cartesian_delta": "#8B5CF6",
                    "end_effector_6d": "#F59E0B",
                    "hybrid": "#10B981",
                }
                try:
                    logic = rect.data(13)
                    if logic is None or not hasattr(logic, "skill_manifest_summary"):
                        return
                    summary = logic.skill_manifest_summary()
                    if not summary:
                        beh_action_badge.setText("—")
                        beh_action_badge.setStyleSheet(
                            "QLabel#behActionBadge {"
                            f"background: {self._button_border}; color: #ffffff; "
                            "font-size: 9px; font-weight: 600; border-radius: 3px; }"
                        )
                        beh_freq_label.setText("")
                        beh_posture_badge.setText("")
                        beh_compat_badge.setText("")
                        beh_compat_badge.setStyleSheet(
                            "QLabel#behCompatBadge {"
                            "background: #9CA3AF; color: #ffffff; "
                            "font-size: 10px; font-weight: 700; border-radius: 9px; }"
                        )
                        return

                    ast = summary.get("action_space_type", "")
                    abbrev = _ACTION_ABBREV.get(ast, ast[:3].upper() if ast else "—")
                    color = _ACTION_COLOR.get(ast, "#6B7280")
                    beh_action_badge.setText(abbrev)
                    beh_action_badge.setToolTip(f"Action space: {ast}")
                    beh_action_badge.setStyleSheet(
                        "QLabel#behActionBadge {"
                        f"background: {color}; color: #ffffff; "
                        "font-size: 9px; font-weight: 600; border-radius: 3px; }"
                    )

                    freq = summary.get("control_frequency_hz", 0)
                    beh_freq_label.setText(f"{freq:.0f}Hz" if freq else "")

                    pre = summary.get("precondition_posture", "")
                    post = summary.get("postcondition_posture", "")
                    if pre or post:
                        beh_posture_badge.setText(f"{pre} → {post}")
                    else:
                        beh_posture_badge.setText("")

                    # Compat badge: green circle (manifest loaded)
                    beh_compat_badge.setText("✓")
                    beh_compat_badge.setToolTip(
                        f"Manifest loaded: {summary.get('skill_id', '')}"
                    )
                    beh_compat_badge.setStyleSheet(
                        "QLabel#behCompatBadge {"
                        "background: #22C55E; color: #ffffff; "
                        "font-size: 10px; font-weight: 700; border-radius: 9px; }"
                    )
                except Exception:
                    pass

            def _on_behavior_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                _update_behavior_manifest_badge()
                self.regenerate_code()

            def _on_behavior_scale_change(value: int):
                speed_value.setText(f"{int(value)}%")
                _on_behavior_change()

            # Register as a pure-select policy picker alongside CheckpointNode
            self._policy_select_combos.append(beh_pid_setting.combo)
            self._refresh_policy_select_combos()
            beh_pid_setting.combo.currentIndexChanged.connect(lambda _i: _on_behavior_change())
            speed_slider.valueChanged.connect(_on_behavior_scale_change)

        elif name == "Sensor":
            sensor_row, sensor_picker = _make_combo_setting_row(
                "sensor_type_row",
                "read",
                items=[
                    "Read IMU",
                    "Read Camera",
                    "Read Ultrasonic",
                    "Read Infrared",
                    "Read Odometry",
                ],
                show_left_dot=False,
                show_right_dot=False,
            )
            alias_row, alias_setting = _make_input_setting_row(
                "sensor_alias_row",
                "as",
                "sensor_data",
                show_left_dot=False,
                show_right_dot=True,
            )
            alias_input = alias_setting.input_edit

            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            sensor_out = _mk_port(w - 12, h / 2, "out", "out", radius=4, channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_sensor():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(sensor_row)
                if y is not None:
                    flow_in.setPos(0, y)
                    flow_out.setPos(w_now, y)
                y_alias = _port_y(alias_row)
                if y_alias is not None:
                    out_center = _widget_center_in_node(alias_setting.right_box) if alias_setting.right_box else None
                    sensor_out.setPos(out_center.x(), out_center.y()) if out_center is not None else sensor_out.setPos(w_now - 12, y_alias)

            def _sync_layout_sensor():
                _resize_to_fit()
                _sync_ports_sensor()

            QTimer.singleShot(0, _sync_layout_sensor)
            rect._sync_layout_fn = _sync_layout_sensor
            resize_ports_cb = _sync_ports_sensor
            rect._sensor_type_picker = sensor_picker
            rect._sensor_output_alias_input = alias_input

            def _on_sensor_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            sensor_picker.combo.currentIndexChanged.connect(lambda _i: _on_sensor_change())
            alias_input.textChanged.connect(lambda _t: _on_sensor_change())

        elif name == "Trigger":
            event_row, event_setting = _make_combo_setting_row(
                "trigger_event_row",
                "event",
                items=[],
                show_left_dot=False,
                show_right_dot=False,
            )
            event_combo = event_setting.combo
            self._wire_symbol_combo(event_combo, "event")
            event_input = event_combo.lineEdit()

            target_row, target_setting = _make_combo_setting_row(
                "trigger_target_row",
                "target",
                items=[],
                show_left_dot=False,
                show_right_dot=False,
            )
            target_combo = target_setting.combo
            self._wire_symbol_combo(target_combo, "script")
            target_input = target_combo.lineEdit()

            timeout_row, timeout_setting = _make_input_setting_row(
                "trigger_timeout_row",
                "timeout",
                "0",
                show_left_dot=False,
                show_right_dot=False,
            )
            timeout_input = timeout_setting.input_edit
            timeout_input.setValidator(QDoubleValidator(0.0, 300.0, 3, timeout_input))

            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")

            def _sync_ports_trigger():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(event_row)
                if y is not None:
                    flow_in.setPos(0, y)
                    flow_out.setPos(w_now, y)

            def _sync_layout_trigger():
                _resize_to_fit()
                _sync_ports_trigger()

            QTimer.singleShot(0, _sync_layout_trigger)
            rect._sync_layout_fn = _sync_layout_trigger
            resize_ports_cb = _sync_ports_trigger
            rect._trigger_event_input = event_input
            rect._trigger_target_input = target_input
            rect._trigger_timeout_input = timeout_input
            rect._trigger_event_combo = event_combo
            rect._trigger_target_combo = target_combo
            # Keep legacy aliases so save/load and IR lowering stay compatible.
            rect._event_input = event_input
            rect._timeout_input = timeout_input

            def _on_trigger_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            event_combo.currentTextChanged.connect(lambda _t: _on_trigger_change())
            target_combo.currentTextChanged.connect(lambda _t: _on_trigger_change())
            timeout_input.textChanged.connect(lambda _t: _on_trigger_change())

        elif name == "Checkpoint":
            # Pure select dropdown — items populated by refresh_policy_registry()
            # (Circle 3+).  No free-text entry; format is read from the manifest
            # automatically and never exposed to the user.
            policy_id_row, policy_id_setting = _make_combo_setting_row(
                "ckpt_policy_id_row",
                "Checkpoint",
                items=[],
                show_left_dot=False,
                show_right_dot=True,
            )
            status_badge = QLabel("❌")
            status_badge.setObjectName("ckptValidationBadge")
            status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_badge.setFixedSize(20, 20)
            status_badge.setToolTip("未选择 Checkpoint。")
            policy_id_setting.set_middle_widget(status_badge)

            checkpoint_out_ckpt = _mk_port(w - 12, h / 2, "out", "checkpoint", radius=4,
                                           channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_ckpt():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_pid = _port_y(policy_id_row)
                if y_pid is not None:
                    out_center = _widget_center_in_node(policy_id_setting.right_box) if policy_id_setting.right_box else None
                    checkpoint_out_ckpt.setPos(out_center.x(), out_center.y()) if out_center is not None else checkpoint_out_ckpt.setPos(w_now - 12, y_pid)

            def _sync_layout_ckpt():
                _resize_to_fit()
                _sync_ports_ckpt()

            QTimer.singleShot(0, _sync_layout_ckpt)
            rect._sync_layout_fn = _sync_layout_ckpt
            resize_ports_cb = _sync_ports_ckpt
            rect._ckpt_policy_id_picker = policy_id_setting
            rect._ckpt_validation_badge = status_badge

            def _on_ckpt_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                _apply_ckpt_validation_state()
                self.regenerate_code()

            # Register as a pure-select (non-editable) policy picker and populate now
            self._policy_select_combos.append(policy_id_setting.combo)
            self._refresh_policy_select_combos()
            policy_id_setting.combo.currentIndexChanged.connect(lambda _i: _on_ckpt_change())

            # "Open Training Ground" action button (Phase A2)
            train_btn_row_w = QWidget()
            train_btn_row_w.setObjectName("ckpt_train_btn_row")
            train_btn_row_w.setStyleSheet("QWidget { border: 0px; background: transparent; }")
            train_btn_row_layout = QHBoxLayout(train_btn_row_w)
            train_btn_row_layout.setContentsMargins(8, 2, 8, 4)
            train_btn = QPushButton("Open Training Ground")
            train_btn.setObjectName("ckptTrainBtn")
            train_btn.setStyleSheet(self._button_style)
            train_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            train_btn.setEnabled(False)
            train_btn_row_layout.addWidget(train_btn)
            row_stack.add_row("ckpt_train_btn_row", train_btn_row_w)
            rect._ckpt_train_btn = train_btn

            def _apply_ckpt_validation_state():
                state = self._checkpoint_validation_state(policy_id_setting.combo.currentText())
                symbol = state.get("symbol", "❌")
                level = state.get("level", "invalid")
                tooltip = state.get("tooltip", "")
                color = {
                    "runnable": "#22C55E",
                    "trainable": "#F59E0B",
                    "invalid": "#EF4444",
                }.get(level, "#9CA3AF")
                status_badge.setText(symbol)
                status_badge.setToolTip(tooltip)
                status_badge.setStyleSheet(
                    "QLabel#ckptValidationBadge {"
                    f"background: {color}; color: #ffffff; "
                    f"border: 1px solid {self._button_border};"
                    "font-size: 11px; font-weight: 700; }"
                )
                train_btn.setEnabled(True)
                train_btn.setToolTip("打开 Training Ground。")

            def _update_train_btn():
                _apply_ckpt_validation_state()

            def _on_train_btn_clicked():
                pid = policy_id_setting.combo.currentText().strip()
                self.workspace_open_requested.emit(pid)

            policy_id_setting.combo.currentTextChanged.connect(lambda _t: _update_train_btn())
            train_btn.clicked.connect(_on_train_btn_clicked)
            _update_train_btn()

        elif name == "Train Input":
            # Fixed canvas anchor: outputs robot_type → EnvConfigNode robot_type input
            ti_robot_row, ti_robot_setting = _make_input_setting_row(
                "ti_robot_row", "robot_type", "go2",
                show_left_dot=False, show_right_dot=True,
            )

            ti_out = _mk_port(w - 12, h / 2, "out", "robot_type", radius=4,
                              channel="data", data_type="string", dot_kind="sub_dot")

            def _sync_ports_ti():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(ti_robot_row)
                if y is not None:
                    out_center = _widget_center_in_node(ti_robot_setting.right_box) if ti_robot_setting.right_box else None
                    ti_out.setPos(out_center.x(), out_center.y()) if out_center else ti_out.setPos(w_now - 12, y)

            def _sync_layout_ti():
                _resize_to_fit()
                _sync_ports_ti()

            QTimer.singleShot(0, _sync_layout_ti)
            rect._sync_layout_fn = _sync_layout_ti
            resize_ports_cb = _sync_ports_ti
            rect._ti_robot_input = ti_robot_setting.input_edit
            rect._is_protected = True
            delete_btn.setVisible(False)
            header_title.setText("INPUT")

            def _on_ti_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)

            ti_robot_setting.input_edit.textChanged.connect(lambda _t: _on_ti_change())

        elif name == "Train Output":
            # Fixed canvas anchor: receives training result dict from TrainNode
            to_cp_row, to_cp_setting = _make_input_setting_row(
                "to_cp_row", "checkpoint", "trained_policy",
                show_left_dot=True, show_right_dot=False,
                input_slot="checkpoint_name",
            )

            to_in = _mk_port(12, h / 2, "in", "checkpoint_name", radius=4,
                             channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_to():
                if not isValid(rect):
                    return
                y = _port_y(to_cp_row)
                if y is not None:
                    in_center = _widget_center_in_node(to_cp_setting.left_box) if to_cp_setting.left_box else None
                    to_in.setPos(in_center.x(), in_center.y()) if in_center else to_in.setPos(12, y)

            def _sync_layout_to():
                _resize_to_fit()
                _sync_ports_to()

            QTimer.singleShot(0, _sync_layout_to)
            rect._sync_layout_fn = _sync_layout_to
            resize_ports_cb = _sync_ports_to
            rect._to_cp_input = to_cp_setting.input_edit
            rect._is_protected = True
            delete_btn.setVisible(False)
            header_title.setText("OUTPUT")

        elif name == "Env Config":
            # Rows: robot_type (with left input dot), scene_xml, obs_components, reward_fn, max_steps
            ec_robot_row, ec_robot_setting = _make_input_setting_row(
                "ec_robot_row", "robot_type", "go2",
                show_left_dot=True, show_right_dot=False,
            )
            ec_scene_row, ec_scene_setting = _make_input_setting_row(
                "ec_scene_row", "scene_xml", "go2.xml",
                show_left_dot=False, show_right_dot=False,
            )
            ec_obs_row, ec_obs_setting = _make_input_setting_row(
                "ec_obs_row", "obs_components", "joint_pos joint_vel imu",
                show_left_dot=False, show_right_dot=False,
            )
            ec_reward_row, ec_reward_setting = _make_input_setting_row(
                "ec_reward_row", "reward_fn", "default",
                show_left_dot=False, show_right_dot=False,
            )
            ec_steps_row, ec_steps_setting = _make_input_setting_row(
                "ec_steps_row", "max_steps", "1000",
                show_left_dot=False, show_right_dot=True,
            )

            robot_type_in_port = _mk_port(12, h / 4, "in", "robot_type", radius=4,
                                          channel="data", data_type="string", dot_kind="sub_dot")
            env_config_out = _mk_port(w - 12, h / 2, "out", "env_config", radius=4,
                                      channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_ec():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_robot = _port_y(ec_robot_row)
                if y_robot is not None:
                    in_center = _widget_center_in_node(ec_robot_setting.left_box) if ec_robot_setting.left_box else None
                    robot_type_in_port.setPos(in_center.x(), in_center.y()) if in_center else robot_type_in_port.setPos(12, y_robot)
                y = _port_y(ec_steps_row)
                if y is not None:
                    out_center = _widget_center_in_node(ec_steps_setting.right_box) if ec_steps_setting.right_box else None
                    env_config_out.setPos(out_center.x(), out_center.y()) if out_center else env_config_out.setPos(w_now - 12, y)

            def _sync_layout_ec():
                _resize_to_fit()
                _sync_ports_ec()

            QTimer.singleShot(0, _sync_layout_ec)
            rect._sync_layout_fn = _sync_layout_ec
            resize_ports_cb = _sync_ports_ec
            rect._ec_robot_input = ec_robot_setting.input_edit
            rect._ec_scene_input = ec_scene_setting.input_edit
            rect._ec_obs_input = ec_obs_setting.input_edit
            rect._ec_reward_input = ec_reward_setting.input_edit
            rect._ec_steps_input = ec_steps_setting.input_edit
            header_title.setText("ENV CONFIG")

            def _on_ec_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            for _ec_w in (ec_robot_setting.input_edit, ec_scene_setting.input_edit,
                          ec_obs_setting.input_edit, ec_reward_setting.input_edit,
                          ec_steps_setting.input_edit):
                _ec_w.textChanged.connect(lambda _t: _on_ec_change())

        elif name == "Train Config":
            # Combo for algorithm + text inputs for key hyperparams
            tc_algo_row, tc_algo_setting = _make_combo_setting_row(
                "tc_algo_row", "algorithm", ["SAC", "PPO", "TD3"],
                show_left_dot=True, show_right_dot=False,
            )
            tc_steps_row, tc_steps_setting = _make_input_setting_row(
                "tc_steps_row", "total_timesteps", "1000000",
                show_left_dot=True, show_right_dot=False,
            )
            tc_pid_row, tc_pid_setting = _make_input_setting_row(
                "tc_pid_row", "policy_id_out", "trained_policy",
                show_left_dot=False, show_right_dot=True,
            )

            # Input ports: env_config (top-left) and checkpoint (mid-left)
            env_cfg_in_port = _mk_port(12, h * 0.35, "in", "env_config", radius=4,
                                       channel="data", data_type="dict", dot_kind="sub_dot")
            checkpoint_in_port = _mk_port(12, h * 0.55, "in", "checkpoint", radius=4,
                                          channel="data", data_type="dict", dot_kind="sub_dot")
            train_cfg_out = _mk_port(w - 12, h / 2, "out", "train_config", radius=4,
                                     channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_tc():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y_algo = _port_y(tc_algo_row)
                y_steps = _port_y(tc_steps_row)
                y_pid = _port_y(tc_pid_row)
                if y_algo is not None:
                    in_center = _widget_center_in_node(tc_algo_setting.left_box) if tc_algo_setting.left_box else None
                    env_cfg_in_port.setPos(in_center.x(), in_center.y()) if in_center else env_cfg_in_port.setPos(12, y_algo)
                if y_steps is not None:
                    in_center2 = _widget_center_in_node(tc_steps_setting.left_box) if tc_steps_setting.left_box else None
                    checkpoint_in_port.setPos(in_center2.x(), in_center2.y()) if in_center2 else checkpoint_in_port.setPos(12, y_steps)
                if y_pid is not None:
                    out_center = _widget_center_in_node(tc_pid_setting.right_box) if tc_pid_setting.right_box else None
                    train_cfg_out.setPos(out_center.x(), out_center.y()) if out_center else train_cfg_out.setPos(w_now - 12, y_pid)

            def _sync_layout_tc():
                _resize_to_fit()
                _sync_ports_tc()

            QTimer.singleShot(0, _sync_layout_tc)
            rect._sync_layout_fn = _sync_layout_tc
            resize_ports_cb = _sync_ports_tc
            rect._tc_algo_picker = tc_algo_setting
            rect._tc_steps_input = tc_steps_setting.input_edit
            rect._tc_pid_input = tc_pid_setting.input_edit
            header_title.setText("TRAIN CONFIG")

            def _on_tc_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            tc_algo_setting.combo.currentIndexChanged.connect(lambda _i: _on_tc_change())
            tc_steps_setting.input_edit.textChanged.connect(lambda _t: _on_tc_change())
            tc_pid_setting.input_edit.textChanged.connect(lambda _t: _on_tc_change())

        elif name == "Task Config":
            # Rows: task_type, command_mode, curriculum (output: task_config)
            tsk_type_row, tsk_type_setting = _make_combo_setting_row(
                "tsk_type_row", "task_type",
                ["velocity_tracking", "locomotion", "manipulation", "custom"],
                show_left_dot=False, show_right_dot=False,
            )
            tsk_cmd_row, tsk_cmd_setting = _make_combo_setting_row(
                "tsk_cmd_row", "command_mode", ["fixed", "variable"],
                show_left_dot=False, show_right_dot=False,
            )
            tsk_curric_row, tsk_curric_setting = _make_combo_setting_row(
                "tsk_curric_row", "curriculum", ["false", "true"],
                show_left_dot=False, show_right_dot=True,
            )

            task_config_out = _mk_port(w - 12, h / 2, "out", "task_config", radius=4,
                                       channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_tsk():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(tsk_curric_row)
                if y is not None:
                    out_center = _widget_center_in_node(tsk_curric_setting.right_box) if tsk_curric_setting.right_box else None
                    task_config_out.setPos(out_center.x(), out_center.y()) if out_center else task_config_out.setPos(w_now - 12, y)

            def _sync_layout_tsk():
                _resize_to_fit()
                _sync_ports_tsk()

            QTimer.singleShot(0, _sync_layout_tsk)
            rect._sync_layout_fn = _sync_layout_tsk
            resize_ports_cb = _sync_ports_tsk
            rect._tsk_type_picker = tsk_type_setting
            rect._tsk_cmd_picker = tsk_cmd_setting
            rect._tsk_curric_picker = tsk_curric_setting
            header_title.setText("TASK CONFIG")

            def _on_tsk_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            tsk_type_setting.combo.currentIndexChanged.connect(lambda _i: _on_tsk_change())
            tsk_cmd_setting.combo.currentIndexChanged.connect(lambda _i: _on_tsk_change())
            tsk_curric_setting.combo.currentIndexChanged.connect(lambda _i: _on_tsk_change())

        elif name == "Eval Config":
            # Rows: eval_episodes, success_threshold, deterministic (output: eval_config)
            ev_eps_row, ev_eps_setting = _make_input_setting_row(
                "ev_eps_row", "eval_episodes", "10",
                show_left_dot=False, show_right_dot=False,
            )
            ev_thresh_row, ev_thresh_setting = _make_input_setting_row(
                "ev_thresh_row", "success_threshold", "0.8",
                show_left_dot=False, show_right_dot=False,
            )
            ev_det_row, ev_det_setting = _make_combo_setting_row(
                "ev_det_row", "deterministic", ["true", "false"],
                show_left_dot=False, show_right_dot=True,
            )

            eval_config_out = _mk_port(w - 12, h / 2, "out", "eval_config", radius=4,
                                       channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_ev():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(ev_det_row)
                if y is not None:
                    out_center = _widget_center_in_node(ev_det_setting.right_box) if ev_det_setting.right_box else None
                    eval_config_out.setPos(out_center.x(), out_center.y()) if out_center else eval_config_out.setPos(w_now - 12, y)

            def _sync_layout_ev():
                _resize_to_fit()
                _sync_ports_ev()

            QTimer.singleShot(0, _sync_layout_ev)
            rect._sync_layout_fn = _sync_layout_ev
            resize_ports_cb = _sync_ports_ev
            rect._ev_eps_input = ev_eps_setting.input_edit
            rect._ev_thresh_input = ev_thresh_setting.input_edit
            rect._ev_det_picker = ev_det_setting
            header_title.setText("EVAL CONFIG")

            def _on_ev_change():
                self._update_node_params(rect)
                self._sync_node_parameters(rect)
                self.regenerate_code()

            ev_eps_setting.input_edit.textChanged.connect(lambda _t: _on_ev_change())
            ev_thresh_setting.input_edit.textChanged.connect(lambda _t: _on_ev_change())
            ev_det_setting.combo.currentIndexChanged.connect(lambda _i: _on_ev_change())

        elif name == "SB3 Train":
            # Minimal shell: input port (train_config) + output port (result)
            train_tag_row, train_tag_layout = _make_row("none")
            train_tag_layout.addWidget(make_tag("train_config \u2192 result", self._tag_style))
            _finish_row(train_tag_layout, "none")
            row_stack.add_row("train_tag_row", train_tag_row)

            train_cfg_in = _mk_port(12, h / 2, "in", "train_config", radius=4,
                                    channel="data", data_type="dict", dot_kind="sub_dot")
            result_out = _mk_port(w - 12, h / 2, "out", "result", radius=4,
                                  channel="data", data_type="dict", dot_kind="sub_dot")

            def _sync_ports_train():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(train_tag_row)
                if y is not None:
                    train_cfg_in.setPos(12, y)
                    result_out.setPos(w_now - 12, y)

            def _sync_layout_train():
                _resize_to_fit()
                _sync_ports_train()

            QTimer.singleShot(0, _sync_layout_train)
            rect._sync_layout_fn = _sync_layout_train
            resize_ports_cb = _sync_ports_train
            header_title.setText("TRAIN")

        else:
            default_text = "< logic >"
            if "Action Execution" in name:
                default_text = "< action >"
            elif "Sensor Input" in name:
                default_text = "< sensor >"
            elif name == "Sensor":
                default_text = "< sensor >"
            elif name == "Trigger":
                default_text = "< trigger >"
            elif "Math" in name:
                default_text = "< math >"

            default_cond = ConditionSet(
                self._condition_set_left_bg, self._condition_set_right_bg, self._button_border,
                self._condition_set_text, self._hover_bg, default_text,
            )

            selection_row, selection_layout = _make_row("none")
            selection_layout.addWidget(default_cond)
            _finish_row(selection_layout, "input")
            row_stack.add_row("selection_row", selection_row)

            flow_in = _mk_port(0, h / 2, "in", "flow_in", radius=6, channel="flow")
            flow_out = _mk_port(w, h / 2, "out", "flow_out", radius=6, channel="flow")
            default_cond_port = _mk_port(12, h / 2, "in", "condition", channel="data", data_type="bool", dot_kind="sub_dot")

            def _sync_ports_default():
                if not isValid(rect):
                    return
                w_now = rect.rect().width()
                y = _port_y(selection_row)
                if y is not None:
                    flow_in.setPos(0, y)
                    flow_out.setPos(w_now, y)
                    cc = _widget_center_in_node(default_cond.left_box)
                    if cc is not None:
                        default_cond_port.setPos(cc.x(), cc.y())

            def _sync_layout_default():
                _resize_to_fit()
                _sync_ports_default()

            QTimer.singleShot(0, _sync_layout_default)
            rect._sync_layout_fn = _sync_layout_default
            resize_ports_cb = _sync_ports_default
            rect._condition_set = default_cond
            rect._condition_input = default_cond.button
            default_cond.button.clicked.connect(lambda: self._update_node_params(rect))

        _FLOW_BRANCH_LAYOUT_NODES = {
            "Logic Control",
            "Loop",
            "Gate",
            "Timeout",
            "Retry",
            "Switch",
            "Try",
            "Parallel",
            "Join",
        }
        if name not in _FLOW_BRANCH_LAYOUT_NODES:
            align_main_ports_cb = _align_main_flow_ports_to_title_row
            rect._align_main_ports_fn = _align_main_flow_ports_to_title_row

            # Ensure any node-specific sync layout (including delayed timers and
            # future mode-triggered sync calls) always ends with title-row main
            # port alignment for canonical flow_in/flow_out.
            sync_layout_fn = getattr(rect, "_sync_layout_fn", None)
            if callable(sync_layout_fn):
                def _sync_layout_with_main_align(_base=sync_layout_fn):
                    _base()
                    _align_main_flow_ports_to_title_row()
                rect._sync_layout_fn = _sync_layout_with_main_align

        def _on_node_resized():
            _apply_content_width()
            if callable(resize_ports_cb):
                resize_ports_cb()
            if callable(align_main_ports_cb):
                align_main_ports_cb()

        rect.set_resize_callbacks(min_size_fn=_min_size_for_content, resized_cb=_on_node_resized)
        if callable(resize_ports_cb):
            _resize_to_fit()
            resize_ports_cb()
        if callable(align_main_ports_cb):
            align_main_ports_cb()
            # Run once more after pending QTimer(0) sync callbacks to prevent
            # node-local sync functions from leaving main ports in content rows.
            QTimer.singleShot(0, align_main_ports_cb)

        node_id = self._node_seq
        self._node_seq += 1
        rect.setData(10, "node")
        rect.setData(11, name)
        rect.setData(12, node_id)
        delete_btn.clicked.connect(lambda _=False, item=rect: self._delete_items([item]))

        if open_subgraph_controls:
            def _open_nested_editor():
                node_title = (rect.data(11) or "")
                if node_title == "Script Box":
                    kind = "script"
                elif node_title == "Start":
                    kind = "settings"
                else:
                    kind = node_title.lower()
                cond_set = getattr(rect, "_condition_set", None)
                if kind == "script":
                    ref = getattr(rect, "_script_ref", "") or f"script_{node_id}"
                    rect._script_ref = ref
                    ref_input = getattr(rect, "_script_ref_input", None)
                    if ref_input is not None:
                        ref_input.setText(ref)
                    self._record_symbol("script", ref)
                elif kind == "settings":
                    ref = "settings"
                elif kind == "behavior":
                    ref = self._ensure_behavior_ref(rect)
                    display_name = cond_set.logic_text() if cond_set else ""
                    combo = getattr(cond_set, "combo", None) if cond_set else None
                    combo_data = combo.currentData(Qt.ItemDataRole.UserRole) if combo is not None else {}
                    if not isinstance(combo_data, dict):
                        combo_data = {}
                    intent_id = str(combo_data.get("intent_id") or "").strip()
                else:
                    ref = cond_set.logic_text() if cond_set else ""
                if callable(self._subgraph_opener):
                    kwargs = {"kind": kind, "node_id": node_id, "ref": ref, "node_item": rect}
                    if kind == "behavior":
                        kwargs["display_name"] = display_name
                        kwargs["intent_id"] = intent_id
                    self._subgraph_opener(**kwargs)
                else:
                    log_info(f"Open nested editor requested: kind={kind}, node_id={node_id}, ref={ref}")

            for _ctrl in open_subgraph_controls:
                _ctrl.clicked.connect(_open_nested_editor)

        logic_node = self._create_logic_node(name, node_id, rect)
        if logic_node:
            self._logic_nodes[node_id] = logic_node
            rect.setData(13, logic_node)

        self.regenerate_code()
        log_info(f"Node created: {name} (ID: {node_id})")

        # Dirty + undo history (no-op during bulk load / suppress_history)
        self._touch_content()

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
        if node_type == "opaque_code":
            # Script Box is currently a shell node; logic/behavior is delegated
            # to a future Behavior-layer editor.
            return None

        try:
            logic_node = create_logic_node(node_type, str(node_id))
            log_debug(f"Created logic node: {node_type} (ID: {node_id})")
            return logic_node
        except Exception as e:
            log_error(f"Failed to create logic node: {e}")
            return None

    def _on_combo_changed(self, rect_item):
        """Handle combo box selection change"""
        self._sync_node_parameters(rect_item)
        self.regenerate_code()

    def _sync_node_parameters(self, rect_item):
        """Sync UI values to logic node parameters"""
        node_id = rect_item.data(12)
        logic_node = self._logic_nodes.get(node_id)
        if not logic_node:
            return

        name = rect_item.data(11)
        cond_set_generic = getattr(rect_item, '_condition_set', None)

        # Sync based on node type
        if name == "Start":
            brand_combo = getattr(rect_item, '_brand_picker', None)
            model_combo = getattr(rect_item, '_model_picker', None)
            if brand_combo:
                logic_node.set_parameter(
                    'robot_brand',
                    brand_combo.currentData(Qt.ItemDataRole.UserRole) or brand_combo.currentText(),
                )
            if model_combo:
                logic_node.set_parameter('robot_type', model_combo.currentText())
            return

        elif name == "End":
            return  # End node has no parameters to sync

        elif "Action Execution" in name and cond_set_generic:
            action = cond_set_generic.logic_text()
            robot_action = self._action_mapping.get(action, action.lower().replace(" ", "_"))
            logic_node.set_parameter('action', robot_action)

        elif name == "Behavior":
            pid_picker = getattr(rect_item, '_beh_policy_id_picker', None)
            if pid_picker:
                logic_node.set_parameter('policy_id', pid_picker.combo.currentText())
            scale_slider = getattr(rect_item, '_beh_command_scale_slider', None)
            if scale_slider is not None:
                logic_node.set_parameter('command_scale', float(scale_slider.value()) / 100.0)

        elif "Sensor Input" in name and cond_set_generic:
            sensor_map = {
                "Read Ultrasonic": "ultrasonic",
                "Read Infrared": "infrared",
                "Read Camera": "camera",
                "Read IMU": "imu",
                "Read Odometry": "odometry"
            }
            sensor_type = sensor_map.get(cond_set_generic.logic_text(), "imu")
            logic_node.set_parameter('sensor_type', sensor_type)

        elif name == "Sensor":
            sensor_map = {
                "Read Ultrasonic": "ultrasonic",
                "Read Infrared": "infrared",
                "Read Camera": "camera",
                "Read IMU": "imu",
                "Read Odometry": "odometry",
            }
            sensor_picker = getattr(rect_item, "_sensor_type_picker", None)
            selected = sensor_picker.current_text() if sensor_picker else ""
            if not selected and cond_set_generic:
                selected = cond_set_generic.logic_text()
            logic_node.set_parameter('sensor_type', sensor_map.get(selected, "imu"))

        elif name == "Trigger":
            trigger_event_input = getattr(rect_item, "_trigger_event_input", None)
            trigger_timeout_input = getattr(rect_item, "_trigger_timeout_input", None)
            trigger_event = trigger_event_input.text().strip() if trigger_event_input else ""
            logic_node.set_parameter('mode', 'event')
            logic_node.set_parameter('event_name', trigger_event or "trigger_event")
            try:
                timeout_val = float(trigger_timeout_input.text() or 0.0) if trigger_timeout_input else 0.0
            except ValueError:
                timeout_val = 0.0
            logic_node.set_parameter('timeout', timeout_val)

        elif "Logic Control" in name:
            cond_set = getattr(rect_item, '_condition_set', None)
            cond_input = getattr(rect_item, '_condition_input', None)
            if cond_set:
                logic_node.set_parameter('condition_expr', cond_set.logic_text())
            elif cond_input:
                logic_node.set_parameter('condition_expr', cond_input.text())
            elif_inputs = getattr(rect_item, '_elif_inputs', [])
            logic_node.set_parameter('elif_conditions', [w.logic_text() if hasattr(w, 'logic_text') else w.text() for w in elif_inputs])

        elif name == "Loop":
            type_picker = getattr(rect_item, '_loop_type_picker', None)
            loop_type = type_picker.combo.currentText().lower() if type_picker else 'while'
            logic_node.set_parameter('loop_type', loop_type)
            if loop_type == 'while':
                cond_set = getattr(rect_item, '_condition_set', None)
                cond_input = getattr(rect_item, '_condition_input', None)
                if cond_set:
                    logic_node.set_parameter('condition_expr', cond_set.logic_text())
                elif cond_input:
                    logic_node.set_parameter('condition_expr', cond_input.text())
            else:
                start_inp = getattr(rect_item, '_for_start_input', None)
                end_inp = getattr(rect_item, '_for_end_input', None)
                step_inp = getattr(rect_item, '_for_step_input', None)
                try:
                    logic_node.set_parameter('for_start', int(start_inp.text()) if start_inp and start_inp.text() else 0)
                except ValueError:
                    logic_node.set_parameter('for_start', 0)
                try:
                    logic_node.set_parameter('for_end', int(end_inp.text()) if end_inp and end_inp.text() else 10)
                except ValueError:
                    logic_node.set_parameter('for_end', 10)
                try:
                    logic_node.set_parameter('for_step', int(step_inp.text()) if step_inp and step_inp.text() else 1)
                except ValueError:
                    logic_node.set_parameter('for_step', 1)

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
            cond_set = getattr(rect_item, '_condition_set', None)
            if cond_set:
                op_text = cond_set.logic_text()
                logic_node.set_parameter('condition_expr', op_text)
                op_map = {
                    "Equal": "==", "Not Equal": "!=",
                    "Greater Than": ">", "Less Than": "<",
                    "Greater Equal": ">=", "Less Equal": "<=",
                }
                logic_node.set_parameter('operator', op_map.get(op_text, op_text))

        elif "Math" in name and cond_set_generic:
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
            logic_node.set_parameter('operation', op_map.get(cond_set_generic.logic_text(), "add"))
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

        elif name == "Wait":
            mode_picker = getattr(rect_item, '_wait_mode_picker', None)
            if mode_picker:
                logic_node.set_parameter('mode', mode_picker.current_text())
            dur_input = getattr(rect_item, '_duration_input', None)
            if dur_input:
                try:
                    logic_node.set_parameter('duration', float(dur_input.text() or 1.0))
                except ValueError:
                    logic_node.set_parameter('duration', 1.0)
            ev_input = getattr(rect_item, '_event_input', None)
            if ev_input:
                logic_node.set_parameter('event_name', ev_input.text())
            to_input = getattr(rect_item, '_timeout_input', None)
            if to_input:
                try:
                    logic_node.set_parameter('timeout', float(to_input.text() or 0))
                except ValueError:
                    logic_node.set_parameter('timeout', 0)

        elif name == "Gate":
            mode_picker = getattr(rect_item, '_gate_mode_picker', None)
            if mode_picker:
                logic_node.set_parameter('mode', mode_picker.current_text())
            cond_input = getattr(rect_item, '_condition_input', None)
            if cond_input:
                logic_node.set_parameter('condition_expr', cond_input.text())
            ev_input = getattr(rect_item, '_event_input', None)
            if ev_input:
                logic_node.set_parameter('event_name', ev_input.text())

        elif name == "Timeout":
            dur_input = getattr(rect_item, '_duration_input', None)
            if dur_input:
                try:
                    logic_node.set_parameter('duration', float(dur_input.text() or 5.0))
                except ValueError:
                    logic_node.set_parameter('duration', 5.0)

        elif name == "Retry":
            att_input = getattr(rect_item, '_attempts_input', None)
            if att_input:
                try:
                    logic_node.set_parameter('max_attempts', int(att_input.text() or 3))
                except ValueError:
                    logic_node.set_parameter('max_attempts', 3)
            bo_input = getattr(rect_item, '_backoff_input', None)
            if bo_input:
                try:
                    logic_node.set_parameter('backoff_sec', float(bo_input.text() or 1.0))
                except ValueError:
                    logic_node.set_parameter('backoff_sec', 1.0)
            strat_picker = getattr(rect_item, '_strategy_picker', None)
            if strat_picker:
                logic_node.set_parameter('strategy', strat_picker.current_text())

        elif name == "Cancel":
            target_picker = getattr(rect_item, '_cancel_target_picker', None)
            if target_picker:
                logic_node.set_parameter('target', target_picker.current_text())
            tag_input = getattr(rect_item, '_cancel_tag_input', None)
            if tag_input:
                logic_node.set_parameter('tag', tag_input.text())

        elif name == "Abort":
            reason_input = getattr(rect_item, '_abort_reason_input', None)
            if reason_input:
                logic_node.set_parameter('reason', reason_input.text())
            em_picker = getattr(rect_item, '_abort_emergency_picker', None)
            if em_picker:
                logic_node.set_parameter('emergency', em_picker.current_text() == "true")

        elif name == "Break":
            level_input = getattr(rect_item, '_break_level_input', None)
            try:
                level = int(level_input.text()) if level_input and level_input.text() else 1
            except ValueError:
                level = 1
            logic_node.set_parameter('break_level', max(1, level))

        elif name == "Checkpoint":
            pid_picker = getattr(rect_item, '_ckpt_policy_id_picker', None)
            if pid_picker:
                logic_node.set_parameter('policy_id', pid_picker.combo.currentText())

        elif name == "Env Config":
            for attr, param in (
                ("_ec_robot_input", "robot_type"),
                ("_ec_scene_input", "scene_xml"),
                ("_ec_obs_input", "obs_components"),
                ("_ec_reward_input", "reward_fn"),
                ("_ec_steps_input", "max_steps"),
            ):
                w = getattr(rect_item, attr, None)
                if w:
                    logic_node.set_parameter(param, w.text())

        elif name == "Train Config":
            algo_picker = getattr(rect_item, "_tc_algo_picker", None)
            if algo_picker:
                logic_node.set_parameter("algorithm", algo_picker.combo.currentText())
            steps_inp = getattr(rect_item, "_tc_steps_input", None)
            if steps_inp:
                logic_node.set_parameter("total_timesteps", steps_inp.text())
            pid_inp = getattr(rect_item, "_tc_pid_input", None)
            if pid_inp:
                logic_node.set_parameter("policy_id_out", pid_inp.text())

        elif name == "SB3 Train":
            pass  # No user-editable parameters in S1

        elif name == "Task Config":
            for attr, param in (
                ("_tsk_type_picker", "task_type"),
                ("_tsk_cmd_picker", "command_mode"),
                ("_tsk_curric_picker", "curriculum"),
            ):
                picker = getattr(rect_item, attr, None)
                if picker and hasattr(picker, "combo"):
                    logic_node.set_parameter(param, picker.combo.currentText())

        elif name == "Eval Config":
            for attr, param in (
                ("_ev_eps_input", "eval_episodes"),
                ("_ev_thresh_input", "success_threshold"),
            ):
                w = getattr(rect_item, attr, None)
                if w:
                    logic_node.set_parameter(param, w.text())
            det_picker = getattr(rect_item, "_ev_det_picker", None)
            if det_picker and hasattr(det_picker, "combo"):
                logic_node.set_parameter("deterministic", det_picker.combo.currentText())

        elif "Switch" in name:
            expr_inp = getattr(rect_item, '_switch_expr_input', None)
            if expr_inp:
                logic_node.set_parameter('input_expr', expr_inp.text())
            case_inputs = getattr(rect_item, '_switch_case_inputs', [])
            logic_node.set_parameter('cases', [inp.text() for inp in case_inputs])

        elif "Try" in name:
            exc_inp = getattr(rect_item, '_exc_type_input', None)
            if exc_inp:
                logic_node.set_parameter('exception_types', exc_inp.text() or 'Exception')

        elif "Parallel" in name:
            cnt_picker = getattr(rect_item, '_branch_count_picker', None)
            if cnt_picker:
                try:
                    logic_node.set_parameter('branch_count', int(cnt_picker.current_text()))
                except ValueError:
                    logic_node.set_parameter('branch_count', 2)

        elif "Join" in name or name == "Join":
            cnt_picker = getattr(rect_item, '_join_count_picker', None)
            if cnt_picker:
                try:
                    logic_node.set_parameter('branch_count', int(cnt_picker.current_text()))
                except ValueError:
                    logic_node.set_parameter('branch_count', 2)

    def _get_node_port(self, rect_item, slot: str, io: str):
        """Return the port QGraphicsEllipseItem matching slot name and io direction."""
        for child in rect_item.childItems():
            if not child or not isValid(child):
                continue
            try:
                if child.data(0) == "port" and child.data(3) == slot and child.data(1) == io:
                    return child
            except RuntimeError:
                continue
        return None

    def _find_port_near(self, pos, radius=14):
        """Find port near position"""
        search_rect = QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        candidates = []
        max_dist2 = float(radius * radius)

        for it in self.items(search_rect):
            if self._is_port(it):
                c = self._port_center(it)
                dist2 = (c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2
                if dist2 <= max_dist2:
                    candidates.append((dist2, it))

        return min(candidates, key=lambda t: t[0])[1] if candidates else None

    def get_port_interaction_radius(self) -> int:
        return int(getattr(self, "_port_interaction_radius", 11))

    def _is_port(self, item):
        """Check if item is a port"""
        if not item:
            return False
        try:
            if not isValid(item):
                return False
            return item.data(0) == "port"
        except RuntimeError:
            return False

    def _apply_port_visual(self, port_item, state: str = "normal"):
        """Apply consistent main/sub dot visuals with preview state."""
        if not self._is_port(port_item):
            return
        meta = self._port_meta(port_item)
        dot_kind = meta.get("dot_kind", "main_dot")
        base_border = (
            QColor(self._main_dot_color)
            if dot_kind == "main_dot"
            else _sub_dot_type_color(meta.get("data_type", "any"))
        )
        port_bg = get_color("port_bg", "#1f2937")
        if state == "valid":
            green = QColor("#22c55e")
            pen = QPen(green, 3.5)
            port_item.setOpacity(1.0)
            port_item.setBrush(QBrush(QColor(port_bg)))
            port_item.setPen(pen)
            return
        if state == "invalid":
            gray = QColor(self._port_invalid_color)
            gray.setAlpha(180)
            fill = QColor(self._port_invalid_color)
            fill.setAlpha(80)
            port_item.setOpacity(0.55)
            port_item.setBrush(QBrush(fill))
            port_item.setPen(QPen(gray, 2))
            return
        if state == "hover":
            glow = QColor(base_border).lighter(160)
            port_item.setOpacity(1.0)
            port_item.setBrush(QBrush(QColor(port_bg)))
            port_item.setPen(QPen(glow, 3.5))
            return
        port_item.setOpacity(1.0)
        port_item.setBrush(QBrush(QColor(port_bg)))
        port_item.setPen(QPen(QColor(base_border), 2))

    def _create_dynamic_port(
        self,
        node_item,
        x: float,
        y: float,
        io: str,
        slot: str,
        *,
        channel: str = "data",
        data_type: str = "any",
        max_connections: Optional[int] = None,
        dot_kind: str = "sub_dot",
        radius: Optional[int] = None,
    ):
        """Create a port on an existing node (for dynamic script IO rows)."""
        if not node_item or not isValid(node_item):
            return None
        resolved_dot = dot_kind or ("main_dot" if channel == "flow" else "sub_dot")
        resolved_max = max_connections
        if resolved_max is None and io == "in":
            resolved_max = 1
        is_fan_in = (resolved_max is not None and resolved_max < 0)
        resolved_radius = radius if radius is not None else (6 if resolved_dot == "main_dot" else 4)
        if is_fan_in:
            resolved_radius += 1
        port_cls = _FanInPortItem if is_fan_in else QGraphicsEllipseItem
        p = port_cls(
            -resolved_radius, -resolved_radius, resolved_radius * 2, resolved_radius * 2, node_item
        )
        p.setPos(x, y)
        p.setData(0, "port")
        p.setData(1, io)
        p.setData(2, [])
        p.setData(3, slot)
        p.setData(self._port_meta_key, {
            "channel": channel,
            "data_type": _normalize_data_type(data_type),
            "max_connections": resolved_max,
            "dot_kind": resolved_dot,
        })
        p.setZValue(3)
        p.setAcceptedMouseButtons(Qt.LeftButton)
        p.setAcceptHoverEvents(True)
        p.setCursor(Qt.PointingHandCursor)
        self._apply_port_visual(p, "normal")
        return p

    def _set_port_data_type(self, port_item, data_type: str):
        if not self._is_port(port_item):
            return
        meta = dict(port_item.data(self._port_meta_key) or {})
        meta["data_type"] = _normalize_data_type(data_type)
        port_item.setData(self._port_meta_key, meta)
        self._apply_port_visual(port_item, "normal")
        for conn in list(port_item.data(2) or []):
            if conn and isValid(conn):
                conn.setPen(ConnectionStyleDelegate.make_pen(conn, "normal"))

    @staticmethod
    def _script_next_slot(entries: List[Dict[str, Any]], prefix: str) -> str:
        used = set()
        for e in entries:
            s = str(e.get("slot", ""))
            if s.startswith(prefix):
                used.add(s)
        idx = 0
        while f"{prefix}{idx}" in used:
            idx += 1
        return f"{prefix}{idx}"

    def _script_find_input_entry_by_slot(self, node_item, slot: str) -> Optional[Dict[str, Any]]:
        for entry in list(getattr(node_item, "_script_input_rows", []) or []):
            if str(entry.get("slot", "")) == str(slot or ""):
                return entry
        return None

    def _script_find_input_entry_by_port(self, node_item, port_item) -> Optional[Dict[str, Any]]:
        for entry in list(getattr(node_item, "_script_input_rows", []) or []):
            if entry.get("port") is port_item:
                return entry
        return None

    def _script_find_output_entry_by_slot(self, node_item, slot: str) -> Optional[Dict[str, Any]]:
        for entry in list(getattr(node_item, "_script_output_rows", []) or []):
            if str(entry.get("slot", "")) == str(slot or ""):
                return entry
        return None

    def _script_set_row_dot_style(self, entry: Dict[str, Any], *, io: str, data_type: str):
        box_key = "left_box" if io == "in" else "right_box"
        box = entry.get(box_key)
        if box is None or not isValid(box):
            return
        color = _sub_dot_type_color(data_type).name()
        border_left = "0px" if io == "out" else f"1px solid {color}"
        style = (
            "QFrame { "
            f"border: 1px solid {color}; border-left: {border_left}; "
            f"background: {get_color('port_bg', '#1f2937')}; border-radius: 0px; }}"
        )
        box.setStyleSheet(style)

    def _script_sync_io_spec(self, node_item):
        if not node_item or not isValid(node_item):
            return
        inputs = []
        for entry in list(getattr(node_item, "_script_input_rows", []) or []):
            type_box = entry.get("data_type_box")
            name_box = entry.get("name_box")
            inputs.append({
                "slot": str(entry.get("slot", "")),
                "name": name_box.text().strip() if name_box is not None and isValid(name_box) else "",
                "data_type": type_box.text().strip() if type_box is not None and isValid(type_box) else "any",
            })
        outputs = []
        for entry in list(getattr(node_item, "_script_output_rows", []) or []):
            type_box = entry.get("data_type_box")
            name_box = entry.get("name_box")
            outputs.append({
                "slot": str(entry.get("slot", "")),
                "name": name_box.text().strip() if name_box is not None and isValid(name_box) else "",
                "data_type": type_box.text().strip() if type_box is not None and isValid(type_box) else "any",
            })
        node_item._script_io_spec = {"inputs": inputs, "outputs": outputs}
        if getattr(self, "_loading_workflow", False):
            params = node_item.data(20) or {}
            params["script_io_spec"] = node_item._script_io_spec
            node_item.setData(20, params)
        else:
            self._update_node_params(node_item)
        # Notify script editor tab (if open) so it can sync the signature
        on_spec_update = getattr(node_item, "_on_io_spec_updated", None)
        if callable(on_spec_update):
            try:
                on_spec_update(node_item._script_io_spec)
            except Exception:
                pass

    def _script_add_input_row(
        self,
        node_item,
        *,
        data_type: str = "any",
        var_name: Optional[str] = None,
        slot: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is None or not isValid(io_widget):
            return None
        entries = list(getattr(node_item, "_script_input_rows", []) or [])
        resolved_slot = str(slot or self._script_next_slot(entries, "script_in_"))
        existing = self._script_find_input_entry_by_slot(node_item, resolved_slot)
        if existing:
            name_box = existing.get("name_box")
            type_box = existing.get("data_type_box")
            if var_name and name_box is not None and isValid(name_box):
                name_box.setText(str(var_name))
            if type_box is not None and isValid(type_box):
                type_box.setText(_normalize_data_type(data_type))
            port = existing.get("port")
            if port:
                self._set_port_data_type(port, data_type)
            self._script_set_row_dot_style(existing, io="in", data_type=data_type)
            self._script_sync_io_spec(node_item)
            return existing

        next_idx = len(entries)
        resolved_name = str(var_name) if var_name is not None else f"in_{next_idx}"
        ui_entry = io_widget.add_input_row(_normalize_data_type(data_type), resolved_name)
        port = self._create_dynamic_port(
            node_item,
            12,
            0,
            "in",
            resolved_slot,
            channel="data",
            data_type=_normalize_data_type(data_type),
            max_connections=1,
            dot_kind="sub_dot",
            radius=4,
        )
        if port is None:
            return None
        row_entry = dict(ui_entry)
        row_entry["slot"] = resolved_slot
        row_entry["port"] = port
        entries.append(row_entry)
        node_item._script_input_rows = entries
        self._script_set_row_dot_style(row_entry, io="in", data_type=data_type)

        name_box = row_entry.get("name_box")
        if name_box is not None:
            name_box.textChanged.connect(lambda _t, n=node_item: self._script_sync_io_spec(n))
        self._script_sync_io_spec(node_item)

        sync_layout = getattr(node_item, "_sync_layout_fn", None)
        if callable(sync_layout):
            QTimer.singleShot(0, sync_layout)
        return row_entry

    def _script_add_output_row(
        self,
        node_item,
        *,
        var_name: Optional[str] = None,
        data_type: str = "any",
        slot: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is None or not isValid(io_widget):
            return None
        entries = list(getattr(node_item, "_script_output_rows", []) or [])
        resolved_slot = str(slot or self._script_next_slot(entries, "script_out_"))
        existing = self._script_find_output_entry_by_slot(node_item, resolved_slot)
        if existing:
            name_box = existing.get("name_box")
            type_box = existing.get("data_type_box")
            if name_box is not None and isValid(name_box):
                name_box.setText(str(var_name or ""))
            if type_box is not None and isValid(type_box):
                type_box.setText(_normalize_data_type(data_type))
            port = existing.get("port")
            if port:
                self._set_port_data_type(port, data_type)
            self._script_set_row_dot_style(existing, io="out", data_type=data_type)
            self._script_sync_io_spec(node_item)
            return existing

        next_idx = len(entries)
        resolved_name = str(var_name) if var_name is not None else f"out_{next_idx}"
        ui_entry = io_widget.add_output_row(resolved_name, _normalize_data_type(data_type))
        port = self._create_dynamic_port(
            node_item,
            0,
            0,
            "out",
            resolved_slot,
            channel="data",
            data_type=_normalize_data_type(data_type),
            max_connections=None,
            dot_kind="sub_dot",
            radius=4,
        )
        if port is None:
            return None
        row_entry = dict(ui_entry)
        row_entry["slot"] = resolved_slot
        row_entry["port"] = port
        entries.append(row_entry)
        node_item._script_output_rows = entries
        self._script_set_row_dot_style(row_entry, io="out", data_type=data_type)
        self._script_sync_io_spec(node_item)

        sync_layout = getattr(node_item, "_sync_layout_fn", None)
        if callable(sync_layout):
            QTimer.singleShot(0, sync_layout)
        return row_entry

    def _script_clear_rows(self, node_item):
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is None or not isValid(io_widget):
            return
        for entries in (
            list(getattr(node_item, "_script_input_rows", []) or []),
            list(getattr(node_item, "_script_output_rows", []) or []),
        ):
            for entry in entries:
                port = entry.get("port")
                if port and isValid(port):
                    conns = list(port.data(2) or [])
                    for conn in conns:
                        if conn and isValid(conn):
                            self._detach_connection(conn)
                            if conn.scene() is self:
                                self.removeItem(conn)
                    if port.scene() is self:
                        self.removeItem(port)
        io_widget.clear_input_rows()
        io_widget.clear_output_rows()
        node_item._script_input_rows = []
        node_item._script_output_rows = []

    def _script_restore_io_spec(self, node_item, spec: Dict[str, Any]):
        if not node_item or not isValid(node_item):
            return
        self._script_clear_rows(node_item)

        input_items = list((spec or {}).get("inputs", []) or [])
        output_items = list((spec or {}).get("outputs", []) or [])
        for idx, item in enumerate(input_items):
            if isinstance(item, dict):
                slot = item.get("slot")
                name = item.get("name", item.get("var_name", item.get("id", "")))
                data_type = item.get("data_type", item.get("type", "any"))
            else:
                slot = None
                name = str(item or "")
                data_type = "any"
            if not name:
                name = f"in_{idx}"
            self._script_add_input_row(node_item, data_type=str(data_type or "any"), var_name=str(name), slot=slot)

        for idx, item in enumerate(output_items):
            if isinstance(item, dict):
                slot = item.get("slot")
                name = item.get("name", item.get("var_name", item.get("id", "")))
                data_type = item.get("data_type", item.get("type", "any"))
            else:
                slot = None
                name = str(item or "")
                data_type = "any"
            if not name:
                name = f"out_{idx}"
            self._script_add_output_row(node_item, var_name=str(name), data_type=str(data_type or "any"), slot=slot)

        self._script_sync_io_spec(node_item)
        io_widget = getattr(node_item, "_script_inout", None)
        if isinstance(io_widget, ScriptInAndOut):
            io_widget.refresh_style()

    def script_update_outputs(self, node_id: int, outputs: list):
        """Called by ScriptEditor on Save & Compile.

        Replaces the Script Box node's output ports with *outputs*
        (list of ``{"name": str, "data_type": str}``).  Existing output
        connections are removed with a log warning so the user knows to
        reconnect them.
        """
        node_item = self.get_node_item(node_id)
        if not node_item or not isValid(node_item):
            return
        if node_item.data(11) != "Script Box":
            return

        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is None or not isValid(io_widget):
            return

        current_output_rows = list(getattr(node_item, "_script_output_rows", []) or [])

        # Warn about output rows that have active connections and will be removed
        new_names = {o.get("name", "") for o in outputs}
        for entry in current_output_rows:
            name_box = entry.get("name_box")
            current_name = name_box.text().strip() if name_box and isValid(name_box) else ""
            if current_name not in new_names:
                port = entry.get("port")
                if port and isValid(port) and port.data(2):
                    log_warning(
                        f"[Script] Output '{current_name}' has active connections "
                        f"and will be removed 鈥?please reconnect."
                    )

        # Remove all existing output ports and their connections
        for entry in current_output_rows:
            port = entry.get("port")
            if port and isValid(port):
                for conn in list(port.data(2) or []):
                    if conn and isValid(conn):
                        self._detach_connection(conn)
                        if conn.scene() is self:
                            self.removeItem(conn)
                if port.scene() is self:
                    self.removeItem(port)
        io_widget.clear_output_rows()
        node_item._script_output_rows = []

        # Add new output rows from compiled return
        for out in outputs:
            self._script_add_output_row(
                node_item,
                var_name=out.get("name", "result"),
                data_type=out.get("data_type", "any"),
            )

        self._script_sync_io_spec(node_item)
        sync_layout = getattr(node_item, "_sync_layout_fn", None)
        if callable(sync_layout):
            QTimer.singleShot(0, sync_layout)

    # 鈹€鈹€ Step 4: name-stable IO sync from compile result 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _script_entry_name(entry: Dict[str, Any]) -> str:
        """Return the current display name of a script IO row entry."""
        nb = entry.get("name_box")
        return nb.text().strip() if nb is not None and isValid(nb) else ""

    def _script_remove_input_entry(self, node_item, entry: Dict[str, Any]):
        """Remove one input row entry: detach connections, remove port, remove widget."""
        port = entry.get("port")
        if port and isValid(port):
            for conn in list(port.data(2) or []):
                if conn and isValid(conn):
                    self._detach_connection(conn)
                    if conn.scene() is self:
                        self.removeItem(conn)
            if port.scene() is self:
                self.removeItem(port)
        row_widget = entry.get("row")
        if row_widget is not None and isValid(row_widget):
            row_widget.setParent(None)
            row_widget.deleteLater()
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is not None and isValid(io_widget):
            try:
                io_widget.input_rows.remove(entry)
            except ValueError:
                pass
        entries = list(getattr(node_item, "_script_input_rows", []) or [])
        if entry in entries:
            entries.remove(entry)
        node_item._script_input_rows = entries
        if io_widget is not None and isValid(io_widget) and hasattr(io_widget, "notify_content_changed"):
            io_widget.notify_content_changed()

    def _script_remove_output_entry(self, node_item, entry: Dict[str, Any]):
        """Remove one output row entry: detach connections, remove port, remove widget."""
        port = entry.get("port")
        if port and isValid(port):
            for conn in list(port.data(2) or []):
                if conn and isValid(conn):
                    self._detach_connection(conn)
                    if conn.scene() is self:
                        self.removeItem(conn)
            if port.scene() is self:
                self.removeItem(port)
        row_widget = entry.get("row")
        if row_widget is not None and isValid(row_widget):
            row_widget.setParent(None)
            row_widget.deleteLater()
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is not None and isValid(io_widget):
            try:
                io_widget.output_rows.remove(entry)
            except ValueError:
                pass
        entries = list(getattr(node_item, "_script_output_rows", []) or [])
        if entry in entries:
            entries.remove(entry)
        node_item._script_output_rows = entries
        if io_widget is not None and isValid(io_widget) and hasattr(io_widget, "notify_content_changed"):
            io_widget.notify_content_changed()

    def script_update_io(self, node_id: int, compile_result: dict):
        """Update Script Box input AND output ports from a compile_result dict.

        Called by ScriptEditor on Save & Compile (Step 4+). The compile result
        is authoritative:
        - names present in compile_result are added or type-updated
        - names absent from compile_result are removed with connection cleanup

        compile_result keys (from analyze_script):
            inputs  : [{"name": str, "data_type": str, "slot": str|None}, ...]
            outputs : [{"name": str, "data_type": str}, ...]
        """
        node_item = self.get_node_item(node_id)
        if not node_item or not isValid(node_item):
            return
        if node_item.data(11) != "Script Box":
            return
        io_widget = getattr(node_item, "_script_inout", None)
        if io_widget is None or not isValid(io_widget):
            return

        new_inputs = list(compile_result.get("inputs") or [])
        new_outputs = list(compile_result.get("outputs") or [])

        # 鈹€鈹€ inputs diff 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        current_input_entries = list(getattr(node_item, "_script_input_rows", []) or [])
        current_in_by_name: Dict[str, Any] = {
            self._script_entry_name(e): e for e in current_input_entries
        }
        new_input_names = {
            str(inp.get("name") or "").strip() for inp in new_inputs if str(inp.get("name") or "").strip()
        }
        for name, entry in list(current_in_by_name.items()):
            if name in new_input_names:
                continue
            port = entry.get("port")
            connection_count = len(list(port.data(2) or [])) if port and isValid(port) else 0
            if connection_count:
                log_warning(
                    f"[GraphScene] Removing Script Box input '{name}' with "
                    f"{connection_count} existing connection(s) after compile sync."
                )
            self._script_remove_input_entry(node_item, entry)

        # Add new / update existing in declaration order
        for inp in new_inputs:
            name = str(inp.get("name") or "").strip()
            if not name:
                continue
            data_type = inp.get("data_type", "any")
            if name in current_in_by_name:
                entry = current_in_by_name[name]
                # Update data_type in-place (connection preserved)
                type_box = entry.get("data_type_box")
                if type_box is not None and isValid(type_box):
                    type_box.setText(_normalize_data_type(data_type))
                port = entry.get("port")
                if port and isValid(port):
                    self._set_port_data_type(port, data_type)
                self._script_set_row_dot_style(entry, io="in", data_type=data_type)
            else:
                self._script_add_input_row(node_item, data_type=data_type, var_name=name)

        # 鈹€鈹€ outputs diff 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        current_output_entries = list(getattr(node_item, "_script_output_rows", []) or [])
        current_out_by_name: Dict[str, Any] = {
            self._script_entry_name(e): e for e in current_output_entries
        }
        new_output_names = {
            str(out.get("name") or "").strip() for out in new_outputs if str(out.get("name") or "").strip()
        }
        for name, entry in list(current_out_by_name.items()):
            if name in new_output_names:
                continue
            port = entry.get("port")
            connection_count = len(list(port.data(2) or [])) if port and isValid(port) else 0
            if connection_count:
                log_warning(
                    f"[GraphScene] Removing Script Box output '{name}' with "
                    f"{connection_count} existing connection(s) after compile sync."
                )
            self._script_remove_output_entry(node_item, entry)

        # Add new / update existing in declaration order
        for out in new_outputs:
            name = str(out.get("name") or "").strip()
            if not name:
                continue
            data_type = out.get("data_type", "any")
            if name in current_out_by_name:
                entry = current_out_by_name[name]
                type_box = entry.get("data_type_box")
                if type_box is not None and isValid(type_box):
                    type_box.setText(_normalize_data_type(data_type))
                port = entry.get("port")
                if port and isValid(port):
                    self._set_port_data_type(port, data_type)
                self._script_set_row_dot_style(entry, io="out", data_type=data_type)
            else:
                self._script_add_output_row(node_item, var_name=name, data_type=data_type)

        # 鈹€鈹€ sync spec and layout 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self._script_sync_io_spec(node_item)
        sync_layout = getattr(node_item, "_sync_layout_fn", None)
        if callable(sync_layout):
            QTimer.singleShot(0, sync_layout)

    def _resolve_script_proxy_input_target(self, out_port, in_port):
        """Route wires targeting ScriptSetting proxy input to a real ScriptInAndOut row."""
        if not self._is_port(out_port) or not self._is_port(in_port):
            return in_port
        node_item = in_port.parentItem()
        if not node_item or not isValid(node_item):
            return in_port
        proxy_port = getattr(node_item, "_script_proxy_in_port", None)
        if proxy_port is None or in_port is not proxy_port:
            return in_port
        source_type = self._port_meta(out_port).get("data_type", "any")
        entry = self._script_add_input_row(node_item, data_type=source_type)
        return entry.get("port") if isinstance(entry, dict) else in_port

    def _sync_script_input_row_from_connection(self, node_item, in_port, out_port):
        if not node_item or not isValid(node_item):
            return
        entry = self._script_find_input_entry_by_port(node_item, in_port)
        if not entry:
            return
        source_type = self._port_meta(out_port).get("data_type", "any")
        type_box = entry.get("data_type_box")
        if type_box is not None and isValid(type_box):
            type_box.setText(_normalize_data_type(source_type))
        self._set_port_data_type(in_port, source_type)
        self._script_set_row_dot_style(entry, io="in", data_type=source_type)
        self._script_sync_io_spec(node_item)

    def _clear_script_input_row_binding(self, node_item, in_port):
        if not node_item or not isValid(node_item):
            return
        entry = self._script_find_input_entry_by_port(node_item, in_port)
        if not entry:
            return
        type_box = entry.get("data_type_box")
        if type_box is not None and isValid(type_box):
            type_box.setText("any")
        self._set_port_data_type(in_port, "any")
        self._script_set_row_dot_style(entry, io="in", data_type="any")
        self._script_sync_io_spec(node_item)

    def _reset_connection_preview(self):
        if self._preview_port and isValid(self._preview_port):
            self._apply_port_visual(self._preview_port, "normal")
        self._preview_port = None
        self._snap_target_port = None

    def _clear_port_hover(self):
        """Clear passive hover highlight for ports."""
        if self._hover_port and isValid(self._hover_port):
            if self._hover_port != self._preview_port:
                self._apply_port_visual(self._hover_port, "normal")
        self._hover_port = None
        self._cancel_sub_dot_tooltip()

    def _update_port_hover(self, scene_pos):
        """Apply passive hover highlight when cursor is near a port."""
        if self._preview_port:
            self._cancel_sub_dot_tooltip()
            return
        candidate = self._find_port_near(scene_pos, radius=self.get_port_interaction_radius())
        if not candidate:
            self._clear_port_hover()
            return
        if self._hover_port and candidate == self._hover_port:
            self._update_sub_dot_tooltip_candidate(candidate, scene_pos)
            return
        self._clear_port_hover()
        self._hover_port = candidate
        self._apply_port_visual(candidate, "hover")
        self._update_sub_dot_tooltip_candidate(candidate, scene_pos)

    def _preview_connection_target(self, pos, *, ignore_connection=None) -> QPointF:
        """Preview nearest target port and return snapped endpoint if valid."""
        self._reset_connection_preview()
        if not self._temp_start_port or not isValid(self._temp_start_port):
            return pos

        candidate = self._find_port_near(pos, radius=self.get_port_interaction_radius())
        if not candidate or candidate == self._temp_start_port:
            return pos

        start_io = self._temp_start_port.data(1)
        if start_io == candidate.data(1):
            self._preview_port = candidate
            self._apply_port_visual(candidate, "invalid")
            return pos

        out_port = self._temp_start_port if start_io == "out" else candidate
        in_port = candidate if start_io == "out" else self._temp_start_port
        if self._can_connect_ports(out_port, in_port, ignore_connection=ignore_connection, log_fail=False):
            self._preview_port = candidate
            self._snap_target_port = candidate
            self._apply_port_visual(candidate, "valid")
            return self._port_center(candidate)

        self._preview_port = candidate
        self._apply_port_visual(candidate, "invalid")
        return pos

    def _port_meta(self, port_item) -> Dict[str, Any]:
        """Return normalized port metadata for type-safe connection checks."""
        if not port_item:
            return {
                "io": "",
                "slot": "",
                "channel": "data",
                "data_type": "any",
                "max_connections": None,
                "dot_kind": "sub_dot",
            }
        meta = {}
        try:
            if not isValid(port_item):
                raise RuntimeError("invalid port")
            meta = dict(port_item.data(self._port_meta_key) or {})
        except Exception:
            meta = {}
        try:
            slot = str(port_item.data(3) or "")
        except Exception:
            slot = ""
        try:
            io = str(port_item.data(1) or "")
        except Exception:
            io = ""
        channel = meta.get("channel")
        if not channel:
            channel = "flow" if slot in _FLOW_SLOTS or slot.startswith("out_elif") else "data"
        return {
            "io": io,
            "slot": slot,
            "channel": channel,
            "data_type": _normalize_data_type(meta.get("data_type", "any")),
            "max_connections": meta.get("max_connections"),
            "dot_kind": meta.get("dot_kind", "main_dot" if channel == "flow" else "sub_dot"),
        }

    @staticmethod
    def _is_data_type_compatible(source_type: str, target_type: str) -> bool:
        # IL training port compatibility: fan-in and semantic aliases.
        _il_compat = {
            ("actuator_config", "actuator_configs"),
            ("velocity_command","command_source"),
            ("action_entity",   "action_source"),
        }
        src = _normalize_data_type(source_type)
        tgt = _normalize_data_type(target_type)
        if src == tgt:
            return True
        # Keep protocol strictly typed; do not widen via "any".
        if src == "protocol" or tgt == "protocol":
            return False
        # "any" behaves as wildcard for regular data channels.
        if src == "any" or tgt == "any":
            return True
        # IL training port fan-in / semantic alias compatibility.
        if (src, tgt) in _il_compat:
            return True
        return False

    def _can_connect_ports(self, out_port, in_port, ignore_connection=None, log_fail: bool = True) -> bool:
        """Protocol-level connection validator to prevent wrong/messy wiring."""
        if not out_port or not in_port:
            return False
        try:
            if not isValid(out_port) or not isValid(in_port):
                return False
        except Exception:
            return False

        out_meta = self._port_meta(out_port)
        in_meta = self._port_meta(in_port)

        if out_meta["io"] != "out" or in_meta["io"] != "in":
            if log_fail:
                log_warning("Invalid connection direction")
            return False
        if out_meta["channel"] != in_meta["channel"]:
            if log_fail:
                log_warning(
                    f"Port channel mismatch: {out_meta['slot']}[{out_meta['channel']}] -> "
                    f"{in_meta['slot']}[{in_meta['channel']}]"
                )
            return False
        if out_meta["dot_kind"] != in_meta["dot_kind"]:
            if log_fail:
                log_warning(
                    f"Port dot mismatch: {out_meta['slot']}[{out_meta['dot_kind']}] -> "
                    f"{in_meta['slot']}[{in_meta['dot_kind']}]"
                )
            return False
        if not self._is_data_type_compatible(out_meta["data_type"], in_meta["data_type"]):
            # Migration-compat gate (Step 5): when loading an older mission file,
            # Behavior protocol input may still use the legacy "condition" slot.
            # Allow them to restore so the user sees a protocol_invalid border
            # rather than a silent missing-connection 鈥?without widening type rules
            # for interactive use.
            is_migration_load = getattr(self, "_loading_workflow", False)
            is_legacy_behavior_cond = (
                in_meta["slot"] in ("condition", "control_pipe")
                and in_meta["data_type"] == "protocol"
                and out_meta["data_type"] in ("bool", "any")
            )
            if not (is_migration_load and is_legacy_behavior_cond):
                if log_fail:
                    log_warning(
                        f"Port type mismatch: {out_meta['slot']}[{out_meta['data_type']}] -> "
                        f"{in_meta['slot']}[{in_meta['data_type']}]"
                    )
                return False

        max_in = in_meta.get("max_connections")
        if isinstance(max_in, int) and max_in >= 0:
            try:
                current_conns = list(in_port.data(2) or [])
            except Exception:
                current_conns = []
            if ignore_connection is not None:
                current_conns = [c for c in current_conns if c is not ignore_connection]
            current = len(current_conns)
            if current >= max_in:
                if log_fail:
                    log_warning(f"Input port '{in_meta['slot']}' reached max connections ({max_in})")
                return False
        return True

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

            in_slot = self._normalize_input_slot(in_port.data(3))
            label = self._format_connection_label(out_port)
            handled_bound_widget = self._set_bound_input_widget_state(node_item, in_slot, label)

            if in_slot == "condition":
                cond_set = getattr(node_item, "_condition_set", None)
                if cond_set:
                    cond_set.set_logic_text(label)
                    sync_layout = getattr(node_item, "_sync_layout_fn", None)
                    if callable(sync_layout):
                        sync_layout()
                    self._update_node_params(node_item)
                    return
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
                    inp = elif_inputs[idx]
                    if hasattr(inp, 'set_logic_text'):
                        inp.set_logic_text(label)
                    elif not handled_bound_widget:
                        inp.setText(label)
            elif in_slot in ("for_start", "for_end", "for_step") and not handled_bound_widget:
                field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
                if field:
                    field.setText(label)
            elif in_slot == "duration" and not handled_bound_widget:
                duration_input = getattr(node_item, "_duration_input", None)
                if duration_input:
                    duration_input.setText(label)
            elif in_slot in ("left", "right"):
                left_input = getattr(node_item, "_left_input", None)
                right_input = getattr(node_item, "_right_input", None)
                if left_input or right_input:
                    if in_slot == "left" and left_input and not handled_bound_widget:
                        left_input.setText(label)
                    if in_slot == "right" and right_input and not handled_bound_widget:
                        right_input.setText(label)
                elif not handled_bound_widget:
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
            elif isinstance(in_slot, str) and in_slot.startswith("se_") and not handled_bound_widget:
                field_key = in_slot[3:]
                se_fields = getattr(node_item, "_state_event_fields", {}) or {}
                widget = se_fields.get(field_key)
                if widget is not None and hasattr(widget, "setText"):
                    widget.setText(label)
            elif isinstance(in_slot, str) and in_slot.startswith("script_in_"):
                self._sync_script_input_row_from_connection(node_item, in_port, out_port)

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
            in_slot = self._normalize_input_slot(in_port.data(3))
            handled_bound_widget = self._set_bound_input_widget_state(node_item, in_slot, "")

            if in_slot == "condition":
                cond_set = getattr(node_item, "_condition_set", None)
                if cond_set:
                    cond_set.set_logic_text("< logic >")
                    sync_layout = getattr(node_item, "_sync_layout_fn", None)
                    if callable(sync_layout):
                        sync_layout()
                    self._update_node_params(node_item)
                    return
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
                    inp = elif_inputs[idx]
                    if hasattr(inp, 'set_logic_text'):
                        inp.set_logic_text("< logic >")
                    elif not handled_bound_widget:
                        inp.setText("")
            elif in_slot in ("for_start", "for_end", "for_step") and not handled_bound_widget:
                field = getattr(node_item, f"_for_{in_slot.split('_')[1]}_input", None)
                if field:
                    field.setText("")
            elif in_slot == "duration" and not handled_bound_widget:
                duration_input = getattr(node_item, "_duration_input", None)
                if duration_input:
                    duration_input.setText("")
            elif in_slot in ("left", "right"):
                left_input = getattr(node_item, "_left_input", None)
                right_input = getattr(node_item, "_right_input", None)
                if left_input or right_input:
                    if in_slot == "left" and left_input and not handled_bound_widget:
                        left_input.setText("")
                    if in_slot == "right" and right_input and not handled_bound_widget:
                        right_input.setText("")
                elif not handled_bound_widget:
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
            elif isinstance(in_slot, str) and in_slot.startswith("se_") and not handled_bound_widget:
                field_key = in_slot[3:]
                se_fields = getattr(node_item, "_state_event_fields", {}) or {}
                widget = se_fields.get(field_key)
                if widget is not None and hasattr(widget, "setText"):
                    widget.setText("")
            elif isinstance(in_slot, str) and in_slot.startswith("script_in_"):
                self._clear_script_input_row_binding(node_item, in_port)

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

    def _normalize_input_slot(self, in_slot):
        aliases = {
            "for_start_in": "for_start",
            "for_end_in": "for_end",
            "for_step_in": "for_step",
        }
        return aliases.get(in_slot, in_slot)

    def _register_bound_input_widget(self, node_item, slot: str, widget) -> None:
        if node_item is None or not slot or widget is None:
            return
        mapping = getattr(node_item, "_bound_input_widgets", None)
        if not isinstance(mapping, dict):
            mapping = {}
            node_item._bound_input_widgets = mapping
        mapping[str(slot)] = widget

    def _set_bound_input_widget_state(self, node_item, in_slot, label: str = "") -> bool:
        if node_item is None:
            return False
        mapping = getattr(node_item, "_bound_input_widgets", None)
        if not isinstance(mapping, dict):
            return False
        widget = mapping.get(str(in_slot))
        if widget is None:
            return False
        if label:
            if hasattr(widget, "set_external_binding"):
                widget.set_external_binding(label)
            else:
                return False
        else:
            if hasattr(widget, "clear_external_binding"):
                widget.clear_external_binding()
            else:
                return False
        return True

    def _update_node_params(self, node_item):
        """Collect UI values into node metadata for later execution"""
        if not node_item or node_item.data(10) != "node":
            return

        params = node_item.data(20) or {}

        if hasattr(node_item, "_condition_set"):
            params["condition_expr"] = node_item._condition_set.logic_text()
        elif hasattr(node_item, "_condition_input"):
            params["condition_expr"] = node_item._condition_input.text()
        if hasattr(node_item, "_elif_inputs"):
            params["elif_conditions"] = [w.logic_text() if hasattr(w, 'logic_text') else w.text() for w in node_item._elif_inputs]
        cond_set_ui = getattr(node_item, "_condition_set", None)
        if cond_set_ui:
            params["ui_selection"] = cond_set_ui.logic_text()

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
        if hasattr(node_item, "_break_level_input"):
            params["break_level"] = node_item._break_level_input.text()
        if hasattr(node_item, "_script_ref"):
            params["script_ref"] = getattr(node_item, "_script_ref", "")
        if hasattr(node_item, "_script_io_spec"):
            params["script_io_spec"] = getattr(node_item, "_script_io_spec", {"inputs": [], "outputs": []})
        if hasattr(node_item, "_sensor_type_picker"):
            params["ui_selection"] = node_item._sensor_type_picker.current_text()
        if hasattr(node_item, "_sensor_output_alias_input"):
            params["sensor_output_alias"] = node_item._sensor_output_alias_input.text()
        if hasattr(node_item, "_trigger_event_input"):
            params["trigger_event"] = node_item._trigger_event_input.text()
            params["event_name"] = node_item._trigger_event_input.text()
        if hasattr(node_item, "_trigger_target_input"):
            params["trigger_target"] = node_item._trigger_target_input.text()
        if hasattr(node_item, "_trigger_timeout_input"):
            params["trigger_timeout"] = node_item._trigger_timeout_input.text()
            params["timeout"] = node_item._trigger_timeout_input.text()
        if hasattr(node_item, "_state_event_fields"):
            se_fields = getattr(node_item, "_state_event_fields", {}) or {}
            params["state_event_fields"] = {
                k: (w.text() if hasattr(w, "text") else "")
                for k, w in se_fields.items()
            }
        if hasattr(node_item, "_behavior_extension_notes"):
            params["behavior_extension_notes"] = getattr(node_item, "_behavior_extension_notes", {})
        if hasattr(node_item, "_behavior_param_inputs"):
            params["behavior_action_params"] = {
                str(k): w.text()
                for k, w in (getattr(node_item, "_behavior_param_inputs", {}) or {}).items()
                if hasattr(w, "text")
            }
        if hasattr(node_item, "_ckpt_policy_id_picker"):
            params["ckpt_policy_id"] = node_item._ckpt_policy_id_picker.combo.currentText()
        # Behavior node params
        if hasattr(node_item, "_beh_policy_id_picker"):
            params["beh_policy_id"] = node_item._beh_policy_id_picker.combo.currentText()
        if hasattr(node_item, "_beh_command_scale_slider"):
            params["beh_command_scale"] = f"{float(node_item._beh_command_scale_slider.value()) / 100.0:.2f}"
        # Training nodes (Circle 7 S1)
        if hasattr(node_item, "_ec_robot_input"):
            params["ec_robot_type"] = node_item._ec_robot_input.text()
        if hasattr(node_item, "_ec_scene_input"):
            params["ec_scene_xml"] = node_item._ec_scene_input.text()
        if hasattr(node_item, "_ec_obs_input"):
            params["ec_obs_components"] = node_item._ec_obs_input.text()
        if hasattr(node_item, "_ec_reward_input"):
            params["ec_reward_fn"] = node_item._ec_reward_input.text()
        if hasattr(node_item, "_ec_steps_input"):
            params["ec_max_steps"] = node_item._ec_steps_input.text()
        if hasattr(node_item, "_tc_algo_picker"):
            params["tc_algorithm"] = node_item._tc_algo_picker.combo.currentText()
        if hasattr(node_item, "_tc_steps_input"):
            params["tc_total_timesteps"] = node_item._tc_steps_input.text()
        if hasattr(node_item, "_tc_pid_input"):
            params["tc_policy_id_out"] = node_item._tc_pid_input.text()

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
                in_port_name = item.in_port.data(3)

                if out_id is None or in_id is None:
                    continue
                if in_port_name == "loop_break_in":
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
        data_only_entries = {"End", "Checkpoint"}
        for node_id, item in graph['nodes'].items():
            if not item or not isValid(item):
                continue
            incoming = graph['incoming'].get(node_id, {})
            # Check if has flow_in connection
            has_flow_in = 'flow_in' in incoming and len(incoming['flow_in']) > 0
            # Exclude pure data-provider nodes from entry points.
            node_name = item.data(11) if item else ""
            if not has_flow_in and node_name not in data_only_entries:
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
            cond_set = getattr(item, '_condition_set', None)
            if cond_set:
                text = cond_set.logic_text()
                if text:
                    return text
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

        # Handle Start node
        if node_name == "Start":
            model_combo = getattr(item, '_model_picker', None)
            rt = model_combo.currentText() if model_combo else self._robot_type
            code_lines.append(f"{indent_str}RobotContext.set_robot_type('{rt}')")
            flow_targets = outgoing.get('flow_out', [])
            for target_id, _ in flow_targets:
                code_lines.extend(self._generate_node_code(target_id, graph, indent, generated))
            return code_lines

        # Handle End node
        elif node_name == "End":
            return code_lines  # No code generated

        # Handle If nodes
        elif "Logic Control" in node_name:
            condition_expr = self._get_condition_from_connection(node_id, graph)
            code_lines.append(f"{indent_str}if {condition_expr}:")

            true_targets = outgoing.get('out_if', [])
            if true_targets:
                for target_id, _ in true_targets:
                    code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
            else:
                code_lines.append(f"{indent_str}    pass")

            # Elif rows are now output-only; use generated branch labels.
            elif_ports = sorted(
                [k for k in outgoing.keys() if isinstance(k, str) and k.startswith("out_elif_")],
                key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0,
            )
            for idx, elif_port in enumerate(elif_ports):
                code_lines.append(f"{indent_str}elif elif_branch_{idx}:")
                elif_targets = outgoing.get(elif_port, [])
                if elif_targets:
                    for target_id, _ in elif_targets:
                        code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
                else:
                    code_lines.append(f"{indent_str}    pass")

            code_lines.append(f"{indent_str}else:")
            false_targets = outgoing.get('out_else', [])
            if false_targets:
                for target_id, _ in false_targets:
                    code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
            else:
                code_lines.append(f"{indent_str}    pass")

        # Handle Loop nodes (while / for)
        elif node_name == "Loop":
            logic_node_item = item  # alias for clarity
            self._sync_node_parameters(item)
            type_picker = getattr(item, '_loop_type_picker', None)
            loop_mode = type_picker.combo.currentText().lower() if type_picker else 'while'
            if loop_mode == 'for':
                if logic_node:
                    fs = logic_node.parameters.get('for_start', 0)
                    fe = logic_node.parameters.get('for_end', 10)
                    fst = logic_node.parameters.get('for_step', 1)
                    code_lines.append(f"{indent_str}for _i in range({fs}, {fe}, {fst}):")
                else:
                    code_lines.append(f"{indent_str}for _i in range(0, 10, 1):")
            else:
                condition_expr = self._get_condition_from_connection(node_id, graph)
                code_lines.append(f"{indent_str}while {condition_expr}:")

            body_targets = outgoing.get('loop_body', [])
            if body_targets:
                for target_id, _ in body_targets:
                    code_lines.extend(self._generate_node_code(target_id, graph, indent + 1, generated))
            else:
                code_lines.append(f"{indent_str}    pass")

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

        elif node_name == "Break":
            code_lines.append(f"{indent_str}break")

        elif node_name == "Script Box":
            script_ref = getattr(item, "_script_ref", "") or f"script_{node_id}"
            code_lines.append(f"{indent_str}# Script Box placeholder: {script_ref}")
            flow_targets = outgoing.get('flow_out', [])
            for target_id, _ in flow_targets:
                code_lines.extend(self._generate_node_code(target_id, graph, indent, generated))

        elif node_name in ("OnEvent", "PublishEvent", "SetState", "CheckState", "HumanConfirm"):
            params = item.data(20) or {}
            se_fields = params.get("state_event_fields", {}) if isinstance(params, dict) else {}
            code_lines.append(f"{indent_str}# {node_name} shell (Behavior extension pending)")
            if isinstance(se_fields, dict) and se_fields:
                # Keep literal params in generated code for review/debug until
                # Behavior-layer handlers are implemented.
                code_lines.append(f"{indent_str}# params: {se_fields}")
            flow_targets = outgoing.get('flow_out', [])
            for target_id, _ in flow_targets:
                code_lines.extend(self._generate_node_code(target_id, graph, indent, generated))

        # Handle Action/Sensor nodes
        else:
            if logic_node:
                self._sync_node_parameters(item)
                node_code = logic_node.to_code()
                for line in node_code.strip().split('\n'):
                    code_lines.append(f"{indent_str}{line}")
            else:
                # Fallback
                cs = getattr(item, '_condition_set', None)
                if cs and "Action Execution" in node_name:
                    action = cs.logic_text()
                    robot_action = self._action_mapping.get(action, action.lower().replace(" ", "_"))
                    code_lines.append(f"{indent_str}# Action: {action}")
                    code_lines.append(f"{indent_str}robot.run_action('{robot_action}')")
                elif cs and "Sensor Input" in node_name:
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

        # Mission-canvas dirty-state hook: the regenerate_code() path is the
        # only unified flow for widget-driven parameter edits, so touching
        # here gives us parameter-change dirty tracking without instrumenting
        # every individual _on_*_change callback. Structural mutations
        # (create/delete/connect/move/...) also call regenerate_code() but
        # _touch_content has a snapshot dedup guard that prevents double
        # entries. Load paths are protected upstream by ``_loading_workflow``
        # (short-circuited above) and by ``suppress_history`` wrappers.
        self._touch_content()

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
            from src.system.compiler.lowering.canvas_to_ir import CanvasToIR
            from src.system.compiler.codegen.ir_to_code import IRToCode
            from src.system.compiler.semantic.validator import SemanticValidator

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
            if name in ("Sensor", "Trigger"):
                node_entry["external_kind"] = name.lower()
            if name == "Behavior":
                node_entry["behavior_ref"] = self._ensure_behavior_ref(item)
                cond_set = getattr(item, "_condition_set", None)
                combo = getattr(cond_set, "combo", None) if cond_set is not None else None
                combo_data = combo.currentData(Qt.ItemDataRole.UserRole) if combo is not None else {}
                if isinstance(combo_data, dict):
                    node_entry["behavior_intent_id"] = str(combo_data.get("intent_id") or "").strip()
            if name == "Script Box":
                node_entry["node_type"] = "opaque_code"
                node_entry["external_kind"] = "script"
                node_entry["script_ref"] = getattr(item, "_script_ref", "") or f"script_{node_id}"
                node_entry["script_io_spec"] = getattr(item, "_script_io_spec", {"inputs": [], "outputs": []})
            if name in ("OnEvent", "PublishEvent", "SetState", "CheckState", "HumanConfirm"):
                node_entry["node_type"] = "opaque_code"
                node_entry["external_kind"] = "state_event"
                node_entry["shell_name"] = name
                se_fields = getattr(item, "_state_event_fields", {}) or {}
                node_entry["state_event_fields"] = {
                    k: (w.text() if hasattr(w, "text") else "")
                    for k, w in se_fields.items()
                }
                node_entry["behavior_extension_notes"] = getattr(item, "_behavior_extension_notes", {})

            # Start node: serialize brand, robot_type, field_id, and
            # the v1.5 render_enabled viewer toggle. v1.4 collapsed
            # actor + field selection onto StartNode itself.
            brand_combo = getattr(item, '_brand_picker', None)
            model_combo = getattr(item, '_model_picker', None)
            field_combo = getattr(item, '_field_picker', None)
            render_combo = getattr(item, '_render_picker', None)
            if brand_combo:
                node_entry["robot_brand"] = (
                    brand_combo.currentData(Qt.ItemDataRole.UserRole) or brand_combo.currentText()
                )
            if model_combo:
                node_entry["ui_selection"] = model_combo.currentText()
                node_entry["robot_type"] = model_combo.currentText()
            if field_combo:
                node_entry["field_id"] = field_combo.currentText()
            if render_combo:
                node_entry["render_enabled"] = (
                    str(render_combo.currentText() or "On").strip().lower() == "on"
                )

            # ConditionSet text as ui_selection
            cond_set = getattr(item, '_condition_set', None)
            if cond_set:
                node_entry["ui_selection"] = cond_set.logic_text()
                node_entry["condition_expr"] = cond_set.logic_text()
            else:
                cond_input = getattr(item, '_condition_input', None)
                if cond_input:
                    node_entry["condition_expr"] = cond_input.text()

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
                node_entry["elif_conditions"] = [inp.logic_text() if hasattr(inp, 'logic_text') else inp.text() for inp in elif_inputs]

            # Wait node
            wait_mode = getattr(item, '_wait_mode_picker', None)
            if wait_mode:
                node_entry["wait_mode"] = wait_mode.current_text()
            event_input = getattr(item, '_event_input', None)
            if event_input:
                node_entry["event_name"] = event_input.text()
            timeout_input = getattr(item, '_timeout_input', None)
            if timeout_input:
                node_entry["timeout"] = timeout_input.text()
            sensor_picker = getattr(item, "_sensor_type_picker", None)
            if sensor_picker:
                node_entry["ui_selection"] = sensor_picker.current_text()
            sensor_alias_input = getattr(item, "_sensor_output_alias_input", None)
            if sensor_alias_input:
                node_entry["sensor_output_alias"] = sensor_alias_input.text()
            trigger_event_input = getattr(item, "_trigger_event_input", None)
            if trigger_event_input:
                node_entry["trigger_event"] = trigger_event_input.text()
                node_entry["event_name"] = trigger_event_input.text()
            trigger_target_input = getattr(item, "_trigger_target_input", None)
            if trigger_target_input:
                node_entry["trigger_target"] = trigger_target_input.text()
            trigger_timeout_input = getattr(item, "_trigger_timeout_input", None)
            if trigger_timeout_input:
                node_entry["trigger_timeout"] = trigger_timeout_input.text()
                node_entry["timeout"] = trigger_timeout_input.text()

            # Gate node
            gate_mode = getattr(item, '_gate_mode_picker', None)
            if gate_mode:
                node_entry["gate_mode"] = gate_mode.current_text()

            # Retry node
            att_input = getattr(item, '_attempts_input', None)
            if att_input:
                node_entry["max_attempts"] = att_input.text()
            bo_input = getattr(item, '_backoff_input', None)
            if bo_input:
                node_entry["backoff_sec"] = bo_input.text()
            strat_picker = getattr(item, '_strategy_picker', None)
            if strat_picker:
                node_entry["strategy"] = strat_picker.current_text()

            # Cancel node
            cancel_target = getattr(item, '_cancel_target_picker', None)
            if cancel_target:
                node_entry["cancel_target"] = cancel_target.current_text()
            cancel_tag = getattr(item, '_cancel_tag_input', None)
            if cancel_tag:
                node_entry["cancel_tag"] = cancel_tag.text()

            # Abort node
            abort_reason = getattr(item, '_abort_reason_input', None)
            if abort_reason:
                node_entry["abort_reason"] = abort_reason.text()
            abort_em = getattr(item, '_abort_emergency_picker', None)
            if abort_em:
                node_entry["abort_emergency"] = abort_em.current_text()
            break_level = getattr(item, '_break_level_input', None)
            if break_level:
                node_entry["break_level"] = break_level.text()

            # Control pipeline nodes
            ckpt_policy_id = getattr(item, "_ckpt_policy_id_picker", None)
            if ckpt_policy_id is not None:
                node_entry["ckpt_policy_id"] = ckpt_policy_id.combo.currentText()
            beh_pid = getattr(item, "_beh_policy_id_picker", None)
            if beh_pid is not None:
                node_entry["beh_policy_id"] = beh_pid.combo.currentText()
            beh_scale = getattr(item, "_beh_command_scale_slider", None)
            if beh_scale is not None:
                node_entry["beh_command_scale"] = round(float(beh_scale.value()) / 100.0, 2)
            # Training nodes (Circle 7 S1)
            for _attr, _key in (
                ("_ec_robot_input", "ec_robot_type"),
                ("_ec_scene_input", "ec_scene_xml"),
                ("_ec_obs_input", "ec_obs_components"),
                ("_ec_reward_input", "ec_reward_fn"),
                ("_ec_steps_input", "ec_max_steps"),
            ):
                _w = getattr(item, _attr, None)
                if _w is not None:
                    node_entry[_key] = _w.text()
            _tc_algo = getattr(item, "_tc_algo_picker", None)
            if _tc_algo is not None:
                node_entry["tc_algorithm"] = _tc_algo.combo.currentText()
            _tc_steps = getattr(item, "_tc_steps_input", None)
            if _tc_steps is not None:
                node_entry["tc_total_timesteps"] = _tc_steps.text()
            _tc_pid = getattr(item, "_tc_pid_input", None)
            if _tc_pid is not None:
                node_entry["tc_policy_id_out"] = _tc_pid.text()

            # Loop node (unified While / For)
            loop_type_picker = getattr(item, '_loop_type_picker', None)
            if loop_type_picker is not None:
                lt = loop_type_picker.combo.currentText()   # "While" or "For"
                node_entry["loop_type"] = lt
            for_start = getattr(item, '_for_start_input', None)
            if for_start is not None:
                node_entry["for_start"] = for_start.text()
            for_end = getattr(item, '_for_end_input', None)
            if for_end is not None:
                node_entry["for_end"] = for_end.text()
            for_step = getattr(item, '_for_step_input', None)
            if for_step is not None:
                node_entry["for_step"] = for_step.text()

            # Switch node
            switch_expr = getattr(item, '_switch_expr_input', None)
            if switch_expr is not None:
                node_entry["switch_expr"] = switch_expr.text()
            switch_cases = getattr(item, '_switch_case_inputs', None)
            if switch_cases is not None:
                node_entry["switch_cases"] = [inp.text() for inp in switch_cases]

            # Try node
            exc_type = getattr(item, '_exc_type_input', None)
            if exc_type is not None:
                node_entry["exception_types"] = exc_type.text()

            # Parallel node
            branch_cnt = getattr(item, '_branch_count_picker', None)
            if branch_cnt is not None:
                node_entry["branch_count"] = branch_cnt.current_text()

            # Join node
            join_cnt = getattr(item, '_join_count_picker', None)
            if join_cnt is not None:
                node_entry["join_count"] = join_cnt.current_text()

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

        # Serialize group containers
        groups = []
        for gid, g_item in list(self._groups.items()):
            if not isValid(g_item):
                continue
            gpos = g_item.pos()
            grect = g_item.rect()
            groups.append({
                "id": gid,
                "title": g_item._title,
                "node_ids": sorted(g_item._member_ids),
                "position": {"x": round(gpos.x(), 1), "y": round(gpos.y(), 1)},
                "width": round(grect.width(), 1),
                "height": round(grect.height(), 1),
            })

        return {
            "version": "1.0",
            "robot_brand": self._robot_brand,
            "robot_type": self._robot_type,
            "nodes": nodes,
            "connections": connections,
            "groups": groups,
        }

    def load_workflow(self, data: Dict[str, Any]):
        """
        Load a workflow from a serialized dict (as produced by serialize_workflow).
        Clears the current scene and recreates all nodes and connections.

        Args:
            data: Workflow dict with nodes and connections.
        """
        # Detect whether this load is the user opening a file (top-level) or
        # an internal restore called from undo()/redo() (already inside a
        # suppress_history frame). Only the top-level case should seed a
        # fresh clean baseline — undo restores must keep the existing
        # history/clean indices intact.
        seed_baseline = self._history_suppress_depth == 0

        # Suppress history pushes for the entire load path.
        with self.suppress_history():
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

        if seed_baseline:
            # Top-level load: the freshly loaded canvas becomes the new
            # clean baseline and wipes any stale undo stack.
            self.reset_history()

    def _apply_node_data(self, rect, node_data: Dict[str, Any]):
        """Apply serialized node_data parameters to an existing node item.
        Used by both load_workflow and duplicate_selected_nodes.
        Caller must handle signal blocking/unblocking around this call.
        """
        # Set condition expression from condition_expr or ui_selection
        cond_expr = node_data.get("condition_expr") or node_data.get("ui_selection")
        cond_set = getattr(rect, '_condition_set', None)
        cond_input = getattr(rect, '_condition_input', None)
        if cond_expr and cond_set:
            cond_set.set_logic_text(cond_expr)
            sync_layout = getattr(rect, "_sync_layout_fn", None)
            if callable(sync_layout):
                sync_layout()
        elif cond_expr and cond_input:
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
                    inp = elif_inputs[i]
                    if hasattr(inp, 'set_logic_text'):
                        inp.set_logic_text(cond_text)
                    else:
                        inp.setText(cond_text)

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

        # Restore Wait node
        wait_mode_val = node_data.get("wait_mode")
        wait_mode_widget = getattr(rect, '_wait_mode_picker', None)
        if wait_mode_val and wait_mode_widget:
            idx = wait_mode_widget.combo.findText(wait_mode_val)
            if idx >= 0:
                wait_mode_widget.combo.setCurrentIndex(idx)

        event_name_val = node_data.get("event_name")
        event_widget = getattr(rect, '_event_input', None)
        if event_name_val is not None and event_widget:
            event_widget.setText(str(event_name_val))
            self._record_symbol("event", str(event_name_val))

        timeout_val = node_data.get("timeout")
        timeout_widget = getattr(rect, '_timeout_input', None)
        if timeout_val is not None and timeout_widget:
            timeout_widget.setText(str(timeout_val))
        sensor_sel_val = node_data.get("ui_selection")
        sensor_picker_widget = getattr(rect, "_sensor_type_picker", None)
        if sensor_sel_val and sensor_picker_widget:
            idx = sensor_picker_widget.combo.findText(str(sensor_sel_val))
            if idx >= 0:
                sensor_picker_widget.combo.setCurrentIndex(idx)
        sensor_alias_val = node_data.get("sensor_output_alias")
        sensor_alias_widget = getattr(rect, "_sensor_output_alias_input", None)
        if sensor_alias_val is not None and sensor_alias_widget:
            sensor_alias_widget.setText(str(sensor_alias_val))
        trigger_event_val = node_data.get("trigger_event", node_data.get("event_name"))
        trigger_event_widget = getattr(rect, "_trigger_event_input", None)
        if trigger_event_val is not None and trigger_event_widget:
            trigger_event_widget.setText(str(trigger_event_val))
            self._record_symbol("event", str(trigger_event_val))
        trigger_target_val = node_data.get("trigger_target")
        trigger_target_widget = getattr(rect, "_trigger_target_input", None)
        if trigger_target_val is not None and trigger_target_widget:
            trigger_target_widget.setText(str(trigger_target_val))
            self._record_symbol("script", str(trigger_target_val))
        trigger_timeout_val = node_data.get("trigger_timeout", node_data.get("timeout"))
        trigger_timeout_widget = getattr(rect, "_trigger_timeout_input", None)
        if trigger_timeout_val is not None and trigger_timeout_widget:
            trigger_timeout_widget.setText(str(trigger_timeout_val))
        # Restore Gate node
        gate_mode_val = node_data.get("gate_mode")
        gate_mode_widget = getattr(rect, '_gate_mode_picker', None)
        if gate_mode_val and gate_mode_widget:
            idx = gate_mode_widget.combo.findText(gate_mode_val)
            if idx >= 0:
                gate_mode_widget.combo.setCurrentIndex(idx)

        # Restore Retry node
        max_att_val = node_data.get("max_attempts")
        att_widget = getattr(rect, '_attempts_input', None)
        if max_att_val is not None and att_widget:
            att_widget.setText(str(max_att_val))

        backoff_val = node_data.get("backoff_sec")
        bo_widget = getattr(rect, '_backoff_input', None)
        if backoff_val is not None and bo_widget:
            bo_widget.setText(str(backoff_val))

        strategy_val = node_data.get("strategy")
        strat_widget = getattr(rect, '_strategy_picker', None)
        if strategy_val and strat_widget:
            idx = strat_widget.combo.findText(strategy_val)
            if idx >= 0:
                strat_widget.combo.setCurrentIndex(idx)

        # Restore Cancel node
        cancel_target_val = node_data.get("cancel_target")
        cancel_target_widget = getattr(rect, '_cancel_target_picker', None)
        if cancel_target_val and cancel_target_widget:
            idx = cancel_target_widget.combo.findText(cancel_target_val)
            if idx >= 0:
                cancel_target_widget.combo.setCurrentIndex(idx)

        cancel_tag_val = node_data.get("cancel_tag")
        cancel_tag_widget = getattr(rect, '_cancel_tag_input', None)
        if cancel_tag_val is not None and cancel_tag_widget:
            cancel_tag_widget.setText(str(cancel_tag_val))

        # Restore Abort node
        abort_reason_val = node_data.get("abort_reason")
        abort_reason_widget = getattr(rect, '_abort_reason_input', None)
        if abort_reason_val is not None and abort_reason_widget:
            abort_reason_widget.setText(str(abort_reason_val))

        abort_em_val = node_data.get("abort_emergency")
        abort_em_widget = getattr(rect, '_abort_emergency_picker', None)
        if abort_em_val and abort_em_widget:
            idx = abort_em_widget.combo.findText(str(abort_em_val))
            if idx >= 0:
                abort_em_widget.combo.setCurrentIndex(idx)
        break_level_val = node_data.get("break_level")
        break_level_widget = getattr(rect, '_break_level_input', None)
        if break_level_val is not None and break_level_widget:
            break_level_widget.setText(str(break_level_val))

        # Restore Checkpoint node
        ckpt_pid_val = node_data.get("ckpt_policy_id")
        ckpt_pid_widget = getattr(rect, '_ckpt_policy_id_picker', None)
        if ckpt_pid_val is not None and ckpt_pid_widget:
            idx = ckpt_pid_widget.combo.findText(str(ckpt_pid_val))
            if idx >= 0:
                ckpt_pid_widget.combo.setCurrentIndex(idx)
            else:
                ckpt_pid_widget.combo.addItem(str(ckpt_pid_val))
                ckpt_pid_widget.combo.setCurrentText(str(ckpt_pid_val))

        # Restore training nodes (Circle 7 S1)
        for _key, _attr in (
            ("ec_robot_type", "_ec_robot_input"),
            ("ec_scene_xml", "_ec_scene_input"),
            ("ec_obs_components", "_ec_obs_input"),
            ("ec_reward_fn", "_ec_reward_input"),
            ("ec_max_steps", "_ec_steps_input"),
        ):
            _val = node_data.get(_key)
            _w = getattr(rect, _attr, None)
            if _val is not None and _w is not None:
                _w.setText(str(_val))
        _tc_algo_val = node_data.get("tc_algorithm")
        _tc_algo_w = getattr(rect, "_tc_algo_picker", None)
        if _tc_algo_val and _tc_algo_w:
            _idx = _tc_algo_w.combo.findText(str(_tc_algo_val))
            if _idx >= 0:
                _tc_algo_w.combo.setCurrentIndex(_idx)
        _tc_steps_val = node_data.get("tc_total_timesteps")
        _tc_steps_w = getattr(rect, "_tc_steps_input", None)
        if _tc_steps_val is not None and _tc_steps_w is not None:
            _tc_steps_w.setText(str(_tc_steps_val))
        _tc_pid_val = node_data.get("tc_policy_id_out")
        _tc_pid_w = getattr(rect, "_tc_pid_input", None)
        if _tc_pid_val is not None and _tc_pid_w is not None:
            _tc_pid_w.setText(str(_tc_pid_val))

        # Restore Behavior node
        beh_pid_val = node_data.get("beh_policy_id")
        beh_pid_widget = getattr(rect, '_beh_policy_id_picker', None)
        if beh_pid_val is not None and beh_pid_widget:
            idx = beh_pid_widget.combo.findText(str(beh_pid_val))
            if idx >= 0:
                beh_pid_widget.combo.setCurrentIndex(idx)
            else:
                beh_pid_widget.combo.addItem(str(beh_pid_val))
                beh_pid_widget.combo.setCurrentText(str(beh_pid_val))
        beh_scale_val = node_data.get("beh_command_scale")
        beh_scale_widget = getattr(rect, "_beh_command_scale_slider", None)
        beh_scale_label = getattr(rect, "_beh_command_scale_label", None)
        if beh_scale_val is not None and beh_scale_widget is not None:
            try:
                slider_value = int(round(float(beh_scale_val) * 100.0))
            except (TypeError, ValueError):
                slider_value = 100
            slider_value = max(0, min(200, slider_value))
            beh_scale_widget.setValue(slider_value)
            if beh_scale_label is not None:
                beh_scale_label.setText(f"{slider_value}%")
        script_ref_val = node_data.get("script_ref")
        if script_ref_val is not None:
            rect._script_ref = str(script_ref_val)
            script_ref_widget = getattr(rect, "_script_ref_input", None)
            if script_ref_widget is not None:
                script_ref_widget.setText(str(script_ref_val))
            node_id_local = int(rect.data(12) or 0)
            if callable(self._script_tab_renamer) and node_id_local > 0:
                try:
                    self._script_tab_renamer(node_id_local, str(script_ref_val))
                except Exception:
                    pass
            self._record_symbol("script", str(script_ref_val))
        script_io_spec_val = node_data.get("script_io_spec")
        if isinstance(script_io_spec_val, dict):
            self._script_restore_io_spec(rect, script_io_spec_val)
            # Keep load path aligned with create-node path: only enforce layout sync.
            if rect.data(11) == "Script Box":
                sync_layout = getattr(rect, "_sync_layout_fn", None)
                if callable(sync_layout):
                    sync_layout()
                    QTimer.singleShot(0, sync_layout)
        se_fields_val = node_data.get("state_event_fields")
        se_widgets = getattr(rect, "_state_event_fields", None)
        if isinstance(se_fields_val, dict) and isinstance(se_widgets, dict):
            for _k, _v in se_fields_val.items():
                _w = se_widgets.get(_k)
                if _w is not None and hasattr(_w, "setText"):
                    _w.setText(str(_v))
                if _k == "event_name":
                    self._record_symbol("event", str(_v))
        se_notes_val = node_data.get("behavior_extension_notes")
        if isinstance(se_notes_val, dict):
            rect._behavior_extension_notes = se_notes_val

        # Restore Loop node (unified While / For)
        loop_type_val = node_data.get("loop_type", "while")
        # Normalise legacy "for"/"while" (lowercase) and "For"/"While" (capitalised)
        lt_norm = str(loop_type_val).capitalize()   # "For" or "While"
        loop_type_picker_widget = getattr(rect, '_loop_type_picker', None)
        if loop_type_picker_widget is not None:
            idx = loop_type_picker_widget.combo.findText(lt_norm)
            if idx >= 0:
                loop_type_picker_widget.combo.setCurrentIndex(idx)
            # Apply mode immediately (show/hide rows) without waiting for signals.
            sync_fn = getattr(rect, '_sync_layout_fn', None)
            row_stack = getattr(rect, "_row_stack", None)
            is_for_mode = lt_norm == "For"
            for attr, visible in [
                ('_loop_while_rows', not is_for_mode),
                ('_loop_for_rows', is_for_mode),
            ]:
                rows = getattr(rect, attr, None)
                if rows:
                    for r in rows:
                        row_id = r.property("_row_id") if hasattr(r, "property") else None
                        if row_stack is not None and row_id and hasattr(row_stack, "set_row_collapsed"):
                            row_stack.set_row_collapsed(str(row_id), not visible)
                        else:
                            r.setVisible(visible)
                            if visible:
                                r.setMinimumHeight(24)
                                r.setMaximumHeight(24)
                            else:
                                r.setMinimumHeight(0)
                                r.setMaximumHeight(0)
            if callable(sync_fn):
                sync_fn()

        for_start_val = node_data.get("for_start")
        for_start_widget = getattr(rect, '_for_start_input', None)
        if for_start_val is not None and for_start_widget:
            for_start_widget.setText(str(for_start_val))

        for_end_val = node_data.get("for_end")
        for_end_widget = getattr(rect, '_for_end_input', None)
        if for_end_val is not None and for_end_widget:
            for_end_widget.setText(str(for_end_val))

        for_step_val = node_data.get("for_step")
        for_step_widget = getattr(rect, '_for_step_input', None)
        if for_step_val is not None and for_step_widget:
            for_step_widget.setText(str(for_step_val))

        # Restore Switch node
        switch_expr_val = node_data.get("switch_expr")
        switch_expr_widget = getattr(rect, '_switch_expr_input', None)
        if switch_expr_val is not None and switch_expr_widget:
            switch_expr_widget.setText(str(switch_expr_val))

        switch_cases_val = node_data.get("switch_cases", [])
        add_case_fn = getattr(rect, '_add_switch_case', None)
        if switch_cases_val and add_case_fn:
            for _ in switch_cases_val:
                add_case_fn()
            case_inputs = getattr(rect, '_switch_case_inputs', [])
            for _i, _case_text in enumerate(switch_cases_val):
                if _i < len(case_inputs):
                    case_inputs[_i].setText(str(_case_text))

        # Restore Try node
        exc_type_val = node_data.get("exception_types")
        exc_type_widget = getattr(rect, '_exc_type_input', None)
        if exc_type_val is not None and exc_type_widget:
            exc_type_widget.setText(str(exc_type_val))

        # Restore Parallel node
        branch_cnt_val = node_data.get("branch_count")
        branch_cnt_widget = getattr(rect, '_branch_count_picker', None)
        if branch_cnt_val and branch_cnt_widget:
            _idx = branch_cnt_widget.combo.findText(str(branch_cnt_val))
            if _idx >= 0:
                branch_cnt_widget.combo.setCurrentIndex(_idx)

        # Restore Join node
        join_cnt_val = node_data.get("join_count") or node_data.get("branch_count")
        join_cnt_widget = getattr(rect, '_join_count_picker', None)
        if join_cnt_val and join_cnt_widget:
            _idx = join_cnt_widget.combo.findText(str(join_cnt_val))
            if _idx >= 0:
                join_cnt_widget.combo.setCurrentIndex(_idx)

        # Restore Start node brand and robot type
        brand_val = node_data.get("robot_brand")
        brand_combo = getattr(rect, '_brand_picker', None)
        if brand_val and brand_combo:
            brand_id = normalize_brand_id(str(brand_val))
            idx = brand_combo.findData(brand_id, Qt.ItemDataRole.UserRole)
            if idx < 0:
                idx = brand_combo.findText(str(brand_val))
            if idx >= 0:
                brand_combo.setCurrentIndex(idx)
            self._robot_brand = brand_id or str(brand_val)

        robot_type_val = node_data.get("robot_type") or node_data.get("ui_selection")
        model_combo = getattr(rect, '_model_picker', None)
        if robot_type_val and model_combo:
            idx = model_combo.findText(robot_type_val)
            if idx >= 0:
                model_combo.setCurrentIndex(idx)
            self._robot_type = robot_type_val

        # v1.4: restore field_id selection. Tolerates the legacy
        # ``default_field_id`` key for canvases saved pre-v1.4.
        field_val = node_data.get("field_id") or node_data.get("default_field_id")
        field_combo = getattr(rect, '_field_picker', None)
        if field_val and field_combo:
            idx = field_combo.findText(str(field_val))
            if idx >= 0:
                field_combo.setCurrentIndex(idx)
            self._start_field_id = str(field_val)

        # v1.5: restore render_enabled toggle. Missing key → True
        # (matches the v1.5 default for legacy canvases).
        render_combo = getattr(rect, '_render_picker', None)
        if render_combo is not None:
            render_val = node_data.get("render_enabled", True)
            if isinstance(render_val, str):
                render_val = render_val.strip().lower() in ("true", "1", "yes", "on")
            render_combo.setCurrentText("On" if bool(render_val) else "Off")
            self._start_render_enabled = bool(render_val)

    def _load_workflow_impl(self, data: Dict[str, Any]):
        """Internal implementation of workflow loading (signals suppressed)."""
        # Clear existing nodes
        self.clear_all_nodes()

        # Set robot type
        if "robot_brand" in data:
            self._robot_brand = normalize_brand_id(data["robot_brand"])
        if "robot_type" in data:
            self._robot_type = data["robot_type"]

        # Map old node IDs to new graphics items for connection wiring
        id_to_item: Dict[int, Any] = {}

        for node_data in data.get("nodes", []):
            old_id = node_data["id"]
            node_type = str(node_data.get("node_type", "") or "").lower()
            name = node_data.get("display_name", "")
            external_kind = str(node_data.get("external_kind", "") or "").lower()
            pos_data = node_data.get("position", {})
            x = pos_data.get("x", 100)
            y = pos_data.get("y", 100)
            w = node_data.get("width", 180)
            h = node_data.get("height", 110)
            features = node_data.get("features", [])
            ui_selection = node_data.get("ui_selection")

            # Prefer explicit node_type when restoring from compiler output.
            # This avoids ambiguity between logic nodes that share display names.
            if node_type == "start":
                name = "Start"
            elif node_type == "end":
                name = "End"
            elif node_type == "action_execution" and external_kind == "behavior":
                name = "Behavior"
            elif node_type == "sensor_input" and external_kind == "sensor":
                name = "Sensor"
            elif node_type == "wait" and external_kind == "trigger":
                name = "Trigger"
            elif node_type == "if":
                name = "Logic Control"
            elif node_type == "while_loop":
                name = "Loop"
            elif node_type == "switch":
                name = "Switch"
            elif node_type == "try_catch":
                name = "Try"
            elif node_type == "parallel":
                name = "Parallel Branch"
            elif node_type == "join":
                name = "Join"
            elif node_type == "opaque_code":
                if external_kind == "state_event":
                    shell_name = str(node_data.get("shell_name", "") or "")
                    if shell_name in ("OnEvent", "PublishEvent", "SetState", "CheckState", "HumanConfirm"):
                        name = shell_name
                    else:
                        name = "OnEvent"
                else:
                    name = "Script Box"
            elif node_type == "wait":
                name = "Wait"
            elif node_type == "gate":
                name = "Gate"
            elif node_type == "timeout":
                name = "Timeout"
            elif node_type == "retry":
                name = "Retry"
            elif node_type == "cancel":
                name = "Cancel"
            elif node_type == "abort":
                name = "Abort"
            elif node_type == "break":
                name = "Break"
            elif node_type == "checkpoint":
                name = "Checkpoint"
            elif node_type == "policy":
                name = "Policy"
            elif node_type == "conductor":
                name = "Conductor"

            # create_node internally does setPos(scene_pos - QPointF(50, 40))
            # using the fixed initial node size (100×80). Pass scene_pos such
            # that the final setPos lands exactly on the saved (x, y) position,
            # avoiding coordinate drift caused by using the saved (w, h) offsets.
            pos = QPointF(x + 50, y + 40)
            rect = self.create_node(name, pos, features)

            # Block signals on all child widgets to prevent triggering
            # regenerate_code during batch restoration
            self._set_node_widgets_silent(rect, True)

            try:
                self._apply_node_data(rect, node_data)
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

        # Restore group containers (backward compat: missing key 鈫?empty list)
        for group_data in data.get("groups", []):
            try:
                gid = int(group_data.get("id", 0))
                if gid <= 0:
                    continue
                title = str(group_data.get("title", "") or "")
                member_ids = set(group_data.get("node_ids", []))
                pos_data = group_data.get("position", {})
                gx = float(pos_data.get("x", 0))
                gy = float(pos_data.get("y", 0))
                gw = float(group_data.get("width", 120))
                gh = float(group_data.get("height", 80))

                g_item = GroupContainerItem(gid, title)
                self.addItem(g_item)

                # Restore position without triggering member moves
                # (_prev_pos is None on fresh item 鈫?first setPos is a no-op for members)
                g_item.setPos(QPointF(gx, gy))
                g_item.setRect(0, 0, gw, gh)

                # Re-attach group membership to member node items
                for new_node_id, new_item in id_to_item.items():
                    if new_node_id in member_ids:
                        new_item._group_id = gid
                        g_item._member_ids.add(new_node_id)

                self._groups[gid] = g_item
                # Keep _group_seq at least as high as any restored id
                if gid > self._group_seq:
                    self._group_seq = gid
            except Exception as _ge:
                log_debug(f"Error restoring group: {_ge}")

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
        """Clear all nodes, connections, and group containers from the scene."""
        # Remove all items
        items_to_remove = []
        for item in self.items():
            if not isValid(item):
                continue
            if (isinstance(item, ConnectionItem)
                    or item.data(10) == "node"
                    or item.data(10) == "group"):
                items_to_remove.append(item)

        for item in items_to_remove:
            if isValid(item) and item.scene() is not None:
                self.removeItem(item)

        self._logic_nodes.clear()
        self._groups.clear()
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
            cs = getattr(item, '_condition_set', None)
            if cs:
                node_data['ui_selection'] = cs.logic_text()

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

            # Ensure UI-driven parameters (combos, text fields) are flushed
            # to the logic node before we snapshot parameters below.
            self._sync_node_parameters(item)

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
            cs = getattr(item, '_condition_set', None)
            if cs:
                node_data['ui_selection'] = cs.logic_text()

            # Add condition input for Logic Control nodes
            cond_set = getattr(item, '_condition_set', None)
            cond_input = getattr(item, '_condition_input', None)
            if cond_set:
                node_data['condition_expr'] = cond_set.logic_text()
            elif cond_input:
                node_data['condition_expr'] = cond_input.text()

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
