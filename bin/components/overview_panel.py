#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OverviewPanel — floating console bar + expandable ControlPanel overlay on the canvas.

Collapsed form:  a draggable floating bar with mini page switcher.
Expanded form:   a resizable floating panel with three-zone layout.

**Usage**::

    self._overview_bar = OverviewPanel(self._view)
    self._overview_bar.show()
"""
from __future__ import annotations

from enum import Flag, auto
from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from bin.pages.layout.misc import PageSwitcher
from src.system.core.theme_manager import get_color


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BTN_SIZE = 30
_ICON_SIZE = QSize(20, 20)
_EXTRA_WIDTH = 10
_PANEL_DEFAULT_W = 820
_PANEL_DEFAULT_H = 300
_PANEL_MIN_W = 400
_PANEL_MIN_H = 200
_PANEL_RADIUS = 20
_PANEL_BOTTOM_MARGIN = 60
_AGENT_BAR_H = 32
_EDGE_MARGIN = 8  # px from edge to trigger resize cursor
_ZONE_SIDE_W = 200  # fixed width for left_zone / right_zone


# ---------------------------------------------------------------------------
# Edge flags for resize detection
# ---------------------------------------------------------------------------

class _Edge(Flag):
    NONE = 0
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


def _detect_edges(pos: QPoint, rect: QRect, margin: int = _EDGE_MARGIN) -> _Edge:
    """Return which edges the cursor is near."""
    edges = _Edge.NONE
    if pos.x() <= margin:
        edges |= _Edge.LEFT
    if pos.x() >= rect.width() - margin:
        edges |= _Edge.RIGHT
    if pos.y() <= margin:
        edges |= _Edge.TOP
    if pos.y() >= rect.height() - margin:
        edges |= _Edge.BOTTOM
    return edges


def _cursor_for_edges(edges: _Edge) -> Qt.CursorShape:
    """Return the appropriate resize cursor for the given edge combination."""
    if edges == (_Edge.LEFT | _Edge.TOP) or edges == (_Edge.RIGHT | _Edge.BOTTOM):
        return Qt.CursorShape.SizeFDiagCursor
    if edges == (_Edge.RIGHT | _Edge.TOP) or edges == (_Edge.LEFT | _Edge.BOTTOM):
        return Qt.CursorShape.SizeBDiagCursor
    if _Edge.LEFT in edges or _Edge.RIGHT in edges:
        return Qt.CursorShape.SizeHorCursor
    if _Edge.TOP in edges or _Edge.BOTTOM in edges:
        return Qt.CursorShape.SizeVerCursor
    return Qt.CursorShape.ArrowCursor


# ---------------------------------------------------------------------------
# Agent Input Bar (placeholder v1)
# ---------------------------------------------------------------------------

class _AgentInputBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_AGENT_BAR_H)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(12, 2, 12, 6)
        hl.setSpacing(4)

        icon = QLabel("\U0001f916")
        icon.setFixedWidth(20)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setEnabled(False)
        hl.addWidget(icon, 0)

        inp = QLineEdit()
        inp.setPlaceholderText("Agent assistant coming soon...")
        inp.setEnabled(False)
        inp.setObjectName("agentInput")
        hl.addWidget(inp, 1)

        send = QPushButton("Send")
        send.setFixedWidth(48)
        send.setEnabled(False)
        send.setObjectName("agentSend")
        hl.addWidget(send, 0)


# ---------------------------------------------------------------------------
# Mini Page Switcher for the collapsed bar
# ---------------------------------------------------------------------------

class _MiniPageSwitcher(QWidget):
    """Compact M / T toggle for the collapsed floating bar."""

    page_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_page = "mission"
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self._m_btn = QPushButton("M")
        self._m_btn.setObjectName("miniSwitcherM")
        self._m_btn.setCheckable(True)
        self._m_btn.setChecked(True)
        self._m_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._m_btn.setFixedSize(26, 22)
        self._group.addButton(self._m_btn)
        hl.addWidget(self._m_btn)

        self._t_btn = QPushButton("T")
        self._t_btn.setObjectName("miniSwitcherT")
        self._t_btn.setCheckable(True)
        self._t_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._t_btn.setFixedSize(26, 22)
        self._group.addButton(self._t_btn)
        hl.addWidget(self._t_btn)

        self._m_btn.clicked.connect(lambda checked: self._on_clicked("mission", checked))
        self._t_btn.clicked.connect(lambda checked: self._on_clicked("training", checked))

    def _on_clicked(self, page: str, checked: bool) -> None:
        if not checked or page == self._current_page:
            return
        self._current_page = page
        self.page_selected.emit(page)

    def set_current_page(self, page: str) -> None:
        page = "training" if page == "training" else "mission"
        self._current_page = page
        if page == "training":
            self._t_btn.setChecked(True)
        else:
            self._m_btn.setChecked(True)

    def apply_theme(self) -> None:
        base_bg = get_color("tab_bg", "#334155")
        base_text = get_color("text_secondary", "#9ca3af")
        checked_bg = get_color("tab_bg_checked", "#4a6741")
        checked_text = get_color("tab_text_checked", "#ffffff")
        hover_bg = get_color("tab_bg_hover", "#3d4f63")
        self.setStyleSheet(f"""
            QPushButton#miniSwitcherM, QPushButton#miniSwitcherT {{
                background: {base_bg};
                color: {base_text};
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton#miniSwitcherM:hover, QPushButton#miniSwitcherT:hover {{
                background: {hover_bg};
            }}
            QPushButton#miniSwitcherM:checked, QPushButton#miniSwitcherT:checked {{
                background: {checked_bg};
                color: {checked_text};
            }}
        """)


# ---------------------------------------------------------------------------
# ControlPanel — independent resizable floating panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """Resizable floating control panel with three-zone layout.

    Layout::

        +------------------------------------------+
        |  Title bar  [title] ... [dot] [close]    |
        +----------+------------------+------------+
        |          | [PageSwitcher]   |            |
        | left_zone| function_zone    | right_zone |
        |  (fixed) | (stretch)        |  (fixed)   |
        |          | + content slot   |            |
        +----------+------------------+------------+
        |  Agent Input Bar                         |
        +------------------------------------------+
    """

    collapse_requested = Signal()
    page_selected = Signal(str)
    robot_changed = Signal(str, str)  # (brand_id, model_id)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(_PANEL_MIN_W, _PANEL_MIN_H)
        self.resize(_PANEL_DEFAULT_W, _PANEL_DEFAULT_H)

        # Page mode — controls border accent color
        self._page_mode = "mission"

        # Resize state
        self._resize_edges = _Edge.NONE
        self._resize_dragging = False
        self._resize_origin = QPoint()
        self._resize_geom = QRect()
        self.setMouseTracking(True)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        # Main vertical layout
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # ── Title bar ────────────────────────────────────────────
        title_bar = QWidget(self)
        title_bar.setFixedHeight(32)
        title_bar.setObjectName("controlPanelTitle")
        thl = QHBoxLayout(title_bar)
        thl.setContentsMargins(16, 0, 8, 0)
        thl.setSpacing(6)

        self._title = QLabel("Control Panel")
        self._title.setStyleSheet("font-weight: bold; font-size: 12px; color: #e0e0e0;")
        thl.addWidget(self._title, 0)
        thl.addStretch(1)

        dot = QWidget(title_bar)
        dot.setFixedSize(8, 8)
        dot.setStyleSheet("background: #555; border-radius: 4px;")
        thl.addWidget(dot, 0)

        close_btn = QPushButton("\u2715", title_bar)
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFlat(True)
        close_btn.setObjectName("controlPanelClose")
        close_btn.clicked.connect(self.collapse_requested.emit)
        thl.addWidget(close_btn, 0)

        vl.addWidget(title_bar, 0)

        # ── Separator ────────────────────────────────────────────
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3d3d3d;")
        vl.addWidget(sep, 0)

        # ── Three-zone content area ──────────────────────────────
        content_row = QWidget(self)
        content_hl = QHBoxLayout(content_row)
        content_hl.setContentsMargins(8, 4, 8, 4)
        content_hl.setSpacing(6)

        # Left zone — hosts RobotPanel
        self._left_zone = QWidget(content_row)
        self._left_zone.setObjectName("controlPanelLeftZone")
        self._left_zone.setFixedWidth(_ZONE_SIDE_W)
        self._left_zone.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_vl = QVBoxLayout(self._left_zone)
        left_vl.setContentsMargins(0, 0, 0, 0)
        left_vl.setSpacing(0)

        from bin.components.robot_panel import RobotPanel
        self._robot_panel = RobotPanel(self._left_zone)
        self._robot_panel.robot_changed.connect(self.robot_changed.emit)
        left_vl.addWidget(self._robot_panel, 1)

        content_hl.addWidget(self._left_zone, 0)

        # Left separator
        sep_l = QFrame(content_row)
        sep_l.setFrameShape(QFrame.Shape.VLine)
        sep_l.setStyleSheet("color: #3d3d3d;")
        content_hl.addWidget(sep_l, 0)

        # Function zone (center, stretch)
        self._function_zone = QWidget(content_row)
        self._function_zone.setObjectName("controlPanelFuncZone")
        func_vl = QVBoxLayout(self._function_zone)
        func_vl.setContentsMargins(0, 4, 0, 0)
        func_vl.setSpacing(4)

        # PageSwitcher at top of function zone
        self._page_switcher = PageSwitcher(self._function_zone)
        self._page_switcher.page_selected.connect(self.page_selected.emit)
        func_vl.addWidget(self._page_switcher, 0, Qt.AlignmentFlag.AlignHCenter)

        # Content slot below PageSwitcher
        self._content: Optional[QWidget] = None
        self._content_slot = QWidget(self._function_zone)
        self._content_layout = QVBoxLayout(self._content_slot)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        func_vl.addWidget(self._content_slot, 1)

        content_hl.addWidget(self._function_zone, 1)

        # Right separator
        sep_r = QFrame(content_row)
        sep_r.setFrameShape(QFrame.Shape.VLine)
        sep_r.setStyleSheet("color: #3d3d3d;")
        content_hl.addWidget(sep_r, 0)

        # Right zone
        self._right_zone = QWidget(content_row)
        self._right_zone.setObjectName("controlPanelRightZone")
        self._right_zone.setFixedWidth(_ZONE_SIDE_W)
        self._right_zone.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        QVBoxLayout(self._right_zone).setContentsMargins(0, 0, 0, 0)
        content_hl.addWidget(self._right_zone, 0)

        vl.addWidget(content_row, 1)

        # ── Agent bar ────────────────────────────────────────────
        self._agent_bar = _AgentInputBar(self)
        vl.addWidget(self._agent_bar, 0)

        # ── Side dock boxes (siblings, hidden by default) ────────
        # Created lazily in showEvent once self.parent() is settled.
        self._left_dock: Optional[QWidget] = None
        self._right_dock: Optional[QWidget] = None
        self._docks_created = False

        # Apply initial theme
        self.apply_theme()

    def _ensure_docks(self) -> None:
        """Create left/right dock boxes as siblings (children of same parent)."""
        if self._docks_created:
            return
        p = self.parent()
        if p is None:
            return
        self._docks_created = True

        for attr, name in [("_left_dock", "controlPanelLeftDock"),
                           ("_right_dock", "controlPanelRightDock")]:
            dock = QWidget(p)
            dock.setObjectName(name)
            dock.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            dock.hide()
            vl = QVBoxLayout(dock)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(0)
            setattr(self, attr, dock)

    def _sync_dock_positions(self) -> None:
        """Align dock boxes to the edges of the ControlPanel."""
        if self._left_dock is not None and self._left_dock.isVisible():
            dw = self._left_dock.width() or 260
            dh = self.height()
            self._left_dock.setFixedHeight(dh)
            self._left_dock.move(self.x() - dw - 4, self.y())
            self._left_dock.raise_()
        if self._right_dock is not None and self._right_dock.isVisible():
            dh = self.height()
            self._right_dock.setFixedHeight(dh)
            self._right_dock.move(self.x() + self.width() + 4, self.y())
            self._right_dock.raise_()

    def show_left_dock(self, widget: QWidget) -> None:
        """Show *widget* in the left dock box, positioned to the panel's left."""
        self._ensure_docks()
        if self._left_dock is None:
            return
        layout = self._left_dock.layout()
        # Clear previous content
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        widget.setParent(self._left_dock)
        layout.addWidget(widget)
        # Size the dock to the widget's preferred width
        w = widget.sizeHint().width() or 260
        self._left_dock.setFixedWidth(w)
        self._left_dock.show()
        self._sync_dock_positions()

    def hide_left_dock(self) -> None:
        if self._left_dock is not None:
            self._left_dock.hide()

    def show_right_dock(self, widget: QWidget) -> None:
        """Show *widget* in the right dock box."""
        self._ensure_docks()
        if self._right_dock is None:
            return
        layout = self._right_dock.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        widget.setParent(self._right_dock)
        layout.addWidget(widget)
        w = widget.sizeHint().width() or 260
        self._right_dock.setFixedWidth(w)
        self._right_dock.show()
        self._sync_dock_positions()

    def hide_right_dock(self) -> None:
        if self._right_dock is not None:
            self._right_dock.hide()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._sync_dock_positions()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_dock_positions()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_docks()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.hide_left_dock()
        self.hide_right_dock()

    # ------------------------------------------------------------------
    # Public zone accessors
    # ------------------------------------------------------------------

    @property
    def left_zone(self) -> QWidget:
        return self._left_zone

    @property
    def right_zone(self) -> QWidget:
        return self._right_zone

    @property
    def function_zone(self) -> QWidget:
        return self._function_zone

    @property
    def page_switcher(self) -> PageSwitcher:
        return self._page_switcher

    @property
    def robot_panel(self):
        """Return the embedded RobotPanel instance."""
        return self._robot_panel

    # ------------------------------------------------------------------
    # Content management
    # ------------------------------------------------------------------

    def set_content(self, widget: QWidget) -> None:
        if self._content is not None:
            self._content_layout.removeWidget(self._content)
            self._content.setParent(None)
        self._content = widget
        self._content_layout.addWidget(widget, 1)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    # ------------------------------------------------------------------
    # Positioning
    # ------------------------------------------------------------------

    def place_center_bottom(self) -> None:
        parent = self.parent()
        if parent is None:
            return
        pw, ph = parent.width(), parent.height()
        x = max(0, (pw - self.width()) // 2)
        y = max(0, ph - self.height() - _PANEL_BOTTOM_MARGIN)
        self.move(x, y)

    # ------------------------------------------------------------------
    # Edge-drag resize
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edges = _detect_edges(event.position().toPoint(), self.rect())
            if edges != _Edge.NONE:
                self._resize_dragging = True
                self._resize_edges = edges
                self._resize_origin = event.globalPosition().toPoint()
                self._resize_geom = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._resize_dragging:
            delta = event.globalPosition().toPoint() - self._resize_origin
            new_geom = QRect(self._resize_geom)
            if _Edge.LEFT in self._resize_edges:
                new_geom.setLeft(new_geom.left() + delta.x())
            if _Edge.RIGHT in self._resize_edges:
                new_geom.setRight(new_geom.right() + delta.x())
            if _Edge.TOP in self._resize_edges:
                new_geom.setTop(new_geom.top() + delta.y())
            if _Edge.BOTTOM in self._resize_edges:
                new_geom.setBottom(new_geom.bottom() + delta.y())
            # Enforce min size
            if new_geom.width() < _PANEL_MIN_W:
                if _Edge.LEFT in self._resize_edges:
                    new_geom.setLeft(new_geom.right() - _PANEL_MIN_W)
                else:
                    new_geom.setRight(new_geom.left() + _PANEL_MIN_W)
            if new_geom.height() < _PANEL_MIN_H:
                if _Edge.TOP in self._resize_edges:
                    new_geom.setTop(new_geom.bottom() - _PANEL_MIN_H)
                else:
                    new_geom.setBottom(new_geom.top() + _PANEL_MIN_H)
            self.setGeometry(new_geom)
            event.accept()
            return
        # Hover cursor
        edges = _detect_edges(event.position().toPoint(), self.rect())
        if edges != _Edge.NONE:
            self.setCursor(_cursor_for_edges(edges))
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._resize_dragging:
            self._resize_dragging = False
            self._resize_edges = _Edge.NONE
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def set_page_mode(self, mode: str) -> None:
        """Switch the panel accent (border color) between mission / training."""
        mode = "training" if str(mode or "").strip().lower() == "training" else "mission"
        if mode != self._page_mode:
            self._page_mode = mode
            self.apply_theme()

    def apply_theme(self) -> None:
        bg = get_color("training_float_bar_bg", "#1e2d3d")
        # Border color follows the active PageSwitcher button accent
        if self._page_mode == "training":
            border = get_color("tab_bg_training", "#F6D393")
        else:
            border = get_color("tab_bg_checked", "#FF7844")
        btn_text = get_color("training_float_btn_text", "#d1d5db")
        btn_hover = get_color("training_float_btn_hover", "#374151")

        self.setStyleSheet(f"""
            #controlPanel {{
                background: {bg};
                border: 1px solid {border};
                border-radius: {_PANEL_RADIUS}px;
            }}
            #controlPanel QLabel {{
                background: transparent;
                border: none;
            }}
            #controlPanelTitle {{
                background: transparent;
            }}
            #controlPanelClose {{
                color: #888; font-size: 14px; border: none;
            }}
            #controlPanelClose:hover {{ color: #fff; }}
            #agentInput {{
                background: rgba(0,0,0,0.3);
                color: #666;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 2px 6px;
            }}
            #agentSend {{
                background: rgba(0,0,0,0.3);
                color: #666;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 2px 8px;
            }}
        """)
        self._page_switcher.apply_theme()
        self._robot_panel.apply_theme()

        # Dock boxes share the same bg/border look
        dock_ss = f"""
            background: {bg};
            border: 1px solid {border};
            border-radius: {_PANEL_RADIUS}px;
        """
        for dock in (self._left_dock, self._right_dock):
            if dock is not None:
                dock.setStyleSheet(dock_ss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        g = self.geometry()
        return {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}

    def restore_state(self, state: dict) -> None:
        w = state.get("w", _PANEL_DEFAULT_W)
        h = state.get("h", _PANEL_DEFAULT_H)
        self.resize(max(w, _PANEL_MIN_W), max(h, _PANEL_MIN_H))


# ---------------------------------------------------------------------------
# OverviewPanel — floating bar (collapsed) + ControlPanel (expanded)
# ---------------------------------------------------------------------------

class OverviewPanel(QWidget):
    """Floating console bar on the canvas, expandable into a ControlPanel.

    Collapsed: ``[drag] [M|T] [Control Panel]``  -- draggable, bottom-center default.
    Expanded:  resizable ControlPanel with three-zone layout.
    """

    expanded_changed = Signal(bool)
    page_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("overviewFloatBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._dragging = False
        self._drag_offset = QPoint()
        self._expanded = False
        self._position_initialized = False
        self._content: Optional[QWidget] = None
        self._title_text = "Control Panel"

        # ── Expanded panel (child of same parent, hidden initially) ───
        self._panel = ControlPanel(parent)
        self._panel.collapse_requested.connect(self.collapse)
        self._panel.page_selected.connect(self.page_selected.emit)
        self._panel.hide()

        # ── Collapsed bar UI ─────────────────────────────────────────
        self._build_bar_ui()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._apply_measured_size()
        self.raise_()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Bar UI (collapsed form)
    # ------------------------------------------------------------------

    def _build_bar_ui(self):
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(10, 6, 10, 6)
        hbox.setSpacing(6)

        self._drag_btn = QPushButton()
        self._drag_btn.setObjectName("overviewDragHandle")
        self._drag_btn.setFlat(True)
        self._drag_btn.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_btn.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self._drag_btn.setToolTip("Drag to reposition")
        self._drag_btn.installEventFilter(self)
        hbox.addWidget(self._drag_btn)

        hbox.addSpacing(2)

        # Mini page switcher on collapsed bar
        self._mini_switcher = _MiniPageSwitcher(self)
        self._mini_switcher.page_selected.connect(self._on_mini_page_selected)
        hbox.addWidget(self._mini_switcher)

        hbox.addSpacing(2)

        self._open_btn = QPushButton("Control Panel")
        self._open_btn.setObjectName("overviewOpenBtn")
        self._open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_btn.clicked.connect(self.expand)
        hbox.addWidget(self._open_btn)

        self._apply_icons()

    def _on_mini_page_selected(self, page: str) -> None:
        """Sync mini switcher → full PageSwitcher and emit."""
        self._panel.page_switcher.set_current_page(page)
        self.page_selected.emit(page)

    def _apply_icons(self):
        from src.system.core.theme_manager import get_icon
        icon = get_icon("move")
        if not icon.isNull():
            self._drag_btn.setIcon(icon)
            self._drag_btn.setIconSize(_ICON_SIZE)
            self._drag_btn.setText("")
        else:
            self._drag_btn.setText(":::")
        self._apply_measured_size()

    def _apply_measured_size(self):
        hint = self.sizeHint()
        self.setFixedSize(hint.width() + _EXTRA_WIDTH, hint.height())

    # ------------------------------------------------------------------
    # Content management
    # ------------------------------------------------------------------

    def set_content(self, widget: QWidget, title: str = "Control Panel") -> None:
        self._content = widget
        self._title_text = title
        self._panel.set_content(widget)
        self._panel.set_title(title)

    # ------------------------------------------------------------------
    # Page switcher access
    # ------------------------------------------------------------------

    @property
    def page_switcher(self) -> PageSwitcher:
        return self._panel.page_switcher

    @property
    def control_panel(self) -> ControlPanel:
        return self._panel

    def set_current_page(self, page: str) -> None:
        """Sync both mini switcher and full PageSwitcher."""
        self._panel.page_switcher.set_current_page(page)
        self._mini_switcher.set_current_page(page)

    # ------------------------------------------------------------------
    # Expand / Collapse
    # ------------------------------------------------------------------

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def expand(self) -> None:
        if self._expanded:
            return
        self._expanded = True
        self.hide()
        self._panel.place_center_bottom()
        self._panel.show()
        self._panel.raise_()
        self.expanded_changed.emit(True)

    def collapse(self) -> None:
        if not self._expanded:
            return
        self._expanded = False
        self._panel.hide()
        self.show()
        self.raise_()
        self.expanded_changed.emit(False)

    def toggle(self) -> None:
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    # ------------------------------------------------------------------
    # Drag (identical to TrainingFloatControlBar)
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if watched is self._drag_btn:
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                self._dragging = True
                self._drag_offset = event.position().toPoint()
                self._drag_btn.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if event.type() == QEvent.Type.MouseMove and self._dragging:
                parent = self.parent()
                if parent is not None:
                    gpos = event.globalPosition().toPoint()
                    local = parent.mapFromGlobal(gpos)
                    new_pos = local - self._drag_offset
                    self.move(self._clamp(new_pos))
                return True
            if (event.type() == QEvent.Type.MouseButtonRelease
                    and self._dragging):
                self._dragging = False
                self._drag_btn.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)

    def _clamp(self, pos: QPoint) -> QPoint:
        parent = self.parent()
        if parent is None:
            return pos
        max_x = max(0, parent.width() - self.width())
        max_y = max(0, parent.height() - self.height())
        return QPoint(max(0, min(pos.x(), max_x)),
                      max(0, min(pos.y(), max_y)))

    # ------------------------------------------------------------------
    # Positioning
    # ------------------------------------------------------------------

    def place_default(self):
        """Position bottom-center of parent, 50 px above bottom edge."""
        parent = self.parent()
        if parent is None:
            return
        self._apply_measured_size()
        pw, ph = parent.width(), parent.height()
        if pw == 0:
            return
        x = max(0, (pw - self.width()) // 2)
        y = max(0, ph - self.height() - 50)
        self.move(x, y)
        self._position_initialized = True

    def reposition(self) -> None:
        """Called on parent resize — keep bar/panel in bounds."""
        if self._expanded:
            self._panel.place_center_bottom()
        else:
            if not self._position_initialized:
                self.place_default()
            else:
                self.move(self._clamp(self.pos()))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._position_initialized:
            self.place_default()

    # ------------------------------------------------------------------
    # Theme (mirrors TrainingFloatControlBar.apply_theme)
    # ------------------------------------------------------------------

    def set_page_mode(self, mode: str) -> None:
        """Switch accent color on both the bar and expanded panel."""
        self._panel.set_page_mode(mode)
        self.apply_theme()

    def apply_theme(self):
        bg = get_color("training_float_bar_bg", "#1e2d3d")
        # Re-use the same accent logic as ControlPanel
        if self._panel._page_mode == "training":
            border = get_color("tab_bg_training", "#F6D393")
        else:
            border = get_color("tab_bg_checked", "#FF7844")
        btn_bg = get_color("training_float_btn_bg", "transparent")
        btn_hover = get_color("training_float_btn_hover", "#374151")
        btn_text = get_color("training_float_btn_text", "#d1d5db")

        self.setStyleSheet(f"""
            #overviewFloatBar {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            #overviewDragHandle {{
                background: transparent;
                border: none;
                color: {btn_text};
            }}
            #overviewDragHandle:hover {{ background: {btn_hover}; border-radius: 4px; }}
            #overviewOpenBtn {{
                background: {btn_bg};
                border: none;
                border-radius: 4px;
                color: {btn_text};
                font-size: 11px;
                font-weight: bold;
                padding: 4px 8px;
            }}
            #overviewOpenBtn:hover {{ background: {btn_hover}; }}
        """)
        self._mini_switcher.apply_theme()
        self._panel.apply_theme()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"expanded": self._expanded, "panel": self._panel.save_state()}

    def restore_state(self, state: dict) -> None:
        self._panel.restore_state(state.get("panel", {}))
        if state.get("expanded", False):
            self.expand()
