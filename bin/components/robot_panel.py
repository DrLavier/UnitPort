#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotPanel — robot identity card for the ControlPanel left_zone.

Shows the active robot portrait, name, category, and key parameters.
Clicking the portrait opens a rich choice selector to change the robot model.
Selection syncs bidirectionally with the canvas Start node.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.system.core.theme_manager import get_color


# ---------------------------------------------------------------------------
# Static robot metadata (attributes not available in ModelSpec)
# ---------------------------------------------------------------------------

_ROBOT_PROFILE: Dict[str, Dict[str, Any]] = {
    "go2": {
        "display_name": "Go2",
        "category": "Quadruped",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Foot Force"],
        "control": ["Joint Position", "Joint Torque"],
    },
    "go2w": {
        "display_name": "Go2W",
        "category": "Quadruped (Wheeled)",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Foot Force", "Wheel Encoder"],
        "control": ["Joint Position", "Wheel Velocity"],
    },
    "go1": {
        "display_name": "Go1",
        "category": "Quadruped",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder"],
        "control": ["Joint Position"],
    },
    "a1": {
        "display_name": "A1",
        "category": "Quadruped",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Foot Force"],
        "control": ["Joint Position", "Joint Torque"],
    },
    "b2": {
        "display_name": "B2",
        "category": "Quadruped",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Foot Force"],
        "control": ["Joint Position", "Joint Torque"],
    },
    "b2w": {
        "display_name": "B2W",
        "category": "Quadruped (Wheeled)",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Wheel Encoder"],
        "control": ["Joint Position", "Wheel Velocity"],
    },
    "g1": {
        "display_name": "G1",
        "category": "Humanoid",
        "limbs": 4,
        "dof": 23,
        "sensors": ["IMU", "Joint Encoder", "Force/Torque"],
        "control": ["Joint Position", "Joint Torque"],
    },
    "h1": {
        "display_name": "H1",
        "category": "Humanoid",
        "limbs": 4,
        "dof": 19,
        "sensors": ["IMU", "Joint Encoder", "Force/Torque"],
        "control": ["Joint Position"],
    },
    "h1_2": {
        "display_name": "H1-2",
        "category": "Humanoid",
        "limbs": 4,
        "dof": 19,
        "sensors": ["IMU", "Joint Encoder", "Force/Torque"],
        "control": ["Joint Position"],
    },
    "spot": {
        "display_name": "Spot",
        "category": "Quadruped",
        "limbs": 4,
        "dof": 12,
        "sensors": ["IMU", "Joint Encoder", "Foot Contact"],
        "control": ["Joint Position", "Body Velocity"],
    },
}

_DEFAULT_PROFILE: Dict[str, Any] = {
    "display_name": "Unknown",
    "category": "Robot",
    "limbs": 0,
    "dof": 0,
    "sensors": [],
    "control": [],
}


def _get_profile(robot_type: str) -> Dict[str, Any]:
    key = str(robot_type or "").strip().lower()
    return _ROBOT_PROFILE.get(key, _DEFAULT_PROFILE)


# ---------------------------------------------------------------------------
# Robot portrait widget (square, icon or letter fallback)
# ---------------------------------------------------------------------------

class _RobotPortrait(QWidget):
    """Square portrait: shows robot icon or a 2-letter fallback badge."""

    clicked = Signal()

    def __init__(self, size: int = 56, parent=None):
        super().__init__(parent)
        self._size = size
        self._icon: Optional[QIcon] = None
        self._letters = "?"
        self._bg_color = QColor("#2a3a4a")
        self._text_color = QColor("#e0e0e0")
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_robot(self, robot_type: str) -> None:
        key = str(robot_type or "").strip().lower()
        # Try to load icon
        self._icon = self._load_icon(key)
        # Fallback letters: first 2 chars, uppercased
        self._letters = key[:2].upper() if key else "?"
        self.update()

    def _load_icon(self, key: str) -> Optional[QIcon]:
        from src.system.core.theme_manager import get_icon
        icon = get_icon(key)
        if not icon.isNull():
            return icon
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rounded rect background
        r = 8
        path = QPainterPath()
        path.addRoundedRect(0, 0, self._size, self._size, r, r)
        painter.fillPath(path, self._bg_color)

        if self._icon is not None:
            # Draw icon centered with margin
            margin = 6
            icon_size = self._size - 2 * margin
            self._icon.paint(painter, margin, margin, icon_size, icon_size)
        else:
            # Draw letter fallback
            font = QFont()
            font.setPixelSize(self._size // 3)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(self._text_color)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._letters)

    def apply_theme(self):
        self._bg_color = QColor(get_color("training_float_bar_bg", "#2a3a4a"))
        self._text_color = QColor(get_color("text_primary", "#e0e0e0"))
        self.update()


# ---------------------------------------------------------------------------
# Parameter list item
# ---------------------------------------------------------------------------

class _ParamRow(QWidget):
    """Single key-value row in the parameter list."""

    def __init__(self, label: str, value: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        self._label = QLabel(label)
        self._label.setObjectName("robotParamLabel")
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hl.addWidget(self._label, 0)

        hl.addStretch(1)

        self._value = QLabel(value)
        self._value.setObjectName("robotParamValue")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hl.addWidget(self._value, 0)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


# ---------------------------------------------------------------------------
# Robot selector popup (rich choice cards)
# ---------------------------------------------------------------------------

class RobotSelectorWidget(QWidget):
    """Embeddable robot selector — shown inside ControlPanel's left dock."""

    robot_selected = Signal(str, str)  # (brand_id, model_id)

    def __init__(self, current_model: str = "", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("robotSelectorPopup")
        self._current_model = current_model.lower()

        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(4)

        title = QLabel("Select Robot")
        title.setObjectName("robotSelectorTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #e0e0e0; padding: 2px 4px;")
        vl.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        container = QWidget()
        self._cards_layout = QVBoxLayout(container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(4)

        self._build_cards()

        scroll.setWidget(container)
        vl.addWidget(scroll, 1)

        self.setFixedWidth(260)

    def _build_cards(self) -> None:
        from src.system.models.model_registry import list_brand_items, get_model_ids, get_model_spec

        for brand_id, brand_display in list_brand_items():
            for model_id in get_model_ids(brand_id):
                spec = get_model_spec(brand_id, model_id)
                if spec is None:
                    continue
                profile = _get_profile(model_id)
                title = spec.display_name
                category = profile.get("category", "Robot")
                desc = f"{category}  |  {profile.get('dof', '?')} DOF"
                selected = model_id.lower() == self._current_model

                card = _RobotCard(
                    brand_id=brand_id,
                    model_id=model_id,
                    title=title,
                    desc=desc,
                    selected=selected,
                    parent=self,
                )
                card.card_clicked.connect(self._on_card_clicked)
                self._cards_layout.addWidget(card)

        self._cards_layout.addStretch(1)

    def _on_card_clicked(self, brand_id: str, model_id: str) -> None:
        self.robot_selected.emit(brand_id, model_id)
        # Ask the ControlPanel to hide the dock
        ctrl = self.parent()
        while ctrl is not None:
            if hasattr(ctrl, 'hide_left_dock'):
                ctrl.hide_left_dock()
                break
            ctrl = ctrl.parent()


class _RobotCard(QFrame):
    """Single robot card in the selector popup."""

    card_clicked = Signal(str, str)  # (brand_id, model_id)

    _ICON_SIZE = 40

    def __init__(
        self,
        brand_id: str,
        model_id: str,
        title: str,
        desc: str,
        selected: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._brand_id = brand_id
        self._model_id = model_id
        self._selected = selected
        self.setObjectName("robotCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(52)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(8, 4, 8, 4)
        hl.setSpacing(8)

        # Icon / letter badge
        icon_label = QLabel(self)
        icon_label.setFixedSize(self._ICON_SIZE, self._ICON_SIZE)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from src.system.core.theme_manager import get_icon
        icon = get_icon(model_id)
        if not icon.isNull():
            icon_label.setPixmap(icon.pixmap(QSize(self._ICON_SIZE - 8, self._ICON_SIZE - 8)))
        else:
            icon_label.setText(model_id[:2].upper())
            icon_label.setStyleSheet(
                f"background: #2a3a4a; border-radius: 6px; color: #ccc; "
                f"font-weight: bold; font-size: {self._ICON_SIZE // 3}px;"
            )
        hl.addWidget(icon_label)

        # Text
        text_vl = QVBoxLayout()
        text_vl.setContentsMargins(0, 0, 0, 0)
        text_vl.setSpacing(1)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #e0e0e0;")
        text_vl.addWidget(title_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet("font-size: 10px; color: #888;")
        text_vl.addWidget(desc_lbl)

        hl.addLayout(text_vl, 1)

        # Selected indicator
        if selected:
            check = QLabel("\u2713")
            check.setStyleSheet("color: #84994F; font-size: 14px; font-weight: bold;")
            hl.addWidget(check)

        self._apply_style()

    def _apply_style(self) -> None:
        if self._selected:
            bg = get_color("hover_bg", "#3d4f63")
            border = get_color("tab_bg_checked", "#FF7844")
        else:
            bg = "transparent"
            border = "transparent"
        hover_bg = get_color("hover_bg", "#3d4f63")
        self.setStyleSheet(f"""
            #robotCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 6px;
            }}
            #robotCard:hover {{
                background: {hover_bg};
                border: 1px solid #555;
                border-radius: 6px;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_clicked.emit(self._brand_id, self._model_id)
            event.accept()
            return
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# RobotPanel — main component for ControlPanel left_zone
# ---------------------------------------------------------------------------

class RobotPanel(QWidget):
    """Robot identity card showing portrait, name, category, and parameters.

    Emits ``robot_changed(brand_id, model_id)`` when the user selects a
    different robot from the portrait popup.  Call ``set_robot(brand, model)``
    to update the panel programmatically (e.g. from a canvas node change).
    """

    robot_changed = Signal(str, str)  # (brand_id, model_id)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("robotPanel")
        self._brand_id = ""
        self._model_id = ""

        vl = QVBoxLayout(self)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.setSpacing(6)

        # ── Top row: portrait + name/category ────────────────────
        top_row = QWidget(self)
        top_hl = QHBoxLayout(top_row)
        top_hl.setContentsMargins(0, 0, 0, 0)
        top_hl.setSpacing(8)

        self._portrait = _RobotPortrait(56, self)
        self._portrait.clicked.connect(self._open_selector)
        top_hl.addWidget(self._portrait, 0)

        name_vl = QVBoxLayout()
        name_vl.setContentsMargins(0, 0, 0, 0)
        name_vl.setSpacing(2)

        self._name_label = QLabel("--")
        self._name_label.setObjectName("robotName")
        self._name_label.setWordWrap(True)
        name_vl.addWidget(self._name_label)

        self._category_label = QLabel("--")
        self._category_label.setObjectName("robotCategory")
        self._category_label.setWordWrap(True)
        name_vl.addWidget(self._category_label)

        name_vl.addStretch(1)
        top_hl.addLayout(name_vl, 1)
        vl.addWidget(top_row, 0)

        # ── Separator ────────────────────────────────────────────
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3d3d3d;")
        vl.addWidget(sep, 0)

        # ── Parameter list ───────────────────────────────────────
        self._param_limbs = _ParamRow("Limbs", "--", self)
        self._param_dof = _ParamRow("DOF", "--", self)
        self._param_control = _ParamRow("Control", "--", self)
        self._param_sensors = _ParamRow("Sensors", "--", self)

        vl.addWidget(self._param_limbs)
        vl.addWidget(self._param_dof)
        vl.addWidget(self._param_control)
        vl.addWidget(self._param_sensors)

        vl.addStretch(1)

        self.apply_theme()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_robot(self, brand_id: str, model_id: str) -> None:
        """Update the panel to show the given robot (no signal emitted)."""
        brand_id = str(brand_id or "").strip().lower()
        model_id = str(model_id or "").strip().lower()
        if brand_id == self._brand_id and model_id == self._model_id:
            return
        self._brand_id = brand_id
        self._model_id = model_id
        self._refresh_display()

    def set_robot_type(self, robot_type: str) -> None:
        """Convenience: set by robot_type, auto-resolving brand."""
        robot_type = str(robot_type or "").strip().lower()
        if robot_type == self._model_id:
            return
        from src.system.models.model_registry import get_robot_brand_map
        brand_map = get_robot_brand_map()
        brand = brand_map.get(robot_type, self._brand_id or "unitree")
        self.set_robot(brand, robot_type)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def brand_id(self) -> str:
        return self._brand_id

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        profile = _get_profile(self._model_id)

        # Portrait
        self._portrait.set_robot(self._model_id)

        # Name — use ModelSpec.display_name if available
        display_name = profile.get("display_name", self._model_id.upper() or "--")
        from src.system.models.model_registry import get_model_spec
        spec = get_model_spec(self._brand_id, self._model_id)
        if spec is not None:
            display_name = spec.display_name
        self._name_label.setText(display_name)

        # Category
        category = profile.get("category", "Robot")
        self._category_label.setText(category)

        # Parameters
        limbs = profile.get("limbs", 0)
        self._param_limbs.set_value(str(limbs) if limbs else "--")

        dof = profile.get("dof", 0)
        self._param_dof.set_value(str(dof) if dof else "--")

        control = profile.get("control", [])
        self._param_control.set_value(", ".join(control) if control else "--")

        sensors = profile.get("sensors", [])
        self._param_sensors.set_value(", ".join(sensors) if sensors else "--")

    # ------------------------------------------------------------------
    # Selector popup
    # ------------------------------------------------------------------

    def _open_selector(self) -> None:
        ctrl_panel = self._find_control_panel()
        if ctrl_panel is None:
            return

        # If already open, toggle off
        if ctrl_panel._left_dock is not None and ctrl_panel._left_dock.isVisible():
            ctrl_panel.hide_left_dock()
            return

        selector = RobotSelectorWidget(current_model=self._model_id)
        selector.robot_selected.connect(self._on_robot_selected)
        ctrl_panel.show_left_dock(selector)

    def _find_control_panel(self) -> Optional[QWidget]:
        """Walk up the parent chain to find the ControlPanel."""
        w = self.parent()
        while w is not None:
            if w.objectName() == "controlPanel":
                return w
            w = w.parent()
        return None

    def _on_robot_selected(self, brand_id: str, model_id: str) -> None:
        if brand_id == self._brand_id and model_id == self._model_id:
            return
        self.set_robot(brand_id, model_id)
        self.robot_changed.emit(brand_id, model_id)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        text_primary = get_color("text_primary", "#e0e0e0")
        text_secondary = get_color("text_secondary", "#9ca3af")

        self.setStyleSheet(f"""
            #robotPanel {{
                background: transparent;
            }}
            #robotName {{
                font-weight: bold;
                font-size: 13px;
                color: {text_primary};
            }}
            #robotCategory {{
                font-size: 11px;
                color: {text_secondary};
            }}
            #robotParamLabel {{
                font-size: 10px;
                color: {text_secondary};
            }}
            #robotParamValue {{
                font-size: 10px;
                color: {text_primary};
                font-weight: 600;
            }}
        """)
        self._portrait.apply_theme()
