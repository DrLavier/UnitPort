#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable row-based UI building blocks for node widgets."""

from typing import Dict, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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

    def center_y(self, proxy) -> float:
        geo = self.geometry()
        return proxy.pos().y() + geo.y() + geo.height() / 2


def make_tag(text: str, style: str) -> QLabel:
    """Create a consistent tag label used on node rows."""
    lbl = QLabel(text)
    lbl.setObjectName("nodeTag")
    lbl.setStyleSheet(style)
    return lbl


def build_row(
    left: Optional[QWidget] = None,
    center: Optional[QWidget] = None,
    right: Optional[QWidget] = None,
    *,
    right_align: bool = True,
    spacing: int = 4,
) -> QWidget:
    """Build a single row with optional left/center/right widgets."""
    row_widget = QWidget()
    row_widget.setStyleSheet("background: transparent;")
    row = QHBoxLayout(row_widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    if left is not None:
        row.addWidget(left)
    if center is not None:
        row.addWidget(center)
    if right_align:
        row.addStretch(1)
    if right is not None:
        row.addWidget(right)
    return row_widget


class NodeRowStack:
    """Manage node rows with strict order, fixed metrics, and safe collapsing.

    Layout contract:
    - Rows are added by id and keep a stable logical order.
    - Optional fixed row height is enforced for all visible rows.
    - Dynamic rows must be collapsed/expanded through this class
      (`set_row_collapsed`) instead of only toggling `setVisible(False)`,
      otherwise Qt can retain phantom spacing in QVBoxLayout.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        spacing: int = 6,
        row_height: int = 0,
        show_row_border: bool = False,
        row_border_color: str = "#ff2bd6",
    ):
        self.container = QWidget(parent)
        self.container.setStyleSheet("background: transparent;")
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(spacing)
        # Keep rows top-anchored so adding/removing rows does not vertically
        # recenter existing content.
        self.layout.setAlignment(Qt.AlignTop)
        self._row_height = int(row_height) if row_height else 0
        self._show_row_border = bool(show_row_border)
        self._row_border_color = row_border_color
        self._rows: Dict[str, QWidget] = {}
        self._row_order = []

    def add_row(self, row_id: str, row_widget: QWidget, before_row_id: Optional[str] = None):
        """Insert a row widget and track it by id."""
        if row_id in self._rows:
            self.remove_row(row_id)

        # Enforce fixed row height when configured.
        if self._row_height > 0:
            row_widget.setMinimumHeight(self._row_height)
            row_widget.setMaximumHeight(self._row_height)

        # Debug aid: visualize each row container boundary.
        if self._show_row_border:
            base_style = row_widget.styleSheet() or ""
            if base_style and not base_style.strip().endswith(";"):
                base_style = base_style.strip() + ";"
            # Apply border directly on the row widget itself.
            row_widget.setStyleSheet(
                base_style + f" border: 1px solid {self._row_border_color};"
            )
            row_widget.setAttribute(Qt.WA_StyledBackground, True)
        row_widget.setProperty("_row_id", row_id)

        if before_row_id and before_row_id in self._rows:
            before_widget = self._rows[before_row_id]
            insert_at = self.layout.indexOf(before_widget)
            if insert_at >= 0:
                self.layout.insertWidget(insert_at, row_widget)
            else:
                self.layout.addWidget(row_widget)
        else:
            self.layout.addWidget(row_widget)
        self._rows[row_id] = row_widget
        if before_row_id and before_row_id in self._row_order:
            insert_pos = self._row_order.index(before_row_id)
            self._row_order.insert(insert_pos, row_id)
        else:
            self._row_order.append(row_id)
        # Ensure the row is visible — inside a QGraphicsProxyWidget the
        # layout does not auto-show newly added children.
        row_widget.show()

    def remove_row(self, row_id: str):
        """Remove a row widget by id."""
        row_widget = self._rows.pop(row_id, None)
        if row_widget is None:
            return
        self.layout.removeWidget(row_widget)
        if row_id in self._row_order:
            self._row_order.remove(row_id)
        row_widget.setParent(None)
        row_widget.deleteLater()

    def row(self, row_id: str) -> Optional[QWidget]:
        return self._rows.get(row_id)

    def set_row_collapsed(self, row_id: str, collapsed: bool):
        """Collapse/expand a row by removing/inserting it in layout order.

        This physically detaches hidden rows from the layout, which prevents
        hidden-row spacing artifacts seen in dynamic nodes.
        """
        row_widget = self._rows.get(row_id)
        if row_widget is None:
            return

        if collapsed:
            if self.layout.indexOf(row_widget) >= 0:
                self.layout.removeWidget(row_widget)
            row_widget.hide()
            row_widget.setMinimumHeight(0)
            row_widget.setMaximumHeight(0)
            self.layout.invalidate()
            return

        if self.layout.indexOf(row_widget) < 0:
            insert_at = self.layout.count()
            if row_id in self._row_order:
                cur_idx = self._row_order.index(row_id)
                for next_id in self._row_order[cur_idx + 1:]:
                    next_widget = self._rows.get(next_id)
                    if next_widget is None:
                        continue
                    next_pos = self.layout.indexOf(next_widget)
                    if next_pos >= 0:
                        insert_at = next_pos
                        break
            self.layout.insertWidget(insert_at, row_widget)

        if self._row_height > 0:
            row_widget.setMinimumHeight(self._row_height)
            row_widget.setMaximumHeight(self._row_height)
        row_widget.show()
        self.layout.invalidate()


class NodeWidgetFactory:
    """Centralized widget factory for consistent node control look/size."""

    def __init__(self, input_style: str, button_style: str, combo_style: str, combo_view_style: str):
        self._input_style = input_style
        self._button_style = button_style
        self._combo_style = combo_style
        self._combo_view_style = combo_view_style

    def button(self, text: str, size: int = 20) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(size, size)
        btn.setStyleSheet(self._button_style)
        return btn

    def line_edit(self, placeholder: str = "") -> QLineEdit:
        line = QLineEdit()
        line.setPlaceholderText(placeholder)
        line.setStyleSheet(self._input_style)
        return line

    def combo_box(self, items=None) -> QComboBox:
        combo = QComboBox()
        if items:
            combo.addItems(items)
        combo.setStyleSheet(self._combo_style + self._combo_view_style)
        return combo


class ConditionSet(QWidget):
    """Boolean condition input chip: [sub_dot square] [logic button].

    The left square hosts a connection dot (port) at its centre;
    the right button displays the current logic expression.
    Left and right segments can use different background colours.
    """

    def __init__(
        self,
        left_bg_color: str,
        right_bg_color: str,
        border_color: str,
        text_color: str,
        hover_bg: str,
        text: str = "< logic >",
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # --- left square (port host) ---
        self.left_box = QFrame()
        self.left_box.setFixedSize(self._h, self._h)
        self.left_box.setStyleSheet(
            f"QFrame {{ border: 1px solid {border_color}; "
            f"background: {left_bg_color}; border-radius: 0px; }}"
        )
        row.addWidget(self.left_box)

        # --- right logic button ---
        self.button = QPushButton(text)
        self.button.setProperty("keep_own_style", True)
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setFixedHeight(self._h)
        self.button.setStyleSheet(
            f"QPushButton {{ background: {right_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }} "
            f"QPushButton:hover {{ background: {hover_bg}; }}"
        )
        row.addWidget(self.button)
        row.addStretch(1)

    def set_logic_text(self, text: str):
        self.button.setText(text or "< logic >")

    def logic_text(self) -> str:
        return self.button.text().strip()


class OutputSet(QWidget):
    """Output/status chip: [sub_dot square] [read-only text box]."""

    def __init__(
        self,
        left_bg_color: str,
        right_bg_color: str,
        border_color: str,
        text_color: str,
        text: str = "if Break",
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # Left square hosts an input sub_dot.
        self.left_box = QFrame()
        self.left_box.setFixedSize(self._h, self._h)
        self.left_box.setStyleSheet(
            f"QFrame {{ border: 1px solid {border_color}; "
            f"background: {left_bg_color}; border-radius: 0px; }}"
        )
        row.addWidget(self.left_box)

        # Right segment is read-only so users understand this is an output tag.
        self.readonly_text = QLineEdit()
        self.readonly_text.setReadOnly(True)
        self.readonly_text.setText(text or "if Break")
        self.readonly_text.setFixedHeight(self._h)
        self.readonly_text.setStyleSheet(
            f"QLineEdit {{ background: {right_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
        )
        row.addWidget(self.readonly_text)
        row.addStretch(1)

    def set_text(self, text: str):
        self.readonly_text.setText(text or "")

    def text(self) -> str:
        return self.readonly_text.text().strip()


class InputSetting(QWidget):
    """Structured IO row: [opt in-box] [key] [input] [opt out-box]."""

    def __init__(
        self,
        dot_bg_color: str,
        key_bg_color: str,
        input_bg_color: str,
        border_color: str,
        text_color: str,
        *,
        key_text: str = "key",
        placeholder: str = "",
        show_left_dot: bool = False,
        show_right_dot: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.left_box: Optional[QFrame] = None
        self.right_box: Optional[QFrame] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        if show_left_dot:
            self.left_box = QFrame()
            self.left_box.setFixedSize(self._h, self._h)
            self.left_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {border_color}; "
                f"background: {dot_bg_color}; border-radius: 0px; }}"
            )
            row.addWidget(self.left_box)

        self.key_box = QLabel()
        self.key_box.setText(key_text or "key")
        self.key_box.setFixedHeight(self._h)
        self.key_box.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        key_left_border = "0px" if show_left_dot else f"1px solid {border_color}"
        self.key_box.setStyleSheet(
            f"QLabel {{ background: {key_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: {key_left_border}; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
        )
        self._fit_key_box_width()
        row.addWidget(self.key_box)

        self.input_edit = QLineEdit()
        self.input_edit.setProperty("keep_own_style", True)
        self.input_edit.setPlaceholderText(placeholder or "")
        self.input_edit.setFixedHeight(self._h)
        self.input_edit.setStyleSheet(
            f"QLineEdit {{ background: {input_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
        )
        row.addWidget(self.input_edit, 1)

        if show_right_dot:
            self.right_box = QFrame()
            self.right_box.setFixedSize(self._h, self._h)
            self.right_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {border_color}; border-left: 0px; "
                f"background: {dot_bg_color}; border-radius: 0px; }}"
            )
            row.addWidget(self.right_box)

    def set_key_text(self, text: str):
        self.key_box.setText(text or "")
        self._fit_key_box_width()

    def key_text(self) -> str:
        return self.key_box.text().strip()

    def set_value(self, text: str):
        self.input_edit.setText(text or "")

    def value(self) -> str:
        return self.input_edit.text().strip()

    def _fit_key_box_width(self):
        txt = self.key_box.text() or "key"
        fm = QFontMetrics(self.key_box.font())
        # text width + horizontal paddings + a small safety for border rendering
        w = int(fm.horizontalAdvance(txt) + 16)
        self.key_box.setFixedWidth(max(34, w))


class ComboSetting(QWidget):
    """Structured IO row: [opt in-box] [key] [combo] [opt out-box]."""

    def __init__(
        self,
        dot_bg_color: str,
        key_bg_color: str,
        combo_bg_color: str,
        border_color: str,
        text_color: str,
        *,
        key_text: str = "key",
        items: Optional[Sequence[str]] = None,
        show_left_dot: bool = False,
        show_right_dot: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.left_box: Optional[QFrame] = None
        self.right_box: Optional[QFrame] = None
        self._middle_widget: Optional[QWidget] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._row_layout = row

        if show_left_dot:
            self.left_box = QFrame()
            self.left_box.setFixedSize(self._h, self._h)
            self.left_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {border_color}; "
                f"background: {dot_bg_color}; border-radius: 0px; }}"
            )
            row.addWidget(self.left_box)

        self.key_box = QLabel()
        self.key_box.setText(key_text or "key")
        self.key_box.setFixedHeight(self._h)
        self.key_box.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        key_left_border = "0px" if show_left_dot else f"1px solid {border_color}"
        self.key_box.setStyleSheet(
            f"QLabel {{ background: {key_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: {key_left_border}; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
        )
        self._fit_key_box_width()
        row.addWidget(self.key_box)

        self.combo = QComboBox()
        self.combo.setProperty("keep_own_style", True)
        self.combo.setFixedHeight(self._h)
        if items:
            self.combo.addItems([str(x) for x in items])
        right_border = "0px" if show_right_dot else "1px"
        self.combo.setStyleSheet(
            f"QComboBox {{ background: {combo_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: 0px; border-right: {right_border}; "
            f"border-radius: 0px; padding: 1px 6px; font-size: 10px; }} "
            f"QComboBox::item {{ color: {text_color}; background: {combo_bg_color}; }} "
            f"QComboBox::item:selected {{ background: {border_color}; color: {text_color}; }} "
            f"QComboBox QAbstractItemView {{ color: {text_color}; }}"
        )
        row.addWidget(self.combo, 1)

        if show_right_dot:
            self.right_box = QFrame()
            self.right_box.setFixedSize(self._h, self._h)
            self.right_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {border_color}; border-left: 0px; "
                f"background: {dot_bg_color}; border-radius: 0px; }}"
            )
            row.addWidget(self.right_box)

    def set_middle_widget(self, widget: Optional[QWidget]) -> None:
        """Insert a compact widget between the combo and the optional right dot."""
        if self._middle_widget is widget:
            return
        if self._middle_widget is not None:
            self._row_layout.removeWidget(self._middle_widget)
            self._middle_widget.setParent(None)
        self._middle_widget = widget
        if widget is None:
            return
        widget.setParent(self)
        insert_at = self._row_layout.count()
        if self.right_box is not None:
            insert_at = max(0, self._row_layout.indexOf(self.right_box))
        self._row_layout.insertWidget(insert_at, widget)

    def set_key_text(self, text: str):
        self.key_box.setText(text or "")
        self._fit_key_box_width()

    def key_text(self) -> str:
        return self.key_box.text().strip()

    def add_items(self, items: Sequence[str]):
        self.combo.addItems([str(x) for x in items])

    def set_items(self, items: Sequence[str]):
        self.combo.clear()
        self.add_items(items)

    def set_current_text(self, text: str):
        self.combo.setCurrentText(text)

    def current_text(self) -> str:
        return self.combo.currentText()

    def _fit_key_box_width(self):
        txt = self.key_box.text() or "key"
        fm = QFontMetrics(self.key_box.font())
        w = int(fm.horizontalAdvance(txt) + 16)
        self.key_box.setFixedWidth(max(34, w))


class BehaviorSetting(QWidget):
    """Behavior row: [opt in-box] [combo] [advanced button]."""

    def __init__(
        self,
        dot_bg_color: str,
        combo_bg_color: str,
        border_color: str,
        text_color: str,
        hover_bg: str,
        *,
        items: Optional[Sequence[str]] = None,
        show_left_dot: bool = False,
        show_advanced_button: bool = True,
        combo_editable: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dot_bg_color = dot_bg_color
        self._combo_bg_color = combo_bg_color
        self._border_color = border_color
        self._text_color = text_color
        self._hover_bg = hover_bg
        self._show_left_dot = bool(show_left_dot)
        self._show_advanced_button = bool(show_advanced_button)

        self.left_box: Optional[QFrame] = None
        self.advanced_btn: Optional[QPushButton] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        if show_left_dot:
            self.left_box = QFrame()
            self.left_box.setFixedSize(self._h, self._h)
            self.left_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {border_color}; "
                f"background: {dot_bg_color}; border-radius: 0px; }}"
            )
            row.addWidget(self.left_box)

        self.combo = QComboBox()
        self.combo.setProperty("keep_own_style", True)
        self.combo.setFixedHeight(self._h)
        self.combo.setEditable(bool(combo_editable))
        if combo_editable:
            self.combo.setInsertPolicy(QComboBox.NoInsert)
        if items:
            self.combo.addItems([str(x) for x in items])
        left_border = "0px" if show_left_dot else f"1px solid {border_color}"
        right_border = "0px" if show_advanced_button else f"1px solid {border_color}"
        self.combo.setStyleSheet(
            f"QComboBox {{ background: {combo_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: {left_border}; border-right: {right_border}; "
            f"border-radius: 0px; padding: 1px 6px; font-size: 10px; }} "
            f"QComboBox::item {{ color: {text_color}; background: {combo_bg_color}; }} "
            f"QComboBox::item:selected {{ background: {border_color}; color: {text_color}; }} "
            f"QComboBox QAbstractItemView {{ color: {text_color}; }}"
        )
        row.addWidget(self.combo, 1)

        if show_advanced_button:
            self.advanced_btn = QPushButton("⚙")
            self.advanced_btn.setProperty("keep_own_style", True)
            self.advanced_btn.setCursor(Qt.PointingHandCursor)
            self.advanced_btn.setFixedSize(self._h, self._h)
            self.advanced_btn.setStyleSheet(
                f"QPushButton {{ background: {dot_bg_color}; color: {text_color}; "
                f"border: 1px solid {border_color}; border-left: 0px; "
                f"border-radius: 0px; padding: 0px; font-size: 11px; }} "
                f"QPushButton:hover {{ background: {hover_bg}; }}"
            )
            row.addWidget(self.advanced_btn)
        self.refresh_style()

    def add_items(self, items: Sequence[str]):
        self.combo.addItems([str(x) for x in items])

    def set_items(self, items: Sequence[str]):
        self.combo.clear()
        self.add_items(items)

    def set_current_text(self, text: str):
        self.combo.setCurrentText(text)

    def current_text(self) -> str:
        return self.combo.currentText()

    # Keep ConditionSet-like compatibility so existing graph_scene sync paths
    # can continue using `_condition_set.logic_text()` for Behavior nodes.
    def set_logic_text(self, text: str):
        value = str(text or "").strip()
        if not value:
            value = "stand"
        if self.combo.findText(value) < 0:
            self.combo.addItem(value)
        self.combo.setCurrentText(value)

    def logic_text(self) -> str:
        return self.current_text().strip()

    def set_theme_colors(
        self,
        dot_bg_color: str,
        combo_bg_color: str,
        border_color: str,
        text_color: str,
        hover_bg: str,
    ) -> None:
        self._dot_bg_color = dot_bg_color
        self._combo_bg_color = combo_bg_color
        self._border_color = border_color
        self._text_color = text_color
        self._hover_bg = hover_bg
        self.refresh_style()

    def refresh_style(self) -> None:
        left_border = "0px" if self._show_left_dot else f"1px solid {self._border_color}"
        right_border = "0px" if self._show_advanced_button else f"1px solid {self._border_color}"

        if self.left_box is not None:
            self.left_box.setStyleSheet(
                f"QFrame {{ border: 1px solid {self._border_color}; "
                f"background: {self._dot_bg_color}; border-radius: 0px; }}"
            )
        self.combo.setStyleSheet(
            f"QComboBox {{ background: {self._combo_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-left: {left_border}; border-right: {right_border}; "
            f"border-radius: 0px; padding: 1px 6px; font-size: 10px; }} "
            f"QComboBox::item {{ color: {self._text_color}; background: {self._combo_bg_color}; }} "
            f"QComboBox::item:selected {{ background: {self._border_color}; color: {self._text_color}; }} "
            f"QComboBox QAbstractItemView {{ color: {self._text_color}; }}"
            f"QComboBox QLineEdit {{ background: {self._combo_bg_color}; color: {self._text_color}; "
            f"border: 0px; padding: 0px; }}"
        )
        if self.advanced_btn is not None:
            self.advanced_btn.setStyleSheet(
                f"QPushButton {{ background: {self._dot_bg_color}; color: {self._text_color}; "
                f"border: 1px solid {self._border_color}; border-left: 0px; "
                f"border-radius: 0px; padding: 0px; font-size: 11px; }} "
                f"QPushButton:hover {{ background: {self._hover_bg}; }}"
            )


class ScriptSetting(BehaviorSetting):
    """Specialized script selector row (same visual contract as BehaviorSetting)."""

    def __init__(self, *args, **kwargs):
        hint_bg_color = kwargs.pop("hint_bg_color", None)
        border_color = kwargs.get("border_color", args[2] if len(args) > 2 else "#4b5563")
        text_color = kwargs.get("text_color", args[3] if len(args) > 3 else "#9ca3af")
        super().__init__(*args, **kwargs)
        self._hint_bg_color = str(hint_bg_color or "#334155")
        self.proxy_hint = QLabel("Params")
        self.proxy_hint.setFixedHeight(self._h)
        self.proxy_hint.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        fm = QFontMetrics(self.proxy_hint.font())
        self.proxy_hint.setFixedWidth(max(48, int(fm.horizontalAdvance(self.proxy_hint.text()) + 12)))
        lay = self.layout()
        if lay is not None and self.left_box is not None:
            lay.insertWidget(1, self.proxy_hint)
        self.refresh_style()

    def set_theme_colors(
        self,
        dot_bg_color: str,
        combo_bg_color: str,
        border_color: str,
        text_color: str,
        hover_bg: str,
        *,
        hint_bg_color: Optional[str] = None,
    ) -> None:
        if hint_bg_color is not None:
            self._hint_bg_color = str(hint_bg_color)
        super().set_theme_colors(dot_bg_color, combo_bg_color, border_color, text_color, hover_bg)

    def refresh_style(self) -> None:
        super().refresh_style()
        if not hasattr(self, "proxy_hint"):
            return
        self.proxy_hint.setStyleSheet(
            f"QLabel {{ "
            f"background: {self._hint_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-left: 0px; border-right: 0px; "
            "border-radius: 0px; padding: 2px 4px; font-size: 10px; }"
        )


class ScriptInAndOut(QWidget):
    """Script IO editor: left=input rows, right=output rows."""

    def __init__(
        self,
        dot_bg_color: str,
        field_bg_color: str,
        border_color: str,
        text_color: str,
        area_bg_color: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self._h = 20
        self._dot_bg_color = dot_bg_color
        self._field_bg_color = field_bg_color
        self._border_color = border_color
        self._text_color = text_color
        self._area_bg_color = area_bg_color
        self.input_rows = []
        self.output_rows = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.input_box = QFrame()
        self.input_box.setObjectName("scriptInputArea")
        self.input_box.setStyleSheet(
            f"QFrame#scriptInputArea {{ border: 0px; background: {self._area_bg_color}; }}"
        )
        self.input_layout = QVBoxLayout(self.input_box)
        self.input_layout.setContentsMargins(4, 4, 4, 4)
        self.input_layout.setSpacing(2)
        self.input_title = QLabel("Inputs")
        self.input_title.setStyleSheet(
            f"QLabel {{ color: {self._text_color}; background: transparent; "
            "font-size: 10px; font-weight: 700; padding: 0px 2px; }"
        )
        self.input_layout.addWidget(self.input_title)
        root.addWidget(self.input_box, 1)

        self.output_box = QFrame()
        self.output_box.setObjectName("scriptOutputArea")
        self.output_box.setStyleSheet(
            f"QFrame#scriptOutputArea {{ border: 0px; background: {self._area_bg_color}; }}"
        )
        self.output_layout = QVBoxLayout(self.output_box)
        self.output_layout.setContentsMargins(4, 4, 4, 4)
        self.output_layout.setSpacing(2)
        self.output_title = QLabel("Outputs")
        self.output_title.setStyleSheet(
            f"QLabel {{ color: {self._text_color}; background: transparent; "
            "font-size: 10px; font-weight: 700; padding: 0px 2px; }"
        )
        self.output_layout.addWidget(self.output_title)
        root.addWidget(self.output_box, 1)
        self.refresh_style()

    def notify_content_changed(self) -> None:
        """Invalidate layout/geometry caches after row add/remove."""
        for layout in (self.layout(), self.input_layout, self.output_layout):
            if layout is not None:
                layout.invalidate()
                layout.activate()
        for widget in (self.input_box, self.output_box, self):
            if widget is not None:
                widget.updateGeometry()

    def add_input_row(self, data_type: str = "any", var_name: str = "") -> Dict[str, QWidget]:
        row = QWidget()
        row.setFixedHeight(self._h)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        left_box = QFrame()
        left_box.setFixedSize(self._h, self._h)
        left_box.setStyleSheet(
            f"QFrame {{ border: 1px solid {self._border_color}; "
            f"background: {self._dot_bg_color}; border-radius: 0px; }}"
        )
        layout.addWidget(left_box)

        data_type_box = QLineEdit()
        data_type_box.setProperty("keep_own_style", True)
        data_type_box.setReadOnly(True)
        data_type_box.setText(str(data_type or "any"))
        data_type_box.setFixedHeight(self._h)
        data_type_box.setFixedWidth(64)
        data_type_box.setStyleSheet(
            f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 4px; font-size: 10px; }}"
        )
        layout.addWidget(data_type_box)

        name_box = QLineEdit()
        name_box.setProperty("keep_own_style", True)
        name_box.setPlaceholderText("input var")
        name_box.setText(str(var_name or ""))
        name_box.setFixedHeight(self._h)
        name_box.setMinimumWidth(108)
        name_box.setStyleSheet(
            f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
        )
        layout.addWidget(name_box, 1)

        self.input_layout.addWidget(row)
        row.show()

        entry = {
            "row": row,
            "left_box": left_box,
            "data_type_box": data_type_box,
            "name_box": name_box,
        }
        self.input_rows.append(entry)
        self.notify_content_changed()
        return entry

    def add_output_row(self, var_name: str = "", data_type: str = "any") -> Dict[str, QWidget]:
        row = QWidget()
        row.setFixedHeight(self._h)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        name_box = QLineEdit()
        name_box.setProperty("keep_own_style", True)
        name_box.setReadOnly(True)
        name_box.setText(str(var_name or ""))
        name_box.setFixedHeight(self._h)
        name_box.setMinimumWidth(108)
        name_box.setStyleSheet(
            f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-radius: 0px; "
            f"padding: 2px 6px; font-size: 10px; }}"
        )
        layout.addWidget(name_box, 1)

        data_type_box = QLineEdit()
        data_type_box.setProperty("keep_own_style", True)
        data_type_box.setReadOnly(True)
        data_type_box.setText(str(data_type or "any"))
        data_type_box.setFixedHeight(self._h)
        data_type_box.setFixedWidth(64)
        data_type_box.setStyleSheet(
            f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
            f"border: 1px solid {self._border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 2px 4px; font-size: 10px; }}"
        )
        layout.addWidget(data_type_box)

        right_box = QFrame()
        right_box.setFixedSize(self._h, self._h)
        right_box.setStyleSheet(
            f"QFrame {{ border: 1px solid {self._border_color}; border-left: 0px; "
            f"background: {self._dot_bg_color}; border-radius: 0px; }}"
        )
        layout.addWidget(right_box)

        self.output_layout.addWidget(row)
        row.show()

        entry = {
            "row": row,
            "name_box": name_box,
            "data_type_box": data_type_box,
            "right_box": right_box,
        }
        self.output_rows.append(entry)
        self.notify_content_changed()
        return entry

    def clear_input_rows(self):
        for entry in self.input_rows:
            row = entry.get("row")
            if row is not None:
                row.setParent(None)
                row.deleteLater()
        self.input_rows.clear()
        self.notify_content_changed()

    def clear_output_rows(self):
        for entry in self.output_rows:
            row = entry.get("row")
            if row is not None:
                row.setParent(None)
                row.deleteLater()
        self.output_rows.clear()
        self.notify_content_changed()

    def refresh_style(self):
        """Re-apply Script Box dedicated colors after theme/load mutations."""
        self.input_box.setStyleSheet(
            f"QFrame#scriptInputArea {{ border: 0px; background: {self._area_bg_color}; }}"
        )
        self.output_box.setStyleSheet(
            f"QFrame#scriptOutputArea {{ border: 0px; background: {self._area_bg_color}; }}"
        )
        self.input_title.setStyleSheet(
            f"QLabel {{ color: {self._text_color}; background: transparent; "
            "font-size: 10px; font-weight: 700; padding: 0px 2px; }"
        )
        self.output_title.setStyleSheet(
            f"QLabel {{ color: {self._text_color}; background: transparent; "
            "font-size: 10px; font-weight: 700; padding: 0px 2px; }"
        )

        for entry in list(self.input_rows):
            left_box = entry.get("left_box")
            data_type_box = entry.get("data_type_box")
            name_box = entry.get("name_box")
            if left_box is not None:
                left_box.setStyleSheet(
                    f"QFrame {{ border: 1px solid {self._border_color}; "
                    f"background: {self._dot_bg_color}; border-radius: 0px; }}"
                )
            if data_type_box is not None:
                data_type_box.setStyleSheet(
                    f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
                    f"border: 1px solid {self._border_color}; border-left: 0px; "
                    f"border-radius: 0px; padding: 2px 4px; font-size: 10px; }}"
                )
            if name_box is not None:
                name_box.setStyleSheet(
                    f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
                    f"border: 1px solid {self._border_color}; border-left: 0px; "
                    f"border-radius: 0px; padding: 2px 6px; font-size: 10px; }}"
                )

        for entry in list(self.output_rows):
            name_box = entry.get("name_box")
            data_type_box = entry.get("data_type_box")
            right_box = entry.get("right_box")
            if name_box is not None:
                name_box.setStyleSheet(
                    f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
                    f"border: 1px solid {self._border_color}; border-radius: 0px; "
                    f"padding: 2px 6px; font-size: 10px; }}"
                )
            if data_type_box is not None:
                data_type_box.setStyleSheet(
                    f"QLineEdit {{ background: {self._field_bg_color}; color: {self._text_color}; "
                    f"border: 1px solid {self._border_color}; border-left: 0px; "
                    f"border-radius: 0px; padding: 2px 4px; font-size: 10px; }}"
                )
            if right_box is not None:
                right_box.setStyleSheet(
                    f"QFrame {{ border: 1px solid {self._border_color}; border-left: 0px; "
                    f"background: {self._dot_bg_color}; border-radius: 0px; }}"
                )

    def set_theme_colors(
        self,
        dot_bg_color: str,
        field_bg_color: str,
        border_color: str,
        text_color: str,
        area_bg_color: str,
    ) -> None:
        self._dot_bg_color = dot_bg_color
        self._field_bg_color = field_bg_color
        self._border_color = border_color
        self._text_color = text_color
        self._area_bg_color = area_bg_color
        self.refresh_style()

    def preferred_height(self) -> int:
        """Return a robust content-driven height for parent row sync.

        Uses both deterministic row math and real Qt size hints to avoid
        clipping when fonts/styles differ after load/theme changes.
        """
        self.notify_content_changed()
        title_h = max(self.input_title.sizeHint().height(), self.output_title.sizeHint().height(), 14)
        row_count = max(len(self.input_rows), len(self.output_rows), 1)
        # top+bottom margins (4+4) + title + spacing below title (2) + rows + inter-row spacing.
        rows_h = row_count * self._h + max(0, row_count - 1) * 2
        computed = int(8 + title_h + 2 + rows_h)
        in_h = int(self.input_box.minimumSizeHint().height()) if self.input_box is not None else 0
        out_h = int(self.output_box.minimumSizeHint().height()) if self.output_box is not None else 0
        return max(computed, in_h, out_h) + 4

class MainPicker(QWidget):
    """Picker chip: [combo(without native arrow)] [square trigger with '▼']."""

    def __init__(
        self,
        combo_style: str,
        combo_view_style: str,
        right_bg_color: str,
        border_color: str,
        text_color: str,
        hover_bg: str,
        combo_text_color: Optional[str] = None,
        items: Optional[Sequence[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._h = 20
        self.setFixedHeight(self._h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.combo = QComboBox()
        self.combo.setProperty("hide_native_arrow", True)
        self.combo.setProperty("main_picker_combo", True)
        self.combo.setFixedHeight(self._h)
        if items:
            self.combo.addItems([str(x) for x in items])

        # Hide native drop-down arrow; expansion is controlled by the right square.
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
        combo_text_override = ""
        if combo_text_color:
            combo_text_override = (
                "QComboBox {"
                f" color: {combo_text_color};"
                "} "
                "QComboBox QAbstractItemView {"
                f" color: {combo_text_color};"
                "}"
            )
        self.combo.setStyleSheet(combo_style + combo_view_style + combo_text_override + combo_no_arrow)
        row.addWidget(self.combo, 1)

        self.trigger = QPushButton("▼")
        self.trigger.setProperty("keep_own_style", True)
        self.trigger.setCursor(Qt.PointingHandCursor)
        self.trigger.setFixedSize(self._h, self._h)
        self.trigger.setStyleSheet(
            f"QPushButton {{ background: {right_bg_color}; color: {text_color}; "
            f"border: 1px solid {border_color}; border-left: 0px; "
            f"border-radius: 0px; padding: 0px; font-size: 10px; }} "
            f"QPushButton:hover {{ background: {hover_bg}; }}"
        )
        self.trigger.clicked.connect(self.combo.showPopup)
        row.addWidget(self.trigger)

    def add_items(self, items: Sequence[str]):
        self.combo.addItems([str(x) for x in items])

    def set_items(self, items: Sequence[str]):
        self.combo.clear()
        self.add_items(items)

    def set_current_text(self, text: str):
        self.combo.setCurrentText(text)

    def current_text(self) -> str:
        return self.combo.currentText()
