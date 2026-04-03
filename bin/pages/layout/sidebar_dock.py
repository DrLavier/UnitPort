#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidebar dock component with fixed rail and expandable side content area.
"""

from pathlib import Path

from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class ProjectFilesPanel(QWidget):
    """Sidebar panel that exposes workflow files under a root directory."""

    file_activated = Signal(str)
    load_requested = Signal()
    open_folder_requested = Signal()

    def __init__(self, workflows_root: str, parent=None):
        super().__init__(parent)
        self._workflows_root = Path(workflows_root)
        self._all_files = []

        self.setObjectName("projectFilesPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("projectFilesSearchInput")
        self.search_input.setPlaceholderText("Search .unitport files")
        self.search_input.setFixedHeight(28)
        self.search_input.textChanged.connect(self._apply_filter)
        controls_layout.addWidget(self.search_input, 1)

        self.open_button = QPushButton("📂")
        self.load_button = QPushButton("+")
        self.load_button.setObjectName("projectFilesLoadButton")
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.setFixedSize(28, 28)
        self.load_button.clicked.connect(self.load_requested.emit)
        controls_layout.addWidget(self.load_button)

        self.open_button.setObjectName("projectFilesOpenButton")
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.open_button.setFixedSize(28, 28)
        self.open_button.clicked.connect(self.open_folder_requested.emit)
        controls_layout.addWidget(self.open_button)
        layout.addLayout(controls_layout)

        self.file_list = QListWidget()
        self.file_list.setObjectName("projectFilesList")
        self.file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.file_list, 1)

        self.refresh_files()

    def set_root(self, new_root: str) -> None:
        """Change the root directory and refresh file list."""
        self._workflows_root = Path(new_root)
        self.refresh_files()

    def refresh_files(self):
        self._workflows_root.mkdir(parents=True, exist_ok=True)
        # Scan both legacy .unitport and new .workflow.json files
        all_files = list(self._workflows_root.rglob("*.unitport"))
        all_files.extend(self._workflows_root.rglob("*.workflow.json"))
        self._all_files = sorted(set(all_files), key=lambda p: p.as_posix().lower())
        self._apply_filter()

    def _apply_filter(self):
        query = self.search_input.text().strip().lower()
        self.file_list.clear()
        for path in self._all_files:
            rel_path = path.relative_to(self._workflows_root).as_posix()
            if query and query not in rel_path.lower():
                continue
            item = QListWidgetItem(rel_path)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.file_list.addItem(item)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.file_activated.emit(path)


class SidebarHoverButton(QPushButton):
    """Sidebar button with animated icon scaling on hover."""

    hover_changed = Signal(bool)

    def __init__(self, title: str = "", base_icon_extent: int = 24, parent=None):
        super().__init__(parent)
        self._title = title
        self._base_icon_extent = max(16, int(base_icon_extent))
        self._hover_icon_extent = max(self._base_icon_extent + 2, int(round(self._base_icon_extent * 1.1)))
        self._icon_extent = self._base_icon_extent
        self._hover_animation = QPropertyAnimation(self, b"icon_extent", self)
        self._hover_animation.setDuration(10)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.setIconSize(QSize(self._icon_extent, self._icon_extent))

    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title or ""

    def get_icon_extent(self) -> int:
        return self._icon_extent

    def set_icon_extent(self, extent: int):
        self._icon_extent = max(16, int(extent))
        self.setIconSize(QSize(self._icon_extent, self._icon_extent))

    icon_extent = Property(int, get_icon_extent, set_icon_extent)

    def enterEvent(self, event):
        self._animate_icon(self._hover_icon_extent if not self.icon().isNull() else self._base_icon_extent)
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_icon(self._base_icon_extent)
        self.hover_changed.emit(False)
        super().leaveEvent(event)

    def _animate_icon(self, target_extent: int):
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._icon_extent)
        self._hover_animation.setEndValue(target_extent)
        self._hover_animation.start()


class SidebarDock(QWidget):
    """Fixed left rail plus a contextual content panel that expands on demand."""

    panel_requested = Signal(str)
    panel_changed = Signal(str)
    panel_closed = Signal()
    theme_toggle_requested = Signal()
    language_toggle_requested = Signal()

    def __init__(
        self,
        nav_items=None,
        rail_width: int = 55,
        panel_width: int = 320,
        config=None,
        parent=None,
    ):
        super().__init__(parent)
        w_gap = 10
        h_gap = 20
        spacing = 2
        
        self._button_size = 55
        self._icon_button_extent = 29
        rail_width = self._button_size+1
        
        self.config = config
        self._rail_width = rail_width
        self._panel_width = panel_width
        self._active_key = ""
        self._panel_meta = {}
        self._nav_buttons = {}
        self._nav_icon_bindings = {}
        self._panel_visible = False

        self.setObjectName("sidebarDock")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        self.rail = QWidget()
        self.rail.setObjectName("sidebarRail")
        self.rail.setFixedWidth(self._rail_width)
        rail_layout = QVBoxLayout(self.rail)
        rail_layout.setContentsMargins(0, h_gap, 1, h_gap)
        rail_layout.setSpacing(spacing)

        self.content_panel = QFrame()
        self.content_panel.setObjectName("sidebarContentPanel")
        self.content_panel.setVisible(False)
        self.content_panel.setFixedWidth(0)
        self.content_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        content_layout = QVBoxLayout(self.content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.header_container = QWidget()
        self.header_container.setObjectName("sidebarContentHeader")
        header_layout = QHBoxLayout(self.header_container)
        header_layout.setContentsMargins(14, 14, 14, 10)
        header_layout.setSpacing(8)

        body_container = QWidget()
        body_layout = QVBoxLayout(body_container)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(10)

        self.content_title = QLabel("Panel")
        self.content_title.setObjectName("sidebarContentTitle")
        self.content_title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(self.content_title, 1)
        content_layout.addWidget(self.header_container)

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("sidebarContentStack")
        body_layout.addWidget(self.content_stack, 1)
        content_layout.addWidget(body_container, 1)

        self.hover_tooltip = QLabel("", self)
        self.hover_tooltip.setObjectName("sidebarHoverTooltip")
        self.hover_tooltip.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.hover_tooltip.setVisible(False)
        self.hover_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hover_tooltip.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._panel_animation = QPropertyAnimation(self, b"panel_width", self)
        self._panel_animation.setDuration(260)
        self._panel_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._panel_animation.finished.connect(self._on_panel_animation_finished)

        items = nav_items or [
            ("user", "User", "acc"),
            ("projects", "Project Files", "prj"),
            ("nodes", "Nodes", "nod"),
        ]
        for key, title, icon_name in items:
            btn = SidebarHoverButton(title=title, base_icon_extent=self._icon_size().width())
            btn.setObjectName("sidebarNavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedSize(self._button_size, self._button_size)
            btn.setIconSize(self._icon_size())
            btn.clicked.connect(lambda checked=False, item_key=key: self._on_nav_clicked(item_key))
            btn.hover_changed.connect(lambda hovered, button=btn: self._on_button_hover_changed(button, hovered))
            rail_layout.addWidget(btn)
            self._nav_buttons[key] = btn
            self._nav_icon_bindings[key] = icon_name
            self.register_panel(key, title)

        rail_layout.addStretch()

        self.theme_button = SidebarHoverButton(title="Theme", base_icon_extent=self._icon_size().width())
        self.theme_button.setObjectName("sidebarUtilityButton")
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.setFixedSize(self._button_size, self._button_size)
        self.theme_button.setIconSize(self._icon_size())
        self.theme_button.clicked.connect(self.theme_toggle_requested.emit)
        self.theme_button.hover_changed.connect(
            lambda hovered, button=self.theme_button: self._on_button_hover_changed(button, hovered)
        )
        rail_layout.addWidget(self.theme_button)
        self._apply_theme_toggle_icon("light")

        self.language_button = SidebarHoverButton(title="Language", base_icon_extent=self._icon_size().width())
        self.language_button.setText("EN")
        self.language_button.setObjectName("sidebarUtilityButton")
        self.language_button.setCursor(Qt.PointingHandCursor)
        self.language_button.setFixedSize(self._button_size, self._button_size)
        self.language_button.clicked.connect(self.language_toggle_requested.emit)
        self.language_button.hover_changed.connect(
            lambda hovered, button=self.language_button: self._on_button_hover_changed(button, hovered)
        )
        rail_layout.addWidget(self.language_button)

        root_layout.addWidget(self.rail)
        root_layout.addWidget(self.content_panel)
        root_layout.setAlignment(self.rail, Qt.AlignmentFlag.AlignLeft)
        root_layout.setAlignment(self.content_panel, Qt.AlignmentFlag.AlignLeft)
        self.setFixedWidth(self._rail_width)
        self.refresh_icons("light")

    def register_panel(self, key: str, title: str, widget: QWidget = None):
        """Register or replace a content page for the specified navigation key."""
        if key in self._panel_meta:
            old_widget = self._panel_meta[key]["widget"]
            index = self.content_stack.indexOf(old_widget)
            if index >= 0:
                self.content_stack.removeWidget(old_widget)
            if old_widget is not None:
                old_widget.setParent(None)

        panel_widget = widget or self._build_placeholder_panel(title)
        self.content_stack.addWidget(panel_widget)
        self._panel_meta[key] = {"title": title, "widget": panel_widget}

    def set_panel_widget(self, key: str, widget: QWidget, title: str = None):
        """Replace one panel page with caller-provided content."""
        meta = self._panel_meta.get(key, {})
        resolved_title = title or meta.get("title") or key.title()
        self.register_panel(key, resolved_title, widget)
        if self._active_key == key:
            self.show_panel(key)

    def show_panel(self, key: str):
        """Open the side content panel and switch to the selected page."""
        self._hide_hover_tooltip()
        meta = self._panel_meta.get(key)
        if not meta:
            self.register_panel(key, key.title())
            meta = self._panel_meta[key]

        self._active_key = key
        self.content_title.setText(meta["title"])
        self.content_stack.setCurrentWidget(meta["widget"])
        self._panel_visible = True
        self.content_panel.setVisible(True)
        self._animate_panel(self._panel_width)
        self._sync_button_states()
        self.panel_changed.emit(key)

    def close_panel(self):
        """Collapse the side content panel without altering registered pages."""
        self._hide_hover_tooltip()
        self._active_key = ""
        self._panel_visible = False
        self._animate_panel(0)
        self._sync_button_states()
        self.panel_closed.emit()

    def active_panel_key(self) -> str:
        return self._active_key

    def _on_nav_clicked(self, key: str):
        self._hide_hover_tooltip()
        if self._panel_visible and self._active_key == key:
            self.close_panel()
            return
        self.show_panel(key)
        self.panel_requested.emit(key)

    def _sync_button_states(self):
        for key, btn in self._nav_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(self._panel_visible and key == self._active_key)
            btn.blockSignals(False)

    def _animate_panel(self, target_width: int):
        self._panel_animation.stop()
        self._panel_animation.setStartValue(self.panel_width)
        self._panel_animation.setEndValue(target_width)
        self._panel_animation.start()

    def _on_panel_animation_finished(self):
        if not self._panel_visible and self.panel_width == 0:
            self.content_panel.setVisible(False)

    def get_panel_width(self) -> int:
        return self.content_panel.width()

    def set_panel_width(self, width: int):
        safe_width = max(0, int(width))
        self.content_panel.setFixedWidth(safe_width)
        self.setFixedWidth(self._rail_width + safe_width)

    panel_width = Property(int, get_panel_width, set_panel_width)

    def _build_placeholder_panel(self, title: str) -> QWidget:
        page = QWidget()
        page.setObjectName("sidebarPlaceholderPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        label = QLabel(f"{title} panel placeholder")
        label.setObjectName("sidebarPlaceholderLabel")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(label)
        layout.addStretch()
        return page

    def _on_button_hover_changed(self, button: SidebarHoverButton, hovered: bool):
        if hovered:
            self._show_hover_tooltip(button)
            return
        if self.hover_tooltip.isVisible():
            self._hide_hover_tooltip()

    def _show_hover_tooltip(self, button: SidebarHoverButton):
        title = button.title().strip()
        if not title:
            return
        host = self.window()
        if self.hover_tooltip.parentWidget() is not host:
            self.hover_tooltip.setParent(host)
        self.hover_tooltip.setText(title)
        self.hover_tooltip.adjustSize()
        tooltip_width = max(96, self.hover_tooltip.width() + 22)
        tooltip_height = max(self._button_size - 2, self.hover_tooltip.height() + 10)
        self.hover_tooltip.resize(tooltip_width, tooltip_height)
        button_top_left = button.mapTo(host, button.rect().topLeft())
        x = button_top_left.x() + button.width() + 14
        y = button_top_left.y() + max(0, (button.height() - tooltip_height) // 2)
        max_x = max(0, host.width() - tooltip_width - 8)
        max_y = max(0, host.height() - tooltip_height - 8)
        self.hover_tooltip.move(min(max(0, x), max_x), min(max(0, y), max_y))
        self.hover_tooltip.raise_()
        self.hover_tooltip.show()

    def _hide_hover_tooltip(self):
        self.hover_tooltip.hide()

    def leaveEvent(self, event):
        self._hide_hover_tooltip()
        super().leaveEvent(event)

    def refresh_icons(self, theme: str):
        dark_mode = str(theme or "").lower() == "dark"
        from src.system.core.theme_manager import get_icon
        for key, button in self._nav_buttons.items():
            icon_name = self._nav_icon_bindings.get(key, "")
            if icon_name:
                icon = get_icon(icon_name)
                if not icon.isNull():
                    button.setIcon(icon)
                    button.setText("")
                else:
                    button.setIcon(QIcon())
        self._apply_theme_toggle_icon(theme)

    def _apply_theme_toggle_icon(self, theme: str):
        from src.system.core.theme_manager import get_icon
        icon = get_icon("L&D")
        if not icon.isNull():
            self.theme_button.setIcon(icon)
            self.theme_button.setText("")

    def _icon_size(self):
        icon_px = max(20, self._icon_button_extent)
        return QSize(icon_px, icon_px)
