#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homepage UI components — all startup-screen widgets live here.

Classes (public)
----------------
PageSelector    — 主页三区布局组件（顶部 Logo / 中间按钮 / 底部版本）
LoadingScreen   — 启动加载屏
HomepageWidget  — 全屏主页（渐变背景 + PageSelector）

Startup flow
------------
  LoadingScreen  →  HomepageWidget  →  MainWindow (Mission / Training / Exit)
"""

import configparser
import math
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, Signal, Property, QPropertyAnimation, QEasingCurve,
    QTimer, QUrl, QSize, QRectF, QObject, QEvent,
)
from PySide6.QtGui import (
    QColor, QLinearGradient, QPainter, QPen, QFont, QFontMetrics,
    QBrush, QBrush as _QBrush, QPixmap, QPainterPath, QIcon,
)
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QApplication, QGraphicsScene, QGraphicsBlurEffect, QLineEdit,
    QStackedWidget, QFrame,
)

from src.system.core.theme_manager import get_color, get_font_size, get_color_slot

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).parent.parent.parent
_SOUND_DIR = _ROOT / "assets" / "sound"
_SYSTEM_INI = _ROOT / "config" / "system.ini"

SOUND_SET: dict[str, str] = {
    "SLIP": str(_SOUND_DIR / "slip.wav"),
    "BEEP": str(_SOUND_DIR / "beep.wav"),
}

# ---------------------------------------------------------------------------
# Navigation item registry
# ---------------------------------------------------------------------------
_ITEMS = ["Continue", "New Project", "Load Project", "Settings", "Credits", "Exit"]
_IDX_CONTINUE    = 0
_IDX_NEW_PROJECT = 1
_IDX_LOAD        = 2
_IDX_SETTINGS    = 3
_IDX_CREDITS     = 4
_IDX_EXIT        = 5


class WindowControlButtons(QWidget):
    """Compact window controls with theme-aware hover states."""

    minimize_requested = Signal()
    fullscreen_requested = Signal()
    close_requested = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = (theme or "dark").lower()
        self._is_fullscreen = False
        self._button_extent = 18
        self._setup_ui()
        self.apply_theme()

    def _setup_ui(self) -> None:
        self.setObjectName("windowControlStrip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.minimize_button = self._build_button("Minimize")
        self.minimize_button.setObjectName("windowControlMinimizeButton")
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        self.fullscreen_button = self._build_button("Enter Full Screen")
        self.fullscreen_button.setObjectName("windowControlFullscreenButton")
        self.fullscreen_button.clicked.connect(self.fullscreen_requested.emit)
        layout.addWidget(self.fullscreen_button)

        self.close_button = self._build_button("Close")
        self.close_button.setObjectName("windowControlCloseButton")
        self.close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_button)

        self._refresh_button_icons()

    def _build_button(self, tooltip: str) -> QPushButton:
        button = QPushButton(self)
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setContentsMargins(0, 0, 0, 0)
        return button

    @staticmethod
    def _svg_has_visible_content(path: Path) -> bool:
        if not path.exists() or path.suffix.lower() != ".svg":
            return path.exists()
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False
        visible_tags = ("<path", "<rect", "<circle", "<ellipse", "<line",
                        "<polyline", "<polygon", "<image", "<text", "<use")
        return any(tag in content for tag in visible_tags)

    def _icon_size(self) -> QSize:
        icon_dim = max(10, self._button_extent - 6)
        return QSize(icon_dim, icon_dim)

    @staticmethod
    def _wc_icon(icon_name: str) -> QIcon:
        from src.system.core.theme_manager import get_icon
        return get_icon(icon_name)

    def _refresh_button_icons(self) -> None:
        icon_size = self._icon_size()
        self.minimize_button.setIcon(self._wc_icon("ICON_MIN"))
        self.minimize_button.setIconSize(icon_size)
        fs_name = "ICON_WW" if self._is_fullscreen else "ICON_FW"
        self.fullscreen_button.setIcon(self._wc_icon(fs_name))
        self.fullscreen_button.setIconSize(icon_size)
        self.close_button.setIcon(self._wc_icon("ICON_CL"))
        self.close_button.setIconSize(icon_size)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_button_geometry()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_button_geometry()

    def _sync_button_geometry(self) -> None:
        extent = max(18, self.contentsRect().height())
        self._button_extent = extent
        icon_size = self._icon_size()
        for button in (self.minimize_button, self.fullscreen_button, self.close_button):
            button.setFixedSize(extent, extent)
            button.setIconSize(icon_size)
        self.setFixedSize(extent * 3, extent)

    def set_fullscreen(self, is_fullscreen: bool) -> None:
        self._is_fullscreen = bool(is_fullscreen)
        self.fullscreen_button.setToolTip(
            "Restore Window" if self._is_fullscreen else "Enter Full Screen"
        )
        fs_name = "ICON_WW" if self._is_fullscreen else "ICON_FW"
        self.fullscreen_button.setIcon(self._wc_icon(fs_name))

    def apply_theme(self, theme: Optional[str] = None) -> None:
        current_theme = theme or getattr(get_color_slot(), "_current_theme", self._theme)
        self._theme = (current_theme or "dark").lower()
        text_color = get_color("text_primary", "#ffffff")
        hover_close_bg = get_color("danger", "#c94f4f")
        pressed_bg = get_color("sidebar_button_checked_bg", "#2d3748")

        hover_neutral = get_color("button_hover")
        self.setStyleSheet(
            f"""
            QWidget#windowControlStrip {{
                background: transparent;
            }}
            QWidget#windowControlStrip QPushButton {{
                background: transparent;
                border: none;
                border-radius: 4px;
                color: {text_color};
                padding: 0px;
                margin: 0px;
            }}
            QWidget#windowControlStrip QPushButton#windowControlMinimizeButton:hover,
            QWidget#windowControlStrip QPushButton#windowControlFullscreenButton:hover {{
                background: {hover_neutral};
            }}
            QWidget#windowControlStrip QPushButton#windowControlMinimizeButton:pressed,
            QWidget#windowControlStrip QPushButton#windowControlFullscreenButton:pressed,
            QWidget#windowControlStrip QPushButton#windowControlCloseButton:pressed {{
                background: {pressed_bg};
            }}
            QWidget#windowControlStrip QPushButton#windowControlCloseButton:hover {{
                background: {hover_close_bg};
                color: #ffffff;
            }}
            """
        )
        self._refresh_button_icons()


# ===========================================================================
# PageSelector — 主页三区布局组件
# ===========================================================================

def _read_version_display() -> str:
    """从 config/system.ini 读取 [APP] version_display。"""
    try:
        cfg = configparser.ConfigParser()
        cfg.read(str(_SYSTEM_INI), encoding="utf-8")
        return cfg.get("APP", "version_display", fallback="")
    except Exception:
        return ""


class _MenuButton(QWidget):
    """
    单个菜单项按钮。
    - 普通态：20pt，home_text_main 颜色，无背景
    - hover 态：21pt，home_text_hover 颜色，home_hover_bg 半透明圆角背景
    - active 态：21pt + bold，home_hover_bg 背景常驻
    - 点击：emit clicked
    """

    clicked = Signal()
    entered = Signal()

    _FONT_NORMAL = 20
    _FONT_HOVER  = 21

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False
        self._active = False

        self._label = QLabel(text, self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: transparent;")

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(self._label)

        big_f = QFont()
        big_f.setPointSize(self._FONT_HOVER)
        self.setFixedHeight(QFontMetrics(big_f).height() + 20)
        self.setMinimumWidth(300)

        self._apply_state()

    def set_active(self, active: bool):
        self._active = active
        self._apply_state()

    def is_active(self) -> bool:
        return self._active

    def _apply_state(self):
        if self._active:
            fs = self._FONT_HOVER
            color = get_color("home_text_hover")
            f = QFont()
            f.setPointSize(fs)
            f.setBold(True)
            self._label.setFont(f)
        elif self._hovered:
            fs = self._FONT_HOVER
            color = get_color("home_text_hover")
            f = QFont()
            f.setPointSize(fs)
            self._label.setFont(f)
        else:
            fs = self._FONT_NORMAL
            color = get_color("home_text_main")
            f = QFont()
            f.setPointSize(fs)
            self._label.setFont(f)
        self._label.setStyleSheet(f"color: {color}; background: transparent;")
        self.update()

    def paintEvent(self, event):
        if self._active or self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            bg_color = QColor(get_color("home_hover_bg", "#F6D393"))
            if self._hovered and not self._active:
                bg_color.setAlpha(180)
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.PenStyle.NoPen)
            cx = self.rect().center().x()
            r = QRectF(cx - 150, 2, 300, self.height() - 4)
            painter.drawRoundedRect(r, 8, 8)
        super().paintEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self._apply_state()
        self.entered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_state()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class _SeparatorLine(QWidget):
    """水平分隔线：1px 高，300px 宽，颜色 home_text_main。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(150, 1)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(get_color("home_text_sec")))
        painter.drawRect(self.rect())


class _VerticalDivider(QWidget):
    """纵向分割线，高度由父级设置。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(1)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(get_color("home_text_sec", "#888888")))
        painter.drawRect(self.rect())


class _NewProjectForm(QWidget):
    """
    新建项目表单：居中显示标题、输入框、Confirm / Cancel 按钮。
    Confirm 后切换为 Loading 文字。
    """

    confirmed = Signal(str)   # project name
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addStretch(2)

        # ── 表单容器（confirm 时整体隐藏） ──
        self._form_container = QWidget()
        self._form_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        form_lo = QVBoxLayout(self._form_container)
        form_lo.setContentsMargins(0, 0, 0, 0)
        form_lo.setSpacing(0)

        title = QLabel("Your Project Name:")
        title_font = QFont()
        title_font.setPointSize(20)
        title.setFont(title_font)
        title.setStyleSheet(
            f"color: {get_color('home_text_main', '#ffffff')}; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form_lo.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)
        form_lo.addSpacing(24)

        self._input = QLineEdit()
        self._input.setFixedWidth(360)
        self._input.setFixedHeight(40)
        input_font = QFont()
        input_font.setPointSize(14)
        self._input.setFont(input_font)
        self._input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._input.setStyleSheet(
            f"background: rgba(255,255,255,0.08); "
            f"color: {get_color('home_text_main', '#ffffff')}; "
            f"border: 1px solid {get_color('home_text_sec', '#888888')}; "
            f"border-radius: 6px; padding: 4px 12px;"
        )
        self._input.setPlaceholderText("e.g. My Robot Project")
        form_lo.addWidget(self._input, alignment=Qt.AlignmentFlag.AlignHCenter)
        form_lo.addSpacing(32)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        btn_style_base = (
            "QPushButton {{ "
            "background: rgba(255,255,255,0.06); "
            "color: {color}; border: 1px solid {border}; "
            "border-radius: 6px; padding: 8px 28px; font-size: 14px; }}"
            "QPushButton:hover {{ background: rgba(255,255,255,0.14); }}"
        )

        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._confirm_btn.setStyleSheet(btn_style_base.format(
            color=get_color("home_text_hover", "#F6D393"),
            border=get_color("home_hover_bg", "#F6D393"),
        ))
        self._confirm_btn.clicked.connect(self._on_confirm)

        danger_bg = get_color("btn_danger_bg", "#c94f4f")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setStyleSheet(btn_style_base.format(
            color=danger_bg,
            border=danger_bg,
        ))
        self._cancel_btn.clicked.connect(self.cancelled.emit)

        btn_row.addStretch()
        btn_row.addWidget(self._confirm_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        form_lo.addLayout(btn_row)

        root.addWidget(self._form_container, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Loading 文字（初始隐藏） ──
        self._loading_label = QLabel("Loading")
        self._loading_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        loading_font = QFont()
        loading_font.setPointSize(22)
        self._loading_label.setFont(loading_font)
        self._loading_label.setStyleSheet(
            f"color: {get_color('home_text_main', '#ffffff')}; background: transparent;"
        )
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setVisible(False)
        root.addWidget(self._loading_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addStretch(3)

    def _on_confirm(self):
        name = self._input.text().strip()
        if name:
            self._form_container.setVisible(False)
            self._loading_label.setVisible(True)
            self.confirmed.emit(name)

    def reset(self):
        self._loading_label.setVisible(False)
        self._form_container.setVisible(True)
        self._input.clear()
        self._input.setFocus()


class PageSelector(QWidget):
    """
    主页层级菜单布局：
      ┌─ Logo ──────────────────────────────────┐
      ├─ Primary Menu  │ divider │ WheelSelector ┤  (二级展开时)
      └─ separator + version ───────────────────┘

    States:
      - MENU:        仅显示初始菜单
      - SUBMENU:     初始菜单 + 分割线 + WheelSelector
      - NEW_PROJECT: 新建项目表单（替换菜单区域）
    """

    continue_requested    = Signal()
    new_project_created   = Signal(str)   # project name
    load_project_selected = Signal(str)   # project_id
    settings_requested    = Signal()
    credits_requested     = Signal()
    exit_requested        = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme.lower()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._slip_effect: Optional[QSoundEffect] = None
        self._active_index: int = -1       # 当前激活的菜单项 index（-1 = 无）
        self._projects: list = []          # cached ProjectMeta list
        self._project_ids: list[str] = []  # parallel list of project_id
        self._buttons: list[_MenuButton] = []
        self._setup_ui()
        self._refresh_projects()

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addStretch(2)

        # ── Logo ─────────────────────────────────────────────────────────────
        logo = _svg_logo(self._theme, width=800)
        root.addWidget(logo, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(80)

        # ── Content stack (menu / new-project form) ──────────────────────────
        self._content_stack = QStackedWidget()
        self._content_stack.setFrameShape(QFrame.Shape.NoFrame)
        self._content_stack.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        root.addWidget(self._content_stack, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Page 0: menu area (primary + optional submenu)
        self._menu_page = QWidget()
        self._menu_page.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._build_menu_page()
        self._content_stack.addWidget(self._menu_page)

        # Page 1: new project form
        self._new_project_form = _NewProjectForm()
        self._new_project_form.confirmed.connect(self._on_new_project_confirmed)
        self._new_project_form.cancelled.connect(self._back_to_menu)
        self._content_stack.addWidget(self._new_project_form)

        self._content_stack.setCurrentIndex(0)

        root.addSpacing(40)

        # ── Notes: separator + version ───────────────────────────────────────
        sep = _SeparatorLine()
        root.addWidget(sep, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(8)

        ver_text = _read_version_display()
        ver_label = QLabel(ver_text)
        ver_font = QFont()
        ver_font.setPointSize(10)
        ver_label.setFont(ver_font)
        ver_label.setStyleSheet(
            f"color: {get_color('home_text_sec', '#888888')}; background: transparent;"
        )
        ver_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(ver_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addStretch(3)

    def _build_menu_page(self):
        """构建菜单页：左侧 primary buttons + 右侧可展开的 submenu panel。"""
        layout = QHBoxLayout(self._menu_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 左侧：primary menu column ───────────────────────────────────────
        self._primary_col = QWidget()
        self._primary_col.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        primary_lo = QVBoxLayout(self._primary_col)
        primary_lo.setContentsMargins(0, 0, 0, 0)
        primary_lo.setSpacing(20)

        menu_items = [
            (_IDX_CONTINUE,    "Continue"),
            (_IDX_NEW_PROJECT, "New Project"),
            (_IDX_LOAD,        "Load Project"),
            (_IDX_SETTINGS,    "Settings"),
            (_IDX_CREDITS,     "Credits"),
            (_IDX_EXIT,        "Exit"),
        ]
        self._buttons = []
        for idx, text in menu_items:
            btn = _MenuButton(text)
            btn.entered.connect(self._play_slip)
            btn.clicked.connect(lambda _idx=idx: self._on_primary_clicked(_idx))
            primary_lo.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._buttons.append(btn)

        layout.addStretch()
        layout.addWidget(self._primary_col)

        # ── 右侧：submenu panel (divider + WheelSelector) ───────────────────
        self._submenu_panel = QWidget()
        self._submenu_panel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        sub_lo = QHBoxLayout(self._submenu_panel)
        sub_lo.setContentsMargins(20, 0, 0, 0)
        sub_lo.setSpacing(16)

        self._divider = _VerticalDivider()
        sub_lo.addWidget(self._divider)

        from bin.pages.layout.misc import WheelSelector
        self._wheel = WheelSelector(
            items=[],
            visible_count=5,
            radius=110,
            font_size=13,
        )
        self._wheel.setMinimumWidth(400)
        self._wheel.item_activated.connect(self._on_wheel_activated)
        sub_lo.addWidget(self._wheel)

        self._submenu_panel.setVisible(False)
        layout.addWidget(self._submenu_panel)
        layout.addStretch()

    # ── Project data ─────────────────────────────────────────────────────────

    def _refresh_projects(self):
        """从 ProjectStore 加载项目列表并刷新 UI 可见性。"""
        try:
            from src.system.core.project_store import ProjectStore
            store = ProjectStore()
            self._projects = store.list_projects()
        except Exception:
            self._projects = []

        has_projects = len(self._projects) > 0

        # Continue (idx 0) 和 Load Project (idx 2) — 无存档时 hidden
        if len(self._buttons) > _IDX_CONTINUE:
            self._buttons[_IDX_CONTINUE].setVisible(has_projects)
        if len(self._buttons) > _IDX_LOAD:
            self._buttons[_IDX_LOAD].setVisible(has_projects)

        # 构建 WheelSelector 项和 project_id 映射
        self._project_ids = []
        wheel_items = []
        from datetime import datetime
        for meta in self._projects:
            brand = meta.robot.brand or "—"
            model = meta.robot.model or "—"
            # 解析 updated_at 时间
            try:
                dt = datetime.fromisoformat(meta.updated_at)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                time_str = meta.updated_at[:16] if meta.updated_at else "?"
            main_line = f"{meta.name}: {model}({brand})"
            sub_line = time_str
            wheel_items.append((main_line, sub_line))
            self._project_ids.append(meta.project_id)
        self._wheel.setItems(wheel_items)

    # ── Primary menu click handler ───────────────────────────────────────────

    def _on_primary_clicked(self, idx: int):
        # 如果二级菜单已展开 → 收起并清除 active
        if self._active_index >= 0:
            self._collapse_submenu()
            # 如果点击的是同一个项，只收起不执行
            if idx == self._active_index:
                self._active_index = -1
                return
            self._active_index = -1

        # 根据 idx 执行对应动作
        if idx == _IDX_CONTINUE:
            self.continue_requested.emit()
        elif idx == _IDX_NEW_PROJECT:
            self._show_new_project_form()
        elif idx == _IDX_LOAD:
            self._expand_submenu(idx)
        elif idx == _IDX_SETTINGS:
            self.settings_requested.emit()
        elif idx == _IDX_CREDITS:
            self.credits_requested.emit()
        elif idx == _IDX_EXIT:
            self.exit_requested.emit()

    # ── Submenu expand / collapse ────────────────────────────────────────────

    def _expand_submenu(self, idx: int):
        self._active_index = idx
        # 激活按钮高亮
        for i, btn in enumerate(self._buttons):
            btn.set_active(i == idx)
        # 设置分割线高度 = primary menu 总高度
        menu_h = self._primary_col.sizeHint().height()
        self._divider.setFixedHeight(max(menu_h, 100))
        self._submenu_panel.setVisible(True)

    def _collapse_submenu(self):
        self._submenu_panel.setVisible(False)
        for btn in self._buttons:
            btn.set_active(False)
        self._active_index = -1

    # ── WheelSelector activation ─────────────────────────────────────────────

    def _on_wheel_activated(self, idx: int, text: str):
        if 0 <= idx < len(self._project_ids):
            project_id = self._project_ids[idx]
            self._collapse_submenu()
            self.load_project_selected.emit(project_id)

    # ── New Project form ─────────────────────────────────────────────────────

    def _show_new_project_form(self):
        self._new_project_form.reset()
        self._content_stack.setCurrentIndex(1)

    def _back_to_menu(self):
        self._content_stack.setCurrentIndex(0)

    def _on_new_project_confirmed(self, name: str):
        # 表单已切换到 Loading 状态（由 _NewProjectForm._on_confirm 处理）
        try:
            from src.system.core.project_store import ProjectStore
            store = ProjectStore()
            store.create_project(name)
        except Exception:
            pass
        self._refresh_projects()
        self.new_project_created.emit(name)

    # ── Sound ────────────────────────────────────────────────────────────────

    def _play_slip(self):
        try:
            path = SOUND_SET.get("BEEP", "")
            if not path:
                return
            if self._slip_effect is None:
                self._slip_effect = QSoundEffect(self)
                self._slip_effect.setSource(QUrl.fromLocalFile(path))
                self._slip_effect.setVolume(0.45)
            self._slip_effect.play()
        except Exception:
            pass


# ===========================================================================
# Gradient base — shared background for LoadingScreen and HomepageWidget
# ===========================================================================

class _GradientBase(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._wave_phase = 0.0
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(33)
        self._wave_timer.timeout.connect(self._advance_wave_phase)
        self._wave_timer.start()

    def _advance_wave_phase(self):
        self._wave_phase = (self._wave_phase + 0.045) % (math.tau * 4.0)
        self.update()

    @staticmethod
    def _mix_colors(c1: QColor, c2: QColor, ratio: float, alpha: int) -> QColor:
        r = max(0.0, min(1.0, ratio))
        return QColor(
            int(c1.red() * (1.0 - r) + c2.red() * r),
            int(c1.green() * (1.0 - r) + c2.green() * r),
            int(c1.blue() * (1.0 - r) + c2.blue() * r),
            alpha,
        )

    def _build_wave_path(
        self,
        width: float,
        height: float,
        base_y: float,
        amplitude: float,
        wavelength: float,
        thickness: float,
        phase_shift: float,
    ) -> QPainterPath:
        path = QPainterPath()
        step = max(18.0, width / 28.0)
        x = -step
        points_top = []
        points_bottom = []
        while x <= width + step:
            wave = math.sin((x / wavelength) * math.tau + self._wave_phase + phase_shift)
            crest = base_y + amplitude * wave
            points_top.append((x, crest - thickness * 0.5))
            points_bottom.append((x, crest + thickness * 0.5))
            x += step

        if not points_top:
            return path

        path.moveTo(*points_top[0])
        for point in points_top[1:]:
            path.lineTo(*point)
        for point in reversed(points_bottom):
            path.lineTo(*point)
        path.closeSubpath()
        return path

    def _paint_wave_layers(self, painter: QPainter, c1: QColor, c2: QColor):
        w = float(self.width())
        h = float(self.height())
        if w <= 0 or h <= 0:
            return

        diagonal = math.hypot(w, h)
        span = diagonal * 1.5
        band_height = diagonal * 0.24
        amplitude = diagonal * 0.035
        wavelength = diagonal * 0.42

        painter.save()
        painter.translate(w * 0.5, h * 0.5)
        painter.rotate(45)
        painter.translate(-span * 0.5, -span * 0.5)

        layers = [
            (-span * 0.05, 0.0, self._mix_colors(c1, c2, 0.28, 54), self._mix_colors(c1, c2, 0.82, 8)),
            (span * 0.28, 1.2, self._mix_colors(c1, c2, 0.58, 42), self._mix_colors(c1, c2, 0.12, 6)),
            (span * 0.6, 2.4, self._mix_colors(c1, c2, 0.78, 34), self._mix_colors(c1, c2, 0.35, 5)),
        ]
        for base_y, phase_shift, color_a, color_b in layers:
            path = self._build_wave_path(
                span,
                span,
                base_y=base_y + (self._wave_phase * diagonal * 0.03),
                amplitude=amplitude,
                wavelength=wavelength,
                thickness=band_height,
                phase_shift=phase_shift,
            )
            fill = QLinearGradient(0, base_y, span, base_y + band_height)
            fill.setColorAt(0.0, color_b)
            fill.setColorAt(0.5, color_a)
            fill.setColorAt(1.0, color_b)
            painter.fillPath(path, QBrush(fill))

        painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c1 = QColor(get_color("homepage_1", "#1e1e1e"))
        c2 = QColor(get_color("homepage_2", "#2d2d2d"))
        # c3 = QColor(get_color("homepage_3", get_color("homepage_2", "#2d2d2d")))
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, c1)
        grad.setColorAt(1.0, c2)
        # grad.setColorAt(1.0, c3)
        painter.fillRect(self.rect(), QBrush(grad))
        self._paint_wave_layers(painter, c1, c2)
        super().paintEvent(event)


class _SvgLogo(QSvgWidget):
    """SVG widget: fixed to *logo_width* pixels wide, height auto from aspect ratio."""

    def __init__(self, svg_path: str, logo_width: int = 800, parent=None):
        super().__init__(str(svg_path), parent)
        self._logo_width = logo_width
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._apply_size()

    def _apply_size(self):
        r = self.renderer()
        if r and r.isValid():
            ds = r.defaultSize()
            if ds.width() > 0:
                self.setFixedSize(self._logo_width,
                                  int(self._logo_width * ds.height() / ds.width()))
                return
        self.setFixedSize(self._logo_width, self._logo_width)

    def sizeHint(self) -> QSize:
        return self.size()


def _svg_logo(theme: str, width: int = 800) -> _SvgLogo:
    from src.system.core.theme_manager import get_icon_path
    path = get_icon_path("START_LOGO")
    return _SvgLogo(path, logo_width=width)


# ===========================================================================
# LoadingScreen
# ===========================================================================

class LoadingScreen(_GradientBase):
    """
    Shown immediately while the application initialises.
    Call ``close()`` from main.py when init is complete.
    """

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme.lower()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._setup_ui()

    def _setup_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: transparent;")

        from src.system.core.theme_manager import _get_icon_dir
        logo = QSvgWidget(str(_get_icon_dir() / "logo_w.svg"), self)
        logo.setFixedSize(200, 200)
        lo.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

    def close(self):
        super().close()


# ===========================================================================
# HomepageWidget  (full-screen gradient window containing PageSelector)
# ===========================================================================

class HomepageWidget(_GradientBase):
    """
    Full-screen homepage.

    Signals
    -------
    continue_requested    — user clicked "Continue"
    new_project_created   — user created a new project (name)
    load_project_selected — user picked a project from Load Project (project_id)
    settings_requested    — user clicked "Settings"
    credits_requested     — user clicked "Credits"
    exit_requested        — user clicked "Exit"
    """

    continue_requested    = Signal()
    new_project_created   = Signal(str)
    load_project_selected = Signal(str)
    settings_requested    = Signal()
    credits_requested     = Signal()
    exit_requested        = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme.lower()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._setup_ui()

    def _setup_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        self._selector = PageSelector(theme=self._theme)
        self._selector.continue_requested.connect(self.continue_requested)
        self._selector.new_project_created.connect(self.new_project_created)
        self._selector.load_project_selected.connect(self.load_project_selected)
        self._selector.settings_requested.connect(self.settings_requested)
        self._selector.credits_requested.connect(self.credits_requested)
        self._selector.exit_requested.connect(self.exit_requested)
        lo.addWidget(self._selector)


# ===========================================================================
# EscOverlay — Esc 键全屏覆盖导航层
# ===========================================================================

class EscOverlay(QWidget):
    """
    按下 Esc 键后弹出的全屏覆盖层。

    效果：截图当前屏幕 → 模糊化 → 黑色 80% 不透明遮罩 → 居中 PageSelector。
    按 Esc 或选择任意选项后关闭覆盖层。

    Signals
    -------
    closed                — 覆盖层关闭（未选择任何项）
    continue_requested    — 用户选择 "Continue"
    new_project_created   — 用户新建项目
    load_project_selected — 用户选择加载项目
    exit_requested        — 用户选择 "Exit"
    """

    closed                = Signal()
    continue_requested    = Signal()
    new_project_created   = Signal(str)
    load_project_selected = Signal(str)
    exit_requested        = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme
        self._blurred_bg: Optional[QPixmap] = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self._setup_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        selector = PageSelector(theme=self._theme)
        selector.continue_requested.connect(self._on_continue)
        selector.new_project_created.connect(self._on_new_project)
        selector.load_project_selected.connect(self._on_load_project)
        selector.exit_requested.connect(self._on_exit)
        lo.addWidget(selector)

    # ── Navigation handlers ──────────────────────────────────────────────────

    def _on_continue(self):
        self.hide()
        self.continue_requested.emit()

    def _on_new_project(self, name: str):
        self.hide()
        self.new_project_created.emit(name)

    def _on_load_project(self, project_id: str):
        self.hide()
        self.load_project_selected.emit(project_id)

    def _on_exit(self):
        self.hide()
        self.exit_requested.emit()

    def _resolve_target_screen(self, anchor=None):
        candidates = [anchor]
        try:
            candidates.append(QApplication.activeWindow())
        except Exception:
            pass
        candidates.append(self)

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                window = candidate.window() if hasattr(candidate, "window") else None
            except Exception:
                window = None
            for obj in (candidate, window):
                if obj is None:
                    continue
                try:
                    handle = obj.windowHandle() if hasattr(obj, "windowHandle") else None
                    if handle is not None and handle.screen() is not None:
                        return handle.screen()
                except Exception:
                    pass
                try:
                    if hasattr(obj, "screen") and obj.screen() is not None:
                        return obj.screen()
                except Exception:
                    pass
        return QApplication.primaryScreen()

    # ── Show / hide ──────────────────────────────────────────────────────────

    def show_over_screen(self, anchor=None):
        """截图当前屏幕、模糊化后作为背景，然后全屏覆盖显示。"""
        screen = self._resolve_target_screen(anchor)
        self._capture_blurred_bg(screen)
        if screen is None:
            screen = QApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    # ── Blur capture ─────────────────────────────────────────────────────────

    def _capture_blurred_bg(self, screen=None):
        try:
            screen = screen or QApplication.primaryScreen()
            if screen is None:
                self._blurred_bg = None
                return
            pixmap = screen.grabWindow(0)
            scene = QGraphicsScene()
            item = scene.addPixmap(pixmap)
            effect = QGraphicsBlurEffect()
            effect.setBlurRadius(20)
            item.setGraphicsEffect(effect)
            result = QPixmap(pixmap.size())
            result.fill(Qt.GlobalColor.transparent)
            p = QPainter(result)
            scene.render(p)
            p.end()
            self._blurred_bg = result
        except Exception:
            self._blurred_bg = None

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        # 1. blurred screenshot (or solid black fallback)
        if self._blurred_bg:
            painter.drawPixmap(
                0, 0,
                self._blurred_bg.scaled(
                    self.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ),
            )
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
        # 2. Black 80% overlay (vignette glass)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 204))

    # ── Keyboard ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.closed.emit()
        else:
            super().keyPressEvent(event)


# ===========================================================================
# EscKeyFilter — QApplication 级别 Esc 键拦截器
# ===========================================================================

class EscKeyFilter(QObject):
    """
    安装到 QApplication 上，当 EscOverlay 未显示时拦截 Esc 键并呼出覆盖层。
    EscOverlay 可见时不拦截（让覆盖层自身的 keyPressEvent 处理关闭）。
    """

    def __init__(self, overlay: EscOverlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay

    def eventFilter(self, obj, event) -> bool:
        if (event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape
                and not self._overlay.isVisible()):
            self._overlay.show_over_screen(obj)
            return True
        return False
