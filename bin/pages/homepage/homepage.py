#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homepage UI components — all startup-screen widgets live here.

Classes (public)
----------------
WheelSelector   — 转轮式选择器（原 wheel_selector.py，已内联）
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
    QApplication, QGraphicsScene, QGraphicsBlurEffect,
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
_ITEMS       = ["Mission Control", "Training Ground", "Exit"]
_IDX_MISSION  = 0
_IDX_TRAINING = 1
_IDX_EXIT     = 2


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
# WheelSelector  (moved from wheel_selector.py)
# ===========================================================================
class WheelSelector(QWidget):
    """
    转轮式选择器（Wheel Selector）

    特性：
    - 垂直半圆转轮布局，选中项居中
    - 上下项逐级缩放 + 虚化
    - hover 高亮（选中项 hover 时显示背景；非选中项 hover 半透明背景）
    - 鼠标点击：非选中项 → 旋转；选中项 → 触发 item_activated
    - 滚轮 / 键盘支持
    """

    currentIndexChanged = Signal(int, str)
    item_activated      = Signal(int, str)

    def __init__(
        self,
        items=None,
        parent=None,
        visible_count: int = 5,
        radius: int = 100,
        font_size: Optional[int] = None,
    ):
        super().__init__(parent)

        self._items: list[str] = items or []
        self._current_index: int = 0
        self._offset: float = 0.0
        self._hover_index: int = -1

        self.visible_count = visible_count
        self.radius        = radius
        self._custom_font_size = font_size

        self.setMinimumWidth(120)
        self.setMinimumHeight(120)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)

        self._anim = QPropertyAnimation(self, b"offset")
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._angle_step = math.pi / self.visible_count

        self._pending_steps  = 0
        self._anim_running   = False

        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(200)
        self._emit_timer.timeout.connect(self._emit_current_index)

        self._slip_effect: Optional[QSoundEffect] = None
        self._beep_effect: Optional[QSoundEffect] = None

        self.apply_theme()

        if self._items:
            QTimer.singleShot(0, self._emit_current_index)

    # ── Public API ──────────────────────────────────────────────────────────

    def setItems(self, items: list[str]):
        self._items = items or []
        self._current_index = min(self._current_index, max(0, len(self._items) - 1))
        self.update()

    def items(self)        -> list[str]: return self._items
    def currentIndex(self) -> int:       return self._current_index
    def currentText(self)  -> str:
        return self._items[self._current_index] if self._items else ""

    def selectNext(self): self._enqueue_step(1)
    def selectPrev(self): self._enqueue_step(-1)

    def setCurrentIndex(self, index: int):
        if not self._items:
            return
        index = max(0, min(index, len(self._items) - 1))
        diff  = index - self._current_index
        if diff:
            self._enqueue_step(diff)

    # ── Animation property (PySide6) ────────────────────────────────────────

    def _get_offset(self) -> float: return self._offset
    def _set_offset(self, v: float):
        self._offset = max(-1.5, min(1.5, v))
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    # ── Internal animation ──────────────────────────────────────────────────

    def _enqueue_step(self, step: int):
        if not self._items:
            return
        self._pending_steps += step
        self._pending_steps = max(
            -self._current_index,
            min(self._pending_steps, len(self._items) - 1 - self._current_index),
        )
        if not self._anim_running:
            self._dequeue_and_animate()

    def _dequeue_and_animate(self):
        if self._pending_steps == 0:
            self._anim_running = False
            return
        step = 1 if self._pending_steps > 0 else -1
        self._pending_steps -= step
        target = max(0, min(self._current_index + step, len(self._items) - 1))
        self._anim_running = True
        self._play_sound("SLIP")
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(self._offset - step)
        try:
            self._anim.finished.disconnect()
        except RuntimeError:
            pass
        self._anim.finished.connect(lambda idx=target: self._commit_and_continue(idx))
        self._anim.start()

    def _commit_and_continue(self, index: int):
        self._offset = 0.0
        self._current_index = index
        self.update()
        self._emit_timer.start()
        self._dequeue_and_animate()

    def _emit_current_index(self):
        self.currentIndexChanged.emit(self._current_index, self.currentText())

    # ── Sound ───────────────────────────────────────────────────────────────

    def _play_sound(self, key: str):
        try:
            path = SOUND_SET.get(key, "")
            if not path:
                return
            if key == "SLIP":
                if self._slip_effect is None:
                    self._slip_effect = QSoundEffect(self)
                    self._slip_effect.setSource(QUrl.fromLocalFile(path))
                    self._slip_effect.setVolume(0.45)
                self._slip_effect.play()
            elif key == "BEEP":
                if self._beep_effect is None:
                    self._beep_effect = QSoundEffect(self)
                    self._beep_effect.setSource(QUrl.fromLocalFile(path))
                    self._beep_effect.setVolume(0.55)
                self._beep_effect.play()
        except Exception:
            pass

    # ── Input events ────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._enqueue_step(-1 if delta > 0 else 1)
        event.accept()

    def mouseMoveEvent(self, event):
        new = self._item_at_pos(event.pos().y())
        if new != self._hover_index:
            self._hover_index = new
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if new >= 0
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._item_at_pos(event.pos().y())
        if idx < 0:
            return
        if idx == self._current_index:
            self._play_sound("BEEP")
            self.item_activated.emit(idx, self._items[idx])
        else:
            self.setCurrentIndex(idx)

    def _item_at_pos(self, mouse_y: int) -> int:
        cy       = self.height() / 2
        hit_half = max(self._center_item_height / 2, 22)
        for i in range(-(self.visible_count // 2), self.visible_count // 2 + 1):
            idx   = self._current_index + i
            if not (0 <= idx < len(self._items)):
                continue
            theta = (i + self._offset) * self._angle_step
            if abs(theta) > math.pi / 2:
                continue
            y = cy + math.sin(theta) * self.radius
            if abs(mouse_y - y) <= hit_half:
                return idx
        return -1

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        bg = QColor(self._bg_color)
        if bg.alpha() > 0:
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect())

        if self._side_border_width > 0:
            painter.save()
            pen = QPen(QColor(self._border_color))
            pen.setWidth(self._side_border_width)
            painter.setPen(pen)
            half = self._side_border_width // 2
            painter.drawLine(half, 0, half, h)
            painter.drawLine(w - half, 0, w - half, h)
            painter.restore()

        if not self._items:
            return

        cy, cx    = h / 2, w / 2
        item_half_w: float = cx - 12
        h_half    = self._center_item_height / 2

        # 中心高亮背景：仅 hover 时显示
        if self._hover_index == self._current_index:
            cr = QRectF(cx - item_half_w, cy - h_half,
                        item_half_w * 2, self._center_item_height)
            painter.setBrush(QColor(self._hover_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(cr, self._center_item_radius, self._center_item_radius)

        for i in range(-(self.visible_count // 2), self.visible_count // 2 + 1):
            idx = self._current_index + i
            if 0 <= idx < len(self._items):
                self._draw_item(painter, idx, i + self._offset, cx, cy, item_half_w)

    def _draw_item(self, painter, index, offset, cx, cy, item_half_w):
        theta = offset * self._angle_step
        if abs(theta) > math.pi / 2:
            return
        y       = cy + math.sin(theta) * self.radius
        scale   = max(0.65, 1.0 - abs(offset) * 0.18)
        opacity = max(0.20, 1.0 - abs(offset) * 0.28)
        is_current = (index == self._current_index and abs(offset) < 0.01)
        is_hovered = (index == self._hover_index)
        h_half     = self._center_item_height / 2

        painter.save()
        painter.translate(cx, y)
        painter.scale(scale, scale)
        painter.setOpacity(opacity)

        if is_hovered and not is_current:
            painter.save()
            hb = QColor(self._hover_bg)
            hb.setAlpha(60)
            painter.setBrush(QBrush(hb))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(-item_half_w, -h_half, item_half_w * 2, self._center_item_height),
                self._center_item_radius, self._center_item_radius,
            )
            painter.restore()

        font = QFont(self.font())
        font.setBold(is_current)
        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size
        font.setPointSize(max(1, int(fs)))
        painter.setFont(font)

        if is_current:
            color = QColor(self._text_hover if is_hovered else self._text_main)
        elif is_hovered:
            color = QColor(self._text_main)
            color.setAlpha(190)
        else:
            color = QColor(self._text_sec)

        painter.setPen(color)
        painter.drawText(
            QRectF(-item_half_w - 10, -h_half, (item_half_w + 10) * 2, self._center_item_height),
            Qt.AlignmentFlag.AlignCenter, self._items[index],
        )
        painter.restore()

    # ── Theme ────────────────────────────────────────────────────────────────

    def apply_theme(self):
        self._text_main  = get_color("home_text_main",  "#ffffff")
        self._text_sec   = get_color("home_text_sec",   "#888888")
        self._text_hover = get_color("home_text_hover", "#F6D393")
        self._hover_bg   = get_color("home_hover_bg",   "#F6D393")
        self._bg_color   = "transparent"
        self._border_color = self._hover_bg
        self._theme_font_size = get_font_size("size_normal") or 12

        self._side_border_width   = 0
        self._center_item_radius  = 8

        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size
        f  = QFont()
        f.setPointSize(max(1, int(fs)))
        f.setBold(True)
        self._center_item_height = float(QFontMetrics(f).height() + 18)
        self.update()

    def refresh_style(self):
        self.apply_theme()


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
    - 普通态：24pt，home_text_sec 颜色，无背景
    - hover 态：28pt，home_text_main 颜色，home_hover_bg 半透明圆角背景
    - 点击：emit clicked
    """

    clicked = Signal()
    entered = Signal()   # 鼠标首次进入时（供父级播放音效）

    _FONT_NORMAL = 20
    _FONT_HOVER  = 21

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hovered = False

        self._label = QLabel(text, self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: transparent;")

        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(self._label)

        # 固定高度以防止 hover 时字体放大导致布局跳动；最小宽度保证背景色块有足够空间
        big_f = QFont()
        big_f.setPointSize(self._FONT_HOVER)
        self.setFixedHeight(QFontMetrics(big_f).height() + 20)
        self.setMinimumWidth(300)

        self._apply_state(hovered=False)

    def _apply_state(self, hovered: bool):
        self._hovered = hovered
        fs    = self._FONT_HOVER if hovered else self._FONT_NORMAL
        color = get_color("home_text_hover") if hovered \
                else get_color("home_text_main")
        f = QFont()
        f.setPointSize(fs)
        self._label.setFont(f)
        self._label.setStyleSheet(f"color: {color}; background: transparent;")
        self.update()   # 触发 paintEvent 刷新背景

    def paintEvent(self, event):
        if self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(get_color("home_hover_bg", "#F6D393"))))
            painter.setPen(Qt.PenStyle.NoPen)
            cx = self.rect().center().x()
            r = QRectF(cx - 150, 2, 300, self.height() - 4)
            painter.drawRoundedRect(r, 8, 8)
        super().paintEvent(event)

    def enterEvent(self, event):
        self._apply_state(hovered=True)
        self.entered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_state(hovered=False)
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


class PageSelector(QWidget):
    """
    主页三区布局：
      ┌─ Title  ─────────────────┐  Logo SVG
      ├─ Buttons ────────────────┤  垂直菜单项（24→28pt，hover 音效）
      └─ Notes  ─────────────────┘  分隔线 + 版本号
    """

    mission_requested  = Signal()
    training_requested = Signal()
    exit_requested     = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme.lower()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._slip_effect: Optional[QSoundEffect] = None
        self._setup_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addStretch(2)

        # ── Title: Logo ──────────────────────────────────────────────────────
        logo = _svg_logo(self._theme, width=800)
        root.addWidget(logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addSpacing(80)

        # ── Buttons ──────────────────────────────────────────────────────────
        labels = [
            ("Mission Control",  self.mission_requested),
            ("Training Ground",  self.training_requested),
            ("Exit",             self.exit_requested),
        ]
        for text, sig in labels:
            btn = _MenuButton(text)
            btn.entered.connect(self._play_slip)
            btn.clicked.connect(sig)
            root.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            root.addSpacing(20)

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
    mission_requested  — user clicked "Mission Control"
    training_requested — user clicked "Training Ground"
    exit_requested     — user clicked "Exit"
    """

    mission_requested  = Signal()
    training_requested = Signal()
    exit_requested     = Signal()

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = theme.lower()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._setup_ui()

    def _setup_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)

        selector = PageSelector(theme=self._theme)
        selector.mission_requested.connect(self.mission_requested)
        selector.training_requested.connect(self.training_requested)
        selector.exit_requested.connect(self.exit_requested)
        lo.addWidget(selector)


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
    closed             — 覆盖层关闭（未选择任何项）
    mission_requested  — 用户选择 "Mission Control"
    training_requested — 用户选择 "Training Ground"
    exit_requested     — 用户选择 "Exit"
    """

    closed             = Signal()
    mission_requested  = Signal()
    training_requested = Signal()
    exit_requested     = Signal()

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
        selector.mission_requested.connect(self._on_mission)
        selector.training_requested.connect(self._on_training)
        selector.exit_requested.connect(self._on_exit)
        lo.addWidget(selector)

    # ── Navigation handlers ──────────────────────────────────────────────────

    def _on_mission(self):
        self.hide()
        self.mission_requested.emit()

    def _on_training(self):
        self.hide()
        self.training_requested.emit()

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
