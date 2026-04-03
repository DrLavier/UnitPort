import math
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QEasingCurve, QPropertyAnimation, Property, QRectF
from PySide6.QtGui import QFont, QFontMetrics, QColor, QPainter, QPainterPath, QBrush
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QMenu, QPushButton,
    QSizePolicy, QTabBar, QVBoxLayout, QWidget,
)

from src.system.core.theme_manager import get_color, get_font_size

# ---------------------------------------------------------------------------
# Sound assets (shared with WheelSelector)
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).parent.parent.parent
_SOUND_DIR = _ROOT / "assets" / "sound"

SOUND_SET: dict[str, str] = {
    "SLIP": str(_SOUND_DIR / "slip.wav"),
    "BEEP": str(_SOUND_DIR / "beep.wav"),
}


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

        self._items: list = items or []
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

    def setItems(self, items: list):
        self._items = items or []
        self._current_index = min(self._current_index, max(0, len(self._items) - 1))
        self._recalc_item_height()
        self.update()

    def items(self)        -> list: return self._items
    def currentIndex(self) -> int:       return self._current_index
    def currentText(self)  -> str:
        if not self._items:
            return ""
        item = self._items[self._current_index]
        return item[0] if isinstance(item, tuple) else item

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
        if self._items:
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
            item = self._items[idx]
            text = item[0] if isinstance(item, tuple) else item
            self.item_activated.emit(idx, text)
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

        item = self._items[index]
        is_two_line = isinstance(item, tuple)

        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size

        if is_current:
            color = QColor(self._text_hover if is_hovered else self._text_main)
        elif is_hovered:
            color = QColor(self._text_main)
            color.setAlpha(190)
        else:
            color = QColor(self._text_sec)

        text_rect = QRectF(-item_half_w - 10, -h_half,
                           (item_half_w + 10) * 2, self._center_item_height)

        if is_two_line:
            main_text, sub_text = item[0], item[1]
            # 主行
            main_font = QFont(self.font())
            main_font.setBold(is_current)
            main_font.setPointSize(max(1, int(fs)))
            main_h = QFontMetrics(main_font).height()
            # 副行
            sub_font = QFont(self.font())
            sub_font.setBold(False)
            sub_font.setPointSize(max(1, int(fs) - 2))
            sub_h = QFontMetrics(sub_font).height()

            total_h = main_h + sub_h + 2
            top_y = -total_h / 2

            painter.setFont(main_font)
            painter.setPen(color)
            main_rect = QRectF(text_rect.left(), top_y, text_rect.width(), main_h)
            painter.drawText(main_rect, Qt.AlignmentFlag.AlignCenter, main_text)

            sub_color = QColor(color)
            sub_color.setAlpha(max(80, color.alpha() - 60))
            painter.setFont(sub_font)
            painter.setPen(sub_color)
            sub_rect = QRectF(text_rect.left(), top_y + main_h + 2,
                              text_rect.width(), sub_h)
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, sub_text)
        else:
            font = QFont(self.font())
            font.setBold(is_current)
            font.setPointSize(max(1, int(fs)))
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, item)

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

        self._recalc_item_height()
        self.update()

    def _recalc_item_height(self):
        fs = self._custom_font_size if self._custom_font_size else self._theme_font_size
        has_two_line = any(isinstance(it, tuple) for it in self._items)
        if has_two_line:
            main_f = QFont()
            main_f.setPointSize(max(1, int(fs)))
            main_f.setBold(True)
            sub_f = QFont()
            sub_f.setPointSize(max(1, int(fs) - 2))
            self._center_item_height = float(
                QFontMetrics(main_f).height()
                + QFontMetrics(sub_f).height()
                + 2 + 18
            )
        else:
            f = QFont()
            f.setPointSize(max(1, int(fs)))
            f.setBold(True)
            self._center_item_height = float(QFontMetrics(f).height() + 18)

    def refresh_style(self):
        self.apply_theme()


class SwitchButton(QWidget):
    """Animated switch button adapted for the shared UI theme."""

    toggled = Signal(bool)

    def __init__(self, parent=None, checked: bool = False):
        super().__init__(parent)
        self._checked = checked
        self._slider_pos = 1.0 if checked else 0.0

        self._width = 44
        self._height = 22
        self._slider_margin = 2
        self._animation_duration = 180

        self._off_bg = get_color("switch_off_bg", "#111111")
        self._on_bg = get_color("switch_on_bg", "#84994F")
        self._slider_off_color = get_color("switch_slider_off", "#ffffff")
        self._slider_on_color = get_color("switch_slider_on", "#ffffff")

        self.setFixedSize(self._width, self._height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation = QPropertyAnimation(self, b"sliderPos")
        self._animation.setDuration(self._animation_duration)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, animated: bool = True):
        if self._checked == checked:
            return

        self._checked = checked
        target = 1.0 if checked else 0.0

        if animated:
            self._animation.stop()
            self._animation.setStartValue(self._slider_pos)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self._slider_pos = target
            self.update()

        self.toggled.emit(checked)

    def toggle(self):
        self.setChecked(not self._checked)

    def get_slider_pos(self) -> float:
        return self._slider_pos

    def set_slider_pos(self, value: float):
        self._slider_pos = float(value)
        self.update()

    sliderPos = Property(float, get_slider_pos, set_slider_pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        radius = rect.height() / 2.0

        off_color = QColor(self._off_bg)
        on_color = QColor(self._on_bg)
        r = int(off_color.red() + (on_color.red() - off_color.red()) * self._slider_pos)
        g = int(off_color.green() + (on_color.green() - off_color.green()) * self._slider_pos)
        b = int(off_color.blue() + (on_color.blue() - off_color.blue()) * self._slider_pos)
        bg_color = QColor(r, g, b)

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)
        painter.fillPath(path, QBrush(bg_color))

        slider_diameter = rect.height() - 2 * self._slider_margin
        slider_x = self._slider_margin + self._slider_pos * (
            rect.width() - slider_diameter - 2 * self._slider_margin
        )
        slider_y = self._slider_margin

        slider_path = QPainterPath()
        slider_path.addEllipse(QRectF(slider_x, slider_y, slider_diameter, slider_diameter))

        off_slider = QColor(self._slider_off_color)
        on_slider = QColor(self._slider_on_color)
        sr = int(off_slider.red() + (on_slider.red() - off_slider.red()) * self._slider_pos)
        sg = int(off_slider.green() + (on_slider.green() - off_slider.green()) * self._slider_pos)
        sb = int(off_slider.blue() + (on_slider.blue() - off_slider.blue()) * self._slider_pos)
        painter.fillPath(slider_path, QBrush(QColor(sr, sg, sb)))

    def refresh_style(self):
        self._off_bg = get_color("switch_off_bg", "#111111")
        self._on_bg = get_color("switch_on_bg", "#84994F")
        self._slider_off_color = get_color("switch_slider_off", "#ffffff")
        self._slider_on_color = get_color("switch_slider_on", "#ffffff")
        self.update()


class PageSwitcher(QWidget):
    """Shared Mission/Training page switcher."""

    page_selected    = Signal(str)
    page_mode_changed = Signal(str)   # fired on every set_current_page (programmatic or click)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageSwitcher")
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._sound_effect: Optional[QSoundEffect] = None
        self._current_page = "mission"
        self._normal_font_px = 12
        self._active_font_px = self._normal_font_px + 2
        self._button_height = 32
        self._button_horizontal_chrome = 34

        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        self._mission_btn = QPushButton("MISSION")
        self._mission_btn.setObjectName("pageSwitcherMissionButton")
        self._mission_btn.setCheckable(True)
        self._mission_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._mission_btn)
        self._group.addButton(self._mission_btn)

        self._training_btn = QPushButton("TRAINING")
        self._training_btn.setObjectName("pageSwitcherTrainingButton")
        self._training_btn.setCheckable(True)
        self._training_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._training_btn)
        self._group.addButton(self._training_btn)

        self._mission_btn.clicked.connect(lambda checked: self._on_button_clicked("mission", checked))
        self._training_btn.clicked.connect(lambda checked: self._on_button_clicked("training", checked))
        self._update_button_widths()
        self.set_current_page("mission")
        self.apply_theme()

    def set_current_page(self, page: str) -> None:
        target = str(page or "mission").strip().lower()
        new_page = "training" if target == "training" else "mission"
        self._current_page = new_page
        if new_page == "training":
            self._training_btn.setChecked(True)
        else:
            self._mission_btn.setChecked(True)
        # No apply_theme() call needed — Qt applies :checked rules per button automatically
        self.page_mode_changed.emit(new_page)

    def apply_theme(self) -> None:
        container_bg = get_color("tab_bg", "#334155")
        base_text = get_color("text_secondary", "#9ca3af")
        tab_hover_bg = get_color("tab_bg_hover", get_color("hover_bg", "#3d4f63"))
        tab_hover_text = get_color("tab_text_hover", get_color("text_primary", "#e2e8f0"))
        # Per-button checked colors — Qt selects the right rule by button identity (index),
        # no runtime page-string branch required.
        mission_checked_bg  = get_color("tab_bg_checked")
        training_checked_bg = get_color("tab_bg_training")
        tab_checked_text = get_color("tab_text_checked", "#ffffff")
        r = 6
        self.setStyleSheet(
            f"""
            #pageSwitcher {{
                background-color: {container_bg};
                border-radius: 8px;
                padding: 0px;
            }}
            QPushButton#pageSwitcherMissionButton, QPushButton#pageSwitcherTrainingButton {{
                background-color: transparent;
                color: {base_text};
                border: none;
                border-radius: {r}px;
                padding: 6px 16px;
                font-weight: 600;
                font-size: {self._normal_font_px}px;
            }}
            QPushButton#pageSwitcherMissionButton:hover, QPushButton#pageSwitcherTrainingButton:hover {{
                background-color: {tab_hover_bg};
                color: {tab_hover_text};
            }}
            QPushButton#pageSwitcherMissionButton:checked {{
                background-color: {mission_checked_bg};
                font-weight: bold;
                color: {tab_checked_text};
            }}
            QPushButton#pageSwitcherTrainingButton:checked {{
                background-color: {training_checked_bg};
                font-weight: bold;
                color: {tab_checked_text};
            }}
            """
        )

    def _on_button_clicked(self, page: str, checked: bool) -> None:
        if not checked:
            return
        if page == self._current_page:
            return
        self._current_page = page
        self._play_pick_sound()
        self.page_selected.emit(page)

    def _play_pick_sound(self) -> None:
        try:
            sound_path = Path(__file__).resolve().parents[2] / "assets" / "sound" / "picked.wav"
            if not sound_path.exists():
                return
            if self._sound_effect is None:
                self._sound_effect = QSoundEffect(self)
                self._sound_effect.setSource(QUrl.fromLocalFile(str(sound_path)))
                self._sound_effect.setVolume(0.55)
            self._sound_effect.play()
        except Exception:
            pass

    def _update_button_widths(self) -> None:
        self._mission_btn.setFixedHeight(self._button_height)
        self._training_btn.setFixedHeight(self._button_height)
        font = QFont(self._mission_btn.font())
        font.setPixelSize(self._normal_font_px)
        metrics = QFontMetrics(font)
        mission_width = metrics.horizontalAdvance(self._mission_btn.text()) + self._button_horizontal_chrome
        training_width = metrics.horizontalAdvance(self._training_btn.text()) + self._button_horizontal_chrome
        self._mission_btn.setFixedWidth(mission_width)
        self._training_btn.setFixedWidth(training_width)
        layout = self.layout()
        total_width = mission_width + training_width
        if layout is not None:
            margins = layout.contentsMargins()
            total_width += margins.left() + margins.right()
            total_width += layout.spacing()
        self.setFixedWidth(total_width)


class CanvasTabBar(QWidget):
    """Browser-style tab bar: QTabBar + "+" button in a plain QWidget.

    Same layout approach as ``_right_host``: QWidget with QHBoxLayout,
    contentsMargins(0, 4, 0, 4), children placed directly.

    Signals: tab_activated(str), tab_close_requested(str), new_tab_requested(str)
    """

    tab_activated = Signal(str)
    tab_close_requested = Signal(str)
    new_tab_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasTabBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#canvasTabBar { background: transparent; border: none; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)

        self._bar = QTabBar(self)
        self._bar.setObjectName("canvasTabBarInner")
        self._bar.setTabsClosable(True)
        self._bar.setExpanding(False)
        self._bar.setUsesScrollButtons(False)
        self._bar.setElideMode(Qt.TextElideMode.ElideRight)
        self._bar.setDrawBase(False)
        layout.addWidget(self._bar, 0)

        self._add_btn = QPushButton("+", self)
        self._add_btn.setObjectName("canvasTabAddBtn")
        self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_btn.setFixedSize(40, 40)
        self._add_btn.clicked.connect(lambda: self.new_tab_requested.emit(""))
        layout.addWidget(self._add_btn, 0)

        layout.addStretch(1)

        self._tab_ids: List[str] = []
        self._block = False

        self._bar.currentChanged.connect(self._on_current_changed)
        self._bar.tabCloseRequested.connect(self._on_close)
        self.apply_theme()

    # -- public API --

    def add_tab(self, tab_id: str, display_name: str) -> None:
        if tab_id in self._tab_ids:
            self.activate_tab(tab_id)
            return
        self._block = True
        idx = self._bar.addTab(display_name)
        self._tab_ids.append(tab_id)
        self._bar.setCurrentIndex(idx)
        self._block = False

    def remove_tab(self, tab_id: str) -> None:
        if tab_id not in self._tab_ids:
            return
        idx = self._tab_ids.index(tab_id)
        self._block = True
        self._bar.removeTab(idx)
        self._tab_ids.pop(idx)
        self._block = False

    def activate_tab(self, tab_id: str) -> None:
        if tab_id in self._tab_ids:
            self._bar.setCurrentIndex(self._tab_ids.index(tab_id))

    def rename_tab(self, tab_id: str, new_name: str) -> None:
        if tab_id in self._tab_ids:
            self._bar.setTabText(self._tab_ids.index(tab_id), new_name)

    def clear_tabs(self) -> None:
        self._block = True
        while self._bar.count() > 0:
            self._bar.removeTab(0)
        self._tab_ids.clear()
        self._block = False

    def active_tab_id(self) -> str:
        idx = self._bar.currentIndex()
        if 0 <= idx < len(self._tab_ids):
            return self._tab_ids[idx]
        return ""

    def tab_count(self) -> int:
        return len(self._tab_ids)

    # -- internal --

    def _on_current_changed(self, index: int) -> None:
        if self._block or index < 0:
            return
        if 0 <= index < len(self._tab_ids):
            self.tab_activated.emit(self._tab_ids[index])

    def _on_close(self, index: int) -> None:
        if 0 <= index < len(self._tab_ids):
            self.tab_close_requested.emit(self._tab_ids[index])

    # -- theme --

    def set_page_mode(self, page: str) -> None:
        self._page_mode = page
        self.apply_theme()

    def apply_theme(self) -> None:
        mode = getattr(self, "_page_mode", "training")
        if mode == "training":
            bg = get_color("canvas_tab_training_bg", "#2a2c33")
            sel_bg = get_color("canvas_tab_training_bg_active", "#F6D393")
            text = get_color("canvas_tab_training_text", "#aaaaaa")
            sel_text = get_color("canvas_tab_training_text_active", "#1E1E1E")
        else:
            bg = get_color("canvas_tab_bg", "#2a2c33")
            sel_bg = get_color("canvas_tab_bg_active", "#A390FC")
            text = get_color("canvas_tab_text", "#aaaaaa")
            sel_text = get_color("canvas_tab_text_active", "#1E1E1E")
        hover = get_color("canvas_tab_bg_hover", "#3d3d3d")
        self._bar.setStyleSheet(f"""
            QTabBar {{
                background: transparent;
                border: none;
            }}
            QTabBar::tab {{
                background: {bg};
                color: {text};
                border: none;
                padding: 6px 14px;
                margin-right: 2px;
            }}
            QTabBar::tab:hover {{
                background: {hover};
            }}
            QTabBar::tab:selected {{
                background: {sel_bg};
                color: {sel_text};
                font-weight: bold;
            }}
        """)
        self._add_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {text};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
        """)


class MainRow(QWidget):
    """Top navigation row — WorkSpace selector + PageSwitcher (left),
    canvas file tabs (center), window controls (right).

    Signals
    -------
    workspace_selected(project_id: str)
        User selected a workspace from the dropdown.
    """

    workspace_selected = Signal(str)

    _OBJECT_NAME = "mainRowHeader"

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = (theme or "dark").lower()
        self._current_page = "mission"
        self._projects: List[dict] = []   # [{id, name}, ...]
        self._active_project_id: str = ""
        self.setObjectName(self._OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 10, 0)
        self._layout.setSpacing(0)

        # ── Left zone: WorkSpace selector ─────────────────────────────
        self._left_zone = QWidget(self)
        self._left_zone.setObjectName("mainRowLeftZone")
        left_layout = QHBoxLayout(self._left_zone)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self._left_zone.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        self._workspace_btn = QPushButton("<NEW PROJECT>", self._left_zone)
        self._workspace_btn.setObjectName("workspaceDropdown")
        self._workspace_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._workspace_btn.setFixedHeight(46)
        self._workspace_btn.clicked.connect(self._show_workspace_menu)
        left_layout.addWidget(self._workspace_btn)

        self._layout.addWidget(self._left_zone, 0)

        # ── Center zone: Canvas file tabs (training mode only) ─────────
        self.tab_bar = CanvasTabBar(self)
        self._layout.addWidget(self.tab_bar, 0)

        # Push right zone to the far right
        self._layout.addStretch(1)

        # ── Right zone: host for window controls ───────────────────────
        self._right_host = QWidget(self)
        self._right_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._right_host.setStyleSheet("background: transparent; border: none;")
        self._right_host_layout = QHBoxLayout(self._right_host)
        self._right_host_layout.setContentsMargins(0, 4, 0, 4)
        self._right_host_layout.setSpacing(0)
        self._right_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._layout.addWidget(self._right_host, 0)

        self._right_widget: Optional[QWidget] = None
        self.apply_page_bg()
        self._apply_left_zone_style()
        # Mission mode at init: show project name, hide tab bar
        self.tab_bar.setVisible(False)

    # -- workspace dropdown --

    def set_projects(self, projects: List[dict]) -> None:
        """Update the workspace dropdown choices. Each dict: {id, name}."""
        self._projects = list(projects)

    def set_active_project(self, project_id: str, project_name: str = "") -> None:
        """Update the workspace button label to show active project."""
        self._active_project_id = project_id
        label = project_name or project_id or "[NEW PROJECT]"
        self._workspace_btn.setText(f" {label} \u25bc")

    def _show_workspace_menu(self) -> None:
        menu = QMenu(self)
        menu.setObjectName("workspaceMenu")
        menu.setStyleSheet(self._menu_stylesheet())
        if not self._projects:
            action = menu.addAction("(No projects)")
            action.setEnabled(False)
        else:
            for proj in self._projects:
                pid = proj.get("id", "")
                name = proj.get("name", pid)
                act = menu.addAction(name)
                act.setData(pid)
                if pid == self._active_project_id:
                    font = act.font()
                    font.setBold(True)
                    act.setFont(font)
        chosen = menu.exec(self._workspace_btn.mapToGlobal(
            self._workspace_btn.rect().bottomLeft()
        ))
        if chosen is not None and chosen.data():
            self.workspace_selected.emit(chosen.data())

    def _menu_stylesheet(self) -> str:
        bg = get_color("dropdown_bg", "#1e293b")
        text = get_color("text_primary", "#e2e8f0")
        hover = get_color("hover_bg", "#334155")
        border = get_color("border", "#475569")
        return f"""
            QMenu {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {hover};
            }}
        """

    # -- page mode --

    def set_current_page(self, page: str) -> None:
        """Update header to match the active page.

        Mission  → show project name, hide tab bar.
        Training → hide project name, show tab bar.
        """
        page = "training" if str(page or "").strip().lower() == "training" else "mission"
        if page != self._current_page:
            self._current_page = page
            self.apply_page_bg()
            self.tab_bar.set_page_mode(page)
        # Toggle visibility
        self._left_zone.setVisible(page == "mission")
        self.tab_bar.setVisible(page == "training")

    def apply_page_bg(self) -> None:
        """Update MainRow background and border to match the active page."""
        if self._current_page == "training":
            bg = get_color("main_row_training_bg")
        else:
            bg = get_color("main_row_bg")
        border = get_color("border")
        self.setStyleSheet(
            f"#{self._OBJECT_NAME} {{ background-color: {bg}; border-bottom: 1px solid {border}; }}"
        )

    # -- right widget --

    def set_right_widget(self, widget: QWidget) -> None:
        if self._right_widget is widget:
            return
        if self._right_widget is not None:
            self._right_host_layout.removeWidget(self._right_widget)
            self._right_widget.setParent(None)
        self._right_widget = widget
        if self._right_widget is not None:
            self._right_host_layout.addWidget(self._right_widget)

    # -- theme --

    def apply_theme(self) -> None:
        self.apply_page_bg()
        self._apply_left_zone_style()
        self.tab_bar.apply_theme()

    def _apply_left_zone_style(self) -> None:
        border = get_color("border", "#475569")
        ws_bg = get_color("workspace_btn_bg", "transparent")
        ws_text = get_color("workspace_btn_text", get_color("text_primary", "#e2e8f0"))
        ws_hover = get_color("workspace_btn_hover_bg", get_color("hover_bg", "#3d4f63"))
        ws_border = get_color("workspace_btn_border", border)
        self._left_zone.setStyleSheet(f"""
            #mainRowLeftZone {{
                background-color: transparent;
                border: none;
            }}
        """)
        self._workspace_btn.setStyleSheet(f"""
            #workspaceDropdown {{
                background: {ws_bg};
                color: {ws_text};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            #workspaceDropdown:hover {{
                background-color: {ws_hover};
            }}
        """)
