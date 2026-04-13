import math
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QUrl, QTimer, QSize, Signal, QEasingCurve, QPropertyAnimation, Property, QRectF
from PySide6.QtGui import QFont, QFontMetrics, QColor, QPainter, QPainterPath, QBrush
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QHBoxLayout, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
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


class ThemeSwitchButton(QWidget):
    """Theme toggle implemented as an animated switch with an SVG icon baked
    into the slider knob. Colors are delegated to the ``theme_switch_*`` slots
    in ``ui.ini`` so they can be themed independently of the generic SwitchButton.
    """

    toggled = Signal(bool)

    def __init__(self, parent=None, checked: bool = False, height: int = 22):
        super().__init__(parent)
        self._checked = checked
        self._slider_pos = 1.0 if checked else 0.0

        # Frame height equals the slider diameter (no vertical margin), and
        # frame width is exactly twice the height → a compact pill with the
        # knob traversing the full length. ``height`` is parameterised so the
        # caller (e.g. MainRow) can match adjacent control sizes.
        self._height = int(height)
        self._width = self._height * 2
        self._slider_margin = 0
        self._animation_duration = 200

        self._refresh_colors()

        self.setFixedSize(self._width, self._height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation = QPropertyAnimation(self, b"sliderPos")
        self._animation.setDuration(self._animation_duration)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Cached SVG renderers for each state
        self._off_icon_renderer: Optional[QSvgRenderer] = None
        self._on_icon_renderer: Optional[QSvgRenderer] = None
        self._reload_icons()

    def _refresh_colors(self) -> None:
        self._off_bg = get_color("theme_switch_off_bg", "#e5e7eb")
        self._on_bg = get_color("theme_switch_on_bg", "#1f2937")
        self._slider_off_color = get_color("theme_switch_slider_off", "#ffffff")
        self._slider_on_color = get_color("theme_switch_slider_on", "#ffffff")
        self._icon_off_color = get_color("theme_switch_icon_off", "#f59e0b")
        self._icon_on_color = get_color("theme_switch_icon_on", "#e5e7eb")

    def _reload_icons(self) -> None:
        """Load SVG renderers for the off (light/sun) and on (dark/moon) states.

        Prefers dedicated ``theme_switch_light``/``theme_switch_dark`` assets.
        Falls back to the existing ``L&D`` icon pair, using the light variant
        for the off state and the dark (``_w``) variant for the on state.
        """
        self._off_icon_renderer = self._load_svg_candidates([
            "icon_theme_sun.svg",
            "icon_L&D.svg",
        ])
        self._on_icon_renderer = self._load_svg_candidates([
            "icon_theme_moon.svg",
            "icon_L&D_w.svg",
        ])

    @staticmethod
    def _load_svg_candidates(filenames) -> Optional[QSvgRenderer]:
        icon_dir = Path(__file__).resolve().parents[2] / "assets" / "icon"
        for name in filenames:
            path = icon_dir / name
            if path.exists():
                renderer = QSvgRenderer(str(path))
                if renderer.isValid():
                    return renderer
        return None

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

    @staticmethod
    def _interp_color(off_color: QColor, on_color: QColor, t: float) -> QColor:
        r = int(off_color.red() + (on_color.red() - off_color.red()) * t)
        g = int(off_color.green() + (on_color.green() - off_color.green()) * t)
        b = int(off_color.blue() + (on_color.blue() - off_color.blue()) * t)
        return QColor(r, g, b)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        radius = rect.height() / 2.0

        # Track background
        bg_color = self._interp_color(
            QColor(self._off_bg), QColor(self._on_bg), self._slider_pos
        )
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)
        painter.fillPath(path, QBrush(bg_color))

        # Slider knob
        slider_diameter = rect.height() - 2 * self._slider_margin
        slider_x = self._slider_margin + self._slider_pos * (
            rect.width() - slider_diameter - 2 * self._slider_margin
        )
        slider_y = self._slider_margin

        slider_path = QPainterPath()
        slider_rect = QRectF(slider_x, slider_y, slider_diameter, slider_diameter)
        slider_path.addEllipse(slider_rect)

        slider_color = self._interp_color(
            QColor(self._slider_off_color), QColor(self._slider_on_color), self._slider_pos
        )
        painter.fillPath(slider_path, QBrush(slider_color))

        # SVG icon inside the knob — pick whichever renderer corresponds to
        # the "dominant" state (crossfade across the animation).
        active_renderer = self._on_icon_renderer if self._slider_pos >= 0.5 else self._off_icon_renderer
        if active_renderer is not None and active_renderer.isValid():
            pad = 3.0
            icon_rect = QRectF(
                slider_x + pad,
                slider_y + pad,
                slider_diameter - 2 * pad,
                slider_diameter - 2 * pad,
            )
            active_renderer.render(painter, icon_rect)

    def refresh_style(self):
        self._refresh_colors()
        self._reload_icons()
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
        # Track (pill) background — must contrast against MainRow.
        track_bg = get_color("button_bg", "#2a2c33")
        border_color = get_color("border", "#475569")
        base_text = get_color("text_secondary", "#9ca3af")
        tab_hover_bg = get_color("tab_bg_hover", get_color("hover_bg", "#3d4f63"))
        tab_hover_text = get_color("tab_text_hover", get_color("text_primary", "#e2e8f0"))
        mission_checked_bg  = get_color("tab_bg_checked")
        training_checked_bg = get_color("tab_bg_training")
        tab_checked_text = get_color("tab_text_checked", "#ffffff")
        r = 6
        self.setStyleSheet(
            f"""
            #pageSwitcher {{
                background-color: {track_bg};
                border: 1px solid {border_color};
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


class SidebarPageToggleButton(QPushButton):
    """Single-button Mission/Training toggle used at the top of the sidebar rail.

    Replaces the two-button PageSwitcher with a form-shifting button: clicking
    it toggles the current page and re-styles itself using the same color slots
    as PageSwitcher (``tab_bg_checked`` for Mission, ``tab_bg_training`` for
    Training).
    """

    page_selected = Signal(str)
    page_mode_changed = Signal(str)

    def __init__(self, parent=None, button_size: int = 55):
        super().__init__(parent)
        self.setObjectName("sidebarPageToggleButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(button_size, button_size)
        self._current_page = "mission"
        self._sound_effect: Optional[QSoundEffect] = None
        self.clicked.connect(self._on_click)
        self.apply_theme()

    def current_page(self) -> str:
        return self._current_page

    def set_current_page(self, page: str) -> None:
        target = "training" if str(page or "mission").strip().lower() == "training" else "mission"
        self._current_page = target
        self.apply_theme()
        self.page_mode_changed.emit(target)

    def _on_click(self) -> None:
        next_page = "training" if self._current_page == "mission" else "mission"
        self._current_page = next_page
        self._play_pick_sound()
        self.apply_theme()
        self.page_mode_changed.emit(next_page)
        self.page_selected.emit(next_page)

    def apply_theme(self) -> None:
        text = get_color("tab_text_checked", "#ffffff")
        if self._current_page == "training":
            bg = get_color("tab_bg_training")
            hover = get_color("tab_bg_hover", get_color("hover_bg", "#3d4f63"))
            label = "TRN"
        else:
            bg = get_color("tab_bg_checked")
            hover = get_color("tab_bg_hover", get_color("hover_bg", "#3d4f63"))
            label = "MIS"
        self.setText(label)
        self.setToolTip(
            "Mission Control — click to switch to Training"
            if self._current_page == "mission"
            else "Training Ground — click to switch to Mission"
        )
        self.setStyleSheet(f"""
            QPushButton#sidebarPageToggleButton {{
                background-color: {bg};
                color: {text};
                border: none;
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 1px;
            }}
            QPushButton#sidebarPageToggleButton:hover {{
                background-color: {hover};
            }}
        """)

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


class MainRow(QWidget):
    """Full-width header row spanning both the workspace and CMD columns.

    Layout (left → right):
        (10px) [PageSwitcher] (5px) | (5px) [FilesRow] (10px) [utility cluster] (20px) [window controls]

    The FilesRow is embedded with zero vertical margin so it shares the
    exact same height as MainRow (48 px).
    """

    _OBJECT_NAME = "mainRowHeader"

    _GAP_PX = 20          # gap between utility cluster and window controls
    _UTIL_BTN_SIZE = 32   # size of circular utility buttons in the cluster
    _LANG_COMBO_WIDTH = 63  # ~30% narrower than the previous default sizeHint
    _THEME_SWITCH_HEIGHT = _UTIL_BTN_SIZE - 5

    user_clicked = Signal()
    page_selected = Signal(str)      # forwarded from embedded PageSwitcher
    theme_switched = Signal(bool)    # True = dark, False = light
    language_changed = Signal(str)   # emits language code ("en", "zh", …)

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = (theme or "dark").lower()
        self._current_page = "mission"
        self.setObjectName(self._OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 10, 0)
        self._layout.setSpacing(0)

        # ── Left zone: PageSwitcher ───────────────────────────────────
        self._layout.addSpacing(10)
        self.page_switcher = PageSwitcher(parent=self)
        self.page_switcher.page_selected.connect(self.page_selected.emit)
        self._layout.addWidget(self.page_switcher, 0, Qt.AlignmentFlag.AlignVCenter)

        # 5px + 1px separator + 5px
        self._layout.addSpacing(5)
        self._separator = QWidget(self)
        self._separator.setFixedWidth(1)
        self._separator.setFixedHeight(28)
        self._layout.addWidget(self._separator, 0, Qt.AlignmentFlag.AlignVCenter)
        self._layout.addSpacing(5)

        # ── FilesRow slot — caller sets via set_files_row() ───────────
        # Placeholder stretches until the real FilesRow is injected.
        self._files_row: Optional[QWidget] = None
        self._files_row_stretch_index = self._layout.count()
        self._layout.addStretch(1)

        # ── Utility cluster: Theme switch / Language ──────────────────
        self._layout.addSpacing(10)
        self._util_host = QWidget(self)
        self._util_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._util_host.setObjectName("mainRowUtilHost")
        util_layout = QHBoxLayout(self._util_host)
        util_layout.setContentsMargins(0, 0, 0, 0)
        util_layout.setSpacing(10)

        self.theme_switch = ThemeSwitchButton(
            parent=self._util_host,
            checked=(self._theme == "dark"),
            height=self._THEME_SWITCH_HEIGHT,
        )
        self.theme_switch.setToolTip("Toggle theme")
        self.theme_switch.toggled.connect(self.theme_switched.emit)
        util_layout.addWidget(self.theme_switch, 0, Qt.AlignmentFlag.AlignVCenter)

        self.language_combo = QComboBox(self._util_host)
        self.language_combo.setObjectName("mainRowLanguageCombo")
        self.language_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.language_combo.addItem("EN", "en")
        self.language_combo.addItem("中文", "zh")
        self.language_combo.setFixedHeight(self._UTIL_BTN_SIZE)
        self.language_combo.setFixedWidth(self._LANG_COMBO_WIDTH)
        self.language_combo.currentIndexChanged.connect(self._emit_language)
        util_layout.addWidget(self.language_combo)

        self.user_button = None

        self._layout.addWidget(self._util_host, 0, Qt.AlignmentFlag.AlignVCenter)

        # 20 px gap between utility cluster and the window-controls host
        self._layout.addSpacing(self._GAP_PX)

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
        self._apply_util_styles()
        self._refresh_user_icon()

    # -- language / utility helpers --

    def _emit_language(self, _index: int) -> None:
        code = self.language_combo.currentData()
        if code:
            self.language_changed.emit(str(code))

    def set_language(self, code: str) -> None:
        """Programmatic language select (does not re-emit the signal loop)."""
        code = str(code or "").lower()
        idx = self.language_combo.findData(code)
        if idx >= 0:
            was_blocked = self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(idx)
            self.language_combo.blockSignals(was_blocked)

    # -- page mode --

    def set_current_page(self, page: str) -> None:
        """Update header background to match the active page."""
        page = "training" if str(page or "").strip().lower() == "training" else "mission"
        if page != self._current_page:
            self._current_page = page
            self.apply_page_bg()
            # Language combo bg tracks the row bg → re-stylize on page swap.
            self._apply_util_styles()

    def apply_page_bg(self) -> None:
        """Update MainRow background and separator to match the active page."""
        if self._current_page == "training":
            bg = get_color("main_row_training_bg")
        else:
            bg = get_color("main_row_bg")
        self.setStyleSheet(
            f"#{self._OBJECT_NAME} {{ background-color: {bg}; border: none; }}"
        )
        # Separator line colour
        sep_color = get_color("border", "#475569")
        if hasattr(self, "_separator") and self._separator is not None:
            self._separator.setStyleSheet(f"background-color: {sep_color};")

    # -- files row --

    def set_files_row(self, files_row: QWidget) -> None:
        """Embed *files_row* into MainRow between the separator and util cluster.

        The widget is inserted with zero vertical margin so it shares the
        exact 48 px height of MainRow.
        """
        if self._files_row is files_row:
            return
        if self._files_row is not None:
            self._layout.removeWidget(self._files_row)
            self._files_row.setParent(None)
        self._files_row = files_row
        if self._files_row is not None:
            # Remove the placeholder stretch and insert the real widget
            stretch_item = self._layout.itemAt(self._files_row_stretch_index)
            if stretch_item is not None and stretch_item.spacerItem() is not None:
                self._layout.removeItem(stretch_item)
            self._layout.insertWidget(self._files_row_stretch_index, self._files_row, 1)

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
        self._apply_util_styles()
        if hasattr(self, "theme_switch"):
            self.theme_switch.refresh_style()
        if hasattr(self, "page_switcher"):
            self.page_switcher.apply_theme()
        self._refresh_user_icon()

    def _refresh_user_icon(self) -> None:
        # User button has been removed from the MainRow layout. Method kept
        # as a no-op so external callers / theme refresh paths don't break.
        if self.user_button is None:
            return
        from src.system.core.theme_manager import get_icon
        icon = get_icon("acc")
        if not icon.isNull():
            self.user_button.setIcon(icon)
            self.user_button.setIconSize(
                QSize(self._UTIL_BTN_SIZE - 10, self._UTIL_BTN_SIZE - 10)
            )
            self.user_button.setText("")
        else:
            self.user_button.setText("U")

    def set_theme_state(self, theme: str) -> None:
        """Sync the theme switch visual to an external theme change without
        re-emitting the toggled signal."""
        is_dark = str(theme or "dark").lower() == "dark"
        if hasattr(self, "theme_switch"):
            was_blocked = self.theme_switch.blockSignals(True)
            self.theme_switch.setChecked(is_dark, animated=True)
            self.theme_switch.blockSignals(was_blocked)

    def _apply_util_styles(self) -> None:
        # Language combo: bg matches the MainRow itself (so the closed combo
        # blends into the row), hover uses ``button_hover`` to match the
        # window-control buttons on the right edge of the row.
        if self._current_page == "training":
            row_bg = get_color("main_row_training_bg", get_color("main_row_bg", "#1f2937"))
        else:
            row_bg = get_color("main_row_bg", "#1f2937")
        hover = get_color("button_hover", get_color("hover_bg", "#3d3d3d"))
        popup_bg = get_color("button_bg", "#2a2a2a")
        text = get_color("text_primary", "#e2e8f0")
        border = get_color("border", "#475569")
        self._util_host.setStyleSheet(f"""
            #mainRowUtilHost {{
                background: transparent;
                border: none;
            }}
            QComboBox#mainRowLanguageCombo {{
                background-color: {row_bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 2px 6px 2px 8px;
                font-weight: 600;
            }}
            QComboBox#mainRowLanguageCombo:hover {{
                background-color: {hover};
            }}
            QComboBox#mainRowLanguageCombo::drop-down {{
                border: none;
                width: 16px;
            }}
            QComboBox#mainRowLanguageCombo QAbstractItemView {{
                background-color: {popup_bg};
                color: {text};
                selection-background-color: {hover};
                border: 1px solid {border};
            }}
        """)
