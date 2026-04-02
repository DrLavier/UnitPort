from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal, QEasingCurve, QPropertyAnimation, Property, QRectF
from PySide6.QtGui import QFont, QFontMetrics, QColor, QPainter, QPainterPath, QBrush
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QSizePolicy, QWidget

from src.system.core.theme_manager import get_color


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


class MainRow(QWidget):
    """Top navigation row — hosts window controls (right side).

    PageSwitcher has moved to ControlPanel; this row now only provides
    the header background and a slot for the right-side widget.
    """

    _OBJECT_NAME = "mainRowHeader"

    def __init__(self, theme: str = "dark", parent=None):
        super().__init__(parent)
        self._theme = (theme or "dark").lower()
        self._current_page = "mission"
        self.setObjectName(self._OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 4, 10, 4)
        self._layout.setSpacing(10)

        self._layout.addStretch(1)

        self._right_host = QWidget(self)
        self._right_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._right_host.setStyleSheet("background: transparent; border: none;")
        self._right_host_layout = QHBoxLayout(self._right_host)
        self._right_host_layout.setContentsMargins(0, 0, 0, 0)
        self._right_host_layout.setSpacing(0)
        self._right_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._layout.addWidget(self._right_host, 0)

        self._right_widget: Optional[QWidget] = None
        self.apply_page_bg()

    def set_current_page(self, page: str) -> None:
        """Update header background color to match the active page."""
        page = "training" if str(page or "").strip().lower() == "training" else "mission"
        if page != self._current_page:
            self._current_page = page
            self.apply_page_bg()

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

    def set_right_widget(self, widget: QWidget) -> None:
        if self._right_widget is widget:
            return
        if self._right_widget is not None:
            self._right_host_layout.removeWidget(self._right_widget)
            self._right_widget.setParent(None)
        self._right_widget = widget
        if self._right_widget is not None:
            self._right_host_layout.addWidget(self._right_widget)

    def apply_theme(self) -> None:
        self.apply_page_bg()
