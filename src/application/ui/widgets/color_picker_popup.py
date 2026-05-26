# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Frameless RGB / HSV color picker popup for chart line overrides.

Used by ``TrainingChartPanel`` — clicking a per-series palette swatch
opens this popup, the user picks a color, and on confirm the chart panel
persists the choice via ``Config.set_value("Theme", slot, hex_)``.

Layout (Photoshop style):

    +-----------------+ +-+
    |                 | |H|
    |   SV pad        | |u|
    |   200x200       | |e|
    |                 | |
    +-----------------+ +-+
    [#RRGGBB     ] [yes][no]

Behaviour:
  - Qt.WindowType.Popup → clicks outside auto-close (== cancel)
  - SV pad / Hue slider / hex input three-way live sync
  - color_changed(hex) fires on every interactive change (live preview
    on the chart)
  - accepted_color(hex) fires only when the user clicks the yes button
  - cancelled() fires when the popup closes without confirm
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRegularExpressionValidator,
)
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Assets, Config, tr


# ---------------------------------------------------------------------------
# Hex / QColor helpers
# ---------------------------------------------------------------------------

def _normalize_hex(s: str) -> Optional[str]:
    """Return canonical ``#RRGGBB`` (upper) or None if not parseable."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return None
    try:
        int(s, 16)
    except ValueError:
        return None
    return "#" + s.upper()


def _qcolor_to_hex(c: QColor) -> str:
    return "#{:02X}{:02X}{:02X}".format(c.red(), c.green(), c.blue())


# ---------------------------------------------------------------------------
# _SVPad — saturation x value square
# ---------------------------------------------------------------------------

class _SVPad(QWidget):
    """200x200 square painting white->hue horizontally and transparent->black
    vertically. Click/drag updates (s, v) in 0..1.
    """

    sv_changed = pyqtSignal(float, float)  # s, v in 0..1

    _SIZE = 200
    _MARKER_R = 6

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self._hue = 0.0
        self._s = 1.0
        self._v = 1.0
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_hue(self, hue: float) -> None:
        self._hue = max(0.0, min(1.0, float(hue)))
        self.update()

    def set_sv(self, s: float, v: float) -> None:
        self._s = max(0.0, min(1.0, float(s)))
        self._v = max(0.0, min(1.0, float(v)))
        self.update()

    def current_sv(self) -> tuple:
        return (self._s, self._v)

    # ---- paint -----------------------------------------------------------
    def paintEvent(self, _e: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        # Layer 1: white -> pure hue, horizontal
        pure = QColor.fromHsvF(self._hue, 1.0, 1.0)
        g1 = QLinearGradient(0, 0, w, 0)
        g1.setColorAt(0.0, QColor(255, 255, 255))
        g1.setColorAt(1.0, pure)
        p.fillRect(rect, g1)

        # Layer 2: transparent -> black, vertical
        g2 = QLinearGradient(0, 0, 0, h)
        g2.setColorAt(0.0, QColor(0, 0, 0, 0))
        g2.setColorAt(1.0, QColor(0, 0, 0, 255))
        p.fillRect(rect, g2)

        # Marker (ring) at (s*w, (1-v)*h)
        mx = int(round(self._s * (w - 1)))
        my = int(round((1.0 - self._v) * (h - 1)))
        outer = QColor(Config.get_color("main_t2"))
        inner = QColor(Config.get_color("picker_marker_inner"))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(outer, 2.0)
        p.setPen(pen)
        p.drawEllipse(QRect(mx - self._MARKER_R, my - self._MARKER_R,
                            self._MARKER_R * 2, self._MARKER_R * 2))
        pen2 = QPen(inner, 1.0)
        p.setPen(pen2)
        p.drawEllipse(QRect(mx - self._MARKER_R + 1, my - self._MARKER_R + 1,
                            (self._MARKER_R - 1) * 2, (self._MARKER_R - 1) * 2))

    # ---- input -----------------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._update_from_pos(e.position().x(), e.position().y())
            e.accept()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._update_from_pos(e.position().x(), e.position().y())
            e.accept()

    def _update_from_pos(self, x: float, y: float) -> None:
        w = max(1, self.width() - 1)
        h = max(1, self.height() - 1)
        s = max(0.0, min(1.0, x / w))
        v = max(0.0, min(1.0, 1.0 - y / h))
        if (s, v) != (self._s, self._v):
            self._s = s
            self._v = v
            self.update()
            self.sv_changed.emit(s, v)


# ---------------------------------------------------------------------------
# _HueSlider — vertical hue bar
# ---------------------------------------------------------------------------

class _HueSlider(QWidget):
    """Vertical 16x200 hue slider. Top = red (h=0), bottom = red (h=1)
    via the standard 6-stop hue cycle.
    """

    hue_changed = pyqtSignal(float)  # 0..1

    _W = 18
    _H = 200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._W, self._H)
        self._hue = 0.0
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def set_hue(self, hue: float) -> None:
        self._hue = max(0.0, min(1.0, float(hue)))
        self.update()

    def current_hue(self) -> float:
        return self._hue

    def paintEvent(self, _e: QPaintEvent) -> None:
        p = QPainter(self)
        rect = self.rect()
        w, h = rect.width(), rect.height()

        g = QLinearGradient(0, 0, 0, h)
        # Six-stop hue cycle: r y g c b m r
        g.setColorAt(0.000, QColor(255, 0, 0))
        g.setColorAt(0.167, QColor(255, 255, 0))
        g.setColorAt(0.333, QColor(0, 255, 0))
        g.setColorAt(0.500, QColor(0, 255, 255))
        g.setColorAt(0.667, QColor(0, 0, 255))
        g.setColorAt(0.833, QColor(255, 0, 255))
        g.setColorAt(1.000, QColor(255, 0, 0))
        p.fillRect(rect, g)

        # Marker: short horizontal line spanning the slider at hue*h
        y = int(round(self._hue * (h - 1)))
        outer = QColor(Config.get_color("main_t2"))
        inner = QColor(Config.get_color("picker_marker_inner"))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(outer, 2.0))
        p.drawLine(0, y, w - 1, y)
        p.setPen(QPen(inner, 1.0))
        p.drawLine(0, y - 1, w - 1, y - 1)
        p.drawLine(0, y + 1, w - 1, y + 1)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._update_from_y(e.position().y())
            e.accept()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._update_from_y(e.position().y())
            e.accept()

    def _update_from_y(self, y: float) -> None:
        h = max(1, self.height() - 1)
        new = max(0.0, min(1.0, y / h))
        if new != self._hue:
            self._hue = new
            self.update()
            self.hue_changed.emit(new)


# ---------------------------------------------------------------------------
# ColorPickerPopup
# ---------------------------------------------------------------------------

class ColorPickerPopup(QFrame):
    """Frameless RGB picker. See module docstring for layout / behaviour."""

    color_changed = pyqtSignal(str)
    accepted_color = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, initial_hex: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("colorPickerPopup")

        self._confirmed = False
        self._suppress_hex_emit = False

        normalized = _normalize_hex(initial_hex) or Config.get_color(
            "chart_line_default"
        )
        c0 = QColor(normalized)
        h0, s0, v0, _ = c0.getHsvF()
        if h0 < 0:  # achromatic
            h0 = 0.0

        self._build_ui()
        self._apply_theme()
        self._set_from_qcolor(c0, emit=False, sync_widgets=True)
        # Re-derive HSV again from the canonical hex so internal state is exact
        self._sv_pad.set_hue(h0)
        self._sv_pad.set_sv(s0, v0)
        self._hue_slider.set_hue(h0)

    # ---- construction ----------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- top: SV pad + hue slider ---
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self._sv_pad = _SVPad(self)
        self._sv_pad.sv_changed.connect(self._on_sv_changed)
        top.addWidget(self._sv_pad, 0)

        self._hue_slider = _HueSlider(self)
        self._hue_slider.hue_changed.connect(self._on_hue_changed)
        top.addWidget(self._hue_slider, 0)

        root.addLayout(top)

        # --- bottom: hex input + yes/no ---
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)

        self._hex_input = QLineEdit(self)
        self._hex_input.setObjectName("colorPickerHexInput")
        self._hex_input.setPlaceholderText(
            tr("missioncontrol.color.hex_input.placeholder")
        )
        self._hex_input.setMaxLength(7)
        rx = QRegularExpression(r"^#?[0-9A-Fa-f]{0,6}$")
        self._hex_input.setValidator(QRegularExpressionValidator(rx, self))
        self._hex_input.textEdited.connect(self._on_hex_edited)
        self._hex_input.editingFinished.connect(self._on_hex_finalised)
        bottom.addWidget(self._hex_input, 1)

        self._yes_btn = QToolButton(self)
        self._yes_btn.setObjectName("colorPickerYesBtn")
        self._yes_btn.setFixedSize(28, 28)
        self._yes_btn.setIconSize(QSize(18, 18))
        self._yes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._yes_btn.setToolTip(tr("missioncontrol.color.btn.confirm.tip"))
        self._yes_btn.clicked.connect(self._on_confirm)
        self._apply_icon(self._yes_btn, "icon_yes")
        bottom.addWidget(self._yes_btn, 0)

        self._no_btn = QToolButton(self)
        self._no_btn.setObjectName("colorPickerNoBtn")
        self._no_btn.setFixedSize(28, 28)
        self._no_btn.setIconSize(QSize(18, 18))
        self._no_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._no_btn.setToolTip(tr("missioncontrol.color.btn.cancel.tip"))
        self._no_btn.clicked.connect(self._on_cancel)
        self._apply_icon(self._no_btn, "icon_no")
        bottom.addWidget(self._no_btn, 0)

        root.addLayout(bottom)

    @staticmethod
    def _apply_icon(btn: QToolButton, stem: str) -> None:
        try:
            path = Assets.find_icon(stem)
        except Exception:
            path = None
        if path is not None:
            btn.setIcon(QIcon(str(path)))

    # ---- theme -----------------------------------------------------------
    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        border = Config.get_color("picker_popup_border")
        ie_bg = Config.get_color("picker_hex_input_bg")
        ie_border = Config.get_color("picker_hex_input_border")
        ie_focus = Config.get_color("checked_1")
        ie_text = Config.get_color("main_t1")
        btn_bg = Config.get_color("row_1")
        btn_bg_hover = Config.get_color("picker_btn_bg_hover")
        btn_border = Config.get_color("picker_btn_border")
        font_small = Config.get_font_size("size_small")

        self.setStyleSheet(
            f"QFrame#colorPickerPopup {{"
            f"  background: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 6px;"
            f"}}"
            f"QLineEdit#colorPickerHexInput {{"
            f"  background: {ie_bg};"
            f"  color: {ie_text};"
            f"  border: 1px solid {ie_border};"
            f"  border-radius: 3px;"
            f"  padding: 4px 6px;"
            f"  font-size: {font_small}px;"
            f"}}"
            f"QLineEdit#colorPickerHexInput:focus {{"
            f"  border: 1px solid {ie_focus};"
            f"}}"
            f"QToolButton#colorPickerYesBtn, QToolButton#colorPickerNoBtn {{"
            f"  background: {btn_bg};"
            f"  border: 1px solid {btn_border};"
            f"  border-radius: 4px;"
            f"}}"
            f"QToolButton#colorPickerYesBtn:hover, QToolButton#colorPickerNoBtn:hover {{"
            f"  background: {btn_bg_hover};"
            f"}}"
        )

    def _set_hex_input_invalid(self, invalid: bool) -> None:
        # Toggle border colour by overriding the input border slot inline.
        # Cheaper than rebuilding the full stylesheet on every keystroke.
        if invalid:
            border = Config.get_color("picker_hex_input_border_invalid")
        else:
            border = Config.get_color("picker_hex_input_border")
        focus = Config.get_color("checked_1")
        self._hex_input.setStyleSheet(
            f"QLineEdit#colorPickerHexInput {{ border: 1px solid {border}; }}"
            f"QLineEdit#colorPickerHexInput:focus {{ border: 1px solid {focus}; }}"
        )

    # ---- positioning -----------------------------------------------------
    def show_at(self, anchor_global: QPoint) -> None:
        """Show with top-left at *anchor_global*, clamped to screen."""
        self.adjustSize()
        self.move(anchor_global)
        self.show()
        screen = self.screen() if hasattr(self, "screen") else None
        if screen is not None:
            avail = screen.availableGeometry()
            x, y = self.x(), self.y()
            new_x, new_y = x, y
            if x + self.width() > avail.right():
                new_x = max(avail.left(), avail.right() - self.width() - 2)
            if y + self.height() > avail.bottom():
                new_y = max(avail.top(), avail.bottom() - self.height() - 2)
            if x < avail.left():
                new_x = avail.left() + 2
            if y < avail.top():
                new_y = avail.top() + 2
            if (new_x, new_y) != (x, y):
                self.move(new_x, new_y)

    # ---- internal sync ---------------------------------------------------
    def _current_color(self) -> QColor:
        h = self._hue_slider.current_hue()
        s, v = self._sv_pad.current_sv()
        return QColor.fromHsvF(h, s, v)

    def _set_from_qcolor(self, c: QColor, *, emit: bool,
                         sync_widgets: bool = True) -> None:
        if sync_widgets:
            self._suppress_hex_emit = True
            self._hex_input.setText(_qcolor_to_hex(c))
            self._suppress_hex_emit = False
            self._set_hex_input_invalid(False)
        if emit:
            self.color_changed.emit(_qcolor_to_hex(c))

    def _on_sv_changed(self, s: float, v: float) -> None:
        c = QColor.fromHsvF(self._hue_slider.current_hue(), s, v)
        self._set_from_qcolor(c, emit=True)

    def _on_hue_changed(self, h: float) -> None:
        self._sv_pad.set_hue(h)
        s, v = self._sv_pad.current_sv()
        c = QColor.fromHsvF(h, s, v)
        self._set_from_qcolor(c, emit=True)

    def _on_hex_edited(self, text: str) -> None:
        if self._suppress_hex_emit:
            return
        normalized = _normalize_hex(text)
        if normalized is None:
            # Visual warning but no value push — wait for the user to finish.
            if len(text.strip().lstrip("#")) >= 6:
                self._set_hex_input_invalid(True)
            return
        self._set_hex_input_invalid(False)
        c = QColor(normalized)
        h, s, v, _ = c.getHsvF()
        if h < 0:
            h = self._hue_slider.current_hue()
        self._hue_slider.set_hue(h)
        self._sv_pad.set_hue(h)
        self._sv_pad.set_sv(s, v)
        self.color_changed.emit(normalized)

    def _on_hex_finalised(self) -> None:
        # On commit: if the field is invalid, snap back to current color.
        normalized = _normalize_hex(self._hex_input.text())
        if normalized is None:
            c = self._current_color()
            self._suppress_hex_emit = True
            self._hex_input.setText(_qcolor_to_hex(c))
            self._suppress_hex_emit = False
            self._set_hex_input_invalid(False)

    # ---- confirm / cancel ------------------------------------------------
    def _on_confirm(self) -> None:
        self._confirmed = True
        final = _normalize_hex(self._hex_input.text()) or _qcolor_to_hex(
            self._current_color()
        )
        self.accepted_color.emit(final)
        self.close()

    def _on_cancel(self) -> None:
        self.close()

    def closeEvent(self, e) -> None:  # noqa: D401 — Qt signature
        if not self._confirmed:
            self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e) -> None:  # noqa: D401 — Qt signature
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_confirm()
            return
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(e)


__all__ = ["ColorPickerPopup"]
