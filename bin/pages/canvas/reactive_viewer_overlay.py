"""Reactive viewer overlay — HUD for real-time command visualization.

Displays velocity command arrows, yaw indicator, input source badge,
and live command values when a ReactiveLoop is active.

This is a transparent PySide6 widget intended to be overlaid on top of
the MuJoCo viewer or any simulation viewport.

Usage::

    overlay = ReactiveViewerOverlay(parent_widget)
    overlay.update_commands(vx=0.5, vy=0.0, vyaw=-0.3)
    overlay.set_input_source("gamepad")
    overlay.set_active(True)
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
from PySide6.QtWidgets import QWidget


class ReactiveViewerOverlay(QWidget):
    """Transparent overlay showing real-time reactive control state."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        self._active = False
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._input_source = "gamepad"
        self._gait_state = ""

        # Colors
        self._bg_color = QColor(0, 0, 0, 120)
        self._arrow_color = QColor(0, 200, 100, 220)
        self._text_color = QColor(255, 255, 255, 200)
        self._yaw_color = QColor(100, 180, 255, 200)
        self._badge_color = QColor(60, 60, 60, 180)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Show/hide the overlay."""
        self._active = active
        self.setVisible(active)
        self.update()

    def update_commands(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Update displayed command values."""
        self._vx = vx
        self._vy = vy
        self._vyaw = vyaw
        if self._active:
            self.update()

    def set_input_source(self, source: str) -> None:
        """Set input source badge text."""
        self._input_source = source
        if self._active:
            self.update()

    def set_gait_state(self, state: str) -> None:
        """Set gait state indicator (standing/walking/trotting)."""
        self._gait_state = state
        if self._active:
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # --- Command display panel (bottom-left) ---
        panel_w = 200
        panel_h = 140
        panel_x = 10
        panel_y = h - panel_h - 10

        # Panel background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._bg_color))
        painter.drawRoundedRect(panel_x, panel_y, panel_w, panel_h, 8, 8)

        # Title
        font = QFont("Consolas", 10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(self._text_color))
        painter.drawText(panel_x + 8, panel_y + 20, "REACTIVE LIVE")

        # Command values
        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        y_offset = panel_y + 40
        for label, val in [("vx", self._vx), ("vy", self._vy), ("vyaw", self._vyaw)]:
            # Bar background
            bar_x = panel_x + 50
            bar_w = 130
            bar_h = 14
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(40, 40, 40, 180)))
            painter.drawRoundedRect(bar_x, y_offset, bar_w, bar_h, 3, 3)

            # Bar fill
            normalized = max(-1.0, min(1.0, val))
            fill_w = int(abs(normalized) * bar_w / 2)
            fill_x = bar_x + bar_w // 2
            if normalized < 0:
                fill_x = fill_x - fill_w
            color = self._arrow_color if label != "vyaw" else self._yaw_color
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(fill_x, y_offset, fill_w, bar_h, 3, 3)

            # Label
            painter.setPen(QPen(self._text_color))
            painter.drawText(panel_x + 8, y_offset + 11, f"{label}")
            painter.drawText(bar_x + bar_w + 5, y_offset + 11, f"{val:+.2f}")

            y_offset += 26

        # --- Input source badge (bottom-left, above panel) ---
        badge_text = f"[{self._input_source.upper()}]"
        if self._gait_state:
            badge_text += f"  {self._gait_state}"
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._badge_color))
        badge_w = max(100, len(badge_text) * 9 + 16)
        badge_y = panel_y - 30
        painter.drawRoundedRect(panel_x, badge_y, badge_w, 24, 6, 6)
        painter.setPen(QPen(self._text_color))
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(panel_x + 8, badge_y + 16, badge_text)

        # --- Direction arrow (bottom-right) ---
        arrow_cx = w - 80
        arrow_cy = h - 80
        arrow_r = 50

        # Circle background
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(0, 0, 0, 80)))
        painter.drawEllipse(QPointF(arrow_cx, arrow_cy), arrow_r, arrow_r)

        # Velocity arrow (vx, vy)
        mag = math.sqrt(self._vx ** 2 + self._vy ** 2)
        if mag > 0.01:
            angle = math.atan2(-self._vy, self._vx)  # screen coords
            arrow_len = min(mag, 1.0) * (arrow_r - 5)
            end_x = arrow_cx + arrow_len * math.cos(angle)
            end_y = arrow_cy - arrow_len * math.sin(angle)

            pen = QPen(self._arrow_color, 3)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(arrow_cx, arrow_cy), QPointF(end_x, end_y))

            # Arrowhead
            head_len = 8
            for da in [2.5, -2.5]:
                hx = end_x - head_len * math.cos(angle + da)
                hy = end_y + head_len * math.sin(angle + da)
                painter.drawLine(QPointF(end_x, end_y), QPointF(hx, hy))

        # Yaw arc
        if abs(self._vyaw) > 0.01:
            pen = QPen(self._yaw_color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            span = int(min(abs(self._vyaw), 1.0) * 180)
            start_angle = 90 * 16 if self._vyaw > 0 else (90 - span) * 16
            arc_rect = QRectF(arrow_cx - arrow_r, arrow_cy - arrow_r,
                              arrow_r * 2, arrow_r * 2)
            painter.drawArc(arc_rect, start_angle, span * 16 * (1 if self._vyaw > 0 else -1))

        painter.end()
