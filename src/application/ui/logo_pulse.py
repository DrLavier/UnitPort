"""Lightweight pulsing-logo widget for the startup loading overlay.

Renders a static SVG once and modulates a ``QGraphicsOpacityEffect`` between
0.9 and 1.0 along a sine wave whose full peak-to-peak period is 2 seconds.
No background thread, no frame cache, no rendering loop -- just one timer
driving a single float.

PyQt6 port of ``DEMO/bin/widgets/logo_pulse.py`` (originally PySide6).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import QWidget, QGraphicsOpacityEffect


class LogoPulse(QWidget):
    """Static SVG with a slow opacity pulse (0.9-1.0, 2 s peak-to-peak)."""

    _MIN_OPACITY = 0.9
    _MAX_OPACITY = 1.0
    _PERIOD_S = 2.0
    _UPDATE_MS = 50  # 20 Hz is plenty for a 0.1-amplitude pulse

    def __init__(
        self,
        svg_path: str,
        size: QSize,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("startupLogoPulse")
        self.setFixedSize(size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._svg = QSvgWidget(svg_path, self)
        self._svg.setFixedSize(size)
        self._svg.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._effect = QGraphicsOpacityEffect(self._svg)
        self._effect.setOpacity(self._MAX_OPACITY)
        self._svg.setGraphicsEffect(self._effect)

        self._t0 = time.monotonic()
        self._playing = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_opacity)
        self._timer.start(self._UPDATE_MS)

    def _update_opacity(self) -> None:
        if not self._playing:
            return
        amp = (self._MAX_OPACITY - self._MIN_OPACITY) / 2.0
        mid = (self._MAX_OPACITY + self._MIN_OPACITY) / 2.0
        phase = 2.0 * math.pi * (time.monotonic() - self._t0) / self._PERIOD_S
        self._effect.setOpacity(mid + amp * math.sin(phase))

    def tick(self) -> None:
        """Drive the pulse forward when the GUI event loop is busy."""
        self._update_opacity()

    def stop(self) -> None:
        self._playing = False
        self._timer.stop()
        self._effect.setOpacity(self._MAX_OPACITY)
        # Detach the child opacity effect so a parent-level
        # QGraphicsOpacityEffect (used by MainWindow to fade the loading
        # page) can be applied without nesting -- nested effects trigger
        # Qt's "QPainter: paint device painted by one painter at a
        # time" warning.
        self._svg.setGraphicsEffect(None)
