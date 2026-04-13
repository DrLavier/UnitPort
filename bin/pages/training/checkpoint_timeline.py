#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckpointTimeline — horizontal timeline strip showing training checkpoints.

Phase A: renders with mock data, no real filesystem watcher.
Each circular node represents a saved checkpoint with iteration number
and reward value.  Clicking emits preview_requested(path).
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget


class _Checkpoint:
    """Internal data for a single checkpoint node."""

    __slots__ = ("iteration", "reward", "path", "timestamp")

    def __init__(self, iteration: int, reward: float, path: str, timestamp: str = ""):
        self.iteration = iteration
        self.reward = reward
        self.path = path
        self.timestamp = timestamp


class CheckpointTimeline(QWidget):
    """Horizontal timeline strip with circular checkpoint nodes.

    Signals
    -------
    preview_requested(str)
        Emits the checkpoint path when a node is clicked.
    """

    preview_requested = Signal(str)

    # Layout
    _NODE_RADIUS = 8
    _MARGIN_X = 24
    _MARGIN_Y = 14
    _LINE_Y_OFFSET = 20  # y-centre of the timeline rail
    _MAX_DISPLAY = 20

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("checkpointTimeline")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(70)
        self.setMouseTracking(True)

        self._checkpoints: List[_Checkpoint] = []
        self._active_index: int = -1  # currently previewing
        self._hover_index: int = -1

        # Populate mock data for Phase A
        self._populate_mock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_checkpoint(self, iteration: int, reward: float, path: str) -> None:
        """Add a new checkpoint to the timeline."""
        self._checkpoints.append(_Checkpoint(iteration, reward, path))
        self.update()

    def clear(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()
        self._active_index = -1
        self._hover_index = -1
        self.update()

    def set_active(self, index: int) -> None:
        """Highlight the checkpoint currently being previewed."""
        self._active_index = index
        self.update()

    # ------------------------------------------------------------------
    # Mock data (Phase A)
    # ------------------------------------------------------------------

    def _populate_mock(self) -> None:
        mock = [
            (100,  5.2,  "mock/checkpoints/model_100.pt"),
            (200,  12.8, "mock/checkpoints/model_200.pt"),
            (400,  18.3, "mock/checkpoints/model_400.pt"),
            (700,  22.1, "mock/checkpoints/model_700.pt"),
            (1000, 24.6, "mock/checkpoints/model_1000.pt"),
            (1300, 25.9, "mock/checkpoints/model_1300.pt"),
            (1500, 26.4, "mock/checkpoints/model_1500.pt"),
        ]
        for it, rew, path in mock:
            self._checkpoints.append(_Checkpoint(it, rew, path))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _visible_checkpoints(self) -> List[_Checkpoint]:
        cps = self._checkpoints
        if len(cps) <= self._MAX_DISPLAY:
            return list(cps)
        step = max(1, len(cps) // self._MAX_DISPLAY)
        return [cps[i] for i in range(0, len(cps), step)]

    def _node_positions(self) -> List[QPointF]:
        """Return centre positions for each visible checkpoint node."""
        vis = self._visible_checkpoints()
        if not vis:
            return []
        w = self.width() - 2 * self._MARGIN_X
        y = self._LINE_Y_OFFSET + self._MARGIN_Y
        if len(vis) == 1:
            return [QPointF(self._MARGIN_X + w / 2, y)]
        positions = []
        for i in range(len(vis)):
            x = self._MARGIN_X + w * i / (len(vis) - 1)
            positions.append(QPointF(x, y))
        return positions

    def _hit_test(self, pos) -> int:
        """Return the index in _visible_checkpoints() under pos, or -1."""
        positions = self._node_positions()
        r2 = (self._NODE_RADIUS + 4) ** 2
        for i, p in enumerate(positions):
            dx = pos.x() - p.x()
            dy = pos.y() - p.y()
            if dx * dx + dy * dy <= r2:
                return i
        return -1

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        vis = self._visible_checkpoints()
        positions = self._node_positions()
        if not positions:
            painter.setPen(QPen(QColor("#555"), 1))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No checkpoints yet")
            painter.end()
            return

        # Rail line
        y = positions[0].y()
        painter.setPen(QPen(QColor("#444"), 2))
        painter.drawLine(
            QPointF(positions[0].x(), y),
            QPointF(positions[-1].x(), y),
        )

        r = self._NODE_RADIUS
        font_small = QFont("Arial", 7)
        font_label = QFont("Arial", 8)

        for i, (cp, pos) in enumerate(zip(vis, positions)):
            # Map back to original index for active highlight
            orig_idx = self._checkpoints.index(cp) if cp in self._checkpoints else -1
            is_active = orig_idx == self._active_index
            is_hover = i == self._hover_index

            # Node circle
            if is_active:
                painter.setPen(QPen(QColor("#4FC3F7"), 3))
                painter.setBrush(QBrush(QColor("#0288D1")))
            elif is_hover:
                painter.setPen(QPen(QColor("#81D4FA"), 2))
                painter.setBrush(QBrush(QColor("#01579B")))
            else:
                painter.setPen(QPen(QColor("#78909C"), 1.5))
                painter.setBrush(QBrush(QColor("#263238")))

            painter.drawEllipse(pos, r, r)

            # Reward above
            painter.setPen(QPen(QColor("#B0BEC5"), 1))
            painter.setFont(font_small)
            painter.drawText(
                QRectF(pos.x() - 25, pos.y() - r - 14, 50, 12),
                Qt.AlignmentFlag.AlignCenter,
                f"{cp.reward:.1f}",
            )

            # Iteration below
            painter.setFont(font_label)
            painter.drawText(
                QRectF(pos.x() - 25, pos.y() + r + 2, 50, 14),
                Qt.AlignmentFlag.AlignCenter,
                f"{cp.iteration}",
            )

        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_test(event.position())
            if idx >= 0:
                vis = self._visible_checkpoints()
                cp = vis[idx]
                orig = self._checkpoints.index(cp) if cp in self._checkpoints else -1
                self._active_index = orig
                self.update()
                self.preview_requested.emit(cp.path)
                from src.system.core.logger import log_info
                log_info(f"preview_requested: {cp.path}")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        idx = self._hit_test(event.position())
        if idx != self._hover_index:
            self._hover_index = idx
            self.update()
        if idx >= 0:
            vis = self._visible_checkpoints()
            cp = vis[idx]
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Iteration {cp.iteration}\nReward: {cp.reward:.3f}\nPath: {cp.path}",
                self,
            )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_index = -1
        self.update()
        super().leaveEvent(event)
