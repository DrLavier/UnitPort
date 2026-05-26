# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Cached QPixmap grid background / 缓存 QPixmap 的栅格背景.

DEMO 的 ``GraphScene.drawBackground()`` 每帧重画整片栅格，无缓存。
本模块改为：栅格画到 ``QPixmap`` 一次，以 ``QBrush(pixmap)`` 平铺到 scene rect。
仅在 ``(bg/small/big_color)`` 或 ``(small_step/big_step)`` 变化时重新生成。
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap


class CachedGrid:
    """栅格 ``QBrush`` 缓存 / Cached grid brush.

    使用：
        grid = CachedGrid()
        grid.set_colors(bg, small, big)
        painter.fillRect(rect, grid.brush())
    """

    def __init__(self, small_step: int = 20, big_step: int = 100) -> None:
        self.small_step = max(1, int(small_step))
        self.big_step = max(self.small_step, int(big_step))
        self.bg_color = QColor("#1E1E1E")
        self.small_color = QColor("#2A2A2A")
        self.big_color = QColor("#3d3d3d")
        self._big_only: bool = False
        self._brush: Optional[QBrush] = None
        self._dirty = True

    # ---- 配置 ----

    def set_colors(self, bg: QColor, small: QColor, big: QColor) -> None:
        """三色变更则置脏；颜色相同则无操作 / Mark dirty if any color changed."""
        if self.bg_color == bg and self.small_color == small and self.big_color == big:
            return
        self.bg_color = QColor(bg)
        self.small_color = QColor(small)
        self.big_color = QColor(big)
        self._dirty = True

    def set_steps(self, small_step: int, big_step: int) -> None:
        small_step = max(1, int(small_step))
        big_step = max(small_step, int(big_step))
        if small_step == self.small_step and big_step == self.big_step:
            return
        self.small_step = small_step
        self.big_step = big_step
        self._dirty = True

    def set_visible_tiers(self, big_only: bool) -> None:
        """LOD T0 时关掉小格 / At T0 LOD, suppress the small-step grid lines.

        在远缩放（zoom < 0.3）下，20px 的小格会糊成一片噪点；只画 100px 大格
        反而干净。CanvasPage 监听 ``view.zoomChanged`` 在 T0/T1 边界切换。
        """
        if big_only == self._big_only:
            return
        self._big_only = big_only
        self._dirty = True

    # ---- 输出 ----

    def brush(self) -> QBrush:
        """获取 brush；脏则先重新生成 / Return brush, regenerating if dirty."""
        if self._dirty or self._brush is None:
            self._regenerate()
        assert self._brush is not None  # for type checker
        return self._brush

    # ---- 内部 ----

    def _regenerate(self) -> None:
        size = self.big_step
        pm = QPixmap(size, size)
        pm.fill(self.bg_color)

        painter = QPainter(pm)
        try:
            if not self._big_only:
                small_pen = QPen(self.small_color)
                small_pen.setWidthF(0.5)
                painter.setPen(small_pen)
                i = self.small_step
                while i < size:
                    painter.drawLine(i, 0, i, size)
                    painter.drawLine(0, i, size, i)
                    i += self.small_step

            big_pen = QPen(self.big_color)
            big_pen.setWidthF(1.0)
            painter.setPen(big_pen)
            painter.drawLine(0, 0, 0, size)
            painter.drawLine(0, 0, size, 0)
        finally:
            painter.end()

        self._brush = QBrush(pm)
        self._dirty = False


__all__ = ["CachedGrid"]
