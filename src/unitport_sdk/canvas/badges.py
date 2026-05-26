# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.badges — Node 状态 / 极性徽章 widget.

DEMO 对应（``bin/nodes/training_node_items.py``）：
    _ModulePolarityBadge (888-930)  → Node_ModulePolarityBadge
    _ClipRefIndicator               → Node_ClipRefIndicator

视觉契约：
    18×18 状态徽章 —— 圆 / 方 / 菱形（draw_polarity_badge）+ 单击循环切换。
    Node_ClipRefIndicator 是 LaviLabel 包装的小角标 ``×N``。

注意：本文件只导出 ``Node_*`` 类，绝不允许 ``_`` 私有前缀。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from unitport_sdk.sys import Config
from unitport_sdk.ui import setText

from .badge import PolarityKind, draw_polarity_badge


# =============================================================================
# Node_ModulePolarityBadge —— 极性徽章
# =============================================================================

class Node_ModulePolarityBadge(QWidget):
    """模块极性徽章 / Module polarity badge.

    DEMO 行 888-930（``_ModulePolarityBadge``）。点击循环切换三态：
    reward → penalty → bidirectional → reward。

    信号：
        kindChanged(str) —— 当前 kind 变更
    """

    kindChanged = pyqtSignal(str)

    _CYCLE: List[PolarityKind] = ["reward", "penalty", "bidirectional"]

    def __init__(
        self,
        kind: PolarityKind = "reward",
        *,
        size: int = 18,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._kind: PolarityKind = kind if kind in self._CYCLE else "reward"
        self.setFixedSize(QSize(size, size))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def kind(self) -> PolarityKind:
        return self._kind

    def setKind(self, kind: PolarityKind) -> None:
        if kind in self._CYCLE and kind != self._kind:
            self._kind = kind
            self.update()
            self.kindChanged.emit(kind)

    def cycle(self) -> None:
        idx = self._CYCLE.index(self._kind)
        self.setKind(self._CYCLE[(idx + 1) % len(self._CYCLE)])

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.cycle()
            e.accept()
            return
        super().mousePressEvent(e)

    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        draw_polarity_badge(p, rect, kind=self._kind)
        p.end()

    def refresh_style(self) -> None:
        self.update()


# =============================================================================
# Node_ClipRefIndicator —— 引用计数小角标
# =============================================================================

class Node_ClipRefIndicator(QWidget):
    """clip 引用计数角标 / Reference-count indicator (e.g. ``×3``).

    用 ``LaviLabel``（``setText``）显示文字，外层套底色徽章。
    """

    def __init__(
        self,
        count: int = 0,
        *,
        widget_id: str = "node.clip_ref",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._count = max(0, int(count))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 6, 1)
        layout.setSpacing(0)
        self._label = setText(
            widget_id,
            default=self._format_text(),
            kind="caption",
        )
        layout.addWidget(self._label)
        self.refresh_style()

    def count(self) -> int:
        return self._count

    def setCount(self, n: int) -> None:
        self._count = max(0, int(n))
        self._label.setText(self._format_text())
        self.update()

    def _format_text(self) -> str:
        return f"×{self._count}" if self._count else ""

    def refresh_style(self) -> None:
        bg = Config.get_color("btn_1", "#2D2F33")
        fg = Config.get_color("main_t1", "#9CA3AF")
        self.setStyleSheet(
            f"Node_ClipRefIndicator {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border-radius: 7px;"
            f"}}"
        )


__all__ = ["Node_ModulePolarityBadge", "Node_ClipRefIndicator"]
