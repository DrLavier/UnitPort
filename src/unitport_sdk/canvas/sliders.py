"""unitport_sdk.canvas.sliders — Node 滑块家族 / Node-internal slider family.

DEMO 对应（``bin/nodes/training_node_items.py`` / ``node_ui_rows.py``）：
    _RangeHandleSlider     (1473-1628)  → Node_RangeHandleSlider
    _DualHandleRangeSlider                → Node_DualHandleRangeSlider
    _NodeSlider            (node_ui_rows) → Node_NodeSlider
    _ModuleCurveSlider                    → Node_ModuleCurveSlider

视觉契约：paint 全部走 SDK ``draw_range_slider`` + 命中走
``RangeSliderHitTest``；数值回显用 ``LaviDoubleSpinBox``，标签用 ``LaviLabel``
（``setText``）。

约束：所有 UI class 以 ``Node_`` 开头，禁止 ``_`` 私有前缀。
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPaintEvent
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from unitport_sdk.sys import Config
from unitport_sdk.ui import setText

from .range_slider import RangeSliderHitTest, draw_range_slider
from .widgets import LaviDoubleSpinBox, setDoubleSpinBox


# =============================================================================
# 几何常量
# =============================================================================

NODE_SLIDER_H = 18
NODE_SLIDER_MIN_W = 80


# =============================================================================
# Node_RangeHandleSlider —— 单 handle 滑块
# =============================================================================

class Node_RangeHandleSlider(QWidget):
    """单 handle 滑块 / Single-handle slider widget.

    DEMO 行 1473-1628（``_RangeHandleSlider``）。
    paint 调 ``draw_range_slider(dual=False)``，鼠标命中走 ``RangeSliderHitTest``.

    信号：
        valueChanged(float)
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        value: float = 0.0,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._vmin = float(minimum)
        self._vmax = float(maximum)
        self._value = self._clamp(value)
        self._dragging = False
        self._hover = False

        self.setMinimumHeight(NODE_SLIDER_H)
        self.setMinimumWidth(NODE_SLIDER_MIN_W)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ----- value api -------------------------------------------------
    def value(self) -> float:
        return self._value

    def setValue(self, v: float) -> None:
        v = self._clamp(v)
        if v != self._value:
            self._value = v
            self.update()
            self.valueChanged.emit(v)

    def setRange(self, lo: float, hi: float) -> None:
        if hi > lo:
            self._vmin = float(lo)
            self._vmax = float(hi)
            self._value = self._clamp(self._value)
            self.update()

    def _clamp(self, v: float) -> float:
        return max(self._vmin, min(self._vmax, float(v)))

    def _rect(self) -> QRectF:
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _hit(self) -> RangeSliderHitTest:
        return RangeSliderHitTest(rect=self._rect(), vmin=self._vmin, vmax=self._vmax, dual=False)

    # ----- mouse events ----------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            ht = self._hit()
            kind = ht.handle_at(QPointF(e.position()), self._value)
            if kind is not None:
                self._dragging = True
                self.setValue(ht.value_at(e.position().x()))
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging:
            self.setValue(self._hit().value_at(e.position().x()))
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def enterEvent(self, e):
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        super().leaveEvent(e)

    # ----- paint -----------------------------------------------------
    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        draw_range_slider(
            p,
            self._rect(),
            lo=self._value,
            vmin=self._vmin,
            vmax=self._vmax,
            dual=False,
            hover_handle="lo" if (self._hover or self._dragging) else None,
        )
        p.end()

    def refresh_style(self) -> None:
        self.update()


# =============================================================================
# Node_DualHandleRangeSlider —— 双 handle 滑块
# =============================================================================

class Node_DualHandleRangeSlider(QWidget):
    """区间滑块 / Dual-handle range slider.

    paint 调 ``draw_range_slider(dual=True)``。

    信号：
        rangeChanged(float, float)  —— (lo, hi)
    """

    rangeChanged = pyqtSignal(float, float)

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._vmin = float(minimum)
        self._vmax = float(maximum)
        self._lo = self._clamp(lo)
        self._hi = self._clamp(hi)
        if self._hi < self._lo:
            self._lo, self._hi = self._hi, self._lo
        self._dragging: Optional[str] = None  # "lo" / "hi"
        self._hover_handle: Optional[str] = None

        self.setMinimumHeight(NODE_SLIDER_H)
        self.setMinimumWidth(NODE_SLIDER_MIN_W)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def lo(self) -> float:
        return self._lo

    def hi(self) -> float:
        return self._hi

    def setRangeValues(self, lo: float, hi: float) -> None:
        new_lo = self._clamp(lo)
        new_hi = self._clamp(hi)
        if new_hi < new_lo:
            new_lo, new_hi = new_hi, new_lo
        if (new_lo, new_hi) != (self._lo, self._hi):
            self._lo = new_lo
            self._hi = new_hi
            self.update()
            self.rangeChanged.emit(new_lo, new_hi)

    def setRange(self, vmin: float, vmax: float) -> None:
        if vmax > vmin:
            self._vmin = float(vmin)
            self._vmax = float(vmax)
            self.setRangeValues(self._lo, self._hi)

    def _clamp(self, v: float) -> float:
        return max(self._vmin, min(self._vmax, float(v)))

    def _rect(self) -> QRectF:
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _hit(self) -> RangeSliderHitTest:
        return RangeSliderHitTest(rect=self._rect(), vmin=self._vmin, vmax=self._vmax, dual=True)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            ht = self._hit()
            kind = ht.handle_at(QPointF(e.position()), self._lo, self._hi)
            if kind in ("lo", "hi"):
                self._dragging = kind
                e.accept()
                return
            if kind == "track":
                # click-to-jump: 离哪个 handle 近就拖哪个
                v = ht.value_at(e.position().x())
                self._dragging = "lo" if abs(v - self._lo) < abs(v - self._hi) else "hi"
                self._set_handle(self._dragging, v)
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        v = self._hit().value_at(e.position().x())
        if self._dragging:
            self._set_handle(self._dragging, v)
            e.accept()
            return
        # hover detection
        kind = self._hit().handle_at(QPointF(e.position()), self._lo, self._hi)
        new_hover = kind if kind in ("lo", "hi") else None
        if new_hover != self._hover_handle:
            self._hover_handle = new_hover
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        self._hover_handle = None
        self.update()
        super().leaveEvent(e)

    def _set_handle(self, which: str, v: float) -> None:
        v = self._clamp(v)
        if which == "lo":
            v = min(v, self._hi)
            if v != self._lo:
                self._lo = v
                self.update()
                self.rangeChanged.emit(self._lo, self._hi)
        else:
            v = max(v, self._lo)
            if v != self._hi:
                self._hi = v
                self.update()
                self.rangeChanged.emit(self._lo, self._hi)

    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        draw_range_slider(
            p,
            self._rect(),
            lo=self._lo,
            hi=self._hi,
            vmin=self._vmin,
            vmax=self._vmax,
            dual=True,
            hover_handle=self._hover_handle or self._dragging,
        )
        p.end()

    def refresh_style(self) -> None:
        self.update()


# =============================================================================
# Node_NodeSlider —— LaviLabel + 滑块 + LaviDoubleSpinBox 复合行
# =============================================================================

class Node_NodeSlider(QWidget):
    """带标签 + 数值回显的滑块 / Slider with label and spin box.

    DEMO 行（``node_ui_rows.py``）的 ``_NodeSlider``。
    内部组合：
        左侧：LaviLabel（``setText``）
        中部：Node_RangeHandleSlider
        右侧：LaviDoubleSpinBox（``setDoubleSpinBox``）
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        value: float = 0.0,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        decimals: int = 3,
        widget_id: str = "node.slider",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = setText(f"{widget_id}.label", default=label, kind="caption")
        layout.addWidget(self._label)

        self._slider = Node_RangeHandleSlider(
            value=value, minimum=minimum, maximum=maximum, parent=self,
        )
        layout.addWidget(self._slider, 1)

        self._spin = setDoubleSpinBox(
            value, minimum=minimum, maximum=maximum, decimals=decimals, parent=self,
        )
        self._spin.setMaximumWidth(72)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def value(self) -> float:
        return self._slider.value()

    def setValue(self, v: float) -> None:
        self._slider.setValue(v)
        self._spin.setValue(self._slider.value())

    def _on_slider(self, v: float) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self.valueChanged.emit(v)

    def _on_spin(self, v: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(v)
        self._slider.blockSignals(False)
        self.valueChanged.emit(self._slider.value())

    def refresh_style(self) -> None:
        self._slider.refresh_style()
        self._spin.refresh_style()


# =============================================================================
# Node_ModuleCurveSlider —— 滑块 + 曲线预览面板
# =============================================================================

class Node_ModuleCurveSlider(QWidget):
    """滑块 + 曲线预览 / Slider with curve preview panel.

    DEMO 中给奖励曲线 / 课程曲线模块用的复合控件。本期骨架：
    顶部 ``Node_DualHandleRangeSlider``，下方放一个 paint-only 占位区。
    """

    rangeChanged = pyqtSignal(float, float)

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        widget_id: str = "node.curve_slider",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._slider = Node_DualHandleRangeSlider(
            lo=lo, hi=hi, minimum=minimum, maximum=maximum, parent=self,
        )
        self._slider.rangeChanged.connect(self.rangeChanged.emit)
        layout.addWidget(self._slider)

        self._preview = Node_CurvePreviewArea(parent=self)
        self._preview.setMinimumHeight(36)
        layout.addWidget(self._preview)

    def values(self) -> tuple:
        return self._slider.lo(), self._slider.hi()

    def setValues(self, lo: float, hi: float) -> None:
        self._slider.setRangeValues(lo, hi)

    def refresh_style(self) -> None:
        self._slider.refresh_style()
        self._preview.update()


class Node_CurvePreviewArea(QWidget):
    """曲线预览面板 / Curve preview area.

    给 ``Node_ModuleCurveSlider`` 用的占位 paint 区，用 Config 主题色绘背景
    + 一条参考基线。后续可由调用方注入 curve points 重写 paintEvent。
    """

    def paintEvent(self, e: QPaintEvent) -> None:
        from PyQt6.QtGui import QBrush, QColor, QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QColor(Config.get_color("bg_1", "#1c1c1c"))
        baseline = QColor(Config.get_color("border_2", "#3a3a3a"))
        rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        p.fillRect(rect, QBrush(bg))
        p.setPen(QPen(baseline, 1.0))
        y = rect.top() + rect.height() * 0.5
        p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        p.end()


__all__ = [
    "Node_RangeHandleSlider",
    "Node_DualHandleRangeSlider",
    "Node_NodeSlider",
    "Node_ModuleCurveSlider",
    "Node_CurvePreviewArea",
]
