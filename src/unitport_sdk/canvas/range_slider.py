"""unitport_sdk.canvas.range_slider — 单/双手柄 range slider paint helper.

替换 DEMO ``_RangeHandleSlider.paintEvent``（training_node_items.py:1473）的
视觉契约。两类滑块共用同一函数：
    dual=False → 单值 slider（vmin..vmax 上一个 thumb）
    dual=True  → 区间 slider（lo..hi 两个 thumb，中间 fill bar）

颜色 **全部** 走 ``Config.get_color`` 取自 ``system.ini``，本模块不允许定义
自己的颜色字段。新增视觉 slot 必须先加进 ``system.ini[Theme]``。

调用方负责：
- 几何 / 位置（``rect`` 是 item-local QRectF）
- 当前值（lo / hi）
- 鼠标命中（用 ``RangeSliderHitTest`` 决定按下哪个 thumb）

坐标系：rect 用 item-local；track 走 rect 中线。
LOD 建议：
    T0/T1 → 调用方跳过，画一根静态条 + 数值文本即可
    T2+   → 走本函数完整视觉
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen

from unitport_sdk.sys import Config


# =============================================================================
# Paint
# =============================================================================

def draw_range_slider(
    painter: QPainter,
    rect: QRectF,
    *,
    lo: float,
    hi: Optional[float] = None,
    vmin: float,
    vmax: float,
    dual: bool = False,
    hover_handle: Optional[Literal["lo", "hi"]] = None,
) -> None:
    """画 range slider / Paint a range slider.

    Args:
        painter: ``QPainter``
        rect: slider 整体矩形（item-local）
        lo: 下界值 / 单值
        hi: 上界值（dual=True 时必传；dual=False 时忽略）
        vmin / vmax: 值域
        dual: 是否双手柄
        hover_handle: 哪个 thumb 处于 hover（"lo" / "hi" / None）
    """
    if vmax <= vmin:
        return  # 无效值域，静默不画

    # ---- 颜色全部走 system.ini，禁止本地常量 ----
    track_color        = QColor(Config.get_color("border_2"))
    track_active_color = QColor(Config.get_color("highlight"))
    handle_color       = QColor(Config.get_color("main_t2"))
    handle_border      = QColor(Config.get_color("bg_3"))
    handle_hover_color = QColor(Config.get_color("main_t2"))

    track_y = rect.top() + rect.height() * 0.5
    track_h = max(3.0, rect.height() * 0.18)
    handle_r = max(5.0, rect.height() * 0.32)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 1. 轨道背景
    track_rect = QRectF(rect.left(), track_y - track_h * 0.5, rect.width(), track_h)
    painter.setBrush(QBrush(track_color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(track_rect, track_h * 0.5, track_h * 0.5)

    # 2. 选中段
    def _x(value: float) -> float:
        t = (value - vmin) / (vmax - vmin)
        return rect.left() + max(0.0, min(1.0, t)) * rect.width()

    if dual and hi is not None:
        x_lo, x_hi = _x(lo), _x(hi)
        active_rect = QRectF(min(x_lo, x_hi), track_rect.top(), abs(x_hi - x_lo), track_h)
    else:
        x_lo = rect.left()
        x_hi = _x(lo)
        active_rect = QRectF(x_lo, track_rect.top(), x_hi - x_lo, track_h)
    painter.setBrush(QBrush(track_active_color))
    painter.drawRoundedRect(active_rect, track_h * 0.5, track_h * 0.5)

    # 3. handles
    # 视觉契约：
    #   单把手 (dual=False) → 圆形 thumb，半径 handle_r
    #   双把手 (dual=True)  → 扁圆角矩形 thumb，高度 = 2*handle_r（与圆形等高），
    #                          宽度 = handle_r（圆形直径一半），圆角 = handle_r/2
    def _draw_round_handle(cx: float, hovered: bool) -> None:
        fill = handle_hover_color if hovered else handle_color
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(handle_border, 1.2))
        painter.drawEllipse(QPointF(cx, track_y), handle_r, handle_r)

    def _draw_flat_handle(cx: float, hovered: bool) -> None:
        fill = handle_hover_color if hovered else handle_color
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(handle_border, 1.2))
        h = handle_r * 2.0
        w = handle_r
        rect_h = QRectF(cx - w * 0.5, track_y - h * 0.5, w, h)
        painter.drawRoundedRect(rect_h, w * 0.5, w * 0.5)

    if dual and hi is not None:
        _draw_flat_handle(_x(lo), hover_handle == "lo")
        _draw_flat_handle(_x(hi), hover_handle == "hi")
    else:
        _draw_round_handle(_x(lo), hover_handle == "lo" or hover_handle == "hi")

    painter.restore()


# =============================================================================
# Hit-test 工具类
# =============================================================================

@dataclass
class RangeSliderHitTest:
    """range slider 鼠标命中工具 / Hit-test helper for range slider.

    用法：
        ht = RangeSliderHitTest(rect, vmin=0.0, vmax=1.0, dual=True)
        if (h := ht.handle_at(point, lo, hi)):
            # h == "lo" / "hi" → 按住对应 thumb 拖动
        v = ht.value_at(point.x())  # 把 x 映射回值域

    几何与 ``draw_range_slider`` 完全对齐。
    """

    rect: QRectF
    vmin: float
    vmax: float
    dual: bool = False
    handle_radius: Optional[float] = None  # None → 与 paint 一致按 rect 高度推导

    def _handle_r(self) -> float:
        return self.handle_radius if self.handle_radius is not None \
            else max(5.0, self.rect.height() * 0.32)

    def value_at(self, x: float) -> float:
        """把 x（item-local）映射回 [vmin, vmax]，越界 clamp."""
        if self.rect.width() <= 0:
            return self.vmin
        t = (x - self.rect.left()) / self.rect.width()
        t = max(0.0, min(1.0, t))
        return self.vmin + t * (self.vmax - self.vmin)

    def x_at(self, value: float) -> float:
        """把值映射到 x（item-local）."""
        if self.vmax <= self.vmin:
            return self.rect.left()
        t = (value - self.vmin) / (self.vmax - self.vmin)
        t = max(0.0, min(1.0, t))
        return self.rect.left() + t * self.rect.width()

    def handle_at(
        self,
        point: QPointF,
        lo: float,
        hi: Optional[float] = None,
    ) -> Optional[Literal["lo", "hi", "track"]]:
        """命中检测：返回 'lo' / 'hi' / 'track' / None.

        优先 thumb；否则若在轨道矩形内返回 'track'（让调用方做 click-to-jump）；
        否则 None。

        几何与 ``draw_range_slider`` 完全对齐：
            单把手 (dual=False) → 圆 thumb，半径 r → 圆内命中。
            双把手 (dual=True)  → 扁圆角矩形 thumb，宽 r、高 2r → 矩形内命中。
        两种模式都额外给 1.5px 的 hit margin。
        """
        r = self._handle_r()
        margin = 1.5
        cy = self.rect.top() + self.rect.height() * 0.5
        if self.dual and hi is not None:
            half_w = r * 0.5 + margin
            half_h = r + margin
            cx_lo, cx_hi = self.x_at(lo), self.x_at(hi)
            if (
                abs(point.x() - cx_lo) <= half_w
                and abs(point.y() - cy) <= half_h
            ):
                return "lo"
            if (
                abs(point.x() - cx_hi) <= half_w
                and abs(point.y() - cy) <= half_h
            ):
                return "hi"
        else:
            hit_r = r + margin
            cx = self.x_at(lo)
            if (point.x() - cx) ** 2 + (point.y() - cy) ** 2 <= hit_r * hit_r:
                return "lo"
        if self.rect.contains(point):
            return "track"
        return None


__all__ = ["draw_range_slider", "RangeSliderHitTest"]
