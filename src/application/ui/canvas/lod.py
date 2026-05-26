# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.lod — LOD tier 工具 / Level-of-detail tier helpers.

4-tier 划分（按 zoom 因子，``view.transform().m11()`` 或 painter world m11）：

    T0 Overview    0.10 – 0.30   实心矩形 + layer 配色，跳过端口/参数行/边
    T1 Minimap     0.30 – 0.80   节点标题 + 小端口；边走直线
    T2 Working     0.80 – 2.50   完整渲染；正常 Bezier 边
    T3 Detail      2.50 – 4.00   + 选中光晕 + 端口名标签 + 1px 微格

Qt 钩子：``QStyleOptionGraphicsItem.levelOfDetailFromTransform(painter.worldTransform())``。
我们用更直接的 ``painter.worldTransform().m11()`` 等价：因为画布只做均匀缩放，
m11 == m22 == zoom factor。

``CanvasView`` 顶层维护 ``_zoom_tier``，跨 tier 时发 ``zoomChanged(int)`` 信号给
``CanvasScene`` / ``CachedGrid`` 调整背景层 LOD。
"""

from __future__ import annotations

from PyQt6.QtGui import QPainter


T0_MAX = 0.30
T1_MAX = 0.80
T2_MAX = 2.50
# T3 = > T2_MAX

TIER_OVERVIEW = 0  # T0
TIER_MINIMAP = 1   # T1
TIER_WORKING = 2   # T2
TIER_DETAIL = 3    # T3

TIER_NAMES = {
    TIER_OVERVIEW: "T0_OVERVIEW",
    TIER_MINIMAP: "T1_MINIMAP",
    TIER_WORKING: "T2_WORKING",
    TIER_DETAIL: "T3_DETAIL",
}


def tier_for_zoom(zoom: float) -> int:
    """zoom 因子 → tier 序号 / Map zoom factor to tier id."""
    if zoom < T0_MAX:
        return TIER_OVERVIEW
    if zoom < T1_MAX:
        return TIER_MINIMAP
    if zoom < T2_MAX:
        return TIER_WORKING
    return TIER_DETAIL


def tier_for_painter(painter: QPainter) -> int:
    """从 painter 当前 world transform 取 LOD tier / Tier from painter transform."""
    return tier_for_zoom(painter.worldTransform().m11())


def lod_for_painter(painter: QPainter) -> float:
    """原始 LOD 浮点数 / Raw LOD scalar.

    画布只做均匀缩放（缩放在 view 中通过 ``scale(f, f)`` 同时改 m11/m22），
    因此 ``m11`` 即为 ``levelOfDetailFromTransform`` 的等价值。
    """
    return painter.worldTransform().m11()


__all__ = [
    "T0_MAX",
    "T1_MAX",
    "T2_MAX",
    "TIER_OVERVIEW",
    "TIER_MINIMAP",
    "TIER_WORKING",
    "TIER_DETAIL",
    "TIER_NAMES",
    "tier_for_zoom",
    "tier_for_painter",
    "lod_for_painter",
]
