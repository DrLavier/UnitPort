# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.port_visual — PortSpec → 视觉描述 dict.

把"端口长什么样"也变成工厂分发——保持端口 / 参数行的「自动转换」对称：
``PortItem.__init__`` 调一次 ``make_port_visual(spec)`` 拿到颜色 / 半径 /
形状 / 通道，避免每次 paint 都重新查 palette。

未来可扩展形状（square / diamond）以区分 channel 或某类特殊端口（fan-in
聚合口、reconfigurable 接口等）。当前仅 ``circle``。
"""

from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtGui import QColor

from .port_palette import get_port_color


def make_port_visual(spec: Any) -> Dict[str, Any]:
    """从 PortSpec 派生视觉描述 / Derive visual descriptor from PortSpec.

    Args:
        spec: ``application.compiler.nodes.PortSpec`` 实例（保留 Any 类型避免
            SDK 反向依赖业务 dataclass）。需要 ``spec.type`` 和 ``spec.optional``
            两个字段。

    Returns:
        字典含：
            ``color``    : QColor，类型主色
            ``radius``   : float，视觉半径（默认 6.0）
            ``shape``    : str，``"circle"`` | 未来扩展
            ``channel``  : str，``"flow"`` | ``"data"``
            ``optional`` : bool，对应 PortSpec.optional
    """
    data_type = getattr(spec, "type", "any")
    optional = bool(getattr(spec, "optional", False))
    channel = "flow" if data_type == "flow" else "data"
    return {
        "color": get_port_color(data_type),
        "radius": 6.0,
        "shape": "circle",
        "channel": channel,
        "optional": optional,
    }


__all__ = ["make_port_visual"]
