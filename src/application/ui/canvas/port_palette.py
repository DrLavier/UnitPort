"""application.ui.canvas.port_palette — 兼容 re-export / Backward-compat shim.

端口类型 → 颜色字典已迁到 ``unitport_sdk.canvas.port_palette``。本文件保留
是为了不打破现有 ``from .port_palette import get_port_color`` 等 import。
新代码请直接 ``from unitport_sdk import get_port_color, PORT_TYPE_COLORS``。
"""

from __future__ import annotations

from unitport_sdk.canvas.port_palette import (
    PORT_TYPE_COLORS,
    default_color_hex,
    get_port_color,
    known_types,
)


# 兼容旧名 _PALETTE
_PALETTE = PORT_TYPE_COLORS


__all__ = [
    "PORT_TYPE_COLORS",
    "default_color_hex",
    "get_port_color",
    "known_types",
    "_PALETTE",
]
