# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas — Training Ground 画布子包.

子模块：
- ``page.py``           ``CanvasPage(QWidget)`` 页面壳；挂 view + scene
- ``scene.py``          ``CanvasScene(QGraphicsScene)`` 场景；含栅格背景
- ``view.py``           ``CanvasView(QGraphicsView)`` 视口；pan/zoom
- ``grid.py``           ``CachedGrid`` 缓存 ``QBrush`` 栅格生成器
- ``items.py``          ``NodeItem`` / ``PortItem`` / ``ConnectionItem``（custom paint）
- ``port_palette.py``   端口 ``data_type`` → 颜色映射

Phase 0: 仅骨架——栅格 + pan/zoom。
Phase 1a (本阶段): items.py 三件套 + 训练画布第一节点端到端。
Phase 1b: lod.py + 10 种 ParamRow + edit_dialogs.py。
Phase 1c: lowering.py 实装。

节点实现下沉到 ``RELEASE/src/nodes/<id>/``，由 ``registers.nodes.load()`` 自动发现。
"""

from .grid import CachedGrid
from .items import (
    EDGE_PEN_DATA,
    EDGE_PEN_HOVER,
    NODE_W,
    PARAM_ROW_H,
    PORT_HIT_R,
    PORT_R,
    PORT_ROW_H,
    ROLE_CONN,
    ROLE_DIR,
    ROLE_KIND,
    ROLE_META,
    ROLE_NAME,
    SEP_H,
    TITLE_H,
    ConnectionItem,
    NodeItem,
    PortItem,
    connect_ports,
)
from .lod import (
    T0_MAX,
    T1_MAX,
    T2_MAX,
    TIER_DETAIL,
    TIER_MINIMAP,
    TIER_NAMES,
    TIER_OVERVIEW,
    TIER_WORKING,
    lod_for_painter,
    tier_for_painter,
    tier_for_zoom,
)
from .page import CanvasPage
from .param_rows import (
    BadgeRow,
    BoolRow,
    ChoiceRow,
    CodeRow,
    IndexRow,
    NumberRow,
    ParamRow,
    PathRow,
    RangeRow,
    StringRow,
    TableReadOnlyRow,
    create_param_row,
)
from .port_palette import default_color_hex, get_port_color, known_types
from .scene import CanvasScene
from .view import CanvasView


__all__ = [
    # 页面壳
    "CachedGrid",
    "CanvasPage",
    "CanvasScene",
    "CanvasView",
    # items
    "NodeItem",
    "PortItem",
    "ConnectionItem",
    "connect_ports",
    # data() roles
    "ROLE_KIND",
    "ROLE_DIR",
    "ROLE_CONN",
    "ROLE_NAME",
    "ROLE_META",
    # geometry constants
    "TITLE_H",
    "NODE_W",
    "PORT_ROW_H",
    "PARAM_ROW_H",
    "SEP_H",
    "PORT_R",
    "PORT_HIT_R",
    "EDGE_PEN_DATA",
    "EDGE_PEN_HOVER",
    # palette
    "default_color_hex",
    "get_port_color",
    "known_types",
    # LOD
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
    # ParamRow
    "ParamRow",
    "BoolRow",
    "NumberRow",
    "StringRow",
    "ChoiceRow",
    "PathRow",
    "IndexRow",
    "RangeRow",
    "BadgeRow",
    "CodeRow",
    "TableReadOnlyRow",
    "create_param_row",
]
