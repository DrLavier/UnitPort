"""application.ui.canvas.items — 画布 Item 类（custom paint，零 ProxyWidget）.

Phase 1a 三件套：
    NodeItem          ``QGraphicsItem``      节点矩形 + 标题栏 + 端口承载
    PortItem          ``QGraphicsItem``      端口圆点（与 DEMO data() 协议对齐）
    ConnectionItem    ``QGraphicsPathItem``  Bezier 连线 + dirty flag

设计原则：
- 全部 ``paint()`` 自绘，绝不嵌套 ``QGraphicsProxyWidget``（DEMO 反模式）
- 主题色一律走 ``Config.get_color``；端口类型色走 ``port_palette.get_port_color``
- ``ConnectionItem`` 持脏标记：``update_path()`` 不脏直接 return
- LOD 钩子留空，Phase 1b 在 ``lod.py`` 落地后再接入

ParamRow 10 种子类与编辑弹窗（Bool/Number/String/Choice/Path/Index/Range/Badge/
Code/TableReadOnly）在 Phase 1b 实现，本文件仅保留接入位（``NodeItem`` 已为
"参数行区域" 预留几何，但当前不构造 ParamRow 实例）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QLineEdit,
    QStyleOptionGraphicsItem,
    QWidget,
)


class _TitleLineEdit(QLineEdit):
    """Inline node-title editor — Enter → commit, Esc → cancel, focusOut → commit.

    Why subclass rather than the previous ``eventFilter`` + ``clearFocus()``
    chain: a QLineEdit wrapped in QGraphicsProxyWidget does not reliably emit
    ``editingFinished`` from ``clearFocus()`` — the scene focus item state is
    decoupled from the inner widget's focus state, so the chain silently
    dead-ends and the cursor stays parked. Same shape as ``note_item._BodyEdit``.
    """

    def __init__(self, on_commit, on_cancel, parent=None) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._on_cancel = on_cancel
        self._firing = False

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            if not self._firing:
                self._firing = True
                try:
                    self._on_cancel()
                finally:
                    self._firing = False
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self._fire_commit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self._fire_commit()

    def _fire_commit(self) -> None:
        if self._firing:
            return
        self._firing = True
        try:
            self._on_commit()
        finally:
            self._firing = False

from unitport_sdk import Assets, Config, draw_dot_port, get_port_color, make_port_visual, tr

from application.compiler.nodes import NodeManifest, PortSpec

from .lod import (
    TIER_DETAIL,
    TIER_MINIMAP,
    TIER_WORKING,
    tier_for_painter,
)
from .param_rows import ParamRow, create_param_row


# =============================================================================
# 数据角色 / data() roles —— 与 DEMO PortItem 协议对齐，便于 lowering 直接复用
# =============================================================================

ROLE_KIND = 0      # str: "port" | "node"
ROLE_DIR = 1       # str: "in" | "out"
ROLE_CONN = 2      # list: connections (ConnectionItem 列表)
ROLE_NAME = 3      # str: port slot name
ROLE_META = 20     # dict: {"channel": "data"|"flow", "data_type": str, ...}


# =============================================================================
# 几何常量 / geometry constants —— 与 DEMO 训练画布对齐
# =============================================================================

TITLE_H = 30          # 标题栏高度（DEMO TrainingNodeItem.TITLE_H）
NODE_W = 324          # 节点宽度 — 268 → 324（+21%），多出宽度全部给到 value 半边
                      # （param_rows.KEY_WIDTH 不变 → 右侧控件区可用宽度
                      # = NODE_W - SEP_X - VALUE_PAD_RIGHT 同步加宽）
PORT_ROW_H = 26       # 端口行高（DEMO Training，比 Mission 高 4px）
PARAM_ROW_H = 24      # 参数行高
SEP_H = 8             # 分隔条高度
H_PAD = 14            # 水平内边距
PORT_MARGIN = 10      # 端口距右边距
LABEL_GAP = 17        # 端口圆点中心到 label 起点的水平距离（≈ x=23 - x=6）
MIN_BODY_H = 20       # 节点最小 body 高度
CORNER_R = 6          # body / 标题栏圆角

PORT_R = 6            # 端口可视半径
PORT_HIT_R = 11       # 端口默认命中半径（Phase 1b LOD-aware）

EXPAND_BTN_ROW_H = 22 # 收纳态展开按钮行高度(Title → in → [chevron] → out)

EDGE_PEN_DATA = 1.6   # 数据边线宽
EDGE_PEN_HOVER = 2.6  # hover 加粗

ICON_SIZE = 16        # 标题栏右侧节点 icon 尺寸（16×16）
BADGE_SIZE = 14       # Coverage / health Badge 尺寸（rewards 节点）
BADGE_GAP = 5         # Badge 与 ICON 之间的水平间距


# 进程级 SVG 渲染器缓存：stem → QSvgRenderer | None（None = 此 stem 在 Assets 中
# 找不到，免得每帧重新查盘）。Layer 字母占位无需缓存，QFont 走 paint 内常量。
_SVG_RENDERER_CACHE: Dict[str, Optional[QSvgRenderer]] = {}


def _icon_renderer(stem: str) -> Optional[QSvgRenderer]:
    """``stem`` → 缓存的 QSvgRenderer；找不到对应 Assets 文件返回 None.

    缺图静默返回 None（不 log_error）—— 启动日志干净是契约。
    """
    if not stem:
        return None
    if stem in _SVG_RENDERER_CACHE:
        return _SVG_RENDERER_CACHE[stem]
    path = Assets.find_icon(stem)
    if path is None:
        _SVG_RENDERER_CACHE[stem] = None
        return None
    try:
        r = QSvgRenderer(str(path))
        if not r.isValid():
            _SVG_RENDERER_CACHE[stem] = None
            return None
        _SVG_RENDERER_CACHE[stem] = r
        return r
    except Exception:
        _SVG_RENDERER_CACHE[stem] = None
        return None


# =============================================================================
# NodeItem
# =============================================================================

class NodeItem(QGraphicsItem):
    """节点矩形 / Node rectangle item.

    布局（自顶向下，DEMO Training Node 风格）：
        ┌────────────────────────────────────┐  y = 0
        │ Title Bar (TITLE_H, gradient)      │
        ├────────────────────────────────────┤  y = TITLE_H
        │●in_1   <input row, type tint>      │
        │●in_2   <input row, type tint>      │
        │ ── separator (#2E2E2E) ──          │  SEP_H
        │     param_1                  ▾     │  param row
        │     param_2                  ▾     │
        │ ── separator (#2E2E2E) ──          │  SEP_H
        │            <output row>     out_1●│
        │            <output row>     out_2●│
        └────────────────────────────────────┘
    """

    def __init__(
        self,
        manifest: NodeManifest,
        node_id: str,
        params: Optional[dict] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.manifest = manifest
        self.node_id = node_id
        self.params: dict = dict(params or {})

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setData(ROLE_KIND, "node")

        # DeviceCoordinateCache 在 pan/zoom 下每次变换缓存失效，反而比直接重画慢；
        # 性能优化由 BspTreeIndex + 视口剔除（Phase 1b）+ LOD 共同承担。
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        self._width = NODE_W
        # Title override 必须在 _resolve_title 调用前初始化 —— _resolve_title 读
        # self._display_name_override。其它 rename widgets 留到 __init__ 末尾再建。
        self._display_name_override: str = ""
        # Per-instance color override (hex string like "#FF6B6B" or "")。空值表示
        # 走默认 layer 色 (_title_bar_color()) ;非空时该 hex 直接作为标题栏色。
        # 仅影响本画布上这一个 NodeItem,不动 system.ini[Theme]。
        self._color_override: str = ""
        self._title_rename_edit = None
        self._title_rename_proxy = None
        self._editing_title: bool = False
        self._title_text = self._resolve_title()
        self._opt_suffix = tr("canvas.port.optional", "(opt)")
        # Coverage Badge state (rewards × motion_phase isolation feature).
        # Populated by ``_paint_coverage_badge`` so ``mouseDoubleClickEvent``
        # can dispatch the Inspector dialog without re-computing the report.
        self._coverage_report_cache: Any = None
        # Dump-health Badge state (Robot Node only — per-format dump
        # state + cross-format gap + base-height offset). Populated by
        # ``_paint_dump_health_badge`` for symmetry with the coverage
        # cache so a future tooltip / context-menu can reuse the same
        # snapshot without re-querying the registry.
        self._dump_health_cache: Any = None

        # 构建端口
        self._in_ports: List["PortItem"] = []
        self._out_ports: List["PortItem"] = []
        # Dynamic inline ports — created by ``_rebuild_inline_ports`` from
        # ParamRow.iter_inline_port_anchors() during _compute_layout. Keyed by
        # port name so we can diff-update across layout passes (preserve
        # existing PortItems + their connections rather than tearing down and
        # rebuilding on every redraw).
        self._inline_in_ports: Dict[str, "PortItem"] = {}
        self._build_ports()

        # 构建参数行
        self._param_rows: List[ParamRow] = []
        self._build_param_rows()

        # 段落 y 起点（None 表示该段不存在）
        self._in_y: Optional[float] = None
        self._params_y: Optional[float] = None
        self._out_y: Optional[float] = None
        self._sep_ys: List[float] = []
        self._body_h: float = MIN_BODY_H

        # 收纳 / 禁用 状态(ComfyUI-style bypass + visual fold)。
        # 收纳态:ParamRow 全部 hide,布局变 Title → in → [chevron] → out。
        # 禁用态:画灰色遮罩,连到本节点的边在 paint 时也变灰;编译时整条节点 + 边
        # 被 WorkflowIR.from_dict 过滤掉,不进入下游 backend。
        # 双方持久化到 canvas JSON,通过 set_collapsed/set_disabled 公共方法变更。
        self._collapsed: bool = False
        self._disabled: bool = False
        # Diagnostic state — set by canvas error reporter when a self-check
        # (spec_validator / env_cfg_compiler) flags this node. Non-empty
        # strings drive ``paint`` to draw a danger-zone outline. Cleared
        # by ``mark_diagnostic(None)`` / next training submit.
        self._diag_state: Optional[str] = None
        # 收纳态时由 _compute_layout 设置:展开按钮(chevron)的命中矩形(item-local)
        self._chevron_y: Optional[float] = None
        self._chevron_rect: Optional[QRectF] = None
        # NOTE: _display_name_override / 标题改名 inline widgets 已在更早位置
        # 初始化(_resolve_title 在 __init__ 早期被调用,需要 override 已就绪)。

        self._compute_layout()
        self._bounding = QRectF(0, 0, self._width, TITLE_H + self._body_h)

    # ---- 公开 API ----

    def port(self, name: str, direction: str = "in") -> Optional["PortItem"]:
        """按名取端口；direction ∈ {'in', 'out'}.

        Inline (row-bound) input ports are looked up in
        ``_inline_in_ports`` first so canvas-IR connection loading can find
        ``reward_in__<item_id>`` ports that were created dynamically by the
        TrainingItemsInlineRow.
        """
        if direction == "in":
            p = self._inline_in_ports.get(name)
            if p is not None:
                return p
        coll = self._in_ports if direction == "in" else self._out_ports
        for p in coll:
            if p.port_name == name:
                return p
        return None

    def iter_ports(self) -> Iterable["PortItem"]:
        """遍历所有端口（先 in 后 out；包含动态 inline ports）."""
        yield from self._in_ports
        yield from self._inline_in_ports.values()
        yield from self._out_ports

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return TITLE_H + self._body_h

    # ---- Qt overrides ----

    def boundingRect(self) -> QRectF:
        # 端口圆点（PortItem）作为子项贴在 x=0 / x=width 上，绘制半径会越过 _bounding
        # 左右各 PORT_HIT_R；选中光晕也外扩 3px。把这些纳入父 boundingRect，否则
        # BoundingRectViewportUpdate 模式下节点拖动后旧位置端点会留 dot 残影。
        pad = max(PORT_HIT_R, 3) + 1
        return self._bounding.adjusted(-pad, -3, pad, 3)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        tier = tier_for_painter(painter)
        title_color = self._title_bar_color()

        # 任何缩放档都走完整绘制——节点装饰禁止提前消失。
        body_color = QColor(Config.get_color("bg_1", "#2A2A2A"))
        title_text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        border_color = QColor(Config.get_color("border_1", "#444444"))
        sep_color = QColor(Config.get_color("canvas_node_sep", "#2E2E2E"))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # T3 Detail: 选中光晕（在 body 之前画，避免被覆盖）
        if tier == TIER_DETAIL and self.isSelected():
            glow_color = QColor(Config.get_color("highlight", "#F6D393"))
            glow_color.setAlpha(80)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                self._bounding.adjusted(-3, -3, 3, 3),
                CORNER_R + 2,
                CORNER_R + 2,
            )

        # 1. body 背景（圆角矩形 + 1px crisp border）
        painter.setBrush(QBrush(body_color))
        painter.setPen(QPen(border_color, 1.0))
        painter.drawRoundedRect(self._bounding, CORNER_R, CORNER_R)

        # 2. 标题栏：clip 到 body 圆角形状，画 lighter→base 渐变（DEMO Training Node）
        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(self._bounding, CORNER_R, CORNER_R)
        painter.setClipPath(clip_path)
        grad = QLinearGradient(0.0, 0.0, 0.0, TITLE_H)
        grad.setColorAt(0.0, title_color.lighter(145))
        grad.setColorAt(1.0, title_color)
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(0, 0, self._width, TITLE_H))
        painter.restore()

        # 3. 标题栏底分割线（用 title_color.darker(150)，比 sep 略深）
        painter.setPen(QPen(title_color.darker(150), 0.8))
        painter.drawLine(QPointF(0, TITLE_H), QPointF(self._width, TITLE_H))

        # 4. 标题文字(编辑中由 QLineEdit 顶替,不画)
        if not self._editing_title:
            painter.setPen(QPen(title_text_color))
            font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
            font.setPixelSize(int(Config.get_font_size("size_small", 12)))
            font.setBold(True)
            painter.setFont(font)
            text_rect = QRectF(H_PAD, 0, self._width - H_PAD - 40, TITLE_H)
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self._title_text,
            )

        # 4b. 标题栏右侧 icon —— 优先 manifest.icon → Assets.find_icon → QSvgRenderer
        # 缺图回落到 layer 字母占位（A/B/C/D/IL）。
        icon_x = self._width - H_PAD - ICON_SIZE
        icon_y = (TITLE_H - ICON_SIZE) * 0.5
        icon_rect = QRectF(icon_x, icon_y, ICON_SIZE, ICON_SIZE)
        self._paint_icon(painter, icon_rect, title_text_color)

        # 4c. Coverage Badge — rewards × motion-phase isolation feature.
        # Only drawn on phase-aware nodes (rewards node with reward_terms
        # meta.phase_aware = true) and gated by [Canvas] phase_lanes_enabled.
        # The Badge color reflects worst-phase severity (✗ red / ⚠ yellow /
        # ✓ green); double-click opens the Coverage Inspector dialog. See
        # registers.rewards_coverage.evaluate for the rule semantics.
        self._paint_coverage_badge(painter)

        # UX-3: Robot Node dump-health badge. Same visual pattern (tri-
        # state coloured circle at the title-bar right edge) but on the
        # robot node it reflects per-format dump state + cross-format
        # IR-role coverage + base-height calibration. The Robot Node
        # used to be "always green" because the only visible state was
        # "asset_id picked", which let users sail past a stale MJCF
        # table into a multi-hour training run that produced an
        # undeployable bundle. Now the badge tells them up-front.
        self._paint_dump_health_badge(painter)

        # 5. 端口行 tint + label（任何 tier 都画）
        self._paint_port_rows(painter, tier)

        # 6. 分隔线（任何 tier 都画）
        if self._sep_ys:
            painter.setPen(QPen(sep_color, 1.0))
            for sep_y in self._sep_ys:
                y = sep_y + SEP_H * 0.5
                painter.drawLine(
                    QPointF(H_PAD, y),
                    QPointF(self._width - H_PAD, y),
                )

        # 7. 选中：1.5px 白色虚线 (alpha=210)（DEMO Training Node 选中样式）
        if self.isSelected():
            sel_pen = QPen(QColor(255, 255, 255, 210), 1.5)
            sel_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                self._bounding.adjusted(0.75, 0.75, -0.75, -0.75),
                CORNER_R,
                CORNER_R,
            )

        # 8. 收纳态 chevron（▼，提示"点我展开"）
        if self._collapsed and self._chevron_rect is not None:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            chev_color = QColor(Config.get_color("canvas_node_chevron", "#9CA3AF"))
            painter.setPen(QPen(chev_color, 1.6, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            cx = self._chevron_rect.center().x()
            cy = self._chevron_rect.center().y()
            painter.drawLine(QPointF(cx - 6, cy - 3), QPointF(cx, cy + 3))
            painter.drawLine(QPointF(cx, cy + 3), QPointF(cx + 6, cy - 3))
            painter.restore()

        # 9. 禁用遮罩(画在最上,盖住 title/ports/params/selection halo)
        # —— 与 ComfyUI bypass 视觉一致。crisp 1px 边框已经在 step 1 画过,
        # 用户仍能看到选中态依稀边框透出。
        if self._disabled:
            overlay = QColor(Config.get_color(
                "canvas_node_disabled_overlay", "#1F1F1FE6",
            ))
            painter.setBrush(QBrush(overlay))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self._bounding, CORNER_R, CORNER_R)

        # 10. Diagnostic overlay — drawn on top of everything except the
        # disable mask so the user immediately sees which node a
        # self-check error refers to. Theme slot ``danger_zone`` carries
        # the same red used by the placeholder-node border (consistent
        # "something is wrong here" signal across canvas surfaces).
        if self._diag_state == "danger":
            danger_color = QColor(Config.get_color("danger_zone"))
            danger_pen = QPen(danger_color, 2.4)
            painter.setPen(danger_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                self._bounding.adjusted(-1.0, -1.0, 1.0, 1.0),
                CORNER_R + 1,
                CORNER_R + 1,
            )

    def mark_diagnostic(self, state: Optional[str]) -> None:
        """Set the diagnostic overlay state for this node and trigger a repaint.

        ``state="danger"`` paints a danger-zone outline; ``None`` clears it.
        Called by the canvas error reporter when self-check flags this node;
        cleared on the next training submit.
        """
        if state == self._diag_state:
            return
        self._diag_state = state
        self.update()

    def _paint_icon(
        self,
        painter: QPainter,
        rect: QRectF,
        fallback_color: QColor,
    ) -> None:
        """画标题栏右侧 16×16 icon —— 找到 SVG 走 QSvgRenderer，缺则画 layer 字母."""
        icon_field = (self.manifest.icon or "").strip()
        renderer: Optional[QSvgRenderer] = None
        if icon_field:
            # manifest.icon 如带 .svg 后缀，剥掉后传 stem，让 Assets.find_icon 走
            # ``<assets>/icon/<stem>.{svg,png,ico}`` 路径；否则按相对路径处理。
            stem = icon_field
            for ext in (".svg", ".png", ".ico"):
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            renderer = _icon_renderer(stem)
            if renderer is None and stem != icon_field:
                # 罕见：manifest.icon 本身就是相对路径（含分隔符）。直接交给 Assets.find_icon。
                renderer = _icon_renderer(icon_field)
        if renderer is not None:
            renderer.render(painter, rect)
            return
        # ---- 缺图回落：layer 字母 ----
        layer_text = (self.manifest.layer or "").upper()
        if not layer_text:
            layer_text = "?"
        # IL 是双字符，其它单字符。任何情况下都画在 icon_rect 中央。
        painter.save()
        painter.setPen(QPen(fallback_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(11 if len(layer_text) > 1 else 12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignCenter),
            layer_text,
        )
        painter.restore()

    # ------------------------------------------------------------------
    # Coverage Badge — rewards × motion-phase isolation
    # ------------------------------------------------------------------

    def _is_coverage_node(self) -> bool:
        """Phase-aware coverage Badge applies to this node.

        Three preconditions: the node's id is ``"rewards"``, the
        ``reward_terms`` ParamSpec has ``meta.phase_aware = true``, and
        the ``[Canvas] phase_lanes_enabled`` feature flag is on. The
        manifest id check keeps the check cheap on every paint of every
        non-rewards node — exiting after a single string compare.
        """
        if self.manifest.id != "rewards":
            return False
        try:
            enabled = Config.get_value(
                "Canvas", "phase_lanes_enabled", True, value_type=bool,
            )
        except Exception:
            enabled = True
        if not enabled:
            return False
        for spec in self.manifest.parameters:
            if spec.key == "reward_terms" and (spec.meta or {}).get("phase_aware"):
                return True
        return False

    def _coverage_badge_rect(self) -> QRectF:
        """Badge sits just left of the title-bar icon."""
        icon_x = self._width - H_PAD - ICON_SIZE
        return QRectF(
            icon_x - BADGE_GAP - BADGE_SIZE,
            (TITLE_H - BADGE_SIZE) * 0.5,
            BADGE_SIZE, BADGE_SIZE,
        )

    def _compute_coverage(self):
        """Evaluate the current reward_terms against the phase layout.

        Returns ``None`` when the node is not coverage-aware or the
        registers layer failed to load (caller skips badge painting).
        Heavy: this re-runs every paint — kept cheap by short-circuiting
        on ``has_terms = False`` and by the evaluator's reliance on the
        small canonical phase list (~3 phases).
        """
        if not self._is_coverage_node():
            return None
        try:
            from registers import motion_phases as _mp
            from registers import rewards_coverage as _rc
        except Exception:
            return None
        raw = self.params.get("reward_terms", "{}")
        if isinstance(raw, str):
            try:
                terms = json.loads(raw) if raw.strip() else {}
            except Exception:
                terms = {}
        elif isinstance(raw, dict):
            terms = raw
        else:
            terms = {}
        phase_objs = _mp.list_phases()
        phases = [
            {
                "id": p.id,
                "display_name": p.display_name_default,
                "polarity_required": p.polarity_required,
            }
            for p in phase_objs
        ]
        if not phases:
            return None
        backend = str(self.params.get("backend", "sb3") or "sb3").strip()
        try:
            return _rc.evaluate(terms, phases, backend=backend)
        except Exception:
            return None

    def _paint_coverage_badge(self, painter: QPainter) -> None:
        report = self._compute_coverage()
        self._coverage_report_cache = report
        if report is None or not report.has_terms:
            return
        sev = report.severity
        if sev == "critical":
            bg_slot = "canvas_badge_critical"
            glyph = "!"
        elif sev == "warning":
            bg_slot = "canvas_badge_warning"
            glyph = "!"
        else:
            bg_slot = "safe_zone"
            glyph = "OK"
        rect = self._coverage_badge_rect()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QColor(Config.get_color(bg_slot, "#999999"))
        fg = QColor(Config.get_color("canvas_badge_text", "#0A0A0A"))
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(rect)
        painter.setPen(QPen(fg))
        f = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        f.setPixelSize(9 if len(glyph) > 1 else 11)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), glyph)
        painter.restore()
        # Refresh the QGraphicsItem tooltip so hovering the node shows a
        # summary of which phases failed. The tooltip is item-wide rather
        # than badge-only — Qt's QGraphicsItem tooltip API doesn't support
        # per-region tooltips without a hoverMoveEvent state machine, and
        # a single short summary on hover is enough for the entry point.
        tip_lines = [tr("canvas.rewards.coverage.title", "Reward Coverage")]
        if sev == "critical":
            tip_lines.append(tr(
                "canvas.rewards.coverage.red",
                "Critical: one or more phases are missing required constraint reward.",
            ))
        elif sev == "warning":
            tip_lines.append(tr(
                "canvas.rewards.coverage.yellow",
                "Warning: at least one phase is missing a recommended reward.",
            ))
        else:
            tip_lines.append(tr(
                "canvas.rewards.coverage.green",
                "All phases have positive + negative reward coverage.",
            ))
        for ph in report.phases:
            if ph.severity == "ok":
                continue
            tip_lines.append(
                f"  • {ph.display_name}: +{len(ph.positive_terms)} / "
                f"−{len(ph.negative_terms)} — {ph.message}"
            )
        tip_lines.append("")
        tip_lines.append(tr(
            "canvas.rewards.coverage.double_click_hint",
            "Double-click the badge to open the Coverage Inspector.",
        ))
        self.setToolTip("\n".join(tip_lines))

    # ------------------------------------------------------------------
    # Robot Node dump-health badge (UX-3)
    # ------------------------------------------------------------------

    def _is_robot_node(self) -> bool:
        """True iff this node is the Robot Node — the only node carrying
        a SKU and therefore eligible for the dump-health badge.
        """
        return self.manifest.id == "robot"

    def _current_robot_sku(self) -> str:
        """Resolve the canonical SKU from the Robot Node's asset_id param.

        ``asset_id`` is the user-picked id (brand.model or alias); the
        registry's ``resolve_id`` turns it into the canonical hex SKU
        the rest of the pipeline keys off of. Empty string when the
        param is unset or resolution fails — badge skips painting.
        """
        if not self._is_robot_node():
            return ""
        raw = ""
        param_entry = self.params.get("asset_id")
        if isinstance(param_entry, dict):
            raw = str(param_entry.get("value") or "")
        else:
            raw = str(param_entry or "")
        raw = raw.strip()
        if not raw:
            return ""
        try:
            from registers import resolve_id
            return resolve_id(raw) or raw
        except Exception:
            return raw

    def _dump_health_badge_rect(self) -> QRectF:
        """Badge sits just left of the icon, mirroring the coverage badge slot.

        Robot Node never carries a coverage badge (not phase-aware), so
        the two badges never overlap; we can share the rect formula.
        """
        icon_x = self._width - H_PAD - ICON_SIZE
        return QRectF(
            icon_x - BADGE_GAP - BADGE_SIZE,
            (TITLE_H - BADGE_SIZE) * 0.5,
            BADGE_SIZE, BADGE_SIZE,
        )

    def _paint_dump_health_badge(self, painter: "QPainter") -> None:
        """Tri-state Robot Node badge: per-format dump + cross-format gap.

        Cached health stays on ``self._dump_health_cache`` so the
        mouseDoubleClickEvent handler can re-use it without recomputing.
        """
        if not self._is_robot_node():
            return
        sku = self._current_robot_sku()
        if not sku:
            self._dump_health_cache = None  # type: ignore[attr-defined]
            return
        try:
            from application.ui.dialogs.joint_mapping_dialog import (
                JointMappingHealth,
            )
            health = JointMappingHealth.build(sku)
        except Exception:
            # Best-effort: a registry/import failure must NOT block node
            # paint. Skip the badge silently — the user's canvas keeps
            # rendering even if introspection breaks.
            self._dump_health_cache = None  # type: ignore[attr-defined]
            return
        self._dump_health_cache = health  # type: ignore[attr-defined]

        # Decide severity.
        # red    — neither format dumped, OR active_format empty
        # yellow — one format only / cross-format gap
        # green  — both formats present and IR-role sets agree
        n_mjcf = len(health.mjcf_rows)
        n_usd = len(health.usd_rows)
        if not n_mjcf and not n_usd:
            sev = "critical"
        elif health.coverage_has_gap or (n_mjcf == 0) != (n_usd == 0):
            sev = "warning"
        else:
            sev = "ok"

        if sev == "critical":
            bg_slot, glyph = "canvas_badge_critical", "!"
        elif sev == "warning":
            bg_slot, glyph = "canvas_badge_warning", "!"
        else:
            bg_slot, glyph = "safe_zone", "OK"

        rect = self._dump_health_badge_rect()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QColor(Config.get_color(bg_slot, "#999999"))
        fg = QColor(Config.get_color("canvas_badge_text", "#0A0A0A"))
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(rect)
        painter.setPen(QPen(fg))
        f = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        f.setPixelSize(9 if len(glyph) > 1 else 11)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), glyph)
        painter.restore()

        # Tooltip carries the actionable summary. Reusing setToolTip
        # here is safe because Robot Node is not phase-aware and the
        # coverage badge tooltip writer never fires on it.
        offset_overlay = health.height_offset or {}
        offset_z = offset_overlay.get("offset_z")
        if offset_z is None:
            offset_str = tr(
                "canvas.robot.health.offset_uncalibrated",
                "(not calibrated)",
            )
        else:
            try:
                offset_str = f"{float(offset_z):.4f} m"
            except (TypeError, ValueError):
                offset_str = str(offset_z)
        tip_lines = [
            tr("canvas.robot.health.title", "Robot dump health"),
            tr(
                "canvas.robot.health.counts",
                "MJCF: {m} joints  ·  USD: {u} joints  ·  base offset: {off}",
            ).format(m=n_mjcf, u=n_usd, off=offset_str),
        ]
        if sev == "critical":
            tip_lines.append(tr(
                "canvas.robot.health.crit",
                "Neither format has been dumped — open Robot Assets and "
                "click Dump MJCF / Dump USD before training.",
            ))
        elif sev == "warning":
            if n_mjcf == 0 or n_usd == 0:
                missing_fmt = "MJCF" if n_mjcf == 0 else "USD"
                tip_lines.append(tr(
                    "canvas.robot.health.single",
                    "Only {present} is dumped — the {missing} deploy target "
                    "will be unavailable for bundles trained against this robot.",
                ).format(
                    present=("USD" if n_mjcf == 0 else "MJCF"),
                    missing=missing_fmt,
                ))
            else:
                aff = ", ".join(health.coverage_affected_targets) or "(see View joint mapping)"
                tip_lines.append(tr(
                    "canvas.robot.health.gap",
                    "MJCF and USD declare different IR-role sets — affected: {aff}.",
                ).format(aff=aff))
        else:
            tip_lines.append(tr(
                "canvas.robot.health.ok",
                "MJCF and USD agree on joint set — both deploy targets available.",
            ))
        tip_lines.append("")
        tip_lines.append(tr(
            "canvas.robot.health.double_click_hint",
            "Double-click the badge for the full joint mapping table.",
        ))
        self.setToolTip("\n".join(tip_lines))

    def _open_joint_mapping_dialog(self, event) -> None:
        """Mouse-double-click dispatch: open the joint mapping dialog."""
        sku = self._current_robot_sku()
        if not sku:
            return
        try:
            from application.ui.dialogs import JointMappingDialog
            JointMappingDialog.open_for(None, sku)
        except Exception:
            # Dialog open failure must not crash the canvas. The dump
            # health was already visible via the badge tooltip.
            return

    def mouseDoubleClickEvent(self, event):  # type: ignore[override]
        """收纳态:双击节点体任意处展开。展开态:title 区域 → 改名;coverage badge 区 → Inspector。"""
        # 收纳态双击 → 展开(三种入口之一,与 chevron / toolbar 共用 _toggle_collapse_for_node)
        if self._collapsed and event.button() == Qt.MouseButton.LeftButton:
            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is not None and hasattr(page, "_toggle_collapse_for_node"):
                page._toggle_collapse_for_node(self.node_id)
            event.accept()
            return
        # Robot Node dump-health badge dispatch (priority over title rename,
        # because the badge sits in the right portion of the title bar).
        if event.button() == Qt.MouseButton.LeftButton and self._is_robot_node():
            rect = self._dump_health_badge_rect()
            if rect.contains(event.pos()):
                self._open_joint_mapping_dialog(event)
                event.accept()
                return
        # 展开态:Coverage badge dispatch(优先于 title rename,因为 badge 在右侧)
        if event.button() == Qt.MouseButton.LeftButton and self._is_coverage_node():
            rect = self._coverage_badge_rect()
            if rect.contains(event.pos()):
                self._open_coverage_inspector(event)
                event.accept()
                return
        # title 区域双击 → inline 改名(避开 icon 区,与 paint text_rect 对齐)
        if event.button() == Qt.MouseButton.LeftButton:
            title_text_rect = QRectF(H_PAD, 0, self._width - H_PAD - 40, TITLE_H)
            if title_text_rect.contains(event.pos()):
                self._enter_title_rename()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------
    # Title rename (inline QLineEdit via QGraphicsProxyWidget)
    # ------------------------------------------------------------------
    def set_display_name_override(self, value: str) -> None:
        """覆盖标题(空字符串恢复 manifest 默认)。Undo cmd 走此入口。"""
        new = str(value or "")
        if new == self._display_name_override:
            return
        self._display_name_override = new
        self._title_text = self._resolve_title()
        self.update()

    def _build_title_rename_widgets(self) -> None:
        from PyQt6.QtWidgets import QGraphicsProxyWidget

        edit = _TitleLineEdit(
            on_commit=self._commit_title_rename,
            on_cancel=self._cancel_title_rename,
        )
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        size = int(Config.get_font_size("size_small", 12))
        bg = Config.get_color("btn_1", "#343636")
        fg = Config.get_color("main_t1", "#D6D3C7")
        bd = Config.get_color("checked_1", "#67CCF5")
        edit.setStyleSheet(
            f"QLineEdit {{ background: {bg}; color: {fg}; border: 1px solid {bd};"
            f" border-radius: 3px; padding: 1px 4px;"
            f" font-family: '{family}'; font-size: {size}px; font-weight: bold; }}"
        )
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(edit)
        proxy.setVisible(False)
        # 定位 = title text_rect
        proxy.setPos(H_PAD - 4, 3)
        proxy.resize(self._width - H_PAD - 40 + 8, TITLE_H - 6)
        self._title_rename_edit = edit
        self._title_rename_proxy = proxy

    def _enter_title_rename(self) -> None:
        if self._title_rename_edit is None:
            self._build_title_rename_widgets()
        if self._title_rename_edit is None or self._title_rename_proxy is None:
            return
        self._title_rename_edit.setText(self._title_text)
        self._editing_title = True
        self._title_rename_proxy.setVisible(True)
        self._title_rename_edit.setFocus(Qt.FocusReason.MouseFocusReason)
        self._title_rename_edit.selectAll()
        self.update()

    def _cancel_title_rename(self) -> None:
        if not self._editing_title or self._title_rename_edit is None:
            return
        self._editing_title = False
        if self._title_rename_proxy is not None:
            self._title_rename_proxy.setVisible(False)
        self.update()

    def _commit_title_rename(self) -> None:
        if not self._editing_title or self._title_rename_edit is None:
            return
        new_text = (self._title_rename_edit.text() or "").strip()
        old_override = self._display_name_override
        self._editing_title = False
        if self._title_rename_proxy is not None:
            self._title_rename_proxy.setVisible(False)
        # 用户清空 → 回到 manifest default(_display_name_override = "")
        if new_text == old_override or (not new_text and not old_override):
            self.update()
            return
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None or getattr(page, "_in_undo", False):
            self.set_display_name_override(new_text)
            return
        from .undo import RenameCmd
        page._undo_stack.push(
            RenameCmd(page, "node", self.node_id, old_override, new_text)
        )

    def _open_coverage_inspector(self, event) -> None:
        """Launch the Coverage Inspector dialog for this node.

        Late import keeps the dialog code out of the cold-paint path.
        """
        try:
            from .edit_dialogs import open_coverage_inspector
        except Exception as exc:  # pragma: no cover — degrade gracefully
            try:
                from unitport_sdk import log_warning
                log_warning(
                    f"[canvas] Coverage Inspector unavailable: {exc!r}"
                )
            except Exception:
                pass
            return
        try:
            open_coverage_inspector(event, self)
        except Exception as exc:  # pragma: no cover
            try:
                from unitport_sdk import log_warning
                log_warning(
                    f"[canvas] Coverage Inspector failed to open: {exc!r}"
                )
            except Exception:
                pass

    def _paint_port_rows(self, painter: QPainter, tier: int) -> None:
        """画端口行 tint + label（label 文字色 = 端口 type 色）."""
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        draw_label = True  # 端口名标签：任何 tier 都画

        # input 行（仅可见端口参与排布与绘制）
        visible_in = [p for p in self._in_ports if p.isVisible()]
        visible_out = [p for p in self._out_ports if p.isVisible()]

        if self._in_y is not None:
            for i, port in enumerate(visible_in):
                y = self._in_y + i * PORT_ROW_H
                tint = QColor(port.color)
                tint.setAlpha(18)
                painter.setBrush(QBrush(tint))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(0, y, self._width, PORT_ROW_H))
                if draw_label:
                    label = port._display_label
                    if port.spec.optional:
                        label = f"{label}  {self._opt_suffix}"
                    painter.setPen(QPen(QColor(port.color)))
                    label_rect = QRectF(
                        PORT_R + LABEL_GAP,
                        y,
                        self._width - (PORT_R + LABEL_GAP) - H_PAD,
                        PORT_ROW_H,
                    )
                    painter.drawText(
                        label_rect,
                        int(
                            Qt.AlignmentFlag.AlignVCenter
                            | Qt.AlignmentFlag.AlignLeft
                        ),
                        label,
                    )

        # output 行
        if self._out_y is not None:
            text_right = self._width - PORT_MARGIN - PORT_R - 7
            for i, port in enumerate(visible_out):
                y = self._out_y + i * PORT_ROW_H
                tint = QColor(port.color)
                tint.setAlpha(18)
                painter.setBrush(QBrush(tint))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(0, y, self._width, PORT_ROW_H))
                if draw_label:
                    painter.setPen(QPen(QColor(port.color)))
                    label_rect = QRectF(
                        H_PAD,
                        y,
                        text_right - H_PAD,
                        PORT_ROW_H,
                    )
                    painter.drawText(
                        label_rect,
                        int(
                            Qt.AlignmentFlag.AlignVCenter
                            | Qt.AlignmentFlag.AlignRight
                        ),
                        port._display_label,
                    )

    def itemChange(self, change, value):
        # 节点位置改变时，所有连接它的边都需要重画
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._mark_edges_dirty()
            # 选区悬浮工具栏跟随被拖动的节点
            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is not None and hasattr(page, "_reposition_selection_toolbar"):
                try:
                    page._reposition_selection_toolbar()
                except Exception:  # pragma: no cover
                    pass
        return super().itemChange(change, value)

    # ---- Undo 去抖：mousePress 记 old pos，mouseRelease 终态 push MoveNodeCmd ----

    def mousePressEvent(self, event):
        # Click outside the title text rect while renaming → commit first.
        # NodeItem lacks ItemIsFocusable, so a click on its own body would not
        # transfer scene focus and the editor would otherwise stay parked.
        if self._editing_title:
            title_text_rect = QRectF(H_PAD, 0, self._width - H_PAD - 40, TITLE_H)
            if not title_text_rect.contains(event.pos()):
                self._commit_title_rename()

        # 收纳态:点击 chevron 按钮 → 切换为展开,直接吞掉 event,不进入 drag/move 路径
        if (
            self._collapsed
            and self._chevron_rect is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._chevron_rect.contains(event.pos())
        ):
            sc = self.scene()
            page = getattr(sc, "_page_ref", None) if sc is not None else None
            if page is not None and hasattr(page, "_toggle_collapse_for_node"):
                page._toggle_collapse_for_node(self.node_id)
            event.accept()
            return

        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None or getattr(page, "_in_undo", False):
            return
        moves: List[tuple] = []
        seen = set()
        for it in sc.selectedItems():
            if isinstance(it, NodeItem):
                moves.append((it.node_id, QPointF(it.pos())))
                seen.add(it.node_id)
        if self.node_id not in seen:
            moves.append((self.node_id, QPointF(self.pos())))
        page._pending_move = moves

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is None:
            return
        moves = getattr(page, "_pending_move", None)
        if not moves:
            return
        page._pending_move = None
        if getattr(page, "_in_undo", False):
            return
        actual = []
        for inst, old in moves:
            item = page._instances.get(inst)
            if item is None:
                continue
            new = item.pos()
            if QPointF(old) != QPointF(new):
                actual.append((inst, QPointF(old), QPointF(new)))
        if not actual:
            return
        from .undo import MoveNodeCmd
        page._undo_stack.push(MoveNodeCmd(page, actual))
        # 节点拖动结束 → 重算所有 group 的几何成员(节点可能被拖入/出 group 框)
        if hasattr(page, "_recompute_all_group_memberships"):
            try:
                page._recompute_all_group_memberships()
            except Exception:  # pragma: no cover
                pass

    # ---- 内部 ----

    def _resolve_title(self) -> str:
        # 用户改名 override 优先于 manifest.display_name_key 的 tr() 结果。
        if self._display_name_override:
            return self._display_name_override
        # display_name_key → tr()；缺翻译时回落到 manifest.id
        # display_name_key → tr(); fall back to manifest.id when missing.
        key = self.manifest.display_name_key or ""
        if key:
            v = tr(key, self.manifest.id)
            if v:
                return v
        return self.manifest.id

    def retranslate(self) -> None:
        """Re-resolve the cached title under the current language and repaint.

        语种切换时由 CanvasPage 调用：title 文本是 __init__ 一次性 cache 在
        _title_text 里的，不重算就一直显示旧语种。同时刷新 _opt_suffix 与所有
        子 PortItem 的 _display_label —— 端口标签由本 NodeItem 在 _paint_port_rows
        里读 port._display_label 绘制，必须先让子端口重算才能 repaint 正确。"""
        dirty = False
        new = self._resolve_title()
        if new != self._title_text:
            self._title_text = new
            dirty = True
        new_opt = tr("canvas.port.optional", "(opt)")
        if new_opt != self._opt_suffix:
            self._opt_suffix = new_opt
            dirty = True
        for port in list(self.iter_ports()):
            if port.retranslate():
                dirty = True
        if dirty:
            self.update()

    def _title_bar_color(self) -> QColor:
        # Per-instance override 优先。非空时直接作为标题栏色,不再查 layer slot。
        # 非法 hex 会让 QColor.isValid() == False —— 此时退回默认 layer 色,
        # 等价于 "用户输入坏数据 → 当作没设过 override"。
        if self._color_override:
            c = QColor(self._color_override)
            if c.isValid():
                return c
        layer = (self.manifest.layer or "").upper()
        slot_map = {
            "A": "canvas_layer_a",
            "B": "canvas_layer_b",
            "C": "canvas_layer_c",
            "D": "canvas_layer_d",
            "IL": "canvas_layer_il",
        }
        slot = slot_map.get(layer, "btn_1")
        return QColor(Config.get_color(slot, "#343636"))

    # ------------------------------------------------------------------
    # Per-instance color override (SelectionToolbar 色盘按钮 → page.py 入口)
    # ------------------------------------------------------------------
    def set_color_override(self, value: Optional[str]) -> None:
        """设置/清空 per-canvas 颜色覆写。

        - 传入 ``""`` / ``None`` → 清空覆写,回到默认 layer 色
        - 传入 ``"#RRGGBB"`` 等合法 hex → 覆写标题栏色
        - 仅影响本画布上这一个 NodeItem;持久化由 page.py 序列化层负责
        """
        new = "" if value is None else str(value).strip()
        if new == self._color_override:
            return
        self._color_override = new
        self.update()

    @property
    def color_override(self) -> str:
        return self._color_override

    def _build_ports(self) -> None:
        for spec in self.manifest.inputs:
            p = PortItem(spec, direction="in", parent=self)
            self._in_ports.append(p)
        for spec in self.manifest.outputs:
            p = PortItem(spec, direction="out", parent=self)
            self._out_ports.append(p)
        # Initial conditional_on visibility for ports (PortSpec.meta).
        # Mirrors the param-row pass in _build_param_rows so PPO-mode
        # nodes start without the AMP-only ports visible.
        for p in self.iter_ports():
            try:
                ParamRow._eval_conditional  # noqa — sentinel; reuse static helper
                visible = ParamRow._eval_conditional(
                    getattr(p.spec, "meta", None), self.params
                )
                if not visible:
                    p.setVisible(False)
            except Exception:
                pass

    def _build_param_rows(self) -> None:
        for spec in self.manifest.parameters:
            # ``meta.hidden = true`` 的参数仅在 IR 中存在，不渲染任何 UI 行。
            # 用于 base_asset.checkpoint_file 这类"完全由其它 picker 写入、用户
            # 不直接编辑"的字段——匹配 DEMO ``NODE_UI_ROWS`` 不声明该行的行为。
            if isinstance(spec.meta, dict) and spec.meta.get("hidden"):
                continue
            row = create_param_row(spec, self, self._width)
            self._param_rows.append(row)
            # Variable-height rows (e.g. BodyMappingTableRow) notify us when
            # they grow / shrink so we can reflow downstream rows + body height.
            try:
                row.on_height_changed(lambda _h, _r=row: self._on_row_height_changed(_r))
            except Exception:
                pass
        # Initial conditional_on visibility pass — rows whose ParamSpec.meta
        # contains a conditional_on hint may start hidden depending on
        # gating param defaults (e.g. il_ppo_trainer.training_mode=PPO hides
        # the 7 AMP-only rows).
        for row in self._param_rows:
            try:
                row.update_visibility(self.params)
            except Exception:
                pass

    def _compute_layout(self) -> None:
        """段落式布局：input 段 → 参数段 → output 段；段间填 SEP_H 分隔条.

        参数段内每行使用 ``ParamRow.preferred_height()`` 累加位置（默认 24，
        BodyMappingTableRow 这类 inline 整行 widget 会返回更大值）。

        收纳态(``self._collapsed=True``): ParamRow 全部 hide,布局变成
        ``Title → in → [chevron 行] → out``,chevron 命中矩形 ``_chevron_rect``
        交给 mousePressEvent 处理点击。
        """
        # 收纳态:隐藏未 pin 的 ParamRow,保留 ``meta.pin_on_collapse=true`` 的"关键行"。
        # 展开态:按每行 ``ParamSpec.meta['conditional_on']`` 重新评估可见性,
        # 而非粗暴 setVisible(True) —— 后者会清掉 update_visibility 刚设的隐藏
        # (例:base_asset.checkpoint_id 应只在 start_point=__load__ 时显示,
        # load_mode 应只在 start_point != __new__ 时显示)。
        if self._collapsed:
            for row in self._param_rows:
                row.setVisible(bool(getattr(row, "is_pinned_on_collapse", False)))
            visible_param_rows: List[ParamRow] = [r for r in self._param_rows if r.isVisible()]
        else:
            for row in self._param_rows:
                row.setVisible(ParamRow._eval_conditional(row.spec.meta, self.params))
            visible_param_rows = [r for r in self._param_rows if r.isVisible()]

        visible_in_ports = [p for p in self._in_ports if p.isVisible()]
        visible_out_ports = [p for p in self._out_ports if p.isVisible()]

        sections: List[str] = []
        if visible_in_ports:
            sections.append("in")
        # 收纳态:有 pinned 行就先画 params 段;无论是否有 pinned,chevron 段都画。
        # 展开态:仅当存在 visible param rows 时画 params 段。
        if visible_param_rows:
            sections.append("params")
        if self._collapsed:
            sections.append("chevron")
        if visible_out_ports:
            sections.append("out")

        y = TITLE_H
        self._sep_ys = []
        self._in_y = self._params_y = self._out_y = None
        self._chevron_y = None
        params_h = sum(
            float(getattr(r, "preferred_height", lambda: PARAM_ROW_H)())
            for r in visible_param_rows
        )
        for idx, kind in enumerate(sections):
            if idx > 0:
                self._sep_ys.append(y)
                y += SEP_H
            if kind == "in":
                self._in_y = y
                y += len(visible_in_ports) * PORT_ROW_H
            elif kind == "params":
                self._params_y = y
                y += params_h
            elif kind == "chevron":
                self._chevron_y = y
                y += EXPAND_BTN_ROW_H
            elif kind == "out":
                self._out_y = y
                y += len(visible_out_ports) * PORT_ROW_H

        body_h = y - TITLE_H if y > TITLE_H else MIN_BODY_H
        self._body_h = max(MIN_BODY_H, body_h)

        # chevron 命中矩形(item-local) —— 仅收纳态有效
        if self._collapsed and self._chevron_y is not None:
            self._chevron_rect = QRectF(
                self._width * 0.5 - 14,
                self._chevron_y + 2,
                28, EXPAND_BTN_ROW_H - 4,
            )
        else:
            self._chevron_rect = None

        # 端口位置：圆点中心在所属行的中线（仅可见端口参与排布；隐藏端口
        # 仍持原位但 setVisible(False) 不绘制 + 边连随之消失）
        if self._in_y is not None:
            for i, p in enumerate(visible_in_ports):
                p.setPos(0.0, self._in_y + (i + 0.5) * PORT_ROW_H)
        if self._out_y is not None:
            for i, p in enumerate(visible_out_ports):
                p.setPos(self._width, self._out_y + (i + 0.5) * PORT_ROW_H)
        if self._params_y is not None:
            cursor_y = self._params_y
            for row in self._param_rows:
                if not row.isVisible():
                    continue
                row.setPos(0.0, cursor_y)
                cursor_y += float(
                    getattr(row, "preferred_height", lambda: PARAM_ROW_H)()
                )

        # Reconcile dynamic inline-bound ports (e.g. per-item reward inputs on
        # training_motion). Done after rows have been positioned so anchors
        # translate to the correct node-local coordinates.
        # 收纳态:节点没有参数行可见,inline ports 也跟着退出(rebuild 内部按
        # row.isVisible() 过滤,自然得到空集 → 现有 inline port 被清理)。
        self._rebuild_inline_ports()

    def _rebuild_inline_ports(self) -> None:
        """Sync ``_inline_in_ports`` with each row's iter_inline_port_anchors().

        Diff-based: existing PortItems are reused (preserves their attached
        ConnectionItems); ports no longer yielded by any row are removed and
        their connections detached. Newly yielded ports are created in place.
        Position = row.pos() + anchor_local. All inline ports are inputs
        (output-side inline ports are not part of v2's contract).
        """
        wanted: Dict[str, tuple] = {}  # name → (spec, node_local_pt)
        for row in self._param_rows:
            if not row.isVisible():
                continue
            anchors = getattr(row, "iter_inline_port_anchors", None)
            if not callable(anchors):
                continue
            try:
                row_pos = row.pos()
                for spec, anchor in anchors():
                    node_pt = QPointF(
                        row_pos.x() + anchor.x(),
                        row_pos.y() + anchor.y(),
                    )
                    wanted[spec.name] = (spec, node_pt)
            except Exception:
                continue

        # Remove orphaned inline ports (no longer wanted). Detaching
        # connections first prevents stale ConnectionItems pointing at a
        # freed PortItem.
        for name in list(self._inline_in_ports.keys()):
            if name in wanted:
                continue
            port = self._inline_in_ports.pop(name)
            for conn in tuple(port.connections):
                try:
                    sc = conn.scene()
                    if sc is not None:
                        sc.removeItem(conn)
                except Exception:
                    pass
            sc = port.scene()
            if sc is not None:
                try:
                    sc.removeItem(port)
                except Exception:
                    pass

        # Create / reposition wanted ports.
        for name, (spec, pt) in wanted.items():
            port = self._inline_in_ports.get(name)
            if port is None:
                port = PortItem(spec, direction="in", parent=self)
                self._inline_in_ports[name] = port
            port.setPos(pt)

    def _on_row_height_changed(self, row: "ParamRow") -> None:
        """A param row reported a new preferred_height — reflow + grow node."""
        self.prepareGeometryChange()
        self._compute_layout()
        self._bounding = QRectF(0, 0, self._width, TITLE_H + self._body_h)
        self._mark_edges_dirty()
        self.update()

    def apply_canvas_backend(self, kind: str) -> None:
        """Inject the canvas-bound backend kind into ``self.params['backend']``.

        Backend is no longer a user-toggleable param on rewards / terminations
        / domain_rand — it derives from the parent CanvasPage's bound
        backend via ``registers.backends.node_backend_kind``. Called by
        CanvasPage after ``bind_backend`` / ``spawn_node`` /
        ``from_workflow_dict`` so ``conditional_on`` visibility evaluates
        against the canvas's kind, not a user-edited stale value.

        Idempotent: no-ops when the value is already correct.
        """
        if not isinstance(kind, str) or not kind:
            return
        if self.params.get("backend") == kind:
            return
        self.params["backend"] = kind
        self._refresh_conditional_visibility()
        # Registry-aware inline rows (RegistryModuleInlineRow, TrainingMotionRow,
        # ...) cache per-key TaskModuleItem lookups in ``_row_payloads`` at row
        # ``__init__`` time — which happens inside ``NodeItem.__init__``, BEFORE
        # CanvasPage gets a chance to inject the canvas-bound backend here.
        # Without re-resolution, those caches keep the items they resolved with
        # the default ``"sb3"`` backend; IL-only keys (e.g. ``time_out`` /
        # ``base_height`` / ``bad_orientation``) look up to None, slider ranges
        # fall back to [-10, +10], and handles render in nonsense positions.
        # The first user-triggered ``set_value`` (e.g. dragging any slider)
        # then re-runs ``_refresh_payloads`` with the now-correct backend, all
        # ranges snap to their real bounds, and every handle visibly jumps.
        # Surface the refresh here so the visual state is correct from the
        # first paint.
        for row in self._param_rows:
            cb = getattr(row, "_refresh_payloads", None)
            if callable(cb):
                try:
                    cb()
                except Exception:
                    pass
        self.update()

    # ------------------------------------------------------------------
    # 收纳 / 禁用 公共 API
    # ------------------------------------------------------------------
    def set_collapsed(self, value: bool) -> None:
        """切换收纳态。仅视觉影响:布局重排 + ParamRow 隐藏/恢复 + 边线重连。"""
        new = bool(value)
        if new == self._collapsed:
            return
        self._collapsed = new
        self.prepareGeometryChange()
        self._compute_layout()
        self._bounding = QRectF(0, 0, self._width, TITLE_H + self._body_h)
        self._mark_edges_dirty()
        self.update()
        # 节点尺寸变了 → 工具栏需要重新定位 + icon mode 切换
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is not None and hasattr(page, "_reposition_selection_toolbar"):
            try:
                page._reposition_selection_toolbar()
            except Exception:  # pragma: no cover
                pass

    def set_disabled(self, value: bool) -> None:
        """切换禁用态(ComfyUI bypass)。仅视觉影响 + 编译时通过 lowering.py 过滤。"""
        new = bool(value)
        if new == self._disabled:
            return
        self._disabled = new
        self.update()
        # 连到本节点的边需要刷新(变灰 / 复原)
        for p in self.iter_ports():
            for c in tuple(p.connections):
                c.update()
        # toolbar 选区可能不变但 disable 模式要切 icon
        sc = self.scene()
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        if page is not None and hasattr(page, "_reposition_selection_toolbar"):
            try:
                page._reposition_selection_toolbar()
            except Exception:  # pragma: no cover
                pass

    def on_param_changed(self, key: str, value: object) -> None:
        """Sibling rows can react to a parameter commit (e.g. BodyMappingTableRow
        watches asset_id changes). Best-effort dispatch — never raises.

        Also re-evaluates ``ParamSpec.meta['conditional_on']`` on every row so
        gating params (e.g. il_ppo_trainer.training_mode) flip dependent rows
        in or out of view. When any row's visibility changes, the node body
        is re-laid-out + edges marked dirty so connections track the new
        port positions.

        Additionally rebuilds inline (row-bound) ports so dict-shaped params
        like ``training_motion.training_items`` can add / remove per-item
        PortItems on the fly (enable an item → reward_pipe input appears;
        disable / rename → it disappears + its connections detach).
        """
        for row in self._param_rows:
            cb = getattr(row, "maybe_refresh_for_asset_change", None)
            if callable(cb):
                try:
                    cb()
                except Exception:
                    pass
        self._refresh_conditional_visibility()
        try:
            self._rebuild_inline_ports()
        except Exception:
            pass
        self._mark_edges_dirty()
        # Canvas-derived-keys rule: when a Robot Node's asset_id changes,
        # every downstream ActorSetting whose init_joint_angles key set
        # is derived from that SKU must auto-reconcile. We dispatch from
        # here so any pathway that mutates Robot Node params (UI edit,
        # undo/redo, programmatic load) goes through the same hook.
        try:
            if (
                str(getattr(self.manifest, "id", "") or "") == "robot"
                and str(key) == "asset_id"
            ):
                from .param_rows import (
                    _reconcile_downstream_actor_settings,
                )
                _reconcile_downstream_actor_settings(self)
        except Exception:
            pass

    def _refresh_conditional_visibility(self) -> None:
        """Apply ``ParamSpec.meta['conditional_on']`` and ``PortSpec.meta
        ['conditional_on']`` visibility + reflow.

        Iterates every param row + every port (cheap — most have no
        conditional_on and short-circuit). If any flipped, recomputes
        layout once and marks all attached edges dirty.
        """
        flipped = False
        # Param rows
        for row in self._param_rows:
            try:
                if row.update_visibility(self.params):
                    flipped = True
            except Exception:
                pass
        # Ports — share the same _eval_conditional helper (PortSpec.meta
        # mirrors ParamSpec.meta schema). Hidden ports' connections track
        # their port's visibility so dangling edges don't render.
        for port in self.iter_ports():
            try:
                meta = getattr(port.spec, "meta", None)
                visible = ParamRow._eval_conditional(meta, self.params)
                if visible == port.isVisible():
                    continue
                port.setVisible(visible)
                # Sync attached connections so they don't render against an
                # invisible port. ConnectionItem(s) live in the scene, not
                # under the port — explicit sync needed.
                for c in tuple(port.connections):
                    try:
                        c.setVisible(visible and self._other_endpoint_visible(c, port))
                    except Exception:
                        pass
                flipped = True
            except Exception:
                pass
        if not flipped:
            return
        self.prepareGeometryChange()
        self._compute_layout()
        self._bounding = QRectF(0, 0, self._width, TITLE_H + self._body_h)
        self._mark_edges_dirty()
        self.update()

    @staticmethod
    def _other_endpoint_visible(connection, this_port) -> bool:
        """Return True if the other side of *connection* is on a visible port.

        A connection is only renderable when *both* its endpoints are visible
        (this port flipping back to visible alone isn't enough if the peer
        node still has its port hidden).
        """
        for p in (
            getattr(connection, "src_port", None),
            getattr(connection, "dst_port", None),
            getattr(connection, "source", None),
            getattr(connection, "target", None),
        ):
            if p is None or p is this_port:
                continue
            try:
                return bool(p.isVisible())
            except Exception:
                return True
        return True

    def _mark_edges_dirty(self) -> None:
        for p in self.iter_ports():
            for c in tuple(p.connections):
                c.mark_dirty()


# =============================================================================
# PlaceholderNodeItem —— 缺失 manifest 的降级渲染（红框 + Missing: <schema_id>）
# =============================================================================

class PlaceholderNodeItem(QGraphicsItem):
    """加载画布时遇到 ``schema_id`` 没在 registry 注册的节点，构造一个 placeholder
    保留位置 + 原始 params + ui 数据，``to_workflow_dict`` 时原样吐回，避免数据丢失。

    视觉契约：
    - 标题 ``"Missing: <schema_id>"``
    - 红色边框（``canvas_placeholder_border`` 主题槽，缺则用 ``danger_zone`` 兜底）
    - 文字色（``canvas_placeholder_text`` 主题槽）

    与 NodeItem 的鸭子类型对齐（让 page 的 ``_instances`` / ``to_workflow_dict``
    / 选择 / 移动 / Undo 路径无差别处理）：
        node_id / params / pos() / width / height / iter_ports() / setSelected /
        is_placeholder = True / original_dict 字段
    """

    is_placeholder = True

    def __init__(
        self,
        schema_id: str,
        instance_id: str,
        original_dict: Optional[dict] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.schema_id = schema_id
        self.node_id = instance_id
        # 原始 IR-shape dict（含 ui / params / kind / opaque_code）；to_workflow_dict
        # 时只更新 ui.x / ui.y，其余字段一字不动地透传，做到数据零丢失。
        self.original_dict: dict = dict(original_dict or {})
        # 兼容 NodeItem 的 params 字段（少量代码路径会读 .params 直接取值）
        params_raw = self.original_dict.get("params") or {}
        self.params: dict = {
            k: (v.get("value") if isinstance(v, dict) else v)
            for k, v in params_raw.items()
        }

        ui = self.original_dict.get("ui") or {}
        self._width = float(ui.get("width") or NODE_W)
        self._height = float(ui.get("height") or 56.0)
        self._title_text = f"Missing: {schema_id}"

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setData(ROLE_KIND, "node")  # 与 NodeItem 一致让命中测试一致
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        self._bounding = QRectF(0, 0, self._width, self._height)

    # ---- NodeItem 鸭子 API ----

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def iter_ports(self) -> Iterable["PortItem"]:
        return iter(())

    def port(self, name: str, direction: str = "in") -> Optional["PortItem"]:
        return None

    # ---- Qt overrides ----

    def boundingRect(self) -> QRectF:
        return self._bounding.adjusted(-3, -3, 3, 3)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = QColor(Config.get_color("canvas_placeholder_bg", "#2A1A1A"))
        border = QColor(Config.get_color(
            "canvas_placeholder_border",
            Config.get_color("danger_zone", "#FF6B6B"),
        ))
        text_color = QColor(Config.get_color("canvas_placeholder_text", "#FFB3B3"))

        # 1. body
        painter.setBrush(QBrush(bg))
        pen = QPen(border, 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(self._bounding, CORNER_R, CORNER_R)

        # 2. text
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            self._bounding.adjusted(H_PAD, 0, -H_PAD, 0),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._title_text,
        )

        # 3. 选中虚线边框
        if self.isSelected():
            sel_pen = QPen(QColor(255, 255, 255, 210), 1.5)
            sel_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                self._bounding.adjusted(0.75, 0.75, -0.75, -0.75),
                CORNER_R,
                CORNER_R,
            )

    # 与 NodeItem 一致：拖动结束触发 MoveNodeCmd
    mousePressEvent = NodeItem.mousePressEvent  # type: ignore[assignment]
    mouseReleaseEvent = NodeItem.mouseReleaseEvent  # type: ignore[assignment]


# =============================================================================
# PortItem
# =============================================================================

class PortItem(QGraphicsItem):
    """端口圆点 / Port circle item.

    自绘 ``QGraphicsItem``（不继承 ``QGraphicsEllipseItem`` —— 后者每帧都重设
    brush/pen，自绘更省）；boundingRect 与 shape 保留 hit margin 让点击宽容些。
    """

    def __init__(
        self,
        spec: PortSpec,
        direction: str,
        parent: "NodeItem",
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.port_name = spec.name
        self.port_type = spec.type
        self.direction = direction  # "in" | "out"
        self._connections: List["ConnectionItem"] = []
        # i18n display label cached at construction; NodeItem reads it in
        # _paint_port_rows. retranslate() rebuilds on language change.
        self._display_label: str = self._resolve_label()

        # 视觉描述统一走 SDK make_port_visual，保留 self._color 兼容老代码
        self._visual = make_port_visual(spec)
        self._color: QColor = self._visual["color"]
        self._hover = False
        self._snap_status: str = "normal"  # "normal" | "snap_valid" | "snap_invalid"

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 与 DEMO 端口协议对齐
        self.setData(ROLE_KIND, "port")
        self.setData(ROLE_DIR, direction)
        self.setData(ROLE_CONN, self._connections)
        self.setData(ROLE_NAME, spec.name)
        self.setData(ROLE_META, {
            "channel": self._visual["channel"],
            "data_type": spec.type,
            "max_connections": None if spec.multi else 1,
            "type_color": self._color.name(),
        })

    # ---- 公开 API ----

    @property
    def connections(self) -> List["ConnectionItem"]:
        return self._connections

    @property
    def color(self) -> QColor:
        return QColor(self._color)

    def add_connection(self, c: "ConnectionItem") -> None:
        if c not in self._connections:
            self._connections.append(c)

    def remove_connection(self, c: "ConnectionItem") -> None:
        if c in self._connections:
            self._connections.remove(c)

    def is_input(self) -> bool:
        return self.direction == "in"

    def parent_node(self) -> Optional["NodeItem"]:
        p = self.parentItem()
        return p if isinstance(p, NodeItem) else None

    def _resolve_label(self) -> str:
        """Translate spec.name via node.<id>.port.<name>; fall back to spec.name.

        和 NodeItem._resolve_title 一个套路：i18n 缺翻译时回落 id，保证视觉不空。"""
        parent = self.parent_node()
        node_id = parent.manifest.id if parent is not None else ""
        if node_id:
            return tr(f"node.{node_id}.port.{self.spec.name}", self.spec.name)
        return self.spec.name

    def retranslate(self) -> bool:
        """Re-resolve cached display label under current language.

        Returns True iff label actually changed (caller—NodeItem.retranslate—
        uses this to decide whether to schedule a repaint)."""
        new = self._resolve_label()
        if new != self._display_label:
            self._display_label = new
            return True
        return False

    # ---- Qt overrides ----

    def boundingRect(self) -> QRectF:
        r = PORT_HIT_R
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QPointF(0, 0), PORT_HIT_R, PORT_HIT_R)
        return path

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        tier = tier_for_painter(painter)
        # 端口圆点：任何 tier 都画（禁止提前消失）

        # 状态优先级：snap_valid / snap_invalid > hover > normal
        if self._snap_status == "snap_valid":
            status = "snap_valid"
        elif self._snap_status == "snap_invalid":
            status = "snap_invalid"
        elif self._hover:
            status = "hover"
        else:
            status = "normal"

        # T1 Minimap: 半径减半（仍可见但不抢眼）；其他 tier 用视觉默认半径
        radius = PORT_R * 0.6 if tier == TIER_MINIMAP else PORT_R
        border = QColor(Config.get_color("border_1", "#444444"))

        draw_dot_port(
            painter,
            QPointF(0.0, 0.0),
            self._color,
            radius=radius,
            status=status,  # type: ignore[arg-type]
            border_color=border,
        )
        # NOTE: 端口 label 由父 NodeItem 在 _paint_port_rows 里统一绘制
        # （需要按 type_color 着色 + 行 tint 背景），这里只画圆点本身。

    def set_snap_status(self, status: str) -> None:
        """drag-connect 期间由 PortDragController 调，触发重绘.

        ``"normal"`` / ``"snap_valid"`` / ``"snap_invalid"``。
        """
        if status != self._snap_status:
            self._snap_status = status
            self.update()

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)


# =============================================================================
# ConnectionItem
# =============================================================================

class ConnectionItem(QGraphicsPathItem):
    """节点连线 / Edge between two ports.

    Bezier 路径：起点水平切线 + 终点水平切线，控制点 dx*0.5 偏移（DEMO 一致）。

    **Dirty flag**：``update_path()`` 不脏直接 return；NodeItem 移动时会调
    ``mark_dirty()``，下一次绘制前才真正重算 path。这是 DEMO ConnectionItem
    缺失的关键优化（DEMO 每次 move 都 setPath，对 50 节点 200+ 边明显）。
    """

    def __init__(
        self,
        src_port: PortItem,
        dst_port: PortItem,
        edge_id: Optional[str] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        import uuid as _uuid
        self.edge_id: str = edge_id or f"e_{_uuid.uuid4().hex[:12]}"
        self.src_port = src_port
        self.dst_port = dst_port
        self._dirty = True
        self._hover = False
        self._color = src_port.color

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1.0)  # 在 NodeItem 之下
        self.setData(ROLE_KIND, "connection")

        src_port.add_connection(self)
        dst_port.add_connection(self)
        self._refresh_pen()
        self.update_path()

    # ---- 公开 API ----

    def mark_dirty(self) -> None:
        """标记 path 需要重算；下一次 update() 前若被询问 path 才重建."""
        self._dirty = True
        self.update_path()

    def update_path(self) -> None:
        """重建 Bezier path；不脏直接 return.

        显式调 ``prepareGeometryChange()`` —— 通知 Qt 旧 boundingRect 区域需要
        invalidate，否则 ``SmartViewportUpdate`` 模式下节点拖动时会留残影边。
        """
        if not self._dirty:
            return
        # 关键：先通知 Qt 我们要改 bounding，让 scene 把旧位置的 region 标记为脏
        self.prepareGeometryChange()
        s = self.src_port.scenePos()
        e = self.dst_port.scenePos()
        dx = e.x() - s.x()
        path = QPainterPath()
        path.moveTo(s)
        path.cubicTo(
            QPointF(s.x() + dx * 0.5, s.y()),
            QPointF(e.x() - dx * 0.5, e.y()),
            e,
        )
        self.setPath(path)
        self._dirty = False
        # 强制重绘当前 boundingRect 区域，弥补 SmartViewportUpdate 漏判
        self.update()

    def boundingRect(self) -> QRectF:
        """从端点 + pen padding 计算，不依赖 path 内部状态.

        默认 ``QGraphicsPathItem.boundingRect()`` 会从 ``self.path()`` 推算，
        但 LOD T1 的直线 paint 不走 path、T0 完全不画——boundingRect 与实际 paint
        区域可能脱节，导致残影。这里用一个**保守一点的并集**：包含 src/dst
        scenePos 与 path bounding 的最大外包 + pen 宽度 padding。
        """
        path_rect = self.path().boundingRect() if not self.path().isEmpty() else QRectF()
        s = self.src_port.scenePos()
        e = self.dst_port.scenePos()
        endpoints = QRectF(s, e).normalized()
        rect = path_rect.united(endpoints) if not path_rect.isEmpty() else endpoints
        pad = max(EDGE_PEN_HOVER, self.pen().widthF()) * 0.5 + 2.0
        return rect.adjusted(-pad, -pad, pad, pad)

    def shape(self) -> QPainterPath:
        """供命中测试 / for hit-testing — 让 stroke 范围内可点中边."""
        from PyQt6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(max(8.0, self.pen().widthF() + 6.0))
        return stroker.createStroke(self.path())

    def disconnect(self) -> None:
        """从场景移除并清理两端 port 的 connections 列表."""
        self.src_port.remove_connection(self)
        self.dst_port.remove_connection(self)
        sc = self.scene()
        if sc is not None:
            sc.removeItem(self)

    # ---- LOD-aware paint（覆盖 QGraphicsPathItem 默认 paint）----

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        tier = tier_for_painter(painter)
        # 连线永不消失——任何 tier 都画。LOD 仅切换抗锯齿（不影响形状），
        # 严禁因缩放把 Bezier 退化为直线。
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing, tier >= TIER_WORKING
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = self.pen()
        # Bypass 视觉:任一端点节点 disabled → 边变灰 + 减透明度。
        # 仅 transient 替换 paint 用的 pen,不持久化(节点状态变化时 paint 会
        # 重新检查;NodeItem.set_disabled 已经主动调每条挂的边 update())。
        src_node = self.src_port.parent_node() if self.src_port else None
        dst_node = self.dst_port.parent_node() if self.dst_port else None
        if bool(getattr(src_node, "_disabled", False)) or bool(getattr(dst_node, "_disabled", False)):
            dim = QColor(Config.get_color("canvas_edge_disabled", "#5A5A5A"))
            dim.setAlpha(110)
            pen = QPen(dim, pen.widthF())
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        # 任何 tier 都画完整 Bezier（cached path）。
        painter.drawPath(self.path())

    # ---- Qt overrides ----

    def hoverEnterEvent(self, event):
        self._hover = True
        self._refresh_pen()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hover = False
        self._refresh_pen()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._refresh_pen()
        return super().itemChange(change, value)

    # ---- 内部 ----

    def _refresh_pen(self) -> None:
        active = self._hover or self.isSelected()
        width = EDGE_PEN_HOVER if active else EDGE_PEN_DATA
        color = self._color.lighter(140) if active else self._color
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)


# =============================================================================
# 便捷函数 / convenience
# =============================================================================

def connect_ports(
    src: PortItem,
    dst: PortItem,
    edge_id: Optional[str] = None,
) -> ConnectionItem:
    """根据 src/dst 端口建立连线；自动推断方向（out → in）.

    Raises:
        ValueError: 两端方向相同（in→in / out→out）
    """
    if src.direction == dst.direction:
        raise ValueError(
            f"无法连接同方向端口: {src.direction}={src.port_name} → {dst.direction}={dst.port_name}"
        )
    if src.direction == "in":
        src, dst = dst, src
    return ConnectionItem(src, dst, edge_id=edge_id)


__all__ = [
    "NodeItem",
    "PlaceholderNodeItem",
    "PortItem",
    "ConnectionItem",
    "connect_ports",
    "ROLE_KIND",
    "ROLE_DIR",
    "ROLE_CONN",
    "ROLE_NAME",
    "ROLE_META",
    "TITLE_H",
    "NODE_W",
    "PORT_ROW_H",
    "PARAM_ROW_H",
    "SEP_H",
    "MIN_BODY_H",
    "CORNER_R",
    "PORT_R",
    "PORT_HIT_R",
    "EDGE_PEN_DATA",
    "EDGE_PEN_HOVER",
]
