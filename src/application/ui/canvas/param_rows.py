# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.param_rows — 节点参数行（custom paint）/ Node parameter rows.

20 种 ``ParamRow`` 子类，全部 ``QGraphicsItem`` 自绘，绝无 ``QGraphicsProxyWidget``。
编辑触发时**绝不弹 modal QDialog**——所有编辑器走 ``popups.py`` 里的锚定
``Qt.WindowType.Popup``，紧贴行右侧弹出，点击外部即关闭（DEMO ``DataInput`` /
``_RichChoicePopup`` / ``_TextInputPopup`` 行为一致）。

通用 10 种 — 类型驱动（``ParamSpec.widget`` 优先；否则按 ``ParamSpec.type``）：

    bool                  → BoolRow            （单击 toggle）
    int / float           → NumberRow          （双击 → open_number_popup）
    string                → StringRow          （双击 → open_text_popup）
    enum (with choices)   → ChoiceRow          （单击 → open_choice_popup / Node_RichChoicePopup）
    widget="path"         → PathRow            （单击按钮区 → QFileDialog）
    widget="index"        → IndexRow           （单击 → open_number_popup int）
    widget="range"        → RangeRow           （单滑块；行内 draw_range_slider）
    widget="range_list"   → RangeListRow       （[lo, hi] 双滑块；draw_range_slider dual=True）
    widget="badge"        → BadgeRow           （单击 cycle；行内 draw_polarity_badge）
    widget="code", json   → CodeRow            （单击 → open_code_popup）
    widget="table_readonly" → TableReadOnlyRow （不可编辑）

节点专属 10 种 — 弹出 SDK ``Node_*`` 高阶 picker / editor 装在锚定 popup 里：

    widget="picker_scene"            → SceneTypeRow         → Node_SceneTypePicker
    widget="picker_task_type"        → TaskTypeRow          → Node_TaskTypePicker
    widget="picker_start_point"      → StartPointRow        → Node_StartPointChoicePicker
    widget="picker_export_bundle"    → ExportBundleRow      → Node_ExportBundlePicker
    widget="picker_motion_library"   → MotionLibraryRow     → Node_MotionLibraryPicker
    widget="picker_obs_components"   → ObsComponentsRow     → Node_ObsComponentsPicker (multi)
    widget="registry_module"         → RegistryModuleInlineRow (inline rows; click a
                                       value → Node_DataInput; obs scale is per-channel
                                       for multi-axis terms)
    widget="training_items"          → TrainingItemsRow     → Node_TrainingItemEditor
    widget="stage_editor"            → StageEditorRow       → Node_StageSwitchEditorWidget
    widget="curve"                   → CurveRow             → Node_ModuleCurveSlider
    widget="buffer_size"             → BufferSizeRow        → Node_BufferSizePickerWidget

行内绘制走 SDK paint helper（draw_choice_dropdown / draw_edit_button /
draw_polarity_badge / draw_range_slider）+ ``RangeSliderHitTest`` 命中测试。

LOD：T0/T1 不绘；T2 简化（无图标/箭头）；T3 + 注释。
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QMessageBox,
    QStyleOptionGraphicsItem,
    QWidget,
)

from unitport_sdk import (
    Config,
    Node_BufferSizePickerWidget,
    Node_NodeTableRowsWidget,
    Node_RegistryModuleEditor,
    Node_TrainingItemEditor,
    RangeSliderHitTest,
    draw_choice_dropdown,
    draw_edit_button,
    draw_polarity_badge,
    draw_range_slider,
    log_error,
    log_info,
    log_warning,
    setText,
    tr,
)

from application.compiler.nodes import ParamSpec

from .popups import (
    open_choice_popup,
    open_code_popup,
    open_number_array_popup,
    open_number_popup,
    open_path_dialog,
    open_text_popup,
    open_widget_popup,
)
from .lod import (
    TIER_DETAIL,
    TIER_WORKING,
    tier_for_painter,
)


PARAM_ROW_H = 24
KEY_PAD_LEFT = 14
KEY_WIDTH = 96
SEP_X = KEY_PAD_LEFT + KEY_WIDTH + 4
VALUE_PAD_RIGHT = 12


def _view_of(event: QGraphicsSceneMouseEvent) -> Optional[QGraphicsView]:
    """从 GraphicsScene mouse event 找回当前 QGraphicsView / Resolve view from event.

    ``event.widget()`` 在 PyQt6 是触发 paint 的 viewport (QWidget)，
    其 ``parent()`` 才是 QGraphicsView。
    """
    w = event.widget()
    if w is None:
        return None
    if isinstance(w, QGraphicsView):
        return w
    parent = w.parent()
    if isinstance(parent, QGraphicsView):
        return parent
    return None


# =============================================================================
# ParamRow 基类
# =============================================================================

class ParamRow(QGraphicsItem):
    """所有参数行的基类 / Base class for all parameter rows.

    子类必须实现：
        ``_paint_value(painter, tier)`` 画 value 半边
        ``_handle_click(event)`` / ``_handle_double_click(event)`` 至少一个

    数据流：
        owner_node.params[spec.key] ↔ ParamRow._value（双向同步）

    LOD 行为（基类统一）：
        T0/T1 不绘
        T2 标准
        T3 加边框淡化 + 描述 tooltip 准备
    """

    def __init__(
        self,
        spec: ParamSpec,
        owner_node: "NodeItem",
        width: float,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent if parent is not None else owner_node)
        self.spec = spec
        self._owner = owner_node
        self._width = float(width)
        self._height = float(PARAM_ROW_H)
        self._value: Any = self._read_value()
        # i18n display label cached at construction; rebuilt by retranslate()
        # on language change. Same pattern as NodeItem._title_text.
        self._display_label: str = self._resolve_label()
        # 区域级 hover：只在落入 _interactive_regions() 命中的子矩形时被赋值；
        # 整行 hover / 整行光标均已废弃。
        self._hover_rect: Optional[QRectF] = None
        self._cursor_active = False
        self.setAcceptHoverEvents(True)
        # 行不接受 ItemIsMovable —— 节点整体移动；行只响应自己内部点击
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    # ---- i18n ----

    def _resolve_label(self) -> str:
        """Translate spec.key via node.<id>.param.<key>; fall back to spec.key.

        和 NodeItem._resolve_title 一个套路：缺翻译时回落 id，保证视觉不空。"""
        node_id = ""
        try:
            node_id = str(getattr(self._owner.manifest, "id", "") or "")
        except Exception:
            node_id = ""
        if node_id:
            return tr(f"node.{node_id}.param.{self.spec.key}", self.spec.key)
        return self.spec.key

    def _resolve_choice_text(self, raw_value: Any) -> str:
        """Translate an enum/choice raw value via node.<id>.param.<key>.choice.<value>.

        Lookups use ``str(raw_value)`` as the final key segment. Missing key
        falls back to ``str(raw_value)`` so未翻译的枚举仍能可读地显示原 id。
        None / "" returns empty string (matches prior _draw_value_text behaviour
        when value is None)."""
        if raw_value is None or raw_value == "":
            return ""
        node_id = ""
        try:
            node_id = str(getattr(self._owner.manifest, "id", "") or "")
        except Exception:
            node_id = ""
        fallback = str(raw_value)
        if not node_id:
            return fallback
        key = f"node.{node_id}.param.{self.spec.key}.choice.{fallback}"
        return tr(key, fallback)

    def retranslate(self) -> None:
        """Re-resolve cached display label under current language and repaint.

        Called by CanvasPage._retranslate_canvas via scene.items() sweep on
        language change. Enum value text is not cached (read directly through
        _resolve_choice_text every paint), so a plain self.update() is enough
        to pick up new translations for value-side text as well."""
        new = self._resolve_label()
        changed = new != self._display_label
        if changed:
            self._display_label = new
        # repaint unconditionally — value-side enum text reads tr() at paint
        # time and also needs to refresh under the new language.
        self.update()

    # ---- 公开 ----

    @property
    def value(self) -> Any:
        return self._value

    def set_value(self, v: Any) -> None:
        if v == self._value:
            return
        old = self._value
        self._value = v
        self._owner.params[self.spec.key] = v
        self.update()
        # Notify owner so siblings (e.g. BodyMappingTableRow watching asset_id)
        # can react. Best-effort: ignore if NodeItem has no hook.
        notify = getattr(self._owner, "on_param_changed", None)
        if callable(notify):
            try:
                notify(self.spec.key, v)
            except Exception:
                pass
        # Scene-level fan-out — DEMO 对应 ``TrainingGraphScene.node_param_changed``
        # (training_workspace_window.py:3013). 让别的节点上的 widget 也能感知
        # 参数变化（例：Export 节点 OutputFileListRow 监听 bundle_name 变化以
        # 自动刷新文件清单 / Launch 按钮收集最新 review_backend / scene_id）。
        sc = self.scene()
        sig = getattr(sc, "node_param_changed", None) if sc is not None else None
        if sig is not None:
            try:
                node_type = str(getattr(self._owner.manifest, "id", "") or "")
                sig.emit(node_type, str(self.spec.key), str(v))
            except Exception:
                pass
        # Push to CanvasPage._undo_stack (skipped during command replay so
        # ParamChangeCmd._apply doesn't re-push itself).
        self._maybe_push_undo(old, v)

    def _maybe_push_undo(self, old: Any, new: Any) -> None:
        sc = self.scene()
        if sc is None:
            return
        page = getattr(sc, "_page_ref", None)
        if page is None or getattr(page, "_in_undo", False):
            return
        try:
            from .undo import ParamChangeCmd
            page._undo_stack.push(ParamChangeCmd(
                page, self._owner.node_id, self.spec.key, old, new,
            ))
        except Exception:
            # 不让 undo bookkeeping 失败影响 set_value 的副作用
            pass

    # ---- conditional_on visibility ----

    @staticmethod
    def _eval_one_conditional(cond: Any, host_params: dict) -> bool:
        """Evaluate a single ``{key, op, value}`` condition. Malformed → True."""
        if not isinstance(cond, dict):
            return True
        key = cond.get("key")
        op = cond.get("op")
        expected = cond.get("value")
        if not key or not op:
            return True
        actual = host_params.get(key)
        try:
            if op == "==":
                return actual == expected
            if op == "!=":
                return actual != expected
            if op == "in":
                seq = expected if isinstance(expected, (list, tuple, set)) else (expected,)
                return actual in seq
            if op == "not in":
                seq = expected if isinstance(expected, (list, tuple, set)) else (expected,)
                return actual not in seq
        except Exception:
            return True
        return True

    @staticmethod
    def _eval_conditional(meta: Any, host_params: dict) -> bool:
        """Evaluate ``meta['conditional_on']`` against host params.

        ``conditional_on`` is either a single ``{"key","op","value"}`` dict OR a
        LIST of such dicts, in which case ALL must pass (logical AND). The list
        form lets a row require both a backend gate (``backend == isaac_lab``)
        AND an intra-node toggle (``gait_enabled == true``) — needed because
        gating only the parent toggle does not hide a child whose stored parent
        value already satisfies the child's condition. Missing/malformed →
        visible (default to showing the row).
        """
        if not isinstance(meta, dict):
            return True
        cond = meta.get("conditional_on")
        if isinstance(cond, (list, tuple)):
            return all(
                ParamRow._eval_one_conditional(c, host_params) for c in cond
            )
        return ParamRow._eval_one_conditional(cond, host_params)

    def update_visibility(self, host_params: dict) -> bool:
        """Apply conditional_on visibility against host_params.

        Returns True if visibility flipped (caller can reflow on True).
        """
        visible = self._eval_conditional(self.spec.meta, host_params)
        if visible == self.isVisible():
            return False
        self.setVisible(visible)
        return True

    # ---- Inline port anchors (overridable; default = none) ----

    def iter_inline_port_anchors(self):
        """Yield ``(PortSpec, anchor_local_pt)`` tuples for row-bound ports.

        Override in subclasses that need to mount per-item PortItems on
        canvas inside the row body — e.g. ``TrainingItemsInlineRow`` mounts a
        ``reward_pipe`` input on each enabled training item. The anchor point
        is in **row-local** coordinates (the row's setPos() origin); the
        NodeItem layout pass converts it to node-local before assigning the
        PortItem position. Default = empty (no inline ports).
        """
        return ()

    # ---- Height contract (overridable for full-row rows like body_mapping) ----

    def preferred_height(self) -> float:
        """Return the row's preferred height. Default = ``PARAM_ROW_H`` (24).

        Honours ``spec.meta['row_height']`` first (DEMO ``training_node_ui.json``
        协议 — 让 ParamSpec 直接声明非 24 的行高，无需子类覆写)；否则回退到
        ``self._height``。Subclasses with dynamic content (BodyMappingTableRow,
        OutputFileListRow) still override this for content-dependent height
        and call ``self.prepareGeometryChange()`` + invoke their height-change
        callbacks when the value changes.
        """
        meta = self.spec.meta or {}
        rh = meta.get("row_height")
        if isinstance(rh, (int, float)) and rh > 0:
            return float(rh)
        return float(self._height)

    @property
    def is_full_width(self) -> bool:
        """``meta['full_width_widget']`` — value 半边占满整行（DEMO 协议）.

        当前 ParamRow 的 paint 已经把 value 半边一直延伸到 ``self._width``，
        子类可读这个属性决定是否绘制 key 标签外的扩展内容（如 OutputFileListRow
        的全宽文件清单 / ReviewLaunchButtonRow 的全宽按钮）。
        """
        meta = self.spec.meta or {}
        return bool(meta.get("full_width_widget", False))

    @property
    def is_pinned_on_collapse(self) -> bool:
        """``meta['pin_on_collapse']`` — 节点收纳态时仍保持可见的"关键行"。

        NodeItem._compute_layout 在 ``_collapsed=True`` 时把 ParamRow 全部 hide,
        但读到本 flag 的会留下来 —— 用于 Robot.asset_id / Rewards.reward_terms
        / Trainer 的 mode/envs/iterations 等业务认定的"标识性参数",收纳后仍
        能一眼看清。
        """
        meta = self.spec.meta or {}
        return bool(meta.get("pin_on_collapse", False))

    def on_height_changed(self, callback) -> None:
        """No-op default; BodyMappingTableRow overrides with a real registry."""
        return None

    # ---- Qt overrides ----

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        tier = tier_for_painter(painter)
        # 任何缩放档都画——节点内容不允许提前消失。
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 行底色透明：仅当 hover 命中某个具体可交互子矩形时，在该矩形上画一层
        # 淡高亮；平时（包括停在 key 标签 / 行间隙 / value 半边非交互区域上）
        # 完全不画，让节点 body 底色透出。
        if self._hover_rect is not None:
            hover_bg = QColor(Config.get_color("bg_1", "#2A2A2A")).lighter(115)
            painter.setBrush(QBrush(hover_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self._hover_rect)

        # key 标签
        text_color = QColor(Config.get_color("canvas_node_param_text", "#C8C8C8"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        key_rect = QRectF(KEY_PAD_LEFT, 0, KEY_WIDTH, self._height)
        painter.drawText(
            key_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._display_label,
        )

        # 分隔细线
        sep_color = QColor(Config.get_color("border_1", "#444444"))
        sep_color.setAlpha(120)
        painter.setPen(QPen(sep_color, 0.6))
        painter.drawLine(
            QPointF(SEP_X, 4), QPointF(SEP_X, self._height - 4)
        )

        # value 半边（子类实现）
        self._paint_value(painter, tier)

    def _is_hovering(self, rect: Optional[QRectF] = None) -> bool:
        """当前是否正在 hover；可选传入具体子矩形精确比对."""
        if self._hover_rect is None:
            return False
        if rect is None:
            return True
        return self._hover_rect == rect

    def _interactive_regions(self) -> List[QRectF]:
        """子类声明本行内真正可交互的子矩形列表 / Hit-rects of in-row controls.

        默认实现：整个 value 半边（多数行的「双击/单击 弹 popup」触发区）。
        子类按真实控件几何缩小（BoolRow 复选框 14×14、IndexRow 大按钮、
        RangeRow [滑块, 末端按钮]、TableReadOnlyRow 空列表）。
        """
        return [self._value_rect()]

    def _set_hover_rect(self, rect: Optional[QRectF]) -> None:
        if rect is self._hover_rect:
            return
        if (
            rect is not None
            and self._hover_rect is not None
            and rect == self._hover_rect
        ):
            return
        self._hover_rect = rect
        # 同步光标：命中可交互子区域才显示 PointingHand，否则回到默认 Arrow。
        if rect is not None:
            if not self._cursor_active:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self._cursor_active = True
        else:
            if self._cursor_active:
                self.unsetCursor()
                self._cursor_active = False
        self.update()

    def _hit_region(self, pos: QPointF) -> Optional[QRectF]:
        for r in self._interactive_regions():
            if r.contains(pos):
                return r
        return None

    def hoverEnterEvent(self, event):
        self._set_hover_rect(self._hit_region(event.pos()))
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        self._set_hover_rect(self._hit_region(event.pos()))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self._set_hover_rect(None)
        super().hoverLeaveEvent(event)

    # 子类可设 True 表示 click 在 release 触发（QPushButton 风格）。
    # 重要：popup-opening 行必须 release 触发，否则 QGraphicsScene 的 implicit
    # mouse grab 会与 Qt.WindowType.Popup 的 popup grab 冲突，导致 popup 弹出
    # 后无法被外部点击关闭（"卡住收不回去"）。drag 类（RangeRow）必须 press
    # 触发，所以保持默认 False。
    _OPEN_ON_RELEASE = False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # 仅当点击落在某个可交互子矩形内才进入处理；落在 key 标签 / 行间隙上的
        # 点击与 hover/光标的「不响应」语义保持一致，事件冒泡给节点（用于拖拽）。
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._hit_region(event.pos()) is not None
        ):
            if self._OPEN_ON_RELEASE:
                self._press_marked = True
                event.accept()
                return
            handled = self._handle_click(event)
            if handled:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._OPEN_ON_RELEASE
            and getattr(self, "_press_marked", False)
        ):
            self._press_marked = False
            # 释放的位置仍落在某个可交互子矩形内才触发（QPushButton 行为）
            if self._hit_region(event.pos()) is not None:
                handled = self._handle_click(event)
                if handled:
                    event.accept()
                    return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._hit_region(event.pos()) is not None
        ):
            handled = self._handle_double_click(event)
            if handled:
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ---- 子类钩子 ----

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        """画 value 半边；子类实现."""
        ...

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        """单击；返回 True 表示已消化，不再向上冒泡."""
        return False

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        """双击；返回 True 表示已消化."""
        return False

    # ---- 内部 ----

    def _read_value(self) -> Any:
        v = self._owner.params.get(self.spec.key, None)
        if v is None:
            v = self.spec.default
        return v

    def _value_rect(self) -> QRectF:
        x = SEP_X + 6
        return QRectF(
            x, 0, max(0.0, self._width - x - VALUE_PAD_RIGHT), self._height
        )

    def _draw_value_text(
        self,
        painter: QPainter,
        text: str,
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        # 截断超长 value
        rect = self._value_rect()
        fm = QFontMetricsF(font)
        elided = fm.elidedText(str(text), Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(
            rect,
            int(align | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )


# =============================================================================
# 1. BoolRow —— 复选框
# =============================================================================

class BoolRow(ParamRow):
    """单击 toggle / Single-click toggle."""

    _OPEN_ON_RELEASE = True

    def _checkbox_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 4, rect.center().y() - 7, 14, 14)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._checkbox_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = self._value_rect()
        # 复选框 14×14，竖直居中
        box = self._checkbox_rect()
        painter.setBrush(QBrush(QColor(Config.get_color("bg_1", "#2A2A2A"))))
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(box, 2, 2)
        if bool(self._value):
            check_color = QColor(Config.get_color("highlight", "#F6D393"))
            painter.setPen(QPen(check_color, 2.0))
            # 勾
            cx, cy = box.center().x(), box.center().y()
            painter.drawLine(
                QPointF(cx - 4, cy + 0),
                QPointF(cx - 1, cy + 3),
            )
            painter.drawLine(
                QPointF(cx - 1, cy + 3),
                QPointF(cx + 5, cy - 4),
            )
        # T3: 在 box 右侧补 True/False 文字
        if tier == TIER_DETAIL:
            label_rect = QRectF(box.right() + 6, rect.top(), rect.right() - box.right() - 6, rect.height())
            label_color = QColor(Config.get_color("canvas_node_param_text", "#C8C8C8"))
            painter.setPen(QPen(label_color))
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                tr("canvas.row.true", "True") if self._value
                else tr("canvas.row.false", "False"),
            )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        self.set_value(not bool(self._value))
        return True


# =============================================================================
# 2. NumberRow —— int / float
# =============================================================================

def _draw_input_chrome(
    painter: QPainter, rect: QRectF, *, hover: bool,
) -> QRectF:
    """画 input-box 风格 chrome（bg + border + hover）/ Input-box chrome.

    与 ChoiceRow 的 ``draw_choice_dropdown`` 同款颜色规范，让所有可编辑 cell
    在画布上视觉一致。返回内缩 4px 后的内容矩形（供文字绘制）。
    """
    bg = QColor(Config.get_color("bg_1", "#1A1A1A"))
    if hover:
        bg = bg.lighter(115)
    border = QColor(Config.get_color("border_2", "#3d3d3d"))
    painter.setBrush(QBrush(bg))
    painter.setPen(QPen(border, 1.0))
    painter.drawRoundedRect(rect, 3, 3)
    return rect.adjusted(4, 0, -4, 0)


class NumberRow(ParamRow):
    """单击 → NumberEditDialog / Single-click opens NumberEditDialog.

    视觉为 input-box 风格 chrome（描边 + 淡底色），与 ChoiceRow 同款 —— 让
    用户一眼看出这是可编辑控件，而不是只读 label。
    """

    _OPEN_ON_RELEASE = True

    def _input_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 4, rect.top() + 3, rect.width() - 8, rect.height() - 6)

    def _interactive_regions(self) -> List[QRectF]:
        # 整个 value cell 全宽都可点 —— 不要求用户对准内缩 input box。
        return [self._value_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        box = self._input_rect()
        text_rect = _draw_input_chrome(painter, box, hover=self._is_hovering(self._value_rect()))
        text = self._format_value()
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        fm = QFontMetricsF(font)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

    def _format_value(self) -> str:
        v = self._value
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        is_int = self.spec.type == "int"
        meta = self.spec.meta or {}
        try:
            v = float(self._value if self._value is not None else 0)
        except (TypeError, ValueError):
            v = 0.0
        open_number_popup(
            view=_view_of(event),
            row=self,
            value=int(v) if is_int else v,
            is_int=is_int,
            minimum=meta.get("min"),
            maximum=meta.get("max"),
            step=meta.get("step"),
            on_commit=self.set_value,
        )
        return True

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        # 兼容老的双击习惯 —— 行为与单击一致。
        return self._handle_click(event)


# =============================================================================
# 3. StringRow —— 单行字符串
# =============================================================================

class StringRow(ParamRow):
    """单击 → StringEditDialog.

    视觉为 input-box 风格 chrome（描边 + 淡底色），与 ChoiceRow / NumberRow
    一致。
    """

    _OPEN_ON_RELEASE = True

    def _input_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 4, rect.top() + 3, rect.width() - 8, rect.height() - 6)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._value_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        box = self._input_rect()
        text_rect = _draw_input_chrome(painter, box, hover=self._is_hovering(self._value_rect()))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        fm = QFontMetricsF(font)
        elided = fm.elidedText(
            str(self._value or ""), Qt.TextElideMode.ElideRight, text_rect.width(),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        open_text_popup(
            view=_view_of(event),
            row=self,
            value=str(self._value or ""),
            placeholder=self.spec.description or "",
            on_commit=self.set_value,
        )
        return True

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        return self._handle_click(event)


# =============================================================================
# 4. ChoiceRow —— enum (with choices)
# =============================================================================

class ChoiceRow(ParamRow):
    """单击弹 SDK Node_RichChoicePopup（卡片选单）/ Single-click opens popup."""

    _OPEN_ON_RELEASE = True

    def _dropdown_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 4, rect.top() + 3, rect.width() - 8, rect.height() - 6)

    def _interactive_regions(self) -> List[QRectF]:
        # 整个 value cell 全宽都可点 —— ``_dropdown_rect`` 仅作为视觉 chrome；
        # 用户点 cell 任意位置（含 4px 边缘留白）都能弹 popup。
        return [self._value_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        text = self._resolve_choice_text(self._value)
        # 行内 dropdown（高度收 4px 给上下留呼吸）
        box = self._dropdown_rect()
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        draw_choice_dropdown(
            painter,
            box,
            text,
            hover=self._is_hovering(box),
            font=font,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        if not self.spec.choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=self.spec.choices,
            current=self._value,
            multi=False,
            on_commit=self.set_value,
            label_resolver=self._resolve_choice_text,
        )
        return True


# =============================================================================
# 5. PathRow —— 文件 / 目录路径
# =============================================================================

class PathRow(ParamRow):
    """单击右侧浏览按钮 → QFileDialog; 路径文本左侧只读显示."""

    BROWSE_W = 22  # ⋯ 按钮宽度

    def _browse_btn_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.right() - self.BROWSE_W, rect.top() + 3, self.BROWSE_W, rect.height() - 6)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._browse_btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = self._value_rect()
        # 按钮在右侧
        btn_rect = self._browse_btn_rect()
        # 路径文本在左侧（截断）
        text_rect = QRectF(rect.left(), rect.top(), rect.width() - self.BROWSE_W - 4, rect.height())
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        fm = QFontMetricsF(font)
        elided = fm.elidedText(
            str(self._value or ""), Qt.TextElideMode.ElideMiddle, text_rect.width()
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        # ⋯ 浏览按钮（SDK helper）
        draw_edit_button(
            painter,
            btn_rect,
            kind="browse",
            hover=self._is_hovering(btn_rect),
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        # 区分：只在 ⋯ 按钮区域内才弹 file dialog（基类已收紧，但保留显式判断
        # 以便文本区即便被未来扩展也不会误触发 dialog）。
        if not self._browse_btn_rect().contains(event.pos()):
            return False
        meta = self.spec.meta or {}
        path = open_path_dialog(
            title=self.spec.key,
            current=str(self._value or ""),
            is_dir=bool(meta.get("is_dir")),
            name_filter=str(meta.get("filter", "")),
            parent=event.widget(),
        )
        if path is not None:
            self.set_value(path)
        return True


# =============================================================================
# 6. IndexRow —— 数字按钮（int 专用，带 step）
# =============================================================================

class IndexRow(ParamRow):
    """大号数字按钮，单击弹锚定 number popup."""

    _OPEN_ON_RELEASE = True

    def _btn_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 4, rect.top() + 2, rect.width() - 8, rect.height() - 4)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        btn = self._btn_rect()
        btn_color = QColor(Config.get_color("btn_1", "#343636"))
        if self._is_hovering(btn):
            btn_color = btn_color.lighter(115)
        painter.setBrush(QBrush(btn_color))
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(btn, 3, 3)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(btn, int(Qt.AlignmentFlag.AlignCenter), str(self._value if self._value is not None else 0))

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        meta = self.spec.meta or {}
        try:
            v = int(self._value if self._value is not None else 0)
        except (TypeError, ValueError):
            v = 0
        open_number_popup(
            view=_view_of(event),
            row=self,
            value=v,
            is_int=True,
            minimum=int(meta.get("min", 0)),
            maximum=int(meta.get("max", 1000)),
            on_commit=lambda val: self.set_value(int(val)),
        )
        return True


# =============================================================================
# 7. RangeRow —— 单滑块范围（meta: min/max/step）
# =============================================================================

class RangeRow(ParamRow):
    """滑块拖拽编辑数值；几何 + 视觉走 SDK ``draw_range_slider`` + ``RangeSliderHitTest``."""

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._dragging = False
        meta = spec.meta or {}
        self._min = float(meta.get("min", 0.0))
        self._max = float(meta.get("max", 1.0))
        if self._max <= self._min:
            self._max = self._min + 1.0
        self._step = meta.get("step")

    # 几何契约（单把手 NodeSlider：[slider][hi_btn]，仅右端 1 按钮显示当前值）。
    # 严格 fit content：按钮宽高 = 当前文本的 font metrics 测量值，
    # 零内边距、零最小/最大限制、不随容器拉伸。

    # 与按钮右边沿到 value_rect 右边沿之间留 2px 的视觉间距（属于 row 布局，
    # 不是按钮内部 padding）；slider 与按钮左边沿之间留 6px gap。
    _BTN_RIGHT_OFFSET = 2
    _BTN_SLIDER_GAP = 6

    def _btn_font(self) -> QFont:
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        return font

    # 仅水平方向左右各 1px padding（避免边框圆角切到字形端点）；
    # 垂直方向严格 fit content（高度 = 文本 fm.height）。
    _BTN_PAD_X = 1

    def _measure_btn_size(self, text: str) -> tuple:
        """fit content：返回 (width, height)。
        width = ceil(fm.horizontalAdvance) + 2 * _BTN_PAD_X
        height = ceil(fm.height)（无垂直 padding）"""
        fm = QFontMetricsF(self._btn_font())
        from math import ceil
        return (float(ceil(fm.horizontalAdvance(text)) + 2 * self._BTN_PAD_X),
                float(ceil(fm.height())))

    def _hi_btn_rect(self, text: Optional[str] = None) -> QRectF:
        rect = self._value_rect()
        if text is None:
            text = self._format(self._normalized_value())
        w, h = self._measure_btn_size(text)
        # 右锚定 + 在 value_rect 中竖直居中（高度 = 文本高度，不再拉满行高）
        x = rect.right() - w - self._BTN_RIGHT_OFFSET
        y = rect.top() + (rect.height() - h) * 0.5
        return QRectF(x, y, w, h)

    def _slider_rect(self) -> QRectF:
        rect = self._value_rect()
        btn_w, _ = self._measure_btn_size(
            self._format(self._normalized_value())
        )
        # 左侧仅 2px 起手 padding；右侧让出 btn 宽度 + 间距
        left = rect.left() + 2
        right = rect.right() - btn_w - self._BTN_RIGHT_OFFSET - self._BTN_SLIDER_GAP
        width = max(20.0, right - left)
        return QRectF(left, rect.top() + 4, width, rect.height() - 8)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._slider_rect(), self._hi_btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        try:
            v = float(self._value)
        except (TypeError, ValueError):
            v = self._min

        slider_rect = self._slider_rect()
        draw_range_slider(
            painter,
            slider_rect,
            lo=v,
            vmin=self._min,
            vmax=self._max,
            dual=False,
            hover_handle="lo" if self._is_hovering(slider_rect) or self._dragging else None,
        )
        # 单把手只放 1 个按钮：右端显示当前值（DEMO IndexButton 风格）
        text = self._format(v)
        btn_rect = self._hi_btn_rect(text)
        self._paint_end_button(painter, btn_rect, text=text, accent=True)

    def _format(self, x: float) -> str:
        try:
            return f"{float(x):g}"
        except (TypeError, ValueError):
            return str(x)

    def _paint_end_button(self, painter: QPainter, rect: QRectF, *, text: str, accent: bool) -> None:
        """画 [ value ] 风格 mini 按钮 / Paint a clickable mini-button at slider end.

        严格 fit content：``rect`` 由 ``_measure_btn_size(text)`` 给出，宽高
        都等于文本 font metrics，零 padding、不随容器伸缩。无 elide —— 文本
        永远刚好填满按钮。
        """
        bg_slot = "btn_1" if not accent else "btn_1"
        bg = QColor(Config.get_color(bg_slot, "#343636"))
        if self._is_hovering(rect):
            bg = bg.lighter(110)
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(rect, 3, 3)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        font = self._btn_font()
        if not accent:
            font.setBold(False)
        painter.setFont(font)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)

    def _hit(self) -> RangeSliderHitTest:
        return RangeSliderHitTest(
            rect=self._slider_rect(),
            vmin=self._min,
            vmax=self._max,
            dual=False,
        )

    def _commit_x(self, x: float) -> None:
        # 拖拽必须顺滑 —— 不吸附 step，避免 thumb 抢用户光标。
        # step 只作用于：number popup 输入校验、键盘 ±step 微调。
        ht = self._hit()
        self.set_value(ht.value_at(x))

    def _open_value_popup(self, event: QGraphicsSceneMouseEvent, *, initial: float) -> None:
        """点端点按钮 → 弹 number popup 编辑 current value."""
        open_number_popup(
            view=_view_of(event),
            row=self,
            value=initial,
            is_int=False,
            minimum=self._min,
            maximum=self._max,
            step=float(self._step) if self._step else None,
            on_commit=self.set_value,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        # 1. 端点按钮（右 current value）→ 弹 number popup 编辑当前值
        if self._hi_btn_rect().contains(event.pos()):
            try:
                cur = float(self._value)
            except (TypeError, ValueError):
                cur = self._min
            self._open_value_popup(event, initial=cur)
            return True
        # 2. slider 轨道 / thumb → 跳点 + 拖拽
        ht = self._hit()
        hit = ht.handle_at(event.pos(), self._normalized_value())
        if hit is None:
            return False
        self._commit_x(event.pos().x())
        self._dragging = True
        return True

    def _normalized_value(self) -> float:
        try:
            return float(self._value)
        except (TypeError, ValueError):
            return self._min

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging:
            self._commit_x(event.pos().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


# =============================================================================
# 7b. RangeListRow —— 双手柄区间 [lo, hi]（widget="range_list"）
# =============================================================================

class RangeListRow(RangeRow):
    """``widget="range_list"`` → ``[lo, hi]`` 双手柄 slider.

    用于 DR / motion / task 节点中所有 ``[a, b]`` 范围参数（mass_range,
    static_friction_range, push_velocity_x_range, gait_frequency_range, …）。
    DEMO 用 ``_RangeHandleSlider`` 双 thumb；这里复用 SDK ``draw_range_slider(dual=True)``
    + ``RangeSliderHitTest(dual=True)``。

    几何：[lo_btn][slider][hi_btn] —— 两端各一按钮显示当前 lo/hi 值，单击弹
    number popup 编辑该端点；拖拽 thumb 即时更新对应端点。

    持久化：``ParamSpec.type == "list"`` 时直接 set ``[lo, hi]`` Python list；
    其它（json / 容错路径）回退为 ``json.dumps([lo, hi])``，与 DEMO 兼容。
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        # 重置为双手柄拖拽状态
        self._drag_handle: Optional[str] = None  # "lo" / "hi" / None
        # 把 super().__init__ 的 ``_dragging`` 钩子保留以兼容 mouseMove/Release。
        self._dragging = False

    # ---- 值解析 / Serialisation ----

    def _parse_pair(self, v: Any) -> tuple:
        """解析存储值为 (lo, hi)。容忍 list / tuple / JSON-string / None."""
        arr = None
        if isinstance(v, (list, tuple)):
            arr = list(v)
        elif isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (list, tuple)):
                    arr = list(parsed)
            except Exception:
                arr = None
        if arr is None or len(arr) < 2:
            return (self._min, self._max)
        try:
            lo = float(arr[0])
            hi = float(arr[1])
        except (TypeError, ValueError):
            return (self._min, self._max)
        # 保证 lo <= hi
        if lo > hi:
            lo, hi = hi, lo
        # clamp
        lo = max(self._min, min(self._max, lo))
        hi = max(self._min, min(self._max, hi))
        return (lo, hi)

    def _serialize_pair_raw(self, lo: float, hi: float) -> Any:
        """序列化 [lo, hi]，原样保留用户值（不吸附 step）.

        拖拽与 popup 键入共用同一路径。早先 popup 路径会按 ``meta.step`` 做
        ``round(v/step)*step`` 吸附（拖拽路径反而不吸附），结果只把键入值篡改
        （键入 0.01、step=0.05 → 0.0），属 §8 静默改输入。step 仅是滑块视觉
        粒度，不应改写用户键入的数值。"""
        if lo > hi:
            lo, hi = hi, lo
        t = (self.spec.type or "").lower()
        if t in ("list", "json", "dict"):
            return [lo, hi]
        try:
            return json.dumps([lo, hi], ensure_ascii=False)
        except Exception:
            return [lo, hi]

    # ---- 几何 ----

    def _lo_btn_rect(self, text: Optional[str] = None) -> QRectF:
        rect = self._value_rect()
        if text is None:
            lo, _hi = self._parse_pair(self._value)
            text = self._format(lo)
        w, h = self._measure_btn_size(text)
        x = rect.left() + 2
        y = rect.top() + (rect.height() - h) * 0.5
        return QRectF(x, y, w, h)

    def _hi_btn_rect(self, text: Optional[str] = None) -> QRectF:  # type: ignore[override]
        rect = self._value_rect()
        if text is None:
            _lo, hi = self._parse_pair(self._value)
            text = self._format(hi)
        w, h = self._measure_btn_size(text)
        x = rect.right() - w - self._BTN_RIGHT_OFFSET
        y = rect.top() + (rect.height() - h) * 0.5
        return QRectF(x, y, w, h)

    def _slider_rect(self) -> QRectF:  # type: ignore[override]
        rect = self._value_rect()
        lo, hi = self._parse_pair(self._value)
        lo_w, _ = self._measure_btn_size(self._format(lo))
        hi_w, _ = self._measure_btn_size(self._format(hi))
        left = rect.left() + 2 + lo_w + self._BTN_SLIDER_GAP
        right = rect.right() - hi_w - self._BTN_RIGHT_OFFSET - self._BTN_SLIDER_GAP
        width = max(20.0, right - left)
        return QRectF(left, rect.top() + 4, width, rect.height() - 8)

    def _interactive_regions(self) -> List[QRectF]:  # type: ignore[override]
        return [self._slider_rect(), self._lo_btn_rect(), self._hi_btn_rect()]

    # ---- Paint ----

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        lo, hi = self._parse_pair(self._value)
        slider_rect = self._slider_rect()
        # hover 计算：拖拽中显示对应 thumb 的 hover 高亮
        hover = None
        if self._drag_handle in ("lo", "hi"):
            hover = self._drag_handle  # type: ignore[assignment]
        draw_range_slider(
            painter,
            slider_rect,
            lo=lo,
            hi=hi,
            vmin=self._min,
            vmax=self._max,
            dual=True,
            hover_handle=hover,
        )
        # 端点按钮：左 = lo，右 = hi
        lo_text = self._format(lo)
        hi_text = self._format(hi)
        self._paint_end_button(painter, self._lo_btn_rect(lo_text), text=lo_text, accent=True)
        self._paint_end_button(painter, self._hi_btn_rect(hi_text), text=hi_text, accent=True)

    # ---- Hit-test / 拖拽 ----

    def _hit(self) -> RangeSliderHitTest:  # type: ignore[override]
        return RangeSliderHitTest(
            rect=self._slider_rect(),
            vmin=self._min,
            vmax=self._max,
            dual=True,
        )

    def _commit_x(self, x: float) -> None:  # type: ignore[override]
        if self._drag_handle not in ("lo", "hi"):
            return
        # 拖拽顺滑 —— 不吸附 step。
        ht = self._hit()
        new_val = ht.value_at(x)
        lo, hi = self._parse_pair(self._value)
        if self._drag_handle == "lo":
            lo = min(new_val, hi)
        else:
            hi = max(new_val, lo)
        self.set_value(self._serialize_pair_raw(lo, hi))

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        lo, hi = self._parse_pair(self._value)
        # 1. 左端按钮 → 编辑 lo
        if self._lo_btn_rect().contains(event.pos()):
            self._open_handle_popup(event, which="lo", initial=lo)
            return True
        # 2. 右端按钮 → 编辑 hi
        if self._hi_btn_rect().contains(event.pos()):
            self._open_handle_popup(event, which="hi", initial=hi)
            return True
        # 3. slider thumb / track → 决定哪个 thumb 被命中并启动拖拽
        ht = self._hit()
        hit = ht.handle_at(event.pos(), lo, hi)
        if hit is None:
            return False
        if hit == "track":
            # 点轨道：把更近的那个 thumb 拉到点击位置（DEMO 行为）
            click_v = ht.value_at(event.pos().x())
            hit = "lo" if abs(click_v - lo) <= abs(click_v - hi) else "hi"
        self._drag_handle = hit  # type: ignore[assignment]
        self._dragging = True
        self._commit_x(event.pos().x())
        return True

    def _open_handle_popup(
        self,
        event: QGraphicsSceneMouseEvent,
        *,
        which: str,
        initial: float,
    ) -> None:
        def _commit(val: Any) -> None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return
            lo, hi = self._parse_pair(self._value)
            if which == "lo":
                lo = min(v, hi)
            else:
                hi = max(v, lo)
            self.set_value(self._serialize_pair_raw(lo, hi))

        open_number_popup(
            view=_view_of(event),
            row=self,
            value=initial,
            is_int=False,
            minimum=self._min,
            maximum=self._max,
            step=float(self._step) if self._step else None,
            on_commit=_commit,
        )

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        if self._dragging:
            self._dragging = False
            self._drag_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


# =============================================================================
# 8. BadgeRow —— 极性徽章（圆 / 方 / 菱形）
# =============================================================================

class BadgeRow(ParamRow):
    """``meta.badge_states`` 列出所有可选徽章；单击 cycle / Click cycles through states.

    缺省 states：``["circle", "square", "diamond"]``。
    单击 → 取下一个 → set_value(state_name)。
    """

    _OPEN_ON_RELEASE = True
    DEFAULT_STATES = ("circle", "square", "diamond")

    # state 名 → SDK draw_polarity_badge 的 kind
    _STATE_TO_KIND = {
        "circle":   "reward",
        "square":   "penalty",
        "diamond":  "bidirectional",
        "reward":   "reward",
        "penalty":  "penalty",
        "bidirectional": "bidirectional",
        "neutral":  "neutral",
    }

    def _states(self) -> tuple:
        meta = self.spec.meta or {}
        st = meta.get("badge_states")
        if isinstance(st, (list, tuple)) and st:
            return tuple(st)
        return self.DEFAULT_STATES

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = self._value_rect()
        size = 18
        badge_rect = QRectF(rect.left() + 4, rect.center().y() - size * 0.5, size, size)
        state = str(self._value if self._value is not None else self._states()[0])
        state_label = self._resolve_choice_text(state)
        kind = self._STATE_TO_KIND.get(state, "neutral")
        color = QColor(Config.get_color("highlight", "#F6D393"))
        draw_polarity_badge(
            painter,
            badge_rect,
            kind=kind,  # type: ignore[arg-type]
            color=color,
            border_color=QColor(Config.get_color("border_1", "#444444")),
        )
        # state 文字
        label_rect = QRectF(badge_rect.right() + 8, rect.top(), rect.right() - badge_rect.right() - 10, rect.height())
        text_color = QColor(Config.get_color("canvas_node_param_text", "#C8C8C8"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), state_label)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        states = self._states()
        try:
            idx = states.index(str(self._value))
        except ValueError:
            idx = -1
        nxt = states[(idx + 1) % len(states)]
        self.set_value(nxt)
        return True


# =============================================================================
# 9. CodeRow —— 多行代码 / JSON
# =============================================================================

class CodeRow(ParamRow):
    """单击弹锚定 code popup（多行 ``LaviCodeEditor``）.

    视觉契约：整个 value 区域是一个按钮，按钮内直接显示 JSON / 代码文本
    （单行截断）。无独立 ``{·}`` icon 按钮。点击任意位置都打开编辑器。
    """

    _OPEN_ON_RELEASE = True

    def _btn_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 2, rect.top() + 2, rect.width() - 4, rect.height() - 4)

    def _interactive_regions(self) -> List[QRectF]:
        return [self._btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        btn = self._btn_rect()
        # 按钮 bg / border
        bg = QColor(Config.get_color("btn_1", "#343636"))
        if self._is_hovering(btn):
            bg = bg.lighter(115)
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(btn, 3, 3)
        # 直接显示 JSON / 代码文本（单行截断）
        v = self._value
        if isinstance(v, (dict, list)):
            try:
                preview = json.dumps(v, ensure_ascii=False)
            except Exception:
                preview = str(v)
        else:
            preview = str(v or "")
        preview = preview.replace("\n", " ").replace("  ", " ").strip()
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        # 内边距 6px 给 elided text
        text_rect = btn.adjusted(6, 0, -6, 0)
        elided = fm.elidedText(preview or "Edit…", Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        meta = self.spec.meta or {}
        v = self._value
        if isinstance(v, (dict, list)):
            try:
                text = json.dumps(v, indent=2, ensure_ascii=False)
            except Exception:
                text = str(v)
            language = "json"
        else:
            text = str(v or "")
            language = str(meta.get("language", "python"))
            # Pretty-print compact JSON *strings* too (e.g. the default "{}",
            # or a value reloaded from disk as a single-line string) so the
            # editor opens with indentation/newlines instead of one cramped
            # line. If the string isn't valid JSON (user mid-edit), leave it
            # untouched so they can see and fix the raw text.
            if language == "json" and text.strip():
                try:
                    text = json.dumps(
                        json.loads(text), indent=2, ensure_ascii=False
                    )
                except (json.JSONDecodeError, ValueError):
                    pass

        def _commit(new_text: str) -> None:
            if language == "json":
                try:
                    self.set_value(json.loads(new_text))
                    return
                except (json.JSONDecodeError, ValueError):
                    self.set_value(new_text)
                    return
            self.set_value(new_text)

        open_code_popup(
            view=_view_of(event),
            row=self,
            text=text,
            language=language,
            validate_json=(language == "json"),
            on_commit=_commit,
        )
        return True


# =============================================================================
# 10. TableReadOnlyRow —— 只读多列预览
# =============================================================================

class TableReadOnlyRow(ParamRow):
    """只读 ``[key] [glyph] [value]`` 行；不可编辑.

    常用于显示运行时状态（DEMO ``NodeTableRowsWidget``）。
    Phase 1b 阶段只画结构；运行态 update 由后续 backend wiring 推送。
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)

    def _interactive_regions(self) -> List[QRectF]:
        # 只读行无任何可交互组件 —— 光标保持默认 Arrow，hover 永不高亮。
        return []

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = self._value_rect()
        # glyph (small dot)
        meta = self.spec.meta or {}
        glyph = str(meta.get("glyph", "•"))
        glyph_color_hex = str(meta.get("glyph_color", "#9CA3AF"))
        glyph_color = QColor(glyph_color_hex)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        painter.setPen(QPen(glyph_color))
        glyph_rect = QRectF(rect.left() + 4, rect.top(), 18, rect.height())
        painter.drawText(glyph_rect, int(Qt.AlignmentFlag.AlignCenter), glyph)
        # value
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        val_rect = QRectF(glyph_rect.right() + 2, rect.top(), rect.right() - glyph_rect.right() - 4, rect.height())
        fm = QFontMetricsF(font)
        elided = fm.elidedText(str(self._value or ""), Qt.TextElideMode.ElideRight, val_rect.width())
        painter.drawText(
            val_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

    # 只读：默认 _handle_click / _handle_double_click 都返回 False


# =============================================================================
# 节点专属 Row 子类 —— 弹锚定 SDK Node_* picker / editor
# =============================================================================
# 模式：行内画 summary（_draw_value_text），双击/单击 → open_choice_popup
# (单/多选) 或 open_widget_popup (复杂 editor)。
# 所有 popup 都是 Qt.WindowType.Popup —— 紧贴行右侧、点外即关、不挡画布。
#
# Node_TaskTypePicker / Node_SceneTypePicker / ... 等都是 Node_RichChoicePicker
# 的预设子类（封装了一个"按钮 + popup"两段式 widget）。在画布场景中我们直接
# 跳过其按钮、用 open_choice_popup 弹同样的卡片选单——视觉与 DEMO 一致。


class _PickerRowMixin:
    """通用：summary 文字行内绘制 + 双击弹 popup.

    视觉契约：行内画 *按钮* 而不是裸文字 —— 与 CodeRow / IndexRow / RangeRow
    端点按钮完全一致的按钮配色（bg=canvas_node_title, border=canvas_node_border,
    text=canvas_node_title_text, hover=lighter(115)）。这样 asset_id picker、
    SceneType / TaskType / ExportBundle / MotionLibrary / ObsComponents 等所有
    选单按钮的视觉与画布上其它输入按钮严格一致。
    """

    def _summary_text(self) -> str:
        v = self._value
        if isinstance(v, (list, tuple)):
            return tr("canvas.row.selected", "{n} selected").replace("{n}", str(len(v)))
        if isinstance(v, dict):
            return tr("canvas.row.entries", "{n} entries").replace("{n}", str(len(v)))
        if v is None:
            return tr("canvas.row.empty", "—")
        return str(v)

    def _picker_btn_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(rect.left() + 2, rect.top() + 2, rect.width() - 4, rect.height() - 4)

    def _interactive_regions(self) -> List[QRectF]:  # type: ignore[override]
        return [self._picker_btn_rect()]

    def _paint_picker_chrome(self, painter: QPainter, btn: QRectF) -> None:
        bg = QColor(Config.get_color("btn_1"))
        if self._is_hovering(btn):
            bg = bg.lighter(115)
        border = QColor(Config.get_color("border_1"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(btn, 3, 3)

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        btn = self._picker_btn_rect()
        self._paint_picker_chrome(painter, btn)
        # 文字
        text_color = QColor(Config.get_color("main_t1"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        text_rect = btn.adjusted(8, 0, -18, 0)  # 右侧让 12px 给箭头
        elided = fm.elidedText(
            self._summary_text(), Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        # 下拉箭头 ▾
        arrow_color = QColor(Config.get_color("main_t1"))
        painter.setBrush(QBrush(arrow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = btn.right() - 10.0
        cy = btn.top() + btn.height() * 0.5
        tri = QPolygonF([
            QPointF(cx, cy + 3.0),
            QPointF(cx - 4.5, cy - 2.5),
            QPointF(cx + 4.5, cy - 2.5),
        ])
        painter.drawPolygon(tri)

    def _resolve_choices(self) -> list:
        meta = self.spec.meta or {}
        choices = meta.get("choices") or self.spec.choices or []
        return list(choices)


class _ChoicePickerRow(_PickerRowMixin, ParamRow):
    """单选 picker rows 共通基类（直接 open_choice_popup）.

    子类只需重载类属性 ``LEADING_MODE`` 来切换图标 / checkbox 样式。
    """

    _OPEN_ON_RELEASE = True
    LEADING_MODE = "checkbox"

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._open(event)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        # 单击也展开（与 DEMO 选单 widget 一致）。NodeItem 的拖拽不会因为
        # ChoiceRow 处理了 click 而失效（行只占行高，节点其它区仍可拖）。
        return self._open(event)

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:
        choices = self._resolve_choices()
        if not choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            on_commit=self.set_value,
        )
        return True


#: Sentinel scene_id in the scene picker that opens the terrain import
#: dialog instead of selecting a scene (handled in SceneTypeRow.set_value).
_IMPORT_TERRAIN_SENTINEL = "__import_terrain__"


class SceneTypeRow(_ChoicePickerRow):
    """``widget="picker_scene"`` → registry-backed scene_id picker.

    Lives on play_ground_setting. The ParamSpec carries no static
    choices because the legal scene catalogue depends on the active
    training backend (sb3 vs isaac_lab) + the robot family on the
    canvas. We resolve choices from
    ``application.training.scene_registry.list_scenes`` on every popup
    open so backend / robot swaps refresh immediately.

    On commit we also patch the sibling ``scene_type`` and any
    ``defaults`` the registry attaches to the scene — the
    play_ground_setting node docstring promises scene_type is
    auto-synced from scene_id, and env_cfg_compiler depends on that
    being the actual stored value (it rejects scene_type values
    outside {"flat","rough"}).
    """

    LEADING_MODE = "checkbox"
    _OPEN_ON_RELEASE = True

    # Canvas trainer-node ids that imply the Isaac Lab backend. Any
    # other case (or none) → sb3 / MuJoCo path; the registry filters
    # rough variants out for sb3 so the user sees only the runnable
    # subset.
    _IL_TRAINER_IDS = frozenset({"il_ppo_trainer", "amp_trainer"})

    def _detect_backend(self) -> str:
        sc = self.scene()
        if sc is None:
            return "sb3_mujoco"
        try:
            for it in sc.items():
                manifest = getattr(it, "manifest", None)
                if manifest is None:
                    continue
                mid = str(getattr(manifest, "id", "") or "")
                if mid in self._IL_TRAINER_IDS:
                    return "isaac_lab"
        except Exception:
            pass
        return "sb3_mujoco"

    def _detect_family(self) -> Optional[str]:
        sc = self.scene()
        if sc is None:
            return None
        try:
            for it in sc.items():
                manifest = getattr(it, "manifest", None)
                params = getattr(it, "params", None)
                if manifest is None or params is None:
                    continue
                mid = str(getattr(manifest, "id", "") or "")
                if mid != "robot":
                    continue
                asset_id = str(params.get("asset_id", "") or "").strip()
                if not asset_id:
                    continue
                try:
                    from registers import robots as _robots_registry
                    rob = _robots_registry.get_robot(asset_id)
                    if rob is None:
                        rob = _robots_registry.get_robot(
                            _robots_registry.resolve_id(asset_id) or ""
                        )
                    if rob is not None:
                        fams = getattr(rob, "families", None) or []
                        if fams:
                            return str(fams[0])
                except Exception:
                    pass
                return None
        except Exception:
            return None
        return None

    def _resolve_scene_choices(self) -> tuple:
        backend = self._detect_backend()
        family = self._detect_family()
        try:
            from application.training.scene_registry import list_scenes
            scenes = list_scenes(backend=backend, family=family)
            ids: List[str] = []
            meta: Dict[str, Dict[str, str]] = {}
            for s in scenes:
                sid = str(getattr(s, "scene_id", ""))
                if not sid:
                    continue
                ids.append(sid)
                meta[sid] = {
                    "title": str(getattr(s, "name", sid)) or sid,
                    "desc": str(getattr(s, "description", "")),
                }
            # Sentinel entry → opens the terrain import dialog (handled in
            # set_value) rather than selecting a scene. Always last.
            ids.append(_IMPORT_TERRAIN_SENTINEL)
            meta[_IMPORT_TERRAIN_SENTINEL] = {
                "title": tr("terrain.import.menu", "➕ Import terrain…"),
                "desc": tr(
                    "terrain.import.menu_desc",
                    "Import a PNG / .npy / .npz height-map as a custom terrain.",
                ),
            }
            if ids:
                return ids, meta
        except Exception as exc:
            log_warning(f"[play_ground.scene_id] list_scenes failed: {exc}")
        # Final safety net — at minimum the flat ground baseline so the
        # picker never opens empty even if the registry import broke.
        return ["flat_ground"], {
            "flat_ground": {"title": "Flat Ground", "desc": ""},
        }

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        choices, meta = self._resolve_scene_choices()
        if not choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            meta_map=meta,
            on_commit=self.set_value,
        )
        return True

    def set_value(self, v: Any) -> None:  # type: ignore[override]
        # Sentinel: open the terrain import dialog instead of selecting a
        # scene. On success, re-enter set_value with the freshly-imported
        # scene_id (now registered) so its defaults cascade as normal.
        if v == _IMPORT_TERRAIN_SENTINEL:
            new_sid = self._launch_terrain_import()
            if new_sid:
                self.set_value(new_sid)
            return
        super().set_value(v)
        if not isinstance(v, str) or not v:
            return
        try:
            from application.training.scene_registry import get_scene
            scene = get_scene(v)
        except Exception:
            return
        # Sibling-param patch — scene_type is the load-bearing sync;
        # other defaults are advisory (only applied when the row exists
        # so we don't pollute params with keys the manifest doesn't
        # declare).
        patch: Dict[str, Any] = {"scene_type": str(getattr(scene, "scene_type", "flat"))}
        for k, val in (getattr(scene, "defaults", None) or {}).items():
            patch[str(k)] = val

        rows_by_key = {r.spec.key: r for r in self._owner._param_rows}
        notify = getattr(self._owner, "on_param_changed", None)
        for k, val in patch.items():
            sib = rows_by_key.get(k)
            if sib is not None:
                # Coerce string "true"/"false" → bool for the bool rows
                # (scene defaults dict stores them as strings to stay
                # JSON-friendly for the future filesystem rescan).
                spec_type = str(getattr(sib.spec, "type", "") or "").lower()
                if spec_type == "bool" and isinstance(val, str):
                    val = val.strip().lower() == "true"
                try:
                    sib.set_value(val)
                except Exception:
                    pass
                continue
            # No row for this key — write directly + manual notify so
            # conditional_on visibility on other rows re-evaluates.
            if self._owner.params.get(k) == val:
                continue
            self._owner.params[k] = val
            if callable(notify):
                try:
                    notify(k, val)
                except Exception:
                    pass

    def _launch_terrain_import(self) -> str:
        """Open the terrain import dialog; return the imported scene_id (or "").

        Self-contained launch point for custom-terrain import — keeps the
        SDK choice popup untouched (no footer-action hook there). On success
        the new scene is already registered (import_terrain_scene), so the
        caller re-selects it to cascade defaults.
        """
        try:
            from application.ui.dialogs.terrain_import_dialog import (
                TerrainImportDialog,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[play_ground.scene_id] terrain import dialog unavailable: {exc}")
            return ""
        sc = self.scene()
        parent = None
        try:
            views = sc.views() if sc is not None else []
            parent = views[0].window() if views else None
        except Exception:
            parent = None
        dlg = TerrainImportDialog(parent)
        try:
            from PyQt6.QtWidgets import QDialog
            if dlg.exec() == QDialog.DialogCode.Accepted:
                return str(dlg.imported_scene_id or "")
        finally:
            dlg.deleteLater()
        return ""


class TaskTypeRow(_ChoicePickerRow):
    """``widget="picker_task_type"`` → 单选 + icon 模式."""
    LEADING_MODE = "icon"


# ---------------------------------------------------------------------------
# StartPointRow helpers — mirror DEMO ``_StartPointChoicePicker``
# (training_node_items.py:4034-4106). Module-level so the popup logic stays
# decoupled from the row class itself; both the popup-build and the
# multi-write side-effects share the same token grammar.
# ---------------------------------------------------------------------------


# Start Point grammar (3 semantic top-tier tokens only).
# Specific run-dir checkpoints and exported policies are picked via a
# secondary CheckpointPickerRow that writes ``checkpoint_id`` —
# ``start_point`` itself never holds a concrete checkpoint path.
_SP_NEW = "__new__"
_SP_LATEST = "__latest__"
_SP_LOAD = "__load__"


def _iter_step(name: str) -> int:
    """``model_4000.pt`` → 4000; unparseable names sort last via -1."""
    from pathlib import Path as _P
    stem = _P(name).stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return -1


def _canvas_has_latest_checkpoint(owner: Any) -> bool:
    """True iff ``base_asset.last_run_id`` resolves to a real run dir
    with at least one ``checkpoints/*.pt`` file under the active project.

    Latest is hidden from the start_point picker whenever this returns
    False — never silently rewritten to another option. Empty
    ``last_run_id`` (canvas has never trained yet), missing run dir
    (manually deleted), and empty ``checkpoints/`` all collapse to False.
    """
    rid = str(owner.params.get("last_run_id") or "").strip()
    if not rid:
        return False
    from application.service.training_assets import get_training_assets
    cache = get_training_assets()
    if cache.project() is None:
        return False
    asset = cache.find_run(rid)
    if asset is None or not asset.path.is_dir():
        return False
    cps = asset.path / "checkpoints"
    if not cps.is_dir():
        return False
    return any(cps.glob("*.pt"))


def _start_point_param_patch(token: str) -> Dict[str, Any]:
    """Coherent sibling-param writes whenever start_point changes. The
    new grammar has only 3 top-level tokens; secondary checkpoint
    selection is driven by ``CheckpointPickerRow`` (which writes its own
    ``checkpoint_id`` + ``load_mode`` patch independently).
    """
    if token == _SP_NEW:
        return {"checkpoint_id": "", "load_mode": "scratch"}
    if token == _SP_LATEST:
        # Latest resolves at compile time via last_run_id; clear
        # checkpoint_id so a stale __load__ pick doesn't bleed in.
        return {"checkpoint_id": "", "load_mode": "resume"}
    if token == _SP_LOAD:
        # Don't touch checkpoint_id here — user picks one via the
        # secondary CheckpointPickerRow which fires its own patch.
        # load_mode lands at warm_start_actor as the safer default;
        # the run/export pick will override to the canonical mode.
        return {"load_mode": "warm_start_actor"}
    return {}


def _resolve_start_point_label(token: Any) -> str:
    """Translate a 3-token start_point into a human button label.

    Unrecognised tokens (legacy ``run:<path>`` / ``asset:<id>`` /
    ``__latest_export__``) shouldn't reach here because the migration
    shim in ``page.from_workflow_dict`` rewrites them on load; if one
    slips through we display it verbatim to surface the data issue.
    """
    if not isinstance(token, str) or not token:
        return "—"
    if token == _SP_NEW:
        return tr("base_asset.start_point.new", "New")
    if token == _SP_LATEST:
        return tr("base_asset.start_point.latest", "Latest")
    if token == _SP_LOAD:
        return tr("base_asset.start_point.load", "Load")
    return token


def _build_start_point_choices(owner: Any) -> tuple:
    """Return ``(choices_keys, meta_map)`` for the start_point popup.

    Top-tier categories only — exactly 2 or 3 entries depending on
    whether this canvas has a resolvable Latest. Specific checkpoint
    picks live in CheckpointPickerRow under the Load branch, not here.
    """
    choices: List[str] = [_SP_NEW]
    meta_map: Dict[str, Dict[str, str]] = {
        _SP_NEW: {
            "title": tr("base_asset.start_point.new", "New"),
            "desc": tr(
                "base_asset.start_point.new_desc",
                "Start training from scratch with random initialization.",
            ),
        },
    }
    if _canvas_has_latest_checkpoint(owner):
        choices.append(_SP_LATEST)
        meta_map[_SP_LATEST] = {
            "title": tr("base_asset.start_point.latest", "Latest"),
            "desc": tr(
                "base_asset.start_point.latest_desc",
                "Continue cumulative training from this canvas's most recent checkpoint.",
            ),
        }
    choices.append(_SP_LOAD)
    meta_map[_SP_LOAD] = {
        "title": tr("base_asset.start_point.load", "Load"),
        "desc": tr(
            "base_asset.start_point.load_desc",
            "Pick any checkpoint or exported policy in the current project.",
        ),
    }
    return choices, meta_map


def _build_project_checkpoint_choices() -> tuple:
    """Enumerate every ``.pt`` under every run + every exported policy
    in the active project. Returns ``(choices, meta_map)`` with the
    ``run:<abs>`` / ``export:<abs>`` token grammar the spec_compiler
    consumes.

    Run-dir listing comes from the project's TrainingAssetsCache
    singleton (already populated at canvas-load / refreshed on
    ``training_run_finished``). Per-step ``.pt`` enumeration isn't
    cached, so we step into ``<run>/checkpoints/`` per row — cheap
    (one ``listdir`` per run dir).
    """
    from pathlib import Path as _P
    from application.service.training_assets import get_training_assets

    cache = get_training_assets()
    choices: List[str] = []
    meta_map: Dict[str, Dict[str, str]] = {}
    for ra in cache.runs():
        cps = ra.path / "checkpoints"
        if not cps.is_dir():
            continue
        pts = sorted(
            cps.glob("*.pt"),
            key=lambda p: _iter_step(p.name),
            reverse=True,
        )
        for pt in pts:
            token = f"run:{pt}"
            choices.append(token)
            meta_map[token] = {
                "title": f"{ra.run_id} · {pt.name}",
                "desc": tr(
                    "base_asset.checkpoint.run_desc",
                    "Run-dir checkpoint (.pt). load_mode auto-snaps to 'resume'.",
                ),
            }
    for pa in cache.policies():
        policy_pt = pa.path / "policy.onnx"
        if not policy_pt.is_file():
            continue
        token = f"export:{policy_pt}"
        choices.append(token)
        algo = pa.algorithm or "?"
        meta_map[token] = {
            "title": f"{pa.policy_id} ({algo})",
            "desc": tr(
                "base_asset.checkpoint.export_desc",
                "Exported policy (.onnx). load_mode auto-snaps to 'warm_start_actor'.",
            ),
        }
    return choices, meta_map


class StartPointRow(_ChoicePickerRow):
    """``widget="picker_start_point"`` → 3-token semantic picker.

    The popup offers ``New`` / ``Latest`` / ``Load`` (Latest is omitted
    when this canvas has no resolvable on-disk checkpoint via
    ``last_run_id``). Specific checkpoint selection happens via the
    secondary ``CheckpointPickerRow`` (``picker_project_checkpoint``)
    which is conditionally visible only under the Load branch.
    """

    LEADING_MODE = "checkbox"
    _OPEN_ON_RELEASE = True

    def _summary_text(self) -> str:  # type: ignore[override]
        return _resolve_start_point_label(self._value)

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        choices, meta_map = _build_start_point_choices(self._owner)
        if not choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            meta_map=meta_map,
            on_commit=self.set_value,
        )
        return True

    def set_value(self, v: Any) -> None:  # type: ignore[override]
        super().set_value(v)
        if not isinstance(v, str):
            return
        patch = _start_point_param_patch(v)
        if not patch:
            return
        # Route patched keys through the sibling ParamRow's own
        # set_value() so the row's internal _value cache and paint sync.
        # Hidden params (no UI row instance) take the direct-write +
        # manual on_param_changed path so conditional_on visibility on
        # other rows still re-evaluates.
        rows_by_key = {r.spec.key: r for r in self._owner._param_rows}
        notify = getattr(self._owner, "on_param_changed", None)
        for k, val in patch.items():
            sib = rows_by_key.get(k)
            if sib is not None:
                sib.set_value(val)
                continue
            if self._owner.params.get(k) == val:
                continue
            self._owner.params[k] = val
            if callable(notify):
                notify(k, val)


class CheckpointPickerRow(_ChoicePickerRow):
    """``widget="picker_project_checkpoint"`` → flat list of every
    project-local checkpoint, runs first then exports.

    Stored value uses the ``run:<abs>`` / ``export:<abs>`` grammar so
    the spec_compiler routes it to the right CheckpointConfig field
    without re-scanning disk. Picking a row auto-snaps the sibling
    ``load_mode`` to the per-source canonical default (``resume`` for
    run-dir .pt, ``warm_start_actor`` for exports).
    """

    LEADING_MODE = "checkbox"
    _OPEN_ON_RELEASE = True

    def _summary_text(self) -> str:  # type: ignore[override]
        v = str(self._value or "")
        if not v:
            return tr("base_asset.checkpoint.empty", "— pick checkpoint")
        from pathlib import Path as _P
        if v.startswith("run:"):
            p = _P(v[4:])
            # <run_id>/checkpoints/<model_N.pt> → "<run_id> · <model_N.pt>"
            return f"{p.parent.parent.name} · {p.name}"
        if v.startswith("export:"):
            p = _P(v[7:])
            return f"export · {p.parent.name}"
        return v

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        choices, meta_map = _build_project_checkpoint_choices()
        if not choices:
            # Loud over silent — log so the user knows there's nothing
            # to pick. No fallback popup, no auto-flip back to __new__.
            log_warning(
                "[base_asset.checkpoint] active project has no run-dir "
                "checkpoints or exported policies to pick from"
            )
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            meta_map=meta_map,
            on_commit=self.set_value,
        )
        return True

    def set_value(self, v: Any) -> None:  # type: ignore[override]
        super().set_value(v)
        if not isinstance(v, str):
            return
        if v.startswith("run:"):
            lm = "resume"
        elif v.startswith("export:"):
            lm = "warm_start_actor"
        else:
            return
        rows_by_key = {r.spec.key: r for r in self._owner._param_rows}
        sib = rows_by_key.get("load_mode")
        if sib is not None:
            sib.set_value(lm)


class ExportBundleRow(_ChoicePickerRow):
    """``widget="picker_export_bundle"`` → 单选."""
    LEADING_MODE = "checkbox"


class MotionLibraryRow(_ChoicePickerRow):
    """``widget="picker_motion_library"`` → 单选."""
    LEADING_MODE = "checkbox"


class ObsComponentsRow(_PickerRowMixin, ParamRow):
    """``widget="picker_obs_components"`` → 多选卡片选单 (multi=True)."""

    _OPEN_ON_RELEASE = True

    def _summary_text(self) -> str:
        v = self._value
        if isinstance(v, (list, tuple)):
            return tr("canvas.row.selected", "{n} selected").replace("{n}", str(len(v)))
        if isinstance(v, str) and v.strip():
            return tr("canvas.row.selected", "{n} selected").replace("{n}", str(len(v.split())))
        return tr("canvas.row.empty", "—")

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._open(event)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._open(event)

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:
        choices = self._resolve_choices()
        if not choices:
            return False
        v = self._value
        if isinstance(v, (list, tuple)):
            current = list(v)
        elif isinstance(v, str) and v.strip():
            current = v.split()
        else:
            current = []

        def _commit(sel: list) -> None:
            # JSON / list 字段：返回 list；自由文本字段：空格分隔
            if (self.spec.type or "").lower() in ("list", "json", "dict"):
                self.set_value(list(sel))
            else:
                self.set_value(" ".join(str(s) for s in sel))

        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=current,
            multi=True,
            leading_mode="checkbox",
            on_commit=_commit,
        )
        return True


# ---- 复杂 editor rows：用 open_widget_popup 把 Node_* QWidget 装入锚定 popup ----

def _parse_dict_value(value: Any) -> dict:
    """Decode a dict-shaped canvas param value. Strict-mode contract:
    JSON parse failure is logged at ERROR level — the empty-dict fallback
    keeps the row widget renderable, but the compile path's ``_as_json``
    will re-detect and emit a ``Severity.ERROR`` issue that ``raise_if_errors``
    halts the play submission on.
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            d = json.loads(value)
        except Exception as exc:
            preview = value if len(value) <= 120 else (value[:117] + "...")
            log_error(
                f"[param_rows] _parse_dict_value: JSON decode failed "
                f"({exc}); raw={preview!r}"
            )
            return {}
        if isinstance(d, dict):
            return d
        log_error(
            f"[param_rows] _parse_dict_value: expected JSON object, got "
            f"{type(d).__name__}"
        )
    return {}


def _parse_list_value(value: Any) -> list:
    """Decode a list-shaped canvas param value. See :func:`_parse_dict_value`
    for the strict-mode contract."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            arr = json.loads(value)
        except Exception as exc:
            preview = value if len(value) <= 120 else (value[:117] + "...")
            log_error(
                f"[param_rows] _parse_list_value: JSON decode failed "
                f"({exc}); raw={preview!r}"
            )
            return []
        if isinstance(arr, list):
            return arr
        log_error(
            f"[param_rows] _parse_list_value: expected JSON array, got "
            f"{type(arr).__name__}"
        )
    return []


def _serialize_for_spec(spec: ParamSpec, value: Any) -> Any:
    """根据 ParamSpec.type 决定回写 dict/list 还是 JSON 字符串.

    Strict-mode contract: a serialization failure here is a schema bug
    (the value the widget produced cannot be encoded as the param's
    declared type). Raise ``ValueError`` rather than silently writing a
    raw Python object back into a JSON-typed param — that would only
    surface downstream as a confusing ``_as_json`` parse failure.
    """
    t = (spec.type or "").lower()
    if t in ("dict", "json", "list"):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(
            f"_serialize_for_spec(spec={getattr(spec, 'name', '?')!r}, "
            f"type={spec.type!r}): cannot encode {type(value).__name__} "
            f"as JSON: {exc}"
        ) from exc


_GEAR_BTN_W = 22.0


class _GearedPickerRowMixin(_PickerRowMixin):
    """``_PickerRowMixin`` + 右侧 ⚙ 按钮入口 / Picker chrome with right-side ⚙.

    主按钮（左半，inline popup）+ ⚙ 按钮（右侧 22px，打开 modal Editor 面板）。
    DEMO 在 ``_TrainingItemEditor`` / ``_RegistryModuleEditor`` 旁边各放了一个
    ⚙ 按钮，分别打开 ``TrainingMotionEditorPanel`` 和 ``RewardEditorPanel``
    两个 modal。本类把这个行内分裂结构在 RELEASE 里复刻一遍，并由子类
    实现 ``_open_modal_editor`` 决定打开哪个 modal。
    """

    def _picker_btn_rect(self) -> QRectF:  # type: ignore[override]
        rect = self._value_rect()
        return QRectF(
            rect.left() + 2, rect.top() + 2,
            rect.width() - _GEAR_BTN_W - 4 - 2, rect.height() - 4,
        )

    def _gear_btn_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(
            rect.right() - _GEAR_BTN_W - 2, rect.top() + 2,
            _GEAR_BTN_W, rect.height() - 4,
        )

    def _interactive_regions(self) -> List[QRectF]:  # type: ignore[override]
        return [self._picker_btn_rect(), self._gear_btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        # Main picker chrome (delegates to mixin's normal flow).
        super()._paint_value(painter, tier)
        # Gear button on the right.
        gear = self._gear_btn_rect()
        bg = QColor(Config.get_color("btn_1"))
        if self._is_hovering(gear):
            bg = bg.lighter(115)
        border = QColor(Config.get_color("border_1"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(gear, 3, 3)
        text_color = QColor(Config.get_color("main_t1"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            gear,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
            "⚙",
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        if self._gear_btn_rect().contains(event.pos()):
            return self._open_modal_editor(event)
        return self._handle_double_click(event)

    def _open_modal_editor(self, event: QGraphicsSceneMouseEvent) -> bool:
        """Subclass hook — open the kind-specific modal editor panel."""
        return False


# Per-channel labels for fixed-dimension obs terms — drives the per-channel
# scale editor (one labeled spinbox per channel). The DIMENSION authority is
# ``application.training.envs.obs_term_engine._TERM_DIM``; this map adds the
# human channel names for the small fixed-dim terms where per-channel scaling is
# meaningful (legged_gym scales the velocity command by [2, 2, 0.25]). Terms not
# listed here (joint_pos / joint_vel / last_action = num_joints / action_dim,
# and any unknown term) fall back to a single scalar spinbox — per-joint obs
# scaling is impractical and effectively never wanted.
_OBS_TERM_CHANNELS: Dict[str, tuple] = {
    "velocity_command": ("vx", "vy", "yaw"),
    "command": ("vx", "vy", "yaw", "head"),
    "base_ang_vel": ("wx", "wy", "wz"),
    "base_lin_vel": ("vx", "vy", "vz"),
    "projected_gravity": ("gx", "gy", "gz"),
    "imu": ("ax", "ay", "az", "wx", "wy", "wz"),
}


def obs_term_channel_labels(term: str) -> Optional[tuple]:
    """Channel labels for a fixed-dim obs term, or ``None`` (→ scalar field)."""
    return _OBS_TERM_CHANNELS.get(str(term))


def obs_scale_to_channels(value: Any, n: int) -> List[float]:
    """Expand a stored scale (scalar / list / ``{"scale": ...}``) to ``n`` floats.

    A scalar broadcasts to every channel; a length-``n`` list maps 1:1; a length
    mismatch is padded/truncated to ``n`` with 1.0 (the editor surfaces the real
    value, never silently drops a per-channel list the way the old float cell
    did). Missing / unparseable → all 1.0."""
    s: Any = value.get("scale") if isinstance(value, dict) else value
    if isinstance(s, (list, tuple)) and s:
        try:
            vals = [float(x) for x in s]
        except (TypeError, ValueError):
            return [1.0] * n
        if len(vals) == n:
            return vals
        if len(vals) == 1:
            return [vals[0]] * n
        return (vals + [1.0] * n)[:n]
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return [float(s)] * n
    return [1.0] * n


def channels_to_obs_scale(vals: List[float]) -> Any:
    """Collapse per-channel floats to a scalar (all equal) or a list.

    All-equal channels round-trip to a bare scalar (minimal canvas diff); any
    difference keeps the per-channel list the compiler emits as
    ``scale=(v0, v1, …)`` (legged_gym ``commands_scale``)."""
    if not vals:
        return 1.0
    first = float(vals[0])
    if all(abs(float(v) - first) < 1e-12 for v in vals):
        return first
    return [float(v) for v in vals]


# Per-obs-item "Data Type" (set in the gear editor's panel): which scale input
# shape the inline row offers. ``scalar`` = one number; ``list`` = a vector
# (per-channel boxes for a small fixed-dim term, JSON for a variable-dim one);
# ``both`` = the row's ⬤ checkbox toggles between the two. Stored in the obs term
# payload as ``{"scale": <number|list>, "data_type": ...}``; the compiler reads
# ``scale`` only (see env_cfg_compiler._OBS_UI_META_FIELDS).
_OBS_DATA_TYPES = ("scalar", "list", "both")


def parse_obs_term_value(value: Any) -> tuple:
    """Decode a stored obs term value into ``(scale_value, data_type)``.

    Accepts the canonical ``{"scale":…, "data_type":…}`` dict, or a bare
    scalar / list (legacy + un-typed) whose ``data_type`` is inferred from the
    shape (scalar→``"scalar"``, list→``"list"``). There is "no default" — an
    item's type is exactly what the panel stored, or the shape it already has.
    """
    if isinstance(value, dict):
        scale = value.get("scale")
        dt = value.get("data_type")
        if dt not in _OBS_DATA_TYPES:
            dt = "list" if isinstance(scale, (list, tuple)) else "scalar"
        scale_v = [float(x) for x in scale] if isinstance(scale, (list, tuple)) else scale
        return scale_v, dt
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value], "list"
    return value, "scalar"


def serialize_obs_term_value(scale_value: Any, data_type: str) -> dict:
    """Encode ``(scale_value, data_type)`` into the canonical obs payload dict."""
    dt = data_type if data_type in _OBS_DATA_TYPES else (
        "list" if isinstance(scale_value, (list, tuple)) else "scalar"
    )
    if isinstance(scale_value, (list, tuple)):
        scale_value = [float(x) for x in scale_value]
    return {"scale": scale_value, "data_type": dt}


class TrainingItemsRow(_GearedPickerRowMixin, ParamRow):
    """``widget="training_items"`` → inline popup + ⚙ modal editor.

        - 主按钮：双击弹锚定 popup（``Node_TrainingItemEditor``）
        - ⚙ 按钮：打开 modal ``RegistryModuleEditorPanel`` (kind="training_motion") —
          DEMO ``TrainingMotionEditorPanel`` 的等价入口。
    """

    def _summary_text(self) -> str:
        d = _parse_dict_value(self._value)
        if not d:
            return "—"
        total = len(d)
        enabled_items = [
            (k, v) for k, v in d.items()
            if isinstance(v, dict) and v.get("enabled", True)
        ]
        enabled = len(enabled_items)
        # §1D — show first 2 enabled items with channel/unit context so the
        # user can see what speed[lo, hi] actually controls. Falls back to
        # the legacy "N/M enabled" form if registers.commands isn't loaded
        # or the items aren't in the registry.
        try:
            from registers import commands as _commands_reg
            previews: list[str] = []
            for item_id, item_cfg in enabled_items[:2]:
                reg = _commands_reg.get_item(item_id)
                speed = item_cfg.get("speed") if isinstance(item_cfg, dict) else None
                if not (isinstance(speed, (list, tuple)) and len(speed) >= 2):
                    speed = None
                if reg is not None and speed is not None:
                    unit = "rad/s" if str(reg.speed_channel).startswith("ang_") else "m/s"
                    lo = float(speed[0])
                    hi = float(speed[1])
                    previews.append(f"{item_id}({lo:g}–{hi:g} {unit})")
                else:
                    previews.append(str(item_id))
            if previews:
                rest = enabled - len(previews)
                tail = f" + {rest} more" if rest > 0 else ""
                return f"{' + '.join(previews)}{tail} · {enabled}/{total}"
        except Exception:
            pass
        return f"{enabled}/{total} enabled"

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        items = _parse_dict_value(self._value)
        view = _view_of(event)
        try:
            body = Node_TrainingItemEditor(items=items, parent=view)
        except TypeError:
            try:
                body = Node_TrainingItemEditor(parent=view)
            except Exception:
                return False

        def _getter(picker: QWidget) -> Any:
            for attr in ("items", "value", "to_dict"):
                fn = getattr(picker, attr, None)
                if callable(fn):
                    try:
                        out = fn()
                        if isinstance(out, dict):
                            return _serialize_for_spec(self.spec, out)
                    except Exception:
                        continue
            return None

        open_widget_popup(
            view=view, row=self, body=body,
            value_getter=_getter, on_commit=self.set_value,
            min_width=520, min_height=380,
        )
        return True

    def _open_modal_editor(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        from application.ui.dialogs import open_training_motion_editor

        items = _parse_dict_value(self._value)
        view = _view_of(event)
        backend = str(self._owner.params.get("backend", "sb3_mujoco") or "sb3_mujoco").strip()
        result = open_training_motion_editor(
            view, dict(items), canvas_backend=backend,
        )
        if result is None:
            return True
        self.set_value(_serialize_for_spec(self.spec, result))
        return True


class StageEditorRow(_PickerRowMixin, ParamRow):
    """``widget="stage_editor"`` → 锚定 popup 装 ``Node_StageSwitchEditorWidget``.

    SDK widget 构造失败（依赖 stage_switch 节点 edge 拓扑）时，降级为 JSON
    code popup —— 仍然是锚定的 ``Qt.WindowType.Popup``。
    """

    def _summary_text(self) -> str:
        arr = _parse_list_value(self._value)
        return f"{len(arr)} stages" if arr else "—"

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        view = _view_of(event)
        stages = _parse_list_value(self._value)
        # 优先尝试 SDK 的 stage editor widget
        try:
            from unitport_sdk import Node_StageSwitchEditorWidget
            body = Node_StageSwitchEditorWidget(stages=stages, parent=view)
        except Exception:
            # 降级：JSON code popup
            text = json.dumps(stages, indent=2, ensure_ascii=False)

            def _commit_text(new_text: str) -> None:
                try:
                    self.set_value(_serialize_for_spec(self.spec, json.loads(new_text)))
                except Exception:
                    self.set_value(new_text)

            open_code_popup(
                view=view, row=self,
                text=text, language="json", validate_json=True,
                on_commit=_commit_text,
            )
            return True

        def _getter(picker: QWidget) -> Any:
            for attr in ("stages", "value", "to_list"):
                fn = getattr(picker, attr, None)
                if callable(fn):
                    try:
                        out = fn()
                        if isinstance(out, list):
                            return _serialize_for_spec(self.spec, out)
                    except Exception:
                        continue
            return None

        open_widget_popup(
            view=view, row=self, body=body,
            value_getter=_getter, on_commit=self.set_value,
            min_width=560, min_height=420,
        )
        return True


class CurveRow(_PickerRowMixin, ParamRow):
    """``widget="curve"`` → 锚定 JSON code popup（待 SDK curve widget 上线再切换）."""

    def _summary_text(self) -> str:
        return "edit curve…"

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        view = _view_of(event)
        v = self._value
        if isinstance(v, str):
            text = v
            language = "json" if v.strip().startswith(("{", "[")) else "python"
        else:
            try:
                text = json.dumps(v, indent=2, ensure_ascii=False)
                language = "json"
            except Exception:
                text = str(v or "")
                language = "python"

        def _commit(new_text: str) -> None:
            if language == "json":
                try:
                    self.set_value(_serialize_for_spec(self.spec, json.loads(new_text)))
                    return
                except Exception:
                    pass
            self.set_value(new_text)

        open_code_popup(
            view=view, row=self,
            text=text, language=language,
            validate_json=(language == "json"),
            on_commit=_commit,
        )
        return True


class BufferSizeRow(_PickerRowMixin, ParamRow):
    """``widget="buffer_size"`` → 锚定 popup 装 ``Node_BufferSizePickerWidget``."""

    def _summary_text(self) -> str:
        return str(self._value or "auto")

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        meta = self.spec.meta or {}
        view = _view_of(event)
        try:
            v = int(self._value if self._value not in (None, "auto", "") else 1_000_000)
        except (TypeError, ValueError):
            v = 1_000_000
        try:
            body = Node_BufferSizePickerWidget(
                value=v,
                minimum=int(meta.get("min", 1)),
                maximum=int(meta.get("max", 10_000_000)),
                widget_id=f"node.buffer_size.{self.spec.key}",
                parent=view,
            )
        except TypeError:
            try:
                body = Node_BufferSizePickerWidget(parent=view)
            except Exception:
                return False

        def _getter(picker: QWidget) -> Any:
            for attr in ("value", "currentValue"):
                fn = getattr(picker, attr, None)
                if callable(fn):
                    try:
                        return int(fn())
                    except Exception:
                        continue
            return None

        open_widget_popup(
            view=view, row=self, body=body,
            value_getter=_getter, on_commit=self.set_value,
            min_width=320, min_height=160,
        )
        return True


# =============================================================================
# RobotAssetPickerRow — widget="picker_robot_asset"
# 单选 SKU picker，choices 来自 registers.robots.list_skus()，meta_map 显示 brand
# 副标题。行内 summary 显示 "{brand} · {name}"，未解析显示红色 "— no asset —"。
# =============================================================================


class RobotAssetPickerRow(_PickerRowMixin, ParamRow):
    """``widget="picker_robot_asset"`` → 锚定 popup 选 robot SKU.

    对照 DEMO ``RobotNode`` 的 asset_id 行：用户必须从机器人注册表
    (``registers.robots``) 选一个 SKU，输入框直接显示 ``brand · name``，
    未解析显示红字提示。
    """

    _OPEN_ON_RELEASE = True

    def _resolve_entry(self):
        """Look up the registry entry for the current value (SKU or alias).

        Returns ``(sku, entry_dict)`` or ``(None, None)``.
        """
        try:
            from registers import robots as _robots_registry
        except Exception:
            return None, None
        raw = str(self._value or "").strip()
        if not raw:
            return None, None
        sku = raw if _robots_registry.get_robot(raw) is not None else (
            _robots_registry.resolve_id(raw)
        )
        if sku is None:
            return None, None
        return sku, _robots_registry.get_robot(sku)

    def _summary_text(self) -> str:
        sku, entry = self._resolve_entry()
        if entry is None:
            return str(self._value or "— select asset —")
        brand = str(entry.get("brand", "")).strip()
        name = str(entry.get("name", entry.get("model", sku))).strip()
        return f"{brand} · {name}" if brand else name

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        sku, entry = self._resolve_entry()
        # 行内按钮 chrome 走 _PickerRowMixin 标准（与 CodeRow / IndexRow / 其它
        # picker 完全一致的色组）。仅"未解析资产"时把按钮内文字改红色提示。
        btn = self._picker_btn_rect()
        self._paint_picker_chrome(painter, btn)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        if entry is None and self._value:
            text_color = QColor(Config.get_color("canvas_safety_error"))
            text = f"— no asset: {self._value} —"
        else:
            text_color = QColor(Config.get_color("main_t1"))
            text = self._summary_text()
        painter.setPen(QPen(text_color))
        fm = QFontMetricsF(font)
        text_rect = btn.adjusted(8, 0, -18, 0)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        # 下拉箭头
        arrow_color = QColor(Config.get_color("main_t1"))
        painter.setBrush(QBrush(arrow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = btn.right() - 10.0
        cy = btn.top() + btn.height() * 0.5
        tri = QPolygonF([
            QPointF(cx, cy + 3.0),
            QPointF(cx - 4.5, cy - 2.5),
            QPointF(cx + 4.5, cy - 2.5),
        ])
        painter.drawPolygon(tri)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        try:
            from registers import robots as _robots_registry
        except Exception:
            return False
        skus = list(_robots_registry.list_skus())
        if not skus:
            return False
        meta: dict = {}
        for sku in skus:
            entry = _robots_registry.get_robot(sku) or {}
            brand = str(entry.get("brand", "")).strip()
            name = str(entry.get("name", entry.get("model", sku))).strip()
            meta[sku] = {
                "title": name or sku,
                "desc": brand,
            }
        # The current value may be an alias — resolve to the canonical SKU so
        # the popup highlights the right card.
        current_sku, _ = self._resolve_entry()
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=skus,
            current=current_sku if current_sku is not None else self._value,
            multi=False,
            leading_mode="checkbox",
            meta_map=meta,
            on_commit=self.set_value,
        )
        return True


# =============================================================================
# BodyMappingTableRow — widget="body_mapping_table"
# Inline self-painted segmented table that mirrors the DEMO body-IR validator
# (Header: Current / Role / St + per-canonical-role rows + unmapped rows +
# warning banner + Refresh button). Variable height — relies on
# ParamRow.preferred_height + NodeItem reflow callback.
# =============================================================================


# Layout constants — equal-weight two-column table (Current | Role).
# Status column removed; role text colour now encodes status (green/red/gray).
_BM_HDR_H = 18
_BM_ROW_H = 20
_BM_BTN_H = 20
_BM_FRAME_PAD = 4
_BM_COL_GAP = 6
_BM_FOOTER_GAP = 4
_BM_PERSIST_DEBOUNCE_MS = 0  # fire on next event-loop tick

_BM_LABEL_UNASSIGNED = "(Unassigned)"
_BM_LABEL_OUT_OF_SCOPE = "(Out of Scope)"
# Translation slots for the labels above. The constants stay as the dict keys
# / serialised payload values used by BodyMappingTableRow's selector logic
# (they show up in graph.json + the popup choices list, so changing them
# would break round-tripping); the slots are looked up at paint time only.
_BM_LABEL_UNASSIGNED_KEY = "canvas.body_mapping.label_unassigned"
_BM_LABEL_OUT_OF_SCOPE_KEY = "canvas.body_mapping.label_out_of_scope"


def _resolve_active_format(params: Dict[str, Any], asset: Any) -> str:
    """Effective asset format for a Robot node — single source of truth.

    Returns one of ``"MJCF"`` / ``"USD"`` / ``"URDF"`` (uppercase), or ``""``
    when nothing is available. Algorithm mirrors
    :meth:`ActiveFileTableRow._current_active`:

      1. Explicit ``active_override`` param (``"mjcf"`` / ``"usd"`` /
         ``"urdf"``) wins unconditionally — user clicked a row in the
         upper Asset Files table.
      2. Otherwise pick by ``backend`` param: ``isaac_lab`` prefers
         USD > URDF > MJCF (matches Isaac Sim's native asset
         pipeline); everything else (sb3 / mujoco / ...) prefers
         MJCF > USD > URDF.
      3. Within the preferred order, return the first format the asset
         actually has (either local path or, for USD, a Nucleus URL).

    The Body Mapping table and the Asset Files table MUST agree on
    which format is live — otherwise the user picks USD in the upper
    table but sees MJCF joints below, which is a hard contract
    violation. Always go through this helper; never read
    ``active_override`` raw.
    """
    raw = str(params.get("active_override", "") or "").strip().lower()
    if raw in ("mjcf", "usd", "urdf"):
        return raw.upper()
    if asset is None:
        return ""
    backend = str(params.get("backend", "sb3_mujoco") or "sb3_mujoco").strip()
    if backend == "isaac_lab":
        order = ("usd", "urdf", "mjcf")
    else:
        order = ("mjcf", "usd", "urdf")
    for kind in order:
        if getattr(asset, f"{kind}_path", None) is not None:
            return kind.upper()
        if kind == "usd" and getattr(asset, "usd_url", None):
            return kind.upper()
    return ""


# Public alias: Mission Control's _RobotConfigSection (Training Config
# Perspective → Joint Mapping table) MUST use the same format resolution
# as the canvas, otherwise canvas-side overrides written under e.g. "USD"
# are read back from "MJCF" and the table looks all-Unassigned.
resolve_active_format = _resolve_active_format


class BodyMappingTableRow(ParamRow):
    """Inline body→IR-role validator table for the Robot node.

    Rebuilds itself from the owner Robot node's ``asset_id`` parameter:
        asset_id  →  registers.robots.resolve_id  →  sku
                 →  RobotAssetService.resolve(sku)  →  RobotAsset
                 →  BodyIRMapper.from_robot_asset(asset)
                 →  apply_user_overrides(mapper, get_body_ir_overrides(sku))

    User edits go through:
        mapper.reassign_role / mark_out_of_scope / clear_out_of_scope
        → extract_user_overrides(mapper)
        → RobotAssetService.set_body_ir_overrides(sku, overrides)   ← single source of truth
        → set_value(json.dumps(mapper.to_dict()))                    ← transient param mirror
    """

    _OPEN_ON_RELEASE = True

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._mapper = None  # BodyIRMapper | None
        self._sku: str = ""
        self._asset_resolved: bool = False
        self._family: str = "generic"
        self._visible_roles: list = []   # IRRole list
        self._unmapped: list = []        # body name list
        self._out_of_scope: list = []    # body name list
        self._catalog_labels: list = []  # canonical role labels for picker
        self._label_to_role_id: dict = {}
        self._row_rects: list = []       # [(kind, payload, rect_role_col, rect_full)]
        # Tooltip targets: [(rect, full_text)] for any cell whose drawn text was
        # elided. hoverMoveEvent looks these up to drive QGraphicsItem.toolTip.
        self._tooltip_cells: list = []
        self._refresh_btn_rect: QRectF = QRectF()
        self._asset_id_last: str = ""    # cached for poll detection
        self._height = float(self._compute_height_empty())
        self._height_callbacks: list = []
        # 来自 mission_panel Robot Config 卡片对 RobotAssetService.set_body_ir_overrides
        # 的写入不经过 canvas_param_changed —— 直接订阅 RobotAssetService.changed
        # 以便外部覆写立即在画布表格里反映出来。
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            get_robot_asset_service().changed.connect(self._schedule_rebuild)
        except Exception:
            pass
        # Initial build — defer one tick so owner_node finishes constructing.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

    # ------------------------------------------------------------------ height

    def preferred_height(self) -> float:
        return float(self._height)

    def on_height_changed(self, callback) -> None:
        """Register a callback notified whenever this row's height changes.

        NodeItem subscribes during reflow so it can rearrange downstream rows.
        """
        if callable(callback) and callback not in self._height_callbacks:
            self._height_callbacks.append(callback)

    def _set_height(self, h: float) -> None:
        h = float(max(PARAM_ROW_H, h))
        if abs(h - self._height) < 0.5:
            return
        self.prepareGeometryChange()
        self._height = h
        self.update()
        for cb in list(self._height_callbacks):
            try:
                cb(h)
            except Exception:
                pass

    def _compute_height_empty(self) -> int:
        # Header + one "select asset" row + footer button.
        return (
            _BM_FRAME_PAD * 2 + _BM_HDR_H + _BM_ROW_H
            + _BM_FOOTER_GAP + _BM_BTN_H + 2
        )

    def _compute_height(self, n_role_rows: int, has_warn: bool) -> int:
        warn_h = _BM_ROW_H if has_warn else 0
        if n_role_rows == 0:
            n_role_rows = 1  # "no roles" placeholder
        return (
            _BM_FRAME_PAD * 2
            + _BM_HDR_H
            + n_role_rows * _BM_ROW_H
            + warn_h
            + _BM_FOOTER_GAP + _BM_BTN_H + 2
        )

    # ---------------------------------------------------------- data resolve

    def _resolve_asset_and_mapper(self):
        """Return ``(sku, mapper)`` or ``("", None)`` if unresolved.

        Stage 3 (per-format schema): the widget targets a single format
        per render. Today we use the Robot node's ``active_override`` to
        pin the format (matches what the canvas will train against);
        Stage 5 adds a MJCF/USD/URDF tab so the user can edit any
        declared format.
        """
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            from application.training.body_ir import (
                BodyIRMapper, apply_user_overrides,
            )
            from registers import resolve_robot_sku
            from registers import robots as _robots_registry
        except Exception:
            return "", None

        raw = str(self._owner.params.get("asset_id", "") or "").strip()
        self._asset_id_last = raw
        if not raw:
            return "", None
        sku = raw if _robots_registry.get_robot(raw) is not None else (
            _robots_registry.resolve_id(raw)
        )
        if sku is None and "." in raw:
            brand, _, model = raw.partition(".")
            sku = resolve_robot_sku(brand, model)
        if sku is None:
            return "", None
        svc = get_robot_asset_service()
        asset = svc.resolve(sku)
        if asset is None:
            return "", None

        # The display format MUST match the upper Asset Files table —
        # _resolve_active_format is the single source of truth (honors
        # active_override, falls back to backend-aware default).
        display_fmt = _resolve_active_format(self._owner.params, asset) or None

        mapper = BodyIRMapper.from_robot_asset(asset, active_format=display_fmt)
        # Read per-format overrides (legacy flat overrides auto-migrate
        # to MJCF; new per-format overrides keyed by format).
        if display_fmt:
            overrides = svc.get_body_ir_overrides(sku, fmt=display_fmt)
        else:
            overrides = svc.get_body_ir_overrides(sku)
        if overrides:
            apply_user_overrides(mapper, overrides)
        return sku, mapper

    def _refresh_catalog_choices(self):
        try:
            from application.training.body_ir import get_canonical_roles
        except Exception:
            self._catalog_labels = []
            self._label_to_role_id = {}
            return
        catalog = get_canonical_roles(self._family)
        self._catalog_labels = [c.label for c in catalog]
        self._label_to_role_id = {c.label: c.role_id for c in catalog}

    # ------------------------------------------------------------ rebuild

    def _rebuild(self) -> None:
        """Recompute mapper + visible role rows; trigger geometry change."""
        sku, mapper = self._resolve_asset_and_mapper()
        self._sku = sku
        self._mapper = mapper
        self._asset_resolved = mapper is not None
        if mapper is None:
            self._visible_roles = []
            self._unmapped = []
            self._out_of_scope = []
            self._family = "generic"
            self._refresh_catalog_choices()
            self._set_height(self._compute_height_empty())
            self.update()
            return

        self._family = mapper.family
        # Joint-centric filter: the table only ever surfaces canonical roles
        # that map to ACTUATED joints for this morphology (hips/thighs/calves
        # for quadruped, hips/knees/shoulders/elbows/wrists for biped, etc.).
        # Structural / kinematic roles (`base`, `feet`, `head`, `hands`) are
        # NOT joints the user maps from actuated DOF — they were producing a
        # "—|Base" noise row whenever the floating-base body went null. Drop
        # them at the source; mission-control body_mapping table follows the
        # same rule (training_config_card._BodyMappingTable._rebuild_rows).
        try:
            from application.training.body_ir import get_joint_ir_roles, get_role
            joint_role_ids = set(get_joint_ir_roles(mapper.family))
        except Exception:
            joint_role_ids = set()
            get_role = None
        all_roles = list(mapper.roles)

        def _is_required(role_id: str) -> bool:
            if get_role is None:
                return False
            try:
                canon = get_role(self._family, role_id)
            except Exception:
                return False
            return bool(canon and canon.required)

        if joint_role_ids:
            # Joint-centric filter: only canonical roles that map to ACTUATED
            # joints for this morphology. AND only roles that are either
            # (a) resolved (`role.body` filled) or (b) declared required in
            # the IR catalog — empty optional rows are noise (every newly
            # added humanoid multi-DOF / finger role is `required=false`,
            # so without this guard the table balloons to ~40 empty rows).
            self._visible_roles = [
                r for r in all_roles
                if r.role_id in joint_role_ids
                and (r.body is not None or _is_required(r.role_id))
            ]
        else:
            # Family has no canonical actuated-joint categories (wheeled /
            # generic). Fall back to any role with a body so the table isn't
            # blank for those rigs.
            self._visible_roles = [r for r in all_roles if r.body is not None]
        self._unmapped = list(mapper.unmapped_bodies())
        self._out_of_scope = list(mapper.out_of_scope_bodies())
        self._refresh_catalog_choices()

        # Mirror current snapshot into the transient param string. This is NOT
        # persistence — just keeps body_mapping param readable to consumers.
        try:
            self._owner.params[self.spec.key] = json.dumps(
                mapper.to_dict(), ensure_ascii=False
            )
            self._value = self._owner.params[self.spec.key]
        except Exception:
            pass

        n_rows = len(self._visible_roles) + len(self._unmapped) + len(self._out_of_scope)
        self._set_height(self._compute_height(n_rows, bool(self._unmapped)))
        self.update()

    def _schedule_rebuild(self) -> None:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(_BM_PERSIST_DEBOUNCE_MS, self._rebuild)

    def _on_refresh_clicked(self) -> None:
        """Refresh button — dump the active format from the live asset and
        rebuild.

        Default behaviour: pick the format from the Robot node's
        ``active_override`` param (USD when unset — Isaac Lab training uses
        USD), bump the live asset via
        :meth:`RobotAssetService.dump_and_persist` (USD path spawns Isaac
        Lab venv subprocess for 5-30s under a busy cursor), then rebuild.
        The persist step writes ``<USER_CONFIG_DIR>/registers/robots_custom.json``
        and auto-emits the service ``changed`` signal so Mission Control
        and any other listeners also refresh.

        On any dump failure (no asset path / subprocess crash / validation
        error) we fall back to :meth:`_do_reload_only` so the click is
        never a no-op — at minimum any out-of-band overlay write is
        picked up. Errors go to the log (canvas context: no dialogs).
        """
        raw = str(self._owner.params.get("asset_id", "") or "").strip()
        sku = self._sku
        if not sku:
            log_warning(
                f"[body_mapping] refresh: no sku resolved for asset_id={raw!r}, "
                f"falling back to reload-only"
            )
            self._do_reload_only()
            return

        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
        except Exception as exc:
            log_warning(f"[body_mapping] refresh import failed: {exc!r}")
            return

        # Resolve fmt via the shared helper so Refresh dumps the EXACT
        # same format the table is going to display (and the trainer
        # is going to consume). Anything else breaks the contract that
        # the upper Asset Files table and the lower Body Mapping table
        # always agree.
        svc = get_robot_asset_service()
        asset = svc.resolve(sku)
        fmt = _resolve_active_format(self._owner.params, asset)
        if not fmt:
            log_warning(
                f"[body_mapping] refresh: no usable format for sku={sku!r} "
                f"(asset has no MJCF/USD/URDF path/url), falling back to reload-only"
            )
            self._do_reload_only()
            return

        log_info(
            f"[body_mapping] refresh dump {fmt} start: asset_id={raw!r} sku={sku!r}"
        )

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            result = svc.dump_and_persist(sku, fmt)
        finally:
            QApplication.restoreOverrideCursor()

        if not result.ok:
            kind = result.error_kind or "unknown"
            msg = result.error or ""
            level = log_error if kind == "unexpected" else log_warning
            level(
                f"[body_mapping] refresh dump {fmt} {kind} sku={sku!r}: {msg}; "
                f"falling back to reload-only"
            )
            self._do_reload_only()
            return

        log_info(
            f"[body_mapping] refresh dump {fmt} sku={sku!r}: "
            f"{len(result.joints)} joints ({result.n_joints_resolved} resolved), "
            f"{len(result.bodies)} bodies ({result.n_bodies_resolved} resolved)"
        )

        # set_discovered_bodies already reloaded the registry and emitted
        # changed; rebuild this widget explicitly so the table re-renders
        # immediately rather than waiting for the next paint tick.
        self._rebuild()
        n_resolved = (
            sum(1 for r in self._mapper.roles if r.resolved)
            if self._mapper is not None else 0
        )
        n_unmapped = len(self._unmapped)
        log_info(
            f"[body_mapping] refresh OK asset_id={raw!r} sku={sku!r} fmt={fmt!r}: "
            f"{n_resolved} roles resolved, {n_unmapped} unmapped bodies"
        )

    def _do_reload_only(self) -> None:
        """Fallback for Refresh: re-read user-overlay registry then rebuild.

        Used when dump_and_persist isn't possible (no SKU, no asset path,
        subprocess crash, etc.) so the Refresh click still performs the
        minimum useful work — picking up any out-of-band overlay edits.
        """
        try:
            from registers import RegistryHub
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
        except Exception as exc:
            log_warning(f"[body_mapping] reload-only import failed: {exc!r}")
            self._rebuild()
            return

        try:
            RegistryHub.reload()
        except Exception as exc:
            log_error(f"[body_mapping] reload-only: RegistryHub.reload failed: {exc!r}")
            self._rebuild()
            return

        try:
            get_robot_asset_service().changed.emit()
        except Exception:
            pass

        self._rebuild()

    # ------------------------------------------------------------ persistence

    def _persist(self) -> None:
        if self._mapper is None or not self._sku:
            return
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            from application.training.body_ir import extract_user_overrides
        except Exception:
            return
        overrides = extract_user_overrides(self._mapper)
        # Persist into the per-format slot the widget is displaying so
        # MJCF and USD overrides don't clobber each other (Stage 3).
        # Resolve through the shared helper so persist + display + dump
        # never disagree on which format is live.
        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )
        svc = get_robot_asset_service()
        asset = svc.resolve(self._sku) if self._sku else None
        fmt = _resolve_active_format(self._owner.params, asset) or None
        svc.set_body_ir_overrides(
            self._sku, overrides if overrides else None, fmt=fmt,
        )

    # ------------------------------------------------------------ painting

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        # The body table is a full-row widget — we paint over the entire row
        # (including the key column) but keep the standard ParamRow chrome
        # (key label + separator) drawn by the base class. Our drawing starts
        # from x=KEY_PAD_LEFT below the param-key label, but actually the
        # table reads better full-width: skip the standard separator on this
        # row by occupying the value half plus a chunk of the key half.
        # However ParamRow.paint() always draws the key label + separator
        # before calling us. We accept that — DEMO does the same (key shown
        # to the left, table on the right).
        # Constrain the table to the value half (SEP_X..width-VALUE_PAD_RIGHT).
        # No third (status) column — status is encoded by the role text colour.
        rect_full = QRectF(
            SEP_X + 4, _BM_FRAME_PAD,
            self._width - (SEP_X + 4) - VALUE_PAD_RIGHT,
            self._height - _BM_FRAME_PAD * 2,
        )
        bg = QColor(Config.get_color("bg_1", "#1A1A1A"))
        border = QColor(Config.get_color("border_2", "#3d3d3d"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(rect_full, 4, 4)

        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        fm = QFontMetricsF(font)

        muted = QColor(Config.get_color("canvas_node_param_text", "#aaaaaa"))
        text_c = QColor(Config.get_color("main_t1", "#cccccc"))
        ok_c = QColor(Config.get_color("safe_zone", "#4CAF50"))
        err_c = QColor(Config.get_color("danger", "#F44336"))
        skip_c = QColor(Config.get_color("border_2", "#777777"))

        # Two equal-weight columns. No min-width clamp — Qt's elidedText
        # handles overflow; full text is surfaced via hover tooltip.
        inner_x = rect_full.left() + 4
        inner_w = max(0.0, rect_full.width() - 8)
        col_w = max(0.0, (inner_w - _BM_COL_GAP) / 2.0)
        col1_x = inner_x
        col2_x = inner_x + col_w + _BM_COL_GAP

        def _draw_cell(
            x: float, y: float, w: float, h: float,
            text: str, colour: QColor,
            *, padding: float = 4.0,
        ) -> bool:
            """Draw text inside (x,y,w,h) with elide-right; return True if elided.

            Caller appends an entry to ``self._tooltip_cells`` for elided cells
            so hoverMoveEvent can surface the full string via tooltip.
            """
            cell_rect = QRectF(x, y, w, h)
            text_rect = cell_rect.adjusted(padding, 0, -padding, 0)
            avail = max(0.0, text_rect.width())
            elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, avail)
            painter.setPen(QPen(colour))
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                elided,
            )
            return elided != text

        # ---- Header row ----
        y = rect_full.top() + 2
        painter.setPen(QPen(muted))
        hdr_font = QFont(font)
        hdr_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(hdr_font)
        painter.drawText(
            QRectF(col1_x + 4, y, col_w, _BM_HDR_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            tr("canvas.body_mapping.col_current", "Current"),
        )
        painter.drawText(
            QRectF(col2_x + 4, y, col_w, _BM_HDR_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            tr("canvas.body_mapping.col_role", "Role"),
        )
        painter.setFont(font)
        y += _BM_HDR_H

        # Reset hit/tooltip caches; they are rebuilt every paint call.
        self._row_rects = []
        self._tooltip_cells = []

        if not self._asset_resolved:
            painter.setPen(QPen(err_c))
            painter.drawText(
                QRectF(inner_x, y, inner_w, _BM_ROW_H),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                tr("canvas.body_mapping.select_asset", "Select an asset_id"),
            )
            self._paint_refresh_button(painter, rect_full, font)
            return

        # ---- Status colour helpers ----
        try:
            from application.training.body_ir import get_role
        except Exception:
            get_role = None

        def _role_status_colour(role) -> QColor:
            """Green=resolved, red=required & unfilled, gray=optional & unfilled."""
            if role.resolved:
                return ok_c
            required = False
            if get_role is not None:
                canon = get_role(self._family, role.role_id)
                if canon is not None:
                    required = bool(canon.required)
            return err_c if required else skip_c

        zebra = QColor(Config.get_color("bg_1", "#1A1A1A")).lighter(108)

        # ---- Visible canonical role rows ----
        idx = 0
        for role in self._visible_roles:
            if idx % 2:
                painter.setBrush(QBrush(zebra))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(inner_x, y, inner_w, _BM_ROW_H))
            # col1: body name (untinted; status lives on the role label)
            body_txt = role.body if role.body else "—"
            body_rect = QRectF(col1_x, y, col_w, _BM_ROW_H)
            if _draw_cell(
                col1_x, y, col_w, _BM_ROW_H,
                body_txt, text_c if role.body else muted,
            ):
                self._tooltip_cells.append((body_rect, body_txt))
            # col2: role label, colour encodes status. Hit-rect only when the
            # slot is filled (re-routing requires an existing body to move).
            role_rect = QRectF(col2_x, y, col_w, _BM_ROW_H)
            role_colour = _role_status_colour(role)
            if role.body is not None:
                btn_bg = QColor(Config.get_color("btn_1", "#343636"))
                if self._hover_rect is not None and self._hover_rect == role_rect:
                    btn_bg = btn_bg.lighter(115)
                painter.setBrush(QBrush(btn_bg))
                painter.setPen(QPen(border, 0.8))
                painter.drawRoundedRect(role_rect.adjusted(0, 2, 0, -2), 3, 3)
                if _draw_cell(
                    col2_x, y, col_w, _BM_ROW_H, role.label, role_colour,
                ):
                    self._tooltip_cells.append((role_rect, role.label))
                self._row_rects.append(
                    ("role_assigned", role.body, role.role_id, role_rect)
                )
            else:
                if _draw_cell(
                    col2_x, y, col_w, _BM_ROW_H, role.label, role_colour,
                ):
                    self._tooltip_cells.append((role_rect, role.label))
            y += _BM_ROW_H
            idx += 1

        # ---- Unmapped + out-of-scope rows ----
        for body in self._unmapped + self._out_of_scope:
            is_oos = body in self._out_of_scope
            if idx % 2:
                painter.setBrush(QBrush(zebra))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(inner_x, y, inner_w, _BM_ROW_H))
            # col1: body name
            body_rect = QRectF(col1_x, y, col_w, _BM_ROW_H)
            if _draw_cell(col1_x, y, col_w, _BM_ROW_H, body, text_c):
                self._tooltip_cells.append((body_rect, body))
            # col2: role label — red for unmapped (needs adapt), gray for OOS.
            role_rect = QRectF(col2_x, y, col_w, _BM_ROW_H)
            btn_bg = QColor(Config.get_color("btn_1", "#343636"))
            if self._hover_rect is not None and self._hover_rect == role_rect:
                btn_bg = btn_bg.lighter(115)
            painter.setBrush(QBrush(btn_bg))
            painter.setPen(QPen(border, 0.8))
            painter.drawRoundedRect(role_rect.adjusted(0, 2, 0, -2), 3, 3)
            label = _BM_LABEL_OUT_OF_SCOPE if is_oos else _BM_LABEL_UNASSIGNED
            label_display = tr(
                _BM_LABEL_OUT_OF_SCOPE_KEY if is_oos else _BM_LABEL_UNASSIGNED_KEY,
                label,
            )
            label_colour = skip_c if is_oos else err_c
            if _draw_cell(col2_x, y, col_w, _BM_ROW_H, label_display, label_colour):
                self._tooltip_cells.append((role_rect, label_display))
            self._row_rects.append(
                ("body_extra", body, "oos" if is_oos else "unmapped", role_rect)
            )
            y += _BM_ROW_H
            idx += 1

        # ---- Warning banner (12px, same colour as required-unfilled) ----
        if self._unmapped:
            preview = ", ".join(self._unmapped[:3])
            more = f" (+{len(self._unmapped) - 3} more)" if len(self._unmapped) > 3 else ""
            warn_text = (
                tr(
                    "canvas.body_mapping.warn_joints",
                    "⚠ {n} joint(s) not matched — assign or mark out of scope: {preview}",
                )
                .replace("{n}", str(len(self._unmapped)))
                .replace("{preview}", f"{preview}{more}")
            )
            warn_rect = QRectF(inner_x + 2, y, inner_w - 4, _BM_ROW_H)
            warn_font = QFont(font)
            warn_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(warn_font)
            painter.setPen(QPen(err_c))
            avail = max(0.0, warn_rect.width())
            elided_warn = fm.elidedText(warn_text, Qt.TextElideMode.ElideRight, avail)
            painter.drawText(
                warn_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                elided_warn,
            )
            if elided_warn != warn_text:
                self._tooltip_cells.append((warn_rect, warn_text))
            painter.setFont(font)
            y += _BM_ROW_H

        if not self._visible_roles and not self._unmapped and not self._out_of_scope:
            painter.setPen(QPen(muted))
            painter.drawText(
                QRectF(inner_x, y, inner_w, _BM_ROW_H),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                tr("canvas.body_mapping.no_roles", "No body roles for this asset"),
            )
            y += _BM_ROW_H

        # ---- Refresh button ----
        self._paint_refresh_button(painter, rect_full, font)

    def _paint_refresh_button(
        self, painter: QPainter, rect_full: QRectF, font: QFont,
    ) -> None:
        btn_y = rect_full.bottom() - _BM_BTN_H - 2
        btn_x = rect_full.left() + 4
        btn_w = min(80.0, rect_full.width() - 8)
        self._refresh_btn_rect = QRectF(btn_x, btn_y, btn_w, _BM_BTN_H)
        bg = QColor(Config.get_color("btn_1", "#343636"))
        if self._hover_rect is not None and self._hover_rect == self._refresh_btn_rect:
            bg = bg.lighter(120)
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(self._refresh_btn_rect, 3, 3)
        painter.setPen(QPen(QColor(Config.get_color("main_t1", "#D6D3C7"))))
        painter.setFont(font)
        painter.drawText(
            self._refresh_btn_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            tr("canvas.body_mapping.refresh", "Refresh"),
        )

    # ------------------------------------------------------------ hit-test

    def _interactive_regions(self) -> List[QRectF]:  # type: ignore[override]
        regions: List[QRectF] = []
        for entry in self._row_rects:
            regions.append(entry[3])
        if self._refresh_btn_rect.width() > 0:
            regions.append(self._refresh_btn_rect)
        return regions

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        pos = event.pos()
        # Refresh button?
        if self._refresh_btn_rect.contains(pos):
            self._on_refresh_clicked()
            return True
        # Row role-cell?
        for entry in self._row_rects:
            kind, payload, payload2, rect = entry
            if not rect.contains(pos):
                continue
            self._open_role_picker(event, kind, payload, payload2)
            return True
        return False

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._handle_click(event)

    # ------------------------------------------------------------ tooltips
    # Per-cell tooltip: when hovering a cell whose drawn text was elided we
    # set the QGraphicsItem.toolTip to the full string. Move outside any
    # tooltip cell → clear tooltip. This lives next to ParamRow's hover-rect
    # logic — we keep the base class behaviour (highlight on interactive
    # cells) by always invoking super.

    def hoverMoveEvent(self, event):  # type: ignore[override]
        pos = event.pos()
        full = ""
        for cell_rect, cell_text in self._tooltip_cells:
            if cell_rect.contains(pos):
                full = cell_text
                break
        # Only mutate when the value actually changes — avoids fighting Qt's
        # own tooltip-show timer.
        if self.toolTip() != full:
            self.setToolTip(full)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):  # type: ignore[override]
        if self.toolTip():
            self.setToolTip("")
        super().hoverLeaveEvent(event)

    def _open_role_picker(
        self, event: QGraphicsSceneMouseEvent,
        kind: str, payload, payload2,
    ) -> None:
        if self._mapper is None:
            return
        # Build choice list: (Unassigned) + catalog labels + (Out of Scope)
        choices = [_BM_LABEL_UNASSIGNED] + self._catalog_labels + [_BM_LABEL_OUT_OF_SCOPE]
        # current selection
        if kind == "role_assigned":
            body, role_id = payload, payload2
            current = next(
                (r.label for r in self._mapper.roles if r.role_id == role_id),
                _BM_LABEL_UNASSIGNED,
            )
        else:
            body = payload
            current = (
                _BM_LABEL_OUT_OF_SCOPE if payload2 == "oos"
                else _BM_LABEL_UNASSIGNED
            )

        def _on_pick(selected):
            if selected is None:
                return
            if isinstance(selected, list):
                if not selected:
                    return
                val = selected[0]
            else:
                val = selected
            self._apply_role_pick(body, val)

        def _label_for(raw):
            if raw == _BM_LABEL_UNASSIGNED:
                return tr(_BM_LABEL_UNASSIGNED_KEY, raw)
            if raw == _BM_LABEL_OUT_OF_SCOPE:
                return tr(_BM_LABEL_OUT_OF_SCOPE_KEY, raw)
            return str(raw)

        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=current,
            multi=False,
            leading_mode="checkbox",
            on_commit=_on_pick,
            label_resolver=_label_for,
        )

    def _apply_role_pick(self, body: str, label: str) -> None:
        if self._mapper is None or not body:
            return
        if label == _BM_LABEL_UNASSIGNED:
            if body in self._mapper.out_of_scope_bodies():
                self._mapper.clear_out_of_scope(body)
            else:
                self._mapper.reassign_role(body, None)
        elif label == _BM_LABEL_OUT_OF_SCOPE:
            self._mapper.mark_out_of_scope(body)
        else:
            new_role_id = self._label_to_role_id.get(label)
            if new_role_id:
                self._mapper.reassign_role(body, new_role_id)
            else:
                return
        self._persist()
        self._schedule_rebuild()

    # ---------------------------------------- owner asset_id change polling

    def maybe_refresh_for_asset_change(self) -> None:
        """Called by NodeItem after a sibling row's set_value (asset_id)
        committed. Compares current asset_id against the cached one and
        reschedules rebuild if they differ. Cheap no-op when unchanged.
        """
        cur = str(self._owner.params.get("asset_id", "") or "").strip()
        if cur != self._asset_id_last:
            self._schedule_rebuild()


# =============================================================================
# ActiveFileTableRow — widget="active_file_table"
# Robot.active_override 的多行表格控件。三行（MJCF / USD / URDF），每行显示文件
# 类型标签 + 已注册路径状态 + ✔ 标记。点击 Loaded 行切换 active_override 到该
# 类型；点击当前 active 行恢复 "auto"。布局与 DEMO
# ``training_node_items.py:_make_active_file_table`` 视觉契约对齐，但走 SDK
# Config.get_color 主题与 QGraphicsItem self-paint。
# =============================================================================

_AF_HDR_H = 14
_AF_ROW_H = 20
_AF_FRAME_PAD = 4
_AF_COL_GAP = 6


class MjcfOffsetBadgeRow(ParamRow):
    """Read-only status badge for the per-SKU MJCF base spawn-Z offset.

    Lives on the Robot Node. Polls
    :class:`RobotAssetService.get_mjcf_base_offset` at paint time and renders
    one of three states:

        - ``"+XX.X mm (auto)"``         — calibration succeeded; offset
                                           is applied at every MuJoCo
                                           spawn site.
        - ``"not applicable"``          — family declares no landmark set
                                           (fixed-base manipulator); no
                                           correction needed.
        - ``"uncalibrated"``            — overlay missing; runtime
                                           spawns will WARN. The next
                                           Robot Node refresh re-submits
                                           the one-shot calibration task
                                           via
                                           :func:`ensure_mjcf_base_offset_cached`.

    Click is a no-op — this is informational; manual recalibration is
    available via the Robot Asset card. See plan
    ``curious-nibbling-plum.md`` and
    :mod:`application.training.validation.mjcf_base_calibration`.
    """

    _OPEN_ON_RELEASE = False

    def _interactive_regions(self) -> List[QRectF]:
        return []

    def _resolve_sku(self) -> str:
        params = getattr(self._owner, "params", None) or {}
        raw = str(params.get("asset_id", "") or "").strip()
        if not raw:
            return ""
        try:
            from registers import robots as _robots_registry
            sku = _robots_registry.resolve_to_sku(raw)
            return str(sku or raw)
        except Exception:
            return raw

    def _read_overlay(self) -> Optional[Dict[str, Any]]:
        sku = self._resolve_sku()
        if not sku:
            return None
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            return get_robot_asset_service().get_mjcf_base_offset(sku)
        except Exception:
            return None

    def _summary_text(self) -> str:
        overlay = self._read_overlay()
        if not isinstance(overlay, dict):
            return tr(
                "canvas.robot.mjcf_offset.uncalibrated",
                default="uncalibrated",
            )
        status = str(overlay.get("status", "") or "")
        if status == "not_applicable":
            return tr(
                "canvas.robot.mjcf_offset.not_applicable",
                default="not applicable",
            )
        if status == "calibrated":
            try:
                off_z = float(overlay.get("offset_z", 0.0) or 0.0)
            except (TypeError, ValueError):
                off_z = 0.0
            return f"+{off_z * 1000.0:.1f} mm (auto)"
        return tr(
            "canvas.robot.mjcf_offset.uncalibrated",
            default="uncalibrated",
        )

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = self._value_rect()
        text_color = QColor(Config.get_color("canvas_node_param_text"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        text_rect = rect.adjusted(8, 0, -8, 0)
        elided = fm.elidedText(
            self._summary_text(),
            Qt.TextElideMode.ElideRight,
            text_rect.width(),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        return False  # read-only


class ActiveFileTableRow(ParamRow):
    """Robot.active_override 三行 file-type 表格 / 3-row file-type table.

    动态高度（preferred_height）= 4 (frame top) + 14 (header) + 3*20 (rows) + 4 (frame bottom)
    通过 ``asset_id`` 解析当前 robot 的 RobotAsset；点击 Loaded 行写入
    ``active_override``；点击 active 行恢复 ``"auto"``。
    """

    _OPEN_ON_RELEASE = True
    _ROWS_SPEC = (
        ("MJCF", "mjcf"),
        ("USD",  "usd"),
        ("URDF", "urdf"),
    )

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._asset = None              # RobotAsset | None
        self._asset_id_last: str = ""
        self._row_hits: List[tuple] = []   # [(file_kind, rect, loaded)]
        self._height = float(self._compute_height())
        self._height_callbacks: list = []
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

    # ---- height contract ----

    def preferred_height(self) -> float:
        return float(self._height)

    def on_height_changed(self, callback) -> None:
        if callable(callback) and callback not in self._height_callbacks:
            self._height_callbacks.append(callback)

    def _set_height(self, h: float) -> None:
        h = float(max(PARAM_ROW_H, h))
        if abs(h - self._height) < 0.5:
            return
        self.prepareGeometryChange()
        self._height = h
        self.update()
        for cb in list(self._height_callbacks):
            try:
                cb(h)
            except Exception:
                pass

    @staticmethod
    def _compute_height() -> int:
        return _AF_FRAME_PAD * 2 + _AF_HDR_H + len(ActiveFileTableRow._ROWS_SPEC) * _AF_ROW_H + 2

    # ---- asset resolve ----

    def _resolve_asset(self):
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            from registers import resolve_robot_sku
            from registers import robots as _robots_registry
        except Exception:
            return None
        raw = str(self._owner.params.get("asset_id", "") or "").strip()
        self._asset_id_last = raw
        if not raw:
            return None
        sku = raw if _robots_registry.get_robot(raw) is not None else (
            _robots_registry.resolve_id(raw)
        )
        if sku is None and "." in raw:
            brand, _, model = raw.partition(".")
            sku = resolve_robot_sku(brand, model)
        if sku is None:
            return None
        try:
            return get_robot_asset_service().resolve(sku)
        except Exception:
            return None

    @staticmethod
    def _kind_available(asset, kind: str) -> bool:
        """True iff ``asset`` has a usable resource for ``kind``.

        For USD, a cloud Nucleus URL (``usd_url``) counts as available
        even when the local ``usd_path`` is None — that's how IsaacLab
        ships its built-in USD library, and the user must be able to
        select USD as the active format on those assets.
        """
        if asset is None:
            return False
        if getattr(asset, f"{kind}_path", None) is not None:
            return True
        if kind == "usd" and getattr(asset, "usd_url", None):
            return True
        return False

    def _current_active(self) -> str:
        """Return the file kind that currently owns the ✔ marker.

        Delegates to module-level :func:`_resolve_active_format` so the
        Body Mapping table below this one always agrees on which format
        is live — see that function's docstring for the algorithm.
        Returns lowercase (``"mjcf"``/``"usd"``/``"urdf"``) for backward
        compat with the existing paint code that does ``file_kind ==
        active`` comparisons against lowercase row spec.
        """
        return _resolve_active_format(self._owner.params, self._asset).lower()

    def _rebuild(self) -> None:
        self._asset = self._resolve_asset()
        self._set_height(self._compute_height())
        self.update()

    def _schedule_rebuild(self) -> None:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

    # ---- painting ----

    def _paint_value(self, painter: QPainter, tier: int) -> None:  # type: ignore[override]
        # NOTE: 表格内容必须在所有 LOD 档位下都渲染 —— 用户明确要求"任何内容
        # 不允许因为缩放消失"。这里不做 tier 早返回（与 _InlineTableRow 同。
        rect_full = QRectF(
            SEP_X + 4, _AF_FRAME_PAD,
            self._width - (SEP_X + 4) - VALUE_PAD_RIGHT,
            self._height - _AF_FRAME_PAD * 2,
        )
        bg = QColor(Config.get_color("bg_1", "#1A1A1A"))
        border = QColor(Config.get_color("border_2", "#3d3d3d"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(rect_full, 4, 4)

        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        fm = QFontMetricsF(font)

        muted = QColor(Config.get_color("canvas_node_param_text", "#aaaaaa"))
        text_c = QColor(Config.get_color("main_t1", "#cccccc"))
        ok_c = QColor(Config.get_color("safe_zone", "#4CAF50"))
        skip_c = QColor(Config.get_color("border_2", "#777777"))
        cloud_c = QColor(Config.get_color("robot_asset_status_cloud", "#4A9EFF"))

        inner_x = rect_full.left() + 4
        inner_w = max(0.0, rect_full.width() - 8)
        # Three columns: Type | Status | ✔
        col_chk_w = 22.0
        col_status_w = max(40.0, inner_w * 0.32)
        col_type_w = max(0.0, inner_w - col_chk_w - col_status_w - _AF_COL_GAP * 2)
        col_type_x = inner_x
        col_status_x = col_type_x + col_type_w + _AF_COL_GAP
        col_chk_x = col_status_x + col_status_w + _AF_COL_GAP

        # ---- header ----
        y = rect_full.top() + 1
        hdr_font = QFont(font)
        hdr_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(hdr_font)
        painter.setPen(QPen(muted))
        painter.drawText(
            QRectF(col_type_x + 2, y, col_type_w, _AF_HDR_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            tr("canvas.active_file.col_type", "Type"),
        )
        painter.drawText(
            QRectF(col_status_x, y, col_status_w, _AF_HDR_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            tr("canvas.active_file.col_status", "Status"),
        )
        painter.setFont(font)
        y += _AF_HDR_H

        active = self._current_active()
        zebra = QColor(Config.get_color("bg_1", "#1A1A1A")).lighter(108)
        self._row_hits = []

        for idx, (label, file_kind) in enumerate(self._ROWS_SPEC):
            path_value = (
                getattr(self._asset, f"{file_kind}_path", None)
                if self._asset is not None else None
            )
            cloud_only = (
                file_kind == "usd"
                and path_value is None
                and self._asset is not None
                and bool(getattr(self._asset, "usd_url", None))
            )
            loaded = path_value is not None
            available = loaded or cloud_only
            is_active = available and (file_kind == active)
            row_rect = QRectF(inner_x, y, inner_w, _AF_ROW_H)

            # zebra stripe + hover highlight
            if self._hover_rect is not None and self._hover_rect == row_rect:
                painter.setBrush(QBrush(QColor(Config.get_color("bg_1", "#2A2A2A")).lighter(120)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(row_rect)
            elif idx % 2 == 0:
                painter.setBrush(QBrush(zebra))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(row_rect)

            # type label
            painter.setPen(QPen(text_c if available else skip_c))
            label_rect = QRectF(col_type_x + 2, y, col_type_w, _AF_ROW_H)
            label_text = fm.elidedText(label, Qt.TextElideMode.ElideRight, label_rect.width())
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label_text,
            )

            # status: Loaded (local) | Cloud (URL only) | — (missing)
            if loaded:
                status_text, status_pen = tr("canvas.active_file.status_loaded", "Loaded"), ok_c
            elif cloud_only:
                status_text, status_pen = tr("canvas.active_file.status_cloud", "Cloud"), cloud_c
            else:
                status_text, status_pen = tr("canvas.row.empty", "—"), skip_c
            painter.setPen(QPen(status_pen))
            painter.drawText(
                QRectF(col_status_x, y, col_status_w, _AF_ROW_H),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                status_text,
            )

            # ✔ marker
            if is_active:
                painter.setPen(QPen(cloud_c if cloud_only else ok_c))
                check_font = QFont(font)
                check_font.setPixelSize(int(Config.get_font_size("size_small", 12)) + 2)
                check_font.setBold(True)
                painter.setFont(check_font)
                painter.drawText(
                    QRectF(col_chk_x, y, col_chk_w, _AF_ROW_H),
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                    "✔",
                )
                painter.setFont(font)

            # hit-test record (cloud-only rows are clickable too)
            self._row_hits.append((file_kind, QRectF(row_rect), available, is_active))
            y += _AF_ROW_H

    # ---- interaction ----

    def _interactive_regions(self) -> List[QRectF]:
        return [r for (_kind, r, loaded, _act) in self._row_hits if loaded]

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        pos = event.pos()
        for kind, rect, loaded, is_active in self._row_hits:
            if not loaded:
                continue
            if not rect.contains(pos):
                continue
            self.set_value("auto" if is_active else kind)
            self._schedule_rebuild()
            return True
        return False

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        return self._handle_click(event)

    # ---- asset_id sibling poll ----

    def maybe_refresh_for_asset_change(self) -> None:
        cur = str(self._owner.params.get("asset_id", "") or "").strip()
        if cur != self._asset_id_last:
            self._schedule_rebuild()


# =============================================================================
# Inline registry / training-item TABLES — DEMO `_RegistryModuleEditor` /
# `_TrainingItemEditor` faithful migration
# =============================================================================
#
# DEMO 在节点体内直接渲染多行表格（顶部 selector + ⚙，下方加边框的行表格，
# 每行 [badge | name | NodeSlider]）。RELEASE 之前是单行 summary "N items" +
# 弹 popup —— 视觉与交互完全偏离 DEMO。本节把两个表格做 inline 自绘 ParamRow，
# 与 ``BodyMappingTableRow`` 同款变高 + 自绘 + 命中测试，**绝不**用
# QGraphicsProxyWidget（项目硬性禁止）。
#
# 几何（与 DEMO 像素对齐 ± SDK 主题约束）：
#   header 行：24px，含 selector chrome（"N items ▾"）+ 22px ⚙ 按钮
#   frame  ：bordered ``canvas_widget_input_bg`` / ``canvas_widget_border``
#   per row：32px，[18px badge | 96px name (elide) | flex slider | 44px value]
# =============================================================================

_INL_HDR_H = 24
_INL_ROW_H = 32
_INL_FRAME_PAD = 4
_INL_BADGE_W = 18
_INL_NAME_W = 110
_INL_VAL_W = 56
_INL_WEIGHT_W = 46     # TrainingItemsInlineRow "Weight" column (far right of Max Spd)
_INL_GAP = 4
_INL_KEY_GAP = 8       # gap between header "key:" text and the selector picker
_INL_HDR_TOP_PAD = 2   # top breathing space above header
_INL_HDR_BOT_PAD = 4   # gap between header and the table frame
_INL_COLHDR_H = 15     # column-header strip height inside the frame (Name/Value/Grace labels)
_INL_LANE_HDR_H = 20   # phase-lane header height (when phase_aware=true)
_INL_LANE_EMPTY_H = 26 # phase-lane "uncovered" placeholder height (warning state)
_INL_LANE_COVERED_H = 14  # phase-lane "covered by unconditional" hint height (informational)
_INL_LANE_BAR_W = 4    # left color bar inside a phase-lane header
_INL_CHIP_H = 14       # phase-id chip height drawn on multi-phase reward rows
_INL_CHIP_PAD_X = 6    # chip horizontal padding around its text
_INL_CHIP_GAP = 3      # gap between adjacent chips
_INL_PHASE_BTN_W = 30  # phase-summary chip button width on each reward row (phase-aware mode)
_INL_PHASE_BTN_GAP = 4 # gap between phase-summary chip button and slider/value
_INL_GRACE_BTN_W = 48  # grace-period chip width on termination rows (fits "0.25s" + unit)
_INL_VALUE_BTN_W = 48  # value-param chip width on reward rows (fits "0.45m" + unit)
_PHASE_UNCONDITIONAL = "__unconditional__"  # synthetic lane id for applies_to == []


def _kind_from_registry_id(registry_id: str) -> str:
    """Map ``meta.registry_id`` → scripts.lookup ``kind`` argument."""
    rid = (registry_id or "").lower()
    if "termination" in rid:
        return "termination"
    if "obs" in rid:
        return "observation"
    if "discriminator" in rid:
        return "discriminator"
    return "reward"


# ---------------------------------------------------------------------------
# Registry-row slider curve
# ---------------------------------------------------------------------------
#
# Almost every registry-row slider has an extreme dynamic range relative to
# the typical value the user actually sets. For instance:
#   * IL terminations: ``illegal_contact`` ∈ [0.1, 200]  default 1.0
#   * IL terminations: ``time_out``        ∈ [1, 300]    default 20.0
#   * IL rewards:      ``joint_accel_penalty`` ∈ [-1e-3, 0] default -2.5e-7
#   * IL rewards:      ``joint_torque_penalty`` ∈ [-0.01, 0] default -2e-4
#
# On a linear slider, those defaults render at <1% / ~99% of the track — the
# user has no usable resolution near the working point. The audit confirmed
# every reward / termination / observation registry item is single-signed
# (vmin >= 0 OR vmax <= 0) except SB3's ``goal_distance`` [-5, +10]; that
# single span-zero entry stays linear because 0 is a meaningful midpoint
# there.
#
# Curve contract: ``curve(0.5) == 0.2`` — the first half of the slider
# track covers 20 % of the value range, the second half covers the
# remaining 80 %. Implemented as a pure power law ``v_frac = pos^k`` with
# ``k = log(0.2)/log(0.5) ≈ 2.32``. POS ranges are dense at the ``vmin``
# end (where small thresholds / small positive weights live); NEG ranges
# mirror so they stay dense at ``vmax == 0`` (where small-magnitude
# penalty weights live). The displayed numeric value is always the true
# value (not the position) — only the handle's *horizontal placement* is
# warped, so users still see "0.0002" rather than "78 %".
_SLIDER_CURVE_K = math.log(0.2) / math.log(0.5)


def _slider_curve_mode(vmin: float, vmax: float) -> str:
    """Pick curve direction from the sign of the [vmin, vmax] interval."""
    if vmax <= vmin:
        return "linear"
    if vmin >= 0.0:
        return "pos"
    if vmax <= 0.0:
        return "neg"
    return "linear"


def _slider_value_to_pos(value: float, vmin: float, vmax: float) -> float:
    """Map a value in [vmin, vmax] → normalized slider position in [0, 1]."""
    mode = _slider_curve_mode(vmin, vmax)
    if mode == "linear":
        if vmax <= vmin:
            return 0.0
        return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if mode == "pos":
        frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        return frac ** (1.0 / _SLIDER_CURVE_K)
    # "neg" — dense at vmax (== 0 for our penalty terms).
    frac = max(0.0, min(1.0, (vmax - value) / (vmax - vmin)))
    return 1.0 - frac ** (1.0 / _SLIDER_CURVE_K)


def _slider_pos_to_value(pos: float, vmin: float, vmax: float) -> float:
    """Inverse of :func:`_slider_value_to_pos`."""
    pos = max(0.0, min(1.0, pos))
    mode = _slider_curve_mode(vmin, vmax)
    if mode == "linear":
        return vmin + pos * (vmax - vmin)
    if mode == "pos":
        return vmin + (vmax - vmin) * (pos ** _SLIDER_CURVE_K)
    return vmax - (vmax - vmin) * ((1.0 - pos) ** _SLIDER_CURVE_K)


class _InlineTableRow(ParamRow):
    """变高内联表格基类 / Variable-height inline-table base.

    布局（full-width，跨 key + value 两列；不渲染 base ParamRow 的 key 列 +
    分隔线）：

        ┌──────────────────────────────────────────────────────────────┐
        │  reward_terms:  [N items ▾]                            [⚙]   │  ← header (24px)
        ├──────────────────────────────────────────────────────────────┤
        │  ┌────────────────────────────────────────────────────────┐  │
        │  │ ◉  Velocity Tracking         ━━━●━━━━━━━━━━━━━  1.000  │  │
        │  │ ◉  Alive Bonus               ━●━━━━━━━━━━━━━━  0.500  │  │
        │  │ ▣  Energy Penalty            ━━━━━━━━━●━━━━━━ -0.0001 │  │
        │  └────────────────────────────────────────────────────────┘  │  ← frame (full width)
        └──────────────────────────────────────────────────────────────┘

    Header 第 1 行宽度 = ``KEY_PAD_LEFT .. width-VALUE_PAD_RIGHT``（即原 key 半 + value 半合并）。
    子类实现：
        - ``_refresh_payloads()`` 从 ``self._value`` 重建 row payload list
        - ``_paint_row(painter, idx, payload)`` 行内自绘（badge + name + slider + value）
        - ``_row_min/_max/_step/_value/_set_value`` 单行数值契约
        - ``_header_summary_text()`` selector 按钮上的文本（"3 items" 等）
        - ``_on_selector_clicked(event)`` 顶部 picker 点击 → multi-choice popup
        - ``_on_modal_clicked(event)`` ⚙ 点击 → 模态 Editor
    """

    _OPEN_ON_RELEASE = False  # slider drag 必须 press 触发

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._row_payloads: list = []
        self._dragging_row: Optional[int] = None
        self._height_callbacks: list = []
        self._height = float(self._compute_height())

    # ---- 高度契约 ----

    def preferred_height(self) -> float:
        return float(self._height)

    def on_height_changed(self, callback) -> None:
        if callable(callback) and callback not in self._height_callbacks:
            self._height_callbacks.append(callback)

    def _set_height(self, h: float) -> None:
        h = float(max(PARAM_ROW_H, h))
        if abs(h - self._height) < 0.5:
            return
        self.prepareGeometryChange()
        self._height = h
        self.update()
        for cb in list(self._height_callbacks):
            try:
                cb(h)
            except Exception:
                pass

    def _compute_height(self) -> int:
        n = len(self._row_payloads)
        body = (n if n > 0 else 1) * _INL_ROW_H + _INL_FRAME_PAD * 2
        if n > 0:
            body += _INL_COLHDR_H  # column-header strip inside the frame
        return _INL_HDR_TOP_PAD + _INL_HDR_H + _INL_HDR_BOT_PAD + body + 2

    # ---- 几何（full-width，跨 key + value 两列）----

    def _full_rect(self) -> QRectF:
        """整行可用矩形 — 从节点 ``KEY_PAD_LEFT`` 到 ``width - VALUE_PAD_RIGHT``，
        跨原 key 列 + value 列。
        """
        return QRectF(
            KEY_PAD_LEFT, 0,
            max(0.0, self._width - KEY_PAD_LEFT - VALUE_PAD_RIGHT),
            self._height,
        )

    def _key_text(self) -> str:
        """Header 左边的 "title text"：直接用 ``spec.key``（与 base ParamRow 一致）。"""
        return self.spec.key

    def _key_font(self) -> QFont:
        f = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        f.setPixelSize(int(Config.get_font_size("size_small", 12)))
        return f

    def _key_text_width(self) -> float:
        fm = QFontMetricsF(self._key_font())
        return fm.horizontalAdvance(self._key_text() + ":")

    def _hdr_band_rect(self) -> QRectF:
        full = self._full_rect()
        return QRectF(
            full.left(),
            full.top() + _INL_HDR_TOP_PAD,
            full.width(),
            float(_INL_HDR_H),
        )

    def _hdr_key_rect(self) -> QRectF:
        band = self._hdr_band_rect()
        return QRectF(band.left(), band.top(), self._key_text_width(), band.height())

    def _hdr_picker_rect(self) -> QRectF:
        band = self._hdr_band_rect()
        x_start = band.left() + self._key_text_width() + _INL_KEY_GAP
        x_end = band.right() - _GEAR_BTN_W - _INL_GAP
        return QRectF(
            x_start, band.top() + 2,
            max(40.0, x_end - x_start), band.height() - 4,
        )

    def _hdr_gear_rect(self) -> QRectF:
        band = self._hdr_band_rect()
        return QRectF(
            band.right() - _GEAR_BTN_W, band.top() + 2,
            float(_GEAR_BTN_W), band.height() - 4,
        )

    def _frame_rect(self) -> QRectF:
        full = self._full_rect()
        top = full.top() + _INL_HDR_TOP_PAD + _INL_HDR_H + _INL_HDR_BOT_PAD
        return QRectF(
            full.left(), top,
            full.width(), self._height - (top - full.top()) - 2,
        )

    def _row_rect(self, idx: int) -> QRectF:
        f = self._frame_rect()
        y = f.top() + _INL_FRAME_PAD + _INL_COLHDR_H + idx * _INL_ROW_H
        return QRectF(
            f.left() + _INL_FRAME_PAD, y,
            f.width() - _INL_FRAME_PAD * 2, _INL_ROW_H,
        )

    def _col_header_rect(self) -> QRectF:
        """The column-header strip at the top of the frame (Name/Value/…)."""
        f = self._frame_rect()
        return QRectF(
            f.left() + _INL_FRAME_PAD, f.top() + _INL_FRAME_PAD,
            f.width() - _INL_FRAME_PAD * 2, float(_INL_COLHDR_H),
        )

    def _row_badge_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        bw = min(float(_INL_BADGE_W), r.height() - 8)
        return QRectF(r.left(), r.top() + (r.height() - bw) * 0.5, bw, bw)

    def _row_right_reserved(self, idx: int) -> float:
        """Width reserved at the row's right edge for optional chips drawn to
        the RIGHT of the value button (e.g. the termination grace chip).
        Default 0 — the value button sits flush at the right. Subclasses
        override to shift the value column left and make room."""
        return 0.0

    def _row_name_right(self, idx: int) -> float:
        """X coordinate where the (flexible) name column must stop. Default:
        just left of the value button. Subclasses that draw a chip to the
        LEFT of the value (e.g. the reward phase chip) override to stop the
        name before that chip."""
        return self._row_value_rect(idx).left() - _INL_GAP

    def _row_name_rect(self, idx: int) -> QRectF:
        # v2 (slider removed): columns are FIXED-width with priority
        # value > grace > name. The value/grace columns reserve their fixed
        # widths first; the name column takes exactly the remainder (no
        # min/max clamp) and truncates with an ellipsis (full text on hover).
        r = self._row_rect(idx)
        x0 = r.left() + _INL_BADGE_W + _INL_GAP
        x1 = self._row_name_right(idx)
        return QRectF(x0, r.top(), x1 - x0, r.height())

    def _row_slider_rect(self, idx: int) -> QRectF:
        # v2: per-row sliders were removed (node width is tight; the value
        # button is the sole weight editor). Empty rect → no draw, no drag
        # hit-test in mousePressEvent. Kept as a method so callers that still
        # reference it degrade to a no-op.
        return QRectF()

    def _row_value_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        x_right = r.right() - self._row_right_reserved(idx)
        return QRectF(x_right - _INL_VAL_W, r.top(), float(_INL_VAL_W), r.height())

    # ---- 命中测试 ----

    def _interactive_regions(self) -> List[QRectF]:
        regions = [self._hdr_picker_rect(), self._hdr_gear_rect()]
        for i in range(len(self._row_payloads)):
            regions.append(self._row_slider_rect(i))
            regions.append(self._row_value_rect(i))
        return regions

    # ---- paint ----

    def paint(self, painter, option, widget=None):  # type: ignore[override]
        """Full-width override — 不渲染 base ParamRow 的 key 列文字 + 中部分隔线。

        ``_InlineTableRow`` 自己负责整行的 header（含 inline key text + picker +
        ⚙）以及下方的 frame/rows，跨原 key + value 两列。
        """
        tier = tier_for_painter(painter)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Hover 高亮（保留 base 行为，但不画 key/separator）
        if self._hover_rect is not None:
            hover_bg = QColor(Config.get_color("bg_1", "#2A2A2A")).lighter(115)
            painter.setBrush(QBrush(hover_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self._hover_rect)
        # 全部交给 _paint_value（header + frame + rows）
        self._paint_value(painter, tier)

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        # NOTE: 表格内容必须在所有 LOD 档位下都渲染 —— 用户明确要求"任何内容
        # 不允许因为缩放消失"。这里不做 tier 早返回。
        self._paint_header(painter)
        if not self._row_payloads:
            self._paint_frame(painter)
            self._paint_empty_text(painter)
            return
        self._paint_frame(painter)
        self._paint_col_header(painter)
        for idx, payload in enumerate(self._row_payloads):
            try:
                self._paint_row(painter, idx, payload)
            except Exception:
                pass
        self._paint_row_separators(painter)

    def _paint_row_separators(self, painter: QPainter) -> None:
        """每两行 item 之间画 1px 水平分隔线 / Horizontal dividers between rows."""
        n = len(self._row_payloads)
        if n < 2:
            return
        sep = QColor(Config.get_color("border_1", "#444444"))
        sep.setAlpha(140)
        painter.setPen(QPen(sep, 0.6))
        f = self._frame_rect()
        x_l = f.left() + _INL_FRAME_PAD + 4
        x_r = f.right() - _INL_FRAME_PAD - 4
        for i in range(1, n):
            r = self._row_rect(i)
            y = r.top()
            painter.drawLine(QPointF(x_l, y), QPointF(x_r, y))

    def _paint_header(self, painter: QPainter) -> None:
        # Inline key text (e.g. "reward_terms:") at far left of header band.
        key_rect = self._hdr_key_rect()
        text_color = QColor(Config.get_color("canvas_node_param_text", "#C8C8C8"))
        painter.setPen(QPen(text_color))
        painter.setFont(self._key_font())
        painter.drawText(
            key_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            self._key_text() + ":",
        )
        # Picker chrome
        btn = self._hdr_picker_rect()
        bg = QColor(Config.get_color("btn_1"))
        if self._is_hovering(btn):
            bg = bg.lighter(115)
        border = QColor(Config.get_color("border_1"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(btn, 3, 3)
        text_color = QColor(Config.get_color("main_t1"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        text_rect = btn.adjusted(8, 0, -18, 0)
        elided = fm.elidedText(self._header_summary_text(), Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        # ▾
        painter.setBrush(QBrush(text_color))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = btn.right() - 10.0
        cy = btn.top() + btn.height() * 0.5
        painter.drawPolygon(QPolygonF([
            QPointF(cx, cy + 3.0),
            QPointF(cx - 4.5, cy - 2.5),
            QPointF(cx + 4.5, cy - 2.5),
        ]))
        # ⚙
        gear = self._hdr_gear_rect()
        gbg = QColor(Config.get_color("btn_1"))
        if self._is_hovering(gear):
            gbg = gbg.lighter(115)
        painter.setBrush(QBrush(gbg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(gear, 3, 3)
        painter.setPen(QPen(text_color))
        gear_font = QFont(font)
        gear_font.setBold(True)
        painter.setFont(gear_font)
        painter.drawText(gear, int(Qt.AlignmentFlag.AlignCenter), "⚙")

    def _paint_frame(self, painter: QPainter) -> None:
        f = self._frame_rect()
        bg = QColor(Config.get_color("bg_1", "#1A1A1A"))
        border = QColor(Config.get_color("border_2", "#3d3d3d"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(f, 4, 4)

    def _paint_empty_text(self, painter: QPainter) -> None:
        f = self._frame_rect()
        muted = QColor(Config.get_color("main_c2", "#888888"))
        painter.setPen(QPen(muted))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        painter.drawText(f, int(Qt.AlignmentFlag.AlignCenter), self._empty_text())

    def _empty_text(self) -> str:
        return tr("canvas.row.no_items", "(no items)")

    # ---- 列表头 / column header ----

    def _col_name_label(self) -> str:
        return tr("canvas.inline.col_name", "Name")

    def _col_value_label(self) -> str:
        return tr("canvas.inline.col_value", "Value")

    def _col_grace_label(self) -> str:
        return tr("canvas.inline.col_grace", "Grace")

    def _has_grace_col(self) -> bool:
        """Subclass override — True when a grace column is drawn."""
        return False

    def _paint_col_header(self, painter: QPainter) -> None:
        """Column titles aligned to the value / name / grace columns.

        Labels are right/left aligned to match their column and elide when
        the (fixed) column is too narrow — same truncation contract as the
        cells. A thin rule separates the header from the rows.
        """
        if not self._row_payloads:
            return
        hdr = self._col_header_rect()
        # Column x-positions are identical to row 0 (only y differs).
        name_r = self._row_name_rect(0)
        val_r = self._row_value_rect(0)
        color = QColor(Config.get_color("main_c2", "#888888"))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color))
        fm = QFontMetricsF(font)
        # Name (left-aligned over the name column).
        nx = QRectF(name_r.left(), hdr.top(), name_r.width(), hdr.height())
        painter.drawText(
            nx, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            fm.elidedText(self._col_name_label(), Qt.TextElideMode.ElideRight, max(0.0, nx.width())),
        )
        # Value (right-aligned over the value column).
        vx = QRectF(val_r.left(), hdr.top(), val_r.width(), hdr.height())
        painter.drawText(
            vx, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
            fm.elidedText(self._col_value_label(), Qt.TextElideMode.ElideRight, max(0.0, vx.width())),
        )
        # Grace (centered over the grace column), when present.
        if self._has_grace_col():
            try:
                g = self._row_grace_btn_rect(0)
                gx = QRectF(g.left(), hdr.top(), g.width(), hdr.height())
                painter.drawText(
                    gx, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                    fm.elidedText(self._col_grace_label(), Qt.TextElideMode.ElideRight, max(0.0, gx.width())),
                )
            except Exception:
                pass
        # Separator rule under the header.
        sep = QColor(Config.get_color("border_1", "#444444"))
        sep.setAlpha(160)
        painter.setPen(QPen(sep, 0.8))
        painter.drawLine(QPointF(hdr.left() + 2, hdr.bottom()), QPointF(hdr.right() - 2, hdr.bottom()))

    # ---- hover tooltip (full text for truncated cells) ----

    def _row_tooltip(self, idx: int) -> str:
        """Subclass override — full (un-truncated) text for the hovered row."""
        return ""

    def hoverMoveEvent(self, event):  # type: ignore[override]
        self._update_inline_tooltip(event.pos())
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):  # type: ignore[override]
        self.setToolTip("")
        super().hoverLeaveEvent(event)

    def _update_inline_tooltip(self, pos: QPointF) -> None:
        for idx in range(len(self._row_payloads)):
            if self._row_rect(idx).contains(pos):
                self.setToolTip(self._row_tooltip(idx))
                return
        self.setToolTip("")

    # ---- 子类钩子 ----

    def _header_summary_text(self) -> str:
        return f"{len(self._row_payloads)} items"

    def _paint_row(self, painter: QPainter, idx: int, payload: dict) -> None:
        pass

    def _row_value(self, idx: int, payload: dict) -> float:
        return 0.0

    def _row_min(self, idx: int, payload: dict) -> float:
        return 0.0

    def _row_max(self, idx: int, payload: dict) -> float:
        return 1.0

    def _row_set_value(self, idx: int, payload: dict, v: float) -> None:
        pass

    # ---- per-channel (array) value hooks ----
    # A row whose value is a vector (e.g. an obs term scaled per channel —
    # legged_gym ``commands_scale = [2, 2, 0.25]``) overrides these so clicking
    # the value cell opens N labeled number fields instead of one. Default None
    # ⇒ scalar value (current behavior, every non-array row).
    def _row_array_labels(self, idx: int) -> Optional[Sequence[str]]:
        return None

    def _row_array_values(self, idx: int, payload: dict, n: int) -> List[float]:
        return [float(self._row_value(idx, payload))] * n

    def _row_set_array_value(self, idx: int, payload: dict, values: Sequence[float]) -> None:
        pass

    def _on_selector_clicked(self, event: QGraphicsSceneMouseEvent) -> bool:
        return False

    def _on_modal_clicked(self, event: QGraphicsSceneMouseEvent) -> bool:
        return False

    # ---- click + drag ----

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        pos = event.pos()
        if self._hdr_gear_rect().contains(pos):
            return self._on_modal_clicked(event)
        if self._hdr_picker_rect().contains(pos):
            return self._on_selector_clicked(event)
        for idx in range(len(self._row_payloads)):
            if self._row_value_rect(idx).contains(pos):
                self._open_value_popup(idx, event)
                return True
        return False

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        return self._handle_click(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            for idx in range(len(self._row_payloads)):
                if self._row_slider_rect(idx).contains(event.pos()):
                    self._dragging_row = idx
                    self._commit_drag(idx, event.pos().x())
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_row is not None:
            self._commit_drag(self._dragging_row, event.pos().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._dragging_row is not None:
            self._dragging_row = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _commit_drag(self, idx: int, x: float) -> None:
        payload = self._row_payloads[idx]
        ht = RangeSliderHitTest(
            self._row_slider_rect(idx),
            vmin=self._row_min(idx, payload),
            vmax=self._row_max(idx, payload),
            dual=False,
        )
        v = ht.value_at(x)
        self._row_set_value(idx, payload, v)
        self.update()

    def _open_value_popup(self, idx: int, event: QGraphicsSceneMouseEvent) -> None:
        payload = self._row_payloads[idx]
        # 锚定到单条 row 的 value cell —— 让 popup 高度 = 行高，而不是整张表的高度。
        sub = self._row_value_rect(idx)
        labels = self._row_array_labels(idx)
        if labels:
            # Per-channel vector value (e.g. obs velocity_command scale) → N
            # labeled number fields instead of a single DataInput.
            n = len(labels)
            open_number_array_popup(
                view=_view_of(event), row=self,
                values=self._row_array_values(idx, payload, n), labels=labels,
                minimum=self._row_min(idx, payload),
                maximum=self._row_max(idx, payload),
                on_commit=lambda vals, _i=idx, _p=payload: (
                    self._row_set_array_value(_i, _p, vals), self.update(),
                ),
                sub_rect=sub,
            )
            return
        cur = self._row_value(idx, payload)
        open_number_popup(
            view=_view_of(event), row=self,
            value=float(cur), is_int=False,
            minimum=self._row_min(idx, payload),
            maximum=self._row_max(idx, payload),
            on_commit=lambda v, _i=idx, _p=payload: (
                self._row_set_value(_i, _p, float(v)), self.update(),
            ),
            sub_rect=sub,
        )

    # ---- 共享 painters：value 按钮（与 RangeRow._paint_end_button 同款） ----

    _VAL_BTN_PAD_X = 2  # 与 RangeRow._BTN_PAD_X 同款 fit-content padding

    def _val_btn_rect(self, idx: int, text: str) -> QRectF:
        """fit-content 风格 value 按钮：在 ``_row_value_rect`` 内右对齐 + 垂直居中.

        宽度 = ``advance(text) + 2 * pad_x``；高度 = ``fm.height()``；与
        ``RangeRow._paint_end_button`` 几何契约一致。值串过长（``-0.0001`` 等）
        会往左侵占一点 slider gap —— 视觉上仍贴在 value 列右端，可接受。
        """
        cell = self._row_value_rect(idx)
        font = self._val_btn_font()
        fm = QFontMetricsF(font)
        from math import ceil
        w = float(ceil(fm.horizontalAdvance(text)) + 2 * self._VAL_BTN_PAD_X)
        h = float(ceil(fm.height()))
        x = cell.right() - w - 2
        y = cell.top() + (cell.height() - h) * 0.5
        return QRectF(x, y, w, h)

    def _val_btn_font(self) -> QFont:
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        return font

    def _paint_value_button(
        self, painter: QPainter, idx: int, text: str, *, accent: bool = True,
    ) -> QRectF:
        """画 [value] 按钮 —— 与 ``RangeRow._paint_end_button`` 同款样式."""
        rect = self._val_btn_rect(idx, text)
        bg_slot = "btn_1" if accent else "btn_1"
        bg = QColor(Config.get_color(bg_slot, "#343636"))
        if self._is_hovering(self._row_value_rect(idx)):
            bg = bg.lighter(110)
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(rect, 3, 3)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        painter.setFont(self._val_btn_font())
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        return rect


class RegistryModuleInlineRow(_InlineTableRow):
    """``widget="registry_module"`` — DEMO ``_RegistryModuleEditor`` 内联表格.

    用于 reward_terms / termination_conditions / obs_terms / disc_modules。
    每行 ``[polarity badge | TaskModuleItem.title | NodeSlider | weight]``，
    元数据 (title/min/max/step/polarity) 来自 ``scripts.lookup(key, kind, backend)``。

    Phase-aware mode (``meta.phase_aware = true`` + ``[Canvas]
    phase_lanes_enabled = true``):
        Rows are grouped into Motion-Phase lanes per
        :mod:`registers.motion_phases`. Each reward entry may carry an
        optional ``applies_to: [phase_id, ...]`` field; terms with empty
        or missing ``applies_to`` land in the synthetic ``Unconditional``
        lane (applied to every phase at runtime). Empty phase lanes are
        rendered with an "under-bound" placeholder so the user spots
        missing coverage at editing time — this is the canvas-side half
        of the Reward × MotionPhase isolation feature targeting the Go2
        idle limb-flailing bug.
    """

    def __init__(self, spec, owner_node, width, parent=None):
        # Lane state built by ``_rebuild_lane_segments``; consumed by
        # ``_row_rect`` / ``_compute_height`` / ``_paint_value``.
        # Each segment: dict(
        #     id=<phase_id_or_unconditional>, display=<str>, color_slot=<str>,
        #     start=<int>, end=<int>,             # payload index range [start, end)
        #     rows_y=<float>, header_y=<float>,   # paint-time geometry
        #     placeholder=<bool>,                 # True ⇒ empty lane placeholder
        # )
        self._lane_segments: list = []
        # Cached ordered list of (phase_id_or_unconditional, MotionPhase|None)
        # rebuilt on every ``_refresh_payloads``. The synthetic
        # ``__unconditional__`` entry is always last and only present when
        # at least one term has empty applies_to.
        self._phase_order: list = []
        super().__init__(spec, owner_node, width, parent)
        self._refresh_payloads()

    def set_value(self, v):  # type: ignore[override]
        # 缺口③ — page-aware: when meta.paged, the widget edits ONE page
        # (reward_active_page) of a paged reward_terms ``{page: {func: payload}}``.
        # Internal editors produce the flat page terms ``v``; wrap them back into
        # the stored paged dict so the persisted param stays the single source of
        # truth (the compiler reads it page-by-page). Non-paged → unchanged.
        if self._is_paged():
            from application.compiler.term_payload import (
                migrate_reward_terms_to_paged,
            )
            page_terms = _parse_dict_value(v) or {}
            paged = migrate_reward_terms_to_paged(_parse_dict_value(self._value) or {})
            paged[self._active_page()] = page_terms
            super().set_value(_serialize_for_spec(self.spec, paged))
        else:
            super().set_value(v)
        self._refresh_payloads()

    # ── 缺口③ paging helpers ──────────────────────────────────────────
    def _is_paged(self) -> bool:
        return bool((self.spec.meta or {}).get("paged", False))

    def _active_page(self) -> str:
        key = str((self.spec.meta or {}).get("active_page_key", "") or "")
        if not key:
            return "__global__"
        return str(self._owner.params.get(key, "__global__") or "__global__")

    def _current_terms(self) -> dict:
        """The flat term dict the widget displays/edits — the active page when
        paged, else the whole value (legacy / terminations / obs)."""
        if self._is_paged():
            from application.compiler.term_payload import (
                migrate_reward_terms_to_paged,
            )
            paged = migrate_reward_terms_to_paged(_parse_dict_value(self._value) or {})
            return dict(paged.get(self._active_page(), {}))
        return _parse_dict_value(self._value) or {}

    def _kind(self) -> str:
        meta = self.spec.meta or {}
        return _kind_from_registry_id(str(meta.get("registry_id", "")))

    def _backend(self) -> str:
        return str(self._owner.params.get("backend", "sb3_mujoco") or "sb3_mujoco").strip()

    def _lookup_item(self, key: str):
        try:
            from scripts import lookup as _lookup
            return _lookup(key, kind=self._kind(), backend=self._backend())
        except Exception:
            return None

    def _refresh_payloads(self) -> None:
        d = self._current_terms()
        payloads = []
        kind = self._kind()
        applies_to_key = str((self.spec.meta or {}).get("applies_to_key", "applies_to"))
        for key, val in d.items():
            data_type = "scalar"
            scale_value: Any = None
            if kind == "observation":
                # obs term value = scale (scalar or vector), optionally wrapped
                # as {"scale", "data_type"}. ``scale_list`` drives the value
                # cell's vector display/editor; ``weight`` keeps a scalar fallback
                # for any shared math. ``data_type`` (scalar/list/both) governs
                # whether the row's ⬤ checkbox can toggle the shape.
                scale_value, data_type = parse_obs_term_value(val)
                scale_list = scale_value if isinstance(scale_value, list) else None
                if isinstance(scale_value, list):
                    weight = float(scale_value[0]) if scale_value else 0.0
                else:
                    try:
                        weight = float(scale_value if scale_value is not None else 0.0)
                    except (TypeError, ValueError):
                        weight = 0.0
                applies = []
            else:
                if isinstance(val, dict):
                    weight = val.get("weight")
                    applies_raw = val.get(applies_to_key, []) or []
                else:
                    weight = val
                    applies_raw = []
                scale_list = None
                if isinstance(weight, (list, tuple)):
                    try:
                        scale_list = [float(x) for x in weight]
                    except (TypeError, ValueError):
                        scale_list = None
                    weight = float(scale_list[0]) if scale_list else 0.0
                else:
                    try:
                        weight = float(weight if weight is not None else 0.0)
                    except (TypeError, ValueError):
                        weight = 0.0
                if isinstance(applies_raw, str):
                    applies = [s.strip() for s in applies_raw.split(",") if s.strip()]
                elif isinstance(applies_raw, (list, tuple)):
                    applies = [str(s).strip() for s in applies_raw if str(s).strip()]
                else:
                    applies = []
            # Per-condition grace_period_s (termination grace window). 0 / absent
            # for the legacy scalar form and non-grace kinds.
            try:
                from application.compiler.term_payload import parse_grace_period
                grace_s = float(parse_grace_period(val))
            except Exception:
                grace_s = 0.0
            # Per-item "Value" (a reward's function-internal param, e.g.
            # base_height target_height). None = unset → reward default.
            try:
                from application.compiler.term_payload import parse_item_value
                item_value = parse_item_value(val)
            except Exception:
                item_value = None
            payloads.append({
                "key": key,
                "weight": weight,
                "scale_list": scale_list,
                "scale_value": scale_value,
                "data_type": data_type,
                "item": self._lookup_item(key),
                "meta": val if isinstance(val, dict) else {},
                "applies_to": applies,
                "grace_period_s": grace_s,
                "value": item_value,
            })
        if self._is_phase_aware():
            payloads = self._sort_payloads_by_phase(payloads)
        self._row_payloads = payloads
        if self._is_phase_aware():
            self._rebuild_lane_segments()
        else:
            self._lane_segments = []
            self._phase_order = []
        self._set_height(self._compute_height())

    # ------------------------------------------------------------------
    # Phase-aware helpers
    # ------------------------------------------------------------------

    def _is_phase_aware(self) -> bool:
        """Phase Lane visualization is enabled.

        Three gates must all pass: the param meta opts in via
        ``phase_aware = true``, the global ``[Canvas] phase_lanes_enabled``
        feature flag is on, and the registers.motion_phases registry was
        loadable. Any failure path falls back to the flat-tile layout so
        legacy canvases keep rendering.
        """
        meta = self.spec.meta or {}
        if not bool(meta.get("phase_aware", False)):
            return False
        try:
            from unitport_sdk import Config as _Cfg
            enabled = _Cfg.get_value("Canvas", "phase_lanes_enabled", True, value_type=bool)
        except Exception:
            enabled = True
        if not enabled:
            return False
        try:
            from registers import motion_phases as _mp
            return bool(_mp.list_ids())
        except Exception:
            return False

    def _phase_layout_for_node(self) -> list:
        """Resolve the phase entries this row should render with.

        Order of precedence:
            1. Per-canvas ``phase_layout`` parameter on the owner node
               (JSON list of phase dicts) — usually empty.
            2. Default factory phases from :mod:`registers.motion_phases`.

        Returns a list of dicts with at least ``id``, ``display_name``,
        ``theme_slot``, ``polarity_required``. Never raises — fall back
        to an empty list on any error so the caller can degrade.
        """
        owner_params = getattr(self._owner, "params", {}) or {}
        raw = owner_params.get("phase_layout", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else []
            except Exception:
                raw = []
        if isinstance(raw, list) and raw:
            out = []
            for entry in raw:
                if not isinstance(entry, dict) or "id" not in entry:
                    continue
                out.append({
                    "id": str(entry["id"]),
                    "display_name": str(entry.get("display_name") or entry["id"]),
                    "theme_slot": str(entry.get("theme_slot") or "main_c2"),
                    "polarity_required": str(entry.get("polarity_required") or "any"),
                })
            if out:
                return out
        try:
            from registers import motion_phases as _mp
            return [
                {
                    "id": p.id,
                    "display_name": p.display_name_default,
                    "theme_slot": p.theme_slot,
                    "polarity_required": p.polarity_required,
                }
                for p in _mp.list_phases()
            ]
        except Exception:
            return []

    def _sort_payloads_by_phase(self, payloads: list) -> list:
        """Stable-sort reward payloads so phase lanes group contiguously.

        Within a lane the original insertion order is preserved.
        Unconditional terms (empty applies_to) land last in the synthetic
        ``__unconditional__`` lane.
        """
        layout = self._phase_layout_for_node()
        order: dict = {entry["id"]: idx for idx, entry in enumerate(layout)}
        order[_PHASE_UNCONDITIONAL] = len(layout)

        def _bucket(payload: dict) -> int:
            applies = payload.get("applies_to") or []
            if not applies:
                return order[_PHASE_UNCONDITIONAL]
            # Use the term's first declared phase that exists in the
            # layout — that defines which lane it visually lives in.
            for pid in applies:
                if pid in order:
                    return order[pid]
            # No declared phase matches the layout → group with the
            # unconditional bucket so it still renders (the lane header
            # paints an "unmapped" warning chip for these entries).
            return order[_PHASE_UNCONDITIONAL]

        indexed = list(enumerate(payloads))
        indexed.sort(key=lambda pair: (_bucket(pair[1]), pair[0]))
        return [pair[1] for pair in indexed]

    def _rebuild_lane_segments(self) -> None:
        """Compute lane segments + assign paint-time y offsets.

        Called from ``_refresh_payloads`` after ``_sort_payloads_by_phase``
        leaves ``_row_payloads`` contiguously grouped. Empty lanes get a
        ``placeholder_kind`` field:

        * ``"uncovered"`` — no phase-specific reward AND no unconditional
          reward. Lane renders a warning-coloured dashed box: this is
          the under-constrained / Go2 limb-flailing pattern.
        * ``"covered"``   — no phase-specific reward, but at least one
          reward has empty applies_to (Unconditional). Lane renders a
          muted "Covered by Unconditional" pill — informational only,
          since the unconditional rewards apply to every phase at runtime.
        * ``""``           — lane has phase-specific members (normal rows).
        """
        layout = self._phase_layout_for_node()
        layout_ids = [entry["id"] for entry in layout]
        # First pass: bucket payloads by phase id (without geometry yet).
        buckets: Dict[str, List[int]] = {pid: [] for pid in layout_ids}
        buckets[_PHASE_UNCONDITIONAL] = []
        for i, payload in enumerate(self._row_payloads):
            applies = payload.get("applies_to") or []
            target = _PHASE_UNCONDITIONAL
            if applies:
                for pid in applies:
                    if pid in buckets:
                        target = pid
                        break
                else:
                    target = _PHASE_UNCONDITIONAL  # explicit fallback
            buckets[target].append(i)

        has_unconditional = bool(buckets.get(_PHASE_UNCONDITIONAL))

        segments: list = []
        cursor = 0
        for entry in layout:
            pid = entry["id"]
            members = buckets.get(pid, [])
            placeholder = len(members) == 0
            if not placeholder:
                placeholder_kind = ""
            elif has_unconditional:
                placeholder_kind = "covered"
            else:
                placeholder_kind = "uncovered"
            segments.append({
                "id": pid,
                "display": entry["display_name"],
                "color_slot": entry["theme_slot"],
                "polarity_required": entry["polarity_required"],
                "start": cursor,
                "end": cursor + len(members),
                "placeholder": placeholder,
                "placeholder_kind": placeholder_kind,
            })
            cursor += len(members)
        # Always emit the Unconditional lane when it has members so the user
        # can see what stays globally on; omit it when empty (no need to
        # advertise an empty bucket below the real phases).
        uncond_members = buckets.get(_PHASE_UNCONDITIONAL, [])
        if uncond_members:
            segments.append({
                "id": _PHASE_UNCONDITIONAL,
                "display": "Unconditional",
                "color_slot": "main_c2",
                "polarity_required": "any",
                "start": cursor,
                "end": cursor + len(uncond_members),
                "placeholder": False,
                "placeholder_kind": "",
            })
            cursor += len(uncond_members)

        # Second pass: paint-time y geometry. The frame's top-edge is at
        # ``_frame_rect().top() + _INL_FRAME_PAD``; for each segment, lay
        # down a lane header then either the row span or the placeholder
        # (uncovered = full warning box, covered = small italic hint),
        # with no extra padding between segments (the lane header itself
        # provides visual breathing room).
        y = 0.0
        for seg in segments:
            seg["header_y"] = y
            y += _INL_LANE_HDR_H
            seg["rows_y"] = y
            if seg["placeholder"]:
                kind = seg.get("placeholder_kind", "uncovered")
                y += _INL_LANE_COVERED_H if kind == "covered" else _INL_LANE_EMPTY_H
            else:
                y += (seg["end"] - seg["start"]) * _INL_ROW_H
        self._lane_segments = segments
        self._phase_order = [seg["id"] for seg in segments]

    def _header_summary_text(self) -> str:
        n = len(self._row_payloads)
        return f"{n} items" if n else "—"

    # ------------------------------------------------------------------
    # Phase-aware geometry + paint overrides
    # ------------------------------------------------------------------

    def _compute_height(self) -> int:
        if not self._is_phase_aware() or not self._lane_segments:
            return super()._compute_height()
        body = _INL_FRAME_PAD * 2
        for seg in self._lane_segments:
            body += _INL_LANE_HDR_H
            if seg["placeholder"]:
                kind = seg.get("placeholder_kind", "uncovered")
                body += _INL_LANE_COVERED_H if kind == "covered" else _INL_LANE_EMPTY_H
            else:
                body += (seg["end"] - seg["start"]) * _INL_ROW_H
        return _INL_HDR_TOP_PAD + _INL_HDR_H + _INL_HDR_BOT_PAD + body + 2

    def _row_rect(self, idx: int) -> QRectF:
        if not self._is_phase_aware() or not self._lane_segments:
            return super()._row_rect(idx)
        f = self._frame_rect()
        for seg in self._lane_segments:
            if seg["placeholder"]:
                continue
            if seg["start"] <= idx < seg["end"]:
                local = idx - seg["start"]
                y = f.top() + _INL_FRAME_PAD + seg["rows_y"] + local * _INL_ROW_H
                return QRectF(
                    f.left() + _INL_FRAME_PAD, y,
                    f.width() - _INL_FRAME_PAD * 2, _INL_ROW_H,
                )
        # idx is past the last segment (shouldn't happen) — degrade safely.
        return super()._row_rect(idx)

    def _lane_header_rect(self, seg: dict) -> QRectF:
        f = self._frame_rect()
        y = f.top() + _INL_FRAME_PAD + seg["header_y"]
        return QRectF(
            f.left() + _INL_FRAME_PAD, y,
            f.width() - _INL_FRAME_PAD * 2, float(_INL_LANE_HDR_H),
        )

    def _lane_placeholder_rect(self, seg: dict) -> QRectF:
        f = self._frame_rect()
        y = f.top() + _INL_FRAME_PAD + seg["rows_y"]
        kind = seg.get("placeholder_kind", "uncovered")
        if kind == "covered":
            return QRectF(
                f.left() + _INL_FRAME_PAD + 8 + _INL_LANE_BAR_W + 8,
                y,
                f.width() - _INL_FRAME_PAD * 2 - 16 - _INL_LANE_BAR_W - 8,
                float(_INL_LANE_COVERED_H),
            )
        return QRectF(
            f.left() + _INL_FRAME_PAD + 8, y + 3,
            f.width() - _INL_FRAME_PAD * 2 - 16, float(_INL_LANE_EMPTY_H) - 6,
        )

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        if not self._is_phase_aware() or not self._lane_segments:
            super()._paint_value(painter, tier)
            return
        self._paint_header(painter)
        self._paint_frame(painter)
        self._paint_phase_lanes(painter)
        for idx, payload in enumerate(self._row_payloads):
            try:
                self._paint_row(painter, idx, payload)
                self._paint_row_phase_button(painter, idx, payload)
            except Exception:
                pass
        self._paint_row_separators_phase(painter)

    # ------------------------------------------------------------------
    # Phase-aware geometry: shrink slider to make room for the phase
    # chip button (slider right edge = chip-button left edge − gap).
    # ------------------------------------------------------------------

    def _row_right_reserved(self, idx: int) -> float:
        # Reserve the RIGHTMOST chip column TABLE-WIDE so the weight column
        # aligns vertically across every row. Terminations reserve the grace
        # column; rewards reserve the value-param column. The two never
        # coexist on one node (reward vs termination kind).
        if self._supports_grace():
            return float(_INL_GRACE_BTN_W) + float(_INL_PHASE_BTN_GAP)
        if self._has_value_col():
            return float(_INL_VALUE_BTN_W) + float(_INL_PHASE_BTN_GAP)
        return 0.0

    def _row_name_right(self, idx: int) -> float:
        # Reward phase chip sits to the LEFT of the weight button → name stops
        # before it. Otherwise the name stops just left of the weight column
        # (the value-param / grace chips live to the RIGHT of weight, handled
        # via _row_right_reserved, so they don't affect the name's right edge).
        if self._is_phase_aware():
            return self._row_phase_btn_rect(idx).left() - _INL_GAP
        return super()._row_name_right(idx)

    # ------------------------------------------------------------------
    # Grace-period chip (termination rows on grace-supporting nodes).
    # Mirrors the phase-button affordance: an always-visible clickable
    # chip in the slider→value gap that opens a numeric popup for the
    # per-condition ``grace_period_s`` (seconds). time_out is excluded
    # (its grace is meaningless — it IS the episode-length cutoff).
    # ------------------------------------------------------------------

    def _supports_grace(self) -> bool:
        # Grace applies to terminations (a transient settle window before the
        # condition can fire). Gate on kind alone — robust to whether the
        # node manifest's ``supports_grace_period`` meta flag reached this
        # spec (the flag is informational; kind is the contract).
        return self._kind() == "termination"

    def _row_supports_grace(self, payload: dict) -> bool:
        return self._supports_grace() and str(payload.get("key", "")) != "time_out"

    def _row_grace_btn_rect(self, idx: int) -> QRectF:
        # Rightmost column: [name | value | grace].
        r = self._row_rect(idx)
        x_right = r.right()
        x_left = x_right - float(_INL_GRACE_BTN_W)
        y = r.top() + (r.height() - float(_INL_CHIP_H)) * 0.5
        return QRectF(x_left, y, float(_INL_GRACE_BTN_W), float(_INL_CHIP_H))

    def _paint_row_grace_button(self, painter: QPainter, idx: int, payload: dict) -> None:
        rect = self._row_grace_btn_rect(idx)
        grace = float(payload.get("grace_period_s", 0.0) or 0.0)
        active = grace > 0.0
        # Always show the unit (seconds) so the field is self-explanatory:
        # "0s" (unset) / "0.5s" (set). `:g` keeps it compact in the column.
        label = f"{grace:g}s"
        # Background is the ORIGINAL chip bg (btn_1) in EVERY state — the
        # "set" state is conveyed by HIGHLIGHTING THE TEXT (system.ini
        # ``highlight`` slot), never by recolouring the background (rule §5).
        bg = QColor(Config.get_color("btn_1", "#343636"))
        fg = QColor(Config.get_color("highlight", "#F6D393")) if active else \
            QColor(Config.get_color("main_c2", "#999999"))
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 0.8))
        painter.drawRoundedRect(rect, 3, 3)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _set_grace_period(self, idx: int, payload: dict, grace_s: float) -> None:
        """Write ``grace_period_s`` back into the conditions dict.

        Mirrors :meth:`_set_applies_to` — upgrades a legacy scalar to the
        structured ``{"weight": ...}`` shape, then sets/clears grace. Collapses
        back to a bare scalar when only ``weight`` remains, to keep canvas-JSON
        diffs minimal (matches ``serialize_term_payload``).
        """
        grace_s = max(0.0, float(grace_s))
        payload["grace_period_s"] = grace_s
        d = self._current_terms()
        cur = d.get(payload["key"])
        if isinstance(cur, dict):
            cur = dict(cur)
        else:
            cur = {"weight": float(payload.get("weight", 0.0) or 0.0)}
        if grace_s > 0.0:
            cur["grace_period_s"] = grace_s
        else:
            cur.pop("grace_period_s", None)
        if set(cur.keys()) == {"weight"}:
            d[payload["key"]] = cur["weight"]
        else:
            d[payload["key"]] = cur
        self.set_value(_serialize_for_spec(self.spec, d))

    def _open_grace_popup(self, idx: int, event: QGraphicsSceneMouseEvent) -> None:
        payload = self._row_payloads[idx]
        cur = float(payload.get("grace_period_s", 0.0) or 0.0)
        open_number_popup(
            view=_view_of(event), row=self,
            value=cur, is_int=False,
            minimum=0.0, maximum=10.0,
            on_commit=lambda v, _i=idx, _p=payload: (
                self._set_grace_period(_i, _p, float(v)), self.update(),
            ),
            sub_rect=self._row_grace_btn_rect(idx),
        )

    def _row_phase_btn_rect(self, idx: int) -> QRectF:
        """Phase-summary chip button on a reward row, immediately left of the
        weight button (the value-param column sits to the RIGHT of weight,
        reserved table-wide via _row_right_reserved). Single hit target for
        opening the phase multi-select popup.
        """
        r = self._row_rect(idx)
        val = self._row_value_rect(idx)
        x_right = val.left() - float(_INL_PHASE_BTN_GAP)
        x_left = x_right - float(_INL_PHASE_BTN_W)
        y = r.top() + (r.height() - float(_INL_CHIP_H)) * 0.5
        return QRectF(x_left, y, float(_INL_PHASE_BTN_W), float(_INL_CHIP_H))

    # ------------------------------------------------------------------
    # Value-param chip (reward rows whose registry item declares a
    # function-internal tunable param, e.g. base_height target_height).
    # Mirrors the grace chip: an always-visible clickable chip just left
    # of the weight button that opens a numeric popup for the per-item
    # "Value". std/threshold reward-shaping knobs are NOT this field.
    # ------------------------------------------------------------------

    def _supports_value(self) -> bool:
        # Gate on kind=="reward" AND any row carrying a value-bearing item.
        if self._kind() != "reward":
            return False
        return any(self._row_supports_value(p) for p in self._row_payloads)

    def _row_supports_value(self, payload: dict) -> bool:
        if self._kind() != "reward":
            return False
        item = payload.get("item")
        return bool(
            item is not None
            and str(getattr(item, "il_value_label", "") or "").strip()
        )

    def _has_value_col(self) -> bool:
        return self._supports_value()

    def _row_value_param_rect(self, idx: int) -> QRectF:
        # Reward value-param chip: RIGHTMOST column [name | weight | value].
        # Reserved table-wide (when the table has a Value column) via
        # _row_right_reserved so the weight column stays vertically aligned;
        # only value-bearing rows actually paint a chip here.
        r = self._row_rect(idx)
        x_right = r.right()
        x_left = x_right - float(_INL_VALUE_BTN_W)
        y = r.top() + (r.height() - float(_INL_CHIP_H)) * 0.5
        return QRectF(x_left, y, float(_INL_VALUE_BTN_W), float(_INL_CHIP_H))

    def _paint_row_value_param_button(
        self, painter: QPainter, idx: int, payload: dict
    ) -> None:
        rect = self._row_value_param_rect(idx)
        item = payload.get("item")
        unit = str(getattr(item, "il_value_unit", "") or "")
        default = float(getattr(item, "il_value_default", 0.0) or 0.0)
        raw = payload.get("value")
        cur = float(raw) if raw is not None else default
        # "Set" = an explicit value differing from the registry default.
        active = raw is not None and float(raw) != default
        label = f"{cur:g}{unit}"
        # Background = btn_1 in EVERY state; "set" is conveyed by HIGHLIGHTING
        # THE TEXT (system.ini ``highlight``), never recolouring bg (rule §5).
        bg = QColor(Config.get_color("btn_1", "#343636"))
        fg = QColor(Config.get_color("highlight", "#F6D393")) if active else \
            QColor(Config.get_color("main_c2", "#999999"))
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 0.8))
        painter.drawRoundedRect(rect, 3, 3)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        fm = QFontMetricsF(font)
        painter.drawText(
            rect, int(Qt.AlignmentFlag.AlignCenter),
            fm.elidedText(label, Qt.TextElideMode.ElideRight, max(0.0, rect.width() - 4)),
        )

    def _set_item_value(self, idx: int, payload: dict, value: float) -> None:
        """Write the per-item ``value`` back into the reward_terms dict.

        Mirrors :meth:`_set_grace_period`: upgrade a legacy scalar to the
        structured ``{"weight": ...}`` shape, store ``value`` only when it
        DIFFERS from the registry default (collapse back to a bare scalar
        otherwise) so canvas-JSON diffs stay minimal and ``value == default``
        round-trips to "auto" (matches ``serialize_term_payload``).
        """
        item = payload.get("item")
        default = float(getattr(item, "il_value_default", 0.0) or 0.0)
        value = float(value)
        d = self._current_terms()
        cur = d.get(payload["key"])
        if isinstance(cur, dict):
            cur = dict(cur)
        else:
            cur = {"weight": float(payload.get("weight", 0.0) or 0.0)}
        if value != default:
            cur["value"] = value
            payload["value"] = value
        else:
            cur.pop("value", None)
            payload["value"] = None
        if set(cur.keys()) == {"weight"}:
            d[payload["key"]] = cur["weight"]
        else:
            d[payload["key"]] = cur
        self.set_value(_serialize_for_spec(self.spec, d))

    def _open_item_value_popup(self, idx: int, event: QGraphicsSceneMouseEvent) -> None:
        # NOTE: deliberately NOT named ``_open_value_popup`` — that base-class
        # method opens the WEIGHT popup (called from _handle_click on the
        # weight cell). Shadowing it routed weight clicks into the value editor.
        payload = self._row_payloads[idx]
        item = payload.get("item")
        default = float(getattr(item, "il_value_default", 0.0) or 0.0)
        lo = float(getattr(item, "il_value_min", 0.0) or 0.0)
        hi = float(getattr(item, "il_value_max", 0.0) or 0.0)
        raw = payload.get("value")
        cur = float(raw) if raw is not None else default
        open_number_popup(
            view=_view_of(event), row=self,
            value=cur, is_int=False,
            minimum=lo, maximum=hi,
            on_commit=lambda v, _i=idx, _p=payload: (
                self._set_item_value(_i, _p, float(v)), self.update(),
            ),
            sub_rect=self._row_value_param_rect(idx),
        )

    def _col_param_label(self) -> str:
        return tr("canvas.inline.col_param", "Value")

    def _paint_col_header(self, painter: QPainter) -> None:  # type: ignore[override]
        super()._paint_col_header(painter)
        # Add the value-param column title (centered over the chip column).
        if not self._row_payloads or not self._has_value_col():
            return
        try:
            hdr = self._col_header_rect()
            pr = self._row_value_param_rect(0)
            color = QColor(Config.get_color("main_c2", "#888888"))
            font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
            font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(color))
            fm = QFontMetricsF(font)
            px = QRectF(pr.left(), hdr.top(), pr.width(), hdr.height())
            painter.drawText(
                px, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                fm.elidedText(self._col_param_label(), Qt.TextElideMode.ElideRight, max(0.0, px.width())),
            )
        except Exception:
            pass

    def _paint_phase_lanes(self, painter: QPainter) -> None:
        """Paint lane headers + empty-lane placeholders.

        Each header is a thin band with a left color bar (slot from
        :func:`_phase_layout_for_node`) and the phase's display name.
        Empty lanes draw a dashed rectangle + "no reward — under-bound"
        message styled with ``canvas_safety_warning`` to flag the Go2
        idle-limb-flailing pattern at editing time.
        """
        if not self._lane_segments:
            return
        title_font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        title_font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        title_font.setBold(True)
        body_font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        body_font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        body_font.setItalic(True)
        title_text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        warn_color = QColor(Config.get_color("canvas_safety_warning", "#F59E0B"))

        for seg in self._lane_segments:
            header_rect = self._lane_header_rect(seg)
            # 1. Left color bar (4 px wide) — phase identity at a glance.
            bar_color = QColor(Config.get_color(seg["color_slot"], "#999999"))
            painter.setBrush(QBrush(bar_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(
                    header_rect.left(), header_rect.top() + 4,
                    float(_INL_LANE_BAR_W), header_rect.height() - 8,
                ),
                1.5, 1.5,
            )
            # 2. Header label (display name).
            painter.setPen(QPen(title_text_color))
            painter.setFont(title_font)
            label_rect = QRectF(
                header_rect.left() + _INL_LANE_BAR_W + 8,
                header_rect.top(),
                header_rect.width() - _INL_LANE_BAR_W - 8,
                header_rect.height(),
            )
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                seg["display"],
            )
            # 3. Empty-lane placeholder — two kinds:
            #    * "uncovered": no specific AND no unconditional reward.
            #      Warning-coloured dashed box — the Go2 limb-flailing
            #      pattern; the user MUST address it.
            #    * "covered":   no specific reward, but unconditional
            #      rewards exist. They apply to every phase at runtime,
            #      so this is informational only — render a small muted
            #      "Covered by Unconditional" hint, no warning frame.
            if seg["placeholder"]:
                kind = seg.get("placeholder_kind", "uncovered")
                ph = self._lane_placeholder_rect(seg)
                if kind == "covered":
                    muted = QColor(Config.get_color("canvas_node_param_text", "#9CA3AF"))
                    painter.setPen(QPen(muted))
                    painter.setFont(body_font)
                    msg = tr(
                        "node.rewards.covered_lane",
                        "covered by Unconditional",
                    )
                    fm = QFontMetricsF(body_font)
                    elided = fm.elidedText(
                        msg, Qt.TextElideMode.ElideRight, ph.width() - 8,
                    )
                    painter.drawText(
                        ph,
                        int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                        elided,
                    )
                else:
                    dash_pen = QPen(warn_color, 1.0)
                    dash_pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(dash_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(ph, 3, 3)
                    painter.setPen(QPen(warn_color))
                    painter.setFont(body_font)
                    empty_msg = tr(
                        "node.rewards.empty_lane",
                        "no reward — phase under-constrained",
                    )
                    fm = QFontMetricsF(body_font)
                    elided = fm.elidedText(
                        empty_msg, Qt.TextElideMode.ElideRight, ph.width() - 8,
                    )
                    painter.drawText(
                        ph,
                        int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                        elided,
                    )

    def _paint_row_separators_phase(self, painter: QPainter) -> None:
        """Like the base separator pass, but never crosses a lane boundary."""
        n = len(self._row_payloads)
        if n < 2 or not self._lane_segments:
            return
        sep = QColor(Config.get_color("border_1", "#444444"))
        sep.setAlpha(140)
        painter.setPen(QPen(sep, 0.6))
        f = self._frame_rect()
        x_l = f.left() + _INL_FRAME_PAD + 4
        x_r = f.right() - _INL_FRAME_PAD - 4
        for seg in self._lane_segments:
            if seg["placeholder"]:
                continue
            for i in range(seg["start"] + 1, seg["end"]):
                r = self._row_rect(i)
                y = r.top()
                painter.drawLine(QPointF(x_l, y), QPointF(x_r, y))

    def _phase_button_summary(self, applies: List[str]) -> tuple:
        """Compute the chip label + bg color for a row's applies_to value.

        Mapping (single source of truth for both paint and tooltip):
            []                              -> ("All",  canvas_phase_global)
            ["static"]                      -> ("S",    canvas_phase_static)
            ["locomotion"]                  -> ("L",    canvas_phase_locomotion)
            ["agile"]                       -> ("A",    canvas_phase_agile)
            ["static", "locomotion"]        -> ("S+L",  canvas_phase_chip_bg)
            ["locomotion", "agile"]         -> ("L+A",  canvas_phase_chip_bg)
            three or more                   -> ("3 ph", canvas_phase_chip_bg)
            any unmapped phase id           -> ("!?",   canvas_phase_unmapped)
        """
        layout = self._phase_layout_for_node()
        layout_ids = [entry["id"] for entry in layout]
        layout_set = set(layout_ids)
        slot_by_id = {entry["id"]: entry["theme_slot"] for entry in layout}
        if not applies:
            return ("All", "main_c2")
        unmapped = [pid for pid in applies if pid not in layout_set]
        if unmapped:
            return ("!?", "canvas_phase_unmapped")
        mapped = [pid for pid in applies if pid in layout_set]
        if len(mapped) == 1:
            pid = mapped[0]
            initial = pid[:1].upper() if pid else "?"
            return (initial, slot_by_id.get(pid, "btn_1"))
        if len(mapped) == 2:
            # Keep the canonical phase order from the layout so "S+L" and
            # "L+S" never both appear for the same selection.
            ordered = [pid for pid in layout_ids if pid in mapped]
            label = "+".join(p[:1].upper() for p in ordered)
            return (label, "btn_1")
        return (f"{len(mapped)} ph", "btn_1")

    def _paint_row_phase_button(self, painter: QPainter, idx: int, payload: dict) -> None:
        """Always-visible clickable chip showing the row's applies_to summary.

        Single click on this rect opens the phase multi-select popup.
        The chip is the only visible affordance for re-assigning a
        reward term across phases — without it, the user can see the
        lane grouping but has no way to move terms between lanes.
        """
        applies: List[str] = list(payload.get("applies_to") or [])
        label, slot = self._phase_button_summary(applies)
        rect = self._row_phase_btn_rect(idx)
        bg = QColor(Config.get_color(slot, "#343636"))
        if self._is_hovering(rect):
            bg = bg.lighter(125)
        fg = QColor(Config.get_color("main_t1", "#D6D3C7"))
        border = QColor(Config.get_color("border_1", "#444444"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 0.8))
        painter.drawRoundedRect(rect, 3, 3)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _interactive_regions(self) -> List[QRectF]:  # type: ignore[override]
        regions = super()._interactive_regions()
        # observation: the leftmost ⬤ is a clickable dimension checkbox — it must
        # be a hit region or mousePressEvent never routes the click to
        # _handle_click → _toggle_obs_dimension (the "can't toggle" bug).
        if self._kind() == "observation":
            for i in range(len(self._row_payloads)):
                regions.append(self._row_badge_rect(i))
        if self._is_phase_aware():
            for i in range(len(self._row_payloads)):
                regions.append(self._row_phase_btn_rect(i))
        elif self._supports_grace():
            for i, p in enumerate(self._row_payloads):
                if self._row_supports_grace(p):
                    regions.append(self._row_grace_btn_rect(i))
        # Value-param chips coexist with the phase chip (rewards) — always
        # register them for value-bearing rows when the table has the column.
        if self._has_value_col():
            for i, p in enumerate(self._row_payloads):
                if self._row_supports_value(p):
                    regions.append(self._row_value_param_rect(i))
        return regions

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # type: ignore[override]
        # Phase-aware: a click on the row's phase chip button opens the
        # multi-select popup *before* the base class hits its slider
        # drag detection. Drop straight through to ``super()`` outside
        # phase-aware mode.
        # Value-param chip click (reward rows) — checked first since it can
        # coexist with the phase chip in phase-aware reward mode.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._has_value_col()
        ):
            for idx, p in enumerate(self._row_payloads):
                if self._row_supports_value(p) and self._row_value_param_rect(idx).contains(event.pos()):
                    self._open_item_value_popup(idx, event)
                    event.accept()
                    return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._is_phase_aware()
        ):
            for idx in range(len(self._row_payloads)):
                if self._row_phase_btn_rect(idx).contains(event.pos()):
                    self._open_phase_selector_popup(idx, event)
                    event.accept()
                    return
        # Grace chip click (termination rows) — intercept before the base
        # slider-drag detection, since the chip sits in the slider→value gap.
        elif (
            event.button() == Qt.MouseButton.LeftButton
            and self._supports_grace()
        ):
            for idx, p in enumerate(self._row_payloads):
                if self._row_supports_grace(p) and self._row_grace_btn_rect(idx).contains(event.pos()):
                    self._open_grace_popup(idx, event)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _open_phase_selector_popup(
        self, idx: int, event: QGraphicsSceneMouseEvent
    ) -> None:
        """Open a multi-select popup of phase ids for one reward row.

        The popup lists every phase in the active phase layout. Toggling
        rewrites the row's ``applies_to`` field — empty selection = ``[]``
        (Unconditional). On commit the underlying dict is re-serialized
        through the same setter the slider drag uses, so undo + canvas
        invalidation work the same way as a weight change.
        """
        payload = self._row_payloads[idx]
        layout = self._phase_layout_for_node()
        if not layout:
            return  # No phases registered — popup would be empty; skip.
        choices = [entry["id"] for entry in layout]
        labels = {
            entry["id"]: f"{entry['display_name']}"
            for entry in layout
        }
        descriptions = {
            entry["id"]: tr(
                f"motion.phase.{entry['id']}.desc",
                f"Apply this reward only when the active task item maps to the "
                f"{entry['display_name']!r} phase.",
            )
            for entry in layout
        }
        meta_map = {
            pid: {"title": labels[pid], "description": descriptions[pid]}
            for pid in choices
        }
        current = list(payload.get("applies_to") or [])

        def _on_pick(selected_ids):
            sel = selected_ids if isinstance(selected_ids, list) else (
                [selected_ids] if selected_ids else []
            )
            # Preserve the canonical layout order so undo diffs stay
            # readable ("static, locomotion" rather than user click order).
            ordered = [pid for pid in choices if pid in sel]
            self._set_applies_to(idx, payload, ordered)

        open_choice_popup(
            view=_view_of(event), row=self,
            choices=choices, current=current,
            multi=True, leading_mode="checkbox",
            meta_map=meta_map,
            on_commit=_on_pick,
        )

    def _set_applies_to(
        self, idx: int, payload: dict, applies: List[str]
    ) -> None:
        """Write ``applies_to`` back into the reward_terms dict + reflow.

        Mirrors the structure of ``_row_set_value`` (weight commit) so
        the rest of the row machinery — Coverage Badge re-evaluation,
        edge invalidation, undo wiring — picks the change up via the
        single ``set_value`` entrypoint.
        """
        payload["applies_to"] = list(applies)
        d = self._current_terms()
        cur = d.get(payload["key"])
        if isinstance(cur, dict):
            cur = dict(cur)
        else:
            # Legacy flat weight — upgrade to dict shape so applies_to
            # can be carried alongside.
            cur = {"weight": float(payload.get("weight", 0.0) or 0.0)}
        cur["applies_to"] = list(applies)
        d[payload["key"]] = cur
        self.set_value(_serialize_for_spec(self.spec, d))

    def _row_value(self, idx, payload):
        return float(payload.get("weight", 0.0))

    def _row_min(self, idx, payload):
        # Observation terms: the value cell is a SCALE, not a weight — the
        # registry item's [min_value, max_value] (default [0, 1]) is the weight
        # slider range and would wrongly clamp/reject a scale like 2.0 or 0.25
        # (legged_gym commands_scale). Use a wide bound so the input/validator
        # accepts any realistic scale (this was the "can't confirm / input
        # intercepted" bug). reward/termination keep their declared ranges.
        if self._kind() == "observation":
            return -1.0e6
        item = payload.get("item")
        if item is not None:
            return float(getattr(item, "min_value", 0.0))
        return -10.0

    def _row_max(self, idx, payload):
        if self._kind() == "observation":
            return 1.0e6
        item = payload.get("item")
        if item is not None:
            return float(getattr(item, "max_value", 1.0))
        return 10.0

    def _row_set_value(self, idx, payload, v):
        # Store the typed value verbatim. The old round(v/step)*step snapping
        # was a slider affordance; per-row sliders were removed in v2
        # (_row_slider_rect → empty rect), so snapping a popup-typed weight only
        # ever corrupts it — e.g. 0.01 against a registry step≥0.02 collapses to
        # 0.0, silently discarding the user's input (§8 "illusion of success").
        v = float(v)
        payload["weight"] = v
        d = self._current_terms()
        cur = d.get(payload["key"])
        if isinstance(cur, dict):
            cur["weight"] = v
        else:
            d[payload["key"]] = v
        self.set_value(_serialize_for_spec(self.spec, d))

    # ---- obs scale: scalar / per-channel array / JSON (variable-dim) ----

    def _obs_mode(self, payload: dict) -> str:
        """Current editor mode for an obs row: 'scalar' | 'array' | 'json'.

        Driven by the CURRENT value shape (not data_type — that gates toggling):
        a list value on a small fixed-dim term (velocity_command etc.) edits as
        per-channel boxes; a list on a variable-dim term (joint_pos ≈ 29) edits
        as JSON; anything scalar edits as one number."""
        if self._kind() != "observation":
            return "scalar"
        if not isinstance(payload.get("scale_value"), list):
            return "scalar"
        return "array" if obs_term_channel_labels(str(payload.get("key", ""))) else "json"

    def _row_array_labels(self, idx):  # type: ignore[override]
        """Channel labels — only when the row is CURRENTLY a fixed-dim vector
        (so a scalar obs term opens the single-number popup, not array boxes)."""
        try:
            payload = self._row_payloads[idx]
        except (IndexError, TypeError):
            return None
        if self._obs_mode(payload) != "array":
            return None
        return obs_term_channel_labels(str(payload.get("key", "")))

    def _row_array_values(self, idx, payload, n):  # type: ignore[override]
        sl = payload.get("scale_list")
        if sl is not None:
            return obs_scale_to_channels(sl, n)
        return obs_scale_to_channels(payload.get("weight", 1.0), n)

    def _row_set_array_value(self, idx, payload, values):  # type: ignore[override]
        # Keep the LIST shape (data_type governs shape now, not value equality —
        # collapsing an all-equal vector to a scalar would silently flip the
        # row's ⬤ checkbox). Preserve the term's data_type.
        vals = [float(v) for v in values]
        payload["scale_list"] = vals
        payload["scale_value"] = vals
        payload["weight"] = vals[0] if vals else 0.0
        d = self._current_terms()
        d[payload["key"]] = serialize_obs_term_value(vals, payload.get("data_type", "both"))
        self.set_value(_serialize_for_spec(self.spec, d))

    def _open_value_popup(self, idx, event):  # type: ignore[override]
        """obs JSON-mode value → anchored code popup (like il_policy_network's
        actor_hidden); scalar/array fall through to the base dispatch."""
        if self._kind() == "observation":
            payload = self._row_payloads[idx]
            if self._obs_mode(payload) == "json":
                self._open_obs_json_popup(idx, payload, event)
                return
        super()._open_value_popup(idx, event)

    def _open_obs_json_popup(self, idx, payload, event) -> None:
        import json as _json
        cur = payload.get("scale_value")
        if not isinstance(cur, list):
            cur = [float(payload.get("weight", 1.0) or 0.0)]
        text = _json.dumps([float(x) for x in cur])

        def _commit(s: str, _i=idx, _p=payload) -> None:
            try:
                parsed = _json.loads(s)
            except Exception:  # noqa: BLE001 — invalid JSON: keep prior value
                return
            if isinstance(parsed, (int, float)):
                vals = [float(parsed)]
            elif isinstance(parsed, list):
                try:
                    vals = [float(x) for x in parsed]
                except (TypeError, ValueError):
                    return
            else:
                return
            _p["scale_list"] = vals
            _p["scale_value"] = vals
            _p["weight"] = vals[0] if vals else 0.0
            d = self._current_terms()
            d[_p["key"]] = serialize_obs_term_value(vals, _p.get("data_type", "both"))
            self.set_value(_serialize_for_spec(self.spec, d))
            self.update()

        open_code_popup(
            view=_view_of(event), row=self,
            text=text, language="json", validate_json=True, on_commit=_commit,
        )

    # ---- ⬤ checkbox: toggle scalar <-> dimension (data_type == "both") ----

    def _toggle_obs_dimension(self, idx: int) -> None:
        try:
            payload = self._row_payloads[idx]
        except (IndexError, TypeError):
            return
        if payload.get("data_type") != "both":
            return  # locked — only "both" rows are user-togglable (req3)
        cur = payload.get("scale_value")
        if isinstance(cur, list):
            new: Any = float(cur[0]) if cur else 1.0       # vector → scalar
        else:
            base = float(cur) if isinstance(cur, (int, float)) else 1.0
            labels = obs_term_channel_labels(str(payload.get("key", "")))
            n = len(labels) if labels else 1               # json term → length 1
            new = [base] * n                                # scalar → vector
        d = self._current_terms()
        d[payload["key"]] = serialize_obs_term_value(new, "both")
        self.set_value(_serialize_for_spec(self.spec, d))   # → _refresh_payloads → repaint (req2)
        self.update()

    def _handle_click(self, event):  # type: ignore[override]
        # obs: clicking the leftmost ⬤ checkbox toggles dimension (req1/req2).
        if self._kind() == "observation":
            pos = event.pos()
            for idx in range(len(self._row_payloads)):
                if self._row_badge_rect(idx).contains(pos):
                    self._toggle_obs_dimension(idx)
                    return True
        return super()._handle_click(event)

    # ---- column header labels + hover tooltip ----

    def _col_value_label(self) -> str:
        k = self._kind()
        if k == "termination":
            return tr("canvas.inline.col_threshold", "Threshold")
        if k == "observation":
            return tr("canvas.inline.col_scale", "Scale")
        return tr("canvas.inline.col_weight", "Weight")

    def _has_grace_col(self) -> bool:
        return self._supports_grace() and any(
            self._row_supports_grace(p) for p in self._row_payloads
        )

    def _row_tooltip(self, idx: int) -> str:
        try:
            payload = self._row_payloads[idx]
        except (IndexError, TypeError):
            return ""
        item = payload.get("item")
        meta = payload.get("meta") or {}
        title = str(meta.get("title", "") or "").strip() if isinstance(meta, dict) else ""
        if not title and item is not None:
            title = str(getattr(item, "title", "") or "").strip()
        if not title:
            title = str(payload.get("key", ""))
        lines = [title, f"{self._col_value_label()}: {payload.get('weight', 0.0)}"]
        if self._row_supports_value(payload):
            item = payload.get("item")
            label = str(getattr(item, "il_value_label", "") or "Value")
            unit = str(getattr(item, "il_value_unit", "") or "")
            raw = payload.get("value")
            if raw is None:
                lines.append(f"{label}: auto")
            else:
                lines.append(f"{label}: {float(raw):g}{unit}")
        if self._row_supports_grace(payload):
            g = float(payload.get("grace_period_s", 0.0) or 0.0)
            lines.append(
                f"{self._col_grace_label()}: {g:.2f}s" if g > 0
                else f"{self._col_grace_label()}: off"
            )
        return "\n".join(lines)

    def _paint_obs_dim_checkbox(self, painter, rect, payload) -> None:
        """⬤ checkbox: filled green = dimension/list scale, hollow = scalar.

        A row locked to one type (data_type != 'both') is dimmed to hint it is
        not togglable here (change it in the gear editor's Data Type)."""
        is_list = isinstance(payload.get("scale_value"), list)
        togglable = payload.get("data_type") == "both"
        on = QColor(Config.get_color("safe_zone", "#3fae6a"))
        off = QColor(Config.get_color("border_1", "#444444"))
        if not togglable:
            on.setAlpha(150)
            off.setAlpha(150)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if is_list:
            painter.setBrush(QBrush(on))
            painter.setPen(QPen(on, 1.0))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(off, 1.4))
        painter.drawEllipse(rect)
        painter.restore()

    def _paint_row(self, painter, idx, payload):
        kind = self._kind()
        item = payload.get("item")

        # Badge. observation: a ⬤ CHECKBOX (green filled = dimension/list scale,
        # hollow = scalar) — click toggles when data_type=="both" (req1/req2).
        # reward/termination keep the polarity dot.
        badge_rect = self._row_badge_rect(idx)
        if kind == "observation":
            self._paint_obs_dim_checkbox(painter, badge_rect, payload)
        else:
            if kind == "reward":
                polarity = str(getattr(item, "polarity", "reward") or "reward")
                badge_kind = polarity if polarity in (
                    "reward", "penalty", "bidirectional", "neutral"
                ) else "reward"
            elif kind == "termination":
                badge_kind = "penalty"
            else:
                badge_kind = "neutral"
            try:
                draw_polarity_badge(painter, badge_rect, kind=badge_kind)
            except Exception:
                pass

        # Name —— 解析顺序：items dict 覆写 title > 注册表 title > dict key。
        # 这保证 Canvas 行、外部 picker、Editor 三处显示同一个 human-readable
        # 标签；只有 TaskModuleItem 不存在且 dict 没覆写时才退化到 id。
        name_rect = self._row_name_rect(idx)
        title = ""
        row_meta = payload.get("meta") or {}
        if isinstance(row_meta, dict):
            title = str(row_meta.get("title", "") or "").strip()
        if not title and item is not None:
            title = str(getattr(item, "title", "") or "").strip()
        if not title:
            title = payload["key"]
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        elided = fm.elidedText(str(title), Qt.TextElideMode.ElideRight, name_rect.width())
        painter.drawText(
            name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided,
        )

        # v2: per-row slider removed (tight node width). Weight is edited via
        # the value button (click → number popup). Value 按钮 —— 与 RangeRow
        # 端点按钮同款。Trim trailing zeros so "-2.000" reads "-2", "0.500" → "0.5".
        scale_list = payload.get("scale_list")
        if scale_list is not None and self._obs_mode(payload) == "json":
            # Variable-dim obs vector (joint_pos ≈ 29) — DON'T render the whole
            # list (would blow the column). Show a compact "[n] first…" summary;
            # the value cell click opens the anchored JSON editor.
            n = len(scale_list)
            head = f"{float(scale_list[0]):g}" if scale_list else ""
            text = f"[{n}] {head}…" if n > 1 else f"[{head}]"
        elif scale_list is not None:
            # Per-channel obs scale (velocity_command = [2, 2, 0.25]) — show the
            # vector; the value cell click opens N labeled number fields.
            text = "[" + ", ".join(f"{float(v):g}" for v in scale_list) + "]"
        else:
            weight = payload["weight"]
            if abs(weight) >= 1000 or (weight != 0 and abs(weight) < 0.01):
                text = f"{weight:.2e}"
            else:
                text = f"{weight:.3f}".rstrip("0").rstrip(".")
                if text in ("", "-", "-0"):
                    text = "0"
        self._paint_value_button(painter, idx, text, accent=True)

        # Value-param chip (reward rows whose item declares a tunable param).
        if self._row_supports_value(payload):
            self._paint_row_value_param_button(painter, idx, payload)

        # Grace-period chip (termination rows; not time_out).
        if self._row_supports_grace(payload):
            self._paint_row_grace_button(painter, idx, payload)

        # Column dividers [name | weight | value] on reward value-tables.
        # Drawn after the cells so the thin rule sits on top of the row bg.
        if self._has_value_col() and not self._is_phase_aware():
            self._paint_row_col_dividers(painter, idx)

    def _paint_row_col_dividers(self, painter: QPainter, idx: int) -> None:
        """Thin vertical rules separating the Name | Weight | Value columns."""
        r = self._row_rect(idx)
        wr = self._row_value_rect(idx)
        vr = self._row_value_param_rect(idx)
        sep = QColor(Config.get_color("border_1", "#444444"))
        sep.setAlpha(110)
        painter.setPen(QPen(sep, 0.8))
        top, bot = r.top() + 3.0, r.bottom() - 3.0
        for x in (wr.left() - _INL_GAP * 0.5, vr.left() - _INL_PHASE_BTN_GAP * 0.5):
            painter.drawLine(QPointF(x, top), QPointF(x, bot))

    def _commit_drag(self, idx: int, x: float) -> None:  # type: ignore[override]
        """Curve-aware drag commit — mirrors the warping done in ``_paint_row``.

        The base class implementation feeds ``x`` straight through a linear
        ``RangeSliderHitTest`` over [vmin, vmax]. Here we hit-test against
        the normalized [0, 1] slider space (matching what we drew), then
        un-warp the resulting position into the true value via
        :func:`_slider_pos_to_value`. (Per-row sliders were removed in v2, so
        this drag path is dead — _row_slider_rect returns an empty rect; the
        value is edited via the typed number popup, which stores verbatim.)
        """
        payload = self._row_payloads[idx]
        vmin = self._row_min(idx, payload)
        vmax = self._row_max(idx, payload)
        ht = RangeSliderHitTest(
            self._row_slider_rect(idx),
            vmin=0.0, vmax=1.0,
            dual=False,
        )
        pos = ht.value_at(x)
        v = _slider_pos_to_value(pos, vmin, vmax)
        self._row_set_value(idx, payload, v)
        self.update()

    def _on_selector_clicked(self, event) -> bool:
        try:
            from scripts.query import query_registry
            registry = query_registry(kind=self._kind(), backend=self._backend())
        except Exception:
            return False
        if not registry:
            return False
        d = self._current_terms()
        # 缺口③ — on a joint page, only joint-paginable rewards may be added;
        # the global page offers all. Filtering the popup's candidate list keeps
        # the canvas in sync with what the compiler will accept (§8).
        if self._is_paged() and self._active_page() != "__global__":
            registry = {
                k: v for k, v in registry.items()
                if str(getattr(v, "il_partition_source", "") or "")
            }
            if not registry:
                return False
        # popup 卡片标题用 TaskModuleItem.title，desc 当副标题 —— 与 Canvas 行
        # 和 Editor 列表的标签规则一致。
        meta_map: dict = {}
        for k, v in registry.items():
            label = str(getattr(v, "title", "") or "").strip() or k
            meta_map[k] = {
                "title": label,
                "description": str(getattr(v, "desc", "") or ""),
            }
        # A-Z 排序：用户在大量奖励/观测条目里找项时，按显示 title 字母排序
        # 显著加快查找；用 casefold 做 locale-agnostic 大小写折叠，相同 title
        # fallback 到 key 保稳定。
        choices = sorted(
            registry.keys(),
            key=lambda k: (meta_map[k]["title"].casefold(), k),
        )
        current = list(d.keys())

        def _on_pick(selected):
            sel_list = selected if isinstance(selected, list) else (
                [selected] if selected else []
            )
            new_d: dict = {}
            for k in sel_list:
                if k in d:
                    new_d[k] = d[k]
                else:
                    item = registry.get(k)
                    new_d[k] = float(getattr(item, "default", 1.0)) if item is not None else 1.0
            self.set_value(_serialize_for_spec(self.spec, new_d))

        open_choice_popup(
            view=_view_of(event), row=self,
            choices=choices, current=current,
            multi=True, leading_mode="checkbox",
            meta_map=meta_map,
            on_commit=_on_pick,
        )
        return True

    def _on_modal_clicked(self, event) -> bool:
        from application.ui.dialogs import open_reward_function_editor

        meta = self.spec.meta or {}
        backend = self._backend()
        if backend == "isaac_lab":
            registry_id = str(meta.get("registry_id_il") or meta.get("registry_id") or self.spec.key)
        else:
            registry_id = str(meta.get("registry_id") or self.spec.key)
        items = self._current_terms()
        result = open_reward_function_editor(
            _view_of(event), dict(items),
            registry_id=registry_id, kind=self._kind(),
            canvas_backend=backend,
        )
        if result is not None:
            self.set_value(_serialize_for_spec(self.spec, result))
        return True

    def maybe_refresh_for_asset_change(self) -> None:
        # Rebuild on canvas backend swap or sibling param change. 缺口③ — re-read
        # self._value from params first so an external write to reward_terms /
        # reward_active_page (by the joint pager) is reflected: the pager
        # manipulates the FULL paged dict + the active-page cursor, both of
        # which land in params, and this row must project the (possibly new)
        # active page.
        self._value = self._read_value()
        self._refresh_payloads()
        self.update()


class TrainingItemsInlineRow(_InlineTableRow):
    """``widget="training_items"`` — DEMO ``_TrainingItemEditor`` 内联表格.

    每行 ``[clip indicator | item_id | smax slider | smax]``。Items dict 形如
    ``{item_id: {"enabled": bool, "speed": [lo, hi], "clip": str|None, "advanced": dict}}``，
    smax 在 [0, 1.5] 区间，step=0.05 —— 与 DEMO 完全一致。
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._refresh_payloads()

    def set_value(self, v):  # type: ignore[override]
        super().set_value(v)
        self._refresh_payloads()

    def _refresh_payloads(self) -> None:
        # 仅显示 ``enabled=True`` 的 item —— 用户明确要求"checked 的 item 才在
        # 表格中显示"。被取消勾选的 item 仍在 dict 中保留 ``enabled=False``，
        # 但不进 row payload 列表。
        d = _parse_dict_value(self._value) or {}
        payloads = []
        for item_id, entry in d.items():
            if not isinstance(entry, dict):
                entry = {"enabled": True, "speed": [0.0, float(entry or 1.0)],
                         "clip": None, "advanced": {}}
            if not bool(entry.get("enabled", True)):
                continue
            speed = entry.get("speed") or [0.0, 1.0]
            try:
                smax = float(speed[1]) if len(speed) >= 2 else 1.0
            except (TypeError, ValueError):
                smax = 1.0
            smax = max(0.0, min(1.5, smax))
            # Display label: the user-typed ``title`` wins; when blank we fall
            # back to the item id (matches the modal editor's ``_display_label``
            # semantics). ``key`` remains the authoritative id used for the
            # ``reward_in__<id>`` ports and sampling — only the rendered label
            # changes.
            payloads.append({
                "key": item_id,
                "label": str(entry.get("title") or "").strip() or item_id,
                "smax": smax,
                "has_clip": bool(entry.get("clip")),
                "enabled": True,
                "entry": entry,
            })
        self._row_payloads = payloads
        self._eff_weights = self._compute_effective_weights()
        self._set_height(self._compute_height())

    def _compute_effective_weights(self) -> list:
        """Per-row normalised sampling weight, aligned to ``_row_payloads``.

        Mirrors ``env_cfg_compiler._resolve_initial_item_weights``: each enabled
        item's optional ``weight`` (missing → the uniform share ``1/N``),
        normalised to sum 1. This is exactly what the env samples with, so the
        Weight column shows the effective distribution even before the user has
        opened the pie editor (when all items are still uniform)."""
        n = len(self._row_payloads)
        if n == 0:
            return []
        uniform = 1.0 / n
        raw = []
        for p in self._row_payloads:
            entry = p.get("entry") or {}
            w = entry.get("weight")
            try:
                raw.append(uniform if w is None else max(0.0, float(w)))
            except (TypeError, ValueError):
                raw.append(uniform)
        total = sum(raw)
        if total <= 0.0:
            return [uniform] * n
        return [w / total for w in raw]

    def _header_summary_text(self) -> str:
        n = len(self._row_payloads)
        if n == 0:
            return "—"
        enabled = sum(1 for p in self._row_payloads if p.get("enabled", True))
        return f"{enabled}/{n} enabled"

    def iter_inline_port_anchors(self):
        """One ``reward_pipe`` input port per enabled training item.

        Anchor = centre of the row's leading badge slot (formerly the clip
        indicator). Returned in row-local coordinates — NodeItem translates
        to node-local during layout. The PortSpec name is built from
        ``training_motion.reward_port_name(item_id)`` so the canvas, the
        IR loader, and the training-spec compiler agree on the key.

        Each per-item port is fan-in capable (``multi=True``): a single
        training item may receive several reward feeds from different
        RewardsNodes that contribute to the same item_id. The spec is
        produced by ``training_motion.iter_reward_port_specs`` so that
        the canvas, the compiler, and migrations all share one source of
        truth for port shape (type / multi / optional) and can't drift.
        """
        from nodes.training_motion.node import (
            iter_reward_port_specs,
            reward_port_name,
        )

        items_dict = {
            str(p.get("key", "")): {"enabled": True}
            for p in self._row_payloads
            if str(p.get("key", "") or "")
        }
        specs_by_name = {
            s.name: s
            for s in iter_reward_port_specs({"training_items": items_dict})
        }

        for idx, payload in enumerate(self._row_payloads):
            item_id = str(payload.get("key", "") or "")
            if not item_id:
                continue
            spec = specs_by_name.get(reward_port_name(item_id))
            if spec is None:
                continue
            badge = self._row_badge_rect(idx)
            anchor = QPointF(badge.left(), badge.top() + badge.height() * 0.5)
            yield spec, anchor

    def _row_value(self, idx, payload):
        return float(payload.get("smax", 1.0))

    def _row_min(self, idx, payload):
        return 0.0

    def _row_max(self, idx, payload):
        return 1.5

    def _row_set_value(self, idx, payload, v):
        # Clamp to the valid speed domain [0, 1.5] m/s (a real bound), but store
        # the typed value verbatim otherwise. The old round(v/0.05)*0.05
        # snapping was a slider affordance and the per-row slider was removed in
        # v2 — silently rewriting 0.62 → 0.60 is the same §8 input-mutation bug.
        v = max(0.0, min(1.5, float(v)))
        payload["smax"] = v
        d = self._current_terms()
        cur = d.get(payload["key"])
        if isinstance(cur, dict):
            speed = list(cur.get("speed") or [0.0, 1.0])
            lo = float(speed[0]) if speed else 0.0
            if lo > v:
                lo = v
            cur["speed"] = [lo, v]
        else:
            d[payload["key"]] = {"enabled": True, "speed": [0.0, v], "clip": None, "advanced": {}}
        self.set_value(_serialize_for_spec(self.spec, d))

    def _col_name_label(self) -> str:
        return tr("canvas.inline.col_item", "Item")

    def _col_value_label(self) -> str:
        return tr("canvas.inline.col_maxspeed", "Max Spd")

    def _row_tooltip(self, idx: int) -> str:
        try:
            p = self._row_payloads[idx]
        except (IndexError, TypeError):
            return ""
        return f"{p.get('label') or p.get('key', '')}\n{self._col_value_label()}: {float(p.get('smax', 0.0)):.2f}"

    def _paint_row(self, painter, idx, payload):
        # v2: the legacy circular clip indicator was removed to free up the
        # row-leading anchor for a per-item ``reward_pipe`` input port (the
        # PortItem is laid out by NodeItem against ``_row_badge_rect(idx)``).
        # Clip-bound state now drives the item name color instead: bound →
        # theme_2 (the "reference highlight" accent), unbound → normal text.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Name color: enabled+clip-bound → theme_2 reference highlight;
        # enabled+unbound → default title color; disabled → dim main_c2.
        name_rect = self._row_name_rect(idx)
        if not payload.get("enabled", True):
            text_color = QColor(Config.get_color("main_c2"))
        elif payload.get("has_clip"):
            text_color = QColor(Config.get_color("theme_2"))
        else:
            text_color = QColor(Config.get_color("main_t1"))
        painter.setPen(QPen(text_color))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        elided = fm.elidedText(str(payload.get("label") or payload["key"]), Qt.TextElideMode.ElideRight, name_rect.width())
        painter.drawText(
            name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided,
        )

        # v2: per-row slider removed (tight node width). smax is edited via
        # the value button (click → number popup). Value 按钮 —— 与 RangeRow
        # 端点按钮同款。
        smax = payload["smax"]
        self._paint_value_button(painter, idx, f"{smax:.2f}", accent=True)

        # Weight column (far right of Max Spd). Shows the effective sampling
        # weight %; clicking it opens the Weight Set pie editor (= the node's
        # Weight Set button). Drawn as an accent-coloured number so it reads as
        # interactive; a subtle hover background reinforces it.
        self._paint_weight_cell(painter, idx)

    # ---- Weight column ----------------------------------------------------
    # A "Weight" column is appended to the RIGHT of the Max-Spd value cell. The
    # reserved width shifts the value cell left and (via _row_name_right) shrinks
    # the item-name column — the row's total width is unchanged.

    def _row_right_reserved(self, idx: int) -> float:
        return float(_INL_WEIGHT_W + _INL_GAP)

    def _row_weight_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        return QRectF(
            r.right() - _INL_WEIGHT_W, r.top(), float(_INL_WEIGHT_W), r.height(),
        )

    def _col_weight_label(self) -> str:
        return tr("canvas.inline.col_weight", "Weight")

    def _row_weight_pct_text(self, idx: int) -> str:
        eff = getattr(self, "_eff_weights", None) or []
        if idx < 0 or idx >= len(eff):
            return "—"
        return f"{eff[idx] * 100.0:.0f}%"

    def _paint_weight_cell(self, painter: QPainter, idx: int) -> None:
        rect = self._row_weight_rect(idx)
        if self._is_hovering(rect):
            bg = QColor(Config.get_color("btn_1", "#343636")).lighter(110)
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(QColor(Config.get_color("border_1", "#444444")), 1.0))
            painter.drawRoundedRect(rect.adjusted(1, 4, -1, -4), 3, 3)
        painter.setPen(QPen(QColor(Config.get_color("theme_2", "#F6D393"))))
        painter.setFont(self._val_btn_font())
        painter.drawText(
            rect.adjusted(0, 0, -4, 0),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
            self._row_weight_pct_text(idx),
        )

    def _paint_col_header(self, painter: QPainter) -> None:
        super()._paint_col_header(painter)
        if not self._row_payloads:
            return
        hdr = self._col_header_rect()
        w_r = self._row_weight_rect(0)
        color = QColor(Config.get_color("main_c2", "#888888"))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color))
        fm = QFontMetricsF(font)
        wx = QRectF(w_r.left(), hdr.top(), w_r.width(), hdr.height())
        painter.drawText(
            wx, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
            fm.elidedText(self._col_weight_label(), Qt.TextElideMode.ElideRight,
                          max(0.0, wx.width())),
        )

    def _interactive_regions(self) -> List[QRectF]:
        regions = super()._interactive_regions()
        for i in range(len(self._row_payloads)):
            regions.append(self._row_weight_rect(i))
        return regions

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        pos = event.pos()
        for idx in range(len(self._row_payloads)):
            if self._row_weight_rect(idx).contains(pos):
                items_raw = _parse_dict_value(self._value) or {}
                updated = _run_weight_set_dialog(_view_of(event), items_raw)
                if updated is not None:
                    self.set_value(_serialize_for_spec(self.spec, updated))
                return True
        return super()._handle_click(event)

    def _on_selector_clicked(self, event) -> bool:
        # Multi-toggle the existing item_ids on/off (DEMO selector for training items
        # gates each known item by checkbox; real registry of training_items lives in
        # scripts.training_motion.library, not exposed here yet — fall back to
        # toggling the entries currently in the dict).
        d = _parse_dict_value(self._value) or {}
        if not d:
            return False
        # A-Z 排序——和 RegistryModuleInlineRow 的 picker 行为对齐，便于查找。
        choices = sorted(d.keys(), key=lambda k: str(k).casefold())
        current = [k for k, v in d.items() if isinstance(v, dict) and v.get("enabled", True)]

        def _on_pick(selected):
            sel = selected if isinstance(selected, list) else ([selected] if selected else [])
            sel_set = set(sel)
            new_d = dict(d)
            for k, v in new_d.items():
                if isinstance(v, dict):
                    v["enabled"] = (k in sel_set)
            self.set_value(_serialize_for_spec(self.spec, new_d))

        open_choice_popup(
            view=_view_of(event), row=self,
            choices=choices, current=current,
            multi=True, leading_mode="checkbox",
            on_commit=_on_pick,
        )
        return True

    def _on_modal_clicked(self, event) -> bool:
        from application.ui.dialogs import open_training_motion_editor

        items = _parse_dict_value(self._value)
        backend = str(self._owner.params.get("backend", "sb3_mujoco") or "sb3_mujoco").strip()
        result = open_training_motion_editor(
            _view_of(event), dict(items), canvas_backend=backend,
        )
        if result is not None:
            self.set_value(_serialize_for_spec(self.spec, result))
        return True

    def maybe_refresh_for_asset_change(self) -> None:
        self._refresh_payloads()
        self.update()


# =============================================================================
# Export node — DEMO §1 / §2 / §3 widgets
# =============================================================================
#
# DEMO 对应 (training_node_items.py 行号见各类 docstring)：
#   _make_output_file_list           → OutputFileListRow            (widget="output_file_list")
#   _make_start_point_compat_panel   → StartPointCompatPanelRow     (widget="start_point_compat_panel")
#   _make_review_backend_picker      → ReviewBackendPickerRow       (widget="review_backend_picker")
#   _make_review_scene_picker        → ReviewScenePickerRow         (widget="review_scene_picker")
#   _make_review_launch_button       → ReviewLaunchButtonRow        (widget="review_launch_button")
#
# 所有颜色走 Config.get_color；行高动态（OutputFileListRow / StartPointCompatPanelRow
# 通过 `preferred_height` + on_height_changed 触发 NodeItem reflow）。
# 后端依赖（ExportedBundleRegistry / list_review_backends / list_review_scenes /
# TrainingContext.compat_report / scene.review_launch_requested 信号）走防御性
# import — Stage C/D 未到位时降级显示，不抛错。

# ---- shared per-row constants ----
_OFL_FRAME_PAD = 4
_OFL_HDR_H = 18
_OFL_FILE_H = 16
_OFL_SEP_H = 1
_OFL_OPEN_BTN_W = 42.0
_OFL_OPEN_BTN_H = 14.0
_OFL_OPEN_BTN_PAD = 6.0

_SPC_FRAME_PAD = 4
_SPC_HDR_H = 18
_SPC_ROW_H = 16
_SPC_FOOTER_H = 16

_RLB_FRAME_PAD = 2  # outer pad inside the row's full-width value rect
_RLB_TEXT = "▶  Launch Review"


def _open_path_in_explorer(path) -> None:
    """Open ``path`` (file or directory) in the OS file manager.

    Cross-platform via QDesktopServices.openUrl(QUrl.fromLocalFile(...)). Mirrors
    DEMO ``_open_path`` (training_node_items.py:8756-8761). Silently no-ops on
    failure — caller already gates on path existence.
    """
    try:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    except Exception:
        pass


def _bundle_root_for(
    bundle_name: str, version: str, overwrite: bool, backend_id: str = "",
):
    """Resolve the bundle directory the next training run will write to.

    Layout (mirrors :class:`BundleExporter.export_from_artifacts`)::

        <project>/training/exported/<backend_id>/<bundle_name>[_<version>]/

    When no project is bound this returns a non-existent sentinel Path
    (``<no project>/<bundle_name>``) — the UI row treats that as
    "no real path yet" and ``.is_dir()`` reports False; exporting in this
    state is rejected by BundleExporter itself.

    ``backend_id`` should be the canvas-bound backend (the per-node
    ``params['backend']`` hint, or ``current_backend()``). Falls back to
    ``"unknown"`` when empty so paths stay deterministic.

    Overwrite=True → ``<bundle_name>``; Overwrite=False →
    ``<bundle_name>_<version>``. Returns ``Path`` even when bundle_name
    is the ``<NEW>`` sentinel — caller should treat that case as
    "no real path yet" and skip exists() checks.
    """
    from pathlib import Path

    name = (bundle_name or "").strip()
    if not overwrite and version:
        name = f"{name}_{version}"

    # Project-scoped path mirrors BundleExporter — partitioned by backend.
    try:
        from application.service.projects import current_project_info
        proj = current_project_info()
    except Exception:
        proj = None
    if proj is None:
        return Path("<no project>") / name

    bid = (backend_id or "").strip()
    if not bid:
        try:
            from application.service.signals import current_backend
            bid = (current_backend() or "").strip()
        except Exception:
            bid = ""
    if not bid:
        bid = "unknown"
    return proj.path / "training" / "exported" / bid / name


def _planned_export_files(host_params: dict) -> list:
    """Return ``[(filename, required_bool), ...]`` ordered as DEMO renders them.

    Mirrors DEMO ``_planned_files`` (training_node_items.py:8724-8740):
    manifest.yaml + source.json are always emitted; ONNX / TorchScript / norm
    stats are gated by their include_* toggles. Required flag is informational
    — used to mark *.yaml/*.json as "always written" in a future tier.
    """
    files = [("manifest.yaml", True), ("source.json", True)]
    if bool(host_params.get("include_onnx", True)):
        files.append(("policy.onnx", False))
    if bool(host_params.get("include_torchscript", True)):
        files.append(("policy.pt", False))
    if bool(host_params.get("include_normalization", True)):
        files.append(("normalization.pkl", False))
    return files


# =============================================================================
# OutputFileListRow — DEMO §1 Output Manifest
# =============================================================================

class OutputFileListRow(ParamRow):
    """Read-only file manifest for the Export node.

    DEMO 对应 ``_make_output_file_list`` (training_node_items.py:8595-8915).

    Header row : <bundle dir name> [⚠ exists] [Open]
    Separator  : 1px line, slot ``canvas_node_border``
    File rows  : <filename> [⚠ Will Overwrite] <Exists / Planned> [Open]

    Files come from ``_planned_export_files(host_params)`` so the include_*
    toggles drive the visible list. Auto-rebuilds via the existing
    ``maybe_refresh_for_asset_change`` hook NodeItem fires whenever any sibling
    param commits.
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._planned: list = []      # [(filename, required), ...]
        self._existing: set = set()   # filenames present on disk
        self._bundle_dir = None       # Path | None
        self._bundle_dir_exists: bool = False
        self._open_hits: list = []    # [(QRectF, Path), ...]
        self._height_callbacks: list = []
        self._height = float(self._compute_height_empty())
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

    # ---- height contract ----

    def preferred_height(self) -> float:
        return float(self._height)

    def on_height_changed(self, callback) -> None:
        if callable(callback) and callback not in self._height_callbacks:
            self._height_callbacks.append(callback)

    def _compute_height_empty(self) -> int:
        return _OFL_FRAME_PAD * 2 + _OFL_HDR_H + _OFL_SEP_H + _OFL_FILE_H + 2

    def _compute_height(self, n_files: int) -> int:
        return (
            _OFL_FRAME_PAD * 2
            + _OFL_HDR_H + _OFL_SEP_H
            + max(1, n_files) * _OFL_FILE_H
            + 2
        )

    def _set_height(self, h: float) -> None:
        h = float(max(_OFL_FILE_H, h))
        if abs(h - self._height) < 0.5:
            return
        self.prepareGeometryChange()
        self._height = h
        self.update()
        for cb in list(self._height_callbacks):
            try:
                cb(h)
            except Exception:
                pass

    # ---- auto-refresh hook ----

    def maybe_refresh_for_asset_change(self) -> None:
        """NodeItem.on_param_changed fan-out — rebuild whenever any sibling
        commits a value (cheap: planned-list compute + 1 Path.exists per file).
        """
        self._rebuild()

    def _rebuild(self) -> None:
        params = self._owner.params
        bundle_name = str(params.get("bundle_name", "<NEW>") or "<NEW>").strip()
        version = str(params.get("version", "v1") or "v1")
        overwrite = bool(params.get("overwrite", True))
        backend_id = str(params.get("backend", "") or "").strip()
        bundle_dir = _bundle_root_for(bundle_name, version, overwrite, backend_id)
        self._bundle_dir = bundle_dir
        self._bundle_dir_exists = (
            bundle_name not in ("", "<NEW>") and bundle_dir.is_dir()
        )
        self._planned = _planned_export_files(params)
        self._existing = set()
        if self._bundle_dir_exists:
            for fname, _req in self._planned:
                try:
                    if (bundle_dir / fname).exists():
                        self._existing.add(fname)
                except OSError:
                    pass
        self._set_height(self._compute_height(len(self._planned)))
        self.update()

    # ---- click handling ----

    def _interactive_regions(self) -> List[QRectF]:
        return [r for r, _p in self._open_hits]

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        for r, path in self._open_hits:
            if r.contains(event.pos()):
                _open_path_in_explorer(path)
                return True
        return False

    # ---- paint ----

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        # OutputFileListRow 是 full-width — value 半边自 SEP_X 一直到 _width。
        rect = QRectF(SEP_X + 4, 0, self._width - SEP_X - 8, self._height)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("canvas_node_param_text", "#9CA3AF"))
        muted = QColor(Config.get_color("canvas_node_export_muted_text", "#888888"))
        warn = QColor(Config.get_color("canvas_node_export_warn_text", "#F59E0B"))
        ok = QColor(Config.get_color("canvas_node_export_ok_text", "#4CAF50"))
        sep = QColor(Config.get_color("border_1", "#444444"))

        self._open_hits = []

        y = float(_OFL_FRAME_PAD)
        # ---- Header row ----
        hdr_rect = QRectF(rect.left(), y, rect.width(), _OFL_HDR_H)
        bundle_label = (
            self._bundle_dir.name if self._bundle_dir is not None
            else tr("node.export.manifest.placeholder", "<NEW>")
        )
        # bundle name (bold-ish)
        bold = QFont(font)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QPen(text_color))
        fm = QFontMetricsF(bold)
        # Reserve right side for ⚠ + Open if dir exists
        right_x = hdr_rect.right()
        if self._bundle_dir_exists and self._bundle_dir is not None:
            btn_rect = QRectF(
                right_x - _OFL_OPEN_BTN_W,
                hdr_rect.center().y() - _OFL_OPEN_BTN_H * 0.5,
                _OFL_OPEN_BTN_W, _OFL_OPEN_BTN_H,
            )
            self._draw_open_button(painter, btn_rect)
            self._open_hits.append((btn_rect, self._bundle_dir))
            right_x = btn_rect.left() - 4.0
            # ⚠ tag just left of the button
            painter.setPen(QPen(warn))
            small = QFont(font); small.setPixelSize(max(9, int(font.pixelSize()) - 1))
            painter.setFont(small)
            warn_text = tr("node.export.manifest.exists_warn", "⚠ exists")
            wfm = QFontMetricsF(small)
            ww = wfm.horizontalAdvance(warn_text)
            warn_rect = QRectF(right_x - ww - 4, hdr_rect.top(), ww + 4, hdr_rect.height())
            painter.drawText(
                warn_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                warn_text,
            )
            right_x = warn_rect.left() - 4.0
            painter.setFont(bold)
            painter.setPen(QPen(text_color))
        name_rect = QRectF(hdr_rect.left(), hdr_rect.top(), max(0.0, right_x - hdr_rect.left()), hdr_rect.height())
        elided = fm.elidedText(bundle_label, Qt.TextElideMode.ElideMiddle, name_rect.width())
        painter.drawText(
            name_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        painter.setFont(font)

        # ---- separator ----
        y += _OFL_HDR_H
        sep_pen = QPen(sep, 0.6); sep_pen.setCosmetic(True)
        painter.setPen(sep_pen)
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        y += _OFL_SEP_H

        # ---- file rows ----
        for fname, _required in self._planned:
            row_rect = QRectF(rect.left(), y, rect.width(), _OFL_FILE_H)
            exists = fname in self._existing
            # name on the left
            painter.setPen(QPen(text_color if exists else muted))
            name_w = max(0.0, row_rect.width() * 0.55)
            painter.drawText(
                QRectF(row_rect.left(), row_rect.top(), name_w, row_rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                fname,
            )
            # right side: status + Open
            right = row_rect.right()
            if exists and self._bundle_dir is not None:
                btn_rect = QRectF(
                    right - _OFL_OPEN_BTN_W,
                    row_rect.center().y() - _OFL_OPEN_BTN_H * 0.5,
                    _OFL_OPEN_BTN_W, _OFL_OPEN_BTN_H,
                )
                self._draw_open_button(painter, btn_rect)
                self._open_hits.append((btn_rect, self._bundle_dir / fname))
                right = btn_rect.left() - 4.0
            # status text
            status_text = (
                tr("node.export.manifest.status.exists", "Exists") if exists
                else tr("node.export.manifest.status.planned", "Planned")
            )
            painter.setPen(QPen(warn if exists else ok))
            sfm = QFontMetricsF(font)
            sw = sfm.horizontalAdvance(status_text)
            status_rect = QRectF(right - sw - 4, row_rect.top(), sw + 4, row_rect.height())
            painter.drawText(
                status_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                status_text,
            )
            y += _OFL_FILE_H

    def _draw_open_button(self, painter: QPainter, btn: QRectF) -> None:
        bg = QColor(Config.get_color("bg_1", "#1E1E1E"))
        if self._is_hovering(btn):
            bg = bg.lighter(125)
        border = QColor(Config.get_color("border_2", "#3d3d3d"))
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.0))
        painter.drawRoundedRect(btn, 2, 2)
        text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        painter.setPen(QPen(text_color))
        small = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        small.setPixelSize(9)
        painter.setFont(small)
        painter.drawText(
            btn,
            int(Qt.AlignmentFlag.AlignCenter),
            tr("node.export.manifest.open", "Open"),
        )


# =============================================================================
# StartPointCompatPanelRow — DEMO §2 Start Point Compatibility
# =============================================================================

_SPC_FIELDS = ("Framework", "Action dim", "Decimation", "Joint order")
# i18n key per field name. The English labels above are kept as lookup keys
# matching ``StartPointCompatReport.fields[*].label`` (data source emits English
# labels); the dict maps them to translation slots for display only.
_SPC_FIELD_KEYS = {
    "Framework": "node.export.compat.field.framework",
    "Action dim": "node.export.compat.field.action_dim",
    "Decimation": "node.export.compat.field.decimation",
    "Joint order": "node.export.compat.field.joint_order",
}


class StartPointCompatPanelRow(ParamRow):
    """Read-only Start Point compatibility diff for the Export node.

    DEMO 对应 ``_make_start_point_compat_panel`` (training_node_items.py
    :8921-9200).

    Header           : "Start Point Compatibility"
    4 fixed rows     : Framework / Action dim / Decimation / Joint order
                       each painted with [✓ / ⚠ / ✗] icon + old → new
    Footer summary   : one-liner pulled from StartPointCompatReport.summary

    Data source: ``application.training.training_context.get_training_context()``
    + ``ctx.compat_report(graph)``. Stage B falls back to the placeholder
    "Backend wiring pending" string until Stage D ports TrainingContext.
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._report = None         # StartPointCompatReport | None
        self._error: str = ""
        self._height_callbacks: list = []
        self._height = float(self._compute_height())
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)

    def preferred_height(self) -> float:
        return float(self._height)

    def on_height_changed(self, callback) -> None:
        if callable(callback) and callback not in self._height_callbacks:
            self._height_callbacks.append(callback)

    def _compute_height(self) -> int:
        return (
            _SPC_FRAME_PAD * 2
            + _SPC_HDR_H
            + len(_SPC_FIELDS) * _SPC_ROW_H
            + _SPC_FOOTER_H + 2
        )

    def _set_height(self, h: float) -> None:
        h = float(max(_SPC_ROW_H, h))
        if abs(h - self._height) < 0.5:
            return
        self.prepareGeometryChange()
        self._height = h
        self.update()
        for cb in list(self._height_callbacks):
            try:
                cb(h)
            except Exception:
                pass

    def maybe_refresh_for_asset_change(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        self._report = None
        self._error = ""
        scene = self.scene()
        graph = None
        if scene is not None:
            try:
                graph = scene.serialize_training_graph(for_compiler=True)
            except Exception:
                graph = None
        if graph is None:
            self._error = ""  # tier-1 fallback: render the static field skeleton
            self._set_height(self._compute_height())
            self.update()
            return
        try:
            from application.training.training_context import (
                get_training_context,
            )
            ctx = get_training_context()
            self._report = ctx.compat_report(graph)
        except Exception as e:
            self._error = f"compat report unavailable: {e}"
        self._set_height(self._compute_height())
        self.update()

    def _interactive_regions(self) -> List[QRectF]:
        return []  # read-only panel

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        rect = QRectF(SEP_X + 4, 0, self._width - SEP_X - 8, self._height)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        painter.setFont(font)
        text_color = QColor(Config.get_color("canvas_node_param_text", "#9CA3AF"))
        muted = QColor(Config.get_color("canvas_node_export_muted_text", "#888888"))
        ok = QColor(Config.get_color("canvas_node_export_ok_text", "#4CAF50"))
        warn = QColor(Config.get_color("canvas_node_export_warn_text", "#F59E0B"))
        err = QColor(Config.get_color("canvas_safety_error", "#EF4444"))

        y = float(_SPC_FRAME_PAD)
        # ---- header ----
        bold = QFont(font); bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QPen(text_color))
        painter.drawText(
            QRectF(rect.left(), y, rect.width(), _SPC_HDR_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            tr("node.export.compat.title", "Start Point Compatibility"),
        )
        painter.setFont(font)
        y += _SPC_HDR_H

        report = self._report
        # Build a quick label→(status, old, new, detail) dict from the report.
        field_data = {}
        if report is not None and getattr(report, "fields", None):
            for f in report.fields:
                field_data[str(getattr(f, "label", ""))] = (
                    str(getattr(f, "status", "")),
                    str(getattr(f, "old_value", "")),
                    str(getattr(f, "new_value", "")),
                    str(getattr(f, "detail", "")),
                )

        for label in _SPC_FIELDS:
            row_rect = QRectF(rect.left(), y, rect.width(), _SPC_ROW_H)
            status, old_v, new_v, _detail = field_data.get(label, ("", "—", "—", ""))
            if status == "ok":
                icon, color = "✓", ok
            elif status == "warning":
                icon, color = "⚠", warn
            elif status == "error":
                icon, color = "✗", err
            else:
                icon, color = "·", muted
            # icon column (12px)
            painter.setPen(QPen(color))
            painter.drawText(
                QRectF(row_rect.left(), row_rect.top(), 12.0, row_rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                icon,
            )
            # label
            painter.setPen(QPen(text_color))
            label_w = max(60.0, row_rect.width() * 0.32)
            label_key = _SPC_FIELD_KEYS.get(label)
            label_display = tr(label_key, label) if label_key else label
            painter.drawText(
                QRectF(row_rect.left() + 14.0, row_rect.top(), label_w, row_rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label_display,
            )
            # value (old → new)
            painter.setPen(QPen(muted))
            val_x = row_rect.left() + 14.0 + label_w + 4.0
            val_rect = QRectF(val_x, row_rect.top(), max(0.0, row_rect.right() - val_x), row_rect.height())
            value_text = old_v if old_v == new_v else f"{old_v} → {new_v}"
            fm = QFontMetricsF(font)
            painter.drawText(
                val_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                fm.elidedText(value_text, Qt.TextElideMode.ElideRight, val_rect.width()),
            )
            y += _SPC_ROW_H

        # ---- footer summary ----
        if report is not None:
            summary = str(getattr(report, "summary", "") or "")
            overall = str(getattr(report, "overall_status", "") or "")
        elif self._error:
            summary, overall = self._error, "error"
        else:
            summary = tr(
                "node.export.compat.no_connection",
                "No Start Point connected — connect train_pipe to populate",
            )
            overall = "none"
        if overall == "ok":
            sc = ok
        elif overall == "warning":
            sc = warn
        elif overall == "error":
            sc = err
        else:
            sc = muted
        painter.setPen(QPen(sc))
        painter.drawText(
            QRectF(rect.left(), y, rect.width(), _SPC_FOOTER_H),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            summary,
        )


# =============================================================================
# ReviewBackendPickerRow — DEMO §3 review_backend dropdown
# =============================================================================

_REVIEW_BACKEND_FALLBACK = ("mujoco", "isaac_sim", "newton")
_REVIEW_BACKEND_FALLBACK_META = {
    "mujoco": {
        "title": "MuJoCo",
        "desc": "Fast local viewer (MJCF). Default.",
    },
    "isaac_sim": {
        "title": "Isaac Sim",
        "desc": "Subprocess viewer matching the training renderer 1:1.",
    },
    "newton": {
        "title": "Newton",
        "desc": "Reserved for a future UnitPort release — not yet available.",
    },
}


class ReviewBackendPickerRow(_ChoicePickerRow):
    """``widget="review_backend_picker"`` — single-select rich choice picker.

    DEMO 对应 ``_make_review_backend_picker`` (training_node_items.py
    :9202-9269). Choices come from ``registers.review_backends.list_review_
    backends()`` when available; falls back to a static 3-entry list so the
    picker still works before Stage C registers are wired up. Unavailable
    backends are tagged ``(not yet available)`` in their description.
    """

    LEADING_MODE = "checkbox"
    _OPEN_ON_RELEASE = True

    def _resolve_backend_choices(self) -> tuple:
        try:
            from registers.review_backends import list_review_backends
            backends = list_review_backends()
            ids = [str(getattr(b, "backend_id", "")) for b in backends]
            meta = {}
            for b in backends:
                bid = str(getattr(b, "backend_id", ""))
                if not bid:
                    continue
                desc = str(getattr(b, "description", ""))
                if not bool(getattr(b, "available", True)):
                    desc = (desc + "  (not yet available)").strip()
                meta[bid] = {
                    "title": str(getattr(b, "display_name", bid)) or bid,
                    "desc": desc,
                }
            if ids:
                return ids, meta
        except Exception:
            pass
        # Fallback when Stage C registers not yet ported.
        return list(_REVIEW_BACKEND_FALLBACK), dict(_REVIEW_BACKEND_FALLBACK_META)

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        choices, meta = self._resolve_backend_choices()
        if not choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            meta_map=meta,
            on_commit=self.set_value,
        )
        return True


# =============================================================================
# ReviewScenePickerRow — DEMO §3 review_scene_id dropdown
# =============================================================================

_REVIEW_SCENE_FALLBACK = ("flat_ground", "rough", "stairs")


class ReviewScenePickerRow(_ChoicePickerRow):
    """``widget="review_scene_picker"`` — backend-filtered scene dropdown.

    DEMO 对应 ``_make_review_scene_picker`` (training_node_items.py
    :9275-9367). Choices are filtered by the sibling ``review_backend`` param
    and by the robot family detected from a Robot node on the canvas. Falls
    back to a static 3-entry list before Stage C ``scene_registry`` is wired.
    Re-resolves on every popup open so backend switches refresh immediately.
    """

    LEADING_MODE = "checkbox"
    _OPEN_ON_RELEASE = True

    def _detect_family(self) -> Optional[str]:
        scene = self.scene()
        if scene is None:
            return None
        try:
            for it in scene.items():
                manifest = getattr(it, "manifest", None)
                params = getattr(it, "params", None)
                if manifest is None or params is None:
                    continue
                mid = str(getattr(manifest, "id", "") or "")
                if mid != "robot":
                    continue
                asset_id = str(params.get("asset_id", "") or "").strip()
                if not asset_id:
                    continue
                # Best-effort: try registers.robots.resolve to get family.
                try:
                    from registers import robots as _robots_registry
                    rob = _robots_registry.get_robot(asset_id)
                    if rob is None:
                        rob = _robots_registry.get_robot(
                            _robots_registry.resolve_id(asset_id) or ""
                        )
                    if rob is not None:
                        fams = getattr(rob, "families", None) or []
                        if fams:
                            return str(fams[0])
                except Exception:
                    pass
                return None
        except Exception:
            return None
        return None

    def _resolve_scene_choices(self) -> tuple:
        backend = str(self._owner.params.get("review_backend", "mujoco") or "mujoco")
        family = self._detect_family()
        try:
            from application.training.scene_registry import list_review_scenes
            scenes = list_review_scenes(review_backend=backend, family=family)
            ids = [str(getattr(s, "scene_id", "")) for s in scenes]
            meta = {}
            for s in scenes:
                sid = str(getattr(s, "scene_id", ""))
                if not sid:
                    continue
                meta[sid] = {
                    "title": str(getattr(s, "name", sid)) or sid,
                    "desc": str(getattr(s, "description", "")),
                }
            if ids:
                return ids, meta
        except Exception:
            pass
        return list(_REVIEW_SCENE_FALLBACK), {
            sid: {"title": sid, "desc": ""} for sid in _REVIEW_SCENE_FALLBACK
        }

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        choices, meta = self._resolve_scene_choices()
        if not choices:
            return False
        open_choice_popup(
            view=_view_of(event),
            row=self,
            choices=choices,
            current=self._value,
            multi=False,
            leading_mode=self.LEADING_MODE,
            meta_map=meta,
            on_commit=self.set_value,
        )
        return True


# =============================================================================
# ReviewLaunchButtonRow — DEMO §3 "▶ Launch Review" button
# =============================================================================

class ReviewLaunchButtonRow(ParamRow):
    """``widget="review_launch_button"`` — full-width teal launch button.

    DEMO 对应 ``_make_review_launch_button`` (training_node_items.py
    :9373-9501). Click pipeline:
        1. Walk scene → view → window for ``_run_safety_review(blocking=True)``
        2. If SafetyReview passes (or wiring not yet there) → emit
           ``scene.review_launch_requested(payload)`` so the workspace can
           spawn the review subprocess.
    Defensive: missing ``_run_safety_review`` / missing signal both degrade
    to a ``log_warning``; the button never raises into Qt.
    """

    _OPEN_ON_RELEASE = True

    def _btn_rect(self) -> QRectF:
        # Full-width — span from KEY_PAD_LEFT all the way to the right edge.
        # The base ParamRow paint still draws the key label in the left half;
        # we paint the button on the right half (SEP_X..width-pad) so the row
        # reads as "review_launch  [▶  Launch Review]".
        x = SEP_X + _RLB_FRAME_PAD
        return QRectF(
            x,
            _RLB_FRAME_PAD,
            max(0.0, self._width - x - _RLB_FRAME_PAD - VALUE_PAD_RIGHT + 4),
            self._height - _RLB_FRAME_PAD * 2,
        )

    def _interactive_regions(self) -> List[QRectF]:
        return [self._btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        btn = self._btn_rect()
        idle = QColor(Config.get_color("canvas_node_review_launch_bg", "#00695C"))
        hover = QColor(
            Config.get_color("canvas_node_review_launch_hover_bg", "#00897B")
        )
        text_color = QColor(
            Config.get_color("main_t2", "#FFFFFF")
        )
        bg = hover if self._is_hovering(btn) else idle
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(btn, 4, 4)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        painter.drawText(
            btn,
            int(Qt.AlignmentFlag.AlignCenter),
            "▶  " + tr("canvas.review_launch.button", "Launch Review"),
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        from unitport_sdk import log_warning

        # 1. Resolve owning window.
        scene = self.scene()
        window = None
        if scene is not None:
            try:
                views = scene.views()
                if views:
                    window = views[0].window()
            except Exception:
                window = None

        # 2. SafetyReview gate (optional — degrades to "skip" when missing).
        ok = True
        runner = getattr(window, "_run_safety_review", None) if window else None
        if callable(runner):
            try:
                ok = bool(runner(blocking=True))
            except Exception as e:
                log_warning(f"review_launch: safety review raised: {e}")
                ok = False
        if not ok:
            log_warning("review_launch: SafetyReview reported errors — aborted")
            return True

        # 3. Build payload + emit scene signal.
        params = self._owner.params
        payload = {
            "backend": str(params.get("review_backend", "mujoco") or "mujoco"),
            "scene_id": str(params.get("review_scene_id", "flat_ground") or "flat_ground"),
            "bundle_name": str(params.get("bundle_name", "") or ""),
            "version": str(params.get("version", "v1") or "v1"),
            "overwrite": bool(params.get("overwrite", True)),
        }
        sig = getattr(scene, "review_launch_requested", None) if scene else None
        if sig is None:
            log_warning(
                "review_launch: scene.review_launch_requested signal not yet wired "
                "(Stage D pending) — payload would be %r" % (payload,)
            )
            return True
        try:
            sig.emit(payload)
        except Exception as e:
            log_warning(f"review_launch: signal emit failed: {e}")
        return True


# =============================================================================
# JointPoseTableRow / ActorReviewPoseButtonRow —— ActorSettingNode 专用
# =============================================================================
#
# ActorSettingNode.init_joint_angles 旧 widget="code" → 改 widget="joint_pose_table"。
# 单击打开 ``JointPoseEditorDialog``，每个 IR role 一行 slider+spinbox，slider
# 物理量程取自上游 RobotNode → MJCF ``jnt_range``（``jnt_limited=False`` 回落 ±π）。
#
# 配套 widget="actor_review_pose_button" 全宽 teal 按钮，复用
# ``canvas_node_review_launch_*`` theme slot —— 视觉与 Export 节点的 Review
# Launch 按钮一致，行为不同：这里走 RobotReviewTask（纯加载 MJCF + 注入 pose），
# 不走 Export 的 SafetyReview / training subprocess pipeline。

_JOINT_POSE_ROW_PAD = 2  # outer pad inside the row's full-width value rect


def _upstream_robot_sku(actor_node: "NodeItem") -> str:
    """Walk ``actor_node``'s ``robot_pipe`` input back to the RobotNode and
    resolve its ``asset_id`` to a canonical SKU.

    Raises ``RuntimeError`` with a user-facing description (caught by
    JointPoseTableRow._handle_click and surfaced via QMessageBox) when:
        - actor_node has no ``robot_pipe`` input port
        - that port is not connected to a RobotNode
        - the upstream RobotNode has no ``asset_id`` set
        - ``asset_id`` cannot be resolved through ``registers.robots.resolve_id``

    CLAUDE.md §1.8: previous implementation returned ``None`` on every
    failure path, so the canvas widget would silently fall back to a
    raw JSON editor with no explanation to the user. Replaced with
    descriptive raises.
    """
    in_ports = getattr(actor_node, "_in_ports", None)
    if not in_ports:
        raise RuntimeError(
            "Actor 节点没有 robot_pipe 输入端口 — 这通常表示节点 schema "
            "或图渲染层 bug。"
        )
    target_port = None
    for p in in_ports:
        if getattr(getattr(p, "spec", None), "name", "") == "robot_pipe":
            target_port = p
            break
    if target_port is None:
        raise RuntimeError(
            "Actor 节点缺少 robot_pipe 输入端口。请检查节点 schema 是否"
            "正确声明该端口。"
        )
    connections = list(getattr(target_port, "connections", []) or [])
    if not connections:
        raise RuntimeError(
            "请先把 Robot 节点连接到 Actor Setting 节点的 robot_pipe 端口，"
            "然后再设置初始关节角度。"
        )
    for conn in connections:
        src_port = getattr(conn, "src_port", None)
        if src_port is None:
            continue
        src_node = src_port.parent_node()
        if src_node is None:
            continue
        node_id = str(getattr(src_node.manifest, "id", "") or "")
        if node_id != "robot":
            raise RuntimeError(
                f"Actor 节点的 robot_pipe 端口连到了 {node_id!r} 节点，但应该"
                f"连到 Robot 节点（schema_id='robot'）。"
            )
        params = getattr(src_node, "params", None) or {}
        asset_id = str(params.get("asset_id", "") or "")
        if not asset_id:
            raise RuntimeError(
                "上游 Robot 节点的 asset_id 为空 — 请先在 Robot 节点选择一"
                "个机器人。"
            )
        from registers.robots import resolve_id
        sku = resolve_id(asset_id)
        if sku:
            return sku
        # WHY KEPT (Rule 1.b — cross-format identifier): some Robot
        # nodes may already store the canonical SKU verbatim in
        # asset_id (e.g. from registry editor round-trip); accept that
        # path so RobotNode → SKU resolution survives both storage
        # conventions.
        return asset_id
    raise RuntimeError(
        "Actor 节点的 robot_pipe 端口虽有连接，但所有 connection 的 src_port "
        "都解析失败 — 画布数据可能损坏，请重新连接 Robot 节点。"
    )


def _scene_robot_sku(node: "NodeItem") -> Optional[str]:
    """Resolve the canvas's RobotNode SKU by scanning the scene (single robot
    per canvas). Returns ``None`` when no Robot node / asset_id is present.

    Unlike :func:`_upstream_robot_sku` (which walks a ``robot_pipe`` edge),
    this is for nodes that have no direct robot edge — e.g. the Rewards node,
    whose joint-subset partition picker (缺口③) needs the robot family. Soft
    by design: a missing robot is not an error here (the picker shows a
    "bind a robot" hint), so this returns None rather than raising.
    """
    from .items import NodeItem
    sc = node.scene() if node is not None else None
    if sc is None:
        return None
    for it in sc.items():
        if isinstance(it, NodeItem) and str(getattr(it.manifest, "id", "") or "") == "robot":
            asset_id = str((getattr(it, "params", None) or {}).get("asset_id", "") or "")
            if not asset_id:
                return None
            from registers.robots import resolve_id
            return resolve_id(asset_id) or asset_id
    return None


def _upstream_robot_family(actor_node: "NodeItem") -> str:
    """Resolve the upstream RobotNode's family for ``actor_node``.

    Walks the same ``robot_pipe`` edge as :func:`_upstream_robot_sku`,
    then dereferences the canonical SKU into the first declared family
    via :func:`registers.robots.get_robot_spec`. Used by
    :class:`RegistryPresetPickerRow` to dispatch the family-keyed
    registry lookup (init_pose_presets / gait_presets).

    Raises ``RuntimeError`` with a user-facing description (caught by
    :class:`RegistryPresetPickerRow._handle_click` and surfaced via
    QMessageBox) when:
        - actor_node has no ``robot_pipe`` input port / it is not connected
          (propagated from :func:`_upstream_robot_sku`)
        - the upstream RobotNode's SKU does not resolve
          (propagated from :func:`_upstream_robot_sku`)
        - the canonical entry for the SKU has no ``families`` declared
          (the SKU is registered but family-less, which makes a
          family-keyed preset dropdown undefined; the user is told to
          fill the families field)

    PRINCIPLES R7: the first key is ``families[0]`` (NOT brand / SKU).
    Mirrors :func:`application.training.isaac_lab.env_cfg_compiler.
    _resolve_gait_spec` and
    :func:`application.training.command_schema._build_gait_channels`
    so widget-side and training-side family dispatch agree.
    """
    sku = _upstream_robot_sku(actor_node)  # may raise; same diagnostics
    from registers.robots import get_robot_spec
    spec = get_robot_spec(sku)
    families = list(getattr(spec, "families", []) or [])
    if not families:
        raise RuntimeError(
            f"上游 Robot 节点的 SKU={sku!r} 在 registers 中没有 families 声明 — "
            f"无法按家族过滤 preset。请打开 robots_canonical.json 把该 SKU "
            f"的 families 字段填全。"
        )
    return str(families[0])


def _reconcile_actor_init_joint_angles(actor_node: "NodeItem") -> bool:
    """Reconcile an ActorSetting node's ``init_joint_angles`` against the
    upstream Robot Node's authoritative IR-role set.

    Behaviour (canvas-derived-keys rule):
      - No upstream Robot Node connected (or SKU unresolvable / registry
        missing joint table) → set ``init_joint_angles`` to ``{}``. The
        editor will refuse to open until a Robot Node is connected, and
        any prior canvas keys are dropped because they can no longer
        claim authority.
      - Upstream SKU resolved → keys become exactly the SKU's USD IR-role
        set. Stale keys are dropped, missing keys are filled with 0.0,
        existing keys keep their user-set values.

    Returns True iff the node's stored value mutated (caller may use this
    to skip redundant logs / undo entries).

    Lookup is purely registry-driven (canvas-ir-only rule); no MJCF/USD
    asset loading happens here.
    """
    row = None
    for r in getattr(actor_node, "_param_rows", []) or []:
        spec_key = str(getattr(getattr(r, "spec", None), "key", "") or "")
        if spec_key == "init_joint_angles":
            row = r
            break
    if row is None:
        return False

    existing_raw = (actor_node.params or {}).get("init_joint_angles")
    existing: Dict[str, float] = {}
    if isinstance(existing_raw, dict):
        for k, v in existing_raw.items():
            try:
                existing[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(existing_raw, str) and existing_raw.strip():
        try:
            parsed = json.loads(existing_raw)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    try:
                        existing[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        except (json.JSONDecodeError, ValueError):
            pass

    # Resolve the authoritative IR role set from the upstream Robot Node.
    # If we cannot determine it (no upstream / SKU unknown / registry
    # missing joint table), DO NOT mutate user values. The previous
    # behaviour was "reset to {}", which silently wiped the user's
    # carefully-set init pose whenever a transient registry / pipe
    # state made the lookup throw — see user's "用户设置永远绝对权限"
    # mandate. The editor will refuse to open without an authoritative
    # set; that's enough signal to the user without destroying data.
    try:
        sku = _upstream_robot_sku(actor_node)
        ir_roles, _ranges, _name = _resolve_active_format_joint_ranges(sku)
        authoritative_list = list(ir_roles)
    except RuntimeError as exc:
        log_warning(
            f"[actor_setting] init_joint_angles reconcile aborted: cannot "
            f"resolve authoritative IR role set ({exc}); leaving existing "
            f"user values untouched."
        )
        return False
    if not authoritative_list:
        log_warning(
            "[actor_setting] init_joint_angles reconcile aborted: "
            "upstream Robot Node returned an empty IR role list; "
            "leaving existing user values untouched."
        )
        return False

    # Pull SKU-specific standing-pose blueprint when available; falls back
    # to 0.0 for IR roles the registry doesn't declare a default for.
    # Without this, Spot's knee IR roles (calf_*) got reconciled to 0.0
    # which is outside Spot's [-2.79, -0.25] joint range — the env_cfg
    # joint-range validator then rejected training with an actionable
    # error pointing back here. Per-SKU blueprints land in
    # registers/data/robots_canonical.json under default_init_joint_angles.
    sku_blueprint: Dict[str, float] = {}
    try:
        sku_val = locals().get("sku")
        if sku_val:
            from registers.robots import get_robot_spec
            rs = get_robot_spec(sku_val)
            if rs is not None and rs.default_init_joint_angles:
                sku_blueprint = dict(rs.default_init_joint_angles)
    except Exception:  # noqa: BLE001  WHY KEPT: registry lookup must not block canvas UX; fallback is 0.0
        sku_blueprint = {}

    def _seed_value(role: str) -> float:
        if role in existing:
            return existing[role]
        return float(sku_blueprint.get(role, 0.0))

    reconciled: Dict[str, float] = {
        role: _seed_value(role) for role in authoritative_list
    }
    if reconciled == existing:
        return False

    authoritative_set = set(authoritative_list)
    dropped = sorted(k for k in existing if k not in authoritative_set)
    added = sorted(k for k in authoritative_list if k not in existing)
    seeded_from_blueprint = sorted(
        k for k in added if k in sku_blueprint
    )
    sku_repr = locals().get("sku", "<no upstream RobotNode>")
    log_info(
        f"[actor_setting] reconciled init_joint_angles against upstream "
        f"Robot SKU={sku_repr!r}: dropped={dropped} added={added} "
        f"(seeded from registry blueprint: {seeded_from_blueprint}; "
        f"remaining defaults are 0.0)"
    )
    row.set_value(dict(reconciled))
    return True


def _reconcile_downstream_actor_settings(robot_node: "NodeItem") -> None:
    """Walk ``robot_node``'s ``robot_pipe`` outgoing edges and reconcile
    every connected ActorSetting node.

    Called when:
      - Robot Node's ``asset_id`` parameter changes (NodeItem.on_param_changed)
      - A new edge is connected to / disconnected from an ActorSetting's
        ``robot_pipe`` input (page._connect_raw / _disconnect_edge_by_id_raw)

    Best-effort — never raises; an exception in one downstream's
    reconcile must not block reconciling the others.
    """
    # The actuator caps (effort/velocity) now live on the RobotNode itself,
    # so reconcile them once here (not per-downstream).
    try:
        _reconcile_robot_actuator_params(robot_node)
    except Exception as exc:
        log_warning(
            f"[robot] actuator-caps reconcile failed for {robot_node!r}: {exc}"
        )

    out_ports = getattr(robot_node, "_out_ports", None) or []
    for port in out_ports:
        if getattr(getattr(port, "spec", None), "name", "") != "robot_pipe":
            continue
        for conn in list(getattr(port, "connections", []) or []):
            dst_port = getattr(conn, "dst_port", None)
            if dst_port is None:
                continue
            dst_node = dst_port.parent_node()
            if dst_node is None:
                continue
            if str(getattr(dst_node.manifest, "id", "") or "") != "actor_setting":
                continue
            try:
                _reconcile_actor_init_joint_angles(dst_node)
            except Exception as exc:
                log_warning(
                    f"[actor_setting] init_joint_angles reconcile from "
                    f"upstream Robot Node failed for downstream "
                    f"{dst_node!r}: {exc}"
                )


# RobotNode hidden param keys that mirror RobotSpec.default_actuator_params
# entries. effort_limit / velocity_limit moved from ActorSetting to RobotNode
# in 2026-05 (stiffness/damping were dropped — gains derive from (omega_n,
# zeta), §10). Names match the RobotNode ParamSpec keys exactly.
_ACTUATOR_PARAM_KEYS: tuple = (
    "effort_limit",
    "velocity_limit",
)


def _pd_group_registry_config(sku: str) -> Optional[Dict[str, Any]]:
    """Force-load the model-bound PD-Group config for ``sku`` from the registry.

    The SAME resolver the AUTO button uses
    (``pd_preview.resolve_recommended_actuators``): per-group (omega_n, zeta)
    + actuator caps, sourced from the brand manifest / captured USD drive /
    MJCF. Returns a dict of canvas RobotNode keys, or ``None`` when nothing is
    resolvable (the training-time compiler then fails loud, CLAUDE.md §8).
    """
    try:
        from application.physics.pd_preview import resolve_recommended_actuators
    except Exception:  # noqa: BLE001
        return None
    rec = resolve_recommended_actuators(sku)
    if rec is None:
        return None
    out: Dict[str, Any] = {"pd_param_mode": "natural_freq"}
    groups = getattr(rec, "groups", None) or {}
    if groups:
        out["pd_groups"] = {
            str(gid): {"omega_n": float(wn), "zeta": float(z)}
            for gid, (wn, z) in groups.items()
        }
    if rec.effort_limit is not None:
        out["effort_limit"] = float(rec.effort_limit)
    if rec.velocity_limit is not None:
        out["velocity_limit"] = float(rec.velocity_limit)
    return out if (len(out) > 1) else None


def _reconcile_robot_actuator_params(robot_node: "NodeItem") -> bool:
    """Reconcile the RobotNode's PD-Group config against its bound SKU.

    The PD-Group config (``pd_param_mode`` / ``pd_groups`` (omega_n, zeta) /
    ``effort_limit`` / ``velocity_limit``) is MODEL-BOUND: a 42 kg Spot and a
    15 kg G1 want different ceilings/gains. It is force-loaded from the registry
    when a model is first selected, but a per-SKU in-memory cache slot preserves
    the user's hand edits when they flip the robot model and back
    (user-approved: "每个机型对应一个缓存槽位，切换机型切换对应缓存，画布落盘只
    保存选中的那个机型"). The cache lives on the node instance (NOT in params), so
    only the active SKU's config is serialized to the canvas.

    Called when ``asset_id`` changes (UI edit, undo/redo, programmatic load).
    The previous behaviour force-OVERWROTE effort/velocity from the registry on
    every call — which CLOBBERED a user's saved caps on canvas LOAD (the reported
    data loss) and never preserved per-SKU edits. Returns True iff mutated.

    These are HIDDEN params (edited via the pd_param_table panel), so updates are
    written directly to ``robot_node.params`` + ``on_param_changed`` (the
    hidden-param write path).
    """
    raw_id = str((robot_node.params or {}).get("asset_id", "") or "").strip()
    if not raw_id:
        return False
    try:
        from registers.robots import resolve_to_sku
        from application.physics.actuator_caps import plan_pd_group_transition
        sku = resolve_to_sku(raw_id)
    except Exception:  # noqa: BLE001
        return False
    if not sku:
        return False

    cache = getattr(robot_node, "_pd_group_cache", None)
    if not isinstance(cache, dict):
        cache = {}
    prev_sku = getattr(robot_node, "_pd_group_prev_sku", None)

    # The target family's valid PD-group ids — lets the transition drop a
    # cross-morphology contamination (e.g. a 'finger' override on a quadruped)
    # before it reaches the canvas / PD panel. None = couldn't resolve → no-op.
    valid_group_ids: Optional[set] = None
    try:
        from registers.robots import get_robot
        from registers import pd_groups as _pd_groups
        entry = get_robot(sku) or {}
        fams = list(entry.get("families", []) or [])
        if fams:
            valid_group_ids = set(_pd_groups.list_group_ids(str(fams[0])))
    except Exception:  # noqa: BLE001 — registry hiccup → skip sanitize, not fail
        valid_group_ids = None

    updates, new_cache = plan_pd_group_transition(
        params=robot_node.params or {},
        cache=cache,
        prev_sku=prev_sku,
        new_sku=sku,
        resolve=_pd_group_registry_config,
        valid_group_ids=valid_group_ids,
    )
    # Stash the session-only cache + the SKU the params now describe.
    robot_node._pd_group_cache = new_cache
    robot_node._pd_group_prev_sku = sku

    notify = getattr(robot_node, "on_param_changed", None)
    for key, val in updates.items():
        robot_node.params[key] = val
        if callable(notify):
            notify(key, val)

    if updates:
        log_info(
            f"[robot] reconciled PD-Group config for SKU={sku!r} "
            f"(prev={prev_sku!r}): keys={sorted(updates)}"
        )
    return bool(updates)


def _resolve_active_format_joint_ranges(sku: str):
    """Build (ir_roles_in_order, ranges_by_role, display_name) from the
    registry alone — no MJCF / USD / URDF asset loading.

    Canvas IR-only rule (memory: canvas-ir-only): widgets edit IR Roles,
    not physical joint names; therefore the editor must never reach into
    a specific format's asset file to populate its slider list. The IR
    role set comes from the registry's per-format joint declaration; the
    slider range is a uniform IR-level default. Per-IR-role precision
    ranges (via an ir_canonical.json ``default_range`` field) is a
    larger, separate task and not required to unblock training.

    Active format defaults to ``USD`` — Isaac Lab is RELEASE's sole
    training backend, so the canvas's IR-role coverage requirement
    aligns with the USD joint table. Robots whose USD table is absent
    (legacy / MJCF-only menagerie entries) fall back to MJCF with a
    loud WARN so the user knows the editor is showing a different
    format than the eventual training target.

    Raises (CLAUDE.md §1.8) when the SKU is unknown OR neither USD nor
    MJCF declares any joints — every silent ``return None`` path of the
    previous implementation surfaced as an undiagnosable "raw JSON
    editor pops up instead of the slider table" UX bug.
    """
    from registers.robots import get_robot

    entry = get_robot(sku)
    if not entry:
        raise RuntimeError(
            f"robot sku {sku!r} is not registered — cannot build "
            f"joint-pose editor."
        )
    per_format = entry.get("joints_per_format", {}) or {}
    if not isinstance(per_format, dict):
        per_format = {}

    fmt = "USD"
    joints_block = per_format.get(fmt)
    if not isinstance(joints_block, dict) or not joints_block:
        # WHY KEPT (Rule 1.c — on-disk legacy compat): older robot
        # entries only declared MJCF before Phase 5 added USD support.
        # WARN loudly (not silent) so the user knows the editor is
        # showing a different format than the training target.
        mjcf_block = per_format.get("MJCF")
        if isinstance(mjcf_block, dict) and mjcf_block:
            log_warning(
                f"[joint_pose_table] robot {sku!r} has no USD "
                f"joints_per_format; falling back to MJCF for the editor "
                f"— add a USD table to registers/data/robots_canonical.json "
                f"(or the user overlay) so the editor matches the Isaac "
                f"Lab training target."
            )
            joints_block = mjcf_block
            fmt = "MJCF"
        else:
            raise RuntimeError(
                f"robot {sku!r} has no joints_per_format entries for "
                f"USD or MJCF — registers/data/robots_canonical.json (or "
                f"the user overlay) is missing a joint table. Cannot "
                f"build joint-pose editor."
            )

    ir_roles_in_order: List[str] = []
    seen: set = set()
    for jspec in joints_block.values():
        if not isinstance(jspec, dict):
            continue
        ir = str(jspec.get("ir_role", "") or "")
        if not ir:
            continue
        if ir in seen:
            continue
        ir_roles_in_order.append(ir)
        seen.add(ir)

    if not ir_roles_in_order:
        raise RuntimeError(
            f"robot {sku!r} joints_per_format[{fmt!r}] has zero entries "
            f"with a non-empty ir_role — registry data is broken."
        )

    # Phase 1 IR-only range: uniform [-π, π] (radians). Phase 2 will
    # source precision ranges from ir_canonical.json ``default_range``
    # once the IR catalog grows that field.
    ranges: Dict[str, Optional[tuple]] = {
        ir: (-math.pi, math.pi) for ir in ir_roles_in_order
    }

    display_name = str(entry.get("name", "") or sku)
    return ir_roles_in_order, ranges, display_name


def _submit_actor_pose_review(
    actor_node: "NodeItem",
    joint_pos_by_ir: Dict[str, float],
    *,
    parent_widget: Optional[QWidget] = None,
) -> None:
    """Construct an :class:`InitPoseOverride` from ``actor_node``'s current
    base_pos params + ``joint_pos_by_ir`` and submit a ReviewTask in the
    engine selected on the node (``review_pose_engine``).

    Dispatch:
        * ``review_pose_engine == "mujoco"`` (default) → ``RobotReviewTask``
          (in-process MuJoCo passive viewer)
        * ``review_pose_engine == "isaac_sim"`` → ``IsaacSimPoseReviewTask``
          (out-of-process Isaac Sim Kit subprocess)

    Failures surface as ``QMessageBox.warning`` against ``parent_widget``
    so the user sees the specific reason — CLAUDE.md §1.8 forbids the
    previous "log_warning and silently return" pattern that left the
    user wondering why the viewer never opened.
    """
    from unitport_sdk import get_tasks_manager, log_info

    def _warn(msg: str) -> None:
        QMessageBox.warning(
            parent_widget,
            tr("dialog.review_pose_unavailable",
               default="无法启动 Review Pose"),
            msg,
        )

    try:
        sku = _upstream_robot_sku(actor_node)
    except RuntimeError as exc:
        _warn(str(exc))
        return

    from application.service.robot_init_poses import InitPoseOverride

    params = getattr(actor_node, "params", None) or {}
    engine = str(params.get("review_pose_engine") or "mujoco").strip()
    try:
        bx = float(params.get("init_pos_x", 0.0) or 0.0)
        by = float(params.get("init_pos_y", 0.0) or 0.0)
        bz = float(params.get("init_pos_z", 0.4) or 0.4)
    except (TypeError, ValueError) as exc:
        _warn(
            f"Actor Setting 的 init_pos_x/y/z 数值无效：{exc}。请检查后重试。"
        )
        return

    cleaned: Dict[str, float] = {}
    for k, v in (joint_pos_by_ir or {}).items():
        try:
            cleaned[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    override = InitPoseOverride(
        base_pos=(bx, by, bz),
        joint_pos_by_ir=cleaned,
    )

    if engine == "isaac_sim":
        try:
            from registers import backends as _backends
        except Exception as exc:
            _warn(
                f"无法访问 backends registry，IsaacSim 不可用：{exc}"
            )
            return
        if not _backends.is_available("isaac_lab"):
            _warn(
                "Isaac Lab 未安装或未在 Engine Settings 中注册路径。\n"
                "请打开 Engine Settings 注册 Isaac Lab 安装目录，或在 "
                "Actor Setting 节点上将 review_pose_engine 切回 mujoco。"
            )
            return
        try:
            from application.service.runtime.simulation.isaac_sim import (
                IsaacSimPoseReviewTask,
            )
        except Exception as exc:
            _warn(f"IsaacSimPoseReviewTask 导入失败：{exc}")
            return
        task = IsaacSimPoseReviewTask(
            sku=sku,
            scene_id="flat_ground",
            init_pose_override=override,
        )
        tid = get_tasks_manager().submit(task)
        log_info(
            f"[actor_setting] Review Pose: submitted IsaacSimPoseReviewTask "
            f"id={tid} sku={sku} joints={len(cleaned)}"
        )
        return

    if engine != "mujoco":
        _warn(
            f"未知的 review_pose_engine 值：{engine!r}。"
            f"支持的取值为 mujoco / isaac_sim。"
        )
        return

    from application.service.runtime.simulation.mujoco.robot_review_session import (
        RobotReviewTask,
    )
    task = RobotReviewTask(
        sku,
        "flat_ground",
        live_physics=False,
        init_pose_override=override,
    )
    tid = get_tasks_manager().submit(task)
    log_info(
        f"[actor_setting] Review Pose: submitted RobotReviewTask "
        f"id={tid} sku={sku} joints={len(cleaned)}"
    )


def _resolve_robot_ir_role_choices(
    actor_node: "NodeItem", source: str
) -> List[str]:
    """Pull the upstream Robot Node's IR-role list for picker widgets.

    ``source`` ∈ {"joint", "body"}:
        * "joint" → ``joints_per_format[fmt]`` (USD with MJCF legacy fallback)
        * "body"  → ``bodies_per_format[fmt]`` (USD with MJCF legacy fallback)

    Walks ``actor_node.robot_pipe`` back to the RobotNode the same way
    :func:`_upstream_robot_sku` does. Returns the IR roles in declaration
    order, deduplicated.

    Raises ``RuntimeError`` with a user-facing message — surfaced via
    QMessageBox by callers so the user always knows why the picker came
    up empty (no upstream Robot, asset_id unset, registry missing
    table). CLAUDE.md §1.8 — no silent empty-list fallback.
    """
    sku = _upstream_robot_sku(actor_node)
    from registers.robots import get_robot

    entry = get_robot(sku)
    if not entry:
        raise RuntimeError(
            f"robot sku {sku!r} is not registered — cannot resolve "
            f"IR-role picker choices."
        )
    block_key = "joints_per_format" if source == "joint" else "bodies_per_format"
    per_format = entry.get(block_key, {}) or {}
    if not isinstance(per_format, dict):
        per_format = {}
    fmt = "USD"
    spec_block = per_format.get(fmt)
    if not isinstance(spec_block, dict) or not spec_block:
        # WHY KEPT (CLAUDE.md §1.8 (c) — on-disk legacy compat): older
        # robots only declare MJCF. WARN loudly so the user knows the
        # picker is showing a different format than the training target.
        mjcf_block = per_format.get("MJCF")
        if isinstance(mjcf_block, dict) and mjcf_block:
            log_warning(
                f"[ir_role_picker] robot {sku!r} has no USD {block_key}; "
                f"falling back to MJCF — add a USD table to "
                f"registers/data/robots_canonical.json (or the user "
                f"overlay) so the picker matches Isaac Lab."
            )
            spec_block = mjcf_block
            fmt = "MJCF"
        else:
            raise RuntimeError(
                f"robot {sku!r} has no {block_key} entries for USD or "
                f"MJCF — cannot populate IR-role picker."
            )
    out: List[str] = []
    seen: set = set()
    for spec in spec_block.values():
        if not isinstance(spec, dict):
            continue
        ir = str(spec.get("ir_role", "") or "")
        if not ir or ir in seen:
            continue
        out.append(ir)
        seen.add(ir)
    if not out:
        raise RuntimeError(
            f"robot {sku!r} {block_key}[{fmt!r}] has zero entries with "
            f"a non-empty ir_role — registry data is broken."
        )
    return out


class IRRolePickerRow(_PickerRowMixin, ParamRow):
    """``widget="ir_joint_picker" / "ir_body_picker"`` — multi-select dropdown
    backed by the upstream RobotNode's IR-role catalog.

    Used by ActorSettingNode for Contact Bodies (body IR roles, default
    empty) and Action Joints (joint IR roles, default = all selected).

    Meta keys:
        * ``source`` ∈ {"joint", "body"} — which block to read from the
          upstream Robot Node's registry entry. Default: "joint".
        * ``default_select_all`` (bool) — if True, an empty stored value
          (``[]``) renders as "全选 (N)" and the popup pre-checks every
          role; the user deselects to exclude. Default: False.

    Persistence: stores a JSON list of selected IR roles. When
    ``default_select_all=True`` the popup carries an explicit ``"All"`` menu
    item whose id is the backend ``.*`` regex, so "every joint" is a VISIBLE,
    SELECTABLE choice and the stored value (``[".*"]``) always corresponds to
    a menu entry the user can see and pick — never an invisible sentinel
    (frontend⇄backend consistency). A legacy empty list ``[]`` is still read as
    "All" for backward compatibility and auto-migrates to ``[".*"]`` on the
    next edit. ``.*`` and ``[]`` are both passed through by the env_cfg compiler
    as "all joints", and ``spec_validator._check_action_joints`` treats ``.*``
    as a regex (no IR-membership check).
    """

    _OPEN_ON_RELEASE = True

    # Backend "all joints" form: the Isaac Lab JointPositionActionCfg regex.
    # Surfaced in the popup as an explicit "All" item (id == this string) so
    # the stored value is always a displayable, selectable menu choice.
    _ALL_SENTINEL = ".*"

    def _source(self) -> str:
        meta = self.spec.meta or {}
        return "body" if str(meta.get("source", "joint")).startswith("body") else "joint"

    def _default_select_all(self) -> bool:
        meta = self.spec.meta or {}
        return bool(meta.get("default_select_all", False))

    def _current_list(self) -> List[str]:
        v = self._value
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    def _summary_text(self) -> str:  # type: ignore[override]
        cur = self._current_list()
        if self._default_select_all() and (not cur or self._ALL_SENTINEL in cur):
            # Explicit "All" (the .* regex) — or a legacy empty value the
            # backend treats as all. Both render as the single "All" menu item
            # so the inline summary matches what the picker shows.
            return tr("canvas.row.ir_picker_all", "All")
        if not cur:
            return tr("canvas.row.empty", "—")
        return tr("canvas.row.selected", "{n} selected").replace("{n}", str(len(cur)))

    def _handle_double_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._open(event)

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        return self._open(event)

    def _open(self, event: QGraphicsSceneMouseEvent) -> bool:
        view = _view_of(event)
        parent_widget = view.window() if view is not None else None
        try:
            choices = _resolve_robot_ir_role_choices(self._owner, self._source())
        except RuntimeError as exc:
            QMessageBox.warning(
                parent_widget,
                tr(
                    "dialog.ir_role_picker_unavailable",
                    default="无法打开 IR 角色选单",
                ),
                str(exc),
            )
            return True

        supports_all = self._default_select_all()
        if supports_all:
            # Prepend an explicit "All" menu item whose id IS the backend ".*"
            # regex, so "every joint" is a visible, checkable choice and the
            # stored value (".*") is always a selectable menu entry — no
            # invisible sentinel (frontend⇄backend consistency).
            choices = [self._ALL_SENTINEL] + [c for c in choices if c != self._ALL_SENTINEL]
        choices_set = set(choices)

        cur = self._current_list()
        # Map the stored value onto the menu. ".*" (or a legacy empty value the
        # backend treats as all) → the "All" item is checked. Otherwise check
        # exactly the specific roles, dropping any stale entries no longer in
        # the upstream catalog (keys auto-reconcile against upstream).
        if supports_all and (not cur or self._ALL_SENTINEL in cur):
            initial = [self._ALL_SENTINEL]
        else:
            initial = [c for c in cur if c in choices_set]

        def _label(c: Any) -> str:
            if supports_all and str(c) == self._ALL_SENTINEL:
                return tr("canvas.row.ir_picker_all", "All")
            return str(c)

        def _commit(sel: list) -> None:
            sel = [str(c) for c in sel]
            # "All" is exclusive and canonical: selecting it stores the explicit
            # ".*" (every joint), ignoring any individually-checked roles.
            if supports_all and self._ALL_SENTINEL in sel:
                self.set_value([self._ALL_SENTINEL])
                return
            cleaned = [c for c in sel if c in choices_set and c != self._ALL_SENTINEL]
            self.set_value(cleaned)

        open_choice_popup(
            view=view,
            row=self,
            choices=choices,
            current=initial,
            multi=True,
            leading_mode="checkbox",
            on_commit=_commit,
            label_resolver=_label,
        )
        return True


class JointPoseTableRow(CodeRow):
    """``widget="joint_pose_table"`` — IR-role keyed slider table editor.

    Inline paint inherits CodeRow (single-line JSON preview); click pops the
    :class:`JointPoseEditorDialog`. The slider set is built purely from the
    registry's IR-role declarations (memory: canvas-ir-only); we never
    reach into a specific format's asset file from the canvas layer.

    Failure (no upstream RobotNode, SKU unknown, registry missing joint
    table) surfaces as a ``QMessageBox.warning`` with the specific reason
    — CLAUDE.md §1.8 forbids the previous silent "switch to raw JSON
    editor" fallback, which made the editor pop up the wrong UI without
    telling the user why.
    """

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        view = _view_of(event)
        parent_widget = view.window() if view is not None else None
        try:
            sku = _upstream_robot_sku(self._owner)
            ir_roles_in_order, ranges, display_name = (
                _resolve_active_format_joint_ranges(sku)
            )
        except RuntimeError as exc:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.joint_pose_editor_unavailable",
                   default="无法打开关节角度编辑器"),
                str(exc),
            )
            return True

        # Reconcile canvas value against upstream Robot Node before
        # populating the dialog. The reconcile may have already happened
        # via NodeItem.on_param_changed or _connect_raw hooks (so this
        # call typically is a no-op), but staying defensive here ensures
        # the dialog never displays stale keys even if a hook is missed
        # (e.g. canvas loaded from disk and not yet routed through hook).
        _reconcile_actor_init_joint_angles(self._owner)

        # Now self._value reflects the reconciled state. CRITICAL: the
        # stored param may be EITHER a dict (after a reconcile / dialog
        # round-trip) OR a JSON string (fresh from disk load — the ParamSpec
        # type is "json" with default "{}"). The old code only handled the
        # dict case and silently dropped JSON-string user values, which
        # surfaced as "I set 0/0.9/-1.8 in the inline view but every
        # slider opens at 0.0" — user values absolutely preserved is the
        # contract (forbid silent overwrite / drop).
        reconciled: Dict[str, float] = {}
        v = self._value
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    v = parsed
            except (json.JSONDecodeError, ValueError):
                v = {}
        if isinstance(v, dict):
            for k, val in v.items():
                try:
                    reconciled[str(k)] = float(val)
                except (TypeError, ValueError):
                    continue

        owner = self._owner
        review_parent = parent_widget

        def _on_review(snapshot: Dict[str, float]) -> None:
            _submit_actor_pose_review(
                owner, snapshot, parent_widget=review_parent
            )

        from .joint_pose_dialog import JointPoseEditorDialog

        dialog = JointPoseEditorDialog(
            sku=sku,
            sku_display_name=display_name,
            ir_roles_in_order=ir_roles_in_order,
            joint_ranges=ranges,
            initial_joints=reconciled,
            on_review=_on_review,
            parent=parent_widget,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Final write: filter dialog result through the authoritative
            # set one more time so a future dialog bug that sneaks extra
            # keys in cannot corrupt the canvas.
            result = {
                role: float(dialog.result_joint_pos_by_ir.get(role, 0.0))
                for role in ir_roles_in_order
            }
            self.set_value(result)
        return True


class PDParamTableRow(CodeRow):
    """``widget="pd_param_table"`` — per-joint PD / actuator panel (RobotNode).

    Inline paint inherits CodeRow (single-line JSON preview of ``pd_groups``);
    click opens :class:`PDParamTableDialog`, which shows the live per-joint
    kp/kd derived from (omega_n, zeta) + the MJCF mass matrix (CLAUDE.md §10).
    Editing (omega_n, zeta) writes back into ``pd_groups`` (this bound param);
    ``effort_limit`` / ``velocity_limit`` are written to the RobotNode's hidden
    sibling params (no ParamRow → direct ``params`` write + on_param_changed).

    Failure (no bound SKU, missing MJCF, solver error) surfaces as a
    ``QMessageBox.warning`` — never a silent fallback (§1.8).
    """

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:  # type: ignore[override]
        view = _view_of(event)
        parent_widget = view.window() if view is not None else None

        from registers.robots import resolve_to_sku, get_robot

        raw_id = str((self._owner.params or {}).get("asset_id", "") or "").strip()
        sku = resolve_to_sku(raw_id) if raw_id else ""
        if not sku:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.pd_table_unavailable", default="Cannot open PD panel"),
                tr("dialog.pd_table_no_sku",
                   default="The Robot Node has no valid robot bound (asset_id). "
                           "Select a robot in the Robot Asset card first."),
            )
            return True

        # Parse current pd_groups overrides (may be a JSON string from disk or
        # a dict after a prior dialog round-trip).
        overrides: Dict[str, Dict[str, float]] = {}
        v = self._value
        if isinstance(v, str) and v.strip():
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    v = parsed
            except (json.JSONDecodeError, ValueError):
                v = {}
        if isinstance(v, dict):
            for gid, vals in v.items():
                if isinstance(vals, dict):
                    overrides[str(gid)] = dict(vals)

        # Hidden sibling caps (default 30.0 mirrors the RobotNode ParamSpec).
        params = self._owner.params or {}

        def _cap(key: str) -> float:
            try:
                return float(params.get(key, 30.0) or 30.0)
            except (TypeError, ValueError):
                return 30.0

        effort = _cap("effort_limit")
        velocity = _cap("velocity_limit")

        from application.physics.pd_preview import (
            compute_pd_preview,
            PDPreviewError,
        )
        from PyQt6.QtWidgets import QApplication

        preview = None
        err: Optional[Exception] = None
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            preview = compute_pd_preview(
                sku,
                pd_groups_overrides=overrides,
                effort_limit=effort,
                velocity_limit=velocity,
            )
        except PDPreviewError as exc:
            err = exc
        finally:
            QApplication.restoreOverrideCursor()
        if err is not None or preview is None:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.pd_table_unavailable", default="Cannot open PD panel"),
                str(err) if err is not None else "unknown error",
            )
            return True

        entry = get_robot(sku) or {}
        display_name = str(entry.get("name", "") or sku)

        from .pd_param_dialog import PDParamTableDialog

        dialog = PDParamTableDialog(
            sku=sku,
            sku_display_name=display_name,
            preview=preview,
            initial_overrides=overrides,
            parent=parent_widget,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # pd_groups is this row's bound param.
            self.set_value(dict(dialog.result_pd_groups))
            # effort/velocity → hidden sibling params on the RobotNode.
            self._write_sibling("effort_limit", float(dialog.result_effort_limit))
            self._write_sibling("velocity_limit", float(dialog.result_velocity_limit))
        return True

    def _write_sibling(self, key: str, value: float) -> None:
        """Write a sibling param on the owner node + notify.

        effort_limit / velocity_limit are hidden params (no ParamRow), so they
        take the direct-write path (param_rows StartPointRow precedent). If a
        row ever exists for the key, route through it instead.
        """
        rows_by_key = {
            str(getattr(getattr(r, "spec", None), "key", "") or ""): r
            for r in (getattr(self._owner, "_param_rows", []) or [])
        }
        sib = rows_by_key.get(key)
        if sib is not None:
            sib.set_value(value)
            return
        if self._owner.params.get(key) == value:
            return
        self._owner.params[key] = value
        notify = getattr(self._owner, "on_param_changed", None)
        if callable(notify):
            notify(key, value)


class ActorReviewPoseButtonRow(ParamRow):
    """``widget="actor_review_pose_button"`` — full-width teal Review Pose.

    Visual identical to :class:`ReviewLaunchButtonRow` (shares theme slots).
    Click submits a ``RobotReviewTask`` constructed from the owner's current
    ``init_pos_{x,y,z}`` + ``init_joint_angles`` params, against the upstream
    RobotNode's resolved SKU. SKU unresolved → log_warning, no Qt raise.
    """

    _OPEN_ON_RELEASE = True

    def _btn_rect(self) -> QRectF:
        x = SEP_X + _JOINT_POSE_ROW_PAD
        return QRectF(
            x,
            _JOINT_POSE_ROW_PAD,
            max(
                0.0,
                self._width
                - x
                - _JOINT_POSE_ROW_PAD
                - VALUE_PAD_RIGHT
                + 4,
            ),
            self._height - _JOINT_POSE_ROW_PAD * 2,
        )

    def _interactive_regions(self) -> List[QRectF]:
        return [self._btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        btn = self._btn_rect()
        idle = QColor(
            Config.get_color("canvas_node_review_launch_bg", "#00695C")
        )
        hover = QColor(
            Config.get_color("canvas_node_review_launch_hover_bg", "#00897B")
        )
        text_color = QColor(
            Config.get_color("main_t2", "#FFFFFF")
        )
        bg = hover if self._is_hovering(btn) else idle
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(btn, 4, 4)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        painter.drawText(
            btn,
            int(Qt.AlignmentFlag.AlignCenter),
            "▶  " + tr(
                "canvas.actor_setting.review_pose.button",
                "Review Pose",
            ),
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        params = getattr(self._owner, "params", None) or {}
        raw = params.get("init_joint_angles")
        joints: Dict[str, float] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    joints[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    try:
                        joints[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        view = _view_of(event)
        parent_widget = view.window() if view is not None else None
        _submit_actor_pose_review(
            self._owner, joints, parent_widget=parent_widget
        )
        return True


# =============================================================================
# RegistryPresetPickerRow —— canvas widget that filters presets by the
# upstream RobotNode's family from a family-keyed registry, then writes
# the selected preset name into the bound param. Optionally cascades
# sibling fields (init pose preset selection also fills init_pos_{x,y,z}
# + init_joint_angles on the same ActorSetting node).
# =============================================================================
#
# Adapter dispatch (PRINCIPLES R4 / R1):
#   - The widget itself carries no ``if registry == ...`` /
#     ``if family == ...`` branch. It looks up the bound registry
#     adapter from ``meta.registry`` (a single dispatch keyed by the
#     registry id string) and calls the adapter's two-method contract:
#       * ``list_names(family) -> list[str]``: choices for the popup
#       * ``on_select(owner, name, family) -> None``: side-effects
#         (sibling field population for init pose; no-op for gait)
#   - Both adapters route their data through a family-keyed registry
#     API (init_pose_presets.list_presets /
#     gait_presets.default_presets_for_family) -- no literal family
#     compare anywhere on the dispatch path.


class _PresetAdapter:
    """Two-method contract for a family-keyed preset registry.

    Concrete subclasses know which registry to query and what
    side-effects to perform on selection. The widget dispatches by
    ``meta.registry`` (a string id) through
    :data:`_PRESET_ADAPTERS`.
    """

    def list_names(self, family: str) -> List[str]:
        """Return preset names available for ``family`` in display order."""
        raise NotImplementedError

    def on_select(
        self, owner: "NodeItem", preset_name: str, family: str,
    ) -> None:  # default: no sibling side-effect
        """Optional sibling-param cascade fired AFTER the bound param
        was set to ``preset_name``. Default is no-op (the bound param
        already carries the preset name; sibling fields are only
        populated when the registry semantically owns more than one
        field on the node)."""


class _InitPosePresetAdapter(_PresetAdapter):
    """Adapter for the ``init_pose_presets`` family-keyed registry.

    list: family-default preset names from the registry (no SKU
    filtering -- the dropdown shows every name the family declares;
    selecting one whose SKU has no override fails loud at apply).

    on_select: looks up the three-layer-merged preset via
    :class:`InitPoseService` (family-default <- SKU-override <- user)
    and cascades the selected pose into the sibling fields
    ``init_pos_{x,y,z}`` + ``init_joint_angles`` via
    :func:`_write_sibling_param` (which routes through the sibling
    row's ``set_value`` to trigger the standard canvas reactive update
    -- repaint + on_param_changed fan-out + undo stack). When the SKU
    has no SKU-override for the selected name, ``joint_pos_by_ir`` is
    empty and we raise rather than write zeros (§8: init pose joint
    angles are per-robot physical quantities; the registry
    intentionally has nothing to fall back to).
    """

    def list_names(self, family: str) -> List[str]:
        from registers import init_pose_presets
        return [p.name for p in init_pose_presets.list_presets(family)]

    def on_select(
        self, owner: "NodeItem", preset_name: str, family: str,
    ) -> None:
        sku = _upstream_robot_sku(owner)  # may raise; same diagnostics
        from application.service.robot_init_poses import (
            get_init_pose_service,
        )
        service = get_init_pose_service()
        preset = None
        for p in service.list_presets(sku):
            if p.name == preset_name:
                preset = p
                break
        if preset is None:
            raise RuntimeError(
                f"Preset {preset_name!r} 在 SKU={sku!r} 的合并 preset 列表里"
                f"找不到（已查询 family-default + SKU-override + user 三层）。"
                f"请确认 init_pose_presets / robots_canonical / user "
                f"preset state 任一层已声明该 preset。"
            )
        if not preset.joint_pos_by_ir:
            raise RuntimeError(
                f"Preset {preset_name!r} 在家族 {family!r} 注册表中只有 "
                f"base_pos 没有关节角；该 SKU={sku!r} 在 robots_canonical "
                f"中也没有对应的 SKU-override 关节角条目。init pose 关节角"
                f"是 per-robot 物理量，无法静默填空 — 请在 robots_canonical "
                f"中补 SKU 的 init_pose_presets 条目，或在 User Preset 中"
                f"保存一份关节角。"
            )
        bx, by, bz = preset.base_pos
        _write_sibling_param(owner, "init_pos_x", float(bx))
        _write_sibling_param(owner, "init_pos_y", float(by))
        _write_sibling_param(owner, "init_pos_z", float(bz))
        _write_sibling_param(
            owner, "init_joint_angles",
            {str(k): float(v) for k, v in preset.joint_pos_by_ir.items()},
        )


class _GaitPresetAdapter(_PresetAdapter):
    """Adapter for the bundled gait presets keyed by family.

    list: preset names from
    :func:`application.training.isaac_lab.gait_presets.default_presets_for_family`.
    Aliased families (humanoid -> biped) resolve through the same
    helper so the dropdown shows the biped names for a humanoid SKU.

    on_select: no sibling side-effect. The bound param
    ``gait_presets`` carries the selected name; the canvas node's
    other gait fields (gait_enabled / gait_frequency_range / etc) stay
    user-controlled. Gait selection writes only the name -- the
    canvas-level explicit range fields stay authoritative.
    """

    def list_names(self, family: str) -> List[str]:
        from application.training.isaac_lab.gait_presets import (
            default_presets_for_family,
        )
        return [p.name for p in default_presets_for_family(family)]


_PRESET_ADAPTERS: Dict[str, _PresetAdapter] = {
    "init_pose_presets": _InitPosePresetAdapter(),
    "gait_presets": _GaitPresetAdapter(),
}


def _write_sibling_param(
    owner: "NodeItem", key: str, value: Any,
) -> None:
    """Write ``value`` to ``owner.params[key]`` and propagate the
    standard canvas reactive update.

    Mirrors :meth:`PDParamTableRow._write_sibling`: prefers routing
    through the sibling ``ParamRow.set_value`` (which fires repaint +
    on_param_changed + scene fan-out + undo push) when a row exists for
    the key; falls back to direct dict mutation + ``on_param_changed``
    for hidden params (no ParamRow).
    """
    rows_by_key = {
        str(getattr(getattr(r, "spec", None), "key", "") or ""): r
        for r in (getattr(owner, "_param_rows", []) or [])
    }
    sib = rows_by_key.get(key)
    if sib is not None:
        sib.set_value(value)
        return
    if owner.params.get(key) == value:
        return
    owner.params[key] = value
    notify = getattr(owner, "on_param_changed", None)
    if callable(notify):
        notify(key, value)


class RegistryPresetPickerRow(ParamRow):
    """``widget="registry_preset_picker"`` -- family-filtered preset
    dropdown backed by a family-keyed registry adapter.

    Meta keys:
        * ``registry`` (str, required) -- adapter key in
          :data:`_PRESET_ADAPTERS`. Currently:
          ``"init_pose_presets"`` (cascades into init_pos_{x,y,z} +
          init_joint_angles on selection) or ``"gait_presets"``
          (writes the selected name only).
        * ``family_source`` (str, optional) -- where the family comes
          from. Currently the only supported value is ``"robot_pipe"``
          (walk the upstream RobotNode connection -- mirrors
          :class:`IRRolePickerRow`).

    The bound param value is the selected preset name (string).
    Selecting nothing (empty string) is legal -- the apply-side
    consumers treat empty as "preset not selected; use canvas-level
    explicit fields".
    """

    _OPEN_ON_RELEASE = True

    def _dropdown_rect(self) -> QRectF:
        rect = self._value_rect()
        return QRectF(
            rect.left() + 4, rect.top() + 3,
            rect.width() - 8, rect.height() - 6,
        )

    def _interactive_regions(self) -> List[QRectF]:
        return [self._value_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        text = str(self._value or "") or tr(
            "canvas.row.preset_unselected", "—",
        )
        box = self._dropdown_rect()
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        draw_choice_dropdown(
            painter,
            box,
            text,
            hover=self._is_hovering(box),
            font=font,
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        view = _view_of(event)
        parent_widget = view.window() if view is not None else None
        meta = self.spec.meta or {}
        registry_key = str(meta.get("registry", "") or "")
        adapter = _PRESET_ADAPTERS.get(registry_key)
        if adapter is None:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.registry_preset_picker_unavailable",
                   default="无法打开 Preset 选单"),
                tr("dialog.registry_preset_picker_unknown",
                   default="未知 registry={key!r}；支持的 registry: {keys}").replace(
                    "{key!r}", repr(registry_key),
                ).replace(
                    "{keys}", ", ".join(sorted(_PRESET_ADAPTERS.keys())),
                ),
            )
            return True
        try:
            family = _upstream_robot_family(self._owner)
        except RuntimeError as exc:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.registry_preset_picker_unavailable",
                   default="无法打开 Preset 选单"),
                str(exc),
            )
            return True
        try:
            choices = list(adapter.list_names(family))
        except Exception as exc:
            QMessageBox.warning(
                parent_widget,
                tr("dialog.registry_preset_picker_unavailable",
                   default="无法打开 Preset 选单"),
                str(exc),
            )
            return True
        # Empty preset list -- soft no-op: family declared no presets.
        if not choices:
            QMessageBox.information(
                parent_widget,
                tr("dialog.registry_preset_picker_empty",
                   default="该家族没有可选 preset"),
                tr("dialog.registry_preset_picker_empty_detail",
                   default="family={family} 在 registry={registry} 里"
                   "没有声明 preset。").replace(
                    "{family}", family,
                ).replace(
                    "{registry}", registry_key,
                ),
            )
            return True

        def _commit(selected) -> None:
            name = str(selected or "")
            self.set_value(name)
            if not name:
                return
            try:
                adapter.on_select(self._owner, name, family)
            except Exception as exc:  # fail-loud surface to the user
                QMessageBox.warning(
                    parent_widget,
                    tr("dialog.registry_preset_apply_failed",
                       default="Preset 应用失败"),
                    str(exc),
                )

        open_choice_popup(
            view=view,
            row=self,
            choices=choices,
            current=str(self._value or ""),
            multi=False,
            on_commit=_commit,
        )
        return True


# =============================================================================
# 缺口③ — Reward joint-partition pager (widget="reward_joint_pager")
# =============================================================================

_RJP_GLOBAL = "__global__"
_RJP_OPS_W = 34.0       # Operations column (delete button)
_RJP_COUNT_W = 96.0     # Rewards-count column


class RewardJointPagerRow(_InlineTableRow):
    """Joint-partition pager for the Rewards node (above ``reward_terms``).

    Same layout as the reward_terms inline table:
      header  ``joint_group: [ {n} Groups v ]``  (no gear -- selector only)
      table   columns ``[ Joints Group | Rewards Count | Operations ]``,
              one row per page (``Global`` + each pd_group page) showing the
              page reward count + a delete button (Global is not deletable).
              Clicking a row switches ``reward_terms`` to that page via the
              ``reward_active_page`` cursor; the active row is drawn with the
              ``highlight`` background + ``alt_t1`` text.

    Stores no data of its own -- pages live in ``reward_terms`` keys (single
    source of truth). The dropdown multi-selects the robot family pd_groups.
    """

    def __init__(self, spec, owner_node, width, parent=None):
        super().__init__(spec, owner_node, width, parent)
        self._refresh_payloads()

    # ---- sibling param keys ----
    def _terms_key(self) -> str:
        return str((self.spec.meta or {}).get("terms_key", "reward_terms"))

    def _active_key(self) -> str:
        return str((self.spec.meta or {}).get("active_page_key", "reward_active_page"))

    # ---- header (label + picker; no gear) ----
    def _key_text(self) -> str:
        return "joint_group"

    def _header_summary_text(self) -> str:
        n = sum(1 for p in self._row_payloads if p.get("deletable"))
        return tr("canvas.reward_pager.summary", "{n} Groups").replace("{n}", str(n))

    def _hdr_gear_rect(self) -> QRectF:
        return QRectF()

    def _hdr_picker_rect(self) -> QRectF:
        band = self._hdr_band_rect()
        x_start = band.left() + self._key_text_width() + _INL_KEY_GAP
        x_end = band.right() - _INL_GAP
        return QRectF(x_start, band.top() + 2, max(40.0, x_end - x_start), band.height() - 4)

    def _paint_header(self, painter: QPainter) -> None:
        key_rect = self._hdr_key_rect()
        painter.setPen(QPen(QColor(Config.get_color("canvas_node_param_text", "#C8C8C8"))))
        painter.setFont(self._key_font())
        painter.drawText(key_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         self._key_text() + ":")
        btn = self._hdr_picker_rect()
        bg = QColor(Config.get_color("btn_1"))
        if self._is_hovering(btn):
            bg = bg.lighter(115)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(QColor(Config.get_color("border_1")), 1.0))
        painter.drawRoundedRect(btn, 3, 3)
        tcol = QColor(Config.get_color("main_t1"))
        painter.setPen(QPen(tcol))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        text_rect = btn.adjusted(8, 0, -18, 0)
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         fm.elidedText(self._header_summary_text(), Qt.TextElideMode.ElideRight, text_rect.width()))
        painter.setBrush(QBrush(tcol))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = btn.right() - 10.0
        cy = btn.top() + btn.height() * 0.5
        painter.drawPolygon(QPolygonF([
            QPointF(cx, cy + 3.0), QPointF(cx - 4.5, cy - 2.5), QPointF(cx + 4.5, cy - 2.5),
        ]))

    # ---- 3-column geometry: [ group name | count | ops ] ----
    def _row_ops_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        return QRectF(r.right() - _RJP_OPS_W, r.top(), _RJP_OPS_W, r.height())

    def _row_count_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        ops = self._row_ops_rect(idx)
        return QRectF(ops.left() - _RJP_COUNT_W, r.top(), _RJP_COUNT_W, r.height())

    def _row_groupname_rect(self, idx: int) -> QRectF:
        r = self._row_rect(idx)
        cnt = self._row_count_rect(idx)
        return QRectF(r.left() + 6, r.top(), max(20.0, cnt.left() - r.left() - 10), r.height())

    def _row_del_btn_rect(self, idx: int) -> QRectF:
        ops = self._row_ops_rect(idx)
        s = min(16.0, ops.height() - 6)
        return QRectF(ops.center().x() - s / 2, ops.center().y() - s / 2, s, s)

    def _paint_col_header(self, painter: QPainter) -> None:
        if not self._row_payloads:
            return
        hdr = self._col_header_rect()
        color = QColor(Config.get_color("main_c2", "#888888"))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_mini", 10)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color))
        fm = QFontMetricsF(font)
        nr = self._row_groupname_rect(0)
        cr = self._row_count_rect(0)
        orr = self._row_ops_rect(0)
        painter.drawText(QRectF(nr.left(), hdr.top(), nr.width(), hdr.height()),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         fm.elidedText(tr("canvas.reward_pager.col_group", "Joints Group"),
                                       Qt.TextElideMode.ElideRight, max(0.0, nr.width())))
        painter.drawText(QRectF(cr.left(), hdr.top(), cr.width(), hdr.height()),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                         fm.elidedText(tr("canvas.reward_pager.col_count", "Rewards Count"),
                                       Qt.TextElideMode.ElideRight, max(0.0, cr.width())))
        painter.drawText(QRectF(orr.left(), hdr.top(), orr.width(), hdr.height()),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
                         fm.elidedText(tr("canvas.reward_pager.col_ops", "Ops"),
                                       Qt.TextElideMode.ElideRight, max(0.0, orr.width())))
        sep = QColor(Config.get_color("border_1", "#444444"))
        sep.setAlpha(160)
        painter.setPen(QPen(sep, 0.8))
        painter.drawLine(QPointF(hdr.left() + 2, hdr.bottom()), QPointF(hdr.right() - 2, hdr.bottom()))

    def _paint_row(self, painter: QPainter, idx: int, payload: dict) -> None:
        row = self._row_rect(idx)
        active = (payload.get("page_id") == self._active_page())
        if active:
            painter.setBrush(QBrush(QColor(Config.get_color("highlight", "#F6D393"))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(row, 3, 3)
            text_color = QColor(Config.get_color("alt_t1", "#1E1E1E"))
        else:
            text_color = QColor(Config.get_color("main_t1", "#D6D3C7"))
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 11)))
        painter.setFont(font)
        fm = QFontMetricsF(font)
        nr = self._row_groupname_rect(idx)
        if payload.get("page_id") == _RJP_GLOBAL:
            label = tr("canvas.reward_pager.global", "Global (all joints)")
        else:
            label = str(payload.get("page_id"))
        painter.setPen(QPen(text_color))
        painter.drawText(nr, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         fm.elidedText(label, Qt.TextElideMode.ElideRight, max(0.0, nr.width())))
        cr = self._row_count_rect(idx)
        painter.drawText(cr, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                         str(int(payload.get("count", 0))))
        if payload.get("deletable"):
            db = self._row_del_btn_rect(idx)
            bg = QColor(Config.get_color("btn_1", "#343636"))
            if self._is_hovering(self._row_ops_rect(idx)):
                bg = bg.lighter(115)
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(QColor(Config.get_color("border_1", "#444444")), 1.0))
            painter.drawRoundedRect(db, 3, 3)
            painter.setPen(QPen(QColor(Config.get_color("danger_zone", "#C84B4B")), 1.4))
            m = 4.0
            painter.drawLine(QPointF(db.left() + m, db.top() + m), QPointF(db.right() - m, db.bottom() - m))
            painter.drawLine(QPointF(db.left() + m, db.bottom() - m), QPointF(db.right() - m, db.top() + m))

    # ---- payloads / pages ----
    def _paged_terms(self) -> dict:
        from application.compiler.term_payload import migrate_reward_terms_to_paged
        raw = self._owner.params.get(self._terms_key())
        return migrate_reward_terms_to_paged(_parse_dict_value(raw) or {})

    def _active_page(self) -> str:
        return str(self._owner.params.get(self._active_key(), _RJP_GLOBAL) or _RJP_GLOBAL)

    def _ordered_pages(self):
        paged = self._paged_terms()
        groups = sorted(k for k in paged if k != _RJP_GLOBAL)
        return [_RJP_GLOBAL] + groups, paged

    def _refresh_payloads(self) -> None:
        pages, paged = self._ordered_pages()
        self._row_payloads = [
            {"page_id": pid, "count": len(paged.get(pid, {}) or {}),
             "deletable": pid != _RJP_GLOBAL}
            for pid in pages
        ]
        self._set_height(self._compute_height())

    def maybe_refresh_for_asset_change(self) -> None:
        self._refresh_payloads()
        self.update()

    # ---- robot family + partition choices ----
    def _robot_family(self) -> Optional[str]:
        try:
            sku = _scene_robot_sku(self._owner)
        except Exception:                                         # noqa: BLE001
            sku = None
        if not sku:
            return None
        try:
            from registers.robots import get_robot_spec
            rs = get_robot_spec(sku)
            fams = list(getattr(rs, "families", []) or []) if rs is not None else []
            return fams[0] if fams else None
        except Exception:                                         # noqa: BLE001
            return None

    def _robot_joint_ir_roles(self) -> List[str]:
        """The upstream RobotNode's ACTUAL joint IR roles (not the family's).

        Source of truth for which joint groups this specific robot has. Tries
        the resolved RobotSpec first, then falls back to the registry
        ``joints_per_format`` table (USD with MJCF legacy fallback — the same
        source the action-joints picker uses).
        """
        try:
            sku = _scene_robot_sku(self._owner)
        except Exception:                                         # noqa: BLE001
            sku = None
        if not sku:
            return []
        try:
            from registers.robots import get_robot_spec
            rs = get_robot_spec(sku)
            roles = list(getattr(rs, "joint_ir_roles", []) or [])
            if roles:
                return roles
        except Exception:                                         # noqa: BLE001
            pass
        try:
            from registers.robots import get_robot
            entry = get_robot(sku) or {}
            jpf = entry.get("joints_per_format") or {}
            blk = jpf.get("USD") or jpf.get("MJCF") or {}
            out: List[str] = []
            seen: set = set()
            for spec in (blk or {}).values():
                ir = str((spec or {}).get("ir_role", "") or "")
                if ir and ir not in seen:
                    seen.add(ir)
                    out.append(ir)
            return out
        except Exception:                                         # noqa: BLE001
            return []

    def _partition_choices(self) -> List[str]:
        """Joint groups the CURRENT robot actually has — never the whole family.

        The pager must only offer pages for groups the upstream RobotNode owns:
        a humanoid H1 has a single-DOF ankle (ankle_pitch, NO ankle_roll), so
        offering ankle_roll would let the user page rewards onto a joint group
        the robot lacks — a frontend⇄backend mismatch (the compiler would have
        no joints for that page). We resolve the robot's actual IR roles to
        their pd_group via ``find_group_for_role`` and keep only those groups,
        preserving the family's declaration order.
        """
        fam = self._robot_family()
        if not fam:
            return []
        roles = self._robot_joint_ir_roles()
        if not roles:
            # Robot unresolved: do NOT fall back to the full family list (that
            # is exactly the bug — offering groups the robot may not have).
            return []
        try:
            from registers import pd_groups
            return pd_groups.groups_for_roles(fam, roles)
        except Exception:                                         # noqa: BLE001
            return []

    # ---- interaction ----
    def _interactive_regions(self) -> List[QRectF]:
        regions = [self._hdr_picker_rect()]
        for i in range(len(self._row_payloads)):
            regions.append(self._row_rect(i))
            if self._row_payloads[i].get("deletable"):
                regions.append(self._row_del_btn_rect(i))
        return regions

    def _handle_click(self, event) -> bool:
        pos = event.pos()
        if self._hdr_picker_rect().contains(pos):
            return self._on_selector_clicked(event)
        for idx, payload in enumerate(self._row_payloads):
            if payload.get("deletable") and self._row_del_btn_rect(idx).contains(pos):
                self._delete_page(str(payload.get("page_id")))
                return True
        for idx, payload in enumerate(self._row_payloads):
            if self._row_rect(idx).contains(pos):
                self._switch_active_page(str(payload.get("page_id")))
                return True
        return False

    def _handle_double_click(self, event) -> bool:
        return self._handle_click(event)

    def _on_selector_clicked(self, event) -> bool:
        choices = self._partition_choices()
        if not choices:
            return False
        _pages, paged = self._ordered_pages()
        current = [k for k in paged if k != _RJP_GLOBAL]

        def _commit(selected):
            sel = list(selected) if isinstance(selected, (list, tuple)) else (
                [selected] if selected else [])
            self._apply_partition_selection(sel)

        open_choice_popup(
            view=_view_of(event), row=self,
            choices=choices, current=current,
            multi=True, leading_mode="checkbox",
            on_commit=_commit,
        )
        return True

    def _on_modal_clicked(self, event) -> bool:
        return False

    # ---- mutations ----
    def _apply_partition_selection(self, selected) -> None:
        sel = set(str(s) for s in selected)
        paged = self._paged_terms()
        new = {_RJP_GLOBAL: paged.get(_RJP_GLOBAL, {})}
        for g in sel:
            new[g] = paged.get(g, {})
        self._write_full_terms(new)
        if self._active_page() not in new:
            self._switch_active_page(_RJP_GLOBAL)
        self._refresh_payloads()
        self.update()

    def _delete_page(self, page_id: str) -> None:
        if page_id == _RJP_GLOBAL:
            return
        paged = self._paged_terms()
        if page_id not in paged:
            return
        paged.pop(page_id, None)
        self._write_full_terms(paged)
        if self._active_page() == page_id:
            self._switch_active_page(_RJP_GLOBAL)
        self._refresh_payloads()
        self.update()

    def _write_full_terms(self, paged_dict: dict) -> None:
        key = self._terms_key()
        self._owner.params[key] = paged_dict
        notify = getattr(self._owner, "on_param_changed", None)
        if callable(notify):
            notify(key, paged_dict)

    def _switch_active_page(self, page_id: str) -> None:
        _write_sibling_param(self._owner, self._active_key(), str(page_id))
        self.update()


_WPE_FRAME_PAD = 2  # outer pad inside the row's full-width value rect


def _run_weight_set_dialog(view, items_raw: dict):
    """Open the Weight Set pie editor for the enabled items of a ``training_items``
    dict and return the dict with the edited per-item ``weight`` written back
    (disabled items untouched), or ``None`` if there are no enabled items or the
    user cancelled. Shared by :class:`WeightPieEditorRow` (the node's Weight Set
    button) and :class:`TrainingItemsInlineRow` (clicking a Weight cell) so both
    entry points behave identically. The caller persists the returned dict via
    its own seam (sibling-write vs. ``set_value``)."""
    items_raw = items_raw or {}
    enabled = [
        (iid, cfg)
        for iid, cfg in items_raw.items()
        if isinstance(cfg, dict) and cfg.get("enabled")
    ]
    if not enabled:
        return None
    # Legend label: user-typed ``title`` wins, blank → fall back to the id
    # (matches the inline row + modal editor display semantics). ``iid`` stays
    # the authoritative sampling key threaded back into ``weight``.
    item_pairs = [(iid, str(cfg.get("title") or "").strip() or iid) for iid, cfg in enabled]
    current: dict = {}
    for iid, cfg in enabled:
        w = cfg.get("weight")
        if w is None:
            continue
        try:
            current[iid] = float(w)
        except (TypeError, ValueError):
            pass  # malformed → treated as unset (uniform share) in dialog
    parent_widget = view.window() if view else None
    from application.ui.dialogs import WeightSetDialog

    dlg = WeightSetDialog(item_pairs, current, parent=parent_widget)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    result = dlg.result_weights or {}
    updated = dict(items_raw)
    for iid, cfg in enabled:
        new_cfg = dict(cfg)
        if iid in result:
            new_cfg["weight"] = float(result[iid])
        updated[iid] = new_cfg
    return updated


class WeightPieEditorRow(ParamRow):
    """``widget="weight_pie_editor"`` — full-width button opening the Weight Set
    pie editor for per-item initial sampling weights.

    Action-only: its own ``weight_set`` param holds no value (like
    :class:`ReviewLaunchButtonRow`). On click it reads the **sibling**
    ``training_items`` param, opens :class:`WeightSetDialog` for the enabled
    items + their current ``weight``, and on accept writes the edited weights
    back into each enabled item's ``weight`` via :func:`_write_sibling_param`
    (so ``training_items`` stays the single data of record). Disabled items are
    left untouched. Shares the review-button theme slots.
    """

    _OPEN_ON_RELEASE = True

    def _btn_rect(self) -> QRectF:
        x = SEP_X + _WPE_FRAME_PAD
        return QRectF(
            x,
            _WPE_FRAME_PAD,
            max(0.0, self._width - x - _WPE_FRAME_PAD - VALUE_PAD_RIGHT + 4),
            self._height - _WPE_FRAME_PAD * 2,
        )

    def _interactive_regions(self) -> List[QRectF]:
        return [self._btn_rect()]

    def _paint_value(self, painter: QPainter, tier: int) -> None:
        btn = self._btn_rect()
        idle = QColor(Config.get_color("canvas_node_review_launch_bg", "#00695C"))
        hover = QColor(
            Config.get_color("canvas_node_review_launch_hover_bg", "#00897B")
        )
        text_color = QColor(Config.get_color("main_t2", "#FFFFFF"))
        bg = hover if self._is_hovering(btn) else idle
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(btn, 4, 4)
        font = QFont(Config.get_value("Font", "family", "Microsoft YaHei"))
        font.setPixelSize(int(Config.get_font_size("size_small", 12)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        painter.drawText(
            btn,
            int(Qt.AlignmentFlag.AlignCenter),
            "⚖  " + tr("canvas.training_motion.weight_set.button", "Weight Set"),
        )

    def _handle_click(self, event: QGraphicsSceneMouseEvent) -> bool:
        from unitport_sdk import log_warning

        items_raw = _parse_dict_value(self._owner.params.get("training_items"))
        if not any(
            isinstance(cfg, dict) and cfg.get("enabled")
            for cfg in (items_raw or {}).values()
        ):
            log_warning(
                "[weight_set] no enabled training items to weight — open the "
                "items editor and enable at least one item first."
            )
            return True
        updated = _run_weight_set_dialog(_view_of(event), items_raw)
        if updated is not None:
            _write_sibling_param(self._owner, "training_items", updated)
        return True


# =============================================================================
# Factory / dispatcher —— 节点脚本 → 图形的「自动转换」入口
# =============================================================================

# 分发表（widget hint 优先；否则按 type）
WIDGET_DISPATCH = {
    # 通用 7 种
    "path":                       PathRow,
    "code":                       CodeRow,
    "range":                      RangeRow,
    "range_list":                 RangeListRow,
    "index":                      IndexRow,
    "badge":                      BadgeRow,
    "table_readonly":             TableReadOnlyRow,
    # 节点专属 11 种 — SDK Node_* picker / editor
    "picker_scene":               SceneTypeRow,
    "picker_task_type":           TaskTypeRow,
    "picker_start_point":         StartPointRow,
    "picker_project_checkpoint":  CheckpointPickerRow,
    "picker_export_bundle":       ExportBundleRow,
    "picker_motion_library":      MotionLibraryRow,
    "picker_obs_components":      ObsComponentsRow,
    "registry_module":            RegistryModuleInlineRow,
    "training_items":             TrainingItemsInlineRow,
    "registry_preset_picker":     RegistryPresetPickerRow,
    "reward_joint_pager":         RewardJointPagerRow,  # 缺口③
    "stage_editor":               StageEditorRow,
    "curve":                      CurveRow,
    "buffer_size":                BufferSizeRow,
    "picker_robot_asset":         RobotAssetPickerRow,
    "body_mapping_table":         BodyMappingTableRow,
    "active_file_table":          ActiveFileTableRow,
    "mjcf_offset_badge":          MjcfOffsetBadgeRow,
    # Export node — DEMO §1 / §2 / §3 widgets
    "output_file_list":           OutputFileListRow,
    "start_point_compat_panel":   StartPointCompatPanelRow,
    "review_backend_picker":      ReviewBackendPickerRow,
    "review_scene_picker":        ReviewScenePickerRow,
    "review_launch_button":       ReviewLaunchButtonRow,
    # TrainingMotionNode — initial sampling-weight pie editor button
    "weight_pie_editor":          WeightPieEditorRow,
    # ActorSettingNode — joint pose slider table + Review Pose button
    "joint_pose_table":           JointPoseTableRow,
    "actor_review_pose_button":   ActorReviewPoseButtonRow,
    # RobotNode — per-joint PD / actuator panel
    "pd_param_table":             PDParamTableRow,
    # ActorSettingNode — IR-role multi-select pickers (joints / bodies)
    "ir_joint_picker":            IRRolePickerRow,
    "ir_body_picker":             IRRolePickerRow,
}

TYPE_DISPATCH = {
    "bool":     BoolRow,
    "int":      NumberRow,
    "float":    NumberRow,
    "number":   NumberRow,
    "string":   StringRow,
    "enum":     ChoiceRow,
    "json":     CodeRow,
    # 兼容老/容错
    "dict":     CodeRow,
    "list":     CodeRow,
}

VALID_WIDGETS = frozenset(WIDGET_DISPATCH.keys())
VALID_TYPES = frozenset(("bool", "int", "float", "string", "enum", "json"))


def validate_param_spec(spec: ParamSpec) -> Optional[str]:
    """启动期 schema 校验 / Validate ParamSpec at discovery time.

    返回 ``None`` 表示通过；返回非空 str 表示错误信息。

    检查项：
    - ``spec.widget`` 必须在 ``VALID_WIDGETS`` 内（或 None）
    - ``spec.type`` 必须在 ``VALID_TYPES`` 内
    - ``spec.type == "enum"`` 时 ``spec.choices`` 必须非空
    - ``spec.widget == "range"`` 时 ``spec.meta`` 必须含 ``min`` / ``max``
    - ``spec.widget == "index"`` 时 ``spec.type`` 必须是 ``int``
    """
    widget = (spec.widget or "").strip().lower()
    if widget and widget not in VALID_WIDGETS:
        return (
            f"param {spec.key!r}: widget {spec.widget!r} 未识别；"
            f"有效值 {sorted(VALID_WIDGETS)} 或 None"
        )
    t = (spec.type or "").strip().lower()
    if t not in VALID_TYPES:
        # 留个口子接受 dict/list/number 旧拼法（warn 但不阻断）
        if t not in ("dict", "list", "number"):
            return (
                f"param {spec.key!r}: type {spec.type!r} 未识别；"
                f"有效值 {sorted(VALID_TYPES)}"
            )
    if t == "enum" and not spec.choices:
        return f"param {spec.key!r}: type='enum' 时 choices 不能为空"
    if widget == "range":
        meta = spec.meta or {}
        if "min" not in meta or "max" not in meta:
            return f"param {spec.key!r}: widget='range' 时 meta 必须含 min / max"
    if widget == "range_list":
        meta = spec.meta or {}
        if "min" not in meta or "max" not in meta:
            return f"param {spec.key!r}: widget='range_list' 时 meta 必须含 min / max"
        if t not in ("list", "json"):
            return f"param {spec.key!r}: widget='range_list' 时 type 必须是 'list' 或 'json'"
    if widget == "index" and t != "int":
        return f"param {spec.key!r}: widget='index' 时 type 必须是 'int'"
    if widget == "picker_robot_asset" and t != "string":
        return f"param {spec.key!r}: widget='picker_robot_asset' 时 type 必须是 'string'"
    if widget == "body_mapping_table" and t != "json":
        return f"param {spec.key!r}: widget='body_mapping_table' 时 type 必须是 'json'"
    if widget == "active_file_table" and t != "enum":
        return f"param {spec.key!r}: widget='active_file_table' 时 type 必须是 'enum'"
    # Export node §1 / §2 / §3 widgets
    if widget in ("output_file_list", "start_point_compat_panel",
                  "review_launch_button", "review_scene_picker") and t != "string":
        return (
            f"param {spec.key!r}: widget={spec.widget!r} 时 type 必须是 'string'"
        )
    if widget == "review_backend_picker":
        if t != "enum":
            return (
                f"param {spec.key!r}: widget='review_backend_picker' 时 type 必须是 'enum'"
            )
        if not spec.choices:
            return (
                f"param {spec.key!r}: widget='review_backend_picker' 时 choices 不能为空"
            )
    return None


def create_param_row(
    spec: ParamSpec,
    owner_node: "NodeItem",
    width: float,
) -> ParamRow:
    """根据 ``spec.widget`` / ``spec.type`` 选 ParamRow 子类 / Pick row class.

    优先级：``widget`` > ``type``。``choices`` 非空且 ``type`` 不是 enum 时也走 ChoiceRow.

    本函数容忍未知 type（兜底 ``StringRow``）；启动期严格校验由
    ``validate_param_spec`` 在 discovery 阶段执行。
    """
    widget = (spec.widget or "").strip().lower()
    if widget in WIDGET_DISPATCH:
        return WIDGET_DISPATCH[widget](spec, owner_node, width)

    t = (spec.type or "").strip().lower()
    # enum-via-choices 兼容（type=string but has choices → ChoiceRow）
    if t == "string" and spec.choices:
        return ChoiceRow(spec, owner_node, width)
    if t in TYPE_DISPATCH:
        return TYPE_DISPATCH[t](spec, owner_node, width)
    # 兜底
    return StringRow(spec, owner_node, width)


__all__ = [
    "ParamRow",
    # 通用 10
    "BoolRow",
    "NumberRow",
    "StringRow",
    "ChoiceRow",
    "PathRow",
    "IndexRow",
    "RangeRow",
    "RangeListRow",
    "BadgeRow",
    "CodeRow",
    "TableReadOnlyRow",
    # 节点专属（SDK Node_* picker / editor）
    "SceneTypeRow",
    "TaskTypeRow",
    "StartPointRow",
    "ExportBundleRow",
    "MotionLibraryRow",
    "ObsComponentsRow",
    "TrainingItemsRow",
    "StageEditorRow",
    "CurveRow",
    "BufferSizeRow",
    "RobotAssetPickerRow",
    "BodyMappingTableRow",
    # Export node §1 / §2 / §3 widgets
    "OutputFileListRow",
    "StartPointCompatPanelRow",
    "ReviewBackendPickerRow",
    "ReviewScenePickerRow",
    "ReviewLaunchButtonRow",
    # ActorSettingNode init pose slider editor + Review Pose button
    "JointPoseTableRow",
    "ActorReviewPoseButtonRow",
    # RobotNode per-joint PD / actuator panel
    "PDParamTableRow",
    # factory + 分发表
    "create_param_row",
    "validate_param_spec",
    "WIDGET_DISPATCH",
    "TYPE_DISPATCH",
    "VALID_WIDGETS",
    "VALID_TYPES",
    "PARAM_ROW_H",
]
