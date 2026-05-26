# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.ui.canvas.popups — Popup 工厂调度 / Popup factory dispatchers.

ParamRow 触发就地编辑时调用本模块的 ``open_*_popup`` 工厂；工厂只负责：
    1. ``screen_anchor_for_row(view, row)`` 计算锚点
    2. instantiate **SDK 的 Node_* popup class**
    3. ``popup.show_at(anchor, row_h)``
    4. ``_retain_popup(view, popup)`` 防 Python GC

popup 视觉契约 / 行为合约的唯一来源在 ``unitport_sdk/canvas/popups.py``：
    Node_DataInput        — 单行 [LineEdit][✓][✗] 输入（string / int / float 三态）
    Node_CodeEditorPopup  — 多行 code/JSON 编辑（OK/Cancel）
    Node_EditorPopupHost  — 通用 widget 容器（OK/Cancel + value_getter）
    Node_RichChoicePopup  — 选单（已存在）

本文件不再定义任何 widget / popup class —— 所有 Node 用的 UI 组件均由 SDK
提供。本文件保留的仅是 ui-layer 几何粘合（QGraphicsView ↔ QPoint 锚点
计算、防 GC bag、QFileDialog 调度），它们与 ``QGraphicsView`` / ``QGraphicsItem``
强耦合，不应进入 SDK。

公开接口（``param_rows.py`` 调用）：
    screen_anchor_for_row(view, row)  — 计算行右上角的全局屏幕坐标
    open_choice_popup(...)            — 单/多选；body=Node_RichChoicePopup
    open_text_popup(...)              — 单行文本；body=Node_DataInput(dtype='string')
    open_number_popup(...)            — int/float；body=Node_DataInput(dtype='int'/'float')
    open_code_popup(...)              — 多行；body=Node_CodeEditorPopup
    open_widget_popup(...)            — 通用；body=Node_EditorPopupHost
    open_path_dialog(...)             — 文件选择走系统 QFileDialog（无 Node_* 等价物）
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from PyQt6.QtCore import QPoint, QPointF, QRectF
from PyQt6.QtWidgets import (
    QFileDialog,
    QGraphicsItem,
    QGraphicsView,
    QWidget,
)

from unitport_sdk import (
    Node_CodeEditorPopup,
    Node_DataInput,
    Node_EditorPopupHost,
    Node_RichChoicePopup,
)


# =============================================================================
# Python 引用保持 / Python ref retention
# =============================================================================
#
# 即使 Qt parent 持有 C++ 对象，PyQt 包装层仍可能被 Python GC 回收，导致
# 信号回调失效（"sip wrapper has been deleted" 类问题）。把每个活跃 popup
# 挂到宿主 view 的属性 list 上，destroyed 时自动摘除。

_RETAIN_ATTR = "_canvas_active_popups"


def _retain_popup(host: Optional[QWidget], popup: QWidget) -> None:
    """把 popup 挂到 host 的属性上防 GC；popup destroyed 时自动摘掉."""
    if host is None or popup is None:
        return
    bag = getattr(host, _RETAIN_ATTR, None)
    if bag is None:
        bag = []
        try:
            setattr(host, _RETAIN_ATTR, bag)
        except Exception:
            return
    bag.append(popup)

    def _drop(*_a: Any) -> None:
        try:
            bag.remove(popup)
        except (ValueError, RuntimeError):
            pass

    popup.destroyed.connect(_drop)


# =============================================================================
# 锚点计算 / Anchor computation
# =============================================================================

def screen_anchor_for_row(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    offset_x: int = 2,
    sub_rect: Optional["QRectF"] = None,
) -> tuple[QPoint, int]:
    """返回行 (右上角的全局屏幕 QPoint, 行高 px) / Global screen anchor for row.

    锚点 = 行右上角 + ``offset_x`` 像素；行高按 view 缩放后的实际像素。
    无 view 时回落到 (0,0)。

    ``sub_rect`` 可选 — 若给定（item-local 坐标），锚点改为该子矩形的右上角，
    返回的行高也按子矩形高度计算。用于变高表格行（``_InlineTableRow``）锚定
    单条 row 的 value 单元格而不是整个 N 行 boundingRect。

    DEMO 同等实现见 ``training_node_items.py:_popup_anchor`` (1163-1184)。
    """
    if view is None or row is None:
        return QPoint(0, 0), 24
    rect = sub_rect if sub_rect is not None else row.boundingRect()
    scene_tr = row.mapToScene(QPointF(rect.right(), rect.top()))
    scene_br = row.mapToScene(QPointF(rect.right(), rect.bottom()))
    view_tr = view.mapFromScene(scene_tr)
    view_br = view.mapFromScene(scene_br)
    global_pos = view.viewport().mapToGlobal(view_tr)
    row_h = max(1, view_br.y() - view_tr.y())
    return QPoint(global_pos.x() + offset_x, global_pos.y()), int(row_h)


# =============================================================================
# 单行文本 popup —— body = SDK Node_DataInput(dtype='string')
# =============================================================================

def open_text_popup(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    value: str,
    placeholder: str = "",
    on_commit: Callable[[str], None],
) -> Node_DataInput:
    # parent=None：让 popup 成为干净的 top-level Qt.WindowType.Popup，避免
    # Qt 把它当作 view 的子窗导致 popup grab 失效。GC 由 _retain_popup 兜底。
    popup = Node_DataInput(
        value=value,
        dtype="string",
        placeholder=placeholder,
        on_commit=on_commit,
        parent=None,
    )
    # row_h 是 view 缩放后的实际像素行高 → popup 跟随 zoom 缩放
    anchor, row_h = screen_anchor_for_row(view, row)
    popup.show_at(anchor, row_h)
    _retain_popup(view, popup)
    return popup


# =============================================================================
# 数字 popup —— body = SDK Node_DataInput(dtype='int'/'float')
# =============================================================================

def open_number_popup(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    value: float | int,
    is_int: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    step: Optional[float] = None,  # noqa: ARG001 — 兼容旧签名；DataInput 不用 step
    on_commit: Callable[[Any], None],
    sub_rect: Optional[QRectF] = None,
) -> Node_DataInput:
    popup = Node_DataInput(
        value=value,
        dtype="int" if is_int else "float",
        minimum=minimum,
        maximum=maximum,
        on_commit=on_commit,
        parent=None,
    )
    # row_h 跟随 view 缩放 → popup 同步缩放（DEMO DataInput.show_at 行为）
    # 变高表格行可传 ``sub_rect`` 锚定单 cell，避免 popup 拉到整张表的高度。
    anchor, row_h = screen_anchor_for_row(view, row, sub_rect=sub_rect)
    popup.show_at(anchor, row_h)
    _retain_popup(view, popup)
    return popup


# =============================================================================
# 选单 popup —— 直接复用 SDK Node_RichChoicePopup（已是 Qt.WindowType.Popup）
# =============================================================================

def open_choice_popup(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    choices: Sequence[Any],
    current: Any = None,
    multi: bool = False,
    leading_mode: str = "checkbox",
    meta_map: Optional[dict] = None,
    on_commit: Callable[[Any], None],
    label_resolver: Optional[Callable[[Any], str]] = None,
) -> Node_RichChoicePopup:
    """SDK Node_RichChoicePopup 锚定到 row 右侧 / Anchored choice popup with item cards.

    单选：选中即 emit ``on_commit(value)`` 并自动 close（DEMO 行为一致）。
    多选：每次 toggle 都 emit ``on_commit(list)``——所见即所得。

    ``label_resolver``（i18n）：可选回调，接收一个 raw choice，返回展示用的
    title 字符串。提供后会自动合成 ``meta_map[choice]["title"] = resolver(choice)``，
    显式传入的 ``meta_map[choice]["title"]`` 优先。raw choice 仍以 id 形态
    用于选中态匹配与 ``on_commit`` 回调，确保未触碰的旧调用方零变更。
    """
    str_choices = [str(c) for c in choices]
    if multi:
        if isinstance(current, (list, tuple)):
            str_current: Any = [str(c) for c in current]
        elif isinstance(current, str) and current.strip():
            # 兼容空格分隔字符串（DEMO 风格）
            str_current = current.split()
        else:
            str_current = []
    else:
        str_current = str(current) if current is not None else None

    # Merge i18n titles from label_resolver into meta_map; do not clobber an
    # explicit title the caller already supplied for a given choice.
    merged_meta: Optional[dict] = None
    if label_resolver is not None:
        merged_meta = {}
        for raw, s in zip(choices, str_choices):
            existing = (meta_map or {}).get(s, {})
            entry = dict(existing) if isinstance(existing, dict) else {}
            if "title" not in entry:
                try:
                    entry["title"] = label_resolver(raw)
                except Exception:
                    entry["title"] = s
            merged_meta[s] = entry
        # carry through any other entries the caller passed that don't
        # correspond to a current choice id (rare; keeps API surface stable)
        if meta_map:
            for k, v in meta_map.items():
                merged_meta.setdefault(k, v)
    else:
        merged_meta = meta_map

    popup = Node_RichChoicePopup(
        choices=str_choices,
        meta_map=merged_meta,
        leading_mode=leading_mode,
        multi_select=multi,
        current=str_current,
        parent=None,
    )

    def _map_back(s: str) -> Any:
        for v in choices:
            if str(v) == s:
                return v
        return s

    if multi:
        popup.selection_changed.connect(
            lambda sel: on_commit([_map_back(s) for s in sel])
        )
    else:
        popup.selection_committed.connect(lambda s: on_commit(_map_back(s)))

    anchor, _ = screen_anchor_for_row(view, row)
    # 与 SDK Node_RichChoicePicker._on_button_clicked 完全一致：
    # 仅 move + show；**不要** activateWindow / raise_，它们会破坏
    # Qt.WindowType.Popup 的 popup grab，导致点击外部无法 auto-close。
    popup.move(anchor)
    popup.show()
    # 防 Python GC：popup 是 Qt.WindowType.Popup 顶层窗，parent=view 让 Qt
    # 持引用直到 WA_DeleteOnClose 触发。这里也把 popup 暂存到 view 上做双保险。
    _retain_popup(view, popup)
    return popup


# =============================================================================
# 多行代码 / JSON popup —— body = SDK Node_CodeEditorPopup
# =============================================================================

def open_code_popup(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    text: str,
    language: str = "json",
    validate_json: bool = False,
    on_commit: Callable[[str], None],
) -> Node_CodeEditorPopup:
    popup = Node_CodeEditorPopup(
        text=text,
        language=language,
        validate_json=validate_json,
        on_commit=on_commit,
        parent=None,
    )
    anchor, _ = screen_anchor_for_row(view, row)
    popup.show_at(anchor)
    _retain_popup(view, popup)
    return popup


# =============================================================================
# 通用 widget popup —— body = SDK Node_EditorPopupHost(任意 Node_* widget)
# =============================================================================

def open_widget_popup(
    view: Optional[QGraphicsView],
    row: QGraphicsItem,
    *,
    body: QWidget,
    value_getter: Callable[[QWidget], Any],
    on_commit: Callable[[Any], None],
    min_width: int = 360,
    min_height: int = 240,
) -> Node_EditorPopupHost:
    popup = Node_EditorPopupHost(
        body=body,
        value_getter=value_getter,
        on_commit=on_commit,
        min_width=min_width,
        min_height=min_height,
        parent=None,
    )
    anchor, _ = screen_anchor_for_row(view, row)
    popup.show_at(anchor)
    _retain_popup(view, popup)
    return popup


# =============================================================================
# 文件选择 —— 系统 QFileDialog（无对应 Node_*）
# =============================================================================

def open_path_dialog(
    title: str,
    current: str,
    *,
    is_dir: bool = False,
    name_filter: str = "",
    parent: Optional[QWidget] = None,
) -> Optional[str]:
    """系统文件选择器 / System file picker (no Node_* equivalent).

    跨平台一致体验比统一外观更重要——Windows/Linux/Mac 用户更熟悉系统对话框。
    """
    if is_dir:
        path = QFileDialog.getExistingDirectory(parent, title, current)
    else:
        path, _filter = QFileDialog.getOpenFileName(parent, title, current, name_filter)
    return path or None


__all__ = [
    "screen_anchor_for_row",
    "open_text_popup",
    "open_number_popup",
    "open_choice_popup",
    "open_code_popup",
    "open_widget_popup",
    "open_path_dialog",
]
