# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.pickers — Node 选单家族 / Node-internal picker family.

DEMO 对应（``bin/nodes/training_node_items.py``）：
    _RichChoiceCard       (2272-2371)  → Node_RichChoiceCard
    _RichChoicePopup      (2374-2556)  → Node_RichChoicePopup
    _RichChoicePicker     (2558-2752)  → Node_RichChoicePicker
    _TaskTypePicker       (2755-2764)  → Node_TaskTypePicker
    _ObsComponentsPicker  (2767-2782)  → Node_ObsComponentsPicker
    _DropdownChoicePicker (2785-2795)  → Node_DropdownChoicePicker
    _SceneTypePicker      (2798-2855)  → Node_SceneTypePicker
    _ExportBundlePicker   (3943+)      → Node_ExportBundlePicker
    _StartPointChoicePicker (4034+)    → Node_StartPointChoicePicker
    _MotionLibraryPicker  (5410+)      → Node_MotionLibraryPicker
DEMO 对应（``bin/nodes/node_ui_rows.py``）：
    MainPicker            (1352-1436)  → Node_MainPicker

视觉契约（一比一对齐 DEMO）将在阶段 2 完成；本文件先把骨架登记到 SDK，
内部用 ``LaviButton`` / ``LaviLabel`` 做可点击区域 + 文本。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk.sys import Config
from unitport_sdk.ui import I18nButton, I18nLabel, LaviButton, LaviLabel, setButton, setText


# =============================================================================
# 几何常量（DEMO Training Node 契约）
# =============================================================================

NODE_PICKER_W = 134
NODE_PICKER_H = 24
NODE_POPUP_MIN_W = 320
NODE_POPUP_MAX_H = 320
NODE_CARD_PAD_H = 10
NODE_CARD_PAD_V = 8
NODE_CARD_INNER_GAP = 12
NODE_CARD_LEADING_W = 22
NODE_CARD_RADIUS = 8


# =============================================================================
# Node_RichChoiceCard —— 单个选项卡片
# =============================================================================

class Node_RichChoiceCard(QFrame):
    """选单 popup 内单个 choice 卡片 / Single choice card inside picker popup.

    DEMO 行 2272-2371。布局：[checkbox/icon 22px][gap 12px][title + desc 列].

    视觉契约：
        - 圆角 8px，内边距 (10/8)px
        - 未选中：透明 bg + 1px 透明 border（占位）
        - 悬停：bg=canvas_widget_popup_selected_bg, border=canvas_widget_focus_border
        - 选中：bg=canvas_widget_focus_border + 文字 white
        - title  13px / weight 600
        - desc   11px / regular / wrap
    """

    clicked = pyqtSignal(str)

    def __init__(
        self,
        choice: str,
        *,
        title: str = "",
        desc: str = "",
        leading_mode: str = "checkbox",  # "checkbox" | "icon" | "text"
        selected: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._choice = choice
        self._selected = selected
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(NODE_CARD_PAD_H, NODE_CARD_PAD_V, NODE_CARD_PAD_H, NODE_CARD_PAD_V)
        layout.setSpacing(NODE_CARD_INNER_GAP)

        # leading: checkbox / icon / 空
        self._leading: Optional[QWidget] = None
        if leading_mode == "checkbox":
            cb = QCheckBox(self)
            cb.setChecked(selected)
            cb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            cb.setFixedWidth(NODE_CARD_LEADING_W)
            self._leading = cb
        elif leading_mode == "icon":
            icon_label = QFrame(self)
            icon_label.setFixedSize(NODE_CARD_LEADING_W, NODE_CARD_LEADING_W)
            self._leading = icon_label
        if self._leading is not None:
            layout.addWidget(self._leading)

        # text column: title + desc
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        self._title_label = setText(f"choice.{choice}.title", default=title or choice)
        self._desc_label = setText(f"choice.{choice}.desc", default=desc, kind="caption")
        self._desc_label.setWordWrap(True)
        text_col.addWidget(self._title_label)
        text_col.addWidget(self._desc_label)
        layout.addLayout(text_col, 1)

        self.setProperty("selected", "true" if selected else "false")
        self.refresh_style()
        self._apply_label_colors()

    @property
    def choice(self) -> str:
        return self._choice

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, sel: bool) -> None:
        self._selected = bool(sel)
        if isinstance(self._leading, QCheckBox):
            self._leading.setChecked(self._selected)
        self.setProperty("selected", "true" if self._selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self._apply_label_colors()
        self.update()

    def _apply_label_colors(self) -> None:
        """把 selected 字体色推到 title/desc LaviLabel 自身 QSS 上.

        LaviLabel 在自己控件上 setStyleSheet 写死 color；这优先级高于父
        QFrame 的后代选择器。所以 Node_RichChoiceCard QSS 里写
        ``QLabel[selected=true] { color: alt_t1 }`` 不会生效——必须直接改
        子标签的 ``_color_override``、再 refresh_style 重建它自身的 QSS。
        """
        if self._selected:
            sel_fg = Config.get_color("bg_1", "#1E1E1E")
            override = sel_fg
        else:
            override = None  # 回退到 LaviLabel kind 默认色
        for lbl in (self._title_label, self._desc_label):
            try:
                lbl._color_override = override  # type: ignore[attr-defined]
                lbl.refresh_style()
            except Exception:
                pass

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._choice)
            e.accept()
            return
        super().mousePressEvent(e)

    def refresh_style(self) -> None:
        # 视觉契约（用户严格指定）：
        # - hover bg     : canvas_widget_popup_selected_bg
        # - selected bg  : "highlight"   (#F6D393, 主题黄)
        # - selected text: "bg_1"       (#1E1E1E, 暗色文字)
        # - checkbox     : 未勾常规边；勾选用 "safe_zone" (#36E38E)
        hover_bg = Config.get_color("btn_1", "#2d4a7a")
        accent = Config.get_color("checked_1", "#4f7ecc")
        sel_bg = Config.get_color("highlight", "#F6D393")
        sel_fg = Config.get_color("bg_1", "#1E1E1E")
        check_bg = Config.get_color("safe_zone", "#36E38E")
        normal_fg = Config.get_color("main_t1", "#D6D3C7")
        cb_border = Config.get_color("border_1", "#444444")
        self.setStyleSheet(f"""
            Node_RichChoiceCard {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: {NODE_CARD_RADIUS}px;
            }}
            Node_RichChoiceCard:hover {{
                background: {hover_bg};
                border-color: {accent};
            }}
            Node_RichChoiceCard[selected="true"] {{
                background: {sel_bg};
                border-color: {sel_bg};
            }}
            Node_RichChoiceCard QLabel {{
                background: transparent;
                color: {normal_fg};
            }}
            Node_RichChoiceCard[selected="true"] QLabel {{
                color: {sel_fg};
            }}
            Node_RichChoiceCard QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid {cb_border};
                border-radius: 3px;
                background: transparent;
            }}
            Node_RichChoiceCard QCheckBox::indicator:checked {{
                background: {check_bg};
                border-color: {check_bg};
            }}
        """)


# =============================================================================
# Node_RichChoicePopup —— 选单 popup 容器
# =============================================================================

class Node_RichChoicePopup(QFrame):
    """frameless popup 容器 / Frameless popup hosting Node_RichChoiceCards.

    DEMO 行 2374-2556。锚右 +2px、min 320 宽、max 320 高、可滚、QSS 圆角 8px。
    """

    selection_changed = pyqtSignal(list)
    selection_committed = pyqtSignal(str)  # 单选模式：选中即 emit + 关闭

    def __init__(
        self,
        choices: Sequence[str],
        *,
        meta_map: Optional[Dict[str, Dict[str, str]]] = None,
        leading_mode: str = "checkbox",
        multi_select: bool = False,
        current: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint,
        )
        self._multi = multi_select
        self._leading_mode = leading_mode
        self._meta_map = dict(meta_map or {})
        self._cards: List[Node_RichChoiceCard] = []
        self._selection: List[str] = []
        if current is not None:
            self._selection = list(current) if isinstance(current, (list, tuple)) else [str(current)]

        self.setMinimumWidth(NODE_POPUP_MIN_W)
        self.setMaximumHeight(NODE_POPUP_MAX_H)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget(scroll)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)

        for c in choices:
            meta = self._meta_map.get(c, {})
            card = Node_RichChoiceCard(
                c,
                title=meta.get("title", c.replace("_", " ").title()),
                desc=meta.get("desc", ""),
                leading_mode=leading_mode,
                selected=(c in self._selection),
                parent=host,
            )
            card.clicked.connect(self._on_card_clicked)
            host_layout.addWidget(card)
            self._cards.append(card)
        host_layout.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll)

        self.refresh_style()

    def _on_card_clicked(self, choice: str) -> None:
        if self._multi:
            if choice in self._selection:
                self._selection.remove(choice)
            else:
                self._selection.append(choice)
            for c in self._cards:
                c.set_selected(c.choice in self._selection)
            self.selection_changed.emit(list(self._selection))
        else:
            self._selection = [choice]
            for c in self._cards:
                c.set_selected(c.choice == choice)
            self.selection_committed.emit(choice)
            self.close()

    def refresh_style(self) -> None:
        bg = Config.get_color("bg_1", "#1c1c1c")
        border = Config.get_color("border_2", "#3a3a3a")
        self.setStyleSheet(f"""
            Node_RichChoicePopup {{
                background: {bg};
                border: 1px solid {border};
                border-radius: {NODE_CARD_RADIUS}px;
            }}
        """)


# =============================================================================
# Node_RichChoicePicker —— 触发按钮（单击弹 popup）
# =============================================================================

class Node_RichChoicePicker(QWidget):
    """选单按钮 / Picker button that opens Node_RichChoicePopup.

    DEMO 行 2558-2752。比 Qt QComboBox 更紧凑：134×24，文字省略号截断。
    内部用 LaviButton 做可点击区域；popup 复用 Node_RichChoicePopup。
    """

    currentTextChanged = pyqtSignal(str)
    selectionChanged = pyqtSignal(list)

    def __init__(
        self,
        choices: Sequence[Any],
        *,
        current: Any = None,
        meta_map: Optional[Dict[str, Dict[str, str]]] = None,
        leading_mode: str = "checkbox",
        multi_select: bool = False,
        choices_provider: Optional[Callable[[], tuple]] = None,
        widget_id: str = "node.picker",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._choices = [str(c) for c in choices]
        self._current: List[str] = []
        if current is not None:
            self._current = list(current) if isinstance(current, (list, tuple)) else [str(current)]
        self._meta_map = dict(meta_map or {})
        self._leading_mode = leading_mode
        self._multi = multi_select
        self._choices_provider = choices_provider

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # 触发按钮（LaviButton）
        self._button = setButton(
            widget_id,
            NODE_PICKER_W,
            NODE_PICKER_H,
            kind="border",
            spec="none",
            default=self._format_label(),
            parent=self,
        )
        self._button.clicked.connect(self._on_button_clicked)
        layout.addWidget(self._button)

        self.setMaximumWidth(NODE_PICKER_W)

    def currentText(self) -> str:
        return self._current[0] if self._current else ""

    def currentSelection(self) -> List[str]:
        return list(self._current)

    def setCurrent(self, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            self._current = [str(v) for v in value]
        else:
            self._current = [str(value)] if value else []
        self._button.setText(self._format_label())

    def _format_label(self) -> str:
        if self._multi:
            n = len(self._current)
            return f"{n} selected" if n else "Select…"
        return self._current[0] if self._current else "Select…"

    def _on_button_clicked(self) -> None:
        # 动态刷新 choices（DEMO 的 choices_provider 模式）
        if self._choices_provider is not None:
            try:
                new_choices, new_meta = self._choices_provider()
                self._choices = [str(c) for c in new_choices]
                self._meta_map = dict(new_meta or {})
            except Exception:
                pass
        popup = Node_RichChoicePopup(
            self._choices,
            meta_map=self._meta_map,
            leading_mode=self._leading_mode,
            multi_select=self._multi,
            current=self._current,
            parent=self,
        )
        popup.selection_changed.connect(self._on_multi_changed)
        popup.selection_committed.connect(self._on_single_committed)
        # 锚点：button 右下 +2px
        anchor = self._button.mapToGlobal(self._button.rect().topRight())
        popup.move(anchor.x() + 2, anchor.y())
        popup.show()

    def _on_multi_changed(self, sel: list) -> None:
        self._current = [str(s) for s in sel]
        self._button.setText(self._format_label())
        self.selectionChanged.emit(list(self._current))

    def _on_single_committed(self, value: str) -> None:
        self._current = [value]
        self._button.setText(self._format_label())
        self.currentTextChanged.emit(value)


# =============================================================================
# 子类预设 —— DEMO 把不同节点的 picker 命名好让代码更清楚
# =============================================================================

class Node_TaskTypePicker(Node_RichChoicePicker):
    """task type 选择（icon mode）/ Task type picker. DEMO 行 2755-2764."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "icon")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.task_type")
        super().__init__(choices, **kw)


class Node_ObsComponentsPicker(Node_RichChoicePicker):
    """observation 组件多选（checkbox mode）. DEMO 行 2767-2782."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", True)
        kw.setdefault("widget_id", "node.picker.obs_components")
        super().__init__(choices, **kw)


class Node_DropdownChoicePicker(Node_RichChoicePicker):
    """通用单选下拉（checkbox mode 单选）. DEMO 行 2785-2795."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.dropdown")
        super().__init__(choices, **kw)


class Node_SceneTypePicker(Node_RichChoicePicker):
    """scene 选择（PlayGround 节点用）. DEMO 行 2798-2855."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.scene_type")
        super().__init__(choices, **kw)


class Node_ExportBundlePicker(Node_RichChoicePicker):
    """export 目标选择. DEMO 行 3943+."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.export_bundle")
        super().__init__(choices, **kw)


class Node_StartPointChoicePicker(Node_RichChoicePicker):
    """start_point 选择（training_motion 节点用）. DEMO 行 4034+."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.start_point")
        super().__init__(choices, **kw)


class Node_MotionLibraryPicker(Node_RichChoicePicker):
    """motion 资产选择. DEMO 行 5410+."""

    def __init__(self, choices, **kw):
        kw.setdefault("leading_mode", "checkbox")
        kw.setdefault("multi_select", False)
        kw.setdefault("widget_id", "node.picker.motion_library")
        super().__init__(choices, **kw)


# =============================================================================
# Node_MainPicker —— DEMO node_ui_rows.py:1352-1436 的紧凑选单变体
# =============================================================================

class Node_MainPicker(QWidget):
    """combo 隐藏箭头 + 方块触发 ▼ 的紧凑版选单 / Compact picker.

    DEMO 行 1352-1436（``node_ui_rows.py``）。视觉与 Node_RichChoicePicker 不同：
    - 主体是 QComboBox（隐藏 native ::drop-down）
    - 右侧 LaviButton（▼ 方块）作为弹出触发
    - 主要给 Behavior 节点用
    """

    currentIndexChanged = pyqtSignal(int)

    def __init__(
        self,
        items: Sequence[str],
        *,
        widget_id: str = "node.main_picker",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        from unitport_sdk.ui import LaviComboBox  # 延迟避免循环
        self._combo = LaviComboBox(list(items), parent=self)
        self._combo.setStyleSheet(self._combo.styleSheet() + " QComboBox::drop-down { width: 0px; }")
        self._combo.currentIndexChanged.connect(self.currentIndexChanged.emit)
        layout.addWidget(self._combo, 1)

        self._trigger = setButton(
            widget_id + ".trigger",
            20,
            20,
            kind="border",
            spec="none",
            default="▼",
            parent=self,
        )
        self._trigger.clicked.connect(self._combo.showPopup)
        layout.addWidget(self._trigger)


__all__ = [
    "Node_RichChoiceCard",
    "Node_RichChoicePopup",
    "Node_RichChoicePicker",
    "Node_TaskTypePicker",
    "Node_ObsComponentsPicker",
    "Node_DropdownChoicePicker",
    "Node_SceneTypePicker",
    "Node_ExportBundlePicker",
    "Node_StartPointChoicePicker",
    "Node_MotionLibraryPicker",
    "Node_MainPicker",
]
