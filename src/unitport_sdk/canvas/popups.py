"""unitport_sdk.canvas.popups — Node 用 frameless popup 组件 / Frameless popup widgets.

ParamRow 触发就地编辑时**绝不**弹 modal QDialog——而是按 DEMO 行为弹一个
``Qt.WindowType.Popup`` 的 frameless 面板，紧贴节点行右侧。点击外部即关闭，
回车提交。视觉与 DEMO ``DataInput`` / ``_TextInputPopup`` / ``_RichChoicePopup``
一致。

为什么不是 QDialog：
- modal 会让画布失焦、覆盖节点上下文，破坏"节点旁就地编辑"的心智模型
- 系统标题栏与节点视觉断层
- 用户实际反馈：DEMO 的体验是"卡片在节点旁展开"，QDialog 是错误模式

公开 class（统一 ``Node_`` 前缀，归属 SDK 唯一来源）：
    Node_AnchoredPopup   — frameless popup 基类（``_MINIMAL_STYLE`` 子类开关）
    Node_DataInput       — 单行 [LineEdit][✓][✗] 输入 popup（string / int / float 三态）
    Node_CodeEditorPopup — 多行 code/JSON 编辑 popup（盒式 + LaviCodeEditor + OK/Cancel）
    Node_EditorPopupHost — 通用 widget 容器 popup（任意 Node_* widget + OK/Cancel）

DEMO 对应：
    DataInput（training_node_items.py:932-1076）             → Node_DataInput
    _AnchoredPopup（application/ui/canvas/popups.py 历史）   → Node_AnchoredPopup
    _CodeAnchoredPopup（同上）                               → Node_CodeEditorPopup
    _WidgetHostPopup（同上）                                 → Node_EditorPopupHost

所有 popup 都：
1. ``Qt.WindowType.Popup`` —— 点外部自动关
2. ``WA_DeleteOnClose=True`` —— 关闭即销毁
3. 主题走 ``Config.get_color`` 的 ``canvas_widget_*`` 槽
4. ``on_commit(value)`` 回调（非阻塞）；取消 → 不调
5. ``show_at(anchor, row_h)`` 跟随 view 缩放（DEMO ``DataInput.show_at`` 公式），
   末尾含越屏校正
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QDoubleValidator, QFont, QIcon, QIntValidator, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk.sys import Assets, Config, tr
from unitport_sdk.ui import setButton

from .widgets import LaviCodeEditor, setCodeEditor


# =============================================================================
# 几何 / 字体常量
# =============================================================================

POPUP_INPUT_H = 26  # 简约 popup 默认初始行高（show_at 会用 row_h 覆盖）


# =============================================================================
# 私有 helper —— SVG icon 挂载
# =============================================================================
# 函数命名允许 ``_`` 前缀（用户规则只约束 class）。

def _set_svg_icon(btn: QPushButton, stem: str) -> None:
    """在 QPushButton 上挂 SVG icon（无 stem → 跳过）."""
    try:
        p = Assets.find_icon(stem)
    except Exception:
        p = None
    if p is not None:
        btn.setIcon(QIcon(str(p)))


# =============================================================================
# Node_AnchoredPopup —— frameless popup 基类
# =============================================================================

class Node_AnchoredPopup(QFrame):
    """frameless 弹窗基类 / Frameless popup base.

    ``Qt.WindowType.Popup`` 让点击外部自动 close（与 QMenu 同一行为）。
    子类决定是否画外框：
      - 简约（DataInput / TextInput）：无 bg / 无 border / 透明
      - 盒式（Code / WidgetHost）：有 bg + 圆角 + border

    取代 ``application/ui/canvas/popups.py:_AnchoredPopup``（历史违规位置）。
    """

    _MINIMAL_STYLE = False  # True: 透明无边；False: 盒式

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        # 显式去掉 QFrame 的默认 frame 形状（PyQt 某些版本在 Popup 窗类下会画一
        # 个 1px 的窗框，这里强制清零）。
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(0)
        self.setMidLineWidth(0)
        self.setContentsMargins(0, 0, 0, 0)
        if self._MINIMAL_STYLE:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAutoFillBackground(False)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def show_at(self, anchor: QPoint, row_h: int = 0) -> None:
        """定位到 anchor（行右上角全局坐标）并显示.

        关键：**不要**调 ``activateWindow()`` / ``raise_()``——它们会破坏
        ``Qt.WindowType.Popup`` 的内部 popup grab（导致点击外部不再 auto-close）。
        DEMO 同等做法见 ``training_node_items.py:1185``（DataInput.show_at）。
        """
        self.move(anchor)
        self.show()
        # 越屏校正（show 后 size 才是真实值）
        screen = self.screen() if hasattr(self, "screen") else None
        if screen is not None:
            avail = screen.availableGeometry()
            x, y = self.x(), self.y()
            new_x, new_y = x, y
            if x + self.width() > avail.right():
                new_x = max(avail.left(), avail.right() - self.width() - 2)
            if y + self.height() > avail.bottom():
                new_y = max(avail.top(), avail.bottom() - self.height() - 2)
            if (new_x, new_y) != (x, y):
                self.move(new_x, new_y)

    def _apply_palette(self) -> None:
        # 仅盒式样式（_MINIMAL_STYLE=False）有 bg/border；简约样式什么都不画。
        if self._MINIMAL_STYLE:
            self.setObjectName("anchored_popup_minimal")
            self.setStyleSheet(
                "QFrame#anchored_popup_minimal {"
                "  background: transparent;"
                "  border: none;"
                "}"
            )
            return
        bg = Config.get_color("bg_1", "#1c1c1c")
        border = Config.get_color("border_2", "#3a3a3a")
        self.setObjectName("anchored_popup")
        self.setStyleSheet(
            f"QFrame#anchored_popup {{"
            f"  background: {bg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 6px;"
            f"}}"
        )


# =============================================================================
# Node_DataInput —— 单行 [LineEdit][✓][✗] 输入 popup（DEMO DataInput 本体）
# =============================================================================

class Node_DataInput(QFrame):
    """DEMO ``DataInput`` 直接移植 / Direct port of DEMO ``DataInput``.

    DEMO 源：``D:/Unitport/EXE/DEMO/bin/nodes/training_node_items.py:932-1076``。
    结构：``[QLineEdit][QPushButton ✓][QPushButton ✗] addStretch(1)`` 左对齐。
    popup 自身 ``#dataInputPopup { background: transparent; border: none; }``，
    所有可见样式都画在子 widget 上（QLineEdit 自己的 border、按钮自己的 border）；
    两个按钮共用同一个 selector ``QPushButton[dataInputBtn="true"]``——视觉
    保证 100% 一致。

    ✓/✗ 按钮的字符 glyph 在 RELEASE 改用 ``icon_yes.svg`` / ``icon_no.svg``
    （``Assets.find_icon``）。其他全部与 DEMO 行为/几何一致：
        - ``setSpacing(1)``
        - ``addStretch(1)`` 末尾左对齐
        - ``show_at(pos, row_h)`` 跟随 zoom 缩放
        - ``selectAll()`` 显示后聚焦+全选
        - 无效输入 → line edit 红边

    NOTE：刻意 **不** 继承 ``Node_AnchoredPopup``。后者会在构造里调
    ``setFrameShape/Shadow/LineWidth/setAutoFillBackground`` 等 QFrame 塑形动作；
    在 PyQt6 + ``Qt.WindowType.Popup`` 下这些会让 QFrame 仍保留 1px 默认窗框，
    与 DEMO 完全无边框的视觉不符。直接继承 QFrame 并按 DEMO 顺序铺开窗体属性，
    再让 QSS ``#dataInputPopup { border: none }`` 命中 objectName，效果与 DEMO 一致。

    取代历史违规：
      - ``application/ui/canvas/popups.py:_DataInputPopup``
      - ``application/ui/canvas/popups.py:_TextInputAnchoredPopup``
      - ``application/ui/canvas/popups.py:_NumberInputAnchoredPopup``
      - 旧 SDK ``unitport_sdk/canvas/inputs.py:Node_TextInputPopup``
    """

    def __init__(
        self,
        value: Any = 0,
        *,
        dtype: str = "string",
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        decimals: int = 6,
        placeholder: str = "",
        on_commit: Optional[Callable[[Any], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        if dtype not in ("string", "int", "float"):
            dtype = "string"
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("dataInputPopup")  # 与 DEMO 同名，QSS selector 一致
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        # 显式压平 QFrame 自身的 frame 绘制 —— PyQt6 在 Popup 窗类下，即使 QSS
        # `border: none`，QFrame 的 paintEvent 里仍会按 frameShape/lineWidth 画一
        # 圈 1px 默认窗框。下面 4 行强制清零，DEMO PySide6 的默认值不同所以不需要。
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setLineWidth(0)
        self.setMidLineWidth(0)
        self.setAutoFillBackground(False)
        self.setContentsMargins(0, 0, 0, 0)
        self._on_commit = on_commit
        self._committed = False
        self._dtype = dtype
        self._decimals = int(decimals)
        self._min = minimum
        self._max = maximum

        bg = Config.get_color("bg_1", "#1A1A1A")
        fg = Config.get_color("main_t1", "#cccccc")
        border = Config.get_color("border_2", "#3d3d3d")
        btn_bg = Config.get_color("btn_1", "#252525")
        hover = Config.get_color("hover_1", "#2e2e2e")
        err = Config.get_color("canvas_safety_error", "#f87171")

        self.setStyleSheet(
            f"""
            QFrame#dataInputPopup {{
                background: transparent;
                background-color: transparent;
                border: 0;
                border-style: none;
                border-width: 0;
            }}
            QLineEdit[dataInput="true"] {{
                background: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 0px 4px;
            }}
            QLineEdit[dataInput="true"][invalid="true"] {{
                border-color: {err};
            }}
            QPushButton[dataInputBtn="true"] {{
                background: {btn_bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 0px 2px;
            }}
            QPushButton[dataInputBtn="true"]:hover {{
                background: {hover};
            }}
            """
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        # 让 layout 自身严格收缩到子控件之和——避免任何额外 1px 余地被 QFrame
        # paintEvent 当成"窗框"画出来。配合下方删除的 addStretch + _apply_scale
        # 末尾的 setFixedSize(精确总和) 三重夹紧，popup 边界完全贴合内容。
        row.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._row_layout = row

        # ---- LineEdit -----------------------------------------------------
        self._input = QLineEdit(self)
        self._input.setProperty("dataInput", True)
        if self._dtype == "string":
            self._input.setText(str(value or ""))
            if placeholder:
                self._input.setPlaceholderText(placeholder)
        elif self._dtype == "int":
            lo = int(minimum) if minimum is not None else -2**31
            hi = int(maximum) if maximum is not None else 2**31 - 1
            self._input.setValidator(QIntValidator(lo, hi, self))
            try:
                self._input.setText(str(int(value)))
            except (TypeError, ValueError):
                self._input.setText("0")
        else:  # float
            lo = float(minimum) if minimum is not None else -1e15
            hi = float(maximum) if maximum is not None else 1e15
            self._input.setValidator(QDoubleValidator(lo, hi, self._decimals, self))
            try:
                self._input.setText(f"{float(value):.6g}")
            except (TypeError, ValueError):
                self._input.setText("0")
        self._input.setFixedWidth(96)
        self._input.selectAll()
        self._input.returnPressed.connect(self._accept_if_valid)
        row.addWidget(self._input, 0)

        # ---- ✓ 按钮 -------------------------------------------------------
        self._ok_btn = QPushButton("", self)
        self._ok_btn.setProperty("dataInputBtn", True)
        self._ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ok_btn.clicked.connect(self._accept_if_valid)
        _set_svg_icon(self._ok_btn, "icon_yes")
        row.addWidget(self._ok_btn, 0)

        # ---- ✗ 按钮 -------------------------------------------------------
        self._cancel_btn = QPushButton("", self)
        self._cancel_btn.setProperty("dataInputBtn", True)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self.close)
        _set_svg_icon(self._cancel_btn, "icon_no")
        row.addWidget(self._cancel_btn, 0)

        # NOTE：DEMO 末尾有 addStretch(1) 让 3 个组件左对齐；RELEASE 改成
        # 严格固定 layout 尺寸（SetFixedSize + 精确算 total_w），让 popup 外框
        # 完全贴合 [LineEdit][✓][✗] 的总宽——这才能在 PyQt6 Popup 窗类下
        # 视觉上彻底"消除"边框。

        # 初始尺寸；show_at 会用 view 缩放后的 row_h 覆盖
        self._apply_scale(POPUP_INPUT_H)

    # -- DEMO 同名公式 ----------------------------------------------------

    def _apply_scale(self, row_h: int) -> None:
        """跟随 view 缩放：参数全部按 row_h 等比缩放（DEMO show_at 公式）.

        DEMO ``DataInput.show_at`` (training_node_items.py:1048-1066)。

        RELEASE 额外约束：popup 外框严格 = LineEdit 宽 + 2 × 按钮宽 + 2 × spacing
        + 0 × margins。让 QFrame 没有"剩余像素"可画窗框，视觉上彻底贴合内容。
        """
        h = max(24, int(row_h))
        font_px = max(12, round(h * 0.45))
        f = QFont()
        f.setPixelSize(font_px)
        input_w = max(96, round(h * 3.2))
        self._input.setFixedSize(input_w, h)
        self._input.setFont(f)
        self._ok_btn.setFixedSize(h, h)
        self._ok_btn.setFont(f)
        self._cancel_btn.setFixedSize(h, h)
        self._cancel_btn.setFont(f)
        ico_sz = QSize(int(h * 0.6), int(h * 0.6))
        self._ok_btn.setIconSize(ico_sz)
        self._cancel_btn.setIconSize(ico_sz)

        # popup 总宽 = LineEdit + ✓ + ✗ + 2 个 spacing 间隙；高度直接 = h
        # （contents margins 已 setContentsMargins(0,0,0,0)，spacing=1 见构造）
        spacing = self._row_layout.spacing()
        total_w = input_w + h + h + spacing * 2
        self.setFixedSize(total_w, h)

    def show_at(self, anchor: QPoint, row_h: int = 0) -> None:
        """Anchor + show. 等价 DEMO ``DataInput.show_at``，外加越屏校正.

        DEMO 同步实现：``training_node_items.py:1048-1066``。RELEASE 在末尾
        多一段 screen.availableGeometry 越屏 reposition —— 通用 popup grab
        的边缘兜底，DEMO 没有这段是因为它默认锚点不会越屏。
        """
        if row_h:
            self._apply_scale(row_h)
        # 先 move 后 show（与 SDK Node_RichChoicePicker._on_button_clicked 一致）；
        # **不要** 调 activateWindow() / raise_()——它们会破坏 Qt.WindowType.Popup
        # 的 popup grab，导致点击外部不再 auto-close。
        self.move(anchor)
        self.show()
        screen = self.screen() if hasattr(self, "screen") else None
        if screen is not None:
            avail = screen.availableGeometry()
            x, y = self.x(), self.y()
            new_x, new_y = x, y
            if x + self.width() > avail.right():
                new_x = max(avail.left(), avail.right() - self.width() - 2)
            if y + self.height() > avail.bottom():
                new_y = max(avail.top(), avail.bottom() - self.height() - 2)
            if (new_x, new_y) != (x, y):
                self.move(new_x, new_y)
        self._input.setFocus()
        self._input.selectAll()

    # -- 提交逻辑（与 DEMO _accept_if_valid 一致）-------------------------

    def _accept_if_valid(self) -> None:
        if self._committed:
            return
        raw = (self._input.text() or "").strip()
        if self._dtype == "string":
            self._mark_invalid(False)
            self._committed = True
            if self._on_commit is not None:
                self._on_commit(raw)
            self.close()
            return
        try:
            num: Any = int(raw) if self._dtype == "int" else float(raw)
        except (TypeError, ValueError):
            self._mark_invalid(True)
            return
        if self._min is not None and num < self._min:
            self._mark_invalid(True)
            return
        if self._max is not None and num > self._max:
            self._mark_invalid(True)
            return
        self._mark_invalid(False)
        self._committed = True
        if self._on_commit is not None:
            self._on_commit(num)
        self.close()

    def _mark_invalid(self, invalid: bool) -> None:
        self._input.setProperty("invalid", "true" if invalid else "false")
        self._input.style().unpolish(self._input)
        self._input.style().polish(self._input)
        self._input.setFocus()
        self._input.selectAll()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._accept_if_valid()
            return
        super().keyPressEvent(event)


# =============================================================================
# Node_CodeEditorPopup —— 多行代码 / JSON 编辑 popup
# =============================================================================

class Node_CodeEditorPopup(Node_AnchoredPopup):
    """多行代码 / JSON 编辑 popup / Multi-line code editor popup.

    比单行 popup 大，但仍是 frameless ``Qt.WindowType.Popup``——点外关。
    取代 ``application/ui/canvas/popups.py:_CodeAnchoredPopup``。
    """

    _MIN_W = 540
    _MIN_H = 360

    def __init__(
        self,
        text: str = "",
        *,
        language: str = "json",
        validate_json: bool = False,
        on_commit: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_commit = on_commit
        self._committed = False
        self._validate_json = bool(validate_json)
        self.setMinimumSize(self._MIN_W, self._MIN_H)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._editor: LaviCodeEditor = setCodeEditor(text=text, language=language, parent=self)
        layout.addWidget(self._editor, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        cancel = setButton(
            "popup.code.cancel", 76, 26, kind="border", spec="none",
            default=tr("dialog.cancel", "Cancel"),
        )
        ok = setButton(
            "popup.code.ok", 76, 26, kind="normal", spec="save",
            default=tr("dialog.ok", "OK"),
        )
        cancel.clicked.connect(self.close)
        ok.clicked.connect(self._commit)
        bar.addWidget(cancel)
        bar.addWidget(ok)
        layout.addLayout(bar)

        self._apply_palette()
        self._editor.setFocus()

    def _commit(self) -> None:
        if self._committed:
            return
        text = self._editor.toPlainText()
        if self._validate_json:
            try:
                json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # 红边/标题提示由 widget 自身负责；这里阻止提交
                return
        self._committed = True
        if self._on_commit is not None:
            self._on_commit(text)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


# =============================================================================
# Node_EditorPopupHost —— 通用 widget 容器 popup
# =============================================================================

class Node_EditorPopupHost(Node_AnchoredPopup):
    """把任意 Node_* QWidget 装进 popup / Generic Node_* host popup.

    用于 RegistryItemListRow / TrainingItemsRow / StageEditorRow / BufferSizeRow
    等需要复杂 SDK widget 的场景。在 widget 下方留一栏 OK/Cancel。

    取代 ``application/ui/canvas/popups.py:_WidgetHostPopup``。
    """

    def __init__(
        self,
        body: QWidget,
        *,
        value_getter: Callable[[QWidget], Any],
        on_commit: Optional[Callable[[Any], None]] = None,
        min_width: int = 360,
        min_height: int = 240,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._body = body
        self._value_getter = value_getter
        self._on_commit = on_commit
        self._committed = False
        self.setMinimumSize(min_width, min_height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        body.setParent(self)
        layout.addWidget(body, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        cancel = setButton(
            "popup.widget.cancel", 76, 26, kind="border", spec="none",
            default=tr("dialog.cancel", "Cancel"),
        )
        ok = setButton(
            "popup.widget.ok", 76, 26, kind="normal", spec="save",
            default=tr("dialog.ok", "OK"),
        )
        cancel.clicked.connect(self.close)
        ok.clicked.connect(self._commit)
        bar.addWidget(cancel)
        bar.addWidget(ok)
        layout.addLayout(bar)

        self._apply_palette()

    def _commit(self) -> None:
        if self._committed:
            return
        try:
            v = self._value_getter(self._body)
        except Exception:
            v = None
        self._committed = True
        if v is not None and self._on_commit is not None:
            self._on_commit(v)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


__all__ = [
    "POPUP_INPUT_H",
    "Node_AnchoredPopup",
    "Node_DataInput",
    "Node_CodeEditorPopup",
    "Node_EditorPopupHost",
]
