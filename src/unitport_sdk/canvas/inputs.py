"""unitport_sdk.canvas.inputs — Node 输入 / 编辑触发 widget.

DEMO 对应（``bin/nodes/training_node_items.py`` / ``node_ui_rows.py``）：
    _IndexButton        → Node_IndexButton
    _NodeInputer        → Node_NodeInputer
    _CodeEdit           → Node_CodeEdit
    _ModuleValueButton  → Node_ModuleValueButton

约束：所有 UI class 以 ``Node_`` 开头；内部一律组合 SDK ``LaviButton`` /
``LaviLabel`` / ``LaviLineEdit`` / ``LaviSpinBox`` / ``LaviDoubleSpinBox`` /
``LaviCodeEditor``。

NOTE：``Node_DataInput`` （frameless [LineEdit][✓][✗] popup）和其它 popup 类
统一存放在 ``unitport_sdk/canvas/popups.py``，不在本文件。
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk.sys import Config, tr
from unitport_sdk.ui import setButton, setText

from .widgets import (
    LaviCodeEditor,
    LaviDoubleSpinBox,
    LaviSpinBox,
    setCodeEditor,
    setDoubleSpinBox,
    setLineEdit,
    setSpinBox,
)


# =============================================================================
# 几何常量
# =============================================================================

NODE_INPUT_H = 24
NODE_INPUT_MIN_W = 80
NODE_BUTTON_SQ = 22


# =============================================================================
# Node_IndexButton —— ± LaviButton 包裹 LaviSpinBox
# =============================================================================

class Node_IndexButton(QWidget):
    """整数索引按钮 / Index button (LaviSpinBox + ± LaviButtons).

    布局：[ − 22×22 ] [ LaviSpinBox ] [ + 22×22 ].
    """

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        value: int = 0,
        *,
        minimum: int = 0,
        maximum: int = 1_000_000,
        widget_id: str = "node.index",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._dec = setButton(
            f"{widget_id}.dec", NODE_BUTTON_SQ, NODE_BUTTON_SQ,
            kind="border", spec="none", default="−", parent=self,
        )
        self._spin: LaviSpinBox = setSpinBox(
            value, minimum=minimum, maximum=maximum, step=1, parent=self,
        )
        self._spin.setMinimumWidth(NODE_INPUT_MIN_W)
        self._inc = setButton(
            f"{widget_id}.inc", NODE_BUTTON_SQ, NODE_BUTTON_SQ,
            kind="border", spec="none", default="+", parent=self,
        )

        layout.addWidget(self._dec)
        layout.addWidget(self._spin, 1)
        layout.addWidget(self._inc)

        self._dec.clicked.connect(lambda: self._spin.setValue(self._spin.value() - 1))
        self._inc.clicked.connect(lambda: self._spin.setValue(self._spin.value() + 1))
        self._spin.valueChanged.connect(self.valueChanged.emit)

    def value(self) -> int:
        return self._spin.value()

    def setValue(self, v: int) -> None:
        self._spin.setValue(int(v))

    def refresh_style(self) -> None:
        self._spin.refresh_style()


# =============================================================================
# Node_NodeInputer —— 按 dtype 切换 LineEdit / SpinBox / DoubleSpinBox
# =============================================================================

class Node_NodeInputer(QWidget):
    """按 dtype 自动切换的输入条 / Auto-typed inputer (string / int / float).

    DEMO 行（``_NodeInputer``）。dtype 在构造时确定，运行时不切换。

    信号：
        valueChanged(object) —— 内部值变（int/float/str）
    """

    valueChanged = pyqtSignal(object)

    def __init__(
        self,
        dtype: str = "string",
        value: Any = None,
        *,
        placeholder: str = "",
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        decimals: int = 6,
        widget_id: str = "node.inputer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._dtype = dtype if dtype in ("string", "int", "float") else "string"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if self._dtype == "int":
            self._inner = setSpinBox(
                int(value or 0),
                minimum=int(minimum) if minimum is not None else None,
                maximum=int(maximum) if maximum is not None else None,
                parent=self,
            )
            self._inner.valueChanged.connect(self.valueChanged.emit)
        elif self._dtype == "float":
            self._inner = setDoubleSpinBox(
                float(value or 0.0),
                minimum=minimum, maximum=maximum, decimals=decimals, parent=self,
            )
            self._inner.valueChanged.connect(self.valueChanged.emit)
        else:
            self._inner = setLineEdit(
                str(value or ""), placeholder=placeholder, parent=self,
            )
            self._inner.textChanged.connect(self.valueChanged.emit)

        self._inner.setMinimumHeight(NODE_INPUT_H)
        layout.addWidget(self._inner)

    def value(self) -> Any:
        if self._dtype in ("int", "float"):
            return self._inner.value()
        return self._inner.text()

    def setValue(self, v: Any) -> None:
        if self._dtype == "int":
            self._inner.setValue(int(v))
        elif self._dtype == "float":
            self._inner.setValue(float(v))
        else:
            self._inner.setText(str(v))

    def refresh_style(self) -> None:
        self._inner.refresh_style()


# =============================================================================
# Node_CodeEdit —— 触发 LaviButton + 弹窗 LaviCodeEditor
# =============================================================================

class Node_CodeEdit(QWidget):
    """JSON / 代码编辑触发按钮 / Code edit trigger button.

    DEMO 行（``_CodeEdit``）。点击触发按钮 → 弹模态 ``QDialog`` 内置
    ``LaviCodeEditor``，OK/Cancel 走 SDK ``setButton``。

    信号：
        textCommitted(str) —— OK 后的文本
    """

    textCommitted = pyqtSignal(str)

    def __init__(
        self,
        value: str = "",
        *,
        language: str = "json",
        title: str = "",
        widget_id: str = "node.code_edit",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._value = value
        self._language = language
        self._title = title or tr(f"{widget_id}.title", "Edit Code")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._trigger = setButton(
            widget_id, 96, NODE_INPUT_H,
            kind="border", spec="none",
            default=tr(f"{widget_id}.trigger", "Edit…"),
            parent=self,
        )
        self._trigger.clicked.connect(self._open_dialog)
        layout.addWidget(self._trigger)

    def text(self) -> str:
        return self._value

    def setText(self, value: str) -> None:
        self._value = str(value)

    def _open_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._title)
        dlg.setModal(True)
        dlg.setMinimumSize(560, 380)

        bg = Config.get_color("bg_1", "#1c1c1c")
        fg = Config.get_color("main_t1", "#D6D3C7")
        dlg.setStyleSheet(f"QDialog {{ background: {bg}; color: {fg}; }}")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(setText(
            "node.code_edit.header",
            default=f"{self._title}    [{self._language}]",
            kind="title",
        ))

        editor: LaviCodeEditor = setCodeEditor(self._value, language=self._language, parent=dlg)
        layout.addWidget(editor, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        cancel = setButton(
            "dialog.cancel", 96, 30,
            kind="border", spec="none",
            default=tr("dialog.cancel", "Cancel"),
        )
        ok = setButton(
            "dialog.ok", 96, 30,
            kind="normal", spec="save",
            default=tr("dialog.ok", "OK"),
        )
        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)
        bar.addWidget(cancel)
        bar.addWidget(ok)
        layout.addLayout(bar)

        if dlg.exec():
            self._value = editor.toPlainText()
            self.textCommitted.emit(self._value)

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_ModuleValueButton —— 显示当前值的按钮（点击弹 popup）
# =============================================================================

class Node_ModuleValueButton(QWidget):
    """模块值显示按钮 / Module value button.

    DEMO 行（``_ModuleValueButton``）。仅显示当前值，点击由调用方接信号
    决定怎么弹（典型：弹 ``Node_RichChoicePopup`` / ``Node_DataInput``）。

    信号：
        clicked() —— 按钮被点
    """

    clicked = pyqtSignal()

    def __init__(
        self,
        value: str = "",
        *,
        width: int = 120,
        widget_id: str = "node.module_value",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._value = str(value)
        self._button = setButton(
            widget_id, width, NODE_INPUT_H,
            kind="border", spec="none",
            default=self._value or "—",
            parent=self,
        )
        self._button.clicked.connect(self.clicked.emit)
        layout.addWidget(self._button)

    def value(self) -> str:
        return self._value

    def setValue(self, v: str) -> None:
        self._value = str(v)
        self._button.setText(self._value or "—")

    def refresh_style(self) -> None:
        pass


__all__ = [
    "Node_IndexButton",
    "Node_NodeInputer",
    "Node_CodeEdit",
    "Node_ModuleValueButton",
]
