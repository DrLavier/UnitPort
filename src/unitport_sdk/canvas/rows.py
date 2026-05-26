# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.rows — Node 内部行控件 / Node-internal row widgets.

DEMO 对应（``bin/nodes/node_ui_rows.py`` + ``training_node_items.py``）：
    _NodeRow                → Node_NodeRow
    _PortInputRow           → Node_PortInputRow
    _ConditionSet           → Node_ConditionSet
    _OutputSet              → Node_OutputSet
    _InputSetting           → Node_InputSetting
    _ComboSetting           → Node_ComboSetting
    _BehaviorSetting        → Node_BehaviorSetting
    _ScriptSetting          → Node_ScriptSetting
    _ScriptInAndOut         → Node_ScriptInAndOut
    _NodeTableRowsWidget    → Node_NodeTableRowsWidget
    _RegistryModuleRow      → Node_RegistryModuleRow

约束：所有 UI class 以 ``Node_`` 开头，**不允许** ``_`` 私有前缀；内部一律
组合 ``LaviLabel`` (``setText``) / ``LaviButton`` (``setButton``) / ``LaviLineEdit``
等 SDK 部件。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk.sys import Config
from unitport_sdk.ui import setButton, setText

from .inputs import (
    NODE_BUTTON_SQ,
    NODE_INPUT_H,
    Node_CodeEdit,
    Node_IndexButton,
    Node_ModuleValueButton,
    Node_NodeInputer,
)
from .pickers import Node_DropdownChoicePicker, Node_RichChoicePicker
from .sliders import Node_NodeSlider


# =============================================================================
# 几何常量
# =============================================================================

NODE_ROW_H = 28
NODE_ROW_LABEL_W = 96
NODE_ROW_GAP = 6


# =============================================================================
# Node_NodeRow —— 基类，左 LaviLabel + 右 slot widget
# =============================================================================

class Node_NodeRow(QWidget):
    """节点行基类 / Base class for node rows.

    布局：``[LaviLabel : NODE_ROW_LABEL_W][gap][slot widget : 1]``。
    子类 / 调用方通过 ``set_slot(widget)`` 把右侧 slot 替换为任意控件。
    """

    def __init__(
        self,
        label: str,
        *,
        widget_id: str = "node.row",
        slot: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedHeight(NODE_ROW_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(NODE_ROW_GAP)

        self._label = setText(f"{widget_id}.label", default=label, kind="caption")
        self._label.setFixedWidth(NODE_ROW_LABEL_W)
        self._layout.addWidget(self._label)

        self._slot: Optional[QWidget] = None
        if slot is not None:
            self.set_slot(slot)

    def set_slot(self, widget: QWidget) -> None:
        if self._slot is not None:
            self._layout.removeWidget(self._slot)
            self._slot.setParent(None)
            self._slot.deleteLater()
        self._slot = widget
        self._layout.addWidget(widget, 1)

    def slot(self) -> Optional[QWidget]:
        return self._slot

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def refresh_style(self) -> None:
        if hasattr(self._slot, "refresh_style"):
            self._slot.refresh_style()  # type: ignore[union-attr]


# =============================================================================
# Node_PortInputRow —— Node_NodeRow + 左侧端口锚点
# =============================================================================

class Node_PortInputRow(Node_NodeRow):
    """带端口锚点的节点行 / Node row with port anchor on the left.

    端口图元（``Node_TrainingNodePort``）由调用方在 NodeItem 层挂；本类只
    在 boundingRect 左缘留出空位（mark 用 LaviLabel 形式占位即可）。
    """

    portClicked = pyqtSignal()

    def __init__(
        self,
        label: str,
        *,
        port_color: Optional[QColor] = None,
        widget_id: str = "node.port_row",
        slot: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(label, widget_id=widget_id, slot=slot, parent=parent)
        self._port_color = port_color or QColor(
            Config.get_color("checked_1", "#67CCF5")
        )
        # 在最左侧塞一个 12px 占位（端口实际由 NodeItem.paint 时画在外面）
        anchor = QFrame(self)
        anchor.setFixedSize(12, NODE_ROW_H)
        self._layout.insertWidget(0, anchor)
        self._port_anchor = anchor

    @property
    def port_color(self) -> QColor:
        return self._port_color


# =============================================================================
# Node_ConditionSet —— 一组 condition 表单
# =============================================================================

class Node_ConditionSet(QWidget):
    """条件集合编辑器 / Condition set editor.

    每个条件一行：``[Node_DropdownChoicePicker(op)][Node_NodeInputer(rhs)]``，
    底部一个 LaviButton 加行。

    信号：
        conditionsChanged(list)  —— 条件列表 [{"op": str, "value": Any}, ...]
    """

    conditionsChanged = pyqtSignal(list)

    def __init__(
        self,
        ops: Sequence[str] = ("==", "!=", ">", ">=", "<", "<="),
        *,
        widget_id: str = "node.condition_set",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ops = list(ops)
        self._widget_id = widget_id
        self._rows: List[QWidget] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._add_btn = setButton(
            f"{widget_id}.add", 96, NODE_INPUT_H,
            kind="border", spec="add", default="+ Add",
            parent=self,
        )
        self._add_btn.clicked.connect(lambda: self.add_condition())
        self._layout.addWidget(self._add_btn)

    def add_condition(self, op: str = "==", value: Any = "") -> None:
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(NODE_ROW_GAP)
        op_picker = Node_DropdownChoicePicker(self._ops, current=op, parent=row)
        rhs = Node_NodeInputer("string", value, parent=row)
        op_picker.currentTextChanged.connect(lambda *_: self._emit())
        rhs.valueChanged.connect(lambda *_: self._emit())
        h.addWidget(op_picker)
        h.addWidget(rhs, 1)
        self._rows.append(row)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._emit()

    def conditions(self) -> List[dict]:
        result: List[dict] = []
        for row in self._rows:
            picker: Node_DropdownChoicePicker = row.findChild(Node_DropdownChoicePicker)  # type: ignore[assignment]
            rhs: Node_NodeInputer = row.findChild(Node_NodeInputer)  # type: ignore[assignment]
            if picker is not None and rhs is not None:
                result.append({"op": picker.currentText(), "value": rhs.value()})
        return result

    def _emit(self) -> None:
        self.conditionsChanged.emit(self.conditions())

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_OutputSet / Node_InputSetting / Node_ComboSetting —— 简单包装行集合
# =============================================================================

class Node_OutputSet(QWidget):
    """输出端口列表行 / Output port list row.

    每行一个端口名（LaviLabel）+ 删除按钮（LaviButton）。
    """

    portsChanged = pyqtSignal(list)

    def __init__(
        self,
        ports: Sequence[str] = (),
        *,
        widget_id: str = "node.output_set",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._widget_id = widget_id
        self._ports: List[str] = list(ports)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._rebuild()

    def _rebuild(self) -> None:
        while self._layout.count():
            it = self._layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for name in self._ports:
            row = QWidget(self)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(NODE_ROW_GAP)
            h.addWidget(setText(f"{self._widget_id}.{name}", default=name))
            del_btn = setButton(
                f"{self._widget_id}.del.{name}", NODE_BUTTON_SQ, NODE_BUTTON_SQ,
                kind="border", spec="del", default="×", parent=row,
            )
            del_btn.clicked.connect(lambda _=False, p=name: self._remove(p))
            h.addWidget(del_btn)
            self._layout.addWidget(row)

    def _remove(self, name: str) -> None:
        if name in self._ports:
            self._ports.remove(name)
            self._rebuild()
            self.portsChanged.emit(list(self._ports))

    def ports(self) -> List[str]:
        return list(self._ports)

    def setPorts(self, names: Sequence[str]) -> None:
        self._ports = list(names)
        self._rebuild()

    def refresh_style(self) -> None:
        pass


class Node_InputSetting(Node_NodeRow):
    """单输入字段编辑行 / Single input field row (label + Node_NodeInputer)."""

    valueChanged = pyqtSignal(object)

    def __init__(
        self,
        label: str,
        value: Any = "",
        *,
        dtype: str = "string",
        widget_id: str = "node.input_setting",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(label, widget_id=widget_id, parent=parent)
        self._inputer = Node_NodeInputer(dtype, value, widget_id=widget_id, parent=self)
        self._inputer.valueChanged.connect(self.valueChanged.emit)
        self.set_slot(self._inputer)

    def value(self) -> Any:
        return self._inputer.value()

    def setValue(self, v: Any) -> None:
        self._inputer.setValue(v)


class Node_ComboSetting(Node_NodeRow):
    """选单字段行 / Choice field row (label + Node_DropdownChoicePicker)."""

    currentTextChanged = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        choices: Sequence[str],
        *,
        current: Any = None,
        widget_id: str = "node.combo_setting",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(label, widget_id=widget_id, parent=parent)
        self._picker = Node_DropdownChoicePicker(
            choices, current=current, widget_id=widget_id, parent=self,
        )
        self._picker.currentTextChanged.connect(self.currentTextChanged.emit)
        self.set_slot(self._picker)

    def currentText(self) -> str:
        return self._picker.currentText()


class Node_BehaviorSetting(Node_NodeRow):
    """Behavior 节点设置行 / Behavior config row.

    DEMO ``_BehaviorSetting``：左 label，右一组 picker + spin 组合，本期骨架
    暂用 ``Node_DropdownChoicePicker`` 占位；调用方可换 slot。
    """

    def __init__(
        self,
        label: str,
        choices: Sequence[str] = (),
        *,
        widget_id: str = "node.behavior_setting",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(label, widget_id=widget_id, parent=parent)
        if choices:
            picker = Node_DropdownChoicePicker(choices, widget_id=widget_id, parent=self)
            self.set_slot(picker)


class Node_ScriptSetting(Node_NodeRow):
    """脚本字段行 / Script field row (label + Node_CodeEdit trigger)."""

    textCommitted = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        value: str = "",
        *,
        language: str = "python",
        widget_id: str = "node.script_setting",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(label, widget_id=widget_id, parent=parent)
        self._code = Node_CodeEdit(value, language=language, widget_id=widget_id, parent=self)
        self._code.textCommitted.connect(self.textCommitted.emit)
        self.set_slot(self._code)

    def text(self) -> str:
        return self._code.text()


class Node_ScriptInAndOut(QWidget):
    """脚本 in/out 端口对 / Script in/out port pair editor.

    DEMO ``_ScriptInAndOut``：上下两组 ``Node_OutputSet`` —— 顶上是 inputs
    的端口列表，下面是 outputs 的端口列表。
    """

    portsChanged = pyqtSignal(dict)

    def __init__(
        self,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        *,
        widget_id: str = "node.script_in_out",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(setText(f"{widget_id}.inputs", default="Inputs", kind="caption"))
        self._inputs = Node_OutputSet(inputs, widget_id=f"{widget_id}.inputs", parent=self)
        self._inputs.portsChanged.connect(lambda _: self._emit())
        layout.addWidget(self._inputs)

        layout.addWidget(setText(f"{widget_id}.outputs", default="Outputs", kind="caption"))
        self._outputs = Node_OutputSet(outputs, widget_id=f"{widget_id}.outputs", parent=self)
        self._outputs.portsChanged.connect(lambda _: self._emit())
        layout.addWidget(self._outputs)

    def _emit(self) -> None:
        self.portsChanged.emit({
            "inputs": self._inputs.ports(),
            "outputs": self._outputs.ports(),
        })

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_NodeTableRowsWidget —— 多行可增删的表格行集合
# =============================================================================

class Node_NodeTableRowsWidget(QWidget):
    """节点表格行 / Node table rows.

    每行 = ``[LaviLabel][Node_NodeInputer]…[del LaviButton]``；底部 +Add
    LaviButton。column 由 ``columns`` 参数（``(label, dtype)`` 列表）定义。

    信号：
        rowsChanged(list)  —— 表格当前内容（每行一个 dict）
    """

    rowsChanged = pyqtSignal(list)

    def __init__(
        self,
        columns: Sequence[tuple],
        *,
        rows: Sequence[dict] = (),
        widget_id: str = "node.table_rows",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._columns = list(columns)
        self._widget_id = widget_id

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._row_widgets: List[QWidget] = []

        self._add_btn = setButton(
            f"{widget_id}.add", 96, NODE_INPUT_H,
            kind="border", spec="add", default="+ Add",
            parent=self,
        )
        self._add_btn.clicked.connect(lambda: self.add_row())
        self._layout.addWidget(self._add_btn)

        for r in rows:
            self.add_row(r)

    def add_row(self, values: Optional[dict] = None) -> None:
        values = values or {}
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(NODE_ROW_GAP)
        for label, dtype in self._columns:
            h.addWidget(setText(f"{self._widget_id}.{label}", default=label, kind="caption"))
            inp = Node_NodeInputer(dtype, values.get(label, ""),
                                   widget_id=f"{self._widget_id}.{label}", parent=row)
            inp.valueChanged.connect(lambda *_: self._emit())
            inp.setProperty("col_label", label)
            h.addWidget(inp, 1)
        del_btn = setButton(
            f"{self._widget_id}.del", NODE_BUTTON_SQ, NODE_BUTTON_SQ,
            kind="border", spec="del", default="×", parent=row,
        )
        del_btn.clicked.connect(lambda _=False, w=row: self._remove(w))
        h.addWidget(del_btn)
        self._row_widgets.append(row)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._emit()

    def _remove(self, row: QWidget) -> None:
        if row in self._row_widgets:
            self._row_widgets.remove(row)
            row.setParent(None)
            row.deleteLater()
            self._emit()

    def rows(self) -> List[dict]:
        result: List[dict] = []
        for row in self._row_widgets:
            row_dict: dict = {}
            for inp in row.findChildren(Node_NodeInputer):
                key = inp.property("col_label")
                if key:
                    row_dict[str(key)] = inp.value()
            result.append(row_dict)
        return result

    def _emit(self) -> None:
        self.rowsChanged.emit(self.rows())

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_RegistryModuleRow —— 注册模块单行展示 + 编辑入口
# =============================================================================

class Node_RegistryModuleRow(Node_NodeRow):
    """注册模块行 / Registry module row.

    左侧 LaviLabel 显示 module name，右侧 ``Node_ModuleValueButton`` 显示当前
    选中值（点击由调用方接 picker / popup）。
    """

    valueClicked = pyqtSignal()

    def __init__(
        self,
        module_name: str,
        value: str = "",
        *,
        widget_id: str = "node.registry_module",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(module_name, widget_id=widget_id, parent=parent)
        self._value_btn = Node_ModuleValueButton(
            value, widget_id=widget_id, parent=self,
        )
        self._value_btn.clicked.connect(self.valueClicked.emit)
        self.set_slot(self._value_btn)

    def value(self) -> str:
        return self._value_btn.value()

    def setValue(self, v: str) -> None:
        self._value_btn.setValue(v)


__all__ = [
    "Node_NodeRow",
    "Node_PortInputRow",
    "Node_ConditionSet",
    "Node_OutputSet",
    "Node_InputSetting",
    "Node_ComboSetting",
    "Node_BehaviorSetting",
    "Node_ScriptSetting",
    "Node_ScriptInAndOut",
    "Node_NodeTableRowsWidget",
    "Node_RegistryModuleRow",
]
