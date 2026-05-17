"""unitport_sdk.canvas.composites — Training Node 复合控件 / Composite widgets.

DEMO 对应（``bin/nodes/training_node_items.py`` + ``node_ui_rows.py``）：
    _SaveListButton           → Node_SaveListButton
    _BufferSizePickerWidget   → Node_BufferSizePickerWidget
    _MotionFilePickerWidget   → Node_MotionFilePickerWidget
    _FPSSliderWidget          → Node_FPSSliderWidget
    _PreviewButtonWidget      → Node_PreviewButtonWidget
    _AMPResultsPanel          → Node_AMPResultsPanel
    _TrainingItemRow          → Node_TrainingItemRow
    _TrainingItemEditor       → Node_TrainingItemEditor
    _RegistryModuleEditor     → Node_RegistryModuleEditor

约束：所有 UI class 以 ``Node_`` 开头；内部一律用 SDK ``setButton`` /
``setText`` / Lavi 输入控件 + 已有 ``Node_*`` 原子件。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk.sys import Config, tr
from unitport_sdk.ui import setButton, setText

from .inputs import (
    NODE_INPUT_H,
    Node_IndexButton,
    Node_ModuleValueButton,
    Node_NodeInputer,
)
from .pickers import (
    Node_DropdownChoicePicker,
    Node_MotionLibraryPicker,
    Node_RichChoicePicker,
)
from .rows import (
    NODE_ROW_GAP,
    Node_NodeRow,
    Node_NodeTableRowsWidget,
)
from .sliders import Node_NodeSlider, Node_RangeHandleSlider
from .widgets import LaviCodeEditor, setCodeEditor


# =============================================================================
# Node_SaveListButton —— 弹出保存列表的按钮
# =============================================================================

class Node_SaveListButton(QWidget):
    """保存列表触发按钮 / Save-list trigger button.

    DEMO ``_SaveListButton``：LaviButton 显示 ``Save…``，点击后由调用方接信号
    决定弹列表（典型 ``Node_RichChoicePopup``）。
    """

    clicked = pyqtSignal()

    def __init__(
        self,
        *,
        widget_id: str = "node.save_list",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._button = setButton(
            widget_id, 96, NODE_INPUT_H,
            kind="normal", spec="save",
            default=tr(f"{widget_id}.label", "Save…"),
            parent=self,
        )
        self._button.clicked.connect(self.clicked.emit)
        layout.addWidget(self._button)

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_BufferSizePickerWidget —— LaviLabel + Node_IndexButton
# =============================================================================

class Node_BufferSizePickerWidget(Node_NodeRow):
    """buffer size 选择 / Buffer size picker.

    DEMO ``_BufferSizePickerWidget``。Index button 输入大小，单位由调用方加。
    """

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        value: int = 1024,
        *,
        minimum: int = 1,
        maximum: int = 1_000_000,
        widget_id: str = "node.buffer_size",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            tr(f"{widget_id}.label", "Buffer Size"),
            widget_id=widget_id,
            parent=parent,
        )
        self._idx = Node_IndexButton(
            value, minimum=minimum, maximum=maximum,
            widget_id=widget_id, parent=self,
        )
        self._idx.valueChanged.connect(self.valueChanged.emit)
        self.set_slot(self._idx)

    def value(self) -> int:
        return self._idx.value()


# =============================================================================
# Node_MotionFilePickerWidget —— motion 文件 picker + LaviLabel
# =============================================================================

class Node_MotionFilePickerWidget(Node_NodeRow):
    """motion 文件 picker / Motion file picker.

    DEMO ``_MotionFilePickerWidget``。右侧用 ``Node_MotionLibraryPicker``。
    """

    selectionChanged = pyqtSignal(str)

    def __init__(
        self,
        choices: Sequence[str] = (),
        *,
        current: Any = None,
        widget_id: str = "node.motion_file",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            tr(f"{widget_id}.label", "Motion"),
            widget_id=widget_id,
            parent=parent,
        )
        self._picker = Node_MotionLibraryPicker(
            choices, current=current, widget_id=widget_id, parent=self,
        )
        self._picker.currentTextChanged.connect(self.selectionChanged.emit)
        self.set_slot(self._picker)

    def currentText(self) -> str:
        return self._picker.currentText()


# =============================================================================
# Node_FPSSliderWidget —— LaviLabel + Node_NodeSlider
# =============================================================================

class Node_FPSSliderWidget(QWidget):
    """FPS 滑块 / FPS slider widget.

    DEMO ``_FPSSliderWidget``。组合 ``Node_NodeSlider``，1..120 默认。
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        value: float = 30.0,
        *,
        minimum: float = 1.0,
        maximum: float = 120.0,
        widget_id: str = "node.fps_slider",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._slider = Node_NodeSlider(
            tr(f"{widget_id}.label", "FPS"),
            value=value, minimum=minimum, maximum=maximum, decimals=1,
            widget_id=widget_id, parent=self,
        )
        self._slider.valueChanged.connect(self.valueChanged.emit)
        layout.addWidget(self._slider)

    def value(self) -> float:
        return self._slider.value()

    def setValue(self, v: float) -> None:
        self._slider.setValue(v)

    def refresh_style(self) -> None:
        self._slider.refresh_style()


# =============================================================================
# Node_PreviewButtonWidget —— 预览触发按钮
# =============================================================================

class Node_PreviewButtonWidget(QWidget):
    """预览按钮 / Preview trigger button.

    DEMO ``_PreviewButtonWidget``。LaviButton spec=``play``，对外发 ``clicked``。
    """

    clicked = pyqtSignal()

    def __init__(
        self,
        *,
        widget_id: str = "node.preview",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._button = setButton(
            widget_id, 96, NODE_INPUT_H,
            kind="normal", spec="play",
            default=tr(f"{widget_id}.label", "Preview"),
            parent=self,
        )
        self._button.clicked.connect(self.clicked.emit)
        layout.addWidget(self._button)

    def refresh_style(self) -> None:
        pass


# =============================================================================
# Node_AMPResultsPanel —— LaviLabel 标题 + LaviCodeEditor 只读 JSON
# =============================================================================

class Node_AMPResultsPanel(QWidget):
    """AMP 训练结果展示面板 / AMP training result panel.

    DEMO ``_AMPResultsPanel``。顶部 LaviLabel 标题；下方 LaviCodeEditor
    （JSON 高亮，只读模式）展示结果。
    """

    def __init__(
        self,
        result: str = "",
        *,
        widget_id: str = "node.amp_results",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(setText(
            f"{widget_id}.title",
            default=tr(f"{widget_id}.title", "AMP Results"),
            kind="title",
        ))

        self._editor: LaviCodeEditor = setCodeEditor(result, language="json", parent=self)
        self._editor.setReadOnly(True)
        self._editor.setMinimumHeight(120)
        layout.addWidget(self._editor, 1)

    def setResult(self, text: str) -> None:
        self._editor.setPlainText(text)

    def result(self) -> str:
        return self._editor.toPlainText()

    def refresh_style(self) -> None:
        self._editor.refresh_style()


# =============================================================================
# Node_TrainingItemRow / Node_TrainingItemEditor —— 训练项目编辑
# =============================================================================

class Node_TrainingItemRow(QWidget):
    """训练项目单行 / Single training item row.

    DEMO ``_TrainingItemRow``。一行：LaviLabel name + Node_NodeInputer weight
    + Node_ModuleValueButton（点击展开详情）。
    """

    valueChanged = pyqtSignal(dict)

    def __init__(
        self,
        name: str,
        *,
        weight: float = 1.0,
        detail: str = "",
        widget_id: str = "node.training_item_row",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._widget_id = widget_id
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(NODE_ROW_GAP)

        self._label = setText(f"{widget_id}.{name}", default=name)
        h.addWidget(self._label)

        self._weight = Node_NodeInputer("float", weight, widget_id=f"{widget_id}.weight", parent=self)
        self._weight.valueChanged.connect(self._emit)
        h.addWidget(self._weight, 1)

        self._detail = Node_ModuleValueButton(
            detail, widget_id=f"{widget_id}.detail", parent=self,
        )
        h.addWidget(self._detail)

    @property
    def name(self) -> str:
        return self._name

    def detail(self) -> str:
        return self._detail.value()

    def setDetail(self, v: str) -> None:
        self._detail.setValue(v)

    def _emit(self, *_: Any) -> None:
        self.valueChanged.emit({
            "name": self._name,
            "weight": self._weight.value(),
            "detail": self._detail.value(),
        })

    def refresh_style(self) -> None:
        pass


class Node_TrainingItemEditor(QWidget):
    """训练项目集合编辑器 / Training items editor.

    DEMO ``_TrainingItemEditor``。一组 ``Node_TrainingItemRow``，底部 +Add
    LaviButton。
    """

    itemsChanged = pyqtSignal(list)

    def __init__(
        self,
        items: Sequence[dict] = (),
        *,
        widget_id: str = "node.training_item_editor",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._widget_id = widget_id
        self._items: List[Node_TrainingItemRow] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._add_btn = setButton(
            f"{widget_id}.add", 96, NODE_INPUT_H,
            kind="border", spec="add",
            default=tr(f"{widget_id}.add", "+ Add"),
            parent=self,
        )
        self._add_btn.clicked.connect(lambda: self.add_item("new_item"))
        self._layout.addWidget(self._add_btn)

        for it in items:
            self.add_item(it.get("name", ""), weight=it.get("weight", 1.0), detail=it.get("detail", ""))

    def add_item(self, name: str, *, weight: float = 1.0, detail: str = "") -> None:
        row = Node_TrainingItemRow(
            name, weight=weight, detail=detail,
            widget_id=self._widget_id, parent=self,
        )
        row.valueChanged.connect(lambda *_: self._emit())
        self._items.append(row)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._emit()

    def items(self) -> List[dict]:
        result: List[dict] = []
        for row in self._items:
            result.append({
                "name": row.name,
                "detail": row.detail(),
            })
        return result

    def _emit(self) -> None:
        self.itemsChanged.emit(self.items())

    def refresh_style(self) -> None:
        for it in self._items:
            it.refresh_style()


# =============================================================================
# Node_RegistryModuleEditor —— 注册模块批量编辑
# =============================================================================

class Node_RegistryModuleEditor(QWidget):
    """注册模块编辑器 / Registry module editor.

    DEMO ``_RegistryModuleEditor``。组合 ``Node_NodeTableRowsWidget`` 做表格
    式编辑，列由 ``columns`` 决定。
    """

    rowsChanged = pyqtSignal(list)

    def __init__(
        self,
        columns: Sequence[tuple] = (("name", "string"), ("value", "string")),
        *,
        rows: Sequence[dict] = (),
        widget_id: str = "node.registry_module_editor",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(setText(
            f"{widget_id}.title",
            default=tr(f"{widget_id}.title", "Registry Modules"),
            kind="title",
        ))

        self._table = Node_NodeTableRowsWidget(
            columns, rows=rows, widget_id=widget_id, parent=self,
        )
        self._table.rowsChanged.connect(self.rowsChanged.emit)
        layout.addWidget(self._table, 1)

    def rows(self) -> List[dict]:
        return self._table.rows()

    def refresh_style(self) -> None:
        self._table.refresh_style()


__all__ = [
    "Node_SaveListButton",
    "Node_BufferSizePickerWidget",
    "Node_MotionFilePickerWidget",
    "Node_FPSSliderWidget",
    "Node_PreviewButtonWidget",
    "Node_AMPResultsPanel",
    "Node_TrainingItemRow",
    "Node_TrainingItemEditor",
    "Node_RegistryModuleEditor",
]
