# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""unitport_sdk.canvas.dialogs — Node 弹窗对话框 / Node-internal dialogs.

DEMO 对应（``bin/nodes/training_node_items.py``）：
    _StageOverrideDialog        → Node_StageOverrideDialog
    _StageSwitchCriteriaDialog  → Node_StageSwitchCriteriaDialog
    _StageSwitchEditorWidget    → Node_StageSwitchEditorWidget

约束：所有 UI class 以 ``Node_`` 开头；Ok/Cancel 用 SDK ``setButton``，
内容区用 ``Node_NodeTableRowsWidget`` / ``Node_NodeInputer`` / ``Node_DropdownChoicePicker``。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from unitport_sdk.sys import Config, tr
from unitport_sdk.ui import setButton, setText

from .inputs import NODE_INPUT_H, Node_NodeInputer
from .pickers import Node_DropdownChoicePicker
from .rows import Node_NodeTableRowsWidget


# =============================================================================
# 通用 dialog 工具
# =============================================================================

def _build_node_dialog_button_bar(dialog: QDialog) -> QHBoxLayout:
    """OK/Cancel button bar — SDK setButton 风格，与 edit_dialogs 一致."""
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
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    bar.addWidget(cancel)
    bar.addWidget(ok)
    return bar


def _apply_node_dialog_palette(dialog: QDialog) -> None:
    bg = Config.get_color("bg_1", "#1c1c1c")
    fg = Config.get_color("main_t1", "#D6D3C7")
    dialog.setStyleSheet(f"QDialog {{ background: {bg}; color: {fg}; }}")


# =============================================================================
# Node_StageOverrideDialog —— 阶段覆写对话框
# =============================================================================

class Node_StageOverrideDialog(QDialog):
    """阶段覆写设置对话框 / Stage override settings dialog.

    DEMO ``_StageOverrideDialog``。中部一个 ``Node_NodeTableRowsWidget``
    （列 = ``[("key", "string"), ("value", "string")]``），底部 OK/Cancel。
    """

    def __init__(
        self,
        rows: Sequence[dict] = (),
        *,
        title: str = "",
        widget_id: str = "node.stage_override_dialog",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or tr(f"{widget_id}.title", "Stage Override"))
        self.setModal(True)
        self.setMinimumSize(560, 360)
        _apply_node_dialog_palette(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addWidget(setText(
            f"{widget_id}.header",
            default=tr(f"{widget_id}.header", "Override values for this stage:"),
            kind="title",
        ))

        self._table = Node_NodeTableRowsWidget(
            (("key", "string"), ("value", "string")),
            rows=rows, widget_id=widget_id, parent=self,
        )
        layout.addWidget(self._table, 1)

        layout.addLayout(_build_node_dialog_button_bar(self))

    def values(self) -> List[dict]:
        return self._table.rows()


# =============================================================================
# Node_StageSwitchCriteriaDialog —— 阶段切换条件对话框
# =============================================================================

class Node_StageSwitchCriteriaDialog(QDialog):
    """阶段切换条件对话框 / Stage switch criteria dialog.

    DEMO ``_StageSwitchCriteriaDialog``。封装 ``Node_StageSwitchEditorWidget``
    + OK/Cancel。
    """

    def __init__(
        self,
        metrics: Sequence[str] = (),
        *,
        criteria: Sequence[dict] = (),
        title: str = "",
        widget_id: str = "node.stage_switch_criteria",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or tr(f"{widget_id}.title", "Stage Switch Criteria"))
        self.setModal(True)
        self.setMinimumSize(560, 360)
        _apply_node_dialog_palette(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._editor = Node_StageSwitchEditorWidget(
            metrics, criteria=criteria, widget_id=widget_id, parent=self,
        )
        layout.addWidget(self._editor, 1)

        layout.addLayout(_build_node_dialog_button_bar(self))

    def criteria(self) -> List[dict]:
        return self._editor.criteria()


# =============================================================================
# Node_StageSwitchEditorWidget —— 阶段切换条件编辑器
# =============================================================================

class Node_StageSwitchEditorWidget(QWidget):
    """阶段切换条件编辑器 / Stage switch criteria editor.

    DEMO ``_StageSwitchEditorWidget``。一个表：
        [metric picker][op picker][threshold inputer][del]
    底部 +Add。
    """

    criteriaChanged = pyqtSignal(list)

    _OPS = (">=", ">", "<=", "<", "==", "!=")

    def __init__(
        self,
        metrics: Sequence[str] = (),
        *,
        criteria: Sequence[dict] = (),
        widget_id: str = "node.stage_switch_editor",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._metrics = list(metrics)
        self._widget_id = widget_id
        self._rows: List[QWidget] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        layout_header = setText(
            f"{widget_id}.header",
            default=tr(f"{widget_id}.header", "Switch when:"),
            kind="caption",
        )
        self._layout.addWidget(layout_header)

        self._add_btn = setButton(
            f"{widget_id}.add", 96, NODE_INPUT_H,
            kind="border", spec="add",
            default=tr(f"{widget_id}.add", "+ Add"),
            parent=self,
        )
        self._add_btn.clicked.connect(lambda: self.add_criterion())
        self._layout.addWidget(self._add_btn)

        for c in criteria:
            self.add_criterion(
                metric=c.get("metric", metrics[0] if metrics else ""),
                op=c.get("op", ">="),
                threshold=c.get("threshold", 0.0),
            )

    def add_criterion(
        self,
        *,
        metric: str = "",
        op: str = ">=",
        threshold: float = 0.0,
    ) -> None:
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        metric_picker = Node_DropdownChoicePicker(
            self._metrics, current=metric or (self._metrics[0] if self._metrics else None),
            widget_id=f"{self._widget_id}.metric", parent=row,
        )
        op_picker = Node_DropdownChoicePicker(
            self._OPS, current=op,
            widget_id=f"{self._widget_id}.op", parent=row,
        )
        threshold_input = Node_NodeInputer(
            "float", threshold,
            widget_id=f"{self._widget_id}.threshold", parent=row,
        )
        del_btn = setButton(
            f"{self._widget_id}.del", 22, 22,
            kind="border", spec="del", default="×", parent=row,
        )

        metric_picker.currentTextChanged.connect(lambda *_: self._emit())
        op_picker.currentTextChanged.connect(lambda *_: self._emit())
        threshold_input.valueChanged.connect(lambda *_: self._emit())
        del_btn.clicked.connect(lambda _=False, w=row: self._remove(w))

        h.addWidget(metric_picker)
        h.addWidget(op_picker)
        h.addWidget(threshold_input, 1)
        h.addWidget(del_btn)

        self._rows.append(row)
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._emit()

    def _remove(self, row: QWidget) -> None:
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()
            self._emit()

    def criteria(self) -> List[dict]:
        result: List[dict] = []
        for row in self._rows:
            metric = row.findChildren(Node_DropdownChoicePicker)
            inputer = row.findChild(Node_NodeInputer)
            if len(metric) >= 2 and inputer is not None:
                result.append({
                    "metric": metric[0].currentText(),
                    "op": metric[1].currentText(),
                    "threshold": inputer.value(),
                })
        return result

    def _emit(self) -> None:
        self.criteriaChanged.emit(self.criteria())

    def refresh_style(self) -> None:
        pass


__all__ = [
    "Node_StageOverrideDialog",
    "Node_StageSwitchCriteriaDialog",
    "Node_StageSwitchEditorWidget",
]
