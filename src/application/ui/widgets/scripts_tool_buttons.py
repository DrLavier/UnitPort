"""application.ui.widgets.scripts_tool_buttons — Scripts 模式工具row 左侧按钮组。

Scripts 模式下出现在 MissionControlPanel 工具row 左侧的 3 按钮集合：

    [icon_undo] [icon_redo] [Save 按钮]

- undo / redo: light icon-only button —— 直接驱动绑定 CodeEditorWidget 的
  ``undo()`` / ``redo()`` slot（继承自 QPlainTextEdit），可用性跟随
  ``undoAvailable`` / ``redoAvailable`` 信号自动切换。
- Save: border+save button —— enable 跟 CodeEditorWidget.dirtyChanged 信号
  （SHA-1 baseline 判脏，比 QTextDocument.isModified 更准）。点击发送
  ``save_requested`` 信号让 MissionControlPanel 决定写盘目标（文件路径 vs
  registry 虚拟 id 落到 <project>/scripts/<kind>/<key>.py 覆盖文件）。

本 widget 完全自包含：bind_editor(editor) 完成所有信号挂载；editor=None 时
全部 disable。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from unitport_sdk import (
    Assets,
    CodeEditorWidget,
    Config,
    LaviButton,
    i18n_bind,
    log_warning,
    setButton,
    tr,
)


BUTTON_H = 30                # 与 TCToolButtonsCluster 保持一致（top_row 43 - 6*2 padding）
ICON_BTN_W = 30
ICON_SIDE = max(12, BUTTON_H - 12)


# ----------------------------------------------------------------------------
# SVG icon helpers (与 tc_tool_buttons 同款；后续可抽到共享模块)
# ----------------------------------------------------------------------------

def _build_dual_mode_icon(
    svg_path: Path, side: int, disabled_color: str
) -> QIcon:
    """生成 Normal / Disabled 双模态 QIcon。"""
    icon = QIcon(str(svg_path))
    renderer = QSvgRenderer(str(svg_path))
    pm = QPixmap(side, side)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    renderer.render(p)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(pm.rect(), QColor(disabled_color))
    p.end()
    icon.addPixmap(pm, QIcon.Mode.Disabled)
    return icon


# ----------------------------------------------------------------------------
# Cluster
# ----------------------------------------------------------------------------

class ScriptsToolbarCluster(QWidget):
    """Scripts 模式工具row 左侧 3 按钮集合。"""

    # 点击 Save 时发出；MissionControlPanel 据 current_script 前缀决定落盘目标。
    save_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._editor: Optional[CodeEditorWidget] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._undo_btn: LaviButton = setButton(
            "missioncontrol.script.btn.undo",
            ICON_BTN_W, BUTTON_H,
            kind="light",
            icon="icon_undo",
            default="",
            icon_only=True,
            disabled=True,
        )
        i18n_bind(self._undo_btn, "setToolTip",
                  "missioncontrol.script.btn.undo.tip", "Undo (Ctrl+Z)")
        layout.addWidget(self._undo_btn)

        self._redo_btn: LaviButton = setButton(
            "missioncontrol.script.btn.redo",
            ICON_BTN_W, BUTTON_H,
            kind="light",
            icon="icon_redo",
            default="",
            icon_only=True,
            disabled=True,
        )
        self._redo_btn.setToolTip(
            tr("missioncontrol.script.btn.redo.tip", "Redo (Ctrl+Shift+Z)")
        )
        layout.addWidget(self._redo_btn)

        self._save_btn: LaviButton = setButton(
            "missioncontrol.script.btn.save",
            0, BUTTON_H,
            kind="border", spec="save",
            default="Save",
            disabled=True,
        )
        self._save_btn.setFixedHeight(BUTTON_H)
        i18n_bind(self._save_btn, "setToolTip",
                  "missioncontrol.script.btn.save.tip", "Save script")
        self._save_btn.clicked.connect(self.save_requested.emit)
        self._patch_save_disabled_qss()
        layout.addWidget(self._save_btn)

        self._apply_undo_redo_icons()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind_editor(self, editor: Optional[CodeEditorWidget]) -> None:
        """绑定 / 解绑 CodeEditorWidget。重复绑同一 editor 时安全。"""
        if self._editor is editor:
            self._sync_initial_state()
            return

        old = self._editor
        if old is not None:
            try:
                old.undoAvailable.disconnect(self._undo_btn.setEnabled)
                old.redoAvailable.disconnect(self._redo_btn.setEnabled)
                self._undo_btn.clicked.disconnect(old.undo)
                self._redo_btn.clicked.disconnect(old.redo)
                old.dirtyChanged.disconnect(self._save_btn.setEnabled)
            except (TypeError, RuntimeError):
                pass

        self._editor = editor
        if editor is None:
            self._undo_btn.setEnabled(False)
            self._redo_btn.setEnabled(False)
            self._save_btn.setEnabled(False)
            return

        editor.undoAvailable.connect(self._undo_btn.setEnabled)
        editor.redoAvailable.connect(self._redo_btn.setEnabled)
        self._undo_btn.clicked.connect(editor.undo)
        self._redo_btn.clicked.connect(editor.redo)
        editor.dirtyChanged.connect(self._save_btn.setEnabled)

        self._sync_initial_state()

    def apply_theme(self) -> None:
        """主题切换。"""
        for btn in (self._undo_btn, self._redo_btn, self._save_btn):
            try:
                btn.refresh_style()
            except Exception:
                pass
        self._patch_save_disabled_qss()
        self._apply_undo_redo_icons()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_initial_state(self) -> None:
        editor = self._editor
        if editor is None:
            return
        # QPlainTextEdit 没有暴露 canUndo()/canRedo() 公共方法，但有等价的
        # document().isUndoAvailable() / isRedoAvailable()。
        try:
            doc = editor.document()
            self._undo_btn.setEnabled(doc.isUndoAvailable())
            self._redo_btn.setEnabled(doc.isRedoAvailable())
        except Exception:
            self._undo_btn.setEnabled(False)
            self._redo_btn.setEnabled(False)
        try:
            self._save_btn.setEnabled(bool(editor.is_modified()))
        except Exception:
            self._save_btn.setEnabled(False)

    def _patch_save_disabled_qss(self) -> None:
        disabled_fg = Config.get_color("main_c2", "#999999")
        base = self._save_btn.styleSheet() or ""
        extra = f"QPushButton:disabled {{ border-color: {disabled_fg}; }}"
        self._save_btn.setStyleSheet(base + "\n" + extra)

    def _apply_undo_redo_icons(self) -> None:
        disabled_color = Config.get_color("bg_3", "#101010")
        for btn, name in ((self._undo_btn, "icon_undo"),
                          (self._redo_btn, "icon_redo")):
            svg_path = Assets.find_icon(name)
            if svg_path is None:
                log_warning(f"[scripts_tool_buttons] icon not found: {name}")
                continue
            icon = _build_dual_mode_icon(svg_path, ICON_SIDE, disabled_color)
            btn.setIcon(icon)
            btn.setIconSize(QSize(ICON_SIDE, ICON_SIDE))


__all__ = ["ScriptsToolbarCluster"]
