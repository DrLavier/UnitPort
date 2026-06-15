# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasMiniMenu — 画布右键迷你菜单 / Canvas right-click mini context menu.

单击鼠标右键（按下后未拖动）时由 ``CanvasView`` 弹出。按住右键拖动则用于
平移画布，不弹菜单（见 ``application.ui.canvas.view`` 的鼠标契约）。

当前为占位实现：仅承载若干禁用的占位条目，预留后续填充
（新建节点 / 居中视图 / 适配窗口 / 复制粘贴 等画布上下文操作）。落地具体
动作时通过 ``pyqtSignal`` 回传给 ``CanvasView`` / ``CanvasPage``，本组件不
直接触碰场景或磁盘。

主题：所有颜色 / 字号 use-time 经 ``Config.get_color`` / ``Config.get_font_size``
读取（RELEASE/CLAUDE.md §5），主题切换由持有者调用 :meth:`apply_theme` 传播。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtWidgets import QMenu, QWidget

from unitport_sdk import Config, tr


class CanvasMiniMenu(QMenu):
    """画布右键上下文菜单 / Canvas right-click context menu.

    行为型组件：占位条目暂为禁用项；后续动作落地时改为 enabled 并经
    ``action_triggered`` 信号上抛由 ``CanvasView`` / ``CanvasPage`` 执行副作用。

    用法::

        menu = CanvasMiniMenu(view)
        menu.popup_at(global_pos, scene_pos)

    Signals:
        action_triggered(str)  -- 占位动作 key（如 ``"add_node"``），由持有者解释
    """

    action_triggered = pyqtSignal(str)

    _MIN_WIDTH = 180

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasMiniMenu")
        self.setMinimumWidth(self._MIN_WIDTH)

        # 弹出时所处的场景坐标，供后续 "在此处新建节点" 之类的动作消费。
        self._scene_pos: QPointF = QPointF()

        self._build_placeholder_items()
        self.apply_theme()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def popup_at(self, global_pos, scene_pos: Optional[QPointF] = None) -> None:
        """在屏幕坐标 ``global_pos`` 处弹出，记录对应场景坐标 ``scene_pos``."""
        self._scene_pos = QPointF(scene_pos) if scene_pos is not None else QPointF()
        self.popup(global_pos)

    def scene_pos(self) -> QPointF:
        """弹出时所处的场景坐标 / Scene position where the menu opened."""
        return QPointF(self._scene_pos)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _build_placeholder_items(self) -> None:
        """构建占位条目（暂禁用）/ Build placeholder (disabled) entries."""
        placeholder = self.addAction(
            tr("canvas.minimenu.placeholder", "(coming soon)")
        )
        placeholder.setEnabled(False)

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        fg_disabled = Config.get_color("sub_t2")
        hover = Config.get_color("hover_2")
        font_mini = Config.get_font_size("size_mini")
        self.setStyleSheet(
            f"QMenu#canvasMiniMenu {{ "
            f"background-color: {bg}; "
            f"border: 1px solid {border}; "
            f"color: {fg}; "
            f"font-size: {font_mini}pt; "
            f"padding: 2px; "
            f"}}"
            f"QMenu#canvasMiniMenu::item {{ "
            f"padding: 4px 16px; "
            f"background: transparent; "
            f"}}"
            f"QMenu#canvasMiniMenu::item:selected {{ "
            f"background-color: {hover}; "
            f"}}"
            f"QMenu#canvasMiniMenu::item:disabled {{ "
            f"color: {fg_disabled}; "
            f"}}"
            f"QMenu#canvasMiniMenu::separator {{ "
            f"height: 1px; "
            f"background-color: {border}; "
            f"margin: 4px 0px; "
            f"}}"
        )


__all__ = ["CanvasMiniMenu"]
