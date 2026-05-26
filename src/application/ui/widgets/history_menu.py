# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HistoryMenu -- main_row [History ▾] popup with per-row delete + Clear All.

Replaces the previous QMenu-of-QActions list (MainWindow._populate_history_menu)
with a tailored widget so each run row carries a square delete button on its
right, and the popup has a "Clear All" footer button at the bottom (border +
danger).

Behaviour-only widget: emits signals back to MainWindow, which owns all
side-effects (deleting run directories, garbage-collecting canvas snapshots,
swapping canvas content). HistoryMenu never touches disk.

Signals:
    run_clicked(backend_id, run_id)     -- user clicked the row body
    delete_clicked(backend_id, run_id)  -- user clicked the row's delete btn
    clear_all_clicked()                 -- user clicked the footer button
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from unitport_sdk import Config, tr, setButton

from application.service.training_assets import RunAsset


class _HistoryRow(QFrame):
    """One run row: square delete button + ``[backend_id] run_id`` label.

    Delete button sits on the LEFT (in front of the label) so the user can
    aim at the same x-coordinate across rows when bulk-pruning. Mouse press
    on the row body emits ``row_clicked``; the delete button has its own
    clicked signal which we surface as ``row_delete_clicked``. The button
    consumes its own click so the row body doesn't also fire.
    """

    row_clicked = pyqtSignal(str, str)
    row_delete_clicked = pyqtSignal(str, str)

    _ROW_H = 24
    _DELETE_SIDE = 18

    def __init__(self, run: RunAsset, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._backend_id = run.backend_id
        self._run_id = run.run_id
        self.setObjectName("historyMenuRow")
        self.setFixedHeight(self._ROW_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoFillBackground(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 8, 2)
        layout.setSpacing(6)

        self._delete_btn = setButton(
            "history.row.delete",
            self._DELETE_SIDE, self._DELETE_SIDE,
            kind="light",
            spec="danger",
            icon="icon_delete",
            default="",
            icon_only=True,
            parent=self,
        )
        self._delete_btn.setToolTip(tr("history.row.delete.tip", "Delete this run"))
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self._delete_btn, 0)

        self._label = QLabel(f"[{run.backend_id}] {run.run_id}", self)
        self._label.setObjectName("historyMenuRowLabel")
        self._label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addWidget(self._label, 1)

        self.apply_theme()

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        bg_hover = Config.get_color("hover_2")
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        font_mini = Config.get_font_size("size_mini")
        self.setStyleSheet(
            f"QFrame#historyMenuRow {{ "
            f"background-color: {bg}; "
            f"border: none; "
            f"border-bottom: 1px solid {border}; "
            f"}}"
            f"QFrame#historyMenuRow:hover {{ background-color: {bg_hover}; }}"
            f"QLabel#historyMenuRowLabel {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-size: {font_mini}pt; "
            f"}}"
        )
        self._delete_btn.refresh_style()

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: D401
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.row_clicked.emit(self._backend_id, self._run_id)
            return
        super().mousePressEvent(ev)

    def _on_delete_clicked(self) -> None:
        self.row_delete_clicked.emit(self._backend_id, self._run_id)


class _HistoryPlaceholderRow(QFrame):
    """Disabled placeholder row, e.g. ``(empty)`` / ``(no project)``."""

    _ROW_H = 24

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("historyMenuPlaceholder")
        self.setFixedHeight(self._ROW_H)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        label = QLabel(text, self)
        label.setObjectName("historyMenuPlaceholderLabel")
        layout.addWidget(label, 1)

        self.apply_theme()

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        fg = Config.get_color("sub_t2")
        font_mini = Config.get_font_size("size_mini")
        self.setStyleSheet(
            f"QFrame#historyMenuPlaceholder {{ "
            f"background-color: {bg}; "
            f"border: none; "
            f"}}"
            f"QLabel#historyMenuPlaceholderLabel {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-size: {font_mini}pt; "
            f"font-style: italic; "
            f"}}"
        )


class HistoryMenu(QMenu):
    """Custom popup for the main_row [History ▾] button.

    Hosts a single ``QWidgetAction`` whose widget is the rows container; a
    second ``QWidgetAction`` carries the Clear All footer button. Theming is
    re-applied from MainWindow on theme switch via :meth:`apply_theme`.
    """

    run_clicked = pyqtSignal(str, str)
    delete_clicked = pyqtSignal(str, str)
    clear_all_clicked = pyqtSignal()

    # Fixed (non-growing) width — matches the visual footprint of the
    # neighbouring Policies QMenu so the two dropdowns line up. Switching
    # away from setMinimumWidth so a long run_id no longer stretches the
    # popup wider than the Policies one beside it.
    _FIXED_WIDTH = 260
    _MAX_HEIGHT = 360

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("historyMenu")
        self.setFixedWidth(self._FIXED_WIDTH)

        # Rows container — re-built on every populate().
        self._rows_host = QWidget(self)
        self._rows_host.setObjectName("historyMenuRowsHost")
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea(self)
        scroll.setObjectName("historyMenuScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(self._FIXED_WIDTH)
        scroll.setMaximumHeight(self._MAX_HEIGHT)
        scroll.setWidget(self._rows_host)
        self._scroll = scroll

        rows_action = QWidgetAction(self)
        rows_action.setDefaultWidget(scroll)
        self.addAction(rows_action)

        self.addSeparator()

        # Footer — Clear All. Built once, enabled/disabled per populate().
        footer = QWidget(self)
        footer.setObjectName("historyMenuFooter")
        footer.setFixedWidth(self._FIXED_WIDTH)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(6, 4, 6, 4)
        footer_layout.setSpacing(0)
        self._clear_all_btn = setButton(
            "history.clear_all",
            self._FIXED_WIDTH - 16, 26,
            kind="border",
            spec="danger",
            default="Clear All",
            parent=footer,
        )
        self._clear_all_btn.clicked.connect(self.clear_all_clicked)
        footer_layout.addWidget(self._clear_all_btn, 1)
        self._footer = footer

        footer_action = QWidgetAction(self)
        footer_action.setDefaultWidget(footer)
        self.addAction(footer_action)

        self.apply_theme()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def populate(
        self,
        runs: List[RunAsset],
        *,
        placeholder_text: Optional[str] = None,
    ) -> None:
        """Rebuild the rows from ``runs``.

        ``placeholder_text`` (e.g. ``"(no project)"``) replaces the rows when
        ``runs`` is empty; when ``placeholder_text`` is None and ``runs`` is
        empty, a generic ``"(empty)"`` is shown. Clear All is disabled
        whenever the runs list is empty.
        """
        # Drop existing rows.
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        if not runs:
            text = placeholder_text or tr("progressrow.menu.empty", "(empty)")
            self._rows_layout.addWidget(_HistoryPlaceholderRow(text, self._rows_host))
            self._clear_all_btn.setEnabled(False)
            return

        for run in runs:
            row = _HistoryRow(run, self._rows_host)
            row.row_clicked.connect(self.run_clicked)
            row.row_delete_clicked.connect(self.delete_clicked)
            self._rows_layout.addWidget(row)
        self._clear_all_btn.setEnabled(True)

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        font_mini = Config.get_font_size("size_mini")
        self.setStyleSheet(
            f"QMenu#historyMenu {{ "
            f"background-color: {bg}; "
            f"border: 1px solid {border}; "
            f"color: {fg}; "
            f"font-size: {font_mini}pt; "
            f"}}"
            f"QMenu#historyMenu::separator {{ "
            f"height: 1px; "
            f"background-color: {border}; "
            f"margin: 0px; "
            f"}}"
            f"QWidget#historyMenuRowsHost {{ background-color: {bg}; }}"
            f"QScrollArea#historyMenuScroll {{ background-color: {bg}; }}"
            f"QWidget#historyMenuFooter {{ background-color: {bg}; }}"
        )
        self._clear_all_btn.refresh_style()
        # Re-skin any live rows so a theme switch propagates.
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            fn = getattr(w, "apply_theme", None)
            if callable(fn):
                fn()


__all__ = ["HistoryMenu"]
