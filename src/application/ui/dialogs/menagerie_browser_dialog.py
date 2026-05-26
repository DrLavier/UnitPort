# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""MenagerieBrowserDialog — sidebar entry to browse and download Menagerie packages.

All git/network logic lives in :mod:`application.service.models.menagerie_manager`;
this dialog is the Qt wrapper that turns its callable surface into Task-based
worker dispatches:

* :class:`MenagerieSparseAddTask` — sparse-checkout the selected packages.
* :class:`MenagerieRefreshTask`   — refresh the live package list from GitHub.
* :class:`MenagerieIconFetchTask` — populate the per-package PNG cache.

Lifecycle:
  1. On open: populate the grid from the snapshot, kick an icon-fetch task
     for any cards still missing thumbnails.
  2. User filters / picks packages -> clicks Download -> sparse-add task
     runs; progress streams into ``CmdLogWidget`` via ``Task.log_info``.
  3. On success: emits ``download_finished(list[str])`` for the panel to
     auto-match / auto-register the new packages.

Concurrent-click guard: each task type tracks its in-flight ID; re-clicking
the same button while a task runs is ignored (the progress label surfaces a
"already in progress" hint). Closing the dialog mid-download cancels
in-flight Tasks via the SDK's task signal — git operations cooperate at each
``on_output`` boundary.

All colors come from ``system.ini[Theme]`` via :func:`Config.get_color` per
CLAUDE.md §1.5; all strings go through :func:`tr` / :func:`i18n_bind`.
"""

from __future__ import annotations

from typing import List, Optional, Set

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviLineEdit,
    get_task_signal,
    get_tasks_manager,
    i18n_bind,
    log_error,
    log_info,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.menagerie import (
    MenagerieIconFetchTask,
    MenagerieRefreshTask,
    MenagerieSparseAddTask,
)
from application.service.models import menagerie_manager as _mm
from application.ui.widgets.menagerie import MenagerieCardGrid


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


class MenagerieBrowserDialog(QDialog):
    """Browse + selectively sparse-checkout MuJoCo Menagerie packages."""

    download_finished = pyqtSignal(list)   # List[str] of installed dir names

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumSize(1320, 620)
        i18n_bind(
            self, "setWindowTitle",
            "menagerie.dialog_title", "MuJoCo Menagerie",
        )
        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

        self._packages: List[str] = list(_mm.MENAGERIE_PACKAGES_SNAPSHOT)
        self._installed: Set[str] = _mm.scan_installed_packages()
        self._registered: Set[str] = _mm.registered_package_dirs()

        self._task_id_sparse: Optional[str] = None
        self._task_id_refresh: Optional[str] = None
        self._task_id_icons: Optional[str] = None

        self._build_ui()
        self._wire_signals()

        # Initial population + kick async icon fetch for any cards still
        # missing a thumbnail.
        self._populate_grid()
        self._kick_icon_fetch()

    # ----- UI build -----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        # Header: status line + filter input
        header = QHBoxLayout()
        header.setSpacing(10)

        self._status_label = setText(
            "menagerie.status_line",
            default="0 / 0 installed · 0 selected",
            kind="content", size=_ss(),
        )
        header.addWidget(self._status_label, 1)

        self._filter_edit = LaviLineEdit(
            text="",
            placeholder=tr("menagerie.filter_ph", "Filter packages…"),
        )
        self._filter_edit.setFixedWidth(260)
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        header.addWidget(self._filter_edit)

        root.addLayout(header)

        # Grid
        self._grid = MenagerieCardGrid(self)
        self._grid.selection_changed.connect(self._refresh_status)
        root.addWidget(self._grid, 1)

        # Progress label (single line; verbose log streams to CmdLogWidget)
        self._progress_label = setText(
            "menagerie.progress_idle", default="",
            kind="content", size=_ss(),
        )
        self._progress_label.setWordWrap(True)
        root.addWidget(self._progress_label)

        # Footer button row
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)

        self._btn_refresh = setButton(
            "menagerie.refresh_btn", 130, 28,
            kind="border", spec="none",
            default=tr("menagerie.refresh_btn", "Refresh from GitHub"),
        )
        self._btn_refresh.clicked.connect(self._on_refresh_clicked)
        footer.addWidget(self._btn_refresh)

        self._btn_download = setButton(
            "menagerie.download_btn", 130, 28,
            kind="normal", spec="confirm",
            default=tr("menagerie.download_btn", "Download selected"),
        )
        self._btn_download.clicked.connect(self._on_download_clicked)
        footer.addWidget(self._btn_download)

        self._btn_close = setButton(
            "menagerie.close_btn", 100, 28,
            kind="border", spec="none",
            default=tr("menagerie.close_btn", "Close"),
        )
        self._btn_close.clicked.connect(self.accept)
        footer.addWidget(self._btn_close)

        root.addLayout(footer)

    def _wire_signals(self) -> None:
        signal = get_task_signal()
        signal.task_finished.connect(self._on_task_finished)

    # ----- population helpers -------------------------------------------

    def _populate_grid(self) -> None:
        self._grid.populate(
            self._packages,
            installed=self._installed,
            registered=self._registered,
            pre_checked=set(),
        )
        self._refresh_status()

    def _refresh_status(self) -> None:
        total = len(self._packages)
        installed = len(self._installed)
        selected = len(self._grid.selected_packages())
        self._status_label.setText(
            tr(
                "menagerie.status_line",
                "{installed} / {total} installed · {selected} selected",
            ).replace("{installed}", str(installed))
             .replace("{total}", str(total))
             .replace("{selected}", str(selected))
        )

    def _on_filter_changed(self, text: str) -> None:
        self._grid.set_filter(text)

    # ----- task dispatch ------------------------------------------------

    def _kick_icon_fetch(self) -> None:
        if self._task_id_icons is not None:
            return
        missing = self._grid.cards_missing_icon()
        if not missing:
            return
        task = MenagerieIconFetchTask(missing)
        self._task_id_icons = get_tasks_manager().submit(task)
        self._set_progress(
            tr("menagerie.progress_icons",
               "Fetching {n} preview icons…").replace("{n}", str(len(missing))),
        )

    def _on_refresh_clicked(self) -> None:
        if self._task_id_refresh is not None:
            self._set_progress(
                tr("menagerie.refresh_busy", "Already refreshing; please wait.")
            )
            return
        task = MenagerieRefreshTask()
        self._task_id_refresh = get_tasks_manager().submit(task)
        self._set_progress(
            tr("menagerie.refreshing", "Fetching live package list from GitHub…")
        )

    def _on_download_clicked(self) -> None:
        if self._task_id_sparse is not None:
            self._set_progress(
                tr("menagerie.download_busy", "Already downloading; please wait.")
            )
            return
        targets = self._grid.selected_packages()
        if not targets:
            self._set_progress(
                tr("menagerie.download_empty",
                   "Tick at least one package before clicking Download.")
            )
            return
        task = MenagerieSparseAddTask(targets)
        self._task_id_sparse = get_tasks_manager().submit(task)
        self._btn_download.setEnabled(False)
        self._set_progress(
            tr("menagerie.downloading",
               "Sparse-checkout {n} package(s)…").replace("{n}", str(len(targets)))
        )

    # ----- task callback ------------------------------------------------

    @pyqtSlot(str, bool, object)
    def _on_task_finished(self, task_id: str, success: bool, result: object) -> None:
        if task_id == self._task_id_icons:
            self._task_id_icons = None
            self._handle_icons_finished(success, result)
            return
        if task_id == self._task_id_refresh:
            self._task_id_refresh = None
            self._handle_refresh_finished(success, result)
            return
        if task_id == self._task_id_sparse:
            self._task_id_sparse = None
            self._handle_sparse_finished(success, result)
            return
        # Not our task — ignore.

    def _handle_icons_finished(self, success: bool, result: object) -> None:
        if not success or not isinstance(result, dict):
            self._set_progress(
                tr("menagerie.icons_offline",
                   "Icon fetch failed; cached icons still render.")
            )
            return
        for name, path in result.items():
            if path:
                self._grid.update_card_icon_from_path(str(name), str(path))
        self._set_progress("")

    def _handle_refresh_finished(self, success: bool, result: object) -> None:
        if not success or not isinstance(result, list):
            self._set_progress(
                tr("menagerie.refresh_failed",
                   "Could not reach GitHub — keeping the bundled snapshot.")
            )
            return
        live = [str(n) for n in result if n]
        if not live:
            self._set_progress(
                tr("menagerie.refresh_empty",
                   "GitHub returned an empty package list; keeping the snapshot.")
            )
            return
        self._packages = live
        self._populate_grid()
        self._kick_icon_fetch()
        self._set_progress(
            tr("menagerie.refresh_done",
               "Refreshed: {n} packages.").replace("{n}", str(len(live)))
        )

    def _handle_sparse_finished(self, success: bool, result: object) -> None:
        self._btn_download.setEnabled(True)
        if not success:
            err = str(result) if result else "unknown error"
            log_error(f"[menagerie] download failed: {err}")
            self._set_progress(
                tr("menagerie.download_failed",
                   "Download failed: {err}").replace("{err}", err)
            )
            return
        installed_now = [str(n) for n in (result or [])]
        # Refresh installed snapshot, repopulate cards (cards now flip to
        # installed-green and lock their checkboxes).
        self._installed = _mm.scan_installed_packages()
        self._registered = _mm.registered_package_dirs()
        self._populate_grid()
        log_info(f"[menagerie] download complete: {installed_now}")
        self._set_progress(
            tr("menagerie.download_done",
               "Downloaded {n} package(s).").replace("{n}", str(len(installed_now)))
        )
        if installed_now:
            self.download_finished.emit(installed_now)

    # ----- helpers ------------------------------------------------------

    def _set_progress(self, text: str) -> None:
        self._progress_label.setText(text or "")

    def closeEvent(self, event) -> None:  # noqa: N802
        # Best-effort cancel of any in-flight task.
        mgr = get_tasks_manager()
        for tid in (self._task_id_sparse, self._task_id_refresh, self._task_id_icons):
            if not tid:
                continue
            try:
                mgr.cancel(tid)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[menagerie] cancel({tid}) failed: {exc}")
        try:
            get_task_signal().task_finished.disconnect(self._on_task_finished)
        except Exception:
            pass
        super().closeEvent(event)


__all__ = ["MenagerieBrowserDialog"]
