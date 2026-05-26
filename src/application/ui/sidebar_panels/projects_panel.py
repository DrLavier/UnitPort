# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ProjectsPanel — project picker + flat Canvas files list.

The panel is two-pieced:

    +---------------------------------------+
    | [ project dropdown ▾ ] [↻] [folder] |
    +---------------------------------------+
    |                                       |
    |       files_list (Canvas only)        |
    |       — owned single-tab LaviTabTable, |
    |         tab strip hidden, transparent |
    |                                       |
    +---------------------------------------+

* The dropdown enumerates every project under ``Paths.PROJECTS_DIR``
  via ``ProjectStore.snapshot()``. Selecting an entry emits
  ``project_selected(Path)``; ``MainWindow`` bridges it to ``open_project``,
  which re-runs ``_bind_project`` and feeds the new ``set_canvas_groups``
  payload back into this panel.
* The lower half is a panel-owned :class:`LaviTabTable` with a single tab
  (Canvas). The tab strip is hidden so it reads as a flat grouped list.
  Selecting a row emits ``canvas_selected(file_id)`` for MainWindow to
  forward into ``MissionControlPanel.open_canvas``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviTabTable,
    i18n_bind,
    log_debug,
    log_warning,
    setButton,
    setComboBox,
    setLaviTabTable,
    tr,
)

from application.service.projects import ProjectInfo, get_project_store
from application.ui.sidebar_panels.scripts_panels import (
    restyle_tab_table_for_sidebar,
)


_FILES_TAB_ID = "projects.tab.canvas"


class ProjectsPanel(QWidget):
    """Project dropdown + refresh + open-folder + flat canvas list."""

    file_activated = pyqtSignal(Path)
    project_selected = pyqtSignal(Path)
    # Emitted when the user picks a canvas row. Payload is the canvas file_id
    # (e.g. ``"canvas/sb3_mujoco/main.canvas.json"``) — the same identifier
    # ``MissionControlPanel.open_canvas`` consumes.
    canvas_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._store = get_project_store()
        self._projects_cache: List[ProjectInfo] = []
        self._suspend_dropdown_signal: bool = False

        self._canvas_groups: List[Dict] = []
        self._current_canvas: str = ""
        self._files_table: Optional[LaviTabTable] = None
        self._files_table_layout: Optional[QVBoxLayout] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addLayout(self._build_header())
        layout.addWidget(self._build_files_host(), 1)

        self._store.snapshot_changed.connect(self._on_snapshot_changed)
        self._repopulate_dropdown()

    # ------------------------------------------------------------------
    # Public refresh hook (forwarded by Sidebar.refresh -> MainWindow.refresh)
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._store.refresh_snapshot()

    def set_current_project(self, path: Optional[Path]) -> None:
        """Sync the dropdown to ``path`` without emitting selection signals."""
        if not path:
            return
        try:
            target = path.resolve()
        except OSError:
            target = path
        for info in self._projects_cache:
            try:
                same = info.path.resolve() == target
            except OSError:
                same = info.path == path
            if same:
                self._suspend_dropdown_signal = True
                try:
                    self._dropdown.setCurrentKey(info.project_id)
                finally:
                    self._suspend_dropdown_signal = False
                return

    def set_canvas_groups(self, groups: List[Dict]) -> None:
        """Replace the panel's canvas list with ``groups``.

        Called by ``MainWindow._bind_project_info`` whenever the active
        project changes (or on initial bind). Groups follow the spec built
        by :func:`application.service.projects.list_canvas_groups`.
        """
        self._canvas_groups = list(groups or [])
        self._rebuild_files_table()

    def set_current_canvas(self, file_id: str) -> None:
        """Programmatically highlight ``file_id`` in the list.

        Driven by ``MainWindow._on_canvas_loaded`` so the highlight
        follows the actively-loaded canvas (including programmatic opens
        from the picker / last-canvas auto-open).
        """
        self._current_canvas = file_id or ""
        if self._files_table is None or not self._current_canvas:
            return
        self._files_table.setSelection(_FILES_TAB_ID, [self._current_canvas])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self._dropdown = setComboBox([], height=26, i18n=False, parent=self)
        self._dropdown.currentIndexChanged.connect(self._on_dropdown_changed)
        header.addWidget(self._dropdown, 1)

        self._btn_refresh = setButton(
            "projects.refresh", 26, 26,
            kind="light", spec="none",
            icon="icon_reset", icon_only=True,
            default="",
        )
        i18n_bind(self._btn_refresh, "setToolTip",
                  "projects.refresh_tip", "Rescan projects folder")
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)

        self._btn_open_folder = setButton(
            "projects.open_folder", 26, 26,
            kind="light", spec="none",
            icon="icon_setting", icon_only=True,
            default="",
        )
        i18n_bind(
            self._btn_open_folder, "setToolTip",
            "projects.open_folder_tip", "Open the projects folder in OS file browser",
        )
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        header.addWidget(self._btn_open_folder)

        return header

    def _build_files_host(self) -> QWidget:
        """Container that hosts the panel-owned single-tab LaviTabTable."""
        host = QWidget(self)
        host.setObjectName("projectsFilesHost")
        host.setStyleSheet(
            "QWidget#projectsFilesHost { background: transparent;"
            " border: none; }"
        )
        self._files_table_layout = QVBoxLayout(host)
        self._files_table_layout.setContentsMargins(0, 0, 0, 0)
        self._files_table_layout.setSpacing(0)
        self._rebuild_files_table()
        return host

    def _rebuild_files_table(self) -> None:
        """Destroy + rebuild the LaviTabTable with the cached groups."""
        if self._files_table_layout is None:
            return
        if self._files_table is not None:
            try:
                self._files_table.selectionChanged.disconnect(
                    self._on_files_selection_changed
                )
            except (TypeError, RuntimeError):
                pass
            self._files_table_layout.removeWidget(self._files_table)
            self._files_table.setParent(None)
            self._files_table.deleteLater()
            self._files_table = None

        spec = [{
            "id": _FILES_TAB_ID,
            "default": "Canvas",
            "groups": list(self._canvas_groups),
        }]
        self._files_table = setLaviTabTable(
            spec, kind="single", parent=self,
        )
        self._files_table.setObjectName("projectsFilesTable")
        self._files_table.setTabEmptyHint(
            _FILES_TAB_ID, "projects.list.empty.canvas", "(no canvas files)",
        )
        # Hide the tab strip — degenerate 1-tab case; LaviTabTable always
        # paints ``_tab_bar``. Private-attr access acknowledged; follow-up
        # SDK request: add a public ``tab_bar_visible`` constructor flag.
        try:
            self._files_table._tab_bar.setVisible(False)
        except AttributeError:
            pass
        self._files_table.setStyleSheet(
            (self._files_table.styleSheet() or "")
            + "\nQWidget#projectsFilesTable { background: transparent; }"
        )
        # Strip the inner list-bg autofill and re-paint group headers with
        # the ``highlight`` slot — same treatment as the Scripts-mode
        # sidebar panels so the sidebar reads consistently across modes.
        restyle_tab_table_for_sidebar(self._files_table)
        self._files_table.selectionChanged.connect(self._on_files_selection_changed)
        self._files_table_layout.addWidget(self._files_table, 1)

        # Restore selection if the previously-loaded canvas survived the
        # rebuild (e.g. opened canvas + then project rescan refreshed groups).
        if self._current_canvas:
            self._files_table.setSelection(
                _FILES_TAB_ID, [self._current_canvas]
            )
            survivors = self._files_table.selectionMap().get(
                _FILES_TAB_ID, []
            )
            if not survivors:
                self._current_canvas = ""

    # ------------------------------------------------------------------
    # Snapshot rendering
    # ------------------------------------------------------------------
    def _on_snapshot_changed(self, _snapshot: list) -> None:
        self._repopulate_dropdown()

    def _repopulate_dropdown(self) -> None:
        snapshot = self._store.snapshot()
        self._projects_cache = list(snapshot)

        if not snapshot:
            placeholder = tr(
                "projects.empty",
                "No projects found",
            )
            items = [("__empty__", placeholder)]
        else:
            items = [(p.project_id, p.name) for p in snapshot]

        prev_key = self._dropdown.currentKey()
        self._suspend_dropdown_signal = True
        try:
            self._dropdown.setItems(items)
            if snapshot:
                if prev_key and not self._dropdown.setCurrentKey(prev_key):
                    self._dropdown.setCurrentIndex(0)
        finally:
            self._suspend_dropdown_signal = False

    def _current_info(self) -> Optional[ProjectInfo]:
        key = self._dropdown.currentKey()
        if not key or key == "__empty__":
            return None
        for info in self._projects_cache:
            if info.project_id == key:
                return info
        return None

    def _emit_current_selection(self) -> None:
        info = self._current_info()
        if info is None:
            return
        log_debug(f"[projects] selected {info.name} ({info.project_id})")
        self._store.add_recent(info.path)
        self.project_selected.emit(info.path)
        self.file_activated.emit(info.path)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def refresh_style(self) -> None:
        if self._files_table is not None:
            try:
                self._files_table.refresh_style()
            except Exception:
                pass
            self._files_table.setStyleSheet(
                (self._files_table.styleSheet() or "")
                + "\nQWidget#projectsFilesTable { background: transparent; }"
            )
            # SDK refresh_style re-runs ``_apply_list_bg`` (which calls
            # setAutoFillBackground(True)) and won't touch group-header
            # paintEvents. Re-strip + re-tint so theme switches stay clean.
            restyle_tab_table_for_sidebar(self._files_table)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_open_folder(self) -> None:
        if not self._store.open_in_explorer():
            log_warning(f"[projects] cannot open {self._store.projects_root()}")

    def _on_dropdown_changed(self, _index: int) -> None:
        if self._suspend_dropdown_signal:
            return
        self._emit_current_selection()

    def _on_files_selection_changed(
        self, _tab_id: str, items: list,
    ) -> None:
        item_id = items[0] if items else ""
        self._current_canvas = item_id
        if item_id:
            self.canvas_selected.emit(item_id)


__all__ = ["ProjectsPanel"]
