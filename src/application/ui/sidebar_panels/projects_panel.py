"""ProjectsPanel — project picker + Canvas/Script files_list host.

The panel is two-pieced:

    +---------------------------------------+
    | [ project dropdown ▾ ] [↻] [folder] |
    +---------------------------------------+
    |                                       |
    |       files_list (Canvas / Script)    |
    |       — adopted from MissionControlPanel —
    |                                       |
    +---------------------------------------+

* The dropdown enumerates every project under ``Paths.PROJECTS_DIR``
  via ``ProjectStore.snapshot()``. Selecting an entry emits
  ``project_selected(Path)``; ``MainWindow`` bridges it to ``open_project``,
  which re-runs ``_bind_project`` and feeds the new ``set_canvas_groups`` /
  ``set_script_groups`` payload into the mounted files_list.
* The lower half is a vacant container until ``mount_files_list(widget)``
  is called by ``MainWindow`` (after both Sidebar and MissionControlPanel
  are built). The widget is the same ``files_list`` MissionControlPanel
  used to host on its left column — its UI and its signal connections are
  unchanged; only its parent chain moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, log_debug, log_warning, setButton, setComboBox, tr

from application.service.projects import ProjectInfo, get_project_store


class ProjectsPanel(QWidget):
    """Project dropdown + refresh + open-folder + mounted files_list."""

    # Same name and (Path,) signature as before — MainWindow's bridge
    # (``open_project``) does not need to change.
    file_activated = pyqtSignal(Path)
    project_selected = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._store = get_project_store()
        self._projects_cache: List[ProjectInfo] = []
        self._suspend_dropdown_signal: bool = False

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
        """Sync the dropdown to ``path`` without emitting selection signals.

        ``MainWindow`` calls this on every ``_bind_project`` so the dropdown
        always reflects the currently bound project. The selection is
        round-tripped through ``project_id`` (the LaviComboBox key) so it
        survives snapshot rescans where the underlying list is rebuilt.
        """
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

    def mount_files_list(self, widget: QWidget) -> None:
        """Adopt the orphan ``files_list`` widget into the panel.

        Called once by ``MainWindow._build_main_page`` after MissionControlPanel
        has been constructed. The widget's parent chain changes; its inner
        signal wiring (LaviTabTable → MissionControlPanel slots) is preserved.
        """
        if self._files_host_layout is None or widget is None:
            return
        # Idempotent: removing first lets callers re-mount without leaks.
        for i in reversed(range(self._files_host_layout.count())):
            item = self._files_host_layout.itemAt(i)
            if item is None:
                continue
            existing = item.widget()
            if existing is not None:
                self._files_host_layout.removeWidget(existing)
                existing.setParent(None)
        self._files_host_layout.addWidget(widget, 1)
        widget.setVisible(True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        # Initial empty list — repopulated from the snapshot below. We use
        # i18n=False because project names / ids aren't translation keys.
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
        """Empty container the relocated files_list will be parented into."""
        host = QWidget(self)
        host.setObjectName("projectsFilesHost")
        self._files_host = host
        self._files_host_layout = QVBoxLayout(host)
        self._files_host_layout.setContentsMargins(0, 0, 0, 0)
        self._files_host_layout.setSpacing(0)
        # Transparent host — the inner files_list owns its own theming via
        # MissionControlPanel, so the sidebar panel doesn't double up a
        # background/border around it.
        host.setStyleSheet(
            "QWidget#projectsFilesHost { background: transparent;"
            " border: none; }"
        )
        return host

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
            # i18n=False: ``default`` is what's displayed. project_id is the
            # stable key we use to round-trip selection across rescans.
            items = [(p.project_id, p.name) for p in snapshot]

        # Preserve current selection (by project_id) across the rebuild.
        # We never auto-emit here: snapshot refreshes happen as a side
        # effect of ``open_project -> refresh -> Sidebar.refresh`` and
        # re-emitting would loop back into open_project.
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
        # Backwards-compatible alias for the previous activation signal,
        # so MainWindow's existing ``file_activated -> open_project``
        # bridge keeps firing.
        self.file_activated.emit(info.path)

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


__all__ = ["ProjectsPanel"]
