"""FieldEditor — Mission Runtime field authoring sub-page (P6-T3 v1).

A minimum-viable Qt page that lists every field discoverable through
:mod:`src.system.runtime.simulation.mujoco.field_loader` (project-local
+ shared), shows the active field's YAML in a read-only text view,
displays its rendered thumbnail (P6-T1), and lets the user create a
new project-local field by cloning a shared default.

What v1 does NOT do (deferred — see future_todos):
    * Drag-handles for prop placement on a 3D viewport
    * Live re-render as the user edits prop pos/size
    * Project-local field deletion / rename UI
    * Yaml validation feedback
    * Direct serialization of in-memory edits back to disk

These belong to a follow-up "Field Editor v2" pass; the v1 page is
intentionally a thin viewer + cloner so users can SEE what fields
exist without leaving the app.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class FieldEditor(QWidget):
    """Sub-page that lets users browse + clone Mission Runtime fields.

    Public signals
    --------------
    field_selected(str)
        Emitted with the active field_id whenever the user picks a
        new entry. Mission canvases can listen for this if they want
        to thread the selection back into MjFieldNode parameters.

    field_cloned(str)
        Emitted with the new field_id after a successful clone-to-
        project-local operation. Useful for Mission canvas refresh.
    """

    field_selected = Signal(str)
    field_cloned = Signal(str)

    def __init__(self, project_root: Optional[Path] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project_root: Optional[Path] = project_root
        self._current_field_id: str = ""
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project_root(self, project_root: Optional[Path]) -> None:
        """Switch the active project. Refreshes the listing so any
        project-local fields under the new root become visible.
        """
        self._project_root = project_root
        self.refresh()

    def current_field_id(self) -> str:
        return self._current_field_id

    def refresh(self) -> None:
        """Re-scan project-local + shared field libraries and rebuild
        the list view. Preserves the current selection when possible.
        """
        from src.system.runtime.simulation.mujoco import (
            list_all_fields,
            list_project_fields,
            list_shared_fields,
        )

        previous = self._current_field_id
        self._list.clear()

        try:
            shared = set(list_shared_fields())
            local = set(list_project_fields(self._project_root))
        except Exception as exc:  # noqa: BLE001
            log.warning("FieldEditor.refresh: listing failed: %s", exc)
            shared, local = set(), set()

        merged = sorted(shared | local)
        for fid in merged:
            scope = "local" if fid in local else "shared"
            label = f"{fid}    [{scope}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, fid)
            self._list.addItem(item)

        if previous:
            self._select_field_id(previous)
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal, self)
        outer.addWidget(splitter)

        # ── Left: field list ─────────────────────────────────────────
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Fields", left))
        self._list = QListWidget(left)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh", left)
        self._refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self._refresh_btn)
        self._clone_btn = QPushButton("Clone to project", left)
        self._clone_btn.clicked.connect(self._on_clone_clicked)
        btn_row.addWidget(self._clone_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # ── Right: detail panel ──────────────────────────────────────
        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._title_lbl = QLabel("(no field selected)", right)
        self._title_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self._title_lbl)

        self._thumb_lbl = QLabel(right)
        self._thumb_lbl.setFixedSize(256, 256)
        self._thumb_lbl.setAlignment(Qt.AlignCenter)
        self._thumb_lbl.setStyleSheet(
            "background-color: #1a1a22; border: 1px solid #2a2a35;"
        )
        right_layout.addWidget(self._thumb_lbl, alignment=Qt.AlignLeft)

        self._yaml_view = QPlainTextEdit(right)
        self._yaml_view.setReadOnly(True)
        self._yaml_view.setPlaceholderText("(field YAML will appear here)")
        right_layout.addWidget(self._yaml_view, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    # ------------------------------------------------------------------
    # Selection / detail rendering
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._current_field_id = ""
            self._title_lbl.setText("(no field selected)")
            self._yaml_view.clear()
            self._thumb_lbl.clear()
            return
        field_id = str(items[0].data(Qt.UserRole) or "")
        self._current_field_id = field_id
        self.field_selected.emit(field_id)
        self._render_detail(field_id)

    def _select_field_id(self, field_id: str) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if str(item.data(Qt.UserRole) or "") == field_id:
                self._list.setCurrentRow(i)
                return

    def _render_detail(self, field_id: str) -> None:
        from src.system.runtime.simulation.mujoco import resolve_field_path
        from src.system.tools.field_thumbnail import field_thumbnail_qimage

        self._title_lbl.setText(field_id)

        # YAML view
        try:
            path = resolve_field_path(field_id, project_root=self._project_root)
            self._yaml_view.setPlainText(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._yaml_view.setPlainText(f"# failed to read field yaml: {exc}")

        # Thumbnail
        try:
            img = field_thumbnail_qimage(
                field_id,
                size=(256, 256),
                project_root=self._project_root,
            )
            if img is not None and not img.isNull():
                self._thumb_lbl.setPixmap(QPixmap.fromImage(img))
            else:
                self._thumb_lbl.setText("(no thumbnail)")
        except Exception as exc:  # noqa: BLE001
            log.warning("FieldEditor: thumbnail render failed: %s", exc)
            self._thumb_lbl.setText("(thumbnail error)")

    # ------------------------------------------------------------------
    # Clone-to-project
    # ------------------------------------------------------------------

    def _on_clone_clicked(self) -> None:
        """Copy the selected field's yaml into
        ``<project_root>/scenarios/fields/<field_id>.yaml`` so the user
        can edit it locally. P6-T3 v1 doesn't ship an in-app yaml
        editor — users open the file in their text editor of choice.
        """
        if not self._current_field_id:
            return
        if self._project_root is None:
            QMessageBox.information(
                self,
                "No active project",
                "Cannot clone — no active project root. Open a project "
                "via the workspace dropdown first.",
            )
            return

        from src.system.runtime.simulation.mujoco import resolve_field_path

        try:
            src_path = resolve_field_path(
                self._current_field_id, project_root=self._project_root
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Clone failed", f"Resolve failed: {exc}")
            return

        dst_dir = Path(self._project_root) / "scenarios" / "fields"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{self._current_field_id}.yaml"

        if src_path.resolve() == dst_path.resolve():
            QMessageBox.information(
                self,
                "Already local",
                f"'{self._current_field_id}' is already a project-local field.",
            )
            return

        if dst_path.exists():
            QMessageBox.warning(
                self,
                "Already exists",
                f"A project-local field named '{self._current_field_id}' "
                f"already exists at {dst_path}.",
            )
            return

        try:
            dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Clone failed", f"Write failed: {exc}")
            return

        self.field_cloned.emit(self._current_field_id)
        self.refresh()
        QMessageBox.information(
            self,
            "Cloned",
            f"Cloned to {dst_path}. Edit the file in your text editor of "
            f"choice and click Refresh to reload.",
        )
