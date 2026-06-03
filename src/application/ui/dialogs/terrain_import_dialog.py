# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""TerrainImportDialog — import an external height-map as a custom terrain.

Lets the user pick a PNG / ``.npy`` / ``.npz`` height-map, supply the
physical tile size (and, for normalised PNGs, the white-pixel elevation),
and registers it as a reusable custom-terrain scene. All heavy lifting is
in :func:`application.training.scene_registry.import_terrain_scene` (which
converts to a canonical self-contained ``.npz`` under USER_CONFIG_DIR, runs
the cross-engine consistency gate at export, and adds it to the scene
picker) — this dialog is a thin form over it.

On accept the imported scene appears in the Play Ground Setting scene
picker; selecting it sets ``scene_type='custom'`` + ``custom_terrain_path``
automatically (the scene's ``defaults``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviLineEdit,
    i18n_bind,
    setButton,
    setLineEdit,
    tr,
)


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


def _mini_edit(edit: "LaviLineEdit") -> "LaviLineEdit":
    f = QFont(edit.font())
    f.setPixelSize(_mini())
    edit.setFont(f)
    return edit


class TerrainImportDialog(QDialog):
    """Modal dialog. ``exec()`` returns ``Accepted`` on a successful import;
    the imported scene_id is available via :attr:`imported_scene_id`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("terrainImportDialog")
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setWindowTitle(tr("terrain.import.title", "Import terrain"))
        self.setFixedWidth(480)

        self.imported_scene_id: str = ""
        self._src_edit: Optional[LaviLineEdit] = None
        self._id_edit: Optional[LaviLineEdit] = None
        self._name_edit: Optional[LaviLineEdit] = None
        self._sx_edit: Optional[LaviLineEdit] = None
        self._sy_edit: Optional[LaviLineEdit] = None
        self._elev_edit: Optional[LaviLineEdit] = None
        self._vscale_edit: Optional[LaviLineEdit] = None

        self._init_ui()

    # ----- UI ----------------------------------------------------------------

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        row = 0
        grid.addWidget(self._label("terrain.import.source", "Height-map file"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._src_edit = _mini_edit(setLineEdit(
            placeholder=tr("terrain.import.source_ph", ".png / .npy / .npz"),
            parent=self,
        ))
        grid.addWidget(self._src_edit, row, 1)
        browse = setButton(
            "terrain.import.browse", 72, 28, kind="border", spec="none",
            default=tr("terrain.import.browse", "Browse"),
        )
        browse.clicked.connect(self._on_browse)
        grid.addWidget(browse, row, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.scene_id", "Scene id"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._id_edit = _mini_edit(setLineEdit(
            placeholder=tr("terrain.import.scene_id_ph", "e.g. my_hills"),
            parent=self,
        ))
        grid.addWidget(self._id_edit, row, 1, 1, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.name", "Display name"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._name_edit = _mini_edit(setLineEdit(placeholder="", parent=self))
        grid.addWidget(self._name_edit, row, 1, 1, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.size_x", "Size X (m)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._sx_edit = _mini_edit(setLineEdit(placeholder="8.0", parent=self))
        grid.addWidget(self._sx_edit, row, 1, 1, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.size_y", "Size Y (m)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._sy_edit = _mini_edit(setLineEdit(placeholder="8.0", parent=self))
        grid.addWidget(self._sy_edit, row, 1, 1, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.elevation", "Elevation (m, PNG only)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._elev_edit = _mini_edit(setLineEdit(placeholder="0.5", parent=self))
        grid.addWidget(self._elev_edit, row, 1, 1, 2)

        row += 1
        grid.addWidget(self._label("terrain.import.vscale", "Height step (m)"),
                       row, 0, Qt.AlignmentFlag.AlignRight)
        self._vscale_edit = _mini_edit(setLineEdit(placeholder="0.005", parent=self))
        grid.addWidget(self._vscale_edit, row, 1, 1, 2)

        root.addLayout(grid)

        hint = self._label(
            "terrain.import.hint",
            "Size X/Y are required for .png and .npy. Elevation is the "
            "white-pixel height (PNG only). .npz files carry their own size.",
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = setButton(
            "terrain.import.ok", 110, 28, kind="normal", spec="save",
            default=tr("terrain.import.ok", "Import"),
        )
        ok_btn.clicked.connect(self._on_submit)
        btn_row.addWidget(ok_btn)
        cancel_btn = setButton(
            "terrain.import.cancel", 96, 28, kind="border", spec="none",
            default=tr("terrain.import.cancel", "Cancel"),
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        root.addLayout(btn_row)

    def _label(self, key: str, default: str) -> QLabel:
        lbl = QLabel(self)
        i18n_bind(lbl, "setText", key, default)
        lbl.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        return lbl

    # ----- behaviour ---------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("terrain.import.title", "Import terrain"),
            "",
            "Height maps (*.png *.npy *.npz)",
        )
        if path and self._src_edit is not None:
            self._src_edit.setText(path)

    @staticmethod
    def _opt_float(edit: Optional["LaviLineEdit"]) -> Optional[float]:
        if edit is None:
            return None
        txt = edit.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    def _on_submit(self) -> None:
        src = self._src_edit.text().strip() if self._src_edit else ""
        sid = self._id_edit.text().strip() if self._id_edit else ""
        if not src or not Path(src).is_file():
            self._warn(tr("terrain.import.err_src", "Pick a valid height-map file."))
            return
        if not sid:
            self._warn(tr("terrain.import.err_id", "Scene id is required."))
            return

        name = (self._name_edit.text().strip() if self._name_edit else "") or sid
        vscale = self._opt_float(self._vscale_edit) or 0.005

        from application.training.scene_registry import import_terrain_scene

        try:
            self.imported_scene_id = ""
            import_terrain_scene(
                src, sid, name,
                size_x=self._opt_float(self._sx_edit),
                size_y=self._opt_float(self._sy_edit),
                elevation_z=self._opt_float(self._elev_edit),
                vertical_scale=vscale,
            )
        except Exception as e:  # noqa: BLE001 - surface any import failure to the user
            QMessageBox.critical(
                self, tr("terrain.import.title", "Import terrain"),
                f"{type(e).__name__}: {e}",
            )
            return
        self.imported_scene_id = sid
        self.accept()

    def _warn(self, msg: str) -> None:
        QMessageBox.warning(self, tr("terrain.import.title", "Import terrain"), msg)


__all__ = ["TerrainImportDialog"]
