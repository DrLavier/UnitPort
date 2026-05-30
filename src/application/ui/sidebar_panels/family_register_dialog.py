# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""FamilyRegisterDialog — add / edit a user-overlay IR family.

Opened from :class:`RobotEditorDialog` via the "+" button next to the
Families selector. Persists to ``<USER_CONFIG_DIR>/registers/ir_custom.json``
under the new ``families`` block (see ``registers/__init__.py`` →
``_merge_user_overlays``).

Fields:
    family_id  — slug, ``[a-z0-9_]+``; unique against
                 ``registers.ir.list_families()``.
    label      — display name.
    alias_of   — existing family whose roles are inherited at load time.
    roles      — extra roles layered on top of the inherited set
                 (``id / category / label / position / required``).

On Save we round-trip the existing ``ir_custom.json`` via
:func:`registers.ir.persist_user_family` (so the role-level
``extensions`` block is preserved untouched), then trigger
``RegistryHub.reload() + validate()``.

UI follows :class:`RobotEditorDialog` conventions: size_small everywhere,
colors via ``Config.get_color(slot)``, no hex literals or font literals
(CLAUDE.md §1.5).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviComboBox,
    LaviLineEdit,
    i18n_bind,
    log_warning,
    setButton,
    setText,
    tr,
)

# Slug grammar: lower-case alphanumerics + underscore; same shape canonical
# family ids use ("quadruped", "biped"). Rejected at save time so the user
# cannot register a family that resolve_id() can never reach.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _ss() -> int:
    """Current size_small px (CLAUDE.md §1.5)."""
    return int(Config.get_font_size("size_small"))


class _RoleTable(QTableWidget):
    """Five-column table for the user's explicit role list.

    Columns: id / category / label / position / required. Row widgets are
    plain QTableWidgetItem (text) plus a centered QCheckBox in the required
    column. The inherited (alias_of) preview is shown at the top as
    read-only rows the user cannot delete.
    """

    COLS = ("id", "category", "label", "position", "required")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels([
            tr("family_register.col_id", "Role id"),
            tr("family_register.col_category", "Category"),
            tr("family_register.col_label", "Label"),
            tr("family_register.col_position", "Position"),
            tr("family_register.col_required", "Required"),
        ])
        bg = Config.get_color("bg_2")
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        alt = Config.get_color("row_1")
        sz = _ss()
        self.setStyleSheet(
            f"QTableWidget {{ background: {bg}; color: {fg};"
            f" border: 1px solid {border}; gridline-color: {border};"
            f" font-size: {sz}px; }}"
            f"QHeaderView::section {{ background: {alt}; color: {fg};"
            f" border: 0; padding: 3px 6px; font-size: {sz}px; }}"
        )
        self.verticalHeader().setVisible(False)
        for i in range(len(self.COLS) - 1):
            self.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(
            len(self.COLS) - 1, QHeaderView.ResizeMode.ResizeToContents,
        )
        self.setMinimumHeight(140)

    def _required_cell(self, checked: bool, *, enabled: bool = True) -> QWidget:
        host = QWidget(self)
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cb = QCheckBox(host)
        cb.setChecked(bool(checked))
        cb.setEnabled(enabled)
        lay.addWidget(cb)
        host._checkbox = cb  # type: ignore[attr-defined]
        return host

    def add_row(
        self,
        role_id: str = "",
        category: str = "",
        label: str = "",
        position: str = "",
        required: bool = False,
        *,
        read_only: bool = False,
    ) -> None:
        r = self.rowCount()
        self.insertRow(r)
        for c, value in enumerate((role_id, category, label, position)):
            item = QTableWidgetItem(str(value))
            if read_only:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # Greyed-out colour to signal "inherited, not editable here".
                item.setForeground(_grey_brush())
            self.setItem(r, c, item)
        self.setCellWidget(r, 4, self._required_cell(required, enabled=not read_only))
        if read_only:
            for c in range(self.columnCount()):
                item = self.item(r, c)
                if item is not None:
                    item.setData(Qt.ItemDataRole.UserRole, "inherited")

    def is_read_only_row(self, r: int) -> bool:
        item = self.item(r, 0)
        return bool(item and item.data(Qt.ItemDataRole.UserRole) == "inherited")

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        for r in rows:
            if self.is_read_only_row(r):
                continue  # inherited preview rows are not user-deletable
            self.removeRow(r)

    def collect_user_roles(self) -> List[Dict[str, Any]]:
        """Return only the user-editable (non-inherited) roles."""
        out: List[Dict[str, Any]] = []
        for r in range(self.rowCount()):
            if self.is_read_only_row(r):
                continue
            rid = (self.item(r, 0).text() if self.item(r, 0) else "").strip()
            if not rid:
                continue
            host = self.cellWidget(r, 4)
            cb = getattr(host, "_checkbox", None) if host is not None else None
            required = bool(cb.isChecked()) if cb is not None else False
            out.append({
                "id": rid,
                "category": (self.item(r, 1).text() if self.item(r, 1) else "").strip(),
                "label": (self.item(r, 2).text() if self.item(r, 2) else "").strip(),
                "position": (self.item(r, 3).text() if self.item(r, 3) else "").strip(),
                "required": required,
            })
        return out

    def clear_all_inherited(self) -> None:
        """Remove every row currently marked as inherited (preview header)."""
        for r in range(self.rowCount() - 1, -1, -1):
            if self.is_read_only_row(r):
                self.removeRow(r)


def _grey_brush():
    from PyQt6.QtGui import QBrush, QColor
    return QBrush(QColor(Config.get_color("sub_t2")))


class FamilyRegisterDialog(QDialog):
    """Register a new IR family in the user overlay, or edit an existing
    user-overlay family. Returns the persisted family id via
    :attr:`saved_family_id` on Accept."""

    def __init__(
        self,
        existing_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._existing_id = (existing_id or "").strip().lower() or None
        self.saved_family_id: Optional[str] = None

        is_edit = self._existing_id is not None
        if is_edit:
            i18n_bind(
                self, "setWindowTitle",
                "family_register.title_edit", "Edit family",
            )
        else:
            i18n_bind(
                self, "setWindowTitle",
                "family_register.title_new", "Register new family",
            )
        self.setModal(True)
        self.resize(560, 520)

        bg = Config.get_color("bg_3")
        fg = Config.get_color("main_t1")
        sz = _ss()
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; color: {fg}; font-size: {sz}px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        host = QWidget()
        form = QVBoxLayout(host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)

        # ----- intro / why -------------------------------------------------
        intro = setText(
            "family_register.intro",
            default=(
                "Register a custom IR family. Pick a base family to inherit "
                "its roles, or add extra roles below — both are layered into "
                "a single role list for joint mapping."
            ),
            kind="content", size=sz,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            f"color: {Config.get_color('main_c1')};"
            f" background: transparent; padding: 4px 0;"
        )
        form.addWidget(intro)

        # ----- family_id / label / alias_of --------------------------------
        existing_spec: Dict[str, Any] = {}
        if is_edit:
            from registers import ir as _ir
            existing_spec = _ir.get_user_family_spec(self._existing_id) or {}

        self._id_edit = LaviLineEdit(
            text=self._existing_id or "",
            placeholder=tr("family_register.id_ph",
                           "e.g. quadruped_with_tail"),
        )
        if is_edit:
            self._id_edit.setEnabled(False)
        form.addLayout(self._labelled(
            "family_register.id_label", "Family id", self._id_edit,
        ))

        self._label_edit = LaviLineEdit(
            text=str(existing_spec.get("label") or ""),
            placeholder=tr("family_register.label_ph", "Display name"),
        )
        form.addLayout(self._labelled(
            "family_register.label_label", "Label", self._label_edit,
        ))

        # alias_of dropdown — every loaded family (canonical + user) minus
        # the family we're editing (cannot alias_of self).
        alias_items: List[tuple] = [("", tr("family_register.alias_none", "(none)"))]
        try:
            from registers import ir as _ir
            for fam in _ir.list_families():
                if fam == self._existing_id:
                    continue
                alias_items.append((fam, fam))
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[family_register] list_families failed: {exc}")
        self._alias_combo = LaviComboBox(alias_items, i18n=False)
        cur_alias = str(existing_spec.get("alias_of") or "")
        if cur_alias:
            self._alias_combo.setCurrentKey(cur_alias)
        self._alias_combo.currentIndexChanged.connect(self._on_alias_changed)
        form.addLayout(self._labelled(
            "family_register.alias_of_label", "Base family (alias_of)",
            self._alias_combo,
        ))

        # ----- roles table -------------------------------------------------
        roles_header = QHBoxLayout()
        roles_header.addWidget(setText(
            "family_register.roles_section", default="Extra roles",
            kind="title", size=sz,
        ), 1)
        self._btn_add_role = setButton(
            "family_register.add_role", 80, 22,
            kind="border", spec="none",
            default=tr("family_register.add_role", "+ Role"),
        )
        self._btn_add_role.clicked.connect(lambda: self._roles.add_row())
        roles_header.addWidget(self._btn_add_role)
        self._btn_del_role = setButton(
            "family_register.del_role", 80, 22,
            kind="border", spec="danger",
            default=tr("family_register.del_role", "− Role"),
        )
        self._btn_del_role.clicked.connect(lambda: self._roles.remove_selected())
        roles_header.addWidget(self._btn_del_role)
        form.addLayout(roles_header)

        self._roles = _RoleTable()
        form.addWidget(self._roles)

        # Inherited preview rendered first (so the user sees what alias_of
        # brings in), then existing user-added roles below.
        self._refresh_inherited_preview()
        for r in (existing_spec.get("roles") or []):
            if not isinstance(r, dict):
                continue
            self._roles.add_row(
                role_id=str(r.get("id", "")),
                category=str(r.get("category", "")),
                label=str(r.get("label", "")),
                position=str(r.get("position", "")),
                required=bool(r.get("required", False)),
            )

        # ----- buttons -----------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addStretch(1)
        if is_edit:
            self._btn_delete = setButton(
                "family_register.delete", 80, 26,
                kind="border", spec="danger",
                default=tr("family_register.delete", "Delete"),
            )
            self._btn_delete.clicked.connect(self._on_delete)
            btn_row.addWidget(self._btn_delete)
        self._btn_cancel = setButton(
            "family_register.cancel", 80, 26,
            kind="border", spec="none",
            default=tr("family_register.cancel", "Cancel"),
        )
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        self._btn_save = setButton(
            "family_register.save", 80, 26,
            kind="normal", spec="save",
            default=tr("family_register.save", "Save"),
        )
        self._btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self._btn_save)
        outer.addLayout(btn_row)

    # ----- helpers --------------------------------------------------------

    def _labelled(self, i18n_key: str, default: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = setText(i18n_key, default=f"{default}:",
                        kind="content", size=_ss())
        label.setMinimumWidth(140)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return row

    def _on_alias_changed(self) -> None:
        self._refresh_inherited_preview()

    def _refresh_inherited_preview(self) -> None:
        """Re-render the read-only inherited-roles preview rows at the top of
        the table whenever ``alias_of`` changes."""
        self._roles.clear_all_inherited()
        alias = (self._alias_combo.currentKey() or "").strip().lower()
        if not alias:
            return
        try:
            from registers import ir as _ir
            inherited = _ir.list_roles(alias)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[family_register] list_roles({alias!r}) failed: {exc}")
            return
        # Insert inherited rows at the TOP, in declaration order. Because
        # _RoleTable.add_row always appends, we instead build a temporary
        # snapshot of existing user rows, clear, then re-insert inherited
        # first, then user rows.
        user_snapshot = self._roles.collect_user_roles()
        # Wipe everything (read-only and user rows) and rebuild.
        while self._roles.rowCount() > 0:
            self._roles.removeRow(0)
        for role in inherited:
            self._roles.add_row(
                role_id=str(role.get("id", "")),
                category=str(role.get("category", "")),
                label=str(role.get("label", "")),
                position=str(role.get("position", "")),
                required=bool(role.get("required", False)),
                read_only=True,
            )
        for role in user_snapshot:
            self._roles.add_row(
                role_id=str(role.get("id", "")),
                category=str(role.get("category", "")),
                label=str(role.get("label", "")),
                position=str(role.get("position", "")),
                required=bool(role.get("required", False)),
            )

    # ----- actions --------------------------------------------------------

    def _validate(self) -> Optional[str]:
        """Return the validated family id, or None on error (dialog stays
        open after showing a QMessageBox)."""
        if self._existing_id is not None:
            fid = self._existing_id
        else:
            fid = (self._id_edit.text() or "").strip().lower()
            if not _SLUG_RE.match(fid):
                QMessageBox.warning(
                    self,
                    tr("family_register.error_title", "Invalid input"),
                    tr("family_register.error_slug",
                       "Family id must be lower-case ASCII letters / digits / "
                       "underscores and start with a letter (e.g. "
                       "'quadruped_with_tail')."),
                )
                return None
            try:
                from registers import ir as _ir
                if fid in _ir.list_families():
                    QMessageBox.warning(
                        self,
                        tr("family_register.error_title", "Invalid input"),
                        tr("family_register.error_duplicate",
                           "Family id {fid!r} is already registered "
                           "(canonical or user-layer).").format(fid=fid),
                    )
                    return None
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[family_register] uniqueness check failed: {exc}")
        # Role id slug check
        seen_ids: set = set()
        for r in self._roles.collect_user_roles():
            rid = r.get("id", "")
            if not _SLUG_RE.match(rid):
                QMessageBox.warning(
                    self,
                    tr("family_register.error_title", "Invalid input"),
                    tr("family_register.error_role_slug",
                       "Role id {rid!r} must match [a-z][a-z0-9_]*.").format(rid=rid),
                )
                return None
            if rid in seen_ids:
                QMessageBox.warning(
                    self,
                    tr("family_register.error_title", "Invalid input"),
                    tr("family_register.error_role_duplicate",
                       "Role id {rid!r} appears more than once.").format(rid=rid),
                )
                return None
            seen_ids.add(rid)
        return fid

    def _on_save(self) -> None:
        fid = self._validate()
        if fid is None:
            return
        spec: Dict[str, Any] = {
            "label": (self._label_edit.text() or "").strip() or fid,
            "alias_of": (self._alias_combo.currentKey() or "").strip().lower() or "",
            "roles": self._roles.collect_user_roles(),
        }
        if not spec["alias_of"]:
            # Drop the empty alias_of so the on-disk file stays clean —
            # absent vs. empty are equivalent for the merger.
            spec.pop("alias_of", None)
        if not spec["roles"]:
            spec.pop("roles", None)
        try:
            from registers import RegistryHub, RegistryValidationError
            from registers import ir as _ir
            ok = _ir.persist_user_family(fid, spec)
            if not ok:
                raise RuntimeError("persist_user_family returned False")
            RegistryHub.reload()
            RegistryHub.validate()
        except RegistryValidationError as exc:
            QMessageBox.critical(
                self,
                tr("family_register.error_title", "Save failed"),
                str(exc),
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self,
                tr("family_register.error_title", "Save failed"),
                str(exc),
            )
            return
        self.saved_family_id = fid
        # Surface the change to listeners (asset panel will refresh its
        # families dropdown on next dialog open).
        try:
            from application.service.robot_assets import get_robot_asset_service
            get_robot_asset_service().changed.emit()
        except Exception:  # noqa: BLE001
            pass
        self.accept()

    def _on_delete(self) -> None:
        if self._existing_id is None:
            return
        fid = self._existing_id
        confirm = QMessageBox.question(
            self,
            tr("family_register.confirm_delete_title", "Delete family"),
            tr("family_register.confirm_delete_body",
               "Remove the user-overlay family {fid!r}? Canonical families "
               "are not affected. Robots that reference this family will "
               "block the delete.").format(fid=fid),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            from registers import RegistryHub, ir as _ir
            try:
                ok = _ir.delete_user_family(fid)
            except _ir.CascadeProtectionError as exc:
                QMessageBox.warning(
                    self,
                    tr("family_register.cascade_title", "Cannot delete"),
                    str(exc),
                )
                return
            if not ok:
                return
            RegistryHub.reload()
        except Exception as exc:
            QMessageBox.critical(
                self,
                tr("family_register.error_title", "Delete failed"),
                str(exc),
            )
            return
        self.saved_family_id = None
        try:
            from application.service.robot_assets import get_robot_asset_service
            get_robot_asset_service().changed.emit()
        except Exception:  # noqa: BLE001
            pass
        self.accept()


__all__ = ["FamilyRegisterDialog"]
