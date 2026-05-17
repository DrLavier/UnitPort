"""RobotEditorDialog — add / edit a user-layer robot register entry.

Opened from RobotAssetsPanel via the "+" button (new) or per-card "Edit"
(existing user-layer SKU). Saves through RobotAssetService.add_user_robot /
update_user_robot, which persist to ``~/UnitPort/registers/robots_custom.json``
and trigger ``RegistryHub.reload() + validate()``.

Fields:
    Brand        — line edit (required, free text — folded by build_sku).
    Model        — line edit (required).
    Name         — display name shown in UI.
    Families     — TagInput (free-tag, e.g. "quadruped" / "humanoid").
    Adapter      — combo box, candidates from registers.brands.list_built_in_adapters.
    Asset paths  — 4 rows (MJCF / URDF / USD / XACRO) with browse buttons.
    Joints       — QTableWidget (joint_name, ir_role); ir_role candidates pulled
                   from registers.ir.list_roles_for_families(*families).

All text uses size_small per CLAUDE.md §1.5; all colors via Config.get_color
against system.ini[Theme] slots (no fallback hex literals).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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
    TagInput,
    i18n_bind,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.robot_assets import get_robot_asset_service

ASSET_KIND_ORDER = ("MJCF", "URDF", "USD", "XACRO")


def _ss() -> int:
    """Current size_small px."""
    return int(Config.get_font_size("size_small"))


class _PathRow(QWidget):
    """One asset-kind row: kind label + LaviLineEdit + browse button."""

    def __init__(self, kind: str, initial: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._kind = kind

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        label = setText(
            f"robot_editor.kind.{kind.lower()}", default=f"{kind}:",
            kind="content", size=_ss(),
        )
        label.setMinimumWidth(54)
        row.addWidget(label)

        self._edit = LaviLineEdit(text=initial,
                                  placeholder=tr("robot_editor.path_placeholder",
                                                 "Relative or absolute path"))
        row.addWidget(self._edit, 1)

        self._browse = setButton(
            f"robot_editor.browse.{kind.lower()}", 60, 22,
            kind="border", spec="none",
            default=tr("robot_editor.browse", "..."),
        )
        self._browse.clicked.connect(self._on_browse)
        row.addWidget(self._browse)

    def _on_browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            tr("robot_editor.browse_title", "Select {kind} file").format(kind=self._kind),
            self._edit.text() or "",
        )
        if chosen:
            self._edit.setText(chosen)

    def value(self) -> str:
        return (self._edit.text() or "").strip()


class _JointTable(QTableWidget):
    """joint_name + ir_role table; ir_role choices reflect families."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels([
            tr("robot_editor.col_joint", "Joint name"),
            tr("robot_editor.col_role", "IR role"),
        ])
        self._role_choices: List[str] = []

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
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.setMinimumHeight(120)

    def set_role_choices(self, choices: List[str]) -> None:
        self._role_choices = list(choices)
        # Rebuild combos in existing rows so new families take effect.
        for r in range(self.rowCount()):
            current = ""
            old = self.cellWidget(r, 1)
            if isinstance(old, QComboBox):
                current = old.currentText()
            combo = self._make_role_combo(current)
            self.setCellWidget(r, 1, combo)

    def _make_role_combo(self, current: str = "") -> QComboBox:
        combo = QComboBox()
        combo.addItem("")
        for role_id in self._role_choices:
            combo.addItem(role_id)
        if current:
            idx = combo.findText(current)
            if idx < 0:
                combo.addItem(current)
                idx = combo.findText(current)
            combo.setCurrentIndex(idx)
        bg = Config.get_color("bg_2")
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        sz = _ss()
        combo.setStyleSheet(
            f"QComboBox {{ background: {bg}; color: {fg};"
            f" border: 1px solid {border}; padding: 2px 6px; font-size: {sz}px; }}"
        )
        return combo

    def add_row(self, joint_name: str = "", ir_role: str = "") -> None:
        r = self.rowCount()
        self.insertRow(r)
        item = QTableWidgetItem(joint_name)
        self.setItem(r, 0, item)
        self.setCellWidget(r, 1, self._make_role_combo(ir_role))

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        for r in rows:
            self.removeRow(r)

    def collect(self) -> Dict[str, Dict[str, str]]:
        """Return joints dict keyed by build_sku(robot_sku, joint_name).

        Robot SKU is unknown to the table; caller wraps each entry under the
        ``joints`` key of the new robot. Here we just return name+ir_role; the
        service layer rebuilds the joint SKU.
        """
        out: Dict[str, Dict[str, str]] = {}
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            jname = (name_item.text() if name_item else "").strip()
            if not jname:
                continue
            combo = self.cellWidget(r, 1)
            jrole = combo.currentText().strip() if isinstance(combo, QComboBox) else ""
            out[jname] = {"name": jname, "ir_role": jrole}
        return out


class RobotEditorDialog(QDialog):
    """Add / edit a user-layer robot. Returns sku via :attr:`saved_sku` on Accept."""

    def __init__(
        self,
        existing: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._svc = get_robot_asset_service()
        self._existing = existing
        self.saved_sku: Optional[str] = None

        is_edit = existing is not None
        i18n_bind(
            self, "setWindowTitle",
            "robot_editor.title_edit" if is_edit else "robot_editor.title_new",
            "Edit Robot" if is_edit else "Add Robot",
        )
        self.setModal(True)
        self.resize(520, 600)

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

        # Brand / Model / Name
        self._brand = LaviLineEdit(text=str((existing or {}).get("brand", "")),
                                   placeholder=tr("robot_editor.brand_ph", "e.g. unitree"))
        self._model = LaviLineEdit(text=str((existing or {}).get("model", "")),
                                   placeholder=tr("robot_editor.model_ph", "e.g. go2"))
        self._name = LaviLineEdit(text=str((existing or {}).get("name", "")),
                                  placeholder=tr("robot_editor.name_ph", "Display name"))
        if is_edit:
            self._brand.setEnabled(False)
            self._model.setEnabled(False)

        form.addLayout(self._labelled("robot_editor.brand", "Brand", self._brand))
        form.addLayout(self._labelled("robot_editor.model", "Model", self._model))
        form.addLayout(self._labelled("robot_editor.name", "Name", self._name))

        # Families (TagInput)
        self._families = TagInput(
            mode="input",
            placeholder_key="robot_editor.family_placeholder",
            placeholder=tr("robot_editor.family_placeholder", "Add family (humanoid, quadruped, …)"),
        )
        self._families.set_values(list((existing or {}).get("families", []) or []))
        self._families.changed.connect(self._on_families_changed)
        form.addLayout(self._labelled("robot_editor.families", "Families", self._families))

        # Adapter
        adapter_items = self._adapter_items((existing or {}).get("adapter", ""))
        self._adapter = LaviComboBox(adapter_items, i18n=False)
        cur_adapter = str((existing or {}).get("adapter", ""))
        if cur_adapter:
            self._adapter.setCurrentKey(cur_adapter)
        form.addLayout(self._labelled("robot_editor.adapter", "Adapter", self._adapter))

        # Asset paths
        form.addWidget(setText(
            "robot_editor.assets_section", default="Asset paths",
            kind="title", size=sz,
        ))
        existing_assets = dict((existing or {}).get("assets", {}) or {})
        self._path_rows: Dict[str, _PathRow] = {}
        for kind in ASSET_KIND_ORDER:
            initial = existing_assets.get(kind) or existing_assets.get(kind.lower()) or ""
            row = _PathRow(kind, initial=str(initial or ""))
            self._path_rows[kind] = row
            form.addWidget(row)

        # Joints table
        joints_header = QHBoxLayout()
        joints_header.addWidget(setText(
            "robot_editor.joints_section", default="Joints",
            kind="title", size=sz,
        ), 1)
        self._btn_add_joint = setButton(
            "robot_editor.add_joint", 60, 22,
            kind="border", spec="none",
            default=tr("robot_editor.add_joint", "+ Joint"),
        )
        self._btn_add_joint.clicked.connect(lambda: self._joints.add_row())
        joints_header.addWidget(self._btn_add_joint)
        self._btn_del_joint = setButton(
            "robot_editor.del_joint", 60, 22,
            kind="border", spec="danger",
            default=tr("robot_editor.del_joint", "− Joint"),
        )
        self._btn_del_joint.clicked.connect(lambda: self._joints.remove_selected())
        joints_header.addWidget(self._btn_del_joint)
        form.addLayout(joints_header)

        self._joints = _JointTable()
        form.addWidget(self._joints)
        self._refresh_role_choices()
        for jentry in (existing or {}).get("joints", {}).values() or []:
            if isinstance(jentry, dict):
                self._joints.add_row(
                    str(jentry.get("name", "")),
                    str(jentry.get("ir_role", "")),
                )

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addStretch(1)
        if is_edit and self._svc.is_user_extension(str(existing.get("sku", ""))):
            self._btn_delete = setButton(
                "robot_editor.delete", 80, 26,
                kind="border", spec="danger",
                default=tr("robot_editor.delete", "Delete"),
            )
            self._btn_delete.clicked.connect(self._on_delete)
            btn_row.addWidget(self._btn_delete)
        self._btn_cancel = setButton(
            "robot_editor.cancel", 80, 26,
            kind="border", spec="none",
            default=tr("robot_editor.cancel", "Cancel"),
        )
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        self._btn_save = setButton(
            "robot_editor.save", 80, 26,
            kind="normal", spec="save",
            default=tr("robot_editor.save", "Save"),
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
        label.setMinimumWidth(80)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return row

    def _adapter_items(self, current: str) -> List[tuple]:
        try:
            from registers import brands
            built_in = list(brands.list_built_in_adapters().keys())
        except Exception:
            built_in = []
        items: List[tuple] = [("", tr("robot_editor.adapter_none", "(none)"))]
        for aid in built_in:
            items.append((aid, aid))
        if current and not any(k == current for k, _ in items):
            items.append((current, current))
        return items

    def _on_families_changed(self) -> None:
        self._refresh_role_choices()

    def _refresh_role_choices(self) -> None:
        families = list(self._families.get_values() or [])
        choices: List[str] = []
        try:
            from registers import ir as _ir
            for role in _ir.list_roles_for_families(*families, include_user_extensions=True):
                rid = role.get("id", "")
                if rid and rid not in choices:
                    choices.append(rid)
        except Exception as exc:
            log_warning(f"[robot_editor] ir role lookup failed: {exc}")
        self._joints.set_role_choices(choices)

    # ----- actions --------------------------------------------------------

    def _on_save(self) -> None:
        brand = self._brand.text().strip()
        model = self._model.text().strip()
        name = self._name.text().strip() or model
        if not brand or not model:
            QMessageBox.warning(
                self,
                tr("robot_editor.error_title", "Invalid input"),
                tr("robot_editor.error_required", "Brand and Model are required."),
            )
            return
        families = list(self._families.get_values() or [])
        adapter = self._adapter.currentKey() or ""
        assets: Dict[str, str] = {}
        for kind, row in self._path_rows.items():
            v = row.value()
            if v:
                assets[kind] = v

        # Joints: collect raw, then re-key under build_sku(robot_sku, joint_name).
        raw_joints = self._joints.collect()
        from registers import resolve_robot_sku, resolve_joint_sku
        sku = resolve_robot_sku(brand, model)
        joints: Dict[str, Dict[str, str]] = {}
        for jname, jspec in raw_joints.items():
            jsku = resolve_joint_sku(sku, jname)
            joints[jsku] = {"name": jname, "ir_role": jspec.get("ir_role", "")}

        try:
            if self._existing is None:
                self.saved_sku = self._svc.add_user_robot(
                    brand=brand, model=model, name=name,
                    families=families, adapter=adapter,
                    assets=assets, joints=joints,
                )
            else:
                ok = self._svc.update_user_robot(sku, {
                    "name": name,
                    "families": families,
                    "adapter": adapter,
                    "assets": {k: v for k, v in assets.items()},
                    "joints": joints,
                })
                if not ok:
                    raise RuntimeError("update_user_robot returned False")
                self.saved_sku = sku
        except Exception as exc:
            QMessageBox.critical(
                self,
                tr("robot_editor.error_title", "Save failed"),
                str(exc),
            )
            return
        self.accept()

    def _on_delete(self) -> None:
        if self._existing is None:
            return
        sku = str(self._existing.get("sku", ""))
        if not sku:
            return
        confirm = QMessageBox.question(
            self,
            tr("robot_editor.confirm_delete_title", "Delete robot"),
            tr("robot_editor.confirm_delete_body",
               "Remove this user-layer robot? Canonical entries are not affected.").format(),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if self._svc.remove_user_robot(sku):
            self.saved_sku = None
            self.accept()


__all__ = ["RobotEditorDialog"]
