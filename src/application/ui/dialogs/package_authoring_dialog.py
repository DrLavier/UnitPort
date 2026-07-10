# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Training-package authoring dialog (Method A, Slice 1d).

A modal editor for the ``training_motion`` node's ``packages`` param. It authors:
  * packages — add / remove / rename / enable, and the coarse ``package_weight``
    global-rebalance lever;
  * membership — assigns each training item to a package (writes
    ``training_items[id]['package_id']``);
  * per-package inline rewards — delegated to the shared, node-agnostic modal
    ``open_reward_function_editor`` (no bespoke reward editor);
  * a live WYSIWYG preview — the composed reward per item is derived from the SAME
    ``compose_reward_preview`` the exporter lowers, so what you see is what trains (P7).

Presence-gated: when no package is authored the preview/resolution fall back to one
implicit default package, computed live (never persisted — the ``__default__`` id is
reserved). All colors/fonts come from ``Config`` (§5). Fixed-width, self-contained.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, tr

from application.training.package_synthesis import compose_reward_preview
from application.training.training_spec import (
    DEFAULT_PACKAGE_ID,
    MotionConfig,
    resolve_effective_packages,
)

__all__ = ["PackageAuthoringDialog"]

_DEFAULT_OPTION = "— default (implicit) —"


def _parse_json_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _norm_package(pid: str, raw: Any) -> Dict[str, Any]:
    d = raw if isinstance(raw, dict) else {}
    try:
        weight = float(d.get("package_weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "package_id": str(d.get("package_id", pid)),
        "name": str(d.get("name", "")),
        "enabled": bool(d.get("enabled", True)),
        "package_weight": weight,
        "gated_by": str(d.get("gated_by", "") or ""),
        "reward_terms": dict(d.get("reward_terms") or {}),
    }


class PackageAuthoringDialog(QDialog):
    """Edit the training-package layer. Read ``result_packages`` /
    ``result_training_items`` after ``exec() == Accepted``."""

    def __init__(
        self,
        packages_value: Any,
        training_items_value: Any,
        *,
        canvas_backend: str = "sb3_mujoco",
        robot_sku: str = "",
        skill_items_value: Any = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._backend = str(canvas_backend or "sb3_mujoco")
        self._sku = str(robot_sku or "")

        raw_pkgs = _parse_json_obj(packages_value)
        self._packages: Dict[str, Dict[str, Any]] = {
            str(pid): _norm_package(str(pid), v) for pid, v in raw_pkgs.items()
        }
        self._items: Dict[str, Dict[str, Any]] = {
            str(iid): (dict(v) if isinstance(v, dict) else {"enabled": True})
            for iid, v in _parse_json_obj(training_items_value).items()
        }
        # Enabled skill (trigger) ids a package can be gated by (Slice 3) — read
        # from the sibling skill_items param so the gate is authorable here.
        raw_skills = _parse_json_obj(skill_items_value)
        self._skill_ids = [
            str(sid) for sid, v in raw_skills.items()
            if not isinstance(v, dict) or bool(v.get("enabled", True))
        ]
        self._pid_counter = 0

        # Result snapshot — overwritten on accept().
        self.result_packages: Dict[str, Dict[str, Any]] = dict(self._packages)
        self.result_training_items: Dict[str, Dict[str, Any]] = dict(self._items)

        self.setObjectName("packageAuthoringDialog")
        self.setModal(True)
        self.setWindowTitle(tr("training.packages.title", "Training Packages"))
        self.setFixedWidth(560)

        self._pkg_list: Optional[QListWidget] = None
        self._name_edit: Optional[QLineEdit] = None
        self._weight_edit: Optional[QLineEdit] = None
        self._enabled_chk: Optional[QCheckBox] = None
        self._gate_combo: Optional[QComboBox] = None
        self._member_host: Optional[QWidget] = None
        self._preview: Optional[QPlainTextEdit] = None
        self._status: Optional[QLabel] = None
        self._member_combos: Dict[str, QComboBox] = {}

        self._init_ui()
        self.setStyleSheet(self._stylesheet())
        self._reload_package_list()
        self._reload_membership()
        self._refresh_preview()

    # ------------------------------------------------------------------ UI
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        header = QLabel(self)
        header.setObjectName("pkgHeader")
        header.setText(tr("training.packages.title", "Training Packages"))
        root.addWidget(header)

        hint = QLabel(self)
        hint.setObjectName("pkgHint")
        hint.setWordWrap(True)
        hint.setText(tr(
            "training.packages.hint",
            "Group items into packages. package_weight rebalances a whole "
            "package; each package can own an inline reward set. Empty = one "
            "implicit default package.",
        ))
        root.addWidget(hint)

        root.addWidget(self._build_package_section())
        root.addWidget(self._divider())
        root.addWidget(self._build_membership_section())
        root.addWidget(self._divider())
        root.addWidget(self._build_preview_section())

        self._status = QLabel(self)
        self._status.setObjectName("pkgStatus")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        root.addLayout(self._build_buttons())

    def _divider(self) -> QFrame:
        line = QFrame(self)
        line.setObjectName("pkgDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _build_package_section(self) -> QWidget:
        box = QWidget(self)
        lay = QGridLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        self._pkg_list = QListWidget(box)
        self._pkg_list.setObjectName("pkgList")
        self._pkg_list.setFixedHeight(104)
        self._pkg_list.currentItemChanged.connect(lambda *_: self._on_package_selected())
        lay.addWidget(self._pkg_list, 0, 0, 5, 1)

        add_btn = QPushButton(tr("training.packages.add", "Add package"), box)
        add_btn.setObjectName("pkgBtn")
        add_btn.clicked.connect(self._add_package)
        lay.addWidget(add_btn, 0, 1)

        rm_btn = QPushButton(tr("training.packages.remove", "Remove"), box)
        rm_btn.setObjectName("pkgBtn")
        rm_btn.clicked.connect(self._remove_package)
        lay.addWidget(rm_btn, 1, 1)

        # Detail form for the selected package.
        form = QWidget(box)
        fl = QGridLayout(form)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setHorizontalSpacing(8)
        fl.setVerticalSpacing(6)

        fl.addWidget(self._field_label(tr("training.packages.name", "Name")), 0, 0)
        self._name_edit = QLineEdit(form)
        self._name_edit.setObjectName("pkgInput")
        self._name_edit.textEdited.connect(self._on_name_changed)
        fl.addWidget(self._name_edit, 0, 1)

        fl.addWidget(self._field_label(tr("training.packages.weight", "package_weight")), 1, 0)
        self._weight_edit = QLineEdit(form)
        self._weight_edit.setObjectName("pkgInput")
        self._weight_edit.editingFinished.connect(self._on_weight_changed)
        fl.addWidget(self._weight_edit, 1, 1)

        self._enabled_chk = QCheckBox(tr("training.packages.enabled", "Enabled"), form)
        self._enabled_chk.setObjectName("pkgCheck")
        self._enabled_chk.toggled.connect(self._on_enabled_changed)
        fl.addWidget(self._enabled_chk, 2, 1)

        fl.addWidget(self._field_label(tr("training.packages.gated_by", "Gated by skill")), 3, 0)
        self._gate_combo = QComboBox(form)
        self._gate_combo.setObjectName("pkgCombo")
        self._gate_combo.setToolTip(tr(
            "training.packages.gated_by.tip",
            "Restrict this package's rewards to a skill's trigger window — the "
            "rewards only count while that skill is firing (post-pulse decay). "
            "Requires a skill authored in this node's Skill Channels.",
        ))
        self._gate_combo.addItem(tr("training.packages.gate_none", "— none —"), "")
        for sid in self._skill_ids:
            self._gate_combo.addItem(sid, sid)
        self._gate_combo.currentIndexChanged.connect(self._on_gate_changed)
        fl.addWidget(self._gate_combo, 3, 1)

        rew_btn = QPushButton(tr("training.packages.edit_rewards", "Edit rewards…"), form)
        rew_btn.setObjectName("pkgBtn")
        rew_btn.clicked.connect(self._edit_rewards)
        fl.addWidget(rew_btn, 4, 1)

        lay.addWidget(form, 0, 2, 5, 1)
        lay.setColumnStretch(0, 3)
        lay.setColumnStretch(2, 4)
        return box

    def _build_membership_section(self) -> QWidget:
        box = QWidget(self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(tr("training.packages.membership", "Item membership"), box)
        cap.setObjectName("pkgCaption")
        lay.addWidget(cap)

        scroll = QScrollArea(box)
        scroll.setObjectName("pkgScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(120)
        self._member_host = QWidget()
        self._member_host.setObjectName("pkgMemberHost")
        scroll.setWidget(self._member_host)
        lay.addWidget(scroll)
        return box

    def _build_preview_section(self) -> QWidget:
        box = QWidget(self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(tr("training.packages.preview", "Composed reward preview (= export)"), box)
        cap.setObjectName("pkgCaption")
        lay.addWidget(cap)
        self._preview = QPlainTextEdit(box)
        self._preview.setObjectName("pkgPreview")
        self._preview.setReadOnly(True)
        self._preview.setFixedHeight(110)
        lay.addWidget(self._preview)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        ok = QPushButton(tr("common.ok", "OK"), self)
        ok.setObjectName("pkgBtnPrimary")
        ok.clicked.connect(self._on_accept)
        cancel = QPushButton(tr("common.cancel", "Cancel"), self)
        cancel.setObjectName("pkgBtn")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        return row

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setObjectName("pkgFieldLabel")
        return lbl

    # ------------------------------------------------------- selection state
    def _current_pid(self) -> str:
        if self._pkg_list is None:
            return ""
        it = self._pkg_list.currentItem()
        return str(it.data(Qt.ItemDataRole.UserRole)) if it is not None else ""

    def _reload_package_list(self) -> None:
        if self._pkg_list is None:
            return
        self._pkg_list.blockSignals(True)
        self._pkg_list.clear()
        for pid, pkg in self._packages.items():
            label = pkg.get("name") or pid
            item = QListWidgetItem(str(label))
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self._pkg_list.addItem(item)
        self._pkg_list.blockSignals(False)
        if self._pkg_list.count():
            self._pkg_list.setCurrentRow(0)
        self._on_package_selected()

    def _on_package_selected(self) -> None:
        pid = self._current_pid()
        pkg = self._packages.get(pid)
        has = pkg is not None
        for w in (self._name_edit, self._weight_edit, self._enabled_chk, self._gate_combo):
            if w is not None:
                w.setEnabled(has)
        if not has:
            if self._name_edit:
                self._name_edit.setText("")
            if self._weight_edit:
                self._weight_edit.setText("")
            if self._enabled_chk:
                self._enabled_chk.setChecked(False)
            return
        if self._name_edit:
            self._name_edit.setText(str(pkg.get("name", "")))
        if self._weight_edit:
            self._weight_edit.setText(f"{float(pkg.get('package_weight', 1.0)):g}")
        if self._enabled_chk:
            self._enabled_chk.blockSignals(True)
            self._enabled_chk.setChecked(bool(pkg.get("enabled", True)))
            self._enabled_chk.blockSignals(False)
        if self._gate_combo:
            self._gate_combo.blockSignals(True)
            gate = str(pkg.get("gated_by", "") or "")
            idx = self._gate_combo.findData(gate)
            if idx < 0:
                # Stale gate (skill removed/disabled): show it — never drop a
                # stored value silently (§8 / storage must stay UI-visible).
                self._gate_combo.addItem(
                    tr("training.packages.gate_missing", "{s} (missing)").replace("{s}", gate),
                    gate,
                )
                idx = self._gate_combo.findData(gate)
            self._gate_combo.setCurrentIndex(max(0, idx))
            self._gate_combo.blockSignals(False)

    # ------------------------------------------------------- mutations
    def _unique_pid(self) -> str:
        while True:
            self._pid_counter += 1
            pid = f"pkg_{self._pid_counter}"
            if pid != DEFAULT_PACKAGE_ID and pid not in self._packages:
                return pid

    def _add_package(self) -> None:
        pid = self._unique_pid()
        n = len(self._packages) + 1
        self._packages[pid] = {
            "package_id": pid,
            "name": tr("training.packages.new_name", "Package {n}").replace("{n}", str(n)),
            "enabled": True,
            "package_weight": 1.0,
            "gated_by": "",
            "reward_terms": {},
        }
        self._reload_package_list()
        self._select_pid(pid)
        self._reload_membership()
        self._refresh_preview()

    def _remove_package(self) -> None:
        pid = self._current_pid()
        if not pid or pid not in self._packages:
            return
        del self._packages[pid]
        # Orphaned members fall back to the implicit default (membership cleared).
        for cfg in self._items.values():
            if isinstance(cfg, dict) and str(cfg.get("package_id", "")) == pid:
                cfg.pop("package_id", None)
        self._reload_package_list()
        self._reload_membership()
        self._refresh_preview()

    def _select_pid(self, pid: str) -> None:
        if self._pkg_list is None:
            return
        for row in range(self._pkg_list.count()):
            if str(self._pkg_list.item(row).data(Qt.ItemDataRole.UserRole)) == pid:
                self._pkg_list.setCurrentRow(row)
                return

    def _on_name_changed(self, text: str) -> None:
        pid = self._current_pid()
        if pid in self._packages:
            self._packages[pid]["name"] = str(text)
            it = self._pkg_list.currentItem() if self._pkg_list else None
            if it is not None:
                it.setText(str(text) or pid)
            self._rebuild_member_combo_labels()

    def _on_weight_changed(self) -> None:
        pid = self._current_pid()
        if pid not in self._packages or self._weight_edit is None:
            return
        raw = self._weight_edit.text().strip()
        try:
            self._packages[pid]["package_weight"] = float(raw)
            self._set_status("", error=False)
        except ValueError:
            self._weight_edit.setText(f"{float(self._packages[pid].get('package_weight', 1.0)):g}")
            self._set_status(tr("training.packages.bad_weight", "package_weight must be a number"), error=True)
            return
        self._refresh_preview()

    def _on_enabled_changed(self, on: bool) -> None:
        pid = self._current_pid()
        if pid in self._packages:
            self._packages[pid]["enabled"] = bool(on)
            self._refresh_preview()

    def _on_gate_changed(self, _idx: int) -> None:
        pid = self._current_pid()
        if pid in self._packages and self._gate_combo is not None:
            self._packages[pid]["gated_by"] = str(self._gate_combo.currentData() or "")
            self._refresh_preview()

    def _edit_rewards(self) -> None:
        pid = self._current_pid()
        if pid not in self._packages:
            return
        from application.compiler.term_payload import (
            PAGE_GLOBAL,
            migrate_reward_terms_to_paged,
        )
        from application.ui.dialogs import open_reward_function_editor

        pkg = self._packages[pid]
        paged = migrate_reward_terms_to_paged(pkg.get("reward_terms") or {})
        page = dict(paged.get(PAGE_GLOBAL, {}))
        result = open_reward_function_editor(
            self,
            page,
            registry_id=("il_rewards" if self._backend == "isaac_lab" else "rewards"),
            kind="reward",
            canvas_backend=self._backend,
            robot_sku=self._sku or None,
        )
        if result is not None:
            paged[PAGE_GLOBAL] = result
            pkg["reward_terms"] = paged
            self._refresh_preview()

    # ------------------------------------------------------- membership UI
    def _package_options(self):
        return [("", _DEFAULT_OPTION)] + [
            (pid, (pkg.get("name") or pid)) for pid, pkg in self._packages.items()
        ]

    def _reload_membership(self) -> None:
        if self._member_host is None:
            return
        old = self._member_host.layout()
        if old is not None:
            while old.count():
                w = old.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            QWidget().setLayout(old)  # detach the old layout
        self._member_combos = {}
        grid = QGridLayout(self._member_host)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        options = self._package_options()
        row = 0
        for item_id, cfg in self._items.items():
            if isinstance(cfg, dict) and not bool(cfg.get("enabled", True)):
                continue
            lbl = QLabel(str(item_id), self._member_host)
            lbl.setObjectName("pkgMemberLabel")
            combo = QComboBox(self._member_host)
            combo.setObjectName("pkgCombo")
            current = str(cfg.get("package_id", "")) if isinstance(cfg, dict) else ""
            sel = 0
            for idx, (pid, label) in enumerate(options):
                combo.addItem(str(label), pid)
                if pid == current:
                    sel = idx
            combo.setCurrentIndex(sel)
            combo.currentIndexChanged.connect(
                lambda _idx, iid=item_id, c=combo: self._on_member_changed(iid, c)
            )
            grid.addWidget(lbl, row, 0)
            grid.addWidget(combo, row, 1)
            grid.setColumnStretch(1, 1)
            self._member_combos[item_id] = combo
            row += 1

    def _rebuild_member_combo_labels(self) -> None:
        options = self._package_options()
        for combo in self._member_combos.values():
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            sel = 0
            for idx, (pid, label) in enumerate(options):
                combo.addItem(str(label), pid)
                if pid == current:
                    sel = idx
            combo.setCurrentIndex(sel)
            combo.blockSignals(False)

    def _on_member_changed(self, item_id: str, combo: QComboBox) -> None:
        pid = str(combo.currentData() or "")
        cfg = self._items.get(item_id)
        if not isinstance(cfg, dict):
            cfg = {"enabled": True}
            self._items[item_id] = cfg
        if pid:
            cfg["package_id"] = pid
        else:
            cfg.pop("package_id", None)
        self._refresh_preview()

    # ------------------------------------------------------- preview + accept
    def _refresh_preview(self) -> None:
        if self._preview is None:
            return
        try:
            preview = compose_reward_preview(self._items, self._packages)
        except ValueError as exc:
            self._preview.setPlainText("")
            self._set_status(str(exc), error=True)
            return
        self._set_status("", error=False)
        if not preview:
            self._preview.setPlainText(
                tr("training.packages.preview_empty",
                   "No inline package rewards. Items with no package reward keep "
                   "their wired rewards node.")
            )
            return
        lines = []
        for item_id, terms in preview.items():
            pid = str((self._items.get(item_id) or {}).get("package_id", "") or DEFAULT_PACKAGE_ID)
            pkg = self._packages.get(pid, {})
            pname = pkg.get("name") or pid
            pw = float(pkg.get("package_weight", 1.0)) if pkg else 1.0
            lines.append(f"{item_id}  ← {pname} (×{pw:g})")
            for func, payload in terms.items():
                w = payload.get("weight") if isinstance(payload, dict) else payload
                lines.append(f"    {func}: {w}")
        self._preview.setPlainText("\n".join(lines))

    def _collect_result(self) -> None:
        self.result_packages = {pid: dict(p) for pid, p in self._packages.items()}
        self.result_training_items = {iid: dict(c) if isinstance(c, dict) else c
                                      for iid, c in self._items.items()}

    def _on_accept(self) -> None:
        # Validate against the SAME resolver the backend uses (fail-loud membership).
        try:
            from application.training.package_synthesis import packages_from_param
            motion = MotionConfig(
                training_items=self._items,
                packages=packages_from_param(self._packages),
            )
            resolve_effective_packages(motion)
            compose_reward_preview(self._items, self._packages)  # weight/reward fail-louds
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._collect_result()
        self.accept()

    def _set_status(self, text: str, *, error: bool) -> None:
        if self._status is None:
            return
        self._status.setText(text or "")
        slot = "danger_zone" if error else "sub_t2"
        size = Config.get_font_size("size_small", 12)
        self._status.setStyleSheet(
            f"color: {Config.get_color(slot, '#C0504D' if error else '#8A8778')}; "
            f"font-size: {size}px; background: transparent;"
        )

    # ------------------------------------------------------- theme (§5)
    def _stylesheet(self) -> str:
        bg = Config.get_color("bg_1", "#15171C")
        bg2 = Config.get_color("bg_2", "#1A1C22")
        text = Config.get_color("main_t2", "#FFFFFF")
        t1 = Config.get_color("main_t1", "#D6D3C7")
        muted = Config.get_color("sub_t2", "#8A8778")
        accent = Config.get_color("highlight", "#00897B")
        border = Config.get_color("border_1", "#2A2D35")
        size_normal = Config.get_font_size("size_normal", 14)
        size_small = Config.get_font_size("size_small", 12)
        return f"""
            QDialog#packageAuthoringDialog {{ background: {bg}; }}
            QLabel#pkgHeader {{ color: {accent}; font-size: {size_normal}px; font-weight: 600; }}
            QLabel#pkgHint, QLabel#pkgCaption {{ color: {muted}; font-size: {size_small}px; }}
            QLabel#pkgFieldLabel, QLabel#pkgMemberLabel {{ color: {t1}; font-size: {size_small}px; }}
            QFrame#pkgDivider {{ color: {border}; background: {border}; max-height: 1px; }}
            QListWidget#pkgList, QScrollArea#pkgScroll, QPlainTextEdit#pkgPreview,
            QWidget#pkgMemberHost {{
                background: {bg2}; color: {t1}; border: 1px solid {border};
                font-size: {size_small}px;
            }}
            QLineEdit#pkgInput, QComboBox#pkgCombo {{
                background: {bg2}; color: {text}; border: 1px solid {border};
                padding: 2px 4px; font-size: {size_small}px;
            }}
            QCheckBox#pkgCheck {{ color: {t1}; font-size: {size_small}px; }}
            QPushButton#pkgBtn, QPushButton#pkgBtnPrimary {{
                background: {bg2}; color: {text}; border: 1px solid {border};
                padding: 4px 10px; font-size: {size_small}px;
            }}
            QPushButton#pkgBtnPrimary {{ background: {accent}; border: 1px solid {accent}; }}
            QLabel#pkgStatus {{ font-size: {size_small}px; }}
        """
