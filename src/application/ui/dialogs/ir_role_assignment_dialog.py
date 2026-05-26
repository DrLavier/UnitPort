# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IR-Role Assignment Dialog — boot-time bulk mapping fix-up.

Opened from :meth:`UnitPortMain._maybe_open_ir_assignment_dialog` after
``finish_loading()`` whenever
:class:`RobotAssetSelfCheckTask` returned a non-empty pending list. Lets
the user assign IR roles to every joint/body the auto-dump tokeniser
couldn't classify, then persists the assignments to the user-overlay
registry via :meth:`RobotAssetService.update_ir_role`.

Layout (List + Table):

    ┌──────────────────┬──────────────────────────────┐
    │  Pending Robots  │  Joint Mapping — <robot>     │
    │  ─────────────   │  ──────────────────────────  │
    │  • Spot   (4)    │  Joint           Role        │
    │  • Go2    (2)    │  fl_hx     [ hip_FL    ▼]    │
    │  • H1     (3)    │  fr_hx     [ hip_FR    ▼]    │
    │                  │  ...                         │
    ├──────────────────┴──────────────────────────────┤
    │                       [Confirm All]  [Cancel]   │
    └─────────────────────────────────────────────────┘

Modal. Confirm-All becomes enabled only when every row across every
robot has a non-empty role picked — partial saves cause confusion
because the same pending list would re-open on the next boot.

Theme (CLAUDE.md §1.5): every color via ``Config.get_color`` from the
existing slots (``bg_1`` / ``bg_2`` / ``border_1`` / ``main_t1`` /
``sub_t2`` / ``row_1``). Font size is ``size_small`` everywhere
(closed-set rule).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Shared OOS sentinel — the registry validator special-cases the same
# value so user-overlay entries marked Out-of-Scope aren't reported as
# invalid ir_role. Imported (not redeclared) so the two sites can't drift.
from registers.robots import OUT_OF_SCOPE_IR_ROLE as OUT_OF_SCOPE_SENTINEL

_OOS_DISPLAY_LABEL = (
    "(Out of Scope — totally ignored at training time, "
    "use sensor_* roles for physical mount links)"
)

from unitport_sdk import (
    Config,
    log_error,
    log_info,
    log_warning,
    setButton,
    tr,
)

from application.tools.robot_asset_selfcheck import PendingAssignment


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


class IRRoleAssignmentDialog(QDialog):
    """Bulk-assign IR roles for joints/bodies left empty by auto-dump.

    Groups the flat pending list by robot SKU; left list = one entry per
    robot with a "(<n>)" count badge; right table rebuilds when the
    user selects a different robot. The current selections are held in
    ``self._selections[(sku, fmt, kind, uid)] = ir_role`` and only
    written back on Confirm.
    """

    def __init__(
        self,
        pending: List[PendingAssignment],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(
            "ir_assign.dialog_title",
            "Assign IR Roles for New Robots",
        ))
        self.setModal(True)
        # Resizable: explicitly drop the MSWindowsFixedSizeDialogHint
        # flag (some Qt builds turn it on for modal dialogs on Windows,
        # which is what makes edge-drag cursors appear but resize do
        # nothing). Set an explicit large QWIDGETSIZE_MAX upper bound
        # so the layout's default constraint can never accidentally
        # lock min == max. Show a bottom-right size grip as a visible
        # affordance on top of the standard edge hit areas.
        flags = self.windowFlags()
        flags &= ~Qt.WindowType.MSWindowsFixedSizeDialogHint
        self.setWindowFlags(flags)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(560, 380)
        self.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX
        self.resize(820, 520)

        # Group pending entries by SKU, preserving first-seen order.
        self._by_sku: Dict[str, List[PendingAssignment]] = defaultdict(list)
        self._sku_order: List[str] = []
        self._sku_meta: Dict[str, Tuple[str, str]] = {}  # sku -> (robot_name, family)
        for p in pending:
            if p.sku not in self._by_sku:
                self._sku_order.append(p.sku)
                self._sku_meta[p.sku] = (p.robot_name, p.family)
            self._by_sku[p.sku].append(p)

        # In-memory selections — seeded from each entry's `current`
        # ir_role so already-classified rows start with the tokeniser's
        # auto-match pre-selected. Empty string = still needs picking.
        # Keyed by the same (sku, fmt, kind, uid) tuple the registry uses.
        self._selections: Dict[Tuple[str, str, str, str], str] = {
            (p.sku, p.fmt, p.kind, p.uid): p.current for p in pending
        }

        # Cache canonical-role choice lists per family — get_canonical_roles
        # is cheap but called per dropdown rebuild.
        self._role_choices_by_family: Dict[str, List[str]] = {}

        self._apply_dialog_style()
        self._build_ui()
        if self._sku_order:
            self._list.setCurrentRow(0)

    # ------------------------------------------------------------------ chrome

    def _apply_dialog_style(self) -> None:
        bg = Config.get_color("bg_1")
        fg = Config.get_color("main_t1")
        sz = _ss()
        self.setStyleSheet(
            f"QDialog {{ background: {bg}; color: {fg}; font-size: {sz}px; }}"
            f"QLabel {{ color: {fg}; font-size: {sz}px; }}"
            f"QSplitter::handle {{ background: {Config.get_color('border_1')}; }}"
        )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        # Default constraint = layout asks for the widget's hints but
        # NEVER forces them onto the window — i.e. resize is free both
        # directions. SetMinimumSize would lock the window to >= layout
        # minimum, SetFixedSize would lock min == max == hint. We
        # explicitly pin it so a future addLayout/addWidget call can't
        # accidentally tighten the constraint.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Header — small banner so the user understands why the dialog opened.
        banner = QLabel(tr(
            "ir_assign.banner",
            "Some robot joints/bodies could not be auto-classified. "
            "Assign their IR roles below to finish setup.",
        ))
        banner.setWordWrap(True)
        banner.setStyleSheet(f"color: {Config.get_color('sub_t2')};")
        outer.addWidget(banner)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # Left — pending-robots list
        left_wrap = QWidget(self)
        left_l = QVBoxLayout(left_wrap)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(4)
        left_lbl = QLabel(tr("ir_assign.pending_robots", "Pending Robots"))
        left_lbl.setStyleSheet(f"color: {Config.get_color('sub_t2')};")
        left_l.addWidget(left_lbl)
        self._list = QListWidget(left_wrap)
        self._list.setStyleSheet(
            f"QListWidget {{ background: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"font-size: {_ss()}px; }}"
            f"QListWidget::item:selected {{ background: {Config.get_color('row_1')}; }}"
        )
        for sku in self._sku_order:
            self._add_or_refresh_list_item(sku)
        self._list.currentRowChanged.connect(self._on_list_changed)
        left_l.addWidget(self._list, 1)
        splitter.addWidget(left_wrap)

        # Right — per-robot mapping table
        right_wrap = QWidget(self)
        right_l = QVBoxLayout(right_wrap)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)
        self._right_title = QLabel("")
        self._right_title.setStyleSheet(f"color: {Config.get_color('sub_t2')};")
        right_l.addWidget(self._right_title)
        self._table = QTableWidget(0, 3, right_wrap)
        self._table.setHorizontalHeaderLabels([
            tr("ir_assign.col_kind", "Type"),
            tr("ir_assign.col_joint", "Joint / Body"),
            tr("ir_assign.col_role", "IR Role"),
        ])
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"gridline-color: {Config.get_color('border_1')}; "
            f"font-size: {_ss()}px; }}"
            f"QHeaderView::section {{ background: {Config.get_color('row_1')}; "
            f"color: {Config.get_color('main_t1')}; border: 0; "
            f"padding: 3px 6px; font-size: {_ss()}px; }}"
        )
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        right_l.addWidget(self._table, 1)
        splitter.addWidget(right_wrap)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([220, 600])

        # Footer — buttons. Confirm on the LEFT (primary action, leads
        # the eye), Cancel on the RIGHT (secondary/escape). Confirm uses
        # border+save spec so it reads as an affirmative-but-not-loud
        # commit (matches sidebar Dump-style buttons; no green fill).
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self._btn_confirm = setButton(
            "ir_assign.confirm", 140, 28, kind="border", spec="save",
            default=tr("ir_assign.confirm", "Confirm All"),
        )
        self._btn_confirm.clicked.connect(self._on_confirm)
        footer.addWidget(self._btn_confirm)
        footer.addSpacing(8)
        self._btn_cancel = setButton(
            "ir_assign.cancel", 100, 28, kind="border", spec="none",
            default=tr("ir_assign.cancel", "Cancel"),
        )
        self._btn_cancel.clicked.connect(self.reject)
        footer.addWidget(self._btn_cancel)
        footer.addStretch(1)
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(f"color: {Config.get_color('sub_t2')};")
        footer.addWidget(self._lbl_status)
        outer.addLayout(footer)

        self._update_status()

    # ------------------------------------------------------------------ list

    def _sku_status(self, sku: str) -> Tuple[int, int]:
        """Return (assigned_count, total_count) for one sku, using the
        live selections (not the original pending list)."""
        total = 0
        assigned = 0
        for p in self._by_sku.get(sku, []):
            total += 1
            if self._selections.get((p.sku, p.fmt, p.kind, p.uid), ""):
                assigned += 1
        return assigned, total

    def _add_or_refresh_list_item(self, sku: str) -> None:
        """Append a new row (called during build) OR refresh the existing
        row's label + color (called after every dropdown change)."""
        robot_name, _ = self._sku_meta[sku]
        assigned, total = self._sku_status(sku)
        text = f"{robot_name}  ({assigned}/{total})"
        # Find existing row for this sku, if any.
        row_idx = -1
        for r in range(self._list.count()):
            it = self._list.item(r)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == sku:
                row_idx = r
                break
        if row_idx < 0:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, sku)
            self._list.addItem(item)
        else:
            item = self._list.item(row_idx)
            item.setText(text)
        # Dye: fully-assigned → safe_zone (green); any pending → danger_zone (red).
        from PyQt6.QtGui import QColor
        if assigned >= total:
            item.setForeground(QColor(Config.get_color("safe_zone")))
        else:
            item.setForeground(QColor(Config.get_color("danger_zone")))

    # ------------------------------------------------------------------ data

    def _role_choices_for(self, family: str) -> List[str]:
        cached = self._role_choices_by_family.get(family)
        if cached is not None:
            return cached
        try:
            from application.training.body_ir import (
                get_canonical_roles,
                get_joint_ir_roles,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[ir_assign] cannot load role catalog for {family!r}: {exc}")
            self._role_choices_by_family[family] = []
            return []
        # Joints filter to actuated categories only; bodies show every
        # canonical role (base / feet / head / hands plus joints). The
        # dialog dropdown union of "any role usable for this family"
        # is the simplest correct choice — picking the wrong category
        # is caught by the validator next launch.
        all_roles = [r.role_id for r in get_canonical_roles(family)]
        # De-duplicate while preserving order.
        seen: set = set()
        out: List[str] = []
        for rid in all_roles:
            if rid and rid not in seen:
                out.append(rid)
                seen.add(rid)
        self._role_choices_by_family[family] = out
        return out

    def _on_list_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sku_order):
            self._table.setRowCount(0)
            self._right_title.setText("")
            return
        sku = self._sku_order[row]
        robot_name, family = self._sku_meta[sku]
        n = len(self._by_sku[sku])
        self._right_title.setText(tr(
            "ir_assign.right_title",
            "Joint Mapping — {robot} ({family}, {n} entr{plural})",
        ).format(
            robot=robot_name, family=family, n=n,
            plural="ies" if n != 1 else "y",
        ))

        role_choices = self._role_choices_for(family)
        self._table.setRowCount(0)
        for p in self._by_sku[sku]:
            r = self._table.rowCount()
            self._table.insertRow(r)
            kind_label = (
                tr("ir_assign.kind_joint", "Joint")
                if p.kind == "joint"
                else tr("ir_assign.kind_body", "Body")
            )
            kind_item = QTableWidgetItem(f"{kind_label}  [{p.fmt}]")
            kind_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(r, 0, kind_item)
            name_item = QTableWidgetItem(p.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setToolTip(f"uid={p.uid}")
            self._table.setItem(r, 1, name_item)
            combo = self._make_role_combo(role_choices)
            current = self._selections.get((p.sku, p.fmt, p.kind, p.uid), "")
            if current:
                # findData matches userData (the persisted ir_role
                # string), not the display text — handles both regular
                # role_ids AND the OOS sentinel uniformly.
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            key = (p.sku, p.fmt, p.kind, p.uid)
            combo.currentIndexChanged.connect(
                lambda _i, c=combo, k=key: self._on_combo_changed(
                    k, self._combo_value(c)
                )
            )
            self._table.setCellWidget(r, 2, combo)
            self._dye_row(r, current)

    def _make_role_combo(self, role_choices: List[str]) -> QComboBox:
        combo = QComboBox()
        # Row 0 — empty sentinel, the "still needs picking" state.
        combo.addItem("", userData="")
        # Row 1 — Out-of-Scope shortcut. ``setItemData`` stashes the
        # sentinel string so :meth:`_combo_value` can read the actual
        # ir_role value to persist regardless of what the user-facing
        # display label says.
        combo.addItem(
            tr("ir_assign.role_out_of_scope", _OOS_DISPLAY_LABEL),
            userData=OUT_OF_SCOPE_SENTINEL,
        )
        for rid in role_choices:
            combo.addItem(rid, userData=rid)
        bg = Config.get_color("bg_2")
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        combo.setStyleSheet(
            f"QComboBox {{ background: {bg}; color: {fg}; "
            f"border: 1px solid {border}; padding: 2px 6px; "
            f"font-size: {_ss()}px; }}"
            f"QComboBox QAbstractItemView {{ background: {bg}; color: {fg}; "
            f"selection-background-color: {Config.get_color('row_1')}; }}"
        )
        return combo

    @staticmethod
    def _combo_value(combo: QComboBox) -> str:
        """Return the persistable ir_role value for the combo's current
        selection — either the canonical role_id, ``OUT_OF_SCOPE_SENTINEL``,
        or ``""`` for the empty pick. Always prefer ``currentData`` over
        ``currentText`` so the display label (which may be translated)
        never leaks into the registry.
        """
        data = combo.currentData()
        if isinstance(data, str):
            return data
        return ""

    def _dye_row(self, row: int, ir_role: str) -> None:
        """Color row 0/1 cell text by assignment state.

        - Empty ir_role            → ``danger_zone`` (red, needs action)
        - ``OUT_OF_SCOPE_SENTINEL`` → ``sub_t2`` (muted grey, parked)
        - Regular role_id          → ``safe_zone`` (green, classified)

        Combo cell isn't dyed — the dropdown stylesheet already gives a
        clear "I'm interactive" visual signal; redoing its color would
        fight Qt's combo-popup styling on hover/focus.
        """
        from PyQt6.QtGui import QColor
        if not ir_role:
            slot = "danger_zone"
        elif ir_role == OUT_OF_SCOPE_SENTINEL:
            slot = "sub_t2"
        else:
            slot = "safe_zone"
        color = QColor(Config.get_color(slot))
        for col in (0, 1):
            it = self._table.item(row, col)
            if it is not None:
                it.setForeground(color)

    def _on_combo_changed(self, key: Tuple[str, str, str, str], text: str) -> None:
        self._selections[key] = (text or "").strip()
        # Find which row this combo lives in and recolor it; also refresh
        # the left-list count + color for the affected sku.
        sku = key[0]
        if sku in self._sku_meta:
            for r in range(self._table.rowCount()):
                w = self._table.cellWidget(r, 2)
                if w is not None and w.signalsBlocked() is False:
                    # Match by (sku, fmt, kind, uid) — we don't have
                    # row→key mapping stored, but the table is filtered
                    # to one sku at a time, so we walk by_sku order.
                    pass
            # Walk pending in the same order rows were inserted (=order
            # of self._by_sku[sku]) so the row index matches.
            for idx, p in enumerate(self._by_sku.get(sku, [])):
                if (p.sku, p.fmt, p.kind, p.uid) == key:
                    self._dye_row(idx, self._selections[key])
                    break
            self._add_or_refresh_list_item(sku)
        self._update_status()

    def _all_filled(self) -> bool:
        return all(bool(v) for v in self._selections.values())

    def _update_status(self) -> None:
        total = len(self._selections)
        filled = sum(1 for v in self._selections.values() if v)
        if filled == total:
            msg = tr(
                "ir_assign.status_ready",
                "All {n} entries assigned — ready to confirm.",
            ).format(n=total)
        else:
            msg = tr(
                "ir_assign.status_pending",
                "{filled}/{total} assigned",
            ).format(filled=filled, total=total)
        self._lbl_status.setText(msg)
        self._btn_confirm.setEnabled(self._all_filled())

    # ------------------------------------------------------------------ confirm

    def _on_confirm(self) -> None:
        if not self._all_filled():
            return
        try:
            from application.service.robot_assets import (
                get_robot_asset_service,
            )
        except Exception as exc:  # noqa: BLE001
            log_error(f"[ir_assign] cannot import service: {exc}")
            return
        svc = get_robot_asset_service()
        # Build a flat patch list and ship it in ONE call — bulk API
        # groups by sku, persists once per sku, and reloads+validates
        # ONCE at the very end. The per-entry update_ir_role loop we
        # had before issued a reload+validate per call, spraying
        # warning lines and re-firing the changed signal N times.
        patches = [
            (sku, fmt, kind, uid, ir_role)
            for (sku, fmt, kind, uid), ir_role in self._selections.items()
        ]
        try:
            n_ok = svc.bulk_update_ir_roles(patches)
        except Exception as exc:  # noqa: BLE001
            log_error(
                f"[ir_assign] bulk_update_ir_roles crashed: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        n_fail = len(patches) - n_ok
        n_skus = len({p[0] for p in patches})
        log_info(
            f"[ir_assign] confirm: {n_ok} assignments saved, "
            f"{n_fail} failed, across {n_skus} robot(s)"
        )
        self.accept()


__all__ = ["IRRoleAssignmentDialog"]
