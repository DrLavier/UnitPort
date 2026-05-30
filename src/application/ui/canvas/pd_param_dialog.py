# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""PDParamTableDialog — per-joint PD / actuator table for the RobotNode.

Opened from ``PDParamTableRow._handle_click`` (widget=``pd_param_table`` on the
RobotNode's ``pd_groups`` param). Shows the **actual submitted** per-joint gains,
computed live via :func:`application.physics.pd_preview.compute_pd_preview`
(loads the MJCF, mass-weights ``kp = m_eff·omega_n²``).

Editable knobs (CLAUDE.md §10 — PD is parameterized by (omega_n, zeta), never
raw kp/kd):
    * per PD group: ``omega_n`` (rad/s) + ``zeta`` (dimensionless)
    * scalar: ``effort_limit`` (N·m) + ``velocity_limit`` (rad/s)

``kp`` / ``kd`` are **read-only derived** — recomputed live when a group's
(omega_n, zeta) changes. Remotized joints (e.g. Spot knees) show their
angle-dependent torque-table PEAK in the Effort column, not the scalar cap.

On accept:
    * :attr:`result_pd_groups` — ``{group_id: {"omega_n", "zeta"}}`` overrides
    * :attr:`result_effort_limit` / :attr:`result_velocity_limit`

All colors / fonts go through ``Config.get_color`` / ``Config.get_font_size``
(§5); UI strings through ``tr`` / ``I18nLabel``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, I18nLabel, LaviButton, tr

from application.physics.pd_preview import PDPreview


# Per-joint table columns.
_COLS = ["IR role", "Group", "ωn", "ζ", "kp", "kd", "Effort", "Vel"]
_COL_IR, _COL_GROUP, _COL_WN, _COL_ZETA, _COL_KP, _COL_KD, _COL_EFF, _COL_VEL = range(8)


class PDParamTableDialog(QDialog):
    """Modal per-joint PD / actuator editor for a bound RobotNode SKU."""

    def __init__(
        self,
        sku: str,
        sku_display_name: str,
        preview: PDPreview,
        initial_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sku = str(sku)
        self._preview = preview
        # Working override dict — start from the canvas pd_groups; touching a
        # group's (omega_n, zeta) records it here.
        self._overrides: Dict[str, Dict[str, float]] = copy.deepcopy(
            dict(initial_overrides or {})
        )

        # Results (read after exec() == Accepted).
        self.result_pd_groups: Dict[str, Dict[str, float]] = dict(self._overrides)
        self.result_effort_limit: float = float(preview.effort_limit)
        self.result_velocity_limit: float = float(preview.velocity_limit)

        # Per-group editor spinboxes, indexed by group_id.
        self._wn_spins: Dict[str, QDoubleSpinBox] = {}
        self._zeta_spins: Dict[str, QDoubleSpinBox] = {}
        self._effort_spin: Optional[QDoubleSpinBox] = None
        self._vel_spin: Optional[QDoubleSpinBox] = None
        self._table: Optional[QTableWidget] = None

        title = tr("canvas.robot.pd_table.dialog_title", default="PD / Actuator")
        if sku_display_name:
            title = f"{title} — {sku_display_name}"
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(720)
        self.setMinimumHeight(440)

        self._build_layout()
        self._refresh_table()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        header = I18nLabel(
            "canvas.robot.pd_table.header",
            default="Per-joint PD gains (kp = m_eff·ωn²) — derived live from the "
                    "MJCF mass matrix. Edit (ωn, ζ) per group; kp/kd are read-only.",
            parent=self,
        )
        header.setObjectName("pdTableHeader")
        header.setWordWrap(True)
        outer.addWidget(header, 0)

        # ---- AUTO preset button (border + safe/green LaviButton) ----
        # Sits alongside an inline "Re-dump USD" button that appears only when
        # AUTO reports ``status == "need_dump_usd"`` — i.e. the robot has a
        # USD asset declared but no ``physics_per_format.USD`` cache yet (the
        # branch that would feed AUTO's USD-drive reverse-derive layer). The
        # button calls ``RobotAssetService.dump_and_persist(sku, 'USD')`` then
        # re-runs AUTO. Rule §8: no silent fall-through to family when richer
        # data is fetchable — surface the gap and let the user fix it.
        auto_row = QHBoxLayout()
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.addStretch(1)
        self._redump_btn = LaviButton(
            "canvas.robot.pd_table.redump_button",
            default="Re-dump USD",
            kind="border",
            spec="notice",
            parent=self,
        )
        self._redump_btn.setToolTip(tr(
            "canvas.robot.pd_table.redump_tooltip",
            default="Dump the USD asset to refresh the cached drive params, "
                    "then re-fill AUTO from the dumped values.",
        ))
        self._redump_btn.clicked.connect(self._on_redump_clicked)
        self._redump_btn.setVisible(False)
        auto_row.addWidget(self._redump_btn, 0)

        self._auto_btn = LaviButton(
            "canvas.robot.pd_table.auto_button",
            default="AUTO ⤓",
            kind="border",
            spec="save",
            parent=self,
        )
        self._auto_btn.setToolTip(tr(
            "canvas.robot.pd_table.auto_tooltip",
            default="Fill the official recommended (ωn, ζ) + effort/velocity preset. "
                    "Resolution order: brand manifest → USD drive params (cached "
                    "from Dump USD) → family defaults.",
        ))
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        auto_row.addWidget(self._auto_btn, 0)
        outer.addLayout(auto_row, 0)

        # Status line for AUTO results (dynamic; empty until clicked). The
        # three-state UI: ✓ ok / ⚠ need_dump_usd / ⓘ no_usd_family_fallback.
        self._status_lbl = QLabel("", self)
        self._status_lbl.setObjectName("pdTableStatus")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setVisible(False)
        outer.addWidget(self._status_lbl, 0)

        # ---- Group editors (omega_n / zeta per PD group) ----
        groups_box = QWidget(self)
        grid = QGridLayout(groups_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.addWidget(self._mk_caption("Group"), 0, 0)
        grid.addWidget(self._mk_caption("ωn (rad/s)"), 0, 1)
        grid.addWidget(self._mk_caption("ζ"), 0, 2)
        for r, g in enumerate(self._preview.groups, start=1):
            lbl = QLabel(g.group_id, groups_box)
            lbl.setObjectName("pdTableGroupLabel")
            grid.addWidget(lbl, r, 0)

            wn = QDoubleSpinBox(groups_box)
            wn.setDecimals(2)
            wn.setRange(0.5, 200.0)   # PDGroup OMEGA_N_MIN/MAX
            wn.setSingleStep(1.0)
            wn.setValue(float(g.omega_n))
            wn.setKeyboardTracking(False)
            wn.setFixedWidth(110)
            wn.editingFinished.connect(
                lambda gid=g.group_id: self._on_group_changed(gid)
            )
            grid.addWidget(wn, r, 1)
            self._wn_spins[g.group_id] = wn

            zeta = QDoubleSpinBox(groups_box)
            zeta.setDecimals(3)
            zeta.setRange(0.1, 2.0)   # PDGroup ZETA_MIN/MAX
            zeta.setSingleStep(0.05)
            zeta.setValue(float(g.zeta))
            zeta.setKeyboardTracking(False)
            zeta.setFixedWidth(110)
            zeta.editingFinished.connect(
                lambda gid=g.group_id: self._on_group_changed(gid)
            )
            grid.addWidget(zeta, r, 2)
            self._zeta_spins[g.group_id] = zeta

        # Scalar caps to the right of the group grid.
        grid.addWidget(self._mk_caption("effort_limit (N·m)"), 0, 4)
        self._effort_spin = QDoubleSpinBox(groups_box)
        self._effort_spin.setDecimals(2)
        self._effort_spin.setRange(0.0, 100000.0)
        self._effort_spin.setSingleStep(1.0)
        self._effort_spin.setValue(float(self._preview.effort_limit))
        self._effort_spin.setKeyboardTracking(False)
        self._effort_spin.setFixedWidth(120)
        self._effort_spin.editingFinished.connect(self._on_scalar_changed)
        grid.addWidget(self._effort_spin, 1, 4)

        grid.addWidget(self._mk_caption("velocity_limit (rad/s)"), 2, 4)
        self._vel_spin = QDoubleSpinBox(groups_box)
        self._vel_spin.setDecimals(2)
        self._vel_spin.setRange(0.0, 100000.0)
        self._vel_spin.setSingleStep(1.0)
        self._vel_spin.setValue(float(self._preview.velocity_limit))
        self._vel_spin.setKeyboardTracking(False)
        self._vel_spin.setFixedWidth(120)
        self._vel_spin.editingFinished.connect(self._on_scalar_changed)
        grid.addWidget(self._vel_spin, 3, 4)

        grid.setColumnStretch(3, 1)  # spacer column between groups and caps
        outer.addWidget(groups_box, 0)

        # ---- Per-joint table ----
        scroll = QScrollArea(self)
        scroll.setObjectName("pdTableScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._table = QTableWidget(0, len(_COLS), scroll)
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setObjectName("pdTable")
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.verticalHeader().setVisible(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        scroll.setWidget(self._table)
        outer.addWidget(scroll, 1)

        note = I18nLabel(
            "canvas.robot.pd_table.remotized_note",
            default="Remotized joints (linkage-driven) show their angle-dependent "
                    "torque-table PEAK in Effort, not the scalar cap.",
            parent=self,
        )
        note.setObjectName("pdTableNote")
        note.setWordWrap(True)
        outer.addWidget(note, 0)

        ok_cancel = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        ok_cancel.accepted.connect(self.accept)
        ok_cancel.rejected.connect(self.reject)
        outer.addWidget(ok_cancel, 0)

    def _mk_caption(self, text: str) -> QLabel:
        lbl = QLabel(text, self)
        lbl.setObjectName("pdTableCaption")
        return lbl

    # ------------------------------------------------------------------
    # Live recompute
    # ------------------------------------------------------------------

    def _on_group_changed(self, group_id: str) -> None:
        wn = self._wn_spins.get(group_id)
        zeta = self._zeta_spins.get(group_id)
        if wn is None or zeta is None:
            return
        self._overrides[group_id] = {
            "omega_n": float(wn.value()),
            "zeta": float(zeta.value()),
        }
        self._recompute()

    def _on_scalar_changed(self) -> None:
        if self._effort_spin is not None:
            self.result_effort_limit = float(self._effort_spin.value())
        if self._vel_spin is not None:
            self.result_velocity_limit = float(self._vel_spin.value())
        self._recompute()

    def _on_auto_clicked(self) -> None:
        """Fill the official recommended preset (AUTO button), then recompute.

        Three-state outcome (see
        :func:`application.physics.pd_preview.resolve_recommended_actuators`):

        * ``status == "ok"`` — populated; status line is ✓ green.
        * ``status == "need_dump_usd"`` — robot has a USD asset declared
          but no cached drive block yet. AUTO still fills from brand/family
          (so the panel is usable), the status line warns with ⚠, and the
          inline "Re-dump USD" button becomes visible.
        * ``status == "no_usd_family_fallback"`` — no USD asset at all;
          family defaults are the genuine ceiling, status line is ⓘ.

        Per rule §8, AUTO never silently substitutes — every status case
        explicitly tells the user what was filled and what was missing.
        """
        from application.physics.pd_preview import resolve_recommended_actuators

        rec = resolve_recommended_actuators(self._sku)
        if rec is None:
            self._set_status(tr(
                "canvas.robot.pd_table.auto_none",
                default="No recommended preset available for this robot.",
            ))
            self._redump_btn.setVisible(False)
            return

        # Fill (ωn,ζ) for the groups this panel actually shows.
        for gid, (wn, z) in rec.groups.items():
            wn_spin = self._wn_spins.get(gid)
            z_spin = self._zeta_spins.get(gid)
            if wn_spin is None or z_spin is None:
                continue
            for spin, val in ((wn_spin, wn), (z_spin, z)):
                spin.blockSignals(True)
                spin.setValue(float(val))
                spin.blockSignals(False)
            self._overrides[gid] = {"omega_n": float(wn), "zeta": float(z)}

        # Fill caps only when resolved.
        if rec.effort_limit is not None and self._effort_spin is not None:
            self._effort_spin.blockSignals(True)
            self._effort_spin.setValue(float(rec.effort_limit))
            self._effort_spin.blockSignals(False)
            self.result_effort_limit = float(rec.effort_limit)
        if rec.velocity_limit is not None and self._vel_spin is not None:
            self._vel_spin.blockSignals(True)
            self._vel_spin.setValue(float(rec.velocity_limit))
            self._vel_spin.blockSignals(False)
            self.result_velocity_limit = float(rec.velocity_limit)

        self._recompute()

        # Status line + Re-dump visibility, by three-state.
        status_glyph = {
            "ok": "✓",
            "need_dump_usd": "⚠",
            "no_usd_family_fallback": "ⓘ",
        }.get(rec.status, "•")
        self._set_status(f"{status_glyph} {rec.notes}")
        self._redump_btn.setVisible(rec.status == "need_dump_usd")

    def _on_redump_clicked(self) -> None:
        """Spawn the USD dump subprocess for this robot's SKU, then re-fill AUTO.

        Triggered from the inline "Re-dump USD" button that only appears
        when AUTO reported ``status == "need_dump_usd"``. After the dump
        succeeds the registry overlay's ``physics_per_format.USD`` is
        populated and the next AUTO click pulls per-joint drive params
        instead of falling back to brand/family.

        Errors are surfaced in the status label (not a separate modal — the
        user is already in a modal panel). On success the status flips to
        ✓ and the Re-dump button hides itself.
        """
        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )

        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            result = get_robot_asset_service().dump_and_persist(self._sku, "USD")
        except Exception as exc:  # noqa: BLE001
            self._set_status(
                "⚠ " + tr(
                    "canvas.robot.pd_table.redump_failed",
                    default="Re-dump USD failed: {error}",
                ).format(error=f"{type(exc).__name__}: {exc}")
            )
            return
        finally:
            self.unsetCursor()

        if not result.ok:
            self._set_status(
                "⚠ " + tr(
                    "canvas.robot.pd_table.redump_failed",
                    default="Re-dump USD failed: {error}",
                ).format(error=str(result.error or result.error_kind or "unknown"))
            )
            return

        # Refresh the underlying preview (so the per-joint table re-binds
        # against the freshly populated physics block) and re-run AUTO.
        self._on_auto_clicked()

    def _set_status(self, text: str) -> None:
        if getattr(self, "_status_lbl", None) is None:
            return
        self._status_lbl.setText(text)
        self._status_lbl.setVisible(bool(text))

    def _recompute(self) -> None:
        """Re-run the solver with the current overrides + caps and refresh."""
        from application.physics.pd_preview import (
            compute_pd_preview,
            PDPreviewError,
        )

        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self._preview = compute_pd_preview(
                self._sku,
                pd_groups_overrides=self._overrides,
                effort_limit=self.result_effort_limit,
                velocity_limit=self.result_velocity_limit,
            )
        except PDPreviewError:
            # Keep the last good table; the open-time path already validated
            # the SKU/MJCF, so a transient recompute failure (e.g. an
            # out-of-band edit) shouldn't wipe the view.
            from unitport_sdk import log_warning
            log_warning("[pd_param_dialog] live recompute failed — keeping last view")
            return
        finally:
            self.unsetCursor()
        self._refresh_table()

    def _refresh_table(self) -> None:
        if self._table is None:
            return
        joints = self._preview.joints
        self._table.setRowCount(len(joints))
        for r, j in enumerate(joints):
            self._set_cell(r, _COL_IR, j.ir_role)
            self._set_cell(r, _COL_GROUP, j.group_id or "—")
            self._set_cell(r, _COL_WN, f"{j.omega_n:.2f}" if j.group_id else "—")
            self._set_cell(r, _COL_ZETA, f"{j.zeta:.3f}" if j.group_id else "—")
            self._set_cell(r, _COL_KP, f"{j.kp:.4g}")
            self._set_cell(r, _COL_KD, f"{j.kd:.4g}")
            eff_text = f"{j.effort_effective:.2f}"
            if j.remotized:
                eff_text += " ⤵"  # table-driven peak
            eff_item = self._set_cell(r, _COL_EFF, eff_text)
            if j.remotized:
                eff_item.setToolTip(tr(
                    "canvas.robot.pd_table.remotized_tooltip",
                    default="Angle-dependent torque-table peak (remotized joint)",
                ))
            self._set_cell(r, _COL_VEL, f"{self.result_velocity_limit:.2f}")

    def _set_cell(self, row: int, col: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        if col in (_COL_KP, _COL_KD):
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if self._table is not None:
            self._table.setItem(row, col, item)
        return item

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def accept(self) -> None:  # noqa: D401 - QDialog override
        # Flush any in-flight spinbox edits (editingFinished may not have fired
        # if OK is clicked while a box has focus).
        if self._effort_spin is not None:
            self.result_effort_limit = float(self._effort_spin.value())
        if self._vel_spin is not None:
            self.result_velocity_limit = float(self._vel_spin.value())
        for gid, wn in self._wn_spins.items():
            zeta = self._zeta_spins.get(gid)
            if zeta is None:
                continue
            self._overrides[gid] = {
                "omega_n": float(wn.value()),
                "zeta": float(zeta.value()),
            }
        self.result_pd_groups = dict(self._overrides)
        super().accept()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        fg_main = Config.get_color("main_t1")
        fg_sub = Config.get_color("sub_t2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        row_alt = Config.get_color("row_1")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(f"QDialog {{ background-color: {bg}; }}")

        for w in self.findChildren(QLabel, "pdTableHeader"):
            w.setStyleSheet(
                f"QLabel#pdTableHeader {{ color: {fg_main}; "
                f"font-size: {font_normal}px; font-weight: 600; "
                f"background: transparent; }}"
            )
        for name in ("pdTableNote",):
            for w in self.findChildren(QLabel, name):
                w.setStyleSheet(
                    f"QLabel#{name} {{ color: {fg_sub}; "
                    f"font-size: {font_small}px; font-style: italic; "
                    f"background: transparent; }}"
                )
        for w in self.findChildren(QLabel, "pdTableStatus"):
            w.setStyleSheet(
                f"QLabel#pdTableStatus {{ color: {accent}; "
                f"font-size: {font_small}px; background: transparent; }}"
            )
        # LaviButton manages its own themed QSS; refresh on theme change.
        if getattr(self, "_auto_btn", None) is not None:
            self._auto_btn.refresh_style()
        for name in ("pdTableCaption", "pdTableGroupLabel"):
            for w in self.findChildren(QLabel, name):
                w.setStyleSheet(
                    f"QLabel#{name} {{ color: {fg_main}; "
                    f"font-size: {font_small}px; background: transparent; }}"
                )

        spin_qss = (
            f"QDoubleSpinBox {{ color: {fg_main}; "
            f"background-color: transparent; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px 4px; "
            f"font-size: {font_small}px; }}"
            f"QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}"
        )
        for sb in self.findChildren(QDoubleSpinBox):
            sb.setStyleSheet(spin_qss)

        if self._table is not None:
            self._table.setStyleSheet(
                f"QTableWidget#pdTable {{ background-color: {bg}; "
                f"color: {fg_main}; gridline-color: {border}; "
                f"border: 1px solid {border}; font-size: {font_small}px; "
                f"alternate-background-color: {row_alt}; }}"
                f"QHeaderView::section {{ background-color: {bg}; "
                f"color: {fg_sub}; border: none; "
                f"border-bottom: 1px solid {border}; padding: 4px; "
                f"font-size: {font_small}px; }}"
            )
            self._table.setAlternatingRowColors(True)


__all__ = ["PDParamTableDialog"]
