# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""JointMappingDialog — visualise the registry's per-format joint tables.

Opens after a successful Dump (MJCF / USD) cascade, and from the Robot
Asset card's "View joint mapping" button at any time. Shows a side-by-
side table of MJCF and USD joints, IR-role alignment per row, the
cached USD↔MJCF base-height offset (the
``mjcf_base_height_offset`` overlay in ``state.json``), and the deploy-
target coverage summary (computed via
:func:`application.training.trainer_runtime.compute_deploy_coverage`).

Until this dialog existed the user only saw "Dump complete: 30 bodies /
30 joints written" + the Robot Node "all green" indicator, with no way
to spot that MJCF lacked 14 hand IR roles that USD declared. The user
explicitly called that footgun out: "我训练开始的时候没有看到 MJCF 不
支持的声明，结果 simulation 说不支持" — bundles burned compute then
failed at deploy time. This dialog closes that gap.

The IR-role column is an **editable dropdown**: changing a row's role
persists immediately to the user-overlay registry via
:meth:`RobotAssetService.update_ir_role` and live-re-tints the matched /
orphan / unmapped status across both formats. (Wholesale re-dumping still
happens through the Robot Asset card's Dump buttons; the canvas Robot Node
body-mapping table remains the place to edit *body* IR roles.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, log_warning, tr


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------

# Status codes drive row colouring + summary counters. Three buckets:
#   * "matched"   — IR role present in this format AND in the cross-format
#                   table; canonical happy path.
#   * "orphan"    — IR role present in this format but NOT in the other
#                   (e.g. MJCF declares finger joints USD doesn't, or
#                   vice versa). Bundle still loads but the missing-side
#                   deploy target is unavailable.
#   * "unmapped"  — physical joint exists in this format but has empty
#                   ir_role; canvas Body Mapping table must assign one.
#   * "bucket"    — IR role is misc / sensor* / base (legitimately
#                   repeats or differs across formats; not counted
#                   against coverage).
_STATUS_MATCHED = "matched"
_STATUS_ORPHAN = "orphan"
_STATUS_UNMAPPED = "unmapped"
_STATUS_BUCKET = "bucket"

_BUCKET_PREFIXES = ("sensor",)
_BUCKET_EXACT = frozenset({"misc", "base"})


def _is_bucket(ir_role: str) -> bool:
    if not ir_role:
        return False
    if ir_role in _BUCKET_EXACT:
        return True
    return any(ir_role.startswith(p) for p in _BUCKET_PREFIXES)


# ---------------------------------------------------------------------------
# Health snapshot (UI-side aggregation)
# ---------------------------------------------------------------------------

class JointMappingHealth:
    """Per-SKU joint-table + height-offset snapshot built for the dialog.

    Built from registry + RobotAssetService.get_mjcf_base_offset; the
    dialog renders this directly. Separating the snapshot from the
    widget keeps testing trivial (no QApplication needed) and lets
    UX-3 (Robot Node badges) share the same data model.
    """

    def __init__(
        self,
        *,
        sku: str,
        robot_name: str,
        family: str = "",
        mjcf_rows: List[Dict[str, str]],
        usd_rows: List[Dict[str, str]],
        mjcf_asset_path: str,
        usd_asset_path: str,
        height_offset: Optional[Dict[str, Any]],
        coverage_has_gap: bool,
        coverage_missing_in_mjcf: List[str],
        coverage_missing_in_usd: List[str],
        coverage_affected_targets: List[str],
    ) -> None:
        self.sku = sku
        self.robot_name = robot_name
        self.family = family
        self.mjcf_rows = mjcf_rows
        self.usd_rows = usd_rows
        self.mjcf_asset_path = mjcf_asset_path
        self.usd_asset_path = usd_asset_path
        self.height_offset = height_offset
        self.coverage_has_gap = coverage_has_gap
        self.coverage_missing_in_mjcf = coverage_missing_in_mjcf
        self.coverage_missing_in_usd = coverage_missing_in_usd
        self.coverage_affected_targets = coverage_affected_targets

    @classmethod
    def build(cls, sku: str) -> "JointMappingHealth":
        """Pull all data the dialog needs in one pass.

        Reads:
          * registry: ``robots[sku]`` entry — for asset paths +
            joints_per_format[MJCF|USD] tables.
          * RobotAssetService state.json: ``mjcf_base_height_offset``
            overlay — the canonical USD↔MJCF height calibration value.
          * compute_deploy_coverage: the cross-format coverage report.
        """
        from registers import robots as _r
        from application.service.robot_assets import get_robot_asset_service
        from application.training.trainer_runtime import compute_deploy_coverage

        entry = _r.get_robot(sku) or {}
        robot_name = str(entry.get("name") or sku)
        families = list(entry.get("families") or [])
        family = str(families[0]) if families else ""
        assets = entry.get("assets") or {}
        joints_pf = entry.get("joints_per_format") or {}

        mjcf_asset_path = str(assets.get("MJCF") or "") or "(not declared)"
        usd_asset_path = str(assets.get("USD") or assets.get("USD_URL") or "") or "(not declared)"

        # Build the cross-format IR-role sets for orphan detection.
        def _ir_set(fmt: str) -> set:
            block = joints_pf.get(fmt) or {}
            return {
                str((spec or {}).get("ir_role") or "").strip()
                for spec in block.values()
                if isinstance(spec, dict)
                and str((spec or {}).get("ir_role") or "").strip()
            }

        mjcf_ir_set = _ir_set("MJCF")
        usd_ir_set = _ir_set("USD")

        def _rows_for_format(fmt: str, opposite_ir_set: set) -> List[Dict[str, str]]:
            block = joints_pf.get(fmt) or {}
            if not isinstance(block, dict):
                return []
            rows: List[Dict[str, str]] = []
            for uid, spec in block.items():
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name") or "")
                ir_role = str(spec.get("ir_role") or "").strip()
                if _is_bucket(ir_role):
                    status = _STATUS_BUCKET
                elif not ir_role:
                    status = _STATUS_UNMAPPED
                elif ir_role in opposite_ir_set:
                    status = _STATUS_MATCHED
                else:
                    status = _STATUS_ORPHAN
                rows.append({
                    "name": name,
                    "ir_role": ir_role or "—",
                    "ir_role_raw": ir_role,   # "" when unmapped — for the combo
                    "uid": str(uid),           # registry key — for update_ir_role
                    "fmt": fmt,
                    "status": status,
                })
            return rows

        mjcf_rows = _rows_for_format("MJCF", usd_ir_set)
        usd_rows = _rows_for_format("USD", mjcf_ir_set)

        # Height offset (single source of truth: state.json overlay).
        try:
            svc = get_robot_asset_service()
            height_offset = svc.get_mjcf_base_offset(sku)
        except Exception as exc:  # noqa: BLE001
            # State load is best-effort for the dialog — if it fails the
            # dialog just shows "(not calibrated)" rather than crash.
            log_warning(f"[joint_mapping] height-offset read failed: {exc!r}")
            height_offset = None

        # Coverage report (same source of truth as the Play modal).
        try:
            cov = compute_deploy_coverage(sku)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[joint_mapping] coverage compute failed: {exc!r}")
            cov = None

        return cls(
            sku=sku,
            robot_name=robot_name,
            family=family,
            mjcf_rows=mjcf_rows,
            usd_rows=usd_rows,
            mjcf_asset_path=mjcf_asset_path,
            usd_asset_path=usd_asset_path,
            height_offset=height_offset,
            coverage_has_gap=bool(cov and cov.has_gap),
            coverage_missing_in_mjcf=list(cov.missing_in_mjcf) if cov else [],
            coverage_missing_in_usd=list(cov.missing_in_usd) if cov else [],
            coverage_affected_targets=list(cov.affected_targets) if cov else [],
        )


# ---------------------------------------------------------------------------
# Dialog widget
# ---------------------------------------------------------------------------

class JointMappingDialog(QDialog):
    """Scrollable side-by-side MJCF/USD joint table with health summary.

    Use ``JointMappingDialog.open_for(parent, sku)`` rather than
    constructing directly — that classmethod builds the
    :class:`JointMappingHealth` snapshot first, so the dialog always
    reflects the on-disk overlay state at open time (not whatever
    was cached on a previous open).
    """

    def __init__(self, health: JointMappingHealth, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._health = health
        # Per-format editable-row registry: fmt -> [(uid, combo, name_item)].
        # Populated by _build_table; used by _retint_all to recompute the
        # matched/orphan/unmapped status live as the user edits combos.
        self._row_meta: Dict[str, List[Tuple[str, QComboBox, QTableWidgetItem]]] = {}
        # Canonical IR-role choices for this robot's family (dropdown items).
        self._role_choices: List[str] = self._compute_role_choices(health.family)
        self.setWindowTitle(tr(
            "joint_mapping.title",
            "Joint mapping — {name} (sku={sku})",
        ).format(name=health.robot_name, sku=health.sku))
        self.setMinimumSize(820, 560)
        self._build_ui()

    def _compute_role_choices(self, family: str) -> List[str]:
        """Canonical IR role_ids for ``family`` (dropdown options). Empty on
        any failure — the combo then offers just the current value + clear."""
        if not family:
            return []
        try:
            from application.training.body_ir import get_canonical_roles
            out: List[str] = []
            seen: set = set()
            for r in get_canonical_roles(family):
                rid = str(getattr(r, "role_id", "") or "")
                if rid and rid not in seen:
                    out.append(rid)
                    seen.add(rid)
            return out
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[joint_mapping] role catalog load failed for family "
                f"{family!r}: {exc!r}"
            )
            return []

    @classmethod
    def open_for(cls, parent: Optional[QWidget], sku: str) -> "JointMappingDialog":
        """Build a health snapshot for ``sku`` and show the dialog modally."""
        health = JointMappingHealth.build(sku)
        dlg = cls(health, parent=parent)
        dlg.exec()
        return dlg

    # -- layout ------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        root.addWidget(self._build_summary_block())

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_format_panel(
            tr("joint_mapping.mjcf", "MJCF (MuJoCo deploy)"),
            self._health.mjcf_asset_path,
            self._health.mjcf_rows,
        ))
        splitter.addWidget(self._build_format_panel(
            tr("joint_mapping.usd", "USD (IsaacSim / cloud deploy)"),
            self._health.usd_asset_path,
            self._health.usd_rows,
        ))
        splitter.setSizes([400, 400])
        root.addWidget(splitter, 1)

        # Bottom: legend + close button.
        legend = QLabel(tr(
            "joint_mapping.legend",
            "Row colour: <span style='color:#4caf50'>■ matched</span> · "
            "<span style='color:#ff9800'>■ orphan (only in this format)</span> · "
            "<span style='color:#f44336'>■ unmapped (empty IR role)</span> · "
            "<span style='color:#9e9e9e'>■ bucket (misc / sensor / base)</span>",
        ))
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setWordWrap(True)
        root.addWidget(legend)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

    def _build_summary_block(self) -> QWidget:
        """Compact header: deploy-target status + height offset summary."""
        host = QWidget(self)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Deploy-target line.
        if self._health.coverage_has_gap:
            affected_str = "; ".join(self._health.coverage_affected_targets)
            cov_text = tr(
                "joint_mapping.coverage_gap",
                "⚠ Cross-format gap — Unavailable: {affected}",
            ).format(affected=affected_str)
            cov_color = Config.get_color("status_warning", "#ff9800")
        else:
            both_present = bool(self._health.mjcf_rows) and bool(self._health.usd_rows)
            if both_present:
                cov_text = tr(
                    "joint_mapping.coverage_ok",
                    "✓ MJCF and USD agree — both deploy targets available",
                )
            else:
                cov_text = tr(
                    "joint_mapping.coverage_single",
                    "Only one format declared; cross-format check skipped",
                )
            cov_color = Config.get_color("status_ok", "#4caf50")
        cov_label = QLabel(cov_text)
        cov_label.setStyleSheet(f"color: {cov_color}; font-weight: bold;")
        cov_label.setWordWrap(True)
        layout.addWidget(cov_label)

        # Height offset line — single source of truth: state.json overlay.
        layout.addWidget(self._build_height_offset_label())

        return host

    def _build_height_offset_label(self) -> QLabel:
        """Render the cached USD↔MJCF base-height calibration value."""
        overlay = self._health.height_offset
        if not overlay:
            text = tr(
                "joint_mapping.height_uncalibrated",
                "Base-height offset: (not calibrated — run Calibrate from the Robot Asset card)",
            )
            color = Config.get_color("text_secondary", "#888888")
        else:
            status = str(overlay.get("status") or "")
            if status == "not_applicable":
                text = tr(
                    "joint_mapping.height_na",
                    "Base-height offset: not applicable (single-format setup)",
                )
                color = Config.get_color("text_secondary", "#888888")
            else:
                offset_z = overlay.get("offset_z")
                try:
                    offset_str = f"{float(offset_z):.6f} m"
                except (TypeError, ValueError):
                    offset_str = repr(offset_z)
                calib_meta = overlay.get("calibration") or {}
                ts = str(calib_meta.get("calibrated_at") or "").strip()
                text = tr(
                    "joint_mapping.height_value",
                    "Base-height offset (USD↔MJCF): {offset}  ·  calibrated {ts}",
                ).format(offset=offset_str, ts=ts or "(timestamp missing)")
                color = Config.get_color("text_primary", "#cccccc")
        label = QLabel(text)
        label.setStyleSheet(f"color: {color};")
        label.setWordWrap(True)
        return label

    def _build_format_panel(
        self,
        title: str,
        asset_path: str,
        rows: List[Dict[str, str]],
    ) -> QWidget:
        """Per-format panel: title + asset path + joint table."""
        panel = QWidget(self)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold;")
        col.addWidget(title_label)

        path_label = QLabel(tr("joint_mapping.asset_path", "Asset: {path}").format(path=asset_path))
        path_label.setStyleSheet(
            f"color: {Config.get_color('text_secondary', '#888888')};"
        )
        path_label.setWordWrap(True)
        col.addWidget(path_label)

        if not rows:
            empty = QLabel(tr(
                "joint_mapping.no_rows",
                "(no joints dumped — click Dump on the Robot Asset card)",
            ))
            empty.setStyleSheet(
                f"color: {Config.get_color('text_secondary', '#888888')}; font-style: italic;"
            )
            col.addWidget(empty)
            col.addStretch(1)
            return panel

        table = self._build_table(rows)
        col.addWidget(table, 1)

        # Per-table count summary.
        n_matched = sum(1 for r in rows if r["status"] == _STATUS_MATCHED)
        n_orphan = sum(1 for r in rows if r["status"] == _STATUS_ORPHAN)
        n_unmapped = sum(1 for r in rows if r["status"] == _STATUS_UNMAPPED)
        n_bucket = sum(1 for r in rows if r["status"] == _STATUS_BUCKET)
        summary_text = tr(
            "joint_mapping.row_summary",
            "Total: {total}  ·  matched: {m}  ·  orphan: {o}  ·  unmapped: {u}  ·  bucket: {b}",
        ).format(
            total=len(rows), m=n_matched, o=n_orphan, u=n_unmapped, b=n_bucket,
        )
        summary = QLabel(summary_text)
        summary.setStyleSheet(
            f"color: {Config.get_color('text_secondary', '#888888')}; font-size: 11px;"
        )
        col.addWidget(summary)
        return panel

    def _build_table(self, rows: List[Dict[str, str]]) -> QTableWidget:
        table = QTableWidget(len(rows), 2, self)
        table.setHorizontalHeaderLabels([
            tr("joint_mapping.col_joint", "Physical joint"),
            tr("joint_mapping.col_ir", "IR role"),
        ])
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        # Editing is via the per-row IR-role combo cell-widget, not in-cell
        # item editing — so leave item edit-triggers off.
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(False)
        h = table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        fmt = str(rows[0].get("fmt", "")) if rows else ""
        meta: List[Tuple[str, QComboBox, QTableWidgetItem]] = []
        for i, row in enumerate(rows):
            name_item = QTableWidgetItem(str(row["name"]))
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            tint = self._row_tint(row["status"])
            if tint is not None:
                name_item.setBackground(tint)
            tooltip = self._row_tooltip(row["status"])
            if tooltip:
                name_item.setToolTip(tooltip)
            table.setItem(i, 0, name_item)

            raw = str(row.get("ir_role_raw", "") or "")
            uid = str(row.get("uid", "") or "")
            combo = self._make_role_combo(raw)
            # Connect AFTER the initial index is set inside _make_role_combo,
            # so the programmatic preselect doesn't fire a spurious persist.
            combo.currentIndexChanged.connect(
                lambda _i, f=fmt, u=uid, c=combo: self._on_ir_changed(f, u, c)
            )
            table.setCellWidget(i, 1, combo)
            meta.append((uid, combo, name_item))

        if fmt:
            self._row_meta[fmt] = meta
        return table

    def _make_role_combo(self, current: str) -> QComboBox:
        """Themed IR-role dropdown. Item 0 = empty (clear/unmapped); then the
        canonical role_ids. The current value is preselected — and prepended
        if it isn't a canonical role (e.g. a bucket role like ``misc`` or a
        legacy value) so editing never silently drops it."""
        combo = QComboBox()
        combo.addItem("", userData="")
        choices = list(self._role_choices)
        if current and current not in choices:
            choices = [current] + choices
        for rid in choices:
            combo.addItem(rid, userData=rid)
        idx = combo.findData(current or "")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        bg = Config.get_color("bg_2")
        fg = Config.get_color("main_t1")
        border = Config.get_color("border_1")
        sel = Config.get_color("row_1")
        combo.setStyleSheet(
            f"QComboBox {{ background: {bg}; color: {fg}; "
            f"border: 1px solid {border}; padding: 2px 6px; }}"
            f"QComboBox QAbstractItemView {{ background: {bg}; color: {fg}; "
            f"selection-background-color: {sel}; }}"
        )
        return combo

    def _on_ir_changed(self, fmt: str, uid: str, combo: QComboBox) -> None:
        """Persist a single IR-role edit immediately, then re-tint live."""
        data = combo.currentData()
        role = data if isinstance(data, str) else ""
        try:
            from application.service.robot_assets import get_robot_asset_service
            ok = get_robot_asset_service().update_ir_role(
                self._health.sku, fmt, "joint", uid, role,
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[joint_mapping] update_ir_role raised sku={self._health.sku!r} "
                f"fmt={fmt!r} uid={uid!r} role={role!r}: {exc!r}"
            )
            ok = False
        if not ok:
            log_warning(
                f"[joint_mapping] update_ir_role failed sku={self._health.sku!r} "
                f"fmt={fmt!r} uid={uid!r} role={role!r}"
            )
        self._retint_all()

    def _retint_all(self) -> None:
        """Recompute matched/orphan/unmapped status for every row from the
        live combo selections (both formats) and re-tint the name cells."""
        live: Dict[str, set] = {}
        for fmt, meta in self._row_meta.items():
            s: set = set()
            for _uid, combo, _name_item in meta:
                d = combo.currentData()
                v = d.strip() if isinstance(d, str) else ""
                if v and not _is_bucket(v):
                    s.add(v)
            live[fmt] = s
        for fmt, meta in self._row_meta.items():
            opp = live.get("USD" if fmt == "MJCF" else "MJCF", set())
            for _uid, combo, name_item in meta:
                d = combo.currentData()
                v = d.strip() if isinstance(d, str) else ""
                if _is_bucket(v):
                    status = _STATUS_BUCKET
                elif not v:
                    status = _STATUS_UNMAPPED
                elif v in opp:
                    status = _STATUS_MATCHED
                else:
                    status = _STATUS_ORPHAN
                tint = self._row_tint(status)
                name_item.setBackground(tint if tint is not None else QBrush())
                name_item.setToolTip(self._row_tooltip(status) or "")

    @staticmethod
    def _row_tint(status: str) -> Optional[QColor]:
        """Translucent row background by status — readable on dark theme."""
        # Alpha kept low so the cell text colour from the theme palette
        # is still legible underneath. Hex colours pulled from system.ini
        # status_* slots when present.
        mapping = {
            _STATUS_MATCHED: ("status_ok", "#4caf50"),
            _STATUS_ORPHAN: ("status_warning", "#ff9800"),
            _STATUS_UNMAPPED: ("status_error", "#f44336"),
            _STATUS_BUCKET: ("text_secondary", "#9e9e9e"),
        }
        slot_default = mapping.get(status)
        if slot_default is None:
            return None
        slot, default = slot_default
        col = QColor(Config.get_color(slot, default))
        col.setAlpha(60)
        return col

    @staticmethod
    def _row_tooltip(status: str) -> str:
        if status == _STATUS_MATCHED:
            return tr(
                "joint_mapping.tooltip.matched",
                "IR role present in both MJCF and USD tables — bundle deploys cleanly.",
            )
        if status == _STATUS_ORPHAN:
            return tr(
                "joint_mapping.tooltip.orphan",
                "IR role declared in this format only — the OTHER deploy "
                "target won't honor this joint.",
            )
        if status == _STATUS_UNMAPPED:
            return tr(
                "joint_mapping.tooltip.unmapped",
                "Physical joint dumped but no IR role assigned. Open the "
                "Robot Node Body Mapping table to assign one before training.",
            )
        if status == _STATUS_BUCKET:
            return tr(
                "joint_mapping.tooltip.bucket",
                "Bucket role (misc / sensor* / base) — legitimately "
                "repeats or differs across formats; ignored by coverage check.",
            )
        return ""


__all__ = ["JointMappingDialog", "JointMappingHealth"]
