# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RobotAssetsPanel — registry-driven asset browser AND register editor.

Sidebar's Robot Asset panel is the **canonical entrypoint for editing the
registers.robots register**. It surfaces every (brand, model) known to
``registers.robots`` — both canonical (factory ``data/robots_canonical.json``)
and user-layer (``<USER_CONFIG_DIR>/registers/robots_custom.json``) — and lets the
user add / edit / delete user-layer entries through :class:`RobotEditorDialog`.
Per-asset selection / family-tag / body-IR override state still lives in
``<USER_CONFIG_DIR>/robot_assets/state.json`` via ``RobotAssetService``.

UI rules (CLAUDE.md §1.5):
    * every label uses size_small;
    * every color reads from system.ini[Theme] via Config.get_color (no hex
      literals); slots used: bg_2, sidebar_card_bg, sidebar_card_border,
      sidebar_card_active_border, sidebar_hover_overlay, sidebar_canonical_badge,
      sidebar_user_badge, safe_zone, main_c2, etc.;
    * the entire card header is a single click-target with hover feedback and
      a pointing-hand cursor.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviComboBox,
    TagInput,
    i18n_bind,
    log_debug,
    log_error,
    log_info,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.robot_assets import (
    ASSET_KINDS,
    STATUS_LOCAL,
    STATUS_MISSING,
    STATUS_REMOTE,
    AssetRecord,
    AssetRecordGroup,
    get_robot_asset_service,
)
from application.ui.dialogs.menagerie_browser_dialog import MenagerieBrowserDialog
from application.ui.sidebar_panels.robot_editor_dialog import RobotEditorDialog


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


class _CardHeader(QFrame):
    """Whole-row clickable header for an _AssetCard.

    Click anywhere on the header → toggles ``parent.toggle_body``.
    Hover → background flips to sidebar_hover_overlay.
    """

    def __init__(self, on_toggle, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_toggle = on_toggle
        self.setObjectName("assetCardHeader")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Drop the implicit text-driven minimum so the QScrollArea host can
        # shrink to its viewport width instead of expanding to the title's
        # natural width.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply_style(hover=False)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        # Allow horizontal clipping; preserve vertical hint so the layout
        # still allocates enough height for the title row.
        s = super().minimumSizeHint()
        return QSize(0, s.height())

    def _apply_style(self, *, hover: bool) -> None:
        bg = (Config.get_color("row_1")
              if hover else "transparent")
        self.setStyleSheet(
            f"#assetCardHeader {{ background: {bg};"
            f" border-radius: 3px; }}"
        )

    def enterEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=True)
        super().enterEvent(e)

    def leaveEvent(self, e):  # type: ignore[override]
        self._apply_style(hover=False)
        super().leaveEvent(e)

    def mousePressEvent(self, e):  # type: ignore[override]
        if e.button() == Qt.MouseButton.LeftButton:
            self._on_toggle()
            e.accept()
            return
        super().mousePressEvent(e)


class _AssetCard(QWidget):
    """Collapsible card for a single (brand, model) — canonical or user-layer."""

    def __init__(
        self,
        record: AssetRecord,
        on_edit_request,
        on_delete_request,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._record = record
        self._svc = get_robot_asset_service()
        self._on_edit_request = on_edit_request
        self._on_delete_request = on_delete_request
        self._expanded = False
        self._is_user = self._svc.is_user_extension(record.sku)

        self.setObjectName("assetCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Same trick as _CardHeader: the card must be allowed to shrink below
        # its title's natural width, otherwise QScrollArea expands the host
        # widget past the sidebar viewport (280 px → 224 px after group +
        # padding chain) and content visibly overflows.
        self._apply_card_style(active=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # ----- header (whole row clickable) -----
        self._header = _CardHeader(self._toggle_body)
        header_row = QHBoxLayout(self._header)
        header_row.setContentsMargins(4, 2, 4, 2)
        header_row.setSpacing(6)

        self._glyph = QLabel("▶")
        self._glyph.setStyleSheet(
            f"color: {Config.get_color('main_c1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        header_row.addWidget(self._glyph)

        # ``record.name`` already carries brand (e.g. "Unitree Go2"); prefixing
        # ``record.brand`` would yield "unitree Unitree Go2".
        display_name = record.name or f"{record.brand} {record.model}".strip()
        title = setText(
            f"robot_assets.title.{record.sku}",
            default=display_name,
            kind="title", size=_ss(),
            color=Config.get_color("highlight"),
        )
        # ``Ignored`` horizontal policy lets the label clip when the panel is
        # narrower than the title's natural text width. Without this, QLabel's
        # text width feeds back into the parent's minimumSizeHint and the
        # whole card expands past the 224-px sidebar viewport.
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        title.setMinimumWidth(0)
        title.setToolTip(display_name)
        header_row.addWidget(title, 1)
        layout.addWidget(self._header)

        # ----- body (hidden until expanded) -----
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 4, 4, 4)
        body_layout.setSpacing(4)

        for kind in ASSET_KINDS:
            body_layout.addLayout(self._build_kind_row(kind))

        # Family tags
        tags_row = QHBoxLayout()
        tags_row.addWidget(setText(
            "robot_assets.family_tags", default="Tags:",
            kind="content", size=_ss(),
        ))
        self._tag_input = TagInput(
            mode="input",
            placeholder_key="robot_assets.tag_placeholder",
            placeholder=tr("robot_assets.tag_placeholder", "Add tag..."),
            font_size=_ss(),
        )
        # TagInput's internal QLineEdit needs ~28 px to render the placeholder
        # / chips without clipping (its 4-px outer margins + 2-px padding eat
        # most of a 22-px shell). Keep min/max equal so the row doesn't grow
        # vertically as tags wrap.
        self._tag_input.setMinimumHeight(28)
        self._tag_input.setMaximumHeight(28)
        self._tag_input.set_values(record.family_tags)
        self._tag_input.changed.connect(self._on_tags_changed)
        tags_row.addWidget(self._tag_input, 1)
        body_layout.addLayout(tags_row)

        # Unified Dump button. The cascade in ``_on_dump_format`` already
        # auto-runs the second format after the first succeeds, so two
        # buttons were redundant — they did the same thing modulo which
        # format ran first. Pick the available format as the "primary":
        # prefer MJCF (in-process via mujoco, ~0.1s) over USD (Isaac Lab
        # subprocess, ~5-30s) when both are available, since MJCF
        # success warms the cascade fast.
        mjcf_local = record.status.get("MJCF") == "local"
        usd_avail = (
            record.status.get("USD") in ("local", "remote")
            or bool(record.canonical_url.get("USD"))
        )
        if mjcf_local:
            primary_fmt = "MJCF"
        elif usd_avail:
            primary_fmt = "USD"
        else:
            primary_fmt = ""

        dump_btn = setButton(
            f"robot_assets.dump.{record.sku}", 0, 22,
            kind="border", spec="none",
            default=tr("robot_assets.dump", "Dump bodies / joints"),
        )
        dump_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dump_btn.setEnabled(bool(primary_fmt))
        if not primary_fmt:
            dump_btn.setToolTip(
                tr("robot_assets.dump_disabled",
                   "Neither MJCF nor USD asset is available "
                   "(no local file and no Nucleus URL declared)")
            )
        else:
            dump_btn.clicked.connect(
                lambda _c=False, sku=record.sku, fmt=primary_fmt:
                self._on_dump_format(sku, fmt)
            )
        body_layout.addWidget(dump_btn)

        # UX-1: dedicated "View joint mapping" entrypoint so users can
        # inspect the joint table + height-offset state WITHOUT running
        # a fresh dump. Opens the same dialog the dump-end shows.
        view_mapping_btn = setButton(
            f"robot_assets.view_mapping.{record.sku}", 0, 22,
            kind="border", spec="none",
            default=tr("robot_assets.view_mapping", "View joint mapping"),
        )
        view_mapping_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        view_mapping_btn.clicked.connect(
            lambda _c=False, sku=record.sku: self._on_view_joint_mapping(sku)
        )
        body_layout.addWidget(view_mapping_btn)

        # Coming-soon placeholders kept (calibrate xacro etc.)
        coming_soon = tr("robot_assets.coming_soon", "Coming soon")
        for key, default in (
            ("robot_assets.calibrate_xacro", "Calibrate XACRO"),
        ):
            btn = setButton(key, 0, 22, kind="border", spec="none", default=default)
            btn.setEnabled(False)
            btn.setToolTip(coming_soon)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            body_layout.addWidget(btn)

        # Edit / Delete row
        edit_row = QHBoxLayout()
        edit_row.addStretch(1)
        self._btn_edit = setButton(
            f"robot_assets.edit.{record.sku}", 60, 22,
            kind="border", spec="none",
            default=tr("robot_assets.edit", "Edit"),
        )
        if self._is_user:
            self._btn_edit.clicked.connect(
                lambda _c=False, sku=record.sku: self._on_edit_request(sku)
            )
        else:
            self._btn_edit.setEnabled(False)
            i18n_bind(
                self._btn_edit, "setToolTip",
                "robot_assets.edit_canonical_disabled",
                "Canonical robots are read-only. Add a new user entry instead.",
            )
        edit_row.addWidget(self._btn_edit)
        if self._is_user:
            self._btn_delete = setButton(
                f"robot_assets.delete.{record.sku}", 60, 22,
                kind="border", spec="danger",
                default=tr("robot_assets.delete", "Delete"),
            )
            self._btn_delete.clicked.connect(
                lambda _c=False, sku=record.sku: self._on_delete_request(sku)
            )
            edit_row.addWidget(self._btn_delete)
        body_layout.addLayout(edit_row)

        self._body.setVisible(False)
        layout.addWidget(self._body)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        s = super().minimumSizeHint()
        return QSize(0, s.height())

    # ----- card chrome -----------------------------------------------------

    def _apply_card_style(self, *, active: bool) -> None:
        border = (
            Config.get_color("sidebar_card_active_border") if active
            else Config.get_color("border_1")
        )
        self.setStyleSheet(
            f"#assetCard {{ background: transparent;"
            f" border: 1px solid {border}; border-radius: 4px; }}"
        )

    def _build_kind_row(self, kind: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        kind_label = setText(
            f"robot_assets.kind.{kind.lower()}", default=f"{kind}:",
            kind="content", size=_ss(),
        )
        kind_label.setFixedWidth(46)
        row.addWidget(kind_label)

        # Combo items: canonical (read-only) → custom (user paths) → cloud (URL marker).
        # The panel and the canvas Robot Node both read from registers.robots —
        # ``record.canonical`` exposes the same canonical assets that
        # ``RobotAssetService.resolve()`` would resolve, so the dropdown finally
        # surfaces the registry's MJCF/USD/URDF/XACRO instead of "(none)".
        canonical_path = self._record.canonical.get(kind) or ""
        custom_paths = list(self._record.paths.get(kind, []))
        cloud_url = (
            self._record.canonical_url.get(kind) or "" if kind == "USD" else ""
        )

        items: List = []
        if canonical_path:
            items.append((
                canonical_path,
                f"{canonical_path}  [{tr('robot_assets.tag.canonical', 'canonical')}]",
            ))
        for cp in custom_paths:
            items.append((
                cp,
                f"{cp}  [{tr('robot_assets.tag.custom', 'custom')}]",
            ))
        if cloud_url and not canonical_path and not custom_paths:
            # Cloud marker is display-only — same key as label so set_selected_path
            # rejects via the no-empty check; the combo callback also guards below.
            items.append((
                cloud_url,
                f"{cloud_url}  [{tr('robot_assets.tag.cloud', 'cloud')}]",
            ))
        if not items:
            items = [("", tr("robot_assets.no_path", "(none)"))]

        # When the kind has no canonical path, no custom paths, and no cloud
        # marker, the combo would only render a "(none)" placeholder — drop it
        # entirely so the row collapses to "label … pill add-btn" and the user
        # immediately sees the missing-flag plus the "add path" affordance.
        has_any_path = bool(canonical_path or custom_paths or cloud_url)
        if has_any_path:
            combo = LaviComboBox(items, i18n=False)
            # LaviComboBox defaults to setMinimumWidth(120). The 280-px sidebar
            # minus margins / kind label / status pill / add button leaves
            # ~120 px for the combo. Drop the minimum so the combo can shrink.
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            sel = self._record.selected.get(kind, "")
            if sel:
                keys = [k for k, _ in items]
                if sel in keys:
                    combo.setCurrentKey(sel)
            combo.currentIndexChanged.connect(
                lambda _idx, c=combo, k=kind: self._on_combo_changed(k, c.currentKey())
            )
            row.addWidget(combo, 1)
        else:
            # Stretch so the pill + add-btn stay right-aligned even without the
            # combo carrying the expanding role.
            row.addStretch(1)

        # Status pill (LOCAL / REMOTE / MISSING) — colour from system.ini[Theme].
        status = self._record.status.get(kind, STATUS_MISSING)
        if cloud_url and not canonical_path and not custom_paths:
            # Cloud-only USD: render as a distinct "cloud" pill in cloud blue.
            pill_color = Config.get_color("robot_asset_status_cloud")
            pill_text = tr("robot_assets.status.cloud", "cloud")
        elif status == STATUS_LOCAL:
            pill_color = Config.get_color("safe_zone")
            pill_text = tr("robot_assets.status.local", "local")
        elif status == STATUS_REMOTE:
            pill_color = Config.get_color("robot_asset_status_remote")
            pill_text = tr("robot_assets.status.remote", "remote")
        else:
            pill_color = Config.get_color("robot_asset_status_missing")
            pill_text = tr("robot_assets.status.missing", "missing")
        pill = QLabel(pill_text)
        pill.setStyleSheet(
            f"color: {pill_color}; background: transparent;"
            f" font-size: {_ss()}px; padding: 0 4px;"
        )
        pill.setToolTip(pill_text)
        row.addWidget(pill)

        btn_add = setButton(
            f"robot_assets.add.{self._record.sku}.{kind}", 22, 22,
            kind="light", spec="none",
            icon="icon_setting", icon_only=True, default="",
        )
        i18n_bind(btn_add, "setToolTip",
                  "robot_assets.add_path", "Add a custom path")
        btn_add.clicked.connect(lambda _c=False, k=kind: self._on_add_path(k))
        row.addWidget(btn_add)
        return row

    # ----- callbacks ------------------------------------------------------

    def _toggle_body(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._glyph.setText("▼" if self._expanded else "▶")
        self._apply_card_style(active=self._expanded)

    def _on_tags_changed(self) -> None:
        self._svc.set_family_tags(
            self._record.brand, self._record.model, self._tag_input.get_values(),
        )

    def _is_format_dump_available(self, fmt: str) -> bool:
        """True iff this card's robot has the asset path/URL needed to dump ``fmt``.

        Mirrors the per-format enablement logic used at button-construction
        time (lines 226-249) so the cascade in :meth:`_on_dump_format` decides
        the same way the explicit buttons would have.
        """
        f = (fmt or "").upper()
        if f == "MJCF":
            return self._record.status.get("MJCF") == "local"
        if f == "USD":
            return (
                self._record.status.get("USD") in ("local", "remote")
                or bool(self._record.canonical_url.get("USD"))
            )
        return False

    def _dump_one_format(self, sku: str, fmt: str):
        """Execute one dump_and_persist + emit log lines. Returns the DumpResult.

        Extracted from :meth:`_on_dump_format` so the cascade can run the
        same pipeline for the "other" format without duplicating
        error-routing logic. The caller decides how to present results
        (single dialog or combined).
        """
        log_info(f"[robot_assets] dump {fmt} start: sku={sku!r}")
        result = self._svc.dump_and_persist(sku, fmt)
        if result.ok:
            log_info(
                f"[robot_assets] dump {fmt} OK sku={sku!r}: "
                f"{len(result.joints)} joints "
                f"({result.n_joints_resolved} resolved), "
                f"{len(result.bodies)} bodies "
                f"({result.n_bodies_resolved} resolved), "
                f"{result.n_unresolved} need manual IR-role assignment"
            )
        return result

    def _on_view_joint_mapping(self, sku: str) -> None:
        """Open the read-only joint mapping dialog without re-dumping.

        Same dialog the post-dump cascade shows; this entry point lets
        the user inspect the current state of the MJCF / USD per-format
        tables + cached base-height offset at any time (e.g. before
        clicking Play to confirm dump health, or after a registry
        overlay edit to verify the result).
        """
        try:
            from application.ui.dialogs import JointMappingDialog
            JointMappingDialog.open_for(self, sku)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[robot_assets] JointMappingDialog open failed: {exc!r}"
            )
            QMessageBox.warning(
                self,
                tr("robot_assets.view_mapping_error_title",
                   "Joint mapping unavailable"),
                f"{exc}",
            )

    def _on_dump_format(self, sku: str, fmt: str) -> None:
        """Dump MJCF / USD bodies + joints into the per-format registry slot.

        Thin UI wrapper around :meth:`RobotAssetService.dump_and_persist`:
        adds a busy cursor (USD path may block 5-30s) and routes errors
        into QMessageBox for the user. The actual dump+persist pipeline,
        seed map / family lookup, and exception handling all live in the
        service so the canvas Body Mapping Refresh button can share it.

        Single-click both-format sync: after the explicitly-clicked
        format dumps successfully, automatically dump the OTHER format
        too when its asset is declared. The two per-format tables must
        agree on joint count for Isaac Lab → MuJoCo sim2sim (bundle
        export's ``_derive_mujoco_pd_gains_for_bundle`` looks up every IR
        role across BOTH tables); cascading the dump removes the
        "I dumped USD but forgot MJCF (or vice versa) → bundle export
        explodes on IR-role-not-in-MJCF" footgun. Cross-format joint
        count drift after both dumps surfaces as a loud warning so the
        user notices when their asset paths point at incompatible
        variants (e.g. G1 ``scene.xml`` 29-DOF vs. ``g1.usd`` 43-DOF).
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        cascade_fmt = "MJCF" if fmt.upper() == "USD" else "USD"
        do_cascade = self._is_format_dump_available(cascade_fmt)

        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            primary = self._dump_one_format(sku, fmt)
            cascade = None
            if primary.ok and do_cascade:
                log_info(
                    f"[robot_assets] cascading dump {cascade_fmt} after "
                    f"successful {fmt} (single-click both-format sync)"
                )
                cascade = self._dump_one_format(sku, cascade_fmt)
        finally:
            QApplication.restoreOverrideCursor()

        # Primary failure short-circuits everything — show the canonical
        # per-error dialog and exit. Cascade is never attempted when
        # primary fails (see guard above) so we don't need to merge here.
        if not primary.ok:
            self._show_dump_error(sku, fmt, primary)
            return

        # Cascade failure: primary succeeded, but the auto-second-format
        # didn't. Surface as a warning (not a fatal) — the primary write
        # IS already on disk and visible to the rest of the app. User may
        # need to manually fix the cascade format (missing asset path,
        # Isaac kit crash, etc.).
        if cascade is not None and not cascade.ok:
            log_warning(
                f"[robot_assets] cascade dump {cascade_fmt} failed after "
                f"successful {fmt}: kind={cascade.error_kind} "
                f"msg={cascade.error!r}"
            )
            self._show_dump_error(sku, cascade_fmt, cascade, cascade=True)
            # Fall through — still want to show the primary's success and
            # any divergence info, since the primary IS a real change.

        # Build the combined success dialog body.
        lines = [self._format_success_line(fmt, primary)]
        if cascade is not None and cascade.ok:
            lines.append(self._format_success_line(cascade_fmt, cascade))

            # Cross-format joint-count divergence — ADVISORY only. The
            # two formats describing different DOF-variants only limits
            # deploy-target availability (smaller-DOF format can't honor
            # the trained joint set); training itself is unaffected.
            # Hard block belongs at MuJoCo runtime load, not here.
            n_primary = len(primary.joints)
            n_cascade = len(cascade.joints)
            if n_primary != n_cascade:
                # Name the format with fewer joints — its deploy target
                # is the one that loses coverage.
                if n_primary < n_cascade:
                    smaller_fmt, smaller_n, larger_fmt, larger_n = (
                        fmt, n_primary, cascade_fmt, n_cascade,
                    )
                else:
                    smaller_fmt, smaller_n, larger_fmt, larger_n = (
                        cascade_fmt, n_cascade, fmt, n_primary,
                    )
                affected = (
                    "MuJoCo deploy" if smaller_fmt == "MJCF"
                    else "IsaacSim deploy"
                )
                lines.append(
                    tr(
                        "robot_assets.dump_format_mismatch",
                        "\n⚠ {larger_fmt} has {larger_n} joints but "
                        "{smaller_fmt} has {smaller_n}. Training is "
                        "unaffected, but {affected} won't be available "
                        "for bundles trained against the {larger_fmt}-side "
                        "joint set. If you need both deploy targets, "
                        "repoint the {smaller_fmt} asset to a matching "
                        "variant (e.g. for G1 use scene_with_hands.xml "
                        "alongside g1.usd) and re-Dump."
                    ).format(
                        smaller_fmt=smaller_fmt, smaller_n=smaller_n,
                        larger_fmt=larger_fmt, larger_n=larger_n,
                        affected=affected,
                    )
                )
                log_warning(
                    f"[robot_assets] joint-count divergence sku={sku!r}: "
                    f"{smaller_fmt}={smaller_n} < {larger_fmt}={larger_n} "
                    f"— {affected} target unavailable for bundles trained "
                    f"against {larger_fmt}"
                )

        QMessageBox.information(
            self,
            tr("robot_assets.dump_done_title", "Dump complete"),
            "\n".join(lines),
        )

        # After a dump that refreshed BOTH the USD and MJCF body tables, offer
        # the USD→MJCF inertia calibration (closes the sim2sim base-property gap
        # for robots whose MJCF carries placeholder inertia, e.g. Spot's trunk).
        # Gated on both-this-round so calibrate_mjcf_inertia has both
        # bodies_per_format tables to map across; opt-in via a question dialog
        # because the USD harvest can take 5–30 s.
        usd_dumped = (
            (fmt.upper() == "USD" and primary.ok)
            or (cascade is not None and cascade.ok and cascade_fmt == "USD")
        )
        mjcf_dumped = (
            (fmt.upper() == "MJCF" and primary.ok)
            or (cascade is not None and cascade.ok and cascade_fmt == "MJCF")
        )
        if usd_dumped and mjcf_dumped:
            self._offer_inertia_calibration(sku)

        # UX-1: every dump (success or cascade-partial) now hands the user
        # the full joint mapping table — what's matched, what's orphan,
        # what's still unmapped, and the cached USD↔MJCF base-height
        # offset. Closes the "30 joints written" → "user can't see if
        # there's a problem" footgun the user explicitly called out.
        try:
            from application.ui.dialogs import JointMappingDialog
            JointMappingDialog.open_for(self, sku)
        except Exception as exc:  # noqa: BLE001
            # Dialog failure must NOT block the dump itself — the dump
            # already succeeded and registry is updated. Just log so the
            # next dev sees the issue.
            log_warning(f"[robot_assets] JointMappingDialog open failed: {exc!r}")

    def _offer_inertia_calibration(self, sku: str) -> None:
        """Offer (opt-in) the USD→MJCF mass/inertia calibration after a dump.

        Runs ``calibrate_mjcf_inertia`` + ``write_overlay`` under a busy cursor
        when the user accepts. The original MJCF is never modified — a YAML
        overlay is written under USER_CONFIG_DIR and applied to the in-memory
        MjModel at MuJoCo load. Fail-loud calibration errors (unauthored USD
        inertia, frame mismatch) surface in a warning dialog with the directive,
        rather than producing a half-correct overlay.
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        ans = QMessageBox.question(
            self,
            tr("robot_assets.inertia_calib_offer_title",
               "Calibrate inertia from USD?"),
            tr("robot_assets.inertia_calib_offer_body",
               "Compare the training USD's per-body mass / inertia against the "
               "MuJoCo MJCF and write a non-destructive overlay (the original "
               "MJCF is never changed). This closes the IsaacLab→MuJoCo sim2sim "
               "gap for robots whose MJCF ships placeholder inertia. The USD "
               "harvest may take 5–30 s."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        result = None
        rel = ""
        err: Optional[Exception] = None
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            from application.training.validation.usd_mjcf_inertia_calibration import (
                calibrate_mjcf_inertia,
                write_overlay,
            )
            result = calibrate_mjcf_inertia(sku)
            rel = write_overlay(result)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user below
            err = exc
        finally:
            QApplication.restoreOverrideCursor()

        if err is not None:
            log_warning(
                f"[robot_assets] inertia calibration failed sku={sku!r}: {err!r}"
            )
            QMessageBox.warning(
                self,
                tr("robot_assets.inertia_calib_error_title",
                   "Inertia calibration failed"),
                f"{err}",
            )
            return

        lines = [
            tr("robot_assets.inertia_calib_done",
               "Inertia overlay {status}: {n} body override(s) written to {rel}.")
            .format(status=result.status, n=len(result.overrides), rel=rel)
        ]
        for ov in result.overrides:
            lines.append(f"  • {ov.mjcf_body} ({ov.ir_role}): "
                         f"{', '.join(ov.fields_applied)}")
        if result.ignored:
            lines.append(
                tr("robot_assets.inertia_calib_ignored",
                   "{n} body(ies) below tolerance — see the overlay's "
                   "ignored_bodies audit list.").format(n=len(result.ignored))
            )
        QMessageBox.information(
            self,
            tr("robot_assets.inertia_calib_done_title",
               "Inertia calibration complete"),
            "\n".join(lines),
        )

    def _format_success_line(self, fmt: str, result) -> str:
        """Build one human-readable success summary line for a completed dump."""
        if result.n_unresolved == 0:
            key, default = (
                "robot_assets.dump_done_body_complete",
                "{fmt}: {n_bodies} bodies / {n_joints} joints written. "
                "Open the canvas Robot node to review IR-role assignments.",
            )
        else:
            key, default = (
                "robot_assets.dump_done_body_partial",
                "{fmt}: {n_bodies} bodies / {n_joints} joints written. "
                "{unresolved} entries could not be auto-mapped — open the "
                "canvas Robot node BodyMappingTable to assign their IR roles "
                "before training.",
            )
        return tr(key, default).format(
            fmt=fmt,
            n_bodies=len(result.bodies),
            n_joints=len(result.joints),
            unresolved=result.n_unresolved,
        )

    def _show_dump_error(self, sku: str, fmt: str, result, *, cascade: bool = False) -> None:
        """Route a failed DumpResult into the canonical per-error QMessageBox."""
        kind = result.error_kind
        msg = result.error or "unknown error"
        title_key = ("robot_assets.dump_cascade_error_title"
                     if cascade else "robot_assets.dump_error_title")
        title_default = (f"Cascade dump ({fmt}) failed"
                         if cascade else "Dump failed")
        if kind == "no_asset":
            log_warning(f"[robot_assets] dump {fmt} failed: {msg}")
            QMessageBox.warning(
                self,
                tr(title_key, title_default),
                tr("robot_assets.dump_error_no_asset",
                   "Cannot resolve robot asset for SKU {sku}").format(sku=sku),
            )
            return
        if kind == "discovery":
            log_warning(
                f"[robot_assets] dump {fmt} discovery error sku={sku!r}: {msg}"
            )
            QMessageBox.warning(self, tr(title_key, title_default), msg)
            return
        if kind == "validation":
            log_warning(
                f"[robot_assets] dump {fmt} validation failed sku={sku!r}: {msg}"
            )
            QMessageBox.warning(
                self,
                tr(title_key, title_default),
                tr("robot_assets.dump_error_validation",
                   "Validation failed after writing {fmt} mapping:\n\n{exc}\n\n"
                   "Open the canvas Robot node and fix the flagged "
                   "joint/body IR roles, then re-Dump.").format(fmt=fmt, exc=msg),
            )
            return
        if kind == "persist":
            log_warning(
                f"[robot_assets] dump {fmt} persist returned False sku={sku!r}"
            )
            QMessageBox.warning(
                self,
                tr(title_key, title_default),
                tr("robot_assets.dump_error_persist",
                   "Failed to persist discovered bodies for {fmt}").format(fmt=fmt),
            )
            return
        # "unexpected" or any other kind — log error and surface raw msg
        log_error(
            f"[robot_assets] dump {fmt} unexpected error sku={sku!r}: {msg}"
        )
        QMessageBox.warning(
            self,
            tr(title_key, title_default),
            f"Unexpected error: {msg}",
        )

    def _on_combo_changed(self, kind: str, path: str) -> None:
        if not path:
            return
        # Cloud URL markers (nucleus:..., http://...) aren't resolvable as
        # filesystem paths — don't persist them as the user-selected path
        # or resolve() will treat them as missing.
        if "://" in path or path.startswith(("nucleus:", "http:", "https:")):
            return
        self._svc.set_selected_path(self._record.brand, self._record.model, kind, path)

    def _on_add_path(self, kind: str) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            tr("robot_assets.pick_file", "Pick a {kind} file").format(kind=kind),
            "",
        )
        if chosen:
            self._svc.add_custom_path(
                self._record.brand, self._record.model, kind, chosen,
            )


class _ModelGroupCard(QWidget):
    """Card grouping a base + one or more variants under (brand, model_family).

    Renders a header with the family display name, a horizontal chip strip
    naming each variant (``variant_label``), and a body that hosts the
    currently-selected variant's :class:`_AssetCard`. Switching chip
    rebuilds the body in place — no QStackedWidget so each card pays one
    AssetCard's worth of memory.
    """

    def __init__(
        self,
        group: AssetRecordGroup,
        on_edit_request,
        on_delete_request,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._group = group
        self._records: List[AssetRecord] = list(group.records)
        self._on_edit_request = on_edit_request
        self._on_delete_request = on_delete_request
        self._active_sku: str = self._records[0].sku if self._records else ""

        self.setObjectName("modelGroupCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"#modelGroupCard {{ background: transparent;"
            f" border: 1px solid {Config.get_color('sidebar_card_active_border')};"
            f" border-radius: 4px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(2)

        # ----- header (family display name) -----
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 0)
        # ``records[0].name`` carries the base robot's display ("Unitree Go2");
        # we strip the variant tail ("Standard"/"Air") so the family header
        # reads "Unitree Go2" rather than "Unitree Go2 Standard".
        base = self._records[0] if self._records else None
        family_name = (base.name if base else group.model_family) or group.model_family
        title = setText(
            f"robot_assets.family.{group.brand}.{group.model_family}",
            default=family_name,
            kind="title", size=_ss(),
            color=Config.get_color("highlight"),
        )
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        title.setMinimumWidth(0)
        title.setToolTip(family_name)
        header.addWidget(title, 1)

        count_chip = QLabel(f"× {len(self._records)}")
        count_chip.setStyleSheet(
            f"color: {Config.get_color('sub_t2')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        header.addWidget(count_chip)
        outer.addLayout(header)

        # ----- variant chip strip -----
        self._chip_row = QHBoxLayout()
        self._chip_row.setContentsMargins(4, 0, 4, 4)
        self._chip_row.setSpacing(4)
        self._chips: Dict[str, QToolButton] = {}
        for rec in self._records:
            chip = QToolButton(self)
            chip.setText(rec.variant_label or "Standard")
            chip.setCheckable(True)
            chip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            chip.setToolTip(rec.name or rec.sku)
            chip.clicked.connect(
                lambda _c=False, sku=rec.sku: self._on_chip_clicked(sku)
            )
            self._chips[rec.sku] = chip
            self._chip_row.addWidget(chip)
        self._chip_row.addStretch(1)
        outer.addLayout(self._chip_row)
        self._apply_chip_styles()

        # ----- variant body host -----
        self._body_host = QWidget(self)
        body_layout = QVBoxLayout(self._body_host)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self._body_layout = body_layout
        outer.addWidget(self._body_host)

        self._rebuild_body()

    # ----- chip ------------------------------------------------------------

    def _chip_qss(self, *, active: bool) -> str:
        bg = (
            Config.get_color("highlight") if active
            else "transparent"
        )
        fg = (
            Config.get_color("bg_1") if active
            else Config.get_color("main_t2")
        )
        border = (
            Config.get_color("highlight") if active
            else Config.get_color("border_1")
        )
        return (
            f"QToolButton {{ background: {bg}; color: {fg};"
            f" border: 1px solid {border}; border-radius: 3px;"
            f" padding: 1px 8px; font-size: {_ss()}px; }}"
        )

    def _apply_chip_styles(self) -> None:
        for sku, chip in self._chips.items():
            active = (sku == self._active_sku)
            chip.setChecked(active)
            chip.setStyleSheet(self._chip_qss(active=active))

    def _on_chip_clicked(self, sku: str) -> None:
        if sku == self._active_sku:
            # keep selection (toggle-back not allowed)
            self._chips[sku].setChecked(True)
            return
        self._active_sku = sku
        self._apply_chip_styles()
        self._rebuild_body()

    # ----- body ------------------------------------------------------------

    def _rebuild_body(self) -> None:
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        rec = next((r for r in self._records if r.sku == self._active_sku), None)
        if rec is None:
            return
        card = _AssetCard(
            rec,
            on_edit_request=self._on_edit_request,
            on_delete_request=self._on_delete_request,
            parent=self._body_host,
        )
        self._body_layout.addWidget(card)


class RobotAssetsPanel(QWidget):
    """Top-level Robot Asset content."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._svc = get_robot_asset_service()
        self._pending_editor_queue: List[Dict[str, str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ----- header -----
        # NOTE: the leading "Registered robots" title was removed — the sidebar
        # group already carries the panel title, so the inline label was
        # redundant. A leading stretch keeps the action cluster (add /
        # Menagerie / refresh) right-aligned where it used to sit.
        header = QHBoxLayout()
        header.addStretch(1)
        self._btn_add = self._build_add_menu_button()
        header.addWidget(self._btn_add)

        # Width pinned to 110 px so the "M" head and "…" tail of "Menagerie…"
        # render without clipping in both EN and ZH (the ZH localisation also
        # keeps the literal "Menagerie…" word — see robot_assets.txt).
        self._btn_menagerie = setButton(
            "robot_assets.menagerie", 110, 24,
            kind="border", spec="none",
            default=tr("robot_assets.menagerie", "Menagerie…"),
        )
        i18n_bind(
            self._btn_menagerie, "setToolTip",
            "robot_assets.menagerie_tip",
            "Browse and download MuJoCo Menagerie packages",
        )
        self._btn_menagerie.clicked.connect(self._on_open_menagerie)
        header.addWidget(self._btn_menagerie)

        self._btn_refresh = setButton(
            "robot_assets.refresh", 24, 24,
            kind="light", spec="none",
            icon="icon_refresh", icon_only=True, default="",
        )
        i18n_bind(
            self._btn_refresh, "setToolTip",
            "robot_assets.refresh_tip", "Re-scan asset registry",
        )
        self._btn_refresh.clicked.connect(self._on_refresh)
        header.addWidget(self._btn_refresh)
        layout.addLayout(header)

        # ----- body -----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        # QScrollArea's viewport and inner content widget both default to the
        # palette Window colour. Without explicit transparency they paint a
        # solid fill over the sidebarPanel bg, leaving a visible "底色" around
        # the asset cards even when the cards themselves are transparent.
        viewport = scroll.viewport()
        if viewport is not None:
            viewport.setStyleSheet("background: transparent;")
        self._host = QWidget()
        self._host.setStyleSheet("background: transparent;")
        self._host_layout = QVBoxLayout(self._host)
        self._host_layout.setContentsMargins(0, 0, 0, 0)
        self._host_layout.setSpacing(6)
        scroll.setWidget(self._host)
        layout.addWidget(scroll, 1)

        self._svc.changed.connect(self._refresh)
        self._refresh()

    # ----- actions --------------------------------------------------------

    def _build_add_menu_button(self) -> QToolButton:
        btn = QToolButton(self)
        btn.setText("+")
        btn.setFixedSize(QSize(24, 24))
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setStyleSheet(
            f"QToolButton {{ background: transparent;"
            f" color: {Config.get_color('main_t2')};"
            f" border: 1px solid {Config.get_color('border_1')};"
            f" border-radius: 3px; font-size: {_ss() + 1}px; }}"
            f"QToolButton:hover {{ background: {Config.get_color('row_1')}; }}"
            f"QToolButton::menu-indicator {{ image: none; width: 0; }}"
        )
        i18n_bind(
            btn, "setToolTip",
            "robot_assets.add_tip", "Register a new robot or variant",
        )
        menu = QMenu(btn)
        act_new_model = QAction(self)
        i18n_bind(act_new_model, "setText", "robot_assets.add_robot", "New model")
        act_new_model.triggered.connect(self._on_add_robot_new_model)
        menu.addAction(act_new_model)
        act_new_variant = QAction(self)
        i18n_bind(act_new_variant, "setText", "robot_assets.add_variant", "New variant of …")
        act_new_variant.triggered.connect(self._on_add_robot_new_variant)
        menu.addAction(act_new_variant)
        btn.setMenu(menu)
        # Keep a reference so the menu+actions outlive the local scope.
        self._add_menu = menu
        return btn

    def _on_refresh(self) -> None:
        """Re-scan canonical roots AND surface unregistered packages.

        Stage 1: ``scan_and_merge_assets`` walks ``custom_mods/models/`` and
        fills in USD / URDF / XACRO slots on registered SKUs.

        Stage 2: ``scan_unregistered_packages`` walks the same roots looking
        for directories whose name does NOT resolve to any SKU (canonical
        or user-layer) AND that contain at least one MJCF/USD/URDF.
        Each hit is queued into ``_pending_editor_queue`` and the next-tick
        scheduler kicks open the first editor — same plumbing the menagerie
        download flow uses (see ``_on_menagerie_download_finished``). Users
        can dismiss a hit via the "Ignore" button on the editor to suppress
        it from future Refresh sweeps.
        """
        n = self._svc.scan_and_merge()
        log_debug(f"[robot_assets] scan_and_merge -> {n} entries")
        try:
            hints = self._svc.scan_unregistered_packages()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] scan_unregistered_packages failed: {exc}")
            hints = []
        if not hints:
            return
        log_info(
            f"[robot_assets] refresh detected {len(hints)} unregistered "
            f"package(s): "
            f"{', '.join(h.get('dir_name', '?') for h in hints)}"
        )
        was_idle = not self._pending_editor_queue
        self._pending_editor_queue.extend(hints)
        if was_idle:
            QTimer.singleShot(0, self._open_next_pending_editor)

    def _on_add_robot_new_model(self) -> None:
        dlg = RobotEditorDialog(existing=None, parent=self)
        dlg.exec()

    def _on_add_robot_new_variant(self) -> None:
        from registers import robots as _robots
        # Candidates = every registered robot that is NOT itself a variant.
        candidates: List = []
        for sku in _robots.list_skus():
            if _robots.is_variant(sku):
                continue
            entry = _robots.get_robot(sku) or {}
            label = entry.get("name") or f"{entry.get('brand', '')}.{entry.get('model', '')}"
            candidates.append((label, sku))
        if not candidates:
            QMessageBox.information(
                self,
                tr("robot_assets.add_variant", "New variant of …"),
                tr("robot_assets.add_variant_empty",
                   "No base robots registered yet. Add a model first."),
            )
            return
        candidates.sort(key=lambda kv: kv[0].lower())
        labels = [kv[0] for kv in candidates]
        chosen, ok = QInputDialog.getItem(
            self,
            tr("robot_assets.add_variant", "New variant of …"),
            tr("robot_assets.pick_parent",
               "Choose the base model this variant inherits from:"),
            labels, 0, False,
        )
        if not ok or not chosen:
            return
        parent_sku = next(sku for lab, sku in candidates if lab == chosen)
        dlg = RobotEditorDialog(existing=None, parent=self, parent_sku=parent_sku)
        dlg.exec()

    def _on_edit_robot(self, sku: str) -> None:
        from registers import robots as _robots
        # Round-trip via the raw (pre-inheritance) view so variants don't
        # accidentally persist inherited fields back into the user overlay.
        entry = _robots.get_raw_robot(sku) or _robots.get_robot(sku) or {}
        parent_sku = _robots.get_parent_sku(sku)
        dlg = RobotEditorDialog(
            existing=dict(entry, sku=sku),
            parent=self,
            parent_sku=parent_sku,
        )
        dlg.exec()

    def _on_delete_robot(self, sku: str) -> None:
        from registers.robots import CascadeProtectionError
        try:
            removed = self._svc.remove_user_robot(sku)
        except CascadeProtectionError as exc:
            QMessageBox.warning(
                self,
                tr("robot_assets.cascade_title", "Cannot delete"),
                tr("robot_assets.cascade_body",
                   "This robot has user variants that depend on it. "
                   "Delete or detach the variants first.\n\n{detail}")
                .replace("{detail}", str(exc)),
            )
            return
        if removed:
            log_info(f"[robot_assets] removed user robot sku={sku}")

    def _on_open_menagerie(self) -> None:
        dlg = MenagerieBrowserDialog(parent=self)
        dlg.download_finished.connect(self._on_menagerie_download_finished)
        dlg.exec()

    def _on_menagerie_download_finished(self, installed_names: List[str]) -> None:
        """Stage 4: smart-match each newly-installed package.

        Flow per package:
          1. Rescan registry + asset state so canonical entries whose MJCF
             path matches the new dir flip their status from remote/missing
             to local (no editor needed).
          2. ``try_match_menagerie_package`` — if a SKU resolves (canonical
             or user-layer), log and move on.
          3. Otherwise: queue a ``RobotEditorDialog`` pre-filled with brand /
             model / mjcf so the user can register it as a new model. The
             queue uses ``QTimer.singleShot`` so multiple unmatched packages
             open the dialog one-at-a-time instead of stacking modals.
        """
        try:
            self._svc.scan_and_merge()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] post-menagerie rescan failed: {exc}")

        names = [str(n) for n in (installed_names or []) if n]
        if not names:
            return

        unmatched: List[Dict[str, str]] = []
        for dir_name in names:
            matched = self._svc.try_match_menagerie_package(dir_name)
            if matched:
                log_info(f"[robot_assets] menagerie/{dir_name} -> {matched} (auto-attached)")
                continue
            hint = self._svc.menagerie_register_hint(dir_name)
            unmatched.append(hint)

        if not unmatched:
            return
        # Append to the panel's queue; if no editor is currently open, kick
        # the next-tick scheduler so the first dialog opens shortly.
        was_idle = not self._pending_editor_queue
        self._pending_editor_queue.extend(unmatched)
        if was_idle:
            QTimer.singleShot(0, self._open_next_pending_editor)

    def _open_next_pending_editor(self) -> None:
        if not self._pending_editor_queue:
            return
        hint = self._pending_editor_queue.pop(0)
        existing_seed = {
            "brand": hint.get("brand", ""),
            "model": hint.get("model", ""),
            "name": hint.get("name", ""),
            "assets": {"MJCF": hint.get("mjcf", "")},
        }
        # We use existing= with sku missing so RobotEditorDialog treats it as
        # edit-mode display-only for brand/model (disabled). The user can
        # still edit name/families/adapter/joints; on Save the dialog calls
        # add_user_robot (not update) because saved_sku derives from build_sku
        # and the entry won't already be in the registry.
        # Simpler path: open in create-mode and pre-fill via setText.
        dlg = RobotEditorDialog(existing=None, parent=self)
        try:
            dlg._brand.setText(existing_seed["brand"])
            dlg._model.setText(existing_seed["model"])
            dlg._name.setText(existing_seed["name"])
            mjcf_row = dlg._path_rows.get("MJCF")
            if mjcf_row is not None:
                mjcf_row._edit.setText(existing_seed["assets"]["MJCF"])
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[robot_assets] menagerie editor prefill failed: {exc}")
        # "Ignore this package" — only relevant when we have the source
        # dir_name in the hint (Refresh path adds it; the menagerie download
        # path doesn't, since you wouldn't want to ignore a just-downloaded
        # robot). The button writes the dir to state.json's ignore list via
        # the service, so subsequent Refresh sweeps skip it.
        dir_name = str(hint.get("dir_name") or "").strip()
        if dir_name:
            try:
                btn_ignore = setButton(
                    "robot_assets.ignore_package", 100, 26,
                    kind="border", spec="none",
                    default=tr("robot_assets.ignore_package", "Ignore"),
                )
                i18n_bind(
                    btn_ignore, "setToolTip",
                    "robot_assets.ignore_package_tip",
                    "Skip this directory on future Refresh sweeps",
                )

                def _on_ignore(_c: bool = False, _name: str = dir_name, _dlg=dlg) -> None:
                    self._svc.add_ignored_package(_name)
                    log_info(f"[robot_assets] ignoring future Refresh hits for {_name!r}")
                    _dlg.reject()

                btn_ignore.clicked.connect(_on_ignore)
                # The bottom button row of RobotEditorDialog is the last
                # QHBoxLayout under self.layout(); insert the Ignore button
                # at index 1 (after the stretch, before Cancel/Save).
                outer_layout = dlg.layout()
                btn_row_layout = None
                for i in range(outer_layout.count()):
                    sub = outer_layout.itemAt(i)
                    if sub is not None and sub.layout() is not None:
                        # Heuristic: the buttons row is the only QHBoxLayout
                        # at the very bottom of the dialog.
                        btn_row_layout = sub.layout()
                if btn_row_layout is not None:
                    btn_row_layout.insertWidget(1, btn_ignore)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[robot_assets] ignore-button install failed: {exc}")
        dlg.finished.connect(lambda _r: QTimer.singleShot(0, self._open_next_pending_editor))
        dlg.show()

    @pyqtSlot()
    def _refresh(self) -> None:
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        groups = self._svc.list_assets_grouped()
        records_flat: List[AssetRecord] = [r for g in groups for r in g.records]
        if not records_flat:
            placeholder = setText(
                "robot_assets.empty_state",
                default=(
                    "No robots registered yet. Click + to add a custom robot, "
                    "or drop a brand pack — entries will appear here automatically."
                ),
                kind="content", size=_ss(),
            )
            placeholder.setWordWrap(True)
            self._host_layout.addWidget(placeholder)
        else:
            for group in groups:
                if len(group.records) <= 1:
                    self._host_layout.addWidget(_AssetCard(
                        group.records[0],
                        on_edit_request=self._on_edit_robot,
                        on_delete_request=self._on_delete_robot,
                        parent=self._host,
                    ))
                else:
                    self._host_layout.addWidget(_ModelGroupCard(
                        group,
                        on_edit_request=self._on_edit_robot,
                        on_delete_request=self._on_delete_robot,
                        parent=self._host,
                    ))
        self._host_layout.addStretch(1)


__all__ = ["RobotAssetsPanel"]
