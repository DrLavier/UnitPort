"""RobotAssetsPanel — registry-driven asset browser AND register editor.

Sidebar's Robot Asset panel is the **canonical entrypoint for editing the
registers.robots register**. It surfaces every (brand, model) known to
``registers.robots`` — both canonical (factory ``data/robots_canonical.json``)
and user-layer (``~/UnitPort/registers/robots_custom.json``) — and lets the
user add / edit / delete user-layer entries through :class:`RobotEditorDialog`.
Per-asset selection / family-tag / body-IR override state still lives in
``~/UnitPort/robot_assets/state.json`` via ``RobotAssetService``.

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

from typing import List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    LaviComboBox,
    TagInput,
    i18n_bind,
    log_debug,
    log_info,
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
    get_robot_asset_service,
)
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

        # Disabled action buttons (Stage C tooling). Stacked vertically to fit
        # inside the 280-px sidebar without overflow. "Download from HF" is
        # hidden until the HF transport lands — re-add the tuple to surface it.
        coming_soon = tr("robot_assets.coming_soon", "Coming soon")
        for key, default in (
            ("robot_assets.calibrate_xacro", "Calibrate XACRO"),
            ("robot_assets.convert_usd",     "MJCF to USD"),
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


class RobotAssetsPanel(QWidget):
    """Top-level Robot Asset content."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._svc = get_robot_asset_service()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ----- header -----
        header = QHBoxLayout()
        header.addWidget(setText(
            "robot_assets.section", default="Registered robots",
            kind="title", size=_ss(),
        ), 1)
        self._btn_add = setButton(
            "robot_assets.add_robot", 24, 24,
            kind="light", spec="none",
            default="+",
        )
        i18n_bind(
            self._btn_add, "setToolTip",
            "robot_assets.add_robot_tip", "Register a new robot",
        )
        self._btn_add.clicked.connect(self._on_add_robot)
        header.addWidget(self._btn_add)

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

    def _on_refresh(self) -> None:
        n = self._svc.scan_and_merge()
        log_debug(f"[robot_assets] scan_and_merge -> {n} entries")

    def _on_add_robot(self) -> None:
        dlg = RobotEditorDialog(existing=None, parent=self)
        dlg.exec()

    def _on_edit_robot(self, sku: str) -> None:
        from registers import robots as _robots
        entry = _robots.get_robot(sku) or {}
        dlg = RobotEditorDialog(existing=dict(entry, sku=sku), parent=self)
        dlg.exec()

    def _on_delete_robot(self, sku: str) -> None:
        if self._svc.remove_user_robot(sku):
            log_info(f"[robot_assets] removed user robot sku={sku}")

    @pyqtSlot()
    def _refresh(self) -> None:
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        records: List[AssetRecord] = self._svc.list_assets()
        if not records:
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
            for rec in records:
                self._host_layout.addWidget(_AssetCard(
                    rec,
                    on_edit_request=self._on_edit_robot,
                    on_delete_request=self._on_delete_robot,
                ))
        self._host_layout.addStretch(1)


__all__ = ["RobotAssetsPanel"]
