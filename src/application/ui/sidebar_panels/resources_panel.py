# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ResourcesPanel — sidebar > Resources view.

A mature asset browser over four categories that mirror the managed
``custom_mods/`` hierarchy plus the registry-backed robot models:

* **Motion** — downloaded datasets + community AMP packs. Each Motion card
  expands into a :class:`MotionClipTable` listing the clips inside the
  package, with a per-clip crop/label button.
* **Policy** — downloaded bundles + bundles dropped into ``custom_mods/
  policies/`` without the downloader.
* **Model** — robots from ``registers.robots``.
* **Canvas** — shipped/downloaded template canvases under
  ``custom_mods/canvas/``.

The panel renders against the :mod:`application.service.assets_browser`
seam (``PackageRow`` DTOs from ``get_asset_browser_provider()``), never the
concrete sources directly — so the backend round can swap clip enumeration,
unify motion discovery, and feed the Training Motion picker without touching
this widget.

Live download updates still arrive via ``AppSignals`` (``resource_added`` /
``resource_progress`` / ``resource_finished`` / ``resource_removed``);
cards are keyed by ``package_id`` which equals the ``ResourceEntry.id`` for
download-registry packages, so progress routing is unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QCursor, QFontMetrics
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    i18n_bind,
    log_debug,
    log_error,
    setButton,
    tr,
)

from application.service.assets_browser import (
    AssetCategory,
    PackageRow,
    get_asset_browser_provider,
)
from application.service.resources import (
    ResourceState,
    TransportKind,
    get_resource_manager,
)
from application.service.signals import get_app_signals
from application.ui.dialogs.add_resource_dialog import AddResourceDialog


def _ss() -> int:
    return int(Config.get_font_size("size_small"))


def _mini() -> int:
    return int(Config.get_font_size("size_mini"))


# ---------------------------------------------------------------------------
# Static maps (theme slots + i18n keyed by the resources-package enums)
# ---------------------------------------------------------------------------


_TRANSPORT_BADGE_SLOT = {
    TransportKind.GITHUB_CLONE: "resources_badge_github",
    TransportKind.GITHUB_RELEASE: "resources_badge_release",
    TransportKind.HUGGINGFACE: "resources_badge_huggingface",
}

_STATUS_SLOT = {
    ResourceState.LOCAL: "resources_status_local",
    ResourceState.DOWNLOADING: "resources_status_downloading",
    ResourceState.MISSING: "resources_status_missing",
    ResourceState.ERROR: "resources_status_error",
}

_TRANSPORT_I18N = {
    TransportKind.GITHUB_CLONE: ("resources.transport.github_clone", "GitHub"),
    TransportKind.GITHUB_RELEASE: ("resources.transport.github_release", "Release"),
    TransportKind.HUGGINGFACE: ("resources.transport.huggingface", "HuggingFace"),
}

_STATUS_I18N = {
    ResourceState.LOCAL: ("resources.status.local", "local"),
    ResourceState.DOWNLOADING: ("resources.status.downloading", "downloading"),
    ResourceState.MISSING: ("resources.status.missing", "missing"),
    ResourceState.ERROR: ("resources.status.error", "error"),
}

#: Category → section-heading i18n key.
_CATEGORY_I18N = {
    AssetCategory.MOTION: ("resources.kind.motion", "Motion"),
    AssetCategory.POLICY: ("resources.kind.policy", "Policy"),
    AssetCategory.MODEL: ("resources.kind.model", "Model"),
    AssetCategory.CANVAS: ("resources.kind.canvas", "Canvas"),
}

#: Category → empty-state hint i18n key.
_CATEGORY_EMPTY_I18N = {
    AssetCategory.MOTION: (
        "resources.empty_motion",
        "No motion datasets yet — click \"Download Online\" to add one",
    ),
    AssetCategory.POLICY: (
        "resources.empty_policy",
        "No policy bundles yet — click \"Download Online\" to add one",
    ),
    AssetCategory.MODEL: ("resources.empty_model", "No robot models registered"),
    AssetCategory.CANVAS: ("resources.empty_canvas", "No canvas templates found"),
}

#: Provenance-row key → label i18n key. Unknown keys fall back to a
#: title-cased label so a new provider field still renders sensibly.
_PROVENANCE_I18N = {
    "source": ("resources.card.source", "Source"),
    "revision": ("resources.card.revision", "Revision"),
    "subpath": ("resources.card.subpath", "Subpath"),
    "downloaded_at": ("resources.card.downloaded_at", "Downloaded"),
    "size": ("resources.card.size", "Size"),
    "package": ("resources.card.package", "Package"),
    "subdir": ("resources.card.subdir", "Subdir"),
    "clips": ("resources.card.clips", "Clips"),
    "sku": ("resources.card.sku", "SKU"),
    "brand": ("resources.card.brand", "Brand"),
    "model": ("resources.card.model", "Model"),
    "families": ("resources.card.families", "Families"),
    "backend": ("resources.card.backend", "Backend"),
    "path": ("resources.card.path", "Path"),
}


def _badge_label(text: str, color_hex: str, *, parent: Optional[QWidget] = None) -> QLabel:
    """Small rounded label with a coloured background — used for both the
    transport badge and the status pill on each card header."""
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(
        f"QLabel {{"
        f" background: {color_hex}; color: {Config.get_color('alt_t1')};"
        f" padding: 1px 6px; border-radius: 6px;"
        f" font-size: {_mini()}px; }}"
    )
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


class _ElidedLabel(QLabel):
    """Single-line label that elides overflow to its width.

    The narrow sidebar can't fit long provenance values (HuggingFace URLs,
    timestamps); they elide with "…" and the full text shows on hover.
    """

    def __init__(self, full_text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full = full_text or ""
        self.setToolTip(self._full)
        self.setText(self._full)
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        # CRITICAL: a non-wrapping QLabel's minimumSizeHint is the FULL text
        # width, and QSizePolicy.Ignored suppresses sizeHint but NOT
        # minimumSizeHint — so a long clip name / provenance value would force
        # its row (and the enclosing fixed-width sidebar scroll content) wider
        # than the 280-px panel, pushing the right-hand action button out of
        # view. Returning width 0 lets the label collapse and elide instead,
        # so the row never exceeds the sidebar and the button stays visible.
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, e):  # type: ignore[override]
        fm = QFontMetrics(self.font())
        self.setText(
            fm.elidedText(self._full, Qt.TextElideMode.ElideRight, max(0, self.width()))
        )
        super().resizeEvent(e)


class _CardHeader(QFrame):
    """Whole-row clickable header (mirror of robot_assets_panel._CardHeader)."""

    def __init__(self, on_toggle, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_toggle = on_toggle
        self.setObjectName("resourceCardHeader")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply_style(hover=False)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        s = super().minimumSizeHint()
        return QSize(0, s.height())

    def _apply_style(self, *, hover: bool) -> None:
        bg = Config.get_color("row_1") if hover else "transparent"
        self.setStyleSheet(
            f"#resourceCardHeader {{ background: {bg};"
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


class _ResourceCard(QWidget):
    """Collapsible card for one :class:`PackageRow`.

    Works for all four categories: the transport badge / progress bar /
    action buttons appear only for download-registry packages
    (``raw_entry_id`` set); a Motion package additionally embeds a
    :class:`MotionClipTable`, built lazily on first expand.
    """

    def __init__(
        self,
        row: PackageRow,
        *,
        on_remove,
        on_open_folder,
        on_classify,
        provider,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._row = row
        self._on_remove = on_remove
        self._on_open_folder = on_open_folder
        self._on_classify = on_classify
        self._provider = provider
        self._expanded = False
        self._clip_host_layout: Optional[QVBoxLayout] = None
        self._clip_table: Optional[QWidget] = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # ----- header -----
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

        self._title = QLabel(row.name)
        self._title.setStyleSheet(
            f"color: {Config.get_color('highlight')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._title.setMinimumWidth(0)
        self._title.setToolTip(row.name)
        header_row.addWidget(self._title, 1)

        # Transport badge — download packages only.
        tkind = self._transport_kind()
        if tkind is not None:
            t_key, t_default = _TRANSPORT_I18N[tkind]
            header_row.addWidget(_badge_label(
                tr(t_key, t_default),
                Config.get_color(_TRANSPORT_BADGE_SLOT[tkind]),
            ))

        # Status pill.
        state = self._state()
        s_key, s_default = _STATUS_I18N[state]
        header_row.addWidget(_badge_label(
            tr(s_key, s_default),
            Config.get_color(_STATUS_SLOT[state]),
        ))

        layout.addWidget(self._header)

        # ----- body -----
        self._body = QWidget(self)
        body_l = QVBoxLayout(self._body)
        body_l.setContentsMargins(8, 4, 4, 6)
        body_l.setSpacing(3)

        # Progress bar — only for download packages, visible while downloading.
        self._progress: Optional[QProgressBar] = None
        self._progress_label: Optional[QLabel] = None
        if row.raw_entry_id is not None:
            self._progress = QProgressBar()
            self._progress.setMinimum(0)
            self._progress.setMaximum(100)
            self._progress.setValue(0)
            self._progress.setFixedHeight(8)
            self._progress.setTextVisible(False)
            self._progress.setStyleSheet(
                f"QProgressBar {{ background: {Config.get_color('bg_2')};"
                f" border: 1px solid {Config.get_color('border_1')};"
                f" border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: "
                f"{Config.get_color('resources_status_downloading')};"
                f" border-radius: 3px; }}"
            )
            self._progress.setVisible(state == ResourceState.DOWNLOADING)
            body_l.addWidget(self._progress)

            self._progress_label = QLabel("")
            self._progress_label.setStyleSheet(
                f"color: {Config.get_color('sub_t1')};"
                f" background: transparent; font-size: {_mini()}px;"
            )
            self._progress_label.setVisible(state == ResourceState.DOWNLOADING)
            body_l.addWidget(self._progress_label)

        # Provenance detail rows.
        for key, value in row.provenance.items():
            body_l.addLayout(self._kv_row(key, value))

        # Error block.
        if state == ResourceState.ERROR and row.error:
            err = QLabel(row.error)
            err.setWordWrap(True)
            err.setStyleSheet(
                f"color: {Config.get_color('resources_status_error')};"
                f" background: transparent; font-size: {_mini()}px;"
            )
            body_l.addWidget(err)

        # Clip-table host (Motion only) — populated lazily on first expand.
        if row.clip_capable:
            clip_host = QWidget(self._body)
            self._clip_host_layout = QVBoxLayout(clip_host)
            self._clip_host_layout.setContentsMargins(0, 2, 0, 0)
            self._clip_host_layout.setSpacing(0)
            body_l.addWidget(clip_host)

        # Action buttons — stacked VERTICALLY, each full-width, so they always
        # fit the 280-px panel and none is ever clipped (never three on one
        # row). Capability-gated: Classify only for downloads that may need
        # manual classification; Open folder / Remove for ANY package with
        # files (downloads + install-bundled community packs alike) — registry
        # models have no files, shipped canvases aren't removable.
        if row.raw_entry_id is not None:
            self._add_action_button(
                body_l, "resources.classify", "Classify…", "none",
                self._on_classify,
            )
        if row.openable:
            self._add_action_button(
                body_l, "resources.open_folder", "Open folder", "none",
                self._on_open_folder,
            )
        if row.removable:
            self._add_action_button(
                body_l, "resources.remove", "Remove", "danger",
                self._on_remove,
            )

        self._body.setVisible(False)
        layout.addWidget(self._body)

    # ----- helpers ----

    def _transport_kind(self) -> Optional[TransportKind]:
        if not self._row.transport:
            return None
        try:
            return TransportKind(self._row.transport)
        except ValueError:
            return None

    def _state(self) -> ResourceState:
        try:
            return ResourceState(self._row.state)
        except ValueError:
            return ResourceState.LOCAL

    def _kv_row(self, key: str, value: str) -> QHBoxLayout:
        i18n_key, default_label = _PROVENANCE_I18N.get(
            key, (f"resources.card.{key}", key.replace("_", " ").title())
        )
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel()
        i18n_bind(lbl, "setText", i18n_key, default_label + ":")
        lbl.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        lbl.setFixedWidth(72)
        row.addWidget(lbl)
        val = _ElidedLabel(value or "")
        val.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        row.addWidget(val, 1)
        return row

    def _add_action_button(self, layout, i18n_key: str, default: str, spec: str, on_click) -> None:
        """Append a full-width action button (vertical stack — fits 280 px)."""
        btn = setButton(
            f"{i18n_key}.{self._row.package_id}", 0, 26,
            kind="border", spec=spec, default=tr(i18n_key, default),
        )
        # width=0 ⇒ setButton skips setFixedSize; lock height + let it expand to
        # the panel width so the label never overflows / clips.
        btn.setFixedHeight(26)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pid = self._row.package_id
        btn.clicked.connect(lambda _c=False, p=pid, fn=on_click: fn(p))
        layout.addWidget(btn)

    def _toggle_body(self) -> None:
        self._expanded = not self._expanded
        # Build the clip table on first expand (lazy — list_clips walks disk).
        if (
            self._expanded
            and self._row.clip_capable
            and self._clip_table is None
            and self._clip_host_layout is not None
        ):
            from application.ui.sidebar_panels.clip_table import MotionClipTable
            self._clip_table = MotionClipTable(
                self._row.package_id, provider=self._provider, parent=self._body
            )
            self._clip_host_layout.addWidget(self._clip_table)
        self._body.setVisible(self._expanded)
        self._glyph.setText("▼" if self._expanded else "▶")

    # ----- live updates ----

    @property
    def package_id(self) -> str:
        return self._row.package_id

    def apply_progress(self, fraction: float, line: str) -> None:
        if self._progress is None or self._progress_label is None:
            return
        if not self._progress.isVisible():
            self._progress.setVisible(True)
            self._progress_label.setVisible(True)
        self._progress.setValue(max(0, min(int(fraction * 100), 100)))
        self._progress_label.setText(line)


# ---------------------------------------------------------------------------
# Top-level panel
# ---------------------------------------------------------------------------


class ResourcesPanel(QWidget):
    """Sidebar > Resources content widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mgr = get_resource_manager()
        self._provider = get_asset_browser_provider()
        self._cards: Dict[str, _ResourceCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ----- header -----
        header = QHBoxLayout()

        self._btn_add = setButton(
            "resources.add", 0, 24,
            kind="normal", spec="save",
            default=tr("resources.add", "Download Online"),
        )
        i18n_bind(self._btn_add, "setToolTip", "resources.add_tip",
                  "Download a motion dataset or policy bundle")
        # kind="save" ⇒ QSS bolds the label, but sizeHint measures with the
        # widget's own (non-bold) font, clipping the text. Mirror the QSS font
        # so sizeHint matches what's painted (re-fits on language switch).
        add_font = self._btn_add.font()
        add_font.setBold(True)
        add_font.setPixelSize(_ss())
        add_font.setFamily(Config.get_value("Font", "family", "Microsoft YaHei"))
        self._btn_add.setFont(add_font)
        self._btn_add.clicked.connect(self._on_add_clicked)
        header.addWidget(self._btn_add)

        self._btn_refresh = setButton(
            "resources.refresh", 24, 24,
            kind="light", spec="none",
            icon="icon_refresh", icon_only=True, default="",
        )
        i18n_bind(self._btn_refresh, "setToolTip", "resources.refresh_tip",
                  "Re-scan custom_mods/")
        self._btn_refresh.clicked.connect(self._on_refresh_clicked)
        header.addWidget(self._btn_refresh)
        header.addStretch(1)

        layout.addLayout(header)

        # ----- scroll body -----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
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

        # ----- subscribe to global resource signals -----
        sig = get_app_signals()
        sig.resource_added.connect(self._on_resource_added)
        sig.resource_progress.connect(self._on_resource_progress)
        sig.resource_finished.connect(self._on_resource_finished)
        sig.resource_removed.connect(self._on_resource_removed)

        # Reconcile index ↔ disk on first paint, then render.
        self._mgr.refresh_scan_state()
        self._refresh()

    # ----- header actions ----

    def _on_add_clicked(self) -> None:
        dlg = AddResourceDialog(parent=self)
        dlg.exec()

    def _on_refresh_clicked(self) -> None:
        self._mgr.refresh_scan_state()
        self._refresh()

    # ----- rendering ----

    def _refresh(self) -> None:
        log_debug("[resources] refresh")
        for card in list(self._cards.values()):
            card.deleteLater()
        self._cards.clear()
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        for category in self._provider.list_categories():
            key, default = _CATEGORY_I18N.get(
                category, (f"resources.kind.{category.value}", category.value.title())
            )
            self._host_layout.addWidget(self._section_label(key, default))

            rows = self._provider.list_packages(category)
            if not rows:
                e_key, e_default = _CATEGORY_EMPTY_I18N.get(
                    category, ("resources.empty_generic", "Nothing here yet")
                )
                self._host_layout.addWidget(self._empty_hint(e_key, e_default))
                continue

            for row in rows:
                card = _ResourceCard(
                    row,
                    on_remove=self._handle_remove,
                    on_open_folder=self._handle_open_folder,
                    on_classify=self._handle_classify,
                    provider=self._provider,
                )
                self._host_layout.addWidget(card)
                self._cards[row.package_id] = card

        self._host_layout.addStretch(1)

    def _section_label(self, key: str, default: str) -> QLabel:
        lbl = QLabel()
        i18n_bind(lbl, "setText", key, default)
        lbl.setStyleSheet(
            f"color: {Config.get_color('main_c1')};"
            f" background: transparent; font-size: {_ss()}px;"
            f" padding-top: 4px;"
        )
        return lbl

    def _empty_hint(self, key: str, default: str) -> QLabel:
        lbl = QLabel()
        i18n_bind(lbl, "setText", key, default)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
            f" padding: 4px 8px;"
        )
        return lbl

    # ----- card callbacks ----

    def _handle_remove(self, package_id: str) -> None:
        disk = self._provider.package_folder(package_id)
        msg = tr("resources.remove_confirm", "Remove this resource and its files?")
        if disk is not None:
            msg += "\n" + str(disk)
        ans = QMessageBox.question(
            self,
            tr("resources.remove", "Remove"),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self._provider.remove_package(package_id)
        except (RuntimeError, OSError) as e:
            QMessageBox.critical(
                self, tr("resources.remove", "Remove"), f"{type(e).__name__}: {e}"
            )
            return
        self._refresh()

    def _handle_open_folder(self, package_id: str) -> None:
        disk = self._provider.package_folder(package_id)
        if disk is None or not disk.exists():
            QMessageBox.warning(
                self,
                tr("resources.open_folder", "Open folder"),
                f"{disk}\n\n{tr('resources.status.missing', 'missing')}",
            )
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(disk))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(disk)], check=False)
            else:
                subprocess.run(["xdg-open", str(disk)], check=False)
        except OSError as e:
            log_error(f"[resources] open_folder failed: {e}")

    def _handle_classify(self, package_id: str) -> None:
        from application.ui.dialogs.classify_asset_dialog import ClassifyAssetDialog
        dlg = ClassifyAssetDialog(package_id, parent=self)
        dlg.exec()

    # ----- signal slots ----

    @pyqtSlot(str)
    def _on_resource_added(self, entry_id: str) -> None:
        self._refresh()

    @pyqtSlot(str, float, str)
    def _on_resource_progress(self, entry_id: str, fraction: float, line: str) -> None:
        card = self._cards.get(entry_id)
        if card is not None:
            card.apply_progress(fraction, line)

    @pyqtSlot(str, bool, str)
    def _on_resource_finished(self, entry_id: str, ok: bool, msg: str) -> None:
        self._refresh()

    @pyqtSlot(str)
    def _on_resource_removed(self, entry_id: str) -> None:
        self._refresh()


__all__ = ["ResourcesPanel"]
