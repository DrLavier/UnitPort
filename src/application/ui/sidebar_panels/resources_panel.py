# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ResourcesPanel — sidebar > Resources view.

Modeled on :class:`robot_assets_panel.RobotAssetsPanel` (same card-with-
header pattern, same scroll-area transparency tricks). Two sections,
Motion and Policy, each a stacked list of :class:`_ResourceCard`.

The panel reads its content from :class:`ResourceManager`:

* Indexed entries (``index.list_*``) drive most of the cards — they
  carry the source URL / revision / state.
* For policy cards, we additionally cross-reference :class:`PolicyPackScanner`
  results so a bundle that was dropped into ``custom_mods/policies/``
  without going through the downloader still shows up (as "imported",
  status=local, no source URL).

The panel subscribes to ``AppSignals`` for live updates:

* ``resource_added`` -> insert a card
* ``resource_progress`` -> tick the card's progress bar
* ``resource_finished`` -> flip the status pill and refresh disk size
* ``resource_removed`` -> drop the card
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    i18n_bind,
    log_debug,
    log_error,
    setButton,
    setText,
    tr,
)

from application.service.resources import (
    PolicyPack,
    ResourceEntry,
    ResourceKind,
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
# Sub-widgets
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
    """Collapsible card for one downloaded asset.

    The card is driven primarily by a :class:`ResourceEntry`; for
    on-disk-only policy bundles (no index entry) we synthesise a
    pseudo-entry with ``state=local`` and an empty source URL so the
    header still renders consistently.
    """

    def __init__(
        self,
        entry: ResourceEntry,
        *,
        on_remove,
        on_open_folder,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._entry = entry
        self._on_remove = on_remove
        self._on_open_folder = on_open_folder
        self._expanded = False

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

        self._title = QLabel(entry.name or entry.id[:8])
        self._title.setStyleSheet(
            f"color: {Config.get_color('highlight')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._title.setMinimumWidth(0)
        self._title.setToolTip(entry.name or entry.id)
        header_row.addWidget(self._title, 1)

        # Transport badge.
        t_key, t_default = _TRANSPORT_I18N[entry.transport]
        self._transport_badge = _badge_label(
            tr(t_key, t_default),
            Config.get_color(_TRANSPORT_BADGE_SLOT[entry.transport]),
        )
        header_row.addWidget(self._transport_badge)

        # Status pill.
        s_key, s_default = _STATUS_I18N[entry.state]
        self._status_badge = _badge_label(
            tr(s_key, s_default),
            Config.get_color(_STATUS_SLOT[entry.state]),
        )
        header_row.addWidget(self._status_badge)

        layout.addWidget(self._header)

        # ----- body -----
        self._body = QWidget(self)
        body_l = QVBoxLayout(self._body)
        body_l.setContentsMargins(8, 4, 4, 6)
        body_l.setSpacing(3)

        # Progress bar — only visible while downloading.
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
        self._progress.setVisible(entry.state == ResourceState.DOWNLOADING)
        body_l.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        self._progress_label.setVisible(entry.state == ResourceState.DOWNLOADING)
        body_l.addWidget(self._progress_label)

        # Detail rows.
        body_l.addLayout(self._kv_row(
            "resources.card.source", "Source",
            entry.source.url or "(local import)",
        ))
        if entry.source.revision:
            body_l.addLayout(self._kv_row(
                "resources.card.revision", "Revision",
                entry.source.revision,
            ))
        if entry.source.subpath:
            body_l.addLayout(self._kv_row(
                "resources.card.subpath", "Subpath",
                entry.source.subpath,
            ))
        if entry.downloaded_at:
            body_l.addLayout(self._kv_row(
                "resources.card.downloaded_at", "Downloaded",
                entry.downloaded_at,
            ))
        if entry.size_bytes:
            mib = entry.size_bytes // (1024 * 1024)
            body_l.addLayout(self._kv_row(
                "resources.card.size", "Size",
                f"{mib} MiB",
            ))

        if entry.state == ResourceState.ERROR and entry.error:
            err = QLabel(entry.error)
            err.setWordWrap(True)
            err.setStyleSheet(
                f"color: {Config.get_color('resources_status_error')};"
                f" background: transparent; font-size: {_mini()}px;"
            )
            body_l.addWidget(err)

        # Action buttons.
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addStretch(1)

        open_btn = setButton(
            f"resources.open_folder.{entry.id}", 0, 22,
            kind="border", spec="none",
            default=tr("resources.open_folder", "Open folder"),
        )
        open_btn.clicked.connect(lambda _c=False, eid=entry.id: self._on_open_folder(eid))
        actions.addWidget(open_btn)

        remove_btn = setButton(
            f"resources.remove.{entry.id}", 0, 22,
            kind="border", spec="none",
            default=tr("resources.remove", "Remove"),
        )
        remove_btn.clicked.connect(lambda _c=False, eid=entry.id: self._on_remove(eid))
        actions.addWidget(remove_btn)

        body_l.addLayout(actions)

        self._body.setVisible(False)
        layout.addWidget(self._body)

    # ----- helpers ----

    def _kv_row(self, key: str, default_label: str, value: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel()
        i18n_bind(lbl, "setText", key, default_label + ":")
        lbl.setStyleSheet(
            f"color: {Config.get_color('sub_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        lbl.setFixedWidth(72)
        row.addWidget(lbl)
        val = QLabel(value or "")
        val.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_mini()}px;"
        )
        val.setWordWrap(True)
        val.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(val, 1)
        return row

    def _toggle_body(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._glyph.setText("▼" if self._expanded else "▶")

    # ----- live updates ----

    @property
    def entry_id(self) -> str:
        return self._entry.id

    def apply_progress(self, fraction: float, line: str) -> None:
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
        self._cards: Dict[str, _ResourceCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ----- header -----
        header = QHBoxLayout()
        header.addWidget(setText(
            "resources.section", default="Third-party resources",
            kind="title", size=_ss(),
        ), 1)

        self._btn_add = setButton(
            "resources.add", 0, 24,
            kind="normal", spec="save",
            default=tr("resources.add", "Download Online"),
        )
        i18n_bind(self._btn_add, "setToolTip", "resources.add_tip",
                  "Download a motion dataset or policy bundle")
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
        # Clear existing cards.
        for card in list(self._cards.values()):
            card.deleteLater()
        self._cards.clear()
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        motions = self._mgr.list_motion_entries()
        policies = self._mgr.list_policy_entries()

        # Section: Motion
        self._host_layout.addWidget(self._section_label(
            "resources.kind.motion", "Motion",
        ))
        if not motions:
            self._host_layout.addWidget(self._empty_hint(
                "resources.empty_motion",
                "No motion datasets yet — click \"Add resource…\" to download one",
            ))
        else:
            for e in motions:
                card = _ResourceCard(
                    e,
                    on_remove=self._handle_remove,
                    on_open_folder=self._handle_open_folder,
                )
                self._host_layout.addWidget(card)
                self._cards[e.id] = card

        # Section: Policy (entries + unindexed bundles)
        self._host_layout.addWidget(self._section_label(
            "resources.kind.policy", "Policy",
        ))
        indexed_paths = {Path(e.local_path).as_posix() for e in policies}
        unindexed_packs: List[PolicyPack] = []
        for pack in self._mgr.scan_policy_packs():
            rel = f"policies/{pack.pack_id}"
            if rel not in indexed_paths:
                unindexed_packs.append(pack)

        if not policies and not unindexed_packs:
            self._host_layout.addWidget(self._empty_hint(
                "resources.empty_policy",
                "No policy bundles yet — click \"Add resource…\" to download one",
            ))
        else:
            for e in policies:
                card = _ResourceCard(
                    e,
                    on_remove=self._handle_remove,
                    on_open_folder=self._handle_open_folder,
                )
                self._host_layout.addWidget(card)
                self._cards[e.id] = card
            for pack in unindexed_packs:
                self._host_layout.addWidget(self._unindexed_row(pack))

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

    def _unindexed_row(self, pack: PolicyPack) -> QWidget:
        row = QFrame()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(8, 2, 8, 2)
        rl.setSpacing(6)
        title = QLabel(pack.label())
        title.setStyleSheet(
            f"color: {Config.get_color('main_t1')};"
            f" background: transparent; font-size: {_ss()}px;"
        )
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        rl.addWidget(title, 1)
        rl.addWidget(_badge_label(
            tr("resources.unindexed", "imported"),
            Config.get_color("resources_status_local"),
        ))
        return row

    # ----- card callbacks ----

    def _handle_remove(self, entry_id: str) -> None:
        entry = self._mgr.index.get(entry_id)
        if entry is None:
            return
        disk = entry.absolute_path()
        msg = tr(
            "resources.remove",
            "Remove this resource and its files at\n{path}?",
        ).replace("{path}", str(disk))
        ans = QMessageBox.question(
            self,
            tr("resources.remove", "Remove"),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._mgr.remove_entry(entry_id, delete_files=True)

    def _handle_open_folder(self, entry_id: str) -> None:
        entry = self._mgr.index.get(entry_id)
        if entry is None:
            return
        disk = entry.absolute_path()
        if not disk.exists():
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
