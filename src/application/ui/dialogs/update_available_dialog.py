# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UpdateAvailableDialog — release title + Markdown notes + 3-button footer.

Shown both at startup (auto, when a newer release is detected) and on the
sidebar Update-button click — the two paths share this one dialog.

Header shows the GitHub release title (``release.name``) with a
``safe_zone``-green ``✔ vX.Y.Z`` badge beneath it. The body renders the
release notes as Markdown via ``QTextBrowser.setMarkdown`` (Qt6 built-in —
no extra dependency).

Buttons, left → right:

- ``Update and restart`` — emits ``update_and_restart`` then accept(); the
  MainWindow handler applies the update and relaunches the app on success.
- ``Update on exit`` — emits ``update_on_exit`` then accept(); the handler
  arms ``UpdateService.arm_exit_apply`` so the apply runs when the user
  later closes the app. The app stays usable in the meantime.
- ``Skip this version`` — persists the slug to
  ``user.ini[App].update_skipped_version`` (suppresses the *auto* startup
  popup for this version; the sidebar button can still re-open it).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.updater import ReleaseInfo, get_update_service


class UpdateAvailableDialog(QDialog):
    """Three-button dialog: Update-and-restart / Update-on-exit / Skip."""

    # Emitted when the user clicks the matching button. Listeners
    # (MainWindow) drive the apply flow. The dialog accept()s before the
    # signal fires so the apply-progress modal can open cleanly on top.
    update_and_restart = pyqtSignal(object)   # ReleaseInfo
    update_on_exit = pyqtSignal(object)       # ReleaseInfo

    def __init__(
        self,
        release: ReleaseInfo,
        current_version: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._release = release
        self._current_version = current_version
        self.setModal(True)
        self.resize(560, 480)
        i18n_bind(
            self, "setWindowTitle",
            "updater.dialog.title_available", "Update available",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(10)

        # Release title (GitHub release ``name``; fall back to the tag).
        title_text = (self._release.name or "").strip() or f"v{self._release.version}"
        title = QLabel(title_text, self)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(title)

        # Version badge: ✔ vX.Y.Z highlighted in safe_zone green.
        badge = QLabel(f"✔ v{self._release.version}", self)
        badge.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('safe_zone')}; "
            f"font-size: {Config.get_font_size('size_normal')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(badge)

        # Channel hint (e.g. "via git checkout v...").
        channel = get_update_service().channel()
        channel_desc = channel.describe() if channel is not None else "—"
        channel_label = QLabel(
            f"{tr('updater.label.channel', 'Channel')}: {channel_desc}",
            self,
        )
        channel_label.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('sub_t2')}; "
            f"font-size: {Config.get_font_size('size_mini')}pt; }}"
        )
        root.addWidget(channel_label)

        # Release-notes header.
        notes_title = QLabel(
            tr("updater.label.release_notes", "Release notes"),
            self,
        )
        notes_title.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(notes_title)

        # Release-notes body. ``release.body`` is raw Markdown; render it
        # with Qt6's built-in QTextBrowser.setMarkdown (no extra dep).
        notes = QTextBrowser(self)
        notes.setOpenExternalLinks(True)
        body_md = (self._release.body or "").strip()
        if body_md:
            notes.setMarkdown(body_md)
        else:
            notes.setPlainText(
                tr(
                    "updater.label.no_release_notes",
                    "(no release notes for this version)",
                )
            )
        notes.setStyleSheet(
            f"QTextBrowser {{ background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 8px; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(notes, 1)

        # Footer — left → right: Update-and-restart / Update-on-exit / Skip.
        footer = QHBoxLayout()
        footer.setSpacing(8)

        restart_btn = QPushButton(self)
        i18n_bind(
            restart_btn, "setText",
            "updater.btn.update_restart", "Update and restart",
        )
        restart_btn.setStyleSheet(self._button_qss("primary"))
        restart_btn.clicked.connect(self._on_update_restart)
        footer.addWidget(restart_btn)

        on_exit_btn = QPushButton(self)
        i18n_bind(
            on_exit_btn, "setText",
            "updater.btn.update_on_exit", "Update on exit",
        )
        on_exit_btn.setStyleSheet(self._button_qss("muted"))
        on_exit_btn.clicked.connect(self._on_update_on_exit)
        footer.addWidget(on_exit_btn)

        footer.addStretch(1)

        skip_btn = QPushButton(self)
        i18n_bind(skip_btn, "setText", "updater.btn.skip", "Skip this version")
        skip_btn.setStyleSheet(self._button_qss("muted"))
        skip_btn.clicked.connect(self._on_skip)
        footer.addWidget(skip_btn)

        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    # ----- handlers -----
    def _on_update_restart(self) -> None:
        self.update_and_restart.emit(self._release)
        self.accept()

    def _on_update_on_exit(self) -> None:
        self.update_on_exit.emit(self._release)
        self.accept()

    def _on_skip(self) -> None:
        get_update_service().skip_version(self._release.version)
        self.reject()

    @staticmethod
    def _button_qss(kind: str) -> str:
        if kind == "primary":
            return (
                f"QPushButton {{ "
                f"background-color: {Config.get_color('safe_zone')}; "
                f"color: {Config.get_color('bg_1')}; "
                f"border: none; border-radius: 4px; "
                f"padding: 7px 16px; font-weight: bold; "
                f"font-size: {Config.get_font_size('size_small')}pt; }} "
                f"QPushButton:hover {{ "
                f"background-color: {Config.get_color('safe_zone_hover')}; }}"
            )
        return (
            f"QPushButton {{ "
            f"background-color: {Config.get_color('btn_1')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 6px 14px; "
            f"font-size: {Config.get_font_size('size_small')}pt; }} "
            f"QPushButton:hover {{ background-color: {Config.get_color('hover_1')}; }}"
        )


__all__ = ["UpdateAvailableDialog"]
