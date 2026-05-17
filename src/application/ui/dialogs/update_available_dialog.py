"""UpdateAvailableDialog — show release notes + 3-button footer.

Buttons:
- ``Skip this version`` — persists the version slug to
  ``user.ini[App].update_skipped_version`` (the sidebar still shows the
  "Update" state because the cached available_version is still set; the
  startup check just suppresses the success log on re-detection).
- ``Later`` — close and do nothing.
- ``Apply Update`` — accept() with the release on the dialog; the
  MainWindow handler then submits ``ApplyUpdateTask`` and opens the
  progress dialog.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.updater import ReleaseInfo, get_update_service


class UpdateAvailableDialog(QDialog):
    """Three-button dialog: Skip / Later / Apply."""

    # Emitted when the user clicks Apply. Listeners (MainWindow) launch
    # the apply task. The dialog closes via accept() before the signal
    # fires so the apply-progress modal can open on top cleanly.
    apply_requested = pyqtSignal(object)   # ReleaseInfo

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
        self.resize(560, 460)
        i18n_bind(
            self, "setWindowTitle",
            "updater.dialog.title_available", "Update available",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        # Header: vX.Y.Z -> vA.B.C
        header = QLabel(
            f"v{self._current_version}  →  v{self._release.version}",
            self,
        )
        header.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('highlight')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(header)

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

        # Release-notes body. Render as plain text — release.body is
        # raw Markdown and we have no markdown lib dep; plain text is
        # already legible and avoids HTML-injection surface.
        notes = QTextEdit(self)
        notes.setReadOnly(True)
        body_text = (self._release.body or "").strip() or tr(
            "updater.label.no_release_notes",
            "(no release notes for this version)",
        )
        notes.setPlainText(body_text)
        notes.setStyleSheet(
            f"QTextEdit {{ background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 8px; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(notes, 1)

        # Footer.
        footer = QHBoxLayout()
        footer.setSpacing(8)

        skip_btn = QPushButton(self)
        i18n_bind(skip_btn, "setText", "updater.btn.skip", "Skip this version")
        skip_btn.setStyleSheet(self._button_qss("muted"))
        skip_btn.clicked.connect(self._on_skip)
        footer.addWidget(skip_btn)

        footer.addStretch(1)

        later_btn = QPushButton(self)
        i18n_bind(later_btn, "setText", "updater.btn.later", "Later")
        later_btn.setStyleSheet(self._button_qss("muted"))
        later_btn.clicked.connect(self.reject)
        footer.addWidget(later_btn)

        apply_btn = QPushButton(self)
        i18n_bind(apply_btn, "setText", "updater.btn.apply", "Apply update")
        apply_btn.setStyleSheet(self._button_qss("primary"))
        apply_btn.clicked.connect(self._on_apply)
        footer.addWidget(apply_btn)

        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )

    # ----- handlers -----
    def _on_skip(self) -> None:
        get_update_service().skip_version(self._release.version)
        self.reject()

    def _on_apply(self) -> None:
        self.apply_requested.emit(self._release)
        self.accept()

    @staticmethod
    def _button_qss(kind: str) -> str:
        if kind == "primary":
            return (
                f"QPushButton {{ "
                f"background-color: {Config.get_color('highlight')}; "
                f"color: {Config.get_color('bg_1')}; "
                f"border: none; border-radius: 4px; "
                f"padding: 7px 16px; font-weight: bold; "
                f"font-size: {Config.get_font_size('size_small')}pt; }} "
                f"QPushButton:hover {{ "
                f"background-color: {Config.get_color('highlight_checked')}; }}"
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
