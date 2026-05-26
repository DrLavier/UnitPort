# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""VersionNoticeDialog — one-time post-update "what's new" notice.

Shown once, on the first launch after the running app version changes. The
body is the project-root ``CHANGELOG.md`` with its license header stripped
(see ``application.service.updater.local_notice``); the version shown is the
running app version (``system.ini[System].version``). Renders the body as
Markdown via Qt6's built-in ``QTextBrowser.setMarkdown`` — no extra dep.

Purely informational: a single "Got it" button. The caller
(``main.py:_maybe_show_version_notice``) records the shown version in
``user.ini[App].info_notice_seen_version`` so it never re-pops for the
same version.
"""

from __future__ import annotations

from typing import Optional

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


class VersionNoticeDialog(QDialog):
    """Single-button Markdown notice for a freshly-applied version."""

    def __init__(
        self,
        version: str,
        body_markdown: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._version = version
        self._body = body_markdown
        self.setModal(True)
        self.resize(560, 460)
        i18n_bind(
            self, "setWindowTitle",
            "updater.notice.title", "What's new",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(10)

        # Version badge: ✔ vX.Y.Z highlighted in safe_zone green.
        badge = QLabel(f"✔ v{self._version}", self)
        badge.setStyleSheet(
            f"QLabel {{ color: {Config.get_color('safe_zone')}; "
            f"font-size: {Config.get_font_size('size_large')}pt; "
            f"font-weight: bold; }}"
        )
        root.addWidget(badge)

        # Notice body — Markdown via QTextBrowser.setMarkdown (Qt6 built-in).
        notes = QTextBrowser(self)
        notes.setOpenExternalLinks(True)
        notes.setMarkdown(self._body or "")
        notes.setStyleSheet(
            f"QTextBrowser {{ background-color: {Config.get_color('bg_2')}; "
            f"color: {Config.get_color('main_t1')}; "
            f"border: 1px solid {Config.get_color('border_1')}; "
            f"border-radius: 4px; padding: 8px; "
            f"font-size: {Config.get_font_size('size_small')}pt; }}"
        )
        root.addWidget(notes, 1)

        # Footer — single "Got it" button.
        footer = QHBoxLayout()
        footer.addStretch(1)
        ok_btn = QPushButton(self)
        i18n_bind(ok_btn, "setText", "updater.notice.ok", "Got it")
        ok_btn.setMinimumWidth(120)
        ok_btn.setStyleSheet(
            f"QPushButton {{ "
            f"background-color: {Config.get_color('safe_zone')}; "
            f"color: {Config.get_color('bg_1')}; "
            f"border: none; border-radius: 4px; "
            f"padding: 7px 16px; font-weight: bold; "
            f"font-size: {Config.get_font_size('size_small')}pt; }} "
            f"QPushButton:hover {{ "
            f"background-color: {Config.get_color('safe_zone_hover')}; }}"
        )
        ok_btn.clicked.connect(self.accept)
        footer.addWidget(ok_btn)
        root.addLayout(footer)

        self.setStyleSheet(
            f"QDialog {{ background-color: {Config.get_color('bg_1')}; }}"
        )


__all__ = ["VersionNoticeDialog"]
