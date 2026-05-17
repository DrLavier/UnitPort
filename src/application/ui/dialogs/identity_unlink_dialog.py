"""IdentityUnlinkDialog — confirm removal of a linked OAuth identity.

Triggered from UserPanel when the user clicks a Google/GitHub status button
that is currently in the "linked" state. The dialog renders one of three
body texts depending on context:

    - ``is_only_identity``  : refuse — keep the button enabled but only show
      a Close button so the user gets visible feedback.
    - ``is_current_provider``: warn that unlinking the active session
      provider will sign them out.
    - otherwise              : a plain "you'll no longer be able to sign in
      with X" confirmation.

The Supabase server enforces the "can't delete the only identity" rule too,
so the client-side check is a UX nicety, not a security check.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.auth import get_auth_manager


_PROVIDER_DISPLAY = {
    "email": "Email",
    "google": "Google",
    "github": "GitHub",
}


class IdentityUnlinkDialog(QDialog):
    """Modal confirm. Caller passes the contextual flags so this dialog
    doesn't need to re-query AuthManager for state that the caller
    already had at click time."""

    def __init__(
        self,
        *,
        provider: str,
        is_only_identity: bool,
        is_current_provider: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._auth = get_auth_manager()
        self._provider = provider
        self._is_only = is_only_identity
        self._is_current = is_current_provider

        display = _PROVIDER_DISPLAY.get(provider, provider.capitalize())
        title_tpl = tr(
            "user.linked_account_unlink_confirm_title", "Unlink {provider}?",
        )
        self.setWindowTitle(title_tpl.format(provider=display))
        self.setModal(True)
        self.setMinimumWidth(380)

        self._build_ui(display)
        self._apply_styles()

    def _build_ui(self, display: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(12)

        if self._is_only:
            body_tpl = tr(
                "user.linked_account_unlink_only_body",
                "This is your only sign-in method. You can't unlink it.",
            )
        elif self._is_current:
            body_tpl = tr(
                "user.linked_account_unlink_current_body",
                "You'll be signed out after unlinking, because this is your "
                "current sign-in method.",
            )
        else:
            body_tpl = tr(
                "user.linked_account_unlink_normal_body",
                "You'll no longer be able to sign in with {provider}.",
            )
        body = QLabel(body_tpl.format(provider=display))
        body.setWordWrap(True)
        body.setObjectName("authStatusLabel")
        root.addWidget(body)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        if self._is_only:
            self._btn_close = QPushButton(tr("auth.close_button", "Close"))
            self._btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_close.clicked.connect(self.reject)
            btn_row.addWidget(self._btn_close)
        else:
            self._btn_cancel = QPushButton(tr("auth.close_button", "Cancel"))
            self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_cancel.clicked.connect(self.reject)
            btn_row.addWidget(self._btn_cancel)

            unlink_tpl = tr(
                "user.linked_account_unlink_button", "Unlink {provider}",
            )
            self._btn_unlink = QPushButton(unlink_tpl.format(provider=display))
            self._btn_unlink.setObjectName("authDangerButton")
            self._btn_unlink.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_unlink.clicked.connect(self._on_confirm)
            btn_row.addWidget(self._btn_unlink)

        root.addLayout(btn_row)

    def _apply_styles(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        secondary = Config.get_color("main_c2")
        accent = Config.get_color("safe_zone")
        danger = Config.get_color("danger_zone")
        danger_hover = Config.get_color("danger_zone_hover")
        primary_text = Config.get_color("auth_primary_button_text")

        self.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
            f"QPushButton {{"
            f"  background: {bg}; color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px; padding: 6px 14px;"
            f"}}"
            f"QPushButton:hover {{ border-color: {accent}; }}"
            f"QPushButton#authDangerButton {{"
            f"  background: {danger}; color: {primary_text};"
            f"  border: none; font-weight: 600;"
            f"}}"
            f"QPushButton#authDangerButton:hover {{ background: {danger_hover}; }}"
            f"QLabel#authStatusLabel {{ color: {secondary}; font-size: 12px; }}"
        )

    def _on_confirm(self) -> None:
        self._auth.unlink_identity_by_provider(self._provider)
        self.accept()
