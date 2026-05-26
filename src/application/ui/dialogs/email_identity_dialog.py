# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""EmailIdentityDialog — start the email-verification flow for sign-in password.

Triggered by clicking the [Email] status button in UserPanel's
linked-accounts row.

Security model
--------------
The dialog **does not** mutate the password directly inside the app —
password changes are high-risk and must go through Supabase's
email-verification round-trip. The dialog only does one thing: ask
Supabase to send a recovery link to the user's primary email.

Flow
----
1. Show the user's primary email (read-only) — that's where the link
   will land.
2. User clicks "Send link" → ``auth.send_password_reset(primary_email)``
3. Supabase emits ``info_message`` ("Password reset email sent to X...").
4. We optimistically mark the email/password capability as available
   via ``auth.mark_email_password_pending()`` so the [Email] chip in
   UserPanel reflects what the user just set in motion. Server-side
   truth eventually catches up via ``app_metadata.providers`` on the
   next ``refresh_identities`` call.
5. Show "Email sent — check your inbox" and close after a beat.

The user finishes the setup outside the app by clicking the verification
link, which lands on Supabase's hosted password-set page. Next time
they sign in via email + password (or the session refreshes), the
backend reflects the new capability and our optimistic flag remains
consistent.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.auth import get_auth_manager


class EmailIdentityDialog(QDialog):
    """Single-purpose: dispatch a password-reset email to the primary address."""

    def __init__(
        self,
        *,
        email_identity: Optional[dict] = None,   # kept for caller compatibility
        primary_email: str = "",
        available_emails: Optional[list] = None,  # accepted but unused (legacy)
        on_workspace_switch=None,                 # accepted but unused (legacy)
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        del email_identity, available_emails, on_workspace_switch  # legacy params
        self._auth = get_auth_manager()
        self._busy = False
        self._primary_email = (primary_email or "").strip()

        i18n_bind(self, "setWindowTitle",
                  "user.email_identity_title", "Email sign-in")
        self.setModal(True)
        self.setMinimumWidth(440)

        self._build_ui()
        self._apply_styles()
        self._wire_signals()

    # ----- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(12)

        # Primary email — read-only display, formatted as a "to:" line.
        target_row = QVBoxLayout()
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(2)
        target_row.addWidget(QLabel(tr(
            "user.email_identity_send_to",
            "We'll send a verification link to:",
        )))
        target_value = QLabel(
            self._primary_email or tr(
                "user.email_identity_no_email", "(no email available)",
            )
        )
        target_value.setObjectName("authPrimaryEmail")
        target_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        target_row.addWidget(target_value)
        root.addLayout(target_row)

        # Explanation — why we don't let the user change the password
        # in-app, what to do next.
        hint = QLabel(tr(
            "user.email_identity_security_hint",
            "Click the link in your inbox to set or update your sign-in "
            "password. For security, password changes can't be made "
            "inside the app — all changes go through email verification.",
        ))
        hint.setWordWrap(True)
        hint.setObjectName("authStatusLabel")
        root.addWidget(hint)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        self._status.setObjectName("authStatusLabel")
        self._status.setMinimumHeight(28)
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_close = QPushButton(tr("auth.close_button", "Close"))
        self._btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self._btn_close)

        self._btn_send = QPushButton(tr(
            "user.email_identity_send_button", "Send link",
        ))
        self._btn_send.setObjectName("authPrimaryButton")
        self._btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_send.setMinimumHeight(32)
        if not self._primary_email:
            self._btn_send.setEnabled(False)
        btn_row.addWidget(self._btn_send)

        root.addLayout(btn_row)

    def _apply_styles(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        secondary = Config.get_color("main_c2")
        accent = Config.get_color("safe_zone")
        primary_text = Config.get_color("auth_primary_button_text")
        primary_hover = Config.get_color("auth_primary_button_hover_bg")
        disabled_bg = Config.get_color("auth_button_disabled_bg")
        error_color = Config.get_color("auth_status_error_color")

        self.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
            f"QLabel#authPrimaryEmail {{"
            f"  color: {fg}; font-weight: 600; font-size: 14px;"
            f"  padding: 4px 0;"
            f"}}"
            f"QPushButton {{"
            f"  background: {bg}; color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px; padding: 6px 12px;"
            f"}}"
            f"QPushButton:hover {{ border-color: {accent}; }}"
            f"QPushButton#authPrimaryButton {{"
            f"  background: {accent}; color: {primary_text}; border: none;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton#authPrimaryButton:hover {{ background: {primary_hover}; }}"
            f"QPushButton#authPrimaryButton:disabled {{"
            f"  background: {disabled_bg}; color: {secondary};"
            f"}}"
            f"QLabel#authStatusLabel {{ color: {secondary}; font-size: 12px; }}"
            f"QLabel#authStatusLabel[isError=\"true\"] {{ color: {error_color}; }}"
        )

    def _wire_signals(self) -> None:
        self._btn_close.clicked.connect(self.reject)
        self._btn_send.clicked.connect(self._on_send)

        self._auth.info_message.connect(self._on_info)
        self._auth.auth_error.connect(self._on_error)

    # ----- handlers --------------------------------------------------------

    def _on_send(self) -> None:
        if self._busy:
            return
        if not self._primary_email:
            self._show_error(
                tr("auth.error_empty_email", "Email is required.")
            )
            return

        self._set_busy(tr(
            "auth.busy_sending_reset", "Sending reset email...",
        ))
        self._auth.send_password_reset(self._primary_email)

    @pyqtSlot(str)
    def _on_info(self, message: str) -> None:
        if not self._busy:
            return
        # send_password_reset emits ``info_message`` with a message that
        # includes "reset email sent to" — that's our success signal.
        lower = (message or "").lower()
        if "reset" not in lower and "sent" not in lower:
            return

        # Optimistically flip the capability bit. The user has clearly
        # signalled intent to enable email sign-in; the actual password
        # they choose happens out-of-band on Supabase's hosted page,
        # which we can't observe in-session. Server truth reconciles
        # on the next sign-in / refresh.
        try:
            self._auth.mark_email_password_pending()
        except Exception:                                       # noqa: BLE001
            # Older AuthManager builds may not have this method; if so,
            # the [Email] chip just stays "not linked" until next
            # sign-in. Don't fail the dialog over it.
            pass

        self._set_idle()
        self._show_status(message, is_error=False)
        # Slightly longer delay than the password-set flow used to use,
        # so the user has time to read the "check your inbox" message
        # before the dialog dismisses.
        QTimer.singleShot(1500, self.accept)

    @pyqtSlot(str, str)
    def _on_error(self, code: str, message: str) -> None:
        if not self._busy:
            return
        friendly = {
            "over_email_send_rate_limit": tr(
                "auth.error_email_rate_limited",
                "Too many requests. Wait a minute and try again.",
            ),
        }.get(code)
        self._show_error(friendly or message or code)

    # ----- helpers ---------------------------------------------------------

    def _set_busy(self, hint: str) -> None:
        self._busy = True
        self._progress.setVisible(True)
        self._btn_send.setEnabled(False)
        self._show_status(hint, is_error=False)

    def _set_idle(self) -> None:
        self._busy = False
        self._progress.setVisible(False)
        if self._primary_email:
            self._btn_send.setEnabled(True)

    def _show_status(self, message: str, *, is_error: bool) -> None:
        self._status.setText(message)
        self._status.setProperty("isError", "true" if is_error else "false")
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _show_error(self, message: str) -> None:
        self._set_idle()
        self._show_status(message, is_error=True)
