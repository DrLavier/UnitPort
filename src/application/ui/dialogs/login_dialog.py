# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""LoginDialog — Supabase sign-in / sign-up modal for the Sidebar User panel.

Triggered by UserPanel when the user is signed out and clicks Sign in.
Subscribes to AuthManager signals so the dialog auto-closes once auth succeeds
(handles both password flow and deferred OAuth / deep-link flow).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QRegularExpression, pyqtSlot
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, i18n_bind, tr

from application.service.auth import UserProfile, get_auth_manager


_EMAIL_REGEX = QRegularExpression(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginDialog(QDialog):
    """Modal login window. Use exec() — returns Accepted on success."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._auth = get_auth_manager()
        self._busy = False

        i18n_bind(self, "setWindowTitle",
                  "auth.dialog_title", "UnitPort Account")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._build_ui()
        self._apply_styles()
        self._wire_signals()

    # ----- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(10)

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._build_sign_in_tab(), tr("auth.tab_sign_in", "Sign In"))
        self._tabs.addTab(self._build_sign_up_tab(), tr("auth.tab_sign_up", "Sign Up"))
        root.addWidget(self._tabs)

        # OAuth row
        divider_row = QHBoxLayout()
        divider_row.setSpacing(8)
        line_l = QFrame(); line_l.setFrameShape(QFrame.Shape.HLine)
        line_r = QFrame(); line_r.setFrameShape(QFrame.Shape.HLine)
        divider_row.addWidget(line_l, 1)
        self._divider_label = QLabel(tr("auth.continue_with", "or continue with"))
        self._divider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        divider_row.addWidget(self._divider_label)
        divider_row.addWidget(line_r, 1)
        self._divider_lines = (line_l, line_r)
        root.addLayout(divider_row)

        oauth_row = QHBoxLayout()
        oauth_row.setSpacing(10)
        self._btn_google = QPushButton(tr("auth.continue_google", "Continue with Google"))
        self._btn_github = QPushButton(tr("auth.continue_github", "Continue with GitHub"))
        for b in (self._btn_google, self._btn_github):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setMinimumHeight(34)
        oauth_row.addWidget(self._btn_google, 1)
        oauth_row.addWidget(self._btn_github, 1)
        root.addLayout(oauth_row)

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
        root.addLayout(btn_row)

    def _build_sign_in_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 12, 6, 6)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self._signin_email = QLineEdit()
        i18n_bind(self._signin_email, "setPlaceholderText",
                  "auth.email_placeholder", "you@example.com")
        self._signin_email.setValidator(
            QRegularExpressionValidator(_EMAIL_REGEX, self._signin_email)
        )
        form.addRow(tr("auth.email_label", "Email"), self._signin_email)

        self._signin_password = QLineEdit()
        self._signin_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._signin_password.setPlaceholderText(
            tr("auth.password_placeholder", "at least 6 characters")
        )
        form.addRow(tr("auth.password_label", "Password"), self._signin_password)
        layout.addLayout(form)

        forgot_row = QHBoxLayout()
        forgot_row.addStretch(1)
        self._btn_forgot = QPushButton(tr("auth.forgot_password", "Forgot password?"))
        self._btn_forgot.setFlat(True)
        self._btn_forgot.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_forgot.setObjectName("authLinkButton")
        forgot_row.addWidget(self._btn_forgot)
        layout.addLayout(forgot_row)

        self._btn_sign_in = QPushButton(tr("auth.sign_in_button", "Sign in"))
        self._btn_sign_in.setObjectName("authPrimaryButton")
        self._btn_sign_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_sign_in.setMinimumHeight(34)
        self._btn_sign_in.setDefault(True)
        layout.addWidget(self._btn_sign_in)

        layout.addStretch(1)
        return page

    def _build_sign_up_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 12, 6, 6)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self._signup_email = QLineEdit()
        i18n_bind(self._signup_email, "setPlaceholderText",
                  "auth.email_placeholder", "you@example.com")
        self._signup_email.setValidator(
            QRegularExpressionValidator(_EMAIL_REGEX, self._signup_email)
        )
        form.addRow(tr("auth.email_label", "Email"), self._signup_email)

        self._signup_password = QLineEdit()
        self._signup_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._signup_password.setPlaceholderText(
            tr("auth.password_placeholder", "at least 6 characters")
        )
        form.addRow(tr("auth.password_label", "Password"), self._signup_password)

        self._signup_confirm = QLineEdit()
        self._signup_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self._signup_confirm.setPlaceholderText(
            tr("auth.password_placeholder", "at least 6 characters")
        )
        form.addRow(
            tr("auth.confirm_password_label", "Confirm password"), self._signup_confirm
        )
        layout.addLayout(form)

        self._btn_sign_up = QPushButton(tr("auth.sign_up_button", "Create account"))
        self._btn_sign_up.setObjectName("authPrimaryButton")
        self._btn_sign_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_sign_up.setMinimumHeight(34)
        layout.addWidget(self._btn_sign_up)

        layout.addStretch(1)
        return page

    def _apply_styles(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        secondary = Config.get_color("main_c2")
        accent = Config.get_color("safe_zone")
        primary_text = Config.get_color("auth_primary_button_text")
        primary_hover = Config.get_color("auth_primary_button_hover_bg")
        disabled_bg = Config.get_color("auth_button_disabled_bg")

        self.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
            f"QLineEdit {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px;"
            f"  padding: 6px 8px;"
            f"}}"
            f"QLineEdit:focus {{ border-color: {accent}; }}"
            f"QPushButton {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px;"
            f"  padding: 6px 12px;"
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
            f"QPushButton#authLinkButton {{"
            f"  background: transparent; border: none; color: {secondary};"
            f"  text-decoration: underline; padding: 0;"
            f"}}"
            f"QPushButton#authLinkButton:hover {{ color: {accent}; }}"
            f"QLabel#authStatusLabel {{ color: {secondary}; font-size: 12px; }}"
        )

        for line in getattr(self, "_divider_lines", ()):
            line.setStyleSheet(f"background-color: {border}; border: none;")
        self._divider_label.setStyleSheet(f"color: {secondary}; font-size: 11px;")

    def _wire_signals(self) -> None:
        self._btn_close.clicked.connect(self.reject)
        self._btn_sign_in.clicked.connect(self._on_sign_in)
        self._btn_sign_up.clicked.connect(self._on_sign_up)
        self._btn_forgot.clicked.connect(self._on_forgot)
        self._btn_google.clicked.connect(lambda: self._on_oauth("google"))
        self._btn_github.clicked.connect(lambda: self._on_oauth("github"))

        self._signin_password.returnPressed.connect(self._on_sign_in)
        self._signup_confirm.returnPressed.connect(self._on_sign_up)

        self._auth.authenticated.connect(self._on_authenticated)
        self._auth.auth_error.connect(self._on_error)
        self._auth.email_confirmation_pending.connect(self._on_email_pending)
        self._auth.info_message.connect(self._on_info)

    # ----- handlers --------------------------------------------------------

    def _on_sign_in(self) -> None:
        if self._busy:
            return
        email = self._signin_email.text().strip()
        password = self._signin_password.text()
        if not email:
            self._show_error(tr("auth.error_empty_email", "Email is required."))
            return
        if not password:
            self._show_error(tr("auth.error_empty_password", "Password is required."))
            return
        self._set_busy(tr("auth.busy_signing_in", "Signing in..."))
        self._auth.sign_in_password(email, password)

    def _on_sign_up(self) -> None:
        if self._busy:
            return
        email = self._signup_email.text().strip()
        password = self._signup_password.text()
        confirm = self._signup_confirm.text()
        if not email:
            self._show_error(tr("auth.error_empty_email", "Email is required."))
            return
        if not password:
            self._show_error(tr("auth.error_empty_password", "Password is required."))
            return
        if len(password) < 6:
            self._show_error(tr("auth.error_password_too_short",
                                "Password must be at least 6 characters."))
            return
        if password != confirm:
            self._show_error(tr("auth.error_password_mismatch",
                                "Passwords do not match."))
            return
        self._set_busy(tr("auth.busy_signing_up", "Creating account..."))
        self._auth.sign_up(email, password)

    def _on_forgot(self) -> None:
        if self._busy:
            return
        email = self._signin_email.text().strip()
        if not email:
            self._show_error(tr("auth.error_empty_email", "Email is required."))
            self._signin_email.setFocus()
            return
        self._set_busy(tr("auth.busy_sending_reset", "Sending reset email..."))
        self._auth.send_password_reset(email)

    def _on_oauth(self, provider: str) -> None:
        if self._busy:
            return
        self._set_busy(tr("auth.busy_oauth",
                          "Waiting for browser confirmation..."))
        self._auth.sign_in_oauth(provider)

    # ----- AuthManager signals --------------------------------------------

    @pyqtSlot(object)
    def _on_authenticated(self, _user) -> None:
        self._set_busy(None)
        self.accept()

    @pyqtSlot(str, str)
    def _on_error(self, code: str, message: str) -> None:
        self._set_busy(None)
        self._show_error(_friendly_error(code, message))

    @pyqtSlot(str)
    def _on_email_pending(self, email: str) -> None:
        self._set_busy(None)
        self._show_info(
            tr("auth.info_email_confirmation",
               "Account created. Check {email} to confirm your address.",
               ).format(email=email)
        )

    @pyqtSlot(str)
    def _on_info(self, message: str) -> None:
        self._set_busy(None)
        self._show_info(message)

    # ----- helpers ---------------------------------------------------------

    def _set_busy(self, message: Optional[str]) -> None:
        self._busy = message is not None
        self._progress.setVisible(self._busy)
        for btn in (
            self._btn_sign_in, self._btn_sign_up, self._btn_forgot,
            self._btn_google, self._btn_github,
        ):
            btn.setEnabled(not self._busy)
        if message:
            self._show_info(message)

    def _show_info(self, message: str) -> None:
        secondary = Config.get_color("main_c2")
        self._status.setStyleSheet(
            f"QLabel#authStatusLabel {{ color: {secondary}; font-size: 12px; }}"
        )
        self._status.setText(message)

    def _show_error(self, message: str) -> None:
        error_color = Config.get_color("auth_status_error_color")
        self._status.setStyleSheet(
            f"QLabel#authStatusLabel {{ color: {error_color}; font-size: 12px; }}"
        )
        self._status.setText(message)


def _friendly_error(code: str, message: str) -> str:
    c = (code or "").lower()
    if c == "network":
        return tr("auth.error_network", "Network error — check your connection.")
    if c in ("invalid_grant", "invalid_credentials", "invalid_login_credentials"):
        return tr("auth.error_invalid_credentials", "Incorrect email or password.")
    if c in ("user_already_exists", "user_already_registered", "email_exists"):
        return tr("auth.error_email_taken",
                  "An account with this email already exists.")
    return tr("auth.error_generic", "Sign-in failed: {message}",
              ).format(message=message or code or "")


__all__ = ["LoginDialog"]
