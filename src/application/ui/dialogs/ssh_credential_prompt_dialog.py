"""SshCredentialPromptDialog — modal credential collector for AutoConnect.

Phase 3.7 forbids the silent-skip path that earlier diagnostics took when
SSH credentials were missing: every probe that needs SSH must either run
or surface a loud error. When :class:`AutoRepairLoop` notices an
``ssh_required`` finding, it emits ``connection_needs_ssh`` and blocks on
a threading.Event. This dialog is the only thing that resolves that
block — the user enters credentials (saved to
:class:`SecureCredentialStore`) and clicks Save & Continue, which fires
``connection_ssh_response(server, True)``; Cancel fires
``(server, False)`` so the loop can decide to give up gracefully.

The dialog is modal and intentionally cannot be closed by the window's
[X] without going through one of the two response paths — closing via
[X] is treated as Cancel.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import Config, log_warning, setButton, tr

from application.service.auth.secure import get_secure_store
from application.service.signals import get_app_signals


class SshCredentialPromptDialog(QDialog):
    """Modal that collects SSH password (and optional sudo password)."""

    def __init__(
        self,
        server_key: str,
        suggested_user: str,
        reason: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionDiagSshPromptDialog")
        self.setModal(True)
        self.setWindowTitle(
            tr("mission.diag.ssh.title", default="SSH Credentials Required")
        )
        self.resize(440, 260)
        self._server_key = server_key
        self._suggested_user = suggested_user or "ubuntu"
        self._reason = reason or ""
        self._responded = False
        self._build_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        head = QLabel(
            tr(
                "mission.diag.ssh.headline",
                default="Auto-repair needs SSH access to the robot.",
            ),
            self,
        )
        head.setObjectName("missionDiagSshHeadline")
        head.setWordWrap(True)
        outer.addWidget(head, 0)

        sub = QLabel(
            tr(
                "mission.diag.ssh.context",
                default="Server: {server}\nReason: {reason}",
            ).format(server=self._server_key, reason=self._reason),
            self,
        )
        sub.setObjectName("missionDiagSshContext")
        sub.setWordWrap(True)
        outer.addWidget(sub, 0)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self._user_edit = QLineEdit(self._suggested_user, self)
        self._user_edit.setObjectName("missionDiagSshInput")
        form.addRow(
            QLabel(tr("mission.diag.ssh.user", default="SSH user"), self),
            self._user_edit,
        )

        self._password_edit = QLineEdit(self)
        self._password_edit.setObjectName("missionDiagSshInput")
        self._password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_edit.setPlaceholderText(
            tr("mission.diag.ssh.pw_placeholder", default="required")
        )
        form.addRow(
            QLabel(tr("mission.diag.ssh.pw", default="SSH password"), self),
            self._password_edit,
        )

        self._sudo_edit = QLineEdit(self)
        self._sudo_edit.setObjectName("missionDiagSshInput")
        self._sudo_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._sudo_edit.setPlaceholderText(
            tr(
                "mission.diag.ssh.sudo_placeholder",
                default="optional — defaults to SSH password if blank",
            )
        )
        form.addRow(
            QLabel(tr("mission.diag.ssh.sudo", default="Sudo password"), self),
            self._sudo_edit,
        )
        outer.addLayout(form)

        self._error_label = QLabel("", self)
        self._error_label.setObjectName("missionDiagSshError")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        outer.addWidget(self._error_label, 0)

        outer.addStretch(1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addStretch(1)
        self._btn_cancel = setButton(
            "mission.diag.ssh.btn_cancel", 100, 32,
            kind="normal", spec="none", default="Cancel", parent=self,
        )
        self._btn_save = setButton(
            "mission.diag.ssh.btn_save", 180, 32,
            kind="normal", spec="save", default="Save & Continue", parent=self,
        )
        footer.addWidget(self._btn_cancel, 0)
        footer.addWidget(self._btn_save, 0)
        outer.addLayout(footer)

        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_save.clicked.connect(self._on_save)
        self._password_edit.returnPressed.connect(self._on_save)
        self._password_edit.setFocus()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        border = Config.get_color("border_2")
        input_bg = Config.get_color("bg_1")
        input_fg = Config.get_color("main_t1")
        focus = Config.get_color("checked_1")
        err = Config.get_color("auth_status_error_color")
        font_normal = Config.get_font_size("size_normal")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QDialog#missionDiagSshPromptDialog {{ background-color: {bg}; }}"
            f"QLabel#missionDiagSshHeadline {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 600; }}"
            f"QLabel#missionDiagSshContext {{ color: {sub}; "
            f"font-size: {font_small}px; }}"
            f"QLineEdit#missionDiagSshInput {{ background-color: {input_bg}; "
            f"color: {input_fg}; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 4px 6px; }}"
            f"QLineEdit#missionDiagSshInput:focus {{ "
            f"border: 1px solid {focus}; }}"
            f"QLabel#missionDiagSshError {{ color: {err}; "
            f"font-size: {font_small}px; }}"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        password = self._password_edit.text().strip()
        if not password:
            self._error_label.setText(
                tr(
                    "mission.diag.ssh.err_empty_pw",
                    default="SSH password is required.",
                )
            )
            self._error_label.setVisible(True)
            self._password_edit.setFocus()
            return

        sudo_pw = self._sudo_edit.text().strip() or password
        store = get_secure_store()
        try:
            store.ssh_password(self._server_key).set(password)
            store.ssh_sudo_password(self._server_key).set(sudo_pw)
        except Exception as exc:  # noqa: BLE001
            self._error_label.setText(
                tr(
                    "mission.diag.ssh.err_save",
                    default="Could not save credentials: {err}",
                ).format(err=str(exc))
            )
            self._error_label.setVisible(True)
            log_warning(f"[ssh-prompt] save failed: {exc}")
            return

        self._respond(ok=True)
        self.accept()

    def _on_cancel(self) -> None:
        self._respond(ok=False)
        self.reject()

    def _respond(self, *, ok: bool) -> None:
        if self._responded:
            return
        self._responded = True
        try:
            get_app_signals().connection_ssh_response.emit(self._server_key, ok)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[ssh-prompt] response emit failed: {exc}")

    # ------------------------------------------------------------------
    # Lifecycle — close via [X] = cancel
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        if not self._responded:
            self._respond(ok=False)
        super().closeEvent(event)


__all__ = ["SshCredentialPromptDialog"]
