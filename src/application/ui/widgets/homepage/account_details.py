# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AccountDetailsWidget — sign-in methods + Cloud Sync block.

Sidebar-only consumer:

* ``application.ui.sidebar_panels.user_panel.UserPanel``

The homepage uses :class:`HomeAccountCard` instead, which inlines a
different visual layout. This widget stays around because the sidebar
still wants the legacy vertical-stack arrangement: provider chips on
one horizontal row, Cloud Sync section below.

The widget self-subscribes to ``AuthManager`` / ``CloudSyncService`` /
``TaskSignal`` so dropping it into any layout gives the full behaviour
without the host needing to plumb anything.

Public signal:

* :attr:`workspace_refreshed` — emitted after a successful Pull whose
  ``ok > 0``. Hosts that hold project state (MainWindow, sidebar
  ProjectsPanel) should re-read the workspace.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    Paths,
    SliderSwitch,
    get_task_signal,
    get_tasks_manager,
    i18n_bind,
    log_info,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.auth import get_auth_manager
from application.service.cloud_sync import get_cloud_sync_service
from application.tools.cloud_sync_task import CloudSyncTask


class AccountDetailsWidget(QWidget):
    """Reusable Sign-in methods + Cloud Sync block."""

    _LINK_PROVIDERS = ("email", "google", "github")

    workspace_refreshed = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("accountDetailsWidget")
        self._auth = get_auth_manager()

        self._link_buttons: dict[str, QPushButton] = {}
        self._link_status_label: Optional[QLabel] = None
        self._link_status_timer = QTimer(self)
        self._link_status_timer.setSingleShot(True)
        self._link_status_timer.timeout.connect(self._clear_link_status)
        self._last_link_attempt: str = ""

        self._cloud_status_label: Optional[QLabel] = None
        self._cloud_push_btn: Optional[QPushButton] = None
        self._cloud_pull_btn: Optional[QPushButton] = None
        self._cloud_auto_switch: Optional[SliderSwitch] = None
        self._cloud_push_task_id: str = ""
        self._cloud_pull_task_id: str = ""

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(18)
        col.addWidget(self._build_linked_accounts_row())
        col.addWidget(self._build_cloud_sync_row())

        self._auth.identities_changed.connect(self._on_identities_changed)
        self._auth.identity_linked.connect(self._on_identity_linked)
        self._auth.identity_unlinked.connect(self._on_identity_unlinked)
        self._auth.auth_error.connect(self._on_link_auth_error)
        self._auth.info_message.connect(self._on_link_info_message)
        self._auth.authenticated.connect(self._on_auth_changed)
        self._auth.signed_out.connect(self._on_auth_signed_out)
        self._auth.workspace_changed.connect(self._on_workspace_changed)

        get_cloud_sync_service().status_changed.connect(
            self._on_cloud_status_changed
        )
        get_task_signal().task_finished.connect(self._on_cloud_task_finished)

        self._refresh_link_buttons()
        self._refresh_cloud_status_text()

    # ------------------------------------------------------------------
    # Linked accounts
    # ------------------------------------------------------------------

    def _build_linked_accounts_row(self) -> QWidget:
        host = QWidget(self)
        host.setObjectName("accountDetailsLinkedHost")
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        sz = int(Config.get_font_size("size_small"))
        col.addWidget(setText(
            "user.linked_accounts_label", default="Sign-in methods:",
            kind="content", size=sz,
        ))

        chips_row = QHBoxLayout()
        chips_row.setContentsMargins(0, 0, 0, 0)
        chips_row.setSpacing(10)

        for provider in self._LINK_PROVIDERS:
            btn = QPushButton("")
            btn.setObjectName("linkedAccountButton")
            btn.setMinimumHeight(26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("provider", provider)
            btn.setProperty("isPrimary", "false")
            btn.setProperty("isLinked", "false")
            btn.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            icon_path = Assets.find_icon(f"icon_{provider}")
            if icon_path is not None:
                btn.setIcon(QIcon(str(icon_path)))
                btn.setIconSize(QSize(14, 14))
            else:
                log_warning(
                    f"[account_details] provider icon missing: icon_{provider}"
                )
            btn.clicked.connect(
                lambda _checked=False, p=provider: self._on_link_button_clicked(p)
            )
            self._link_buttons[provider] = btn
            chips_row.addWidget(btn)

        chips_row.addStretch(1)
        col.addLayout(chips_row)

        self._link_status_label = QLabel("", host)
        self._link_status_label.setObjectName("authStatusLabel")
        self._link_status_label.setWordWrap(True)
        self._link_status_label.setVisible(False)
        col.addWidget(self._link_status_label)

        self._apply_link_button_styles()
        return host

    def _apply_link_button_styles(self) -> None:
        border = Config.get_color("border_1")
        safe = Config.get_color("safe_zone")
        fg = Config.get_color("main_t1")
        secondary = Config.get_color("main_c2")
        error_color = Config.get_color("auth_status_error_color")
        sz = int(Config.get_font_size("size_small"))

        qss = (
            f"QPushButton#linkedAccountButton {{"
            f"  background: transparent; color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 4px; padding: 4px 10px;"
            f"  font-size: {sz}px;"
            f"  text-align: left;"
            f"}}"
            f"QPushButton#linkedAccountButton:hover {{"
            f"  background: transparent;"
            f"}}"
            f"QPushButton#linkedAccountButton[isLinked=\"true\"] {{"
            f"  border: 1px solid {safe};"
            f"}}"
            f"QPushButton#linkedAccountButton[isPrimary=\"true\"] {{"
            f"  border: 2px solid {safe};"
            f"}}"
            f"QPushButton#linkedAccountButton:disabled {{"
            f"  color: {secondary};"
            f"  background: transparent;"
            f"}}"
            f"QLabel#authStatusLabel {{ color: {secondary}; font-size: 11px; }}"
            f"QLabel#authStatusLabel[isError=\"true\"] {{ color: {error_color}; }}"
        )
        self.setStyleSheet(qss)

    def _refresh_link_buttons(
        self, identities: Optional[list] = None,
    ) -> None:
        if not self._link_buttons:
            return
        if identities is None:
            identities = self._auth.current_identities()
        by_provider = {
            str(i.get("provider")): i
            for i in (identities or [])
            if isinstance(i, dict)
        }
        signed_in = self._auth.is_signed_in()
        current_user = self._auth.current_user() if signed_in else None
        current_provider = current_user.provider if current_user else ""

        for provider, btn in self._link_buttons.items():
            idn = by_provider.get(provider)
            display = (
                provider.capitalize() if provider != "github" else "GitHub"
            )
            email_always_linked = (
                provider == "email"
                and idn is None
                and signed_in
                and current_user is not None
                and bool(current_user.email)
            )
            if not signed_in:
                label = tr(
                    f"user.linked_account_btn_signedout_{provider}",
                    default=f"{display}: —",
                )
                is_linked = False
                btn.setEnabled(False)
            elif idn is None and not email_always_linked:
                label = tr(
                    f"user.linked_account_btn_unlinked_{provider}",
                    default=f"{display}: (not linked)",
                )
                is_linked = False
                btn.setEnabled(True)
            else:
                email = ""
                if isinstance(idn, dict):
                    idata = idn.get("identity_data")
                    if isinstance(idata, dict):
                        email = str(idata.get("email") or "")
                if (
                    not email
                    and email_always_linked
                    and current_user is not None
                ):
                    email = current_user.email
                if not email:
                    email = tr(
                        "user.linked_account_status_linked", default="linked",
                    )
                label = f"{display}: {email}"
                is_linked = True
                btn.setEnabled(True)

            btn.setText(label)
            btn.setToolTip(label)
            is_primary = signed_in and (provider == current_provider)
            btn.setProperty("isLinked", "true" if is_linked else "false")
            btn.setProperty("isPrimary", "true" if is_primary else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_link_button_clicked(self, provider: str) -> None:
        log_info(
            f"[account_details] linked-account button clicked: {provider!r}"
        )
        if not self._auth.is_signed_in():
            return
        identities = self._auth.current_identities()
        by_provider = {
            str(i.get("provider")): i
            for i in identities
            if isinstance(i, dict)
        }

        if provider == "email":
            current_user = self._auth.current_user()
            primary_email = current_user.email if current_user else ""
            from application.ui.dialogs.email_identity_dialog import (
                EmailIdentityDialog,
            )
            EmailIdentityDialog(
                primary_email=primary_email,
                parent=self,
            ).exec()
            return

        idn = by_provider.get(provider)
        if idn is None:
            display = (
                provider.capitalize() if provider != "github" else "GitHub"
            )
            self._last_link_attempt = provider
            self._show_link_status(
                tr("auth.busy_linking", "Linking {provider} account...")
                .format(provider=display),
                is_error=False,
            )
            self._auth.link_identity_oauth(provider)
            return

        from application.ui.dialogs.identity_unlink_dialog import (
            IdentityUnlinkDialog,
        )
        current_user = self._auth.current_user()
        IdentityUnlinkDialog(
            provider=provider,
            is_only_identity=(len(identities) <= 1),
            is_current_provider=(
                current_user is not None
                and current_user.provider == provider
            ),
            parent=self,
        ).exec()

    @pyqtSlot(object)
    def _on_identities_changed(self, identities: object) -> None:
        ids = list(identities) if isinstance(identities, list) else []
        self._refresh_link_buttons(ids)

    @pyqtSlot(str)
    def _on_identity_linked(self, provider: str) -> None:
        display = provider.capitalize() if provider != "github" else "GitHub"
        self._show_link_status(f"{display}: linked.", is_error=False)
        self._refresh_link_buttons()

    @pyqtSlot(str)
    def _on_identity_unlinked(self, provider: str) -> None:
        display = provider.capitalize() if provider != "github" else "GitHub"
        self._show_link_status(f"{display}: unlinked.", is_error=False)
        self._refresh_link_buttons()

    @pyqtSlot(str, str)
    def _on_link_auth_error(self, code: str, message: str) -> None:
        lower_msg = (message or "").lower()
        is_already_linked = (
            code in (
                "identity_already_exists",
                "linked_identity_already_exists",
                "manual_linking_disabled",
            )
            or "already linked" in lower_msg
            or "already a user" in lower_msg
        )
        if is_already_linked:
            self._show_identity_conflict_dialog(message)
            return
        friendly_key = {
            "oauth_busy": "auth.error_oauth_busy",
            "single_identity_not_deletable":
                "auth.error_single_identity_not_deletable",
            "identity_already_exists":
                "auth.error_identity_already_exists",
        }.get(code)
        text = (
            tr(friendly_key, default=message) if friendly_key
            else (message or code)
        )
        self._show_link_status(text, is_error=True)

    def _show_identity_conflict_dialog(self, server_message: str) -> None:
        provider = self._last_link_attempt or ""
        display = (
            provider.capitalize() if provider and provider != "github"
            else ("GitHub" if provider == "github" else "the provider")
        )
        body = tr(
            "auth.identity_conflict_body",
            "Couldn't link {provider}.\n\n"
            "That account already belongs to a separate UnitPort user, "
            "and Supabase can't merge two existing accounts.\n\n"
            "You can sign in with {provider} directly instead, or try a "
            "different {provider} account.\n\n"
            "Provider message: {message}",
        ).format(provider=display, message=server_message or "(none)")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(tr(
            "auth.identity_conflict_title", "Identity already in use",
        ))
        box.setText(body)
        switch_btn = None
        if provider in ("google", "github"):
            switch_btn = box.addButton(
                tr(
                    "auth.identity_conflict_switch",
                    "Sign in with {provider} instead",
                ).format(provider=display),
                QMessageBox.ButtonRole.AcceptRole,
            )
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        attempted = self._last_link_attempt
        self._last_link_attempt = ""
        if (
            switch_btn is not None
            and box.clickedButton() is switch_btn
            and attempted
        ):
            log_info(
                f"[account_details] conflict resolution: signing out + "
                f"sign_in_oauth({attempted!r})"
            )
            self._auth.sign_out()
            self._auth.sign_in_oauth(attempted)

    @pyqtSlot(str)
    def _on_link_info_message(self, message: str) -> None:
        self._show_link_status(message, is_error=False)

    def _show_link_status(self, message: str, *, is_error: bool) -> None:
        if self._link_status_label is None:
            return
        self._link_status_label.setText(message)
        self._link_status_label.setProperty(
            "isError", "true" if is_error else "false",
        )
        self._link_status_label.style().unpolish(self._link_status_label)
        self._link_status_label.style().polish(self._link_status_label)
        self._link_status_label.setVisible(True)
        self._link_status_timer.start(4000)

    def _clear_link_status(self) -> None:
        if self._link_status_label is None:
            return
        self._link_status_label.setVisible(False)
        self._link_status_label.setText("")

    # ------------------------------------------------------------------
    # Cloud Sync
    # ------------------------------------------------------------------

    def _build_cloud_sync_row(self) -> QWidget:
        host = QWidget(self)
        host.setObjectName("accountDetailsCloudHost")
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 4, 0, 4)
        col.setSpacing(10)

        sz = int(Config.get_font_size("size_small"))
        col.addWidget(setText(
            "cloud.section_title", default="Cloud Sync",
            kind="content", size=sz,
        ))

        self._cloud_status_label = QLabel("", host)
        self._cloud_status_label.setObjectName("cloudSyncStatus")
        self._cloud_status_label.setWordWrap(True)
        self._cloud_status_label.setStyleSheet(
            f"color: {Config.get_color('main_c2')}; "
            f"background: transparent; font-size: 11px;"
        )
        col.addWidget(self._cloud_status_label)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self._cloud_push_btn = setButton(
            "cloud.push_button", 0, 28,
            kind="normal", spec="save",
            default="Push",
        )
        i18n_bind(
            self._cloud_push_btn, "setToolTip",
            "cloud.push_tooltip",
            "Upload local workspace files to the cloud",
        )
        self._cloud_push_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cloud_push_btn.clicked.connect(self._on_cloud_push_clicked)
        btn_row.addWidget(self._cloud_push_btn)

        self._cloud_pull_btn = setButton(
            "cloud.pull_button", 0, 28,
            kind="normal", spec="none",
            default="Pull",
        )
        i18n_bind(
            self._cloud_pull_btn, "setToolTip",
            "cloud.pull_tooltip",
            "Download cloud files into the local workspace",
        )
        self._cloud_pull_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cloud_pull_btn.clicked.connect(self._on_cloud_pull_clicked)
        btn_row.addWidget(self._cloud_pull_btn)

        btn_row.addStretch(1)
        col.addLayout(btn_row)

        auto_row = QHBoxLayout()
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.setSpacing(8)
        auto_label = setText(
            "cloud.auto_push_label", default="Auto-push",
            kind="content", size=sz,
        )
        auto_row.addWidget(auto_label)
        self._cloud_auto_switch = SliderSwitch(
            options=[("off", "Off"), ("on", "On")],
            height=24,
            min_segment_width=44,
            font_size=10,
        )
        # Seed from user.ini[Cloud] auto_push; persist on every flip.
        # emit=False guards against the seed firing back into the slot
        # and re-writing the same value during construction.
        auto_on = bool(Config.get_value(
            "Cloud", "auto_push", False, value_type=bool,
        ))
        self._cloud_auto_switch.setCurrentIndex(
            1 if auto_on else 0, animated=False, emit=False,
        )
        self._cloud_auto_switch.current_changed.connect(
            self._on_auto_push_changed
        )
        i18n_bind(
            self._cloud_auto_switch, "setToolTip",
            "cloud.auto_push_tooltip",
            "When on, UnitPort pushes each changed workspace file to the cloud "
            "shortly after you save it.",
        )
        auto_row.addWidget(self._cloud_auto_switch)
        auto_row.addStretch(1)
        col.addLayout(auto_row)

        host.setVisible(self._auth.is_signed_in())
        return host

    def _refresh_cloud_status_text(self) -> None:
        if self._cloud_status_label is None:
            return
        if not self._auth.is_signed_in():
            self._cloud_status_label.setText(
                tr("cloud.status_signed_out", "Sign in to enable cloud sync")
            )
            return
        status = get_cloud_sync_service().get_status()
        synced = int(status.get("synced_files", 0) or 0)
        last = (
            status.get("last_push_ts")
            or status.get("last_pull_ts")
            or status.get("last_self_check_ts")
            or ""
        )
        when = last or tr("cloud.never_synced", "never")
        self._cloud_status_label.setText(
            tr(
                "cloud.status_synced",
                "Synced {n} files · last sync {when}",
            ).format(n=synced, when=when)
        )

    @pyqtSlot(dict)
    def _on_cloud_status_changed(self, _payload: dict) -> None:
        self._refresh_cloud_status_text()

    def _on_auto_push_changed(self, _idx: int, key: str) -> None:
        """Persist the Auto-sync toggle to ``user.ini[Cloud] auto_push``.

        When on, the AutoSyncController pushes each include-set file to the
        cloud shortly after it is saved (persist-time, debounced) — there is no
        batch upload on exit. This slot only stores the preference, so the
        widget itself stays inexpensive to toggle.
        """
        Config.set_value("Cloud", "auto_push", key == "on")
        log_info(f"[account_details] auto-sync set to {key!r}")

    def _on_cloud_push_clicked(self) -> None:
        if self._cloud_push_task_id or self._cloud_pull_task_id:
            return
        if not self._auth.is_signed_in():
            return
        if self._cloud_status_label is not None:
            self._cloud_status_label.setText(
                tr("cloud.busy_push", "Pushing to cloud...")
            )
        if self._cloud_push_btn is not None:
            self._cloud_push_btn.setEnabled(False)
        if self._cloud_pull_btn is not None:
            self._cloud_pull_btn.setEnabled(False)
        try:
            self._cloud_push_task_id = get_tasks_manager().submit(
                CloudSyncTask("push")
            )
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[account_details] cloud push submit failed: {exc}")
            self._reset_cloud_buttons()
            self._refresh_cloud_status_text()

    def _on_cloud_pull_clicked(self) -> None:
        if self._cloud_push_task_id or self._cloud_pull_task_id:
            return
        if not self._auth.is_signed_in():
            return
        if self._cloud_status_label is not None:
            self._cloud_status_label.setText(
                tr("cloud.busy_pull", "Pulling from cloud...")
            )
        if self._cloud_push_btn is not None:
            self._cloud_push_btn.setEnabled(False)
        if self._cloud_pull_btn is not None:
            self._cloud_pull_btn.setEnabled(False)
        try:
            self._cloud_pull_task_id = get_tasks_manager().submit(
                CloudSyncTask("pull")
            )
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[account_details] cloud pull submit failed: {exc}")
            self._reset_cloud_buttons()
            self._refresh_cloud_status_text()

    def _reset_cloud_buttons(self) -> None:
        self._cloud_push_task_id = ""
        self._cloud_pull_task_id = ""
        if self._cloud_push_btn is not None:
            self._cloud_push_btn.setEnabled(True)
        if self._cloud_pull_btn is not None:
            self._cloud_pull_btn.setEnabled(True)

    @pyqtSlot(str, bool, object)
    def _on_cloud_task_finished(
        self, task_id: str, ok: bool, result: object,
    ) -> None:
        if task_id not in (self._cloud_push_task_id, self._cloud_pull_task_id):
            return
        is_push = task_id == self._cloud_push_task_id
        self._reset_cloud_buttons()
        if not ok:
            log_warning(
                f"[account_details] cloud-sync task {task_id!r} failed; "
                f"result={result!r}"
            )
            if self._cloud_status_label is not None:
                self._cloud_status_label.setText(
                    tr(
                        "cloud.status_error", "Sync failed: {msg}",
                    ).format(msg=str(result) if result else "unknown")
                )
            return
        summary = result if isinstance(result, dict) else {}
        ok_n = int(summary.get("ok", 0) or 0)
        total = int(summary.get("total", ok_n) or ok_n)
        skipped = int(summary.get("skipped", 0) or 0) + len(
            summary.get("oversize", []) or []
        )
        if not is_push and ok_n > 0:
            try:
                from application.service.user_workspace import (
                    reload_workspace_data,
                )
                reload_workspace_data()
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[account_details] post-pull reload failed: {exc}"
                )
            try:
                from application.service.engines import get_engine_service
                get_engine_service().refresh()
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[account_details] post-pull engine refresh failed: "
                    f"{exc}"
                )
            new_str = str(Paths.USER_CONFIG_DIR)
            self.workspace_refreshed.emit(new_str, new_str)
        key = "cloud.done_push" if is_push else "cloud.done_pull"
        default = (
            "Pushed {ok}/{total} files. Skipped {skipped}."
            if is_push
            else "Pulled {ok}/{total} files. Skipped {skipped}."
        )
        if self._cloud_status_label is not None:
            self._cloud_status_label.setText(
                tr(key, default).format(ok=ok_n, total=total, skipped=skipped)
            )

    # ------------------------------------------------------------------
    # Auth-state transitions
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_auth_changed(self, _user: object) -> None:
        cloud_host = (
            self._cloud_status_label.parentWidget()
            if self._cloud_status_label else None
        )
        if cloud_host is not None:
            cloud_host.setVisible(True)
        self._refresh_link_buttons()
        self._refresh_cloud_status_text()

    @pyqtSlot()
    def _on_auth_signed_out(self) -> None:
        cloud_host = (
            self._cloud_status_label.parentWidget()
            if self._cloud_status_label else None
        )
        if cloud_host is not None:
            cloud_host.setVisible(False)
        self._refresh_link_buttons()

    @pyqtSlot(str)
    def _on_workspace_changed(self, _uid: str) -> None:
        self._refresh_cloud_status_text()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        self._apply_link_button_styles()
        if self._cloud_status_label is not None:
            self._cloud_status_label.setStyleSheet(
                f"color: {Config.get_color('main_c2')}; "
                f"background: transparent; font-size: 11px;"
            )


__all__ = ["AccountDetailsWidget"]
