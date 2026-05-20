"""UserPanel — sidebar content for the User key.

Layout:

    Account card  (avatar | name + subtitle | sign-in/out button)
    --- divider ---
    Engines section title
    [scrollable list of EngineRow widgets]
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18n,
    Paths,
    i18n_bind,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.auth import UserProfile, get_auth_manager
from application.service.auth.avatar_cache import make_initials_pixmap, round_pixmap
from application.service.engines import get_engine_service
from application.ui.dialogs.engine_settings_dialog import EngineSettingsDialog
from application.ui.dialogs.login_dialog import LoginDialog
from application.ui.widgets.homepage.account_details import AccountDetailsWidget


_AVATAR_PX = 40


class _EngineRow(QWidget):
    """One engine's status block: name + Local row + Cloud row."""

    def __init__(self, engine_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine_id = engine_id
        self._svc = get_engine_service()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(4)

        sz = int(Config.get_font_size("size_small"))
        self._name_label = setText(
            f"engines.{engine_id}.name", default=_pretty_engine_name(engine_id),
            kind="title", size=sz,
        )
        layout.addWidget(self._name_label)

        # Local row
        local_row = QHBoxLayout()
        local_row.setContentsMargins(0, 0, 0, 0)
        local_row.setSpacing(6)
        local_row.addWidget(setText(
            "user.engines_local", default="Local:", kind="content", size=sz,
        ))
        self._local_status = QLabel("")
        self._local_status.setWordWrap(True)
        self._local_status.setStyleSheet(f"font-size: {sz}px; background: transparent;")
        local_row.addWidget(self._local_status, 1)
        self._btn_local = setButton(
            f"user.engines_settings.local.{engine_id}",
            22, 22,
            kind="light", spec="none",
            icon="icon_setting", icon_only=True,
            default="",
        )
        i18n_bind(
            self._btn_local, "setToolTip",
            "user.engines_local_settings", "Configure local installation",
        )
        self._btn_local.clicked.connect(self._on_local_clicked)
        local_row.addWidget(self._btn_local)
        layout.addLayout(local_row)

        # Cloud row
        cloud_row = QHBoxLayout()
        cloud_row.setContentsMargins(0, 0, 0, 0)
        cloud_row.setSpacing(6)
        cloud_row.addWidget(setText(
            "user.engines_cloud", default="Cloud:", kind="content", size=sz,
        ))
        self._cloud_status = QLabel("")
        self._cloud_status.setWordWrap(True)
        self._cloud_status.setStyleSheet(f"font-size: {sz}px; background: transparent;")
        cloud_row.addWidget(self._cloud_status, 1)
        self._btn_cloud = setButton(
            f"user.engines_settings.cloud.{engine_id}",
            22, 22,
            kind="light", spec="none",
            icon="icon_setting", icon_only=True,
            default="",
        )
        i18n_bind(
            self._btn_cloud, "setToolTip",
            "user.engines_cloud_settings", "Configure cloud servers",
        )
        self._btn_cloud.clicked.connect(self._on_cloud_clicked)
        cloud_row.addWidget(self._btn_cloud)
        layout.addLayout(cloud_row)

        self._svc.changed.connect(self._on_changed)
        self.refresh()

    def refresh(self) -> None:
        status = self._svc.status(self._engine_id)
        local = self._svc.get_local(self._engine_id)
        servers = self._svc.list_servers(self._engine_id)
        default_server = self._svc.get_default_server_name(self._engine_id)

        ok_color = Config.get_color("safe_zone")
        warn_color = Config.get_color("danger_zone")
        muted = Config.get_color("main_c2")
        sz = int(Config.get_font_size("size_small"))

        # Local status. Three families:
        #   1. Built-in (pip-installed Python deps that ship with the SDK
        #      requirements.txt: ``sb3`` / ``sb3_mujoco`` / ``mujoco``) —
        #      no user registration; show version only.
        #   2. Externally-installed (``isaac_lab``) — user picks a root via
        #      the gear button; show path AND version.
        #   3. Generic / unknown — fall back to the registered-root pattern.
        local_root = str(local.get("root", "")) if local else ""
        if self._engine_id in ("sb3", "sb3_mujoco", "mujoco"):
            if status["available"]:
                local_text = tr(
                    "engines.local_builtin",
                    "Built-in (pip), v{ver}",
                ).format(ver=status["version"] or "?")
                color = ok_color
            else:
                local_text = tr(
                    "engines.local_missing", "Not installed",
                )
                color = warn_color
        elif self._engine_id == "isaac_lab":
            if local.get("registered") and local_root:
                if status["available"]:
                    ver = status["version"] or "?"
                    local_text = tr(
                        "engines.local_registered_with_ver",
                        "{root} (v{ver})",
                    ).format(root=local_root, ver=ver)
                    color = ok_color
                else:
                    local_text = tr(
                        "engines.local_registered_no_module",
                        "{root} (module not importable)",
                    ).format(root=local_root)
                    color = warn_color
            else:
                local_text = tr(
                    "engines.local_isaac_unregistered",
                    "Not registered — pick install root via the gear",
                )
                color = muted
        else:
            if local.get("registered") and local_root:
                if status["available"]:
                    local_text = local_root
                    color = ok_color
                else:
                    local_text = tr(
                        "engines.local_registered_no_module",
                        "{root} (module not importable)",
                    ).format(root=local_root)
                    color = warn_color
            else:
                local_text = tr("engines.local_unregistered", "Not registered")
                color = muted

        self._local_status.setText(local_text)
        self._local_status.setStyleSheet(
            f"color: {color}; background: transparent; font-size: {sz}px;"
        )

        # Cloud status
        if not servers:
            cloud_text = tr("engines.cloud_none", "No servers")
            cloud_color = muted
        else:
            count_label = tr(
                "engines.cloud_summary",
                "{n} server(s); default: {d}",
            ).format(n=len(servers), d=default_server or "—")
            cloud_text = count_label
            cloud_color = ok_color if default_server else muted
        self._cloud_status.setText(cloud_text)
        self._cloud_status.setStyleSheet(
            f"color: {cloud_color}; background: transparent; font-size: {sz}px;"
        )

    @pyqtSlot(str)
    def _on_changed(self, engine_id: str) -> None:
        if engine_id and engine_id != self._engine_id:
            return
        self.refresh()

    def _on_local_clicked(self) -> None:
        if self._engine_id in ("sb3", "sb3_mujoco", "mujoco"):
            # No directory needed — just refresh availability detection.
            self._svc.refresh()
            return
        if self._engine_id == "isaac_lab":
            start = self._svc.get_local("isaac_lab").get("root", "") or ""
            chosen = QFileDialog.getExistingDirectory(
                self, tr("engines.pick_isaac", "Select Isaac Lab installation root"),
                start,
            )
            if not chosen:
                return
            ok = self._svc.register_isaac_local(chosen)
            if not ok:
                log_warning(f"[user_panel] Isaac Lab validation failed for {chosen}")
            return
        # Generic engine: free-form root pick.
        chosen = QFileDialog.getExistingDirectory(
            self, tr("engines.pick_generic", "Select engine installation root"), "",
        )
        if chosen:
            self._svc.set_local(self._engine_id, root=chosen, registered=True, enabled=True)

    def _on_cloud_clicked(self) -> None:
        dlg = EngineSettingsDialog(self._engine_id, parent=self)
        dlg.exec()


def _pretty_engine_name(engine_id: str) -> str:
    if engine_id == "sb3":
        return "Stable-Baselines3"
    if engine_id == "sb3_mujoco":
        return "Stable-Baselines3 + MuJoCo"
    if engine_id == "mujoco":
        return "MuJoCo"
    if engine_id == "isaac_lab":
        return "NVIDIA Isaac Lab"
    return engine_id.replace("_", " ").title()


# User-facing engines surfaced in the sidebar Engines section, in the
# order they should appear. ``mujoco`` is the SDK's hard requirement (it
# powers the in-process review viewer), ``sb3_mujoco`` is the SB3 training
# canvas owner, ``isaac_lab`` is the high-fidelity training + review path.
# Probe-only rows like raw ``sb3`` / ``gymnasium`` are intentionally
# omitted — they live inside the composite engines and surface there as
# version info, not as separate user-managed engines.
_USER_FACING_ENGINES = ("mujoco", "sb3_mujoco", "isaac_lab")


class UserPanel(QWidget):
    """Top-of-sidebar account card + User Workspace row + Engines section."""

    # Emitted after a successful USER_CONFIG_DIR migration so MainWindow
    # can re-bind the active project (its path moved with the rest) and
    # refresh sidebar snapshots. Args: (old_dir, new_dir) as strings.
    workspace_changed = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._auth = get_auth_manager()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Section spacing — account card / workspace / sign-in methods /
        # divider / engines all sit on this. Bumped from 6 for visual
        # breathing room between the stacked sections.
        layout.setSpacing(10)

        layout.addWidget(self._build_account_card())

        # User Workspace shortcut — single LaviButton that opens the live
        # USER_CONFIG_DIR in the OS file explorer. Previous row had a
        # label + elided path + gear (relocate dialog); relocation now
        # happens implicitly on login / account swap, so the sidebar only
        # needs the "show me my data folder" shortcut.
        self._workspace_cached_path: str = str(Paths.USER_CONFIG_DIR)
        layout.addWidget(self._build_workspace_row())

        # Sign-in methods + Cloud Sync — both sections share an
        # implementation with the homepage UserCard via the reusable
        # AccountDetailsWidget. The widget self-subscribes to all the
        # auth + cloud-sync signals it needs; we just forward its
        # post-pull notification to UserPanel's existing
        # ``workspace_changed`` signal so MainWindow stays wired the
        # same way it was before the extraction.
        self._account_details = AccountDetailsWidget(self)
        self._account_details.workspace_refreshed.connect(
            self.workspace_changed.emit
        )
        layout.addWidget(self._account_details)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            f"background-color: {Config.get_color('border_1')}; border: none;"
        )
        layout.addWidget(divider)

        # Engines section
        layout.addWidget(setText(
            "user.engines_section", default="Engines",
            kind="title",
            size=int(Config.get_font_size("size_small")),
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Three layers to strip transparently: the QScrollArea, its
        # viewport (a separate widget the QScrollArea qss rule does NOT
        # propagate to), and the host widget hanging off setWidget().
        scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget {"
            "  background: transparent;"
            "  border: none;"
            "}"
        )
        viewport = scroll.viewport()
        if viewport is not None:
            viewport.setAutoFillBackground(False)
            viewport.setStyleSheet("background: transparent;")
        engines_host = QWidget()
        engines_host.setAutoFillBackground(False)
        engines_host.setStyleSheet("background: transparent;")
        self._engines_layout = QVBoxLayout(engines_host)
        self._engines_layout.setContentsMargins(0, 0, 0, 0)
        self._engines_layout.setSpacing(6)
        scroll.setWidget(engines_host)
        layout.addWidget(scroll, 1)

        self._populate_engines()

        # Wire auth signals AFTER widgets exist. Identity-linking,
        # cloud-sync status, and task-finished signals are owned by the
        # embedded AccountDetailsWidget — no need to re-subscribe here.
        self._auth.authenticated.connect(self._on_authenticated)
        self._auth.signed_out.connect(self._on_signed_out)
        self._auth.avatar_updated.connect(self._on_avatar_updated)
        self._auth.workspace_changed.connect(self._on_workspace_changed)

        # Re-render auth name + button text when language switches. Both
        # widgets bypass SDK i18n_bind (see _build_account_card) precisely
        # so we can decide which translation to apply based on auth state.
        I18n.instance().language_changed.connect(self._retranslate_auth)

        # Seed from cache so the card renders instantly on cold start.
        cached_user = self._auth.cached_user()
        if cached_user is not None:
            self._render_signed_in(cached_user)
        else:
            self._render_signed_out()
        cached_avatar = self._auth.cached_avatar()
        if cached_avatar is not None:
            self._set_avatar(cached_avatar)

    # ----- account card ---------------------------------------------------

    def _build_account_card(self) -> QWidget:
        card = QWidget()
        row = QHBoxLayout(card)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(10)

        self._avatar = QLabel()
        self._avatar.setFixedSize(_AVATAR_PX, _AVATAR_PX)
        self._avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._avatar.setStyleSheet(
            f"background-color: {Config.get_color('bg_3')};"
            f"border: 1px solid {Config.get_color('border_1')};"
            f"border-radius: {_AVATAR_PX // 2}px;"
        )
        row.addWidget(self._avatar)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        sz = int(Config.get_font_size("size_small"))
        # _name_label 和 _auth_btn 的文本完全由登录态决定（user.label vs
        # i18n hint / "Sign in" vs "Sign out"），不能让 SDK 的 i18n_bind 在
        # language_changed 时把它们覆写回某一个固定 key。所以两者都用「跳过
        # i18n 绑定」的构造形态：setText(id="") / setButton(icon_only=True)。
        # 后续 _retranslate_auth() 在 language_changed 时按当前态重写文本。
        self._name_label = setText("", default="Sign in to UnitPort",
                                   kind="title", size=sz)

        # Sign-in/out button lives on the SAME row as the username (right-
        # aligned, compact) so it never crowds the email subtitle below.
        # width=0 lets it size to its text; padding 2/2 is the visual
        # contract the user asked for. The base padding from
        # LaviButton._build_qss is 4px 12px and gets re-applied each
        # spec switch, so _set_auth_button_spec re-overrides it.
        self._auth_btn = setButton(
            "auth.sidebar_sign_in_button", 0, 22,
            kind="normal", spec="save",
            default="Sign in",
            icon_only=True,                # 跳过 SDK 的文本 i18n_bind
        )
        self._auth_btn.clicked.connect(self._on_auth_button)
        self._apply_auth_btn_compact_padding()

        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(6)
        name_row.addWidget(self._name_label, 1)
        name_row.addWidget(
            self._auth_btn, 0, Qt.AlignmentFlag.AlignRight,
        )
        text_col.addLayout(name_row)

        # Subtitle holds the Supabase UUID once the user is signed in
        # (cloud-storage namespace key; replaces the previous email shown
        # here so cloud-side prefix == local FS slug). Selectable so the
        # user can copy it for support tickets; full UUID rendered in the
        # tooltip even when the visible text is elided.
        self._subtitle = setText("", default="", kind="content", size=sz)
        self._subtitle.setVisible(False)
        self._subtitle.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        text_col.addWidget(self._subtitle)
        row.addLayout(text_col, 1)

        return card

    def _apply_auth_btn_compact_padding(self) -> None:
        """Re-apply the compact padding + radius override that
        ``refresh_style`` wipes.

        LaviButton.refresh_style() regenerates the stylesheet on every
        spec switch (save -> danger when signing out) and reverts to
        ``padding: 4px 12px`` + ``border-radius: 6px``. We re-append
        our overrides every time the spec changes (see
        ``_set_auth_button_spec``) and once at construction.
        """
        existing = self._auth_btn.styleSheet() or ""
        self._auth_btn.setStyleSheet(
            existing
            + " QPushButton { padding: 2px 8px; border-radius: 4px; }"
        )

    # ----- workspace row --------------------------------------------------

    def _build_workspace_row(self) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(0)

        # LaviButton normal-kind; spans the full sidebar width. Clicking
        # reveals the live USER_CONFIG_DIR in the OS file explorer.
        # ``width=0`` lets the button defer to its size policy, which we
        # flip to Expanding so it fills the row.
        self._workspace_btn = setButton(
            "user.open_workspace", 0, 28,
            kind="normal", spec="none",
            default="Open User Workspace",
        )
        self._workspace_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        i18n_bind(
            self._workspace_btn, "setToolTip",
            "user.open_workspace_tip",
            "Open the UnitPort user data directory",
        )
        self._workspace_btn.clicked.connect(self._on_open_workspace)
        row.addWidget(self._workspace_btn, 1)

        return host

    def _on_open_workspace(self) -> None:
        path = str(Paths.USER_CONFIG_DIR)
        self._workspace_cached_path = path
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            log_warning(f"[user_panel] failed to open workspace dir: {path}")

    def _populate_engines(self) -> None:
        # Clear any previous rows (safe even on first call).
        while self._engines_layout.count():
            item = self._engines_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        svc = get_engine_service()
        known = svc.list_known_engines()
        if not known:
            # Force one detection pass so the rows show up on first launch.
            svc.refresh()
            known = svc.list_known_engines()
        # Surface only the user-facing engines (mujoco / sb3_mujoco /
        # isaac_lab) in the panel's preferred order, but never invent
        # rows for engines registers.backends has not detected yet.
        engines = [eid for eid in _USER_FACING_ENGINES if eid in known]
        if not engines:
            placeholder = setText(
                "user.engines_empty",
                default="No engines detected. Click any local gear to scan.",
                kind="content",
                size=int(Config.get_font_size("size_small")),
            )
            self._engines_layout.addWidget(placeholder)
        else:
            for eid in engines:
                self._engines_layout.addWidget(_EngineRow(eid))
        self._engines_layout.addStretch(1)

    # ----- auth state -----------------------------------------------------

    def _on_auth_button(self) -> None:
        if self._auth.is_signed_in():
            self._auth.sign_out()
        else:
            dlg = LoginDialog(parent=self)
            dlg.exec()

    @pyqtSlot(object)
    def _on_authenticated(self, user) -> None:
        if isinstance(user, UserProfile):
            self._render_signed_in(user)

    @pyqtSlot()
    def _on_signed_out(self) -> None:
        self._render_signed_out()

    @pyqtSlot(str, QPixmap)
    def _on_avatar_updated(self, _user_id: str, pixmap: QPixmap) -> None:
        self._set_avatar(pixmap)

    @pyqtSlot(str)
    def _on_workspace_changed(self, _new_uid: str) -> None:
        """USER_CONFIG_DIR was hot-switched (login / logout / account swap).

        Refresh anything in this panel that depends on the live workspace:
        the workspace-path label (sidebar shows the live dir) and emit
        ``workspace_changed`` to MainWindow so projects panel re-binds.
        Auth-state visuals (avatar / name / button) are already handled
        by the ``authenticated`` / ``signed_out`` slots.
        """
        from unitport_sdk import Paths
        old_str = self._workspace_cached_path
        new_str = str(Paths.USER_CONFIG_DIR)
        self._workspace_cached_path = new_str
        if old_str and old_str != new_str:
            self.workspace_changed.emit(old_str, new_str)

    def _render_signed_in(self, user: UserProfile) -> None:
        self._name_label.setText(user.label)
        # Subtitle = Supabase UUID (per-user cloud-storage namespace).
        # The full UUID lands in the tooltip as well so a hover reveals
        # the unabbreviated value even when the sidebar is narrow.
        if user.user_id:
            self._subtitle.setText(user.user_id)
            self._subtitle.setToolTip(user.user_id)
            self._subtitle.setVisible(True)
        else:
            self._subtitle.setVisible(False)
        self._auth_btn.setText(tr("auth.sidebar_sign_out", "Sign out"))
        # Sign-out is a destructive-ish action (drops the live session,
        # forces a restart prompt) — surface that with the danger spec.
        self._set_auth_button_spec("danger")
        # Always synthesize an initials pixmap on sign-in so the avatar
        # is visually instant. If a real provider avatar arrives later
        # via avatar_updated(), it will replace the initials pixmap.
        self._set_avatar(make_initials_pixmap(user.user_id, user.label, 128))
        # AccountDetailsWidget self-subscribes to authenticated /
        # signed_out / identities_changed, so its chip + cloud-sync
        # state is already refreshed by the time we get here.

    def _render_signed_out(self) -> None:
        self._name_label.setText(tr("auth.sidebar_signed_out_hint", "Sign in to UnitPort"))
        self._subtitle.setVisible(False)
        self._auth_btn.setText(tr("auth.sidebar_sign_in_button", "Sign in"))
        # Sign-in is the "safe / encouraged" action — green save spec.
        self._set_auth_button_spec("save")
        self._avatar.clear()
        self._avatar_set = False

    def _set_auth_button_spec(self, spec: str) -> None:
        """Flip the sidebar auth button between safe (sign-in) and danger
        (sign-out). The SDK button stores its spec on ``_spec`` and
        re-resolves the palette via ``refresh_style()``."""
        if getattr(self._auth_btn, "_spec", None) == spec:
            return
        self._auth_btn._spec = spec
        if hasattr(self._auth_btn, "refresh_style"):
            self._auth_btn.refresh_style()
        # refresh_style rewrote the stylesheet from scratch — re-append
        # our compact padding so the button keeps the 2px contract.
        self._apply_auth_btn_compact_padding()

    def _retranslate_auth(self, *_args) -> None:
        # 语种切换：重写 auth 卡片里两块「态相关」文本。其余区域（engines list
        # 标题、按钮 tooltip、workspace tooltip、子行 EngineRow 内的状态字串）
        # 都已经走 i18n_bind 自动重译，不必这里管。
        cached = self._auth.cached_user()
        if cached is not None:
            self._auth_btn.setText(tr("auth.sidebar_sign_out", "Sign out"))
            # 用户名本身（cached.label）不翻译，沿用即可。
        else:
            self._name_label.setText(
                tr("auth.sidebar_signed_out_hint", "Sign in to UnitPort")
            )
            self._auth_btn.setText(tr("auth.sidebar_sign_in_button", "Sign in"))

    def _set_avatar(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            return
        rounded = round_pixmap(pixmap, _AVATAR_PX)
        self._avatar.setPixmap(rounded)
        self._avatar_set = True


__all__ = ["UserPanel"]
