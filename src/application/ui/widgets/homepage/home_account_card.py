# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HomeAccountCard — the Account tile on the homepage grid.

A single class owns the entire homepage Account experience so future
restyles are localised. Two states share the card chrome:

* **Signed-out**: three OAuth CTAs (Google, GitHub, Email) stacked
  vertically. Identical to the legacy ``UserCard`` signed-out state.
* **Signed-in**: a two-column horizontal layout split by a 1px vertical
  border. The left column hosts the account identity (avatar, name,
  UUID) and the three provider chips stacked top-to-bottom. The right
  column hosts the Cloud Sync section, the Auto-push slider, a stretch,
  and the Sign-out button at the bottom.

The chips on the left are custom :class:`_ProviderChip` widgets — they
mimic the LaviButton ``normal`` style (btn_1 fill, hover_1 hover, 6px
radius) but carry two color regions so the bound email/handle text can
render in ``safe_zone`` green when the provider is linked.

The card self-subscribes to ``AuthManager``, ``CloudSyncService``, and
``TaskSignal`` — there is no shared widget dependency, which keeps the
homepage layout independent from the sidebar UserPanel evolution.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QGuiApplication, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    I18n,
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
from application.service.auth.auth_manager import UserProfile
from application.service.auth.avatar_cache import make_initials_pixmap
from application.service.cloud_sync import get_cloud_sync_service
from application.tools.cloud_sync_task import CloudSyncTask

from .card_base import HomepageCard


_AVATAR_PX = 64
_LINK_PROVIDERS = ("email", "google", "github")
_PROVIDER_DISPLAY = {"email": "Email", "google": "Google", "github": "GitHub"}


class _AspectPixmapLabel(QLabel):
    """QLabel that rescales its source pixmap to current size on resize.

    The label expands in both axes; the pixmap is re-scaled with
    ``KeepAspectRatio`` so it fills as much of the assigned area as
    possible without distortion.
    """

    def __init__(self, source: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source = source
        self.setMinimumSize(1, 1)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resizeEvent(self, ev) -> None:  # noqa: N802
        if not self._source.isNull():
            super().setPixmap(self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        super().resizeEvent(ev)


class _ProviderChip(QFrame):
    """LaviButton-styled clickable chip with a two-color text region.

    The chip renders ``[icon] [provider]: [bound]`` where the bound
    region is recolored to ``safe_zone`` green when the provider is
    linked. The whole frame is clickable; ``clicked`` fires on a left
    mouse press release.
    """

    clicked = pyqtSignal()

    def __init__(
        self,
        provider: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self.setObjectName("homeAccountChip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setProperty("provider", provider)
        self.setProperty("isLinked", "false")
        self.setProperty("isPrimary", "false")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setMinimumHeight(42)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        self._icon = QLabel(self)
        self._icon.setObjectName("homeAccountChipIcon")
        self._icon.setFixedSize(16, 16)
        icon_path = Assets.find_icon(f"icon_{provider}")
        if icon_path is not None:
            pm = QPixmap(str(icon_path))
            self._icon.setPixmap(pm.scaled(
                16, 16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            log_warning(
                f"[home_account_card] provider icon missing: icon_{provider}"
            )
        row.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignVCenter)

        self._prefix = QLabel(_PROVIDER_DISPLAY.get(provider, provider), self)
        self._prefix.setObjectName("homeAccountChipPrefix")
        row.addWidget(self._prefix, 0, Qt.AlignmentFlag.AlignVCenter)

        self._bound = QLabel("", self)
        self._bound.setObjectName("homeAccountChipBound")
        self._bound.setProperty("isLinked", "false")
        self._bound.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )
        row.addWidget(self._bound, 1, Qt.AlignmentFlag.AlignVCenter)

    def set_state(
        self,
        *,
        linked: bool,
        primary: bool,
        bound_text: str,
        enabled: bool,
    ) -> None:
        self._bound.setText(bound_text)
        self._bound.setToolTip(bound_text)
        self.setProperty("isLinked", "true" if linked else "false")
        self.setProperty("isPrimary", "true" if primary else "false")
        self._bound.setProperty("isLinked", "true" if linked else "false")
        self.setEnabled(enabled)
        for w in (self, self._bound, self._prefix):
            st = w.style()
            if st is not None:
                st.unpolish(w)
                st.polish(w)

    def mousePressEvent(self, ev) -> None:  # noqa: D401, N802
        if (
            ev.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
        ):
            self._pressed = True
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: D401, N802
        if (
            ev.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
            and self.rect().contains(ev.pos())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(ev)


class HomeAccountCard(HomepageCard):
    """Top-left homepage tile. Owns its own auth + cloud-sync wiring."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            title_key="homepage.user.title",
            title_default="Account",
            parent=parent,
        )
        # Card chrome — slightly wider horizontal padding than the
        # default HomepageCard so the two-column split (avatar / chips
        # on the left, cloud sync / sign-out on the right) has visible
        # margin on both edges. Vertical padding tracks the base card
        # (already generous) so the title row sits at the same height
        # as the other cards in the grid.
        root = self.layout()
        if root is not None:
            root.setContentsMargins(32, 20, 32, 28)

        self._auth = get_auth_manager()

        # ---- chip state ----
        self._chips: dict[str, _ProviderChip] = {}
        self._chips_layout: Optional[QVBoxLayout] = None
        self._methods_prompt: Optional[QLabel] = None
        self._link_status_label: Optional[QLabel] = None
        self._link_status_timer = QTimer(self)
        self._link_status_timer.setSingleShot(True)
        self._link_status_timer.timeout.connect(self._clear_link_status)
        self._last_link_attempt: str = ""

        # ---- cloud-sync state ----
        self._cloud_status_label: Optional[QLabel] = None
        self._cloud_push_btn: Optional[QPushButton] = None
        self._cloud_pull_btn: Optional[QPushButton] = None
        self._cloud_manage_btn: Optional[QPushButton] = None
        self._cloud_auto_switch: Optional[SliderSwitch] = None
        self._cloud_push_task_id: str = ""
        self._cloud_pull_task_id: str = ""
        self._cloud_usage_prefix: Optional[QLabel] = None
        self._cloud_usage_bar: Optional[QProgressBar] = None
        self._cloud_usage_text: Optional[QLabel] = None
        self._cloud_usage_percent: Optional[QLabel] = None
        self._cloud_status_icon: Optional[QLabel] = None
        self._cloud_usage_task_id: str = ""

        # ---- top-level state widgets ----
        self._signed_in_widget: Optional[QWidget] = None
        self._signed_out_widget: Optional[QWidget] = None
        self._avatar_label: Optional[QLabel] = None
        # The username label was removed from the identity row when the
        # welcome banner above the card body absorbed the name. The
        # attribute is kept as None so legacy callers / retranslate paths
        # don't NoneType-explode if they still reach for it.
        self._name_label: Optional[QLabel] = None
        self._welcome_line: Optional[QLabel] = None
        self._welcome_sub: Optional[QLabel] = None
        self._uuid_label: Optional[QLabel] = None
        self._uuid_copy_btn: Optional[QPushButton] = None
        self._btn_signout: Optional[QPushButton] = None

        self._build_signed_out()
        self._build_signed_in()

        # ---- signal wiring ----
        self._auth.authenticated.connect(self._on_authenticated)
        self._auth.signed_out.connect(self._on_signed_out)
        self._auth.avatar_updated.connect(self._on_avatar_updated)
        self._auth.workspace_changed.connect(self._on_workspace_changed)
        self._auth.identities_changed.connect(self._on_identities_changed)
        self._auth.identity_linked.connect(self._on_identity_linked)
        self._auth.identity_unlinked.connect(self._on_identity_unlinked)
        self._auth.auth_error.connect(self._on_link_auth_error)
        self._auth.info_message.connect(self._on_link_info_message)
        get_cloud_sync_service().status_changed.connect(
            self._on_cloud_status_changed
        )
        get_cloud_sync_service().usage_changed.connect(
            self._on_storage_usage_changed
        )
        get_task_signal().task_finished.connect(self._on_cloud_task_finished)
        I18n.instance().language_changed.connect(self._retranslate_state)

        # ---- seed from cache ----
        cached = self._auth.cached_user()
        if cached is not None:
            self._render_signed_in(cached)
        else:
            self._render_signed_out()
        cached_av = self._auth.cached_avatar()
        if cached_av is not None:
            self._set_avatar_pixmap(cached_av)
        self._refresh_chips()
        self._refresh_cloud_status_text()
        cached_usage = get_cloud_sync_service().cached_usage()
        if cached_usage is not None:
            used, total = cached_usage
            self._render_storage_usage(used, total)
        if cached is not None:
            self._maybe_kick_usage_refresh()

    # ==================================================================
    # State builders
    # ==================================================================

    # ---- signed out --------------------------------------------------

    def _build_signed_out(self) -> None:
        host = QWidget(self)
        host.setObjectName("homeAccountSignedOut")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._headline = QLabel(tr(
            "homepage.user.headline_signed_out",
            "Welcome to UnitPort Studio",
        ))
        self._headline.setObjectName("homeAccountHeadline")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._sub = QLabel(tr(
            "homepage.user.sub_signed_out",
            "Sign in to your account to sync projects and unlock cloud features.",
        ))
        self._sub.setObjectName("homeAccountSub")
        self._sub.setWordWrap(True)
        layout.addWidget(self._sub)

        layout.addSpacing(6)

        # Two-column body mirrors the signed-in card: illustration on
        # the left (fills the column, aspect-preserving), 1px divider in
        # the middle, prompt + the three OAuth CTAs stacked on the right
        # (CTAs fill the right column's width).
        mid_row = QHBoxLayout()
        mid_row.setContentsMargins(0, 0, 0, 0)
        mid_row.setSpacing(20)

        pic_path = Assets.find("pictures/welcome_rs.png")
        if pic_path is not None:
            pm = QPixmap(str(pic_path))
        else:
            log_warning(
                "[home_account_card] welcome picture missing: "
                "pictures/welcome_rs.png"
            )
            pm = QPixmap()
        self._welcome_pic = _AspectPixmapLabel(pm, host)
        self._welcome_pic.setObjectName("homeAccountWelcomePic")
        mid_row.addWidget(self._welcome_pic, 1)

        self._signed_out_divider = self._build_vertical_divider(host)
        mid_row.addWidget(self._signed_out_divider, 0)

        right_host = QWidget(host)
        right_host.setObjectName("homeAccountSignedOutRight")
        right_col = QVBoxLayout(right_host)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(12)

        self._signin_prompt = QLabel(tr(
            "homepage.user.signin_prompt",
            "Sign In / Join in",
        ), right_host)
        self._signin_prompt.setObjectName("homeAccountSignInPrompt")
        self._signin_prompt.setWordWrap(True)
        self._signin_prompt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_col.addWidget(self._signin_prompt)

        # The three CTAs are a fixed-height group centered vertically in
        # the space below the prompt — top stretch + buttons + bottom
        # stretch. Each button keeps a comfortable 44px height so the
        # icon + label sit airy rather than cramped, matching the
        # homepage.png reference's account card.
        right_col.addStretch(1)

        self._btn_google = self._make_signin_button(
            "homepage.user.btn_google", "Continue with Google", "icon_google"
        )
        self._btn_google.clicked.connect(lambda: self._start_oauth("google"))
        right_col.addWidget(self._btn_google)

        self._btn_github = self._make_signin_button(
            "homepage.user.btn_github", "Continue with GitHub", "icon_github"
        )
        self._btn_github.clicked.connect(lambda: self._start_oauth("github"))
        right_col.addWidget(self._btn_github)

        self._btn_email = self._make_signin_button(
            "homepage.user.btn_email", "Continue with Email", "icon_email"
        )
        self._btn_email.clicked.connect(self._open_email_dialog)
        right_col.addWidget(self._btn_email)

        right_col.addStretch(1)

        mid_row.addWidget(right_host, 1)

        # mid_row owns all remaining vertical space — picture stretches
        # to fill the column height, no dead area below the buttons.
        layout.addLayout(mid_row, 1)

        self._signed_out_widget = host
        self.content_layout.addWidget(host)
        self._apply_signin_button_theme()

    def _make_signin_button(
        self, key: str, default: str, icon_name: str,
    ) -> QPushButton:
        btn = QPushButton(tr(key, default))
        btn.setObjectName("homeAccountSignInBtn")
        btn.setFixedHeight(44)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        ico = Assets.find_icon(icon_name)
        if ico is not None:
            btn.setIcon(QIcon(str(ico)))
            btn.setIconSize(QSize(20, 20))
        btn.setProperty("home_i18n_key", key)
        btn.setProperty("home_i18n_default", default)
        return btn

    # ---- signed in ---------------------------------------------------

    def _build_signed_in(self) -> None:
        host = QWidget(self)
        host.setObjectName("homeAccountSignedIn")
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)

        # Welcome banner — replaces the card's title-row text while
        # signed in. Two stacked labels (greeting + subtitle). The
        # greeting is rich-text so the username can be tinted with the
        # ``highlight`` slot without dropping out of the run on emoji.
        banner_host = QWidget(host)
        banner_host.setObjectName("homeAccountWelcomeBanner")
        banner_col = QVBoxLayout(banner_host)
        banner_col.setContentsMargins(0, 0, 0, 0)
        banner_col.setSpacing(2)
        self._welcome_line = QLabel("", banner_host)
        self._welcome_line.setObjectName("homeAccountWelcomeLine")
        self._welcome_line.setTextFormat(Qt.TextFormat.RichText)
        self._welcome_line.setWordWrap(False)
        banner_col.addWidget(self._welcome_line)
        self._welcome_sub = QLabel(
            tr(
                "homepage.user.welcome_sub",
                "Manage your account and sync your projects across devices.",
            ),
            banner_host,
        )
        self._welcome_sub.setObjectName("homeAccountWelcomeSub")
        self._welcome_sub.setWordWrap(True)
        banner_col.addWidget(self._welcome_sub)
        outer.addWidget(banner_host, 0)

        # Two-column body — same shape as before, but now nested inside
        # the outer vertical layout under the banner.
        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        # Generous gap between the two columns so the 1px divider has
        # breathing room on either side.
        body_row.setSpacing(20)
        body_row.addWidget(self._build_left_column(host), 1)
        body_row.addWidget(self._build_vertical_divider(host), 0)
        body_row.addWidget(self._build_right_column(host), 1)
        outer.addLayout(body_row, 1)

        self._signed_in_widget = host
        self.content_layout.addWidget(host)
        host.setVisible(False)
        self._apply_signed_in_theme()

    def _build_left_column(self, parent: QWidget) -> QWidget:
        col_host = QWidget(parent)
        col_host.setObjectName("homeAccountLeftCol")
        col = QVBoxLayout(col_host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        sz_small = int(Config.get_font_size("size_small"))

        # ---- Identity row: avatar (left) + 2-row stack (right) ----
        # The right stack holds two lines: [Cloud Usage + MB/MB] and
        # [progress bar + percentage]. The username has been promoted to
        # the welcome banner above the card body, so the avatar pairs
        # only with the cloud-usage block here.
        identity_row = QHBoxLayout()
        identity_row.setContentsMargins(0, 0, 0, 0)
        identity_row.setSpacing(14)

        self._avatar_label = QLabel(col_host)
        self._avatar_label.setFixedSize(_AVATAR_PX, _AVATAR_PX)
        self._avatar_label.setObjectName("homeAccountAvatar")
        self._avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._avatar_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        identity_row.addWidget(
            self._avatar_label, 0, Qt.AlignmentFlag.AlignVCenter
        )

        right_stack_host = QWidget(col_host)
        right_stack_host.setObjectName("homeAccountIdentityRight")
        right_stack_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        right_stack = QVBoxLayout(right_stack_host)
        right_stack.setContentsMargins(0, 0, 0, 0)
        right_stack.setSpacing(6)
        right_stack.addStretch(1)

        # Row 1: "Cloud Usage" label (left) + "X.X MB / Y.Y MB" (right).
        usage_label_row = QHBoxLayout()
        usage_label_row.setContentsMargins(0, 0, 0, 0)
        usage_label_row.setSpacing(8)
        self._cloud_usage_prefix = setText(
            "homepage.user.cloud_usage_label",
            default="Cloud Usage",
            kind="content",
            size=sz_small,
        )
        usage_label_row.addWidget(
            self._cloud_usage_prefix, 0, Qt.AlignmentFlag.AlignVCenter
        )
        usage_label_row.addStretch(1)
        self._cloud_usage_text = QLabel("—", right_stack_host)
        self._cloud_usage_text.setObjectName("homeAccountCloudUsageText")
        usage_label_row.addWidget(
            self._cloud_usage_text, 0, Qt.AlignmentFlag.AlignVCenter
        )
        right_stack.addLayout(usage_label_row)

        # Row 2: progress bar (stretches) + "13%" suffix.
        usage_bar_row = QHBoxLayout()
        usage_bar_row.setContentsMargins(0, 0, 0, 0)
        usage_bar_row.setSpacing(8)
        self._cloud_usage_bar = QProgressBar(right_stack_host)
        self._cloud_usage_bar.setObjectName("homeAccountCloudUsageBar")
        self._cloud_usage_bar.setRange(0, 1000)
        self._cloud_usage_bar.setValue(0)
        self._cloud_usage_bar.setTextVisible(False)
        self._cloud_usage_bar.setFixedHeight(6)
        self._cloud_usage_bar.setMinimumWidth(40)
        self._cloud_usage_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        usage_bar_row.addWidget(
            self._cloud_usage_bar, 1, Qt.AlignmentFlag.AlignVCenter
        )
        self._cloud_usage_percent = QLabel("—", right_stack_host)
        self._cloud_usage_percent.setObjectName(
            "homeAccountCloudUsagePercent"
        )
        self._cloud_usage_percent.setMinimumWidth(32)
        self._cloud_usage_percent.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        usage_bar_row.addWidget(
            self._cloud_usage_percent, 0, Qt.AlignmentFlag.AlignVCenter
        )
        right_stack.addLayout(usage_bar_row)
        right_stack.addStretch(1)

        identity_row.addWidget(right_stack_host, 1)
        col.addLayout(identity_row)

        # ---- UUID row (text + right-aligned copy button) ----
        uuid_row = QHBoxLayout()
        uuid_row.setContentsMargins(0, 0, 0, 0)
        uuid_row.setSpacing(6)
        self._uuid_label = QLabel("", col_host)
        self._uuid_label.setObjectName("homeAccountUuid")
        self._uuid_label.setWordWrap(True)
        self._uuid_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        uuid_row.addWidget(self._uuid_label, 1, Qt.AlignmentFlag.AlignVCenter)
        self._uuid_copy_btn = QPushButton(col_host)
        self._uuid_copy_btn.setObjectName("homeAccountUuidCopy")
        self._uuid_copy_btn.setFixedSize(24, 24)
        self._uuid_copy_btn.setFlat(True)
        self._uuid_copy_btn.setCursor(
            QCursor(Qt.CursorShape.PointingHandCursor)
        )
        copy_ico = Assets.find_icon("icon_copy")
        if copy_ico is not None:
            self._uuid_copy_btn.setIcon(QIcon(str(copy_ico)))
            self._uuid_copy_btn.setIconSize(QSize(14, 14))
        else:
            log_warning(
                "[home_account_card] copy icon missing: icon_copy"
            )
        i18n_bind(
            self._uuid_copy_btn, "setToolTip",
            "homepage.user.uuid_copy_tooltip", "Copy UUID",
        )
        self._uuid_copy_btn.clicked.connect(self._on_uuid_copy_clicked)
        uuid_row.addWidget(
            self._uuid_copy_btn, 0, Qt.AlignmentFlag.AlignVCenter
        )
        col.addLayout(uuid_row)

        col.addSpacing(6)

        # ---- Auto-push (moved above Cloud Sync row) ----
        # Auto-push is a *preference* toggle (off-by-default) that
        # governs the implicit push on exit. Cloud Sync's manual Push /
        # Pull / Manage actions sit underneath, so the explicit actions
        # follow the implicit preference visually.
        autopush_row = QHBoxLayout()
        autopush_row.setContentsMargins(0, 0, 0, 0)
        autopush_row.setSpacing(10)
        autopush_row.addWidget(setText(
            "cloud.auto_push_label", default="Auto-push",
            kind="content", size=sz_small,
        ), 0, Qt.AlignmentFlag.AlignVCenter)
        self._cloud_auto_switch = SliderSwitch(
            options=[("off", "Off"), ("on", "On")],
            height=24,
            min_segment_width=44,
            font_size=10,
        )
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
        autopush_row.addWidget(
            self._cloud_auto_switch, 0, Qt.AlignmentFlag.AlignVCenter
        )
        autopush_row.addStretch(1)
        col.addLayout(autopush_row)

        # Small breathing room between the auto-push preference and the
        # manual cloud-sync actions below.
        col.addSpacing(4)

        # ---- Cloud Sync header + Push/Pull/Manage buttons (one row) ----
        cloud_row = QHBoxLayout()
        cloud_row.setContentsMargins(0, 0, 0, 0)
        cloud_row.setSpacing(8)
        cloud_row.addWidget(setText(
            "cloud.section_title", default="Cloud Sync",
            kind="content", size=sz_small,
        ), 0, Qt.AlignmentFlag.AlignVCenter)
        self._cloud_push_btn = setButton(
            "cloud.push_button", 0, 28,
            kind="border", spec="save",
            default="Push",
        )
        i18n_bind(
            self._cloud_push_btn, "setToolTip",
            "cloud.push_tooltip",
            "Upload local workspace files to the cloud",
        )
        self._cloud_push_btn.setCursor(
            QCursor(Qt.CursorShape.PointingHandCursor)
        )
        self._cloud_push_btn.clicked.connect(self._on_cloud_push_clicked)
        cloud_row.addWidget(self._cloud_push_btn)
        self._cloud_pull_btn = setButton(
            "cloud.pull_button", 0, 28,
            kind="border", spec="notice",
            default="Pull",
        )
        i18n_bind(
            self._cloud_pull_btn, "setToolTip",
            "cloud.pull_tooltip",
            "Download cloud files into the local workspace",
        )
        self._cloud_pull_btn.setCursor(
            QCursor(Qt.CursorShape.PointingHandCursor)
        )
        self._cloud_pull_btn.clicked.connect(self._on_cloud_pull_clicked)
        cloud_row.addWidget(self._cloud_pull_btn)
        self._cloud_manage_btn = setButton(
            "cloud.manage_button", 0, 28,
            kind="border", spec="none",
            default="Manage",
        )
        i18n_bind(
            self._cloud_manage_btn, "setToolTip",
            "cloud.manage_tooltip",
            "Manage cloud sync settings and stored files",
        )
        self._cloud_manage_btn.setCursor(
            QCursor(Qt.CursorShape.PointingHandCursor)
        )
        self._cloud_manage_btn.clicked.connect(self._on_cloud_manage_clicked)
        cloud_row.addWidget(self._cloud_manage_btn)
        cloud_row.addStretch(1)
        col.addLayout(cloud_row)

        # ---- Synced status (clock icon + status text) ----
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        self._cloud_status_icon = QLabel(col_host)
        self._cloud_status_icon.setObjectName("homeAccountCloudStatusIcon")
        self._cloud_status_icon.setFixedSize(14, 14)
        clock_ico = Assets.find_icon("icon_clock")
        if clock_ico is not None:
            self._cloud_status_icon.setPixmap(QPixmap(str(clock_ico)).scaled(
                14, 14,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            log_warning(
                "[home_account_card] clock icon missing: icon_clock"
            )
        status_row.addWidget(
            self._cloud_status_icon, 0, Qt.AlignmentFlag.AlignVCenter
        )
        self._cloud_status_label = QLabel("", col_host)
        self._cloud_status_label.setObjectName("homeAccountCloudStatus")
        self._cloud_status_label.setWordWrap(True)
        status_row.addWidget(
            self._cloud_status_label, 1, Qt.AlignmentFlag.AlignVCenter
        )
        col.addLayout(status_row)

        col.addStretch(1)
        return col_host

    def _build_vertical_divider(self, parent: QWidget) -> QWidget:
        line = QFrame(parent)
        line.setObjectName("homeAccountDivider")
        line.setFrameShape(QFrame.Shape.NoFrame)
        line.setFixedWidth(1)
        line.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        return line

    def _build_right_column(self, parent: QWidget) -> QWidget:
        # Mirrors the signed-out right column: a left-aligned prompt at
        # the top, a vertically centered group of three CTAs (here the
        # link-status chips), and the destructive sign-out at the bottom.
        col_host = QWidget(parent)
        col_host.setObjectName("homeAccountRightCol")
        col = QVBoxLayout(col_host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        self._methods_prompt = QLabel(tr(
            "homepage.user.methods_prompt", "Sign-in methods",
        ), col_host)
        self._methods_prompt.setObjectName("homeAccountSignInPrompt")
        self._methods_prompt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        col.addWidget(self._methods_prompt)

        col.addStretch(1)

        # Chips column — order is rewritten in _refresh_chips so the
        # currently-active provider sits at the top.
        self._chips_layout = QVBoxLayout()
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(10)
        for provider in _LINK_PROVIDERS:
            chip = _ProviderChip(provider, parent=col_host)
            chip.clicked.connect(
                lambda p=provider: self._on_chip_clicked(p)
            )
            self._chips[provider] = chip
            self._chips_layout.addWidget(chip)
        col.addLayout(self._chips_layout)

        self._link_status_label = QLabel("", col_host)
        self._link_status_label.setObjectName("homeAccountLinkStatus")
        self._link_status_label.setWordWrap(True)
        self._link_status_label.setVisible(False)
        col.addWidget(self._link_status_label)

        col.addStretch(1)

        self._btn_signout = setButton(
            "homepage.user.sign_out", 0, 30,
            kind="border", spec="danger",
            default="Sign out",
        )
        self._btn_signout.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_signout.clicked.connect(self._on_sign_out_clicked)
        col.addWidget(self._btn_signout, 0, Qt.AlignmentFlag.AlignRight)
        return col_host

    def _on_auto_push_changed(self, _idx: int, key: str) -> None:
        """Persist the Auto-sync toggle to ``user.ini[Cloud] auto_push``.

        When on, the AutoSyncController pushes each include-set file to the
        cloud shortly after it is saved (persist-time, debounced) — there is no
        batch upload on exit. This slot only stores the preference, so the
        widget itself stays inexpensive to toggle.
        """
        Config.set_value("Cloud", "auto_push", key == "on")
        log_info(f"[home_account_card] auto-sync set to {key!r}")

    # ==================================================================
    # Rendering
    # ==================================================================

    def _render_signed_in(self, user: UserProfile) -> None:
        if self._signed_out_widget is not None:
            self._signed_out_widget.setVisible(False)
        if self._signed_in_widget is not None:
            self._signed_in_widget.setVisible(True)
        # Hide the card's stock title-row — the in-body welcome banner
        # now serves as the heading and the 32px title slot would be
        # dead chrome above it. Reverse it in _render_signed_out so the
        # signed-out state still reads as "Account".
        self._title_label.setVisible(False)
        name = user.display_name or (
            user.email.split("@", 1)[0] if user.email else "User"
        )
        if self._welcome_line is not None:
            highlight = Config.get_color("highlight")
            # Username tinted with highlight; surrounding text + emoji
            # stay in the inherited main_t1 colour. RichText so the run
            # tinting works inline without an extra row of widgets.
            template = tr(
                "homepage.user.welcome_back", "Welcome back, {name}",
            )
            from html import escape as _esc
            safe = _esc(name)
            # The i18n template is trusted (factory + user locale); only
            # the username value is escaped because it can carry user
            # input characters like <, >, &.
            try:
                rendered = template.replace(
                    "{name}",
                    f'<span style="color:{highlight};">{safe}</span>',
                )
            except Exception:                                          # noqa: BLE001
                rendered = f'Welcome back, <span style="color:{highlight};">{safe}</span>'
            self._welcome_line.setText(f"{rendered} 👋")
            self._welcome_line.setToolTip(name)
        if self._welcome_sub is not None:
            self._welcome_sub.setText(tr(
                "homepage.user.welcome_sub",
                "Manage your account and sync your projects across devices.",
            ))
        if self._uuid_label is not None:
            uid = user.user_id or ""
            self._uuid_label.setText(
                tr("homepage.user.uuid_label", "UUID: {uuid}").format(uuid=uid)
                if uid else ""
            )
            self._uuid_label.setToolTip(uid)
        if (
            self._avatar_label is not None
            and not self._avatar_label.pixmap()
        ):
            self._set_avatar_pixmap(
                make_initials_pixmap(
                    user.user_id, name or user.email, _AVATAR_PX * 2,
                )
            )
        self._refresh_chips()
        self._refresh_cloud_status_text()

    def _render_signed_out(self) -> None:
        if self._signed_in_widget is not None:
            self._signed_in_widget.setVisible(False)
        if self._signed_out_widget is not None:
            self._signed_out_widget.setVisible(True)
        # Restore the default card title so the empty state still reads
        # as "Account" rather than the personalised welcome banner.
        self._title_label.setVisible(True)
        self.set_title("homepage.user.title", "Account")
        self._refresh_chips()
        self._refresh_cloud_status_text()

    def _set_avatar_pixmap(self, pm: QPixmap) -> None:
        if self._avatar_label is None or pm.isNull():
            return
        scaled = pm.scaled(
            _AVATAR_PX, _AVATAR_PX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._avatar_label.setPixmap(scaled)

    # ==================================================================
    # Auth signal handlers
    # ==================================================================

    @pyqtSlot(object)
    def _on_authenticated(self, user: object) -> None:
        if isinstance(user, UserProfile):
            self._render_signed_in(user)
            self._maybe_kick_usage_refresh()

    @pyqtSlot()
    def _on_signed_out(self) -> None:
        if self._avatar_label is not None:
            self._avatar_label.clear()
        self._render_storage_usage(0, 0)
        self._render_signed_out()

    def _on_avatar_updated(self, _user_id: str, pm: object) -> None:
        if isinstance(pm, QPixmap):
            self._set_avatar_pixmap(pm)

    def _on_workspace_changed(self, _uid: str) -> None:
        user = self._auth.current_user() or self._auth.cached_user()
        if user is not None and self._auth.is_signed_in():
            self._render_signed_in(user)
        else:
            self._render_signed_out()

    @pyqtSlot(object)
    def _on_identities_changed(self, _identities: object) -> None:
        self._refresh_chips()

    @pyqtSlot(str)
    def _on_identity_linked(self, provider: str) -> None:
        display = _PROVIDER_DISPLAY.get(provider, provider.capitalize())
        self._show_link_status(f"{display}: linked.", is_error=False)
        self._refresh_chips()

    @pyqtSlot(str)
    def _on_identity_unlinked(self, provider: str) -> None:
        display = _PROVIDER_DISPLAY.get(provider, provider.capitalize())
        self._show_link_status(f"{display}: unlinked.", is_error=False)
        self._refresh_chips()

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
            _PROVIDER_DISPLAY.get(provider, provider.capitalize())
            if provider else "the provider"
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
                f"[home_account_card] conflict resolution: signing out + "
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

    # ==================================================================
    # Provider chip refresh + click routing
    # ==================================================================

    def _refresh_chips(self) -> None:
        if not self._chips:
            return
        identities = self._auth.current_identities()
        by_provider = {
            str(i.get("provider")): i
            for i in (identities or [])
            if isinstance(i, dict)
        }
        signed_in = self._auth.is_signed_in()
        current_user = self._auth.current_user() if signed_in else None
        current_provider = current_user.provider if current_user else ""

        for provider, chip in self._chips.items():
            idn = by_provider.get(provider)
            email_always_linked = (
                provider == "email"
                and idn is None
                and signed_in
                and current_user is not None
                and bool(current_user.email)
            )
            if not signed_in:
                bound_text = tr(
                    f"user.linked_account_bound_signedout_{provider}",
                    default="—",
                )
                is_linked = False
                enabled = False
            elif idn is None and not email_always_linked:
                bound_text = tr(
                    f"user.linked_account_bound_unlinked_{provider}",
                    default="(not linked)",
                )
                is_linked = False
                enabled = True
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
                bound_text = email
                is_linked = True
                enabled = True
            chip.set_state(
                linked=is_linked,
                primary=signed_in and (provider == current_provider),
                bound_text=bound_text,
                enabled=enabled,
            )

        # Reorder so the currently-active provider is at the top of the
        # chip stack. When signed out, fall back to the canonical order.
        if self._chips_layout is not None:
            for prov in _LINK_PROVIDERS:
                ch = self._chips.get(prov)
                if ch is not None:
                    self._chips_layout.removeWidget(ch)
            primary = (
                current_provider
                if signed_in and current_provider in self._chips
                else ""
            )
            order = []
            if primary:
                order.append(primary)
            order.extend(p for p in _LINK_PROVIDERS if p != primary)
            for prov in order:
                ch = self._chips.get(prov)
                if ch is not None:
                    self._chips_layout.addWidget(ch)

    def _on_chip_clicked(self, provider: str) -> None:
        log_info(f"[home_account_card] chip clicked: {provider!r}")
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
            display = _PROVIDER_DISPLAY.get(provider, provider.capitalize())
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

    # ==================================================================
    # Cloud Sync handlers
    # ==================================================================

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

    @pyqtSlot(int, int)
    def _on_storage_usage_changed(self, used: int, total: int) -> None:
        self._render_storage_usage(used, total)

    def _render_storage_usage(self, used: int, total: int) -> None:
        if self._cloud_usage_bar is None or self._cloud_usage_text is None:
            return
        if total <= 0:
            self._cloud_usage_bar.setValue(0)
            self._cloud_usage_text.setText("—")
            self._cloud_usage_text.setToolTip("")
            if self._cloud_usage_percent is not None:
                self._cloud_usage_percent.setText("—")
            return
        ratio = max(0.0, min(1.0, float(used) / float(total)))
        self._cloud_usage_bar.setValue(int(ratio * 1000))
        used_mb = float(used) / 1024.0 / 1024.0
        total_mb = float(total) / 1024.0 / 1024.0
        text = f"{used_mb:.1f} MB / {total_mb:.1f} MB"
        self._cloud_usage_text.setText(text)
        self._cloud_usage_text.setToolTip(text)
        if self._cloud_usage_percent is not None:
            self._cloud_usage_percent.setText(f"{int(round(ratio * 100))}%")

    def _maybe_kick_usage_refresh(self) -> None:
        """Submit a one-shot CloudSyncTask('usage') if none is in flight.

        Called when the card first renders signed-in (cache hit at boot
        or on ``authenticated``). The task itself updates the cache and
        emits ``usage_changed``; this method is purely a fire-and-forget
        kick — failures fall through silently and the placeholder stays.
        """
        if self._cloud_usage_task_id:
            return
        if not self._auth.is_signed_in():
            return
        try:
            self._cloud_usage_task_id = get_tasks_manager().submit(
                CloudSyncTask("usage")
            )
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[home_account_card] usage refresh submit failed: {exc}")
            self._cloud_usage_task_id = ""

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
            log_warning(f"[home_account_card] cloud push submit failed: {exc}")
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
            log_warning(f"[home_account_card] cloud pull submit failed: {exc}")
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
        if task_id == self._cloud_usage_task_id:
            # Usage-only refresh: nothing to update in the status row.
            # The service already emitted ``usage_changed`` on success;
            # this slot just releases the in-flight guard.
            self._cloud_usage_task_id = ""
            return
        if task_id not in (self._cloud_push_task_id, self._cloud_pull_task_id):
            return
        is_push = task_id == self._cloud_push_task_id
        self._reset_cloud_buttons()
        if not ok:
            log_warning(
                f"[home_account_card] cloud-sync task {task_id!r} failed; "
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
                    f"[home_account_card] post-pull reload failed: {exc}"
                )
            try:
                from application.service.engines import get_engine_service
                get_engine_service().refresh()
            except Exception as exc:                              # noqa: BLE001
                log_warning(
                    f"[home_account_card] post-pull engine refresh failed: "
                    f"{exc}"
                )
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

    # ==================================================================
    # Misc click + retranslate
    # ==================================================================

    def _start_oauth(self, provider: str) -> None:
        try:
            self._auth.sign_in_oauth(provider)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[home_account_card] sign_in_oauth({provider}) failed: "
                f"{exc!r}"
            )

    def _open_email_dialog(self) -> None:
        from application.ui.dialogs.login_dialog import LoginDialog
        try:
            dlg = LoginDialog(self.window())
            dlg.open()
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[home_account_card] LoginDialog open failed: {exc!r}"
            )

    def _on_sign_out_clicked(self) -> None:
        try:
            self._auth.sign_out()
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[home_account_card] sign_out failed: {exc!r}")

    def _on_cloud_manage_clicked(self) -> None:
        from application.ui.dialogs.cloud_manage_dialog import (
            CloudManageDialog,
        )
        try:
            dlg = CloudManageDialog(self.window())
            dlg.exec()
        except Exception as exc:                                  # noqa: BLE001
            log_warning(
                f"[home_account_card] CloudManageDialog failed: {exc!r}"
            )

    def _on_uuid_copy_clicked(self) -> None:
        user = self._auth.current_user() or self._auth.cached_user()
        uid = (user.user_id if user else "") or ""
        if not uid:
            return
        try:
            cb = QGuiApplication.clipboard()
            if cb is not None:
                cb.setText(uid)
        except Exception as exc:                                  # noqa: BLE001
            log_warning(f"[home_account_card] uuid copy failed: {exc!r}")
            return
        self._show_link_status(
            tr("homepage.user.uuid_copied_toast", "UUID copied to clipboard"),
            is_error=False,
        )

    def _retranslate_state(self) -> None:
        # signed-out static labels
        if hasattr(self, "_headline"):
            self._headline.setText(tr(
                "homepage.user.headline_signed_out",
                "Welcome to UnitPort Studio",
            ))
        if hasattr(self, "_sub"):
            self._sub.setText(tr(
                "homepage.user.sub_signed_out",
                "Sign in to your account to sync projects and unlock cloud features.",
            ))
        if hasattr(self, "_signin_prompt"):
            self._signin_prompt.setText(tr(
                "homepage.user.signin_prompt",
                "Sign In / Join in",
            ))
        for btn in (
            getattr(self, "_btn_google", None),
            getattr(self, "_btn_github", None),
            getattr(self, "_btn_email", None),
        ):
            if btn is None:
                continue
            key = btn.property("home_i18n_key") or ""
            default = btn.property("home_i18n_default") or ""
            if key:
                btn.setText(tr(str(key), str(default)))
        # signed-in static labels
        if self._methods_prompt is not None:
            self._methods_prompt.setText(tr(
                "homepage.user.methods_prompt", "Sign-in methods",
            ))
        if self._welcome_sub is not None:
            self._welcome_sub.setText(tr(
                "homepage.user.welcome_sub",
                "Manage your account and sync your projects across devices.",
            ))
        # signed-in: re-render the welcome line + chip labels so the
        # localised strings flip with the language.
        user = self._auth.current_user() or self._auth.cached_user()
        if (
            user is not None
            and self._signed_in_widget is not None
            and self._signed_in_widget.isVisible()
        ):
            self._render_signed_in(user)
        else:
            self._refresh_chips()
        self._refresh_cloud_status_text()

    # ==================================================================
    # Theme
    # ==================================================================

    def apply_theme(self) -> None:                               # type: ignore[override]
        super().apply_theme()
        self._apply_signin_button_theme()
        self._apply_signed_in_theme()

    def _apply_signin_button_theme(self) -> None:
        bg = Config.get_color("row_1")
        bg_hover = Config.get_color("btn_1")
        border = Config.get_color("border_2")
        fg = Config.get_color("main_t1")
        title = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        divider = Config.get_color("border_1")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        size_large = Config.get_font_size("size_large")
        qss = (
            f"QLabel#homeAccountHeadline {{"
            f"  color: {title};"
            f"  background: transparent;"
            f"  font-size: {size_large}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QLabel#homeAccountSub {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homeAccountWelcomePic {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
            f"QLabel#homeAccountSignInPrompt {{"
            f"  color: {title};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  font-weight: 600;"
            f"  border: none;"
            f"}}"
            f"QFrame#homeAccountDivider {{"
            f"  background: {divider};"
            f"  border: none;"
            f"}}"
            f"QPushButton#homeAccountSignInBtn {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  border: 1px solid {border};"
            f"  border-radius: 8px;"
            f"  padding: 8px 16px;"
            f"  text-align: center;"
            f"  font-size: {size_normal}px;"
            f"}}"
            f"QPushButton#homeAccountSignInBtn:hover {{"
            f"  background: {bg_hover};"
            f"}}"
        )
        if self._signed_out_widget is not None:
            self._signed_out_widget.setStyleSheet(qss)

    def _apply_signed_in_theme(self) -> None:
        if self._signed_in_widget is None:
            return
        avatar_bg = Config.get_color("bg_3")
        avatar_border = Config.get_color("border_1")
        title = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        btn_bg_hover = Config.get_color("btn_1")
        chip_bg = Config.get_color("btn_1")
        chip_hover = Config.get_color("hover_1")
        chip_text = Config.get_color("main_t2")
        chip_bound = Config.get_color("sub_t2")
        chip_bound_linked = Config.get_color("safe_zone")
        chip_border_primary = Config.get_color("safe_zone")
        divider = Config.get_color("border_1")
        status_label_fg = Config.get_color("main_c2")
        status_error = Config.get_color("auth_status_error_color")
        cloud_status_fg = Config.get_color("main_c2")
        usage_track = Config.get_color("row_1")
        usage_border = Config.get_color("border_1")
        usage_fill = Config.get_color("safe_zone")
        usage_text_fg = Config.get_color("sub_t2")
        size_normal = Config.get_font_size("size_normal")
        size_small = Config.get_font_size("size_small")
        size_large = Config.get_font_size("size_large")
        family = Config.get_value("Font", "family", "Microsoft YaHei")

        self._signed_in_widget.setStyleSheet(
            # Avatar
            f"QLabel#homeAccountAvatar {{"
            f"  background: {avatar_bg};"
            f"  border: 1px solid {avatar_border};"
            f"  border-radius: {_AVATAR_PX // 2}px;"
            f"}}"
            # Welcome banner — main greeting line (rich-text, username
            # tint applied via inline span in _render_signed_in).
            f"QLabel#homeAccountWelcomeLine {{"
            f"  color: {title};"
            f"  background: transparent;"
            f"  font-size: {size_large}px;"
            f"  font-weight: 700;"
            f"  border: none;"
            f"}}"
            # Welcome banner — subtitle.
            f"QLabel#homeAccountWelcomeSub {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # UUID
            f"QLabel#homeAccountUuid {{"
            f"  color: {sub};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # Divider
            f"QFrame#homeAccountDivider {{"
            f"  background: {divider};"
            f"  border: none;"
            f"}}"
            # Chip frame (LaviButton-normal look)
            f"QFrame#homeAccountChip {{"
            f"  background-color: {chip_bg};"
            f"  border: 1px solid transparent;"
            f"  border-radius: 6px;"
            f"}}"
            f"QFrame#homeAccountChip:hover {{"
            f"  background-color: {chip_hover};"
            f"}}"
            f"QFrame#homeAccountChip[isPrimary=\"true\"] {{"
            f"  border: 1px solid {chip_border_primary};"
            f"}}"
            f"QFrame#homeAccountChip:disabled {{"
            f"  background-color: {chip_bg};"
            f"}}"
            # Chip prefix (provider name) — no bold per spec
            f"QLabel#homeAccountChipPrefix {{"
            f"  color: {chip_text};"
            f"  background: transparent;"
            f"  font-family: \"{family}\";"
            f"  font-size: {size_normal}px;"
            f"  border: none;"
            f"}}"
            # Chip bound — default tone, no bold
            f"QLabel#homeAccountChipBound {{"
            f"  color: {chip_bound};"
            f"  background: transparent;"
            f"  font-family: \"{family}\";"
            f"  font-size: {size_normal}px;"
            f"  border: none;"
            f"}}"
            # Chip bound — linked → safe_zone green (still no bold)
            f"QLabel#homeAccountChipBound[isLinked=\"true\"] {{"
            f"  color: {chip_bound_linked};"
            f"}}"
            # Link status (toast under chips)
            f"QLabel#homeAccountLinkStatus {{"
            f"  color: {status_label_fg};"
            f"  background: transparent;"
            f"  font-size: 11px;"
            f"  border: none;"
            f"}}"
            f"QLabel#homeAccountLinkStatus[isError=\"true\"] {{"
            f"  color: {status_error};"
            f"}}"
            # Cloud sync status
            f"QLabel#homeAccountCloudStatus {{"
            f"  color: {cloud_status_fg};"
            f"  background: transparent;"
            f"  font-size: 11px;"
            f"  border: none;"
            f"}}"
            # Cloud usage progress bar — thin track + fill, no inline text
            f"QProgressBar#homeAccountCloudUsageBar {{"
            f"  background-color: {usage_track};"
            f"  border: 1px solid {usage_border};"
            f"  border-radius: 3px;"
            f"}}"
            f"QProgressBar#homeAccountCloudUsageBar::chunk {{"
            f"  background-color: {usage_fill};"
            f"  border-radius: 2px;"
            f"}}"
            # Cloud usage suffix label — "23.0 MB / 200.0 MB"
            f"QLabel#homeAccountCloudUsageText {{"
            f"  color: {usage_text_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # Cloud usage percent suffix — "13%"
            f"QLabel#homeAccountCloudUsagePercent {{"
            f"  color: {usage_text_fg};"
            f"  background: transparent;"
            f"  font-size: {size_small}px;"
            f"  border: none;"
            f"}}"
            # UUID copy button — flat, hover background only.
            f"QPushButton#homeAccountUuidCopy {{"
            f"  background: transparent;"
            f"  border: none;"
            f"  border-radius: 4px;"
            f"  padding: 2px;"
            f"}}"
            f"QPushButton#homeAccountUuidCopy:hover {{"
            f"  background: {btn_bg_hover};"
            f"}}"
            # Sync status icon — no chrome, the pixmap is the visual.
            f"QLabel#homeAccountCloudStatusIcon {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
        )


__all__ = ["HomeAccountCard"]
