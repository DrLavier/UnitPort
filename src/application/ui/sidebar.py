# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Sidebar -- left rail + slide-out content panel for MainWindow.

Layout (top to bottom in the rail)::

    User
    --- divider ---
    Node Library   (mode-gated: Training Canva only)
    Project Files  (mode-gated: Mission Control only)
    Robot Asset    (mode-gated: Mission Control only)
    Controller     (mode-gated: Mission Control only)
    (stretch)
    Fullscreen

Each top-rail button is a checkable ``LaviButton`` (kind=light) that
toggles a slide-out content panel anchored to the rail. Re-clicking the
active button collapses the panel; clicking another button swaps the
panel content. Each panel is a ``TitleGroupBox`` (SDK) hosted in a
container with top/bottom padding 15 and left/right padding 5.

The Sidebar itself draws a 1 px right-edge border using the theme slot
``current_theme`` (system.ini[Theme]).

Signals:
    panel_changed(key)            -- emitted after the panel toggled to
                                     ``key`` (or "" when collapsed).
    feature_requested(key)        -- legacy alias kept for MainWindow's
                                     stub dispatcher; emitted on every
                                     nav-button click (also when the
                                     button collapses the panel).
    fullscreen_toggle_requested   -- bottom rail button (paired with F11
                                     on MainWindow).

Lifecycle (mirrors MainWindow's 5-step contract):
    init / load_data / build_ui / refresh / apply_theme.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Assets,
    Config,
    I18n,
    LaviButton,
    TitleGroupBox,
    i18n_bind,
    setButton,
    setTitleGroupBox,
    setWheelSelector,
    tr,
)


# Display labels for known language codes. Unknown codes fall back to the
# code itself so newly-added localisation directories surface immediately
# without a code change.
_LANG_DISPLAY = {
    "EN": "English",
    "FR": "Français",
    "ZH_S": "简体中文",
    "ZH_T": "繁體中文",
    "IT": "Italiano",
    "ES": "Español",
    "DE": "Deutsch",
    "RU": "Русский",
    "PT": "Português",
    "AR": "العربية",
    "JA": "日本語",
    "KO": "한국어",
}


class _LanguagePopup(QFrame):
    """Frameless popup hosting a WheelSelector of available languages.

    UX: 打开时 wheel 直接停在「当前语种」位置（无入场动画，避免 hit-test
    抖动）。用户拨/点其他语种 → wheel 完全静止 ``_SETTLE_MS`` 毫秒后
    ``currentIndexChanged`` 携带新 index → 与初始 index 不同则视为有效
    选择，emit ``language_picked(code)`` 并关闭。落定窗从「最后一次用户
    输入」起算（WheelSelector 保证），所以只要还在滚动就绝不会提交。
    点击已居中的语种 = 显式确认，跳过落定等待立即提交。点 popup 外部
    关闭不提交。
    """

    language_picked = pyqtSignal(str)

    _PAD = 8
    _RADIUS = 70
    _VISIBLE = 5
    # Idle window before a wheel stop counts as "the user finished
    # spinning". Applying a language closes the popup and retranslates
    # the whole app, so err on the deliberate side (SDK default 200ms is
    # tuned for non-committal display-sync consumers, not for commits).
    _SETTLE_MS = 800

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("sidebarLangPopup")

        self._codes: list[str] = list(I18n.available_languages())
        if not self._codes:
            self._codes = [I18n.get_language() or "EN"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self._PAD, self._PAD, self._PAD, self._PAD)
        layout.setSpacing(0)

        items = [_LANG_DISPLAY.get(code, code) for code in self._codes]
        self._wheel = setWheelSelector(
            items,
            visible_count=self._VISIBLE,
            radius=self._RADIUS,
            settle_ms=self._SETTLE_MS,
            parent=self,
        )
        layout.addWidget(self._wheel)

        cur = I18n.get_language()
        self._initial_idx = self._codes.index(cur) if cur in self._codes else 0

        # WheelSelector 没有「不动画即定位」的 public API。我们在 __init__
        # 之后立刻把私有 state 推到目标位，再清掉构造函数里 enqueue 的初始
        # offset。这样 popup 一开就停在当前语种处，而不是滚一段动画过去。
        self._wheel._current_index = self._initial_idx
        self._wheel._offset = 0.0
        self._wheel._pending_steps = 0
        self._wheel.update()

        # 连接 currentIndexChanged：wheel 完全静止 _SETTLE_MS 后才 emit
        # （任何输入——包括触控板不足一格的微量 delta——都会重置这个窗，
        # 见 WheelSelector 落定语义），所以用户还在滚动时绝不会提交。
        # WheelSelector 构造时 schedule 了一次
        # QTimer.singleShot(0, _emit_current_index)，那次 emit 的 idx 就是
        # 我们设过的 _initial_idx，会被下面 handler 自动忽略。
        self._wheel.currentIndexChanged.connect(self._on_settled)
        # 点击已居中的语种 = 显式确认：不必等落定窗，立即提交。
        self._wheel.item_activated.connect(self._on_activated)
        self.apply_theme()

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_3")
        border = Config.get_color("highlight")
        self.setStyleSheet(
            f"QFrame#sidebarLangPopup {{ "
            f"background-color: {bg}; "
            f"border: 1px solid {border}; "
            f"border-radius: 6px; "
            f"}}"
        )

    def _on_settled(self, idx: int, _key: str) -> None:
        if idx == self._initial_idx:
            return
        if 0 <= idx < len(self._codes):
            self.language_picked.emit(self._codes[idx])
        self.close()

    def _on_activated(self, idx: int, _key: str) -> None:
        # Explicit click on the centered item: commit immediately. Clicking
        # the language that is already active just closes the popup
        # (sidebar's _on_lang_picked is a no-op for the current code anyway).
        if idx != self._initial_idx and 0 <= idx < len(self._codes):
            self.language_picked.emit(self._codes[idx])
        self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


class _RailTooltip(QFrame):
    """Frameless hover tooltip anchored to the right of a rail button.

    Replaces Qt's native QToolTip for sidebar buttons so we control the
    placement (button's right edge), size (size_small), color (sidebar_bg
    background, main_t1 text), corners (4px radius) and lack of border.
    Single shared instance owned by the Sidebar — repositioned each show.
    """

    _PAD_H = 10
    _PAD_V = 4
    _GAP = 6

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # ToolTip flag keeps the popup non-activating and click-through;
        # FramelessWindowHint removes the OS chrome.
        super().__init__(
            parent,
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("sidebarRailTooltip")
        self._label = QLabel(self)
        self._label.setObjectName("sidebarRailTooltipText")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(self._PAD_H, self._PAD_V, self._PAD_H, self._PAD_V)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self.apply_theme()

    def apply_theme(self) -> None:
        bg = Config.get_color("sidebar_bg")
        fg = Config.get_color("main_t1")
        size = Config.get_font_size("size_small")
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        self.setStyleSheet(
            f"QFrame#sidebarRailTooltip {{ "
            f"background-color: {bg}; "
            f"border: none; "
            f"border-radius: 4px; "
            f"}}"
            f"QLabel#sidebarRailTooltipText {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-family: \"{family}\"; "
            f"font-size: {size}pt; "
            f"}}"
        )

    def show_for(self, btn: QWidget, text: str) -> None:
        self._label.setText(text)
        self.adjustSize()
        anchor = btn.mapToGlobal(QPoint(btn.width(), 0))
        ph = self.sizeHint().height()
        pos = QPoint(
            anchor.x() + self._GAP,
            anchor.y() + btn.height() // 2 - ph // 2,
        )
        self.move(pos)
        self.show()
        self.raise_()


class Sidebar(QFrame):
    """Pyramid left rail + slide-out content panel."""

    panel_changed = pyqtSignal(str)
    feature_requested = pyqtSignal(str)            # legacy alias
    fullscreen_toggle_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    update_requested = pyqtSignal()
    # Re-emitted from the node_library pop-out panel; routed by MainWindow
    # into the live CanvasPage. Payload = registers.nodes manifest id.
    node_requested = pyqtSignal(str)
    # Emitted after a sidebar panel is constructed in ``_build_panel_page``.
    # MainWindow uses this to (re)seed panel state (project / canvas /
    # backend) without having to know panel construction timing.
    panel_built = pyqtSignal(str, QWidget)

    # (key, icon_id, i18n_key, default_label, tooltip_default)
    # Rail order: User is always first (above divider). After the divider,
    # node_library precedes the other three so the Training Canva entrypoint
    # sits at the top of the mode-gated cluster.
    NAV_ITEMS = (
        ("user",         "icon_account",    "sidebar.user",         "User",          "User"),
        ("node_library", "icon_node",       "sidebar.node_library", "Node Library",  "Node Library"),
        ("projects",     "icon_prj",        "sidebar.projects",     "Project Files", "Project Files"),
        ("robot_assets", "icon_robot",      "sidebar.robot_assets", "Robot Asset",   "Robot Asset"),
        ("controller",   "icon_controller", "sidebar.controller",   "Controller",    "Controller"),
        ("resources",    "icon_resource",   "sidebar.resources",    "Resources",     "Third-party motion datasets & policy bundles"),
        # Scripts-mode rail entries. Hidden in every mode except "scripts"
        # (see ``_MODE_VISIBLE_KEYS``). Order: Rewards first (most-used →
        # default-open in Scripts mode, see ``apply_view_mode``), then
        # Termins, Observs, and finally Training at the bottom.
        ("rewards",      "icon_rewards",     "sidebar.rewards",      "Rewards",       "Reward preset modules"),
        ("terminations", "icon_termination", "sidebar.terminations", "Termins",       "Termination preset modules"),
        ("observations", "icon_observation", "sidebar.observations", "Observs",       "Observation preset modules"),
        ("training",     "icon_build",       "sidebar.training",     "Training",      "Project + Global scripts"),
    )

    # Per-panel TitleGroupBox metadata: (key, i18n_key, default_title).
    PANEL_TITLES = {
        "user":         ("sidebar.user.title",         "User"),
        "projects":     ("sidebar.projects.title",     "Project Files"),
        "robot_assets": ("sidebar.robot_assets.title", "Robot Asset"),
        "controller":   ("sidebar.controller.title",   "Controller"),
        "resources":    ("sidebar.resources.title",    "Resources"),
        "node_library": ("sidebar.node_library.title", "Node Library"),
        "training":     ("sidebar.training.title",     "Training Scripts"),
        "rewards":      ("sidebar.rewards.title",      "Rewards"),
        "terminations": ("sidebar.terminations.title", "Terminations"),
        "observations": ("sidebar.observations.title", "Observations"),
    }

    # Short single-word label rendered beneath each rail button's icon
    # (Project Files → Projects, Robot Asset → Robots, etc.). Drawn with
    # ``size_sidebar`` so it fits a 50px square button without crowding
    # the icon above. Tooltip still carries the long-form name.
    # Stored as ``key → (i18n_key, default_short_label)`` so the bottom
    # label re-translates on ``language_changed``.
    _NAV_SHORT_LABELS = {
        "user":         ("sidebar.user.short",         "User"),
        "node_library": ("sidebar.node_library.short", "Nodes"),
        "projects":     ("sidebar.projects.short",     "Projects"),
        "robot_assets": ("sidebar.robot_assets.short", "Robots"),
        "controller":   ("sidebar.controller.short",   "Controller"),
        "resources":    ("sidebar.resources.short",    "Resources"),
        "training":     ("sidebar.training.short",     "Training"),
        "rewards":      ("sidebar.rewards.short",      "Rewards"),
        "terminations": ("sidebar.terminations.short", "Termins"),
        "observations": ("sidebar.observations.short", "Observs"),
    }
    _FULLSCREEN_SHORT_LABEL = ("sidebar.fullscreen.short", "Fullscreen")
    _LANG_SHORT_LABEL = ("sidebar.language.short", "Language")
    _SETTINGS_SHORT_LABEL = ("sidebar.settings.short", "Settings")
    # Update button has two states. The text-below-icon flips between
    # them via ``_apply_update_state`` as the AppSignals
    # ``update_check_complete`` signal arrives. Initial state = Latest.
    # i18n keys live in [update] of sidebar.txt (short_latest / short_needed).
    _UPDATE_SHORT_LABEL_LATEST = ("sidebar.update.short_latest", "Latest")
    _UPDATE_SHORT_LABEL_NEEDED = ("sidebar.update.short_needed", "Update")

    # Nav keys whose rail icon keeps its raw SVG fill (no theme tint).
    # The User key carries dynamic avatar / initials pixmaps that own their
    # own colour — never tint over it. Every other rail button (Node Library
    # included) is tinted with ``main_t1`` so the rail reads as one family.
    _UNTINTED_KEYS = frozenset({"user"})

    # MissionControlPanel mode → which nav buttons are shown.
    # Keys not listed here (e.g. ``user``, ``controller``, ``resources``) are
    # always visible in every mode.
    _MODE_VISIBLE_KEYS = {
        "training_canva":  frozenset({"node_library", "robot_assets"}),
        # AI Coding edits the same canvas as Training Canva (the AI Build
        # overlay floats over it) — surface the same node-library / robot-asset
        # nav so the user can reference them while the assistant works.
        "ai_coding":       frozenset({"node_library", "robot_assets"}),
        "mission_control": frozenset({"projects", "robot_assets"}),
        # Terminations / Observations are intentionally hidden: their
        # preset entries are declarative (mapped to Isaac Lab ``mdp.*`` via
        # ``il_func``) and carry no editable inline source — the Script
        # editor would only show a synthesized ``def key(): return default``
        # stub, which is misleading. The panel modules still exist so the
        # buttons can be re-surfaced if/when real editable variants land.
        "scripts":         frozenset({"training", "rewards"}),
    }
    # All mode-gated keys (union of the per-mode sets) — convenient for
    # toggling visibility in one pass. ``controller`` / ``resources`` are
    # deliberately excluded: they stay visible under every mode (like ``user``),
    # so ``apply_view_mode`` never hides them.
    _MODE_GATED_KEYS = frozenset({
        "node_library", "projects", "robot_assets",
        "training", "rewards", "terminations", "observations",
    })

    _BTN_SIZE = 60                # square button — icon top, single-word label bottom
    _BTN_W = _BTN_SIZE
    _RAIL_W = _BTN_W + 5
    _BTN_SPACING = 20             # rail-layout spacing between adjacent buttons
    _RAIL_SIDE_MARGIN = (_RAIL_W - _BTN_W) // 2
    _PANEL_W = 280
    _DIVIDER_H = 1
    _DIVIDER_V_MARGIN = 8
    _PANEL_PAD_V = 15
    _PANEL_PAD_H = 5

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.init()
        self.build_ui()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self) -> None:
        self.setObjectName("sidebar")
        self._buttons: dict[str, LaviButton] = {}
        self._fs_btn: Optional[LaviButton] = None
        self._settings_btn: Optional[LaviButton] = None
        self._update_btn: Optional[LaviButton] = None
        # Most recent payload seen by ``_apply_update_state``. Cached so
        # the locale re-fire on ``language_changed`` can replay the same
        # state (Latest vs Update-needed) without inventing one.
        self._last_release_info: object = None
        # Static V{x.y.z} label below the Update button. Filled in by
        # build_ui after the rail is constructed.
        self._version_label: Optional[QLabel] = None
        self._lang_btn: Optional[LaviButton] = None
        self._lang_popup: Optional[_LanguagePopup] = None
        self._rail_tip: Optional[_RailTooltip] = None
        self._rail: Optional[QWidget] = None
        self._panel: Optional[QFrame] = None
        self._panel_stack: Optional[QStackedWidget] = None
        self._panel_pages: dict[str, int] = {}
        self._panel_groups: dict[str, TitleGroupBox] = {}
        self._panel_widgets: dict[str, QWidget] = {}
        self._divider: Optional[QFrame] = None
        self._active_key: str = ""
        # Populated lazily inside _wire_user_icon; remains None when the
        # auth subtree fails to import early in boot.
        self._make_initials_pixmap = None

    def load_data(self) -> None:
        # Placeholder: future read of nav order / visibility from
        # registers/ or user.ini overlay. NAV_ITEMS is the static
        # fallback for now.
        pass

    def build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # ── Rail ──────────────────────────────────────────────────────
        self._rail = QWidget(self)
        self._rail.setObjectName("sidebarRail")
        self._rail.setFixedWidth(self._RAIL_W)
        rail_layout = QVBoxLayout(self._rail)
        # L/R margins centre the 50px square buttons inside the 66px rail.
        # The bumped spacing visually separates the icon+label stacks so
        # the rail no longer reads as a tight column.
        rail_layout.setContentsMargins(
            self._RAIL_SIDE_MARGIN, 12, self._RAIL_SIDE_MARGIN, 12
        )
        rail_layout.setSpacing(self._BTN_SPACING)
        rail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1) User (always first, above the divider).
        user_item = self.NAV_ITEMS[0]
        rail_layout.addWidget(self._make_nav_button(*user_item))

        # 2) Divider line below User.
        self._divider = QFrame(self._rail)
        self._divider.setObjectName("sidebarDivider")
        self._divider.setFrameShape(QFrame.Shape.HLine)
        self._divider.setFixedHeight(self._DIVIDER_H)
        # rail_layout.addSpacing(self._DIVIDER_V_MARGIN)
        rail_layout.addWidget(self._divider)
        # rail_layout.addSpacing(self._DIVIDER_V_MARGIN)

        # 3) Remaining nav buttons.
        for item in self.NAV_ITEMS[1:]:
            rail_layout.addWidget(self._make_nav_button(*item))

        rail_layout.addStretch(1)

        # 4) Language switch (popup with WheelSelector). Lives just above
        # the fullscreen toggle. Label = current language code (EN/ZH/...)
        # so the user can confirm the switch landed.
        # icon_only=True 让 SDK 跳过对按钮文本的 i18n_bind —— 我们要自己
        # 控制文本（=当前语种代码），不能让 i18n 系统把它覆写回 default。
        self._lang_btn = setButton(
            "sidebar.language",
            self._BTN_W,
            self._BTN_SIZE,
            kind="light",
            spec="none",
            checkable=True,
            icon_only=True,
            checked_color=Config.get_color("highlight"),
            checked_fg_color=Config.get_color("bg_1"),
        )
        self._lang_btn.clicked.connect(self._on_lang_clicked)
        self._install_rail_tooltip(self._lang_btn)
        rail_layout.addWidget(self._lang_btn)
        # Lang button has no SVG icon — the language code itself doubles
        # as the icon (rendered into the top slot by ``_refresh_lang_btn``).
        # The bottom slot carries the short word "Language".
        self._install_icon_label_stack(
            self._lang_btn, *self._LANG_SHORT_LABEL,
        )
        self._style_lang_icon_label()
        self._refresh_lang_btn()
        # Re-skin both child labels when the popup opens/closes (which
        # flips the button's checked state).
        self._lang_btn.toggled.connect(self._on_lang_toggled)
        I18n.instance().language_changed.connect(self._refresh_lang_btn)

        # 5) Bottom: fullscreen toggle (no panel). Hidden in UI — F11 and
        # the signal are still wired so MainWindow keeps its keybinding.
        self._fs_btn = setButton(
            "sidebar.fullscreen",
            self._BTN_W,
            self._BTN_SIZE,
            kind="light",
            spec="none",
            icon=None,
            default="Fullscreen",
            icon_only=True,
        )
        i18n_bind(
            self._fs_btn, "setToolTip",
            "sidebar.fullscreen.tip", "Fullscreen",
        )
        self._fs_btn.clicked.connect(self.fullscreen_toggle_requested)
        self._fs_btn._nav_icon_name = "icon_zone_out"
        self._install_icon_label_stack(
            self._fs_btn, *self._FULLSCREEN_SHORT_LABEL,
        )
        self._refresh_fs_btn_icon()
        self._fs_btn.setVisible(False)
        rail_layout.addWidget(self._fs_btn)

        # 6) Bottom: Settings + Update (top to bottom).
        self._settings_btn = setButton(
            "sidebar.settings",
            self._BTN_W,
            self._BTN_SIZE,
            kind="light",
            spec="none",
            icon=None,
            default="Settings",
            icon_only=True,
        )
        i18n_bind(
            self._settings_btn, "setToolTip",
            "sidebar.settings.tip", "Settings",
        )
        self._settings_btn.clicked.connect(self.settings_requested)
        self._settings_btn._nav_icon_name = "icon_setting_alt"
        self._install_icon_label_stack(
            self._settings_btn, *self._SETTINGS_SHORT_LABEL,
        )
        self._install_rail_tooltip(self._settings_btn)
        self._refresh_aux_btn_icon(self._settings_btn)
        self._settings_btn.setVisible(False)
        rail_layout.addWidget(self._settings_btn)

        self._update_btn = setButton(
            "sidebar.update",
            self._BTN_W,
            self._BTN_SIZE,
            kind="light",
            spec="none",
            icon=None,
            default="Update",
            icon_only=True,
        )
        # Update button's tooltip + short label both flip between
        # "Latest" / "Update" states via ``_apply_update_state`` as the
        # AppSignals ``update_check_complete`` signal arrives. To also
        # re-translate on plain language_changed (no update-check event),
        # _apply_update_state is now hooked to I18n.language_changed too.
        i18n_bind(
            self._update_btn, "setToolTip",
            "sidebar.update.tip_latest", "You are on the latest version",
        )
        self._update_btn.clicked.connect(self.update_requested)
        self._update_btn._nav_icon_name = "icon_update"
        # Pass empty key — _apply_update_state owns the imperative text
        # rewrites for this label (it tr()s the right key on every call,
        # including the language_changed re-fire below).
        self._install_icon_label_stack(
            self._update_btn,
            "",
            tr(*self._UPDATE_SHORT_LABEL_LATEST),
        )
        self._install_rail_tooltip(self._update_btn)
        self._refresh_aux_btn_icon(self._update_btn)
        rail_layout.addWidget(self._update_btn)

        # V{x.y.z} version label, sub_t2 colour + size_mini. Sits a few
        # pixels below the Update button. Updated by ``apply_theme`` on
        # theme switch.
        self._version_label = QLabel(self._rail)
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._version_label.setText(
            f"V{Config.get_value('System', 'version', '0.0.0')}"
        )
        self._style_version_label()
        rail_layout.addSpacing(4)
        rail_layout.addWidget(self._version_label)

        root.addWidget(self._rail)

        # Subscribe to the in-app updater's check-complete signal so the
        # button visuals reflect the latest cached state. Submitted in
        # main.py's _submit_provisioning before this widget exists, so
        # the signal might have fired already — read the cached release
        # via UpdateService.latest_release() to catch up.
        from application.service.signals import get_app_signals
        get_app_signals().update_check_complete.connect(
            self._apply_update_state
        )
        # Also re-translate the Update button's text + tooltip on language
        # switch — without this hook, the imperative ``setText`` /
        # ``setToolTip`` calls inside _apply_update_state would only re-run
        # when a new update check fires, so a quiet language switch would
        # leave the old-language label on screen.
        I18n.instance().language_changed.connect(
            self._reapply_update_state_locale
        )
        try:
            from application.service.updater import get_update_service
            self._apply_update_state(get_update_service().latest_release())
        except Exception:
            # Defensive: a broken updater module must not block sidebar
            # construction.
            pass

        # ── Slide-out content panel ───────────────────────────────────
        self._panel = QFrame(self)
        self._panel.setObjectName("sidebarPanel")
        self._panel.setFixedWidth(0)         # collapsed by default
        self._panel.setVisible(False)

        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(
            self._PANEL_PAD_H, self._PANEL_PAD_V,
            self._PANEL_PAD_H, self._PANEL_PAD_V,
        )
        panel_layout.setSpacing(0)

        self._panel_stack = QStackedWidget(self._panel)
        panel_layout.addWidget(self._panel_stack, 1)

        for key, _icon, _i18n, _default, _tip in self.NAV_ITEMS:
            page = self._build_panel_page(key)
            idx = self._panel_stack.addWidget(page)
            self._panel_pages[key] = idx

        # 初始 mode 视作 Mission Control：node_library 隐藏，其它三项可见。
        # MainWindow 在 MissionControlPanel 构造完后会调 apply_view_mode 同步
        # 真实 effective_mode（已保存的 _mode + canvas-loaded 状态），并订阅
        # mode_changed 以后续刷新。
        self.apply_view_mode("mission_control")

        # 让 node_library 弹出面板的 node_requested 信号向外冒泡，MainWindow
        # 负责把它转成 canvas_page.spawn_node 调用。
        nl_panel = self._panel_widgets.get("node_library")
        if nl_panel is not None and hasattr(nl_panel, "node_requested"):
            nl_panel.node_requested.connect(self.node_requested)

        root.addWidget(self._panel)

        # Wire up dynamic user-key icon (sign-in → initials/avatar; sign-out
        # → default SVG). Done at the end so self._buttons["user"] exists.
        self._wire_user_icon()

    def refresh(self) -> None:
        # Selected button state stays in sync via _on_nav_clicked. Forward
        # the refresh into each panel widget that exposes a ``refresh``
        # method (currently ProjectsPanel — re-scans projects/ on the spot).
        for widget in self._panel_widgets.values():
            fn = getattr(widget, "refresh", None)
            if callable(fn):
                fn()
        # Re-stamp the User rail icon from live auth state. Signal-based
        # paths (authenticated / workspace_changed / avatar_updated) cover
        # the common case, but anything that calls MainWindow.refresh()
        # after a login/logout — workspace switch, project rebind — must
        # also see the rail icon reflect the current account.
        self.refresh_user_icon()

    def refresh_user_icon(self) -> None:
        """Re-render the User rail button icon from live auth state.

        Authoritative source: ``auth.current_user()`` (in-memory). Disk
        caches (``cached_user()`` / ``cached_avatar()``) are transiently
        None mid-workspace-switch because session.json is in the middle
        of being moved from ``_guest/`` to ``{slug}/`` — relying on them
        was the reason the rail icon never refreshed on first login.

        Order: real avatar pixmap (when cache has one for the live user)
        > synthesized initials from the live ``UserProfile`` > default
        SVG (signed out). Safe to call before ``_wire_user_icon`` ran.
        """
        if self._make_initials_pixmap is None:
            return
        try:
            from application.service.auth import get_auth_manager
        except Exception:                                         # noqa: BLE001
            return
        auth = get_auth_manager()
        if not auth.is_signed_in():
            self._apply_user_icon(None)
            return
        user = auth.current_user()
        if user is None:
            self._apply_user_icon(None)
            return
        cached_pm = auth.cached_avatar()
        if cached_pm is not None and not cached_pm.isNull():
            self._apply_user_icon(cached_pm)
            return
        self._apply_user_icon(
            self._make_initials_pixmap(user.user_id, user.label, 96)
        )

    def apply_theme(self) -> None:
        rail_bg = Config.get_color("sidebar_bg")
        panel_bg = Config.get_color("bg_3")
        edge = Config.get_color("sidebar_border")
        divider = Config.get_color("border_1")
        # Only the rail carries the highlight border. When the panel
        # is expanded the rail border becomes the rail/panel separator;
        # the panel itself has no border so the slide-out has a flat
        # edge against the work zone.
        self.setStyleSheet(
            f"QFrame#sidebar {{ background-color: transparent; border: none; }}"
            f"QWidget#sidebarRail {{ "
            f"background-color: {rail_bg}; "
            f"border-right: 1px solid {edge}; "
            f"}}"
            f"QFrame#sidebarDivider {{ "
            f"background-color: {divider}; "
            f"border: none; "
            f"}}"
            f"QFrame#sidebarPanel {{ "
            f"background-color: {panel_bg}; "
            f"border: none; "
            f"}}"
        )
        # LaviButton._overrides are hex strings frozen at init — re-resolve
        # them from the (potentially swapped) theme slots before refresh_style
        # rebuilds the QSS, then regenerate the default/checked icon pair.
        highlight_hex = Config.get_color("highlight")
        alt_t1_hex = Config.get_color("bg_1")
        for key, btn in self._buttons.items():
            btn._overrides["checked"] = highlight_hex
            btn._overrides["checked_fg"] = alt_t1_hex
            btn.refresh_style()
            self._refresh_nav_btn_icons(btn, key)
            self._refresh_nav_btn_text(btn)
        if self._fs_btn is not None:
            self._fs_btn.refresh_style()
            self._refresh_fs_btn_icon()
            self._refresh_nav_btn_text(self._fs_btn)
        for aux_btn in (self._settings_btn, self._update_btn):
            if aux_btn is None:
                continue
            aux_btn.refresh_style()
            self._refresh_aux_btn_icon(aux_btn)
            self._refresh_nav_btn_text(aux_btn)
        self._style_version_label()
        if self._lang_btn is not None:
            self._lang_btn._overrides["checked"] = highlight_hex
            self._lang_btn._overrides["checked_fg"] = alt_t1_hex
            self._lang_btn.refresh_style()
            self._style_lang_icon_label()
            self._refresh_nav_btn_text(self._lang_btn)
        if self._lang_popup is not None:
            self._lang_popup.apply_theme()
        if self._rail_tip is not None:
            self._rail_tip.apply_theme()
        for group in self._panel_groups.values():
            # TitleGroupBox inherits SDK theme via its own stylesheet; no
            #-op refresh hook today, but we keep this branch so future
            # theme switches can call group.refresh_style() if added.
            if hasattr(group, "refresh_style"):
                group.refresh_style()
        # Per-panel content widgets that own their own theme cache
        # (node_library tree colors / font sizes) need a refresh on theme
        # switch too.
        for widget in self._panel_widgets.values():
            fn = getattr(widget, "refresh_style", None)
            if callable(fn):
                fn()

    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------
    @property
    def active_key(self) -> str:
        return self._active_key

    def open_panel(self, key: str) -> None:
        """Programmatically expand the panel for ``key`` (or "" to collapse)."""
        if key and key not in self._panel_pages:
            return
        self._set_active(key)

    def panel_widget(self, key: str) -> Optional[QWidget]:
        """Return the panel content widget for ``key`` (or None if absent)."""
        return self._panel_widgets.get(key)

    def apply_view_mode(self, mode: str) -> None:
        """Show/hide the mode-gated nav buttons based on MissionControl mode.

        - ``training_canva``  → 只显示 Node Library，藏 Project / Robot / Controller
        - ``mission_control`` (含未知值) → 反之

        若隐藏的按钮恰好是当前展开的 panel，则同时收起 panel；用户在隐藏
        按钮的状态下没有重新切到该 panel 的入口。
        """
        visible_keys = self._MODE_VISIBLE_KEYS.get(
            mode, self._MODE_VISIBLE_KEYS["mission_control"]
        )
        for key in self._MODE_GATED_KEYS:
            btn = self._buttons.get(key)
            if btn is None:
                continue
            want = key in visible_keys
            # 总是 setVisible：父级未 show 时 isVisible() 报 False 是误导的，
            # 仍需显式调用让 Qt 记录隐藏意图，父级 show 后才会生效。
            btn.setVisible(want)
            if not want and self._active_key == key:
                self._set_active("")

        # Scripts mode default-open: Rewards is the most frequently used
        # of the four script panels, auto-expand it the first time we
        # enter Scripts mode (skip if the user already picked a different
        # one of the four, so we don't yank focus out from under them).
        if mode == "scripts":
            if self._active_key not in self._MODE_VISIBLE_KEYS["scripts"]:
                if "rewards" in visible_keys:
                    self._set_active("rewards")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # User-key icon: dynamic, swaps to initials/avatar pixmap on sign-in.
    # ------------------------------------------------------------------

    def _wire_user_icon(self) -> None:
        """Subscribe to AuthManager signals so the User rail icon mirrors
        the live auth state. Called at the end of ``build_ui``."""
        try:
            from application.service.auth import get_auth_manager
            from application.service.auth.avatar_cache import (
                make_initials_pixmap,
            )
        except Exception:                                         # noqa: BLE001
            # Auth module not importable yet (early boot before httpx is
            # installed); skip dynamic icon — default SVG stays.
            return

        self._make_initials_pixmap = make_initials_pixmap
        auth = get_auth_manager()
        auth.avatar_updated.connect(self._on_auth_avatar_updated)
        auth.signed_out.connect(self._on_auth_signed_out)
        auth.workspace_changed.connect(self._on_auth_workspace_changed)
        # Belt-and-braces：authenticated 直接带 UserProfile，立刻能合成
        # 字母缩写——不依赖 cached_user 文件回读（workspace 切换瞬间
        # session.json 可能还在搬家路上，cached_user() 取到 None）。
        # 真正的头像 pixmap 到达后会通过 avatar_updated 覆盖这块占位。
        auth.authenticated.connect(self._on_auth_authenticated)

        # Cold-start seed: if there is a cached user, synthesize initials
        # right away so the rail doesn't show the generic SVG for a
        # second before avatar_updated fires.
        cached_user = auth.cached_user()
        if cached_user is not None:
            cached_pm = auth.cached_avatar()
            if cached_pm is not None and not cached_pm.isNull():
                self._apply_user_icon(cached_pm)
            else:
                self._apply_user_icon(
                    self._make_initials_pixmap(
                        cached_user.user_id, cached_user.label, 96,
                    )
                )

    def _apply_user_icon(self, pixmap: Optional[QPixmap]) -> None:
        """Replace (or restore) the User rail button's icon.

        ``pixmap=None`` reverts to the default ``icon_account`` SVG. Other
        rail buttons that share the highlight-tint mechanism are
        untouched (``_HIGHLIGHT_TINTED_KEYS`` doesn't include user)."""
        btn = self._buttons.get("user")
        if btn is None:
            return
        side = self._icon_side(btn)
        icon_label = getattr(btn, "_nav_icon_label", None)
        if icon_label is None:                                # pragma: no cover
            return
        if pixmap is None or pixmap.isNull():
            # Signed-out: stamp the cached SVG directly. Do NOT call
            # _refresh_nav_btn_icons here — for the "user" key it now
            # routes through refresh_user_icon, which on the signed-out
            # branch lands back in this function and recurses infinitely.
            is_checked = btn.isChecked()
            cache_attr = "_nav_pixmap_checked" if is_checked else "_nav_pixmap_default"
            pm = getattr(btn, cache_attr, None)
            if pm is None:
                # First-time render — populate both variants so subsequent
                # toggles can swap without re-rendering.
                icon_name = (
                    getattr(btn, "_nav_icon_name", None) or "icon_account"
                )
                btn._nav_pixmap_default = self._render_icon_pixmap(
                    icon_name, side,
                )
                btn._nav_pixmap_checked = self._render_icon_pixmap(
                    icon_name, side, tint=Config.get_color("bg_1"),
                )
                pm = (
                    btn._nav_pixmap_checked
                    if is_checked
                    else btn._nav_pixmap_default
                )
            if pm is not None:
                icon_label.setPixmap(pm)
                icon_label.update()
            return

        # Crop into a circle so the rail thumb stays consistent with
        # UserPanel's avatar (same round_pixmap helper).
        from application.service.auth.avatar_cache import round_pixmap
        rounded = round_pixmap(pixmap, side)
        icon_label.setPixmap(rounded)
        # Force a repaint — setPixmap by itself doesn't always trigger
        # an immediate visual update when fired from inside a signal
        # handler that nests under modal-dialog event flushing.
        icon_label.update()

    @pyqtSlot(str, QPixmap)
    def _on_auth_avatar_updated(self, _user_id: str, pixmap: QPixmap) -> None:
        # Real avatar arrived for the active user — paint it directly
        # (refresh_user_icon would also work via cached_avatar, but the
        # signal carries the pixmap so use it without a re-lookup).
        self._apply_user_icon(pixmap)

    @pyqtSlot(object)
    def _on_auth_authenticated(self, _user) -> None:
        # Funnel through the canonical refresh — reads in-memory auth
        # state, so it works even before session.json lands on disk in
        # the new workspace.
        self.refresh_user_icon()

    @pyqtSlot()
    def _on_auth_signed_out(self) -> None:
        self._apply_user_icon(None)

    @pyqtSlot(str)
    def _on_auth_workspace_changed(self, _new_uid: str) -> None:
        self.refresh_user_icon()

    def eventFilter(self, obj, ev):
        """Re-render an icon label's pixmap when the layout resizes it.

        Static-estimate cold-start renders are clipped vertically by the
        text-label strip; the first Resize that the layout dispatches is
        our trigger to redo the render at the now-correct ``min(w, h)``.
        """
        # Rail-tooltip hooks: only fire for buttons explicitly opted-in via
        # _install_rail_tooltip. The QEvent.ToolTip branch suppresses Qt's
        # native popup so it doesn't paint alongside ours.
        if getattr(obj, "_rail_tip_target", False):
            t = ev.type()
            if t == QEvent.Type.Enter:
                self._show_rail_tooltip(obj)
            elif t in (QEvent.Type.Leave, QEvent.Type.Hide):
                self._hide_rail_tooltip()
            elif t == QEvent.Type.ToolTip:
                return True
        if ev.type() == QEvent.Type.Resize:
            for key, btn in self._buttons.items():
                if getattr(btn, "_nav_icon_label", None) is obj:
                    self._refresh_nav_btn_icons(btn, key)
                    # The user rail icon may carry a live avatar/initials
                    # pixmap (not the default SVG); _refresh_nav_btn_icons
                    # would have stamped the SVG over it — restore.
                    if key == "user":
                        self.refresh_user_icon()
                    return False
            if self._fs_btn is not None and getattr(
                self._fs_btn, "_nav_icon_label", None
            ) is obj:
                self._refresh_fs_btn_icon()
                return False
            for aux_btn in (self._settings_btn, self._update_btn):
                if aux_btn is not None and getattr(
                    aux_btn, "_nav_icon_label", None
                ) is obj:
                    self._refresh_aux_btn_icon(aux_btn)
                    return False
        return super().eventFilter(obj, ev)

    def _icon_side(self, btn: Optional[LaviButton] = None) -> int:
        """Pixmap side length for the icon child label.

        When ``btn`` is supplied and its icon_label has been laid out,
        derive the side from the LIVE icon_label dimensions (``min(w, h)``)
        — height is the constraining axis (lower strip is the text label),
        so a square pixmap sized to the height never gets vertically
        clipped. Without a btn (cold-start / cached fallback), fall back to
        a tight static estimate of the height the layout would assign.
        """
        if btn is not None:
            label = getattr(btn, "_nav_icon_label", None)
            if label is not None:
                w = label.width()
                h = label.height()
                if w > 0 and h > 0:
                    return max(16, min(w, h))
        # Static estimate: top/bottom margins (4+4), spacing (2), and text
        # row (~14px at size_sidebar=8pt + descent). Conservative so the
        # cold-start render never paints a pixmap that the layout will
        # later clip vertically.
        return max(16, self._BTN_SIZE - 24)

    def _make_nav_button(
        self, key: str, icon: str, i18n_key: str, default: str, tip: str,
    ) -> LaviButton:
        btn = setButton(
            i18n_key,
            self._BTN_W,
            self._BTN_SIZE,
            kind="light",
            spec="none",
            # icon/text intentionally not handed to LaviButton — the icon
            # is rendered into a child QLabel so we can stack it above the
            # short-name label without depending on QPushButton's built-in
            # horizontal icon+text layout.
            icon=None,
            default=default,
            icon_only=True,
            checkable=True,
            checked_color=Config.get_color("highlight"),
            checked_fg_color=Config.get_color("bg_1"),
        )
        # Tooltip bound via i18n so language switches re-translate. Each
        # nav item carries an ``<i18n_key>.tip`` entry in sidebar.txt
        # (e.g. ``[user] tip = User``); the ``tip`` argument is the EN
        # fallback when the key has no translation yet.
        i18n_bind(btn, "setToolTip", f"{i18n_key}.tip", tip)
        self._install_rail_tooltip(btn)
        btn.clicked.connect(lambda _c=False, k=key: self._on_nav_clicked(k))
        self._buttons[key] = btn
        btn._nav_icon_name = icon
        short_key, short_default = self._NAV_SHORT_LABELS.get(
            key, (f"{i18n_key}.short", default),
        )
        self._install_icon_label_stack(btn, short_key, short_default)
        # Cache both default + checked pixmaps so toggling between states
        # swaps the SVG fill color (default keeps original / highlight tint,
        # checked is alt_t1 tinted so it reads against the highlight bg).
        self._refresh_nav_btn_icons(btn, key)
        btn.toggled.connect(
            lambda checked, b=btn, k=key: self._on_nav_toggled(b, k, checked)
        )
        return btn

    def _install_rail_tooltip(self, btn: LaviButton) -> None:
        """Route ``btn``'s hover through the shared right-anchored tooltip.

        The button keeps its ``setToolTip(text)`` calls — that text is the
        source we render. We swallow ``QEvent.ToolTip`` to stop Qt's native
        popup from also appearing alongside ours.
        """
        btn._rail_tip_target = True
        btn.installEventFilter(self)

    def _show_rail_tooltip(self, btn: QWidget) -> None:
        text = btn.toolTip() if hasattr(btn, "toolTip") else ""
        if not text:
            return
        if self._rail_tip is None:
            self._rail_tip = _RailTooltip(None)
        self._rail_tip.show_for(btn, text)

    def _hide_rail_tooltip(self) -> None:
        if self._rail_tip is not None and self._rail_tip.isVisible():
            self._rail_tip.hide()

    def _install_icon_label_stack(
        self, btn: LaviButton, short_label_key: str, short_label_default: str,
    ) -> None:
        """Attach an icon-on-top, label-on-bottom layout inside ``btn``.

        Both children are mouse-transparent so clicks still hit the
        underlying LaviButton. The icon QLabel is filled later by
        ``_refresh_nav_btn_icons``; the text label is themed by
        ``_refresh_nav_btn_text`` and bound to i18n via ``i18n_bind``
        so language switches automatically re-translate the short word.
        """
        layout = QVBoxLayout(btn)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel(btn)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        # Re-render the pixmap whenever the layout assigns a new height —
        # cold-start render runs before the icon_label has a size, so the
        # static estimate is the only thing in place until the first
        # Resize fires (see eventFilter).
        icon_label.installEventFilter(self)

        text_label = QLabel(btn)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        # ``i18n_bind`` performs the initial setText and re-runs on every
        # language_changed; empty key skips the binding and uses the raw
        # default (used by the Update button, whose label is rewritten
        # imperatively by ``_apply_update_state``).
        if short_label_key:
            i18n_bind(
                text_label, "setText", short_label_key, short_label_default,
            )
        else:
            text_label.setText(short_label_default)

        layout.addWidget(icon_label, 1)
        layout.addWidget(text_label, 0)

        btn._nav_icon_label = icon_label
        btn._nav_text_label = text_label
        self._refresh_nav_btn_text(btn)

    def _on_nav_toggled(
        self, btn: LaviButton, key: str, checked: bool,
    ) -> None:
        # The User rail icon is special: when signed in it carries the
        # avatar/initials pixmap, NOT the cached _nav_pixmap_default SVG.
        # Swapping in the SVG on toggle wipes that avatar — instead, run
        # refresh_user_icon so the live auth state re-stamps the right
        # pixmap (avatar, initials, or default SVG when signed out).
        if key == "user":
            self.refresh_user_icon()
            self._refresh_nav_btn_text(btn)
            return
        pm = (
            getattr(btn, "_nav_pixmap_checked", None)
            if checked
            else getattr(btn, "_nav_pixmap_default", None)
        )
        label = getattr(btn, "_nav_icon_label", None)
        if pm is not None and label is not None:
            label.setPixmap(pm)
        self._refresh_nav_btn_text(btn)

    def _refresh_nav_btn_text(self, btn: LaviButton) -> None:
        """Apply theme-derived color + ``size_sidebar`` font to the label.

        Default state uses ``main_t1`` (matches LaviButton kind=light fg);
        checked state uses ``alt_t1`` so the label reads against the
        highlight pill. Kept in lock-step with the button's own QSS.
        """
        label = getattr(btn, "_nav_text_label", None)
        if label is None:
            return
        size = Config.get_font_size("size_sidebar")
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        fg = (
            Config.get_color("bg_1")
            if btn.isChecked()
            else Config.get_color("main_t1")
        )
        label.setStyleSheet(
            f"QLabel {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-family: \"{family}\"; "
            f"font-size: {size}pt; "
            f"}}"
        )

    def _refresh_nav_btn_icons(self, btn: LaviButton, key: str) -> None:
        """Rebuild default + checked QPixmap variants and apply the right one."""
        icon_name = getattr(btn, "_nav_icon_name", None)
        if not icon_name:
            return
        side = self._icon_side(btn)
        # Default-state: main_t1-tinted for every rail icon so the whole rail
        # reads as one family. The only exception is the User button, whose
        # icon slot is owned by refresh_user_icon (avatar / initials / SVG
        # depending on auth state) — leave its SVG fallback un-tinted so the
        # signed-out generic glyph keeps its native colours.
        if key in self._UNTINTED_KEYS:
            default_pm = self._render_icon_pixmap(icon_name, side)
        else:
            default_pm = self._render_icon_pixmap(
                icon_name, side, tint=Config.get_color("main_t1"),
            )
        # Checked-state: always alt_t1 so the icon reads against the
        # highlight background slot.
        checked_pm = self._render_icon_pixmap(
            icon_name, side, tint=Config.get_color("bg_1"),
        )
        btn._nav_pixmap_default = default_pm
        btn._nav_pixmap_checked = checked_pm
        # The User rail icon's *visible* pixmap is owned by refresh_user_icon
        # (avatar / initials / SVG depending on auth state). The SVG variants
        # are still cached above for the signed-out fallback path — we just
        # don't stamp them onto the label here, otherwise apply_theme /
        # eventFilter would wipe the avatar on every theme change or resize.
        if key == "user":
            self.refresh_user_icon()
            return
        active = checked_pm if btn.isChecked() else default_pm
        label = getattr(btn, "_nav_icon_label", None)
        if active is not None and label is not None:
            label.setPixmap(active)

    def _render_icon_pixmap(
        self,
        icon_name: str,
        side: int,
        tint: Optional[str] = None,
    ) -> Optional[QPixmap]:
        """Render ``icon_name`` SVG to a square QPixmap (optionally tinted).

        Pure function — does not touch any widget. Returns ``None`` if the
        asset can't be located or QtSvg is unavailable.
        """
        svg_path = Assets.find_icon(icon_name)
        if svg_path is None:
            return None
        try:
            from PyQt6.QtSvg import QSvgRenderer
            renderer = QSvgRenderer(str(svg_path))
        except Exception:
            return None
        pm = QPixmap(side, side)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        try:
            renderer.render(p)
            if tint:
                p.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_SourceIn
                )
                p.fillRect(pm.rect(), QColor(tint))
        finally:
            p.end()
        return pm

    def _build_panel_page(self, key: str) -> QWidget:
        page = QWidget(self._panel_stack)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        i18n_key, default_title = self.PANEL_TITLES.get(key, ("", key.title()))
        group = setTitleGroupBox(i18n_key, default=default_title, parent=page)
        body = QVBoxLayout(group)
        body.setContentsMargins(6, 8, 6, 8)
        body.setSpacing(6)

        # Delegate to the per-panel factory. Lazy imports inside build_panel
        # keep app startup cheap (services are constructed on first access).
        from application.ui.sidebar_panels import build_panel
        panel = build_panel(key, parent=group)
        if panel is not None:
            body.addWidget(panel, 1)
            self._panel_widgets[key] = panel
            # Notify MainWindow so it can (re)seed the panel with live
            # project / backend state (panels are lazy-built — they may
            # come into existence long after _bind_project_info has run).
            self.panel_built.emit(key, panel)
        else:
            body.addStretch(1)

        layout.addWidget(group)
        self._panel_groups[key] = group
        return page

    def _refresh_lang_btn(self, *_args) -> None:
        if self._lang_btn is None:
            return
        code = (I18n.get_language() or "").upper()[:2] or "??"
        # The code (EN/ZH/…) lives in the icon slot of the stacked layout
        # so the bottom slot can carry the static "Language" short label,
        # keeping the lang button visually consistent with the rest of the
        # rail (icon-on-top / one-word-on-bottom).
        icon_label = getattr(self._lang_btn, "_nav_icon_label", None)
        if icon_label is not None:
            icon_label.setText(code)
        else:                                                  # pragma: no cover
            self._lang_btn.setText(code)
        self._lang_btn.setToolTip(
            tr("sidebar.language.tip", "Language") + f" — {code}"
        )

    def _style_lang_icon_label(self) -> None:
        """Style the lang button's icon-slot QLabel (it shows EN/ZH text,
        not an SVG). Re-applied on theme switch via ``apply_theme``.
        """
        if self._lang_btn is None:
            return
        icon_label = getattr(self._lang_btn, "_nav_icon_label", None)
        if icon_label is None:
            return
        size = Config.get_font_size("size_normal")
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        fg = (
            Config.get_color("bg_1")
            if self._lang_btn.isChecked()
            else Config.get_color("main_t1")
        )
        icon_label.setStyleSheet(
            f"QLabel {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-family: \"{family}\"; "
            f"font-size: {size}pt; "
            f"font-weight: bold; "
            f"}}"
        )

    def _refresh_fs_btn_icon(self) -> None:
        """Paint the fullscreen icon into its child label (theme-aware)."""
        if self._fs_btn is None:
            return
        icon_name = getattr(self._fs_btn, "_nav_icon_name", None)
        label = getattr(self._fs_btn, "_nav_icon_label", None)
        if icon_name is None or label is None:
            return
        pm = self._render_icon_pixmap(
            icon_name, self._icon_side(self._fs_btn),
            tint=Config.get_color("main_t1"),
        )
        if pm is not None:
            label.setPixmap(pm)

    def _refresh_aux_btn_icon(self, btn: Optional[LaviButton]) -> None:
        """Paint a non-checkable rail aux button's icon (Settings / Update)."""
        if btn is None:
            return
        icon_name = getattr(btn, "_nav_icon_name", None)
        label = getattr(btn, "_nav_icon_label", None)
        if icon_name is None or label is None:
            return
        # Optional per-button tint slot (e.g. Update button flips to
        # ``highlight`` when a new version is available). Resolved at paint
        # time so a theme switch re-tints correctly. Defaults to ``main_t1``
        # so the rail reads as one family.
        tint_slot = getattr(btn, "_nav_icon_tint_slot", None) or "main_t1"
        pm = self._render_icon_pixmap(
            icon_name, self._icon_side(btn),
            tint=Config.get_color(tint_slot),
        )
        if pm is not None:
            label.setPixmap(pm)

    def _style_version_label(self) -> None:
        """Apply ``sub_t2`` colour + ``size_mini`` font to the V{x} label.

        Called on construction and re-applied on theme switch so the
        label stays in lock-step with the rest of the rail.
        """
        if self._version_label is None:
            return
        family = Config.get_value("Font", "family", "Microsoft YaHei")
        size = Config.get_font_size("size_mini")
        fg = Config.get_color("sub_t2")
        self._version_label.setStyleSheet(
            f"QLabel {{ "
            f"color: {fg}; "
            f"background: transparent; "
            f"font-family: \"{family}\"; "
            f"font-size: {size}pt; "
            f"}}"
        )

    def _apply_update_state(self, release_info: object = None) -> None:
        """Flip the Update button between its Latest / Update-Needed states.

        Connected to ``AppSignals.update_check_complete`` — and indirectly
        to ``I18n.language_changed`` via ``_reapply_update_state_locale``,
        which re-fires this method with the cached release so the short
        label / tooltip re-translate when the language switches in the
        gap between update checks.

        ``release_info`` is a ``ReleaseInfo`` when a newer release exists,
        or ``None`` for "up to date / offline / throttled-cache-empty".
        Cached on ``self._last_release_info`` so the locale re-fire can
        replay the most recent state without inventing one.
        """
        if self._update_btn is None:
            return
        self._last_release_info = release_info
        needed = release_info is not None
        new_icon = "icon_update_needed" if needed else "icon_update"
        if needed:
            new_text = tr(*self._UPDATE_SHORT_LABEL_NEEDED)
            new_tip = tr(
                "sidebar.update.tip_needed",
                "New version available — click to apply",
            )
        else:
            new_text = tr(*self._UPDATE_SHORT_LABEL_LATEST)
            new_tip = tr(
                "sidebar.update.tip_latest",
                "You are on the latest version",
            )
        self._update_btn._nav_icon_name = new_icon
        # Tint the "update needed" icon with the highlight accent so the rail
        # draws the eye; reset to the default rail tint when up to date.
        self._update_btn._nav_icon_tint_slot = "highlight" if needed else None
        self._refresh_aux_btn_icon(self._update_btn)
        text_label = getattr(self._update_btn, "_nav_text_label", None)
        if text_label is not None:
            text_label.setText(new_text)
        self._update_btn.setToolTip(new_tip)

    def _reapply_update_state_locale(self, *_args) -> None:
        """Re-fire ``_apply_update_state`` with the cached release info.

        Bound to ``I18n.language_changed`` so the imperative
        ``setText`` / ``setToolTip`` calls inside ``_apply_update_state``
        re-run with the new language's translations.
        """
        self._apply_update_state(self._last_release_info)

    def _on_lang_clicked(self) -> None:
        # Toggle: re-clicking while popup is up closes it.
        if self._lang_popup is not None and self._lang_popup.isVisible():
            self._lang_popup.close()
            return

        popup = _LanguagePopup(self)
        popup.language_picked.connect(self._on_lang_picked)
        popup.destroyed.connect(self._on_lang_popup_destroyed)
        self._lang_popup = popup

        # Anchor to the right edge of the language button, vertically
        # centred on it. Geometry is read after adjustSize so sizeHint
        # reflects the wheel's content.
        popup.adjustSize()
        anchor = self._lang_btn.mapToGlobal(QPoint(self._lang_btn.width(), 0))
        ph = popup.sizeHint().height()
        pos = QPoint(
            anchor.x() + 4,
            anchor.y() + self._lang_btn.height() // 2 - ph // 2,
        )
        popup.move(pos)
        popup.show()
        popup.setFocus()

    def _on_lang_picked(self, code: str) -> None:
        if code == I18n.get_language():
            return
        # Language is a DEVICE-LEVEL preference — persist to the
        # machine-level locale.ini at the WORKSPACE root so that signing
        # in / out / swapping accounts never changes what the user is
        # looking at. The subsequent ``I18n.set_language`` call updates
        # ``_current_lang`` in memory, writes the slug-scoped
        # ``user.ini`` overlay (harmless duplicate), and emits
        # ``language_changed`` so every bound widget re-translates:
        #   * SDK 工厂控件（I18nLabel / I18nButton / i18n_bind / i18n=True 的
        #     SDK 控件）自挂槽，自动重译；
        #   * MainWindow 在 __init__ 把 self.refresh 也挂到 language_changed，
        #     所以 windowTitle / statusBar 会跟着重算；
        #   * Canvas 自挂 _retranslate_canvas、NodeLibraryPanel 自挂 _refresh、
        #     UserPanel 自挂 _retranslate_auth …
        try:
            from application.service.user_workspace import write_machine_locale
            write_machine_locale(code)
        except Exception:                                          # pragma: no cover
            pass
        I18n.set_language(code)

    def _on_lang_toggled(self, _checked: bool) -> None:
        self._style_lang_icon_label()
        self._refresh_nav_btn_text(self._lang_btn)

    def _on_lang_popup_destroyed(self) -> None:
        self._lang_popup = None
        if self._lang_btn is not None:
            self._lang_btn.setChecked(False)

    def _on_nav_clicked(self, key: str) -> None:
        # Toggle: clicking the active button again collapses the panel.
        new_key = "" if key == self._active_key else key
        self._set_active(new_key)
        self.feature_requested.emit(key)

    def _set_active(self, key: str) -> None:
        if key == self._active_key:
            return

        self._active_key = key
        for btn_key, btn in self._buttons.items():
            btn.setChecked(btn_key == key)

        if key == "":
            if self._panel is not None:
                self._panel.setFixedWidth(0)
                self._panel.setVisible(False)
        else:
            if self._panel_stack is not None and key in self._panel_pages:
                self._panel_stack.setCurrentIndex(self._panel_pages[key])
            if self._panel is not None:
                self._panel.setVisible(True)
                self._panel.setFixedWidth(self._PANEL_W)

        self.panel_changed.emit(key)
