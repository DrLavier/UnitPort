# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ControllerPanel — mode tiles + bindings grid + controls + live values.

Layout (top -> bottom):

    Mode tile row (Off / Keyboard / Gamepad)
    Reset row     (Reset button only, right-aligned)
    Bindings grid (8 keyboard rows OR axes + 17 button rows OR off-hint)
    Controls row  (Invert vy + Invert vyaw checkboxes, gamepad mode only)
    Live values   (vx / vy / vyaw — 100 ms refresh)

Keyboard mode installs a global QApplication.installEventFilter so any key
press anywhere in the app gets routed to GlobalInputManager.forward_key_press.
The same filter intercepts capture-mode key presses to rebind a row.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QTimer,
    Qt,
    pyqtSlot,
)
from PyQt6.QtGui import QAction, QCursor, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    CheckboxItem,
    Config,
    LaviButton,
    i18n_bind,
    log_warning,
    setButton,
    setText,
    tr,
)

from application.service.input import (
    GAMEPAD_BUTTON_ACTIONS,
    get_global_input_manager,
)
from registers import controllers as _controllers


_MODE_TILE_PX = 44
_LIVE_REFRESH_MS = 100

# Keyboard action rows (action_id, default i18n key, default label).
_KEYBOARD_ROWS: List[Tuple[str, str, str]] = [
    ("forward",      "controller.kb.forward",      "Forward"),
    ("backward",     "controller.kb.backward",     "Backward"),
    ("strafe_left",  "controller.kb.strafe_left",  "Strafe left"),
    ("strafe_right", "controller.kb.strafe_right", "Strafe right"),
    ("yaw_left",     "controller.kb.yaw_left",     "Yaw left"),
    ("yaw_right",    "controller.kb.yaw_right",    "Yaw right"),
    ("estop",        "controller.kb.estop",        "Emergency stop"),
    ("boost",        "controller.kb.boost",        "Boost"),
]

# Gamepad rows: read-only axes + editable buttons.
_GAMEPAD_AXIS_ROWS: List[Tuple[str, str]] = [
    ("Stick L Y", "vx"),
    ("Stick L X", "vy"),
    ("Stick R X", "vyaw"),
    ("Stick R Y", "—"),
]
def _gamepad_button_rows() -> List[Tuple[str, str]]:
    """``(hw_id, label)`` for every gamepad button, from the controller registry —
    the single source of the available control set (no hardcoded list; the picker
    offers exactly what the registry declares for this controller type)."""
    return [(i["id"], i["label"]) for i in _controllers.list_inputs("gamepad", "button")]


class _GlobalKeyEventFilter(QObject):
    """Application-wide key event filter; ownership held by the controller panel."""

    def __init__(self, panel: "ControllerPanel") -> None:
        super().__init__(panel)
        self._panel = panel

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.KeyPress:
            try:
                key = int(event.key())
            except Exception:
                return False
            if event.isAutoRepeat():
                return False
            consumed = self._panel.handle_key_press(key)
            return bool(consumed)
        if et == QEvent.Type.KeyRelease:
            if event.isAutoRepeat():
                return False
            try:
                key = int(event.key())
            except Exception:
                return False
            self._panel.handle_key_release(key)
        return False


class _BindingRow(QFrame):
    """A single (action-name | hardware-mapping) row.

    Layout convention:
        LEFT  cell — fixed identifier (action name in keyboard mode, hardware
                     button label in gamepad mode). NO hover, NO cursor change.
        RIGHT cell — the user-editable mapping (bound key in keyboard mode,
                     mapped action in gamepad mode). Hover → background flips
                     to ``sidebar_hover_overlay``; cursor → PointingHand.

    Single click anywhere on an editable row triggers ``on_request_capture``
    (replaced double-click).
    """

    def __init__(
        self,
        action: str,
        left_text: str,
        right_text: str,
        *,
        editable: bool,
        on_request_capture=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.action = action
        self._editable = editable
        self._on_request_capture = on_request_capture
        self._capturing = False
        self._hover = False

        self.setObjectName("controllerBindingRow")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._left = QLabel(left_text)
        self._left.setMinimumWidth(110)
        self._left.setMaximumWidth(140)
        self._left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._left.setStyleSheet(self._cell_style(side="left"))
        row.addWidget(self._left, 0)

        self._right = QLabel(right_text)
        self._right.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._right.setStyleSheet(self._cell_style(side="right"))
        row.addWidget(self._right, 1)

        if editable:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    # ----- input ----------------------------------------------------------

    def mousePressEvent(self, event):  # type: ignore[override]
        if (event.button() == Qt.MouseButton.LeftButton
                and self._editable
                and self._on_request_capture is not None):
            self._on_request_capture(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event):  # type: ignore[override]
        if self._editable:
            self._hover = True
            self._right.setStyleSheet(self._cell_style(side="right"))
        super().enterEvent(event)

    def leaveEvent(self, event):  # type: ignore[override]
        if self._hover:
            self._hover = False
            self._right.setStyleSheet(self._cell_style(side="right"))
        super().leaveEvent(event)

    # ----- API ------------------------------------------------------------

    def set_left(self, text: str) -> None:
        self._left.setText(text)

    def set_right(self, text: str) -> None:
        self._right.setText(text)

    def set_capturing(self, on: bool) -> None:
        self._capturing = on
        if on:
            self._right.setText("…")
        self._right.setStyleSheet(self._cell_style(side="right"))

    # ----- styling --------------------------------------------------------

    def _cell_style(self, *, side: str) -> str:
        bg_default = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        fg = Config.get_color("main_t1")
        size = int(Config.get_font_size("size_small"))
        family = ('"Consolas", "Menlo", monospace'
                  if side == "right" else "inherit")
        bg = bg_default
        if side == "right":
            if self._capturing:
                fg = Config.get_color("highlight")
            elif self._editable and self._hover:
                bg = Config.get_color("row_1")
        return (
            f"QLabel {{ background: {bg}; color: {fg};"
            f" border: 1px solid {border};"
            f" padding: 4px 6px; font-family: {family};"
            f" font-size: {size}px; }}"
        )


class ControllerPanel(QWidget):
    """Sidebar content for the Controller key."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mgr = get_global_input_manager()
        self._key_filter: Optional[_GlobalKeyEventFilter] = None
        self._capturing_row: Optional[_BindingRow] = None
        self._kb_rows: List[_BindingRow] = []
        self._gp_rows: List[_BindingRow] = []
        self._mode_buttons: dict[str, LaviButton] = {}
        self._mode_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_mode_tile_row())
        layout.addLayout(self._build_reset_row())

        # Bindings host (swapped on mode change).
        self._bindings_host = QWidget()
        self._bindings_layout = QVBoxLayout(self._bindings_host)
        self._bindings_layout.setContentsMargins(0, 0, 0, 0)
        self._bindings_layout.setSpacing(2)
        layout.addWidget(self._bindings_host, 1)

        layout.addLayout(self._build_controls_row())
        layout.addWidget(self._build_live_values())

        self._mgr.mode_changed.connect(self._on_mode_changed)
        self._mgr.device_availability_changed.connect(self._on_device_changed)

        # Contract-driven policy channel bindings: the panel renders the
        # loaded bundle's command contract channels (per SKU) instead of
        # the static axis table. Source of truth = the CURRENTLY SELECTED
        # bundle's deploy_contract.commands (policy_contract_changed), which
        # is persistent while a policy is selected — NOT gated on a running
        # live-review (policy_review_started, kept only as a redundant re-send).
        self._policy_sku: str = ""
        self._policy_channels: Optional[list] = None
        self._policy_legacy: bool = False
        self._policy_loaded: bool = False   # a policy is selected (bundle present)
        self._gp_channel_rows: list = []
        self._channel_by_name: dict = {}    # name -> contract channel dict (pickers)
        self._kb_trigger_channels: set = set()   # trigger channels shown in keyboard mode
        try:
            from application.service.signals import get_app_signals
            get_app_signals().policy_contract_changed.connect(
                self._on_policy_contract_changed
            )
            get_app_signals().policy_review_started.connect(
                self._on_policy_review_started
            )
        except Exception as exc:
            log_warning(f"[controller] policy signal wiring failed: {exc}")

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(_LIVE_REFRESH_MS)
        self._refresh_timer.timeout.connect(self._refresh_live_values)
        self._refresh_timer.start()

        self._availability_timer = QTimer(self)
        self._availability_timer.setInterval(2000)
        self._availability_timer.timeout.connect(self._refresh_mode_label_styles)
        self._availability_timer.start()

        # Render the manager's current mode (which is whatever boot mode landed).
        self._sync_to_manager_mode(self._mgr.mode)

    # ----- mode tile row --------------------------------------------------

    def _build_mode_tile_row(self) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        for mode, icon_name, default_label in (
            ("off",      "icon_mode_off",      "Off"),
            ("keyboard", "icon_mode_keyboard", "Keyboard"),
            ("gamepad",  "icon_mode_gamepad",  "Gamepad"),
        ):
            tile = self._make_mode_tile(mode, icon_name, default_label)
            row.addWidget(tile)
        row.addStretch(1)
        return wrap

    def _make_mode_tile(self, mode: str, icon_name: str, default_label: str) -> QWidget:
        column = QWidget()
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # setButton handles: checkable, PointingHandCursor, hover/checked QSS,
        # icon resolution via Assets.find_icon, theme refresh hooks.
        btn = setButton(
            f"controller.mode_btn.{mode}",
            _MODE_TILE_PX, _MODE_TILE_PX,
            kind="light", spec="none",
            icon=icon_name, default=default_label,
            icon_only=True, checkable=True,
            checked_color=Config.get_color("btn_1"),
            hover_color=Config.get_color("hover_1"),
        )
        btn.clicked.connect(lambda _c=False, m=mode: self._on_mode_clicked(m))
        col.addWidget(btn)

        label = QLabel()
        i18n_bind(label, "setText",
                  f"controller.mode.{mode}", default_label)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        label.setStyleSheet(
            f"background: transparent;"
            f" font-size: {Config.get_font_size('size_small')}px;"
        )
        col.addWidget(label)

        self._mode_buttons[mode] = btn
        self._mode_labels[mode] = label
        return column

    # ----- bindings -------------------------------------------------------

    def _populate_off(self) -> None:
        self._clear_bindings()
        hint = setText(
            "controller.off_hint",
            default="Input disabled. Pick Keyboard or Gamepad above.",
            kind="content",
            size=int(Config.get_font_size("size_small")),
        )
        hint.setWordWrap(True)
        self._bindings_layout.addWidget(hint)
        self._bindings_layout.addStretch(1)

    def _populate_keyboard(self) -> None:
        self._clear_bindings()
        self._kb_trigger_channels = set()
        bindings = self._mgr.get_keyboard_bindings()
        self._kb_rows = []
        self._bindings_layout.addWidget(
            self._section_header("controller.system_actions_keys", "Keys → System Actions")
        )
        for action, key_i18n, default_label in _KEYBOARD_ROWS:
            key_code = bindings.get(action, 0)
            key_text = QKeySequence(int(key_code)).toString() if key_code else "—"
            # Layout convention (see _BindingRow): action name on the LEFT
            # (fixed, no hover); bound key on the RIGHT (user-mapped column,
            # hover + cursor + click-to-rebind).
            row = _BindingRow(
                action=action,
                left_text=tr(key_i18n, default_label),
                right_text=key_text or "—",
                editable=True,
                on_request_capture=self._begin_capture,
            )
            self._kb_rows.append(row)
            self._bindings_layout.addWidget(row)
        # Skill trigger channels — keyboard binding parity (capture a key).
        self._populate_keyboard_triggers()
        self._bindings_layout.addStretch(1)

    def _populate_keyboard_triggers(self) -> None:
        """Render the loaded policy's trigger channels as capture-to-bind key rows,
        so a keyboard user can bind + fire a skill (parity with the gamepad picker)."""
        channels = [
            c for c in (self._policy_channels or [])
            if isinstance(c, dict) and str(c.get("kind")) == "trigger"
        ]
        if not channels:
            return
        self._bindings_layout.addWidget(self._section_header(
            "controller.policy_channels", "Policy Command Channels"))
        button_bindings = self._mgr.get_channel_button_bindings(self._policy_sku)
        for ch in channels:
            name = str(ch.get("name", "") or "")
            if not name:
                continue
            self._kb_trigger_channels.add(name)
            self._channel_by_name[name] = ch
            try:
                key = int((button_bindings.get(name) or {}).get("key", 0) or 0)
            except (TypeError, ValueError):
                key = 0
            key_text = QKeySequence(int(key)).toString() if key else "—"
            row = _BindingRow(
                action=name, left_text=name, right_text=key_text or "—",
                editable=True, on_request_capture=self._begin_capture,
            )
            self._kb_rows.append(row)
            self._bindings_layout.addWidget(row)

    def _populate_gamepad(self) -> None:
        self._clear_bindings()
        # Two visually-separate categories (kept apart deliberately — flattening
        # command channels and system actions into one list is what confused
        # users): ① policy command channels (the command source, contract-driven);
        # ② physical buttons → system actions (boost/estop/…, policy-independent).
        self._populate_policy_section()
        self._populate_button_actions()

    def _section_header(self, key: str, default: str):
        return setText(
            key, default=default, kind="content",
            size=int(Config.get_font_size("size_small")),
        )

    def _populate_policy_section(self) -> None:
        """① Policy Command Channels — three states (constraint 4): a v1 contract
        renders bindable rows; a loaded-but-legacy / no-contract bundle shows a
        re-export hint; no policy loaded shows an unavailable placeholder (never a
        pile of unbound rows)."""
        if self._policy_channels is not None:
            self._populate_policy_channels()
            return
        self._bindings_layout.addWidget(
            self._section_header("controller.policy_channels", "Policy Command Channels")
        )
        if self._policy_legacy:
            key, dflt = "controller.legacy_contract_hint", (
                "Loaded policy carries a legacy commands block — re-export the "
                "bundle to bind command channels here."
            )
        elif self._policy_loaded:
            key, dflt = "controller.no_contract_hint", (
                "This policy has no command channels — re-export the bundle after "
                "authoring velocity commands or a skill on the Training Motion node."
            )
        else:
            key, dflt = "controller.no_policy_hint", (
                "No policy loaded. A skill you author on the Training Motion node "
                "becomes a bindable channel HERE only after you train + export a "
                "policy and select it in Mission Control · Simulate — a trigger is a "
                "trained behavior, not just a key mapping. The buttons below are "
                "system actions (boost / estop / …): always available, separate from "
                "any policy, which is why they are not on the Training Motion node."
            )
        hint = setText(
            key, default=dflt, kind="content",
            size=int(Config.get_font_size("size_small")),
        )
        hint.setWordWrap(True)
        self._bindings_layout.addWidget(hint)

    def _populate_button_actions(self) -> None:
        """② Physical buttons → system actions (boost/estop/mode/…). Always shown;
        policy-independent. Kept in a separate section from the command channels."""
        self._bindings_layout.addWidget(
            self._section_header("controller.system_actions", "Buttons → System Actions")
        )
        self._gp_rows = []
        for hw_id, hw_label in _gamepad_button_rows():
            current_action = self._mgr.get_gamepad_button_action(hw_id)
            label = self._action_label(current_action)
            row = _BindingRow(
                action=hw_id,
                left_text=hw_label,
                right_text=label,
                editable=True,
                on_request_capture=self._begin_action_pick,
            )
            self._gp_rows.append(row)
            self._bindings_layout.addWidget(row)
        self._bindings_layout.addStretch(1)

    def _clear_bindings(self) -> None:
        while self._bindings_layout.count():
            item = self._bindings_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._kb_rows = []
        self._gp_rows = []

    def _action_label(self, action_id: str) -> str:
        for aid, label in GAMEPAD_BUTTON_ACTIONS:
            if aid == action_id:
                return label
        return GAMEPAD_BUTTON_ACTIONS[0][1]

    # ----- reset row ------------------------------------------------------

    def _build_reset_row(self) -> QHBoxLayout:
        """Reset button alone, right-aligned, between mode tiles and bindings."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addStretch(1)
        self._reset_btn = setButton(
            "controller.reset", 80, 24,
            kind="border", spec="none",
            default="Reset",
        )
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        row.addWidget(self._reset_btn)
        return row

    # ----- controls row (gamepad invert toggles) --------------------------

    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        small_font = int(Config.get_font_size("size_small"))

        self._invert_vy_check = CheckboxItem(
            "controller.invert_vy", default="Invert vy",
            checked=self._mgr.invert_vy,
            font_size=small_font,
        )
        self._invert_vy_check.toggled.connect(self._on_invert_vy_toggled)
        row.addWidget(self._invert_vy_check)

        self._invert_vyaw_check = CheckboxItem(
            "controller.invert_vyaw", default="Invert vyaw",
            checked=self._mgr.invert_vyaw,
            font_size=small_font,
        )
        self._invert_vyaw_check.toggled.connect(self._on_invert_vyaw_toggled)
        row.addWidget(self._invert_vyaw_check)
        row.addStretch(1)
        return row

    # ----- live values ----------------------------------------------------

    def _build_live_values(self) -> QWidget:
        wrap = QFrame()
        wrap.setFrameShape(QFrame.Shape.NoFrame)
        grid = QGridLayout(wrap)
        grid.setContentsMargins(2, 4, 2, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)

        self._live_value_labels: dict[str, QLabel] = {}
        muted = Config.get_color("main_c2")
        head_size = int(Config.get_font_size("size_mini"))
        val_size = int(Config.get_font_size("size_small"))

        for col, field in enumerate(self._mgr.field_order):
            head = QLabel(field)
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head.setStyleSheet(
                f"color: {muted}; background: transparent;"
                f" font-size: {head_size}px;"
            )
            grid.addWidget(head, 0, col)

            val = QLabel("0.00")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(
                f"color: {Config.get_color('sub_t2')};"
                f" background: transparent;"
                f" font-family: Consolas, Menlo, monospace;"
                f" font-size: {val_size}px;"
            )
            val.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(val, 1, col)
            self._live_value_labels[field] = val
        return wrap

    def _refresh_live_values(self) -> None:
        values = self._mgr.get_live_values()
        muted = Config.get_color("sub_t2")
        val_size = int(Config.get_font_size("size_small"))
        for field, label in self._live_value_labels.items():
            v = float(values.get(field, 0.0))
            label.setText(f"{v:+.2f}")
            mag = min(1.0, abs(v))
            if mag < 0.05:
                label.setStyleSheet(
                    f"color: {muted};"
                    f" background: transparent;"
                    f" font-family: Consolas, Menlo, monospace;"
                    f" font-size: {val_size}px;"
                )
            else:
                # Blend toward safe-zone green proportional to magnitude.
                green = int(120 + (255 - 120) * mag)
                label.setStyleSheet(
                    f"color: rgb(120, {green}, 140);"
                    f" background: transparent;"
                    f" font-family: Consolas, Menlo, monospace;"
                    f" font-size: {val_size}px;"
                )

    # ----- mode lifecycle -------------------------------------------------

    def _on_mode_clicked(self, mode: str) -> None:
        ok = self._mgr.set_mode(mode)
        if not ok and mode == "gamepad":
            log_warning("[controller] gamepad activation failed (device not available)")
        # Sticky: panel reflects the user's choice even when device missing.
        self._sync_to_manager_mode(mode)

    @pyqtSlot(str)
    def _on_mode_changed(self, mode: str) -> None:
        self._sync_to_manager_mode(mode)

    @pyqtSlot(str, bool)
    def _on_device_changed(self, _mode: str, _available: bool) -> None:
        self._refresh_mode_label_styles()

    def _sync_to_manager_mode(self, mode: str) -> None:
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)
        self._refresh_mode_label_styles()

        if mode == "keyboard":
            self._populate_keyboard()
        elif mode == "gamepad":
            self._populate_gamepad()
        else:
            self._populate_off()

        self._reset_btn.setEnabled(mode in ("keyboard", "gamepad"))
        # Command-frame invert is meaningful only for gamepad.
        gamepad_mode = mode == "gamepad"
        self._invert_vy_check.setVisible(gamepad_mode)
        self._invert_vyaw_check.setVisible(gamepad_mode)
        if gamepad_mode:
            # Block to avoid feedback loop during programmatic refresh.
            for chk, current in (
                (self._invert_vy_check, self._mgr.invert_vy),
                (self._invert_vyaw_check, self._mgr.invert_vyaw),
            ):
                chk.blockSignals(True)
                chk.setChecked(bool(current))
                chk.blockSignals(False)

        if mode == "keyboard":
            self._install_key_filter()
        else:
            self._remove_key_filter()

    def _refresh_mode_label_styles(self) -> None:
        active_mode = self._mgr.mode
        muted = Config.get_color("main_c2", "#999999")
        ok_color = Config.get_color("safe_zone", "#36E38E")
        warn_color = Config.get_color("danger_zone", "#FF6B6B")
        for mode, label in self._mode_labels.items():
            if mode != active_mode:
                label.setStyleSheet(f"color: {muted}; background: transparent;")
                continue
            available = self._mgr.is_device_available(mode)
            color = ok_color if available or mode == "off" else warn_color
            label.setStyleSheet(
                f"color: {color}; background: transparent; font-weight: bold;"
            )

    # ----- keyboard event filter -----------------------------------------

    def _install_key_filter(self) -> None:
        if self._key_filter is not None:
            return
        app = QApplication.instance()
        if app is None:
            return
        self._key_filter = _GlobalKeyEventFilter(self)
        app.installEventFilter(self._key_filter)

    def _remove_key_filter(self) -> None:
        if self._key_filter is None:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._key_filter)
        self._key_filter.deleteLater()
        self._key_filter = None

    def handle_key_press(self, key: int) -> bool:
        if self._capturing_row is not None:
            self._consume_capture(key)
            return True
        # Don't swallow Tab/Shift navigation keys etc — only the bound ones.
        # A key bound to a movement action OR to a skill trigger channel forwards.
        bindings = self._mgr.get_keyboard_bindings()
        if key not in bindings.values() and key not in self._bound_trigger_keys():
            return False
        self._mgr.forward_key_press(key)
        return False

    def _bound_trigger_keys(self) -> set:
        keys = set()
        for spec in (self._mgr.get_channel_button_bindings(self._policy_sku) or {}).values():
            try:
                k = int((spec or {}).get("key", 0) or 0)
            except (TypeError, ValueError):
                k = 0
            if k:
                keys.add(k)
        return keys

    def handle_key_release(self, key: int) -> None:
        self._mgr.forward_key_release(key)

    # ----- keyboard binding capture --------------------------------------

    def _begin_capture(self, row: _BindingRow) -> None:
        if self._capturing_row is not None:
            self._capturing_row.set_capturing(False)
        self._capturing_row = row
        row.set_capturing(True)

    def _consume_capture(self, key: int) -> None:
        if self._capturing_row is None:
            return
        row = self._capturing_row
        row.set_capturing(False)
        self._capturing_row = None
        if row.action in self._kb_trigger_channels:
            self._bind_trigger_key(row, int(key))
            return
        ok = self._mgr.set_keyboard_binding(row.action, key)
        if ok:
            row.set_right(QKeySequence(int(key)).toString() or "—")

    def _bind_trigger_key(self, row, key: int) -> None:
        """Bind a captured keyboard key to a skill trigger channel (with the same
        latch gating + conflict guard as the gamepad picker)."""
        channel = row.action
        claim = self._key_claims(exclude=("trigger", channel)).get(int(key))
        if claim is not None:
            self._reject_conflict(QKeySequence(int(key)).toString() or str(key), claim)
            return
        ch = self._channel_by_name.get(channel, {})
        supported = self._supported_latch(ch) if ch else ["pulse"]
        spec = (self._mgr.get_channel_button_bindings(self._policy_sku) or {}).get(channel) or {}
        latch = str(spec.get("latch", "") or "")
        if latch not in supported:
            latch = supported[0]
        try:
            self._mgr.set_channel_key_binding(self._policy_sku, channel, int(key), latch)
        except ValueError as exc:
            log_warning(f"[controller] trigger key binding rejected: {exc}")
            return
        row.set_right(QKeySequence(int(key)).toString() or "—")

    def _kb_action_label(self, action: str) -> str:
        for a, _i18n, label in _KEYBOARD_ROWS:
            if a == action:
                return tr(_i18n, label)
        return str(action)

    def _key_claims(self, *, exclude=None) -> dict:
        """``{key_code: (kind, key, label)}`` across keyboard movement/system
        actions (#1) and keyboard-bound trigger channels (#3) for the current SKU —
        the keyboard analogue of _input_claims (one key serves one binding)."""
        claims: dict = {}
        for action, code in (self._mgr.get_keyboard_bindings() or {}).items():
            try:
                c = int(code)
            except (TypeError, ValueError):
                c = 0
            if not c or exclude == ("action", str(action)):
                continue
            claims[c] = ("action", str(action), self._kb_action_label(action))
        for ch, spec in (self._mgr.get_channel_button_bindings(self._policy_sku) or {}).items():
            try:
                c = int((spec or {}).get("key", 0) or 0)
            except (TypeError, ValueError):
                c = 0
            if not c or exclude == ("trigger", str(ch)):
                continue
            claims[c] = ("trigger", str(ch), str(ch))
        return claims

    # ----- gamepad button action picker ----------------------------------

    def _begin_action_pick(self, row: _BindingRow) -> None:
        if self._mgr.mode != "gamepad":
            return
        catalog = self._mgr.gamepad_button_action_catalog()
        hw_button = row.action
        current_id = self._mgr.get_gamepad_button_action(hw_button)

        bg = Config.get_color("bg_3", "#101010")
        fg = Config.get_color("main_t1", "#D6D3C7")
        border = Config.get_color("border_1", "#444444")
        accent_bg = Config.get_color("bg_2", "#1A1A1A")

        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {bg}; color: {fg}; border: 1px solid {border}; }}"
            f"QMenu::item {{ padding: 4px 14px; }}"
            f"QMenu::item:selected {{ background: {accent_bg}; }}"
        )
        for action_id, action_label in catalog:
            item = QAction(action_label, menu)
            item.setCheckable(True)
            item.setChecked(action_id == current_id)
            item.triggered.connect(
                lambda _checked=False, _hw=hw_button, _id=action_id:
                    self._apply_action_pick(_hw, _id)
            )
            menu.addAction(item)
        pos = row.mapToGlobal(QPoint(0, row.height()))
        menu.exec(pos)

    def _apply_action_pick(self, hw_button: str, action_id: str) -> None:
        # Conflict guard (constraint 2): this physical button may already drive a
        # trigger channel for the loaded policy — refuse rather than double-bind.
        if action_id:
            claim = self._input_claims(exclude=("action", hw_button)).get(hw_button)
            if claim is not None:
                self._reject_conflict(self._button_label(hw_button), claim)
                return
        self._mgr.set_gamepad_button_binding(hw_button, action_id)
        for row in self._gp_rows:
            if row.action == hw_button:
                row.set_right(self._action_label(action_id))
                break

    # ----- policy command channels (contract-driven) ------------------------

    @pyqtSlot(str, object)
    def _on_policy_contract_changed(self, sku: str, commands: object) -> None:
        """policy_contract_changed — the CURRENTLY SELECTED bundle's command
        contract (persistent while a policy is selected; NOT gated on a running
        live-review). ``("", None)`` = no policy selected → unavailable state."""
        loaded = bool(str(sku or "")) or (commands is not None)
        self._set_policy_contract(sku, commands, loaded=loaded)

    @pyqtSlot(str, object)
    def _on_policy_review_started(self, sku: str, commands: object) -> None:
        """policy_review_started — a live-sim replay started; a redundant
        re-send of the same contract (a policy is always loaded at review)."""
        self._set_policy_contract(sku, commands, loaded=True)

    def _set_policy_contract(self, sku, commands, loaded: bool) -> None:
        """Single entry point that ingests a command contract and refreshes the
        binding UI + axis routing. Both the selection signal and the review signal
        route through here so the panel renders AND installs routing on selection
        (bound == usable, not only once a review is running)."""
        self._policy_sku = str(sku or "")
        self._policy_loaded = bool(loaded)
        channels = None
        legacy = False
        if isinstance(commands, dict) and commands:
            if commands.get("contract_version") is not None:
                channels = [
                    c for c in (commands.get("channels") or [])
                    if isinstance(c, dict)
                ]
            else:
                legacy = True
        self._policy_channels = channels
        self._policy_legacy = legacy
        # Refresh whichever mode is active — a trigger channel is bindable in BOTH
        # gamepad (button picker) and keyboard (key capture), so a contract change
        # must re-render either. (Was gamepad-only: keyboard trigger rows never
        # appeared until the user toggled modes.)
        mode = self._mgr.mode
        if mode == "gamepad":
            self._populate_gamepad()
        elif mode == "keyboard":
            self._populate_keyboard()
        self._apply_channel_routing()

    def _populate_policy_channels(self) -> None:
        header = setText(
            "controller.policy_channels",
            default="Policy Command Channels",
            kind="content",
            size=int(Config.get_font_size("size_small")),
        )
        self._bindings_layout.addWidget(header)
        axis_bindings = self._mgr.get_channel_axis_bindings(self._policy_sku)
        button_bindings = self._mgr.get_channel_button_bindings(self._policy_sku)
        known = set(self._mgr.known_axis_ids())
        self._gp_channel_rows = []
        self._channel_by_name = {}
        contract_names = set()
        for i, ch in enumerate(self._policy_channels or []):
            name = str(ch.get("name", "") or "")
            if not name:
                continue
            contract_names.add(name)
            self._channel_by_name[name] = ch
            kind = str(ch.get("kind", "") or "")
            if kind == "trigger":
                # A trigger channel binds to a gamepad BUTTON + a latch MODE; a
                # press fires the channel's envelope into the command vector.
                spec = button_bindings.get(name) or {}
                hw = str(spec.get("button", "") or "") or self._suggested_button(ch)
                supported = self._supported_latch(ch)
                latch = str(spec.get("latch", "") or "")
                if latch not in supported:
                    latch = supported[0]
                right = self._trigger_value_label(hw, latch) if hw else "—"
                row = _BindingRow(
                    action=name, left_text=name, right_text=right,
                    editable=True, on_request_capture=self._begin_button_pick,
                )
            elif i < 3 and kind == "continuous":
                # The live-sim provider feeds velocity from the first 3 continuous
                # channels [vx, vy, vyaw] (stick axes).
                axis = axis_bindings.get(name) or self._suggested_axis(ch)
                if axis and axis not in known:
                    # Never silently drop a stored binding: render it greyed
                    # with a warning and tell the user (§8).
                    right = "⚠ " + axis
                    log_warning(
                        f"[controller] stored binding {name!r} -> {axis!r} for "
                        f"sku={self._policy_sku!r} references an unknown axis "
                        f"(known: {sorted(known)}); re-bind the channel."
                    )
                else:
                    right = self._axis_label(axis) if axis else "—"
                row = _BindingRow(
                    action=name, left_text=name, right_text=right,
                    editable=True, on_request_capture=self._begin_axis_pick,
                )
            else:
                right = tr("controller.channel_not_bindable", "— (not bindable yet)")
                row = _BindingRow(
                    action=name, left_text=name, right_text=right,
                    editable=False, on_request_capture=self._begin_axis_pick,
                )
            self._gp_channel_rows.append(row)
            self._bindings_layout.addWidget(row)
        self._render_stale_channel_bindings(
            contract_names, axis_bindings, button_bindings
        )

    def _render_stale_channel_bindings(
        self, contract_names, axis_bindings, button_bindings
    ) -> None:
        """Saved bindings whose CHANNEL is not in the loaded contract must not be
        silently dropped (§8 / storage stays UI-visible, constraint 4): render them
        greyed with a warning so the user sees + can clear them."""
        stale = []
        for ch_name, axis in (axis_bindings or {}).items():
            if str(ch_name) not in contract_names and axis:
                stale.append((str(ch_name), self._axis_label(str(axis))))
        for ch_name, spec in (button_bindings or {}).items():
            hw = str((spec or {}).get("button", "") or "")
            if str(ch_name) not in contract_names and hw:
                stale.append((str(ch_name), self._button_label(hw)))
        if not stale:
            return
        self._bindings_layout.addWidget(self._section_header(
            "controller.stale_channels", "Saved bindings not in this contract"))
        for ch_name, val in stale:
            log_warning(
                f"[controller] stale binding {ch_name!r} -> {val!r} for "
                f"sku={self._policy_sku!r}: channel not in the loaded contract; "
                f"re-bind or clear it."
            )
            row = _BindingRow(
                action=f"__stale__{ch_name}", left_text=ch_name,
                right_text="⚠ " + val, editable=False,
            )
            self._gp_channel_rows.append(row)
            self._bindings_layout.addWidget(row)

    @staticmethod
    def _suggested_axis(ch: dict) -> str:
        """Default axis from the contract's informational suggested_binding
        (``"gamepad.<axis_id>"``); empty when absent / not a gamepad hint."""
        sb = str(ch.get("suggested_binding") or "")
        if sb.startswith("gamepad."):
            return sb[len("gamepad."):]
        return ""

    @staticmethod
    def _suggested_button(ch: dict) -> str:
        """Default BUTTON from the contract's suggested_binding, only when it names
        a known gamepad button (not an axis)."""
        sb = str(ch.get("suggested_binding") or "")
        if sb.startswith("gamepad."):
            cand = sb[len("gamepad."):]
            if _controllers.has_input("gamepad", cand):
                return cand
        return ""

    def _supported_latch(self, ch: dict) -> list:
        """Latch modes the trained policy supports for this trigger channel, from
        the contract's ``supported_latch`` capability. A pre-capability bundle
        (§8(c) legacy) defaults to ['pulse'] ONLY (never all-supported) + WARN."""
        raw = ch.get("supported_latch")
        modes = [str(m).strip().lower() for m in raw] if isinstance(raw, (list, tuple)) else []
        modes = [m for m in modes if m in ("pulse", "hold")]
        if not modes:
            log_warning(
                f"[controller] trigger channel {ch.get('name')!r} carries no "
                f"supported_latch (legacy bundle) — defaulting to ['pulse'] and "
                f"gating hold OFF; re-export the bundle to declare capabilities."
            )
            return ["pulse"]
        return modes

    def _latch_mode_label(self, mode: str) -> str:
        return {
            "pulse": tr("controller.latch.pulse", "Pulse (press once)"),
            "hold": tr("controller.latch.hold", "Hold (while held)"),
        }.get(mode, mode)

    def _trigger_value_label(self, hw: str, latch: str) -> str:
        return f"{self._button_label(hw)} · {self._latch_mode_label(latch)}"

    def _axis_label(self, axis_id: str) -> str:
        return tr(f"controller.axis.{axis_id}", axis_id)

    def _begin_axis_pick(self, row) -> None:
        menu = QMenu(self)
        act_none = QAction(tr("controller.axis_unbound", "— (unbound)"), menu)
        act_none.triggered.connect(
            lambda _c=False, r=row: self._apply_axis_pick(r.action, "")
        )
        menu.addAction(act_none)
        for axis_id in self._mgr.known_axis_ids():
            act = QAction(self._axis_label(axis_id), menu)
            act.triggered.connect(
                lambda _c=False, r=row, a=axis_id: self._apply_axis_pick(r.action, a)
            )
            menu.addAction(act)
        menu.exec(QCursor.pos())

    def _apply_axis_pick(self, channel: str, axis_id: str) -> None:
        # Conflict guard (constraint 2): another command channel may already own
        # this axis — refuse rather than silently double-bind.
        if axis_id:
            claim = self._input_claims(exclude=("axis", channel)).get(axis_id)
            if claim is not None:
                self._reject_conflict(self._axis_label(axis_id), claim)
                return
        try:
            self._mgr.set_channel_axis_binding(self._policy_sku, channel, axis_id)
        except ValueError as exc:
            log_warning(f"[controller] axis binding rejected: {exc}")
            return
        if self._mgr.mode == "gamepad":
            self._populate_gamepad()
        self._apply_channel_routing()

    # ----- skill trigger channel → button picker (Slice 4) -----------------

    def _button_label(self, hw: str) -> str:
        if not hw:
            return "—"
        return tr(f"controller.button.{hw}", _controllers.input_label("gamepad", hw))

    def _begin_button_pick(self, row) -> None:
        channel = row.action
        ch = self._channel_by_name.get(channel, {})
        supported = self._supported_latch(ch) if ch else ["pulse"]
        spec = (self._mgr.get_channel_button_bindings(self._policy_sku) or {}).get(channel) or {}
        cur_hw = str(spec.get("button", "") or "")
        cur_latch = str(spec.get("latch", "") or "")
        if cur_latch not in supported:
            cur_latch = supported[0]
        menu = QMenu(self)
        # ── Button ──
        menu.addSection(tr("controller.pick.button", "Button"))
        act_none = QAction(tr("controller.button_unbound", "— (unbound)"), menu)
        act_none.triggered.connect(
            lambda _c=False: self._apply_button_pick(channel, "", cur_latch)
        )
        menu.addAction(act_none)
        for hw_id, label in _gamepad_button_rows():
            act = QAction(tr(f"controller.button.{hw_id}", label), menu)
            act.setCheckable(True)
            act.setChecked(hw_id == cur_hw)
            act.triggered.connect(
                lambda _c=False, h=hw_id: self._apply_button_pick(channel, h, cur_latch)
            )
            menu.addAction(act)
        # ── Mode (latch) — only contract-declared modes are selectable ──
        menu.addSection(tr("controller.pick.mode", "Mode"))
        for mode in ("pulse", "hold"):
            label = self._latch_mode_label(mode)
            if mode not in supported:
                label += tr("controller.latch.unsupported", "  (training side didn't provide)")
            act = QAction(label, menu)
            act.setCheckable(True)
            act.setChecked(mode == cur_latch)
            if mode in supported:
                act.triggered.connect(
                    lambda _c=False, m=mode: self._apply_button_pick(channel, cur_hw, m)
                )
            else:
                act.setEnabled(False)
            menu.addAction(act)
        menu.exec(QCursor.pos())

    def _apply_button_pick(self, channel: str, hw: str, latch=None) -> None:
        latch = str(latch or "pulse")
        if hw:
            claim = self._input_claims(exclude=("trigger", channel)).get(hw)
            if claim is not None:
                self._reject_conflict(self._button_label(hw), claim)
                return
        try:
            self._mgr.set_channel_button_binding(self._policy_sku, channel, hw, latch)
        except ValueError as exc:
            log_warning(f"[controller] trigger button binding rejected: {exc}")
            return
        if self._mgr.mode == "gamepad":
            self._populate_gamepad()

    # ----- unified physical-input claim view + conflict guard --------------

    def _input_claims(self, *, exclude=None) -> dict:
        """``{physical_input_id: (kind, key, label)}`` for the current SKU across
        all three binding maps — system actions (buttons, #1), continuous channels
        (axes, #2), trigger channels (buttons, #3). ``exclude=(kind, key)`` skips a
        binding so a rebind of the SAME owner is never a self-conflict. D-Pad is
        button-space only (hat→buttons), so buttons and axes never collide."""
        claims: dict = {}
        sku = self._policy_sku
        for hw, action in (self._mgr.get_gamepad_button_bindings() or {}).items():
            if not action or exclude == ("action", str(hw)):
                continue
            claims[str(hw)] = ("action", str(hw), self._action_label(str(action)))
        for ch, spec in (self._mgr.get_channel_button_bindings(sku) or {}).items():
            hw = str((spec or {}).get("button", "") or "")
            if not hw or exclude == ("trigger", str(ch)):
                continue
            claims[hw] = ("trigger", str(ch), str(ch))
        for ch, axis in (self._mgr.get_channel_axis_bindings(sku) or {}).items():
            if not axis or exclude == ("axis", str(ch)):
                continue
            claims[str(axis)] = ("axis", str(ch), str(ch))
        return claims

    def _reject_conflict(self, input_label: str, claimant) -> None:
        """Fail-loud: one physical input serves one binding. Refuse + tell the user
        to unbind it first (constraint 2 — no silent double-bind)."""
        kind, key, owner = claimant
        cat = {
            "action": tr("controller.conflict.action", "a system action"),
            "trigger": tr("controller.conflict.trigger", "a skill trigger"),
            "axis": tr("controller.conflict.channel", "a command channel"),
        }.get(kind, kind)
        log_warning(
            f"[controller] binding conflict: {input_label!r} already bound to "
            f"{kind}:{key!r} for sku={self._policy_sku!r} — refused."
        )
        msg = tr(
            "controller.conflict.msg",
            "{input} is already bound to {owner} ({cat}). One physical input "
            "serves one binding — unbind it there first, then try again.",
        )
        msg = (msg.replace("{input}", input_label)
                  .replace("{owner}", str(owner)).replace("{cat}", cat))
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, tr("controller.conflict.title", "Binding conflict"), msg
            )
        except Exception:
            pass

    def _apply_channel_routing(self) -> None:
        """Derive the bus-field routing from the contract channels + stored
        bindings and install it on the input manager. The live-sim provider
        reads the positional fields (vx, vy, vyaw) = contract channels
        0/1/2 — the contract orders the velocity trio first."""
        channels = self._policy_channels
        if not self._policy_sku or not channels:
            self._mgr.apply_channel_routing(None)
            return
        bindings = self._mgr.get_channel_axis_bindings(self._policy_sku)
        known = set(self._mgr.known_axis_ids())
        fields = ["vx", "vy", "vyaw"]
        routing = {}
        for i, ch in enumerate(channels[:3]):
            if str(ch.get("kind", "") or "") != "continuous":
                continue
            name = str(ch.get("name", "") or "")
            axis = bindings.get(name) or self._suggested_axis(ch)
            if not axis:
                continue
            if axis not in known:
                log_warning(
                    f"[controller] channel {name!r} binding {axis!r} is not "
                    f"a known axis — the channel stays undriven until "
                    f"re-bound."
                )
                continue
            field = fields[i]
            routing[field] = (axis, self._mgr.default_axis_sign(field, axis))
        self._mgr.apply_channel_routing(routing or None)

    # ----- reset / invert -------------------------------------------------

    def _on_reset_clicked(self) -> None:
        if self._mgr.mode == "keyboard":
            self._mgr.reset_keyboard_bindings()
            self._populate_keyboard()
        elif self._mgr.mode == "gamepad":
            self._mgr.reset_gamepad_button_bindings()
            self._populate_gamepad()

    def _on_invert_vy_toggled(self, checked: bool) -> None:
        self._mgr.set_invert_vy(bool(checked))

    def _on_invert_vyaw_toggled(self, checked: bool) -> None:
        self._mgr.set_invert_vyaw(bool(checked))


__all__ = ["ControllerPanel"]
