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
_GAMEPAD_BUTTON_HW_ROWS: List[Tuple[str, str]] = [
    ("dpad_up",    "D-Pad Up"),
    ("dpad_down",  "D-Pad Down"),
    ("dpad_left",  "D-Pad Left"),
    ("dpad_right", "D-Pad Right"),
    ("a",          "A"),
    ("b",          "B"),
    ("x",          "X"),
    ("y",          "Y"),
    ("lb",         "Left Bumper"),
    ("rb",         "Right Bumper"),
    ("lt",         "Left Trigger"),
    ("rt",         "Right Trigger"),
    ("ls",         "Stick L Click"),
    ("rs",         "Stick R Click"),
    ("start",      "Start"),
    ("back",       "Back"),
    ("guide",      "Guide"),
]


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
        bindings = self._mgr.get_keyboard_bindings()
        self._kb_rows = []
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
        self._bindings_layout.addStretch(1)

    def _populate_gamepad(self) -> None:
        self._clear_bindings()
        # Axis rows (read-only)
        for hw_label, field in _GAMEPAD_AXIS_ROWS:
            row = _BindingRow(
                action=f"axis_{field}",
                left_text=hw_label,
                right_text=field,
                editable=False,
            )
            self._bindings_layout.addWidget(row)

        # Button rows (action picker on double-click)
        self._gp_rows = []
        for hw_id, hw_label in _GAMEPAD_BUTTON_HW_ROWS:
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
        bindings = self._mgr.get_keyboard_bindings()
        if key not in bindings.values():
            return False
        self._mgr.forward_key_press(key)
        return False

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
        ok = self._mgr.set_keyboard_binding(row.action, key)
        if ok:
            row.set_right(QKeySequence(int(key)).toString() or "—")
        row.set_capturing(False)
        self._capturing_row = None

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
        self._mgr.set_gamepad_button_binding(hw_button, action_id)
        for row in self._gp_rows:
            if row.action == hw_button:
                row.set_right(self._action_label(action_id))
                break

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
