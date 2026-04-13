"""Sidebar panel for live keyboard / gamepad input.

The panel is the user-facing front end of
:class:`src.system.runtime.global_input_manager.GlobalInputManager`. It
exposes:

1. A mode selector -- Off / Keyboard / Gamepad -- that switches the
   active input source process-wide. The selected mode is persisted
   in user.ini under the [CONTROLLER] section, so the user's choice
   survives a restart.
2. An editable bindings table mapping each "action" (forward, strafe,
   yaw, ...) to a key on the active device. In keyboard mode, the
   user can double-click any row to clear it and capture the next
   key press as the new binding. The "field" column (vx +, vy -, ...)
   is hidden by default and surfaced as a tooltip on the action cell
   so the table stays compact.
3. A live readout of the actual command values currently on the bus,
   updated by a 10 Hz QTimer so the user can verify their inputs are
   landing.
4. Reset / Invert-Y action controls. Reset restores the default
   bindings; Invert Y flips the sign of the gamepad's vy axis (the
   default Go2 layout makes the robot strafe in the user-perceived
   "wrong" direction without it). Both states are persisted to
   user.ini via the manager.

When keyboard mode is active, the panel installs a Qt application
event filter that captures every key press / release in the running
process and forwards them through the panel (which routes them either
to the manager for live publishing or to a capturing _BindingRow).
This keeps the MainWindow free of input plumbing -- the panel is the
only place that knows how to bridge Qt to the keyboard input source.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QEvent, QObject, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.system.runtime.global_input_manager import (
    GlobalInputManager,
    get_global_input_manager,
)

log = logging.getLogger(__name__)


# Action rows shown in the keyboard bindings table. Order is presentation
# order. Each entry pairs a stable action name (matches the keys in
# ``KeyboardInputSource._DEFAULT_KEY_MAP``) with the policy field it
# affects and a short human label.
_KEYBOARD_ACTIONS = [
    ("forward",       "vx +",      "Forward"),
    ("backward",      "vx -",      "Backward"),
    ("strafe_left",   "vy +",      "Strafe left"),
    ("strafe_right",  "vy -",      "Strafe right"),
    ("yaw_left",      "vyaw +",    "Yaw left"),
    ("yaw_right",     "vyaw -",    "Yaw right"),
    ("stop",          "all 0",     "Emergency stop"),
    ("boost",         "x 2 boost", "Boost"),
]

_GAMEPAD_BINDINGS = [
    ("Left Stick Y",  "vx",   "Forward / Backward"),
    ("Left Stick X",  "vy",   "Strafe"),
    ("Right Stick X", "vyaw", "Yaw"),
]


# Fixed column widths used by every binding row so the table aligns
# properly across rows. Sized to fit "Space" / "Shift" / "Backspace"
# without truncation at the bumped 13px font.
_COL_INPUT_WIDTH = 96
_COL_ACTION_MIN = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qt_key_label(key: int) -> str:
    """Convert a Qt key code into a human-readable label.

    Falls back to the raw hex value when ``QKeySequence`` doesn't have
    a printable form (rare for the standard alphabetic / arrow / modifier
    keys we care about here).
    """
    try:
        text = QKeySequence(int(key)).toString()
        if text:
            return text
    except Exception:
        pass
    return f"0x{int(key):08x}"


# ---------------------------------------------------------------------------
# Bindings row widget -- one row per action, hover + double-click to rebind
# ---------------------------------------------------------------------------

class _BindingRow(QFrame):
    """One editable binding row.

    The row is a two-column strip: key label + action label. The
    "field" tag (``vx +``, ``vy -``, ``all 0``, ...) is NOT shown
    inline -- it's exposed as the row tooltip so hovering reveals
    which obs slot the binding drives without crowding the table.
    The row hover-highlights via QSS and a left double-click hands
    itself to the panel as the active capture target (the panel then
    redirects the next key press into ``apply_capture``).

    ``editable=False`` rows skip the cursor change and ignore double-
    clicks -- used for the gamepad table where rebinding isn't wired
    up yet (axes vs. buttons need their own capture pipeline).
    """

    def __init__(
        self,
        action: str,
        key_label: str,
        action_label: str,
        field_label: str,
        *,
        editable: bool,
        on_request_capture: Callable[["_BindingRow"], None],
    ):
        super().__init__()
        self._action = action
        self._field_label = field_label
        self._editable = editable
        self._on_request_capture = on_request_capture
        self._capturing = False

        self.setObjectName("controllerBindingRow")
        self.setFrameShape(QFrame.NoFrame)
        # Hover-only tooltip showing the obs field this binding drives.
        self.setToolTip(self._tooltip_text(action_label, field_label, editable))
        if editable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(12)

        self._key_lbl = QLabel(key_label)
        self._key_lbl.setObjectName("bindingKey")
        self._key_lbl.setFixedWidth(_COL_INPUT_WIDTH)
        self._key_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self._action_lbl = QLabel(action_label)
        self._action_lbl.setObjectName("bindingAction")
        self._action_lbl.setMinimumWidth(_COL_ACTION_MIN)
        self._action_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        layout.addWidget(self._key_lbl)
        layout.addWidget(self._action_lbl, 1)

        self._apply_normal_style()

    @staticmethod
    def _tooltip_text(action_label: str, field_label: str, editable: bool) -> str:
        body = f"{action_label}  ->  {field_label}"
        if editable:
            return f"{body}\nDouble-click to rebind"
        return body

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    _STYLE_NORMAL = """
        #controllerBindingRow {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        #controllerBindingRow:hover {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        QLabel#bindingKey {
            font-family: Consolas, monospace; font-size: 13px; color: #dddddd;
        }
        QLabel#bindingAction {
            font-size: 13px; color: #cccccc;
        }
    """

    _STYLE_CAPTURING = """
        #controllerBindingRow {
            background: rgba(80, 170, 255, 0.10);
            border: 1px solid rgba(80, 170, 255, 0.65);
            border-radius: 4px;
        }
        QLabel#bindingKey {
            font-family: Consolas, monospace; font-size: 13px; color: #5aaaff;
            font-weight: 600;
        }
        QLabel#bindingAction {
            font-size: 13px; color: #cccccc;
        }
    """

    def _apply_normal_style(self) -> None:
        self.setStyleSheet(self._STYLE_NORMAL)

    def _apply_capturing_style(self) -> None:
        self.setStyleSheet(self._STYLE_CAPTURING)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    @property
    def action(self) -> str:
        return self._action

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def set_key_label(self, label: str) -> None:
        self._key_lbl.setText(label)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 -- Qt API
        if not self._editable:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._on_request_capture(self)

    def enter_capture_mode(self) -> None:
        self._capturing = True
        self._key_lbl.setText("...")
        self._apply_capturing_style()

    def cancel_capture_mode(self, restore_label: str) -> None:
        self._capturing = False
        self._key_lbl.setText(restore_label)
        self._apply_normal_style()

    def apply_capture(self, key: int, label: str) -> None:
        self._capturing = False
        self._key_lbl.setText(label)
        self._apply_normal_style()


# ---------------------------------------------------------------------------
# Global Qt key event filter -- installed only in keyboard mode
# ---------------------------------------------------------------------------

class _GlobalKeyEventFilter(QObject):
    """QApplication event filter that routes key events through the panel.

    The panel's ``handle_key_press`` decides whether a key event should
    be (a) consumed by an active capture row or (b) forwarded to the
    keyboard source as a normal command. The filter just yields the
    decision back to Qt so capture events don't leak into other widgets.
    """

    def __init__(self, panel: "ControllerPanel"):
        super().__init__(panel)
        self._panel = panel

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 -- Qt API
        et = event.type()
        if et == QEvent.Type.KeyPress:
            ke: QKeyEvent = event  # type: ignore[assignment]
            if not ke.isAutoRepeat():
                consumed = self._panel.handle_key_press(int(ke.key()))
                if consumed:
                    return True
        elif et == QEvent.Type.KeyRelease:
            ke = event  # type: ignore[assignment]
            if not ke.isAutoRepeat():
                self._panel.handle_key_release(int(ke.key()))
        return False


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ControllerPanel(QWidget):
    """Sidebar panel for live input control of replay sessions."""

    mode_changed = Signal(str)  # "off" | "keyboard" | "gamepad"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controllerPanel")
        self._manager = get_global_input_manager()
        self._key_filter: Optional[_GlobalKeyEventFilter] = None

        # Mutable state for the editable bindings UI
        self._binding_rows: List[_BindingRow] = []  # current visible rows
        self._capturing_row: Optional[_BindingRow] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("Controller")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Live input control. Feed the active replay session real "
            "commands instead of bundle defaults. Double-click any row "
            "below to rebind it."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a8a8a; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addWidget(self._build_mode_row())
        layout.addWidget(self._build_bindings_section())
        layout.addWidget(self._build_action_buttons())
        layout.addWidget(self._build_live_section())
        layout.addStretch(1)

        # 10 Hz live-values refresh timer (cheap dict read)
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._refresh_live_values)
        self._live_timer.start()

        # Sync UI with whatever mode the manager is currently in -- the
        # manager itself was constructed earlier (process singleton) and
        # may have already auto-activated keyboard mode from user.ini.
        self._sync_to_manager_mode(self._manager.mode)

    # ------------------------------------------------------------------
    # Sub-widgets
    # ------------------------------------------------------------------

    def _build_mode_row(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("controllerModeRow")
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        label = QLabel("Mode:")
        label.setStyleSheet("font-size: 13px; color: #cccccc;")
        h.addWidget(label)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._radio_off = QRadioButton("Off")
        self._radio_kb = QRadioButton("Keyboard")
        self._radio_pad = QRadioButton("Gamepad")
        for rb in (self._radio_off, self._radio_kb, self._radio_pad):
            rb.setStyleSheet("font-size: 13px;")
            self._mode_group.addButton(rb)
            h.addWidget(rb)

        # ``clicked`` only fires for genuine user input, NOT for
        # programmatic ``setChecked`` calls -- this means
        # ``_sync_to_manager_mode`` can flip the radios freely without
        # re-entering the click handler.
        self._radio_off.clicked.connect(lambda: self._on_mode_clicked("off"))
        self._radio_kb.clicked.connect(lambda: self._on_mode_clicked("keyboard"))
        self._radio_pad.clicked.connect(lambda: self._on_mode_clicked("gamepad"))

        h.addStretch(1)
        return wrap

    def _build_bindings_section(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("controllerBindings")
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(2)

        header = QLabel("Bindings")
        header.setStyleSheet("font-size: 13px; color: #aaaaaa;")
        v.addWidget(header)

        # Two-column header strip with the same fixed widths the rows
        # use, so visually it's a real table instead of a misaligned list.
        col_header = QFrame()
        ch = QHBoxLayout(col_header)
        ch.setContentsMargins(8, 2, 8, 2)
        ch.setSpacing(12)
        h_input = QLabel("Input")
        h_input.setFixedWidth(_COL_INPUT_WIDTH)
        h_input.setStyleSheet("font-size: 12px; font-weight: 600; color: #777;")
        h_action = QLabel("Action")
        h_action.setMinimumWidth(_COL_ACTION_MIN)
        h_action.setStyleSheet("font-size: 12px; font-weight: 600; color: #777;")
        ch.addWidget(h_input)
        ch.addWidget(h_action, 1)
        v.addWidget(col_header)

        # Container for the dynamic rows
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        v.addWidget(self._rows_container)

        return wrap

    def _build_action_buttons(self) -> QWidget:
        wrap = QFrame()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 6, 0, 0)
        h.setSpacing(8)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.setToolTip("Restore the default bindings for the current device")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._reset_btn.setStyleSheet(
            "QPushButton {"
            "  background: #2a2a2a;"
            "  color: #d6d6d6;"
            "  border: 1px solid #3a3a3a;"
            "  border-radius: 4px;"
            "  padding: 4px 12px;"
            "  font-size: 13px;"
            "}"
            "QPushButton:hover {"
            "  background: #353535;"
            "  border-color: #4a4a4a;"
            "}"
            "QPushButton:pressed { background: #222; }"
            "QPushButton:disabled {"
            "  background: #232323;"
            "  color: #666;"
            "  border-color: #2c2c2c;"
            "}"
        )
        h.addWidget(self._reset_btn)

        # Invert Y -- shown only in gamepad mode (keyboard's vy
        # convention is already correct). Toggling it flips the sign
        # of every gamepad axis that targets vy and persists the
        # preference to user.ini via the manager.
        self._invert_check = QCheckBox("Invert Y")
        self._invert_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self._invert_check.setStyleSheet("font-size: 13px; color: #d6d6d6;")
        self._invert_check.setToolTip(
            "Negate the gamepad's vy axis so pushing the left stick "
            "right makes the robot strafe right (default Go2 layout "
            "is reversed). Persisted in user.ini."
        )
        self._invert_check.setChecked(self._manager.invert_vy)
        self._invert_check.toggled.connect(self._on_invert_toggled)
        h.addWidget(self._invert_check)

        h.addStretch(1)
        return wrap

    def _build_live_section(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("controllerLive")
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(4)

        header = QLabel("Live Values")
        header.setStyleSheet("font-size: 13px; color: #aaaaaa;")
        v.addWidget(header)

        self._live_grid = QGridLayout()
        self._live_grid.setContentsMargins(4, 0, 4, 0)
        self._live_grid.setHorizontalSpacing(12)
        self._live_grid.setVerticalSpacing(2)
        v.addLayout(self._live_grid)

        self._live_labels: Dict[str, QLabel] = {}
        for col, name in enumerate(self._manager.field_order):
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet("font-size: 12px; color: #888;")
            lbl_val = QLabel("0.000")
            lbl_val.setStyleSheet(
                "font-family: Consolas, monospace; font-size: 14px; color: #e0e0e0;"
            )
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._live_grid.addWidget(lbl_name, 0, col)
            self._live_grid.addWidget(lbl_val, 1, col)
            self._live_labels[name] = lbl_val

        return wrap

    # ------------------------------------------------------------------
    # Bindings table population
    # ------------------------------------------------------------------

    def _clear_rows(self) -> None:
        # Cancel any in-flight capture before swapping rows.
        if self._capturing_row is not None:
            old_label = _qt_key_label(self._current_keymap_value(self._capturing_row.action) or 0)
            self._capturing_row.cancel_capture_mode(old_label)
            self._capturing_row = None
        while self._rows_layout.count():
            it = self._rows_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._binding_rows = []

    def _populate_keyboard_rows(self) -> None:
        self._clear_rows()
        bindings = self._manager.get_keyboard_bindings()
        for action, field_label, action_label in _KEYBOARD_ACTIONS:
            key = bindings.get(action, 0)
            row = _BindingRow(
                action=action,
                key_label=_qt_key_label(key) if key else "-",
                action_label=action_label,
                field_label=field_label,
                editable=True,
                on_request_capture=self._begin_capture,
            )
            self._rows_layout.addWidget(row)
            self._binding_rows.append(row)

    def _populate_gamepad_rows(self) -> None:
        self._clear_rows()
        for axis_label, field_label, action_label in _GAMEPAD_BINDINGS:
            row = _BindingRow(
                action=field_label,
                key_label=axis_label,
                action_label=action_label,
                field_label=field_label,
                editable=False,
                on_request_capture=self._begin_capture,
            )
            self._rows_layout.addWidget(row)
            self._binding_rows.append(row)

    def _populate_off_rows(self) -> None:
        self._clear_rows()
        placeholder = QLabel(
            "Pick Keyboard or Gamepad above to view and edit bindings."
        )
        placeholder.setStyleSheet(
            "color: #777777; font-size: 13px; padding: 8px 4px;"
        )
        placeholder.setWordWrap(True)
        self._rows_layout.addWidget(placeholder)

    def _current_keymap_value(self, action: str) -> Optional[int]:
        bindings = self._manager.get_keyboard_bindings()
        return bindings.get(action)

    # ------------------------------------------------------------------
    # Capture mode
    # ------------------------------------------------------------------

    def _begin_capture(self, row: _BindingRow) -> None:
        # Only meaningful in keyboard mode for now
        if self._manager.mode != "keyboard":
            return
        # Cancel any previous capture row
        if self._capturing_row is not None and self._capturing_row is not row:
            old_label = _qt_key_label(self._current_keymap_value(self._capturing_row.action) or 0)
            self._capturing_row.cancel_capture_mode(old_label)
        self._capturing_row = row
        row.enter_capture_mode()

    def _consume_capture(self, key: int) -> None:
        row = self._capturing_row
        if row is None:
            return
        # Apply the new binding via the manager (which mutates the
        # active KeyboardInputSource's keymap in place).
        self._manager.set_keyboard_binding(row.action, key)
        row.apply_capture(key, _qt_key_label(key))
        self._capturing_row = None

    # ------------------------------------------------------------------
    # Mode wiring
    # ------------------------------------------------------------------

    def _on_mode_clicked(self, mode: str) -> None:
        ok = self._manager.set_mode(mode)
        if not ok and mode == "gamepad":
            log.info("Gamepad activation failed; reverted to Off")
        self._sync_to_manager_mode(self._manager.mode)
        self.mode_changed.emit(self._manager.mode)

    def _sync_to_manager_mode(self, mode: str) -> None:
        # ``setChecked`` is safe to call freely here -- it does NOT
        # fire the ``clicked`` signal we connected above.
        for rb, m in (
            (self._radio_off, "off"),
            (self._radio_kb, "keyboard"),
            (self._radio_pad, "gamepad"),
        ):
            rb.setChecked(m == mode)

        # Swap the rows table
        if mode == "keyboard":
            self._populate_keyboard_rows()
        elif mode == "gamepad":
            self._populate_gamepad_rows()
        else:
            self._populate_off_rows()

        # Action buttons / invert checkbox visibility
        self._reset_btn.setEnabled(mode == "keyboard")
        self._invert_check.setVisible(mode == "gamepad")
        # Re-sync the checkbox state from the persisted preference --
        # the manager keeps it across mode switches even when the
        # widget is hidden.
        self._invert_check.blockSignals(True)
        self._invert_check.setChecked(self._manager.invert_vy)
        self._invert_check.blockSignals(False)

        # Install / remove the global key event filter
        if mode == "keyboard":
            self._install_key_filter()
        else:
            self._remove_key_filter()

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
            try:
                app.removeEventFilter(self._key_filter)
            except Exception:
                pass
        self._key_filter = None

    # ------------------------------------------------------------------
    # Key event entrypoints (called by _GlobalKeyEventFilter)
    # ------------------------------------------------------------------

    def handle_key_press(self, key: int) -> bool:
        """Process a Qt key press. Returns True to swallow the event."""
        if self._capturing_row is not None:
            self._consume_capture(key)
            return True
        self._manager.forward_key_press(key)
        return False

    def handle_key_release(self, key: int) -> None:
        if self._capturing_row is not None:
            return
        self._manager.forward_key_release(key)

    # ------------------------------------------------------------------
    # Reset / Invert
    # ------------------------------------------------------------------

    def _on_reset_clicked(self) -> None:
        if self._manager.mode != "keyboard":
            return
        self._manager.reset_keyboard_bindings()
        self._populate_keyboard_rows()
        log.info("Controller bindings reset to defaults")

    def _on_invert_toggled(self, checked: bool) -> None:
        self._manager.set_invert_vy(bool(checked))

    # ------------------------------------------------------------------
    # Live values refresh
    # ------------------------------------------------------------------

    def _refresh_live_values(self) -> None:
        values = self._manager.get_live_values()
        for name, lbl in self._live_labels.items():
            v = float(values.get(name, 0.0))
            lbl.setText(f"{v:+.3f}")
            mag = min(1.0, abs(v) / 1.0)
            if mag < 0.05:
                lbl.setStyleSheet(
                    "font-family: Consolas, monospace; font-size: 14px; color: #707070;"
                )
            else:
                green = int(180 + 60 * mag)
                lbl.setStyleSheet(
                    f"font-family: Consolas, monospace; font-size: 14px; color: rgb(120, {green}, 140);"
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 -- Qt API
        self._live_timer.stop()
        self._remove_key_filter()
        super().closeEvent(event)
