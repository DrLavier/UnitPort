# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""GlobalInputManager — process-singleton input dispatcher.

Owns one CommandBus + at most one active input source for the whole UnitPort
process. The Controller sidebar panel reads from it; future replay loops
(BehaviorNode, etc.) will plug into ``get_live_values()`` / ``field_order``.

Persistence:
    - mode + invert_vy + invert_vyaw + keyboard bindings + gamepad button
      bindings -> Config.set_value("Controller", ...) in user.ini overlay.
    - Inversion is on the COMMAND-frame output, not the raw joystick axis:
      ``invert_vy`` flips strafe direction (vy from left-stick X);
      ``invert_vyaw`` flips yaw direction (vyaw from right-stick X).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from unitport_sdk import Config, log_debug, log_warning

from .command_bus import CommandBus
from .gamepad_source import GamepadInputSource, is_available as gamepad_is_available
from .keyboard_source import DEFAULT_BINDINGS, KeyboardInputSource


# Default order the policy sees command fields in.
_DEFAULT_FIELD_ORDER: List[str] = ["vx", "vy", "vyaw"]

# user.ini section + key names (preserved from DEMO so migrating profiles work).
_INI_SECTION = "Controller"
_INI_KEY_MODE = "mode"
_INI_KEY_INVERT_VY = "invert_vy"
_INI_KEY_INVERT_VYAW = "invert_vyaw"
_INI_KEY_KB_BINDINGS = "keyboard_bindings"
_INI_KEY_GP_BUTTON_BINDINGS = "gamepad_button_bindings"

# Catalog of semantic actions any gamepad button can be bound to.
GAMEPAD_BUTTON_ACTIONS: List[tuple] = [
    ("",                    "— (unbound)"),
    ("estop",               "Emergency stop"),
    ("stop",                "Zero velocity (soft stop)"),
    ("boost",               "Boost speed while held"),
    ("toggle_invert_vy",    "Toggle Invert vy"),
    ("toggle_invert_vyaw",  "Toggle Invert vyaw"),
    ("mode_off",            "Set mode: Off"),
    ("mode_keyboard",       "Set mode: Keyboard"),
    ("mode_gamepad",        "Set mode: Gamepad"),
    ("start_robot",         "Start robot"),
    ("stop_robot",          "Stop robot"),
]
_VALID_BUTTON_ACTIONS = frozenset(a for a, _ in GAMEPAD_BUTTON_ACTIONS)

_DEFAULT_BOOT_MODE = "keyboard"
_VALID_MODES = frozenset({"off", "keyboard", "gamepad"})


class GlobalInputManager(QObject):
    """Single owner of the live-input CommandBus + active input source."""

    mode_changed = pyqtSignal(str)
    device_availability_changed = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._bus = CommandBus()
        self._active_source: Optional[Any] = None
        self._mode: str = "off"
        self._field_order: List[str] = list(_DEFAULT_FIELD_ORDER)
        # Persisted preferences.
        self._invert_vy: bool = False
        self._invert_vyaw: bool = False
        self._boot_preferred_mode: str = _DEFAULT_BOOT_MODE
        self._saved_kb_bindings: Dict[str, int] = {}
        self._gp_button_bindings: Dict[str, str] = {}
        self._load_config()
        self._auto_activate_boot_mode()

    # ----- persistence -----------------------------------------------------

    def _load_config(self) -> None:
        try:
            saved_mode = Config.get_value(
                _INI_SECTION, _INI_KEY_MODE, fallback=_DEFAULT_BOOT_MODE,
            )
            saved_invert_vy = Config.get_value(
                _INI_SECTION, _INI_KEY_INVERT_VY, fallback=False, value_type=bool,
            )
            saved_invert_vyaw = Config.get_value(
                _INI_SECTION, _INI_KEY_INVERT_VYAW, fallback=False, value_type=bool,
            )
            saved_kb_raw = Config.get_value(
                _INI_SECTION, _INI_KEY_KB_BINDINGS, fallback="",
            )
            saved_gp_btn_raw = Config.get_value(
                _INI_SECTION, _INI_KEY_GP_BUTTON_BINDINGS, fallback="",
            )
        except Exception as exc:
            log_warning(f"[input] failed to read user.ini: {exc}")
            return
        mode = (str(saved_mode) or "").strip().lower() or _DEFAULT_BOOT_MODE
        if mode not in _VALID_MODES:
            mode = _DEFAULT_BOOT_MODE
        self._boot_preferred_mode = mode
        self._invert_vy = bool(saved_invert_vy)
        self._invert_vyaw = bool(saved_invert_vyaw)
        self._saved_kb_bindings = self._parse_kb_bindings(saved_kb_raw)
        self._gp_button_bindings = self._parse_gp_button_bindings(saved_gp_btn_raw)

    @staticmethod
    def _parse_kb_bindings(raw: Any) -> Dict[str, int]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:
            log_warning(f"[input] keyboard_bindings is not valid JSON: {text!r}")
            return {}
        if not isinstance(data, dict):
            return {}
        out: Dict[str, int] = {}
        for action, key in data.items():
            try:
                out[str(action)] = int(key)
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _parse_gp_button_bindings(raw: Any) -> Dict[str, str]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:
            log_warning(f"[input] gamepad_button_bindings is not valid JSON: {text!r}")
            return {}
        if not isinstance(data, dict):
            return {}
        out: Dict[str, str] = {}
        for hw, action in data.items():
            hw_name = str(hw or "").strip().lower()
            action_id = str(action or "").strip().lower()
            if not hw_name or not action_id:
                continue
            if action_id not in _VALID_BUTTON_ACTIONS:
                continue
            out[hw_name] = action_id
        return out

    def _persist_mode(self, mode: str) -> None:
        try:
            Config.set_value(_INI_SECTION, _INI_KEY_MODE, mode)
        except Exception as exc:
            log_warning(f"[input] persist mode failed: {exc}")

    def _persist_keyboard_bindings(self, bindings: Dict[str, int]) -> None:
        try:
            payload = json.dumps(
                {str(k): int(v) for k, v in (bindings or {}).items()},
                ensure_ascii=False,
            )
            Config.set_value(_INI_SECTION, _INI_KEY_KB_BINDINGS, payload)
        except Exception as exc:
            log_warning(f"[input] persist keyboard bindings failed: {exc}")

    def _persist_gp_button_bindings(self, bindings: Dict[str, str]) -> None:
        try:
            payload = json.dumps(
                {
                    str(k).lower(): str(v)
                    for k, v in (bindings or {}).items()
                    if str(v or "").strip()
                },
                ensure_ascii=False,
            )
            Config.set_value(_INI_SECTION, _INI_KEY_GP_BUTTON_BINDINGS, payload)
        except Exception as exc:
            log_warning(f"[input] persist gamepad button bindings failed: {exc}")

    def _persist_invert_vy(self, inverted: bool) -> None:
        try:
            Config.set_value(_INI_SECTION, _INI_KEY_INVERT_VY, bool(inverted))
        except Exception as exc:
            log_warning(f"[input] persist invert_vy failed: {exc}")

    def _persist_invert_vyaw(self, inverted: bool) -> None:
        try:
            Config.set_value(_INI_SECTION, _INI_KEY_INVERT_VYAW, bool(inverted))
        except Exception as exc:
            log_warning(f"[input] persist invert_vyaw failed: {exc}")

    # ----- mode lifecycle --------------------------------------------------

    def _auto_activate_boot_mode(self) -> None:
        target = self._boot_preferred_mode
        if target == "off":
            self._mode = "off"
            return
        self._try_activate(target)
        # Sticky: keep the chosen mode even when device is missing.
        self._mode = target

    def _try_activate(self, mode: str) -> bool:
        with self._lock:
            self._teardown_active_source()
            self._bus.reset(self._field_order)
            self._mode = "off"
            if mode == "off":
                ok = True
            elif mode == "keyboard":
                ok = self._activate_keyboard()
            elif mode == "gamepad":
                ok = self._activate_gamepad()
            else:
                ok = False
        if ok:
            self._mode = mode
            self._apply_invert_to_active_source()
        return ok

    def _apply_invert_to_active_source(self) -> None:
        if self._active_source is None:
            return
        fn = getattr(self._active_source, "set_field_inverted", None)
        if not callable(fn):
            return
        for field, on in (
            ("vy", self._invert_vy),
            ("vyaw", self._invert_vyaw),
        ):
            if not on:
                continue
            try:
                fn(field, True)
            except Exception as exc:
                log_warning(f"[input] apply invert {field} raised: {exc}")

    def _teardown_active_source(self) -> None:
        if self._active_source is None:
            return
        try:
            stop = getattr(self._active_source, "stop", None)
            if callable(stop):
                stop()
        except Exception as exc:
            log_warning(f"[input] teardown raised: {exc}")
        self._active_source = None

    def _activate_keyboard(self) -> bool:
        seed = dict(self._saved_kb_bindings) if self._saved_kb_bindings else None
        self._active_source = KeyboardInputSource(self._bus, key_map=seed)
        return True

    def _activate_gamepad(self) -> bool:
        source = GamepadInputSource(self._bus, parent=self)
        try:
            started = bool(source.start())
        except Exception as exc:
            log_warning(f"[input] gamepad start raised: {exc}")
            started = False
        if not started:
            return False
        self._active_source = source
        return True

    # ----- public mode API -------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def command_bus(self) -> CommandBus:
        return self._bus

    def is_active(self) -> bool:
        return self._mode != "off" and self._active_source is not None

    def set_mode(self, mode: str) -> bool:
        target = (mode or "off").strip().lower()
        if target not in _VALID_MODES:
            log_warning(f"[input] unknown mode {target!r}")
            return False
        ok = self._try_activate(target)
        # Sticky: keep user's chosen mode even when source failed to bind.
        self._mode = target
        self._persist_mode(target)
        log_debug(f"[input] mode = {target} (active={ok})")
        self.mode_changed.emit(target)
        self.device_availability_changed.emit(target, ok)
        return ok

    def is_device_available(self, mode: str) -> bool:
        target = (mode or "").strip().lower()
        if target in {"off", "keyboard"}:
            return True
        if target == "gamepad":
            return gamepad_is_available()
        return False

    # ----- key event forwarding (called by ControllerPanel event filter) ---

    def forward_key_press(self, key: int) -> None:
        if self._mode != "keyboard" or self._active_source is None:
            return
        fn = getattr(self._active_source, "key_pressed", None)
        if callable(fn):
            try:
                fn(int(key))
            except Exception as exc:
                log_warning(f"[input] forward_key_press: {exc}")

    def forward_key_release(self, key: int) -> None:
        if self._mode != "keyboard" or self._active_source is None:
            return
        fn = getattr(self._active_source, "key_released", None)
        if callable(fn):
            try:
                fn(int(key))
            except Exception as exc:
                log_warning(f"[input] forward_key_release: {exc}")

    # ----- keyboard binding mutation --------------------------------------

    def get_keyboard_bindings(self) -> Dict[str, int]:
        if self._mode != "keyboard" or self._active_source is None:
            # Return saved snapshot so the panel can render the current
            # configured map even when the source isn't bound (rare).
            return dict(self._saved_kb_bindings) if self._saved_kb_bindings else dict(DEFAULT_BINDINGS)
        fn = getattr(self._active_source, "get_all_bindings", None)
        if not callable(fn):
            return {}
        try:
            return dict(fn())
        except Exception as exc:
            log_warning(f"[input] get_keyboard_bindings: {exc}")
            return {}

    def set_keyboard_binding(self, action: str, key: int) -> bool:
        if self._mode != "keyboard" or self._active_source is None:
            return False
        fn = getattr(self._active_source, "set_binding", None)
        if not callable(fn):
            return False
        try:
            fn(action, key)
        except Exception as exc:
            log_warning(f"[input] set_keyboard_binding: {exc}")
            return False
        snapshot = self.get_keyboard_bindings()
        if snapshot:
            self._saved_kb_bindings = dict(snapshot)
            self._persist_keyboard_bindings(snapshot)
        return True

    def reset_keyboard_bindings(self) -> bool:
        if self._mode != "keyboard" or self._active_source is None:
            self._saved_kb_bindings = {}
            self._persist_keyboard_bindings({})
            return True
        fn = getattr(self._active_source, "reset_bindings", None)
        if not callable(fn):
            return False
        try:
            fn()
        except Exception as exc:
            log_warning(f"[input] reset_keyboard_bindings: {exc}")
            return False
        self._saved_kb_bindings = {}
        self._persist_keyboard_bindings({})
        return True

    # ----- gamepad button bindings ----------------------------------------

    def get_gamepad_button_bindings(self) -> Dict[str, str]:
        return dict(self._gp_button_bindings)

    def get_gamepad_button_action(self, hw_name: str) -> str:
        return self._gp_button_bindings.get(str(hw_name or "").strip().lower(), "")

    def set_gamepad_button_binding(self, hw_name: str, action_id: str) -> bool:
        hw = str(hw_name or "").strip().lower()
        action = str(action_id or "").strip().lower()
        if not hw:
            return False
        if action and action not in _VALID_BUTTON_ACTIONS:
            log_warning(f"[input] rejecting unknown action {action!r}")
            return False
        if not action:
            self._gp_button_bindings.pop(hw, None)
        else:
            self._gp_button_bindings[hw] = action
        self._persist_gp_button_bindings(self._gp_button_bindings)
        return True

    def reset_gamepad_button_bindings(self) -> None:
        self._gp_button_bindings = {}
        self._persist_gp_button_bindings({})

    @staticmethod
    def gamepad_button_action_catalog() -> List[tuple]:
        return list(GAMEPAD_BUTTON_ACTIONS)

    # ----- command-frame invert (vy strafe / vyaw rotation) ----------------

    @property
    def invert_vy(self) -> bool:
        return self._invert_vy

    @property
    def invert_vyaw(self) -> bool:
        return self._invert_vyaw

    def set_invert_vy(self, inverted: bool) -> None:
        self._set_invert("vy", inverted)

    def set_invert_vyaw(self, inverted: bool) -> None:
        self._set_invert("vyaw", inverted)

    def _set_invert(self, field: str, inverted: bool) -> None:
        target = bool(inverted)
        if field == "vy":
            cur = self._invert_vy
            self._invert_vy = target
            self._persist_invert_vy(target)
        elif field == "vyaw":
            cur = self._invert_vyaw
            self._invert_vyaw = target
            self._persist_invert_vyaw(target)
        else:
            log_warning(f"[input] unknown invert field {field!r}")
            return
        if cur == target or self._active_source is None:
            return
        fn = getattr(self._active_source, "set_field_inverted", None)
        if callable(fn):
            try:
                fn(field, target)
            except Exception as exc:
                log_warning(f"[input] set_invert_{field}: {exc}")

    # ----- live values -----------------------------------------------------

    @property
    def field_order(self) -> List[str]:
        return list(self._field_order)

    def set_field_order(self, fields) -> None:
        with self._lock:
            self._field_order = [str(f) for f in fields if f]

    def get_live_values(self) -> Dict[str, float]:
        return self._bus.read_all(self._field_order, default=0.0)

    # ----- shutdown --------------------------------------------------------

    def shutdown(self) -> None:
        """Tear down the active input source without persisting state.

        Called from ``QApplication.aboutToQuit`` so that the gamepad
        poll thread joins before pygame's atexit handler deinits the
        video subsystem, and before LogSignal's C++ object is destroyed.
        """
        with self._lock:
            self._teardown_active_source()


# ---------------------------------------------------------------------------
# Process singleton
# ---------------------------------------------------------------------------


_singleton: Optional[GlobalInputManager] = None
_singleton_lock = threading.Lock()


def get_global_input_manager() -> GlobalInputManager:
    """Return the process-wide GlobalInputManager (lazy)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = GlobalInputManager()
    return _singleton


__all__ = [
    "GlobalInputManager",
    "get_global_input_manager",
    "GAMEPAD_BUTTON_ACTIONS",
]
