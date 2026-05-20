"""GamepadInputSource — pygame.joystick poll thread feeding a CommandBus.

Polls the first available joystick at 60 Hz on a QThread:
    - left stick X/Y -> vy / vx (negate Y so up = +1)
    - right stick X  -> vyaw
    - right stick Y  -> currently unmapped (read-only row in the UI)
    - command-frame invert: ``set_field_inverted("vy", on)`` flips strafe
      sign; ``set_field_inverted("vyaw", on)`` flips yaw sign.
    - button transitions -> publish ``__btn_<name>`` as 1.0 (held) or 0.0 (released)

Hardware naming (pygame button index -> hw_name) follows the
``GAMEPAD_BUTTON_ACTIONS`` catalog in :mod:`manager`. Different pads expose
different layouts; the mapping below is the common Xbox-style layout. Atypical
controllers can still be operated, but their button names may not match the
catalog labels.

If pygame is unavailable (import error or missing wheel), :func:`is_available`
returns False and ``start()`` returns False without raising — the panel renders
the gamepad mode tile in red and the user can fall back to keyboard.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from unitport_sdk import log_debug, log_warning

from .command_bus import CommandBus


# WHY KEPT: pygame is an optional third-party dep (CLAUDE.md §1.8 case (a)) —
# absent on minimal installs, in which case the gamepad mode tile renders red
# and keyboard fallback is used. The inner warnings filter silences pygame
# 2.6.x's internal `from pkg_resources import ...` UserWarning at import time
# (pygame upstream still uses the deprecated API; we cannot fix it here).
try:
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        import pygame  # type: ignore
    _PYGAME_OK = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    _PYGAME_OK = False


# Xbox-style button index -> semantic hw name (matches DEMO conventions).
_BUTTON_INDEX_TO_NAME = {
    0:  "a",
    1:  "b",
    2:  "x",
    3:  "y",
    4:  "lb",
    5:  "rb",
    6:  "back",
    7:  "start",
    8:  "ls",          # left stick click
    9:  "rs",          # right stick click
    10: "guide",
}

# Stick dead zone to suppress noise around centre.
_STICK_DEADZONE = 0.08
# Trigger threshold above which lt/rt are reported as "pressed".
_TRIGGER_THRESHOLD = 0.5
# Polling cadence (Hz).
_POLL_HZ = 60


def is_available() -> bool:
    """Return True iff pygame is importable AND at least one joystick is connected."""
    if not _PYGAME_OK:
        return False
    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        return int(pygame.joystick.get_count()) > 0
    except Exception:
        return False


class _PollThread(QThread):
    """Pygame polling worker. Runs at ~60 Hz while ``_running`` is True."""

    button_event = pyqtSignal(str, bool)   # hw_name, pressed

    def __init__(self, bus: CommandBus, joystick_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._joystick_index = joystick_index
        self._running = False
        self._invert_vy = False
        self._invert_vyaw = False
        self._lock = threading.Lock()
        self._last_buttons: dict[str, float] = {}
        self._last_dpad: Tuple[int, int] = (0, 0)

    def set_invert_vy(self, inverted: bool) -> None:
        with self._lock:
            self._invert_vy = bool(inverted)

    def set_invert_vyaw(self, inverted: bool) -> None:
        with self._lock:
            self._invert_vyaw = bool(inverted)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:  # type: ignore[override]
        if not _PYGAME_OK:
            return
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            if pygame.joystick.get_count() <= self._joystick_index:
                log_warning(f"[gamepad] joystick {self._joystick_index} not present")
                return
            joy = pygame.joystick.Joystick(self._joystick_index)
            joy.init()
            log_debug(f"[gamepad] opened {joy.get_name()!r} (axes={joy.get_numaxes()}, buttons={joy.get_numbuttons()})")
        except Exception as exc:
            log_warning(f"[gamepad] init failed: {exc}")
            return

        period = 1.0 / _POLL_HZ
        self._running = True
        try:
            while self._running:
                # If pygame was torn down by its atexit handler during app
                # shutdown, the video/joystick subsystem is gone — bail
                # silently instead of spinning on errors.
                if not pygame.get_init():
                    break
                try:
                    pygame.event.pump()
                    self._tick(joy)
                except pygame.error:
                    # Typically "video system not initialized" during shutdown.
                    break
                except Exception as exc:
                    # LogSignal may already be deleted at shutdown — guard the
                    # log call so a dead Qt object doesn't take the thread down.
                    try:
                        log_warning(f"[gamepad] poll error: {exc}")
                    except RuntimeError:
                        break
                time.sleep(period)
        finally:
            try:
                joy.quit()
            except Exception:
                pass
            # Clear axes so a stale "1.0 vx" doesn't linger after the source quits.
            try:
                for f in ("vx", "vy", "vyaw"):
                    self._bus.set(f, 0.0)
            except Exception:
                pass

    def _tick(self, joy) -> None:
        # ----- sticks -> axes ----------------------------------------------
        n_axes = joy.get_numaxes()
        ax_lx = joy.get_axis(0) if n_axes > 0 else 0.0
        ax_ly = joy.get_axis(1) if n_axes > 1 else 0.0
        ax_rx = joy.get_axis(2) if n_axes > 2 else 0.0

        with self._lock:
            inv_vy = self._invert_vy
            inv_vyaw = self._invert_vyaw

        vx = -_apply_deadzone(ax_ly)            # stick up -> +vx
        vy = _apply_deadzone(ax_lx)
        if inv_vy:
            vy = -vy
        vyaw = -_apply_deadzone(ax_rx)          # stick right -> negative yaw
        if inv_vyaw:
            vyaw = -vyaw

        self._bus.set("vx", vx)
        self._bus.set("vy", vy)
        self._bus.set("vyaw", vyaw)

        # ----- buttons -> __btn_<name> + edge events ----------------------
        for idx in range(joy.get_numbuttons()):
            name = _BUTTON_INDEX_TO_NAME.get(idx)
            if not name:
                continue
            pressed = bool(joy.get_button(idx))
            self._publish_button(name, pressed)

        # Triggers (axes 4, 5 on Xbox layout) reported as pseudo-buttons lt/rt.
        if n_axes > 4:
            lt = (joy.get_axis(4) + 1.0) * 0.5
            self._publish_button("lt", lt > _TRIGGER_THRESHOLD)
        if n_axes > 5:
            rt = (joy.get_axis(5) + 1.0) * 0.5
            self._publish_button("rt", rt > _TRIGGER_THRESHOLD)

        # ----- D-pad (hat) -> dpad_up/down/left/right ---------------------
        if joy.get_numhats() > 0:
            hx, hy = joy.get_hat(0)
            self._publish_button("dpad_up", hy > 0)
            self._publish_button("dpad_down", hy < 0)
            self._publish_button("dpad_left", hx < 0)
            self._publish_button("dpad_right", hx > 0)

    def _publish_button(self, name: str, pressed: bool) -> None:
        key = f"__btn_{name}"
        new_val = 1.0 if pressed else 0.0
        prev = self._last_buttons.get(name, 0.0)
        if new_val == prev:
            return
        self._last_buttons[name] = new_val
        self._bus.set(key, new_val)
        self.button_event.emit(name, pressed)


def _apply_deadzone(value: float) -> float:
    if abs(value) < _STICK_DEADZONE:
        return 0.0
    return float(value)


class GamepadInputSource(QObject):
    """Holds the pygame poll thread; presents the same surface as KeyboardInputSource."""

    button_event = pyqtSignal(str, bool)

    def __init__(self, bus: CommandBus, *, parent=None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._thread: Optional[_PollThread] = None

    def start(self) -> bool:
        if not is_available():
            return False
        self._thread = _PollThread(self._bus, joystick_index=0, parent=self)
        self._thread.button_event.connect(self.button_event)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread.deleteLater()
            self._thread = None

    def set_field_inverted(self, field: str, inverted: bool) -> None:
        if self._thread is None:
            return
        if field == "vy":
            self._thread.set_invert_vy(inverted)
        elif field == "vyaw":
            self._thread.set_invert_vyaw(inverted)


__all__ = ["GamepadInputSource", "is_available"]
