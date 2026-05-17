"""KeyboardInputSource — Qt key codes -> command axes published on a CommandBus.

The Controller sidebar panel installs a global ``QApplication.installEventFilter``
that forwards every key press / release to ``GlobalInputManager.forward_key_press``
/ ``forward_key_release``, which in turn calls into this source.

Default mapping (matches the DEMO controller panel rows 1:1):

    forward       -> W
    backward      -> S
    strafe_left   -> A
    strafe_right  -> D
    yaw_left      -> Q
    yaw_right     -> E
    estop         -> Escape
    boost         -> Shift
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from PyQt6.QtCore import Qt

from .command_bus import CommandBus


# Action ids must stay in lock-step with the row labels in controller_panel.py.
_DEFAULT_BINDINGS: Dict[str, int] = {
    "forward":      int(Qt.Key.Key_W),
    "backward":     int(Qt.Key.Key_S),
    "strafe_left":  int(Qt.Key.Key_A),
    "strafe_right": int(Qt.Key.Key_D),
    "yaw_left":     int(Qt.Key.Key_Q),
    "yaw_right":    int(Qt.Key.Key_E),
    "estop":        int(Qt.Key.Key_Escape),
    "boost":        int(Qt.Key.Key_Shift),
}


# Each action drives one or two CommandBus fields; the (field, value) pair
# describes what to publish while the bound key is held.
_ACTION_EFFECT = {
    "forward":      ("vx",  1.0),
    "backward":     ("vx", -1.0),
    "strafe_left":  ("vy", -1.0),
    "strafe_right": ("vy",  1.0),
    "yaw_left":     ("vyaw", 1.0),
    "yaw_right":    ("vyaw", -1.0),
    # estop and boost are handled as semantic flags on the bus.
    "estop":        ("__estop", 1.0),
    "boost":        ("__boost", 1.0),
}


class KeyboardInputSource:
    """Translate Qt key events into CommandBus axis values."""

    def __init__(self, bus: CommandBus, key_map: Optional[Dict[str, int]] = None) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._bindings: Dict[str, int] = dict(_DEFAULT_BINDINGS)
        if key_map:
            for action, key in key_map.items():
                if action in _ACTION_EFFECT:
                    try:
                        self._bindings[action] = int(key)
                    except (TypeError, ValueError):
                        pass
        # Active set: actions currently held down. Recomputed on every press/release.
        self._held: set[str] = set()
        self._reset_axes()

    # ----- bus output -------------------------------------------------------

    def _reset_axes(self) -> None:
        for field in ("vx", "vy", "vyaw", "__estop", "__boost"):
            self._bus.set(field, 0.0)

    def _publish(self) -> None:
        """Resolve held actions into axis values and publish on the bus."""
        vx = vy = vyaw = 0.0
        estop = boost = 0.0
        for action in self._held:
            field, value = _ACTION_EFFECT.get(action, ("", 0.0))
            if field == "vx":
                vx += value
            elif field == "vy":
                vy += value
            elif field == "vyaw":
                vyaw += value
            elif field == "__estop":
                estop = max(estop, value)
            elif field == "__boost":
                boost = max(boost, value)
        # Clamp to [-1, 1] in case opposing keys were both pressed (vy: A + D).
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        vyaw = max(-1.0, min(1.0, vyaw))
        self._bus.set("vx", vx)
        self._bus.set("vy", vy)
        self._bus.set("vyaw", vyaw)
        self._bus.set("__estop", estop)
        self._bus.set("__boost", boost)

    # ----- event entry points ----------------------------------------------

    def key_pressed(self, key: int) -> None:
        action = self._action_for_key(int(key))
        if not action:
            return
        with self._lock:
            self._held.add(action)
            self._publish()

    def key_released(self, key: int) -> None:
        action = self._action_for_key(int(key))
        if not action:
            return
        with self._lock:
            self._held.discard(action)
            self._publish()

    def _action_for_key(self, key: int) -> str:
        for action, bound_key in self._bindings.items():
            if bound_key == key:
                return action
        return ""

    # ----- binding API used by the manager / panel --------------------------

    def get_all_bindings(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._bindings)

    def set_binding(self, action: str, key: int) -> None:
        if action not in _ACTION_EFFECT:
            return
        with self._lock:
            self._bindings[action] = int(key)
            # Drop stale held state — the user may have remapped a held key.
            self._held.discard(action)
            self._publish()

    def reset_bindings(self) -> None:
        with self._lock:
            self._bindings = dict(_DEFAULT_BINDINGS)
            self._held.clear()
            self._reset_axes()

    # ----- compatibility hooks (called by the manager) ---------------------

    def set_field_inverted(self, field: str, inverted: bool) -> None:
        # Keyboard bindings are symmetric (W/S, A/D both bound), so command
        # invert (vy / vyaw) is a no-op here. Accepted for protocol parity.
        pass

    def stop(self) -> None:
        with self._lock:
            self._held.clear()
            self._reset_axes()


__all__ = ["KeyboardInputSource", "DEFAULT_BINDINGS"]

DEFAULT_BINDINGS = dict(_DEFAULT_BINDINGS)
