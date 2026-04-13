"""KeyboardInputSource — converts keyboard state to CommandBus values.

Unlike GamepadInputSource (which polls in its own thread), KeyboardInputSource
is driven by Qt key events forwarded from the MainWindow.  It converts
key-press/release state into velocity commands and publishes to CommandBus.

Usage::

    source = KeyboardInputSource(command_bus)
    # From MainWindow.keyPressEvent:
    source.key_pressed(Qt.Key_W)
    source.key_released(Qt.Key_W)
    # Call periodically to publish current state:
    source.update()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)

# Default key codes (Qt key values).
# Arrow keys are used for translation instead of WSAD because the MuJoCo
# viewer reserves W/S/A/D for its own camera/visualisation hotkeys, which
# would otherwise eat the keypresses before they reach our event filter.
_DEFAULT_KEY_MAP = {
    "forward":      0x01000013,  # Qt.Key_Up
    "backward":     0x01000015,  # Qt.Key_Down
    "strafe_left":  0x01000012,  # Qt.Key_Left
    "strafe_right": 0x01000014,  # Qt.Key_Right
    "yaw_left":     0x51,        # Q
    "yaw_right":    0x45,        # E
    "stop":         0x20,        # Space
    "boost":        0x01000020,  # Shift
}


class KeyboardInputSource:
    """Converts keyboard input to CommandBus velocity commands.

    Parameters
    ----------
    command_bus:
        The CommandBus to publish to.
    max_vx:
        Maximum forward/backward velocity.
    max_vy:
        Maximum lateral velocity.
    max_vyaw:
        Maximum yaw rate.
    boost_multiplier:
        Speed multiplier when boost key is held.
    key_map:
        Mapping of action names to Qt key codes.
    """

    def __init__(
        self,
        command_bus: Any,
        *,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_vyaw: float = 0.8,
        boost_multiplier: float = 2.0,
        key_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self._bus = command_bus
        self._max_vx = max_vx
        self._max_vy = max_vy
        self._max_vyaw = max_vyaw
        self._boost = boost_multiplier
        self._key_map = key_map or dict(_DEFAULT_KEY_MAP)
        self._pressed: Set[int] = set()

    def key_pressed(self, key: int) -> None:
        """Record a key press and update CommandBus."""
        self._pressed.add(key)
        self.update()

    def key_released(self, key: int) -> None:
        """Record a key release and update CommandBus."""
        self._pressed.discard(key)
        self.update()

    def clear(self) -> None:
        """Clear all pressed keys and zero commands."""
        self._pressed.clear()
        self._bus.publish_batch({"vx": 0.0, "vy": 0.0, "vyaw": 0.0})

    # ------------------------------------------------------------------
    # Runtime binding mutation (used by the Controller sidebar panel)
    # ------------------------------------------------------------------

    def set_binding(self, action: str, key: int) -> None:
        """Bind ``action`` (e.g. ``"forward"``) to a Qt key code.

        Replaces any previous mapping for that action and immediately
        re-publishes the current command state so the runtime picks the
        new key up on the next ``key_pressed`` call. Releases any keys
        in the pressed set so the new binding starts from a clean slate.
        """
        if not action:
            return
        self._key_map[str(action)] = int(key)
        # Drop any held keys — the user is mid-rebind, the previous
        # press state is no longer meaningful.
        self._pressed.clear()
        self.update()

    def get_binding(self, action: str) -> Optional[int]:
        """Return the Qt key code currently bound to ``action`` or None."""
        return self._key_map.get(str(action))

    def get_all_bindings(self) -> Dict[str, int]:
        """Return a snapshot of every action → key mapping."""
        return dict(self._key_map)

    def reset_bindings(self) -> None:
        """Restore the default keymap and clear any held-key state."""
        self._key_map = dict(_DEFAULT_KEY_MAP)
        self._pressed.clear()
        self._bus.publish_batch({"vx": 0.0, "vy": 0.0, "vyaw": 0.0})

    def update(self) -> None:
        """Compute velocity from current key state and publish."""
        km = self._key_map

        # Stop override
        if km.get("stop") in self._pressed:
            self._bus.publish_batch({"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
            return

        vx = 0.0
        vy = 0.0
        vyaw = 0.0

        if km.get("forward") in self._pressed:
            vx += self._max_vx
        if km.get("backward") in self._pressed:
            vx -= self._max_vx
        if km.get("strafe_left") in self._pressed:
            vy += self._max_vy
        if km.get("strafe_right") in self._pressed:
            vy -= self._max_vy
        if km.get("yaw_left") in self._pressed:
            vyaw += self._max_vyaw
        if km.get("yaw_right") in self._pressed:
            vyaw -= self._max_vyaw

        # Boost
        if km.get("boost") in self._pressed:
            vx *= self._boost
            vy *= self._boost
            vyaw *= self._boost

        self._bus.publish_batch({"vx": vx, "vy": vy, "vyaw": vyaw})

    @property
    def is_active(self) -> bool:
        """True if any movement key is currently held."""
        return bool(self._pressed)

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(cls, command_bus: Any, profile_path: Path) -> "KeyboardInputSource":
        """Create from a YAML input profile."""
        try:
            import yaml
            with profile_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            cfg = {}

        key_map = {}
        raw_km = cfg.get("key_mapping", {})
        for action_name, key_code in raw_km.items():
            key_map[action_name] = int(key_code)

        return cls(
            command_bus,
            max_vx=float(cfg.get("max_vx", 0.5)),
            max_vy=float(cfg.get("max_vy", 0.3)),
            max_vyaw=float(cfg.get("max_vyaw", 0.8)),
            boost_multiplier=float(cfg.get("boost_multiplier", 2.0)),
            key_map=key_map or None,
        )
