"""UserInputManager — real-time user command capture for manual robot control.

Captures keyboard (WASD) and gamepad inputs and emits actions in the SAME
format as policy inference output.  This makes UserInputManager an alternative
"behaviour source" that can be assigned to any BehaviorNode in Canvas.

The manager uses configurable input profiles (YAML) per robot family to map
physical inputs to velocity commands or joint deltas.

Public API
----------
UserInputManager.key_pressed(key)
UserInputManager.key_released(key)
UserInputManager.get_action() -> np.ndarray
UserInputManager.get_velocity_command() -> Tuple[float, float, float]
UserInputManager.set_profile(profile_name)
UserInputManager.is_active() -> bool
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Key constants (Qt key codes)
_KEY_W = 0x57
_KEY_A = 0x41
_KEY_S = 0x53
_KEY_D = 0x44
_KEY_Q = 0x51  # yaw left
_KEY_E = 0x45  # yaw right
_KEY_SPACE = 0x20  # stop / emergency
_KEY_SHIFT = 0x01000020  # speed boost


# ---------------------------------------------------------------------------
# Input profiles
# ---------------------------------------------------------------------------

_DEFAULT_KEYBOARD_PROFILE: Dict[str, Any] = {
    "name": "keyboard_quadruped",
    "robot_family": "quadruped",
    "input_type": "keyboard",
    "max_vx": 0.5,
    "max_vy": 0.3,
    "max_vyaw": 0.8,
    "boost_multiplier": 2.0,
    "key_mapping": {
        "forward": _KEY_W,
        "backward": _KEY_S,
        "strafe_left": _KEY_A,
        "strafe_right": _KEY_D,
        "yaw_left": _KEY_Q,
        "yaw_right": _KEY_E,
        "stop": _KEY_SPACE,
        "boost": _KEY_SHIFT,
    },
}

_DEFAULT_GAMEPAD_PROFILE: Dict[str, Any] = {
    "name": "gamepad_quadruped",
    "robot_family": "quadruped",
    "input_type": "gamepad",
    "max_vx": 1.0,
    "max_vy": 0.5,
    "max_vyaw": 1.5,
    "deadzone": 0.1,
    "axis_mapping": {
        "left_stick_y": "vx",
        "left_stick_x": "vy",
        "right_stick_x": "vyaw",
    },
}

_PROFILES: Dict[str, Dict[str, Any]] = {
    "keyboard_quadruped": _DEFAULT_KEYBOARD_PROFILE,
    "gamepad_quadruped": _DEFAULT_GAMEPAD_PROFILE,
}


def load_profile(profile_path: Path) -> Dict[str, Any]:
    """Load a YAML input profile from disk."""
    try:
        import yaml
        with profile_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        log.warning("Failed to load input profile %s: %s", profile_path, exc)
        return {}


def register_profile(name: str, profile: Dict[str, Any]) -> None:
    """Register an input profile by name."""
    _PROFILES[name] = profile


# ---------------------------------------------------------------------------
# UserInputManager
# ---------------------------------------------------------------------------

class UserInputManager:
    """Captures real-time user input and converts to robot action commands.

    Usage::

        manager = UserInputManager()
        manager.set_profile("keyboard_quadruped")

        # Called from MainWindow.keyPressEvent / keyReleaseEvent
        manager.key_pressed(Qt.Key_W)
        manager.key_released(Qt.Key_W)

        # Called each control tick
        vx, vy, vyaw = manager.get_velocity_command()
        action = manager.get_action()
    """

    def __init__(self, profile_name: str = "keyboard_quadruped"):
        self._pressed_keys: Set[int] = set()
        self._gamepad_axes: Dict[str, float] = {}
        self._profile: Dict[str, Any] = _PROFILES.get(
            profile_name, _DEFAULT_KEYBOARD_PROFILE
        )
        self._active = False
        self._action_dim = 12  # default for quadruped

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def set_profile(self, profile_name: str) -> None:
        """Switch to a named input profile."""
        if profile_name in _PROFILES:
            self._profile = _PROFILES[profile_name]
            log.info("Input profile switched to: %s", profile_name)
        else:
            log.warning("Unknown input profile: %s", profile_name)

    @property
    def profile(self) -> Dict[str, Any]:
        return dict(self._profile)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value: int) -> None:
        self._action_dim = max(1, int(value))

    # ------------------------------------------------------------------
    # Key input
    # ------------------------------------------------------------------

    def key_pressed(self, key: int) -> None:
        """Record a key press event."""
        self._pressed_keys.add(key)
        self._active = True

    def key_released(self, key: int) -> None:
        """Record a key release event."""
        self._pressed_keys.discard(key)
        if not self._pressed_keys:
            self._active = False

    def clear_keys(self) -> None:
        """Clear all pressed keys (e.g. on focus loss)."""
        self._pressed_keys.clear()
        self._active = False

    # ------------------------------------------------------------------
    # Gamepad input
    # ------------------------------------------------------------------

    def set_gamepad_axis(self, axis_name: str, value: float) -> None:
        """Update a gamepad axis value."""
        deadzone = self._profile.get("deadzone", 0.1)
        self._gamepad_axes[axis_name] = 0.0 if abs(value) < deadzone else value
        self._active = any(v != 0.0 for v in self._gamepad_axes.values())

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True if any input is currently held."""
        return self._active

    def get_velocity_command(self) -> Tuple[float, float, float]:
        """Compute velocity command (vx, vy, vyaw) from current input state.

        Returns (0, 0, 0) when no input is active.
        """
        if self._profile.get("input_type") == "gamepad":
            return self._velocity_from_gamepad()
        return self._velocity_from_keyboard()

    def get_action(self) -> np.ndarray:
        """Produce an action vector in the same format as policy output.

        For velocity-based control, this encodes the velocity command
        into the first 3 elements of an action_dim-sized zero vector.
        """
        vx, vy, vyaw = self.get_velocity_command()
        action = np.zeros(self._action_dim, dtype=np.float32)
        # Encode velocity command in first 3 dims (convention matches training env)
        action[0] = vx
        if self._action_dim > 1:
            action[1] = vy
        if self._action_dim > 2:
            action[2] = vyaw
        return action

    # ------------------------------------------------------------------
    # Internal: keyboard → velocity
    # ------------------------------------------------------------------

    def _velocity_from_keyboard(self) -> Tuple[float, float, float]:
        km = self._profile.get("key_mapping", {})
        max_vx = self._profile.get("max_vx", 0.5)
        max_vy = self._profile.get("max_vy", 0.3)
        max_vyaw = self._profile.get("max_vyaw", 0.8)
        boost = self._profile.get("boost_multiplier", 2.0)

        vx = 0.0
        vy = 0.0
        vyaw = 0.0

        if km.get("forward") in self._pressed_keys:
            vx += max_vx
        if km.get("backward") in self._pressed_keys:
            vx -= max_vx
        if km.get("strafe_left") in self._pressed_keys:
            vy += max_vy
        if km.get("strafe_right") in self._pressed_keys:
            vy -= max_vy
        if km.get("yaw_left") in self._pressed_keys:
            vyaw += max_vyaw
        if km.get("yaw_right") in self._pressed_keys:
            vyaw -= max_vyaw

        # Speed boost
        if km.get("boost") in self._pressed_keys:
            vx *= boost
            vy *= boost
            vyaw *= boost

        # Stop override
        if km.get("stop") in self._pressed_keys:
            return (0.0, 0.0, 0.0)

        return (vx, vy, vyaw)

    # ------------------------------------------------------------------
    # Internal: gamepad → velocity
    # ------------------------------------------------------------------

    def _velocity_from_gamepad(self) -> Tuple[float, float, float]:
        am = self._profile.get("axis_mapping", {})
        max_vx = self._profile.get("max_vx", 1.0)
        max_vy = self._profile.get("max_vy", 0.5)
        max_vyaw = self._profile.get("max_vyaw", 1.5)

        vx_axis = am.get("left_stick_y", "left_stick_y")
        vy_axis = am.get("left_stick_x", "left_stick_x")
        vyaw_axis = am.get("right_stick_x", "right_stick_x")

        vx = self._gamepad_axes.get(vx_axis, 0.0) * max_vx
        vy = self._gamepad_axes.get(vy_axis, 0.0) * max_vy
        vyaw = self._gamepad_axes.get(vyaw_axis, 0.0) * max_vyaw

        return (vx, vy, vyaw)
