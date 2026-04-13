"""GamepadInputSource — polls gamepad and publishes to CommandBus.

Uses ``pygame.joystick`` as the cross-platform backend (pygame is already
a common dependency for sim visualization).  Falls back gracefully when
no gamepad is detected.

Default mapping for quadruped velocity_2d:
    left_stick_y  → vx   (forward/back, inverted)
    left_stick_x  → vy   (lateral)
    right_stick_x → vyaw  (turning)

Publishes to CommandBus at gamepad poll rate (~100 Hz).
CommandBus values are consumed by ReactiveLoop at policy frequency (~50 Hz).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default mapping (matches quadruped_gamepad.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_MAPPING = {
    "left_stick_y": {"target": "vx", "scale": -1.0, "deadzone": 0.1},
    "left_stick_x": {"target": "vy", "scale": 0.5, "deadzone": 0.1},
    "right_stick_x": {"target": "vyaw", "scale": 1.0, "deadzone": 0.15},
}

_DEFAULT_SMOOTHING_ALPHA = 0.3

# Axis indices for common controllers (SDL convention)
_AXIS_MAP = {
    "left_stick_x": 0,
    "left_stick_y": 1,
    "right_stick_x": 2,   # may be 3 on some controllers
    "right_stick_y": 3,
}

# T8: Button index → semantic name mapping for the standard
# Xbox / Xinput layout (matches SDL/pygame defaults).  We publish each
# button's pressed state to the CommandBus under a ``__btn_<name>`` key
# so consumers (IsaacReviewSession) can edge-detect press transitions
# and dispatch IPC ops.  The double-underscore prefix keeps these out
# of the policy command vector (vx/vy/vyaw) which is what makes them
# safe to mix on the same bus.
_BUTTON_NAME_MAP = {
    0: "a",       # Cross   on PlayStation
    1: "b",       # Circle  on PlayStation
    2: "x",       # Square  on PlayStation
    3: "y",       # Triangle on PlayStation
    4: "lb",      # L1
    5: "rb",      # R1
    6: "back",    # Share / Select
    7: "start",   # Options / Menu
    8: "lstick",  # L3
    9: "rstick",  # R3
}


class GamepadInputSource:
    """Polls a gamepad and publishes velocity commands to a CommandBus.

    Parameters
    ----------
    command_bus:
        The CommandBus to publish to.
    mapping:
        Axis → command mapping. If None, uses default quadruped mapping.
    poll_rate_hz:
        How often to read the gamepad (default 100 Hz).
    smoothing_alpha:
        Exponential smoothing factor (0 = no smoothing, 1 = raw values).
    """

    def __init__(
        self,
        command_bus: Any,
        *,
        mapping: Optional[Dict[str, Dict[str, Any]]] = None,
        poll_rate_hz: float = 100.0,
        smoothing_alpha: float = _DEFAULT_SMOOTHING_ALPHA,
    ) -> None:
        self._bus = command_bus
        self._mapping = mapping or dict(_DEFAULT_MAPPING)
        self._poll_dt = 1.0 / max(poll_rate_hz, 1.0)
        self._alpha = smoothing_alpha
        self._smoothed: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._joystick: Any = None
        self._available = False

    @property
    def is_available(self) -> bool:
        """True if a gamepad was detected."""
        return self._available

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Initialize gamepad and start polling thread.

        Returns True if a gamepad was found and polling started.
        """
        if self.is_running:
            return True

        if not self._init_gamepad():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="gamepad_poll", daemon=True,
        )
        self._thread.start()
        log.info("GamepadInputSource started (poll_dt=%.3fs)", self._poll_dt)
        return True

    def stop(self) -> None:
        """Stop polling and release gamepad."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cleanup_gamepad()

    # ------------------------------------------------------------------
    # Runtime axis tweaks (used by the Controller sidebar panel)
    # ------------------------------------------------------------------

    def set_field_inverted(self, field: str, inverted: bool) -> bool:
        """Flip the sign of every axis that targets ``field``.

        The default Go2 layout has ``left_stick_x`` -> ``vy`` with a
        positive scale, but Isaac Lab's body-frame y-axis points to
        the robot's left -- pushing the stick right therefore makes
        the robot strafe in the user-perceived "wrong" direction. The
        Controller panel exposes a per-field invert toggle that calls
        this method, which negates ``scale`` on every mapping entry
        whose ``target`` matches and re-publishes the smoothed state
        so the change takes effect on the next poll tick.

        Returns True when at least one mapping entry was flipped.
        """
        flipped = False
        for axis_name, cfg in self._mapping.items():
            if not isinstance(cfg, dict):
                continue
            target = str(cfg.get("target", "") or "")
            if target != field:
                continue
            current_scale = float(cfg.get("scale", 0.0) or 0.0)
            magnitude = abs(current_scale)
            if magnitude < 1e-12:
                continue
            new_scale = -magnitude if inverted else magnitude
            if abs(new_scale - current_scale) < 1e-12:
                flipped = True  # already in the requested state
                continue
            cfg["scale"] = new_scale
            flipped = True
        # Wipe smoothed state so the next published value reflects the
        # new sign instead of dragging the old one through the EMA.
        self._smoothed.pop(field, None)
        try:
            self._bus.publish(field, 0.0)
        except Exception:
            pass
        return flipped

    def is_field_inverted(self, field: str) -> bool:
        """Return True iff every axis targeting ``field`` has a negative scale.

        The Go2 default has positive scale on left_stick_x -> vy, so
        ``is_field_inverted("vy")`` returns False on a fresh source and
        True after one ``set_field_inverted("vy", True)`` call.
        """
        seen = False
        for cfg in self._mapping.values():
            if not isinstance(cfg, dict):
                continue
            if str(cfg.get("target", "") or "") != field:
                continue
            seen = True
            scale = float(cfg.get("scale", 0.0) or 0.0)
            if scale >= 0:
                return False
        return seen

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(cls, command_bus: Any, profile_path: Path) -> "GamepadInputSource":
        """Create from a YAML input profile."""
        try:
            import yaml
            with profile_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            cfg = {}

        mapping = {}
        raw_mapping = cfg.get("mapping", cfg.get("axis_mapping", {}))
        if isinstance(raw_mapping, dict):
            for axis_key, target in raw_mapping.items():
                if isinstance(target, str):
                    mapping[axis_key] = {"target": target, "scale": 1.0, "deadzone": 0.1}
                elif isinstance(target, dict):
                    mapping[axis_key] = target

        alpha = float(cfg.get("smoothing_alpha", _DEFAULT_SMOOTHING_ALPHA))
        return cls(command_bus, mapping=mapping or None, smoothing_alpha=alpha)

    # ------------------------------------------------------------------
    # Internal: gamepad init / cleanup
    # ------------------------------------------------------------------

    def _init_gamepad(self) -> bool:
        """Try to initialize pygame joystick."""
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            pygame.joystick.init()
            count = pygame.joystick.get_count()
            if count == 0:
                log.info("No gamepad detected")
                self._available = False
                return False
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._available = True
            log.info("Gamepad found: %s", self._joystick.get_name())
            return True
        except ImportError:
            log.warning("pygame not available — gamepad support disabled")
            self._available = False
            return False
        except Exception as exc:
            log.warning("Gamepad init failed: %s", exc)
            self._available = False
            return False

    def _cleanup_gamepad(self) -> None:
        """Release joystick resources."""
        try:
            if self._joystick is not None:
                self._joystick.quit()
                self._joystick = None
        except Exception:
            pass
        self._available = False

    # ------------------------------------------------------------------
    # Internal: poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Thread target: continuously poll gamepad and publish."""
        try:
            import pygame
        except ImportError:
            return

        while not self._stop_event.is_set():
            tick_start = time.monotonic()

            try:
                pygame.event.pump()
                values: Dict[str, float] = {}

                # ── Axes → vx/vy/vyaw ────────────────────────────────
                for axis_name, cfg in self._mapping.items():
                    axis_idx = _AXIS_MAP.get(axis_name)
                    if axis_idx is None:
                        continue
                    if axis_idx >= self._joystick.get_numaxes():
                        continue

                    raw = self._joystick.get_axis(axis_idx)

                    # Deadzone
                    deadzone = float(cfg.get("deadzone", 0.1))
                    if abs(raw) < deadzone:
                        raw = 0.0

                    # Scale
                    scale = float(cfg.get("scale", 1.0))
                    val = raw * scale

                    # Smoothing
                    target = cfg.get("target", axis_name)
                    prev = self._smoothed.get(target, 0.0)
                    smoothed = self._alpha * val + (1.0 - self._alpha) * prev
                    self._smoothed[target] = smoothed
                    values[target] = smoothed

                # ── T8: Buttons → ``__btn_<name>`` keys ──────────────
                # We publish ALL buttons every tick (1.0 = pressed, 0.0
                # = released).  Edge detection happens consumer-side
                # (IsaacReviewSession._poll_button_events) so the bus
                # stays stateless and publishers don't need to track
                # previous values.
                try:
                    n_buttons = self._joystick.get_numbuttons()
                except Exception:
                    n_buttons = 0
                for btn_idx in range(min(n_buttons, len(_BUTTON_NAME_MAP) + 4)):
                    name = _BUTTON_NAME_MAP.get(btn_idx)
                    if name is None:
                        continue
                    try:
                        pressed = bool(self._joystick.get_button(btn_idx))
                    except Exception:
                        pressed = False
                    values[f"__btn_{name}"] = 1.0 if pressed else 0.0

                if values:
                    self._bus.publish_batch(values)

            except Exception:
                log.debug("Gamepad poll error", exc_info=True)

            elapsed = time.monotonic() - tick_start
            sleep_time = self._poll_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
