"""Process-singleton input manager for live policy replay.

Background
----------
The runtime stack already has all the pieces it needs to feed live
keyboard / gamepad input into a running policy:

* :class:`CommandBus` is a thread-safe key-value store for command
  scalars (``vx``, ``vy``, ``vyaw``, ...).
* :class:`KeyboardInputSource` converts Qt key-press events into
  CommandBus publishes.
* :class:`GamepadInputSource` polls a connected gamepad in its own
  thread and publishes axis values to the bus.
* The bundle's ``SkillManifest.command_interface`` declares which obs
  indices the policy expects to see those commands in.

What's missing is a single point of coordination so that

1. the **Controller sidebar panel** can pick which input mode is
   active and read the live values for display,
2. the **Mission BehaviorNode** replay loop can ask "is live input
   on, and if so what should I send the policy this tick?" without
   having to know about Qt, pygame, or input source classes,
3. the **Training workspace Export Review** can ask the same
   question without duplicating any wiring.

That single point of coordination is :class:`GlobalInputManager`. It
holds *one* CommandBus and *at most one* active input source for the
whole UnitPort process. Replay loops never construct their own bus —
they call :func:`get_global_input_manager` and read from it.

This module is intentionally tiny: most of the real work lives in the
underlying CommandBus / KeyboardInputSource / GamepadInputSource. The
manager just owns the lifecycle and exposes a stable façade.

The keyboard input source needs Qt key events to be forwarded into it.
That bridge is set up by the Controller sidebar panel via a Qt event
filter — the manager itself stays Qt-free so it can be unit-tested
without a GUI.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from .command_bus import CommandBus

log = logging.getLogger(__name__)


# Default order the policy sees command fields in. Matches the IL Go2
# velocity-tracking task: vx → vy → vyaw at obs slots 9, 10, 11. Future
# bundles with different ``command_interface`` shapes can override this
# via the optional ``set_field_order`` API.
_DEFAULT_FIELD_ORDER: List[str] = ["vx", "vy", "vyaw"]

# user.ini section + key names. Single source of truth so the panel and
# manager never type raw config keys.
_INI_SECTION = "CONTROLLER"
_INI_KEY_MODE = "mode"
_INI_KEY_INVERT_VY = "invert_vy"
_INI_KEY_KB_BINDINGS = "keyboard_bindings"

# Boot mode used when user.ini has no [CONTROLLER] section yet -- the user
# explicitly asked that fresh installs land in keyboard mode rather than off.
_DEFAULT_BOOT_MODE = "keyboard"


class GlobalInputManager:
    """Single owner of the live-input CommandBus + active input source."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bus = CommandBus()
        self._active_source: Optional[Any] = None
        self._mode: str = "off"
        self._field_order: List[str] = list(_DEFAULT_FIELD_ORDER)
        # Listeners notified whenever the live mode changes — used by the
        # sidebar panel to refresh its UI without polling.
        self._mode_listeners: List[Callable[[str], None]] = []
        # Persisted preferences loaded from user.ini on first boot.
        self._invert_vy: bool = False
        self._boot_preferred_mode: str = _DEFAULT_BOOT_MODE
        # Saved keyboard bindings from user.ini. Empty dict means "no
        # override on disk -- use the source's compiled-in defaults".
        self._saved_kb_bindings: Dict[str, int] = {}
        self._load_config()
        self._auto_activate_boot_mode()

    # ------------------------------------------------------------------
    # Persistence (user.ini round-trip)
    # ------------------------------------------------------------------

    def _config(self) -> Optional[Any]:
        """Return the shared ConfigManager singleton, or None if unavailable.

        Wrapped in a try so unit tests that don't initialise the wider
        UnitPort runtime can still construct a manager.
        """
        try:
            from src.system.core.config_manager import ConfigManager
            return ConfigManager()
        except Exception:
            return None

    def _load_config(self) -> None:
        cfg = self._config()
        if cfg is None:
            return
        try:
            saved_mode = cfg.get(
                _INI_SECTION, _INI_KEY_MODE,
                fallback=_DEFAULT_BOOT_MODE,
                config_type="user",
            )
            saved_invert = cfg.get_bool(
                _INI_SECTION, _INI_KEY_INVERT_VY,
                fallback=False,
                config_type="user",
            )
            saved_kb_raw = cfg.get(
                _INI_SECTION, _INI_KEY_KB_BINDINGS,
                fallback="",
                config_type="user",
            )
        except Exception:
            log.exception("GlobalInputManager: failed to read user.ini")
            return
        mode = (saved_mode or "").strip().lower() or _DEFAULT_BOOT_MODE
        if mode not in {"off", "keyboard", "gamepad"}:
            mode = _DEFAULT_BOOT_MODE
        self._boot_preferred_mode = mode
        self._invert_vy = bool(saved_invert)
        self._saved_kb_bindings = self._parse_kb_bindings(saved_kb_raw)

    @staticmethod
    def _parse_kb_bindings(raw: Any) -> Dict[str, int]:
        """Parse the JSON-encoded keyboard binding map from user.ini.

        Returns ``{}`` for any unparseable / empty payload so the boot
        path falls through to the source's compiled-in defaults.
        """
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            import json
            data = json.loads(text)
        except Exception:
            log.warning("GlobalInputManager: keyboard_bindings is not valid JSON: %r", text)
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

    def _persist_mode(self, mode: str) -> None:
        cfg = self._config()
        if cfg is None:
            return
        try:
            cfg.set(_INI_SECTION, _INI_KEY_MODE, mode, config_type="user")
            cfg.save_user_config()
        except Exception:
            log.exception("GlobalInputManager: failed to persist mode")

    def _persist_keyboard_bindings(self, bindings: Dict[str, int]) -> None:
        """Serialize the keyboard binding map to user.ini as a JSON blob.

        Storing as JSON (rather than one INI key per action) keeps the
        schema future-proof: new actions added later don't need their
        own dedicated key, and the section stays readable.
        """
        cfg = self._config()
        if cfg is None:
            return
        try:
            import json
            payload = json.dumps(
                {str(k): int(v) for k, v in (bindings or {}).items()},
                separators=(",", ":"),
                sort_keys=True,
            )
            cfg.set(_INI_SECTION, _INI_KEY_KB_BINDINGS, payload, config_type="user")
            cfg.save_user_config()
        except Exception:
            log.exception("GlobalInputManager: failed to persist keyboard bindings")

    def _persist_invert_vy(self, inverted: bool) -> None:
        cfg = self._config()
        if cfg is None:
            return
        try:
            cfg.set(
                _INI_SECTION, _INI_KEY_INVERT_VY,
                "true" if inverted else "false",
                config_type="user",
            )
            cfg.save_user_config()
        except Exception:
            log.exception("GlobalInputManager: failed to persist invert_vy")

    def _auto_activate_boot_mode(self) -> None:
        """Try the saved mode at startup; fall back gracefully on failure.

        Fallback chain: saved -> "keyboard" -> "off". A user who saved
        "gamepad" but unplugged the controller still lands in a usable
        keyboard state instead of stuck Off.
        """
        target = self._boot_preferred_mode
        if target == "off":
            self._mode = "off"
            return
        if self._try_activate(target):
            return
        if target != "keyboard" and self._try_activate("keyboard"):
            return
        self._mode = "off"

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
        """Reapply the cached invert_vy flag whenever a new source becomes active."""
        if not self._invert_vy or self._active_source is None:
            return
        fn = getattr(self._active_source, "set_field_inverted", None)
        if callable(fn):
            try:
                fn("vy", True)
            except Exception:
                log.exception("apply_invert_to_active_source: source raised")

    # ──────────────────────────────────────────────────────────────────
    # Mode management
    # ──────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """``"off"``, ``"keyboard"``, or ``"gamepad"``."""
        return self._mode

    def is_active(self) -> bool:
        """True when an input source is currently bound to the bus."""
        return self._mode != "off" and self._active_source is not None

    def set_mode(self, mode: str) -> bool:
        """Switch the active input source.

        Returns ``True`` on success, ``False`` if the requested source
        is unavailable (e.g. ``"gamepad"`` when no controller is plugged
        in). The previous source is always cleanly stopped first.

        Both successful and unsuccessful mode requests are persisted to
        user.ini -- so a user who explicitly clicks Off will land in
        Off on the next launch, and a user who tried gamepad will see
        the panel auto-attempt gamepad on the next launch (with a
        graceful keyboard fallback if the controller is gone).
        """
        target = (mode or "off").strip().lower()
        if target not in {"off", "keyboard", "gamepad"}:
            log.warning("GlobalInputManager.set_mode: unknown mode %r", target)
            return False

        ok = self._try_activate(target)
        # Persist the *user's intent*, even when activation failed --
        # next launch we'll retry that mode and fall back if it still
        # doesn't work. Without this, a temporary disconnect would
        # silently demote the user's preference.
        self._persist_mode(target)
        self._notify_mode_listeners(self._mode)
        return ok

    def _teardown_active_source(self) -> None:
        if self._active_source is None:
            return
        try:
            stop_fn = getattr(self._active_source, "stop", None)
            if callable(stop_fn):
                stop_fn()
        except Exception:
            log.exception("GlobalInputManager: failed to stop active source")
        self._active_source = None

    def _activate_keyboard(self) -> bool:
        try:
            from .input_sources.keyboard_source import KeyboardInputSource
        except Exception as exc:
            log.warning("KeyboardInputSource unavailable: %s", exc)
            return False
        # Seed the source with the persisted user.ini bindings (if any)
        # so a freshly-activated keyboard mode honours the user's saved
        # remap from the very first keypress.
        seed = dict(self._saved_kb_bindings) if self._saved_kb_bindings else None
        self._active_source = KeyboardInputSource(self._bus, key_map=seed)
        return True

    def _activate_gamepad(self) -> bool:
        try:
            from .input_sources.gamepad_source import GamepadInputSource
        except Exception as exc:
            log.warning("GamepadInputSource unavailable: %s", exc)
            return False
        source = GamepadInputSource(self._bus)
        try:
            started = bool(source.start())
        except Exception:
            log.exception("Gamepad start() raised")
            started = False
        if not started:
            log.warning(
                "GamepadInputSource.start() returned False — no gamepad detected"
            )
            return False
        self._active_source = source
        return True

    # ──────────────────────────────────────────────────────────────────
    # Keyboard event forwarding (driven by the Controller sidebar panel)
    # ──────────────────────────────────────────────────────────────────

    def forward_key_press(self, key: int) -> None:
        """Forward a Qt key press into the active keyboard source.

        No-op when the active mode isn't ``"keyboard"`` or no keyboard
        source is bound.
        """
        if self._mode != "keyboard" or self._active_source is None:
            return
        fn = getattr(self._active_source, "key_pressed", None)
        if callable(fn):
            try:
                fn(int(key))
            except Exception:
                log.exception("forward_key_press failed")

    def forward_key_release(self, key: int) -> None:
        """Forward a Qt key release into the active keyboard source."""
        if self._mode != "keyboard" or self._active_source is None:
            return
        fn = getattr(self._active_source, "key_released", None)
        if callable(fn):
            try:
                fn(int(key))
            except Exception:
                log.exception("forward_key_release failed")

    # ──────────────────────────────────────────────────────────────────
    # Keyboard binding mutation (used by the Controller sidebar panel)
    # ──────────────────────────────────────────────────────────────────

    def get_keyboard_bindings(self) -> Dict[str, int]:
        """Return a snapshot of the active keyboard source's action→key map.

        Returns an empty dict when the active mode isn't ``"keyboard"``.
        """
        if self._mode != "keyboard" or self._active_source is None:
            return {}
        fn = getattr(self._active_source, "get_all_bindings", None)
        if not callable(fn):
            return {}
        try:
            return dict(fn())
        except Exception:
            log.exception("get_keyboard_bindings: source raised")
            return {}

    def set_keyboard_binding(self, action: str, key: int) -> bool:
        """Bind ``action`` (e.g. ``"forward"``) to a Qt key code.

        No-op when the active mode isn't ``"keyboard"``. Returns True
        on success, False otherwise. Successful changes are persisted
        to user.ini so the next launch picks the same binding up.
        """
        if self._mode != "keyboard" or self._active_source is None:
            return False
        fn = getattr(self._active_source, "set_binding", None)
        if not callable(fn):
            return False
        try:
            fn(action, key)
        except Exception:
            log.exception("set_keyboard_binding: source raised")
            return False
        # Mirror the change into the manager's cached snapshot and
        # write the full map back to user.ini. We persist the entire
        # snapshot (not just the delta) so user.ini always reflects a
        # complete picture of the user's intended bindings.
        snapshot = self.get_keyboard_bindings()
        if snapshot:
            self._saved_kb_bindings = dict(snapshot)
            self._persist_keyboard_bindings(snapshot)
        return True

    def reset_keyboard_bindings(self) -> bool:
        """Restore the active keyboard source's default keymap.

        Also clears the persisted override in user.ini so subsequent
        launches start from the compiled-in defaults.
        """
        if self._mode != "keyboard" or self._active_source is None:
            return False
        fn = getattr(self._active_source, "reset_bindings", None)
        if not callable(fn):
            return False
        try:
            fn()
        except Exception:
            log.exception("reset_keyboard_bindings: source raised")
            return False
        # Wipe the persisted override so the next boot uses defaults.
        self._saved_kb_bindings = {}
        self._persist_keyboard_bindings({})
        return True

    # ------------------------------------------------------------------
    # Per-field axis invert (currently only "vy" is wired in the panel)
    # ------------------------------------------------------------------

    @property
    def invert_vy(self) -> bool:
        """Persisted preference: True iff the user wants vy negated.

        Cached on the manager so the panel can read it without going
        through the source -- the value survives mode switches and is
        re-applied to whichever source becomes active next.
        """
        return self._invert_vy

    def set_invert_vy(self, inverted: bool) -> None:
        """Update the vy-invert preference, persist it, and apply it now.

        The flag is stored on the manager regardless of the active mode
        (so it survives a mode switch), persisted to user.ini, and
        forwarded to the active source's ``set_field_inverted("vy", ...)``
        when one is bound. Sources that don't expose that hook (the
        keyboard source today, since vy is keyboard-correct already)
        silently ignore the call.
        """
        target = bool(inverted)
        if target == self._invert_vy:
            # Still persist in case user.ini didn't have the key yet --
            # cheap, atomic, no risk.
            self._persist_invert_vy(target)
            return
        self._invert_vy = target
        self._persist_invert_vy(target)
        if self._active_source is None:
            return
        fn = getattr(self._active_source, "set_field_inverted", None)
        if callable(fn):
            try:
                fn("vy", target)
            except Exception:
                log.exception("set_invert_vy: source raised")

    # ──────────────────────────────────────────────────────────────────
    # Command consumption (used by replay loops)
    # ──────────────────────────────────────────────────────────────────

    @property
    def field_order(self) -> List[str]:
        """The ordered command field names the policy expects.

        Defaults to ``["vx", "vy", "vyaw"]`` (Isaac Lab velocity_2d).
        Replay sites can override per-bundle via :meth:`set_field_order`
        when the bundle's ``command_interface`` declares a different
        layout.
        """
        return list(self._field_order)

    def set_field_order(self, fields: Sequence[str]) -> None:
        """Override the field name list (e.g. for a non-velocity_2d bundle)."""
        with self._lock:
            self._field_order = [str(f) for f in fields if f]

    def get_live_values(self) -> Dict[str, float]:
        """Read all current command values as a name→value dict.

        Used by the sidebar panel for the live display. Always returns
        all configured fields with ``0.0`` defaults so the UI doesn't
        have to handle missing keys.
        """
        return self._bus.read_all(self._field_order, default=0.0)

    def make_command_provider(self) -> Callable[[], List[float]]:
        """Return a callable suitable for ``PolicyRunner.run_episode(command_provider=...)``.

        The callable returns a list aligned to :attr:`field_order` —
        i.e. ``[vx, vy, vyaw, ...]`` — read from the live CommandBus.
        It is safe to call from any thread (the underlying bus is
        thread-safe). When the manager is in ``"off"`` mode the
        provider returns zeros, which a replay loop can interpret as
        "no live input, fall back to bundle defaults" or "stand still"
        depending on its policy.
        """
        # Capture the current field_order so the provider doesn't get
        # surprised if a panel mid-flight calls set_field_order.
        order = list(self._field_order)
        bus = self._bus

        def _provider() -> List[float]:
            snapshot = bus.read_all(order, default=0.0)
            return [float(snapshot.get(k, 0.0)) for k in order]

        return _provider

    # ──────────────────────────────────────────────────────────────────
    # Mode-change pub/sub (used by the panel UI)
    # ──────────────────────────────────────────────────────────────────

    def add_mode_listener(self, listener: Callable[[str], None]) -> None:
        if listener not in self._mode_listeners:
            self._mode_listeners.append(listener)

    def remove_mode_listener(self, listener: Callable[[str], None]) -> None:
        try:
            self._mode_listeners.remove(listener)
        except ValueError:
            pass

    def _notify_mode_listeners(self, mode: str) -> None:
        for cb in list(self._mode_listeners):
            try:
                cb(mode)
            except Exception:
                log.exception("mode listener raised")


# ---------------------------------------------------------------------------
# Process singleton
# ---------------------------------------------------------------------------

_singleton: Optional[GlobalInputManager] = None
_singleton_lock = threading.Lock()


def get_global_input_manager() -> GlobalInputManager:
    """Return the process-wide :class:`GlobalInputManager`.

    Lazy-initialized on first call. All replay sites + the Controller
    sidebar panel must use this accessor — never construct a
    :class:`GlobalInputManager` directly.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = GlobalInputManager()
    return _singleton
