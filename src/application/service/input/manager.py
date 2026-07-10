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

from unitport_sdk import Config, Paths, log_debug, log_warning, push_data, read_data

from .command_bus import CommandBus
from .gamepad_source import (
    AXIS_INDEX,
    DEFAULT_AXIS_ROUTING,
    GamepadInputSource,
    is_available as gamepad_is_available,
)
from .keyboard_source import DEFAULT_BINDINGS, KeyboardInputSource


# Default order the policy sees command fields in.
_DEFAULT_FIELD_ORDER: List[str] = ["vx", "vy", "vyaw"]

# Per-(SKU, channel_name) axis bindings for policy command channels —
# user state under USER_CONFIG_DIR (push_data), name-keyed, NEVER
# index-keyed. Shape:
#   {"version": 1, "schema": "controller/channel_bindings",
#    "robots": {<sku>: {<channel_name>: <axis_id>}}}
_CHANNEL_BINDINGS_REL = "controller/channel_bindings.json"

# Per-axis default sign convention for custom pairings: stick-up must be
# positive (pygame y axes are inverted), x axes pass through. The legacy
# (field, axis) pairs keep their historical signs via DEFAULT_AXIS_ROUTING
# so a default-bound stick behaves byte-identically to the pre-contract map.
_AXIS_DEFAULT_SIGN = {
    "left_stick_x": 1.0,
    "left_stick_y": -1.0,
    "right_stick_x": 1.0,
    "right_stick_y": -1.0,
}

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

# Per-SKU trigger channel -> gamepad button binding block key (inside the same
# channel_bindings.json doc as the axis bindings). Shape:
#   doc["skill_triggers"][<sku>][<channel_name>] = {"button": <hw>, "latch": "hold"|"pulse"}
_TRIGGER_BINDINGS_KEY = "skill_triggers"


def _decay_trigger(value: float, dt: float, window_s: float) -> float:
    """Linear decay of a skill trigger envelope 1 -> 0 over ``window_s`` seconds.

    The deploy-side mirror of the training ``SkillTriggerCommand._update_command``
    (``self._command -= dt / window_s`` clamped >= 0), so a bound button drives the
    SAME decaying pulse the policy trained on (train=deploy, skill_command_path_design.md
    Slice 4). A non-positive window collapses to 0 (fail-safe, never negative)."""
    if window_s <= 0.0:
        return 0.0
    return max(0.0, float(value) - float(dt) / float(window_s))


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
        # Contract-driven bus-field routing (installed per loaded policy by
        # the controller panel); None = default axis map.
        self._channel_routing: Optional[Dict[str, Any]] = None
        # Skill trigger channels (Slice 4): per-channel decaying envelope value
        # (rebuilds the training SkillTriggerCommand pulse at deploy), window_s,
        # and the active gamepad-button -> channel map (from per-SKU bindings).
        # Configured per policy-review session via configure_skill_triggers().
        self._skill_trigger_values: Dict[str, float] = {}
        self._skill_trigger_windows: Dict[str, float] = {}
        self._skill_button_to_channel: Dict[str, str] = {}   # gamepad button -> channel
        self._skill_key_to_channel: Dict[int, str] = {}      # keyboard key code -> channel
        # Per-channel resolved latch mode (pulse/hold) + held state. The latch is
        # CLAMPED to the contract's supported_latch at configure time (contract-layer
        # gate), so a hold the trained policy never declared can't reach the runtime.
        self._skill_trigger_latch: Dict[str, str] = {}
        self._skill_trigger_held: Dict[str, bool] = {}
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
                # Fail loud (§8): a binding the user configured must never
                # vanish silently — the stored user.ini entry stays intact,
                # only the ACTIVE map excludes it, and the user is told why.
                log_warning(
                    f"[input] gamepad button binding {hw_name!r} -> "
                    f"{action_id!r} is not a known action "
                    f"({sorted(a for a in _VALID_BUTTON_ACTIONS if a)}); "
                    f"ignoring it for this session — re-bind the button in "
                    f"the Controller panel."
                )
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

    # ----- policy command-channel bindings (per SKU, name-keyed) -----------

    @staticmethod
    def known_axis_ids() -> List[str]:
        """Bindable physical stick axes (Xbox layout)."""
        return list(AXIS_INDEX.keys())

    @staticmethod
    def default_axis_sign(field: str, axis_id: str) -> float:
        """Sign for a (bus field, axis) pairing. Legacy pairings keep the
        historical sign (byte-identical default behaviour); custom pairings
        take the per-axis convention (stick-up positive on y axes)."""
        legacy = DEFAULT_AXIS_ROUTING.get(field)
        if legacy is not None and legacy[0] == axis_id:
            return float(legacy[1])
        return float(_AXIS_DEFAULT_SIGN.get(axis_id, 1.0))

    def _bindings_doc(self) -> Dict[str, Any]:
        """Session-authoritative channel-bindings document.

        Loaded from disk ONCE (read_data caches by path with no mtime
        check, so a read-modify-write cycle through it can resurrect stale
        state); afterwards the in-memory doc is the single source and every
        mutation goes doc-first + push_data. External hand-edits land on
        the next app start — same contract as the user.ini overlay.
        """
        doc = getattr(self, "_channel_bindings_doc", None)
        if doc is None:
            path = Paths.USER_CONFIG_DIR / _CHANNEL_BINDINGS_REL
            try:
                raw = read_data(path) if path.exists() else None
            except Exception as exc:
                log_warning(f"[input] channel bindings read failed: {exc}")
                raw = None
            doc = raw if isinstance(raw, dict) else {}
            self._channel_bindings_doc = doc
        return doc

    def get_channel_axis_bindings(self, sku: str) -> Dict[str, str]:
        """Stored ``{channel_name: axis_id}`` for ``sku`` — verbatim,
        including entries whose axis is no longer known: the panel renders
        those greyed with a warning instead of silently dropping them."""
        doc = self._bindings_doc()
        robots = doc.get("robots")
        block = robots.get(str(sku)) if isinstance(robots, dict) else None
        if not isinstance(block, dict):
            return {}
        return {str(k): str(v) for k, v in block.items()}

    def set_channel_axis_binding(self, sku: str, channel: str, axis_id: str) -> None:
        """Persist one (SKU, channel_name) → axis binding; ``axis_id=""``
        unbinds. Refuses unknown axis ids loudly."""
        sku = str(sku or "").strip()
        channel = str(channel or "").strip()
        if not sku or not channel:
            raise ValueError(
                "set_channel_axis_binding: sku and channel are required "
                f"(got sku={sku!r}, channel={channel!r})"
            )
        axis_id = str(axis_id or "").strip()
        if axis_id and axis_id not in AXIS_INDEX:
            raise ValueError(
                f"set_channel_axis_binding: unknown axis {axis_id!r} "
                f"(known: {sorted(AXIS_INDEX)})"
            )
        doc = self._bindings_doc()
        if not doc:
            doc.update({
                "version": 1,
                "schema": "controller/channel_bindings",
                "robots": {},
            })
        robots = doc.setdefault("robots", {})
        block = robots.setdefault(sku, {})
        if axis_id:
            block[channel] = axis_id
        else:
            block.pop(channel, None)
        if not push_data(_CHANNEL_BINDINGS_REL, doc):
            log_warning(
                f"[input] channel binding persist failed for sku={sku!r} "
                f"channel={channel!r}"
            )

    # ----- skill trigger channels → button / key (Slice 4) -----------------

    def get_channel_button_bindings(self, sku: str) -> Dict[str, Dict[str, Any]]:
        """Stored ``{channel_name: {"button": hw, "key": key_code, "latch": mode}}``
        for ``sku`` — a trigger channel may carry a gamepad button and/or a keyboard
        key (one per controller type)."""
        doc = self._bindings_doc()
        triggers = doc.get(_TRIGGER_BINDINGS_KEY)
        block = triggers.get(str(sku)) if isinstance(triggers, dict) else None
        if not isinstance(block, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for ch, v in block.items():
            if isinstance(v, dict):
                try:
                    key = int(v.get("key", 0) or 0)
                except (TypeError, ValueError):
                    key = 0
                out[str(ch)] = {
                    "button": str(v.get("button", "") or ""),
                    "key": key,
                    "latch": str(v.get("latch", "pulse") or "pulse"),
                }
        return out

    def _update_trigger_binding(
        self, sku: str, channel: str, *, field: str, value, latch: str,
    ) -> None:
        """Read-modify-write one field of a trigger channel's binding spec,
        PRESERVING the other input field (so setting a keyboard key does not wipe
        the gamepad button, and vice-versa). Drops the channel entry when neither a
        button nor a key remains. Fail-loud on a bad latch (§8)."""
        sku = str(sku or "").strip()
        channel = str(channel or "").strip()
        if not sku or not channel:
            raise ValueError(
                f"trigger binding: sku and channel are required (got sku={sku!r}, "
                f"channel={channel!r})"
            )
        latch = str(latch or "pulse").strip().lower()
        if latch not in ("hold", "pulse"):
            raise ValueError(f"trigger binding: latch must be hold/pulse, got {latch!r}")
        doc = self._bindings_doc()
        if not doc:
            doc.update({
                "version": 1,
                "schema": "controller/channel_bindings",
                "robots": {},
            })
        triggers = doc.setdefault(_TRIGGER_BINDINGS_KEY, {})
        block = triggers.setdefault(sku, {})
        spec = block.get(channel)
        spec = dict(spec) if isinstance(spec, dict) else {}
        if value:
            spec[field] = value
        else:
            spec.pop(field, None)
        spec["latch"] = latch
        if spec.get("button") or spec.get("key"):
            block[channel] = spec
        else:
            block.pop(channel, None)
        if not push_data(_CHANNEL_BINDINGS_REL, doc):
            log_warning(
                f"[input] trigger binding persist failed for sku={sku!r} "
                f"channel={channel!r}"
            )

    def set_channel_button_binding(
        self, sku: str, channel: str, hw_button: str, latch: str = "pulse",
    ) -> None:
        """Persist a (SKU, trigger channel) → gamepad BUTTON binding; empty button
        unbinds (the keyboard key, if any, is preserved)."""
        hw = str(hw_button or "").strip().lower()
        self._update_trigger_binding(sku, channel, field="button", value=hw, latch=latch)

    def set_channel_key_binding(
        self, sku: str, channel: str, key_code: int, latch: str = "pulse",
    ) -> None:
        """Persist a (SKU, trigger channel) → keyboard KEY binding; key_code=0
        unbinds (the gamepad button, if any, is preserved)."""
        try:
            key = int(key_code) if key_code else 0
        except (TypeError, ValueError):
            key = 0
        self._update_trigger_binding(sku, channel, field="key", value=key, latch=latch)

    def configure_skill_triggers(self, channels: Any, sku: str) -> None:
        """Set up the live skill-trigger envelopes for a policy-review session.

        ``channels`` = the deploy contract's trigger channels
        (``[{"name": ..., "window_s": ...}, ...]``). Loads the per-SKU
        button→channel bindings so a bound button fires the channel's envelope.
        Idempotent; call on ``policy_review_started``. No triggers → clears state.
        """
        self._skill_trigger_values = {}
        self._skill_trigger_windows = {}
        self._skill_button_to_channel = {}
        self._skill_key_to_channel = {}
        self._skill_trigger_latch = {}
        self._skill_trigger_held = {}
        supported_by_ch: Dict[str, list] = {}
        for ch in (channels or []):
            if not isinstance(ch, dict):
                continue
            name = str(ch.get("name", "") or "").strip()
            if not name:
                continue
            try:
                w = float(ch.get("window_s", 1.0) or 1.0)
            except (TypeError, ValueError):
                w = 1.0
            self._skill_trigger_windows[name] = w if w > 0.0 else 1.0
            self._skill_trigger_values[name] = 0.0
            self._skill_trigger_held[name] = False
            self._skill_trigger_latch[name] = "pulse"
            sl = ch.get("supported_latch")
            modes = [str(m).strip().lower() for m in sl] if isinstance(sl, (list, tuple)) else []
            supported_by_ch[name] = [m for m in modes if m in ("pulse", "hold")] or ["pulse"]
        for channel, spec in self.get_channel_button_bindings(sku).items():
            if channel not in self._skill_trigger_windows:
                continue
            hw = str(spec.get("button", "") or "").strip().lower()
            if hw:
                self._skill_button_to_channel[hw] = channel
            try:
                key = int(spec.get("key", 0) or 0)
            except (TypeError, ValueError):
                key = 0
            if key:
                self._skill_key_to_channel[key] = channel
            latch = str(spec.get("latch", "") or "").strip().lower()
            supported = supported_by_ch.get(channel, ["pulse"])
            # Contract-layer gate: a latch the trained policy did not declare is
            # clamped to a supported mode — never feed the policy an off-distribution
            # sustained hold it never trained on (§8, realized == designed).
            self._skill_trigger_latch[channel] = latch if latch in supported else supported[0]

    def _drive_trigger(self, channel, pressed: bool) -> bool:
        """Advance a trigger channel's envelope for a press/release edge, per its
        latch mode. ``pulse``: a rising edge fires a decaying pulse (1 → 0 over
        window_s). ``hold``: held at 1.0 while pressed, decays after release. ``hold``
        only occurs when the contract declared it supported (clamped in
        configure_skill_triggers) — the runtime never feeds a sustained value the
        trained policy has not seen. Device-agnostic: gamepad button and keyboard key
        both route here."""
        if channel is None:
            return False
        if self._skill_trigger_latch.get(channel, "pulse") == "hold":
            self._skill_trigger_held[channel] = bool(pressed)
            if pressed:
                self._skill_trigger_values[channel] = 1.0
        elif pressed:                                   # pulse — rising edge only
            self._skill_trigger_values[channel] = 1.0
        return True

    def _on_trigger_button(self, hw_name: str, pressed: bool) -> bool:
        """Gamepad button edge → the bound trigger channel's envelope."""
        return self._drive_trigger(
            self._skill_button_to_channel.get(str(hw_name or "").strip().lower()),
            pressed,
        )

    def _on_trigger_key(self, key: int, pressed: bool) -> bool:
        """Keyboard key edge → the bound trigger channel's envelope (skill trigger
        binding parity for the keyboard controller type)."""
        try:
            k = int(key)
        except (TypeError, ValueError):
            return False
        return self._drive_trigger(self._skill_key_to_channel.get(k), pressed)

    def tick_skill_triggers(self, dt: float) -> None:
        """Advance every active trigger envelope one policy step. A held ``hold``
        channel stays at 1.0; everything else decays (matches training decay). The
        deploy/live-sim command provider calls this once per policy step."""
        for ch, val in list(self._skill_trigger_values.items()):
            if self._skill_trigger_latch.get(ch) == "hold" and self._skill_trigger_held.get(ch):
                self._skill_trigger_values[ch] = 1.0
            else:
                self._skill_trigger_values[ch] = _decay_trigger(
                    val, dt, self._skill_trigger_windows.get(ch, 1.0)
                )

    def skill_trigger_values(self) -> Dict[str, float]:
        """Current envelope value per active trigger channel (the deploy command slot)."""
        return dict(self._skill_trigger_values)

    def apply_channel_routing(
        self, routing: Optional[Dict[str, Any]]
    ) -> None:
        """Install a bus-field → (axis_id, sign) routing derived from the
        loaded policy's command contract. ``None`` reverts to the default
        map. Remembered across mode switches (a fresh gamepad source gets
        it re-applied on activation)."""
        self._channel_routing = dict(routing) if routing else None
        src = self._active_source
        fn = getattr(src, "set_axis_routing", None) if src is not None else None
        if callable(fn):
            try:
                fn(self._channel_routing)
            except Exception as exc:
                log_warning(f"[input] apply channel routing raised: {exc}")

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
        self._active_source = KeyboardInputSource(
            self._bus,
            key_map=seed,
            on_estop=lambda: self._emit_system_estop("keyboard"),
        )
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
        # SYSTEM-channel estop (Slice 0): translate a bound estop button-press
        # into an unconditional system estop. button_event fires on the poll
        # thread; this manager is a GUI-thread QObject, so Qt marshals the slot
        # via QueuedConnection (no manual invokeMethod). The bound action is
        # resolved against the manager's own binding table.
        try:
            source.button_event.connect(self._on_gamepad_button)
        except Exception as exc:
            log_warning(f"[input] gamepad button_event connect raised: {exc}")
        # Re-apply the contract-driven channel routing (if a policy set one)
        # — the poll thread is fresh and would otherwise run the default map.
        routing = getattr(self, "_channel_routing", None)
        if routing:
            try:
                source.set_axis_routing(routing)
            except Exception as exc:
                log_warning(f"[input] re-apply channel routing raised: {exc}")
        return True

    # ----- system channel (estop) -----------------------------------------

    def _on_gamepad_button(self, hw_name: str, pressed: bool) -> None:
        """Translate a bound gamepad button edge into a SYSTEM action (Slice 0).

        Only the rising edge of a button bound to ``estop`` fires the
        unconditional system estop — it never touches the CommandBus / command
        vector. Other system actions (stop_robot / start_robot / mode_*) can be
        dispatched from here in a later slice; ``estop`` is the safety-first one.
        """
        # estop fires on the rising edge only.
        if pressed and self.get_gamepad_button_action(hw_name) == "estop":
            self._emit_system_estop("gamepad")
            return
        # A bound skill trigger channel: press AND release both matter (a hold-latch
        # channel sustains while held, decays on release). A button is either an
        # estop OR a trigger, never both.
        self._on_trigger_button(hw_name, pressed)

    def _emit_system_estop(self, source: str) -> None:
        """Broadcast the SYSTEM-channel estop. Loud by design (safety event)."""
        log_warning(f"[input] ESTOP triggered via {source} — dispatching system stop")
        try:
            from application.service.signals import get_app_signals
            get_app_signals().system_estop.emit(str(source))
        except Exception as exc:
            log_warning(f"[input] system_estop emit failed: {exc}")

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
        # A bound skill trigger key fires that channel's envelope (rising edge),
        # in parallel with the source's velocity/action handling.
        self._on_trigger_key(key, True)
        fn = getattr(self._active_source, "key_pressed", None)
        if callable(fn):
            try:
                fn(int(key))
            except Exception as exc:
                log_warning(f"[input] forward_key_press: {exc}")

    def forward_key_release(self, key: int) -> None:
        if self._mode != "keyboard" or self._active_source is None:
            return
        self._on_trigger_key(key, False)   # hold-latch release decays; pulse no-op
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
