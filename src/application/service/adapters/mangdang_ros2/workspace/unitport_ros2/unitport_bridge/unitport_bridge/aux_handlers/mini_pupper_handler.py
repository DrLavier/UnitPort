"""Mini Pupper 2 aux handler — LEDs, LCD, speaker.

Runs on the robot only. Lazily imports ``mini_pupper_driver`` so generic
robots (or Mini Pupper images where the factory driver was removed)
don't crash when the bridge starts — the lookup failure is logged and
the handler falls back to a no-op.

Supported surface/target/payload combinations:

    surface=led    target=left|right|both   payload={r: 0..1, g: 0..1, b: 0..1}
    surface=lcd    target=main              payload={text: "...", line?: 0, color?: "white"}
    surface=speaker target=default          payload={sound: "estop.wav"}
                                            or {frequency_hz: 440, duration_ms: 200}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from unitport_bridge.aux_handlers import BaseAuxHandler


class MiniPupperAuxHandler(BaseAuxHandler):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._led_driver = None
        self._display = None
        self._audio = None
        self._load_drivers()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handle(self, *, surface: str, target: str, payload: Dict[str, Any]) -> None:
        s = (surface or "").lower()
        try:
            if s == "led":
                self._handle_led(target, payload)
            elif s == "lcd":
                self._handle_lcd(target, payload)
            elif s == "speaker":
                self._handle_speaker(target, payload)
            elif s == "button_led":
                self._handle_led(target, payload)  # Mini Pupper treats button LED as LED.
            else:
                self._log_warn(
                    f"[mini_pupper] unknown surface={surface!r}; dropping"
                )
        except Exception as exc:
            # Swallow — the bridge already catches for us, but be defensive
            # so a broken call never interrupts a followup.
            self._log_warn(f"[mini_pupper] {surface} dispatch failed: {exc}")

    # ------------------------------------------------------------------
    # Surface handlers
    # ------------------------------------------------------------------

    def _handle_led(self, target: str, payload: Dict[str, Any]) -> None:
        if self._led_driver is None:
            self._log_warn("[mini_pupper] LED driver unavailable; dropping command")
            return
        r = _clamp01(float(payload.get("r", 0.0)))
        g = _clamp01(float(payload.get("g", 0.0)))
        b = _clamp01(float(payload.get("b", 0.0)))
        tgt = (target or "both").lower()
        # mini_pupper_driver exposes a LedDriver-like class; method names vary
        # between SDK versions (set_color / set / show). Try them in order.
        for method_name in ("set_color", "set", "show", "set_rgb"):
            method = getattr(self._led_driver, method_name, None)
            if callable(method):
                try:
                    method(r, g, b, tgt) if _accepts_four(method) else method(r, g, b)
                    return
                except TypeError:
                    continue
        self._log_warn("[mini_pupper] no known LED method on driver")

    def _handle_lcd(self, target: str, payload: Dict[str, Any]) -> None:
        if self._display is None:
            self._log_warn("[mini_pupper] LCD driver unavailable; dropping command")
            return
        text = str(payload.get("text", ""))
        for method_name in ("show_text", "display_text", "show"):
            method = getattr(self._display, method_name, None)
            if callable(method):
                try:
                    method(text)
                    return
                except TypeError:
                    continue
        self._log_warn("[mini_pupper] no known LCD method on driver")

    def _handle_speaker(self, target: str, payload: Dict[str, Any]) -> None:
        if self._audio is None:
            self._log_warn("[mini_pupper] audio driver unavailable; dropping command")
            return
        sound = payload.get("sound")
        if sound:
            for method_name in ("play_sound", "play_wav", "play"):
                method = getattr(self._audio, method_name, None)
                if callable(method):
                    method(str(sound))
                    return
        freq = payload.get("frequency_hz")
        if freq is not None:
            duration = int(payload.get("duration_ms", 200))
            for method_name in ("beep", "tone"):
                method = getattr(self._audio, method_name, None)
                if callable(method):
                    method(int(freq), duration)
                    return
        self._log_warn("[mini_pupper] no audio action matched payload")

    # ------------------------------------------------------------------
    # Lazy driver import
    # ------------------------------------------------------------------

    def _load_drivers(self) -> None:
        try:
            import mini_pupper_driver  # type: ignore[import]
        except Exception as exc:
            self._log_warn(
                f"[mini_pupper] mini_pupper_driver unavailable ({exc}); "
                "aux commands will be logged only"
            )
            return

        # The Mangdang SDK's class names have shifted over time. Try the
        # public attributes we know about; store whichever resolves.
        for attr in ("LedDriver", "Led", "RgbLed"):
            cls = getattr(mini_pupper_driver, attr, None)
            if cls is not None:
                try:
                    self._led_driver = cls()
                    break
                except Exception:
                    continue

        for attr in ("Display", "Lcd", "Screen"):
            cls = getattr(mini_pupper_driver, attr, None)
            if cls is not None:
                try:
                    self._display = cls()
                    break
                except Exception:
                    continue

        for attr in ("Audio", "Speaker", "Buzzer"):
            cls = getattr(mini_pupper_driver, attr, None)
            if cls is not None:
                try:
                    self._audio = cls()
                    break
                except Exception:
                    continue

        self._log_info(
            f"[mini_pupper] drivers loaded: led={bool(self._led_driver)} "
            f"lcd={bool(self._display)} audio={bool(self._audio)}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _accepts_four(method) -> bool:
    """Best-effort arity probe — avoids calling inspect.signature on C methods."""
    try:
        import inspect
        sig = inspect.signature(method)
        return len(sig.parameters) >= 4
    except (TypeError, ValueError):
        return False


__all__ = ["MiniPupperAuxHandler"]
