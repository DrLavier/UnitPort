#!/usr/bin/env python3
"""Robot-side UnitPort state banner — LCD + LED on mode transitions.

Called by systemd from ``unitport-banner.service``:

    /usr/local/bin/unitport-banner.py on     # on unitport.target activation
    /usr/local/bin/unitport-banner.py off    # on unitport.target deactivation
    /usr/local/bin/unitport-banner.py connected
                                             # live "connected via USB" hint,
                                             # triggered by identity endpoint
                                             # on successful POST /handshake

Because the banner is wired into ``unitport.target``'s lifecycle, every path
into or out of UnitPort mode — host-initiated SSH isolate, robot's own
handshake-probe self-switch, CLI ``systemctl isolate`` — produces the same
visible feedback on the robot's hardware. That makes "did it actually
connect?" obvious at a glance without looking at the host pill.

Brand-agnostic: probes the Mangdang ``mini_pupper_driver`` classes in order
(Display / Lcd / Screen for the LCD; LedDriver / Led / RgbLed for the LEDs)
and no-ops silently on any robot that doesn't ship the driver. Never fails
boot — banner is purely cosmetic, so exit 0 is always the right move.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple


# Human-readable banner text per state. Kept short so 128x-wide LCDs can
# render it without scroll. No localisation — the robot has no locale
# knowledge and these strings double as diagnostic keys in the journal.
TEXTS = {
    "on":         "UnitPort Mode",
    "connected":  "UnitPort Connected",
    "off":        "Factory Mode",
}

# LED colours. (r, g, b) in 0..1. Green on UnitPort, soft red on factory.
# A missing LED driver is a no-op — the LCD text alone suffices.
LED_COLOURS = {
    "on":         (0.0, 0.8, 0.0),
    "connected":  (0.0, 1.0, 0.2),
    "off":        (0.6, 0.0, 0.0),
}


def _log(msg: str) -> None:
    sys.stderr.write(f"[unitport-banner] {msg}\n")
    sys.stderr.flush()


def _try_lcd(text: str) -> bool:
    """Render ``text`` on the Mini Pupper LCD. True on success."""
    for class_name in ("Display", "Lcd", "Screen"):
        try:
            mod = __import__("mini_pupper_driver", fromlist=[class_name])
        except Exception:
            return False
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        try:
            dev = cls()
            for method in ("show_text", "display_text", "text"):
                if hasattr(dev, method):
                    getattr(dev, method)(text)
                    return True
        except Exception as exc:
            _log(f"LCD {class_name} raised: {exc}")
            continue
    return False


def _try_led(rgb: Tuple[float, float, float]) -> bool:
    """Paint both LEDs ``rgb``. True on success."""
    r, g, b = rgb
    for class_name in ("LedDriver", "Led", "RgbLed"):
        try:
            mod = __import__("mini_pupper_driver", fromlist=[class_name])
        except Exception:
            return False
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        try:
            dev = cls()
            # Various SDK revisions use different method names; try the
            # common ones in order of likelihood.
            for method in ("set_color", "set", "write"):
                if hasattr(dev, method):
                    try:
                        getattr(dev, method)(r, g, b)
                    except TypeError:
                        # Some drivers want (r, g, b) tuple instead of three args
                        getattr(dev, method)((r, g, b))
                    return True
            # Target-specific setters (set_left / set_right).
            touched = False
            for side in ("set_left", "set_right"):
                if hasattr(dev, side):
                    getattr(dev, side)(r, g, b)
                    touched = True
            if touched:
                return True
        except Exception as exc:
            _log(f"LED {class_name} raised: {exc}")
            continue
    return False


def _apply(state: str) -> int:
    text = TEXTS.get(state, state.upper())
    lcd_ok = _try_lcd(text)
    led_ok = _try_led(LED_COLOURS.get(state, (0.0, 0.0, 0.0)))
    if not lcd_ok and not led_ok:
        _log(f"no LCD/LED driver present — '{text}' not shown (robot has no display?)")
    else:
        _log(
            f"banner state={state!r} text={text!r} "
            f"lcd={'ok' if lcd_ok else 'miss'} led={'ok' if led_ok else 'miss'}"
        )
    # Always exit 0 — banner failures must never break boot or mode switches.
    return 0


def main(argv: Optional[list] = None) -> int:
    argv = argv if argv is not None else sys.argv
    action = argv[1].strip().lower() if len(argv) > 1 else "on"
    if action not in TEXTS:
        _log(f"unknown action {action!r}, falling through to 'on'")
        action = "on"
    return _apply(action)


if __name__ == "__main__":
    sys.exit(main())
