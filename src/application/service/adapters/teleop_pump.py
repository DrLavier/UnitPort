"""TeleopPump — rate-limited CommandBus → adapter sink pump.

A teleop pump:

1. Polls a caller-supplied ``provider()`` at a fixed rate (default 50 Hz)
2. Applies a per-brand ``safety_clamp(twist) -> twist`` envelope
3. Pushes the clamped twist to ``sink(twist)`` (typically the adapter's
   bridge.publish wrapper)
4. Enforces a **deadman**: if ``provider()`` returns ``None`` for longer
   than ``deadman_s`` continuously, the pump pushes a zero twist + logs a
   warning. Restores normal pumping on the next non-None value.

The pump runs in a background ``threading.Thread`` (not a QTimer) so it
keeps ticking independently of the Qt event loop — important because the
Qt loop may briefly stall (large repaints, modal dialogs) and a missed
teleop tick translates directly to wobbly robot motion.

Lifecycle: ``start()`` / ``stop()``. ``is_running()`` for introspection.
``stop()`` is idempotent and joins the worker; ``start()`` after a ``stop()``
is supported (e.g. user toggles Take Control off-then-on without restarting
the adapter session).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from unitport_sdk import log_warning


# Plain dict matching the geometry_msgs/Twist IDL shape used by NativeDDSBridge.
_ZERO_TWIST = {
    "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
}


def zero_twist() -> dict:
    """Fresh zero twist dict (immutable callers must copy)."""
    return {
        "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


class TeleopPump:
    """Pump live CommandBus values to a brand-specific sink at a fixed rate."""

    DEFAULT_RATE_HZ = 50.0
    DEFAULT_DEADMAN_S = 0.5

    def __init__(
        self,
        provider: Callable[[], Optional[dict]],
        sink: Callable[[dict], None],
        rate_hz: float = DEFAULT_RATE_HZ,
        deadman_s: float = DEFAULT_DEADMAN_S,
        safety_clamp: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        self._provider = provider
        self._sink = sink
        self._period_s = 1.0 / max(1.0, float(rate_hz))
        self._deadman_s = max(0.05, float(deadman_s))
        self._safety_clamp = safety_clamp or (lambda t: t)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="TeleopPump", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        if not self._running:
            return
        self._stop_evt.set()
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        last_input_ts = time.monotonic()
        deadman_warned = False
        while not self._stop_evt.wait(self._period_s):
            now = time.monotonic()
            try:
                twist = self._provider()
            except Exception as exc:  # noqa: BLE001 — provider must never kill pump
                log_warning(f"[teleop_pump] provider raised: {exc}")
                twist = None

            if twist is None:
                # No user input this tick. If we've been idle past the deadman
                # window, publish zero (only once until input resumes) so the
                # robot doesn't keep marching on stale commands.
                if (now - last_input_ts) >= self._deadman_s:
                    if not deadman_warned:
                        log_warning(
                            f"[teleop_pump] deadman triggered "
                            f"(no input {self._deadman_s:.1f}s); pushing zero"
                        )
                        deadman_warned = True
                    self._safe_publish(zero_twist())
                continue

            # Live input — reset deadman bookkeeping and push the clamped value.
            last_input_ts = now
            deadman_warned = False
            try:
                clamped = self._safety_clamp(twist)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[teleop_pump] safety_clamp raised: {exc}; pushing zero")
                clamped = zero_twist()
            self._safe_publish(clamped)

    def _safe_publish(self, twist: dict) -> None:
        try:
            self._sink(twist)
        except Exception as exc:  # noqa: BLE001 — sink errors must not stop the pump
            log_warning(f"[teleop_pump] sink raised: {exc}")


__all__ = ["TeleopPump", "zero_twist"]
