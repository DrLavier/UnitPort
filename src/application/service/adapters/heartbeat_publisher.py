"""HeartbeatPublisher — 2 Hz /unitport/heartbeat for robot-side estop watchdog.

The UnitPort robot-side stack (deployed via Phase 4 bootstrap, but already
present on the user's pupper) runs an estop watchdog systemd service that
listens for ``/unitport/heartbeat`` (``unitport_msgs/Heartbeat``). Missing
heartbeats for >N seconds trigger a safe-stop. The host MUST publish a
heartbeat for the duration of a session — otherwise the robot will
spontaneously go limp after the watchdog timeout.

Published at 2 Hz (DEMO convention) via the active :class:`NativeDDSBridge`.
The Heartbeat IDL is already in the core registry (Phase 2 ported it).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from unitport_sdk import log_warning


class HeartbeatPublisher:
    """Background thread publishing ``/unitport/heartbeat`` at a fixed rate."""

    TOPIC = "/unitport/heartbeat"
    MSG_TYPE = "unitport_msgs/msg/Heartbeat"
    DEFAULT_RATE_HZ = 2.0

    def __init__(
        self,
        bridge: Any,                          # NativeDDSBridge — typed Any to avoid cyc import
        rate_hz: float = DEFAULT_RATE_HZ,
        robot_state: str = "host_alive",
        active_bundle_name: str = "",
    ) -> None:
        self._bridge = bridge
        self._period_s = 1.0 / max(0.1, float(rate_hz))
        self._robot_state = str(robot_state)
        self._active_bundle = str(active_bundle_name)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="HeartbeatPublisher", daemon=True,
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
        while not self._stop_evt.wait(self._period_s):
            try:
                self._publish_one()
            except Exception as exc:  # noqa: BLE001 — never let the loop die
                log_warning(f"[heartbeat] publish failed: {exc}")

    def _publish_one(self) -> None:
        if self._bridge is None or not self._bridge.is_alive():
            return
        # The Heartbeat IDL is hand-written; field shapes match
        # application/service/runtime/ros2/idl_messages/unitport_msgs.py
        now_s = time.time()
        sec = int(now_s)
        nanosec = int((now_s - sec) * 1e9)
        payload = {
            "timestamp_monotonic": {"sec": sec, "nanosec": nanosec},
            "robot_state": self._robot_state,
            "active_bundle_name": self._active_bundle,
            "inference_rate_measured_hz": 0.0,
            "systemd_unit_state": "",
        }
        self._bridge.publish(
            self.TOPIC, self.MSG_TYPE, payload, qos="reliable",
        )


__all__ = ["HeartbeatPublisher"]
