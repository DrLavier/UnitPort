# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Go2WebRTCBridge — synchronous asyncio↔Qt facade over RoboVerse go2_webrtc.

Sibling of :class:`application.service.runtime.ros2.native_dds_bridge.NativeDDSBridge`.
Wraps :class:`go2_webrtc.Go2Connection` (RoboVerse community library,
cloned into ``custom_mods/runtime/sdk_extensions/Unitree/go2_webrtc/`` by
:class:`SdkManager`) and exposes a thread-safe synchronous API so the
rest of RELEASE — Tasks running on QThreadPool workers, widgets on the
Qt main thread — can use it without touching asyncio.

The underlying ``Go2Connection`` is asyncio-only and aiortc requires its
RTCPeerConnection to be constructed inside the event loop that drives
it. To satisfy that without forcing every caller into ``async``, the
bridge owns a daemon thread that runs a private ``asyncio`` event loop;
public methods schedule coroutines onto that loop via
``asyncio.run_coroutine_threadsafe`` and block on the future.

Public API shape intentionally mirrors :class:`NativeDDSBridge`
(``start/stop/is_alive/publish/subscribe/latest/rate_hz``) so downstream
consumers (Connect Task, telemetry sparkline) can be transport-agnostic
once both bridges are available.

Threading contract:

* Constructor: any thread. Does not start the loop.
* :meth:`start`: any thread. Spins up the loop thread, constructs
  ``Go2Connection`` inside the loop, awaits ``connect_robot``. Blocks
  until SDP negotiation finishes or ``timeout`` elapses.
* :meth:`wait_open` / :meth:`wait_validated`: any thread. Wait on the
  matching threading.Event set by aiortc callbacks.
* :meth:`publish` / :meth:`subscribe` / :meth:`stop`: any thread.
  Schedule coroutines onto the loop thread.
* Inbound ``on_message`` callbacks run on the aiortc thread. The bridge
  routes them through a lock-protected last-sample buffer and calls
  user-registered subscriber callbacks inline (callers must keep them
  short and non-blocking — for UI work, emit a queued-connection
  :class:`pyqtSignal`).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from unitport_sdk import log_info, log_warning


_INBOUND_TELEMETRY_THROTTLE_S = 0.2


class Go2WebRTCBridge:
    """In-process WebRTC client for Unitree Go2 sport API.

    One bridge instance manages one peer connection. Re-using an instance
    across reconnects is supported: call :meth:`stop` then :meth:`start`
    again with the new address. The bridge is NOT thread-shared — each
    connection should own its own bridge.
    """

    def __init__(self) -> None:
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_ready = threading.Event()
        self._conn: Any = None              # go2_webrtc.Go2Connection (typed Any to defer import)
        self._closed: bool = True

        # Lifecycle gates surfaced to the Connect Task.
        self._open_event = threading.Event()
        self._validated_event = threading.Event()

        # Inbound buffers.
        self._lock = threading.RLock()
        self._last_msg: Dict[str, Any] = {}
        self._last_ts: Dict[str, float] = {}
        self._timestamps: Dict[str, Deque[float]] = {}
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}
        self._last_inbound_emit_s: Dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle

    def start(self, ip: str, token: str = "", timeout: float = 8.0) -> None:
        """Connect to the robot at ``ip``. Blocks until SDP negotiation finishes.

        Raises ``RuntimeError`` on negotiation failure or timeout. Calling
        :meth:`start` on an already-started bridge raises ``RuntimeError``.
        """
        if not self._closed:
            raise RuntimeError("Go2WebRTCBridge already started; call stop() first")
        if not ip:
            raise ValueError("Go2WebRTCBridge.start requires a non-empty ip")

        self._open_event.clear()
        self._validated_event.clear()

        # Spin up the dedicated event-loop thread.
        self._loop_ready.clear()
        self._loop_thread = threading.Thread(
            target=self._loop_main,
            name="Go2WebRTCBridge-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=2.0):
            raise RuntimeError("Go2WebRTCBridge: event-loop thread failed to start")

        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._async_start(ip, token), self._loop
        )
        try:
            future.result(timeout=timeout)
        except Exception as exc:
            # Negotiation failed. Close the partially-initialised peer
            # (Go2Connection.__init__ already built pc + data channel
            # before connect_robot ran, so aiortc/aioice tasks are
            # gathering ICE candidates in the background) before
            # tearing down the loop, otherwise those tasks get reported
            # as "destroyed while pending" at interpreter shutdown.
            self._tear_down_loop()
            self._conn = None
            raise RuntimeError(f"WebRTC negotiation failed: {exc}") from exc

        self._closed = False
        log_info(f"[webrtc] connected to {ip}")

    def stop(self) -> None:
        """Tear down the peer connection and the loop thread. Idempotent."""
        if self._closed and self._loop_thread is None:
            return

        # _tear_down_loop closes self._conn.pc and drains pending ICE /
        # data-channel tasks before stopping the event loop, so no
        # separate _async_stop dispatch is required here.
        self._tear_down_loop()
        self._conn = None
        self._closed = True
        self._open_event.clear()
        self._validated_event.clear()
        with self._lock:
            self._last_msg.clear()
            self._last_ts.clear()
            self._timestamps.clear()
            self._subs.clear()
            self._last_inbound_emit_s.clear()

    def is_alive(self) -> bool:
        return not self._closed and self._conn is not None

    def wait_open(self, timeout: float = 10.0) -> bool:
        """Block until the data channel signals ``on_open``."""
        return self._open_event.wait(timeout=timeout)

    def wait_validated(self, timeout: float = 5.0) -> bool:
        """Block until the robot accepts our validation challenge response."""
        return self._validated_event.wait(timeout=timeout)

    # ------------------------------------------------------------------ IO

    def publish(self, topic: str, data: Any, msg_type: str = "msg") -> bool:
        """Send a JSON envelope ``{type, topic, data}`` over the data channel.

        Mirrors :meth:`go2_webrtc.Go2Connection.publish`. Returns False
        when the bridge is not alive or the underlying send raised.
        """
        if not self.is_alive():
            return False
        assert self._loop is not None
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_publish(topic, data, msg_type), self._loop
            )
            future.result(timeout=0.5)
        except Exception as exc:
            log_warning(f"[webrtc] publish failed on {topic!r}: {exc}")
            return False

        payload_for_log = data if isinstance(data, dict) else {"value": data}
        _emit_bridge_traffic("out", topic, msg_type, payload_for_log, rate_hz=0.0)
        return True

    def subscribe(self, topic: str, cb: Callable[[Any], None]) -> bool:
        """Register a callback and request the topic from the robot.

        The Go2 firmware streams a topic only after the client sends a
        ``{"type":"subscribe","topic":<topic>}`` envelope. We register
        ``cb`` first, then forward the request; if the request fails the
        callback is removed.
        """
        if not topic:
            return False
        with self._lock:
            self._subs.setdefault(topic, []).append(cb)
        ok = self.publish(topic, "", "subscribe")
        if not ok:
            with self._lock:
                callbacks = self._subs.get(topic)
                if callbacks and cb in callbacks:
                    callbacks.remove(cb)
                    if not callbacks:
                        self._subs.pop(topic, None)
        return ok

    def latest(self, topic: str) -> Optional[Any]:
        """Return the last payload seen on ``topic`` or None."""
        with self._lock:
            return self._last_msg.get(topic)

    def rate_hz(self, topic: str) -> float:
        """Estimate the inbound rate on ``topic`` from the last few samples."""
        with self._lock:
            stamps = self._timestamps.get(topic)
            if stamps is None or len(stamps) < 2:
                return 0.0
            span = stamps[-1] - stamps[0]
            if span <= 0.0:
                return 0.0
            return float(len(stamps) - 1) / float(span)

    # ============================================================== internals

    def _loop_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _tear_down_loop(self) -> None:
        loop = self._loop
        thread = self._loop_thread
        if loop is not None and loop.is_running():
            try:
                # Drain pending coroutines (aiortc / aioice ICE gathering,
                # peer close, etc.) before stopping the loop. Without this
                # step Python emits "Task was destroyed but it is pending"
                # warnings from the asyncio shutdown path whenever start()
                # failed mid-negotiation.
                fut = asyncio.run_coroutine_threadsafe(
                    self._async_cleanup(), loop
                )
                fut.result(timeout=3.0)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=2.0)
        self._loop = None
        self._loop_thread = None
        self._loop_ready.clear()

    async def _async_cleanup(self) -> None:
        """Close the peer (if any) and cancel every other pending task."""
        if self._conn is not None:
            try:
                await self._conn.pc.close()
            except Exception:
                pass
        current = asyncio.current_task()
        pending = [
            t for t in asyncio.all_tasks()
            if t is not current and not t.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _async_start(self, ip: str, token: str) -> None:
        # Deferred import: go2_webrtc is installed by SdkManager into the
        # venv site-packages. Importing at module top would force every
        # consumer of this file to have the SDK installed, even when they
        # are only inspecting type hints.
        from go2_webrtc import Go2Connection  # type: ignore

        def _on_open() -> None:
            self._open_event.set()

        def _on_validated() -> None:
            self._validated_event.set()

        def _on_message(raw_message: Any, msgobj: Any) -> None:
            self._handle_inbound(raw_message, msgobj)

        self._conn = Go2Connection(
            ip=ip,
            token=token,
            on_open=_on_open,
            on_validated=_on_validated,
            on_message=_on_message,
        )
        await self._conn.connect_robot()

    async def _async_publish(self, topic: str, data: Any, msg_type: str) -> None:
        conn = self._conn
        if conn is None:
            raise RuntimeError("publish before start")
        conn.publish(topic, data, msg_type)

    def _handle_inbound(self, raw_message: Any, msgobj: Any) -> None:
        if not isinstance(msgobj, dict):
            return
        topic = msgobj.get("topic") or ""
        if not topic:
            return
        msg_type = str(msgobj.get("type") or "msg")
        data = msgobj.get("data")
        now = time.monotonic()

        with self._lock:
            self._last_msg[topic] = data
            self._last_ts[topic] = now
            stamps = self._timestamps.setdefault(topic, deque(maxlen=10))
            stamps.append(now)
            callbacks = list(self._subs.get(topic, ()))
            last_emit = self._last_inbound_emit_s.get(topic, 0.0)
            should_emit = (now - last_emit) >= _INBOUND_TELEMETRY_THROTTLE_S
            if should_emit:
                self._last_inbound_emit_s[topic] = now
            rate = float(len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) >= 2 and (stamps[-1] - stamps[0]) > 0 else 0.0

        for cb in callbacks:
            try:
                cb(data)
            except Exception as exc:
                log_warning(f"[webrtc] subscriber on {topic!r} raised: {exc}")

        if should_emit:
            payload_for_log = data if isinstance(data, dict) else {"value": data}
            _emit_bridge_traffic("in", topic, msg_type, payload_for_log, rate_hz=rate)


# ------------------------------------------------------------------- telemetry


def _emit_bridge_traffic(
    direction: str,
    topic: str,
    msg_type: str,
    payload: Any,
    rate_hz: float = 0.0,
) -> None:
    """Forward a bridge sample to ``AppSignals.connection_telemetry``.

    Mirrors :func:`application.service.runtime.ros2.native_dds_bridge._emit_bridge_traffic`
    so downstream UI consumers can treat WebRTC and DDS traffic
    uniformly. Lazy-imports AppSignals and swallows every failure — this
    is a UI-only telemetry pipeline and must never affect publish /
    subscribe correctness.
    """
    try:
        from application.service.signals import get_app_signals
        get_app_signals().connection_telemetry.emit(
            {
                "direction": direction,
                "topic": topic,
                "msg_type": msg_type,
                "ts": time.time(),
                "rate_hz": float(rate_hz),
                "summary": _summarise_payload(topic, payload),
                "last_payload": payload if isinstance(payload, dict) else {},
            }
        )
    except Exception:
        pass


def _summarise_payload(topic: str, payload: Any) -> str:
    """Compress a payload dict into a one-line readable string.

    Recognises a few interactive Go2 topics so the UI cmd_log stays
    skimmable; falls back to ``{keys}`` for everything else.
    """
    if not isinstance(payload, dict):
        return str(payload)[:120]
    try:
        if topic == "rt/api/sport/request":
            header = payload.get("header") or {}
            ident = header.get("identity") or {}
            api_id = ident.get("api_id")
            param = payload.get("parameter")
            return f"sport api_id={api_id} param={param!s}"[:120]
        if topic == "rt/lf/lowstate":
            imu = payload.get("imu_state") or {}
            return f"battery={payload.get('bms')} imu={imu!s}"[:120]
        if topic.endswith("/sportmodestate"):
            return f"mode={payload.get('mode')} gait={payload.get('gait_type')}"[:120]
    except Exception:
        pass
    if isinstance(payload, dict):
        return "{" + ", ".join(sorted(payload.keys())[:8]) + "}"
    return str(payload)[:120]


__all__ = ["Go2WebRTCBridge"]
