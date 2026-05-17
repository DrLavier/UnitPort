"""application.service.telemetry — 事件总线 + 数据契约 + 主线程 marshaller（单文件）.

合并自原 / Merged from former application/service/telemetry/ subdir:
- event_bus.py                EventBus（process-singleton pub/sub）
- channels.py                 数据契约 dataclasses + 频道常量
- main_thread_marshaller.py   跨线程 → 主线程 dispatch（PyQt6）

来源 / Source: DEMO/src/system/telemetry/

频道清单（与 ``data_contract.required_channels`` 对齐）:
    heartbeat / joint_states_mirror / ros2_log_stream / diagnostics_stream /
    topic_discovery / shell_session / bundle_ops / estop / deployment_progress
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal


# =============================================================================
# Section 1 — Event bus（来自 event_bus.py）
# =============================================================================

class EventBus:
    """Process-singleton in-process event bus.

    Every ``EventBus()`` call returns the same instance, so a publisher and a
    subscriber in different modules share one subscriber table. Thread-safe:
    subscribe / unsubscribe / publish all take a short critical section;
    callbacks themselves run outside the lock so a slow handler can't block
    other publishers.
    """

    _instance: "EventBus | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._subscribers = {}
                    inst._lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    _subscribers: Dict[str, List[Callable]]
    _lock: threading.Lock

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            subs = self._subscribers.setdefault(event_type, [])
            if callback not in subs:
                subs.append(callback)

    def publish(self, event_type: str, payload: Any = None) -> None:
        """Snapshot subscribers under lock, dispatch outside lock."""
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, []))
        for callback in callbacks:
            try:
                callback(payload)
            except Exception:
                pass

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        with self._lock:
            subs = self._subscribers.get(event_type)
            if not subs:
                return
            self._subscribers[event_type] = [fn for fn in subs if fn is not callback]

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Drop all subscribers — for unit tests only."""
        with cls._instance_lock:
            if cls._instance is not None:
                with cls._instance._lock:
                    cls._instance._subscribers.clear()


# =============================================================================
# Section 2 — Channel constants + dataclass payloads（来自 channels.py）
# =============================================================================

CHANNEL_HEARTBEAT = "heartbeat"
CHANNEL_JOINT_STATES_MIRROR = "joint_states_mirror"
CHANNEL_ROS2_LOG_STREAM = "ros2_log_stream"
CHANNEL_DIAGNOSTICS_STREAM = "diagnostics_stream"
CHANNEL_TOPIC_DISCOVERY = "topic_discovery"
CHANNEL_SHELL_SESSION = "shell_session"
CHANNEL_BUNDLE_OPS = "bundle_ops"
CHANNEL_ESTOP = "estop"
CHANNEL_DEPLOYMENT_PROGRESS = "deployment_progress"


@dataclass
class Heartbeat:
    """robot_to_console · >= 1 Hz, up to 10 Hz.

    Drives the status-strip health dot and the telemetry-rail
    ``heartbeat`` card. The UI classifies ``healthy / degraded /
    disconnected`` from ``rtt_ms`` and ``missed_beats`` against the
    thresholds in the deploy YAML's ``health_thresholds`` block.
    """

    timestamp_monotonic: float
    robot_state: str                    # "idle" | "running" | "faulted"
    active_bundle_name: str = ""
    inference_rate_measured_hz: float = 0.0
    systemd_unit_state: str = ""        # "active" | "inactive" | "failed" | "not-installed"
    rtt_ms: float = 0.0
    missed_beats: int = 0
    # Hardware-info card extension: surfaced in the mission_panel real-robot
    # card. Adapters that don't report these leave them None — the UI shows
    # "—" for missing fields instead of crashing.
    battery_pct: Optional[float] = None       # 0..100
    temperature_c: Optional[float] = None     # main-board temp °C
    uptime_s: Optional[int] = None            # seconds since power-on
    serial: Optional[str] = None              # robot serial number
    firmware_version: Optional[str] = None    # firmware string


@dataclass
class JointStatesMirrorFrame:
    """Pass-through of /joint_states with optional downsampling."""

    stamp_ns: int
    names: list[str] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)


@dataclass
class Ros2LogLine:
    """Per-line /rosout bridged to the console."""

    level: str        # "DEBUG" | "INFO" | "WARN" | "ERROR" | "FATAL"
    node: str
    message: str
    stamp_ns: int = 0


@dataclass
class DiagnosticsEvent:
    """DiagnosticStatus message."""

    name: str
    level: int        # 0 OK, 1 WARN, 2 ERROR, 3 STALE
    message: str
    hardware_id: str = ""
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentProgressEvent:
    """Stepwise progress for first-time setup and redeploys.

    ``kind``:
      * "step_state" → ``step_id`` + ``state`` ("running" | "ok" | "fail" | "skip")
      * "log_line"   → appended to the right-hand log
      * "completed"  → show "Open console" CTA
      * "failed"     → show "Retry from failed step" CTA
    """

    kind: str
    step_id: str = ""
    state: str = ""
    line: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EStopSignal:
    """One-way console → robot with ack. Delivery is redundant over every transport."""

    source: str              # "ui" | "shortcut" | "external"
    reason: str = ""
    ack: bool = False


# =============================================================================
# Section 3 — Main-thread marshaller（来自 main_thread_marshaller.py）
# =============================================================================

class _Marshaller(QObject):
    """QObject that owns a queued signal → slot pair on the GUI thread."""

    _run = pyqtSignal(object, object)

    def __init__(self) -> None:
        super().__init__()
        # QueuedConnection forces the slot to execute on the thread the
        # QObject was created on — i.e. the GUI thread.
        self._run.connect(self._invoke, Qt.ConnectionType.QueuedConnection)

    def _invoke(self, cb: Callable[[Any], None], payload: Any) -> None:
        try:
            cb(payload)
        except Exception:
            # A failing handler must not crash the event dispatcher.
            pass

    def dispatch(self, cb: Callable[[Any], None], payload: Any) -> None:
        self._run.emit(cb, payload)


_singleton: Optional[_Marshaller] = None
_marshaller_lock = threading.Lock()


def install_marshaller() -> _Marshaller:
    """Create (or return) the process-singleton marshaller.

    MUST be called on the GUI thread so the queued connection resolves
    there. Call right after ``QApplication(sys.argv)``.
    """
    global _singleton
    with _marshaller_lock:
        if _singleton is None:
            _singleton = _Marshaller()
        return _singleton


def post_to_main(cb: Callable[[Any], None], payload: Any = None) -> None:
    """Invoke ``cb(payload)`` on the GUI thread.

    If the marshaller hasn't been installed yet (e.g. headless tests),
    falls back to an immediate synchronous call.
    """
    if _singleton is None:
        try:
            cb(payload)
        except Exception:
            pass
        return
    _singleton.dispatch(cb, payload)


__all__ = [
    "EventBus",
    # channels
    "CHANNEL_BUNDLE_OPS",
    "CHANNEL_DEPLOYMENT_PROGRESS",
    "CHANNEL_DIAGNOSTICS_STREAM",
    "CHANNEL_ESTOP",
    "CHANNEL_HEARTBEAT",
    "CHANNEL_JOINT_STATES_MIRROR",
    "CHANNEL_ROS2_LOG_STREAM",
    "CHANNEL_SHELL_SESSION",
    "CHANNEL_TOPIC_DISCOVERY",
    "DeploymentProgressEvent",
    "DiagnosticsEvent",
    "EStopSignal",
    "Heartbeat",
    "JointStatesMirrorFrame",
    "Ros2LogLine",
    # marshaller
    "install_marshaller",
    "post_to_main",
]
