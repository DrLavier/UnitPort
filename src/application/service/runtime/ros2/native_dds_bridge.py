# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""NativeDDSBridge — in-process cyclonedds-python bridge for ROS2-generic brands.

Phase 2 port from DEMO ``src/system/runtime/ros2/native_dds_bridge.py``.

Speaks DDS directly over the network to the robot's ROS2 graph; no subprocess,
no docker. Topic naming: ROS2 ``/cmd_vel`` maps to DDS ``rt/cmd_vel`` (ROS2-on-DDS
convention; see rmw_cyclonedds_cpp). Type naming:
``geometry_msgs/msg/Twist`` resolves via :mod:`idl_registry` to an IdlStruct
dataclass with typename ``geometry_msgs::msg::dds_::Twist_`` — the exact wire
name rmw_cyclonedds_cpp uses, so native↔real-ROS2 interop works.

RELEASE-specific changes vs DEMO:

- Telemetry: ``_emit_bridge_traffic()`` publishes to
  :attr:`application.service.signals.AppSignals.connection_telemetry`
  (Phase 6 sparkline consumer) instead of DEMO's EventBus.
- Brand registry resolution removed — Phase 5 will reintroduce brand-specific
  IDL packages; Phase 2 is core-only.
- Recording (start_recording / stop_recording) raises NotImplementedError —
  the mcap-backed recorder is deferred to Phase 6 telemetry work.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import threading
import time
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from application.service.runtime.ros2.bridge_metrics import Subscription
from application.service.runtime.ros2.bridge_protocol import BridgeProtocol
from application.service.runtime.ros2 import idl_registry, qos_profiles
from application.service.runtime.ros2.discovery import (
    DDSDiscoveryView,
    ParticipantInfo,
    PublicationInfo,
    SubscriptionInfo,
)
from application.service.runtime.ros2.native_dds_errors import (
    CycloneDDSLoadFailed,
    DomainUnreachable,
    NativeDDSBridgeError,
    WriteFailed,
)


# Lazy-imported once per process so tests can mock cyclonedds before import.
_dds: Any = None


def _import_cyclonedds() -> Any:
    """Import cyclonedds on demand, surfacing a structured error on failure."""
    global _dds
    if _dds is not None:
        return _dds
    try:
        from cyclonedds import core as cyc_core  # noqa: F401
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter, Publisher
        from cyclonedds.sub import DataReader, Subscriber
        from cyclonedds.topic import Topic
        from cyclonedds.core import Listener, Policy, Qos
        from cyclonedds.util import duration
    except ImportError as exc:
        raise CycloneDDSLoadFailed.from_import_error(exc) from exc

    class _Namespace:
        pass

    mod = _Namespace()
    mod.DomainParticipant = DomainParticipant
    mod.DataWriter = DataWriter
    mod.Publisher = Publisher
    mod.DataReader = DataReader
    mod.Subscriber = Subscriber
    mod.Topic = Topic
    mod.Listener = Listener
    mod.Policy = Policy
    mod.Qos = Qos
    mod.duration = duration
    _dds = mod
    return _dds


# --------------------------------------------------------------------- config

def _compose_discovery_uri(
    peer_ip: Optional[str],
    host_address: Optional[str] = None,
) -> Optional[str]:
    """Build a minimal CYCLONEDDS_URI that targets ``peer_ip``.

    Only emitted when a peer IP is needed (USB tether, no-multicast network).
    Default discovery (multicast 239.255.0.1) works on most LANs without this.

    When ``host_address`` is given, also emits a ``<General><Interfaces>`` block
    pinning outbound traffic to that local NIC by IP. Required on multi-NIC
    hosts (Windows with both Wi-Fi and a USB-ether NIC active): without the
    binding, cyclonedds follows OS default routing — discovery unicast to the
    named peer succeeds (so GUIDs become visible across hosts), but data
    traffic egresses via the wrong NIC and never reaches the peer.
    """
    if not peer_ip:
        return None
    interfaces_block = ""
    if host_address:
        interfaces_block = (
            "<General>"
            "<Interfaces>"
            f"<NetworkInterface address='{host_address}'/>"
            "</Interfaces>"
            "</General>"
        )
    return (
        "<CycloneDDS>"
        "<Domain>"
        f"{interfaces_block}"
        "<Discovery>"
        "<Peers>"
        f"<Peer address='{peer_ip}'/>"
        "</Peers>"
        "<ParticipantIndex>auto</ParticipantIndex>"
        "</Discovery>"
        "</Domain>"
        "</CycloneDDS>"
    )


def _pick_host_address_for_peer(peer_ip: Optional[str]) -> Optional[str]:
    """Return the local IPv4 in the same /24 as ``peer_ip`` if any.

    Returns ``None`` when ``peer_ip`` is empty or malformed, ``psutil`` is
    unavailable, or no interface on this host shares the peer's /24. In those
    cases, callers fall back to OS-default routing (the historical behaviour) —
    connect doesn't fail; it just risks the multi-NIC data plane bug.

    /24 is the right scope: the USB tether always uses 192.168.55.0/24, and
    Wi-Fi peers live in a single /24 99% of the time.
    """
    if not peer_ip:
        return None
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        peer_octets = [int(x) for x in peer_ip.split(".")]
    except ValueError:
        return None
    if len(peer_octets) != 4:
        return None
    target_prefix = tuple(peer_octets[:3])
    try:
        addrs_by_iface = psutil.net_if_addrs()
    except Exception:
        return None
    for _name, addrs in addrs_by_iface.items():
        for a in addrs:
            ip = getattr(a, "address", "") or ""
            try:
                octets = [int(x) for x in ip.split(".")]
            except ValueError:
                continue
            if (
                len(octets) == 4
                and tuple(octets[:3]) == target_prefix
                and ip != peer_ip
            ):
                return ip
    return None


def _list_network_interfaces() -> List[str]:
    """Enumerate non-loopback interface names for diagnostics."""
    try:
        import socket as _s

        if hasattr(_s, "if_nameindex"):
            return [name for _idx, name in _s.if_nameindex() if name != "lo"]
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        ips = socket.gethostbyname_ex(hostname)[2]
        return [ip for ip in ips if not ip.startswith("127.")]
    except Exception:
        return []


# --------------------------------------------------------------------- QoS

def _build_qos(qos_dict: Dict[str, Any]) -> Any:
    """Translate a 6-key QoS descriptor dict into ``cyclonedds.core.Qos``."""
    dds = _import_cyclonedds()
    Policy = dds.Policy
    duration = dds.duration

    policies: List[Any] = []

    reliability = qos_dict.get("reliability", "best_effort")
    if reliability == "reliable":
        policies.append(
            Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=100))
        )
    else:
        policies.append(Policy.Reliability.BestEffort)

    durability = qos_dict.get("durability", "volatile")
    if durability == "transient_local":
        policies.append(Policy.Durability.TransientLocal)
    else:
        policies.append(Policy.Durability.Volatile)

    history = qos_dict.get("history", "keep_last")
    depth = int(qos_dict.get("depth", 5))
    if history == "keep_all":
        policies.append(Policy.History.KeepAll)
    else:
        policies.append(Policy.History.KeepLast(max(1, depth)))

    deadline = qos_dict.get("deadline")
    if deadline is not None and float(deadline) > 0:
        policies.append(Policy.Deadline(duration(seconds=float(deadline))))

    liveliness = qos_dict.get("liveliness", "automatic")
    lease = duration(seconds=5)
    if liveliness == "manual_by_topic":
        policies.append(Policy.Liveliness.ManualByTopic(lease_duration=lease))
    else:
        policies.append(Policy.Liveliness.Automatic(lease_duration=lease))

    return dds.Qos(*policies)


# --------------------------------------------------------------------- helpers

def _ros_topic_to_dds(topic: str) -> str:
    """``/cmd_vel`` → ``rt/cmd_vel`` per ROS2-on-DDS convention."""
    if not topic:
        raise ValueError("topic MUST be non-empty")
    stripped = topic.lstrip("/")
    return f"rt/{stripped}"


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an IdlStruct dataclass instance to a plain dict recursively."""
    if is_dataclass(obj) and not isinstance(obj, type):
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            out[f.name] = _dataclass_to_dict(v)
        return out
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(v) for v in obj)
    return obj


def _dict_to_dataclass(cls: Type[Any], payload: Dict[str, Any]) -> Any:
    """Instantiate an IdlStruct dataclass from a plain dict."""
    if not is_dataclass(cls):
        return payload
    kwargs: Dict[str, Any] = {}
    hints = getattr(cls, "__idl_type_annotations__", None) or {
        f.name: f.type for f in dataclasses.fields(cls)
    }
    for f in dataclasses.fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        field_type = hints.get(f.name, f.type)
        kwargs[f.name] = _coerce_value(field_type, value)
    return cls(**kwargs)


def _coerce_value(field_type: Any, value: Any) -> Any:
    if isinstance(value, dict) and is_dataclass(field_type):
        return _dict_to_dataclass(field_type, value)
    if isinstance(value, list):
        inner = _inner_type(field_type)
        if inner is not None and is_dataclass(inner):
            return [_dict_to_dataclass(inner, v) if isinstance(v, dict) else v for v in value]
        return list(value)
    return value


def _inner_type(t: Any) -> Optional[Any]:
    """Extract the element type from sequence[X] / List[X] annotations."""
    args = getattr(t, "__args__", None)
    if args:
        return args[0]
    return None


# --------------------------------------------------------------------- UI surface


def _summarise_payload(topic: str, payload: Any) -> str:
    """Compress a ROS2 payload dict into a one-line readable string.

    Keeps the common interactive topics readable; falls back to ``{keys}`` so
    the user still sees shape without a 40-line dump.
    """
    if not isinstance(payload, dict):
        return str(payload)[:120]
    try:
        if topic.endswith("/cmd_vel") or topic == "/cmd_vel":
            lin = payload.get("linear") or {}
            ang = payload.get("angular") or {}
            return (
                f"vx={float(lin.get('x', 0.0)):+.3f} "
                f"vy={float(lin.get('y', 0.0)):+.3f} "
                f"vyaw={float(ang.get('z', 0.0)):+.3f}"
            )
        if topic.endswith("/joint_states"):
            names = payload.get("name") or []
            pos = payload.get("position") or []
            return f"joints={len(names)} pos[0..2]={[round(float(p), 3) for p in pos[:3]]}"
        if topic.endswith("/imu/data") or topic.endswith("/imu"):
            return (
                f"quat=({float(payload.get('orientation_x', 0.0)):+.3f},"
                f"{float(payload.get('orientation_y', 0.0)):+.3f},"
                f"{float(payload.get('orientation_z', 0.0)):+.3f},"
                f"{float(payload.get('orientation_w', 1.0)):+.3f})"
            )
        if topic.endswith("/battery_state"):
            pct = payload.get("percentage")
            volt = payload.get("voltage")
            return f"pct={pct} volt={volt}"
    except Exception:
        pass
    return "{" + ", ".join(sorted(payload.keys())[:8]) + "}"


def _emit_bridge_traffic(
    direction: str,
    topic: str,
    msg_type: str,
    payload: Any,
    rate_hz: float = 0.0,
) -> None:
    """Forward a bridge sample to ``AppSignals.connection_telemetry``.

    Phase 6's sparkline panel will consume these. Swallows every failure —
    this is a UI-only telemetry pipeline and must never affect publish/subscribe
    correctness. Import is lazy so a unit test that exercises the bridge in
    isolation (no QApplication) does not crash.
    """
    try:
        from application.service.signals import get_app_signals
        get_app_signals().connection_telemetry.emit(
            {
                "direction": direction,        # "out" | "in"
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


# --------------------------------------------------------------------- bridge

class NativeDDSBridge(BridgeProtocol):
    """In-process DDS participant + DataReader/DataWriter orchestration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self._domain_id: int = int(cfg.get("domain_id", cfg.get("ros_domain_id", 0)))
        self._brand_id: str = str(cfg.get("brand_id", ""))
        self._robot_hostname: str = str(cfg.get("robot_hostname", ""))
        self._namespace: str = str(cfg.get("namespace", ""))
        self._start_timeout: float = float(cfg.get("start_timeout", 2.0))

        # Discovery peer (USB tether or explicit user override).
        transport = str(cfg.get("transport", "")).lower()
        peer = cfg.get("discovery_peer")
        if not peer:
            if transport == "usb":
                peer = cfg.get("usb_ip")
            else:
                peer = cfg.get("pupper_ip") if transport == "wifi" else None
        self._peer_ip: Optional[str] = peer or None

        # Local NIC binding. When set, cyclonedds emits a
        # <General><Interfaces> block pinning data-plane traffic to this local
        # IP. Auto-detected from the peer's /24 by ``bridge_bringup``; ``None``
        # falls back to OS routing.
        self._host_address: Optional[str] = cfg.get("host_address") or None

        # Internal state — all created in start().
        self._participant: Any = None
        self._publisher: Any = None
        self._subscriber: Any = None
        self._topics: Dict[str, Any] = {}      # dds topic name → Topic
        self._writers: Dict[str, Any] = {}     # "/topic" → DataWriter
        self._readers: Dict[str, Any] = {}     # "/topic" → DataReader
        self._subs: Dict[str, Subscription] = {}  # "/topic" → Subscription
        self._subs_lock = threading.RLock()
        self._closed = True
        self._on_error_cb: Optional[Callable[[str], None]] = None
        self._prev_cyclonedds_uri: Optional[str] = None
        self._uri_override_applied = False
        # Wire-level discovery view — built lazily once the participant is up.
        # Used by Phase 3.6 probes (foreign publishers on cmd_vel,
        # publication_matched_count, etc.). None until start() runs.
        self._discovery: Optional[DDSDiscoveryView] = None
        # Per-topic last-emit timestamp for inbound telemetry throttling. High
        # rate topics like /joint_states would flood the consumer otherwise
        # (100+ Hz). Outbound is left un-throttled so every /cmd_vel tick is
        # visible to the UI.
        self._last_inbound_emit_s: Dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Open the DDS participant. Raises NativeDDSBridgeError on failure."""
        if not self._closed:
            return

        dds = _import_cyclonedds()

        # Apply CYCLONEDDS_URI override BEFORE the participant is constructed —
        # cyclonedds reads env only at participant creation.
        uri = _compose_discovery_uri(self._peer_ip, self._host_address)
        if uri:
            self._prev_cyclonedds_uri = os.environ.get("CYCLONEDDS_URI")
            os.environ["CYCLONEDDS_URI"] = uri
            self._uri_override_applied = True

        deadline = time.monotonic() + self._start_timeout
        try:
            self._participant = dds.DomainParticipant(domain_id=self._domain_id)
        except Exception as exc:  # pragma: no cover — hard init failure
            self._restore_uri()
            raise DomainUnreachable.make(
                self._domain_id,
                self._robot_hostname,
                self._start_timeout,
                _list_network_interfaces(),
            ) from exc

        # Time-budget guard: DomainParticipant construction MUST finish within
        # start_timeout. If the cyclonedds C-side hangs we surface a clear
        # error instead of blocking the UI thread indefinitely.
        if time.monotonic() > deadline:
            self.stop()
            raise DomainUnreachable.make(
                self._domain_id,
                self._robot_hostname,
                self._start_timeout,
                _list_network_interfaces(),
            )

        self._publisher = dds.Publisher(self._participant)
        self._subscriber = dds.Subscriber(self._participant)
        self._discovery = DDSDiscoveryView(self._participant)
        self._closed = False

    def stop(self, timeout: float = 5.0) -> None:
        """Tear down readers, writers, publisher, subscriber, participant.

        Idempotent. The strict reverse-order teardown prevents cyclonedds from
        leaking DDS resources system-wide.
        """
        if self._closed and self._participant is None:
            return

        deadline = time.monotonic() + max(0.1, timeout)
        with self._subs_lock:
            readers = list(self._readers.values())
            self._readers.clear()
            writers = list(self._writers.values())
            self._writers.clear()
            topics = list(self._topics.values())
            self._topics.clear()
            self._subs.clear()

        for entity in readers + writers + topics:
            if entity is None:
                continue
            try:
                entity.close()  # cyclonedds Entity subclasses expose close()
            except Exception:
                pass
            if time.monotonic() > deadline:
                break

        if self._discovery is not None:
            try:
                self._discovery.close()
            except Exception:
                pass
            self._discovery = None

        for entity in (self._subscriber, self._publisher, self._participant):
            if entity is None:
                continue
            try:
                entity.close()
            except Exception:
                pass

        self._subscriber = None
        self._publisher = None
        self._participant = None
        self._closed = True
        self._restore_uri()

    def _restore_uri(self) -> None:
        if not self._uri_override_applied:
            return
        if self._prev_cyclonedds_uri is None:
            os.environ.pop("CYCLONEDDS_URI", None)
        else:
            os.environ["CYCLONEDDS_URI"] = self._prev_cyclonedds_uri
        self._uri_override_applied = False
        self._prev_cyclonedds_uri = None

    def is_alive(self) -> bool:
        return not self._closed and self._participant is not None

    def on_error(self, cb: Callable[[str], None]) -> None:
        self._on_error_cb = cb

    # ------------------------------------------------------------------ IO

    def subscribe(
        self,
        topic: str,
        msg_type: str,
        qos: Optional[Dict[str, Any]] = None,
        timeout: float = 3.0,
    ) -> bool:
        if self._closed:
            raise NativeDDSBridgeError("subscribe called before start()")
        qos_dict = self._normalised_qos(qos)
        cls = idl_registry.resolve(msg_type, brand_id=self._brand_id or None)
        dds_topic = _ros_topic_to_dds(topic)
        dds = _import_cyclonedds()
        topic_entity = self._topics.get(dds_topic)
        if topic_entity is None:
            topic_entity = dds.Topic(self._participant, dds_topic, cls)
            self._topics[dds_topic] = topic_entity

        with self._subs_lock:
            sub = Subscription(topic=topic, msg_type=msg_type, qos=dict(qos_dict))
            self._subs[topic] = sub

        def _on_data(reader: Any, _sub: Subscription = sub, _cls: Type[Any] = cls,
                     _topic: str = topic, _msg_type: str = msg_type) -> None:
            try:
                for sample in reader.take(N=16):
                    payload = _dataclass_to_dict(sample)
                    now = time.time()
                    with self._subs_lock:
                        _sub.update(payload, now)
                        rate = float(_sub.rate_hz)
                    # Throttled telemetry — at most ~5 Hz per topic, otherwise
                    # /joint_states at >100 Hz would flood AppSignals.
                    last = self._last_inbound_emit_s.get(_topic, 0.0)
                    if (now - last) >= 0.2:
                        self._last_inbound_emit_s[_topic] = now
                        _emit_bridge_traffic("in", _topic, _msg_type, payload, rate)
            except Exception as exc:  # pragma: no cover — listener errors
                self._emit_error(f"on_data_available error on {topic}: {exc}")

        listener = dds.Listener(on_data_available=_on_data)
        if hasattr(listener, "set_on_liveliness_lost"):
            listener.set_on_liveliness_lost(
                lambda r, s: self._emit_error(f"liveliness lost on {topic}")
            )
        if hasattr(listener, "set_on_requested_deadline_missed"):
            listener.set_on_requested_deadline_missed(
                lambda r, s: self._emit_error(f"deadline missed on {topic}")
            )
        if hasattr(listener, "set_on_sample_lost"):
            listener.set_on_sample_lost(
                lambda r, s: self._emit_error(f"sample lost on {topic}")
            )

        reader = dds.DataReader(
            self._subscriber,
            topic_entity,
            qos=_build_qos(qos_dict),
            listener=listener,
        )
        self._readers[topic] = reader
        return True

    def publish(
        self,
        topic: str,
        msg_type: str,
        payload: Dict[str, Any],
        qos: Optional[Dict[str, Any]] = None,
        timeout: float = 1.0,
        wait_ack: bool = False,
    ) -> bool:
        if wait_ack:
            raise NotImplementedError(
                "wait_ack=True is deferred — DDS has no synchronous ack; "
                "reliable QoS + publication_matched polling is the path forward"
            )
        if self._closed:
            raise NativeDDSBridgeError("publish called before start()")
        qos_dict = self._normalised_qos(qos)
        cls = idl_registry.resolve(msg_type, brand_id=self._brand_id or None)
        dds_topic = _ros_topic_to_dds(topic)
        dds = _import_cyclonedds()

        writer = self._writers.get(topic)
        if writer is None:
            topic_entity = self._topics.get(dds_topic)
            if topic_entity is None:
                topic_entity = dds.Topic(self._participant, dds_topic, cls)
                self._topics[dds_topic] = topic_entity
            writer = dds.DataWriter(
                self._publisher,
                topic_entity,
                qos=_build_qos(qos_dict),
            )
            self._writers[topic] = writer

        try:
            sample = _dict_to_dataclass(cls, dict(payload))
            writer.write(sample)
        except NativeDDSBridgeError:
            raise
        except Exception as exc:
            raise WriteFailed.make(topic, timeout, str(exc)) from exc
        # Surface the outbound frame so UI subscribers can show it with a
        # timestamp. Fire-and-forget — never blocks publish.
        _emit_bridge_traffic("out", topic, msg_type, payload)
        return True

    def ping(self, timeout: float = 2.0) -> bool:
        """Native has no peer-process to ping — mirrors is_alive()."""
        return self.is_alive()

    def latest(self, topic: str) -> Optional[Dict[str, Any]]:
        with self._subs_lock:
            sub = self._subs.get(topic)
            if sub is None or sub.last_payload is None:
                return None
            return dict(sub.last_payload)

    def rate_hz(self, topic: str) -> float:
        with self._subs_lock:
            sub = self._subs.get(topic)
            return float(sub.rate_hz) if sub is not None else 0.0

    def subscription_stats(self) -> Dict[str, Dict[str, float]]:
        with self._subs_lock:
            return {t: s.stats() for t, s in self._subs.items()}

    # ------------------------------------------------------------------ discovery

    def discovered_participants(self) -> List[ParticipantInfo]:
        """Every DDS participant currently visible in our domain."""
        if self._discovery is None:
            return []
        return self._discovery.participants()

    def discovered_publications(
        self, topic_filter: Optional[str] = None,
    ) -> List[PublicationInfo]:
        """Every DataWriter visible on the wire (optionally filtered to topic).

        ``topic_filter`` accepts either ROS form (``/cmd_vel``) or DDS form
        (``rt/cmd_vel``); both match the same set of publications.
        """
        if self._discovery is None:
            return []
        return self._discovery.publications(topic_filter=topic_filter)

    def discovered_subscriptions(
        self, topic_filter: Optional[str] = None,
    ) -> List[SubscriptionInfo]:
        """Every DataReader visible on the wire (optionally filtered to topic)."""
        if self._discovery is None:
            return []
        return self._discovery.subscriptions(topic_filter=topic_filter)

    def foreign_publishers(self, topic: str) -> List[PublicationInfo]:
        """Publishers on ``topic`` that are NOT our participant.

        The Phase 3.6 multi-publisher detection probe for cmd_vel / heartbeat
        / any other exclusive topic builds on this. Empty list = OK.
        """
        if self._discovery is None:
            return []
        return self._discovery.foreign_publications(topic)

    def foreign_subscribers(self, topic: str) -> List[SubscriptionInfo]:
        """Subscribers on ``topic`` that are NOT our participant.

        Symmetric counterpart to :meth:`foreign_publishers`. Used by
        :class:`MangdangChampSubscriberProbe` to ask "does the robot
        actually have a /cmd_vel reader?" — empty list = CHAMP not
        running on the wire even if systemd thinks the unit is active.
        """
        if self._discovery is None:
            return []
        return self._discovery.foreign_subscriptions(topic)

    def publication_matched_count(self, topic: str) -> Optional[int]:
        """How many remote DataReaders are currently matched to OUR writer.

        Reads the writer's ``publication_matched_status.current_count`` —
        cyclonedds maintains this transparently as discovery sees new readers
        come and go.

        Returns ``None`` when we have **no writer at all** for ``topic`` —
        i.e. nothing has called ``publish()`` on it yet. Probes use this to
        distinguish "writer exists but unmatched" (real failure) from
        "writer doesn't exist yet because user hasn't clicked Take Control"
        (informational only).
        """
        writer = self._writers.get(topic)
        if writer is None:
            return None
        try:
            status = writer.get_publication_matched_status()
        except Exception:
            return None
        return int(getattr(status, "current_count", 0) or 0)

    def subscription_matched_count(self, topic: str) -> Optional[int]:
        """Mirror of :meth:`publication_matched_count` for our reader on ``topic``.

        Returns ``None`` when we have no reader for the topic (subscribe()
        hasn't been called); 0 when a reader exists but no remote writer
        is matched; positive when matched.
        """
        reader = self._readers.get(topic)
        if reader is None:
            return None
        try:
            status = reader.get_subscription_matched_status()
        except Exception:
            return None
        return int(getattr(status, "current_count", 0) or 0)

    # ------------------------------------------------------------------ recording

    def start_recording(
        self, topics: List[str], out_dir: str, timeout: float = 5.0
    ) -> bool:
        """Recording deferred to Phase 6 (telemetry sparkline + mcap recorder)."""
        raise NotImplementedError(
            "rosbag/mcap recording is deferred to Phase 6 — see plan "
            "demo-release-dds-1 Phase 2 OUT-OF-SCOPE list"
        )

    def stop_recording(self, timeout: float = 5.0) -> bool:
        return False

    # ------------------------------------------------------------------ internals

    def _normalised_qos(self, qos: Any) -> Dict[str, Any]:
        """Resolve the caller-provided QoS to a full descriptor dict.

        Accepts: ``None`` (→ sensor_data default), a profile-name string
        (``"reliable"`` / ``"best_effort"`` / ``"sensor_data"`` / ...), or a
        partial override dict that gets merged onto the sensor_data baseline.
        Strings go through :func:`qos_profiles.resolve` so the alias map
        (``"best_effort"`` → SENSOR_DATA, ``"reliable"`` → RELIABLE_COMMAND)
        keeps short-form callers working without the dict ceremony.
        """
        if not qos:
            return qos_profiles.resolve("sensor_data")
        if isinstance(qos, str):
            return qos_profiles.resolve(qos)
        base = qos_profiles.resolve("sensor_data")
        return qos_profiles.merge(base, qos)

    def _emit_error(self, reason: str) -> None:
        cb = self._on_error_cb
        if cb is None:
            return
        try:
            cb(reason)
        except Exception:
            pass
