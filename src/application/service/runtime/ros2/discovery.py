# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""DDSDiscoveryView — wire-level peer/publication/subscription enumeration.

Wraps the cyclonedds-python builtin readers (DCPSParticipant / DCPSPublication
/ DCPSSubscription) so probes can answer questions like "is anyone else
publishing on rt/cmd_vel?" without poking cyclonedds internals from each probe.

The view holds three lazy-initialised BuiltinDataReaders sharing the bridge's
participant. Filtering "is this entity ours?" is done by comparing the
discovered entity's ``participant_instance_handle`` (or ``participant_key``)
against the bridge participant's own handle/key — cheaper and more robust than
maintaining a per-writer GUID table.

Topic naming conventions:
- ROS2 callers use ``/cmd_vel``; cyclonedds wire names are ``rt/cmd_vel``
  (rmw_cyclonedds_cpp). All topic_filter / foreign_publications / etc.
  arguments accept either form; we normalise both ways internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set


# --------------------------------------------------------------------- DTOs


@dataclass(frozen=True)
class ParticipantInfo:
    """Discovered DDS participant (a process speaking DDS in the domain)."""

    key: Any                # uuid.UUID — globally unique participant identity
    instance_handle: int    # cyclonedds-local handle for fast comparison
    hostname: str           # extracted from user_data when present
    process_id: int         # extracted from user_data when present


@dataclass(frozen=True)
class PublicationInfo:
    """Discovered DDS DataWriter announcing on a topic."""

    key: Any                # uuid.UUID — writer's GUID
    participant_key: Any    # uuid.UUID — owning participant
    participant_instance_handle: int
    topic_name: str         # DDS-side name, e.g. "rt/cmd_vel"
    type_name: str          # e.g. "geometry_msgs::msg::dds_::Twist_"
    qos_reliability: str    # "reliable" | "best_effort" | "unknown"


@dataclass(frozen=True)
class SubscriptionInfo:
    """Discovered DDS DataReader requesting a topic."""

    key: Any
    participant_key: Any
    participant_instance_handle: int
    topic_name: str
    type_name: str
    qos_reliability: str


# --------------------------------------------------------------------- helpers


def _ros_to_dds(topic: str) -> str:
    """``/cmd_vel`` -> ``rt/cmd_vel``. Accepts already-DDS names unchanged."""
    if not topic:
        return topic
    if topic.startswith("rt/"):
        return topic
    return "rt/" + topic.lstrip("/")


def _topic_matches(actual: str, want: Optional[str]) -> bool:
    """True when ``actual`` (DDS-side topic) matches the user's filter.

    Accepts both ROS form (``/cmd_vel``) and DDS form (``rt/cmd_vel``).
    """
    if not want:
        return True
    return actual == want or actual == _ros_to_dds(want)


def _read_qos_reliability(qos: Any) -> str:
    """Extract reliability from a cyclonedds Qos blob; "unknown" on miss."""
    if qos is None:
        return "unknown"
    # cyclonedds-python exposes Qos as an iterable of policy instances; the
    # repr of a Reliability policy is e.g. "Policy.Reliability.Reliable(...)".
    try:
        for policy in qos:
            name = type(policy).__name__
            if name.endswith("Reliable"):
                return "reliable"
            if name.endswith("BestEffort"):
                return "best_effort"
    except TypeError:
        pass
    return "unknown"


def _entity_attr(entity: Any, *candidates: str, default: Any = None) -> Any:
    """Read the first attribute that exists on ``entity``."""
    for name in candidates:
        if hasattr(entity, name):
            return getattr(entity, name)
    return default


# --------------------------------------------------------------------- view


class DDSDiscoveryView:
    """Lazy-initialised view onto a participant's discovery state."""

    _BUILTIN_READ_BATCH = 1024

    def __init__(self, participant: Any) -> None:
        self._participant = participant
        self._part_reader: Any = None
        self._pub_reader: Any = None
        self._sub_reader: Any = None
        self._our_handle: Optional[int] = None
        self._our_key: Any = None

    # --- accessors --------------------------------------------------------

    def participants(self) -> List[ParticipantInfo]:
        reader = self._participant_reader()
        if reader is None:
            return []
        out: List[ParticipantInfo] = []
        for sample in self._safe_read(reader):
            qos = _entity_attr(sample, "qos")
            host, pid = _extract_user_data(qos)
            out.append(ParticipantInfo(
                key=_entity_attr(sample, "key"),
                instance_handle=int(_entity_attr(
                    sample, "participant_instance_handle", "instance_handle",
                    default=0,
                )),
                hostname=host,
                process_id=pid,
            ))
        return out

    def publications(
        self, topic_filter: Optional[str] = None,
    ) -> List[PublicationInfo]:
        reader = self._publication_reader()
        if reader is None:
            return []
        out: List[PublicationInfo] = []
        for sample in self._safe_read(reader):
            topic_name = str(_entity_attr(sample, "topic_name", default=""))
            if not _topic_matches(topic_name, topic_filter):
                continue
            out.append(PublicationInfo(
                key=_entity_attr(sample, "key"),
                participant_key=_entity_attr(sample, "participant_key"),
                participant_instance_handle=int(_entity_attr(
                    sample, "participant_instance_handle", default=0,
                )),
                topic_name=topic_name,
                type_name=str(_entity_attr(sample, "type_name", default="")),
                qos_reliability=_read_qos_reliability(
                    _entity_attr(sample, "qos"),
                ),
            ))
        return out

    def subscriptions(
        self, topic_filter: Optional[str] = None,
    ) -> List[SubscriptionInfo]:
        reader = self._subscription_reader()
        if reader is None:
            return []
        out: List[SubscriptionInfo] = []
        for sample in self._safe_read(reader):
            topic_name = str(_entity_attr(sample, "topic_name", default=""))
            if not _topic_matches(topic_name, topic_filter):
                continue
            out.append(SubscriptionInfo(
                key=_entity_attr(sample, "key"),
                participant_key=_entity_attr(sample, "participant_key"),
                participant_instance_handle=int(_entity_attr(
                    sample, "participant_instance_handle", default=0,
                )),
                topic_name=topic_name,
                type_name=str(_entity_attr(sample, "type_name", default="")),
                qos_reliability=_read_qos_reliability(
                    _entity_attr(sample, "qos"),
                ),
            ))
        return out

    def foreign_publications(self, topic: str) -> List[PublicationInfo]:
        """Publications on ``topic`` that don't belong to our participant."""
        own_handle, own_key = self._own_identity()
        return [
            p for p in self.publications(topic_filter=topic)
            if not _is_ours(p, own_handle, own_key)
        ]

    def foreign_subscriptions(self, topic: str) -> List[SubscriptionInfo]:
        own_handle, own_key = self._own_identity()
        return [
            s for s in self.subscriptions(topic_filter=topic)
            if not _is_ours(s, own_handle, own_key)
        ]

    def close(self) -> None:
        """Release the three builtin readers. Idempotent."""
        for attr in ("_part_reader", "_pub_reader", "_sub_reader"):
            entity = getattr(self, attr, None)
            if entity is None:
                continue
            try:
                entity.close()
            except Exception:
                pass
            setattr(self, attr, None)

    # --- internals --------------------------------------------------------

    def _participant_reader(self) -> Any:
        if self._part_reader is None:
            self._part_reader = self._make_builtin_reader("DcpsParticipant")
        return self._part_reader

    def _publication_reader(self) -> Any:
        if self._pub_reader is None:
            self._pub_reader = self._make_builtin_reader("DcpsPublication")
        return self._pub_reader

    def _subscription_reader(self) -> Any:
        if self._sub_reader is None:
            self._sub_reader = self._make_builtin_reader("DcpsSubscription")
        return self._sub_reader

    def _make_builtin_reader(self, kind: str) -> Any:
        """Construct a ``BuiltinDataReader`` for one of the DCPS topics.

        Falls back to None when cyclonedds is missing or the builtin module
        layout differs (older 0.10 / 11.x reorganisations); discovery probes
        treat None as "no data available" and degrade gracefully.
        """
        if self._participant is None:
            return None
        try:
            from cyclonedds import builtin as cyc_builtin  # type: ignore
        except ImportError:
            return None
        topic_obj = getattr(cyc_builtin, "BuiltinTopic" + kind, None)
        reader_cls = getattr(cyc_builtin, "BuiltinDataReader", None)
        if topic_obj is None or reader_cls is None:
            return None
        try:
            return reader_cls(self._participant, topic_obj)
        except Exception:
            return None

    def _safe_read(self, reader: Any) -> List[Any]:
        try:
            samples = reader.read(N=self._BUILTIN_READ_BATCH)
        except Exception:
            return []
        return list(samples or [])

    def _own_identity(self) -> tuple:
        """Cache and return ``(instance_handle, key)`` for our participant."""
        if self._our_handle is None:
            self._our_handle = int(_entity_attr(
                self._participant, "instance_handle", default=0,
            ))
            self._our_key = _entity_attr(self._participant, "guid")
        return self._our_handle, self._our_key


def _extract_user_data(qos: Any) -> tuple:
    """Best-effort hostname/pid extraction from a participant's user_data.

    UnitPort and most ROS2 participants encode ``hostname=...,pid=...`` in
    the participant user_data policy. We parse defensively — anything we can't
    decode just yields ("", 0).
    """
    if qos is None:
        return "", 0
    try:
        for policy in qos:
            name = type(policy).__name__
            if not name.endswith("Userdata"):
                continue
            raw = getattr(policy, "value", b"") or b""
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="replace")
            host = ""
            pid = 0
            for token in str(raw).split(","):
                k, _, v = token.partition("=")
                k = k.strip().lower()
                v = v.strip()
                if k in ("hostname", "host"):
                    host = v
                elif k in ("pid", "process_id"):
                    try:
                        pid = int(v)
                    except ValueError:
                        pass
            return host, pid
    except TypeError:
        pass
    return "", 0


def _is_ours(entity: Any, own_handle: int, own_key: Any) -> bool:
    """True when the discovered entity belongs to our participant."""
    if own_handle and getattr(entity, "participant_instance_handle", 0) == own_handle:
        return True
    if own_key is not None:
        ent_key = getattr(entity, "participant_key", None)
        if ent_key is not None and ent_key == own_key:
            return True
    return False


__all__ = [
    "ParticipantInfo",
    "PublicationInfo",
    "SubscriptionInfo",
    "DDSDiscoveryView",
]
