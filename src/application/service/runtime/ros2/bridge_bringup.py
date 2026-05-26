# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""bridge_bringup — minimal native-bridge factory for Phase 2.

Thin replacement for DEMO's ``bridge_factory.py``: Phase 2 only supports the
native (in-process cyclonedds) path; the docker / external subprocess paths
are intentionally not ported (they were debug-only fallbacks). Phase 5 will
re-introduce a richer factory once :class:`BaseROS2Adapter` and brand packages
are ported.

Entry point: :func:`build_native_bridge` — given a :class:`ConnectionProfile`
and an active :class:`Transport`, assemble the config dict
:class:`NativeDDSBridge` consumes (auto-detecting the local NIC for cyclonedds
discovery binding).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from application.service.connection.profile import ConnectionProfile
from application.service.connection.transport.base import (
    Transport,
    TransportKind,
)
from application.service.runtime.ros2.native_dds_bridge import (
    NativeDDSBridge,
    _pick_host_address_for_peer,
)


def _peer_ip_for(transport: Transport, profile: ConnectionProfile) -> Optional[str]:
    """Resolve the discovery peer IP from the active transport.

    USB transports advertise ``usb_ip`` directly; other transports fall back to
    ``profile.pupper_ip`` (the user-entered Address for the active row in the
    UI). Returns ``None`` when no peer is needed (multicast LAN discovery).
    """
    if transport.kind is TransportKind.USB:
        usb_ip = getattr(transport, "usb_ip", "") or ""
        return str(usb_ip).strip() or None
    pupper = (profile.pupper_ip or "").strip()
    return pupper or None


def build_native_bridge(
    profile: ConnectionProfile,
    transport: Transport,
    *,
    start_timeout: float = 2.0,
) -> NativeDDSBridge:
    """Construct a :class:`NativeDDSBridge` ready for ``.start()``.

    Composes the 10-key config dict from the profile + active transport:

    - ``domain_id``       : ``profile.ros_domain_id``
    - ``brand_id``        : ``profile.brand`` (empty in Phase 2 — Phase 5 fills)
    - ``robot_hostname``  : ``profile.robot_hostname``
    - ``transport``       : transport ``kind`` value (``"usb"`` / ``"ethernet"`` ...)
    - ``usb_ip``          : populated when transport is USB
    - ``pupper_ip``       : ``profile.pupper_ip`` (Wi-Fi/Ethernet path)
    - ``discovery_peer``  : explicit override (None lets the bridge derive it)
    - ``host_address``    : auto-picked local NIC matching the peer's /24
    - ``start_timeout``   : DomainParticipant construction budget
    """
    peer_ip = _peer_ip_for(transport, profile)
    host_address = _pick_host_address_for_peer(peer_ip)

    cfg: Dict[str, Any] = {
        "domain_id": int(profile.ros_domain_id),
        "brand_id": str(profile.brand or ""),
        "robot_hostname": str(profile.robot_hostname or ""),
        "transport": str(transport.kind.value),
        "pupper_ip": str(profile.pupper_ip or ""),
        "discovery_peer": peer_ip,
        "host_address": host_address,
        "start_timeout": float(start_timeout),
    }
    if transport.kind is TransportKind.USB:
        cfg["usb_ip"] = str(getattr(transport, "usb_ip", "") or "")

    return NativeDDSBridge(cfg)


__all__ = ["build_native_bridge"]
