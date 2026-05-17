"""application.service.connection.transport — transport layer.

Phase 2 ships the first concrete transport (USB DATA role); Ethernet / WiFi /
WebRTC arrive in Phase 3+. The :func:`select_data_transport` selector dispatches
on ``profile.connector_type`` + the user's transport choice, but Phase 2 only
honours USB — other choices fall through with a typed
:class:`TransportUnavailable`.
"""

from __future__ import annotations

from application.service.connection.profile import ConnectionProfile
from application.service.connection.transport.base import (
    Capability,
    ExecResult,
    HTTPResult,
    NotSupported,
    Transport,
    TransportKind,
    TransportRole,
    TransportUnavailable,
)
from application.service.connection.transport.usb_transport import (
    DEFAULT_USB_IP,
    DEFAULT_USB_PORT,
    USBDataTransport,
)


_PHASE2_SUPPORTED_TRANSPORTS = frozenset({"USB"})


def select_data_transport(
    profile: ConnectionProfile,
    transport_label: str = "USB",
) -> Transport:
    """Return the active DATA-role transport for the given profile.

    Phase 2 only ships :class:`USBDataTransport`; the controller passes the
    user's UI selection through ``transport_label`` so non-USB choices fail
    fast with a typed :class:`TransportUnavailable` instead of silently
    probing a USB endpoint against a non-USB address. Phase 3 will extend
    this dispatcher with SSH / Ethernet implementations.

    The USB peer IP is taken from ``profile.pupper_ip`` (Phase 1's
    ``build_profile_from_info`` maps the per-transport UI Address into that
    field) so the user's saved address flows through without an extra
    routing layer.
    """
    label = str(transport_label or "USB").strip()
    if label not in _PHASE2_SUPPORTED_TRANSPORTS:
        raise TransportUnavailable(
            "transport_not_in_phase2",
            f"transport {label!r} not implemented in Phase 2 — "
            f"only USB is supported. Phase 3 adds SSH/Ethernet/WebRTC.",
        )
    usb_ip = (profile.pupper_ip or DEFAULT_USB_IP).strip() or DEFAULT_USB_IP
    return USBDataTransport(
        usb_ip=usb_ip,
        port=DEFAULT_USB_PORT,
        expected_hostname=str(profile.robot_hostname or ""),
    )


__all__ = [
    "Capability",
    "ExecResult",
    "HTTPResult",
    "NotSupported",
    "Transport",
    "TransportKind",
    "TransportRole",
    "TransportUnavailable",
    "USBDataTransport",
    "DEFAULT_USB_IP",
    "DEFAULT_USB_PORT",
    "select_data_transport",
]
