# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Transport protocol — polymorphic I/O channel for the connection layer.

Ported verbatim from DEMO ``src/system/service/connection/transport/base.py``
(2026-05-14, Phase 1). RELEASE only takes the type-level contracts in this
file; concrete transports + ``select_*_transport`` selectors land in Phase 2.

Historically the ROS2 connector assumed every phase ran over paramiko SSH,
with a handful of ``if self._ssh is None: skip`` guards that fired when a
USB-identity-only path needed to bypass SSH. That left two problems:

    1. The disconnect UI calls hard-coded ``usb_handshake.send_disconnect``,
       ignoring whether the connect actually ran over SSH or USB.
    2. Adding a third transport (WebRTC, custom bridges) required another
       round of scattered ``if`` branches.

This module turns the transport into a first-class object. P1 selects the
active ``Transport``; later phases ask it ``Capability.SHELL in caps?`` and
either use ``exec_shell`` / ``http_post`` or emit a clean capability skip.

Capabilities are a :class:`enum.Flag` so a single transport can advertise
multiple channels (SSH also supports file-push; a future WebRTC transport
might support SHELL + STREAM). ``TransportUnavailable`` is the typed error
that Phase 2's ``select_transport`` will raise when the user-chosen transport
can't be used (driver missing, not implemented, ...) so the connector can
surface it as a clean P1 failure instead of an unhelpful exception traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class TransportRole(Enum):
    """What lifecycle phase a transport serves.

    BOOTSTRAP transports (SSH, WebRTC) carry the first-time provisioning
    flow: push the workspace tarball, run ``bootstrap.sh``, trigger reboot.
    They are NOT used in steady-state — the connector explicitly tears
    them down after reboot and switches to a DATA transport.

    DATA transports (USB tether, Ethernet/WiFi LAN) carry the steady-state
    DDS pub/sub. They never need shell capability; the robot-side stack
    runs autonomously after bootstrap and exposes its control surface
    purely as ROS2 topics + services.

    The split is deliberate: SSH is overpowered for steady-state (and a
    security/maintenance liability), while DDS bridges have nothing
    useful to say during a fresh-image install.
    """

    BOOTSTRAP = "bootstrap"
    DATA = "data"


class TransportKind(Enum):
    """Concrete transport implementation identifier.

    Used by ``BridgeFactory`` to dispatch to the right ConnectionBridge
    implementation based on the active transport, and by phase logging
    so users see which channel their connection is actually using.
    """

    SSH = "ssh"
    WEBRTC = "webrtc"
    USB = "usb"
    ETHERNET = "ethernet"


class Capability(Flag):
    """Transport capability bits.

    A transport advertises every channel it can drive. Phases check
    ``Capability.X in transport.capabilities`` before calling the matching
    method — calling an unsupported method raises :class:`NotSupported`.
    """

    NONE = 0
    SHELL = auto()       # exec_shell(cmd) — paramiko or equivalent
    HTTP = auto()        # http_post(path, body) — identity-daemon-style
    FILE_PUSH = auto()   # put_text / put_bytes — SFTP or equivalent
    DDS = auto()         # NativeDDSBridge / cyclonedds-python pub/sub


@dataclass
class ExecResult:
    """Result of :meth:`Transport.exec_shell`.

    Shape intentionally matches DEMO ``SSHSession.SSHResult`` so existing
    DEMO call sites reading ``.ok`` / ``.stdout`` / ``.stderr`` keep working
    when their implementations are ported in Phase 2/3.
    """

    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class HTTPResult:
    """Result of :meth:`Transport.http_post`.

    ``body`` is the parsed JSON object when the response was JSON, else
    an empty dict. ``ok`` means HTTP 2xx *and* (when the server returned
    an ``ok`` field) that field was truthy — matching the convention
    DEMO's ``usb_handshake`` already uses.
    """

    ok: bool
    status: int = 0
    body: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class NotSupported(RuntimeError):
    """Raised when a caller invokes a capability the transport doesn't have."""


class TransportUnavailable(RuntimeError):
    """Raised by ``select_transport`` (Phase 2+) when the requested transport
    cannot be constructed (driver missing, not implemented, ...).

    The connector catches this and produces a typed ``PhaseResult.failure``
    so the UI sees a clean error code rather than a stack trace.
    """

    def __init__(self, error_code: str, message: str = "") -> None:
        super().__init__(message or error_code)
        self.error_code = error_code


@runtime_checkable
class Transport(Protocol):
    """Protocol every concrete transport implements.

    The Protocol is runtime-checkable so tests can assert
    ``isinstance(x, Transport)`` for structural conformance without
    sub-classing. Real implementations live in sibling modules from
    Phase 2 onwards.

    Every transport declares both a ``role`` (lifecycle phase it serves)
    and a ``kind`` (concrete implementation identifier). The two fields
    let the connector and BridgeFactory dispatch without if/elif chains
    on transport type.
    """

    capabilities: "Capability"
    role: "TransportRole"
    kind: "TransportKind"

    def connect(self) -> None:
        """Establish the channel. Must be idempotent — calling twice is a no-op."""
        ...

    def disconnect(self) -> "HTTPResult":
        """Tear the channel down. Returns a :class:`HTTPResult` so the
        disconnect helper can surface success / failure uniformly across
        SSH (where the shape is synthetic) and HTTP (where it's native)."""
        ...

    def is_alive(self) -> bool:
        """True when the channel is currently usable."""
        ...

    def exec_shell(self, command: str, timeout: float = 60.0) -> "ExecResult":
        """Run a shell command. Raises :class:`NotSupported` when
        ``Capability.SHELL`` is not in :attr:`capabilities`."""
        ...

    def http_post(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 5.0,
    ) -> "HTTPResult":
        """POST JSON to ``path`` on the transport's base URL. Raises
        :class:`NotSupported` when ``Capability.HTTP`` is not in
        :attr:`capabilities`."""
        ...


__all__ = [
    "Capability",
    "ExecResult",
    "HTTPResult",
    "NotSupported",
    "Transport",
    "TransportKind",
    "TransportRole",
    "TransportUnavailable",
]
