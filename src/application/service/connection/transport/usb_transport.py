# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""USBDataTransport — DATA-role HTTP+DDS channel over the libcomposite tether.

Phase 2 simplification of DEMO ``usb_identity_transport.py``: stdlib-only
(``urllib.request``), no SSH dependency, no inspector subsystem. Probes
``http://192.168.55.1:9999/identity`` to confirm the robot-side daemon is up;
optional hostname validation lands in Phase 3 once the inspector subsystem is
ported.

The transport advertises ``Capability.HTTP | Capability.DDS`` so:

- Phase 2 ``Ros2BrownfieldConnectTask.P0_USB_IDENTITY_PROBE`` calls ``connect()``
  to validate the daemon endpoint.
- ``NativeDDSBridge`` is constructed with ``transport="usb"`` and
  ``usb_ip=<addr>``; cyclonedds discovery then targets the same ``192.168.55.1``
  peer over the USB tether.

Phase 3+ extends this with the handshake POST to flip the robot into
``unitport.target`` (when configured); for now the transport is read-only on
the HTTP side.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from application.service.connection.transport.base import (
    Capability,
    ExecResult,
    HTTPResult,
    NotSupported,
    TransportKind,
    TransportRole,
    TransportUnavailable,
)


DEFAULT_USB_IP = "192.168.55.1"
DEFAULT_USB_PORT = 9999
DEFAULT_PROBE_TIMEOUT_S = 2.0
DEFAULT_HTTP_TIMEOUT_S = 5.0


class USBDataTransport:
    """Transport backed by the robot-side identity HTTP daemon.

    Role: DATA — the USB tether carries steady-state DDS pub/sub via the
    NativeDDSBridge. The HTTP capability here is for control-plane coordination
    (identity probe, future handshake / mode flip) that complements the DDS
    data plane; shell-dependent ops are out of scope on this path.
    """

    role: TransportRole = TransportRole.DATA
    kind: TransportKind = TransportKind.USB
    capabilities: Capability = Capability.HTTP | Capability.DDS

    def __init__(
        self,
        usb_ip: str = DEFAULT_USB_IP,
        port: int = DEFAULT_USB_PORT,
        expected_hostname: str = "",
    ) -> None:
        self._usb_ip = str(usb_ip or DEFAULT_USB_IP).strip()
        self._port = int(port or DEFAULT_USB_PORT)
        self._expected_hostname = str(expected_hostname or "").strip()
        self._connected: bool = False
        self._identity: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Probe ``/identity`` on the robot-side daemon. Idempotent.

        Raises :class:`TransportUnavailable` with a stable error code on
        failure so the connector can render a typed phase error instead of an
        opaque traceback.
        """
        if self._connected:
            return
        url = f"http://{self._usb_ip}:{self._port}/identity"
        try:
            with urllib.request.urlopen(url, timeout=DEFAULT_PROBE_TIMEOUT_S) as resp:
                status = int(getattr(resp, "status", 200))
                if status >= 400:
                    raise TransportUnavailable(
                        "usb_identity_unreachable",
                        f"USB identity probe to {url} returned HTTP {status}",
                    )
                raw = resp.read() or b""
        except urllib.error.URLError as exc:
            raise TransportUnavailable(
                "usb_unreachable",
                f"USB identity probe to {url} failed: {exc.reason}",
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TransportUnavailable(
                "usb_timeout",
                f"USB identity probe to {url} timed out after "
                f"{DEFAULT_PROBE_TIMEOUT_S:.1f}s",
            ) from exc
        except OSError as exc:
            raise TransportUnavailable(
                "usb_unreachable",
                f"USB identity probe to {url} failed: {exc}",
            ) from exc

        # Parse identity payload best-effort. Phase 2 does not enforce any
        # field beyond "the daemon answered"; Phase 3's inspector will validate
        # hostname / serial / brand against the saved ConnectionProfile.
        try:
            identity = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            identity = {}
        self._identity = identity if isinstance(identity, dict) else {}

        # Optional hostname match — only enforced when the caller supplies an
        # expected_hostname AND the daemon advertises one.
        if self._expected_hostname:
            got = str(self._identity.get("hostname", "")).strip()
            if got and got != self._expected_hostname:
                raise TransportUnavailable(
                    "usb_identity_mismatch",
                    f"identity mismatch — expected hostname="
                    f"{self._expected_hostname!r}, got {got!r}",
                )

        self._connected = True

    def disconnect(self) -> HTTPResult:
        """Tear down the channel. The probe is connectionless so this just
        flips the local flag — the matching handshake/disconnect endpoint is
        a Phase 3 concern.
        """
        self._connected = False
        return HTTPResult(ok=True, status=200, body={"detail": "usb_close_local"})

    def is_alive(self) -> bool:
        return self._connected

    def probe_alive(self, timeout_s: float = 1.5) -> bool:
        """Live network probe — does an HTTP GET on ``/identity``.

        Unlike :meth:`is_alive` (which only reads the local connect flag),
        this hits the wire. Used by
        :class:`Ros2ConnectionController`'s health watchdog to detect a
        physical USB unplug while the bridge is up — cyclonedds itself
        only loses the participant after a multi-second lease timeout.

        Returns True on HTTP 2xx; False on any URLError / timeout / OSError /
        non-2xx status. Never raises.
        """
        url = f"http://{self._usb_ip}:{self._port}/identity"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return int(getattr(resp, "status", 200)) < 400
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError):
            return False

    @property
    def usb_ip(self) -> str:
        """The probed peer IP — consumed by ``bridge_bringup`` to seed
        cyclonedds discovery.
        """
        return self._usb_ip

    @property
    def identity(self) -> Dict[str, Any]:
        """Last-fetched identity payload (empty before ``connect()``)."""
        return dict(self._identity or {})

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def exec_shell(self, command: str, timeout: float = 60.0) -> ExecResult:
        raise NotSupported("usb identity transport does not provide shell access")

    def http_post(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
    ) -> HTTPResult:
        url = f"http://{self._usb_ip}:{self._port}{path}"
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read() or b""
        except urllib.error.HTTPError as exc:
            return HTTPResult(ok=False, status=int(exc.code), body={}, error=str(exc))
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            return HTTPResult(ok=False, status=0, body={}, error=str(exc))
        try:
            decoded = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            decoded = {}
        body_out = decoded if isinstance(decoded, dict) else {"result": decoded}
        return HTTPResult(ok=200 <= status < 300, status=status, body=body_out)


__all__ = [
    "USBDataTransport",
    "DEFAULT_USB_IP",
    "DEFAULT_USB_PORT",
]
