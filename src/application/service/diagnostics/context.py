"""DiagnosticContext — runtime resources passed to every probe / repair.

Probes never construct their own SSH session, transport, or bridge handles.
They receive a ``DiagnosticContext`` that exposes whichever of those are
available at the moment of P9. SSH is optional — most host probes don't
need it; brand probes that need it gracefully short-circuit when ``ssh is
None``.

The context is built once by :class:`DiagnosticsTask`; if the task
upgrades the SSH session mid-run it returns a new context via
:meth:`with_ssh` rather than mutating in place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from application.service.adapters.base_adapter import BaseAdapter
    from application.service.connection.profile import ConnectionProfile
    from application.service.connection.ssh_session import SSHSession
    from application.service.connection.transport.base import Transport


@dataclass(frozen=True)
class DiagnosticContext:
    """Snapshot of live runtime resources at the moment a probe runs."""

    profile: "ConnectionProfile"
    transport: Optional["Transport"] = None
    bridge: Optional[object] = None
    adapter: Optional["BaseAdapter"] = None
    ssh: Optional["SSHSession"] = None

    def with_ssh(self, ssh: Optional["SSHSession"]) -> "DiagnosticContext":
        """Return an immutable copy with a different SSH session."""
        return replace(self, ssh=ssh)


__all__ = ["DiagnosticContext"]
