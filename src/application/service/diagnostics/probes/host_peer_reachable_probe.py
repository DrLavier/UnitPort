# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""HostPeerReachableProbe — verify the host can reach the robot's IP.

Single ICMP echo (``ping``) at the OS level — no admin rights needed on
Windows. When the robot is unreachable the rest of the diagnose chain is
moot, so this finding is ERROR severity. The natural "fix" is human:
re-seat the cable, power-cycle the robot, check the NIC. We surface the
checklist in the detail field rather than offering a button.
"""

from __future__ import annotations

import platform
import subprocess
from typing import List

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


class HostPeerReachableProbe(DiagnosticProbe):
    id = "host_peer_reachable"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        target = (ctx.profile.pupper_ip or "").strip()
        if not target:
            return [self.warning(
                "no peer IP configured",
                detail=(
                    "ConnectionProfile.pupper_ip is empty. Set the address "
                    "field in sec1 before re-running diagnostics."
                ),
            )]

        if platform.system() == "Windows":
            cmd = ["ping", "-n", "2", "-w", "1500", target]
        else:
            cmd = ["ping", "-c", "2", "-W", "2", target]

        try:
            proc = subprocess.run(  # noqa: S603 — well-known OS ping
                cmd, capture_output=True, text=True, timeout=8,
            )
        except subprocess.TimeoutExpired:
            return [self.error(
                f"ping {target} timed out",
                detail="Host ICMP echo blocked or robot offline.",
            )]
        except Exception as exc:  # noqa: BLE001
            return [self.warning(
                "ping subprocess failed",
                detail=str(exc),
            )]

        if proc.returncode == 0 and "TTL=" in proc.stdout.upper():
            return [self.ok(
                f"peer reachable: {target}",
                detail=_first_rtt_line(proc.stdout),
            )]

        # Some Windows builds return exit 0 even on full loss when the
        # destination network is unreachable. Be defensive.
        return [self.error(
            f"peer unreachable: {target}",
            detail=(
                "ping returned no successful echo. Check: "
                "(1) USB cable seated; (2) NIC enabled in Network "
                "Connections; (3) robot powered and booted; "
                "(4) profile.pupper_ip matches the robot's tether IP."
            ),
        )]


def _first_rtt_line(out: str) -> str:
    for ln in out.splitlines():
        s = ln.strip()
        if "time=" in s.lower() or "time<" in s.lower():
            return s
    return ""


__all__ = ["HostPeerReachableProbe"]
