# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""DdsDiscoveryProbe — confirm a peer participant exists.

This probe exercises the DDS wire via the **active bridge** that the
adapter opened in P5. Phase 3.6 narrowed its scope to a single, robust
question: "is anything other than us actually speaking DDS in this
domain right now?"

Earlier (Phase 3.5) it also tried to check whether the host's outbound
cmd topic appeared in ``discovered_publications``, but that direction
was wrong: the cmd topic is **host-published**, not robot-published, so
its presence/absence in the pubs list says nothing about robot health.
Whether the robot is actually subscribed to our cmd_vel is owned by the
:class:`OutboundWriterMatchProbe` (and the brand-specific subscriber
probes), which reads ``publication_matched_count`` directly.

When the bridge is not yet up (e.g. user clicked the manual [Diagnose]
button before connecting) the probe emits an INFO finding rather than
failing.
"""

from __future__ import annotations

import time
from typing import List

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


_SETTLE_S = 3.0


class DdsDiscoveryProbe(DiagnosticProbe):
    id = "dds_discovery"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        bridge = ctx.bridge
        if bridge is None:
            return [self.info(
                "bridge not active; skipping DDS discovery",
                detail=(
                    "The diagnostic context does not carry a live "
                    "NativeDDSBridge. Run after a successful connect, or "
                    "let P9 run in-line at the end of the connect chain."
                ),
            )]

        # Give cyclonedds a moment to gather discovery samples.
        time.sleep(_SETTLE_S)

        participants = bridge.discovered_participants()
        n_participants = len(participants)

        if n_participants == 0:
            return [self.error(
                "no peer DDS participants discovered",
                detail=(
                    "The bridge participant is alive but no other "
                    "participant on this DDS domain has been seen in "
                    f"{_SETTLE_S}s. Likely causes: robot-side service is "
                    "down, ROS_DOMAIN_ID mismatch, or robot is using a "
                    "different RMW (FastDDS vs CycloneDDS — these don't "
                    "interoperate at the wire level)."
                ),
                requires_ssh=True,
            )]

        # Best-effort hostname extraction for the detail line.
        hosts = sorted({
            p.hostname for p in participants if p.hostname
        })
        host_str = f" hosts={hosts}" if hosts else ""
        return [self.ok(
            f"DDS discovery healthy (participants={n_participants})",
            detail=f"{n_participants} remote participant(s){host_str}",
        )]


__all__ = ["DdsDiscoveryProbe"]
