# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Host-side probes — brand-agnostic checks the orchestrator always runs.

Each probe inspects something on the host machine: USB tether NIC, host
firewall (Windows-only — others skip with INFO), peer reachability, DDS
discovery via cyclonedds. Brand-specific probes live alongside their
brand adapter (``application/service/adapters/<brand>/probes.py``) and
are added by :meth:`BaseAdapter.diagnostic_probes`.

Phase 3.6 added three wire-level probes that read the bridge's discovery
view (publication_matched_count, foreign_publishers, heartbeat match) to
catch the "connect ok but cmd_vel never reaches the robot" failure mode.
"""

from typing import Tuple, Type

from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.probes.cmd_topic_multi_publisher_probe import (
    CmdTopicMultiPublisherProbe,
)
from application.service.diagnostics.probes.dds_discovery_probe import (
    DdsDiscoveryProbe,
)
from application.service.diagnostics.probes.heartbeat_match_probe import (
    HeartbeatMatchProbe,
)
from application.service.diagnostics.probes.host_firewall_probe import (
    HostFirewallProbe,
)
from application.service.diagnostics.probes.host_peer_reachable_probe import (
    HostPeerReachableProbe,
)
from application.service.diagnostics.probes.host_usb_nic_probe import (
    HostUsbNicProbe,
)
from application.service.diagnostics.probes.outbound_writer_match_probe import (
    OutboundWriterMatchProbe,
)


HOST_PROBES: Tuple[Type[DiagnosticProbe], ...] = (
    HostUsbNicProbe,
    HostFirewallProbe,
    HostPeerReachableProbe,
    DdsDiscoveryProbe,
    OutboundWriterMatchProbe,
    CmdTopicMultiPublisherProbe,
    HeartbeatMatchProbe,
)


__all__ = [
    "HOST_PROBES",
    "CmdTopicMultiPublisherProbe",
    "DdsDiscoveryProbe",
    "HeartbeatMatchProbe",
    "HostFirewallProbe",
    "HostPeerReachableProbe",
    "HostUsbNicProbe",
    "OutboundWriterMatchProbe",
]
