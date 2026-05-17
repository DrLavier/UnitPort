"""OutboundWriterMatchProbe — verify our cmd writers have a robot subscriber.

Phase 3.6 reproduces DEMO's ``handoff_probe`` "estop_racing_zeros" check at
the host side, but flipped: instead of SSH-ing in and counting publishers
on the robot, we ask the local cyclonedds participant
"how many remote DataReaders are matched to our writer right now?"
via :meth:`NativeDDSBridge.publication_matched_count`.

Tri-state semantics (matches the bridge API):

* ``None`` — host has **no writer** for this topic yet. The user hasn't
  pressed [Take Control], or the writer was torn down on disconnect.
  Surface as INFO so it doesn't masquerade as a robot-side failure.
* ``0``   — writer exists but no remote DataReader is matched. **Real
  failure.** Cmd will publish into the void. ERROR with diagnostic.
* ``>=1`` — writer is matched. OK.

No auto-repair is attached here; the brand-specific
``MangdangChampSubscriberProbe`` (or its equivalents on other brands)
owns the "restart the bringup service" repair so the host probe stays
brand-agnostic.
"""

from __future__ import annotations

from typing import List, Optional

from application.service.adapters import topic_registry
from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


class OutboundWriterMatchProbe(DiagnosticProbe):
    id = "outbound_writer_match"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        bridge = ctx.bridge
        adapter = ctx.adapter
        if bridge is None or adapter is None:
            return [self.info(
                "bridge or adapter not active; skipping writer match check",
            )]

        brand_id = getattr(adapter, "BRAND_ID", "") or ""
        if not brand_id:
            return [self.info("adapter has no BRAND_ID; nothing to check")]

        cmd_specs = topic_registry.list_for_brand(brand_id, role="cmd")
        if not cmd_specs:
            return [self.info(
                f"no cmd-role topics declared for brand '{brand_id}'",
            )]

        findings: List[DiagnosticFinding] = []
        for spec in cmd_specs:
            count: Optional[int]
            try:
                count = bridge.publication_matched_count(spec.topic)
            except Exception as exc:
                findings.append(self.warning(
                    f"could not read publication_matched_count for {spec.topic}",
                    detail=str(exc),
                ))
                continue
            if count is None:
                # No host writer yet — user hasn't clicked [Take Control]
                # (or the topic is published lazily on first frame). Not a
                # failure: the connection is healthy, we just haven't tried
                # to talk on it yet.
                findings.append(self.info(
                    f"no host writer on {spec.topic} yet "
                    "(click [Take Control] to create one)",
                ))
                continue
            if count > 0:
                findings.append(self.ok(
                    f"{spec.topic} writer matched to {count} reader(s)",
                ))
                continue
            findings.append(self.error(
                f"{spec.topic} writer has no matched reader",
                detail=(
                    f"Our DataWriter on '{spec.topic}' "
                    f"(msg_type={spec.msg_type}, qos={spec.qos_profile}) "
                    "is created but not paired with any remote DataReader. "
                    "Cmd will publish into the void. Likely causes: "
                    "(1) brand-specific bringup service inactive on the robot "
                    "(or its inner node crashed); (2) QoS RELIABILITY mismatch "
                    "(reliable vs best_effort don't pair); (3) topic namespace "
                    "mismatch between host adapter and robot launch file. "
                    f"Suggested SSH check: 'ros2 topic info {spec.topic} -v'."
                ),
                requires_ssh=True,
            ))
        return findings


__all__ = ["OutboundWriterMatchProbe"]
