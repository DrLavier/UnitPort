"""CmdTopicMultiPublisherProbe — flag competing publishers on cmd topics.

DEMO traced a class of "robot doesn't move even though host publishes
/cmd_vel" bugs to a second publisher on the same topic — typically the
mini_pupper factory web service was still alive alongside our unitport
bringup, both writing zero Twists at conflicting rates and drowning the
real teleop frames.

This probe enumerates every DataWriter the local cyclonedds participant
has discovered for each cmd-role topic, filters out our own writer, and
emits a WARNING (not ERROR) per remaining foreign publisher. It's a
WARNING because legitimate multi-controller scenarios exist (a user
running their own ROS2 node alongside UnitPort); the dialog will still
auto-pop on >=WARNING so the user sees it, but P9 stays "warning" and
the connection isn't rolled back.

The repair for the canonical Mangdang case (mini_pupper_web racing
/cmd_vel) lives on :class:`MangdangServiceConflictProbe`; this probe
just surfaces the symptom and lets the brand probe attach the SSH-side
fix when applicable.
"""

from __future__ import annotations

from typing import List

from application.service.adapters import topic_registry
from application.service.diagnostics.base_probe import DiagnosticProbe
from application.service.diagnostics.context import DiagnosticContext
from application.service.diagnostics.results import DiagnosticFinding


class CmdTopicMultiPublisherProbe(DiagnosticProbe):
    id = "cmd_topic_multi_publisher"
    requires_ssh = False

    def run(self, ctx: DiagnosticContext) -> List[DiagnosticFinding]:
        bridge = ctx.bridge
        adapter = ctx.adapter
        if bridge is None or adapter is None:
            return [self.info(
                "bridge or adapter not active; skipping multi-publisher check",
            )]

        brand_id = getattr(adapter, "BRAND_ID", "") or ""
        if not brand_id:
            return [self.info("adapter has no BRAND_ID; nothing to check")]

        cmd_specs = topic_registry.list_for_brand(brand_id, role="cmd")
        if not cmd_specs:
            return [self.info(
                f"no cmd-role topics declared for brand '{brand_id}'",
            )]

        # Enumerate participants once so we can annotate foreign publishers
        # with hostname/pid when the remote announced itself in user_data.
        try:
            participants = bridge.discovered_participants()
        except Exception:
            participants = []
        by_handle = {p.instance_handle: p for p in participants if p.instance_handle}

        findings: List[DiagnosticFinding] = []
        for spec in cmd_specs:
            try:
                foreign = list(bridge.foreign_publishers(spec.topic))
            except Exception as exc:
                findings.append(self.warning(
                    f"could not enumerate publishers on {spec.topic}",
                    detail=str(exc),
                ))
                continue
            if not foreign:
                findings.append(self.ok(
                    f"{spec.topic} has no competing publisher",
                ))
                continue

            lines = []
            for pub in foreign:
                handle = int(pub.participant_instance_handle or 0)
                origin = by_handle.get(handle)
                origin_str = ""
                if origin is not None and (origin.hostname or origin.process_id):
                    origin_str = f" from {origin.hostname or '?'}:" \
                                 f"{origin.process_id or '?'}"
                key_repr = str(getattr(pub, "key", "?"))[:18]
                lines.append(
                    f"- writer key={key_repr} type={pub.type_name} "
                    f"qos={pub.qos_reliability}{origin_str}"
                )
            findings.append(self.warning(
                f"competing publisher(s) on {spec.topic} "
                f"({len(foreign)} foreign writer(s))",
                detail=(
                    f"Our writer on '{spec.topic}' is sharing the topic with "
                    f"{len(foreign)} other DataWriter(s). They may overwrite "
                    "our cmd samples (especially a robot-side estop/idle "
                    "node publishing zero Twists).\n\nForeign writers:\n"
                    + "\n".join(lines)
                ),
                requires_ssh=False,
            ))
        return findings


__all__ = ["CmdTopicMultiPublisherProbe"]
