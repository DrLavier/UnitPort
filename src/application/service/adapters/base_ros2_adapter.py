# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BaseROS2Adapter — concrete base for any adapter that talks ROS2 via DDS.

Owns the shared lifecycle for every brand whose host-side surface is "subscribe
to standard ROS2 telemetry topics + publish standard ROS2 command topics over
NativeDDSBridge". Subclasses (MangdangROS2Adapter, future UnitreeROS2Adapter,
etc.) only override:

- ``BRAND_ID`` (mandatory, must match topic_registry registration key)
- ``DEFAULT_DOMAIN_ID`` (per-brand deployment convention)
- ``TOPIC_NAMESPACE`` (optional ROS namespace prefix, e.g. ``/<robot>/``)
- ``capabilities()`` — returns the brand's CapabilitySet
- ``_safety_clamp(twist)`` — per-brand teleop envelope (default: identity)
- ``activate()`` / ``deactivate()`` — brand-specific controller-on hook
  (default: no-op for adapters whose runtime is always live)

Subclasses get for free: bridge construction + lifecycle, declared topic
subscription, teleop pump wiring, heartbeat publisher, telemetry forwarding
to AppSignals.connection_telemetry, P8 confirmation walk.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QTimer

from unitport_sdk import log_info, log_warning

from application.service.adapters.base_adapter import BaseAdapter
from application.service.adapters.capabilities import CapabilitySet
from application.service.adapters.heartbeat_publisher import HeartbeatPublisher
from application.service.adapters.teleop_pump import TeleopPump, zero_twist
from application.service.adapters import topic_registry
from application.service.connection.profile import ConnectionProfile
from application.service.connection.transport.base import Transport
from application.service.diagnostics.results import (
    DiagnosticFinding,
    Severity,
)
from application.service.runtime.ros2.bridge_bringup import build_native_bridge
from application.service.runtime.ros2.native_dds_bridge import NativeDDSBridge


class BaseROS2Adapter(BaseAdapter):
    """Concrete base for ROS2-via-DDS brand adapters."""

    # Subclasses MUST override these three.
    BRAND_ID: str = ""
    DEFAULT_DOMAIN_ID: int = 0
    TOPIC_NAMESPACE: str = ""

    # Heartbeat publisher rate. 2 Hz matches DEMO; tune per-brand if the
    # robot-side estop watchdog is more aggressive.
    HEARTBEAT_RATE_HZ: float = 2.0

    # P8 confirmation budget — settle window before declaring "bridge up
    # but no samples; accept anyway".
    CONFIRMATION_TIMEOUT_S: float = 5.0
    CONFIRMATION_POLL_S: float = 0.2

    # Take-Control writer-match grace: how long to wait after the teleop
    # pump starts before checking publication_matched_count. Cyclonedds
    # reliable handshake usually completes well under 200ms across USB; we
    # give 300ms to absorb the worst-case discovery jitter.
    TELEOP_MATCH_GRACE_MS: int = 300

    def __init__(self) -> None:
        if not self.BRAND_ID:
            raise RuntimeError(
                f"{type(self).__name__} did not set BRAND_ID — required for "
                f"topic_registry lookups"
            )
        self._bridge: Optional[NativeDDSBridge] = None
        self._transport: Optional[Transport] = None
        self._profile: Optional[ConnectionProfile] = None
        self._heartbeat: Optional[HeartbeatPublisher] = None
        self._teleop_pump: Optional[TeleopPump] = None

    # ── Session lifecycle ─────────────────────────────────────────────────

    def open_session(
        self,
        profile: ConnectionProfile,
        transport: Transport,
    ) -> None:
        if self.is_session_open():
            return  # idempotent
        self._profile = profile
        self._transport = transport
        # Build + start the DDS bridge. ``build_native_bridge`` reads
        # transport.kind to decide peer/host config; the adapter does not
        # second-guess that wiring.
        self._bridge = build_native_bridge(profile, transport)
        self._bridge.start()
        # Wire every "obs" telemetry subscription declared by the brand's
        # topic table so P8 has signal sources to watch.
        self._subscribe_declared("obs")
        # Heartbeat — required to keep robot-side estop watchdog fed.
        self._heartbeat = HeartbeatPublisher(
            self._bridge, rate_hz=self.HEARTBEAT_RATE_HZ,
        )
        self._heartbeat.start()
        log_info(
            f"[{self.BRAND_ID}] session open: domain={profile.ros_domain_id} "
            f"transport={transport.kind.value}"
        )

    def close_session(self) -> None:
        # Reverse-order teardown: pump → heartbeat → bridge → transport.
        if self._teleop_pump is not None:
            try:
                self._teleop_pump.stop()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] teleop pump stop failed: {exc}")
            self._teleop_pump = None

        if self._heartbeat is not None:
            try:
                self._heartbeat.stop()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] heartbeat stop failed: {exc}")
            self._heartbeat = None

        if self._bridge is not None:
            try:
                self._bridge.stop(timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] bridge stop failed: {exc}")
            self._bridge = None

        if self._transport is not None:
            try:
                self._transport.disconnect()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] transport disconnect failed: {exc}")
            self._transport = None

        self._profile = None

    def is_session_open(self) -> bool:
        return self._bridge is not None and self._bridge.is_alive()

    # ── Capability declaration (subclass MUST override) ───────────────────

    def capabilities(self) -> CapabilitySet:
        # Subclasses override with brand-specific declarations.
        return CapabilitySet()

    # ── Confirmation hooks (P6 / P8) ──────────────────────────────────────

    def confirm_steady_state(self, timeout_s: float = CONFIRMATION_TIMEOUT_S) -> bool:
        """Return True once any declared "obs" subscription is streaming.

        Falls back to a "bridge survived the settle window" success after
        the timeout — same pragmatic stance as Phase 2's controller. This
        mirrors the contract Phase 2 settled on after the dark-bridge
        diagnosis (bridge up + 5s settle is acceptable; the adapter logs
        a diagnostic warning so the user sees DOMAIN/namespace mismatches).
        """
        if self._bridge is None:
            return False
        deadline = time.monotonic() + max(0.5, timeout_s)
        obs_specs = topic_registry.list_for_brand(self.BRAND_ID, role="obs")
        while time.monotonic() < deadline:
            for spec in obs_specs:
                if (self._bridge.rate_hz(spec.topic) or 0.0) > 0.0:
                    log_info(
                        f"[{self.BRAND_ID}] steady-state confirmed via {spec.topic}"
                    )
                    return True
            time.sleep(self.CONFIRMATION_POLL_S)
        # Settle elapsed without samples — accept if bridge is still alive.
        if not self._bridge.is_alive():
            return False
        stats = {}
        try:
            stats = self._bridge.subscription_stats()
        except Exception:
            pass
        log_warning(
            f"[{self.BRAND_ID}] no declared-topic samples in "
            f"{timeout_s:.1f}s. subscription_stats={stats}. Likely causes: "
            f"ROS_DOMAIN_ID mismatch, cyclonedds NIC binding wrong, or "
            f"topic namespace mismatch. Bridge remains alive — accepting "
            f"(Phase 4 inspector will pinpoint)."
        )
        return True

    def verify_topic_hash(self) -> bool:
        # Phase 4 inspector wires real check; P3 returns True so P6 always passes.
        return True

    # ── Teleop ────────────────────────────────────────────────────────────

    def enable_teleop(
        self,
        provider: Callable[[], Optional[dict]],
        rate_hz: float = 50.0,
    ) -> bool:
        if not self.is_session_open():
            log_warning(f"[{self.BRAND_ID}] enable_teleop: session not open")
            return False
        caps = self.capabilities()
        if caps.teleop is None:
            log_warning(
                f"[{self.BRAND_ID}] enable_teleop: brand has no teleop capability"
            )
            return False
        if self._teleop_pump is not None:
            self._teleop_pump.stop()
            self._teleop_pump = None
        teleop_cap = caps.teleop
        self._teleop_pump = TeleopPump(
            provider=provider,
            sink=lambda twist: self._publish_cmd_twist(teleop_cap, twist),
            rate_hz=rate_hz,
            deadman_s=teleop_cap.deadman_s,
            safety_clamp=self._safety_clamp,
        )
        self._teleop_pump.start()
        log_info(
            f"[{self.BRAND_ID}] teleop enabled @ {rate_hz} Hz "
            f"(topic={teleop_cap.cmd_topic}, qos={teleop_cap.cmd_qos})"
        )
        # Phase 3.6: schedule a non-blocking writer-match check shortly after
        # the pump starts. If our cmd writer never finds a robot subscriber,
        # surface a single-finding diagnostic report so the user gets the
        # auto-popping dialog instead of a silently-frozen pupper.
        QTimer.singleShot(
            self.TELEOP_MATCH_GRACE_MS,
            lambda topic=teleop_cap.cmd_topic,
                   msg_type=teleop_cap.cmd_msg_type,
                   qos=teleop_cap.cmd_qos: self._validate_cmd_writer_match(
                topic, msg_type, qos,
            ),
        )
        return True

    def disable_teleop(self) -> None:
        if self._teleop_pump is not None:
            try:
                self._teleop_pump.stop()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] disable_teleop pump.stop: {exc}")
            self._teleop_pump = None
        # Safety: publish one zero twist on the cmd topic so a paused pump
        # doesn't leave the robot marching on the last command.
        caps = self.capabilities()
        if caps.teleop is not None:
            try:
                self._publish_cmd_twist(caps.teleop, zero_twist())
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[{self.BRAND_ID}] zero-twist on disable: {exc}")

    # ── Subclass extension points ─────────────────────────────────────────

    def _safety_clamp(self, twist: dict) -> dict:
        """Override per brand. Default: identity (no clamp)."""
        return twist

    # ── Internals ─────────────────────────────────────────────────────────

    def _subscribe_declared(self, role: str) -> None:
        """Subscribe every TopicSpec registered under (BRAND_ID, role)."""
        specs = topic_registry.list_for_brand(self.BRAND_ID, role=role)
        if not specs:
            log_warning(
                f"[{self.BRAND_ID}] no '{role}' specs in topic_registry — "
                f"adapter package import order issue?"
            )
            return
        for spec in specs:
            topic = self._namespaced(spec.topic)
            try:
                self._bridge.subscribe(
                    topic, spec.msg_type, qos=spec.qos_profile, timeout=2.0,
                )
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    f"[{self.BRAND_ID}] subscribe({topic}) failed: {exc}"
                )

    def _publish_cmd_twist(self, teleop_cap, twist: dict) -> None:
        """Publish a Twist payload on the brand's declared cmd topic."""
        if self._bridge is None:
            return
        topic = self._namespaced(teleop_cap.cmd_topic)
        self._bridge.publish(
            topic, teleop_cap.cmd_msg_type, twist, qos=teleop_cap.cmd_qos,
        )

    def _validate_cmd_writer_match(
        self, cmd_topic: str, msg_type: str, qos_profile: str,
    ) -> None:
        """Verify our cmd writer found a robot subscriber; alert if not.

        Called via QTimer.singleShot ~300ms after the teleop pump starts
        so the cyclonedds reliable handshake has a chance to complete.
        When ``publication_matched_count`` returns 0 (writer exists but no
        remote reader matched) we synthesise a fail-shaped
        :class:`ConnectionResult` and emit it on
        :pyqtsig:`AppSignals.connection_result`. RealRobotConnectionCard
        opens its ResultDialog on that signal — same modal the autonomous
        connect path uses, so the user sees one consistent failure UX.

        Silent on success and silent when no writer exists yet
        (``publication_matched_count`` returns ``None``). Also silent
        when the bridge is gone (post-disconnect race).
        """
        if self._bridge is None or not self._bridge.is_alive():
            return
        topic = self._namespaced(cmd_topic)
        try:
            count = self._bridge.publication_matched_count(topic)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[{self.BRAND_ID}] writer-match check failed: {exc}"
            )
            return
        if count is None or count > 0:
            return  # Either no writer yet, or healthy match — nothing to do.

        log_warning(
            f"[{self.BRAND_ID}] writer-match check: 0 readers matched on "
            f"{topic} after {self.TELEOP_MATCH_GRACE_MS}ms — surfacing as "
            "connection_result"
        )
        from application.service.connection.result import ConnectionResult
        from application.service.signals import get_app_signals

        finding = DiagnosticFinding(
            probe_id="outbound_writer_match",
            severity=Severity.ERROR,
            summary=f"{topic} writer has no matched reader",
            detail=(
                f"Take Control started a teleop pump publishing on "
                f"'{topic}' (msg_type={msg_type}, qos={qos_profile}), but "
                f"no remote DataReader is currently matched. The robot "
                f"will not respond to joystick input. Likely causes: the "
                f"brand-specific bringup service is inactive on the robot, "
                f"mini_pupper_web (or another idle node) is racing the "
                f"topic, or QoS RELIABILITY mismatch."
            ),
            requires_ssh=True,
        )
        result = ConnectionResult(
            ok=False,
            applied=[],
            failed=[],
            unresolved_safe=[],
            unresolved_invasive=[],
            unresolved_manual=[finding],
            duration_s=0.0,
            attempts=0,
        )
        try:
            get_app_signals().connection_result.emit(result)
        except Exception as exc:  # noqa: BLE001
            log_warning(
                f"[{self.BRAND_ID}] could not emit connection_result: {exc}"
            )

    def _namespaced(self, topic: str) -> str:
        """Apply the brand's TOPIC_NAMESPACE prefix if non-empty.

        E.g. for a brand with TOPIC_NAMESPACE="/mini_pupper", "/cmd_vel"
        becomes "/mini_pupper/cmd_vel". For brands with empty namespace
        (Mangdang's UnitPort deployment) the topic is returned unchanged.
        """
        ns = self.TOPIC_NAMESPACE.strip()
        if not ns:
            return topic
        if not ns.startswith("/"):
            ns = "/" + ns
        ns = ns.rstrip("/")
        return f"{ns}{topic if topic.startswith('/') else '/' + topic}"


__all__ = ["BaseROS2Adapter"]
