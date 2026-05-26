# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""WebRTCConnectTask -- brownfield connect over the WebRTC data channel.

Sibling of :class:`application.service.ros2_connection_config.Ros2BrownfieldConnectTask`
but for brands that route the host link through a vendor WebRTC SDK
instead of ROS2/DDS. Today the only such brand is Unitree Go2 Air
(Unitree does not expose DDS for the Air SKU); the task is brand-neutral
and dispatches purely on transport.

Phase mapping (re-using the existing :class:`ConnectionPhase` IDs so the
UI phase bar + label work without any per-task knowledge):

  P0_USB_IDENTITY_PROBE   real -> webrtc_ip_reachable (TCP probe of 8081/9991)
  P1..P4C                 skipped (SSH / mode-switch / inspector irrelevant)
  P5_BRIDGE_UP            real -> adapter.open_session(profile, None)
                                  (adapter internally drives WebRTC
                                  negotiation, validation, lowstate sub)
  P6_MAPPING_HASH_CHECK   skipped (no ROS graph / no topic hash)
  P7_ACTIVATE             real -> adapter.activate() (StandUp)
  P8_CONFIRMATION         real -> adapter.confirm_steady_state(5.0)
  P9_DIAGNOSTICS          skipped (deferred to WP5 follow-up)

Signal contract (emits to :func:`get_app_signals`):

  connection_state_changed("connecting" | "connected" | "error" | "disconnected", info)
  connection_phase_changed(phase_id, "started"|"success"|"error"|"skipped", msg)
  connection_result(ConnectionResult)  on error paths only (WP5 will wire
                                       a full result on the success path
                                       once diagnostics arrive).
"""

from __future__ import annotations

import socket
from dataclasses import asdict
from typing import Any, Dict, Optional

from unitport_sdk import (
    Task,
    TaskCancelledException,
    log_debug,
    log_error,
    log_info,
    log_warning,
)

from application.service.adapters import (
    AdapterFactory,
    AdapterUnavailable,
    BaseAdapter,
)
from application.service.connection.phases import (
    BROWNFIELD_SEQUENCE,
    ConnectionPhase,
)
from application.service.connection.profile import ConnectionProfile
from application.service.connection.result import ConnectionResult
from application.service.signals import get_app_signals


# Per-phase budget (seconds) ------------------------------------------------

_TCP_PROBE_TIMEOUT_S = 1.5
_CONFIRMATION_TIMEOUT_S = 5.0

# Probe targets for P0 -- the firmware listens on 8081 (v1.0) and 9991
# (v1.1.8+). Either being open is enough to proceed: P5 will sort out
# which signaling path actually works.
_WEBRTC_PROBE_PORTS = (8081, 9991)


# Skip reasons -------------------------------------------------------------

_SKIP_REASONS: Dict[str, str] = {
    ConnectionPhase.P1_SSH_HANDSHAKE.value:        "webrtc_no_ssh",
    ConnectionPhase.P2_MODE_SWITCH.value:          "webrtc_no_mode_switch",
    ConnectionPhase.P3_INSPECTOR.value:            "webrtc_no_inspector",
    ConnectionPhase.P4_ROBOT_PROFILE_PUSH.value:   "webrtc_no_profile_push",
    ConnectionPhase.P4B_BRAND_RUNTIME.value:       "webrtc_no_brand_runtime",
    ConnectionPhase.P4C_ENTER_ROS2_MODE.value:     "webrtc_no_ros2_mode",
    ConnectionPhase.P6_MAPPING_HASH_CHECK.value:   "webrtc_no_topic_hash",
    ConnectionPhase.P9_DIAGNOSTICS.value:          "webrtc_diagnostics_deferred",
}


class WebRTCConnectTask(Task):
    """Connect task for vendor-WebRTC adapters (currently Unitree Go2 Air)."""

    def __init__(self, info: Any, profile: ConnectionProfile) -> None:
        super().__init__("WebRTC Connect")
        self._info = info
        self._profile = profile
        self._adapter: Optional[BaseAdapter] = None
        # Kept for API parity with Ros2BrownfieldConnectTask; the WebRTC
        # path does not use the Transport abstraction (the adapter owns
        # its peer connection internally), so this always stays None.
        self._transport = None

    # ------------------------------------------------------------------
    # Properties consumed by Ros2ConnectionController
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> Optional[BaseAdapter]:
        return self._adapter

    @property
    def transport(self):
        return self._transport

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> str:
        signals = get_app_signals()

        # Resolve adapter up front so a missing brand/model surfaces
        # before any I/O fires. The strategy field is honoured for
        # parity with the ROS2 task; the Go2 Air SKU declares its
        # adapter via robot.adapter so the default "ros2_generic"
        # path resolves the WebRTC adapter cleanly.
        try:
            self._adapter = AdapterFactory.build_for_profile(
                self._profile,
                strategy=getattr(self._info, "adapter_strategy", "ros2_generic"),
            )
        except AdapterUnavailable as exc:
            log_error(
                f"[webrtc-connect] adapter unavailable: "
                f"{exc.error_code} -- {exc}"
            )
            signals.connection_state_changed.emit(
                "error",
                {"reason": str(exc), "code": exc.error_code,
                 "phase": "adapter_resolve"},
            )
            return "error"

        signals.connection_state_changed.emit(
            "connecting", self._info.as_dict(),
        )
        try:
            for phase in BROWNFIELD_SEQUENCE:
                self.check_cancelled()
                pid = phase.value
                if pid in _SKIP_REASONS:
                    signals.connection_phase_changed.emit(
                        pid, "skipped", _SKIP_REASONS[pid],
                    )
                    log_debug(
                        f"[webrtc-connect] {pid} skipped ({_SKIP_REASONS[pid]})"
                    )
                    continue

                signals.connection_phase_changed.emit(
                    pid, "started", f"running {pid}",
                )
                log_debug(f"[webrtc-connect] {pid} started")

                if phase is ConnectionPhase.P0_USB_IDENTITY_PROBE:
                    self._do_p0_ip_reachable()
                elif phase is ConnectionPhase.P5_BRIDGE_UP:
                    self._do_p5_open_session()
                elif phase is ConnectionPhase.P7_ACTIVATE:
                    self._do_p7_activate()
                elif phase is ConnectionPhase.P8_CONFIRMATION:
                    self._do_p8_confirmation()
                else:
                    # Defensive: any phase that is not in _SKIP_REASONS
                    # and not handled explicitly above should not exist;
                    # log and treat as success so the bar advances.
                    log_warning(
                        f"[webrtc-connect] unhandled phase {pid}; "
                        "emitting success without action"
                    )

                signals.connection_phase_changed.emit(pid, "success", "ok")
                log_debug(f"[webrtc-connect] {pid} success")

        except TaskCancelledException:
            log_warning("[webrtc-connect] cancelled by user")
            self._teardown()
            signals.connection_state_changed.emit(
                "disconnected", {"cancelled": True},
            )
            raise
        except Exception as exc:  # noqa: BLE001 -- wrap for UI
            log_error(f"[webrtc-connect] failed: {exc}")
            signals.connection_phase_changed.emit(
                ConnectionPhase.P5_BRIDGE_UP.value, "error", str(exc),
            )
            signals.connection_state_changed.emit(
                "error", {"reason": str(exc)},
            )
            self._teardown()
            self._emit_failed_result(f"webrtc connect failed: {exc}")
            return "error"

        # Success -- emit terminal "connected" with the resolved profile
        # so Phase 6 consumers can read brand/robot/ip without re-querying.
        payload = self._info.as_dict()
        payload["profile"] = asdict(self._profile)
        signals.connection_state_changed.emit("connected", payload)
        log_info(
            f"[webrtc-connect] connected: brand={self._profile.brand!r} "
            f"robot={self._profile.robot!r} ip={self._profile.pupper_ip!r}"
        )
        return "connected"

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _do_p0_ip_reachable(self) -> None:
        """P0 -- TCP-probe the WebRTC signaling endpoints.

        The Go2 firmware listens on 8081 (v1.0 path) and 9991 (v1.1.8+
        path). Either one being open is sufficient to proceed; P5 picks
        the right signaling URL inside ``Go2Connection.connect_robot``.
        """
        ip = (self._profile.pupper_ip or "").strip()
        if not ip:
            raise RuntimeError(
                "WebRTC requires a non-empty IP -- enter the robot's "
                "LAN address in the Transport row"
            )
        open_ports = []
        for port in _WEBRTC_PROBE_PORTS:
            try:
                with socket.create_connection(
                    (ip, port), timeout=_TCP_PROBE_TIMEOUT_S
                ):
                    open_ports.append(port)
            except OSError:
                continue
        if not open_ports:
            raise RuntimeError(
                f"webrtc signaling ports closed at {ip} "
                f"(tried {list(_WEBRTC_PROBE_PORTS)}) -- check robot IP / "
                "WiFi link / firmware version"
            )
        log_info(
            f"[webrtc-connect] {ip}: signaling ports open {open_ports}"
        )

    def _do_p5_open_session(self) -> None:
        """P5 -- adapter negotiates WebRTC and arms its subscriptions.

        The adapter (e.g. :class:`Go2AirWebRTCAdapter`) drives:
        SDP offer / answer, AES handshake (v1.1.8+), data channel open,
        validation challenge, lowstate subscribe. Failures surface as
        :class:`RuntimeError` from ``open_session``; the outer try/except
        wraps them into the UI error path.
        """
        if self._adapter is None:
            raise RuntimeError("P5 reached without a resolved adapter")
        self._adapter.open_session(self._profile, transport=None)

    def _do_p7_activate(self) -> None:
        """P7 -- adapter activates (StandUp for Go2 Air)."""
        if self._adapter is None:
            raise RuntimeError("P7 reached without a resolved adapter")
        self._adapter.activate()

    def _do_p8_confirmation(self) -> None:
        """P8 -- adapter confirms steady-state (lowstate sample arrived)."""
        if self._adapter is None:
            raise RuntimeError("P8 reached without a resolved adapter")
        ok = self._adapter.confirm_steady_state(
            timeout_s=_CONFIRMATION_TIMEOUT_S,
        )
        if not ok:
            raise RuntimeError(
                "adapter steady-state confirmation failed "
                "(no rt/lf/lowstate sample within "
                f"{_CONFIRMATION_TIMEOUT_S:.0f}s)"
            )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Release the adapter (which owns the bridge + peer connection).

        Adapter ``close_session`` walks bridge.stop() so the asyncio loop
        thread + aiortc peer connection are torn down cleanly. Idempotent;
        safe to call from cancel paths, error paths, and the controller's
        defensive ``disconnect()`` branch.
        """
        if self._adapter is not None:
            try:
                self._adapter.close_session()
            except Exception as exc:  # noqa: BLE001 -- best-effort teardown
                log_warning(
                    f"[webrtc-connect] adapter close_session raised: {exc}"
                )

    # ------------------------------------------------------------------
    # Error-path result emission (parity with Ros2BrownfieldConnectTask)
    # ------------------------------------------------------------------

    def _emit_failed_result(self, reason: str) -> None:
        """Synthesise a fail-shaped ConnectionResult so the UI dialog pops.

        The WP5 diagnostics work will replace this with a real diagnostic
        report walk; for now it just makes failure visible to the user
        instead of leaving the result dialog dormant.
        """
        from application.service.diagnostics.results import (
            DiagnosticFinding,
            Severity as _Sev,
        )
        finding = DiagnosticFinding(
            probe_id="webrtc_connect",
            severity=_Sev.ERROR,
            summary=reason,
            detail="",
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
                f"[webrtc-connect] connection_result emit failed: {exc}"
            )


__all__ = ["WebRTCConnectTask"]
