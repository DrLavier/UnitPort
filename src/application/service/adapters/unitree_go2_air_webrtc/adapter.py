# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Go2AirWebRTCAdapter -- Unitree Go2 Air host-side adapter.

Sibling of :class:`MangdangROS2Adapter` but sits on a different transport
stack: instead of the cyclonedds bridge, it owns a :class:`Go2WebRTCBridge`
that wraps the RoboVerse ``go2_webrtc`` library, and translates UnitPort
brand-agnostic commands to Unitree's sport-API JSON envelopes.

Why subclass :class:`BaseAdapter` directly (not :class:`BaseROS2Adapter`):
the Go2 Air sport API rides a single negotiated WebRTC data channel and
does not speak DDS, ROS topic naming, or any standard message type. The
ROS2 base class would force a domain_id, cyclonedds participant, and
ROS-style topic munging that have no semantic content on this link.
Unitree does not expose a DDS surface for the Air SKU, which is why
this adapter exists in the first place.
"""

from __future__ import annotations

import datetime
import json
import random
import threading
import time
from typing import Any, Callable, Dict, Optional

from unitport_sdk import log_info, log_warning

from application.service.adapters.base_adapter import BaseAdapter
from application.service.adapters.capabilities import (
    CapabilitySet,
    EstopCapability,
    TeleopCapability,
)
from application.service.adapters.teleop_pump import TeleopPump
from application.service.adapters.unitree_go2_air_webrtc.safety import (
    MAX_VX_MPS,
    MAX_VY_MPS,
    MAX_VYAW_RPS,
    clamp_go2_air_twist,
)
from application.service.connection.profile import ConnectionProfile
from application.service.connection.transport.base import Transport
from application.service.runtime.webrtc import Go2WebRTCBridge


# Sport API constants -----------------------------------------------------

SPORT_TOPIC = "rt/api/sport/request"
LOWSTATE_TOPIC = "rt/lf/lowstate"

SPORT_DAMP = 1001
SPORT_STOP_MOVE = 1003
SPORT_STAND_UP = 1004
SPORT_STAND_DOWN = 1005
SPORT_MOVE = 1008
SPORT_HELLO = 1016


def _next_sport_id() -> int:
    """Match ``Go2Connection.generate_id`` so server-side id-uniqueness holds."""
    return int(
        datetime.datetime.now().timestamp() * 1000 % 2147483648
    ) + random.randint(0, 999)


def _sport_payload(
    api_id: int,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON ``data`` for an ``rt/api/sport/request`` envelope.

    Matches the firmware convention: no-arg commands carry
    ``parameter = json.dumps(api_id)`` (api_id as a JSON-encoded integer);
    arg commands carry ``parameter = json.dumps(params)``.
    """
    if params is None:
        parameter = json.dumps(api_id)
    else:
        parameter = json.dumps(params)
    return {
        "header": {"identity": {"id": _next_sport_id(), "api_id": api_id}},
        "parameter": parameter,
    }


# Adapter ------------------------------------------------------------------


class Go2AirWebRTCAdapter(BaseAdapter):
    """Host-side adapter for Unitree Go2 Air over WebRTC sport API."""

    BRAND_ID: str = "unitree_go2_air"
    # No DDS domain on this link; kept for BaseAdapter shape only.
    DEFAULT_DOMAIN_ID: int = 0

    # Stand-up / stand-down hold times (seconds) -- chosen to leave the
    # robot mechanically settled before the next command. Tune per
    # environment if you observe overlapping motions.
    ACTIVATE_HOLD_S: float = 2.0
    DEACTIVATE_HOLD_S: float = 1.5

    # WebRTC negotiation budgets.
    CONNECT_TIMEOUT_S: float = 12.0
    DATA_CHANNEL_OPEN_TIMEOUT_S: float = 10.0
    VALIDATION_TIMEOUT_S: float = 5.0

    # Teleop pump cadence -- 50 Hz matches the Go2 sport stack's expected
    # command rate; the deadman matches Mangdang's value so the host fires
    # first before any robot-side watchdog (firmware-side estop kicks in
    # around 1 s of silence).
    TELEOP_RATE_HZ: float = 50.0
    TELEOP_DEADMAN_S: float = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._bridge: Optional[Go2WebRTCBridge] = None
        self._profile: Optional[ConnectionProfile] = None
        self._lock = threading.RLock()
        self._latest_lowstate: Optional[Dict[str, Any]] = None
        self._lowstate_seen = threading.Event()
        self._teleop_pump: Optional[TeleopPump] = None

    # -- Session lifecycle -------------------------------------------------

    def open_session(
        self,
        profile: ConnectionProfile,
        transport: Optional[Transport] = None,
    ) -> None:
        """Negotiate WebRTC and arm subscriptions.

        ``transport`` is accepted for ABC compatibility but ignored --
        :class:`Go2WebRTCBridge` owns its own peer connection; the
        bootstrap path other brands use over SSH/USB does not apply.
        """
        if self.is_session_open():
            return

        ip = (profile.pupper_ip or "").strip()
        if not ip:
            raise RuntimeError(
                "Go2 Air WebRTC requires a non-empty IP in profile.pupper_ip"
            )
        # WebRTC token (per-account credential issued by the Unitree app)
        # is not yet wired into ConnectionProfile; treat the empty case as
        # "no token required" which matches Air firmware on a LAN-only
        # connection.
        token = str(getattr(profile, "webrtc_token", "") or "")

        bridge = Go2WebRTCBridge()
        bridge.start(ip, token=token, timeout=self.CONNECT_TIMEOUT_S)
        if not bridge.wait_open(timeout=self.DATA_CHANNEL_OPEN_TIMEOUT_S):
            bridge.stop()
            raise RuntimeError("Go2 Air WebRTC: data channel did not open")
        # Validation is best-effort: older v1.0 firmware skips it entirely.
        if not bridge.wait_validated(timeout=self.VALIDATION_TIMEOUT_S):
            log_warning(
                "[unitree_go2_air] validation timed out -- continuing "
                "(firmware may not require validation on this link)"
            )

        # Wire the lowstate subscription so confirm_steady_state has a
        # signal source and downstream telemetry consumers can read
        # latest() without re-subscribing.
        bridge.subscribe(LOWSTATE_TOPIC, self._on_lowstate)

        self._bridge = bridge
        self._profile = profile
        log_info(f"[unitree_go2_air] session open: ip={ip}")

    def close_session(self) -> None:
        if self._bridge is None:
            return
        # Stop the teleop pump first so a final tick cannot fight the
        # bridge teardown by trying to publish on a half-closed channel.
        self.disable_teleop()
        try:
            self._bridge.stop()
        except Exception as exc:
            log_warning(f"[unitree_go2_air] close raised: {exc}")
        self._bridge = None
        self._profile = None
        with self._lock:
            self._latest_lowstate = None
        self._lowstate_seen.clear()
        log_info("[unitree_go2_air] session closed")

    def is_session_open(self) -> bool:
        return self._bridge is not None and self._bridge.is_alive()

    # -- Capability declaration -------------------------------------------

    def capabilities(self) -> CapabilitySet:
        teleop = TeleopCapability(
            axes=("vx", "vy", "vyaw"),
            max_rates=(MAX_VX_MPS, MAX_VY_MPS, MAX_VYAW_RPS),
            deadman_s=0.5,
            # cmd_topic / cmd_msg_type are advisory: the sport API does
            # not use ROS-style topics. The labels exist so the UI topic
            # picker can render something meaningful next to the teleop
            # widget.
            cmd_topic="unitree/sport/move",
            cmd_msg_type="unitree_sport/Move",
            cmd_qos="best_effort",
        )
        estop = EstopCapability(
            topic="unitree/sport/damp",
            msg_type="unitree_sport/Damp",
            qos="best_effort",
        )
        return CapabilitySet(teleop=teleop, estop=estop)

    # -- Activation hooks (StandUp / StandDown / Damp) --------------------

    def activate(self) -> None:
        if not self.is_session_open():
            log_warning("[unitree_go2_air] activate before session open -- skipped")
            return
        log_info("[unitree_go2_air] activate: StandUp")
        self._send_sport(SPORT_STAND_UP)
        time.sleep(self.ACTIVATE_HOLD_S)

    def deactivate(self) -> None:
        if not self.is_session_open():
            return
        # Stop the teleop pump (WP4) before lowering -- a teleop tick
        # racing the StandDown would fight the motion.
        self.disable_teleop()
        log_info("[unitree_go2_air] deactivate: StandDown -> Damp")
        self._send_sport(SPORT_STAND_DOWN)
        time.sleep(self.DEACTIVATE_HOLD_S)
        self._send_sport(SPORT_DAMP)

    # -- Teleop -----------------------------------------------------------

    def enable_teleop(
        self,
        provider: Callable[[], Optional[Dict[str, Any]]],
        rate_hz: float = TELEOP_RATE_HZ,
    ) -> bool:
        """Pump live teleop twists to ``rt/api/sport/request`` (sport_id=1008).

        ``provider()`` returns a dict-Twist (``{"linear": {...},
        "angular": {...}}``) or ``None`` (deadman tick -- pump auto-pushes
        a zero twist after :attr:`TELEOP_DEADMAN_S` of silence). Each
        non-None value is clamped through :func:`clamp_go2_air_twist`
        before reaching the sport API, so a bad CommandBus value cannot
        send the robot past its datasheet envelope.

        Returns False when the bridge is not alive or a pump is already
        running -- callers should call :meth:`disable_teleop` first to
        swap providers.
        """
        if not self.is_session_open():
            log_warning(
                "[unitree_go2_air] enable_teleop before session open -- skipped"
            )
            return False
        if self._teleop_pump is not None and self._teleop_pump.is_running():
            log_warning(
                "[unitree_go2_air] enable_teleop: pump already running"
            )
            return False

        pump = TeleopPump(
            provider=provider,
            sink=self._publish_teleop_twist,
            rate_hz=float(rate_hz),
            deadman_s=self.TELEOP_DEADMAN_S,
            safety_clamp=clamp_go2_air_twist,
        )
        pump.start()
        self._teleop_pump = pump
        log_info(
            f"[unitree_go2_air] teleop pump started @ {rate_hz:.1f} Hz "
            f"(deadman={self.TELEOP_DEADMAN_S:.2f}s)"
        )
        return True

    def disable_teleop(self) -> None:
        """Stop the teleop pump (if running) and send StopMove + zero twist.

        Idempotent. Safe to call from :meth:`deactivate`,
        :meth:`close_session`, or directly from the Take Control card's
        toggle handler.
        """
        pump = self._teleop_pump
        self._teleop_pump = None
        if pump is not None:
            try:
                pump.stop()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[unitree_go2_air] teleop pump stop raised: {exc}")
        # Best-effort: push one zero-Move + StopMove so the robot does not
        # coast on the last received velocity vector. Both calls are no-ops
        # when the bridge already went down.
        if self.is_session_open():
            self._send_sport(SPORT_MOVE, {"x": 0.0, "y": 0.0, "z": 0.0})
            self._send_sport(SPORT_STOP_MOVE)

    def _publish_teleop_twist(self, twist: Dict[str, Any]) -> None:
        """Pump sink: dict-Twist -> sport Move (1008) JSON envelope.

        :class:`TeleopPump` calls this on every tick (~50 Hz). The twist
        is already safety-clamped by the pump; here we just unwrap the
        axes and translate to Unitree's ``{x, y, z}`` parameter shape
        (z is yaw-rate on the sport API, not vertical velocity).
        """
        if not self.is_session_open():
            return
        linear = twist.get("linear") if isinstance(twist.get("linear"), dict) else {}
        angular = twist.get("angular") if isinstance(twist.get("angular"), dict) else {}
        params = {
            "x": float(linear.get("x", 0.0) or 0.0),
            "y": float(linear.get("y", 0.0) or 0.0),
            "z": float(angular.get("z", 0.0) or 0.0),
        }
        self._send_sport(SPORT_MOVE, params)

    # -- Connection-phase support -----------------------------------------

    def confirm_steady_state(self, timeout_s: float = 5.0) -> bool:
        """True iff a lowstate sample has arrived since session open."""
        if not self.is_session_open():
            return False
        return self._lowstate_seen.wait(timeout=timeout_s)

    # -- Sport API helpers (used by activate / deactivate / WP4 teleop) ---

    def _send_sport(
        self,
        api_id: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a sport-API JSON envelope on ``rt/api/sport/request``.

        Returns False when the bridge is not alive so callers can branch
        on connection state without re-checking :meth:`is_session_open`.
        """
        if not self.is_session_open():
            return False
        assert self._bridge is not None
        return self._bridge.publish(
            topic=SPORT_TOPIC,
            data=_sport_payload(api_id, params),
            msg_type="msg",
        )

    def _on_lowstate(self, data: Any) -> None:
        with self._lock:
            self._latest_lowstate = data if isinstance(data, dict) else None
        self._lowstate_seen.set()

    def latest_lowstate(self) -> Optional[Dict[str, Any]]:
        """Snapshot of the most recent rt/lf/lowstate payload, or None."""
        with self._lock:
            return self._latest_lowstate


__all__ = ["Go2AirWebRTCAdapter"]
