# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service.ros2_connection_config — ROS2 实机连接配置 + 控制器.

职责拆为两部分：

1. ``Ros2ConnectionConfig``  — 配置项的读写门面。所有字段持久化到
   ``user.ini[RobotConn]``，通过 ``Config.set_value`` / ``Config.get_value``
   走 SDK 的 system.ini ⊕ user.ini overlay。字段：

       domain_id    (int, 0..255, default 0)
       middleware   (str, default "DDS (CycloneDDS)")
       address      (str, default "192.168.1.10")
       port         (int, default 11311)

2. ``Ros2ConnectionController`` — 单例 QObject，包装 test/connect/disconnect
   语义。当前 connect/disconnect 都是 stub（log_error WIP + emit error 状态），
   ``test()`` 用 socket 异步连一次目标地址做轻探针，几秒超时返回。所有 lifecycle
   事件通过 ``signals.connection_state_changed(state, info)`` 广播：

       state ∈ {"disconnected", "connecting", "connected", "error", "testing"}

UI 卡片只消费这两个 API；当真实 ROS2 桥接补齐时只需要替换 controller 内部实现，
不需要改 UI。
"""

from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from unitport_sdk import (
    Config,
    Task,
    TaskCancelledException,
    get_tasks_manager,
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)

from application.service.adapters import (
    AdapterFactory,
    AdapterUnavailable,
    BaseAdapter,
)
from application.service.auth.secure import get_secure_store
from application.service.connection.auto_repair_loop import AutoRepairLoop
from application.service.connection.phases import (
    BROWNFIELD_SEQUENCE,
    ConnectionPhase,
)
from application.service.connection.profile import ConnectionProfile
from application.service.connection.result import ConnectionResult
from application.service.connection.transport import (
    Transport,
    TransportUnavailable,
    select_data_transport,
)
from application.service.diagnostics import (
    DiagnosticContext,
    DiagnosticReport,
    DiagnosticsTask,
    RepairAction,
    RepairTask,
    Severity,
)
from application.service.diagnostics.probes import HOST_PROBES
from application.service.signals import get_app_signals


_SECTION = "RobotConn"

# Middleware 选项 — 与 ROS2 实际可选 RMW 实现对齐。展示在 combo 里。
# Phase 3.9: "Auto" leads the list and is the default. Auto = tolerant
# mode (don't push the robot to switch RMW; trust RTPS wire interop) plus
# best-effort SSH-based detection of the robot's actual RMW (filled into
# ``ConnectionProfile.detected_rmw`` by ``MangdangRmwDetectionProbe``).
MIDDLEWARE_CHOICES = (
    "Auto",
    "DDS (CycloneDDS)",
    "DDS (FastDDS)",
    "DDS (Connext)",
    "Zenoh",
)

# Phase 1+ — independent dropdowns for Transport (physical channel) and
# Adapter strategy (top-level adapter family). Sorted wired > wireless.
TRANSPORT_CHOICES = (
    "USB",
    "Ethernet",
    "WiFi (SSH)",
    "WebRTC",
)

# Adapter strategy — Phase 1 only persists the UI choice; AdapterFactory
# starts honouring it in Phase 5.
ADAPTER_CHOICES = (
    "ROS2 Generic",
    "Brand SDK",
)

DEFAULT_DOMAIN_ID = 0
DEFAULT_MIDDLEWARE = "Auto"
DEFAULT_ADDRESS = "192.168.1.10"
DEFAULT_PORT = 11311
DEFAULT_TRANSPORT = "USB"
DEFAULT_ADAPTER = "ROS2 Generic"
DEFAULT_BRAND = ""
DEFAULT_ROBOT_MODEL = ""

# Per-transport (address, port) factory defaults. The address+port pair is
# meaningful only inside the context of a transport (USB tether identity
# daemon vs Ethernet ROS_MASTER vs SSH login vs WebRTC signaling endpoint),
# so each transport keeps its own remembered value in user.ini under
# ``address_<slug>`` / ``port_<slug>`` keys (see ``transport_namespace``).
TRANSPORT_DEFAULTS: Dict[str, tuple] = {
    "USB":        ("192.168.55.1",  9999),   # identity-daemon HTTP endpoint
    "Ethernet":   ("192.168.1.10",  11311),  # legacy ROS_MASTER, retained
    "WiFi (SSH)": ("192.168.1.10",  22),     # SSH login over WLAN
    # WebRTC: no factory IP -- Go2 Air is WiFi-only and gets a different
    # address depending on STA vs AP mode + DHCP lease. The user types
    # the address shown in the Unitree app; the port field is unused
    # (the firmware listens on hardcoded 8081 / 9991, picked by P0).
    "WebRTC":     ("",              0),
}


# Fields that are not meaningful on a given transport. The card reads
# this map on every transport change and hides the matching rows so the
# user is not shown knobs that cannot affect the connection.
_TRANSPORT_HIDDEN_FIELDS: Dict[str, tuple] = {
    "WebRTC": ("middleware", "domain_id", "port"),
}


def transport_specific_visibility(transport: str) -> Dict[str, bool]:
    """Return ``{field_name: visible}`` for the per-transport UI fields.

    Field names match the card's own widget identifiers (``middleware``,
    ``domain_id``, ``port``, ...). Fields not mentioned are visible by
    default; only the ones in :data:`_TRANSPORT_HIDDEN_FIELDS` flip to
    ``False`` for transports that have no use for them.
    """
    hidden = set(_TRANSPORT_HIDDEN_FIELDS.get(str(transport), ()))
    return {
        "middleware": "middleware" not in hidden,
        "domain_id":  "domain_id"  not in hidden,
        "port":       "port"       not in hidden,
    }

# Socket probe timeout (seconds) used by ``test()`` — kept short so the UI
# does not block the user waiting for a hung TCP handshake.
_TEST_TIMEOUT_S = 2.0

# Per-phase sleep in the Phase 1 fake state machine (seconds). 11 phases
# total ~4.4 s walk so the UI progress bar advances visibly without being
# slow enough to frustrate. Phase 3 replaces this constant with real I/O
# durations driven by the live connector.
_FAKE_PHASE_SLEEP_S = 0.4


def transport_namespace(transport: str) -> str:
    """Slugify a transport label for use as an INI key suffix.

    "WiFi (SSH)" → "WiFi_SSH" — strips spaces and parens so the result is
    safe inside ``Config.set_value(_SECTION, address_<slug>, ...)`` regardless
    of the underlying ConfigParser's permitted-character set.
    """
    return (
        str(transport)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def address_key(transport: str) -> str:
    return f"address_{transport_namespace(transport)}"


def port_key(transport: str) -> str:
    return f"port_{transport_namespace(transport)}"


@dataclass
class Ros2ConnectionInfo:
    """Immutable snapshot of one connection configuration.

    Phase 3 extends the Phase 1+2 shape with two robot identity fields
    (``brand`` + ``robot_model``) that flow into ``ConnectionProfile`` and
    AdapterFactory. ``address`` / ``port`` remain per-transport-namespace.
    """

    domain_id: int
    middleware: str
    address: str
    port: int
    transport: str = DEFAULT_TRANSPORT
    adapter_strategy: str = DEFAULT_ADAPTER
    brand: str = DEFAULT_BRAND
    robot_model: str = DEFAULT_ROBOT_MODEL

    def as_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": int(self.domain_id),
            "middleware": str(self.middleware),
            "address": str(self.address),
            "port": int(self.port),
            "transport": str(self.transport),
            "adapter_strategy": str(self.adapter_strategy),
            "brand": str(self.brand),
            "robot_model": str(self.robot_model),
        }


# ---------------------------------------------------------------------------
# Config façade
# ---------------------------------------------------------------------------


class Ros2ConnectionConfig:
    """Read/write façade over the ``user.ini[RobotConn]`` section.

    All methods are stateless static helpers; every call delegates to
    ``Config`` (system.ini ⊕ user.ini overlay).

    Address+Port are persisted **per transport** under
    ``address_<slug>`` / ``port_<slug>`` keys so each connection method
    remembers its own endpoint (USB identity daemon vs Ethernet master vs
    SSH login vs WebRTC signaler are three distinct address spaces).
    """

    @staticmethod
    def load() -> Ros2ConnectionInfo:
        """Read the full configuration; missing fields fall back to defaults.

        The returned ``info.address`` / ``info.port`` are the values
        remembered for the currently-selected ``transport``.
        """
        domain_id = Config.get_value(
            _SECTION, "domain_id", DEFAULT_DOMAIN_ID, value_type=int,
        )
        middleware = Config.get_value(
            _SECTION, "middleware", DEFAULT_MIDDLEWARE,
        )
        transport = str(Config.get_value(
            _SECTION, "transport", DEFAULT_TRANSPORT,
        ))
        adapter_strategy = Config.get_value(
            _SECTION, "adapter_strategy", DEFAULT_ADAPTER,
        )
        brand = str(Config.get_value(_SECTION, "brand", DEFAULT_BRAND))
        robot_model = str(Config.get_value(_SECTION, "robot_model", DEFAULT_ROBOT_MODEL))
        address, port = Ros2ConnectionConfig.load_address_port(transport)
        return Ros2ConnectionInfo(
            domain_id=int(domain_id),
            middleware=str(middleware),
            address=address,
            port=port,
            transport=transport,
            adapter_strategy=str(adapter_strategy),
            brand=brand,
            robot_model=robot_model,
        )

    @staticmethod
    def save(info: Ros2ConnectionInfo) -> None:
        """Persist a full configuration into ``user.ini[RobotConn]``.

        Address+Port are written under the per-transport namespace so they
        do not stomp on values previously remembered for other transports.
        """
        Config.set_value(_SECTION, "domain_id", int(info.domain_id))
        Config.set_value(_SECTION, "middleware", str(info.middleware))
        Config.set_value(_SECTION, "transport", str(info.transport))
        Config.set_value(_SECTION, "adapter_strategy", str(info.adapter_strategy))
        Config.set_value(_SECTION, "brand", str(info.brand))
        Config.set_value(_SECTION, "robot_model", str(info.robot_model))
        Ros2ConnectionConfig.set_address_for(info.transport, info.address)
        Ros2ConnectionConfig.set_port_for(info.transport, info.port)

    @staticmethod
    def set_field(key: str, value: Any) -> None:
        """Single-field write for top-level non-namespaced keys.

        Use ``set_address_for`` / ``set_port_for`` for address+port — those
        live under per-transport namespaces and ``set_field("address", ...)``
        would write to the wrong key.
        """
        Config.set_value(_SECTION, str(key), value)

    @staticmethod
    def load_address_port(transport: str) -> tuple:
        """Return ``(address, port)`` remembered for the given transport.

        Falls back to ``TRANSPORT_DEFAULTS[transport]`` (then to the global
        ``DEFAULT_ADDRESS`` / ``DEFAULT_PORT`` for unknown transports).
        """
        addr_default, port_default = TRANSPORT_DEFAULTS.get(
            str(transport), (DEFAULT_ADDRESS, DEFAULT_PORT),
        )
        address = Config.get_value(
            _SECTION, address_key(transport), addr_default,
        )
        port = Config.get_value(
            _SECTION, port_key(transport), port_default, value_type=int,
        )
        return str(address), int(port)

    @staticmethod
    def set_address_for(transport: str, value: Any) -> None:
        """Persist ``address`` for the given transport namespace."""
        Config.set_value(_SECTION, address_key(transport), str(value))

    @staticmethod
    def set_port_for(transport: str, value: Any) -> None:
        """Persist ``port`` for the given transport namespace."""
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return
        Config.set_value(_SECTION, port_key(transport), ivalue)

    @staticmethod
    def set_robot_sku(sku: str) -> None:
        """Persist ``brand`` + ``robot_model`` from a single SKU.

        The Phase 3.8 sec1 Robot combo enumerates SKUs directly; downstream
        code (ConnectionProfile, AdapterFactory, SSH server key) still keys
        off ``brand`` + ``robot_model``, so we split the SKU on the way in.
        Empty SKU clears both fields. Unknown SKU is a no-op.
        """
        from registers import robots as _robots_reg  # avoid import cycle

        if not sku:
            Config.set_value(_SECTION, "brand", "")
            Config.set_value(_SECTION, "robot_model", "")
            return
        spec = _robots_reg.get_robot(str(sku))
        if spec is None:
            return
        Config.set_value(
            _SECTION, "brand",
            str(spec.get("brand", "")).strip().lower(),
        )
        Config.set_value(
            _SECTION, "robot_model",
            str(spec.get("model", "")).strip(),
        )

    @staticmethod
    def get_robot_sku() -> str:
        """Reverse of :meth:`set_robot_sku` — derive SKU from current config.

        Returns an empty string when brand or robot_model is unset, or when
        the brand+model pair does not resolve to a registered SKU.
        """
        from registers import robots as _robots_reg  # avoid import cycle

        info = Ros2ConnectionConfig.load()
        if not info.brand or not info.robot_model:
            return ""
        sku = _robots_reg.resolve_id(f"{info.brand}.{info.robot_model}")
        return sku or ""


# ---------------------------------------------------------------------------
# Worker thread for non-blocking TCP probe
# ---------------------------------------------------------------------------


class _ProbeWorker(QThread):
    """子线程跑 socket connect — 主线程不阻塞."""

    finished_with_result = pyqtSignal(bool, str)        # (ok, message)

    def __init__(self, host: str, port: int, timeout_s: float, parent=None) -> None:
        super().__init__(parent)
        self._host = str(host)
        self._port = int(port)
        self._timeout = float(timeout_s)

    def run(self) -> None:  # noqa: D401 — Qt thread entry
        try:
            with socket.create_connection(
                (self._host, self._port), timeout=self._timeout
            ):
                self.finished_with_result.emit(
                    True, f"reachable: {self._host}:{self._port}"
                )
        except OSError as exc:
            self.finished_with_result.emit(False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Controller — singleton
# ---------------------------------------------------------------------------


class Ros2BrownfieldConnectTask(Task):
    """Real-bridge brownfield connect task — Phase 3 adapter-driven.

    Walks the 11-phase brownfield sequence, delegating real I/O to the
    brand adapter resolved from ``profile.brand`` + ``profile.robot``.
    SSH/inspector phases (P1-P4C) emit ``"skipped"`` with reasons in the
    SKIPPED_REASONS map; Phase 4 fills them by porting the SSH transport
    + ros2_generic_connector inspector subsystem.

    Phase mapping:
      P0_USB_IDENTITY_PROBE   real -> USBDataTransport.connect()
      P1..P4C                 skipped -> Phase 4
      P5_BRIDGE_UP            real -> adapter.open_session(profile, transport)
      P6_MAPPING_HASH_CHECK   real (best-effort) -> adapter.verify_topic_hash()
      P7_ACTIVATE             real -> adapter.activate()
      P8_CONFIRMATION         real -> adapter.confirm_steady_state(5.0)

    The adapter owns the bridge + transport lifecycle internally; this task
    coordinates phase emission and surface-level error wrapping. On cancel
    or hard error the task delegates teardown to ``adapter.close_session()``
    so cyclonedds DDS resources are released cleanly.
    """

    SKIPPED_REASONS = {
        ConnectionPhase.P1_SSH_HANDSHAKE.value:        "ssh_arrives_in_phase4",
        ConnectionPhase.P2_MODE_SWITCH.value:          "mode_switch_arrives_in_phase4",
        ConnectionPhase.P3_INSPECTOR.value:            "inspector_arrives_in_phase4",
        ConnectionPhase.P4_ROBOT_PROFILE_PUSH.value:   "profile_push_arrives_in_phase4",
        ConnectionPhase.P4B_BRAND_RUNTIME.value:       "brand_runtime_already_running",
        ConnectionPhase.P4C_ENTER_ROS2_MODE.value:     "already_in_ros2_mode",
    }

    # P8 confirmation budget: forwarded to adapter.confirm_steady_state.
    _CONFIRMATION_TIMEOUT_S = 5.0

    def __init__(self, info: Ros2ConnectionInfo, profile: ConnectionProfile) -> None:
        super().__init__("Ros2 Brownfield Connect")
        self._info = info
        self._profile = profile
        self._transport: Optional[Transport] = None
        self._adapter: Optional[BaseAdapter] = None
        # Active AutoRepairLoop instance during P9 — the controller routes
        # SshCredentialPromptDialog responses to this loop via
        # ``submit_ssh_response``. Set/cleared inside _do_p9_diagnostics.
        self._auto_repair: Optional[AutoRepairLoop] = None

    @property
    def auto_repair(self) -> Optional[AutoRepairLoop]:
        """Active AutoRepairLoop (or None) — used by controller SSH routing."""
        return self._auto_repair

    def run(self) -> str:
        signals = get_app_signals()
        # Resolve adapter early so the UI sees an instantiation failure on
        # P0 (cleanest place to surface "you forgot to pick a brand+model")
        # rather than half-walking the phase list.
        try:
            self._adapter = AdapterFactory.build_for_profile(
                self._profile, strategy=self._info.adapter_strategy,
            )
        except AdapterUnavailable as exc:
            log_error(
                f"[connect] adapter unavailable: {exc.error_code} — {exc}"
            )
            signals.connection_state_changed.emit(
                "error",
                {"reason": str(exc), "code": exc.error_code, "phase": "adapter_resolve"},
            )
            return "error"

        # Emit "connecting" first so the UI flips state before any phase fires.
        signals.connection_state_changed.emit("connecting", self._info.as_dict())
        try:
            for phase in BROWNFIELD_SEQUENCE:
                self.check_cancelled()
                pid = phase.value
                if pid in self.SKIPPED_REASONS:
                    reason = self.SKIPPED_REASONS[pid]
                    signals.connection_phase_changed.emit(pid, "skipped", reason)
                    log_debug(f"[connect] {pid} skipped ({reason})")
                    continue
                signals.connection_phase_changed.emit(pid, "started", f"running {pid}")
                log_debug(f"[connect] {pid} started")
                if phase is ConnectionPhase.P0_USB_IDENTITY_PROBE:
                    self._do_p0_identity_probe()
                elif phase is ConnectionPhase.P5_BRIDGE_UP:
                    self._do_p5_bridge_up()
                elif phase is ConnectionPhase.P6_MAPPING_HASH_CHECK:
                    self._do_p6_mapping_hash()
                elif phase is ConnectionPhase.P7_ACTIVATE:
                    self._do_p7_activate()
                elif phase is ConnectionPhase.P8_CONFIRMATION:
                    self._do_p8_confirmation()
                elif phase is ConnectionPhase.P9_DIAGNOSTICS:
                    # P9 has its own status policy: it never raises
                    # connect into "error" — diagnose findings surface via
                    # connection_result and the connection stays up.
                    diag_status, diag_msg = self._do_p9_diagnostics()
                    signals.connection_phase_changed.emit(pid, diag_status, diag_msg)
                    if diag_status == "error":
                        log_error(f"[connect] {pid} {diag_status}: {diag_msg}")
                    elif diag_status == "warning":
                        log_warning(f"[connect] {pid} {diag_status}: {diag_msg}")
                    else:
                        log_debug(f"[connect] {pid} {diag_status} ({diag_msg})")
                    continue
                signals.connection_phase_changed.emit(pid, "success", "ok")
                log_debug(f"[connect] {pid} success")
        except TaskCancelledException:
            log_warning("[connect] cancelled by user")
            self._teardown()
            signals.connection_state_changed.emit("disconnected", {"cancelled": True})
            raise
        except TransportUnavailable as exc:
            log_error(
                f"[connect] transport unavailable: {exc.error_code} — {exc}"
            )
            signals.connection_phase_changed.emit(
                ConnectionPhase.P0_USB_IDENTITY_PROBE.value, "error",
                f"{exc.error_code}: {exc}",
            )
            signals.connection_state_changed.emit(
                "error", {"reason": str(exc), "code": exc.error_code},
            )
            self._teardown()
            # Phase 3.7: emit a fail-shaped ConnectionResult so the UI's
            # ResultDialog opens even when we never reached P9.
            self._emit_failed_result(
                f"transport unavailable: {exc.error_code}",
            )
            return "error"
        except Exception as exc:  # noqa: BLE001 — wrap for UI
            log_error(f"[connect] failed: {exc}")
            signals.connection_phase_changed.emit(
                BROWNFIELD_SEQUENCE[-1].value, "error", str(exc),
            )
            signals.connection_state_changed.emit("error", {"reason": str(exc)})
            self._teardown()
            self._emit_failed_result(f"connect failed: {exc}")
            return "error"

        # All real phases passed — emit terminal "connected" state. The
        # caller-visible payload includes the resolved profile so Phase 6
        # consumers can read brand/robot without re-querying the controller.
        payload = self._info.as_dict()
        payload["profile"] = asdict(self._profile)
        signals.connection_state_changed.emit("connected", payload)
        return "connected"

    def _emit_failed_result(self, reason: str) -> None:
        """Emit a fail-shaped ConnectionResult on the early-error paths.

        AutoRepairLoop normally publishes the result; pre-P9 failures bypass
        it, so we synthesise a manual-only fail finding so the UI's
        ResultDialog opens with a clear message instead of leaving the user
        with a silent red status bar.
        """
        from application.service.diagnostics.results import (
            DiagnosticFinding,
            Severity as _Sev,
        )
        finding = DiagnosticFinding(
            probe_id="connect",
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
            log_warning(f"[connect] connection_result emit failed: {exc}")

    # Adapter accessor — exposed so Ros2ConnectionController.current_adapter()
    # can hand the live adapter to the UI Controller sub-card.
    @property
    def adapter(self) -> Optional[BaseAdapter]:
        return self._adapter

    # Transport accessor — exposed so Ros2ConnectionController's health
    # watchdog can poll ``transport.probe_alive()`` to catch a physical
    # USB unplug while the bridge is up.
    @property
    def transport(self) -> Optional[Transport]:
        return self._transport

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _do_p0_identity_probe(self) -> None:
        """P0 — bring up the active DATA transport and verify identity."""
        self._transport = select_data_transport(
            self._profile, transport_label=self._info.transport,
        )
        self._transport.connect()  # raises TransportUnavailable on failure

    def _do_p5_bridge_up(self) -> None:
        """P5 — adapter opens its session over the active transport.

        The adapter internally builds the NativeDDSBridge, starts it,
        subscribes its declared "obs" topic table, and starts the heartbeat
        publisher. Failures surface as adapter-specific exceptions wrapped
        by the outer try/except.
        """
        if self._adapter is None:
            raise RuntimeError("P5 reached without a resolved adapter")
        if self._transport is None:
            raise RuntimeError("P5 reached without an active transport")
        self._adapter.open_session(self._profile, self._transport)

    def _do_p6_mapping_hash(self) -> None:
        """P6 — adapter best-effort verifies declared topics match the wire.

        Phase 3 default: returns True (no hard verification). Phase 4
        inspector wires a real check against the live ROS2 graph.
        """
        if self._adapter is None:
            raise RuntimeError("P6 reached without a resolved adapter")
        ok = self._adapter.verify_topic_hash()
        if not ok:
            # Best-effort: log warning, don't fail. P8 confirmation gate is
            # the actual "did we see data?" check.
            self.log_warning("P6: declared topic hash differs from live graph")

    def _do_p7_activate(self) -> None:
        """P7 — adapter activates the brand-specific controller-on hook."""
        if self._adapter is None:
            raise RuntimeError("P7 reached without a resolved adapter")
        self._adapter.activate()

    def _do_p8_confirmation(self) -> None:
        """P8 — adapter confirms steady-state (declared subs streaming)."""
        if self._adapter is None:
            raise RuntimeError("P8 reached without a resolved adapter")
        ok = self._adapter.confirm_steady_state(timeout_s=self._CONFIRMATION_TIMEOUT_S)
        if not ok:
            raise RuntimeError("adapter steady-state confirmation failed")

    def _do_p9_diagnostics(self) -> tuple:
        """P9 — autonomous diagnose + safe-repair loop.

        Phase 3.7 replaced the single-shot DiagnosticsTask with
        :class:`AutoRepairLoop`: it diagnoses, applies all safe repairs in
        place, and re-diagnoses up to N attempts until either the report
        passes or the loop runs out of safe fixes (escalating to the user
        via :pyqtsig:`AppSignals.connection_result`).

        Never raises — any internal failure becomes a fail-shaped
        ConnectionResult. Returns ``(phase_status, msg)`` for the outer
        phase emitter; status mapping:

          result.ok            -> "success"
          result.has_partial   -> "warning"
          (anything else)      -> "warning" (bridge IS up; just unresolved)
        """
        if self._adapter is None:
            self._emit_failed_result("adapter unavailable; skipping diagnostics")
            return ("warning", "adapter unavailable; skipping diagnostics")

        adapter = self._adapter
        profile = self._profile

        def _probes_factory():
            probes = [cls() for cls in HOST_PROBES]
            try:
                probes.extend(adapter.diagnostic_probes() or [])
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[connect] adapter.diagnostic_probes raised: {exc}")
            return probes

        def _ctx() -> DiagnosticContext:
            return DiagnosticContext(
                profile=profile,
                transport=getattr(adapter, "_transport", None),
                bridge=getattr(adapter, "_bridge", None),
                adapter=adapter,
                ssh=None,
            )

        server_key = ssh_server_name_for_profile(profile)
        suggested_user = profile.ssh_user or "ubuntu"
        loop = AutoRepairLoop(
            owner_task=self,
            probes_factory=_probes_factory,
            ctx_factory=_ctx,
            ssh_factory=build_ssh_session_for_profile,
            profile=profile,
            server_key=server_key,
            suggested_user=suggested_user,
            max_attempts=3,
        )
        # Expose so controller.submit_ssh_response can route the prompt
        # response back into the loop's threading.Event.
        self._auto_repair = loop
        try:
            try:
                result = loop.run()
            except TaskCancelledException:
                raise
            except Exception as exc:  # noqa: BLE001 — never break P9
                log_error(f"[connect] AutoRepairLoop crashed: {exc}")
                self._emit_failed_result(f"auto-repair crashed: {exc}")
                return ("error", f"auto-repair crashed: {exc}")
        finally:
            self._auto_repair = None

        # Always broadcast result so UI's ResultDialog pops exactly once.
        try:
            get_app_signals().connection_result.emit(result)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[connect] connection_result emit failed: {exc}")

        if result.ok:
            return ("success", f"auto-fixed {len(result.applied)} issue(s)")
        return (
            "warning",
            f"unresolved={result.total_unresolved} "
            f"applied={len(result.applied)} failed={len(result.failed)}",
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Release the adapter (which owns bridge + transport).

        Adapter ``close_session`` walks teleop pump → heartbeat → bridge →
        transport in reverse order so cyclonedds DDS handles are released
        cleanly. Idempotent — safe to call from cancel and from the
        controller's `disconnect()` defensive path.
        """
        if self._adapter is not None:
            try:
                self._adapter.close_session()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                self.log_warning(f"adapter close_session raised: {exc}")
            # The adapter owns the transport, so we drop the ref here too.
            self._transport = None


class Ros2ConnectionController(QObject):
    """Singleton façade over test() / connect() / disconnect() / current_adapter().

    Phase 3 behaviour:
      - ``test()``            async TCP probe against the active address:port.
      - ``connect()``         submits :class:`Ros2BrownfieldConnectTask` which walks
        the 11 brownfield phases driven by the brand adapter resolved from the
        active connection profile (brand+model). P5/P6/P7/P8 delegate to the
        adapter; SSH/inspector phases (P1-P4C) emit "skipped" until Phase 4.
      - ``disconnect()``      cancels the running connect task; the task's
        teardown delegates to the adapter for a clean close_session.
      - ``current_adapter()`` exposes the live BaseAdapter so UI consumers
        (Controller sub-card, telemetry views) can call enable_teleop /
        disable_teleop / capabilities without re-resolving via AdapterFactory.

    The UI signal contract is unchanged from Phase 1 / 2 — the same
    ``RealRobotConnectionCard._on_phase`` and ``_on_state`` slots consume
    every variant of the connect task interchangeably.
    """

    # Health watchdog — polls the active transport while connected so a
    # physical USB unplug is caught within ~10s without waiting for the
    # cyclonedds participant lease to expire.
    _HEALTH_PROBE_INTERVAL_MS = 5000  # 5s tick
    _HEALTH_FAIL_THRESHOLD = 2        # 2 consecutive failures → declare lost

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._current_state: str = "disconnected"
        self._probe: Optional[_ProbeWorker] = None
        # Task is the common base for both Ros2BrownfieldConnectTask and
        # WebRTCConnectTask; controller code only touches the shared
        # surface (.adapter / .transport / ._teardown / .cancel).
        self._connect_task: Optional[Task] = None
        self._connect_task_id: Optional[str] = None
        # Phase 3.7: subscribe to repair_finished so a successful invasive
        # deploy can auto-trigger a fresh connect — the deploy reboots the
        # robot, so the existing bridge/transport are gone and the user
        # would otherwise have to click [Connect] again.
        get_app_signals().diagnostics_repair_finished.connect(
            self._on_repair_finished
        )
        # Phase 3.10: subscribe to our own state-changed signal so we can
        # drive the health watchdog. The task-side emits go directly via
        # signals.connection_state_changed.emit(...) (bypassing _set_state)
        # so subscribing to the signal is the only place that catches both
        # paths uniformly.
        from PyQt6.QtCore import QTimer
        self._health_timer = QTimer(self)
        self._health_timer.setInterval(self._HEALTH_PROBE_INTERVAL_MS)
        self._health_timer.timeout.connect(self._on_health_tick)
        self._health_consecutive_failures = 0
        get_app_signals().connection_state_changed.connect(
            self._on_state_for_watchdog
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def current_state(self) -> str:
        return self._current_state

    def test(self) -> None:
        """Async TCP-reachability probe against the configured address:port.

        Idempotent: if a probe is already in flight, the new call is a noop.
        """
        if self._probe is not None and self._probe.isRunning():
            return
        info = Ros2ConnectionConfig.load()
        log_info(
            f"[ros2-conn] test {info.address}:{info.port} "
            f"(domain={info.domain_id}, mw={info.middleware!r})"
        )
        self._set_state("testing", {"address": info.address, "port": info.port})

        worker = _ProbeWorker(info.address, info.port, _TEST_TIMEOUT_S, self)
        worker.finished_with_result.connect(self._on_probe_finished)
        worker.finished.connect(self._cleanup_probe)
        self._probe = worker
        worker.start()

    def connect(self) -> bool:
        """Submit the brownfield connect task. Returns True on submit success.

        Dispatches by ``info.transport``: WebRTC routes to
        :class:`WebRTCConnectTask` (Unitree Go2 Air over the RoboVerse
        go2_webrtc library); every other transport routes to the
        ROS2/DDS brownfield task. The submitted task exposes the same
        ``.adapter`` / ``.transport`` / ``._teardown`` surface so the
        controller's disconnect + health-watchdog logic is transport-
        agnostic.

        If a connect task is already running, this call is a no-op
        (returns False); callers should ``disconnect()`` first.
        """
        if self._connect_task is not None and self._is_connect_task_running():
            log_warning("[ros2-conn] connect: another task already running")
            return False

        info = Ros2ConnectionConfig.load()
        profile = build_profile_from_info(info)
        log_info(
            f"[ros2-conn] connect: brand={info.brand!r}/{info.robot_model!r} "
            f"transport={info.transport!r} adapter={info.adapter_strategy!r} "
            f"address={info.address} domain={info.domain_id}"
        )

        if str(info.transport) == "WebRTC":
            # Deferred import to avoid pulling go2_webrtc into the top-of-
            # module ROS2 path; this file is imported during app bootstrap
            # and the SDK may not be installed yet on a fresh checkout.
            from application.service.connection.webrtc_connect_task import (
                WebRTCConnectTask,
            )
            task: Task = WebRTCConnectTask(info, profile)
        else:
            task = Ros2BrownfieldConnectTask(info, profile)

        try:
            task_id = get_tasks_manager().submit(task)
        except Exception as exc:  # noqa: BLE001
            log_error(f"[ros2-conn] connect submit failed: {exc}")
            self._set_state("error", {"reason": f"submit failed: {exc}"})
            return False

        self._connect_task = task
        self._connect_task_id = str(task_id) if task_id is not None else None
        # "connecting" is emitted by the task's run() itself, not here.
        return True

    def disconnect(self) -> None:
        """Cancel the running connect task (if any) and emit disconnected.

        Cancel: ``Ros2BrownfieldConnectTask.run()`` wraps the cooperative
        cancel in ``_teardown()`` which delegates to ``adapter.close_session``.
        After-completion: when the task already reached "connected" we still
        need to release adapter resources so the bridge participant is dropped
        — defensive ``_teardown`` covers that path.
        """
        log_info("[ros2-conn] disconnect")
        task = self._connect_task
        if task is not None:
            if self._is_connect_task_running():
                try:
                    task.cancel()
                except Exception as exc:  # noqa: BLE001
                    log_warning(f"[ros2-conn] cancel raised: {exc}")
            else:
                try:
                    task._teardown()
                except Exception as exc:  # noqa: BLE001
                    log_warning(f"[ros2-conn] teardown raised: {exc}")
        self._connect_task = None
        self._connect_task_id = None
        self._set_state("disconnected", {})

    def current_adapter(self) -> Optional[BaseAdapter]:
        """Return the active adapter once the connect task has resolved one.

        Returns ``None`` when no connect is in flight or the task has not yet
        reached the adapter-resolution step. UI consumers (sec3 Controller
        card) bind this lazily on the ``connection_state_changed("connected")``
        signal so they always pick up a live adapter.
        """
        task = self._connect_task
        if task is None:
            return None
        return task.adapter

    # ------------------------------------------------------------------
    # Phase 3.5 — diagnostics + repair + reconnect
    # ------------------------------------------------------------------

    def run_diagnostics(self) -> bool:
        """Manual [Diagnose] button entry point.

        Reuses the live adapter when present (so DDS discovery probe sees
        the actual bridge); falls back to host-only probes when nothing is
        connected. Result lands on ``AppSignals.diagnostics_ready``.
        Returns False when no diagnostics task could be submitted.
        """
        adapter = self.current_adapter()
        info = Ros2ConnectionConfig.load()
        profile = build_profile_from_info(info)
        if adapter is not None:
            try:
                brand_probes = list(adapter.diagnostic_probes() or [])
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[diag] brand probes raised: {exc}")
                brand_probes = []
        else:
            brand_probes = []
        probes = [cls() for cls in HOST_PROBES] + brand_probes

        def _ctx() -> DiagnosticContext:
            return DiagnosticContext(
                profile=profile,
                transport=getattr(adapter, "_transport", None) if adapter else None,
                bridge=getattr(adapter, "_bridge", None) if adapter else None,
                adapter=adapter,
                ssh=None,
            )

        task = DiagnosticsTask(
            probes=probes,
            ctx_factory=_ctx,
            ssh_factory=build_ssh_session_for_profile,
            name="Diagnostics (manual)",
        )
        try:
            get_tasks_manager().submit(task)
        except Exception as exc:  # noqa: BLE001
            log_error(f"[diag] submit failed: {exc}")
            return False
        try:
            get_app_signals().diagnostics_started.emit()
        except Exception:
            pass
        return True

    def apply_repair(self, actions, *, safe_only: bool) -> bool:
        """Submit a RepairTask. Result lands on ``diagnostics_repair_finished``.

        UI dialogs hand a list of :class:`RepairAction` objects (deduped
        from the report). ``safe_only`` filters by the action's ``safe``
        flag so the [Apply All Safe Fixes] button only runs idempotent
        host/restart-only fixes.
        """
        if not actions:
            return False
        adapter = self.current_adapter()
        info = Ros2ConnectionConfig.load()
        profile = build_profile_from_info(info)

        def _ctx() -> DiagnosticContext:
            return DiagnosticContext(
                profile=profile,
                transport=getattr(adapter, "_transport", None) if adapter else None,
                bridge=getattr(adapter, "_bridge", None) if adapter else None,
                adapter=adapter,
                ssh=None,
            )

        task = RepairTask(
            actions=list(actions),
            ctx_factory=_ctx,
            ssh_factory=build_ssh_session_for_profile,
            safe_only=bool(safe_only),
        )
        try:
            get_tasks_manager().submit(task)
        except Exception as exc:  # noqa: BLE001
            log_error(f"[repair] submit failed: {exc}")
            return False
        return True

    def _on_repair_finished(self, report) -> None:
        """Auto-reconnect when an invasive deploy succeeds.

        Triggered by ``AppSignals.diagnostics_repair_finished``. Looks for
        the deploy action by name in ``report.applied``; if present, the
        robot has rebooted and our prior bridge / transport are dead, so
        we kick off a fresh connect after a short delay (give the robot
        the recovery grace it already burned through inside deploy.run).
        """
        if report is None:
            return
        applied_names = {
            getattr(a, "name", "") for a in (getattr(report, "applied", []) or [])
        }
        if "mangdang.deploy_unitport_bringup" not in applied_names:
            return
        log_info(
            "[ros2-conn] deploy applied — auto-reconnecting in 1.5s"
        )
        # Use QTimer over time.sleep so we don't block the signal-emitter
        # thread (RepairTask is on a worker; we need to hop back to the
        # main thread anyway for connect()).
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1500, self.reconnect)

    def submit_ssh_response(self, server_key: str, ok: bool) -> None:
        """Forward an SshCredentialPromptDialog response into the active loop.

        Called by RealRobotConnectionCard after the modal closes. If no
        AutoRepairLoop is running we silently drop the response (the loop
        already moved on or never started).
        """
        task = self._connect_task
        if task is None:
            log_debug(
                "[ros2-conn] submit_ssh_response: no active connect task"
            )
            return
        loop = getattr(task, "auto_repair", None)
        if loop is None:
            log_debug(
                "[ros2-conn] submit_ssh_response: no active AutoRepairLoop"
            )
            return
        loop.notify_ssh_response(server_key, bool(ok))

    def reconnect(self) -> bool:
        """Disconnect (if connected) and immediately re-submit the connect task.

        Used by the diagnostics dialog after a successful repair so the
        user does not need to click Disconnect + Connect manually.
        """
        log_info("[ros2-conn] reconnect")
        self.disconnect()
        # 200ms delay so cyclonedds has a beat to release the participant
        # before we spawn a new one on the same domain. Done synchronously
        # because reconnect() is invoked from the main thread on a button.
        time.sleep(0.2)
        return self.connect()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _is_connect_task_running(self) -> bool:
        task = self._connect_task
        if task is None:
            return False
        # ``Task.status`` may be a string or a TaskStatus enum depending on
        # the SDK version — the tolerant str() comparison covers both.
        status = getattr(task, "status", None)
        return str(status).lower() in {
            "running", "pending", "taskstatus.running", "taskstatus.pending",
        }

    def _on_probe_finished(self, ok: bool, message: str) -> None:
        if ok:
            log_info(f"[ros2-conn] test ok: {message}")
            # Test 仅是探针 — 成功不进 connected 状态，只回 disconnected。
            self._set_state("disconnected", {"test_ok": True, "info": message})
        else:
            log_warning(f"[ros2-conn] test failed: {message}")
            self._set_state("error", {"test_ok": False, "info": message})

    def _cleanup_probe(self) -> None:
        if self._probe is not None:
            self._probe.deleteLater()
            self._probe = None

    def _set_state(self, state: str, info: Dict[str, Any]) -> None:
        self._current_state = state
        get_app_signals().connection_state_changed.emit(state, dict(info))

    # ------------------------------------------------------------------
    # Phase 3.10 — transport health watchdog (USB unplug detection)
    # ------------------------------------------------------------------
    def _on_state_for_watchdog(self, state: str, _info: Dict[str, Any]) -> None:
        """Start/stop the health watchdog based on connection state."""
        s = str(state or "").lower()
        # Mirror the task-side emits into _current_state so other readers
        # (e.g. UI calling current_state()) stay accurate.
        self._current_state = s
        if s == "connected":
            self._health_consecutive_failures = 0
            if not self._health_timer.isActive():
                self._health_timer.start()
                log_debug("[ros2-conn] health watchdog started")
        else:
            if self._health_timer.isActive():
                self._health_timer.stop()
                log_debug("[ros2-conn] health watchdog stopped")
            self._health_consecutive_failures = 0

    def _on_health_tick(self) -> None:
        """Probe the active transport. Two failures in a row → declare lost.

        For USB the probe is an HTTP GET to ``/identity`` with 1.5s
        timeout (see :meth:`USBDataTransport.probe_alive`). For transports
        that don't expose a real network probe we fall back to the local
        ``is_alive()`` flag, which only catches in-process disconnects.
        """
        task = self._connect_task
        transport = task.transport if task is not None else None
        if transport is None:
            # No active transport — we shouldn't be watching. Stop.
            self._health_timer.stop()
            return
        try:
            probe = getattr(transport, "probe_alive", None)
            alive = bool(probe()) if callable(probe) else bool(transport.is_alive())
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[ros2-conn] health probe raised: {exc}")
            alive = False
        if alive:
            self._health_consecutive_failures = 0
            return
        self._health_consecutive_failures += 1
        log_warning(
            f"[ros2-conn] transport probe failed "
            f"({self._health_consecutive_failures}/{self._HEALTH_FAIL_THRESHOLD})"
        )
        if self._health_consecutive_failures < self._HEALTH_FAIL_THRESHOLD:
            return
        # Two consecutive failures — the link is gone. Tear down so the
        # user sees the card revert to disconnected rather than a stale
        # green dot. Use _set_state directly so the watchdog stop happens
        # via the same _on_state_for_watchdog slot.
        log_error(
            "[ros2-conn] transport health probe failed 2x in a row — "
            "declaring connection lost"
        )
        self._health_timer.stop()
        self._health_consecutive_failures = 0
        # Defensive teardown of the task's resources before flipping state.
        if task is not None:
            try:
                task._teardown()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[ros2-conn] teardown after health-loss raised: {exc}")
        self._connect_task = None
        self._connect_task_id = None
        self._set_state("error", {"reason": "transport lost"})


_instance: Optional[Ros2ConnectionController] = None


def get_ros2_connection_controller() -> Ros2ConnectionController:
    """Lazy singleton — must be called after QApplication exists."""
    global _instance
    if _instance is None:
        _instance = Ros2ConnectionController()
    return _instance


def ssh_server_name_for_profile(profile: ConnectionProfile) -> str:
    """Stable per-robot key used to namespace SSH credentials in keyring.

    Format: ``<brand>/<robot>@<ip>``. A brand+model pair on a different IP
    keeps a separate password slot — one OS keyring per robot deployment.
    """
    brand = (profile.brand or "unknown").strip() or "unknown"
    robot = (profile.robot or "unknown").strip() or "unknown"
    ip = (profile.pupper_ip or "").strip() or "0.0.0.0"
    return f"{brand}/{robot}@{ip}"


def build_ssh_session_for_profile(profile: ConnectionProfile):
    """Construct + connect an :class:`SSHSession` for the active profile.

    Returns a connected ``SSHSession`` on success, ``None`` when the
    credentials are missing or the connect fails (DiagnosticsTask treats
    ``None`` as "ssh_unavailable" and emits INFO findings on each
    SSH-required probe rather than crashing).
    """
    from application.service.connection.ssh_session import SSHSession, SshError

    server = ssh_server_name_for_profile(profile)
    store = get_secure_store()
    password = store.ssh_password(server).get()
    sudo_pw = store.ssh_sudo_password(server).get()
    if not password and not sudo_pw:
        # Phase 3.7: do NOT log_info — AutoRepairLoop will surface this
        # as a proper SSH-prompt modal so the user sees + handles it. A
        # log_info noise line here hides under the more relevant ERROR
        # finding the diagnose probe will emit.
        log_debug(
            f"[ssh] no saved credentials for {server!r}; deferring to "
            "AutoRepairLoop SSH prompt"
        )
        return None

    host = profile.pupper_ip or ""
    user = profile.ssh_user or "ubuntu"
    # SSH port is its own field on the profile, distinct from the data
    # transport port (USB identity daemon uses 9999, SSH always 22).
    ssh_port = 22
    timeout = float(profile.ssh_timeout or 30) if profile.ssh_timeout else 30.0
    sess = SSHSession(
        host=host,
        user=user,
        password=password,
        port=ssh_port,
        timeout=timeout,
        sudo_password=sudo_pw,
    )
    try:
        sess.connect()
    except SshError as exc:
        log_error(f"[ssh] connect to {user}@{host}:{ssh_port} FAILED: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        log_error(f"[ssh] unexpected connect failure: {exc}")
        return None
    return sess


def build_profile_from_info(info: Ros2ConnectionInfo) -> ConnectionProfile:
    """Assemble a full ConnectionProfile from the UI-persisted info snapshot.

    Phase 3 mapping (UI-bound fields → ConnectionProfile slots):

      brand          <- info.brand            (Mangdang/Unitree/...; lower-case slug)
      robot          <- info.robot_model      (mini_pupper_v2/go2/...)
      ros_domain_id  <- info.domain_id
      pupper_ip      <- info.address
      ssh_port       <- info.port
      connector_type <- info.adapter_strategy ("ROS2 Generic" -> "ros2_generic",
                                               "Brand SDK"    -> "brand_sdk")
      ros_distro     <- "humble" default (Phase 4 SSH mode-switch overwrites)

    AdapterFactory.build_for_profile reads brand+robot to instantiate the
    right adapter class — that's the single integration point for the
    connection layer with the brand semantic boundary.
    """
    strategy_map = {
        "ROS2 Generic": "ros2_generic",
        "Brand SDK": "brand_sdk",
    }
    connector_type = strategy_map.get(
        str(info.adapter_strategy), "ros2_generic",
    )
    # ssh_user is persisted via the Phase 3.5 SSH credentials section
    # (Ros2ConnectionConfig.set_field) under the same RobotConn section.
    # Read it directly so the profile carries the user the keyring entry
    # was saved against; default "ubuntu" matches the dataclass default.
    ssh_user = str(Config.get_value(_SECTION, "ssh_user", "ubuntu"))
    return ConnectionProfile(
        brand=str(info.brand),
        robot=str(info.robot_model),
        ros_domain_id=int(info.domain_id),
        pupper_ip=str(info.address),
        ssh_port=int(info.port),
        ssh_user=ssh_user,
        connector_type=connector_type,
        middleware=str(info.middleware),
    )


__all__ = [
    "MIDDLEWARE_CHOICES",
    "TRANSPORT_CHOICES",
    "ADAPTER_CHOICES",
    "TRANSPORT_DEFAULTS",
    "DEFAULT_DOMAIN_ID",
    "DEFAULT_MIDDLEWARE",
    "DEFAULT_ADDRESS",
    "DEFAULT_PORT",
    "DEFAULT_TRANSPORT",
    "DEFAULT_ADAPTER",
    "transport_namespace",
    "address_key",
    "port_key",
    "Ros2ConnectionInfo",
    "Ros2ConnectionConfig",
    "Ros2BrownfieldConnectTask",
    "Ros2ConnectionController",
    "get_ros2_connection_controller",
    "build_profile_from_info",
    "ssh_server_name_for_profile",
    "build_ssh_session_for_profile",
]
