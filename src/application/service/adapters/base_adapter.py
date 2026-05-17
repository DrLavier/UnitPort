"""BaseAdapter — abstract robot adapter contract for RELEASE.

An adapter is the **brand semantic boundary**: it translates UnitPort's
brand-agnostic commands (capabilities, teleop, activation, sensor reads) to
the concrete protocol a specific robot understands (ROS2 topics, vendor SDK,
HTTP control panel, ...). The connection layer (transport + bridge) is owned
by the connector and handed to the adapter via :meth:`open_session`; the
adapter does NOT own connection lifecycle.

Every adapter implements this minimal interface; ROS2-backed adapters extend
:class:`BaseROS2Adapter` for shared bridge+topic+teleop plumbing. Vendor-SDK
adapters that bypass the DDS bridge entirely (e.g. unitree_sdk2py wrapping
its own client) extend :class:`BaseAdapter` directly and provide their own
session lifecycle.

This file deliberately stays small. The Phase 1 "legacy" methods from DEMO
(connect / run_action / stop / get_sensor_data / health) are intentionally
NOT ported — RELEASE's lifecycle is the contract; pre-lifecycle adapters
don't fit the new architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, TYPE_CHECKING

from application.service.adapters.capabilities import CapabilitySet
from application.service.connection.profile import ConnectionProfile
from application.service.connection.transport.base import Transport

if TYPE_CHECKING:
    from application.service.diagnostics.base_probe import DiagnosticProbe


class AdapterUnavailable(RuntimeError):
    """Raised when AdapterFactory cannot resolve / instantiate an adapter.

    ``error_code`` mirrors :class:`TransportUnavailable`'s pattern so the
    connector can render typed phase errors in cmd_log instead of opaque
    stack traces.
    """

    def __init__(self, error_code: str, message: str = "") -> None:
        super().__init__(message or error_code)
        self.error_code = error_code


class BaseAdapter(ABC):
    """Minimal contract every robot adapter satisfies.

    Subclass overrides cluster around 6 themes:

    1. ``open_session`` / ``close_session`` — session lifecycle
    2. ``is_session_open`` — readiness query
    3. ``capabilities`` — what this adapter declares it can do
    4. ``activate`` / ``deactivate`` — brand-specific controller-on hook
    5. ``enable_teleop`` / ``disable_teleop`` — controller→robot pump
    6. ``confirm_steady_state`` / ``verify_topic_hash`` — P8 / P6 confirmation
    """

    BRAND_ID: str = ""           # subclass MUST set; used for topic_registry keys
    DEFAULT_DOMAIN_ID: int = 0   # subclass overrides per brand deployment convention

    # ── Session lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    def open_session(
        self,
        profile: ConnectionProfile,
        transport: Transport,
    ) -> None:
        """Start the adapter's session over an already-connected transport.

        Adapter takes ownership of the underlying bridge / SDK client; the
        transport is borrowed (caller owns its lifecycle). Must be idempotent —
        calling twice without an intervening ``close_session`` is a no-op.
        Raises an adapter-specific exception on hard failure.
        """

    @abstractmethod
    def close_session(self) -> None:
        """Release every resource opened in :meth:`open_session`. Idempotent."""

    @abstractmethod
    def is_session_open(self) -> bool:
        """True iff the adapter holds an active session ready for IO."""

    # ── Capability declaration ────────────────────────────────────────────

    @abstractmethod
    def capabilities(self) -> CapabilitySet:
        """Return a snapshot of what this adapter can do for the active SKU.

        Driven by UI (Controller card renders teleop_2d / arm widgets based
        on this) and by the connector (P7 wires telemetry subscriptions based
        on declared cmd/obs roles).
        """

    # ── Activation (brand-specific controller-on / off) ───────────────────

    def activate(self) -> None:
        """Energise the robot-side controller so commands are honoured.

        Default: no-op. ROS2-already-running brownfield path does not need
        this; brand SDK adapters (Unitree FRC start sport, Mangdang Flask
        /pupper/status/start) override.
        """
        return None

    def deactivate(self) -> None:
        """Inverse of :meth:`activate`. Safe to call repeatedly. Default no-op."""
        return None

    # ── Teleop passthrough (CommandBus → robot) ───────────────────────────

    def enable_teleop(
        self,
        provider: Callable[[], Optional[dict]],
        rate_hz: float = 50.0,
    ) -> bool:
        """Start pumping ``provider()`` values to the robot's cmd surface.

        ``provider`` returns a dict-shaped Twist (``{"linear": {...},
        "angular": {...}}``) or ``None`` (deadman tick — adapter publishes
        zero twist + logs warn after deadman timeout).

        Returns True when the pump started, False when the adapter cannot
        teleop (capability missing, session not open). Default: False.
        """
        return False

    def disable_teleop(self) -> None:
        """Stop the teleop pump (if running) and publish one zero twist. Default no-op."""
        return None

    # ── Connection-phase support (P6 / P8 hooks) ──────────────────────────

    def confirm_steady_state(self, timeout_s: float = 5.0) -> bool:
        """P8 confirmation — adapter-specific "data is flowing" check.

        Default: returns :meth:`is_session_open`. Subclasses override to
        wait for declared subscriptions to start streaming (ROS2) or for the
        SDK client to reach a "READY" status (brand SDK).
        """
        return self.is_session_open()

    def verify_topic_hash(self) -> bool:
        """P6 best-effort verification that declared topics match the wire.

        Default: True (no verification). Phase 4's inspector subsystem will
        wire a real check that compares declared TopicSpec set against the
        live ROS2 graph.
        """
        return True

    # ── Diagnostic probes (P9 self-diagnose, Phase 3.5) ───────────────────

    def diagnostic_probes(self) -> List["DiagnosticProbe"]:
        """Return brand-specific probes the orchestrator runs in P9.

        Default: empty list — host-side probes (USB NIC / firewall / peer
        reachable / DDS discovery) always run; subclasses opt-in their own
        SSH-required checks (service active, RMW installed, env injected,
        ...). Each subclass returns fresh probe instances per call so
        probes can hold per-run state if needed.
        """
        return []


__all__ = ["BaseAdapter", "AdapterUnavailable"]
