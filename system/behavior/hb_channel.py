#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heartbeat execution chain channels — Circle 1 Step 1.3.

Defines the interface contract (IHBChannel) and request/response envelopes
for the four frontend execution channels:

  compile          — compile a heartbeat draft payload into an artifact
  dry_run          — simulate N ticks without a live robot
  run              — start live heartbeat execution
  poll_diagnostics — poll current runtime diagnostics

Also provides HBMockChannel: a fully-stubbed implementation that lets the
UI exercise the complete chain before Circle 2 backend work is complete.

Design principles
-----------------
- Pure Python; no Qt, no SDK calls, no I/O.
- Envelopes use plain dataclasses with default factories (no from_dict/to_dict
  needed — they are internal request/response objects, not persisted).
- IHBChannel is a runtime_checkable Protocol so adapters can be duck-typed.
- HBMockChannel maintains minimal mutable state for realistic poll sequences.
"""

from __future__ import annotations

import datetime
import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

try:
    from bin.core.logger import log_warning as _log_warning
except (ImportError, AttributeError):
    def _log_warning(*a, **k): pass  # type: ignore[misc]


# ===========================================================================
# Request / Response Envelopes
# ===========================================================================

@dataclass
class HBCompileRequest:
    """Request to compile a heartbeat draft payload for a behavior node."""
    behavior_ref: str
    draft_payload: Any   # HBDraftPayload (typed Any to avoid circular import)


@dataclass
class HBCompileResponse:
    """Result of a heartbeat compile operation."""
    ok: bool
    artifact_id: str
    error_count: int
    warning_count: int
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class HBDryRunRequest:
    """Request to simulate a fixed number of heartbeat ticks (no live robot)."""
    behavior_ref: str
    draft_payload: Any   # HBDraftPayload
    tick_count: int = 5


@dataclass
class HBDryRunResponse:
    """Result of a heartbeat dry-run / simulation."""
    ok: bool
    simulated_ticks: int
    event_updates: List[Dict[str, Any]] = field(default_factory=list)
    sensor_snapshot_summary: Dict[str, Any] = field(default_factory=dict)
    control_output_summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class HBRunRequest:
    """Request to start live heartbeat execution for a behavior node."""
    behavior_ref: str
    draft_payload: Any   # HBDraftPayload
    scenario: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HBRunResponse:
    """Response from starting or stopping a live heartbeat run."""
    ok: bool
    status: str            # "started" | "running" | "completed" | "failed" | "cancelled"
    heartbeat_status: str  # "inactive" | "active" | "error"
    event_updates: List[Dict[str, Any]] = field(default_factory=list)
    sensor_snapshot_summary: Dict[str, Any] = field(default_factory=dict)
    control_output_summary: Dict[str, Any] = field(default_factory=dict)
    tick_count: int = 0
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class HBDiagnosticsSnapshot:
    """A point-in-time snapshot of heartbeat runtime diagnostics.

    Returned by ``IHBChannel.poll_diagnostics()``; the UI renders this
    directly into the status label and event-state / IO-mapping tables.
    """
    behavior_ref: str
    status: str            # "idle" | "running" | "completed" | "failed"
    heartbeat_status: str  # "inactive" | "active" | "error"
    event_updates: List[Dict[str, Any]] = field(default_factory=list)
    sensor_snapshot_summary: Dict[str, Any] = field(default_factory=dict)
    control_output_summary: Dict[str, Any] = field(default_factory=dict)
    tick_count: int = 0
    last_tick_ms: float = 0.0
    timestamp: str = ""


# ===========================================================================
# Channel Interface (Protocol)
# ===========================================================================

@runtime_checkable
class IHBChannel(Protocol):
    """Protocol defining the frontend <-> backend heartbeat execution channel.

    All four methods are synchronous from the caller's perspective.  Async or
    threaded execution is the implementor's responsibility.  The UI calls these
    methods directly and renders the responses without caring about the backend
    topology (mock, real runtime, remote, etc.).
    """

    def compile(self, request: HBCompileRequest) -> HBCompileResponse:
        """Compile the heartbeat draft payload; return compile result."""
        ...

    def dry_run(self, request: HBDryRunRequest) -> HBDryRunResponse:
        """Simulate N ticks without a live robot; return tick results."""
        ...

    def run(self, request: HBRunRequest) -> HBRunResponse:
        """Start live heartbeat execution; return initial status."""
        ...

    def poll_diagnostics(self, behavior_ref: str) -> HBDiagnosticsSnapshot:
        """Poll current heartbeat runtime diagnostics for a behavior node."""
        ...


# ===========================================================================
# Mock Implementation
# ===========================================================================

class HBMockChannel:
    """Mock heartbeat channel — stubs all four operations.

    Default channel used by BehaviorPanel (Step 1.3) until the real runtime
    channel is wired in Circle 2.  Produces realistic-looking payloads so the
    complete UI authoring and diagnostics flow can be exercised without a
    running robot or live backend.

    State model
    -----------
    - compile() increments _compile_count; each call returns a unique artifact.
    - dry_run() returns deterministic sensor / control snapshots.
    - run() sets _mock_running = True; each poll increments _mock_tick_count.
    - poll_diagnostics() reflects running state; tick_count grows on each poll.
    - reset() clears all mutable state (useful in tests).
    """

    def __init__(self) -> None:
        self._compile_count: int = 0
        self._run_count: int = 0
        self._mock_running: bool = False
        self._mock_tick_count: int = 0

    # ------------------------------------------------------------------
    # IHBChannel implementation
    # ------------------------------------------------------------------

    def compile(self, request: HBCompileRequest) -> HBCompileResponse:
        self._compile_count += 1
        artifact_id = f"mock-{request.behavior_ref}-{uuid.uuid4().hex[:8]}"
        return HBCompileResponse(
            ok=True,
            artifact_id=artifact_id,
            error_count=0,
            warning_count=0,
            diagnostics=[],
            message=(
                f"[mock] HB compile OK — "
                f"ref='{request.behavior_ref}' artifact={artifact_id[:16]}"
            ),
        )

    def dry_run(self, request: HBDryRunRequest) -> HBDryRunResponse:
        n = max(1, min(request.tick_count, 20))
        return HBDryRunResponse(
            ok=True,
            simulated_ticks=n,
            event_updates=[
                {
                    "event_name": "tick",
                    "state": "active",
                    "source": "mock_hb",
                    "timestamp": f"T{i}",
                    "reason": "",
                }
                for i in range(n)
            ],
            sensor_snapshot_summary={
                "imu": {"pitch": 0.02, "roll": -0.01, "yaw": 0.0},
                "joint_state": "nominal",
                "battery": 85,
                "foot_contact": [True, True, True, True],
            },
            control_output_summary={
                "base_action": "walk",
                "weight_delta": {"walk": 0.0, "stand": 0.0},
                "posture_target": {"pitch_comp": 0.02, "roll_comp": -0.01},
                "confidence": 0.95,
            },
            diagnostics=[],
            message=(
                f"[mock] Dry-run '{request.behavior_ref}' — "
                f"{n} tick(s) simulated"
            ),
        )

    def run(self, request: HBRunRequest) -> HBRunResponse:
        self._run_count += 1
        self._mock_running = True
        self._mock_tick_count = 0
        return HBRunResponse(
            ok=True,
            status="started",
            heartbeat_status="active",
            event_updates=[],
            sensor_snapshot_summary={
                "imu": {"pitch": 0.0, "roll": 0.0},
                "battery": 85,
            },
            control_output_summary={
                "base_action": "walk",
                "weight_delta": 0.0,
                "confidence": 1.0,
            },
            tick_count=0,
            diagnostics=[],
            message=(
                f"[mock] Run '{request.behavior_ref}' started "
                f"(run #{self._run_count})"
            ),
        )

    def poll_diagnostics(self, behavior_ref: str) -> HBDiagnosticsSnapshot:
        if self._mock_running:
            self._mock_tick_count += 1
        ts = datetime.datetime.now().isoformat()
        return HBDiagnosticsSnapshot(
            behavior_ref=behavior_ref,
            status="running" if self._mock_running else "idle",
            heartbeat_status="active" if self._mock_running else "inactive",
            event_updates=[],
            sensor_snapshot_summary=(
                {"imu": {"pitch": 0.0, "roll": 0.0}, "battery": 85}
                if self._mock_running else {}
            ),
            control_output_summary=(
                {"base_action": "walk", "confidence": 1.0}
                if self._mock_running else {}
            ),
            tick_count=self._mock_tick_count,
            last_tick_ms=100.0,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Test / debug helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all mutable state.  Useful in unit tests."""
        self._compile_count = 0
        self._run_count = 0
        self._mock_running = False
        self._mock_tick_count = 0

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def mock_running(self) -> bool:
        return self._mock_running

    @property
    def mock_tick_count(self) -> int:
        return self._mock_tick_count


# ===========================================================================
# Real Runtime Implementation  (Circle 2 Step 2.1)
# ===========================================================================

class HBRuntimeChannel:
    """Real heartbeat execution channel backed by BehaviorCompilerBridge + NodeExecutor.

    Provides the same IHBChannel interface as HBMockChannel but drives actual
    compiler and execution machinery so the BehaviorPanel gets real runtime
    semantics.

    Latest-snapshot semantics
    -------------------------
    - compile() and dry_run() are synchronous; results return immediately.
    - run() initialises execution state without blocking; returns "started"
      on success.
    - poll_diagnostics() returns the last-known snapshot and advances tick_count
      by executing one lightweight NodeExecutor tick per call.  It never blocks
      waiting for external data — if the tick raises it is silently swallowed
      and the snapshot reflects the state just before the fault.

    Adapter interface
    -----------------
    adapter.apply_motor_param(target: ParsedTarget) -> bool  (optional)
    When present motor write calls flow through the adapter; when absent the
    channel runs in no-hardware (context-inject-only) mode.

    Parameters
    ----------
    bridge  : BehaviorCompilerBridge instance (duck-typed Any to avoid
              circular imports — only .compile() is called).
    adapter : Optional robot adapter.  None → no-hardware mode.
    """

    def __init__(self, bridge: Any, adapter: Any = None) -> None:
        self._bridge = bridge
        self._adapter = adapter
        # Run state (reset between run() calls) — protected by _lock.
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._trace_id: str = ""
        self._tick_count: int = 0
        self._behavior_ref: str = ""
        self._last_artifact: Any = None          # BehaviorArtifact | None
        self._last_snapshot: Optional[HBDiagnosticsSnapshot] = None
        self._last_tick_ms: float = 0.0
        # Non-blocking poll support — background tick worker.
        self._tick_in_flight: bool = False       # True while _run_tick_bg is executing
        self._tick_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # IHBChannel implementation
    # ------------------------------------------------------------------

    def compile(self, request: HBCompileRequest) -> HBCompileResponse:
        source = self._extract_source(request.draft_payload)
        try:
            artifact = self._bridge.compile(
                source=source,
                behavior_ref=request.behavior_ref,
            )
            if artifact.is_valid:
                self._last_artifact = artifact
            ok = artifact.is_valid
            return HBCompileResponse(
                ok=ok,
                artifact_id=artifact.artifact_id,
                error_count=artifact.error_count,
                warning_count=artifact.warning_count,
                diagnostics=[d.to_dict() for d in artifact.diagnostics],
                message=(
                    f"[runtime] compile {'OK' if ok else 'FAILED'} — "
                    f"ref='{request.behavior_ref}' "
                    f"errors={artifact.error_count} warnings={artifact.warning_count}"
                ),
            )
        except Exception as exc:
            return HBCompileResponse(
                ok=False,
                artifact_id="",
                error_count=1,
                warning_count=0,
                diagnostics=[],
                message=f"[runtime] compile error: {exc}",
            )

    def dry_run(self, request: HBDryRunRequest) -> HBDryRunResponse:
        source = self._extract_source(request.draft_payload)
        n = max(1, min(request.tick_count, 50))
        try:
            artifact = self._bridge.compile(
                source=source,
                behavior_ref=request.behavior_ref,
            )
            if not artifact.is_valid:
                return HBDryRunResponse(
                    ok=False,
                    simulated_ticks=0,
                    diagnostics=[d.to_dict() for d in artifact.diagnostics],
                    message=(
                        f"[runtime] dry_run: compile failed "
                        f"({artifact.error_count} errors)"
                    ),
                )
            tick_outputs = []
            for i in range(n):
                try:
                    out = self._execute_tick(artifact, tick_index=i, trace_id="dry-run")
                    tick_outputs.append(out)
                except Exception:
                    tick_outputs.append({})

            event_updates = [
                {
                    "event_name": "tick",
                    "state": "active",
                    "source": "runtime_hb",
                    "timestamp": f"T{i}",
                    "reason": "",
                }
                for i in range(len(tick_outputs))
            ]
            return HBDryRunResponse(
                ok=True,
                simulated_ticks=len(tick_outputs),
                event_updates=event_updates,
                sensor_snapshot_summary={
                    "source": "runtime",
                    "ticks": len(tick_outputs),
                },
                control_output_summary={
                    "nodes_executed": sum(
                        len(t) for t in tick_outputs if isinstance(t, dict)
                    ),
                },
                diagnostics=[],
                message=(
                    f"[runtime] dry_run '{request.behavior_ref}' — "
                    f"{len(tick_outputs)} tick(s)"
                ),
            )
        except Exception as exc:
            return HBDryRunResponse(
                ok=False,
                simulated_ticks=0,
                diagnostics=[],
                message=f"[runtime] dry_run error: {exc}",
            )

    def run(self, request: HBRunRequest) -> HBRunResponse:
        source = self._extract_source(request.draft_payload)
        try:
            artifact = self._bridge.compile(
                source=source,
                behavior_ref=request.behavior_ref,
            )
            if not artifact.is_valid:
                self._running = False
                return HBRunResponse(
                    ok=False,
                    status="failed",
                    heartbeat_status="error",
                    diagnostics=[d.to_dict() for d in artifact.diagnostics],
                    message=(
                        f"[runtime] run: compile failed "
                        f"({artifact.error_count} errors)"
                    ),
                )
        except Exception as exc:
            self._running = False
            return HBRunResponse(
                ok=False,
                status="failed",
                heartbeat_status="error",
                diagnostics=[],
                message=f"[runtime] run: compile error: {exc}",
            )

        # Artifact is valid — initialise run state
        self._running = True
        self._trace_id = str(uuid.uuid4())
        self._tick_count = 0
        self._behavior_ref = request.behavior_ref
        self._last_artifact = artifact
        self._last_snapshot = HBDiagnosticsSnapshot(
            behavior_ref=request.behavior_ref,
            status="running",
            heartbeat_status="active",
            tick_count=0,
            timestamp=datetime.datetime.now().isoformat(),
        )
        return HBRunResponse(
            ok=True,
            status="started",
            heartbeat_status="active",
            tick_count=0,
            diagnostics=[],
            message=(
                f"[runtime] run '{request.behavior_ref}' started "
                f"trace_id={self._trace_id[:8]}"
            ),
        )

    def poll_diagnostics(self, behavior_ref: str) -> HBDiagnosticsSnapshot:
        """Return latest-known snapshot immediately (non-blocking).

        If a background tick is not already in flight, launches one as a daemon
        thread.  The returned snapshot always reflects the state at the moment
        of the call — the caller never waits for the tick to complete.
        """
        with self._lock:
            if not self._running or self._last_artifact is None:
                return HBDiagnosticsSnapshot(
                    behavior_ref=behavior_ref,
                    status="idle",
                    heartbeat_status="inactive",
                    tick_count=self._tick_count,
                    last_tick_ms=self._last_tick_ms,
                    timestamp=datetime.datetime.now().isoformat(),
                )

            # Build snapshot from current state — returned immediately.
            snap = HBDiagnosticsSnapshot(
                behavior_ref=behavior_ref,
                status="running",
                heartbeat_status="active",
                tick_count=self._tick_count,
                last_tick_ms=self._last_tick_ms,
                timestamp=datetime.datetime.now().isoformat(),
            )

            # Schedule background tick only if none is currently in flight.
            if not self._tick_in_flight:
                self._tick_in_flight = True
                self._tick_thread = threading.Thread(
                    target=self._run_tick_bg,
                    args=(
                        self._last_artifact,
                        self._tick_count,
                        self._trace_id,
                        behavior_ref,
                    ),
                    daemon=True,
                )
                self._tick_thread.start()

            return snap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tick_bg(
        self,
        artifact: Any,
        tick_index: int,
        trace_id: str,
        behavior_ref: str,
    ) -> None:
        """Execute one tick in the background and update shared snapshot state.

        Called exclusively from a daemon thread launched by poll_diagnostics().
        Faults are swallowed — state is updated to reflect completion or failure
        before clearing _tick_in_flight so that the next poll may schedule a
        new tick.
        """
        t0 = _time.monotonic()
        success = False
        elapsed = 0.0
        try:
            self._execute_tick(artifact, tick_index=tick_index, trace_id=trace_id)
            elapsed = (_time.monotonic() - t0) * 1000.0
            success = True
        except Exception:
            pass
        with self._lock:
            self._tick_in_flight = False
            if success:
                self._tick_count += 1
                self._last_tick_ms = elapsed
                self._last_snapshot = HBDiagnosticsSnapshot(
                    behavior_ref=behavior_ref,
                    status="running",
                    heartbeat_status="active",
                    tick_count=self._tick_count,
                    last_tick_ms=elapsed,
                    timestamp=datetime.datetime.now().isoformat(),
                )

    def _execute_tick(
        self,
        artifact: Any,
        tick_index: int = 0,
        trace_id: str = "",
    ) -> Dict[str, Any]:
        """Execute one tick of behavior_ir using a fresh NodeExecutor."""
        # Lazy import to avoid circular dependency.
        from system.runtime.node_executor import NodeExecutor  # noqa: PLC0415

        executor = NodeExecutor()
        for node in artifact.behavior_ir.nodes:
            executor.add_node(node.id, node.schema_id, node.to_dict())
        for edge in artifact.behavior_ir.edges:
            executor.add_connection(
                edge.from_node, edge.from_port,
                edge.to_node, edge.to_port,
                edge_type=edge.edge_type.value,
            )
        ctx: Dict[str, Any] = {
            "tick_index": tick_index,
            "trace_id": trace_id or self._trace_id,
        }
        if self._adapter is not None:
            ctx["robot_model"] = self._adapter
        return executor.execute(context=ctx)

    @staticmethod
    def _extract_source(draft_payload: Any) -> str:
        """Extract Python source string from a draft_payload.

        Handles:
          - HBDraftPayload object (has .script attribute)
          - dict representation (has "script" key)
          - plain string
          - None → empty string
        """
        if draft_payload is None:
            return ""
        if hasattr(draft_payload, "script"):
            return str(draft_payload.script or "")
        if isinstance(draft_payload, dict):
            return str(draft_payload.get("script", ""))
        return str(draft_payload)

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset execution state (useful in tests and after stop).

        Joins any in-flight background tick with a short timeout so that
        the thread can finish cleanly before state is wiped.
        """
        # Capture thread reference before resetting state.
        with self._lock:
            t = self._tick_thread
            self._running = False

        # Let in-flight tick complete (best-effort, short timeout).
        if t is not None and t.is_alive():
            t.join(timeout=0.2)

        with self._lock:
            self._trace_id = ""
            self._tick_count = 0
            self._behavior_ref = ""
            self._last_artifact = None
            self._last_snapshot = None
            self._last_tick_ms = 0.0
            self._tick_in_flight = False
            self._tick_thread = None

    def _join_tick(self, timeout: float = 1.0) -> None:
        """Wait for any in-flight background tick to complete.

        Intended for use in tests to synchronise on tick completion without
        polling.  Not part of IHBChannel — do not call from production code.
        """
        with self._lock:
            t = self._tick_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def trace_id(self) -> str:
        return self._trace_id


# ===========================================================================
# Channel Factory  (Circle 2 Step 2.2)
# ===========================================================================

class HBChannelFactory:
    """Factory that creates the best available IHBChannel for the given backend.

    Priority
    --------
    1. HBRuntimeChannel — when a BehaviorCompilerBridge is available.
    2. HBMockChannel    — fallback when no bridge is available or construction
                          of the runtime channel raises an exception.

    The fallback is always logged at warning level so operators know the UI is
    running against a stub rather than the real runtime.
    """

    @staticmethod
    def create(bridge: Any = None, adapter: Any = None) -> "IHBChannel":
        """Create and return the best available heartbeat channel.

        Parameters
        ----------
        bridge  : BehaviorCompilerBridge (duck-typed; only .compile() is called).
                  When None, returns HBMockChannel immediately.
        adapter : Optional robot adapter.  None → no-hardware mode for runtime
                  channel (values are context-injected but no hardware writes).

        Returns
        -------
        HBRuntimeChannel when bridge is available and construction succeeds;
        HBMockChannel otherwise (fallback).
        """
        if bridge is None:
            _log_warning(
                "hb_channel: HBChannelFactory using mock — reason=no_bridge"
            )
            return HBMockChannel()
        try:
            return HBRuntimeChannel(bridge=bridge, adapter=adapter)
        except Exception as exc:
            _log_warning(
                f"hb_channel: HBChannelFactory using mock — reason=runtime_init_failed exc={exc}"
            )
            return HBMockChannel()
