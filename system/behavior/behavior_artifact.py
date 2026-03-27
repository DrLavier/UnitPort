#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior artifact contract — Phase 2 STAGE-01.

Defines the canonical data structures exchanged at every boundary of
the Behavior layer:

  BehaviorDiagnostic  — single compile-time or runtime diagnostic entry
  BehaviorArtifact    — compiled, executable behavior payload
  BehaviorInvokeInput — Mission → Behavior invocation request
  BehaviorInvokeOutput— Behavior → Mission invocation result

Design constraints
------------------
- Artifact must be serialisable to dict (for audit logging and persistence).
- trace_id propagates unchanged from compile through runtime so that
  every log entry from both phases can be correlated.
- No direct SDK calls are permitted inside this module.
- Compatibility with Phase 1 RuntimeResult fields is maintained by mapping
  BehaviorInvokeOutput.status to RuntimeResult.status vocabulary
  ("success" / "failed" / "blocked").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from system.ir.workflow_ir import WorkflowIR


# ---------------------------------------------------------------------------
# Error/reason code constants
# ---------------------------------------------------------------------------

class BehaviorErrorCode:
    """Machine-readable error codes for BehaviorInvokeOutput.reason.

    Any callers comparing reason strings MUST use these constants to avoid
    literal drift.
    """
    BEHAVIOR_REF_NOT_FOUND      = "behavior_ref_not_found"
    ARTIFACT_INVALID            = "artifact_invalid"
    SUBGRAPH_EXECUTION_FAILED   = "subgraph_execution_failed"
    SUBGRAPH_ABORTED            = "subgraph_aborted"
    BEHAVIOR_COMPILE_ERROR      = "behavior_compile_error"
    # Circle 7 Step 7.1 — semantic intent resolution failure
    SEMANTIC_RESOLUTION_FAILED  = "semantic_resolution_failed"


# ---------------------------------------------------------------------------
# BehaviorDiagnostic
# ---------------------------------------------------------------------------

@dataclass
class BehaviorDiagnostic:
    """Single diagnostic entry from compile or runtime execution.

    Fields
    ------
    level    : "error" | "warning" | "info"
    code     : Machine-readable identifier (e.g. "undefined_behavior_ref").
    message  : Human-readable description.
    location : Optional source reference — file:line or node_id string.
    trace_id : Trace correlation ID (same as the parent artifact or invocation).
    """

    level: str
    code: str
    message: str
    location: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "trace_id": self.trace_id,
        }

    @classmethod
    def error(cls, code: str, message: str,
              location: Optional[str] = None,
              trace_id: Optional[str] = None) -> "BehaviorDiagnostic":
        return cls(level="error", code=code, message=message,
                   location=location,
                   trace_id=trace_id or str(uuid.uuid4()))

    @classmethod
    def warning(cls, code: str, message: str,
                location: Optional[str] = None,
                trace_id: Optional[str] = None) -> "BehaviorDiagnostic":
        return cls(level="warning", code=code, message=message,
                   location=location,
                   trace_id=trace_id or str(uuid.uuid4()))

    @classmethod
    def info(cls, code: str, message: str,
             location: Optional[str] = None,
             trace_id: Optional[str] = None) -> "BehaviorDiagnostic":
        return cls(level="info", code=code, message=message,
                   location=location,
                   trace_id=trace_id or str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# BehaviorArtifact
# ---------------------------------------------------------------------------

@dataclass
class BehaviorArtifact:
    """Compiled, executable behavior payload.

    Produced by BehaviorCompilerBridge.compile(); consumed by
    BehaviorSubgraphInvoker (STAGE-03) at runtime.

    Fields
    ------
    artifact_id      : UUID string, unique per compile invocation.
    behavior_ref     : Canonical name from the Mission node (e.g. "stand",
                       "walk_to_point"). Used as the registry key for lookup.
    behavior_ir      : Compiled WorkflowIR subgraph — the executable payload.
    diagnostics      : Compile-time diagnostics (errors and warnings).
    trace_id         : UUID propagated to every runtime log entry so that
                       compile output and execution output are correlatable.
    compiled_at      : ISO 8601 UTC timestamp of compile completion.
    source_hash      : Optional SHA-256 hex of the source string used to detect
                       stale artifacts; None when source is unavailable.
    is_valid         : True iff diagnostics contains no error-level entries.

    v2 fields (Circle 1 Step 1.2 / Circle 2 Step 2.1)
    ---------------------------------------------------
    core_ir          : Semantic alias for behavior_ir (property).  Exposes the
                       compiled core behavior WorkflowIR under the v2 name so
                       that Circle 3+ callers can distinguish core from heartbeat
                       without renaming the persisted field.
    heartbeat_ir     : Compiled heartbeat WorkflowIR subgraph; None until
                       Circle 3 heartbeat compiler is wired up.
    heartbeat_policy : Heartbeat tick policy dict (tick_ms, timeout_ms,
                       fail_policy, max_override, safety_mode).  On fresh
                       compile via BehaviorCompilerBridge, this is populated
                       with HeartbeatPolicy.default().to_dict() so it is
                       never None in v2 artifacts.
    heartbeat_script : Raw heartbeat source string (script mode); None when
                       heartbeat is canvas-authored or not yet defined.
    """

    artifact_id: str
    behavior_ref: str
    behavior_ir: WorkflowIR
    diagnostics: List[BehaviorDiagnostic] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    compiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_hash: Optional[str] = None
    # v2 — heartbeat reserved fields (Circle 1 Step 1.2)
    heartbeat_ir: Optional[WorkflowIR] = None
    heartbeat_policy: Optional[Dict[str, Any]] = None
    heartbeat_script: Optional[str] = None
    # Phase 1 behavior redesign — optional structured timeline snapshot.
    # When present this is a BehaviorTimeline.to_dict() plain dict attached
    # at compile time so that audit logs and diagnostics can reference the
    # structured timeline that produced this artifact.  None for all pre-Phase-1
    # artifacts; callers must treat None as "no timeline attached".
    timeline_data: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        behavior_ref: str,
        behavior_ir: WorkflowIR,
        diagnostics: List[BehaviorDiagnostic] | None = None,
        trace_id: Optional[str] = None,
        source_hash: Optional[str] = None,
        heartbeat_ir: Optional[WorkflowIR] = None,
        heartbeat_policy: Optional[Dict[str, Any]] = None,
        heartbeat_script: Optional[str] = None,
        timeline_data: Optional[Dict[str, Any]] = None,
    ) -> "BehaviorArtifact":
        """Construct a BehaviorArtifact with auto-generated artifact_id.

        v2 keyword args (heartbeat_ir, heartbeat_policy, heartbeat_script,
        timeline_data) are optional and default to None so all existing callers
        remain unchanged.
        """
        return cls(
            artifact_id=str(uuid.uuid4()),
            behavior_ref=behavior_ref,
            behavior_ir=behavior_ir,
            diagnostics=diagnostics or [],
            trace_id=trace_id or str(uuid.uuid4()),
            source_hash=source_hash,
            heartbeat_ir=heartbeat_ir,
            heartbeat_policy=heartbeat_policy,
            heartbeat_script=heartbeat_script,
            timeline_data=timeline_data,
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def core_ir(self) -> WorkflowIR:
        """Semantic alias for behavior_ir — Circle 2 Step 2.1 v2 name.

        Returns the compiled core behavior WorkflowIR.  Identical to
        accessing behavior_ir directly; the alias exists so that v2 callers
        can use the semantically clear name without a field rename that would
        break existing callers.
        """
        return self.behavior_ir

    @property
    def is_valid(self) -> bool:
        """True iff no error-level diagnostics were emitted."""
        return not any(d.level == "error" for d in self.diagnostics)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.level == "warning")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict for audit logging and persistence.

        behavior_ir is represented by node/edge counts rather than the
        full WorkflowIR object so that the dict is JSON-safe by default.

        v2 fields: heartbeat_ir serialised as summary counts (None → null);
        heartbeat_policy and heartbeat_script included verbatim.
        """
        d: Dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "behavior_ref": self.behavior_ref,
            # Circle 2 Step 2.1: core_ir_summary mirrors behavior_ir_summary under the v2 name.
            # Both keys are present so old readers (behavior_ir_summary) and new readers
            # (core_ir_summary) can consume the same artifact dict without migration.
            "behavior_ir_summary": {
                "node_count": len(self.behavior_ir.nodes),
                "edge_count": len(self.behavior_ir.edges),
            },
            "core_ir_summary": {
                "node_count": len(self.behavior_ir.nodes),
                "edge_count": len(self.behavior_ir.edges),
            },
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "trace_id": self.trace_id,
            "compiled_at": self.compiled_at,
            "source_hash": self.source_hash,
            "is_valid": self.is_valid,
            # v2 — heartbeat reserved (Circle 1 Step 1.2)
            "heartbeat_ir_summary": (
                {
                    "node_count": len(self.heartbeat_ir.nodes),
                    "edge_count": len(self.heartbeat_ir.edges),
                }
                if self.heartbeat_ir is not None
                else None
            ),
            "heartbeat_policy": self.heartbeat_policy,
            "heartbeat_script": self.heartbeat_script,
            # Phase 1 behavior redesign — structured timeline snapshot (None for legacy artifacts)
            "timeline_data": self.timeline_data,
        }
        return d


# ---------------------------------------------------------------------------
# BehaviorInvokeInput / BehaviorInvokeOutput  (Mission ↔ Behavior boundary)
# ---------------------------------------------------------------------------

@dataclass
class BehaviorInvokeInput:
    """Request payload from a Mission behavior node to the behavior invoker.

    Fields
    ------
    behavior_ref : Which behavior to invoke (used for registry lookup).
    artifact_id  : Optional pre-compiled artifact ID; if absent the invoker
                   resolves via behavior_ref at runtime.
    inputs       : Port-mapped input values from upstream Mission nodes.
    context      : Execution context (robot_model, scenario keys, etc.).
    trace_id     : Propagated trace ID; generated fresh if absent.
    """

    behavior_ref: str
    artifact_id: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Circle 2 Step 2.3: explicit v2 context fields surfaced at the boundary
    # so the invoker and sub_context always carry them without dict.get() chains.
    sdk_settings: Optional[Dict[str, Any]] = None
    heartbeat_policy: Optional[Dict[str, Any]] = None
    # Step 1: optional structured motor-weight protocol payload from Mission.
    # When present the invoker validates, parses, and injects protocol_targets
    # into the behavior sub_context.  When absent the legacy path is unchanged.
    protocol_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior_ref": self.behavior_ref,
            "artifact_id": self.artifact_id,
            "inputs": self.inputs,
            "trace_id": self.trace_id,
            # Circle 2 Step 2.3: v2 context summary (sdk_settings content omitted
            # to avoid serialising potentially large/non-serialisable objects).
            "sdk_settings_present": self.sdk_settings is not None,
            "heartbeat_policy": self.heartbeat_policy,
            # Step 1: protocol payload presence indicator (payload content not serialised
            # to avoid storing potentially large mission payloads in logs).
            "protocol_payload_present": self.protocol_payload is not None,
            # context omitted from serialisation (may contain non-serialisable objects)
        }


@dataclass
class BehaviorInvokeOutput:
    """Result payload returned from the behavior invoker to the Mission node.

    status vocabulary aligns with Phase 1 RuntimeResult.status:
      "success"  — subgraph executed without errors
      "failed"   — subgraph encountered a recoverable failure
      "blocked"  — invocation could not start (ref not found, invalid artifact)

    Fields
    ------
    status      : "success" | "failed" | "blocked"
    reason      : BehaviorErrorCode constant on failure; "" on success.
    outputs     : Port-mapped output values for downstream Mission nodes.
    trace_id    : Same trace_id as the corresponding BehaviorInvokeInput.
    diagnostics : Runtime diagnostics collected during subgraph execution.
    node_results: Raw per-node results from the subgraph RuntimeResult
                  (keyed by node_id); empty on blocked.

    v2 fields (Circle 1 Step 1.2 — heartbeat runtime metrics, populated in Circle 3)
    ----------------------------------------------------------------------------------
    heartbeat_status      : Heartbeat loop outcome ("", "running", "completed",
                            "timeout", "cancelled", "not_started").
    heartbeat_tick_count  : Number of heartbeat ticks executed; 0 when not started.
    heartbeat_last_output : Last control-frame dict produced by the heartbeat loop;
                            None when heartbeat was not active.
    heartbeat_diagnostics : Per-tick or loop-level diagnostics from heartbeat executor.
    """

    status: str
    reason: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    diagnostics: List[BehaviorDiagnostic] = field(default_factory=list)
    node_results: Dict[str, Any] = field(default_factory=dict)
    # v2 — heartbeat runtime metrics (Circle 1 Step 1.2)
    heartbeat_status: str = ""
    heartbeat_tick_count: int = 0
    heartbeat_last_output: Optional[Dict[str, Any]] = None
    heartbeat_diagnostics: List[BehaviorDiagnostic] = field(default_factory=list)
    # Circle 2 Step 2.3: resolved heartbeat policy echoed from the artifact so
    # callers can inspect what policy was active without going back to the bridge.
    heartbeat_policy: Optional[Dict[str, Any]] = None
    # Step 1: protocol ingest diagnostics and status.
    # protocol_status: "valid" | "invalid" | "stale" | "absent"
    # protocol_diagnostics: BehaviorDiagnostic entries from protocol validation + parse.
    protocol_status: str = "absent"
    protocol_diagnostics: List["BehaviorDiagnostic"] = field(default_factory=list)
    # Circle 1 Step 2: protocol apply result fields.
    # apply_success_count: number of targets successfully written (or context-injected).
    # apply_skipped_count: number of targets skipped/failed.
    # apply_failure_reasons: human-readable reason per skipped/failed target.
    apply_success_count: int = 0
    apply_skipped_count: int = 0
    apply_failure_reasons: List[str] = field(default_factory=list)
    # Circle 7 Step 7.1: semantic intent resolution diagnostics.
    # resolution_diagnostics: BehaviorDiagnostic entries from intent resolution.
    #   - info-level "resolution.resolved.<intent_id>" for each successful mapping.
    #   - error-level "resolution.unsupported.<intent_id>" for each failed mapping.
    #   Empty when resolution was skipped (no brand / no timeline data).
    resolution_diagnostics: List["BehaviorDiagnostic"] = field(default_factory=list)
    # Step 10: runtime motor-track topology diagnostics.
    # topology_diagnostics: non-blocking BehaviorDiagnostic entries from the
    #   runtime canonical-topology check on artifact.timeline_data.
    #   - warning-level "runtime.topology.unknown_track" for each stale track_name.
    #   - warning-level "runtime.topology.degraded" when topology is unavailable.
    #   Empty when no timeline_data, no motor overlays, or topology check skipped.
    topology_diagnostics: List["BehaviorDiagnostic"] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        trace_id: str,
        outputs: Dict[str, Any] | None = None,
        node_results: Dict[str, Any] | None = None,
        diagnostics: List[BehaviorDiagnostic] | None = None,
        heartbeat_policy: Optional[Dict[str, Any]] = None,              # Circle 2 Step 2.3
        heartbeat_diagnostics: List["BehaviorDiagnostic"] | None = None, # Circle 2 Step 2.3
        protocol_status: str = "absent",                                 # Step 1
        protocol_diagnostics: List["BehaviorDiagnostic"] | None = None, # Step 1
        apply_success_count: int = 0,                                    # Circle 1 Step 2
        apply_skipped_count: int = 0,                                    # Circle 1 Step 2
        apply_failure_reasons: List[str] | None = None,                  # Circle 1 Step 2
        resolution_diagnostics: List["BehaviorDiagnostic"] | None = None,  # Circle 7
        topology_diagnostics: List["BehaviorDiagnostic"] | None = None,    # Step 10
    ) -> "BehaviorInvokeOutput":
        return cls(
            status="success",
            reason="",
            outputs=outputs or {},
            trace_id=trace_id,
            diagnostics=diagnostics or [],
            node_results=node_results or {},
            heartbeat_policy=heartbeat_policy,
            heartbeat_diagnostics=heartbeat_diagnostics or [],
            protocol_status=protocol_status,
            protocol_diagnostics=protocol_diagnostics or [],
            apply_success_count=apply_success_count,
            apply_skipped_count=apply_skipped_count,
            apply_failure_reasons=apply_failure_reasons or [],
            resolution_diagnostics=resolution_diagnostics or [],
            topology_diagnostics=topology_diagnostics or [],
        )

    @classmethod
    def failed(
        cls,
        trace_id: str,
        reason: str,
        outputs: Dict[str, Any] | None = None,
        node_results: Dict[str, Any] | None = None,
        diagnostics: List[BehaviorDiagnostic] | None = None,
        heartbeat_policy: Optional[Dict[str, Any]] = None,              # Circle 2 Step 2.3
        heartbeat_diagnostics: List["BehaviorDiagnostic"] | None = None, # Circle 2 Step 2.3
        protocol_status: str = "absent",                                 # Step 1
        protocol_diagnostics: List["BehaviorDiagnostic"] | None = None, # Step 1
        apply_success_count: int = 0,                                    # Circle 1 Step 2
        apply_skipped_count: int = 0,                                    # Circle 1 Step 2
        apply_failure_reasons: List[str] | None = None,                  # Circle 1 Step 2
        resolution_diagnostics: List["BehaviorDiagnostic"] | None = None,  # Circle 7
        topology_diagnostics: List["BehaviorDiagnostic"] | None = None,    # Step 10
    ) -> "BehaviorInvokeOutput":
        return cls(
            status="failed",
            reason=reason,
            outputs=outputs or {},
            trace_id=trace_id,
            diagnostics=diagnostics or [],
            node_results=node_results or {},
            heartbeat_policy=heartbeat_policy,
            heartbeat_diagnostics=heartbeat_diagnostics or [],
            protocol_status=protocol_status,
            protocol_diagnostics=protocol_diagnostics or [],
            apply_success_count=apply_success_count,
            apply_skipped_count=apply_skipped_count,
            apply_failure_reasons=apply_failure_reasons or [],
            resolution_diagnostics=resolution_diagnostics or [],
            topology_diagnostics=topology_diagnostics or [],
        )

    @classmethod
    def blocked(
        cls,
        trace_id: str,
        reason: str,
        diagnostics: List[BehaviorDiagnostic] | None = None,
        protocol_status: str = "absent",                                 # Circle 1 gap fix
        protocol_diagnostics: List["BehaviorDiagnostic"] | None = None, # Circle 1 gap fix
        resolution_diagnostics: List["BehaviorDiagnostic"] | None = None,  # Circle 7
        topology_diagnostics: List["BehaviorDiagnostic"] | None = None,    # Step 10
    ) -> "BehaviorInvokeOutput":
        return cls(
            status="blocked",
            reason=reason,
            trace_id=trace_id,
            diagnostics=diagnostics or [],
            protocol_status=protocol_status,
            protocol_diagnostics=protocol_diagnostics or [],
            resolution_diagnostics=resolution_diagnostics or [],
            topology_diagnostics=topology_diagnostics or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "outputs": self.outputs,
            "trace_id": self.trace_id,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "node_results": self.node_results,
            # v2 — heartbeat runtime metrics (Circle 1 Step 1.2)
            "heartbeat_status": self.heartbeat_status,
            "heartbeat_tick_count": self.heartbeat_tick_count,
            "heartbeat_last_output": self.heartbeat_last_output,
            "heartbeat_diagnostics": [d.to_dict() for d in self.heartbeat_diagnostics],
            # Circle 2 Step 2.3: echoed resolved policy for audit / diagnostics consumers
            "heartbeat_policy": self.heartbeat_policy,
            # Step 1: protocol ingest result
            "protocol_status": self.protocol_status,
            "protocol_diagnostics": [d.to_dict() for d in self.protocol_diagnostics],
            # Circle 1 Step 2: protocol apply result
            "apply_success_count": self.apply_success_count,
            "apply_skipped_count": self.apply_skipped_count,
            "apply_failure_reasons": self.apply_failure_reasons,
            # Circle 7 Step 7.1: semantic intent resolution diagnostics
            "resolution_diagnostics": [d.to_dict() for d in self.resolution_diagnostics],
            # Step 10: runtime motor-track topology diagnostics
            "topology_diagnostics": [d.to_dict() for d in self.topology_diagnostics],
        }


# ---------------------------------------------------------------------------
# BehaviorResolveResult  (BehaviorCompilerBridge.resolve() return type)
# ---------------------------------------------------------------------------

@dataclass
class BehaviorResolveResult:
    """Structured result from BehaviorCompilerBridge.resolve().

    Always returned instead of bare None so callers get an explicit
    reason code on every failure path.

    Fields
    ------
    ok          : True iff resolution succeeded and artifact is valid.
    reason      : BehaviorErrorCode constant on failure; "" on success.
    artifact    : The resolved BehaviorArtifact, or None on failure.
    diagnostics : Diagnostics for this resolution attempt.
                  - success          → empty list
                  - missing ref      → one error with BEHAVIOR_REF_NOT_FOUND
                  - invalid artifact → resolution error + artifact compile errors
    trace_id    : Correlation ID generated (or propagated) by resolve().

    Reason-code contract
    --------------------
    missing behavior_ref in registry → BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND
    found but artifact.is_valid=False → BehaviorErrorCode.ARTIFACT_INVALID
    success                           → reason == ""
    """

    ok: bool
    reason: str
    artifact: Optional["BehaviorArtifact"]
    diagnostics: List[BehaviorDiagnostic]
    trace_id: str

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        artifact: "BehaviorArtifact",
        trace_id: str,
    ) -> "BehaviorResolveResult":
        """Resolution succeeded — artifact is valid and ready for execution."""
        return cls(
            ok=True,
            reason="",
            artifact=artifact,
            diagnostics=[],
            trace_id=trace_id,
        )

    @classmethod
    def missing(
        cls,
        behavior_ref: str,
        trace_id: str,
    ) -> "BehaviorResolveResult":
        """behavior_ref was not found in the registry."""
        diag = BehaviorDiagnostic.error(
            code=BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
            message=f"behavior_ref '{behavior_ref}' not found in registry",
            trace_id=trace_id,
        )
        return cls(
            ok=False,
            reason=BehaviorErrorCode.BEHAVIOR_REF_NOT_FOUND,
            artifact=None,
            diagnostics=[diag],
            trace_id=trace_id,
        )

    @classmethod
    def invalid(
        cls,
        artifact: "BehaviorArtifact",
        trace_id: str,
    ) -> "BehaviorResolveResult":
        """behavior_ref found but artifact.is_valid is False.

        diagnostics includes a resolution-level error followed by the
        artifact's own compile-time diagnostics so callers have full context.
        """
        resolution_diag = BehaviorDiagnostic.error(
            code=BehaviorErrorCode.ARTIFACT_INVALID,
            message=(
                f"artifact for '{artifact.behavior_ref}' is invalid "
                f"({artifact.error_count} compile error(s))"
            ),
            trace_id=trace_id,
        )
        return cls(
            ok=False,
            reason=BehaviorErrorCode.ARTIFACT_INVALID,
            artifact=artifact,
            diagnostics=[resolution_diag, *artifact.diagnostics],
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "trace_id": self.trace_id,
        }
