#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior subgraph invoker — Phase 2 STAGE-03.

BehaviorSubgraphInvoker executes a compiled BehaviorArtifact as an isolated
subgraph on behalf of a Mission behavior node.

Design constraints
------------------
- No direct SDK calls.
- No circular imports: this module imports only from system.behavior.*;
  the NodeExecutor sub-executor is received as a parameter, not imported.
- Failure propagation is explicit and reason-coded (no silent success).

Invocation flow
---------------
1. Resolve behavior_ref via BehaviorCompilerBridge.resolve()
   → blocked on BEHAVIOR_REF_NOT_FOUND or ARTIFACT_INVALID
2. Load behavior_ir (WorkflowIR subgraph) into the caller-supplied sub_executor.
3. Execute subgraph with isolated context:
   - Parent context keys (robot_model, scenario …) are inherited.
   - trace_id and behavior_inputs are injected as additional context keys.
4. Inspect subgraph results:
   - _abort set → BehaviorInvokeOutput.failed(SUBGRAPH_ABORTED)
   - any node "error" key → BehaviorInvokeOutput.failed(SUBGRAPH_EXECUTION_FAILED)
   - otherwise → BehaviorInvokeOutput.success()
5. Return BehaviorInvokeOutput to the Mission node dispatcher.
"""

from __future__ import annotations

from typing import Any, List, Optional

from system.behavior.behavior_artifact import (
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
)
from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from system.behavior.heartbeat_policy import validate_heartbeat_policy  # Circle 2 Step 2.3
from system.behavior.motor_weight_protocol import (                      # Step 1
    validate_protocol_payload,
    parse_protocol_targets,
)
from system.behavior.protocol_apply_engine import (                      # Circle 1 Step 2
    ProtocolApplyEngine,
    ProtocolApplyResult,
)


class BehaviorSubgraphInvoker:
    """Execute a BehaviorArtifact as an isolated subgraph.

    Called by NodeExecutor._execute_behavior_node() for every behavior-type
    Mission node encountered during WorkflowIR execution.

    The sub_executor parameter is a fresh NodeExecutor instance created by
    the caller; this avoids a circular import between behavior_invoker and
    node_executor.
    """

    def invoke(
        self,
        invoke_input: BehaviorInvokeInput,
        bridge: BehaviorCompilerBridge,
        sub_executor: Any,              # NodeExecutor — typed Any to avoid circular import
        policy: Optional[Any] = None,  # SafetyPolicy — optional, for loop-limit config
    ) -> BehaviorInvokeOutput:
        """Resolve, load, and execute a behavior subgraph.

        Parameters
        ----------
        invoke_input  : BehaviorInvokeInput from the Mission node dispatcher.
        bridge        : BehaviorCompilerBridge holding the in-memory artifact registry.
        sub_executor  : Fresh NodeExecutor instance (no state) for subgraph execution.
        policy        : Optional SafetyPolicy; used to propagate max_loop_iterations.

        Returns
        -------
        BehaviorInvokeOutput — always a typed result; never raises.
        """
        trace_id = invoke_input.trace_id

        # ------------------------------------------------------------------
        # Step 1 — Resolve artifact
        # ------------------------------------------------------------------
        resolve_result = bridge.resolve(invoke_input.behavior_ref, trace_id=trace_id)
        if not resolve_result.ok:
            return BehaviorInvokeOutput.blocked(
                trace_id=trace_id,
                reason=resolve_result.reason,
                diagnostics=resolve_result.diagnostics,
            )

        artifact = resolve_result.artifact

        # ------------------------------------------------------------------
        # Step 2 — Configure sub-executor from policy
        # ------------------------------------------------------------------
        if policy is not None and hasattr(policy, "max_loop_iterations"):
            sub_executor.max_loop_iterations = policy.max_loop_iterations

        # ------------------------------------------------------------------
        # Step 2b — Validate artifact heartbeat_policy (Circle 2 Step 2.3)
        # Invalid policy produces warning-level diagnostics surfaced in
        # heartbeat_diagnostics; invocation is NOT blocked (enforcement is
        # Circle 3's responsibility at tick-time).
        # ------------------------------------------------------------------
        heartbeat_policy = artifact.heartbeat_policy
        heartbeat_diags: List[BehaviorDiagnostic] = []
        if heartbeat_policy is not None:
            _policy_ok, _policy_diags = validate_heartbeat_policy(
                heartbeat_policy, trace_id=trace_id
            )
            if not _policy_ok:
                heartbeat_diags.extend(_policy_diags)

        # ------------------------------------------------------------------
        # Step 2c — Optional motor-weight protocol ingest (Step 1)
        # If a protocol_payload is present on the invoke_input:
        #   - validate schema + staleness,
        #   - parse targets (resolve final values, apply clamping),
        #   - inject protocol_targets into sub_context.
        # If absent: legacy path — sub_context carries no protocol keys.
        # Invalid payload: blocked with INVALID_PROTOCOL (no silent fallback).
        # ------------------------------------------------------------------
        proto_status: str = "absent"
        proto_diags: List[BehaviorDiagnostic] = []
        protocol_targets = None  # List[ParsedTarget] | None

        if invoke_input.protocol_payload is not None:
            # Circle 1 Step 1: compat gate — caller can set protocol_compat_mode=True
            # in the execution context to allow unknown fields (with warnings) rather
            # than blocking.  Default is strict mode (unknown fields are rejected).
            _strict_mode = not bool(
                invoke_input.context.get("protocol_compat_mode", False)
            )
            _proto_valid, _proto_diags = validate_protocol_payload(
                invoke_input.protocol_payload,
                trace_id=trace_id,
                strict_mode=_strict_mode,
            )
            proto_diags.extend(_proto_diags)
            if not _proto_valid:
                # Determine whether it was a staleness rejection or structural failure.
                from system.behavior.motor_weight_protocol import ProtocolDiagCode
                is_stale = any(
                    d.code == ProtocolDiagCode.STALE for d in _proto_diags
                )
                proto_status = "stale" if is_stale else "invalid"
                # No silent fallback — block the invocation.
                # Circle 1 gap fix: pass protocol_status/protocol_diagnostics so that
                # blocked() results carry the same unified protocol fields as success/failed.
                return BehaviorInvokeOutput.blocked(
                    trace_id=trace_id,
                    reason="INVALID_PROTOCOL",
                    diagnostics=_proto_diags,
                    protocol_status=proto_status,
                    protocol_diagnostics=_proto_diags,
                )
            else:
                proto_status = "valid"
                # Extract known_addresses from execution context so that
                # UNKNOWN_ADDRESS diagnostics are emitted for unresolvable targets.
                # The context key "motor_address_registry" is an optional
                # collection (set/list) provided by the adapter or scenario config.
                known_addresses = invoke_input.context.get("motor_address_registry")
                _parsed_targets, _parse_diags = parse_protocol_targets(
                    invoke_input.protocol_payload,
                    trace_id=trace_id,
                    known_addresses=known_addresses,
                )
                proto_diags.extend(_parse_diags)
                protocol_targets = _parsed_targets

        # ------------------------------------------------------------------
        # Step 3 — Load behavior IR into the fresh sub-executor
        # ------------------------------------------------------------------
        behavior_ir = artifact.behavior_ir
        for node in behavior_ir.nodes:
            sub_executor.add_node(node.id, node.schema_id, node.to_dict())
        for edge in behavior_ir.edges:
            sub_executor.add_connection(
                edge.from_node,
                edge.from_port,
                edge.to_node,
                edge.to_port,
                edge_type=edge.edge_type.value,  # "flow" | "data"
            )

        # ------------------------------------------------------------------
        # Step 4 — Execute subgraph with isolated context
        # ------------------------------------------------------------------
        # Inherit parent context so robot_model and scenario keys are available
        # to robot-aware nodes inside the subgraph.
        # Circle 2 Step 2.3: explicitly surface sdk_settings and heartbeat_policy
        # so sub-context consumers can access them directly without dict.get() chains.
        sub_context: dict = {
            **invoke_input.context,
            "trace_id": trace_id,
            "behavior_inputs": invoke_input.inputs,
            "sdk_settings": invoke_input.sdk_settings,      # Circle 2 Step 2.3
            "heartbeat_policy": heartbeat_policy,            # Circle 2 Step 2.3
            # Step 1: resolved protocol targets (None when no protocol payload)
            "protocol_targets": protocol_targets,
            "protocol_status": proto_status,
        }

        try:
            node_results = sub_executor.execute(context=sub_context)
        except Exception as exc:
            diag = BehaviorDiagnostic.error(
                code=BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
                message=str(exc),
                trace_id=trace_id,
            )
            return BehaviorInvokeOutput.failed(
                trace_id=trace_id,
                reason=BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
                diagnostics=[diag],
                heartbeat_policy=heartbeat_policy,
                heartbeat_diagnostics=heartbeat_diags,
                protocol_status=proto_status,        # Step 1
                protocol_diagnostics=proto_diags,    # Step 1
            )

        # ------------------------------------------------------------------
        # Step 5 — Inspect subgraph outcome
        # ------------------------------------------------------------------

        # Abort takes priority (AbortNode was triggered inside the subgraph).
        if getattr(sub_executor, "_abort", False):
            abort_msg = getattr(sub_executor, "_abort_reason", "") or "subgraph aborted"
            diag = BehaviorDiagnostic.error(
                code=BehaviorErrorCode.SUBGRAPH_ABORTED,
                message=abort_msg,
                trace_id=trace_id,
            )
            return BehaviorInvokeOutput.failed(
                trace_id=trace_id,
                reason=BehaviorErrorCode.SUBGRAPH_ABORTED,
                diagnostics=[diag],
                node_results=node_results,
                heartbeat_policy=heartbeat_policy,
                heartbeat_diagnostics=heartbeat_diags,
                protocol_status=proto_status,        # Step 1
                protocol_diagnostics=proto_diags,    # Step 1
            )

        # Per-node execution failures.
        failed_node_ids = [
            nid for nid, out in node_results.items()
            if isinstance(out, dict) and "error" in out
        ]
        if failed_node_ids:
            diag = BehaviorDiagnostic.error(
                code=BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
                message=(
                    f"subgraph node(s) failed: {', '.join(failed_node_ids)}"
                ),
                trace_id=trace_id,
            )
            return BehaviorInvokeOutput.failed(
                trace_id=trace_id,
                reason=BehaviorErrorCode.SUBGRAPH_EXECUTION_FAILED,
                diagnostics=[diag],
                node_results=node_results,
                heartbeat_policy=heartbeat_policy,
                heartbeat_diagnostics=heartbeat_diags,
                protocol_status=proto_status,        # Step 1
                protocol_diagnostics=proto_diags,    # Step 1
            )

        # ------------------------------------------------------------------
        # Step 6 — Apply protocol targets (Circle 1 Step 2)
        # Explicit apply call after successful execution — not just context injection.
        # The adapter (robot_model) is looked up from the execution context so that
        # real hardware writes occur when a backend is connected; otherwise, the
        # engine records a context-inject-only success for each target.
        # ------------------------------------------------------------------
        _apply_result: ProtocolApplyResult = ProtocolApplyResult()
        if protocol_targets is not None:
            _adapter = invoke_input.context.get("robot_model")
            _apply_result = ProtocolApplyEngine().apply(
                protocol_targets, adapter=_adapter, trace_id=trace_id
            )
            proto_diags.extend(_apply_result.diagnostics)

        # ------------------------------------------------------------------
        # Step 7 — Success
        # ------------------------------------------------------------------
        return BehaviorInvokeOutput.success(
            trace_id=trace_id,
            outputs=node_results,       # All subgraph node outputs as behavior outputs.
            node_results=node_results,  # Per-node detail for audit / diagnostics.
            heartbeat_policy=heartbeat_policy,       # Circle 2 Step 2.3
            heartbeat_diagnostics=heartbeat_diags,   # Circle 2 Step 2.3 (policy warnings if any)
            protocol_status=proto_status,            # Step 1
            protocol_diagnostics=proto_diags,        # Step 1
            apply_success_count=_apply_result.apply_success_count,   # Circle 1 Step 2
            apply_skipped_count=_apply_result.apply_skipped_count,   # Circle 1 Step 2
            apply_failure_reasons=_apply_result.apply_failure_reasons, # Circle 1 Step 2
        )
