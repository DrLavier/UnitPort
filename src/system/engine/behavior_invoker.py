#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior subgraph invoker — Phase 2 STAGE-03.

BehaviorSubgraphInvoker executes a compiled BehaviorArtifact as an isolated
subgraph on behalf of a Mission behavior node.

Design constraints
------------------
- No direct SDK calls.
- No circular imports: this module imports only from src.system.behavior.*;
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

from src.system.behavior.behavior_artifact import (
    BehaviorDiagnostic,
    BehaviorErrorCode,
    BehaviorInvokeInput,
    BehaviorInvokeOutput,
)
from src.system.behavior.semantic_resolution import resolve_timeline_intents
from src.system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
from src.system.behavior.heartbeat_policy import validate_heartbeat_policy  # Circle 2 Step 2.3
from src.system.behavior.motor_weight_protocol import (                      # Step 1
    validate_protocol_payload,
    parse_protocol_targets,
)
from src.system.behavior.protocol_apply_engine import (                      # Circle 1 Step 2
    ProtocolApplyEngine,
    ProtocolApplyResult,
)
from src.system.binding.registry import default_registry                     # Step 8
from src.system.binding.output import BACKEND_SDK, BACKEND_MUJOCO            # Step 8


def _determine_execution_backend(scenario: dict, context: dict) -> str:
    """Return the active execution backend from scenario/context.

    Resolution order:
    1. Explicit scenario["backend"] when valid.
    2. scenario["target"] == "hardware"   -> "sdk"
    3. scenario["target"] == "simulation" -> "mujoco"
    4. Legacy context fallback: top-level robot_model present -> "sdk"
    5. Default -> "mujoco"
    """
    backend = str(scenario.get("backend") or "").lower()
    if backend in (BACKEND_SDK, BACKEND_MUJOCO):
        return backend
    target = str(scenario.get("target") or "").lower()
    if target == "hardware":
        return BACKEND_SDK
    if target == "simulation":
        return BACKEND_MUJOCO
    if context.get("robot_model") is not None:
        return BACKEND_SDK
    return BACKEND_MUJOCO


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
                from src.system.behavior.motor_weight_protocol import ProtocolDiagCode
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
        # Step 2d — Semantic intent resolution (Circle 7 Step 7.1)
        # Resolve every intent_id in the artifact's timeline against the
        # currently active brand/model.  This is late binding: the registry
        # is queried now, not at authoring time, so a model switch between
        # compile and execute is always detected.
        #
        # - No brand in context → skip (legacy / unknown model; no error).
        # - All intents resolve → inject resolved_actions into sub_context.
        # - Any unresolvable intent → blocked (deterministic failure with
        #   per-intent error diagnostics).
        # ------------------------------------------------------------------
        _scenario = invoke_input.context.get("scenario") or {}
        _brand     = str(_scenario.get("brand") or invoke_input.context.get("brand") or "")
        _robot_type = str(_scenario.get("robot_type") or invoke_input.context.get("robot_type") or "")

        resolution_result = resolve_timeline_intents(
            artifact.timeline_data,
            brand=_brand,
            robot_type=_robot_type,
            trace_id=trace_id,
        )

        if not resolution_result.is_clean:
            # One or more intents are UNSUPPORTED on the active model.
            # Block the invocation deterministically with structured diagnostics.
            _res_diags = resolution_result.diagnostics
            return BehaviorInvokeOutput.blocked(
                trace_id=trace_id,
                reason=BehaviorErrorCode.SEMANTIC_RESOLUTION_FAILED,
                diagnostics=_res_diags,
                resolution_diagnostics=_res_diags,
            )

        # ------------------------------------------------------------------
        # Step 2e — Backend binding resolution (Step 8)
        # For each semantically resolved intent:
        #   1. Look up the ActionBinding in the default registry.
        #   2. Extract ir_params from the first matching timeline segment.
        #   3. Call binding.bind() to produce a BindingOutput.
        #   4. Inject the BindingOutput into resolved_bindings.
        # A missing binding is a warning, NOT a block — authoring and IR
        # validation are unaffected; execution degrades gracefully.
        # ------------------------------------------------------------------
        _backend = _determine_execution_backend(_scenario, invoke_input.context)
        _binding_diags: List[BehaviorDiagnostic] = []
        _resolved_bindings: dict = {}  # Dict[intent_id, BindingOutput]

        # Build intent_id → ir_params map from the first matching timeline segment.
        # Params carried by the segment (speed, duration, or nested "params" dict)
        # become the ir_params passed to binding.bind().  Default to {} when absent.
        _segment_params: dict = {}
        _segment_names: dict = {}
        _tl_segments = (artifact.timeline_data or {}).get("action_segments") or []
        for _seg in _tl_segments:
            if not isinstance(_seg, dict):
                continue
            if _seg.get("kind", "movement") != "movement":
                continue
            _seg_iid = (_seg.get("intent_id") or "").strip()
            if not _seg_iid or _seg_iid in _segment_params:
                continue  # already have params for this intent, or no intent
            _seg_ir: dict = {}
            for _pk in ("speed", "vx", "duration"):
                if _pk in _seg:
                    _seg_ir[_pk] = _seg[_pk]
            if "params" in _seg and isinstance(_seg["params"], dict):
                _seg_ir.update(_seg["params"])
            _segment_params[_seg_iid] = _seg_ir
            _seg_name = str(_seg.get("name") or "").strip()
            if _seg_name:
                _segment_names[_seg_iid] = _seg_name

        for _intent_id in resolution_result.resolved:
            _binding = default_registry.resolve(
                _intent_id, _brand, _robot_type, _backend
            )
            if _binding is None:
                _binding_diags.append(
                    BehaviorDiagnostic.warning(
                        code=f"binding.no_binding.{_intent_id}",
                        message=(
                            f"No binding found for intent {_intent_id!r} "
                            f"on {_brand}/{_robot_type} backend={_backend!r}"
                        ),
                        location=_intent_id,
                        trace_id=trace_id or None,
                    )
                )
                continue
            _ir_params = _segment_params.get(_intent_id, {})
            try:
                _binding_output = _binding.bind(
                    _intent_id, _ir_params, _brand, _robot_type, _backend
                )
                _resolved_bindings[_intent_id] = _binding_output
                _source_action = _segment_names.get(_intent_id, "").strip()
                if _source_action and _source_action not in _resolved_bindings:
                    _resolved_bindings[_source_action] = _binding_output
                _resolved_raw = str(resolution_result.resolved.get(_intent_id) or "").strip()
                if _resolved_raw and _resolved_raw not in _resolved_bindings:
                    _resolved_bindings[_resolved_raw] = _binding_output
                _binding_diags.append(
                    BehaviorDiagnostic.info(
                        code=f"binding.selected.{_intent_id}",
                        message=(
                            f"intent={_intent_id!r} binding={type(_binding).__name__} "
                            f"operation={_binding_output.operation!r} backend={_binding_output.backend!r}"
                        ),
                        location=_intent_id,
                        trace_id=trace_id or None,
                    )
                )
            except Exception as _bind_exc:
                _binding_diags.append(
                    BehaviorDiagnostic.warning(
                        code=f"binding.bind_error.{_intent_id}",
                        message=(
                            f"binding.bind() raised for intent {_intent_id!r}: {_bind_exc}"
                        ),
                        location=_intent_id,
                        trace_id=trace_id or None,
                    )
                )

        # Merge semantic-resolution diagnostics with binding-resolution diagnostics
        # so callers see the full picture in a single list.
        _all_resolution_diags = list(resolution_result.diagnostics) + _binding_diags

        # ------------------------------------------------------------------
        # Step 2f — Runtime canonical-topology check (Step 10)
        # Validates MotorSegment track_names in the artifact timeline against
        # the active brand/robot_type topology.  Non-blocking: stale tracks
        # emit warning-level diagnostics in topology_diagnostics so operators
        # can diagnose model-switch regressions without silent data loss.
        # ------------------------------------------------------------------
        _topology_diags: List[BehaviorDiagnostic] = []
        if artifact.timeline_data and _brand and _robot_type:
            try:
                from src.system.behavior.action_profile import BehaviorTimeline
                from src.system.behavior.timeline_migration import validate_timeline_topology
                _tl_obj = BehaviorTimeline.from_dict(artifact.timeline_data)
                _topo_raw = validate_timeline_topology(
                    _tl_obj, brand=_brand, robot_type=_robot_type
                )
                for _td in _topo_raw:
                    # Remap compile-level codes to runtime-prefixed codes so
                    # consumers can distinguish compile-time vs runtime findings.
                    _rt_code = _td.code.replace("topology.", "runtime.topology.", 1)
                    # Prefer canonical body-part label in location when available
                    # (Step 10: "prefer canonical body-part / motor labels").
                    if _td.limb_label and _td.track_name:
                        _loc = f"{_td.limb_label} (track:{_td.track_name})"
                    elif _td.track_name:
                        _loc = f"track:{_td.track_name}"
                    else:
                        _loc = None
                    _topology_diags.append(BehaviorDiagnostic.warning(
                        code=_rt_code,
                        message=_td.message,
                        location=_loc,
                        trace_id=trace_id or None,
                    ))
            except Exception:  # noqa: BLE001
                pass  # Topology check is best-effort; never blocks execution.

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
            "robot_model": _scenario.get("robot_model") or invoke_input.context.get("robot_model"),
            "sdk_settings": invoke_input.sdk_settings,      # Circle 2 Step 2.3
            "heartbeat_policy": heartbeat_policy,            # Circle 2 Step 2.3
            # Step 1: resolved protocol targets (None when no protocol payload)
            "protocol_targets": protocol_targets,
            "protocol_status": proto_status,
            # Circle 7 Step 7.1: late-bound intent → raw_action map (empty when skipped)
            "resolved_actions": resolution_result.resolved,
            # Step 8: intent → ActionBinding map; nodes call binding.bind() with their params
            "resolved_bindings": _resolved_bindings,
            # Promote brand/robot_type to top-level so subgraph nodes can read them
            # directly via context.get("brand") / context.get("robot_type") without
            # having to reach into context["scenario"].  Mirrors how robot_model is
            # promoted above; keeps the three runtime-identity keys symmetric.
            "brand": _brand,
            "robot_type": _robot_type,
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
                resolution_diagnostics=_all_resolution_diags,  # Circle 7 + Step 8
                topology_diagnostics=_topology_diags,           # Step 10
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
                resolution_diagnostics=_all_resolution_diags,  # Circle 7 + Step 8
                topology_diagnostics=_topology_diags,           # Step 10
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
                resolution_diagnostics=_all_resolution_diags,  # Circle 7 + Step 8
                topology_diagnostics=_topology_diags,           # Step 10
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

        # Surface executor-path detail as runtime diagnostics so operators can
        # inspect the full chain: intent -> binding -> operation -> executor -> result.
        _execution_diags: List[BehaviorDiagnostic] = []
        for _node_id, _node_out in (node_results or {}).items():
            if not isinstance(_node_out, dict):
                continue
            _flow = _node_out.get("flow_out") or {}
            if not isinstance(_flow, dict) or not _flow.get("executor_path"):
                continue
            _execution_diags.append(
                BehaviorDiagnostic.info(
                    code=f"executor.dispatch.{_flow.get('operation') or 'unknown'}",
                    message=(
                        f"node={_node_id!r} intent={_flow.get('intent_id')!r} "
                        f"binding={_flow.get('binding_name')!r} "
                        f"operation={_flow.get('operation')!r} "
                        f"executor={_flow.get('executor_name')!r} "
                        f"result={_flow.get('status')!r}"
                    ),
                    location=str(_node_id),
                    trace_id=trace_id or None,
                )
            )

        # ------------------------------------------------------------------
        # Step 7 — Success
        # ------------------------------------------------------------------
        return BehaviorInvokeOutput.success(
            trace_id=trace_id,
            outputs=node_results,       # All subgraph node outputs as behavior outputs.
            node_results=node_results,  # Per-node detail for audit / diagnostics.
            diagnostics=_execution_diags,
            heartbeat_policy=heartbeat_policy,       # Circle 2 Step 2.3
            heartbeat_diagnostics=heartbeat_diags,   # Circle 2 Step 2.3 (policy warnings if any)
            protocol_status=proto_status,            # Step 1
            protocol_diagnostics=proto_diags,        # Step 1
            apply_success_count=_apply_result.apply_success_count,   # Circle 1 Step 2
            apply_skipped_count=_apply_result.apply_skipped_count,   # Circle 1 Step 2
            apply_failure_reasons=_apply_result.apply_failure_reasons, # Circle 1 Step 2
            resolution_diagnostics=_all_resolution_diags,    # Circle 7 + Step 8
            topology_diagnostics=_topology_diags,            # Step 10
        )
