"""Runtime engine for workflow execution."""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

try:
    from bin.core.logger import log_warning
except (ImportError, AttributeError):
    def log_warning(*a, **k): pass  # type: ignore[misc]

from .node_executor import NodeExecutor
from .scheduler import Scheduler
from .monitor import Monitor
from .workflow_runner import WorkflowRunner
from .interception.compile_guard import CompileGuard
from .interception.execute_guard import ExecuteGuard
from .safety.safety_checker import SafetyChecker
from .safety.safety_policy import SafetyPolicy
from .safety.emergency_handler import EmergencyHandler
from .safety.audit_logger import SafetyAuditLogger
from .contracts import DiagnosticsKey, ExecutionPath, RuntimeResult
from .migration import MigrationFlags
from system.ir.workflow_ir import NodeKind, WorkflowIR


def _collect_behavior_tracing(
    results: Dict[str, Any],
) -> Tuple[
    Dict[str, str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Extract behavior trace_ids and diagnostics from node results.

    Walks ``results`` (keyed by node_id) and collects tracing data from
    any behavior node outputs (identified by the presence of "behavior_ref").

    Returns
    -------
    behavior_trace_ids    : {node_id: trace_id}  — one entry per behavior node.
    behavior_core_diags   : flat list of BehaviorDiagnostic.to_dict() entries
                            from core subgraph execution (success + failure paths).
    heartbeat_diags       : flat list of heartbeat-section diagnostics (policy
                            validation warnings etc.) from all behavior nodes.
                            Circle 3 will add tick-level diagnostics here.
    timeline_diags        : flat list of MotorSegmentDiagnostic.to_dict() entries
                            from timeline validation across all behavior nodes.
    protocol_diags        : flat list of protocol ingest diagnostics (Step 1)
                            from all behavior nodes (schema errors, stale,
                            unknown address, clamp notices).
    """
    trace_ids: Dict[str, str] = {}
    core_diags: List[Dict[str, Any]] = []
    hb_diags: List[Dict[str, Any]] = []
    timeline_diags: List[Dict[str, Any]] = []
    protocol_diags: List[Dict[str, Any]] = []
    for node_id, out in results.items():
        if not isinstance(out, dict):
            continue
        if "behavior_ref" not in out:
            continue
        if "trace_id" in out:
            trace_ids[node_id] = out["trace_id"]
        # Core behavior subgraph diagnostics (Circle 2 Step 2.3 label)
        if "diagnostics" in out:
            core_diags.extend(out["diagnostics"])
        # Heartbeat section diagnostics (Circle 2 Step 2.3: policy warnings etc.)
        if "heartbeat_diagnostics" in out:
            hb_diags.extend(out["heartbeat_diagnostics"])
        # Timeline motor parameter diagnostics (Fix 4)
        if "timeline_diagnostics" in out:
            timeline_diags.extend(out["timeline_diagnostics"])
        # Step 1: protocol ingest diagnostics (schema/stale/unknown_address/clamp)
        if "protocol_diagnostics" in out:
            protocol_diags.extend(out["protocol_diagnostics"])
    return trace_ids, core_diags, hb_diags, timeline_diags, protocol_diags


@dataclass
class RuntimeEngine:
    """Executes a mission IR within a given scenario context.

    Phase 2 STAGE-03: behavior_bridge
    ----------------------------------
    An optional BehaviorCompilerBridge that enables behavior-type Mission nodes
    to invoke compiled behavior subgraphs.  When None (default), behavior nodes
    return a "skipped" result; non-behavior nodes are completely unaffected.

    Usage:
        engine = RuntimeEngine(behavior_bridge=bridge)
        # or:
        engine = RuntimeEngine()
        engine.behavior_bridge = bridge
    """

    # Public — callers may set after construction.
    behavior_bridge: Any = field(default=None, repr=False)

    _running: bool = field(default=False, init=False, repr=False)
    # Cycle 2 STAGE-06: set by request_cancel(); checked between node executions.
    _cancel_requested: bool = field(default=False, init=False, repr=False)
    # Cycle 3 STAGE-04: robot_model reference held during execution so that
    # request_cancel() can invoke adapter-level cancel_action() for in-flight
    # action interruption.  Cleared in the finally block after every execute().
    _current_robot_model: Any = field(default=None, init=False, repr=False)
    # True when cancel_action() was successfully called on the active adapter
    # during the current (or most recent) execute() call.
    _adapter_cancel_invoked: bool = field(default=False, init=False, repr=False)
    _executor: NodeExecutor = field(default_factory=NodeExecutor, init=False, repr=False)
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)
    _monitor: Monitor = field(default_factory=Monitor, init=False, repr=False)
    _workflow_runner: WorkflowRunner = field(default_factory=WorkflowRunner, init=False, repr=False)
    _compile_guard: CompileGuard = field(default_factory=CompileGuard, init=False, repr=False)
    _execute_guard: ExecuteGuard = field(default_factory=ExecuteGuard, init=False, repr=False)
    _safety_checker: SafetyChecker = field(default_factory=SafetyChecker, init=False, repr=False)
    _emergency_handler: EmergencyHandler = field(default_factory=EmergencyHandler, init=False, repr=False)
    _audit_logger: SafetyAuditLogger = field(default_factory=SafetyAuditLogger, init=False, repr=False)

    def request_cancel(self) -> None:
        """Request best-effort cancellation of the current execution.

        Cycle 3 STAGE-04 upgrade: in addition to setting the cooperative-cancel
        flag checked between node boundaries, this method also attempts an
        *adapter-level* cancel via ``cancel_action()`` when an action is known
        to be in-flight.

        Thread-safety
        -------------
        Safe to call from any thread (typically the UI thread while the worker
        thread runs ``execute()``).  The cooperative flag is a simple bool
        assignment (GIL-protected atomic).  ``cancel_action()`` must be
        thread-safe in every adapter that overrides it.

        Cancel path priority
        --------------------
        1. Adapter cancel (``cancel_action()``): interrupts in-flight action
           at the SDK level if the adapter supports it.
        2. Cooperative cancel: stops traversal at the next node boundary.
        Fallback to cooperative-only when no adapter is active or the adapter
        uses the default no-op.
        """
        self._cancel_requested = True
        # Cycle 3 STAGE-04: attempt adapter-level cancel for in-flight action.
        model = self._current_robot_model
        if model is not None:
            cancel_fn = getattr(model, "cancel_action", None)
            if callable(cancel_fn):
                try:
                    cancel_fn()
                    self._adapter_cancel_invoked = True
                except Exception:
                    pass  # cancel_action() must not raise; guard anyway

    def execute(self, mission_ir: Any, scenario: Any, node_status_callback=None) -> dict:
        """Execute *mission_ir* under *scenario* and return a unified RuntimeResult dict.

        Args:
            node_status_callback: Optional callable(node_id, status) for in-flight
                per-node status updates.  status ∈ {"running", "success", "failed"}.
                When None, behaviour is unchanged (backward-compat).
        """
        # Reset per-execution flags (Cycle 2 STAGE-06 + Cycle 3 STAGE-04).
        self._cancel_requested = False
        self._adapter_cancel_invoked = False
        # Cycle 3 STAGE-04: hold robot_model for the duration of execute() so
        # that request_cancel() can forward an adapter-level interrupt.
        self._current_robot_model = (scenario or {}).get("robot_model")

        # Generate a per-execution mission trace ID so every audit entry, every
        # behavior invocation, and the final RuntimeResult can be correlated.
        mission_trace_id = str(_uuid.uuid4())

        compile_check = self._compile_guard.check(mission_ir)
        if not compile_check["ok"]:
            self._audit_logger.record("compile_blocked", compile_check)
            return RuntimeResult.blocked(
                path=ExecutionPath.BLOCKED_COMPILE,
                reason=compile_check["reason"],
            ).to_dict()

        execute_check = self._execute_guard.check(scenario or {})
        if not execute_check["ok"]:
            self._audit_logger.record("execute_blocked", execute_check)
            return RuntimeResult.blocked(
                path=ExecutionPath.BLOCKED_EXECUTE,
                reason=execute_check["reason"],
            ).to_dict()

        policy = SafetyPolicy.from_dict((scenario or {}).get("safety_policy", {}))
        safety_check = self._safety_checker.check(mission_ir, scenario or {}, policy)
        if not safety_check["ok"]:
            emergency = self._emergency_handler.handle(safety_check["reason"], safety_check)
            self._audit_logger.record("safety_blocked", {"check": safety_check, "emergency": emergency})
            return RuntimeResult.blocked(
                path=ExecutionPath.BLOCKED_SAFETY,
                reason=safety_check["reason"],
                emergency=emergency,
            ).to_dict()

        self._running = True
        self._monitor.start()
        behavior_trace_ids: Dict[str, str] = {}
        behavior_diagnostics: List[Dict[str, Any]] = []
        heartbeat_diagnostics: List[Dict[str, Any]] = []  # Circle 2 Step 2.3
        timeline_diagnostics: List[Dict[str, Any]] = []   # Fix 4
        protocol_diagnostics: List[Dict[str, Any]] = []   # Step 1
        try:
            task_id = self._scheduler.schedule({"scenario": scenario})
            _compat_mode: bool = False

            # Cycle 2 STAGE-06: cancel_check lambda shared with runner/executor.
            cancel_check = lambda: self._cancel_requested

            if self._is_execution_graph(mission_ir):
                self._workflow_runner.max_loop_iterations = policy.max_loop_iterations
                runner_out = self._workflow_runner.run(
                    exec_graph=mission_ir,
                    robot_model=(scenario or {}).get("robot_model"),
                    graph_scene=(scenario or {}).get("graph_scene"),
                    action_mapping=(scenario or {}).get("action_mapping"),
                    node_status_callback=node_status_callback,
                    cancel_check=cancel_check,  # STAGE-06
                )
                node_count = len(mission_ir.get("nodes", {}))
                results = runner_out.get("results", {})
                ok = runner_out.get("ok", False)
                reason = runner_out.get("reason") or ""
                # Cancellation override for exec_graph path (STAGE-06).
                # The shared section below handles the final priority check.
                if self._cancel_requested and ok:
                    ok = False
                    reason = "mission_cancelled"
                executed_count = runner_out.get("executed_count", 0)
                has_action = runner_out.get("has_action", False)
                path = ExecutionPath.EXECUTION_GRAPH
            else:
                # 策略参数统一接入 executor（对齐 exec_graph 路径 WorkflowRunner）
                migration_flags = MigrationFlags.from_env()
                self._audit_logger.record("migration_flags", migration_flags.to_dict())
                _compat_mode = not migration_flags.use_flow_aware_execution
                if not migration_flags.use_flow_aware_execution:
                    log_warning(
                        "RuntimeEngine: topological-sort compat path activated "
                        "(UNITPORT_FLOW_AWARE_EXECUTION=0). "
                        "This is a controlled fallback — not the default path."
                    )
                self._executor.max_loop_iterations = policy.max_loop_iterations
                self._executor.use_flow_aware_execution = migration_flags.use_flow_aware_execution

                # Phase 2 STAGE-03: inject behavior dispatch into executor.
                # A fresh BehaviorSubgraphInvoker is created per execution so
                # that subgraph state never leaks across mission runs.
                from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
                self._executor._behavior_invoker = BehaviorSubgraphInvoker()
                self._executor._behavior_bridge = self.behavior_bridge  # None → skipped
                self._executor._behavior_policy = policy

                self._executor.clear()
                self._load_mission(mission_ir)
                node_count = len(self._executor.nodes)

                # Cycle 2 STAGE-06: inject cancel_check into executor so it
                # can stop traversal at node boundaries on request_cancel().
                self._executor._cancel_fn = cancel_check

                # Phase 2 STAGE-05: inject mission_trace_id into execution
                # context so that all behavior node invocations within this run
                # share the same top-level trace ID for end-to-end correlation.
                results = self._executor.execute(
                    context={"scenario": scenario, "mission_trace_id": mission_trace_id},
                    node_status_callback=node_status_callback,
                )

                # Collect behavior tracing data for RuntimeResult.diagnostics.
                # Circle 2 Step 2.3: also collect heartbeat section diagnostics.
                # Fix 4: also collect timeline motor parameter diagnostics.
                (
                    behavior_trace_ids,
                    behavior_diagnostics,
                    heartbeat_diagnostics,
                    timeline_diagnostics,
                    protocol_diagnostics,
                ) = _collect_behavior_tracing(results)

                # executed_count 取自 executor 实际遍历计数，对齐 exec_graph 路径
                executed_count = self._executor._last_executed_count
                ok = True
                reason = ""
                path = ExecutionPath.WORKFLOW_IR
                # 与 exec_graph 路径对齐：检测 action / stop 节点是否存在
                _ACTION_KINDS = {NodeKind.ACTION, NodeKind.STOP}
                if isinstance(mission_ir, WorkflowIR):
                    has_action = any(n.kind in _ACTION_KINDS for n in mission_ir.nodes)
                else:
                    _nodes_list = mission_ir.get("nodes", []) if isinstance(mission_ir, dict) else []
                    has_action = any(
                        n.get("type") in ("action_execution", "stop") for n in _nodes_list
                    )

            # 统一感知节点级失败（两路径共用）
            failed_nodes = [
                nid for nid, out in results.items()
                if isinstance(out, dict) and "error" in out
            ]
            # Cycle 2 STAGE-06: cancellation is the highest-priority override —
            # checked before abort/failure so a cancelled run always surfaces as
            # "mission_cancelled" rather than a misleading success or failure.
            if self._cancel_requested:
                ok = False
                reason = "mission_cancelled"
            # 优先检查 workflow_aborted（AbortNode 触发），再检查普通节点失败
            elif ok and path == ExecutionPath.WORKFLOW_IR and self._executor._abort:
                ok = False
                reason = "workflow_aborted"
            elif failed_nodes and ok:
                ok = False
                reason = "node_execution_failed"

            loop_limit_exceeded = (
                path == ExecutionPath.WORKFLOW_IR
                and self._executor._loop_limit_exceeded
            )

            self._monitor.bump_event()
            self._audit_logger.record("execution_completed", {
                "task_id": task_id,
                "mission_trace_id": mission_trace_id,
                "node_count": node_count,
                "path": path,
                "executed_count": executed_count,
                "failed_nodes": failed_nodes,
                "reason": reason,
                "loop_limit_exceeded": loop_limit_exceeded,
                "behavior_trace_ids": behavior_trace_ids,
                "behavior_diagnostics_count": len(behavior_diagnostics),
            })

            if ok:
                result = RuntimeResult.success(
                    task_id=task_id,
                    node_count=node_count,
                    results=results,
                    metrics=self._monitor.get_metrics(),
                    path=path,
                    executed_count=executed_count,
                    failed_nodes=failed_nodes,
                    has_action=has_action,
                    mission_trace_id=mission_trace_id,
                    behavior_diagnostics=behavior_diagnostics,
                    behavior_trace_ids=behavior_trace_ids,
                    heartbeat_diagnostics=heartbeat_diagnostics,  # Circle 2 Step 2.3
                    timeline_diagnostics=timeline_diagnostics,    # Fix 4
                    protocol_diagnostics=protocol_diagnostics,    # Step 1
                )
            else:
                result = RuntimeResult.failed(
                    task_id=task_id,
                    node_count=node_count,
                    results=results,
                    metrics=self._monitor.get_metrics(),
                    reason=reason,
                    path=path,
                    executed_count=executed_count,
                    failed_nodes=failed_nodes,
                    has_action=has_action,
                    mission_trace_id=mission_trace_id,
                    behavior_diagnostics=behavior_diagnostics,
                    behavior_trace_ids=behavior_trace_ids,
                    heartbeat_diagnostics=heartbeat_diagnostics,  # Circle 2 Step 2.3
                    timeline_diagnostics=timeline_diagnostics,    # Fix 4
                    protocol_diagnostics=protocol_diagnostics,    # Step 1
                )
            if _compat_mode:
                result.diagnostics[DiagnosticsKey.COMPAT_PATH]   = True
                result.diagnostics[DiagnosticsKey.COMPAT_REASON] = "UNITPORT_FLOW_AWARE_EXECUTION=0"
            # Cycle 3 STAGE-04: surface adapter-cancel info in diagnostics.
            if self._adapter_cancel_invoked:
                result.diagnostics[DiagnosticsKey.ADAPTER_CANCEL_INVOKED] = True
            # Circle 1 Step 1.1: surface WorkflowIR path selection in diagnostics.
            # behavior_enabled_run=True iff NodeExecutor (WorkflowIR) path was used.
            # execution_path_reason: prefer the scenario tag set by the UI; fall back
            # to a derived string so direct-API callers also get a meaningful value.
            result.diagnostics[DiagnosticsKey.BEHAVIOR_ENABLED_RUN] = (
                path == ExecutionPath.WORKFLOW_IR
            )
            _ui_reason = (scenario or {}).get("_behavior_path_reason", "")
            result.diagnostics[DiagnosticsKey.EXECUTION_PATH_REASON] = (
                _ui_reason
                if _ui_reason
                else (
                    "workflowir_direct"
                    if path == ExecutionPath.WORKFLOW_IR
                    else "exec_graph_compat"
                )
            )
            # Circle 3: surface package metadata trace fields in diagnostics.
            # Always a dict with explicit empty strings so callers never get None.
            _raw_pkg = (scenario or {}).get("package_metadata_trace") or {}
            result.diagnostics[DiagnosticsKey.PACKAGE_METADATA_TRACE] = {
                "package_id":      str(_raw_pkg.get("package_id", "") or ""),
                "package_version": str(_raw_pkg.get("package_version", "") or ""),
                "schema_version":  str(_raw_pkg.get("schema_version", "") or ""),
            }
            return result.to_dict()
        finally:
            self._monitor.stop()
            self._running = False
            # Cycle 3 STAGE-04: release robot_model reference so it isn't held
            # after the run and cannot be called by a stale request_cancel().
            self._current_robot_model = None

    @staticmethod
    def _is_execution_graph(mission_ir: Any) -> bool:
        return isinstance(mission_ir, dict) and isinstance(mission_ir.get("nodes"), dict)

    def _load_mission(self, mission_ir: Any) -> None:
        """Load mission data into the runtime executor.

        Supports dict payloads from legacy graph exports.
        """
        if isinstance(mission_ir, WorkflowIR):
            for node in mission_ir.nodes:
                self._executor.add_node(node.id, node.schema_id, node.to_dict())
            for edge in mission_ir.edges:
                self._executor.add_connection(
                    edge.from_node,
                    edge.from_port,
                    edge.to_node,
                    edge.to_port,
                    edge_type=edge.edge_type.value,  # "flow" | "data"
                )
            return

        if isinstance(mission_ir, dict):
            nodes = mission_ir.get("nodes", [])
            connections = mission_ir.get("connections", [])
            for node in nodes:
                node_id = str(node.get("id"))
                node_type = str(node.get("type", "unknown"))
                self._executor.add_node(node_id, node_type, node)
            for conn in connections:
                src = conn.get("from", {})
                dst = conn.get("to", {})
                self._executor.add_connection(
                    str(src.get("node", "")),
                    str(src.get("port", "flow_out")),
                    str(dst.get("node", "")),
                    str(dst.get("port", "flow_in")),
                )
            return

        raise TypeError("RuntimeEngine currently expects mission_ir as dict")

