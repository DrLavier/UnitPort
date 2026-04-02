#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime unified result contract (Phase 1 - STEP-01).

定义 RuntimeEngine.execute() 的统一返回结构，
使两条执行路径（execution graph / WorkflowIR）产出同构结果。

Naming convention
-----------------
  RuntimeResult   — 顶层结果，RuntimeEngine.execute() 的唯一返回类型
  RuntimeStatus   — status 字段的合法值枚举
  DiagnosticsKey  — diagnostics 子字典的标准 key 常量

Compatibility
-------------
  RuntimeResult.to_dict() 保留既有调用方可读取的全部关键字段：
    status / reason / task_id / node_count / results / metrics
  新增 diagnostics 为附加字段，老调用方可安全忽略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class RuntimeStatus:
    """合法 status 字符串常量。"""
    SUCCESS = "success"
    FAILED  = "failed"
    BLOCKED = "blocked"


class DiagnosticsKey:
    """diagnostics 子字典的标准 key 常量。"""
    PATH           = "path"           # str: ExecutionPath.EXECUTION_GRAPH | WORKFLOW_IR | BLOCKED_*
    EXECUTED_COUNT = "executed_count" # int: 实际走过的节点数
    FAILED_NODES   = "failed_nodes"   # List[str]: 含 "error" 字段的节点 ID 列表
    HAS_ACTION     = "has_action"     # bool: 是否含 action_execution / stop 节点
    EMERGENCY      = "emergency"      # Any: 安全拦截时 emergency_handler 的输出

    # Phase 2 STAGE-05 — behavior traceability
    # These keys are present on success/failed results; absent on blocked results.
    MISSION_TRACE_ID     = "mission_trace_id"      # str: UUID generated once per RuntimeEngine.execute() call
    BEHAVIOR_TRACE_IDS   = "behavior_trace_ids"    # Dict[node_id, trace_id]: one entry per behavior node executed
    BEHAVIOR_DIAGNOSTICS = "behavior_diagnostics"  # List[dict]: BehaviorDiagnostic.to_dict() from all behavior nodes

    # Phase 7 STAGE-05 — compatibility guardrails
    # Present only when a retained compat gate was activated during execution.
    # Callers can inspect these keys to detect compat-path usage without parsing
    # log messages.  Absent on default-path executions.
    COMPAT_PATH   = "compat_path"    # bool: True if any compat gate was active
    COMPAT_REASON = "compat_reason"  # str: human-readable reason (e.g. env var name)

    # Circle 1 Step 1.1 — WorkflowIR path selection
    # Present on every success/failed result that was produced by a behavior-enabled run.
    # BEHAVIOR_ENABLED_RUN=True means the WorkflowIR+NodeExecutor path was used.
    # EXECUTION_PATH_REASON is a human-readable string explaining why the path was chosen
    # ("behavior_node_detected", "env_flag", "exec_graph_compat", etc.).
    BEHAVIOR_ENABLED_RUN   = "behavior_enabled_run"    # bool
    EXECUTION_PATH_REASON  = "execution_path_reason"   # str

    # Cycle 3 STAGE-04 — preemptive cancel contract
    # Present (and True) only when RuntimeEngine.request_cancel() successfully
    # called cancel_action() on the active adapter during an in-flight run.
    # Absent (or False) when cancel used only cooperative node-boundary path.
    # Allows diagnostics consumers to distinguish "interrupted mid-action" from
    # "halted at next node boundary".
    ADAPTER_CANCEL_INVOKED = "adapter_cancel_invoked"  # bool: True when adapter cancel was attempted

    # Circle 2 Step 2.3 — behavior_core and heartbeat section keys
    # These provide stable, semantically distinct diagnostic sections so that
    # Circle 3+ callers can separate core behavior subgraph diagnostics from
    # heartbeat loop diagnostics without parsing the legacy BEHAVIOR_DIAGNOSTICS list.
    BEHAVIOR_CORE_DIAGNOSTICS = "behavior_core_diagnostics"  # List[dict]: core subgraph BehaviorDiagnostic entries
    HEARTBEAT_DIAGNOSTICS     = "heartbeat_diagnostics"      # List[dict]: heartbeat loop diagnostics (from all behavior nodes)
    HEARTBEAT_STATUS          = "heartbeat_status"           # str: aggregate heartbeat loop outcome ("" when not yet active)

    # Circle 1 Step 1.7 — model-switch compatibility audit semantic diagnostics
    # Populated by MainWindow._run_compat_audit() on every model-switch event;
    # empty list when the audit is clean (no incompatible nodes detected).
    COMPAT_DIAGNOSTICS = "compat_diagnostics"  # List[dict]: BehaviorDiagnostic.to_dict() from compat audit

    # Phase 1 Behavior redesign — structured timeline motor parameter diagnostics
    # Populated during compile and pre-run validation when a BehaviorTimeline is
    # present.  Each entry is a MotorSegmentDiagnostic.to_dict().
    # Absent when no structured timeline is attached to the current behavior node.
    TIMELINE_DIAGNOSTICS = "timeline_diagnostics"  # List[dict]: MotorSegmentDiagnostic entries

    # Step 1 — motor weight protocol ingest diagnostics
    # PROTOCOL_STATUS  : "valid" | "invalid" | "stale" | "absent"
    # PROTOCOL_DIAGNOSTICS: List[dict] BehaviorDiagnostic entries from protocol
    #   validation and target parse (schema errors, stale, clamp notices).
    #   Empty when no protocol payload was present (status="absent").
    PROTOCOL_STATUS      = "protocol_status"       # str
    PROTOCOL_DIAGNOSTICS = "protocol_diagnostics"  # List[dict]

    # Circle 3 — package metadata traceability
    # Dict with keys: package_id (str), package_version (str), schema_version (str).
    # Always present after Circle 3; empty strings when no package metadata loaded.
    PACKAGE_METADATA_TRACE = "package_metadata_trace"  # Dict[str, str]

    # Circle 7 Step 7.1 — semantic intent resolution diagnostics
    # List of BehaviorDiagnostic.to_dict() entries from the late-binding
    # intent → raw_action resolution performed at invoke time.
    # - info-level "resolution.resolved.<intent_id>" for each successful mapping.
    # - error-level "resolution.unsupported.<intent_id>" for each blocked mapping.
    # Empty list when resolution was skipped (no brand / no timeline data).
    SEMANTIC_RESOLUTION = "semantic_resolution"  # List[dict]

    # Phase 5 Task A4 — DAG transition pre-flight validation
    # TRANSITION_VALIDATION_REPORT: serialised TransitionValidationReport
    #   (list of per-pair dicts with status, reason, transition_id, warnings).
    # TRANSITION_INSERTED_NODES: list of transition skill IDs auto-inserted
    #   into the execution graph to bridge posture/velocity gaps.
    # TRANSITION_VALIDATION_STATUS: summary string for quick inspection —
    #   "valid" | "needs_insertion" | "blocked".
    TRANSITION_VALIDATION_REPORT = "transition_validation_report"   # List[dict]
    TRANSITION_INSERTED_NODES    = "transition_inserted_nodes"      # List[str]
    TRANSITION_VALIDATION_STATUS = "transition_validation_status"   # str


class ExecutionPath:
    """diagnostics[PATH] 的合法值。"""
    EXECUTION_GRAPH  = "execution_graph"
    WORKFLOW_IR      = "workflow_ir"
    BLOCKED_COMPILE  = "blocked_compile"
    BLOCKED_EXECUTE  = "blocked_execute"
    BLOCKED_SAFETY   = "blocked_safety"


# blocked 路径常量 → legacy "phase" 字符串的映射。
# to_dict() 用此表恢复 {status, phase, reason} 三元组，不再做字符串切片。
_BLOCKED_PATH_TO_PHASE: Dict[str, str] = {
    ExecutionPath.BLOCKED_COMPILE: "compile",
    ExecutionPath.BLOCKED_EXECUTE: "execute",
    ExecutionPath.BLOCKED_SAFETY:  "safety",
}


@dataclass
class RuntimeResult:
    """
    RuntimeEngine.execute() 统一返回结构（草案 v0.1）。

    Fields
    ------
    status      : "success" | "failed" | "blocked"
    reason      : 失败/拦截原因码（成功时为空字符串）
    task_id     : Scheduler 分配的任务 ID（blocked 阶段为空字符串）
    node_count  : 工作流中节点总数（blocked 阶段为 0）
    results     : 各节点执行输出，key = node_id，value = 节点输出 dict
    metrics     : Monitor 指标（事件数、耗时等）
    diagnostics : 结构化诊断信息，含执行路径、失败节点列表等附加字段
    """

    status: str
    reason: str = ""
    task_id: str = ""
    node_count: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 工厂方法：各场景快速构建
    # ------------------------------------------------------------------

    @classmethod
    def success(
        cls,
        task_id: str,
        node_count: int,
        results: Dict[str, Any],
        metrics: Dict[str, Any],
        path: str,
        executed_count: int = 0,
        failed_nodes: List[str] | None = None,
        has_action: bool = False,
        mission_trace_id: str = "",
        behavior_diagnostics: List[Dict[str, Any]] | None = None,
        behavior_trace_ids: Dict[str, str] | None = None,
        heartbeat_diagnostics: List[Dict[str, Any]] | None = None,  # Circle 2 Step 2.3
        timeline_diagnostics: List[Dict[str, Any]] | None = None,   # Fix 4
        protocol_diagnostics: List[Dict[str, Any]] | None = None,   # Step 1
    ) -> "RuntimeResult":
        _bdiags = behavior_diagnostics or []
        return cls(
            status=RuntimeStatus.SUCCESS,
            reason="",
            task_id=task_id,
            node_count=node_count,
            results=results,
            metrics=metrics,
            diagnostics={
                DiagnosticsKey.PATH: path,
                DiagnosticsKey.EXECUTED_COUNT: executed_count,
                DiagnosticsKey.FAILED_NODES: failed_nodes or [],
                DiagnosticsKey.HAS_ACTION: has_action,
                DiagnosticsKey.MISSION_TRACE_ID: mission_trace_id,
                DiagnosticsKey.BEHAVIOR_DIAGNOSTICS: _bdiags,
                DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS: _bdiags,  # Circle 2 Step 2.3: alias
                DiagnosticsKey.BEHAVIOR_TRACE_IDS: behavior_trace_ids or {},
                DiagnosticsKey.HEARTBEAT_DIAGNOSTICS: heartbeat_diagnostics or [],  # Circle 2 Step 2.3
                DiagnosticsKey.HEARTBEAT_STATUS: "",  # Circle 2 Step 2.3: populated in Circle 3
                DiagnosticsKey.TIMELINE_DIAGNOSTICS: timeline_diagnostics or [],    # Fix 4
                DiagnosticsKey.PROTOCOL_DIAGNOSTICS: protocol_diagnostics or [],    # Step 1
            },
        )

    @classmethod
    def failed(
        cls,
        task_id: str,
        node_count: int,
        results: Dict[str, Any],
        metrics: Dict[str, Any],
        reason: str,
        path: str,
        executed_count: int = 0,
        failed_nodes: List[str] | None = None,
        has_action: bool = False,
        mission_trace_id: str = "",
        behavior_diagnostics: List[Dict[str, Any]] | None = None,
        behavior_trace_ids: Dict[str, str] | None = None,
        heartbeat_diagnostics: List[Dict[str, Any]] | None = None,  # Circle 2 Step 2.3
        timeline_diagnostics: List[Dict[str, Any]] | None = None,   # Fix 4
        protocol_diagnostics: List[Dict[str, Any]] | None = None,   # Step 1
    ) -> "RuntimeResult":
        _bdiags = behavior_diagnostics or []
        return cls(
            status=RuntimeStatus.FAILED,
            reason=reason,
            task_id=task_id,
            node_count=node_count,
            results=results,
            metrics=metrics,
            diagnostics={
                DiagnosticsKey.PATH: path,
                DiagnosticsKey.EXECUTED_COUNT: executed_count,
                DiagnosticsKey.FAILED_NODES: failed_nodes or [],
                DiagnosticsKey.HAS_ACTION: has_action,
                DiagnosticsKey.MISSION_TRACE_ID: mission_trace_id,
                DiagnosticsKey.BEHAVIOR_DIAGNOSTICS: _bdiags,
                DiagnosticsKey.BEHAVIOR_CORE_DIAGNOSTICS: _bdiags,  # Circle 2 Step 2.3: alias
                DiagnosticsKey.BEHAVIOR_TRACE_IDS: behavior_trace_ids or {},
                DiagnosticsKey.HEARTBEAT_DIAGNOSTICS: heartbeat_diagnostics or [],  # Circle 2 Step 2.3
                DiagnosticsKey.HEARTBEAT_STATUS: "",  # Circle 2 Step 2.3: populated in Circle 3
                DiagnosticsKey.TIMELINE_DIAGNOSTICS: timeline_diagnostics or [],    # Fix 4
                DiagnosticsKey.PROTOCOL_DIAGNOSTICS: protocol_diagnostics or [],    # Step 1
            },
        )

    @classmethod
    def blocked(
        cls,
        path: str,
        reason: str,
        emergency: Any = None,
    ) -> "RuntimeResult":
        """
        构建 blocked 结果。
        path 必须为 ExecutionPath.BLOCKED_* 常量之一。
        """
        diag: Dict[str, Any] = {
            DiagnosticsKey.PATH: path,
        }
        if emergency is not None:
            diag[DiagnosticsKey.EMERGENCY] = emergency
        return cls(
            status=RuntimeStatus.BLOCKED,
            reason=reason,
            task_id="",
            node_count=0,
            results={},
            metrics={},
            diagnostics=diag,
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为普通 dict。

        兼容性保证：以下字段名与 Phase 1 前的 RuntimeEngine 返回完全一致，
        既有调用方可继续读取而无需修改：
          status / reason / task_id / node_count / results / metrics

        新增字段 diagnostics 为附加字段，老调用方安全忽略即可。
        blocked 路径额外携带 phase（从 diagnostics.path 可推导），
        以维持 legacy "{status, phase, reason}" 三元组可读性。
        """
        d: Dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "task_id": self.task_id,
            "node_count": self.node_count,
            "results": self.results,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
        }
        # Legacy: blocked 路径附带 phase 字段（兼容老调用方）
        # 使用 _BLOCKED_PATH_TO_PHASE 常量映射，不做字符串切片。
        path = self.diagnostics.get(DiagnosticsKey.PATH, "")
        phase = _BLOCKED_PATH_TO_PHASE.get(path)
        if phase is not None:
            d["phase"] = phase
            # Legacy: safety blocked 附带 emergency
            emergency = self.diagnostics.get(DiagnosticsKey.EMERGENCY)
            if emergency is not None:
                d["emergency"] = emergency
        return d
