"""Runtime engine for workflow execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
from shared.ir.workflow_ir import WorkflowIR


@dataclass
class RuntimeEngine:
    """Executes a mission IR within a given scenario context."""

    _running: bool = field(default=False, init=False, repr=False)
    _executor: NodeExecutor = field(default_factory=NodeExecutor, init=False, repr=False)
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)
    _monitor: Monitor = field(default_factory=Monitor, init=False, repr=False)
    _workflow_runner: WorkflowRunner = field(default_factory=WorkflowRunner, init=False, repr=False)
    _compile_guard: CompileGuard = field(default_factory=CompileGuard, init=False, repr=False)
    _execute_guard: ExecuteGuard = field(default_factory=ExecuteGuard, init=False, repr=False)
    _safety_checker: SafetyChecker = field(default_factory=SafetyChecker, init=False, repr=False)
    _emergency_handler: EmergencyHandler = field(default_factory=EmergencyHandler, init=False, repr=False)
    _audit_logger: SafetyAuditLogger = field(default_factory=SafetyAuditLogger, init=False, repr=False)

    def execute(self, mission_ir: Any, scenario: Any) -> dict:
        """Execute *mission_ir* under *scenario* and return a result summary."""
        compile_check = self._compile_guard.check(mission_ir)
        if not compile_check["ok"]:
            self._audit_logger.record("compile_blocked", compile_check)
            return {"status": "blocked", "phase": "compile", "reason": compile_check["reason"]}

        execute_check = self._execute_guard.check(scenario or {})
        if not execute_check["ok"]:
            self._audit_logger.record("execute_blocked", execute_check)
            return {"status": "blocked", "phase": "execute", "reason": execute_check["reason"]}

        policy = SafetyPolicy.from_dict((scenario or {}).get("safety_policy", {}))
        safety_check = self._safety_checker.check(mission_ir, scenario or {}, policy)
        if not safety_check["ok"]:
            emergency = self._emergency_handler.handle(safety_check["reason"], safety_check)
            self._audit_logger.record("safety_blocked", {"check": safety_check, "emergency": emergency})
            return {"status": "blocked", "phase": "safety", "reason": safety_check["reason"], "emergency": emergency}

        self._running = True
        self._monitor.start()
        try:
            task_id = self._scheduler.schedule({"scenario": scenario})

            if self._is_execution_graph(mission_ir):
                self._workflow_runner.max_loop_iterations = policy.max_loop_iterations
                runner_out = self._workflow_runner.run(
                    exec_graph=mission_ir,
                    robot_model=(scenario or {}).get("robot_model"),
                    graph_scene=(scenario or {}).get("graph_scene"),
                    action_mapping=(scenario or {}).get("action_mapping"),
                )
                node_count = len(mission_ir.get("nodes", {}))
                results = runner_out.get("results", {})
                status = "success" if runner_out.get("ok") else "failed"
                reason = runner_out.get("reason")
            else:
                self._executor.clear()
                self._load_mission(mission_ir)
                node_count = len(self._executor.nodes)
                results = self._executor.execute(context={"scenario": scenario})
                status = "success"
                reason = ""

            self._monitor.bump_event()
            self._audit_logger.record("execution_completed", {"task_id": task_id, "node_count": node_count})
            return {
                "status": status,
                "task_id": task_id,
                "node_count": node_count,
                "results": results,
                "metrics": self._monitor.get_metrics(),
                "reason": reason,
            }
        finally:
            self._monitor.stop()
            self._running = False

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
