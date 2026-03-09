#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/runtime/contracts.py — RuntimeResult contract.

STEP-06: covers result contract, status constants, blocked-path compat,
         diagnostics structure, and legacy field backward-compat.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.runtime.contracts import (
    DiagnosticsKey,
    ExecutionPath,
    RuntimeResult,
    RuntimeStatus,
    _BLOCKED_PATH_TO_PHASE,
)


class TestRuntimeStatus(unittest.TestCase):
    def test_constants_are_strings(self):
        for val in (RuntimeStatus.SUCCESS, RuntimeStatus.FAILED, RuntimeStatus.BLOCKED):
            self.assertIsInstance(val, str)

    def test_constant_values(self):
        self.assertEqual(RuntimeStatus.SUCCESS, "success")
        self.assertEqual(RuntimeStatus.FAILED, "failed")
        self.assertEqual(RuntimeStatus.BLOCKED, "blocked")


class TestExecutionPath(unittest.TestCase):
    def test_all_constants_defined(self):
        expected = {
            "EXECUTION_GRAPH", "WORKFLOW_IR",
            "BLOCKED_COMPILE", "BLOCKED_EXECUTE", "BLOCKED_SAFETY",
        }
        for name in expected:
            self.assertTrue(hasattr(ExecutionPath, name), f"missing {name}")

    def test_blocked_path_to_phase_mapping(self):
        self.assertEqual(_BLOCKED_PATH_TO_PHASE[ExecutionPath.BLOCKED_COMPILE], "compile")
        self.assertEqual(_BLOCKED_PATH_TO_PHASE[ExecutionPath.BLOCKED_EXECUTE], "execute")
        self.assertEqual(_BLOCKED_PATH_TO_PHASE[ExecutionPath.BLOCKED_SAFETY], "safety")
        # Non-blocked paths must NOT appear in mapping
        self.assertNotIn(ExecutionPath.EXECUTION_GRAPH, _BLOCKED_PATH_TO_PHASE)
        self.assertNotIn(ExecutionPath.WORKFLOW_IR, _BLOCKED_PATH_TO_PHASE)


class TestRuntimeResultSuccess(unittest.TestCase):
    def _make(self, **overrides):
        defaults = dict(
            task_id="t1",
            node_count=3,
            results={"n1": {"ok": True}},
            metrics={"events": 1},
            path=ExecutionPath.WORKFLOW_IR,
            executed_count=3,
            failed_nodes=[],
            has_action=True,
        )
        defaults.update(overrides)
        return RuntimeResult.success(**defaults)

    def test_status_is_success(self):
        r = self._make()
        self.assertEqual(r.status, RuntimeStatus.SUCCESS)

    def test_reason_empty(self):
        r = self._make()
        self.assertEqual(r.reason, "")

    def test_diagnostics_keys(self):
        r = self._make(failed_nodes=["n2"])
        d = r.diagnostics
        self.assertEqual(d[DiagnosticsKey.PATH], ExecutionPath.WORKFLOW_IR)
        self.assertEqual(d[DiagnosticsKey.EXECUTED_COUNT], 3)
        self.assertEqual(d[DiagnosticsKey.FAILED_NODES], ["n2"])
        self.assertTrue(d[DiagnosticsKey.HAS_ACTION])

    def test_to_dict_core_fields(self):
        d = self._make().to_dict()
        for key in ("status", "reason", "task_id", "node_count", "results", "metrics"):
            self.assertIn(key, d, f"missing legacy field '{key}'")

    def test_to_dict_no_phase_field_for_success(self):
        d = self._make().to_dict()
        self.assertNotIn("phase", d)


class TestRuntimeResultFailed(unittest.TestCase):
    def test_status_is_failed(self):
        r = RuntimeResult.failed(
            task_id="t2", node_count=2, results={},
            metrics={}, reason="node_execution_failed",
            path=ExecutionPath.WORKFLOW_IR,
        )
        self.assertEqual(r.status, RuntimeStatus.FAILED)
        self.assertEqual(r.reason, "node_execution_failed")

    def test_failed_nodes_propagated(self):
        r = RuntimeResult.failed(
            task_id="t3", node_count=1, results={"n1": {"error": "boom"}},
            metrics={}, reason="node_execution_failed",
            path=ExecutionPath.EXECUTION_GRAPH,
            failed_nodes=["n1"],
        )
        self.assertEqual(r.diagnostics[DiagnosticsKey.FAILED_NODES], ["n1"])


class TestRuntimeResultBlocked(unittest.TestCase):
    def _check_blocked(self, path, expected_phase):
        r = RuntimeResult.blocked(path=path, reason="blocked_reason")
        d = r.to_dict()
        self.assertEqual(d["status"], RuntimeStatus.BLOCKED)
        self.assertEqual(d["reason"], "blocked_reason")
        self.assertEqual(d.get("phase"), expected_phase,
                         f"phase mismatch for path={path}")
        # Blocked has empty task_id / node_count
        self.assertEqual(d["task_id"], "")
        self.assertEqual(d["node_count"], 0)

    def test_blocked_compile(self):
        self._check_blocked(ExecutionPath.BLOCKED_COMPILE, "compile")

    def test_blocked_execute(self):
        self._check_blocked(ExecutionPath.BLOCKED_EXECUTE, "execute")

    def test_blocked_safety_with_emergency(self):
        r = RuntimeResult.blocked(
            path=ExecutionPath.BLOCKED_SAFETY,
            reason="safety_violation",
            emergency={"action": "halt"},
        )
        d = r.to_dict()
        self.assertEqual(d["phase"], "safety")
        self.assertEqual(d["emergency"], {"action": "halt"})

    def test_blocked_safety_no_emergency_key_when_none(self):
        r = RuntimeResult.blocked(
            path=ExecutionPath.BLOCKED_SAFETY,
            reason="safety_violation",
        )
        d = r.to_dict()
        self.assertNotIn("emergency", d)

    def test_non_blocked_path_has_no_phase_key(self):
        # workflow_ir path is not blocked; to_dict() must not emit phase
        r = RuntimeResult.success(
            task_id="t", node_count=1, results={}, metrics={},
            path=ExecutionPath.WORKFLOW_IR,
        )
        self.assertNotIn("phase", r.to_dict())


class TestRuntimeResultDiagnosticsKey(unittest.TestCase):
    def test_phase1_key_constants_are_strings(self):
        for attr in ("PATH", "EXECUTED_COUNT", "FAILED_NODES", "HAS_ACTION", "EMERGENCY"):
            val = getattr(DiagnosticsKey, attr)
            self.assertIsInstance(val, str)

    # Phase 2 STAGE-05 additions
    def test_phase2_key_constants_are_strings(self):
        for attr in ("MISSION_TRACE_ID", "BEHAVIOR_TRACE_IDS", "BEHAVIOR_DIAGNOSTICS"):
            val = getattr(DiagnosticsKey, attr)
            self.assertIsInstance(val, str, f"DiagnosticsKey.{attr} must be a str")

    def test_phase2_key_constants_are_nonempty(self):
        for attr in ("MISSION_TRACE_ID", "BEHAVIOR_TRACE_IDS", "BEHAVIOR_DIAGNOSTICS"):
            val = getattr(DiagnosticsKey, attr)
            self.assertTrue(val, f"DiagnosticsKey.{attr} must be non-empty")

    def test_all_diagnostics_key_constants_are_distinct(self):
        all_keys = [
            DiagnosticsKey.PATH, DiagnosticsKey.EXECUTED_COUNT,
            DiagnosticsKey.FAILED_NODES, DiagnosticsKey.HAS_ACTION,
            DiagnosticsKey.EMERGENCY, DiagnosticsKey.MISSION_TRACE_ID,
            DiagnosticsKey.BEHAVIOR_TRACE_IDS, DiagnosticsKey.BEHAVIOR_DIAGNOSTICS,
        ]
        self.assertEqual(len(set(all_keys)), len(all_keys))


class TestRuntimeResultPhase2Defaults(unittest.TestCase):
    """Phase 2 factory params default to empty/neutral values (backward compat)."""

    def _success(self, **kw) -> dict:
        defaults = dict(
            task_id="t", node_count=1, results={}, metrics={},
            path=ExecutionPath.WORKFLOW_IR,
        )
        defaults.update(kw)
        return RuntimeResult.success(**defaults).diagnostics

    def _failed(self, **kw) -> dict:
        defaults = dict(
            task_id="t", node_count=1, results={}, metrics={},
            reason="err", path=ExecutionPath.WORKFLOW_IR,
        )
        defaults.update(kw)
        return RuntimeResult.failed(**defaults).diagnostics

    def test_success_default_mission_trace_id_empty(self):
        self.assertEqual(self._success()[DiagnosticsKey.MISSION_TRACE_ID], "")

    def test_success_default_behavior_diagnostics_empty_list(self):
        self.assertEqual(self._success()[DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])

    def test_success_default_behavior_trace_ids_empty_dict(self):
        self.assertEqual(self._success()[DiagnosticsKey.BEHAVIOR_TRACE_IDS], {})

    def test_failed_default_mission_trace_id_empty(self):
        self.assertEqual(self._failed()[DiagnosticsKey.MISSION_TRACE_ID], "")

    def test_failed_default_behavior_diagnostics_empty_list(self):
        self.assertEqual(self._failed()[DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], [])

    def test_failed_default_behavior_trace_ids_empty_dict(self):
        self.assertEqual(self._failed()[DiagnosticsKey.BEHAVIOR_TRACE_IDS], {})

    def test_success_accepts_mission_trace_id(self):
        d = self._success(mission_trace_id="custom-tid")
        self.assertEqual(d[DiagnosticsKey.MISSION_TRACE_ID], "custom-tid")

    def test_success_accepts_behavior_trace_ids(self):
        d = self._success(behavior_trace_ids={"b0": "trace-abc"})
        self.assertEqual(d[DiagnosticsKey.BEHAVIOR_TRACE_IDS], {"b0": "trace-abc"})

    def test_success_accepts_behavior_diagnostics(self):
        diag = [{"level": "error", "code": "E1", "message": "m"}]
        d = self._success(behavior_diagnostics=diag)
        self.assertEqual(d[DiagnosticsKey.BEHAVIOR_DIAGNOSTICS], diag)


# ===========================================================================
# Circle 1 Step 1.1 — DiagnosticsKey.BEHAVIOR_ENABLED_RUN / EXECUTION_PATH_REASON
# and RuntimeEngine diagnostics surface tests
# ===========================================================================

class TestBehaviorEnabledRunDiagnosticsKeys(unittest.TestCase):
    """Verify the two new constants exist and have correct values."""

    def test_behavior_enabled_run_constant_exists(self):
        self.assertEqual(DiagnosticsKey.BEHAVIOR_ENABLED_RUN, "behavior_enabled_run")

    def test_execution_path_reason_constant_exists(self):
        self.assertEqual(DiagnosticsKey.EXECUTION_PATH_REASON, "execution_path_reason")


class TestRuntimeEngineBehaviorPathDiagnostics(unittest.TestCase):
    """RuntimeEngine.execute() must surface BEHAVIOR_ENABLED_RUN + EXECUTION_PATH_REASON."""

    def _run_exec_graph(self, scenario_extra=None):
        """Execute a minimal exec_graph (legacy path) and return the run_result dict."""
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()
        # Minimal exec_graph that starts, runs one action stub, ends.
        exec_graph = {
            "nodes": {
                "s": {"name": "Start",  "type": "start",  "logic_node": None},
                "e": {"name": "End",    "type": "end",    "logic_node": None},
            },
            "outgoing": {"s": {"flow_out": [("e", "flow_in")]}, "e": {}},
            "incoming": {"s": {}, "e": {"flow_in": [("s", "flow_out")]}},
            "entry_nodes": ["s"],
        }
        scenario = {"robot_model": None}
        if scenario_extra:
            scenario.update(scenario_extra)
        return engine.execute(exec_graph, scenario)

    def _run_workflow_ir(self, scenario_extra=None):
        """Execute a minimal WorkflowIR (new path) and return the run_result dict."""
        from system.runtime.runtime_engine import RuntimeEngine
        from compiler.ir.workflow_ir import WorkflowIR, IRNode, IREdge, NodeKind, EdgeType
        engine = RuntimeEngine()
        ir = WorkflowIR(
            nodes=[
                IRNode(id="s0", schema_id="start", kind=NodeKind.START),
                IRNode(id="e0", schema_id="end",   kind=NodeKind.END),
            ],
            edges=[
                IREdge(from_node="s0", from_port="flow_out",
                       to_node="e0", to_port="flow_in",
                       edge_type=EdgeType.FLOW),
            ],
        )
        scenario = {"robot_model": None}
        if scenario_extra:
            scenario.update(scenario_extra)
        return engine.execute(ir, scenario)

    # ── exec_graph path ────────────────────────────────────────────────────────

    def test_exec_graph_behavior_enabled_run_is_false(self):
        result = self._run_exec_graph()
        diag = result.get("diagnostics", {})
        self.assertIn(DiagnosticsKey.BEHAVIOR_ENABLED_RUN, diag)
        self.assertFalse(diag[DiagnosticsKey.BEHAVIOR_ENABLED_RUN])

    def test_exec_graph_execution_path_reason_default(self):
        result = self._run_exec_graph()
        diag = result.get("diagnostics", {})
        self.assertIn(DiagnosticsKey.EXECUTION_PATH_REASON, diag)
        self.assertEqual(diag[DiagnosticsKey.EXECUTION_PATH_REASON], "exec_graph_compat")

    def test_exec_graph_ui_reason_tag_forwarded(self):
        """UI injects _behavior_path_reason into scenario; engine must surface it."""
        result = self._run_exec_graph(
            scenario_extra={"_behavior_path_reason": "workflowir_compile_failed_fallback"}
        )
        diag = result.get("diagnostics", {})
        self.assertEqual(
            diag[DiagnosticsKey.EXECUTION_PATH_REASON],
            "workflowir_compile_failed_fallback",
        )
        # Even though path reason says "failed_fallback", the exec_graph path was used
        # → behavior_enabled_run stays False.
        self.assertFalse(diag[DiagnosticsKey.BEHAVIOR_ENABLED_RUN])

    # ── WorkflowIR path ────────────────────────────────────────────────────────

    def test_workflowir_behavior_enabled_run_is_true(self):
        result = self._run_workflow_ir()
        diag = result.get("diagnostics", {})
        self.assertIn(DiagnosticsKey.BEHAVIOR_ENABLED_RUN, diag)
        self.assertTrue(diag[DiagnosticsKey.BEHAVIOR_ENABLED_RUN])

    def test_workflowir_execution_path_reason_default(self):
        result = self._run_workflow_ir()
        diag = result.get("diagnostics", {})
        self.assertIn(DiagnosticsKey.EXECUTION_PATH_REASON, diag)
        self.assertEqual(diag[DiagnosticsKey.EXECUTION_PATH_REASON], "workflowir_direct")

    def test_workflowir_ui_reason_tag_forwarded(self):
        """UI injects _behavior_path_reason; engine must surface it on WorkflowIR path too."""
        result = self._run_workflow_ir(
            scenario_extra={"_behavior_path_reason": "workflowir_behavior_enabled"}
        )
        diag = result.get("diagnostics", {})
        self.assertEqual(
            diag[DiagnosticsKey.EXECUTION_PATH_REASON],
            "workflowir_behavior_enabled",
        )
        self.assertTrue(diag[DiagnosticsKey.BEHAVIOR_ENABLED_RUN])

    def test_both_new_keys_present_in_to_dict(self):
        """Keys must survive to_dict() serialisation (used by MissionRunThread signal)."""
        result = self._run_exec_graph()
        self.assertIn("diagnostics", result)
        d = result["diagnostics"]
        self.assertIn(DiagnosticsKey.BEHAVIOR_ENABLED_RUN, d)
        self.assertIn(DiagnosticsKey.EXECUTION_PATH_REASON, d)


class TestBehaviorRunFlags(unittest.TestCase):
    """BehaviorRunFlags env-var and default behaviour."""

    def _flags(self, env_val=None):
        import os
        from system.runtime.migration import BehaviorRunFlags, _BEHAVIOR_ENV_KEY
        old = os.environ.pop(_BEHAVIOR_ENV_KEY, None)
        try:
            if env_val is not None:
                os.environ[_BEHAVIOR_ENV_KEY] = env_val
            return BehaviorRunFlags.from_env()
        finally:
            if old is not None:
                os.environ[_BEHAVIOR_ENV_KEY] = old
            elif _BEHAVIOR_ENV_KEY in os.environ:
                del os.environ[_BEHAVIOR_ENV_KEY]

    def test_default_is_false(self):
        self.assertFalse(self._flags().use_workflowir_for_behavior)

    def test_env_0_is_false(self):
        self.assertFalse(self._flags("0").use_workflowir_for_behavior)

    def test_env_1_is_true(self):
        self.assertTrue(self._flags("1").use_workflowir_for_behavior)

    def test_env_true_is_true(self):
        self.assertTrue(self._flags("true").use_workflowir_for_behavior)

    def test_to_dict_has_source_env_key(self):
        from system.runtime.migration import BehaviorRunFlags
        d = BehaviorRunFlags().to_dict()
        self.assertIn("source_env_key", d)
        self.assertIn("use_workflowir_for_behavior", d)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 4 — TIMELINE_DIAGNOSTICS key in success/failed results
# ===========================================================================

class TestTimelineDiagnosticsKey(unittest.TestCase):
    """Verify RuntimeResult.success() and .failed() expose TIMELINE_DIAGNOSTICS."""

    _BASE_KWARGS = dict(
        task_id="t1",
        node_count=1,
        results={},
        metrics={},
        path=ExecutionPath.WORKFLOW_IR,
    )

    def test_success_has_timeline_diagnostics_key_default_empty(self):
        r = RuntimeResult.success(**self._BASE_KWARGS)
        self.assertIn(DiagnosticsKey.TIMELINE_DIAGNOSTICS, r.diagnostics)
        self.assertEqual(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS], [])

    def test_success_timeline_diagnostics_populated(self):
        td = [{"motor_id": "m1", "level": "warning"}]
        r = RuntimeResult.success(**self._BASE_KWARGS, timeline_diagnostics=td)
        self.assertEqual(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS], td)

    def test_failed_has_timeline_diagnostics_key_default_empty(self):
        r = RuntimeResult.failed(**self._BASE_KWARGS, reason="err")
        self.assertIn(DiagnosticsKey.TIMELINE_DIAGNOSTICS, r.diagnostics)
        self.assertEqual(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS], [])

    def test_failed_timeline_diagnostics_populated(self):
        td = [{"motor_id": "m2", "level": "error"}]
        r = RuntimeResult.failed(**self._BASE_KWARGS, reason="err", timeline_diagnostics=td)
        self.assertEqual(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS], td)

    def test_blocked_does_not_have_timeline_diagnostics(self):
        r = RuntimeResult.blocked(path=ExecutionPath.BLOCKED_COMPILE, reason="compile_failed")
        # blocked results don't have TIMELINE_DIAGNOSTICS — that's by design
        self.assertNotIn(DiagnosticsKey.TIMELINE_DIAGNOSTICS, r.diagnostics)


if __name__ == "__main__":
    unittest.main()
