#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7 Test Matrix — STAGE-07.

Cross-cutting evidence that Phase 7 migration and cleanup is complete.

  SECTION A — Default execution path end-to-end
    A1: WorkflowIR through RuntimeEngine — success result with correct path.
    A2: No compat_path marker on default (flow-aware) path.
    A3: RuntimeResult contract fields all present.

  SECTION B — M6 compat gate end-to-end (UNITPORT_FLOW_AWARE_EXECUTION=0)
    B1: Compat diagnostics injected when env var is set.
    B2: compat_reason mentions controlling env var.
    B3: Compat path still produces a valid RuntimeResult.

  SECTION C — M7 bridge-absent gate end-to-end
    C1: Behavior node skipped (not failed) when bridge not injected.
    C2: Skipped behavior node carries compat_path marker.
    C3: Mission succeeds even when behavior nodes are skipped.

  SECTION D — Node migration proof (M1 / M2)
    D1: ActionExecutionNode.execute() routes through RobotContext.run_action().
    D2: SensorInputNode.execute() routes through RobotContext.get_sensor_data().
    D3: Neither node has a direct robot_model fallback path.

  SECTION E — ServiceRouter path distinction
    E1: execute_with_lifecycle emits no DEPRECATED warning.
    E2: Legacy run_action emits DEPRECATED.
    E3: Legacy methods still return correct values (no regression).

  SECTION F — Phase 3/4/5/6 regression proof
    F1: Safety-blocked path still returns blocked result with phase="safety".
    F2: Compile-blocked path still returns blocked result with phase="compile".
    F3: failed_nodes tracking still works correctly.
    F4: adapter_name still propagated on execute_with_lifecycle result.

Isolation
---------
No Qt, no real SDK / hardware.  All robot model and adapter calls are mocked.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Section A: Default execution path end-to-end ─────────────────────────────

class TestDefaultPathEndToEnd(unittest.TestCase):
    """WorkflowIR through RuntimeEngine on the default flow-aware path."""

    def _run_empty_workflowir(self, env: dict | None = None):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        base_env = os.environ.copy()
        base_env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
        if env:
            base_env.update(env)
        with patch.dict(os.environ, base_env, clear=True):
            return engine.execute(WorkflowIR(nodes=[], edges=[]), {})

    def test_empty_workflowir_returns_success(self):
        result = self._run_empty_workflowir()
        self.assertEqual(result["status"], "success")

    def test_default_path_is_workflow_ir(self):
        result = self._run_empty_workflowir()
        self.assertEqual(result["diagnostics"]["path"], "workflow_ir")

    def test_no_compat_path_on_default(self):
        result = self._run_empty_workflowir()
        self.assertNotIn("compat_path", result["diagnostics"])

    def test_result_contract_fields_present(self):
        result = self._run_empty_workflowir()
        for key in ("status", "reason", "task_id", "node_count", "results", "metrics", "diagnostics"):
            self.assertIn(key, result, f"Missing field: {key}")

    def test_diagnostics_contract_fields_present(self):
        result = self._run_empty_workflowir()
        for key in ("path", "executed_count", "failed_nodes", "has_action"):
            self.assertIn(key, result["diagnostics"], f"Missing diagnostics key: {key}")


# ── Section B: M6 compat gate end-to-end ─────────────────────────────────────

class TestCompatGateM6EndToEnd(unittest.TestCase):
    """UNITPORT_FLOW_AWARE_EXECUTION=0 activates topological sort and injects
    compat diagnostics into the RuntimeResult."""

    def _run_compat(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "0"}):
            return engine.execute(WorkflowIR(nodes=[], edges=[]), {})

    def test_compat_path_true_on_env0(self):
        result = self._run_compat()
        self.assertTrue(result["diagnostics"].get("compat_path"))

    def test_compat_reason_mentions_env_var(self):
        result = self._run_compat()
        self.assertIn("UNITPORT_FLOW_AWARE_EXECUTION", result["diagnostics"].get("compat_reason", ""))

    def test_compat_path_result_still_valid(self):
        result = self._run_compat()
        self.assertIn(result["status"], ("success", "failed"))
        self.assertIn("diagnostics", result)

    def test_compat_path_absent_when_env_is_1(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "1"}):
            result = engine.execute(WorkflowIR(nodes=[], edges=[]), {})
        self.assertNotIn("compat_path", result["diagnostics"])


# ── Section C: M7 bridge-absent gate end-to-end ───────────────────────────────

class TestCompatGateM7EndToEnd(unittest.TestCase):
    """When no behavior bridge is injected, behavior nodes return a skipped
    result with compat_path=True and do NOT fail the mission."""

    def _mission_with_behavior_node(self):
        """Dict-based mission IR with one behavior node (no edges → entry node)."""
        return {
            "nodes": [
                {
                    "id": "b1",
                    "type": "behavior",
                    "data": {"schema_id": "behavior", "behavior_ref": "test_walk"},
                    "inputs": [],
                    "outputs": [],
                }
            ],
            "connections": [],
        }

    def _run(self):
        from system.runtime.runtime_engine import RuntimeEngine

        engine = RuntimeEngine()          # behavior_bridge=None (default)
        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
        with patch.dict(os.environ, env, clear=True):
            return engine.execute(self._mission_with_behavior_node(), {})

    def test_behavior_node_skipped_not_failed(self):
        result = self._run()
        node_result = result["results"].get("b1", {})
        self.assertEqual(node_result.get("status"), "skipped")

    def test_behavior_node_carries_compat_path_marker(self):
        result = self._run()
        self.assertTrue(result["results"].get("b1", {}).get("compat_path"))

    def test_mission_succeeds_with_skipped_behavior(self):
        result = self._run()
        self.assertEqual(result["status"], "success")

    def test_skipped_behavior_not_in_failed_nodes(self):
        result = self._run()
        self.assertNotIn("b1", result["diagnostics"].get("failed_nodes", []))


# ── Section D: Node migration proof (M1 / M2) ────────────────────────────────

class TestNodeMigrationProof(unittest.TestCase):
    """ActionExecutionNode and SensorInputNode must route through RobotContext,
    not call robot_model.* directly."""

    def test_action_node_calls_robotcontext_run_action(self):
        from nodes.sys_nodes.action_nodes import ActionExecutionNode

        node = ActionExecutionNode("test_action")
        node.set_parameter("action", "stand")

        with patch("bin.core.robot_context.RobotContext") as mock_rc:
            mock_rc.run_action.return_value = True
            mock_rc.get_robot_type.return_value = "go2"
            node.execute({})

        mock_rc.run_action.assert_called_once_with("stand")

    def test_sensor_node_calls_robotcontext_get_sensor_data(self):
        from nodes.sys_nodes.sensor_nodes import SensorInputNode

        node = SensorInputNode("test_sensor")
        node.set_parameter("sensor_type", "imu")

        with patch("bin.core.robot_context.RobotContext") as mock_rc:
            mock_rc.get_sensor_data.return_value = {"imu": {"roll": 0.0}}
            mock_rc.get_robot_type.return_value = "go2"
            node.execute({})

        mock_rc.get_sensor_data.assert_called_once()

    def test_action_node_no_robot_model_param(self):
        """ActionExecutionNode must not read 'robot_model' from parameters."""
        from nodes.sys_nodes.action_nodes import ActionExecutionNode

        node = ActionExecutionNode("test_action")
        node.set_parameter("action", "walk")
        # Inject a trap as robot_model parameter — if the node calls it, the test fails
        trap = MagicMock(spec=[])  # empty spec: any attribute access raises AttributeError
        node.set_parameter("robot_model", trap)

        with patch("bin.core.robot_context.RobotContext") as mock_rc:
            mock_rc.run_action.return_value = True
            mock_rc.get_robot_type.return_value = "go2"
            result = node.execute({})

        # trap must never have been called
        trap.assert_not_called()

    def test_sensor_node_no_robot_model_param(self):
        """SensorInputNode must not read 'robot_model' from parameters."""
        from nodes.sys_nodes.sensor_nodes import SensorInputNode

        node = SensorInputNode("test_sensor")
        trap = MagicMock(spec=[])
        node.set_parameter("robot_model", trap)

        with patch("bin.core.robot_context.RobotContext") as mock_rc:
            mock_rc.get_sensor_data.return_value = {}
            mock_rc.get_robot_type.return_value = "go2"
            node.execute({})

        trap.assert_not_called()


# ── Section E: ServiceRouter path distinction ─────────────────────────────────

class TestServiceRouterPathDistinction(unittest.TestCase):
    """execute_with_lifecycle is the recommended path; legacy passthroughs are
    deprecated.  Both must still work functionally."""

    def _make_registry_with_mock(self):
        from system.service.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        mock_adapter = MagicMock()
        mock_adapter.open_session.return_value = {"status": "ok"}
        mock_adapter.run_action.return_value = True
        mock_adapter.get_sensor_data.return_value = {"battery": 80}
        mock_adapter.stop.return_value = None
        mock_adapter.health.return_value = {"status": "ok"}
        registry.register("test_adapter", mock_adapter)
        return registry

    def test_execute_with_lifecycle_no_deprecated_warning(self):
        from system.service.service_router import ServiceRouter, RouteOp

        router = ServiceRouter(self._make_registry_with_mock())
        with patch("system.service.service_router._log") as mock_log:
            router.execute_with_lifecycle("test_adapter", RouteOp.RUN_ACTION, {"action": "stand"})

        deprecated = [c for c in mock_log.warning.call_args_list
                      if "DEPRECATED" in str(c.args[0] if c.args else "")]
        self.assertEqual(deprecated, [])

    def test_legacy_run_action_emits_deprecated(self):
        from system.service.service_router import ServiceRouter

        router = ServiceRouter(self._make_registry_with_mock())
        with patch("system.service.service_router._log") as mock_log:
            router.run_action("test_adapter", "stand")

        texts = [str(c.args[0]) for c in mock_log.warning.call_args_list]
        self.assertTrue(any("DEPRECATED" in t for t in texts))

    def test_legacy_run_action_still_returns_value(self):
        from system.service.service_router import ServiceRouter

        router = ServiceRouter(self._make_registry_with_mock())
        with patch("system.service.service_router._log"):
            result = router.run_action("test_adapter", "stand")
        self.assertTrue(result)

    def test_execute_with_lifecycle_result_has_adapter_name(self):
        from system.service.service_router import ServiceRouter, RouteOp

        router = ServiceRouter(self._make_registry_with_mock())
        result = router.execute_with_lifecycle("test_adapter", RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result.get("adapter_name"), "test_adapter")


# ── Section F: Phase 3/4/5/6 regression proof ────────────────────────────────

class TestPhase3456RegressionProof(unittest.TestCase):
    """Critical contracts established in Phase 3/4/5/6 must remain intact."""

    def test_safety_blocked_returns_blocked_status(self):
        """Phase 3 contract: SafetyChecker intercept → status=blocked, phase=safety."""
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        with patch.object(engine._safety_checker, "check",
                          return_value={"ok": False, "reason": "unsafe_action"}):
            with patch.object(engine._emergency_handler, "handle", return_value={"handled": True}):
                result = engine.execute(WorkflowIR(nodes=[], edges=[]), {})

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result.get("phase"), "safety")

    def test_compile_blocked_returns_blocked_status(self):
        """Phase 3 contract: CompileGuard intercept → status=blocked, phase=compile."""
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        with patch.object(engine._compile_guard, "check",
                          return_value={"ok": False, "reason": "compile_error"}):
            result = engine.execute(WorkflowIR(nodes=[], edges=[]), {})

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result.get("phase"), "compile")

    def test_failed_nodes_tracking_still_works(self):
        """Phase 2 contract: nodes returning {"error": ...} appear in failed_nodes."""
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)

        # Inject a node result with error key via mocked executor
        with patch.object(engine._executor, "execute",
                          return_value={"node1": {"error": "something went wrong"}}):
            with patch.dict(os.environ, env, clear=True):
                result = engine.execute(WorkflowIR(nodes=[], edges=[]), {})

        self.assertIn("node1", result["diagnostics"]["failed_nodes"])
        self.assertEqual(result["status"], "failed")

    def test_adapter_name_propagated_on_lifecycle_result(self):
        """Phase 4 contract: adapter_name always present in execute_with_lifecycle result."""
        from system.service.service_router import ServiceRouter, RouteOp
        from system.service.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        mock_adapter = MagicMock()
        mock_adapter.open_session.return_value = {"status": "ok"}
        mock_adapter.run_action.return_value = True
        registry.register("spot_sdk", mock_adapter)
        router = ServiceRouter(registry)

        result = router.execute_with_lifecycle("spot_sdk", RouteOp.RUN_ACTION, {"action": "sit"})
        self.assertEqual(result["adapter_name"], "spot_sdk")

    def test_telemetry_present_on_lifecycle_result(self):
        """Phase 6 contract: telemetry always present in diagnostics."""
        from system.service.service_router import ServiceRouter, RouteOp
        from system.service.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        mock_adapter = MagicMock()
        mock_adapter.open_session.return_value = {"status": "ok"}
        mock_adapter.run_action.return_value = True
        registry.register("unitree_sdk2", mock_adapter)
        router = ServiceRouter(registry)

        result = router.execute_with_lifecycle("unitree_sdk2", RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertIn("telemetry", result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
