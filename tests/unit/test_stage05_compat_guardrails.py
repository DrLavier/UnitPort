"""Phase 7 STAGE-05 — Compatibility guardrails and deprecation policy tests.

Verifies that:
  - DiagnosticsKey exposes COMPAT_PATH and COMPAT_REASON constants.
  - RuntimeResult diagnostics carry compat markers only when compat path is active.
  - Bridge-absent behavior skip result carries "compat_path": True.
  - ServiceRouter legacy passthrough methods emit DEPRECATED log_warning.
  - execute_with_lifecycle does NOT emit DEPRECATED warnings.

Sections:
    A — DiagnosticsKey new constants
    B — Compat diagnostics absent on default (flow-aware) path
    C — Compat diagnostics present on compat (topological) path
    D — Bridge-absent skip result carries compat_path marker
    E — ServiceRouter legacy methods emit deprecation warning
    F — execute_with_lifecycle does not emit DEPRECATED warning
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch


# ── Section A: DiagnosticsKey new constants ───────────────────────────────────

class TestDiagnosticsKeyCompatConstants(unittest.TestCase):

    def test_compat_path_constant_exists(self):
        from system.runtime.contracts import DiagnosticsKey
        self.assertTrue(hasattr(DiagnosticsKey, "COMPAT_PATH"))

    def test_compat_reason_constant_exists(self):
        from system.runtime.contracts import DiagnosticsKey
        self.assertTrue(hasattr(DiagnosticsKey, "COMPAT_REASON"))

    def test_compat_path_value_is_string(self):
        from system.runtime.contracts import DiagnosticsKey
        self.assertIsInstance(DiagnosticsKey.COMPAT_PATH, str)

    def test_compat_reason_value_is_string(self):
        from system.runtime.contracts import DiagnosticsKey
        self.assertIsInstance(DiagnosticsKey.COMPAT_REASON, str)


# ── Section B: Compat diagnostics absent on default path ─────────────────────

class TestCompatDiagnosticsAbsentOnDefault(unittest.TestCase):
    """When flow-aware path is active (default), compat_path / compat_reason
    must NOT appear in diagnostics — they are absent, not False."""

    def _run_workflowir(self, env_override=None):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
        if env_override:
            env.update(env_override)

        with patch.object(engine._executor, "execute", return_value={}):
            with patch.dict(os.environ, env, clear=True):
                return engine.execute(WorkflowIR(nodes=[], edges=[]), {})

    def test_compat_path_absent_when_env_unset(self):
        result = self._run_workflowir()
        self.assertNotIn("compat_path", result["diagnostics"])

    def test_compat_reason_absent_when_env_unset(self):
        result = self._run_workflowir()
        self.assertNotIn("compat_reason", result["diagnostics"])

    def test_compat_path_absent_when_env_is_1(self):
        result = self._run_workflowir({"UNITPORT_FLOW_AWARE_EXECUTION": "1"})
        self.assertNotIn("compat_path", result["diagnostics"])

    def test_compat_reason_absent_when_env_is_1(self):
        result = self._run_workflowir({"UNITPORT_FLOW_AWARE_EXECUTION": "1"})
        self.assertNotIn("compat_reason", result["diagnostics"])


# ── Section C: Compat diagnostics present on compat path ─────────────────────

class TestCompatDiagnosticsPresentOnCompatPath(unittest.TestCase):
    """When UNITPORT_FLOW_AWARE_EXECUTION=0, RuntimeResult diagnostics must carry
    compat_path=True and a compat_reason that names the controlling env var."""

    def _run_workflowir_compat(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        with patch.object(engine._executor, "execute", return_value={}):
            with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "0"}):
                return engine.execute(WorkflowIR(nodes=[], edges=[]), {})

    def test_compat_path_present(self):
        result = self._run_workflowir_compat()
        self.assertIn("compat_path", result["diagnostics"])

    def test_compat_path_is_true(self):
        result = self._run_workflowir_compat()
        self.assertTrue(result["diagnostics"]["compat_path"])

    def test_compat_reason_present(self):
        result = self._run_workflowir_compat()
        self.assertIn("compat_reason", result["diagnostics"])

    def test_compat_reason_is_string(self):
        result = self._run_workflowir_compat()
        self.assertIsInstance(result["diagnostics"]["compat_reason"], str)

    def test_compat_reason_mentions_env_var(self):
        result = self._run_workflowir_compat()
        self.assertIn("UNITPORT_FLOW_AWARE_EXECUTION", result["diagnostics"]["compat_reason"])


# ── Section D: Bridge-absent skip result carries compat_path marker ───────────

class TestBridgeAbsentCompatMarker(unittest.TestCase):
    """When _behavior_bridge is None, _execute_behavior_node() must include
    "compat_path": True in the skipped result to signal the compat gate."""

    def _make_behavior_node(self, behavior_ref: str = "ref1"):
        return {
            "id": "b1",
            "type": "behavior",
            "data": {"schema_id": "behavior", "behavior_ref": behavior_ref},
            "inputs": [],
            "outputs": [],
        }

    def test_bridge_absent_result_has_compat_path_key(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertIn("compat_path", result)

    def test_bridge_absent_compat_path_is_true(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertTrue(result["compat_path"])

    def test_bridge_absent_status_still_skipped(self):
        """Adding compat_path marker must not change the status value."""
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertEqual(result["status"], "skipped")

    def test_bridge_absent_no_error_key(self):
        """compat_path marker must not introduce an error key (regression guard)."""
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertNotIn("error", result)


# ── Section E: ServiceRouter legacy methods emit DEPRECATED warning ───────────

class TestLegacyRouterDeprecationWarnings(unittest.TestCase):
    """Each Phase 1 legacy passthrough must emit a _log.warning that starts
    with 'DEPRECATED' so operators can detect compat-path usage in logs."""

    def _make_router(self):
        from system.service.service_router import ServiceRouter
        from system.service.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        mock_adapter = MagicMock()
        mock_adapter.run_action.return_value = True
        mock_adapter.get_sensor_data.return_value = {}
        mock_adapter.stop.return_value = None
        mock_adapter.health.return_value = {"status": "ok"}
        registry.register("test_adapter", mock_adapter)
        return ServiceRouter(registry)

    def _warning_texts(self, mock_log):
        return [str(c.args[0]) for c in mock_log.warning.call_args_list]

    def test_run_action_emits_deprecated_warning(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.run_action("test_adapter", "stand")
        self.assertTrue(
            any("DEPRECATED" in w for w in self._warning_texts(mock_log)),
            "Expected DEPRECATED warning from run_action",
        )

    def test_stop_emits_deprecated_warning(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.stop("test_adapter")
        self.assertTrue(
            any("DEPRECATED" in w for w in self._warning_texts(mock_log)),
            "Expected DEPRECATED warning from stop",
        )

    def test_get_sensor_data_emits_deprecated_warning(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.get_sensor_data("test_adapter")
        self.assertTrue(
            any("DEPRECATED" in w for w in self._warning_texts(mock_log)),
            "Expected DEPRECATED warning from get_sensor_data",
        )

    def test_health_emits_deprecated_warning(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.health("test_adapter")
        self.assertTrue(
            any("DEPRECATED" in w for w in self._warning_texts(mock_log)),
            "Expected DEPRECATED warning from health",
        )

    def test_run_action_warning_mentions_execute_with_lifecycle(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.run_action("test_adapter", "stand")
        all_text = " ".join(self._warning_texts(mock_log))
        self.assertIn("execute_with_lifecycle", all_text)

    def test_stop_warning_mentions_execute_with_lifecycle(self):
        router = self._make_router()
        with patch("system.service.service_router._log") as mock_log:
            router.stop("test_adapter")
        all_text = " ".join(self._warning_texts(mock_log))
        self.assertIn("execute_with_lifecycle", all_text)

    def test_legacy_methods_still_return_correct_values(self):
        """Deprecation warnings must not affect return values."""
        router = self._make_router()
        with patch("system.service.service_router._log"):
            action_result = router.run_action("test_adapter", "stand")
            sensor_result = router.get_sensor_data("test_adapter")
            health_result = router.health("test_adapter")
        self.assertTrue(action_result)
        self.assertIsInstance(sensor_result, dict)
        self.assertEqual(health_result["status"], "ok")


# ── Section F: execute_with_lifecycle does NOT emit DEPRECATED warning ─────────

class TestExecuteWithLifecycleNoDeprecation(unittest.TestCase):
    """execute_with_lifecycle is the recommended path; it must never emit
    a DEPRECATED warning — only the legacy passthrough methods do."""

    def test_execute_with_lifecycle_no_deprecated_warning(self):
        from system.service.service_router import ServiceRouter, RouteOp
        from system.service.service_registry import ServiceRegistry

        registry = ServiceRegistry()
        mock_adapter = MagicMock()
        mock_adapter.open_session.return_value = {"status": "ok"}
        mock_adapter.run_action.return_value = True
        registry.register("test_adapter", mock_adapter)
        router = ServiceRouter(registry)

        with patch("system.service.service_router._log") as mock_log:
            router.execute_with_lifecycle(
                "test_adapter", RouteOp.RUN_ACTION, {"action": "stand"}
            )

        deprecated_calls = [
            c for c in mock_log.warning.call_args_list
            if "DEPRECATED" in str(c.args[0] if c.args else "")
        ]
        self.assertEqual(deprecated_calls, [])


if __name__ == "__main__":
    unittest.main()
