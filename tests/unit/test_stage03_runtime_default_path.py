"""Phase 7 STAGE-03 — Runtime default-path consolidation tests.

Verifies that:
  - Flow-aware DFS is the stable default execution path.
  - Topological-sort compat path is only activatable via explicit env var.
  - Compat-path activation emits a log_warning (explicit, not silent).
  - Behavior bridge absent → deterministic "skipped" result (M7 compat gate).
  - MigrationFlags.to_dict() contract is stable.

Sections:
    A — MigrationFlags defaults
    B — Env-var gating
    C — MigrationFlags.to_dict() contract
    D — RuntimeEngine sets use_flow_aware_execution=True by default
    E — NodeExecutor default attribute
    F — Compat-path activation emits log_warning
    G — Behavior bridge absent → skipped (M7 compat gate)
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch, call


# ── Section A: MigrationFlags defaults ───────────────────────────────────────

class TestMigrationFlagsDefaults(unittest.TestCase):

    def test_default_use_flow_aware_execution_is_true(self):
        from system.runtime.migration import MigrationFlags
        flags = MigrationFlags()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_from_env_default_is_flow_aware(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_from_env_explicit_1_is_flow_aware(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "1"}):
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_from_env_returns_migrationflags_instance(self):
        from system.runtime.migration import MigrationFlags
        flags = MigrationFlags.from_env()
        self.assertIsInstance(flags, MigrationFlags)


# ── Section B: Env-var gating ─────────────────────────────────────────────────

class TestEnvVarGating(unittest.TestCase):

    def test_env_0_activates_topological_sort(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "0"}):
            flags = MigrationFlags.from_env()
        self.assertFalse(flags.use_flow_aware_execution)

    def test_env_false_activates_compat(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "false"}):
            flags = MigrationFlags.from_env()
        self.assertFalse(flags.use_flow_aware_execution)

    def test_env_off_activates_compat(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "off"}):
            flags = MigrationFlags.from_env()
        self.assertFalse(flags.use_flow_aware_execution)

    def test_env_no_activates_compat(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "no"}):
            flags = MigrationFlags.from_env()
        self.assertFalse(flags.use_flow_aware_execution)

    def test_env_1_keeps_flow_aware(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "1"}):
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_env_true_keeps_flow_aware(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "true"}):
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_env_on_keeps_flow_aware(self):
        from system.runtime.migration import MigrationFlags
        with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": "on"}):
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)

    def test_compat_mode_is_not_default(self):
        """Compat path (False) requires explicit opt-in; unset env → True."""
        from system.runtime.migration import MigrationFlags
        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
        with patch.dict(os.environ, env, clear=True):
            flags = MigrationFlags.from_env()
        self.assertTrue(flags.use_flow_aware_execution)


# ── Section C: MigrationFlags.to_dict() contract ─────────────────────────────

class TestMigrationFlagsToDict(unittest.TestCase):

    def _flags_dict(self, flow_aware: bool) -> dict:
        from system.runtime.migration import MigrationFlags
        flags = MigrationFlags(use_flow_aware_execution=flow_aware)
        return flags.to_dict()

    def test_to_dict_has_use_flow_aware_execution_key(self):
        d = self._flags_dict(True)
        self.assertIn("use_flow_aware_execution", d)

    def test_to_dict_has_source_env_key(self):
        d = self._flags_dict(True)
        self.assertIn("source_env_key", d)

    def test_to_dict_flow_aware_true(self):
        d = self._flags_dict(True)
        self.assertTrue(d["use_flow_aware_execution"])

    def test_to_dict_flow_aware_false(self):
        d = self._flags_dict(False)
        self.assertFalse(d["use_flow_aware_execution"])

    def test_to_dict_source_env_key_is_string(self):
        d = self._flags_dict(True)
        self.assertIsInstance(d["source_env_key"], str)

    def test_to_dict_source_env_key_matches_env_var_name(self):
        d = self._flags_dict(True)
        self.assertEqual(d["source_env_key"], "UNITPORT_FLOW_AWARE_EXECUTION")


# ── Section D: RuntimeEngine sets use_flow_aware_execution=True by default ────

class TestRuntimeEngineDefaultPath(unittest.TestCase):
    """Verify RuntimeEngine passes use_flow_aware_execution=True to executor
    by default (no env override)."""

    def _run_workflowir_mission(self, env_override: dict | None = None):
        """Run a minimal WorkflowIR mission through RuntimeEngine.
        Returns the executor's use_flow_aware_execution value at the time execute() was called.
        patch.object on the instance avoids class-level mutation.
        """
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        captured = {}
        engine = RuntimeEngine()

        def spy(context=None, **_kw):
            captured["use_flow_aware"] = engine._executor.use_flow_aware_execution
            return {}

        mission_ir = WorkflowIR(nodes=[], edges=[])
        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)
        if env_override:
            env.update(env_override)

        with patch.object(engine._executor, "execute", side_effect=spy):
            with patch.dict(os.environ, env, clear=True):
                engine.execute(mission_ir, {})

        return captured.get("use_flow_aware")

    def test_default_executor_gets_flow_aware_true(self):
        result = self._run_workflowir_mission()
        self.assertTrue(result)

    def test_env_0_executor_gets_flow_aware_false(self):
        result = self._run_workflowir_mission({"UNITPORT_FLOW_AWARE_EXECUTION": "0"})
        self.assertFalse(result)

    def test_env_1_executor_gets_flow_aware_true(self):
        result = self._run_workflowir_mission({"UNITPORT_FLOW_AWARE_EXECUTION": "1"})
        self.assertTrue(result)


# ── Section E: NodeExecutor default attribute ─────────────────────────────────

class TestNodeExecutorDefault(unittest.TestCase):

    def test_node_executor_default_use_flow_aware_is_true(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        self.assertTrue(executor.use_flow_aware_execution)

    def test_node_executor_use_flow_aware_can_be_set_false(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor.use_flow_aware_execution = False
        self.assertFalse(executor.use_flow_aware_execution)

    def test_node_executor_reset_after_set(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor.use_flow_aware_execution = False
        executor.use_flow_aware_execution = True
        self.assertTrue(executor.use_flow_aware_execution)


# ── Section F: Compat-path activation emits log_warning ──────────────────────

class TestCompatPathWarning(unittest.TestCase):
    """When UNITPORT_FLOW_AWARE_EXECUTION=0, RuntimeEngine must emit a
    log_warning so the non-default path is always visible in logs.
    patch.object on the executor instance avoids class-level mutation."""

    def _run_with_warning_capture(self, env_value: str):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        warnings_emitted = []

        with patch.object(engine._executor, "execute", return_value={}):
            with patch("system.runtime.runtime_engine.log_warning",
                       side_effect=lambda msg, *a, **kw: warnings_emitted.append(msg)):
                with patch.dict(os.environ, {"UNITPORT_FLOW_AWARE_EXECUTION": env_value}):
                    engine.execute(WorkflowIR(nodes=[], edges=[]), {})

        return warnings_emitted

    def test_compat_path_env0_emits_warning(self):
        warnings = self._run_with_warning_capture("0")
        self.assertTrue(
            any("compat" in w.lower() or "topological" in w.lower() for w in warnings),
            f"Expected compat-path warning, got: {warnings}",
        )

    def test_default_path_no_compat_warning(self):
        from system.runtime.runtime_engine import RuntimeEngine
        from system.ir.workflow_ir import WorkflowIR

        engine = RuntimeEngine()
        warnings_emitted = []

        env = os.environ.copy()
        env.pop("UNITPORT_FLOW_AWARE_EXECUTION", None)

        with patch.object(engine._executor, "execute", return_value={}):
            with patch("system.runtime.runtime_engine.log_warning",
                       side_effect=lambda msg, *a, **kw: warnings_emitted.append(msg)):
                with patch.dict(os.environ, env, clear=True):
                    engine.execute(WorkflowIR(nodes=[], edges=[]), {})

        compat_warnings = [
            w for w in warnings_emitted
            if "compat" in w.lower() or "topological" in w.lower()
        ]
        self.assertEqual(compat_warnings, [])

    def test_warning_message_mentions_env_var(self):
        warnings = self._run_with_warning_capture("0")
        all_text = " ".join(warnings)
        self.assertIn("UNITPORT_FLOW_AWARE_EXECUTION", all_text)


# ── Section G: Behavior bridge absent → skipped (M7 compat gate) ─────────────

class TestBehaviorBridgeAbsentSkip(unittest.TestCase):
    """When _behavior_bridge is None, behavior nodes return {"status": "skipped"}
    and do NOT raise an error.  This is the M7 explicit compat gate."""

    def _make_behavior_node(self, node_id: str = "b1", behavior_ref: str = "my_behavior"):
        return {
            "id": node_id,
            "type": "behavior",
            "data": {"schema_id": "behavior", "behavior_ref": behavior_ref},
            "inputs": [],
            "outputs": [],
        }

    def test_behavior_node_returns_skipped_when_bridge_is_none(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertEqual(result["status"], "skipped")

    def test_behavior_node_skipped_result_has_reason(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertIn("reason", result)

    def test_behavior_node_skipped_reason_contains_ref(self):
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(behavior_ref="walk_forward"),
            inputs={}, context={},
        )
        self.assertIn("walk_forward", result["reason"])

    def test_bridge_none_invoker_set_still_skips(self):
        """Invoker set but bridge=None → still skipped (bridge is the gate)."""
        from system.runtime.node_executor import NodeExecutor
        from system.runtime.behavior_invoker import BehaviorSubgraphInvoker
        executor = NodeExecutor()
        executor._behavior_invoker = BehaviorSubgraphInvoker()
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertEqual(result["status"], "skipped")

    def test_skipped_result_has_no_error_key(self):
        """Skipped result must NOT have an 'error' key — RuntimeEngine would
        add skipped nodes to failed_nodes if it did."""
        from system.runtime.node_executor import NodeExecutor
        executor = NodeExecutor()
        executor._behavior_invoker = None
        executor._behavior_bridge = None

        result = executor._execute_behavior_node(
            self._make_behavior_node(), inputs={}, context={}
        )
        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
