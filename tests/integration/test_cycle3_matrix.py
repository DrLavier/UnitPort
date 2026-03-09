#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration test matrix — Product Cycle 3 (P2).

Covers end-to-end behavior for all Cycle 3 stages:
  01. Settings persistence (save → load → run round-trip)
  02. Policy scoping (run-scoped policy; class-level not mutated)
  03. Adapter cancel contract (RuntimeEngine + adapter integration)
  04. Unsaved-change guardrails (run/open/close gate branches)
  05. Advanced MuJoCo settings (save/load round-trip; runtime handoff)
  06. Live capability sync (refresh after run/abort)
  07. Regression: Phase 3-7 critical suites remain unaffected

Non-regression invariants:
  - Cycle 1 UX paths (diagnostics, status badges, mission save/load) unchanged
  - Cycle 2 async execution, cancel, settings validation unchanged
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)


def _make_window():
    from bin.core.config_manager import ConfigManager
    from bin.ui import MainWindow
    return MainWindow(ConfigManager())


def _close_window(win):
    with patch.object(win, "_handle_unsaved_settings_guard", return_value=True):
        win.close()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 01/02: Settings persistence round-trip (save → load → apply → run)
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_01_SettingsPersistence(unittest.TestCase):
    """Cycle 3 STAGE-01/02: SDK settings survive save/load cycle."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_save_injects_settings_key(self):
        """Serialised mission must contain 'settings' with brand+config."""
        with patch("bin.ui.QFileDialog.getSaveFileName",
                   return_value=("/tmp/c3_test.unitport", "")):
            with patch("builtins.open", mock_open()):
                with patch("json.dump") as mock_dump:
                    self.win._on_save()
        saved = mock_dump.call_args[0][0]
        self.assertIn("settings", saved)
        self.assertIn("brand", saved["settings"])
        self.assertIn("config", saved["settings"])

    def test_load_restores_matching_brand_settings(self):
        """Loading a mission with matching brand must restore SDK settings."""
        from bin.core.robot_context import RobotContext
        current_brand = RobotContext.get_current_brand()

        mission = {
            "nodes": [],
            "connections": [],
            "schema_version": "1.1",
            "settings": {"brand": current_brand, "config": {"test_field": "restored_value"}},
        }
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/c3_test.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(mission))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    with patch.object(
                        self.win.main_zone, "set_sdk_brand"
                    ) as mock_set_brand:
                        self.win._on_open()

        mock_set_brand.assert_called_once()
        called_brand, called_config = mock_set_brand.call_args[0]
        self.assertEqual(called_brand, current_brand)
        self.assertEqual(called_config.get("test_field"), "restored_value")

    def test_load_skips_mismatched_brand_settings(self):
        """Loading a mission with mismatched brand must NOT restore SDK settings."""
        mission = {
            "nodes": [],
            "connections": [],
            "settings": {"brand": "__no_such_brand__", "config": {"key": "val"}},
        }
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/mismatch.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(mission))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    with patch.object(self.win.main_zone, "set_sdk_brand") as mock_set_brand:
                        self.win._on_open()

        mock_set_brand.assert_not_called()

    def test_load_backward_compat_no_settings_key(self):
        """Old mission files without 'settings' key must load without error."""
        mission = {"nodes": [], "connections": []}
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/old.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(mission))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    try:
                        self.win._on_open()
                    except Exception as exc:
                        self.fail(f"_on_open() must not raise on old mission file: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 02: Runtime policy scoping — class-level policy not mutated
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_02_PolicyScoping(unittest.TestCase):
    """Cycle 3 STAGE-02/03: Per-run policy keeps class-level policy immutable."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_on_run_does_not_mutate_class_policy(self):
        """_on_run() must not change the class-level lifecycle policy."""
        from bin.core.robot_context import RobotContext
        policy_before = RobotContext.get_lifecycle_policy()
        brand_before = policy_before.brand if hasattr(policy_before, "brand") else None

        with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
            with patch.object(self.win, "_validate_settings_pre_run", return_value=False):
                # Validation fails → no thread started, but guard + policy logic runs
                self.win._on_run()

        policy_after = RobotContext.get_lifecycle_policy()
        brand_after = policy_after.brand if hasattr(policy_after, "brand") else None
        self.assertEqual(brand_before, brand_after,
                         "Class-level lifecycle policy brand must not change after _on_run()")

    def test_make_run_policy_is_pure_factory(self):
        """make_run_policy() must return a LifecyclePolicy without side effects."""
        from bin.core.robot_context import RobotContext
        policy_before = RobotContext.get_lifecycle_policy()
        new_policy = RobotContext.make_run_policy({"brand": "unitree"})
        policy_after = RobotContext.get_lifecycle_policy()
        # Class-level policy unchanged
        self.assertIs(policy_before, policy_after)
        # Returned policy is a new object
        self.assertIsNot(new_policy, policy_before)

    def test_on_mission_finished_does_not_mutate_class_policy(self):
        """_on_mission_finished() must not change the class-level lifecycle policy."""
        from bin.core.robot_context import RobotContext
        policy_before = RobotContext.get_lifecycle_policy()

        self.win._last_exec_graph = {"nodes": {}, "connections": []}
        run_result = {
            "status": "failed", "reason": "mission_cancelled", "results": {},
            "diagnostics": {"failed_nodes": [], "executed_count": 0},
        }
        with patch.object(self.win, "_refresh_capability_inspector"):
            self.win._on_mission_finished(run_result)

        policy_after = RobotContext.get_lifecycle_policy()
        self.assertIs(policy_before, policy_after)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 03: Adapter cancel contract — integration
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_03_AdapterCancelContract(unittest.TestCase):
    """Cycle 3 STAGE-04: cancel_action() integrated into RuntimeEngine."""

    def test_cancel_action_present_on_all_adapter_bases(self):
        """cancel_action() must exist on all concrete adapter classes."""
        from system.service.adapters.base_adapter import BaseAdapter
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        from system.service.adapters.spot_sdk.adapter import SpotAdapter
        from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter

        for cls in (BaseAdapter, UnitreeAdapter, SpotAdapter, CyberDogAdapter):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(
                    callable(getattr(cls, "cancel_action", None)),
                    f"{cls.__name__}.cancel_action() must exist",
                )

    def test_adapter_cancel_invoked_key_in_diagnostics(self):
        """ADAPTER_CANCEL_INVOKED must appear in diagnostics on adapter cancel."""
        from system.runtime.contracts import DiagnosticsKey
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()
        mock_model = MagicMock()

        def _cancel_and_return(**kw):
            engine.request_cancel()
            return {"ok": False, "reason": "mission_cancelled", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": True}

        exec_graph = {
            "nodes": {"1": {"name": "Start", "type": "start"}},
            "outgoing": {"1": {}},
            "incoming": {"1": {}},
            "entry_nodes": ["1"],
        }
        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_and_return):
            result = engine.execute(exec_graph, {"robot_model": mock_model})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "mission_cancelled")
        self.assertTrue(result["diagnostics"].get(DiagnosticsKey.ADAPTER_CANCEL_INVOKED))

    def test_cooperative_cancel_fallback_absent_from_diagnostics(self):
        """Without adapter model, ADAPTER_CANCEL_INVOKED must NOT appear."""
        from system.runtime.contracts import DiagnosticsKey
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()

        def _cancel_and_return(**kw):
            engine.request_cancel()
            return {"ok": False, "reason": "mission_cancelled", "results": {},
                    "executed_count": 0, "has_action": False, "cancelled": True}

        exec_graph = {
            "nodes": {"1": {"name": "Start", "type": "start"}},
            "outgoing": {"1": {}},
            "incoming": {"1": {}},
            "entry_nodes": ["1"],
        }
        with patch.object(engine._workflow_runner, "run", side_effect=_cancel_and_return):
            result = engine.execute(exec_graph, {})  # no robot_model

        self.assertNotIn(DiagnosticsKey.ADAPTER_CANCEL_INVOKED,
                         result.get("diagnostics", {}))


# ══════════════════════════════════════════════════════════════════════════════
# Stage 04: Unsaved-change guardrails — end-to-end branches
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_04_UnsavedChangeGuardrails(unittest.TestCase):
    """Cycle 3 STAGE-05: guardrails at run/open/close paths."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_run_blocked_on_cancel_choice(self):
        """Cancel in unsaved-changes dialog must block _on_run()."""
        with patch.object(self.win.main_zone.settings_panel,
                          "has_unsaved_changes", return_value=True):
            from tests.unit.test_unsaved_changes_guard import _make_msg_mock
            mock_msg, *_ = _make_msg_mock("cancel")
            with patch("bin.ui.QMessageBox", return_value=mock_msg):
                with patch("bin.ui.MissionRunThread") as mock_thread:
                    self.win._on_run()
        mock_thread.assert_not_called()

    def test_open_blocked_on_cancel_choice(self):
        """Cancel in unsaved-changes dialog must block _on_open() file dialog."""
        with patch.object(self.win.main_zone.settings_panel,
                          "has_unsaved_changes", return_value=True):
            from tests.unit.test_unsaved_changes_guard import _make_msg_mock
            mock_msg, *_ = _make_msg_mock("cancel")
            with patch("bin.ui.QMessageBox", return_value=mock_msg):
                with patch("bin.ui.QFileDialog.getOpenFileName") as mock_dialog:
                    self.win._on_open()
        mock_dialog.assert_not_called()

    def test_run_proceeds_on_discard_choice(self):
        """Discard in unsaved-changes dialog must allow _on_run() to proceed."""
        with patch.object(self.win.main_zone.settings_panel,
                          "has_unsaved_changes", return_value=True):
            from tests.unit.test_unsaved_changes_guard import _make_msg_mock
            mock_msg, *_ = _make_msg_mock("discard")
            with patch("bin.ui.QMessageBox", return_value=mock_msg):
                with patch.object(self.win, "_validate_settings_pre_run", return_value=False):
                    with patch.object(self.win, "_refresh_capability_inspector"):
                        self.win._on_run()
        # Reached _validate_settings_pre_run → guard did not block

    def test_no_dialog_when_no_unsaved_changes(self):
        """With no unsaved changes, no guardrail dialog must appear."""
        with patch.object(self.win.main_zone.settings_panel,
                          "has_unsaved_changes", return_value=False):
            with patch("bin.ui.QMessageBox") as mock_qmb_cls:
                with patch.object(self.win, "_validate_settings_pre_run", return_value=False):
                    with patch.object(self.win, "_refresh_capability_inspector"):
                        self.win._on_run()
        mock_qmb_cls.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 05/06: Advanced MuJoCo settings round-trip + runtime handoff
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_05_AdvancedMuJoCoRoundTrip(unittest.TestCase):
    """Cycle 3 STAGE-06: advanced settings persist and appear in scenario."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_scenario_settings_saved_and_restored(self):
        """Advanced MuJoCo settings must survive a save → load cycle."""
        # Step 1: configure advanced settings
        self.win.main_zone.set_scenario_settings({
            "mujoco_gravity_z": -5.0,
            "mujoco_solver_iterations": 200,
            "mujoco_render_width": 1280,
            "mujoco_render_height": 720,
        })

        # Step 2: save
        saved_data = {}
        with patch("bin.ui.QFileDialog.getSaveFileName",
                   return_value=("/tmp/c3_adv.unitport", "")):
            with patch("builtins.open", mock_open()):
                with patch("json.dump") as mock_dump:
                    self.win._on_save()
        saved_data = mock_dump.call_args[0][0]
        self.assertIn("scenario_settings", saved_data)
        self.assertAlmostEqual(
            saved_data["scenario_settings"].get("mujoco_gravity_z"), -5.0
        )

        # Step 3: load back
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/c3_adv.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(saved_data))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    self.win._on_open()

        # Step 4: verify restored
        loaded_settings = self.win.main_zone.get_scenario_settings()
        self.assertAlmostEqual(loaded_settings.get("mujoco_gravity_z"), -5.0)
        self.assertEqual(loaded_settings.get("mujoco_solver_iterations"), 200)

    def test_scenario_settings_included_in_runtime_scenario(self):
        """Advanced MuJoCo settings must appear in the scenario dict at run time."""
        self.win.main_zone.set_scenario_settings({
            "mujoco_gravity_z": -3.0,
            "mujoco_render_width": 800,
        })
        settings = self.win._get_runtime_scenario_settings()
        self.assertAlmostEqual(settings.get("mujoco_gravity_z"), -3.0)
        self.assertEqual(settings.get("mujoco_render_width"), 800)

    def test_old_mission_without_scenario_settings_loads_cleanly(self):
        """Old mission files without 'scenario_settings' must load without error."""
        old_mission = {"nodes": [], "connections": [], "schema_version": "1.0"}
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/old.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(old_mission))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    try:
                        self.win._on_open()
                    except Exception as exc:
                        self.fail(f"_on_open() must not raise on old file: {exc}")
        # Defaults must be preserved
        loaded = self.win.main_zone.get_scenario_settings()
        self.assertAlmostEqual(loaded.get("mujoco_gravity_z"), -9.81)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 06: Live capability sync — run completion and abort
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_06_LiveCapabilitySync(unittest.TestCase):
    """Cycle 3 STAGE-06: capability inspector refreshed after state transitions."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_refresh_called_after_successful_run(self):
        """Successful mission completion must trigger capability refresh."""
        self.win._last_exec_graph = {"nodes": {}, "connections": []}
        run_result = {
            "status": "success", "reason": "", "results": {},
            "diagnostics": {"failed_nodes": [], "executed_count": 0},
        }
        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            with patch("bin.ui.QMessageBox.warning"):
                self.win._on_mission_finished(run_result)
        mock_refresh.assert_called()

    def test_refresh_called_after_failed_run(self):
        """Failed mission must trigger capability refresh."""
        self.win._last_exec_graph = {"nodes": {}, "connections": []}
        run_result = {
            "status": "failed", "reason": "node_execution_failed",
            "results": {"1": {"error": "fail"}},
            "diagnostics": {"failed_nodes": ["1"], "executed_count": 1},
        }
        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            self.win._on_mission_finished(run_result)
        mock_refresh.assert_called()

    def test_refresh_called_after_abort(self):
        """Aborting a running mission must trigger capability refresh."""
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        mock_thread.wait.return_value = True
        self.win._mission_run_thread = mock_thread

        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            self.win._on_runtime_abort()
        mock_refresh.assert_called()

    def test_inspector_cleared_on_adapter_error(self):
        """When adapter.capabilities() raises, inspector must be cleared."""
        from bin.core.robot_context import RobotContext
        bad_adapter = MagicMock()
        bad_adapter.capabilities.side_effect = RuntimeError("offline")
        with patch.object(RobotContext, "_create_adapter_for_brand", return_value=bad_adapter):
            with patch.object(
                self.win.main_zone, "set_adapter_capabilities"
            ) as mock_set_cap:
                self.win._refresh_capability_inspector()
        mock_set_cap.assert_called_with({}, [])


# ══════════════════════════════════════════════════════════════════════════════
# Stage 07: Cycle 1+2 non-regression spot checks
# ══════════════════════════════════════════════════════════════════════════════


class TestC3_07_Regression(unittest.TestCase):
    """Key Cycle 1/2 behaviors remain stable under Cycle 3 changes."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        _close_window(self.win)

    def test_settings_validation_still_blocks_run(self):
        """Cycle 2 STAGE-05: settings validation must still block _on_run()."""
        from system.service.settings_validator import validate_settings
        with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
            with patch.object(self.win, "_refresh_capability_inspector"):
                with patch(
                    "system.service.settings_validator.validate_settings",
                    return_value={"status": "error", "missing": ["ip_address"], "invalid": []}
                ):
                    with patch("bin.ui.MissionRunThread") as mock_thread:
                        self.win._on_run()
        mock_thread.assert_not_called()

    def test_mission_schema_validation_still_blocks_open(self):
        """Cycle 1 STAGE-05: invalid schema must still block _on_open()."""
        bad_data = {"not_nodes": [], "connections": []}  # missing 'nodes'
        with patch("bin.ui.QFileDialog.getOpenFileName",
                   return_value=("/tmp/bad.unitport", "")):
            with patch("builtins.open", mock_open(read_data=json.dumps(bad_data))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    with patch.object(
                        self.win.graph_scene, "load_workflow"
                    ) as mock_load:
                        with patch("bin.ui.QMessageBox.warning"):
                            self.win._on_open()
        mock_load.assert_not_called()

    def test_request_cancel_sets_flag_cycle2_regression(self):
        """Cycle 2: RuntimeEngine.request_cancel() must set _cancel_requested."""
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()
        self.assertFalse(engine._cancel_requested)
        engine.request_cancel()
        self.assertTrue(engine._cancel_requested)

    def test_adapter_cancel_invoked_false_when_no_model(self):
        """Cycle 3 STAGE-04: no adapter model → _adapter_cancel_invoked stays False."""
        from system.runtime.runtime_engine import RuntimeEngine
        engine = RuntimeEngine()
        engine._current_robot_model = None
        engine.request_cancel()
        self.assertFalse(engine._adapter_cancel_invoked)

    def test_schema_version_in_saved_mission(self):
        """Cycle 3 STAGE-02: schema_version must be present in saved missions."""
        with patch("bin.ui.QFileDialog.getSaveFileName",
                   return_value=("/tmp/schema_test.unitport", "")):
            with patch("builtins.open", mock_open()):
                with patch("json.dump") as mock_dump:
                    self.win._on_save()
        saved = mock_dump.call_args[0][0]
        self.assertIn("schema_version", saved)
        self.assertEqual(saved["schema_version"], "1.4")


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 4 — TIMELINE_DIAGNOSTICS propagated to RuntimeResult
# ===========================================================================

class TestTimelineDiagsIntegration(unittest.TestCase):
    """Integration: timeline motor diags propagate from compile → run result."""

    def test_timeline_diagnostics_key_in_success_result(self):
        """RuntimeResult.success() always exposes TIMELINE_DIAGNOSTICS key."""
        from system.runtime.contracts import DiagnosticsKey, ExecutionPath, RuntimeResult
        r = RuntimeResult.success(
            task_id="t1",
            node_count=1,
            results={},
            metrics={},
            path=ExecutionPath.WORKFLOW_IR,
            timeline_diagnostics=[{"motor_id": "m1", "level": "warning"}],
        )
        self.assertIn(DiagnosticsKey.TIMELINE_DIAGNOSTICS, r.diagnostics)
        self.assertEqual(len(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS]), 1)

    def test_timeline_diagnostics_key_in_failed_result(self):
        from system.runtime.contracts import DiagnosticsKey, ExecutionPath, RuntimeResult
        r = RuntimeResult.failed(
            task_id="t2",
            node_count=1,
            results={},
            metrics={},
            reason="err",
            path=ExecutionPath.WORKFLOW_IR,
            timeline_diagnostics=[{"motor_id": "m2", "level": "error"}],
        )
        self.assertIn(DiagnosticsKey.TIMELINE_DIAGNOSTICS, r.diagnostics)
        self.assertEqual(r.diagnostics[DiagnosticsKey.TIMELINE_DIAGNOSTICS][0]["level"], "error")

    def test_collect_behavior_tracing_extracts_timeline_diags(self):
        """_collect_behavior_tracing must return 5-tuple including timeline_diags."""
        from system.runtime.runtime_engine import _collect_behavior_tracing

        results = {
            "n1": {
                "behavior_ref": "walk",
                "trace_id": "tid1",
                "diagnostics": [],
                "heartbeat_diagnostics": [],
                "timeline_diagnostics": [
                    {"motor_id": "m3", "level": "warning", "code": "timeline.motor_amp"}
                ],
            }
        }
        trace_ids, core_diags, hb_diags, tl_diags, proto_diags = _collect_behavior_tracing(results)
        self.assertEqual(len(tl_diags), 1)
        self.assertEqual(tl_diags[0]["motor_id"], "m3")

    def test_collect_behavior_tracing_empty_timeline_diags(self):
        from system.runtime.runtime_engine import _collect_behavior_tracing

        results = {
            "n1": {
                "behavior_ref": "stand",
                "trace_id": "tid2",
                "diagnostics": [],
                "heartbeat_diagnostics": [],
                # no timeline_diagnostics key
            }
        }
        trace_ids, core_diags, hb_diags, tl_diags, proto_diags = _collect_behavior_tracing(results)
        self.assertEqual(tl_diags, [])

    def test_node_executor_adds_timeline_diags_to_output(self):
        """node_executor must include timeline_diagnostics in behavior node output."""
        from system.runtime.node_executor import NodeExecutor
        from system.behavior.behavior_compiler_bridge import BehaviorCompilerBridge
        from system.behavior.behavior_artifact import BehaviorArtifact, BehaviorDiagnostic
        from system.ir.workflow_ir import IRNode, NodeKind, WorkflowIR
        from unittest.mock import MagicMock

        # Build a behavior node with a timeline diagnostic
        tl_diag = BehaviorDiagnostic(
            level="warning",
            code="timeline.motor_amplitude_out_of_range",
            message="amp out of range",
        )
        artifact = BehaviorArtifact.create(
            behavior_ref="walk",
            behavior_ir=WorkflowIR(nodes=[], edges=[]),
            diagnostics=[tl_diag],
        )

        bridge = BehaviorCompilerBridge()
        bridge._registry["walk"] = artifact

        ex = NodeExecutor()
        node = {
            "id": "b1",
            "data": {"behavior_ref": "walk"},
        }
        invoker = MagicMock()
        from system.behavior.behavior_artifact import BehaviorInvokeOutput
        invoker.invoke.return_value = BehaviorInvokeOutput.success(
            trace_id="t1",
            outputs={},
            node_results={},
            diagnostics=[tl_diag],
        )
        ex._behavior_invoker = invoker
        ex._behavior_bridge = bridge
        result = ex._execute_behavior_node(node, {}, context={})
        self.assertIn("timeline_diagnostics", result)
        tl_diags = result["timeline_diagnostics"]
        self.assertEqual(len(tl_diags), 1)
        self.assertTrue(tl_diags[0]["code"].startswith("timeline."))


if __name__ == "__main__":
    unittest.main()
