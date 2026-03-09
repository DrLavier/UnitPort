#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests — Cycle 3 STAGE-06: Live Capability Sync + Advanced MuJoCo Settings.

Covers:
  A. Advanced MuJoCo scenario defaults
       A1. New fields present in MainZonePanel._scenario_settings defaults
       A2. get_scenario_settings() returns all 8 fields
  B. set_scenario_settings() — restore from mission file
       B1. Known mujoco_* keys are applied
       B2. Unknown keys are silently ignored
  C. build_scenario_payload() and extract_scenario_payload()
       C1. Round-trip: build → extract returns original data
       C2. Missing key → extract returns None (backward compat)
       C3. Malformed key → extract returns None
  D. Mission save/load — scenario settings persistence (bin/ui.py wiring)
       D1. _on_save() injects scenario_settings into the data dict
       D2. _on_open() restores scenario settings from mission data
  E. Live capability refresh triggers
       E1. _on_mission_finished() calls _refresh_capability_inspector (success path)
       E2. _on_mission_finished() calls _refresh_capability_inspector (failure path)
       E3. _on_runtime_abort() calls _refresh_capability_inspector when aborted
  F. Graceful degradation
       F1. _refresh_capability_inspector() clears inspector when adapter is None
       F2. _refresh_capability_inspector() clears inspector on adapter exception

All tests run without a real robot, SDK, or active Qt event loop.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)


def _make_window():
    from bin.core.config_manager import ConfigManager
    from bin.ui import MainWindow
    return MainWindow(ConfigManager())


# ══════════════════════════════════════════════════════════════════════════════
# A — Advanced MuJoCo scenario defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestAdvancedMuJoCoDefaults(unittest.TestCase):
    """New advanced fields present in MainZonePanel's default scenario settings."""

    def setUp(self):
        from bin.layout.main_zone_panel import MainZonePanel
        self.panel = MainZonePanel()

    def tearDown(self):
        self.panel.close()

    # A1 ── New fields exist in _scenario_settings ────────────────────────────

    def test_mujoco_gravity_z_default(self):
        self.assertIn("mujoco_gravity_z", self.panel._scenario_settings)
        self.assertAlmostEqual(self.panel._scenario_settings["mujoco_gravity_z"], -9.81, places=2)

    def test_mujoco_solver_iterations_default(self):
        self.assertIn("mujoco_solver_iterations", self.panel._scenario_settings)
        self.assertEqual(self.panel._scenario_settings["mujoco_solver_iterations"], 100)

    def test_mujoco_render_width_default(self):
        self.assertIn("mujoco_render_width", self.panel._scenario_settings)
        self.assertEqual(self.panel._scenario_settings["mujoco_render_width"], 640)

    def test_mujoco_render_height_default(self):
        self.assertIn("mujoco_render_height", self.panel._scenario_settings)
        self.assertEqual(self.panel._scenario_settings["mujoco_render_height"], 480)

    # A2 ── get_scenario_settings() returns all 8 fields ──────────────────────

    def test_get_scenario_settings_returns_all_fields(self):
        settings = self.panel.get_scenario_settings()
        expected_keys = [
            "mujoco_gl_backend", "mujoco_realtime", "mujoco_timestep",
            "mujoco_scene_xml",
            "mujoco_gravity_z", "mujoco_solver_iterations",
            "mujoco_render_width", "mujoco_render_height",
        ]
        for key in expected_keys:
            self.assertIn(key, settings, f"Key '{key}' missing from get_scenario_settings()")


# ══════════════════════════════════════════════════════════════════════════════
# B — set_scenario_settings() restore
# ══════════════════════════════════════════════════════════════════════════════


class TestSetScenarioSettings(unittest.TestCase):
    """MainZonePanel.set_scenario_settings() applies saved values correctly."""

    def setUp(self):
        from bin.layout.main_zone_panel import MainZonePanel
        self.panel = MainZonePanel()

    def tearDown(self):
        self.panel.close()

    # B1 ── Known mujoco_* keys are applied ───────────────────────────────────

    def test_set_applies_known_mujoco_keys(self):
        self.panel.set_scenario_settings({
            "mujoco_gravity_z": -5.0,
            "mujoco_solver_iterations": 50,
            "mujoco_render_width": 1280,
            "mujoco_render_height": 720,
        })
        s = self.panel._scenario_settings
        self.assertAlmostEqual(s["mujoco_gravity_z"], -5.0)
        self.assertEqual(s["mujoco_solver_iterations"], 50)
        self.assertEqual(s["mujoco_render_width"], 1280)
        self.assertEqual(s["mujoco_render_height"], 720)

    def test_set_applies_all_standard_mujoco_keys(self):
        self.panel.set_scenario_settings({
            "mujoco_gl_backend": "egl",
            "mujoco_realtime": False,
            "mujoco_timestep": 0.005,
        })
        s = self.panel._scenario_settings
        self.assertEqual(s["mujoco_gl_backend"], "egl")
        self.assertFalse(s["mujoco_realtime"])
        self.assertAlmostEqual(s["mujoco_timestep"], 0.005)

    # B2 ── Unknown keys are silently ignored ─────────────────────────────────

    def test_set_ignores_unknown_keys(self):
        original_keys = set(self.panel._scenario_settings.keys())
        self.panel.set_scenario_settings({
            "unknown_key": "value",
            "another_random_key": 42,
        })
        # No new keys added, no error
        current_keys = set(self.panel._scenario_settings.keys())
        self.assertEqual(original_keys, current_keys)

    def test_set_partial_update_preserves_other_fields(self):
        """Updating only gravity_z must not wipe other fields."""
        self.panel.set_scenario_settings({"mujoco_gravity_z": -3.0})
        s = self.panel._scenario_settings
        self.assertAlmostEqual(s["mujoco_gravity_z"], -3.0)
        self.assertEqual(s["mujoco_solver_iterations"], 100)  # unchanged
        self.assertEqual(s["mujoco_render_width"], 640)        # unchanged


# ══════════════════════════════════════════════════════════════════════════════
# C — build_scenario_payload() and extract_scenario_payload()
# ══════════════════════════════════════════════════════════════════════════════


class TestScenarioPayloadHelpers(unittest.TestCase):
    """Round-trip and backward-compat for mission persistence helpers."""

    def setUp(self):
        from bin.core.mission_persistence import (
            build_scenario_payload,
            extract_scenario_payload,
        )
        self.build = build_scenario_payload
        self.extract = extract_scenario_payload

    # C1 ── Round-trip: build → extract ───────────────────────────────────────

    def test_roundtrip_basic(self):
        settings = {
            "mujoco_gl_backend": "egl",
            "mujoco_gravity_z": -5.0,
            "mujoco_solver_iterations": 50,
        }
        payload = self.build(settings)
        data = {"nodes": [], "connections": [], "scenario_settings": payload}
        recovered = self.extract(data)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["mujoco_gl_backend"], "egl")
        self.assertAlmostEqual(recovered["mujoco_gravity_z"], -5.0)
        self.assertEqual(recovered["mujoco_solver_iterations"], 50)

    def test_build_returns_shallow_copy(self):
        """build_scenario_payload must not share references with the original dict."""
        original = {"mujoco_gravity_z": -9.81}
        payload = self.build(original)
        payload["mujoco_gravity_z"] = -1.0
        self.assertAlmostEqual(original["mujoco_gravity_z"], -9.81)  # unchanged

    # C2 ── Missing key → None (backward compat) ──────────────────────────────

    def test_extract_missing_key_returns_none(self):
        data = {"nodes": [], "connections": []}
        self.assertIsNone(self.extract(data))

    def test_extract_none_input_returns_none(self):
        self.assertIsNone(self.extract(None))

    def test_extract_non_dict_input_returns_none(self):
        self.assertIsNone(self.extract("not a dict"))

    # C3 ── Malformed key → None ──────────────────────────────────────────────

    def test_extract_non_dict_value_returns_none(self):
        data = {"nodes": [], "connections": [], "scenario_settings": "bad"}
        self.assertIsNone(self.extract(data))

    def test_extract_list_value_returns_none(self):
        data = {"nodes": [], "connections": [], "scenario_settings": [1, 2, 3]}
        self.assertIsNone(self.extract(data))


# ══════════════════════════════════════════════════════════════════════════════
# D — Mission save/load wiring (bin/ui.py)
# ══════════════════════════════════════════════════════════════════════════════


class TestScenarioSettingsSaveLoad(unittest.TestCase):
    """_on_save() injects scenario_settings; _on_open() restores them."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
            self.win.close()

    # D1 ── _on_save() injects scenario_settings ──────────────────────────────

    def test_on_save_injects_scenario_settings(self):
        """_on_save() must add 'scenario_settings' key to the serialised dict."""
        captured = {}

        def _fake_save(path, mode, **kwargs):
            return captured

        import json

        class _FakeFile:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def write(self, s): pass

        # Intercept the json.dump call to capture what gets written
        with patch("bin.ui.QFileDialog.getSaveFileName", return_value=("/tmp/test.unitport", "")):
            with patch("builtins.open", return_value=_FakeFile()):
                with patch("json.dump") as mock_dump:
                    self.win._on_save()

        self.assertTrue(mock_dump.called, "_on_save must call json.dump")
        saved_data = mock_dump.call_args[0][0]
        self.assertIn("scenario_settings", saved_data,
                      "scenario_settings key must be present in saved mission")

    # D2 ── _on_open() restores scenario settings ─────────────────────────────

    def test_on_open_restores_scenario_settings(self):
        """_on_open() must call set_scenario_settings with the loaded data."""
        import json
        mission_data = {
            "nodes": [],
            "connections": [],
            "schema_version": "1.1",
            "scenario_settings": {
                "mujoco_gravity_z": -5.0,
                "mujoco_solver_iterations": 50,
            },
        }

        with patch("bin.ui.QFileDialog.getOpenFileName", return_value=("/tmp/test.unitport", "")):
            with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(mission_data))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    with patch.object(
                        self.win.main_zone, "set_scenario_settings"
                    ) as mock_set:
                        self.win._on_open()

        mock_set.assert_called_once()
        restored = mock_set.call_args[0][0]
        self.assertAlmostEqual(restored["mujoco_gravity_z"], -5.0)
        self.assertEqual(restored["mujoco_solver_iterations"], 50)

    def test_on_open_skips_restore_when_no_scenario_settings(self):
        """_on_open() must not call set_scenario_settings for old files."""
        import json
        old_mission_data = {
            "nodes": [],
            "connections": [],
            "schema_version": "1.0",
        }

        with patch("bin.ui.QFileDialog.getOpenFileName", return_value=("/tmp/old.unitport", "")):
            with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(old_mission_data))):
                with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
                    with patch.object(
                        self.win.main_zone, "set_scenario_settings"
                    ) as mock_set:
                        self.win._on_open()

        mock_set.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# E — Live capability refresh triggers
# ══════════════════════════════════════════════════════════════════════════════


class TestLiveCapabilityRefreshTriggers(unittest.TestCase):
    """_refresh_capability_inspector is called after run and abort."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
            self.win.close()

    # E1 ── _on_mission_finished (success) ────────────────────────────────────

    def test_on_mission_finished_refreshes_on_success(self):
        """Capability inspector must be refreshed after a successful run."""
        run_result = {
            "status": "success", "reason": "", "results": {},
            "diagnostics": {"failed_nodes": [], "executed_count": 0},
        }
        self.win._last_exec_graph = {"nodes": {}, "connections": []}
        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            with patch("bin.ui.QMessageBox.warning"):
                self.win._on_mission_finished(run_result)
        mock_refresh.assert_called()

    # E2 ── _on_mission_finished (failure) ────────────────────────────────────

    def test_on_mission_finished_refreshes_on_failure(self):
        """Capability inspector must be refreshed after a failed/cancelled run."""
        run_result = {
            "status": "failed", "reason": "mission_cancelled", "results": {},
            "diagnostics": {"failed_nodes": [], "executed_count": 0},
        }
        self.win._last_exec_graph = {"nodes": {}, "connections": []}
        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            self.win._on_mission_finished(run_result)
        mock_refresh.assert_called()

    # E3 ── _on_runtime_abort ─────────────────────────────────────────────────

    def test_on_runtime_abort_refreshes_when_thread_aborted(self):
        """Capability inspector must be refreshed after aborting a mission thread."""
        mock_thread = MagicMock()
        mock_thread.isRunning.return_value = True
        mock_thread.wait.return_value = True
        self.win._mission_run_thread = mock_thread

        with patch.object(self.win, "_refresh_capability_inspector") as mock_refresh:
            self.win._on_runtime_abort()

        mock_refresh.assert_called()


# ══════════════════════════════════════════════════════════════════════════════
# F — Graceful degradation in _refresh_capability_inspector
# ══════════════════════════════════════════════════════════════════════════════


class TestCapabilityInspectorDegradation(unittest.TestCase):
    """_refresh_capability_inspector() clears inspector on failure."""

    def setUp(self):
        self.win = _make_window()

    def tearDown(self):
        with patch.object(self.win, "_handle_unsaved_settings_guard", return_value=True):
            self.win.close()

    # F1 ── Adapter is None → inspector cleared ───────────────────────────────

    def test_clears_inspector_when_adapter_is_none(self):
        """When _create_adapter_for_brand returns None, inspector is set to empty."""
        with patch.object(
            type(self.win).__mro__[0],  # MainWindow
            "_refresh_capability_inspector",
            wraps=self.win._refresh_capability_inspector,
        ):
            pass  # just verify we can wrap

        from bin.core.robot_context import RobotContext
        with patch.object(RobotContext, "_create_adapter_for_brand", return_value=None):
            with patch.object(
                self.win.main_zone, "set_adapter_capabilities"
            ) as mock_set_cap:
                self.win._refresh_capability_inspector()

        mock_set_cap.assert_called_once_with({}, [])

    # F2 ── Adapter.capabilities() raises → inspector cleared ─────────────────

    def test_clears_inspector_on_adapter_exception(self):
        """When adapter.capabilities() raises, inspector is set to empty."""
        mock_adapter = MagicMock()
        mock_adapter.capabilities.side_effect = RuntimeError("connection refused")

        from bin.core.robot_context import RobotContext
        with patch.object(RobotContext, "_create_adapter_for_brand", return_value=mock_adapter):
            with patch.object(
                self.win.main_zone, "set_adapter_capabilities"
            ) as mock_set_cap:
                # Should not raise
                try:
                    self.win._refresh_capability_inspector()
                except Exception as exc:
                    self.fail(f"_refresh_capability_inspector must not raise: {exc}")

        mock_set_cap.assert_called_with({}, [])


if __name__ == "__main__":
    unittest.main()
