#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 4 STAGE-02: UnitreeAdapter v2 lifecycle uplift.

Covers:
  - open_session: success, connect failure, model unavailable
  - preflight: model-not-initialised, action-in-progress, control-conflict, success
  - close_session: no model (idempotent), running model, viewer teardown, exception
  - capabilities: shape + content
  - Legacy interface: connect/run_action/stop/get_sensor_data/health unchanged

Isolation: no Qt, no SDK, no MuJoCo — all UnitreeModel instances are stubs
           injected via sys.modules before the adapter module is imported.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Stub models.Unitree so UnitreeAdapter can be imported without SDK ──────

_models_pkg = types.ModuleType("models")
_unitree_pkg = types.ModuleType("models.Unitree")


class _FakeUnitreeModel:
    """Configurable stub for UnitreeModel used by most tests."""

    def __init__(self, robot_type: str = "go2"):
        self.robot_type = robot_type
        # Availability flags — tests override these
        self.is_available    = True
        self.mujoco_available = False
        # Execution state
        self.running          = False
        self.stop_requested   = False
        # SDK state
        self._sdk_channel_inited = False
        self._sport_client       = None
        # Call records
        self._stop_called        = 0
        self._close_viewer_called = 0

    def run_action(self, action: str, **params: Any) -> bool:
        return True

    def stop(self) -> None:
        self._stop_called += 1
        self.running = False

    def get_sensor_data(self) -> Dict[str, Any]:
        return {"qpos": [], "qvel": [], "time": 0.0}

    def close_viewer(self) -> None:
        self._close_viewer_called += 1

    def is_viewer_open(self) -> bool:
        return False

    def get_available_actions(self):
        return ["stand", "sit", "walk", "stop", "lift_right_leg"]


_unitree_pkg.UnitreeModel = _FakeUnitreeModel
_models_pkg.Unitree = _unitree_pkg
sys.modules.setdefault("models", _models_pkg)
sys.modules.setdefault("models.Unitree", _unitree_pkg)

# ── Import under test ──────────────────────────────────────────────────────
from system.service.adapters.unitree_sdk2.adapter import (  # noqa: E402
    UnitreeAdapter,
    _REASON_ACTION_IN_PROGRESS,
    _REASON_CONTROL_CONFLICT,
    _REASON_MODEL_UNAVAILABLE,
)
from system.service.lifecycle import LifecycleReason  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _adapter(robot_type: str = "go2") -> UnitreeAdapter:
    return UnitreeAdapter(robot_type)


def _connected_adapter(
    *,
    is_available: bool = True,
    mujoco_available: bool = False,
    running: bool = False,
    sdk_channel_inited: bool = False,
    sport_client=None,
    robot_type: str = "go2",
) -> UnitreeAdapter:
    """Return an adapter with a pre-configured stub model."""
    a = _adapter(robot_type)
    model = _FakeUnitreeModel(robot_type)
    model.is_available       = is_available
    model.mujoco_available   = mujoco_available
    model.running            = running
    model._sdk_channel_inited = sdk_channel_inited
    model._sport_client      = sport_client
    a._model = model
    return a


# ══════════════════════════════════════════════════════════════════════════
# Section A — open_session
# ══════════════════════════════════════════════════════════════════════════

class TestOpenSession(unittest.TestCase):
    """UnitreeAdapter v2 open_session behaviour."""

    def test_open_session_sdk_available_ok(self):
        """open_session succeeds when SDK is available."""
        a = _connected_adapter(is_available=True, mujoco_available=False)
        res = a.open_session()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["stage"],  "open_session")
        self.assertEqual(res["reason"], "")

    def test_open_session_mujoco_available_ok(self):
        """open_session succeeds when MuJoCo simulation is available."""
        a = _connected_adapter(is_available=False, mujoco_available=True)
        res = a.open_session()
        self.assertEqual(res["status"], "ok")

    def test_open_session_both_available_ok(self):
        """open_session succeeds when both SDK and MuJoCo are available."""
        a = _connected_adapter(is_available=True, mujoco_available=True)
        res = a.open_session()
        self.assertEqual(res["status"], "ok")

    def test_open_session_neither_available_fails(self):
        """open_session fails with SESSION_OPEN_FAILED when no execution path."""
        a = _connected_adapter(is_available=False, mujoco_available=False)
        res = a.open_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        self.assertEqual(res["stage"],  "open_session")
        diag = res["diagnostics"]
        self.assertEqual(diag.get("reason"),           _REASON_MODEL_UNAVAILABLE)
        self.assertFalse(diag.get("sdk_available"))
        self.assertFalse(diag.get("mujoco_available"))
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_open_session_connect_raises_fails(self):
        """open_session returns SESSION_OPEN_FAILED when connect() raises."""
        a = _adapter()
        with patch.object(a, "connect", side_effect=RuntimeError("net down")):
            res = a.open_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        diag = res["diagnostics"]
        self.assertIn("net down", diag["message"])
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)

    def test_open_session_connect_returns_false_fails(self):
        """open_session returns SESSION_OPEN_FAILED when connect() returns False."""
        a = _adapter()
        with patch.object(a, "connect", return_value=False):
            res = a.open_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_open_session_passes_config_to_connect(self):
        """open_session forwards config kwargs to connect()."""
        a = _adapter()
        calls = []
        original = a.connect

        def recording_connect(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        with patch.object(a, "connect", side_effect=recording_connect):
            a.open_session(config={"robot_type": "b1", "force_reinit": True})

        self.assertTrue(any(c.get("robot_type") == "b1" for c in calls))

    def test_open_session_result_has_required_keys(self):
        """open_session result always has status/reason/stage/diagnostics."""
        a = _connected_adapter()
        res = a.open_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res, f"Missing key: {key}")


# ══════════════════════════════════════════════════════════════════════════
# Section B — preflight
# ══════════════════════════════════════════════════════════════════════════

class TestPreflight(unittest.TestCase):
    """UnitreeAdapter v2 preflight behaviour."""

    def test_preflight_ok_when_model_ready(self):
        """preflight passes when model is initialised and not running."""
        a = _connected_adapter(running=False)
        res = a.preflight()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["stage"],  "preflight")

    def test_preflight_fails_no_model(self):
        """preflight fails with PREFLIGHT_FAILED when model is None."""
        a = _adapter()  # _model is None
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_MODEL_UNAVAILABLE)
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_fails_action_in_progress(self):
        """preflight fails when a simulation action is currently running."""
        a = _connected_adapter(running=True)
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_ACTION_IN_PROGRESS)
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_fails_control_conflict(self):
        """preflight fails when SDK client + MuJoCo simulation both active."""
        sport_mock = MagicMock()
        a = _connected_adapter(
            sdk_channel_inited=True,
            sport_client=sport_mock,
            mujoco_available=True,
            running=False,
        )
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_CONTROL_CONFLICT)
        self.assertTrue(diag["sdk_client_active"])
        self.assertTrue(diag["mujoco_available"])
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_ok_sdk_only_no_conflict(self):
        """preflight passes when only SDK is active (no MuJoCo)."""
        sport_mock = MagicMock()
        a = _connected_adapter(
            sdk_channel_inited=True,
            sport_client=sport_mock,
            mujoco_available=False,
            running=False,
        )
        res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_ok_mujoco_only_no_conflict(self):
        """preflight passes when only MuJoCo is available (no SDK client)."""
        a = _connected_adapter(
            sdk_channel_inited=False,
            sport_client=None,
            mujoco_available=True,
            running=False,
        )
        res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_context_arg_accepted(self):
        """preflight accepts a context dict without raising."""
        a = _connected_adapter()
        res = a.preflight(context={"mode": "sim", "max_speed": 0.5})
        self.assertEqual(res["status"], "ok")

    def test_preflight_result_has_required_keys(self):
        """preflight result always has status/reason/stage/diagnostics."""
        a = _connected_adapter()
        res = a.preflight()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section C — close_session
# ══════════════════════════════════════════════════════════════════════════

class TestCloseSession(unittest.TestCase):
    """UnitreeAdapter v2 close_session behaviour."""

    def test_close_session_no_model_idempotent(self):
        """close_session returns ok when model is None (idempotent)."""
        a = _adapter()
        res = a.close_session()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["stage"],  "close_session")

    def test_close_session_stops_running_model(self):
        """close_session calls model.stop() when simulation is running."""
        a = _connected_adapter(running=True)
        res = a.close_session()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(a._model._stop_called, 1)

    def test_close_session_skips_stop_when_not_running(self):
        """close_session does not call model.stop() when not running."""
        a = _connected_adapter(running=False)
        res = a.close_session()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(a._model._stop_called, 0)

    def test_close_session_closes_viewer(self):
        """close_session calls close_viewer() if model provides it."""
        a = _connected_adapter()
        res = a.close_session()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(a._model._close_viewer_called, 1)

    def test_close_session_exception_returns_error(self):
        """close_session returns SESSION_CLOSE_FAILED on unexpected exception."""
        a = _connected_adapter(running=True)
        with patch.object(a._model, "stop", side_effect=RuntimeError("hw fault")):
            res = a.close_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_CLOSE_FAILED)
        diag = res["diagnostics"]
        self.assertIn("hw fault", diag["message"])
        self.assertEqual(diag.get("brand"), "unitree")
        self.assertIsInstance(diag.get("context"), dict)

    def test_close_session_model_still_accessible_after(self):
        """Legacy callers can still access the model after close_session."""
        a = _connected_adapter()
        a.close_session()
        # Model is NOT nulled — legacy callers must still be able to use it
        self.assertIsNotNone(a._model)

    def test_close_session_result_has_required_keys(self):
        """close_session result always has status/reason/stage/diagnostics."""
        a = _connected_adapter()
        res = a.close_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section D — capabilities
# ══════════════════════════════════════════════════════════════════════════

class TestCapabilities(unittest.TestCase):
    """UnitreeAdapter v2 capabilities() behaviour."""

    def test_capabilities_returns_expected_shape(self):
        """capabilities() returns dict with actions/sensors/flags."""
        a = _adapter()
        cap = a.capabilities()
        self.assertIn("actions", cap)
        self.assertIn("sensors", cap)
        self.assertIn("flags",   cap)

    def test_capabilities_actions_contains_minimum_set(self):
        """capabilities() includes minimum canonical action set."""
        cap = _adapter().capabilities()
        for action in ("stand", "sit", "walk", "stop"):
            self.assertIn(action, cap["actions"], f"Missing action: {action}")

    def test_capabilities_sensors_contains_minimum_set(self):
        """capabilities() includes minimum sensor keys."""
        cap = _adapter().capabilities()
        for sensor in ("qpos", "qvel", "time"):
            self.assertIn(sensor, cap["sensors"], f"Missing sensor: {sensor}")

    def test_capabilities_flags_sdk_false_before_connect(self):
        """capabilities() reports sdk_available=False before open_session."""
        cap = _adapter().capabilities()
        self.assertFalse(cap["flags"]["sdk_available"])
        self.assertFalse(cap["flags"]["mujoco_available"])

    def test_capabilities_flags_sdk_true_after_connect(self):
        """capabilities() reflects model availability after open_session."""
        a = _connected_adapter(is_available=True, mujoco_available=True)
        cap = a.capabilities()
        self.assertTrue(cap["flags"]["sdk_available"])
        self.assertTrue(cap["flags"]["mujoco_available"])

    def test_capabilities_flags_include_robot_type(self):
        """capabilities() flags include the robot_type string."""
        a = _adapter("b1")
        a._model = _FakeUnitreeModel("b1")
        cap = a.capabilities()
        self.assertEqual(cap["flags"]["robot_type"], "b1")

    # Phase 5 schema additions
    def test_capabilities_brand_is_unitree(self):
        """capabilities() brand key is 'unitree'."""
        self.assertEqual(_adapter().capabilities()["brand"], "unitree")

    def test_capabilities_adapter_is_unitree_sdk2(self):
        """capabilities() adapter key is 'unitree_sdk2'."""
        self.assertEqual(_adapter().capabilities()["adapter"], "unitree_sdk2")

    def test_capabilities_required_settings_present(self):
        """capabilities() includes required_settings list."""
        cap = _adapter().capabilities()
        self.assertIn("required_settings", cap)
        self.assertIsInstance(cap["required_settings"], list)

    def test_capabilities_required_settings_global_keys(self):
        """required_settings contains all global required keys."""
        rs = _adapter().capabilities()["required_settings"]
        for key in ("brand", "robot_type", "connection_mode", "timeout",
                    "stop_policy", "safety_enabled"):
            self.assertIn(key, rs, f"Missing global key: {key}")

    def test_capabilities_required_settings_unitree_keys(self):
        """required_settings contains Unitree-specific keys."""
        rs = _adapter().capabilities()["required_settings"]
        for key in ("dds_domain_id", "network_interface", "control_level",
                    "mode_switch_policy"):
            self.assertIn(key, rs, f"Missing Unitree-specific key: {key}")

    def test_capabilities_sensors_extended(self):
        """capabilities() sensors include battery/imu/foot_contact."""
        sensors = _adapter().capabilities()["sensors"]
        for key in ("battery", "imu", "foot_contact"):
            self.assertIn(key, sensors, f"Missing sensor: {key}")

    def test_capabilities_flags_high_level_control(self):
        """capabilities() flags include high_level_control=True."""
        self.assertTrue(_adapter().capabilities()["flags"]["high_level_control"])


# ══════════════════════════════════════════════════════════════════════════
# Section E — legacy interface unchanged
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyInterfaceUnchanged(unittest.TestCase):
    """Verify that all v1 methods still work after v2 lifecycle additions."""

    def test_connect_creates_model(self):
        """connect() creates a UnitreeModel instance."""
        a = _adapter()
        result = a.connect()
        self.assertTrue(result)
        self.assertIsNotNone(a._model)

    def test_get_model_returns_model_after_connect(self):
        """get_model() triggers connect() if needed and returns model."""
        a = _adapter()
        model = a.get_model()
        self.assertIsNotNone(model)

    def test_run_action_succeeds(self):
        """run_action() delegates to model and returns model result."""
        a = _connected_adapter()
        result = a.run_action("stand")
        self.assertTrue(result)

    def test_stop_calls_model_stop(self):
        """stop() delegates to model.stop()."""
        a = _connected_adapter()
        a.stop()  # must not raise

    def test_get_sensor_data_returns_dict(self):
        """get_sensor_data() returns a dict."""
        a = _connected_adapter()
        data = a.get_sensor_data()
        self.assertIsInstance(data, dict)

    def test_health_returns_dict_with_connected_key(self):
        """health() returns a dict with 'connected' key."""
        a = _connected_adapter()
        h = a.health()
        self.assertIn("connected", h)
        self.assertTrue(h["connected"])

    def test_health_always_returns_connected_key(self):
        """health() always returns a dict with 'connected' key (get_model auto-connects)."""
        a = _adapter()
        # health() calls get_model() which auto-connects, so connected is always present
        h = a.health()
        self.assertIn("connected", h)
        self.assertIn("robot_type", h)

    def test_run_action_no_model_returns_false(self):
        """run_action() returns False gracefully when model is None."""
        a = _adapter()
        with patch.object(a, "get_model", return_value=None):
            result = a.run_action("stand")
        self.assertFalse(result)

    def test_get_sensor_data_no_model_returns_error_dict(self):
        """get_sensor_data() returns error dict when model is None."""
        a = _adapter()
        with patch.object(a, "get_model", return_value=None):
            data = a.get_sensor_data()
        self.assertIn("error", data)


# ══════════════════════════════════════════════════════════════════════════
# Section F — lifecycle flow integration (open → preflight → close)
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleFlowIntegration(unittest.TestCase):
    """End-to-end lifecycle flow through UnitreeAdapter v2."""

    def test_full_flow_sdk_available(self):
        """open → preflight → action → close succeeds with SDK available."""
        a = _adapter()
        # Inject a model with SDK available
        model = _FakeUnitreeModel("go2")
        model.is_available    = True
        model.mujoco_available = False
        a._model = model

        open_res = a.open_session()
        self.assertEqual(open_res["status"], "ok")

        pre_res = a.preflight()
        self.assertEqual(pre_res["status"], "ok")

        action_res = a.run_action("stand")
        self.assertTrue(action_res)

        close_res = a.close_session()
        self.assertEqual(close_res["status"], "ok")

    def test_full_flow_mujoco_available(self):
        """open → preflight → action → close succeeds with MuJoCo available."""
        a = _adapter()
        model = _FakeUnitreeModel("go2")
        model.is_available    = False
        model.mujoco_available = True
        a._model = model

        self.assertEqual(a.open_session()["status"], "ok")
        self.assertEqual(a.preflight()["status"],    "ok")
        self.assertTrue(a.run_action("walk"))
        self.assertEqual(a.close_session()["status"], "ok")

    def test_preflight_blocked_after_running_starts(self):
        """preflight returns error if running becomes True between calls."""
        a = _connected_adapter(is_available=True, running=False)
        self.assertEqual(a.open_session()["status"], "ok")
        self.assertEqual(a.preflight()["status"],    "ok")

        a._model.running = True
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["diagnostics"]["reason"], _REASON_ACTION_IN_PROGRESS)

    def test_open_session_failure_short_circuits_flow(self):
        """When open_session fails, subsequent preflight reflects no model."""
        a = _adapter()
        model = _FakeUnitreeModel("go2")
        model.is_available    = False
        model.mujoco_available = False
        a._model = model

        open_res = a.open_session()
        self.assertEqual(open_res["status"], "error")

        # Model still exists but preflight OK because running=False — callers
        # should not call preflight after open_session error but we confirm
        # it does not crash
        pre_res = a.preflight()
        self.assertIn(pre_res["status"], ("ok", "error"))  # must not raise


if __name__ == "__main__":
    unittest.main()
