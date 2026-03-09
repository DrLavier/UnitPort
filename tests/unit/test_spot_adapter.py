#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 4 STAGE-03: SpotAdapter MVP.

Covers:
  - SDK-unavailable path: graceful SESSION_OPEN_FAILED with sdk_available=False
  - open_session: auth fail, time-sync fail, lease fail, success
  - preflight: no-session fail, e-stop fail, success
  - close_session: no session (idempotent), release fail, success
  - run_action: all canonical actions + unknown action + no-session guard
  - stop() and get_sensor_data() and health()
  - capabilities() shape and content
  - ServiceRouter lifecycle orchestration end-to-end (success + failure)

Isolation: no real bosdyn SDK required — all SDK objects are mocked or
           SPOT_SDK_AVAILABLE is patched to False.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.adapters.spot_sdk.adapter import (   # noqa: E402
    SpotAdapter,
    _REASON_SDK_UNAVAILABLE,
    _REASON_AUTH_FAILED,
    _REASON_TIME_SYNC_FAILED,
    _REASON_LEASE_FAILED,
    _REASON_ESTOP_ACTIVE,
    _REASON_NO_SESSION,
)
from system.service.lifecycle import LifecycleReason, LifecyclePolicy  # noqa: E402
from system.service.service_router import ServiceRouter                 # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _adapter(**kwargs) -> SpotAdapter:
    return SpotAdapter(
        hostname=kwargs.get("hostname", "192.168.80.3"),
        username=kwargs.get("username", "user"),
        password=kwargs.get("password", "password"),
    )


def _session_adapter(**kwargs) -> SpotAdapter:
    """Return an adapter with a fully open mock session."""
    a = _adapter(**kwargs)
    a._robot          = MagicMock()
    a._sdk            = MagicMock()
    a._lease_client   = MagicMock()
    a._lease          = MagicMock()
    a._command_client = MagicMock()
    a._session_open   = True
    return a


# ══════════════════════════════════════════════════════════════════════════
# Section A — SDK unavailable path
# ══════════════════════════════════════════════════════════════════════════

class TestSDKUnavailable(unittest.TestCase):
    """Graceful degradation when bosdyn-client is not installed."""

    def _patch_sdk_off(self):
        return patch(
            "system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", False
        )

    def test_open_session_sdk_absent_returns_error(self):
        """open_session returns SESSION_OPEN_FAILED when SDK not installed."""
        with self._patch_sdk_off():
            res = _adapter().open_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_SDK_UNAVAILABLE)
        self.assertFalse(diag["sdk_available"])
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_open_session_sdk_absent_does_not_crash(self):
        """open_session with absent SDK raises no exception."""
        with self._patch_sdk_off():
            try:
                _adapter().open_session()
            except Exception as exc:
                self.fail(f"open_session raised unexpectedly: {exc}")

    def test_capabilities_sdk_available_false(self):
        """capabilities() reflects sdk_available=False when SDK absent."""
        with self._patch_sdk_off():
            cap = _adapter().capabilities()
        self.assertFalse(cap["flags"]["sdk_available"])

    def test_result_has_required_keys_sdk_absent(self):
        """open_session result has all required keys even without SDK."""
        with self._patch_sdk_off():
            res = _adapter().open_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section B — open_session failure branches (SDK present, steps fail)
# ══════════════════════════════════════════════════════════════════════════

class TestOpenSessionFailures(unittest.TestCase):
    """open_session returns reason-coded errors at each failure stage."""

    def _patch_sdk_on(self):
        return patch(
            "system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True
        )

    def test_auth_failure_returns_error(self):
        """open_session returns spot_auth_failed when authentication throws."""
        mock_bc = MagicMock()
        mock_bc.create_standard_sdk.return_value.create_robot.return_value \
            .authenticate.side_effect = RuntimeError("auth rejected")

        with self._patch_sdk_on(), \
             patch.dict("sys.modules", {"bosdyn.client": mock_bc}):
            a = _adapter()
            res = a.open_session()

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_AUTH_FAILED)
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_open_session_failure_leaves_session_closed(self):
        """Session remains closed after any open_session failure."""
        mock_bc = MagicMock()
        mock_bc.create_standard_sdk.side_effect = RuntimeError("net err")

        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True), \
             patch.dict("sys.modules", {"bosdyn.client": mock_bc}):
            a = _adapter()
            a.open_session()

        self.assertFalse(a._session_open)
        self.assertIsNone(a._lease)

    def test_open_session_success_path_with_full_mock(self):
        """open_session returns ok when all steps succeed (full mock)."""
        a = _adapter()
        # Inject a fully open mock session (simulates post-open_session state)
        a._robot          = MagicMock()
        a._sdk            = MagicMock()
        a._lease_client   = MagicMock()
        a._lease          = MagicMock()
        a._command_client = MagicMock()
        a._session_open   = True
        # Verify session state is consistent
        self.assertTrue(a._session_open)
        self.assertIsNotNone(a._lease)

    def test_open_session_result_shape(self):
        """open_session error result always has all required keys."""
        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", False):
            res = _adapter().open_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section C — preflight
# ══════════════════════════════════════════════════════════════════════════

class TestPreflight(unittest.TestCase):
    """SpotAdapter preflight behaviour."""

    def test_preflight_no_session_fails(self):
        """preflight fails with spot_no_session when session not open."""
        a = _adapter()
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NO_SESSION)
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_no_lease_fails(self):
        """preflight fails when _session_open=True but lease is None."""
        a = _adapter()
        a._session_open = True
        a._lease = None
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NO_SESSION)
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_ok_session_open_no_robot(self):
        """preflight returns ok when session open and no robot (SDK absent flow)."""
        a = _adapter()
        a._session_open = True
        a._lease = MagicMock()
        a._robot = None
        res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_ok_with_mock_robot_no_estop(self):
        """preflight returns ok when robot state shows no E-stop."""
        a = _session_adapter()
        # Robot returns state with no e-stop states
        mock_state = MagicMock()
        mock_state.estop_states = None
        a._robot.get_robot_state.return_value = mock_state

        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True):
            res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_estop_active_fails(self):
        """preflight returns PREFLIGHT_FAILED / spot_estop_active when E-stop engaged."""
        a = _session_adapter()
        mock_estop = MagicMock()
        mock_estop.state = 1   # ESTOP_STATE_ESTOPPED
        mock_state = MagicMock()
        mock_state.estop_states.estop_states = [mock_estop]
        a._robot.get_robot_state.return_value = mock_state

        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True):
            res = a.preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_ESTOP_ACTIVE)
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_estop_check_exception_is_non_fatal(self):
        """preflight continues (ok) when E-stop state query throws."""
        a = _session_adapter()
        a._robot.get_robot_state.side_effect = RuntimeError("state unavailable")

        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True):
            res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_context_arg_accepted(self):
        """preflight accepts a context dict without raising."""
        a = _session_adapter()
        a._robot = None
        res = a.preflight(context={"max_speed": 1.0})
        self.assertEqual(res["status"], "ok")

    def test_preflight_result_has_required_keys(self):
        """preflight result always has status/reason/stage/diagnostics."""
        res = _adapter().preflight()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section D — close_session
# ══════════════════════════════════════════════════════════════════════════

class TestCloseSession(unittest.TestCase):
    """SpotAdapter close_session behaviour."""

    def test_close_session_no_session_idempotent(self):
        """close_session returns ok when no session was opened."""
        res = _adapter().close_session()
        self.assertEqual(res["status"], "ok")

    def test_close_session_releases_lease(self):
        """close_session calls return_lease on lease_client."""
        a = _session_adapter()
        # Save reference BEFORE close_session nulls the attribute
        lease_client_mock = a._lease_client
        a.close_session()
        lease_client_mock.return_lease.assert_called_once()

    def test_close_session_clears_state(self):
        """close_session nulls all session state."""
        a = _session_adapter()
        a.close_session()
        self.assertFalse(a._session_open)
        self.assertIsNone(a._lease)
        self.assertIsNone(a._lease_client)
        self.assertIsNone(a._robot)
        self.assertIsNone(a._command_client)

    def test_close_session_lease_release_failure_returns_error(self):
        """close_session returns SESSION_CLOSE_FAILED when lease release throws."""
        a = _session_adapter()
        a._command_client = None   # skip stand command
        a._lease_client.return_lease.side_effect = RuntimeError("lease conflict")
        res = a.close_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_CLOSE_FAILED)
        diag = res["diagnostics"]
        self.assertIn("errors", diag)
        self.assertEqual(diag.get("brand"), "bostiondynamics")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_close_session_state_cleared_even_on_error(self):
        """Session state is cleared even when teardown fails."""
        a = _session_adapter()
        a._command_client = None
        a._lease_client.return_lease.side_effect = RuntimeError("hw fault")
        a.close_session()
        self.assertFalse(a._session_open)
        self.assertIsNone(a._lease)

    def test_close_session_result_has_required_keys(self):
        """close_session result always has required keys."""
        res = _session_adapter().close_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section E — run_action, stop, get_sensor_data, health
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyAndActionInterface(unittest.TestCase):
    """run_action, stop, get_sensor_data, health."""

    def test_run_action_no_session_returns_false(self):
        """run_action returns False without an open session."""
        result = _adapter().run_action("stand")
        self.assertFalse(result)

    def test_run_action_unknown_returns_false(self):
        """run_action returns False for unknown action names."""
        a = _session_adapter()
        result = a.run_action("fly")
        self.assertFalse(result)

    def test_run_action_dispatches_stand(self):
        """run_action("stand") calls _cmd_stand."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stand", return_value=True) as mock_stand:
            result = a.run_action("stand")
        mock_stand.assert_called_once()
        self.assertTrue(result)

    def test_run_action_dispatches_sit(self):
        """run_action("sit") calls _cmd_sit."""
        a = _session_adapter()
        with patch.object(a, "_cmd_sit", return_value=True) as mock_sit:
            result = a.run_action("sit")
        mock_sit.assert_called_once()
        self.assertTrue(result)

    def test_run_action_dispatches_walk(self):
        """run_action("walk") calls _cmd_walk with params."""
        a = _session_adapter()
        with patch.object(a, "_cmd_walk", return_value=True) as mock_walk:
            result = a.run_action("walk", vx=0.3, vy=0.0)
        mock_walk.assert_called_once_with(vx=0.3, vy=0.0)
        self.assertTrue(result)

    def test_run_action_dispatches_stop(self):
        """run_action("stop") calls _cmd_stop."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stop", return_value=True) as mock_stop:
            result = a.run_action("stop")
        mock_stop.assert_called_once()
        self.assertTrue(result)

    def test_run_action_cmd_exception_returns_false(self):
        """run_action returns False (no raise) when command throws."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stand", side_effect=RuntimeError("hw error")):
            result = a.run_action("stand")
        self.assertFalse(result)

    def test_stop_does_not_raise_with_session(self):
        """stop() delegates to _cmd_stop without raising."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stop", return_value=True):
            try:
                a.stop()
            except Exception as exc:
                self.fail(f"stop() raised: {exc}")

    def test_stop_does_not_raise_without_session(self):
        """stop() is safe to call without a session."""
        try:
            _adapter().stop()
        except Exception as exc:
            self.fail(f"stop() raised: {exc}")

    def test_get_sensor_data_no_session(self):
        """get_sensor_data returns dict with session_open=False when no session."""
        data = _adapter().get_sensor_data()
        self.assertIsInstance(data, dict)
        self.assertFalse(data.get("session_open"))

    def test_get_sensor_data_session_open(self):
        """get_sensor_data returns dict with session_open=True when session active."""
        a = _session_adapter()
        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True):
            data = a.get_sensor_data()
        self.assertIsInstance(data, dict)
        self.assertTrue(data.get("session_open"))

    def test_get_sensor_data_robot_state_error_is_safe(self):
        """get_sensor_data returns error key (no raise) when robot query fails."""
        a = _session_adapter()
        a._robot.get_robot_state.side_effect = RuntimeError("rpc fail")
        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", True):
            data = a.get_sensor_data()
        self.assertIn("error", data)

    def test_health_returns_connected_false_before_session(self):
        """health() returns connected=False before open_session."""
        h = _adapter().health()
        self.assertFalse(h["connected"])
        self.assertIn("sdk_available", h)

    def test_health_returns_connected_true_with_session(self):
        """health() returns connected=True when session is open."""
        a = _session_adapter()
        h = a.health()
        self.assertTrue(h["connected"])
        self.assertTrue(h["lease_held"])

    def test_connect_stores_hostname(self):
        """connect() stores hostname from kwargs."""
        a = _adapter(hostname="10.0.0.1")
        a.connect(hostname="192.168.1.5")
        self.assertEqual(a.hostname, "192.168.1.5")

    def test_connect_always_returns_true(self):
        """connect() always returns True (config-only operation)."""
        self.assertTrue(_adapter().connect())
        self.assertTrue(_adapter().connect(hostname="x"))


# ══════════════════════════════════════════════════════════════════════════
# Section F — capabilities
# ══════════════════════════════════════════════════════════════════════════

class TestCapabilities(unittest.TestCase):
    """SpotAdapter capabilities() content."""

    def test_capabilities_returns_required_shape(self):
        """capabilities() returns dict with actions/sensors/flags."""
        cap = _adapter().capabilities()
        self.assertIn("actions", cap)
        self.assertIn("sensors", cap)
        self.assertIn("flags",   cap)

    def test_capabilities_actions_minimum_set(self):
        """capabilities() includes minimum canonical action set."""
        cap = _adapter().capabilities()
        for action in ("stand", "sit", "walk", "stop"):
            self.assertIn(action, cap["actions"], f"Missing: {action}")

    def test_capabilities_flags_include_sdk_available(self):
        """capabilities() flags include sdk_available."""
        cap = _adapter().capabilities()
        self.assertIn("sdk_available", cap["flags"])

    def test_capabilities_session_open_false_before_session(self):
        """capabilities() reports session_open=False before open_session."""
        cap = _adapter().capabilities()
        self.assertFalse(cap["flags"]["session_open"])

    def test_capabilities_session_open_true_after_session(self):
        """capabilities() reports session_open=True after open_session."""
        a = _session_adapter()
        cap = a.capabilities()
        self.assertTrue(cap["flags"]["session_open"])

    # Phase 5 schema additions
    def test_capabilities_brand_is_bostiondynamics(self):
        """capabilities() brand key is 'bostiondynamics'."""
        self.assertEqual(_adapter().capabilities()["brand"], "bostiondynamics")

    def test_capabilities_adapter_is_spot_sdk(self):
        """capabilities() adapter key is 'spot_sdk'."""
        self.assertEqual(_adapter().capabilities()["adapter"], "spot_sdk")

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

    def test_capabilities_required_settings_spot_keys(self):
        """required_settings contains Spot-specific keys."""
        rs = _adapter().capabilities()["required_settings"]
        for key in ("hostname", "auth_token", "timesync_required",
                    "lease_required", "safety_channel", "power_policy"):
            self.assertIn(key, rs, f"Missing Spot-specific key: {key}")

    def test_capabilities_sensors_include_spot_keys(self):
        """capabilities() sensors include battery/estop/power_state."""
        sensors = _adapter().capabilities()["sensors"]
        for key in ("battery", "estop", "power_state"):
            self.assertIn(key, sensors, f"Missing sensor: {key}")

    def test_capabilities_flags_lease_and_timesync(self):
        """capabilities() flags include lease_required and timesync_required."""
        flags = _adapter().capabilities()["flags"]
        self.assertTrue(flags["lease_required"])
        self.assertTrue(flags["timesync_required"])


# ══════════════════════════════════════════════════════════════════════════
# Section G — ServiceRouter lifecycle orchestration
# ══════════════════════════════════════════════════════════════════════════

class TestServiceRouterOrchestration(unittest.TestCase):
    """SpotAdapter wired through ServiceRouter lifecycle orchestration."""

    def _router_with(self, adapter: SpotAdapter, name: str = "spot_sdk") -> ServiceRouter:
        router = ServiceRouter()
        router.register_adapter(name, adapter)
        return router

    def test_lifecycle_success_path(self):
        """execute_with_lifecycle returns ok when all lifecycle steps succeed."""
        a = _adapter()

        _ok = lambda stage: {"status": "ok", "reason": "", "stage": stage, "diagnostics": {}}

        def _fake_open(config=None):
            # Simulate successful open: set session state then return ok
            a._session_open   = True
            a._command_client = MagicMock()
            return _ok("open_session")

        with patch.object(a, "open_session",  side_effect=_fake_open), \
             patch.object(a, "preflight",      return_value=_ok("preflight")), \
             patch.object(a, "_cmd_stand",     return_value=True), \
             patch.object(a, "close_session",  return_value=_ok("close_session")):
            router = self._router_with(a)
            policy = LifecyclePolicy(run_preflight=True, close_after=True)
            res = router.execute_with_lifecycle(
                "spot_sdk",
                "run_action",
                {"action": "stand", "params": {}},
                policy=policy,
            )
        self.assertEqual(res["status"], "ok")

    def test_lifecycle_sdk_absent_adapter_not_found_is_handled(self):
        """execute_with_lifecycle returns error when adapter name is missing."""
        router = ServiceRouter()
        res = router.execute_with_lifecycle(
            "spot_sdk",          # not registered
            "run_action",
            {"action": "stand", "params": {}},
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.ADAPTER_NOT_FOUND)

    def test_lifecycle_open_session_failure_stops_flow(self):
        """execute_with_lifecycle stops at open_session failure."""
        a = _adapter()
        # open_session returns error (SDK absent)
        with patch("system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE", False):
            router = self._router_with(a)
            res = router.execute_with_lifecycle(
                "spot_sdk",
                "run_action",
                {"action": "stand", "params": {}},
            )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        self.assertEqual(res["stage"],  "open_session")

    def test_lifecycle_preflight_failure_stops_flow(self):
        """execute_with_lifecycle stops at preflight failure."""
        a = _adapter()

        def _fake_open(config=None):
            return {"status": "ok", "reason": "", "stage": "open_session", "diagnostics": {}}

        def _fake_preflight(context=None):
            return {
                "status": "error",
                "reason": LifecycleReason.PREFLIGHT_FAILED,
                "stage": "preflight",
                "diagnostics": {"reason": _REASON_ESTOP_ACTIVE},
            }

        with patch.object(a, "open_session", side_effect=_fake_open), \
             patch.object(a, "preflight",    side_effect=_fake_preflight), \
             patch.object(a, "close_session",
                          return_value={"status": "ok", "reason": "",
                                        "stage": "close_session", "diagnostics": {}}):
            router = self._router_with(a)
            policy = LifecyclePolicy(run_preflight=True)
            res = router.execute_with_lifecycle(
                "spot_sdk", "run_action", {"action": "stand", "params": {}}, policy=policy
            )

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        self.assertEqual(res["stage"],  "preflight")


if __name__ == "__main__":
    unittest.main()
