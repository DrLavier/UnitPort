#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 4 STAGE-04: CyberDogAdapter MVP.

Covers:
  - ROS2-unavailable path: graceful SESSION_OPEN_FAILED with ros2_available=False
  - open_session: node init fail, endpoint check, publisher creation, success
  - preflight: no-session fail, namespace mismatch, success
  - close_session: no session (idempotent), vel reset, node teardown, error
  - run_action: all canonical actions + unknown + no-session guard
  - stop(), get_sensor_data(), health()
  - capabilities() shape and content
  - ServiceRouter lifecycle orchestration (success + failure branches)

Isolation: no real rclpy required — ROS2_AVAILABLE is patched to False/True,
           and all ROS2 objects are replaced with MagicMock.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.adapters.cyberdog_sdk.adapter import (   # noqa: E402
    CyberDogAdapter,
    _REASON_ROS2_UNAVAILABLE,
    _REASON_NODE_INIT_FAILED,
    _REASON_NO_SESSION,
    _REASON_NAMESPACE_MISMATCH,
)
from system.service.lifecycle import LifecycleReason, LifecyclePolicy  # noqa: E402
from system.service.service_router import ServiceRouter                 # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _adapter(**kwargs) -> CyberDogAdapter:
    return CyberDogAdapter(
        namespace=kwargs.get("namespace", "cyberdog"),
        node_name=kwargs.get("node_name",  "unitport_cyberdog"),
    )


def _session_adapter(**kwargs) -> CyberDogAdapter:
    """Return an adapter with a fully open mock session."""
    import time
    a = _adapter(**kwargs)
    a._node               = MagicMock()
    a._vel_publisher      = MagicMock()
    a._motion_publisher   = MagicMock()
    a._session_open       = True
    a._session_start_time = time.monotonic()
    # Mock node.get_namespace() to return the configured namespace
    a._node.get_namespace.return_value = f"/{a.namespace}"
    return a


# ══════════════════════════════════════════════════════════════════════════
# Section A — ROS2 unavailable path
# ══════════════════════════════════════════════════════════════════════════

class TestROS2Unavailable(unittest.TestCase):
    """Graceful degradation when rclpy is not installed."""

    def _patch_ros2_off(self):
        return patch(
            "system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", False
        )

    def test_open_session_ros2_absent_returns_error(self):
        """open_session returns SESSION_OPEN_FAILED when ROS2 not available."""
        with self._patch_ros2_off():
            res = _adapter().open_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_ROS2_UNAVAILABLE)
        self.assertFalse(diag["ros2_available"])
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_open_session_ros2_absent_does_not_crash(self):
        """open_session with absent ROS2 raises no exception."""
        with self._patch_ros2_off():
            try:
                _adapter().open_session()
            except Exception as exc:
                self.fail(f"open_session raised unexpectedly: {exc}")

    def test_capabilities_ros2_available_false(self):
        """capabilities() reflects ros2_available=False when ROS2 absent."""
        with self._patch_ros2_off():
            cap = _adapter().capabilities()
        self.assertFalse(cap["flags"]["ros2_available"])

    def test_result_has_required_keys_ros2_absent(self):
        """open_session result has all required keys even without ROS2."""
        with self._patch_ros2_off():
            res = _adapter().open_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)

    def test_run_action_ros2_absent_returns_false(self):
        """_cmd_stand/_cmd_walk return False when ROS2 unavailable."""
        a = _session_adapter()
        with self._patch_ros2_off():
            self.assertFalse(a._cmd_stand())
            self.assertFalse(a._cmd_walk())
            self.assertFalse(a._cmd_stop())
            self.assertFalse(a._cmd_sit())


# ══════════════════════════════════════════════════════════════════════════
# Section B — open_session failure branches
# ══════════════════════════════════════════════════════════════════════════

class TestOpenSessionFailures(unittest.TestCase):
    """open_session returns reason-coded errors at each failure stage."""

    def _patch_ros2_on(self):
        return patch(
            "system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True
        )

    def test_node_init_failure_returns_error(self):
        """open_session returns cyberdog_node_init_failed when create_node throws."""
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True
        mock_rclpy.create_node.side_effect = RuntimeError("ROS2 runtime error")

        with self._patch_ros2_on(), \
             patch.dict("sys.modules", {"rclpy": mock_rclpy, "rclpy.node": MagicMock()}):
            a = _adapter()
            res = a.open_session()

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NODE_INIT_FAILED)
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_node_init_failure_leaves_session_closed(self):
        """Session remains closed after node init failure."""
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True
        mock_rclpy.create_node.side_effect = RuntimeError("crash")

        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True), \
             patch.dict("sys.modules", {"rclpy": mock_rclpy, "rclpy.node": MagicMock()}):
            a = _adapter()
            a.open_session()

        self.assertFalse(a._session_open)
        self.assertIsNone(a._node)

    def test_open_session_success_via_injected_state(self):
        """Injecting session state represents a successful open_session."""
        a = _session_adapter()
        self.assertTrue(a._session_open)
        self.assertIsNotNone(a._node)

    def test_open_session_result_shape(self):
        """open_session error result has all required keys."""
        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", False):
            res = _adapter().open_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)

    def test_open_session_success_with_mock_rclpy(self):
        """open_session returns ok when rclpy mock succeeds."""
        mock_rclpy = MagicMock()
        mock_rclpy.ok.return_value = True
        mock_node   = MagicMock()
        mock_node.get_namespace.return_value = "/cyberdog"
        mock_node.count_publishers.return_value = 0
        mock_rclpy.create_node.return_value = mock_node

        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True), \
             patch.dict("sys.modules", {
                 "rclpy": mock_rclpy,
                 "rclpy.node": MagicMock(),
                 "geometry_msgs": MagicMock(),
                 "geometry_msgs.msg": MagicMock(),
                 "std_msgs": MagicMock(),
                 "std_msgs.msg": MagicMock(),
             }):
            a = _adapter()
            res = a.open_session(config={"skip_endpoint_check": True})

        self.assertEqual(res["status"], "ok")
        self.assertTrue(a._session_open)


# ══════════════════════════════════════════════════════════════════════════
# Section C — preflight
# ══════════════════════════════════════════════════════════════════════════

class TestPreflight(unittest.TestCase):
    """CyberDogAdapter preflight behaviour."""

    def test_preflight_no_session_fails(self):
        """preflight returns PREFLIGHT_FAILED / cyberdog_no_session."""
        res = _adapter().preflight()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NO_SESSION)
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_no_node_fails(self):
        """preflight fails when _session_open=True but node is None."""
        a = _adapter()
        a._session_open = True
        a._node = None
        res = a.preflight()
        self.assertEqual(res["status"], "error")
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NO_SESSION)
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_ok_with_matching_namespace(self):
        """preflight returns ok when node namespace matches config."""
        a = _session_adapter(namespace="cyberdog")
        a._node.get_namespace.return_value = "/cyberdog"
        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True):
            res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_namespace_mismatch_fails(self):
        """preflight fails when node namespace differs from configured namespace."""
        a = _session_adapter(namespace="cyberdog")
        a._node.get_namespace.return_value = "/other_robot"

        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True):
            res = a.preflight()

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        diag = res["diagnostics"]
        self.assertEqual(diag["reason"], _REASON_NAMESPACE_MISMATCH)
        self.assertIn("node_ns",   diag)
        self.assertIn("config_ns", diag)
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_preflight_ok_when_ros2_off_and_session_injected(self):
        """preflight returns ok when ROS2 disabled and session state injected."""
        a = _session_adapter()
        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", False):
            res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_namespace_check_exception_is_non_fatal(self):
        """preflight continues (ok) when get_namespace() throws."""
        a = _session_adapter()
        a._node.get_namespace.side_effect = RuntimeError("node error")
        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", True):
            res = a.preflight()
        self.assertEqual(res["status"], "ok")

    def test_preflight_context_arg_accepted(self):
        """preflight accepts a context dict without raising."""
        a = _session_adapter()
        a._node.get_namespace.return_value = "/cyberdog"
        res = a.preflight(context={"mode": "normal"})
        self.assertEqual(res["status"], "ok")

    def test_preflight_result_has_required_keys(self):
        """preflight result has status/reason/stage/diagnostics."""
        res = _adapter().preflight()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, res)


# ══════════════════════════════════════════════════════════════════════════
# Section D — close_session
# ══════════════════════════════════════════════════════════════════════════

class TestCloseSession(unittest.TestCase):
    """CyberDogAdapter close_session behaviour."""

    def test_close_session_no_session_idempotent(self):
        """close_session returns ok when no session was opened."""
        res = _adapter().close_session()
        self.assertEqual(res["status"], "ok")

    def test_close_session_destroys_node(self):
        """close_session calls node.destroy_node()."""
        a = _session_adapter()
        node_mock = a._node
        a.close_session()
        node_mock.destroy_node.assert_called_once()

    def test_close_session_clears_state(self):
        """close_session nulls all session state."""
        a = _session_adapter()
        a.close_session()
        self.assertFalse(a._session_open)
        self.assertIsNone(a._node)
        self.assertIsNone(a._vel_publisher)
        self.assertIsNone(a._motion_publisher)

    def test_close_session_node_destroy_failure_returns_error(self):
        """close_session returns SESSION_CLOSE_FAILED when node.destroy_node throws."""
        a = _session_adapter()
        a._vel_publisher = None   # skip vel reset
        a._node.destroy_node.side_effect = RuntimeError("node locked")
        res = a.close_session()
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_CLOSE_FAILED)
        diag = res["diagnostics"]
        self.assertIn("errors", diag)
        self.assertEqual(diag.get("brand"), "xiaomi")
        self.assertIsInstance(diag.get("context"), dict)
        self.assertIsInstance(diag.get("message"), str)

    def test_close_session_state_cleared_even_on_error(self):
        """Session state is cleared even when teardown fails."""
        a = _session_adapter()
        a._vel_publisher = None
        a._node.destroy_node.side_effect = RuntimeError("crash")
        a.close_session()
        self.assertFalse(a._session_open)
        self.assertIsNone(a._node)

    def test_close_session_result_has_required_keys(self):
        """close_session result has all required keys."""
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
        self.assertFalse(_adapter().run_action("stand"))

    def test_run_action_unknown_returns_false(self):
        """run_action returns False for unknown action names."""
        a = _session_adapter()
        self.assertFalse(a.run_action("backflip"))

    def test_run_action_dispatches_stand(self):
        """run_action("stand") calls _cmd_stand."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stand", return_value=True) as m:
            result = a.run_action("stand")
        m.assert_called_once()
        self.assertTrue(result)

    def test_run_action_dispatches_sit(self):
        """run_action("sit") calls _cmd_sit."""
        a = _session_adapter()
        with patch.object(a, "_cmd_sit", return_value=True) as m:
            result = a.run_action("sit")
        m.assert_called_once()
        self.assertTrue(result)

    def test_run_action_dispatches_walk_with_params(self):
        """run_action("walk") passes vx/vy kwargs to _cmd_walk."""
        a = _session_adapter()
        with patch.object(a, "_cmd_walk", return_value=True) as m:
            result = a.run_action("walk", vx=0.4, vy=0.1)
        m.assert_called_once_with(vx=0.4, vy=0.1)
        self.assertTrue(result)

    def test_run_action_dispatches_stop(self):
        """run_action("stop") calls _cmd_stop."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stop", return_value=True) as m:
            result = a.run_action("stop")
        m.assert_called_once()
        self.assertTrue(result)

    def test_run_action_exception_returns_false(self):
        """run_action returns False (no raise) when command throws."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stand", side_effect=RuntimeError("hw err")):
            self.assertFalse(a.run_action("stand"))

    def test_stop_safe_with_session(self):
        """stop() delegates to _cmd_stop without raising."""
        a = _session_adapter()
        with patch.object(a, "_cmd_stop", return_value=True):
            try:
                a.stop()
            except Exception as exc:
                self.fail(f"stop() raised: {exc}")

    def test_stop_safe_without_session(self):
        """stop() is safe to call without a session."""
        try:
            _adapter().stop()
        except Exception as exc:
            self.fail(f"stop() raised: {exc}")

    def test_get_sensor_data_returns_dict(self):
        """get_sensor_data returns a dict with required keys."""
        data = _adapter().get_sensor_data()
        self.assertIsInstance(data, dict)
        self.assertIn("ros2_available", data)
        self.assertIn("session_open",   data)

    def test_health_connected_false_before_session(self):
        """health() returns connected=False before open_session."""
        h = _adapter().health()
        self.assertFalse(h["connected"])
        self.assertIn("ros2_available", h)

    def test_health_connected_true_with_session(self):
        """health() returns connected=True when session is open."""
        h = _session_adapter().health()
        self.assertTrue(h["connected"])
        self.assertTrue(h["node_alive"])

    def test_connect_stores_namespace(self):
        """connect() stores namespace from kwargs."""
        a = _adapter(namespace="robot1")
        a.connect(namespace="robot2")
        self.assertEqual(a.namespace, "robot2")

    def test_connect_always_returns_true(self):
        """connect() always returns True."""
        self.assertTrue(_adapter().connect())
        self.assertTrue(_adapter().connect(namespace="x", node_name="y"))


# ══════════════════════════════════════════════════════════════════════════
# Section F — capabilities
# ══════════════════════════════════════════════════════════════════════════

class TestCapabilities(unittest.TestCase):
    """CyberDogAdapter capabilities() content."""

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

    def test_capabilities_flags_include_ros2_available(self):
        """capabilities() flags include ros2_available."""
        cap = _adapter().capabilities()
        self.assertIn("ros2_available", cap["flags"])

    def test_capabilities_session_open_false_before_session(self):
        """capabilities() reports session_open=False before open_session."""
        self.assertFalse(_adapter().capabilities()["flags"]["session_open"])

    def test_capabilities_session_open_true_with_session(self):
        """capabilities() reports session_open=True when session active."""
        self.assertTrue(_session_adapter().capabilities()["flags"]["session_open"])

    def test_capabilities_includes_namespace(self):
        """capabilities() flags include the configured namespace."""
        cap = _adapter(namespace="dog1").capabilities()
        self.assertEqual(cap["flags"]["namespace"], "dog1")

    # Phase 5 schema additions
    def test_capabilities_brand_is_xiaomi(self):
        """capabilities() brand key is 'xiaomi'."""
        self.assertEqual(_adapter().capabilities()["brand"], "xiaomi")

    def test_capabilities_adapter_is_cyberdog_sdk(self):
        """capabilities() adapter key is 'cyberdog_sdk'."""
        self.assertEqual(_adapter().capabilities()["adapter"], "cyberdog_sdk")

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

    def test_capabilities_required_settings_cyberdog_keys(self):
        """required_settings contains CyberDog-specific keys."""
        rs = _adapter().capabilities()["required_settings"]
        for key in ("ros_domain_id", "rmw_impl", "namespace",
                    "action_endpoints", "timestamp_policy"):
            self.assertIn(key, rs, f"Missing CyberDog-specific key: {key}")

    def test_capabilities_sensors_include_cyberdog_keys(self):
        """capabilities() sensors include battery/imu/odometry."""
        sensors = _adapter().capabilities()["sensors"]
        for key in ("battery", "imu", "odometry"):
            self.assertIn(key, sensors, f"Missing sensor: {key}")

    def test_capabilities_flags_ros2_required(self):
        """capabilities() flags include ros2_required=True."""
        self.assertTrue(_adapter().capabilities()["flags"]["ros2_required"])

    def test_capabilities_flags_namespace_isolated(self):
        """capabilities() flags include namespace_isolated=True."""
        self.assertTrue(_adapter().capabilities()["flags"]["namespace_isolated"])


# ══════════════════════════════════════════════════════════════════════════
# Section G — ServiceRouter lifecycle orchestration
# ══════════════════════════════════════════════════════════════════════════

class TestServiceRouterOrchestration(unittest.TestCase):
    """CyberDogAdapter wired through ServiceRouter lifecycle orchestration."""

    _OK = staticmethod(
        lambda stage: {"status": "ok", "reason": "", "stage": stage, "diagnostics": {}}
    )

    def _router_with(self, adapter: CyberDogAdapter, name: str = "cyberdog_sdk") -> ServiceRouter:
        router = ServiceRouter()
        router.register_adapter(name, adapter)
        return router

    def test_lifecycle_success_path(self):
        """execute_with_lifecycle returns ok when all steps succeed."""
        a = _adapter()

        def _fake_open(config=None):
            a._session_open   = True
            a._node           = MagicMock()
            a._vel_publisher  = MagicMock()
            a._motion_publisher = MagicMock()
            return self._OK("open_session")

        with patch.object(a, "open_session",  side_effect=_fake_open), \
             patch.object(a, "preflight",      return_value=self._OK("preflight")), \
             patch.object(a, "_cmd_stand",     return_value=True), \
             patch.object(a, "close_session",  return_value=self._OK("close_session")):
            router = self._router_with(a)
            policy = LifecyclePolicy(run_preflight=True, close_after=True)
            res = router.execute_with_lifecycle(
                "cyberdog_sdk",
                "run_action",
                {"action": "stand", "params": {}},
                policy=policy,
            )
        self.assertEqual(res["status"], "ok")

    def test_lifecycle_adapter_not_registered_returns_error(self):
        """execute_with_lifecycle returns ADAPTER_NOT_FOUND for unregistered name."""
        router = ServiceRouter()
        res = router.execute_with_lifecycle(
            "cyberdog_sdk",
            "run_action",
            {"action": "stand", "params": {}},
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.ADAPTER_NOT_FOUND)

    def test_lifecycle_ros2_absent_open_session_fails(self):
        """execute_with_lifecycle stops at open_session when ROS2 absent."""
        a = _adapter()
        with patch("system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE", False):
            router = self._router_with(a)
            res = router.execute_with_lifecycle(
                "cyberdog_sdk",
                "run_action",
                {"action": "stand", "params": {}},
            )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.SESSION_OPEN_FAILED)
        self.assertEqual(res["stage"],  "open_session")

    def test_lifecycle_preflight_failure_stops_flow(self):
        """execute_with_lifecycle stops at preflight on PREFLIGHT_FAILED."""
        a = _adapter()

        _err = {
            "status": "error",
            "reason": LifecycleReason.PREFLIGHT_FAILED,
            "stage": "preflight",
            "diagnostics": {"reason": _REASON_NO_SESSION},
        }

        with patch.object(a, "open_session",  return_value=self._OK("open_session")), \
             patch.object(a, "preflight",      return_value=_err), \
             patch.object(a, "close_session",  return_value=self._OK("close_session")):
            router = self._router_with(a)
            policy = LifecyclePolicy(run_preflight=True)
            res = router.execute_with_lifecycle(
                "cyberdog_sdk", "run_action",
                {"action": "stand", "params": {}}, policy=policy,
            )

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)
        self.assertEqual(res["stage"],  "preflight")

    def test_lifecycle_walk_action_dispatched(self):
        """execute_with_lifecycle dispatches walk with velocity params."""
        a = _adapter()

        def _fake_open(config=None):
            a._session_open = True
            a._node = MagicMock()
            return self._OK("open_session")

        walk_calls = []

        def _fake_walk(**kw):
            walk_calls.append(kw)
            return True

        with patch.object(a, "open_session", side_effect=_fake_open), \
             patch.object(a, "_cmd_walk",    side_effect=_fake_walk), \
             patch.object(a, "close_session", return_value=self._OK("close_session")):
            router = self._router_with(a)
            router.execute_with_lifecycle(
                "cyberdog_sdk",
                "run_action",
                {"action": "walk", "params": {"vx": 0.5}},
            )

        self.assertTrue(any(c.get("vx") == 0.5 for c in walk_calls))


if __name__ == "__main__":
    unittest.main()
