#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 3 STAGE-03: ServiceRouter session-aware routing.

Covers:
  - RouteOp constants
  - execute_with_lifecycle: happy path (run_action / get_sensor_data / stop)
  - execute_with_lifecycle: adapter-not-found
  - execute_with_lifecycle: open_session failure
  - execute_with_lifecycle: preflight (enabled/disabled, pass/fail)
  - execute_with_lifecycle: close_session (enabled/disabled, pass/fail)
  - execute_with_lifecycle: execute dispatch failure
  - execute_with_lifecycle: result schema on all paths
  - Legacy passthrough methods still work unchanged

Isolation: no Qt, no SDK.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import LifecyclePolicy, LifecycleReason, LifecycleResult  # noqa: E402
from system.service.service_router import RouteOp, ServiceRouter                         # noqa: E402
from system.service.service_registry import ServiceRegistry                              # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter                             # noqa: E402


# ── Mock adapter ───────────────────────────────────────────────────────────

class _MockAdapter(BaseAdapter):
    """Full lifecycle mock adapter for router tests."""

    def __init__(self):
        # Configurable return values for lifecycle methods
        self.open_session_status   = "ok"
        self.preflight_status      = "ok"
        self.close_session_status  = "ok"
        self.run_action_return     = "action_done"
        self.run_action_raises     = None   # set to an exception to simulate failure

        # Call recording
        self.calls: list = []

    # ── Legacy abstract methods ────────────────────────────────────────────

    def connect(self, **kwargs: Any) -> bool:
        self.calls.append(("connect", kwargs))
        return True

    def run_action(self, action: str, **params: Any) -> Any:
        self.calls.append(("run_action", action, params))
        if self.run_action_raises:
            raise self.run_action_raises
        return self.run_action_return

    def stop(self) -> None:
        self.calls.append(("stop",))

    def get_sensor_data(self) -> Dict[str, Any]:
        self.calls.append(("get_sensor_data",))
        return {"x": 42}

    def health(self) -> Dict[str, Any]:
        return {"connected": True}

    # ── Lifecycle overrides ────────────────────────────────────────────────

    def open_session(self, config=None) -> Dict[str, Any]:
        self.calls.append(("open_session", config))
        if self.open_session_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session",
            LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock open_session failure"},
        ).to_dict()

    def preflight(self, context=None) -> Dict[str, Any]:
        self.calls.append(("preflight", context))
        if self.preflight_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight",
            LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock preflight failure"},
        ).to_dict()

    def close_session(self) -> Dict[str, Any]:
        self.calls.append(("close_session",))
        if self.close_session_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session",
            LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock close_session failure"},
        ).to_dict()


# ── Helper: router with a registered mock adapter ─────────────────────────

def _router_with(adapter: _MockAdapter, name: str = "test") -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


# ══════════════════════════════════════════════════════════════════════════
# RouteOp constants
# ══════════════════════════════════════════════════════════════════════════

class TestRouteOpConstants(unittest.TestCase):

    def test_run_action_constant(self):
        self.assertEqual(RouteOp.RUN_ACTION, "run_action")

    def test_get_sensor_data_constant(self):
        self.assertEqual(RouteOp.GET_SENSOR_DATA, "get_sensor_data")

    def test_stop_constant(self):
        self.assertEqual(RouteOp.STOP, "stop")

    def test_all_constants_are_distinct(self):
        vals = [RouteOp.RUN_ACTION, RouteOp.GET_SENSOR_DATA, RouteOp.STOP]
        self.assertEqual(len(vals), len(set(vals)))


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — result schema
# ══════════════════════════════════════════════════════════════════════════

class TestRouteResultSchema(unittest.TestCase):
    """Every result dict from execute_with_lifecycle has the canonical keys."""

    # Phase 4 STAGE-06: adapter_name added as required top-level key
    _REQUIRED_KEYS = frozenset({"status", "reason", "stage", "payload", "diagnostics", "adapter_name"})

    def _assert_schema(self, result: Dict[str, Any]):
        missing = self._REQUIRED_KEYS - result.keys()
        self.assertFalse(missing, f"Missing keys: {missing}")

    def test_ok_result_has_required_keys(self):
        router = _router_with(_MockAdapter())
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}
        )
        self._assert_schema(result)

    def test_adapter_not_found_has_required_keys(self):
        router = ServiceRouter()
        result = router.execute_with_lifecycle("missing", RouteOp.RUN_ACTION)
        self._assert_schema(result)

    def test_open_session_failure_has_required_keys(self):
        adapter = _MockAdapter()
        adapter.open_session_status = "error"
        router = _router_with(adapter)
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION)
        self._assert_schema(result)

    def test_preflight_failure_has_required_keys(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION,
            policy=LifecyclePolicy(run_preflight=True),
        )
        self._assert_schema(result)

    def test_execute_failure_has_required_keys(self):
        adapter = _MockAdapter()
        adapter.run_action_raises = RuntimeError("boom")
        router = _router_with(adapter)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}
        )
        self._assert_schema(result)


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — happy path
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecycleHappyPath(unittest.TestCase):

    def _run(self, op, op_args=None, policy=None):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        result = router.execute_with_lifecycle("test", op, op_args, policy)
        return result, adapter

    def test_run_action_status_ok(self):
        result, _ = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["status"], "ok")

    def test_run_action_reason_empty(self):
        result, _ = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["reason"], "")

    def test_run_action_stage_execute(self):
        result, _ = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["stage"], "execute")

    def test_run_action_payload_is_adapter_return(self):
        result, _ = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["payload"], "action_done")

    def test_run_action_params_forwarded(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "walk", "params": {"speed": 1}}
        )
        # Verify adapter.run_action was called with correct args
        run_calls = [c for c in adapter.calls if c[0] == "run_action"]
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0][1], "walk")        # c[1] = action name
        self.assertEqual(run_calls[0][2], {"speed": 1})  # c[2] = params dict

    def test_get_sensor_data_payload_is_sensor_dict(self):
        result, _ = self._run(RouteOp.GET_SENSOR_DATA)
        self.assertEqual(result["payload"], {"x": 42})

    def test_get_sensor_data_status_ok(self):
        result, _ = self._run(RouteOp.GET_SENSOR_DATA)
        self.assertEqual(result["status"], "ok")

    def test_stop_status_ok(self):
        result, _ = self._run(RouteOp.STOP)
        self.assertEqual(result["status"], "ok")

    def test_stop_payload_is_none(self):
        result, _ = self._run(RouteOp.STOP)
        self.assertIsNone(result["payload"])

    def test_open_session_is_called(self):
        _, adapter = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 1)

    def test_close_session_not_called_by_default(self):
        _, adapter = self._run(RouteOp.RUN_ACTION, {"action": "stand"})
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)

    def test_no_args_defaults_to_empty_op_args(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        result = router.execute_with_lifecycle("test", RouteOp.GET_SENSOR_DATA)
        self.assertEqual(result["status"], "ok")

    def test_session_config_forwarded_to_open_session(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(session_config={"host": "localhost"})
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"}, policy)
        open_calls = [(c[1]) for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(open_calls[0], {"host": "localhost"})


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — adapter not found
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecycleAdapterNotFound(unittest.TestCase):

    def setUp(self):
        self.router = ServiceRouter()  # empty registry
        self.result = self.router.execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)

    def test_status_is_error(self):
        self.assertEqual(self.result["status"], "error")

    def test_reason_is_adapter_not_found(self):
        self.assertEqual(self.result["reason"], LifecycleReason.ADAPTER_NOT_FOUND)

    def test_stage_is_acquire(self):
        self.assertEqual(self.result["stage"], "acquire")

    def test_payload_is_none(self):
        self.assertIsNone(self.result["payload"])

    def test_diagnostics_contains_adapter_name(self):
        self.assertIn("adapter_name", self.result["diagnostics"])
        self.assertEqual(self.result["diagnostics"]["adapter_name"], "ghost")


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — open_session failure
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecycleOpenSessionFailure(unittest.TestCase):

    def setUp(self):
        self.adapter = _MockAdapter()
        self.adapter.open_session_status = "error"
        self.router = _router_with(self.adapter)
        self.result = self.router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}
        )

    def test_status_is_error(self):
        self.assertEqual(self.result["status"], "error")

    def test_reason_is_session_open_failed(self):
        self.assertEqual(self.result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_stage_is_open_session(self):
        self.assertEqual(self.result["stage"], "open_session")

    def test_payload_is_none(self):
        self.assertIsNone(self.result["payload"])

    def test_execute_not_called_after_open_failure(self):
        run_calls = [c for c in self.adapter.calls if c[0] == "run_action"]
        self.assertEqual(len(run_calls), 0)

    def test_close_session_not_called_after_open_failure(self):
        close_calls = [c for c in self.adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — preflight
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecyclePreflight(unittest.TestCase):

    def test_preflight_not_called_by_default(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"})
        pre_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(pre_calls), 0)

    def test_preflight_called_when_policy_run_preflight_true(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"}, policy)
        pre_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(pre_calls), 1)

    def test_preflight_context_forwarded(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True, preflight_context={"timeout": 3})
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"}, policy)
        pre_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(pre_calls[0][1], {"timeout": 3})

    def test_preflight_ok_continues_to_execute(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        self.assertEqual(result["status"], "ok")
        run_calls = [c for c in adapter.calls if c[0] == "run_action"]
        self.assertEqual(len(run_calls), 1)

    def test_preflight_fail_status_error(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        self.assertEqual(result["status"], "error")

    def test_preflight_fail_reason(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_preflight_fail_stage(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        self.assertEqual(result["stage"], "preflight")

    def test_preflight_fail_execute_not_called(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        run_calls = [c for c in adapter.calls if c[0] == "run_action"]
        self.assertEqual(len(run_calls), 0)

    def test_preflight_fail_close_session_called_as_cleanup(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 1)

    def test_preflight_fail_payload_is_none(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(run_preflight=True)
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {}, policy)
        self.assertIsNone(result["payload"])


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — close_session policy
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecycleCloseSession(unittest.TestCase):

    def test_close_session_not_called_when_close_after_false(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=False)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"}, policy)
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)

    def test_close_session_called_when_close_after_true(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"}, policy)
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 1)

    def test_close_after_true_status_still_ok(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        self.assertEqual(result["status"], "ok")

    def test_close_after_true_payload_preserved(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        self.assertEqual(result["payload"], "action_done")

    def test_close_session_fail_does_not_mask_payload(self):
        adapter = _MockAdapter()
        adapter.close_session_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        # Status is still ok — close failure must not mask the action result
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"], "action_done")

    def test_close_session_fail_surfaced_in_diagnostics(self):
        adapter = _MockAdapter()
        adapter.close_session_status = "error"
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        self.assertIn("close_session_failed", result["diagnostics"])
        self.assertTrue(result["diagnostics"]["close_session_failed"])

    def test_close_session_ok_diagnostics_no_close_failure_keys(self):
        # Phase 6 STAGE-03: diagnostics now contains "telemetry"; the assertion
        # is narrowed to confirm no close-session failure keys are present.
        adapter = _MockAdapter()
        router = _router_with(adapter)
        policy = LifecyclePolicy(close_after=True)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        diag = result["diagnostics"]
        self.assertNotIn("close_session_failed", diag)
        self.assertNotIn("close_reason", diag)
        self.assertNotIn("close_diagnostics", diag)


# ══════════════════════════════════════════════════════════════════════════
# execute_with_lifecycle — execute dispatch failure
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteWithLifecycleExecuteFailure(unittest.TestCase):

    def _run_failing(self, policy=None):
        adapter = _MockAdapter()
        adapter.run_action_raises = RuntimeError("hardware_error")
        router = _router_with(adapter)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}, policy
        )
        return result, adapter

    def test_execute_exception_status_error(self):
        result, _ = self._run_failing()
        self.assertEqual(result["status"], "error")

    def test_execute_exception_reason(self):
        result, _ = self._run_failing()
        # Phase 4 STAGE-06: use the canonical constant, not a magic string
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)
        self.assertEqual(result["reason"], "execute_failed")  # value stability

    def test_execute_exception_stage(self):
        result, _ = self._run_failing()
        self.assertEqual(result["stage"], "execute")

    def test_execute_exception_payload_none(self):
        result, _ = self._run_failing()
        self.assertIsNone(result["payload"])

    def test_execute_exception_message_in_diagnostics(self):
        result, _ = self._run_failing()
        self.assertIn("message", result["diagnostics"])
        self.assertIn("hardware_error", result["diagnostics"]["message"])

    def test_execute_exception_op_in_diagnostics(self):
        result, _ = self._run_failing()
        self.assertIn("op", result["diagnostics"])

    def test_execute_exception_with_close_after_calls_close_session(self):
        policy = LifecyclePolicy(close_after=True)
        _, adapter = self._run_failing(policy=policy)
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 1)

    def test_execute_exception_without_close_after_no_close_session(self):
        policy = LifecyclePolicy(close_after=False)
        _, adapter = self._run_failing(policy=policy)
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)

    def test_unknown_op_raises_before_execute(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        result = router.execute_with_lifecycle("test", "unknown_op")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "execute")


# ══════════════════════════════════════════════════════════════════════════
# Legacy passthrough interface (must remain unchanged)
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyServiceRouterInterface(unittest.TestCase):

    def setUp(self):
        self.adapter = _MockAdapter()
        self.router = _router_with(self.adapter)

    def test_run_action_legacy_returns_adapter_value(self):
        result = self.router.run_action("test", "stand")
        self.assertEqual(result, "action_done")

    def test_run_action_legacy_passes_params(self):
        self.router.run_action("test", "walk", speed=2)
        run_calls = [c for c in self.adapter.calls if c[0] == "run_action"]
        self.assertEqual(run_calls[0][2], {"speed": 2})  # c[2] = params dict

    def test_stop_legacy_calls_adapter_stop(self):
        self.router.stop("test")
        stop_calls = [c for c in self.adapter.calls if c[0] == "stop"]
        self.assertEqual(len(stop_calls), 1)

    def test_get_sensor_data_legacy_returns_sensor_dict(self):
        result = self.router.get_sensor_data("test")
        self.assertEqual(result, {"x": 42})

    def test_health_legacy_returns_health_dict(self):
        result = self.router.health("test")
        self.assertIsInstance(result, dict)

    def test_get_adapter_returns_registered_adapter(self):
        self.assertIs(self.router.get_adapter("test"), self.adapter)

    def test_get_adapter_raises_on_missing(self):
        with self.assertRaises(KeyError):
            self.router.get_adapter("does_not_exist")

    def test_register_adapter_then_get(self):
        other = _MockAdapter()
        router = ServiceRouter()
        router.register_adapter("other", other)
        self.assertIs(router.get_adapter("other"), other)

    def test_legacy_methods_do_not_call_open_session(self):
        self.router.run_action("test", "stand")
        open_calls = [c for c in self.adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 0)

    def test_legacy_methods_do_not_call_close_session(self):
        self.router.run_action("test", "stand")
        close_calls = [c for c in self.adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)


# ══════════════════════════════════════════════════════════════════════════
# Default policy reproduces legacy behaviour
# ══════════════════════════════════════════════════════════════════════════

class TestDefaultPolicyBehavior(unittest.TestCase):
    """With default policy, execute_with_lifecycle behaves like legacy passthrough."""

    def test_default_policy_no_preflight(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"})
        pre_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(pre_calls), 0)

    def test_default_policy_no_close_session(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"})
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 0)

    def test_default_policy_open_session_still_called(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"})
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 1)

    def test_default_policy_payload_returned(self):
        adapter = _MockAdapter()
        router = _router_with(adapter)
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"}
        )
        self.assertEqual(result["payload"], "action_done")


# ══════════════════════════════════════════════════════════════════════════
# Phase 5 settings validation integration
# ══════════════════════════════════════════════════════════════════════════

def _valid_unitree_config() -> dict:
    """Minimal valid Unitree session_config for validation tests."""
    return {
        "brand":              "unitree",
        "robot_type":         "go2",
        "connection_mode":    "simulation",
        "timeout":            30,
        "stop_policy":        "graceful",
        "safety_enabled":     True,
        "dds_domain_id":      0,
        "network_interface":  "eth0",
        "control_level":      "high",
        "mode_switch_policy": "immediate",
    }


class TestSettingsValidationRouterIntegration(unittest.TestCase):
    """execute_with_lifecycle Step 0 — settings validation gating."""

    def _router(self):
        adapter = _MockAdapter()
        return _router_with(adapter), adapter

    # ── Validation skipped when no brand in session_config ─────────────────

    def test_no_brand_validation_skipped_status_ok(self):
        """Legacy callers with empty session_config are unaffected."""
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config={}),
        )
        self.assertEqual(result["status"], "ok")

    def test_no_brand_validation_skipped_open_session_called(self):
        """Without brand, validation skipped → open_session still called."""
        router, adapter = self._router()
        router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config={"host": "localhost"}),
        )
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 1)

    def test_default_policy_no_brand_unaffected(self):
        """Default policy (empty session_config) still works after Phase 5."""
        router, _ = self._router()
        result = router.execute_with_lifecycle("test", RouteOp.RUN_ACTION, {"action": "stand"})
        self.assertEqual(result["status"], "ok")

    # ── Valid config proceeds normally ─────────────────────────────────────

    def test_valid_config_status_ok(self):
        """Valid brand config passes validation → execution succeeds."""
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=_valid_unitree_config()),
        )
        self.assertEqual(result["status"], "ok")

    def test_valid_config_open_session_called(self):
        """Valid config → validation passes → open_session is called."""
        router, adapter = self._router()
        router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=_valid_unitree_config()),
        )
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 1)

    def test_valid_config_payload_returned(self):
        """Valid config → action payload returned normally."""
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=_valid_unitree_config()),
        )
        self.assertEqual(result["payload"], "action_done")

    # ── Invalid config blocked before acquire ──────────────────────────────

    def test_invalid_config_status_error(self):
        """Missing required key → validation fails → status error."""
        config = dict(_valid_unitree_config())
        del config["timeout"]
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        self.assertEqual(result["status"], "error")

    def test_invalid_config_reason_preflight_failed(self):
        """Validation failure → reason is PREFLIGHT_FAILED."""
        config = dict(_valid_unitree_config())
        del config["dds_domain_id"]
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_invalid_config_stage_settings_validation(self):
        """Validation failure → stage is 'settings_validation'."""
        config = dict(_valid_unitree_config())
        config["connection_mode"] = "cloud"  # disallowed choice
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        self.assertEqual(result["stage"], "settings_validation")

    def test_invalid_config_open_session_not_called(self):
        """Validation failure → open_session is never called."""
        config = dict(_valid_unitree_config())
        del config["safety_enabled"]
        router, adapter = self._router()
        router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 0)

    def test_invalid_config_execute_not_called(self):
        """Validation failure → run_action is never called."""
        config = dict(_valid_unitree_config())
        del config["control_level"]
        router, adapter = self._router()
        router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        run_calls = [c for c in adapter.calls if c[0] == "run_action"]
        self.assertEqual(len(run_calls), 0)

    def test_invalid_config_diagnostics_contains_validation(self):
        """Validation failure diagnostics contain 'validation' key."""
        config = dict(_valid_unitree_config())
        config["stop_policy"] = "force"  # disallowed choice
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        self.assertIn("validation", result["diagnostics"])

    def test_invalid_config_validation_result_in_diagnostics(self):
        """diagnostics['validation'] is a proper ValidationResult dict."""
        config = dict(_valid_unitree_config())
        del config["network_interface"]
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        val = result["diagnostics"]["validation"]
        for key in ("status", "brand", "errors", "missing", "invalid"):
            self.assertIn(key, val)
        self.assertEqual(val["status"], "error")
        self.assertIn("network_interface", val["missing"])

    def test_invalid_config_adapter_name_in_result(self):
        """Validation failure result still carries adapter_name."""
        config = dict(_valid_unitree_config())
        del config["timeout"]
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        self.assertEqual(result["adapter_name"], "test")

    def test_invalid_config_result_schema_complete(self):
        """Validation failure result has all required result keys."""
        config = dict(_valid_unitree_config())
        config["brand"] = "unknown_brand"
        router, _ = self._router()
        result = router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )
        for key in ("status", "reason", "stage", "payload", "diagnostics", "adapter_name"):
            self.assertIn(key, result)

    # ── Phase 6: mandatory diagnostic keys on settings_validation failure ──

    def _invalid_result(self, drop_key: str | None = None,
                        bad_key: str | None = None, bad_val=None) -> dict:
        """Return a settings_validation failure result with one key missing or bad."""
        config = dict(_valid_unitree_config())
        if drop_key:
            del config[drop_key]
        if bad_key:
            config[bad_key] = bad_val
        router, _ = self._router()
        return router.execute_with_lifecycle(
            "test", RouteOp.RUN_ACTION, {"action": "stand"},
            LifecyclePolicy(session_config=config),
        )

    def test_diagnostics_contains_brand_key(self):
        """settings_validation failure diagnostics include 'brand'."""
        result = self._invalid_result(drop_key="timeout")
        self.assertIn("brand", result["diagnostics"])

    def test_diagnostics_brand_matches_session_brand(self):
        """diagnostics['brand'] equals the brand from session_config."""
        result = self._invalid_result(drop_key="timeout")
        self.assertEqual(result["diagnostics"]["brand"], "unitree")

    def test_diagnostics_contains_message_key(self):
        """settings_validation failure diagnostics include 'message'."""
        result = self._invalid_result(drop_key="dds_domain_id")
        self.assertIn("message", result["diagnostics"])

    def test_diagnostics_message_is_deterministic_string(self):
        """diagnostics['message'] is the canonical text 'Settings validation failed'."""
        result = self._invalid_result(drop_key="network_interface")
        self.assertEqual(result["diagnostics"]["message"], "Settings validation failed")

    def test_diagnostics_message_is_nonempty_string(self):
        result = self._invalid_result(drop_key="control_level")
        msg = result["diagnostics"]["message"]
        self.assertIsInstance(msg, str)
        self.assertTrue(msg)

    def test_diagnostics_contains_context_key(self):
        """settings_validation failure diagnostics include 'context'."""
        result = self._invalid_result(drop_key="stop_policy")
        self.assertIn("context", result["diagnostics"])

    def test_diagnostics_context_is_dict(self):
        result = self._invalid_result(drop_key="timeout")
        self.assertIsInstance(result["diagnostics"]["context"], dict)

    def test_diagnostics_context_stage_is_settings_validation(self):
        """diagnostics['context']['stage'] == 'settings_validation'."""
        result = self._invalid_result(drop_key="timeout")
        self.assertEqual(result["diagnostics"]["context"]["stage"], "settings_validation")

    def test_diagnostics_context_adapter_name_matches(self):
        """diagnostics['context']['adapter_name'] matches the registered adapter key."""
        result = self._invalid_result(drop_key="timeout")
        self.assertEqual(result["diagnostics"]["context"]["adapter_name"], "test")

    def test_diagnostics_context_missing_propagated(self):
        """diagnostics['context']['missing'] matches validation result missing list."""
        result = self._invalid_result(drop_key="timeout")
        ctx = result["diagnostics"]["context"]
        val = result["diagnostics"]["validation"]
        self.assertIn("missing", ctx)
        self.assertEqual(ctx["missing"], val["missing"])
        self.assertIn("timeout", ctx["missing"])

    def test_diagnostics_context_invalid_propagated(self):
        """diagnostics['context']['invalid'] matches validation result invalid list."""
        result = self._invalid_result(bad_key="connection_mode", bad_val="wireless")
        ctx = result["diagnostics"]["context"]
        val = result["diagnostics"]["validation"]
        self.assertIn("invalid", ctx)
        self.assertEqual(ctx["invalid"], val["invalid"])
        self.assertIn("connection_mode", ctx["invalid"])

    def test_diagnostics_validation_status_is_error(self):
        """diagnostics['validation']['status'] == 'error'."""
        result = self._invalid_result(drop_key="safety_enabled")
        self.assertEqual(result["diagnostics"]["validation"]["status"], "error")

    def test_diagnostics_all_mandatory_keys_present(self):
        """All four mandatory diagnostic keys are present: brand/message/context/validation."""
        result = self._invalid_result(drop_key="mode_switch_policy")
        diag = result["diagnostics"]
        for key in ("brand", "message", "context", "validation"):
            with self.subTest(key=key):
                self.assertIn(key, diag)

    def test_diagnostics_context_has_required_subkeys(self):
        """context dict contains stage/adapter_name/missing/invalid."""
        result = self._invalid_result(drop_key="timeout")
        ctx = result["diagnostics"]["context"]
        for key in ("stage", "adapter_name", "missing", "invalid"):
            with self.subTest(key=key):
                self.assertIn(key, ctx)


if __name__ == "__main__":
    unittest.main()
