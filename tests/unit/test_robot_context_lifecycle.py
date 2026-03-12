#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 3 STAGE-04: RobotContext lifecycle integration.

Covers:
  - Public API backward compat (run_action→bool, get_sensor_data→dict, stop→None)
  - Brand/model mapping still resolves Unitree paths unchanged
  - Lifecycle routing wired in (execute_with_lifecycle called per op)
  - Policy management: set/get_lifecycle_policy, reset()
  - BRAND_ADAPTER_MAP Phase 4 extension hooks present
  - Fallback path when lifecycle route returns error

Isolation strategy
------------------
All Qt-dependent and SDK-dependent modules are stubbed in sys.modules
BEFORE any bin.core import occurs.  Only *leaf* modules are replaced;
bin and bin.core are imported from the real filesystem.
Section B (wiring tests) use unittest.mock.patch.object for reliable
observation of execute_with_lifecycle calls.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use offscreen Qt platform so PySide6 modules (including bin.core.logger)
# can be imported without a real display.  Must be set before any PySide6
# import occurs (i.e. before any bin.core import below).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

def _make_mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

# models stubs
class _FakeBrandRegistry:
    def get_robot_brand_map(self):
        return {"go2": "unitree", "a1": "unitree", "b1": "unitree",
                "b2": "unitree", "h1": "unitree"}
    def get_brand_model_map(self):
        return {"unitree": ["go2", "a1", "b1", "b2", "h1"]}
    def discover(self):
        pass
    def get_brands(self):
        return ["Unitree"]

class _FakeBaseRobotModel:
    is_available = True
    def run_action(self, action, **params): return True
    def stop(self): pass
    def get_sensor_data(self): return {"sensor": "data"}
    def get_available_actions(self): return ["stand", "sit"]

class _FakeUnitreeModel(_FakeBaseRobotModel):
    def __init__(self, robot_type="go2"):
        self.robot_type = robot_type

# ── Imports under test ─────────────────────────────────────────────────────
from system.service.lifecycle import LifecyclePolicy, LifecycleReason, LifecycleResult  # noqa: E402
from system.service.service_router import RouteOp, ServiceRouter                        # noqa: E402
from system.service.service_registry import ServiceRegistry                             # noqa: E402
from bin.core.robot_context import RobotContext                                         # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

_OK_RESULT = {
    "status": "ok", "reason": "", "stage": "execute",
    "payload": True, "diagnostics": {},
}
_ERR_RESULT = {
    "status": "error", "reason": LifecycleReason.SESSION_OPEN_FAILED,
    "stage": "open_session", "payload": None, "diagnostics": {},
}


class _RobotContextTestBase(unittest.TestCase):
    def setUp(self):
        RobotContext.reset()

    def tearDown(self):
        RobotContext.reset()


# ══════════════════════════════════════════════════════════════════════════
# A — Public API backward compatibility
# ══════════════════════════════════════════════════════════════════════════

class TestPublicApiBackwardCompat(_RobotContextTestBase):

    def test_run_action_returns_bool(self):
        result = RobotContext.run_action("stand")
        self.assertIsInstance(result, bool)

    def test_run_action_returns_true_on_success(self):
        result = RobotContext.run_action("stand")
        self.assertTrue(result)

    def test_run_action_returns_false_when_adapter_unavailable(self):
        # Patch _ensure_adapter at the class to return None
        with patch.object(RobotContext, '_ensure_adapter', return_value=None):
            result = RobotContext.run_action("stand")
        self.assertFalse(result)

    def test_get_sensor_data_returns_dict(self):
        result = RobotContext.get_sensor_data()
        self.assertIsInstance(result, dict)

    def test_get_sensor_data_no_error_key_on_success(self):
        result = RobotContext.get_sensor_data()
        self.assertNotIn("error", result)

    def test_get_sensor_data_returns_error_dict_when_unavailable(self):
        with patch.object(RobotContext, '_ensure_adapter', return_value=None):
            result = RobotContext.get_sensor_data()
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_stop_returns_none(self):
        result = RobotContext.stop()
        self.assertIsNone(result)

    def test_stop_returns_none_even_when_unavailable(self):
        with patch.object(RobotContext, '_ensure_adapter', return_value=None):
            result = RobotContext.stop()
        self.assertIsNone(result)

    def test_run_action_signature_accepts_kwargs(self):
        result = RobotContext.run_action("walk", speed=1.0, direction="forward")
        self.assertIsInstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════
# B — Lifecycle routing wired into public entry points
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleRoutingWired(_RobotContextTestBase):
    """Verify execute_with_lifecycle is called for each public operation."""

    def _patch_ewl(self, return_value=None):
        """Return a context manager that patches execute_with_lifecycle on the
        CURRENT _service_router instance."""
        rv = return_value or _OK_RESULT
        return patch.object(
            RobotContext._service_router,
            'execute_with_lifecycle',
            return_value=rv,
        )

    def test_run_action_triggers_execute_with_lifecycle(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("stand")
        mock.assert_called()

    def test_run_action_op_is_run_action(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("stand")
        # First positional arg (after adapter_name) is op
        called_op = mock.call_args[0][1]
        self.assertEqual(called_op, RouteOp.RUN_ACTION)

    def test_run_action_adapter_name_is_unitree_sdk2(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("stand")
        adapter_name = mock.call_args[0][0]
        self.assertEqual(adapter_name, "unitree_sdk2")

    def test_run_action_action_name_forwarded_in_op_args(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("sit")
        op_args = mock.call_args[0][2]
        self.assertEqual(op_args["action"], "sit")

    def test_run_action_kwargs_forwarded_as_params(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("walk", speed=2)
        op_args = mock.call_args[0][2]
        self.assertEqual(op_args["params"]["speed"], 2)

    def test_run_action_policy_passed(self):
        with self._patch_ewl() as mock:
            RobotContext.run_action("stand")
        policy_arg = mock.call_args[0][3]
        self.assertIsInstance(policy_arg, LifecyclePolicy)

    def test_get_sensor_data_triggers_execute_with_lifecycle(self):
        sensor_result = dict(_OK_RESULT, payload={"x": 1})
        with self._patch_ewl(return_value=sensor_result) as mock:
            RobotContext.get_sensor_data()
        mock.assert_called()

    def test_get_sensor_data_op_is_get_sensor_data(self):
        sensor_result = dict(_OK_RESULT, payload={"x": 1})
        with self._patch_ewl(return_value=sensor_result) as mock:
            RobotContext.get_sensor_data()
        called_op = mock.call_args[0][1]
        self.assertEqual(called_op, RouteOp.GET_SENSOR_DATA)

    def test_stop_triggers_execute_with_lifecycle(self):
        stop_result = dict(_OK_RESULT, payload=None)
        with self._patch_ewl(return_value=stop_result) as mock:
            RobotContext.stop()
        mock.assert_called()

    def test_stop_op_is_stop(self):
        stop_result = dict(_OK_RESULT, payload=None)
        with self._patch_ewl(return_value=stop_result) as mock:
            RobotContext.stop()
        called_op = mock.call_args[0][1]
        self.assertEqual(called_op, RouteOp.STOP)

    def test_run_action_returns_payload_as_bool(self):
        with self._patch_ewl(return_value=dict(_OK_RESULT, payload=True)):
            result = RobotContext.run_action("stand")
        self.assertTrue(result)

    def test_get_sensor_data_returns_payload(self):
        sensor_data = {"temp": 42.0}
        with self._patch_ewl(return_value=dict(_OK_RESULT, payload=sensor_data)):
            result = RobotContext.get_sensor_data()
        self.assertEqual(result, sensor_data)


# ══════════════════════════════════════════════════════════════════════════
# C — Fallback path when lifecycle route returns error
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleRouteFallback(_RobotContextTestBase):

    def test_run_action_does_not_raise_on_lifecycle_error(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_ERR_RESULT):
            try:
                result = RobotContext.run_action("stand")
                self.assertIsInstance(result, bool)
            except Exception as e:
                self.fail(f"run_action raised on lifecycle error: {e}")

    def test_get_sensor_data_does_not_raise_on_lifecycle_error(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_ERR_RESULT):
            try:
                result = RobotContext.get_sensor_data()
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"get_sensor_data raised on lifecycle error: {e}")

    def test_stop_does_not_raise_on_lifecycle_error(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_ERR_RESULT):
            try:
                result = RobotContext.stop()
                self.assertIsNone(result)
            except Exception as e:
                self.fail(f"stop raised on lifecycle error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# D — Policy management
# ══════════════════════════════════════════════════════════════════════════

class TestLifecyclePolicyManagement(_RobotContextTestBase):

    def test_default_policy_is_lifecycle_policy_instance(self):
        self.assertIsInstance(RobotContext.get_lifecycle_policy(), LifecyclePolicy)

    def test_default_policy_run_preflight_is_false(self):
        self.assertFalse(RobotContext.get_lifecycle_policy().run_preflight)

    def test_default_policy_close_after_is_false(self):
        self.assertFalse(RobotContext.get_lifecycle_policy().close_after)

    def test_set_lifecycle_policy_updates_policy(self):
        new_policy = LifecyclePolicy(run_preflight=True, close_after=True)
        RobotContext.set_lifecycle_policy(new_policy)
        self.assertIs(RobotContext.get_lifecycle_policy(), new_policy)

    def test_set_policy_run_preflight_reflected(self):
        RobotContext.set_lifecycle_policy(LifecyclePolicy(run_preflight=True))
        self.assertTrue(RobotContext.get_lifecycle_policy().run_preflight)

    def test_policy_passed_to_execute_with_lifecycle(self):
        custom = LifecyclePolicy(run_preflight=False, close_after=False)
        RobotContext.set_lifecycle_policy(custom)
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext.run_action("stand")
        policy_arg = mock.call_args[0][3]
        self.assertIs(policy_arg, custom)

    def test_reset_restores_default_policy_run_preflight_false(self):
        RobotContext.set_lifecycle_policy(LifecyclePolicy(run_preflight=True))
        RobotContext.reset()
        self.assertFalse(RobotContext.get_lifecycle_policy().run_preflight)

    def test_reset_restores_default_policy_close_after_false(self):
        RobotContext.set_lifecycle_policy(LifecyclePolicy(close_after=True))
        RobotContext.reset()
        self.assertFalse(RobotContext.get_lifecycle_policy().close_after)

    def test_reset_creates_new_router_instance(self):
        old_router = RobotContext._service_router
        RobotContext.reset()
        self.assertIsNot(RobotContext._service_router, old_router)

    def test_reset_creates_new_registry_instance(self):
        old_reg = RobotContext._service_registry
        RobotContext.reset()
        self.assertIsNot(RobotContext._service_registry, old_reg)


# ══════════════════════════════════════════════════════════════════════════
# E — Brand/model mapping unchanged
# ══════════════════════════════════════════════════════════════════════════

class TestBrandModelMappingUnchanged(_RobotContextTestBase):

    def test_go2_brand_is_unitree(self):
        RobotContext.set_robot_type("go2")
        self.assertEqual(RobotContext.get_current_brand(), "unitree")

    def test_a1_brand_is_unitree(self):
        RobotContext.set_robot_type("a1")
        self.assertEqual(RobotContext.get_current_brand(), "unitree")

    def test_unitree_maps_to_unitree_sdk2_adapter(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP.get("unitree"), "unitree_sdk2")

    def test_set_robot_type_go2_returns_true(self):
        self.assertTrue(RobotContext.set_robot_type("go2"))

    def test_get_robot_type_returns_current(self):
        RobotContext.set_robot_type("a1")
        self.assertEqual(RobotContext.get_robot_type(), "a1")

    def test_unknown_robot_type_defaults_to_go2(self):
        RobotContext.set_robot_type("unknown_bot_9000")
        self.assertEqual(RobotContext.get_robot_type(), "go2")

    def test_brand_adapter_map_is_dict(self):
        self.assertIsInstance(RobotContext.BRAND_ADAPTER_MAP, dict)


# ══════════════════════════════════════════════════════════════════════════
# F — Phase 4 extension hooks present
# ══════════════════════════════════════════════════════════════════════════

class TestPhase4ExtensionHooks(_RobotContextTestBase):

    def test_brand_adapter_map_has_bostiondynamics(self):
        # Phase 4: key corrected from "boston_dynamics" → "bostiondynamics"
        self.assertIn("bostiondynamics", RobotContext.BRAND_ADAPTER_MAP)

    def test_brand_adapter_map_has_xiaomi(self):
        # Phase 4: key corrected from "cyberdog" → "xiaomi"
        self.assertIn("xiaomi", RobotContext.BRAND_ADAPTER_MAP)

    def test_bostiondynamics_adapter_name_is_spot_sdk(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["bostiondynamics"], "spot_sdk")

    def test_xiaomi_adapter_name_is_cyberdog_sdk(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["xiaomi"], "cyberdog_sdk")

    def test_all_brand_adapter_values_are_nonempty_strings(self):
        for brand, adapter in RobotContext.BRAND_ADAPTER_MAP.items():
            with self.subTest(brand=brand):
                self.assertIsInstance(adapter, str)
                self.assertNotEqual(adapter, "")

    def test_unitree_still_present_after_extension(self):
        self.assertIn("unitree", RobotContext.BRAND_ADAPTER_MAP)

    def test_at_least_three_brands_in_map(self):
        self.assertGreaterEqual(len(RobotContext.BRAND_ADAPTER_MAP), 3)


# ══════════════════════════════════════════════════════════════════════════
# G — _lifecycle_route helper contract
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleRouteHelper(_RobotContextTestBase):

    def test_lifecycle_route_returns_two_tuple(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT):
            result = RobotContext._lifecycle_route(
                RouteOp.GET_SENSOR_DATA, "unitree_sdk2"
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_lifecycle_route_ok_result_first_element_true(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT):
            ok, _ = RobotContext._lifecycle_route(
                RouteOp.GET_SENSOR_DATA, "unitree_sdk2"
            )
        self.assertTrue(ok)

    def test_lifecycle_route_ok_result_payload_forwarded(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=dict(_OK_RESULT, payload={"x": 1})):
            _, payload = RobotContext._lifecycle_route(
                RouteOp.GET_SENSOR_DATA, "unitree_sdk2"
            )
        self.assertEqual(payload, {"x": 1})

    def test_lifecycle_route_error_result_first_element_false(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_ERR_RESULT):
            ok, _ = RobotContext._lifecycle_route(
                RouteOp.RUN_ACTION, "unitree_sdk2"
            )
        self.assertFalse(ok)

    def test_lifecycle_route_error_payload_is_none(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_ERR_RESULT):
            _, payload = RobotContext._lifecycle_route(
                RouteOp.RUN_ACTION, "unitree_sdk2"
            )
        self.assertIsNone(payload)

    def test_lifecycle_route_passes_class_policy_by_default(self):
        custom = LifecyclePolicy(run_preflight=False)
        RobotContext.set_lifecycle_policy(custom)
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.GET_SENSOR_DATA, "unitree_sdk2")
        policy_arg = mock.call_args[0][3]
        self.assertIs(policy_arg, custom)

    def test_lifecycle_route_explicit_policy_overrides_class_policy(self):
        explicit = LifecyclePolicy(run_preflight=True)
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(
                RouteOp.GET_SENSOR_DATA, "unitree_sdk2", policy=explicit
            )
        policy_arg = mock.call_args[0][3]
        self.assertIs(policy_arg, explicit)

    def test_lifecycle_route_calls_execute_with_lifecycle(self):
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.STOP, "unitree_sdk2")
        mock.assert_called_once_with("unitree_sdk2", RouteOp.STOP, None,
                                     RobotContext._lifecycle_policy)


# ══════════════════════════════════════════════════════════════════════════
# F — Run-scoped (thread-local) policy (Cycle 3 STAGE-03)
# ══════════════════════════════════════════════════════════════════════════


class TestRunScopedPolicy(_RobotContextTestBase):
    """Thread-local run-scoped policy: set/get/clear and _lifecycle_route precedence."""

    def test_get_run_scoped_policy_returns_none_when_not_set(self):
        """Default state: no thread-local policy → get returns None."""
        self.assertIsNone(RobotContext.get_run_scoped_policy())

    def test_set_run_scoped_policy_stores_policy(self):
        policy = LifecyclePolicy(run_preflight=True)
        RobotContext.set_run_scoped_policy(policy)
        self.assertIs(RobotContext.get_run_scoped_policy(), policy)

    def test_clear_run_scoped_policy_returns_none_after_set(self):
        RobotContext.set_run_scoped_policy(LifecyclePolicy(run_preflight=True))
        RobotContext.clear_run_scoped_policy()
        self.assertIsNone(RobotContext.get_run_scoped_policy())

    def test_set_run_scoped_policy_does_not_mutate_class_policy(self):
        original = RobotContext.get_lifecycle_policy()
        run_policy = LifecyclePolicy(run_preflight=True, session_config={"brand": "unitree"})
        RobotContext.set_run_scoped_policy(run_policy)
        self.assertIs(RobotContext.get_lifecycle_policy(), original)
        self.assertNotEqual(
            RobotContext.get_lifecycle_policy().session_config,
            run_policy.session_config,
        )

    def test_lifecycle_route_uses_scoped_policy_over_class_policy(self):
        """Resolution order: thread-local scoped > class-level default."""
        class_policy = LifecyclePolicy(run_preflight=False)
        scoped_policy = LifecyclePolicy(run_preflight=True, session_config={"brand": "unitree"})
        RobotContext.set_lifecycle_policy(class_policy)
        RobotContext.set_run_scoped_policy(scoped_policy)
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.RUN_ACTION, "unitree_sdk2",
                                         {"action": "stand", "params": {}})
        passed_policy = mock.call_args[0][3]
        self.assertIs(passed_policy, scoped_policy)

    def test_lifecycle_route_falls_back_to_class_policy_when_no_scoped(self):
        """When no thread-local policy is set, class-level default is used."""
        class_policy = LifecyclePolicy(close_after=True)
        RobotContext.set_lifecycle_policy(class_policy)
        # no set_run_scoped_policy call
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.RUN_ACTION, "unitree_sdk2",
                                         {"action": "stand", "params": {}})
        passed_policy = mock.call_args[0][3]
        self.assertIs(passed_policy, class_policy)

    def test_lifecycle_route_explicit_arg_overrides_scoped_and_class(self):
        """Explicit policy argument wins over both thread-local and class-level."""
        class_policy  = LifecyclePolicy(run_preflight=False)
        scoped_policy = LifecyclePolicy(run_preflight=True)
        explicit      = LifecyclePolicy(close_after=True)
        RobotContext.set_lifecycle_policy(class_policy)
        RobotContext.set_run_scoped_policy(scoped_policy)
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.RUN_ACTION, "unitree_sdk2",
                                         {"action": "stand", "params": {}},
                                         policy=explicit)
        passed_policy = mock.call_args[0][3]
        self.assertIs(passed_policy, explicit)

    def test_clear_reverts_to_class_policy_in_route(self):
        """After clearing thread-local policy, class-level is used again."""
        class_policy  = LifecyclePolicy(run_preflight=False)
        scoped_policy = LifecyclePolicy(run_preflight=True)
        RobotContext.set_lifecycle_policy(class_policy)
        RobotContext.set_run_scoped_policy(scoped_policy)
        RobotContext.clear_run_scoped_policy()
        with patch.object(RobotContext._service_router, 'execute_with_lifecycle',
                          return_value=_OK_RESULT) as mock:
            RobotContext._lifecycle_route(RouteOp.RUN_ACTION, "unitree_sdk2",
                                         {"action": "stand", "params": {}})
        passed_policy = mock.call_args[0][3]
        self.assertIs(passed_policy, class_policy)

    def test_run_scoped_policy_thread_isolation(self):
        """Thread-local: policy set in one thread is invisible to another."""
        import threading

        other_thread_policy_seen = {}

        scoped = LifecyclePolicy(run_preflight=True, session_config={"brand": "unitree"})
        RobotContext.set_run_scoped_policy(scoped)  # set in main (test) thread

        def _probe():
            # A different thread must see None (not the main-thread scoped policy)
            other_thread_policy_seen["policy"] = RobotContext.get_run_scoped_policy()

        t = threading.Thread(target=_probe)
        t.start()
        t.join(timeout=3)
        self.assertIsNone(other_thread_policy_seen.get("policy"),
                          "Thread-local policy leaked to another thread")

    def test_reset_clears_thread_local_slot(self):
        """reset() replaces the threading.local() instance, clearing all thread slots."""
        RobotContext.set_run_scoped_policy(LifecyclePolicy(run_preflight=True))
        RobotContext.reset()
        self.assertIsNone(RobotContext.get_run_scoped_policy())


def setup_module(module):
    """Install model stubs into sys.modules for the duration of this test module."""
    sys.modules["models.brand_registry"] = _make_mod(
        "models.brand_registry", BrandRegistry=_FakeBrandRegistry
    )
    sys.modules["models.base"] = _make_mod(
        "models.base", BaseRobotModel=_FakeBaseRobotModel
    )
    sys.modules["models.Unitree"] = _make_mod(
        "models.Unitree", UnitreeModel=_FakeUnitreeModel
    )


def teardown_module(module):
    """Restore stubbed sys.modules entries after this test module so that
    subsequent test modules can import the real models package cleanly."""
    import sys as _sys
    for key in ("models.brand_registry", "models.base", "models.Unitree", "models"):
        _sys.modules.pop(key, None)
    for key in list(_sys.modules.keys()):
        if key.startswith("models."):
            _sys.modules.pop(key, None)
    # Also evict system.brand_packages.unitree modules imported while the fake
    # models.base stub was active so subsequent tests get the real classes.
    for key in list(_sys.modules.keys()):
        if key.startswith("system.brand_packages.unitree"):
            _sys.modules.pop(key, None)


if __name__ == "__main__":
    unittest.main()
