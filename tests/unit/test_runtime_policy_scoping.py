"""Cycle 3 STAGE-03 — Runtime policy scoping unit tests.

Covers:
  - RobotContext.make_run_policy() — pure factory, no class mutation
  - set_lifecycle_policy / get_lifecycle_policy backward-compat
  - _lifecycle_route explicit-policy vs class-level-fallback behavior
  - Snapshot/restore workflow (the pattern used in bin/ui.py _on_run)

All tests are pure Python — no Qt dependency.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bin.core.robot_context import RobotContext
from system.service.lifecycle import LifecyclePolicy, RetryPolicy, SafetyPolicy


def _reset():
    """Reset RobotContext to a clean default state before each test."""
    RobotContext.reset()


# ── Suite A: make_run_policy — pure factory ───────────────────────────────────


class TestMakeRunPolicy(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_creates_policy_with_session_config(self):
        policy = RobotContext.make_run_policy({"brand": "unitree", "host": "10.0.0.1"})
        self.assertEqual(policy.session_config["brand"], "unitree")
        self.assertEqual(policy.session_config["host"], "10.0.0.1")

    def test_inherits_run_preflight_from_base(self):
        base = LifecyclePolicy(run_preflight=True, close_after=False)
        policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=base)
        self.assertTrue(policy.run_preflight)

    def test_inherits_close_after_from_base(self):
        base = LifecyclePolicy(run_preflight=False, close_after=True)
        policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=base)
        self.assertTrue(policy.close_after)

    def test_inherits_safety_policy_from_base(self):
        sp = SafetyPolicy(require_preflight=True)
        base = LifecyclePolicy(safety_policy=sp)
        policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=base)
        self.assertIs(policy.safety_policy, sp)

    def test_inherits_retry_policy_from_base(self):
        rp = RetryPolicy(max_attempts=3)
        base = LifecyclePolicy(retry_policy=rp)
        policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=base)
        self.assertIs(policy.retry_policy, rp)

    def test_inherits_preflight_context_from_base(self):
        base = LifecyclePolicy(preflight_context={"key": "value"})
        policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=base)
        self.assertEqual(policy.preflight_context["key"], "value")

    def test_does_not_mutate_class_level_policy(self):
        original = RobotContext.get_lifecycle_policy()
        original_session = dict(original.session_config)
        RobotContext.make_run_policy({"brand": "unitree", "injected": True})
        after = RobotContext.get_lifecycle_policy()
        self.assertNotIn("injected", after.session_config)
        self.assertEqual(after.session_config, original_session)

    def test_uses_class_level_default_when_no_base_policy(self):
        """make_run_policy() without base_policy inherits class-level policy fields."""
        custom_base = LifecyclePolicy(run_preflight=True, close_after=True)
        RobotContext.set_lifecycle_policy(custom_base)
        policy = RobotContext.make_run_policy({"brand": "unitree"})
        self.assertTrue(policy.run_preflight)
        self.assertTrue(policy.close_after)

    def test_session_config_is_a_copy(self):
        """Mutating the returned session_config does not affect the input dict."""
        cfg = {"brand": "unitree", "port": 8080}
        policy = RobotContext.make_run_policy(cfg)
        policy.session_config["port"] = 9999
        self.assertEqual(cfg["port"], 8080)

    def test_returns_lifecycle_policy_instance(self):
        policy = RobotContext.make_run_policy({"brand": "bostiondynamics"})
        self.assertIsInstance(policy, LifecyclePolicy)

    def test_empty_session_config_produces_empty_dict(self):
        policy = RobotContext.make_run_policy({})
        self.assertEqual(policy.session_config, {})

    def test_multiple_calls_produce_independent_policies(self):
        p1 = RobotContext.make_run_policy({"brand": "unitree", "run": 1})
        p2 = RobotContext.make_run_policy({"brand": "bostiondynamics", "run": 2})
        self.assertEqual(p1.session_config["brand"], "unitree")
        self.assertEqual(p2.session_config["brand"], "bostiondynamics")
        self.assertIsNot(p1, p2)


# ── Suite B: set/get_lifecycle_policy backward-compat ────────────────────────


class TestSetGetLifecyclePolicyCompat(unittest.TestCase):

    def setUp(self):
        _reset()

    def test_set_lifecycle_policy_updates_class_attr(self):
        policy = LifecyclePolicy(run_preflight=True)
        RobotContext.set_lifecycle_policy(policy)
        self.assertIs(RobotContext._lifecycle_policy, policy)

    def test_get_lifecycle_policy_returns_current_policy(self):
        policy = LifecyclePolicy(close_after=True)
        RobotContext.set_lifecycle_policy(policy)
        retrieved = RobotContext.get_lifecycle_policy()
        self.assertIs(retrieved, policy)

    def test_set_then_get_roundtrip(self):
        policy = LifecyclePolicy(run_preflight=True, close_after=True)
        RobotContext.set_lifecycle_policy(policy)
        self.assertIs(RobotContext.get_lifecycle_policy(), policy)

    def test_reset_restores_default_policy(self):
        custom = LifecyclePolicy(run_preflight=True, close_after=True)
        RobotContext.set_lifecycle_policy(custom)
        RobotContext.reset()
        default = RobotContext.get_lifecycle_policy()
        self.assertFalse(default.run_preflight)
        self.assertFalse(default.close_after)
        self.assertEqual(default.session_config, {})


# ── Suite C: _lifecycle_route explicit policy vs class-level fallback ─────────


class TestLifecycleRoutePolicyFallback(unittest.TestCase):
    """Verify _lifecycle_route uses explicit policy when given, else class-level."""

    def setUp(self):
        _reset()

    def _make_mock_router(self, ok=True):
        """Build a mock ServiceRouter whose execute_with_lifecycle returns ok/error."""
        mock_router = MagicMock()
        mock_router.execute_with_lifecycle.return_value = {
            "status": "ok" if ok else "error",
            "reason": "",
            "stage": "execute",
            "payload": True,
            "diagnostics": {},
            "adapter_name": "unitree_sdk2",
        }
        return mock_router

    def test_lifecycle_route_uses_explicit_policy_when_given(self):
        explicit_policy = LifecyclePolicy(run_preflight=True)
        mock_router = self._make_mock_router()
        with patch.object(RobotContext, "_service_router", mock_router):
            RobotContext._lifecycle_route(
                "run_action", "unitree_sdk2",
                op_args={"action": "stand", "params": {}},
                policy=explicit_policy,
            )
        _, kwargs = mock_router.execute_with_lifecycle.call_args
        # policy arg is positional (4th), not keyword
        call_args = mock_router.execute_with_lifecycle.call_args[0]
        passed_policy = call_args[3] if len(call_args) > 3 else kwargs.get("policy")
        self.assertIs(passed_policy, explicit_policy)

    def test_lifecycle_route_falls_back_to_class_policy_when_none(self):
        class_policy = LifecyclePolicy(run_preflight=False, close_after=True)
        RobotContext.set_lifecycle_policy(class_policy)
        mock_router = self._make_mock_router()
        with patch.object(RobotContext, "_service_router", mock_router):
            RobotContext._lifecycle_route(
                "run_action", "unitree_sdk2",
                op_args={"action": "stand", "params": {}},
                policy=None,  # no explicit policy → falls back to class-level
            )
        call_args = mock_router.execute_with_lifecycle.call_args[0]
        passed_policy = call_args[3] if len(call_args) > 3 else None
        self.assertIs(passed_policy, class_policy)

    def test_make_run_policy_can_be_used_in_lifecycle_route(self):
        """make_run_policy result is a valid LifecyclePolicy for _lifecycle_route."""
        run_policy = RobotContext.make_run_policy({"brand": "unitree", "host": "1.2.3.4"})
        mock_router = self._make_mock_router()
        with patch.object(RobotContext, "_service_router", mock_router):
            RobotContext._lifecycle_route(
                "run_action", "unitree_sdk2",
                op_args={"action": "stand", "params": {}},
                policy=run_policy,
            )
        # Verify execute_with_lifecycle was called with our policy
        call_args = mock_router.execute_with_lifecycle.call_args[0]
        passed_policy = call_args[3]
        self.assertEqual(passed_policy.session_config["host"], "1.2.3.4")


# ── Suite D: Snapshot/restore workflow ───────────────────────────────────────


class TestSnapshotRestoreWorkflow(unittest.TestCase):
    """Test the pattern used in bin/ui.py _on_run / _on_mission_finished."""

    def setUp(self):
        _reset()

    def test_snapshot_restore_preserves_original_policy(self):
        original = RobotContext.get_lifecycle_policy()
        # Simulate _on_run: push run policy
        run_policy = RobotContext.make_run_policy({"brand": "unitree"}, base_policy=original)
        RobotContext.set_lifecycle_policy(run_policy)
        snapshot = original
        # Simulate _on_mission_finished: restore
        RobotContext.set_lifecycle_policy(snapshot)
        self.assertIs(RobotContext.get_lifecycle_policy(), original)

    def test_policy_session_config_present_during_run_absent_after_restore(self):
        original = LifecyclePolicy(session_config={})
        RobotContext.set_lifecycle_policy(original)
        sdk_settings = {"brand": "unitree", "host": "10.0.0.1"}
        run_policy = RobotContext.make_run_policy(sdk_settings, base_policy=original)
        RobotContext.set_lifecycle_policy(run_policy)
        # During run: settings are in the active policy
        active = RobotContext.get_lifecycle_policy()
        self.assertEqual(active.session_config["host"], "10.0.0.1")
        # After restore:
        RobotContext.set_lifecycle_policy(original)
        restored = RobotContext.get_lifecycle_policy()
        self.assertNotIn("host", restored.session_config)

    def test_snapshot_is_not_mutated_by_run_policy_creation(self):
        original = LifecyclePolicy(session_config={"pre_existing": "value"})
        RobotContext.set_lifecycle_policy(original)
        run_policy = RobotContext.make_run_policy(
            {"brand": "unitree", "extra": "data"},
            base_policy=original,
        )
        RobotContext.set_lifecycle_policy(run_policy)
        # Original policy object must be unchanged
        self.assertNotIn("extra", original.session_config)
        self.assertEqual(original.session_config, {"pre_existing": "value"})

    def test_consecutive_run_policy_bind_and_restore(self):
        """Two sequential run cycles: each restores properly."""
        initial = LifecyclePolicy(session_config={})
        RobotContext.set_lifecycle_policy(initial)

        # First run
        p1 = RobotContext.make_run_policy({"brand": "unitree", "run": 1}, base_policy=initial)
        RobotContext.set_lifecycle_policy(p1)
        snap1 = initial
        self.assertEqual(RobotContext.get_lifecycle_policy().session_config["run"], 1)
        RobotContext.set_lifecycle_policy(snap1)  # restore
        self.assertIs(RobotContext.get_lifecycle_policy(), initial)

        # Second run
        p2 = RobotContext.make_run_policy({"brand": "bostiondynamics", "run": 2}, base_policy=initial)
        RobotContext.set_lifecycle_policy(p2)
        snap2 = initial
        self.assertEqual(RobotContext.get_lifecycle_policy().session_config["run"], 2)
        RobotContext.set_lifecycle_policy(snap2)  # restore
        self.assertIs(RobotContext.get_lifecycle_policy(), initial)

    def test_make_run_policy_with_all_base_fields_preserved(self):
        """All LifecyclePolicy fields from base_policy survive make_run_policy."""
        sp = SafetyPolicy(require_preflight=True)
        rp = RetryPolicy(max_attempts=5)
        base = LifecyclePolicy(
            run_preflight=True,
            close_after=True,
            session_config={"prior": "data"},
            preflight_context={"ctx": "yes"},
            safety_policy=sp,
            retry_policy=rp,
        )
        run_policy = RobotContext.make_run_policy({"brand": "xiaomi"}, base_policy=base)
        self.assertTrue(run_policy.run_preflight)
        self.assertTrue(run_policy.close_after)
        self.assertEqual(run_policy.preflight_context["ctx"], "yes")
        self.assertIs(run_policy.safety_policy, sp)
        self.assertIs(run_policy.retry_policy, rp)
        # session_config replaced, not merged
        self.assertEqual(run_policy.session_config["brand"], "xiaomi")
        self.assertNotIn("prior", run_policy.session_config)


# ── Suite E: Thread-local run-scoped policy (Cycle 3 STAGE-03 fix) ───────────


class TestRunScopedPolicyThreadLocal(unittest.TestCase):
    """Verify thread-local policy mechanism and _lifecycle_route precedence."""

    def setUp(self):
        _reset()

    def test_no_global_mutation_from_make_run_policy(self):
        """make_run_policy does not set class-level or thread-local policy."""
        original_class = RobotContext.get_lifecycle_policy()
        RobotContext.make_run_policy({"brand": "unitree", "host": "1.2.3.4"})
        self.assertIs(RobotContext.get_lifecycle_policy(), original_class)
        self.assertIsNone(RobotContext.get_run_scoped_policy())

    def test_set_and_get_run_scoped_policy(self):
        policy = LifecyclePolicy(run_preflight=True)
        RobotContext.set_run_scoped_policy(policy)
        self.assertIs(RobotContext.get_run_scoped_policy(), policy)

    def test_clear_run_scoped_policy(self):
        RobotContext.set_run_scoped_policy(LifecyclePolicy())
        RobotContext.clear_run_scoped_policy()
        self.assertIsNone(RobotContext.get_run_scoped_policy())

    def test_scoped_policy_does_not_touch_class_policy(self):
        original = RobotContext.get_lifecycle_policy()
        RobotContext.set_run_scoped_policy(LifecyclePolicy(run_preflight=True))
        self.assertIs(RobotContext.get_lifecycle_policy(), original)

    def test_thread_receives_scoped_policy_not_class_policy(self):
        """Policy set via set_run_scoped_policy is visible in the same thread."""
        run_p = LifecyclePolicy(session_config={"brand": "unitree", "marker": True})
        RobotContext.set_run_scoped_policy(run_p)
        self.assertIs(RobotContext.get_run_scoped_policy(), run_p)
        # Class-level policy is unaffected
        self.assertNotIn("marker", RobotContext.get_lifecycle_policy().session_config)

    def test_thread_isolation(self):
        """Thread-local slot in one thread is invisible to another thread."""
        import threading as _threading
        RobotContext.set_run_scoped_policy(LifecyclePolicy(run_preflight=True))
        other_result = {}

        def _probe():
            other_result["seen"] = RobotContext.get_run_scoped_policy()

        t = _threading.Thread(target=_probe)
        t.start()
        t.join(timeout=3)
        self.assertIsNone(other_result.get("seen"))

    def test_policy_cleared_in_thread_does_not_affect_main_thread(self):
        """Clearing policy in a worker thread does not affect the calling thread."""
        import threading as _threading
        main_policy = LifecyclePolicy(run_preflight=True)
        RobotContext.set_run_scoped_policy(main_policy)

        def _worker():
            # Set and clear policy in the worker thread
            RobotContext.set_run_scoped_policy(LifecyclePolicy(close_after=True))
            RobotContext.clear_run_scoped_policy()

        t = _threading.Thread(target=_worker)
        t.start()
        t.join(timeout=3)
        # Main thread's policy must be unchanged
        self.assertIs(RobotContext.get_run_scoped_policy(), main_policy)

    def test_run_full_lifecycle_without_global_mutation(self):
        """Simulate full run-lifecycle: policy active during run, absent after."""
        original_class = RobotContext.get_lifecycle_policy()
        run_policy = RobotContext.make_run_policy({"brand": "unitree", "host": "10.0.0.1"})

        # Simulate thread start: activate run-scoped policy
        RobotContext.set_run_scoped_policy(run_policy)
        # During run: scoped policy is active; class policy unchanged
        self.assertIs(RobotContext.get_run_scoped_policy(), run_policy)
        self.assertIs(RobotContext.get_lifecycle_policy(), original_class)

        # Simulate thread end: clear run-scoped policy
        RobotContext.clear_run_scoped_policy()
        # After run: scoped slot empty; class policy still unchanged
        self.assertIsNone(RobotContext.get_run_scoped_policy())
        self.assertIs(RobotContext.get_lifecycle_policy(), original_class)


if __name__ == "__main__":
    unittest.main()
