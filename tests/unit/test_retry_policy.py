#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 STAGE-05: Retry policy boundary tests.

Covers:
  A. RetryPolicy dataclass defaults and shape
  B. LifecyclePolicy.retry_policy integration
  C. Router: no-retry paths (backward compatibility)
  D. Router: retry succeeds within budget
  E. Router: budget exhaustion → RETRY_BUDGET_EXHAUSTED
  F. Router: non-retryable operations (STOP, custom)
  G. Telemetry: attempt index increments across retries
"""

import sys
import unittest
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import (                        # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
    RetryPolicy, SafetyPolicy,
)
from system.service.service_router import RouteOp, ServiceRouter  # noqa: E402
from system.service.service_registry import ServiceRegistry        # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter       # noqa: E402


# ── Configurable mock adapter ─────────────────────────────────────────────

class _CountingAdapter(BaseAdapter):
    """Fails for the first *fail_times* calls to run_action, then succeeds."""

    def __init__(self, fail_times: int = 0, fail_exc=None):
        self._fail_times   = fail_times
        self._fail_exc     = fail_exc or RuntimeError("transient")
        self.call_count    = 0

    def connect(self, **kw):  return True
    def stop(self):           pass
    def get_sensor_data(self):return {"x": 1}
    def health(self):         return {"ok": True}

    def run_action(self, action, **params):
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise self._fail_exc
        return {"result": "ok", "call": self.call_count}

    def open_session(self, config=None):
        return LifecycleResult.ok("open_session").to_dict()

    def preflight(self, context=None):
        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self):
        return LifecycleResult.ok("close_session").to_dict()


def _router_with(adapter, name="dev"):
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


def _get_telemetry(result) -> List[dict]:
    return result.get("diagnostics", {}).get("telemetry", [])


def _execute_events(result) -> List[dict]:
    return [e for e in _get_telemetry(result) if e["stage"] == "execute"]


# ══════════════════════════════════════════════════════════════════════════
# A — RetryPolicy dataclass
# ══════════════════════════════════════════════════════════════════════════

class TestRetryPolicyDefaults(unittest.TestCase):

    def test_max_attempts_default_is_1(self):
        self.assertEqual(RetryPolicy().max_attempts, 1)

    def test_backoff_ms_default_is_0(self):
        self.assertEqual(RetryPolicy().backoff_ms, 0)

    def test_retryable_reasons_is_frozenset(self):
        self.assertIsInstance(RetryPolicy().retryable_reasons, frozenset)

    def test_non_retryable_ops_is_frozenset(self):
        self.assertIsInstance(RetryPolicy().non_retryable_ops, frozenset)

    def test_retryable_reasons_contains_execute_failed(self):
        self.assertIn(LifecycleReason.EXECUTE_FAILED, RetryPolicy().retryable_reasons)

    def test_retryable_reasons_contains_spot_timeout(self):
        self.assertIn("spot_command_timeout", RetryPolicy().retryable_reasons)

    def test_retryable_reasons_contains_cyberdog_action_failed(self):
        self.assertIn("cyberdog_action_failed", RetryPolicy().retryable_reasons)

    def test_non_retryable_ops_contains_stop(self):
        self.assertIn(RouteOp.STOP, RetryPolicy().non_retryable_ops)

    def test_custom_max_attempts(self):
        rp = RetryPolicy(max_attempts=5)
        self.assertEqual(rp.max_attempts, 5)

    def test_custom_backoff_ms(self):
        rp = RetryPolicy(backoff_ms=100)
        self.assertEqual(rp.backoff_ms, 100)

    def test_custom_retryable_reasons(self):
        rp = RetryPolicy(retryable_reasons=frozenset({"execute_failed"}))
        self.assertEqual(rp.retryable_reasons, frozenset({"execute_failed"}))

    def test_custom_non_retryable_ops(self):
        rp = RetryPolicy(non_retryable_ops=frozenset({"stop", "get_sensor_data"}))
        self.assertIn("get_sensor_data", rp.non_retryable_ops)

    def test_default_retryable_reasons_has_3_codes(self):
        self.assertEqual(len(RetryPolicy().retryable_reasons), 3)

    def test_default_non_retryable_ops_has_1_op(self):
        self.assertEqual(len(RetryPolicy().non_retryable_ops), 1)


# ══════════════════════════════════════════════════════════════════════════
# B — LifecyclePolicy.retry_policy integration
# ══════════════════════════════════════════════════════════════════════════

class TestLifecyclePolicyRetryIntegration(unittest.TestCase):

    def test_retry_policy_default_is_none(self):
        self.assertIsNone(LifecyclePolicy().retry_policy)

    def test_can_attach_retry_policy(self):
        rp = RetryPolicy(max_attempts=3)
        lp = LifecyclePolicy(retry_policy=rp)
        self.assertIs(lp.retry_policy, rp)

    def test_retry_policy_none_does_not_affect_other_defaults(self):
        lp = LifecyclePolicy()
        self.assertFalse(lp.run_preflight)
        self.assertFalse(lp.close_after)

    def test_retry_policy_is_independent_of_safety_policy(self):
        rp = RetryPolicy(max_attempts=2)
        sp = SafetyPolicy()
        lp = LifecyclePolicy(retry_policy=rp, safety_policy=sp)
        self.assertIsNotNone(lp.retry_policy)
        self.assertIsNotNone(lp.safety_policy)


# ══════════════════════════════════════════════════════════════════════════
# C — Router: no-retry paths
# ══════════════════════════════════════════════════════════════════════════

class TestRouterNoRetry(unittest.TestCase):

    def _run(self, adapter, op=RouteOp.RUN_ACTION, policy=None, name="dev"):
        router = _router_with(adapter, name)
        return router.execute_with_lifecycle(name, op, policy=policy)

    def test_no_retry_policy_execute_fail_returns_execute_failed(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_no_retry_policy_adapter_called_once(self):
        adapter = _CountingAdapter(fail_times=99)
        self._run(adapter)
        self.assertEqual(adapter.call_count, 1)

    def test_retry_policy_max_1_returns_execute_failed(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=1))
        result = self._run(adapter, policy=policy)
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_retry_policy_max_1_adapter_called_once(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=1))
        self._run(adapter, policy=policy)
        self.assertEqual(adapter.call_count, 1)

    def test_non_retryable_reason_not_in_retryable_reasons_returns_execute_failed(self):
        # Custom retryable_reasons that does NOT include execute_failed
        adapter = _CountingAdapter(fail_times=99)
        rp = RetryPolicy(max_attempts=3, retryable_reasons=frozenset({"spot_command_timeout"}))
        policy = LifecyclePolicy(retry_policy=rp)
        result = self._run(adapter, policy=policy)
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_non_retryable_reason_adapter_called_once(self):
        adapter = _CountingAdapter(fail_times=99)
        rp = RetryPolicy(max_attempts=3, retryable_reasons=frozenset({"spot_command_timeout"}))
        policy = LifecyclePolicy(retry_policy=rp)
        self._run(adapter, policy=policy)
        self.assertEqual(adapter.call_count, 1)

    def test_success_with_no_retry_policy(self):
        adapter = _CountingAdapter(fail_times=0)
        result = self._run(adapter)
        self.assertEqual(result["status"], "ok")

    def test_success_adapter_called_once(self):
        adapter = _CountingAdapter(fail_times=0)
        self._run(adapter)
        self.assertEqual(adapter.call_count, 1)


# ══════════════════════════════════════════════════════════════════════════
# D — Router: retry succeeds within budget
# ══════════════════════════════════════════════════════════════════════════

class TestRouterRetrySuccess(unittest.TestCase):

    def _run(self, adapter, policy):
        router = _router_with(adapter)
        return router.execute_with_lifecycle("dev", RouteOp.RUN_ACTION, policy=policy)

    def test_succeeds_on_second_attempt(self):
        adapter = _CountingAdapter(fail_times=1)  # fails once, then succeeds
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = self._run(adapter, policy)
        self.assertEqual(result["status"], "ok")

    def test_adapter_called_twice_on_second_attempt_success(self):
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        self._run(adapter, policy)
        self.assertEqual(adapter.call_count, 2)

    def test_succeeds_on_third_attempt(self):
        adapter = _CountingAdapter(fail_times=2)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = self._run(adapter, policy)
        self.assertEqual(result["status"], "ok")

    def test_adapter_called_three_times(self):
        adapter = _CountingAdapter(fail_times=2)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        self._run(adapter, policy)
        self.assertEqual(adapter.call_count, 3)

    def test_payload_is_present_on_retry_success(self):
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = self._run(adapter, policy)
        self.assertIsNotNone(result["payload"])

    def test_result_has_telemetry_on_retry_success(self):
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = self._run(adapter, policy)
        self.assertIn("telemetry", result["diagnostics"])


# ══════════════════════════════════════════════════════════════════════════
# E — Router: budget exhaustion
# ══════════════════════════════════════════════════════════════════════════

class TestRouterBudgetExhaustion(unittest.TestCase):

    def _run(self, adapter, max_attempts):
        router = _router_with(adapter)
        policy = LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=max_attempts),
        )
        return router.execute_with_lifecycle("dev", RouteOp.RUN_ACTION, policy=policy)

    def test_budget_exhausted_returns_retry_budget_exhausted(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=3)
        self.assertEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)

    def test_budget_exhausted_status_is_error(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=3)
        self.assertEqual(result["status"], "error")

    def test_budget_exhausted_adapter_called_max_attempts_times(self):
        adapter = _CountingAdapter(fail_times=99)
        self._run(adapter, max_attempts=3)
        self.assertEqual(adapter.call_count, 3)

    def test_budget_exhausted_adapter_name_present(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=2)
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], "dev")

    def test_budget_exhausted_has_telemetry(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=2)
        self.assertIn("telemetry", result["diagnostics"])

    def test_budget_exhausted_diagnostics_has_attempts_key(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=3)
        self.assertIn("attempts", result["diagnostics"])

    def test_budget_exhausted_attempts_value_equals_max_attempts(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=3)
        self.assertEqual(result["diagnostics"]["attempts"], 3)

    def test_budget_exhausted_stage_is_execute(self):
        adapter = _CountingAdapter(fail_times=99)
        result = self._run(adapter, max_attempts=2)
        self.assertEqual(result["stage"], "execute")

    def test_max_attempts_2_calls_adapter_twice(self):
        adapter = _CountingAdapter(fail_times=99)
        self._run(adapter, max_attempts=2)
        self.assertEqual(adapter.call_count, 2)

    def test_fails_on_last_attempt_uses_budget_exhausted(self):
        # Fails exactly max_attempts times — budget exhausted
        adapter = _CountingAdapter(fail_times=4)
        result = self._run(adapter, max_attempts=4)
        self.assertEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)

    def test_succeeds_on_last_attempt_is_ok(self):
        # Fails max_attempts-1 times, succeeds on final attempt
        adapter = _CountingAdapter(fail_times=3)
        result = self._run(adapter, max_attempts=4)
        self.assertEqual(result["status"], "ok")


# ══════════════════════════════════════════════════════════════════════════
# F — Router: non-retryable operations
# ══════════════════════════════════════════════════════════════════════════

class TestRouterNonRetryableOps(unittest.TestCase):

    def _run(self, op, adapter, max_attempts=3):
        router = _router_with(adapter)
        policy = LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=max_attempts),
        )
        return router.execute_with_lifecycle("dev", op, policy=policy)

    def test_stop_not_retried_by_default(self):
        # STOP raises from adapter but must not be retried
        class _FailStop(_CountingAdapter):
            def stop(self):
                self.call_count += 1
                raise RuntimeError("stop failed")

        adapter = _FailStop()
        result = self._run(RouteOp.STOP, adapter)
        # STOP is in non_retryable_ops by default → EXECUTE_FAILED, not budget
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_stop_called_only_once_even_with_max_attempts_3(self):
        class _FailStop(_CountingAdapter):
            def stop(self):
                self.call_count += 1
                raise RuntimeError("stop failed")

        adapter = _FailStop()
        self._run(RouteOp.STOP, adapter)
        self.assertEqual(adapter.call_count, 1)

    def test_custom_non_retryable_op_not_retried(self):
        adapter = _CountingAdapter(fail_times=99)
        router = _router_with(adapter)
        rp = RetryPolicy(
            max_attempts=5,
            non_retryable_ops=frozenset({RouteOp.GET_SENSOR_DATA}),
        )
        policy = LifecyclePolicy(retry_policy=rp)
        # GET_SENSOR_DATA raises (monkeypatch)
        class _FailSensor(_CountingAdapter):
            def get_sensor_data(self):
                self.call_count += 1
                raise RuntimeError("sensor fail")
        sensor_adapter = _FailSensor()
        reg = ServiceRegistry()
        reg.register("dev", sensor_adapter)
        router2 = ServiceRouter(registry=reg)
        result = router2.execute_with_lifecycle("dev", RouteOp.GET_SENSOR_DATA, policy=policy)
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)
        self.assertEqual(sensor_adapter.call_count, 1)

    def test_empty_non_retryable_ops_allows_stop_retry(self):
        class _FailStop(_CountingAdapter):
            def stop(self):
                self.call_count += 1
                raise RuntimeError("stop failed")

        adapter = _FailStop()
        router = _router_with(adapter)
        rp = RetryPolicy(max_attempts=3, non_retryable_ops=frozenset())
        policy = LifecyclePolicy(retry_policy=rp)
        result = router.execute_with_lifecycle("dev", RouteOp.STOP, policy=policy)
        # STOP is in retryable with empty non_retryable_ops, budget exhausted
        self.assertEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)
        self.assertEqual(adapter.call_count, 3)


# ══════════════════════════════════════════════════════════════════════════
# G — Telemetry: attempt index across retries
# ══════════════════════════════════════════════════════════════════════════

class TestRetryTelemetry(unittest.TestCase):

    def _run(self, fail_times, max_attempts):
        adapter = _CountingAdapter(fail_times=fail_times)
        router = _router_with(adapter)
        policy = LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=max_attempts),
        )
        return router.execute_with_lifecycle("dev", RouteOp.RUN_ACTION, policy=policy)

    def test_single_attempt_has_one_execute_start(self):
        result = self._run(fail_times=0, max_attempts=1)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(len(starts), 1)

    def test_two_attempts_has_two_execute_starts(self):
        result = self._run(fail_times=99, max_attempts=2)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(len(starts), 2)

    def test_three_attempts_has_three_execute_starts(self):
        result = self._run(fail_times=99, max_attempts=3)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(len(starts), 3)

    def test_first_attempt_index_is_1(self):
        result = self._run(fail_times=99, max_attempts=3)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(starts[0]["attempt"], 1)

    def test_second_attempt_index_is_2(self):
        result = self._run(fail_times=99, max_attempts=3)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(starts[1]["attempt"], 2)

    def test_third_attempt_index_is_3(self):
        result = self._run(fail_times=99, max_attempts=3)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(starts[2]["attempt"], 3)

    def test_success_retry_attempt_index_matches_call_count(self):
        # Fails on attempt 1 and 2, succeeds on attempt 3
        result = self._run(fail_times=2, max_attempts=5)
        end_ok = [e for e in _execute_events(result)
                  if e["event"] == "end" and e["status"] == "ok"]
        self.assertTrue(end_ok)
        self.assertEqual(end_ok[0]["attempt"], 3)

    def test_error_end_events_have_error_status(self):
        result = self._run(fail_times=99, max_attempts=2)
        end_errs = [e for e in _execute_events(result)
                    if e["event"] == "end" and e["status"] == "error"]
        self.assertEqual(len(end_errs), 2)

    def test_all_events_share_same_trace_id(self):
        import uuid
        result = self._run(fail_times=99, max_attempts=3)
        trace_ids = {e["trace_id"] for e in _get_telemetry(result)}
        self.assertEqual(len(trace_ids), 1)
        # And it's a valid UUID4
        tid = trace_ids.pop()
        parsed = uuid.UUID(tid, version=4)
        self.assertEqual(str(parsed), tid)

    def test_execute_events_count_equals_2x_attempts(self):
        # Each attempt: one start + one end
        max_attempts = 3
        result = self._run(fail_times=99, max_attempts=max_attempts)
        execute_evts = _execute_events(result)
        self.assertEqual(len(execute_evts), max_attempts * 2)


if __name__ == "__main__":
    unittest.main()
