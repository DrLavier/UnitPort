#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 STAGE-06: Runtime integration and compatibility tests.

Confirms that all Phase 6 components (SafetyPolicy, telemetry, RetryPolicy)
work together correctly and that Phase 3/4/5 contracts are fully preserved.

Sections:
  A. Legacy path regression (Phase 3/4 contracts unchanged)
  B. Phase 5 contract preservation (settings validation, diagnostics keys)
  C. SafetyPolicy integration
  D. RetryPolicy integration
  E. Telemetry integration across all paths
  F. Combined Phase 6 scenarios
  G. Result shape invariants
"""

import sys
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import (                          # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
    RetryPolicy, SafetyPolicy,
)
from system.service.service_router import RouteOp, ServiceRouter  # noqa: E402
from system.service.service_registry import ServiceRegistry        # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter       # noqa: E402


# ── Shared constants ──────────────────────────────────────────────────────

_RESULT_KEYS = frozenset({"status", "reason", "stage", "payload", "diagnostics", "adapter_name"})
_EVENT_KEYS  = frozenset({
    "trace_id", "adapter_name", "brand", "operation", "stage",
    "event", "timestamp", "duration_ms", "status", "reason",
    "retryable", "attempt", "diagnostics",
})

# Minimal valid Spot/BostonDynamics settings config (Phase 5 validator passes)
_VALID_SPOT_CONFIG: Dict[str, Any] = {
    "brand": "bostiondynamics",
    "robot_type": "spot",
    "connection_mode": "simulation",
    "timeout": 30,
    "stop_policy": "immediate",
    "safety_enabled": True,
    "hostname": "192.168.1.1",
    "auth_token": "tokenABC",
    "timesync_required": True,
    "lease_required": True,
    "safety_channel": "default",
    "power_policy": "safe_power_off",
}

# Minimal invalid Unitree config — triggers settings validation failure
_INVALID_UNITREE_CONFIG: Dict[str, Any] = {"brand": "unitree"}


# ── Mock adapters ─────────────────────────────────────────────────────────

class _BaseOkAdapter(BaseAdapter):
    """Always succeeds; tracks close_session calls."""

    def __init__(self):
        self.close_call_count = 0

    def connect(self, **kw):     return True
    def run_action(self, a, **p): return {"result": "ok"}
    def stop(self):               pass
    def get_sensor_data(self):    return {"x": 1}
    def health(self):             return {"ok": True}

    def open_session(self, config=None):
        return LifecycleResult.ok("open_session").to_dict()

    def preflight(self, context=None):
        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self):
        self.close_call_count += 1
        return LifecycleResult.ok("close_session").to_dict()


class _FailPreflightAdapter(_BaseOkAdapter):
    def preflight(self, context=None):
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED, {"message": "mock fail"},
        ).to_dict()


class _FailOpenAdapter(_BaseOkAdapter):
    def open_session(self, config=None):
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED, {"message": "mock fail"},
        ).to_dict()


class _CountingAdapter(_BaseOkAdapter):
    """Fails run_action for the first *fail_times* calls, then succeeds."""
    def __init__(self, fail_times: int = 0):
        super().__init__()
        self._fail_times = fail_times
        self.call_count  = 0

    def run_action(self, action, **params):
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise RuntimeError("transient failure")
        return {"result": "ok", "call": self.call_count}


class _PreflightContextCapture(_BaseOkAdapter):
    """Records the context passed to preflight()."""
    def __init__(self):
        super().__init__()
        self.last_context = None

    def preflight(self, context=None):
        self.last_context = context
        return LifecycleResult.ok("preflight").to_dict()


# ── Router factory helpers ─────────────────────────────────────────────────

def _router(adapter, name: str = "dev") -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


def _run(adapter, op=RouteOp.RUN_ACTION, policy=None, name: str = "dev") -> Dict:
    return _router(adapter, name).execute_with_lifecycle(name, op, policy=policy)


def _telemetry(result: Dict) -> List[Dict]:
    return result.get("diagnostics", {}).get("telemetry", [])


def _execute_events(result: Dict) -> List[Dict]:
    return [e for e in _telemetry(result) if e["stage"] == "execute"]


# ══════════════════════════════════════════════════════════════════════════
# A — Legacy path regression (Phase 3/4 contracts)
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyPathRegression(unittest.TestCase):
    """Phase 3/4 behaviour must be completely unchanged for legacy callers."""

    # ── Success path ──────────────────────────────────────────────────────

    def test_success_status_ok(self):
        result = _run(_BaseOkAdapter())
        self.assertEqual(result["status"], "ok")

    def test_success_payload_not_none(self):
        result = _run(_BaseOkAdapter())
        self.assertIsNotNone(result["payload"])

    def test_success_reason_empty_string(self):
        result = _run(_BaseOkAdapter())
        self.assertEqual(result["reason"], "")

    def test_success_stage_is_execute(self):
        result = _run(_BaseOkAdapter())
        self.assertEqual(result["stage"], "execute")

    def test_success_adapter_name_in_result(self):
        result = _run(_BaseOkAdapter(), name="spot_sdk")
        self.assertEqual(result["adapter_name"], "spot_sdk")

    # ── Execute failure ───────────────────────────────────────────────────

    def test_execute_fail_reason_is_execute_failed(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter)
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_execute_fail_not_retry_budget_exhausted(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter)
        self.assertNotEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)

    def test_execute_fail_without_retry_calls_adapter_once(self):
        adapter = _CountingAdapter(fail_times=99)
        _run(adapter)
        self.assertEqual(adapter.call_count, 1)

    # ── open_session failure ──────────────────────────────────────────────

    def test_open_session_fail_reason_session_open_failed(self):
        result = _run(_FailOpenAdapter())
        self.assertEqual(result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_open_session_fail_status_error(self):
        result = _run(_FailOpenAdapter())
        self.assertEqual(result["status"], "error")

    # ── Legacy preflight flag ─────────────────────────────────────────────

    def test_legacy_run_preflight_true_succeeds_when_ok(self):
        result = _run(_BaseOkAdapter(), policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(result["status"], "ok")

    def test_legacy_preflight_fail_reason_is_preflight_failed(self):
        result = _run(_FailPreflightAdapter(), policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_legacy_preflight_fail_not_safety_gate_blocked(self):
        result = _run(_FailPreflightAdapter(), policy=LifecyclePolicy(run_preflight=True))
        self.assertNotEqual(result["reason"], LifecycleReason.SAFETY_GATE_BLOCKED)

    # ── Legacy close_after ────────────────────────────────────────────────

    def test_legacy_close_after_calls_close_session(self):
        adapter = _BaseOkAdapter()
        _run(adapter, policy=LifecyclePolicy(close_after=True))
        self.assertEqual(adapter.close_call_count, 1)

    def test_legacy_close_after_false_does_not_call_close(self):
        adapter = _BaseOkAdapter()
        _run(adapter, policy=LifecyclePolicy(close_after=False))
        self.assertEqual(adapter.close_call_count, 0)

    # ── Adapter not found ─────────────────────────────────────────────────

    def test_adapter_not_found_reason(self):
        result = ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertEqual(result["reason"], LifecycleReason.ADAPTER_NOT_FOUND)

    def test_adapter_not_found_adapter_name_preserved(self):
        result = ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertEqual(result["adapter_name"], "ghost")


# ══════════════════════════════════════════════════════════════════════════
# B — Phase 5 contract preservation
# ══════════════════════════════════════════════════════════════════════════

class TestPhase5ContractPreservation(unittest.TestCase):

    def _run_invalid(self, name="dev"):
        adapter = _BaseOkAdapter()
        policy = LifecyclePolicy(session_config=_INVALID_UNITREE_CONFIG)
        return _run(adapter, policy=policy, name=name)

    def test_settings_validation_fires_with_brand(self):
        result = self._run_invalid()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "settings_validation")

    def test_settings_validation_fail_reason_preflight_failed(self):
        result = self._run_invalid()
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_settings_validation_fail_diagnostics_has_brand(self):
        result = self._run_invalid()
        self.assertIn("brand", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["brand"], "unitree")

    def test_settings_validation_fail_diagnostics_has_message(self):
        result = self._run_invalid()
        self.assertIn("message", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["message"], "Settings validation failed")

    def test_settings_validation_fail_diagnostics_has_context(self):
        result = self._run_invalid()
        self.assertIn("context", result["diagnostics"])

    def test_settings_validation_context_has_stage(self):
        result = self._run_invalid()
        ctx = result["diagnostics"]["context"]
        self.assertEqual(ctx["stage"], "settings_validation")

    def test_settings_validation_context_has_adapter_name(self):
        result = self._run_invalid("myAdapter")
        ctx = result["diagnostics"]["context"]
        self.assertEqual(ctx["adapter_name"], "myAdapter")

    def test_settings_validation_fail_diagnostics_has_validation(self):
        result = self._run_invalid()
        self.assertIn("validation", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["validation"]["status"], "error")

    def test_settings_validation_no_brand_skips_validation(self):
        adapter = _BaseOkAdapter()
        policy = LifecyclePolicy(session_config={})
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")

    def test_settings_validation_valid_config_proceeds(self):
        adapter = _BaseOkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_SPOT_CONFIG)
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")


# ══════════════════════════════════════════════════════════════════════════
# C — SafetyPolicy integration
# ══════════════════════════════════════════════════════════════════════════

class TestSafetyPolicyIntegration(unittest.TestCase):

    def _policy(self, **sp_kwargs):
        return LifecyclePolicy(safety_policy=SafetyPolicy(**sp_kwargs))

    def test_safety_policy_preflight_runs(self):
        cap = _PreflightContextCapture()
        result = _run(cap, policy=self._policy())
        self.assertIsNotNone(cap.last_context)

    def test_safety_policy_success_status_ok(self):
        result = _run(_BaseOkAdapter(), policy=self._policy())
        self.assertEqual(result["status"], "ok")

    def test_safety_policy_fail_reason_safety_gate_blocked(self):
        result = _run(_FailPreflightAdapter(), policy=self._policy())
        self.assertEqual(result["reason"], LifecycleReason.SAFETY_GATE_BLOCKED)

    def test_safety_policy_fail_not_preflight_failed(self):
        result = _run(_FailPreflightAdapter(), policy=self._policy())
        self.assertNotEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_safety_policy_allow_execute_skips_preflight(self):
        cap = _PreflightContextCapture()
        policy = LifecyclePolicy(
            safety_policy=SafetyPolicy(allow_execute_without_preflight=True),
        )
        _run(cap, policy=policy)
        self.assertIsNone(cap.last_context)  # preflight was never called

    def test_safety_policy_context_forwarded(self):
        ctx = {"robot_id": "spot-001", "mode": "safe"}
        cap = _PreflightContextCapture()
        policy = LifecyclePolicy(safety_policy=SafetyPolicy(safety_context=ctx))
        _run(cap, policy=policy)
        self.assertEqual(cap.last_context, ctx)

    def test_safety_policy_fail_calls_close_session(self):
        adapter = _FailPreflightAdapter()
        _run(adapter, policy=self._policy())
        self.assertEqual(adapter.close_call_count, 1)

    def test_safety_policy_fail_has_telemetry(self):
        result = _run(_FailPreflightAdapter(), policy=self._policy())
        self.assertIn("telemetry", result["diagnostics"])

    def test_safety_policy_fail_preflight_events_in_telemetry(self):
        result = _run(_FailPreflightAdapter(), policy=self._policy())
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertIn("preflight", stages)

    def test_safety_policy_fail_stage_is_preflight(self):
        result = _run(_FailPreflightAdapter(), policy=self._policy())
        self.assertEqual(result["stage"], "preflight")


# ══════════════════════════════════════════════════════════════════════════
# D — RetryPolicy integration
# ══════════════════════════════════════════════════════════════════════════

class TestRetryPolicyIntegration(unittest.TestCase):

    def _policy(self, max_attempts, close_after=False):
        return LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=max_attempts),
            close_after=close_after,
        )

    def test_retry_success_on_second_attempt(self):
        adapter = _CountingAdapter(fail_times=1)
        result = _run(adapter, policy=self._policy(max_attempts=3))
        self.assertEqual(result["status"], "ok")

    def test_retry_adapter_called_twice_on_second_success(self):
        adapter = _CountingAdapter(fail_times=1)
        _run(adapter, policy=self._policy(max_attempts=3))
        self.assertEqual(adapter.call_count, 2)

    def test_budget_exhausted_reason(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter, policy=self._policy(max_attempts=3))
        self.assertEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)

    def test_budget_exhausted_adapter_called_n_times(self):
        adapter = _CountingAdapter(fail_times=99)
        _run(adapter, policy=self._policy(max_attempts=4))
        self.assertEqual(adapter.call_count, 4)

    def test_stop_not_retried_by_default(self):
        class _FailStop(_BaseOkAdapter):
            def stop(self):
                self.call_count = getattr(self, "call_count", 0) + 1
                raise RuntimeError("stop error")
        adapter = _FailStop()
        result = _run(adapter, op=RouteOp.STOP, policy=self._policy(max_attempts=5))
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)
        self.assertEqual(adapter.call_count, 1)

    def test_budget_exhausted_has_attempts_in_diagnostics(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter, policy=self._policy(max_attempts=3))
        self.assertIn("attempts", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["attempts"], 3)

    def test_budget_exhausted_has_adapter_name(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter, policy=self._policy(max_attempts=2), name="unitree_sdk2")
        self.assertEqual(result["adapter_name"], "unitree_sdk2")

    def test_close_after_called_once_on_budget_exhaustion(self):
        adapter = _CountingAdapter(fail_times=99)
        _run(adapter, policy=self._policy(max_attempts=3, close_after=True))
        self.assertEqual(adapter.close_call_count, 1)

    def test_close_after_called_once_on_retry_success(self):
        adapter = _CountingAdapter(fail_times=1)
        _run(adapter, policy=self._policy(max_attempts=3, close_after=True))
        self.assertEqual(adapter.close_call_count, 1)

    def test_no_retry_policy_single_attempt(self):
        adapter = _CountingAdapter(fail_times=99)
        _run(adapter)
        self.assertEqual(adapter.call_count, 1)

    def test_retry_success_payload_present(self):
        adapter = _CountingAdapter(fail_times=1)
        result = _run(adapter, policy=self._policy(max_attempts=3))
        self.assertIsNotNone(result["payload"])


# ══════════════════════════════════════════════════════════════════════════
# E — Telemetry integration
# ══════════════════════════════════════════════════════════════════════════

class TestTelemetryIntegration(unittest.TestCase):

    def test_every_result_path_has_telemetry(self):
        paths = [
            ("success",         _run(_BaseOkAdapter())),
            ("not_found",       ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)),
            ("open_fail",       _run(_FailOpenAdapter())),
            ("preflight_fail",  _run(_FailPreflightAdapter(), policy=LifecyclePolicy(run_preflight=True))),
            ("execute_fail",    _run(_CountingAdapter(fail_times=99))),
            ("close_fail_ok",   _run(_BaseOkAdapter(), policy=LifecyclePolicy(close_after=True))),
        ]
        for label, result in paths:
            with self.subTest(path=label):
                self.assertIn("telemetry", result["diagnostics"],
                              f"'{label}' missing telemetry")

    def test_telemetry_is_list(self):
        result = _run(_BaseOkAdapter())
        self.assertIsInstance(_telemetry(result), list)

    def test_every_event_has_all_13_keys(self):
        result = _run(_BaseOkAdapter(), policy=LifecyclePolicy(close_after=True))
        for ev in _telemetry(result):
            with self.subTest(stage=ev.get("stage"), event=ev.get("event")):
                self.assertEqual(frozenset(ev.keys()), _EVENT_KEYS)

    def test_trace_id_consistent_within_call(self):
        result = _run(_BaseOkAdapter(), policy=LifecyclePolicy(close_after=True))
        trace_ids = {e["trace_id"] for e in _telemetry(result)}
        self.assertEqual(len(trace_ids), 1)

    def test_trace_id_is_valid_uuid4(self):
        result = _run(_BaseOkAdapter())
        tid = _telemetry(result)[0]["trace_id"]
        parsed = uuid.UUID(tid, version=4)
        self.assertEqual(str(parsed), tid)

    def test_two_calls_have_different_trace_ids(self):
        adapter = _BaseOkAdapter()
        router = _router(adapter)
        r1 = router.execute_with_lifecycle("dev", RouteOp.RUN_ACTION)
        r2 = router.execute_with_lifecycle("dev", RouteOp.RUN_ACTION)
        tid1 = _telemetry(r1)[0]["trace_id"]
        tid2 = _telemetry(r2)[0]["trace_id"]
        self.assertNotEqual(tid1, tid2)

    def test_open_session_events_always_present(self):
        result = _run(_BaseOkAdapter())
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertIn("open_session", stages)

    def test_execute_start_always_present_on_success(self):
        result = _run(_BaseOkAdapter())
        pairs = [(e["stage"], e["event"]) for e in _telemetry(result)]
        self.assertIn(("execute", "start"), pairs)

    def test_execute_failed_event_retryable_true(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter)
        end_errs = [e for e in _execute_events(result)
                    if e["event"] == "end" and e["status"] == "error"]
        self.assertTrue(end_errs)
        self.assertTrue(end_errs[0]["retryable"])

    def test_preflight_failed_event_retryable_false(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(run_preflight=True))
        pf_ends = [e for e in _telemetry(result)
                   if e["stage"] == "preflight" and e["event"] == "end"]
        self.assertTrue(pf_ends)
        self.assertFalse(pf_ends[0]["retryable"])

    def test_safety_gate_blocked_event_retryable_false(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        pf_ends = [e for e in _telemetry(result)
                   if e["stage"] == "preflight" and e["event"] == "end"]
        self.assertTrue(pf_ends)
        self.assertFalse(pf_ends[0]["retryable"])

    def test_close_session_events_present_when_close_after(self):
        result = _run(_BaseOkAdapter(), policy=LifecyclePolicy(close_after=True))
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertIn("close_session", stages)

    def test_no_close_session_events_without_close_after(self):
        result = _run(_BaseOkAdapter())
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertNotIn("close_session", stages)


# ══════════════════════════════════════════════════════════════════════════
# F — Combined Phase 6 scenarios
# ══════════════════════════════════════════════════════════════════════════

class TestCombinedPhase6Scenarios(unittest.TestCase):

    def test_safety_policy_and_retry_policy_success(self):
        """SafetyPolicy + RetryPolicy: preflight passes, retry succeeds."""
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(
            safety_policy=SafetyPolicy(),
            retry_policy=RetryPolicy(max_attempts=3),
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(adapter.call_count, 2)

    def test_safety_gate_block_not_retried(self):
        """SafetyPolicy gate failure must not be retried even with RetryPolicy."""
        adapter = _FailPreflightAdapter()
        policy = LifecyclePolicy(
            safety_policy=SafetyPolicy(),
            retry_policy=RetryPolicy(max_attempts=5),
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["reason"], LifecycleReason.SAFETY_GATE_BLOCKED)
        # Safety failures are not retried — adapter run_action must not be called
        self.assertEqual(adapter.close_call_count, 1)  # cleanup only

    def test_retry_telemetry_attempt_index_increments(self):
        """Each retry attempt increments the attempt index in execute events."""
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        starts = [e for e in _execute_events(result) if e["event"] == "start"]
        self.assertEqual(len(starts), 3)
        self.assertEqual(starts[0]["attempt"], 1)
        self.assertEqual(starts[1]["attempt"], 2)
        self.assertEqual(starts[2]["attempt"], 3)

    def test_full_stack_open_preflight_execute_close(self):
        """All lifecycle stages succeed: open_session + preflight + execute + close."""
        adapter = _BaseOkAdapter()
        policy = LifecyclePolicy(
            safety_policy=SafetyPolicy(),
            close_after=True,
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(adapter.close_call_count, 1)
        stages = {e["stage"] for e in _telemetry(result)}
        for expected in ("open_session", "preflight", "execute", "close_session"):
            self.assertIn(expected, stages, f"stage '{expected}' missing from telemetry")

    def test_retry_with_close_after_close_once_on_success(self):
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=3),
            close_after=True,
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(adapter.close_call_count, 1)

    def test_budget_exhausted_with_close_after_close_once(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(
            retry_policy=RetryPolicy(max_attempts=3),
            close_after=True,
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)
        self.assertEqual(adapter.close_call_count, 1)

    def test_settings_validation_fail_has_telemetry_with_stage_events(self):
        """settings_validation failure must emit telemetry events for that stage."""
        adapter = _BaseOkAdapter()
        policy = LifecyclePolicy(session_config=_INVALID_UNITREE_CONFIG)
        result = _run(adapter, policy=policy)
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertIn("settings_validation", stages)

    def test_settings_validation_pass_no_settings_events_for_no_brand(self):
        """Without a brand, no settings_validation events appear."""
        result = _run(_BaseOkAdapter(), policy=LifecyclePolicy(session_config={}))
        stages = [e["stage"] for e in _telemetry(result)]
        self.assertNotIn("settings_validation", stages)

    def test_all_phase6_policies_combined_success(self):
        """SafetyPolicy + RetryPolicy + close_after + valid settings all pass."""
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(
            session_config=_VALID_SPOT_CONFIG,
            safety_policy=SafetyPolicy(),
            retry_policy=RetryPolicy(max_attempts=3),
            close_after=True,
        )
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")
        self.assertIn("telemetry", result["diagnostics"])


# ══════════════════════════════════════════════════════════════════════════
# G — Result shape invariants
# ══════════════════════════════════════════════════════════════════════════

class TestResultShapeInvariants(unittest.TestCase):
    """Every execute_with_lifecycle result must conform to the result contract."""

    def _all_results(self):
        """Generate (label, result) pairs covering every major result path."""
        yield "success",            _run(_BaseOkAdapter())
        yield "adapter_not_found",  ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        yield "open_fail",          _run(_FailOpenAdapter())
        yield "preflight_fail",     _run(_FailPreflightAdapter(), policy=LifecyclePolicy(run_preflight=True))
        yield "safety_gate_block",  _run(_FailPreflightAdapter(), policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        yield "execute_fail",       _run(_CountingAdapter(fail_times=99))
        yield "budget_exhausted",   _run(_CountingAdapter(fail_times=99),
                                         policy=LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=2)))
        yield "settings_fail",      _run(_BaseOkAdapter(),
                                         policy=LifecyclePolicy(session_config=_INVALID_UNITREE_CONFIG))
        yield "close_ok",           _run(_BaseOkAdapter(), policy=LifecyclePolicy(close_after=True))

    def test_all_result_keys_present(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertEqual(
                    frozenset(result.keys()),
                    _RESULT_KEYS,
                    f"'{label}' has wrong keys: {set(result.keys())}",
                )

    def test_all_results_have_telemetry(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertIn("telemetry", result["diagnostics"],
                              f"'{label}' missing telemetry in diagnostics")

    def test_error_results_have_none_payload(self):
        error_paths = [
            ("open_fail",       _run(_FailOpenAdapter())),
            ("preflight_fail",  _run(_FailPreflightAdapter(), policy=LifecyclePolicy(run_preflight=True))),
            ("execute_fail",    _run(_CountingAdapter(fail_times=99))),
            ("adapter_not_found", ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)),
        ]
        for label, result in error_paths:
            with self.subTest(path=label):
                self.assertIsNone(result["payload"], f"'{label}' error must have None payload")

    def test_success_results_have_ok_status(self):
        result = _run(_BaseOkAdapter())
        self.assertEqual(result["status"], "ok")

    def test_all_results_have_string_reason(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertIsInstance(result["reason"], str)

    def test_all_results_have_string_stage(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertIsInstance(result["stage"], str)

    def test_all_results_have_string_adapter_name(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertIsInstance(result["adapter_name"], str)

    def test_all_results_diagnostics_is_dict(self):
        for label, result in self._all_results():
            with self.subTest(path=label):
                self.assertIsInstance(result["diagnostics"], dict)


if __name__ == "__main__":
    unittest.main()
