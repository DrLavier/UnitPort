#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 STAGE-07: Phase 6 test matrix.

Provides repeatable evidence that safety and observability hardening works
and that prior phases remain stable.

Sections:
  A. Phase 6 deliverable gate  (all new modules and symbols present)
  B. Preflight gate decision matrix  (SafetyPolicy × adapter outcome)
  C. Telemetry schema stability  (13-key contract across all event types)
  D. Reason code + retryable mapping matrix  (all 27 codes classified)
  E. Retry boundary behavior matrix  (parametric max_attempts × fail_times)
  F. End-to-end lifecycle per brand  (unitree / bostiondynamics / xiaomi)
  G. Failure paths at each boundary  (deterministic reason / stage / diagnostics)
  H. Phase 3/4/5 regression gate  (legacy caller behavior unchanged)
"""

import sys
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import (                             # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
    RetryPolicy, SafetyPolicy,
)
from system.service.service_router import RouteOp, ServiceRouter  # noqa: E402
from system.service.service_registry import ServiceRegistry        # noqa: E402
from system.service.adapters.base_adapter import BaseAdapter       # noqa: E402
from system.service.reason_codes import (                          # noqa: E402
    ALL_REASON_CODES, CANONICAL_REASONS, UNITREE_REASONS,
    SPOT_REASONS, CYBERDOG_REASONS, REASON_MAP,
)
from system.service.telemetry import (                             # noqa: E402
    TelemetryEvent, TelemetryCollector, emit_event,
    new_trace_id, _get_retryable,
)


# ── Valid brand settings configs (all fields pass Phase 5 validation) ─────

_VALID_UNITREE = {
    "brand": "unitree", "robot_type": "go2",
    "connection_mode": "simulation", "timeout": 30,
    "stop_policy": "immediate", "safety_enabled": True,
    "dds_domain_id": 0, "network_interface": "eth0",
    "control_level": "high", "mode_switch_policy": "immediate",
}

_VALID_SPOT = {
    "brand": "bostiondynamics", "robot_type": "spot",
    "connection_mode": "simulation", "timeout": 30,
    "stop_policy": "immediate", "safety_enabled": True,
    "hostname": "10.0.0.1", "auth_token": "tok",
    "safety_channel": "default", "power_policy": "safe_power_off",
}

_VALID_XIAOMI = {
    "brand": "xiaomi", "robot_type": "cyberdog2",
    "connection_mode": "simulation", "timeout": 30,
    "stop_policy": "immediate", "safety_enabled": True,
    "ros_domain_id": 0, "rmw_impl": "rmw_fastrtps_cpp",
    "action_endpoints": ["/cyberdog/motion"], "timestamp_policy": "strict",
}

_EVENT_KEYS = frozenset({
    "trace_id", "adapter_name", "brand", "operation", "stage",
    "event", "timestamp", "duration_ms", "status", "reason",
    "retryable", "attempt", "diagnostics",
})

_VALID_EVENT_VALUES = {
    "event":  {"start", "end"},
    "status": {"ok", "error", "skipped", "pending"},
}


# ── Mock adapters ─────────────────────────────────────────────────────────

class _OkAdapter(BaseAdapter):
    def __init__(self): self.close_count = 0
    def connect(self, **kw): return True
    def run_action(self, a, **p): return True
    def stop(self): pass
    def get_sensor_data(self): return {"x": 1}
    def health(self): return {"ok": True}
    def open_session(self, config=None):
        return LifecycleResult.ok("open_session").to_dict()
    def preflight(self, context=None):
        return LifecycleResult.ok("preflight").to_dict()
    def close_session(self):
        self.close_count += 1
        return LifecycleResult.ok("close_session").to_dict()


class _FailPreflightAdapter(_OkAdapter):
    def preflight(self, context=None):
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "gate blocked"},
        ).to_dict()


class _FailOpenAdapter(_OkAdapter):
    def open_session(self, config=None):
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "open failed"},
        ).to_dict()


class _FailCloseAdapter(_OkAdapter):
    def close_session(self):
        self.close_count += 1
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "close failed"},
        ).to_dict()


class _CountingAdapter(_OkAdapter):
    def __init__(self, fail_times=0):
        super().__init__()
        self._fail = fail_times
        self.call_count = 0
    def run_action(self, a, **p):
        self.call_count += 1
        if self.call_count <= self._fail:
            raise RuntimeError("transient")
        return True


# ── Helpers ───────────────────────────────────────────────────────────────

def _router(adapter, name="dev"):
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


def _run(adapter, op=RouteOp.RUN_ACTION, policy=None, name="dev"):
    return _router(adapter, name).execute_with_lifecycle(name, op, policy=policy)


def _tel(result): return result.get("diagnostics", {}).get("telemetry", [])
def _exe_events(result): return [e for e in _tel(result) if e["stage"] == "execute"]


# ══════════════════════════════════════════════════════════════════════════
# A — Phase 6 deliverable gate
# ══════════════════════════════════════════════════════════════════════════

class TestPhase6DeliverableGate(unittest.TestCase):
    """All new Phase 6 symbols must be importable with correct types."""

    # ── lifecycle.py ──────────────────────────────────────────────────────

    def test_safety_policy_class_importable(self):
        self.assertTrue(callable(SafetyPolicy))

    def test_retry_policy_class_importable(self):
        self.assertTrue(callable(RetryPolicy))

    def test_safety_gate_blocked_constant_exists(self):
        self.assertEqual(LifecycleReason.SAFETY_GATE_BLOCKED, "safety_gate_blocked")

    def test_retry_budget_exhausted_constant_exists(self):
        self.assertEqual(LifecycleReason.RETRY_BUDGET_EXHAUSTED, "retry_budget_exhausted")

    def test_lifecycle_policy_has_safety_policy_field(self):
        self.assertIsNone(LifecyclePolicy().safety_policy)

    def test_lifecycle_policy_has_retry_policy_field(self):
        self.assertIsNone(LifecyclePolicy().retry_policy)

    # ── telemetry.py ──────────────────────────────────────────────────────

    def test_telemetry_event_class_importable(self):
        self.assertTrue(callable(TelemetryEvent))

    def test_telemetry_collector_class_importable(self):
        self.assertTrue(callable(TelemetryCollector))

    def test_emit_event_callable(self):
        self.assertTrue(callable(emit_event))

    def test_new_trace_id_callable(self):
        self.assertTrue(callable(new_trace_id))

    def test_get_retryable_callable(self):
        self.assertTrue(callable(_get_retryable))

    # ── reason_codes.py ───────────────────────────────────────────────────

    def test_canonical_reasons_has_8_entries(self):
        self.assertEqual(len(CANONICAL_REASONS), 8)

    def test_all_reason_codes_has_27_entries(self):
        self.assertEqual(len(ALL_REASON_CODES), 27)

    def test_every_reason_map_entry_has_retryable(self):
        missing = [c for c in REASON_MAP if "retryable" not in REASON_MAP[c]]
        self.assertFalse(missing, f"Missing retryable: {missing}")

    def test_safety_gate_blocked_in_reason_map(self):
        self.assertIn("safety_gate_blocked", REASON_MAP)

    def test_retry_budget_exhausted_in_reason_map(self):
        self.assertIn("retry_budget_exhausted", REASON_MAP)

    # ── service_router.py integration ─────────────────────────────────────

    def test_execute_with_lifecycle_returns_telemetry(self):
        result = _run(_OkAdapter())
        self.assertIn("telemetry", result["diagnostics"])

    def test_retry_policy_wired_in_router(self):
        adapter = _CountingAdapter(fail_times=1)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(adapter.call_count, 2)

    def test_safety_policy_wired_in_router(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(result["reason"], LifecycleReason.SAFETY_GATE_BLOCKED)


# ══════════════════════════════════════════════════════════════════════════
# B — Preflight gate decision matrix
# ══════════════════════════════════════════════════════════════════════════

class TestPreflightGateMatrix(unittest.TestCase):
    """
    Parametric matrix over SafetyPolicy settings × adapter preflight outcome.

    Each row: (description, adapter, policy, expected_status, expected_reason)
    """

    def _cases(self):
        ok   = _OkAdapter
        fail = _FailPreflightAdapter

        # SafetyPolicy paths
        yield ("sp_req=T,allow=F,adapter=ok",
               ok(),   LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=True,  allow_execute_without_preflight=False)), "ok",    "")
        yield ("sp_req=T,allow=F,adapter=fail",
               fail(), LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=True,  allow_execute_without_preflight=False)), "error", LifecycleReason.SAFETY_GATE_BLOCKED)
        yield ("sp_req=T,allow=T,adapter=ok",
               ok(),   LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=True,  allow_execute_without_preflight=True)),  "ok",    "")
        yield ("sp_req=T,allow=T,adapter=fail",
               fail(), LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=True,  allow_execute_without_preflight=True)),  "ok",    "")
        yield ("sp_req=F,allow=F,adapter=ok",
               ok(),   LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=False, allow_execute_without_preflight=False)), "ok",    "")
        yield ("sp_req=F,allow=F,adapter=fail",
               fail(), LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=False, allow_execute_without_preflight=False)), "ok",    "")
        # Legacy run_preflight paths
        yield ("legacy_preflight=T,adapter=ok",
               ok(),   LifecyclePolicy(run_preflight=True),  "ok",    "")
        yield ("legacy_preflight=T,adapter=fail",
               fail(), LifecyclePolicy(run_preflight=True),  "error", LifecycleReason.PREFLIGHT_FAILED)
        yield ("legacy_preflight=F,adapter=ok",
               ok(),   LifecyclePolicy(run_preflight=False), "ok",    "")
        yield ("legacy_preflight=F,adapter=fail",
               fail(), LifecyclePolicy(run_preflight=False), "ok",    "")

    def test_gate_matrix_status(self):
        for desc, adapter, policy, exp_status, _ in self._cases():
            with self.subTest(case=desc):
                result = _run(adapter, policy=policy)
                self.assertEqual(result["status"], exp_status, desc)

    def test_gate_matrix_reason(self):
        for desc, adapter, policy, _, exp_reason in self._cases():
            with self.subTest(case=desc):
                result = _run(adapter, policy=policy)
                self.assertEqual(result["reason"], exp_reason, desc)

    def test_safety_gate_blocked_never_equals_preflight_failed(self):
        self.assertNotEqual(
            LifecycleReason.SAFETY_GATE_BLOCKED,
            LifecycleReason.PREFLIGHT_FAILED,
        )

    def test_safety_gate_fail_stage_is_preflight(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(result["stage"], "preflight")

    def test_legacy_preflight_fail_stage_is_preflight(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(result["stage"], "preflight")

    def test_safety_policy_fail_close_session_cleanup(self):
        adapter = _FailPreflightAdapter()
        _run(adapter, policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(adapter.close_count, 1)

    def test_legacy_preflight_fail_close_session_cleanup(self):
        adapter = _FailPreflightAdapter()
        _run(adapter, policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(adapter.close_count, 1)


# ══════════════════════════════════════════════════════════════════════════
# C — Telemetry schema stability
# ══════════════════════════════════════════════════════════════════════════

class TestTelemetrySchemaStability(unittest.TestCase):

    def _full_result(self):
        """Run a lifecycle that exercises all active stages."""
        adapter = _OkAdapter()
        policy = LifecyclePolicy(
            safety_policy=SafetyPolicy(),
            close_after=True,
        )
        return _run(adapter, policy=policy)

    def test_every_event_has_exactly_13_keys(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertEqual(frozenset(ev.keys()), _EVENT_KEYS)

    def test_event_field_values_are_valid(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"]):
                self.assertIn(ev["event"], _VALID_EVENT_VALUES["event"])

    def test_status_field_values_are_valid(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertIn(ev["status"], _VALID_EVENT_VALUES["status"])

    def test_start_events_have_none_duration_ms(self):
        result = self._full_result()
        starts = [e for e in _tel(result) if e["event"] == "start"]
        for ev in starts:
            with self.subTest(stage=ev["stage"]):
                self.assertIsNone(ev["duration_ms"])

    def test_end_events_have_float_duration_ms(self):
        result = self._full_result()
        ends = [e for e in _tel(result) if e["event"] == "end"]
        for ev in ends:
            with self.subTest(stage=ev["stage"]):
                self.assertIsInstance(ev["duration_ms"], float)
                self.assertGreaterEqual(ev["duration_ms"], 0.0)

    def test_timestamp_is_float(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertIsInstance(ev["timestamp"], float)

    def test_retryable_is_bool(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertIsInstance(ev["retryable"], bool)

    def test_attempt_is_positive_int(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertIsInstance(ev["attempt"], int)
                self.assertGreaterEqual(ev["attempt"], 1)

    def test_diagnostics_is_dict(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"], event=ev["event"]):
                self.assertIsInstance(ev["diagnostics"], dict)

    def test_trace_id_is_valid_uuid4(self):
        result = self._full_result()
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"]):
                parsed = uuid.UUID(ev["trace_id"], version=4)
                self.assertEqual(str(parsed), ev["trace_id"])

    def test_all_events_share_one_trace_id(self):
        result = self._full_result()
        ids = {e["trace_id"] for e in _tel(result)}
        self.assertEqual(len(ids), 1)

    def test_adapter_name_in_all_events(self):
        adapter = _OkAdapter()
        result = _run(adapter, name="spot_sdk")
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"]):
                self.assertEqual(ev["adapter_name"], "spot_sdk")

    def test_open_session_both_events_present(self):
        result = self._full_result()
        pairs = {(e["stage"], e["event"]) for e in _tel(result)}
        self.assertIn(("open_session", "start"), pairs)
        self.assertIn(("open_session", "end"), pairs)

    def test_execute_both_events_present(self):
        result = self._full_result()
        pairs = {(e["stage"], e["event"]) for e in _tel(result)}
        self.assertIn(("execute", "start"), pairs)
        self.assertIn(("execute", "end"), pairs)

    def test_credentials_not_in_telemetry_diagnostics(self):
        # Run with a config that contains auth_token
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_SPOT)
        result = _run(adapter, policy=policy)
        for ev in _tel(result):
            with self.subTest(stage=ev["stage"]):
                self.assertNotIn("auth_token", ev["diagnostics"])
                self.assertNotIn("hostname",   ev["diagnostics"])


# ══════════════════════════════════════════════════════════════════════════
# D — Reason code + retryable mapping matrix
# ══════════════════════════════════════════════════════════════════════════

class TestReasonCodeMatrix(unittest.TestCase):

    def test_total_codes_is_27(self):
        self.assertEqual(len(ALL_REASON_CODES), 27)

    def test_canonical_count_is_8(self):
        self.assertEqual(len(CANONICAL_REASONS), 8)

    def test_adapter_tier_counts_unchanged(self):
        self.assertEqual(len(UNITREE_REASONS), 3)
        self.assertEqual(len(SPOT_REASONS), 8)
        self.assertEqual(len(CYBERDOG_REASONS), 8)

    def test_reason_map_keys_equal_all_reason_codes(self):
        self.assertEqual(frozenset(REASON_MAP.keys()), ALL_REASON_CODES)

    def test_every_entry_has_four_keys(self):
        expected = {"adapter", "stage", "tier", "retryable"}
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertEqual(set(meta.keys()), expected)

    def test_retryable_codes_are_exactly_three(self):
        retryable = [c for c, m in REASON_MAP.items() if m["retryable"]]
        self.assertEqual(len(retryable), 3)

    def test_retryable_codes_are_the_correct_three(self):
        retryable = frozenset(c for c, m in REASON_MAP.items() if m["retryable"])
        expected = frozenset({
            LifecycleReason.EXECUTE_FAILED,
            "spot_command_timeout",
            "cyberdog_action_failed",
        })
        self.assertEqual(retryable, expected)

    def test_non_retryable_codes_count_is_24(self):
        non_r = [c for c, m in REASON_MAP.items() if not m["retryable"]]
        self.assertEqual(len(non_r), 24)

    def test_safety_gate_blocked_non_retryable(self):
        self.assertFalse(REASON_MAP["safety_gate_blocked"]["retryable"])

    def test_retry_budget_exhausted_non_retryable(self):
        self.assertFalse(REASON_MAP["retry_budget_exhausted"]["retryable"])

    def test_execute_failed_retryable(self):
        self.assertTrue(REASON_MAP["execute_failed"]["retryable"])

    def test_tier_values_are_canonical_or_adapter(self):
        valid = {"canonical", "adapter"}
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIn(meta["tier"], valid)

    def test_get_retryable_matches_reason_map(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertEqual(
                    _get_retryable(code), meta["retryable"],
                    f"_get_retryable({code!r}) mismatch",
                )

    def test_tiers_are_disjoint(self):
        tiers = [UNITREE_REASONS, SPOT_REASONS, CYBERDOG_REASONS]
        for i, a in enumerate(tiers):
            for j, b in enumerate(tiers):
                if i != j:
                    self.assertFalse(a & b)


# ══════════════════════════════════════════════════════════════════════════
# E — Retry boundary behavior matrix
# ══════════════════════════════════════════════════════════════════════════

class TestRetryBoundaryMatrix(unittest.TestCase):
    """
    Parametric matrix: (max_attempts, fail_times, op) →
    (expected_status, expected_reason, expected_call_count)
    """

    def _cases(self):
        RUN = RouteOp.RUN_ACTION
        STP = RouteOp.STOP
        EF  = LifecycleReason.EXECUTE_FAILED
        RBE = LifecycleReason.RETRY_BUDGET_EXHAUSTED

        # (label, max_attempts, fail_times, op, exp_status, exp_reason, exp_calls)
        yield ("no_policy_success",   None, 0,  RUN, "ok",    "",   1)
        yield ("no_policy_fail",      None, 99, RUN, "error", EF,   1)
        yield ("max1_success",        1,    0,  RUN, "ok",    "",   1)
        yield ("max1_fail",           1,    99, RUN, "error", EF,   1)
        yield ("max2_success_1st",    2,    0,  RUN, "ok",    "",   1)
        yield ("max2_success_2nd",    2,    1,  RUN, "ok",    "",   2)
        yield ("max2_exhausted",      2,    99, RUN, "error", RBE,  2)
        yield ("max3_success_3rd",    3,    2,  RUN, "ok",    "",   3)
        yield ("max3_exhausted",      3,    99, RUN, "error", RBE,  3)
        yield ("stop_not_retried",    3,    99, STP, "error", EF,   1)  # STOP excluded

    def _run_case(self, max_attempts, fail_times, op):
        class _StopFail(_CountingAdapter):
            def stop(self):
                self.call_count += 1
                raise RuntimeError("stop fails")

        if op == RouteOp.STOP:
            adapter = _StopFail(fail_times=0)
        else:
            adapter = _CountingAdapter(fail_times=fail_times)
        if max_attempts is None:
            policy = None
        else:
            policy = LifecyclePolicy(
                retry_policy=RetryPolicy(max_attempts=max_attempts),
            )
        result = _run(adapter, op=op, policy=policy)
        return result, adapter.call_count

    def test_matrix_status(self):
        for label, ma, ft, op, exp_status, _, _ in self._cases():
            with self.subTest(case=label):
                result, _ = self._run_case(ma, ft, op)
                self.assertEqual(result["status"], exp_status, label)

    def test_matrix_reason(self):
        for label, ma, ft, op, _, exp_reason, _ in self._cases():
            with self.subTest(case=label):
                result, _ = self._run_case(ma, ft, op)
                self.assertEqual(result["reason"], exp_reason, label)

    def test_matrix_call_count(self):
        for label, ma, ft, op, _, _, exp_calls in self._cases():
            with self.subTest(case=label):
                _, calls = self._run_case(ma, ft, op)
                self.assertEqual(calls, exp_calls, label)

    def test_budget_exhausted_has_attempts_in_diagnostics(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        self.assertIn("attempts", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["attempts"], 3)

    def test_retry_telemetry_start_count_equals_attempts(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        starts = [e for e in _exe_events(result) if e["event"] == "start"]
        self.assertEqual(len(starts), 3)

    def test_retry_telemetry_attempt_indices(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        starts = [e for e in _exe_events(result) if e["event"] == "start"]
        for i, ev in enumerate(starts, start=1):
            with self.subTest(attempt=i):
                self.assertEqual(ev["attempt"], i)

    def test_execute_failed_events_have_retryable_true(self):
        adapter = _CountingAdapter(fail_times=99)
        result = _run(adapter)
        end_errs = [e for e in _exe_events(result)
                    if e["event"] == "end" and e["status"] == "error"]
        self.assertTrue(end_errs)
        self.assertTrue(end_errs[0]["retryable"])


# ══════════════════════════════════════════════════════════════════════════
# F — End-to-end lifecycle per brand
# ══════════════════════════════════════════════════════════════════════════

class TestPerBrandLifecycle(unittest.TestCase):
    """Settings validation + full lifecycle for each of the three brands."""

    # ── Unitree ───────────────────────────────────────────────────────────

    def test_unitree_valid_config_proceeds(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_UNITREE)
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")

    def test_unitree_invalid_config_settings_fail(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config={"brand": "unitree"})
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "settings_validation")
        self.assertEqual(result["diagnostics"]["brand"], "unitree")

    def test_unitree_valid_config_telemetry_present(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_UNITREE)
        result = _run(adapter, policy=policy)
        self.assertIn("telemetry", result["diagnostics"])

    def test_unitree_brand_in_telemetry_events(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_UNITREE)
        result = _run(adapter, policy=policy)
        brands = {e["brand"] for e in _tel(result)}
        self.assertEqual(brands, {"unitree"})

    # ── BostonDynamics (Spot) ─────────────────────────────────────────────

    def test_spot_valid_config_proceeds(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_SPOT)
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")

    def test_spot_invalid_config_settings_fail(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config={"brand": "bostiondynamics"})
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["diagnostics"]["brand"], "bostiondynamics")

    def test_spot_valid_config_telemetry_present(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_SPOT)
        result = _run(adapter, policy=policy)
        self.assertIn("telemetry", result["diagnostics"])

    def test_spot_settings_telemetry_events_present(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_SPOT)
        result = _run(adapter, policy=policy)
        stages = [e["stage"] for e in _tel(result)]
        self.assertIn("settings_validation", stages)

    # ── Xiaomi (CyberDog) ─────────────────────────────────────────────────

    def test_xiaomi_valid_config_proceeds(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_XIAOMI)
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "ok")

    def test_xiaomi_invalid_config_settings_fail(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config={"brand": "xiaomi"})
        result = _run(adapter, policy=policy)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["diagnostics"]["brand"], "xiaomi")

    def test_xiaomi_valid_config_telemetry_present(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_XIAOMI)
        result = _run(adapter, policy=policy)
        self.assertIn("telemetry", result["diagnostics"])

    def test_xiaomi_brand_in_telemetry_events(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config=_VALID_XIAOMI)
        result = _run(adapter, policy=policy)
        brands = {e["brand"] for e in _tel(result)}
        self.assertEqual(brands, {"xiaomi"})


# ══════════════════════════════════════════════════════════════════════════
# G — Failure paths at each boundary
# ══════════════════════════════════════════════════════════════════════════

class TestBoundaryFailurePaths(unittest.TestCase):
    """
    Each lifecycle boundary failure must produce deterministic
    (reason, stage, diagnostics_keys, payload=None, adapter_name).
    """

    def _assert_boundary(self, result, exp_reason, exp_stage):
        self.assertEqual(result["status"],  "error")
        self.assertEqual(result["reason"],  exp_reason)
        self.assertEqual(result["stage"],   exp_stage)
        self.assertIsNone(result["payload"])
        self.assertIn("adapter_name", result)
        self.assertIn("telemetry", result["diagnostics"])

    def test_settings_validation_boundary(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config={"brand": "unitree"})
        result = _run(adapter, policy=policy)
        self._assert_boundary(result, LifecycleReason.PREFLIGHT_FAILED, "settings_validation")
        for key in ("brand", "message", "context", "validation"):
            self.assertIn(key, result["diagnostics"], f"Missing '{key}' in diag")

    def test_acquire_boundary(self):
        result = ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self._assert_boundary(result, LifecycleReason.ADAPTER_NOT_FOUND, "acquire")

    def test_open_session_boundary(self):
        result = _run(_FailOpenAdapter())
        self._assert_boundary(result, LifecycleReason.SESSION_OPEN_FAILED, "open_session")

    def test_preflight_boundary_legacy(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(run_preflight=True))
        self._assert_boundary(result, LifecycleReason.PREFLIGHT_FAILED, "preflight")

    def test_preflight_boundary_safety_gate(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(safety_policy=SafetyPolicy()))
        self._assert_boundary(result, LifecycleReason.SAFETY_GATE_BLOCKED, "preflight")

    def test_execute_boundary_no_retry(self):
        result = _run(_CountingAdapter(fail_times=99))
        self._assert_boundary(result, LifecycleReason.EXECUTE_FAILED, "execute")
        self.assertIn("message", result["diagnostics"])
        self.assertIn("op",      result["diagnostics"])

    def test_execute_boundary_budget_exhausted(self):
        adapter = _CountingAdapter(fail_times=99)
        policy = LifecyclePolicy(retry_policy=RetryPolicy(max_attempts=3))
        result = _run(adapter, policy=policy)
        self._assert_boundary(result, LifecycleReason.RETRY_BUDGET_EXHAUSTED, "execute")
        self.assertIn("attempts", result["diagnostics"])

    def test_close_session_failure_does_not_mask_payload(self):
        """close_session failure must surface in diagnostics but NOT mask payload."""
        adapter = _FailCloseAdapter()
        policy = LifecyclePolicy(close_after=True)
        result = _run(adapter, policy=policy)
        # Status must still be ok — close failure doesn't override execute success
        self.assertEqual(result["status"], "ok")
        self.assertIsNotNone(result["payload"])
        self.assertIn("close_session_failed", result["diagnostics"])
        self.assertTrue(result["diagnostics"]["close_session_failed"])


# ══════════════════════════════════════════════════════════════════════════
# H — Phase 3/4/5 regression gate
# ══════════════════════════════════════════════════════════════════════════

class TestPriorPhaseRegression(unittest.TestCase):
    """Legacy caller contract: no Phase 6 policies → Phase 3/4/5 behaviour."""

    def test_default_policy_no_preflight(self):
        cap = []
        class _Cap(_OkAdapter):
            def preflight(self, context=None):
                cap.append(True)
                return LifecycleResult.ok("preflight").to_dict()
        _run(_Cap())
        self.assertFalse(cap, "preflight must not run with default policy")

    def test_default_policy_no_close(self):
        adapter = _OkAdapter()
        _run(adapter)
        self.assertEqual(adapter.close_count, 0)

    def test_default_policy_success_result_shape(self):
        result = _run(_OkAdapter())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reason"], "")
        self.assertEqual(result["stage"],  "execute")
        self.assertIsNotNone(result["payload"])
        self.assertIn("adapter_name", result)

    def test_default_policy_execute_fail_reason(self):
        result = _run(_CountingAdapter(fail_times=99))
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)
        self.assertNotEqual(result["reason"], LifecycleReason.RETRY_BUDGET_EXHAUSTED)

    def test_default_policy_single_attempt_only(self):
        adapter = _CountingAdapter(fail_times=99)
        _run(adapter)
        self.assertEqual(adapter.call_count, 1)

    def test_phase4_adapter_name_in_all_paths(self):
        paths = [
            _run(_OkAdapter(),                  name="unitree_sdk2"),
            _run(_FailOpenAdapter(),             name="spot_sdk"),
            _run(_CountingAdapter(fail_times=99),name="cyberdog_sdk"),
            ServiceRouter().execute_with_lifecycle("ghost", RouteOp.RUN_ACTION),
        ]
        names = ["unitree_sdk2", "spot_sdk", "cyberdog_sdk", "ghost"]
        for result, name in zip(paths, names):
            with self.subTest(name=name):
                self.assertEqual(result["adapter_name"], name)

    def test_phase5_settings_validation_contract_preserved(self):
        adapter = _OkAdapter()
        policy = LifecyclePolicy(session_config={"brand": "unitree"})
        result = _run(adapter, policy=policy)
        diag = result["diagnostics"]
        for key in ("brand", "message", "context", "validation"):
            with self.subTest(key=key):
                self.assertIn(key, diag)

    def test_phase5_no_brand_no_validation(self):
        result = _run(_OkAdapter(), policy=LifecyclePolicy(session_config={}))
        self.assertEqual(result["status"], "ok")

    def test_phase3_lifecycle_result_keys_complete(self):
        result = _run(_OkAdapter())
        required = {"status", "reason", "stage", "payload", "diagnostics", "adapter_name"}
        self.assertTrue(required.issubset(result.keys()))

    def test_legacy_close_after_still_closes(self):
        adapter = _OkAdapter()
        _run(adapter, policy=LifecyclePolicy(close_after=True))
        self.assertEqual(adapter.close_count, 1)

    def test_legacy_preflight_still_runs(self):
        result = _run(_FailPreflightAdapter(),
                      policy=LifecyclePolicy(run_preflight=True))
        self.assertEqual(result["reason"], LifecycleReason.PREFLIGHT_FAILED)


if __name__ == "__main__":
    unittest.main()
