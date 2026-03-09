#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests — Circle 2 Step 2.2: HeartbeatPolicy schema and validation.

Coverage
--------
- HeartbeatFailPolicy: constant values, valid_values() set
- HeartbeatSafetyMode: constant values, valid_values() set
- HeartbeatPolicy.default(): produces valid policy with expected defaults
- HeartbeatPolicy.from_dict(): round-trip, missing keys fall back to defaults,
  bad type inputs are coerced safely
- HeartbeatPolicy.to_dict(): all 5 fields present, correct types
- HeartbeatPolicy.validate(): no errors for valid policy
- HeartbeatPolicy.validate(): error per out-of-range / invalid field
- HeartbeatPolicy.validate(): multiple errors when multiple fields invalid
- HeartbeatPolicy.is_valid property: True / False reflection
- validate_heartbeat_policy(): pass-through correctness
- validate_heartbeat_policy(): propagates trace_id into diagnostics
- Range boundary conditions: min/max values pass, min-1/max+1 fail
- Diagnostic code format: all codes match "heartbeat_policy.<field>_<reason>"
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.heartbeat_policy import (   # noqa: E402
    HeartbeatFailPolicy,
    HeartbeatPolicy,
    HeartbeatSafetyMode,
    MAX_OVERRIDE_MAX,
    MAX_OVERRIDE_MIN,
    TICK_MS_MAX,
    TICK_MS_MIN,
    TIMEOUT_MS_MAX,
    TIMEOUT_MS_MIN,
    validate_heartbeat_policy,
)


# ---------------------------------------------------------------------------
# HeartbeatFailPolicy constants
# ---------------------------------------------------------------------------

class TestHeartbeatFailPolicyConstants(unittest.TestCase):
    def test_stop_behavior_value(self):
        self.assertEqual(HeartbeatFailPolicy.STOP_BEHAVIOR, "stop_behavior")

    def test_log_and_continue_value(self):
        self.assertEqual(HeartbeatFailPolicy.LOG_AND_CONTINUE, "log_and_continue")

    def test_abort_mission_value(self):
        self.assertEqual(HeartbeatFailPolicy.ABORT_MISSION, "abort_mission")

    def test_valid_values_contains_all_three(self):
        v = HeartbeatFailPolicy.valid_values()
        self.assertIn("stop_behavior", v)
        self.assertIn("log_and_continue", v)
        self.assertIn("abort_mission", v)
        self.assertEqual(len(v), 3)


# ---------------------------------------------------------------------------
# HeartbeatSafetyMode constants
# ---------------------------------------------------------------------------

class TestHeartbeatSafetyModeConstants(unittest.TestCase):
    def test_strict_value(self):
        self.assertEqual(HeartbeatSafetyMode.STRICT, "strict")

    def test_relaxed_value(self):
        self.assertEqual(HeartbeatSafetyMode.RELAXED, "relaxed")

    def test_valid_values_contains_both(self):
        v = HeartbeatSafetyMode.valid_values()
        self.assertIn("strict", v)
        self.assertIn("relaxed", v)
        self.assertEqual(len(v), 2)


# ---------------------------------------------------------------------------
# HeartbeatPolicy default
# ---------------------------------------------------------------------------

class TestHeartbeatPolicyDefault(unittest.TestCase):
    def setUp(self):
        self.p = HeartbeatPolicy.default()

    def test_default_is_valid(self):
        self.assertTrue(self.p.is_valid)

    def test_default_tick_ms(self):
        self.assertEqual(self.p.tick_ms, 100)

    def test_default_timeout_ms(self):
        self.assertEqual(self.p.timeout_ms, 5_000)

    def test_default_fail_policy(self):
        self.assertEqual(self.p.fail_policy, "stop_behavior")

    def test_default_max_override(self):
        self.assertAlmostEqual(self.p.max_override, 0.2)

    def test_default_safety_mode(self):
        self.assertEqual(self.p.safety_mode, "strict")

    def test_default_validate_returns_empty(self):
        self.assertEqual(self.p.validate(), [])


# ---------------------------------------------------------------------------
# HeartbeatPolicy.from_dict / to_dict round-trip
# ---------------------------------------------------------------------------

class TestHeartbeatPolicyFromDict(unittest.TestCase):
    def _valid_dict(self):
        return {
            "tick_ms": 200,
            "timeout_ms": 8_000,
            "fail_policy": "log_and_continue",
            "max_override": 0.5,
            "safety_mode": "relaxed",
        }

    def test_roundtrip_all_fields(self):
        d = self._valid_dict()
        p = HeartbeatPolicy.from_dict(d)
        self.assertEqual(p.to_dict(), d)

    def test_missing_tick_ms_falls_back_to_default(self):
        p = HeartbeatPolicy.from_dict({})
        self.assertEqual(p.tick_ms, 100)

    def test_missing_all_keys_produces_valid_default(self):
        p = HeartbeatPolicy.from_dict({})
        self.assertTrue(p.is_valid)

    def test_bad_type_tick_ms_falls_back(self):
        """Non-integer tick_ms falls back to 100 (default)."""
        p = HeartbeatPolicy.from_dict({"tick_ms": "not_a_number"})
        self.assertEqual(p.tick_ms, 100)

    def test_bad_type_max_override_falls_back(self):
        p = HeartbeatPolicy.from_dict({"max_override": "bad"})
        self.assertAlmostEqual(p.max_override, 0.2)


class TestHeartbeatPolicyToDict(unittest.TestCase):
    def test_to_dict_has_all_five_fields(self):
        d = HeartbeatPolicy.default().to_dict()
        for key in ("tick_ms", "timeout_ms", "fail_policy", "max_override", "safety_mode"):
            self.assertIn(key, d)

    def test_to_dict_tick_ms_is_int(self):
        d = HeartbeatPolicy.default().to_dict()
        self.assertIsInstance(d["tick_ms"], int)

    def test_to_dict_timeout_ms_is_int(self):
        d = HeartbeatPolicy.default().to_dict()
        self.assertIsInstance(d["timeout_ms"], int)

    def test_to_dict_max_override_is_float(self):
        d = HeartbeatPolicy.default().to_dict()
        self.assertIsInstance(d["max_override"], float)

    def test_to_dict_fail_policy_is_str(self):
        d = HeartbeatPolicy.default().to_dict()
        self.assertIsInstance(d["fail_policy"], str)

    def test_to_dict_safety_mode_is_str(self):
        d = HeartbeatPolicy.default().to_dict()
        self.assertIsInstance(d["safety_mode"], str)


# ---------------------------------------------------------------------------
# HeartbeatPolicy.validate() — per-field errors
# ---------------------------------------------------------------------------

class TestHeartbeatPolicyValidation(unittest.TestCase):
    def test_valid_default_no_errors(self):
        p = HeartbeatPolicy.default()
        self.assertEqual(p.validate(), [])

    # tick_ms
    def test_tick_ms_too_low(self):
        p = HeartbeatPolicy(tick_ms=TICK_MS_MIN - 1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.tick_ms_out_of_range", codes)

    def test_tick_ms_too_high(self):
        p = HeartbeatPolicy(tick_ms=TICK_MS_MAX + 1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.tick_ms_out_of_range", codes)

    def test_tick_ms_at_min_ok(self):
        p = HeartbeatPolicy(tick_ms=TICK_MS_MIN)
        self.assertNotIn("heartbeat_policy.tick_ms_out_of_range",
                         [d.code for d in p.validate()])

    def test_tick_ms_at_max_ok(self):
        p = HeartbeatPolicy(tick_ms=TICK_MS_MAX)
        self.assertNotIn("heartbeat_policy.tick_ms_out_of_range",
                         [d.code for d in p.validate()])

    # timeout_ms
    def test_timeout_ms_too_low(self):
        p = HeartbeatPolicy(timeout_ms=TIMEOUT_MS_MIN - 1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.timeout_ms_out_of_range", codes)

    def test_timeout_ms_too_high(self):
        p = HeartbeatPolicy(timeout_ms=TIMEOUT_MS_MAX + 1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.timeout_ms_out_of_range", codes)

    def test_timeout_ms_at_min_ok(self):
        p = HeartbeatPolicy(timeout_ms=TIMEOUT_MS_MIN)
        self.assertNotIn("heartbeat_policy.timeout_ms_out_of_range",
                         [d.code for d in p.validate()])

    def test_timeout_ms_at_max_ok(self):
        p = HeartbeatPolicy(timeout_ms=TIMEOUT_MS_MAX)
        self.assertNotIn("heartbeat_policy.timeout_ms_out_of_range",
                         [d.code for d in p.validate()])

    # fail_policy
    def test_invalid_fail_policy(self):
        p = HeartbeatPolicy(fail_policy="unknown_action")
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.invalid_fail_policy", codes)

    def test_all_valid_fail_policies_pass(self):
        for fp in HeartbeatFailPolicy.valid_values():
            with self.subTest(fail_policy=fp):
                p = HeartbeatPolicy(fail_policy=fp)
                self.assertNotIn("heartbeat_policy.invalid_fail_policy",
                                 [d.code for d in p.validate()])

    # max_override
    def test_max_override_too_low(self):
        p = HeartbeatPolicy(max_override=MAX_OVERRIDE_MIN - 0.1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.max_override_out_of_range", codes)

    def test_max_override_too_high(self):
        p = HeartbeatPolicy(max_override=MAX_OVERRIDE_MAX + 0.1)
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.max_override_out_of_range", codes)

    def test_max_override_at_min_ok(self):
        p = HeartbeatPolicy(max_override=MAX_OVERRIDE_MIN)
        self.assertNotIn("heartbeat_policy.max_override_out_of_range",
                         [d.code for d in p.validate()])

    def test_max_override_at_max_ok(self):
        p = HeartbeatPolicy(max_override=MAX_OVERRIDE_MAX)
        self.assertNotIn("heartbeat_policy.max_override_out_of_range",
                         [d.code for d in p.validate()])

    # safety_mode
    def test_invalid_safety_mode(self):
        p = HeartbeatPolicy(safety_mode="ultra")
        codes = [d.code for d in p.validate()]
        self.assertIn("heartbeat_policy.invalid_safety_mode", codes)

    def test_all_valid_safety_modes_pass(self):
        for sm in HeartbeatSafetyMode.valid_values():
            with self.subTest(safety_mode=sm):
                p = HeartbeatPolicy(safety_mode=sm)
                self.assertNotIn("heartbeat_policy.invalid_safety_mode",
                                 [d.code for d in p.validate()])

    # multiple errors
    def test_all_fields_invalid_five_errors(self):
        p = HeartbeatPolicy(
            tick_ms=1,
            timeout_ms=50,
            fail_policy="bad",
            max_override=5.0,
            safety_mode="ultra",
        )
        diags = p.validate()
        self.assertEqual(len(diags), 5)

    def test_is_valid_false_when_errors(self):
        p = HeartbeatPolicy(tick_ms=1)
        self.assertFalse(p.is_valid)

    def test_is_valid_true_when_no_errors(self):
        self.assertTrue(HeartbeatPolicy.default().is_valid)

    # diagnostic level
    def test_validation_errors_have_level_error(self):
        p = HeartbeatPolicy(tick_ms=1)
        for d in p.validate():
            self.assertEqual(d.level, "error")

    # trace_id propagation
    def test_validate_trace_id_propagated(self):
        trace_id = "test-trace-001"
        p = HeartbeatPolicy(tick_ms=1)
        for d in p.validate(trace_id=trace_id):
            self.assertEqual(d.trace_id, trace_id)


# ---------------------------------------------------------------------------
# validate_heartbeat_policy() standalone function
# ---------------------------------------------------------------------------

class TestValidateHeartbeatPolicyFunction(unittest.TestCase):
    def test_valid_dict_returns_true(self):
        ok, diags = validate_heartbeat_policy(HeartbeatPolicy.default().to_dict())
        self.assertTrue(ok)
        self.assertEqual(diags, [])

    def test_empty_dict_uses_defaults_and_is_valid(self):
        ok, diags = validate_heartbeat_policy({})
        self.assertTrue(ok)

    def test_invalid_tick_ms_returns_false(self):
        ok, diags = validate_heartbeat_policy({"tick_ms": 1})
        self.assertFalse(ok)
        codes = [d.code for d in diags]
        self.assertIn("heartbeat_policy.tick_ms_out_of_range", codes)

    def test_trace_id_propagated_into_diagnostics(self):
        tid = "trace-xyz"
        _, diags = validate_heartbeat_policy({"tick_ms": 1}, trace_id=tid)
        self.assertTrue(all(d.trace_id == tid for d in diags))

    def test_multiple_invalid_fields_all_reported(self):
        bad = {"tick_ms": 1, "timeout_ms": 1, "fail_policy": "bad"}
        ok, diags = validate_heartbeat_policy(bad)
        self.assertFalse(ok)
        self.assertGreaterEqual(len(diags), 3)

    def test_returns_tuple_of_two(self):
        result = validate_heartbeat_policy({})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
