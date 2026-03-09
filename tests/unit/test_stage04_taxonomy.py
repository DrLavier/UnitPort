#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 STAGE-04: Error taxonomy hardening tests.

Covers:
  A. Phase 6 canonical codes present in CANONICAL_REASONS and ALL_REASON_CODES
  B. retryable field present and typed correctly on every REASON_MAP entry
  C. retryable classification correctness (True / False per specification)
  D. Updated total reason-code counts (8+3+8+8 = 27)
  E. _get_retryable() helper derives correct values
  F. TelemetryCollector.end() auto-derives retryable from REASON_MAP
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import LifecycleReason                         # noqa: E402
from system.service.reason_codes import (                                    # noqa: E402
    ALL_REASON_CODES, CANONICAL_REASONS, UNITREE_REASONS,
    SPOT_REASONS, CYBERDOG_REASONS, REASON_MAP,
)
from system.service.telemetry import TelemetryCollector, _get_retryable      # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# A — Phase 6 canonical codes
# ══════════════════════════════════════════════════════════════════════════

class TestPhase6CanonicalCodes(unittest.TestCase):

    def test_safety_gate_blocked_in_canonical_reasons(self):
        self.assertIn(LifecycleReason.SAFETY_GATE_BLOCKED, CANONICAL_REASONS)

    def test_retry_budget_exhausted_in_canonical_reasons(self):
        self.assertIn(LifecycleReason.RETRY_BUDGET_EXHAUSTED, CANONICAL_REASONS)

    def test_safety_gate_blocked_in_all_reason_codes(self):
        self.assertIn(LifecycleReason.SAFETY_GATE_BLOCKED, ALL_REASON_CODES)

    def test_retry_budget_exhausted_in_all_reason_codes(self):
        self.assertIn(LifecycleReason.RETRY_BUDGET_EXHAUSTED, ALL_REASON_CODES)

    def test_safety_gate_blocked_in_reason_map(self):
        self.assertIn(LifecycleReason.SAFETY_GATE_BLOCKED, REASON_MAP)

    def test_retry_budget_exhausted_in_reason_map(self):
        self.assertIn(LifecycleReason.RETRY_BUDGET_EXHAUSTED, REASON_MAP)

    def test_safety_gate_blocked_tier_is_canonical(self):
        self.assertEqual(REASON_MAP[LifecycleReason.SAFETY_GATE_BLOCKED]["tier"], "canonical")

    def test_retry_budget_exhausted_tier_is_canonical(self):
        self.assertEqual(REASON_MAP[LifecycleReason.RETRY_BUDGET_EXHAUSTED]["tier"], "canonical")

    def test_safety_gate_blocked_adapter_is_router(self):
        self.assertEqual(REASON_MAP[LifecycleReason.SAFETY_GATE_BLOCKED]["adapter"], "router")

    def test_retry_budget_exhausted_adapter_is_router(self):
        self.assertEqual(REASON_MAP[LifecycleReason.RETRY_BUDGET_EXHAUSTED]["adapter"], "router")


# ══════════════════════════════════════════════════════════════════════════
# B — retryable field presence and type
# ══════════════════════════════════════════════════════════════════════════

class TestRetryableFieldPresence(unittest.TestCase):

    def test_every_reason_map_entry_has_retryable(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIn("retryable", meta, f"{code} missing 'retryable'")

    def test_every_retryable_value_is_bool(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIsInstance(
                    meta["retryable"], bool,
                    f"{code}.retryable is not bool: {meta['retryable']!r}",
                )

    def test_reason_map_still_has_adapter_stage_tier(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIn("adapter", meta)
                self.assertIn("stage",   meta)
                self.assertIn("tier",    meta)

    def test_reason_map_entries_have_exactly_four_keys(self):
        expected = {"adapter", "stage", "tier", "retryable"}
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertEqual(set(meta.keys()), expected)


# ══════════════════════════════════════════════════════════════════════════
# C — retryable classification correctness
# ══════════════════════════════════════════════════════════════════════════

class TestRetryableClassification(unittest.TestCase):

    # ── Retryable = True ─────────────────────────────────────────────────

    def test_execute_failed_is_retryable(self):
        self.assertTrue(REASON_MAP[LifecycleReason.EXECUTE_FAILED]["retryable"])

    def test_spot_command_timeout_is_retryable(self):
        self.assertTrue(REASON_MAP["spot_command_timeout"]["retryable"])

    def test_cyberdog_action_failed_is_retryable(self):
        self.assertTrue(REASON_MAP["cyberdog_action_failed"]["retryable"])

    # ── Retryable = False (canonical) ────────────────────────────────────

    def test_session_open_failed_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.SESSION_OPEN_FAILED]["retryable"])

    def test_preflight_failed_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.PREFLIGHT_FAILED]["retryable"])

    def test_session_close_failed_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.SESSION_CLOSE_FAILED]["retryable"])

    def test_capability_unavailable_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.CAPABILITY_UNAVAILABLE]["retryable"])

    def test_adapter_not_found_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.ADAPTER_NOT_FOUND]["retryable"])

    def test_safety_gate_blocked_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.SAFETY_GATE_BLOCKED]["retryable"])

    def test_retry_budget_exhausted_not_retryable(self):
        self.assertFalse(REASON_MAP[LifecycleReason.RETRY_BUDGET_EXHAUSTED]["retryable"])

    # ── Retryable = False (adapter) ───────────────────────────────────────

    def test_spot_action_failed_not_retryable(self):
        self.assertFalse(REASON_MAP["spot_action_failed"]["retryable"])

    def test_spot_auth_failed_not_retryable(self):
        self.assertFalse(REASON_MAP["spot_auth_failed"]["retryable"])

    def test_spot_estop_active_not_retryable(self):
        self.assertFalse(REASON_MAP["spot_estop_active"]["retryable"])

    def test_unitree_model_unavailable_not_retryable(self):
        self.assertFalse(REASON_MAP["unitree_model_unavailable"]["retryable"])

    def test_cyberdog_ros2_unavailable_not_retryable(self):
        self.assertFalse(REASON_MAP["cyberdog_ros2_unavailable"]["retryable"])

    def test_cyberdog_no_session_not_retryable(self):
        self.assertFalse(REASON_MAP["cyberdog_no_session"]["retryable"])

    # ── Counts ───────────────────────────────────────────────────────────

    def test_exactly_three_retryable_codes(self):
        retryable_codes = [c for c, m in REASON_MAP.items() if m["retryable"]]
        self.assertEqual(len(retryable_codes), 3)

    def test_retryable_codes_are_the_expected_three(self):
        retryable_codes = frozenset(c for c, m in REASON_MAP.items() if m["retryable"])
        expected = frozenset({
            LifecycleReason.EXECUTE_FAILED,
            "spot_command_timeout",
            "cyberdog_action_failed",
        })
        self.assertEqual(retryable_codes, expected)

    def test_non_retryable_count_is_24(self):
        non_retryable = [c for c, m in REASON_MAP.items() if not m["retryable"]]
        self.assertEqual(len(non_retryable), 24)


# ══════════════════════════════════════════════════════════════════════════
# D — Updated total counts
# ══════════════════════════════════════════════════════════════════════════

class TestUpdatedCounts(unittest.TestCase):

    def test_canonical_reasons_count_is_8(self):
        self.assertEqual(len(CANONICAL_REASONS), 8)

    def test_all_reason_codes_count_is_27(self):
        self.assertEqual(len(ALL_REASON_CODES), 27)

    def test_reason_map_has_27_entries(self):
        self.assertEqual(len(REASON_MAP), 27)

    def test_reason_map_keys_equal_all_reason_codes(self):
        self.assertEqual(frozenset(REASON_MAP.keys()), ALL_REASON_CODES)

    def test_adapter_tiers_unchanged(self):
        self.assertEqual(len(UNITREE_REASONS), 3)
        self.assertEqual(len(SPOT_REASONS), 8)
        self.assertEqual(len(CYBERDOG_REASONS), 8)

    def test_canonical_plus_adapters_equals_all(self):
        combined = CANONICAL_REASONS | UNITREE_REASONS | SPOT_REASONS | CYBERDOG_REASONS
        self.assertEqual(combined, ALL_REASON_CODES)


# ══════════════════════════════════════════════════════════════════════════
# E — _get_retryable() helper
# ══════════════════════════════════════════════════════════════════════════

class TestGetRetryable(unittest.TestCase):

    def test_execute_failed_is_true(self):
        self.assertTrue(_get_retryable(LifecycleReason.EXECUTE_FAILED))

    def test_spot_command_timeout_is_true(self):
        self.assertTrue(_get_retryable("spot_command_timeout"))

    def test_cyberdog_action_failed_is_true(self):
        self.assertTrue(_get_retryable("cyberdog_action_failed"))

    def test_preflight_failed_is_false(self):
        self.assertFalse(_get_retryable(LifecycleReason.PREFLIGHT_FAILED))

    def test_safety_gate_blocked_is_false(self):
        self.assertFalse(_get_retryable(LifecycleReason.SAFETY_GATE_BLOCKED))

    def test_retry_budget_exhausted_is_false(self):
        self.assertFalse(_get_retryable(LifecycleReason.RETRY_BUDGET_EXHAUSTED))

    def test_empty_reason_returns_default_false(self):
        self.assertFalse(_get_retryable(""))

    def test_unknown_reason_returns_default_false(self):
        self.assertFalse(_get_retryable("unknown_future_code_xyz"))

    def test_unknown_reason_returns_custom_default(self):
        self.assertTrue(_get_retryable("unknown_future_code_xyz", default=True))

    def test_empty_reason_returns_custom_default(self):
        self.assertTrue(_get_retryable("", default=True))


# ══════════════════════════════════════════════════════════════════════════
# F — TelemetryCollector.end() auto-derives retryable
# ══════════════════════════════════════════════════════════════════════════

class TestCollectorRetryableDerivation(unittest.TestCase):

    def _collector(self) -> TelemetryCollector:
        return TelemetryCollector(
            trace_id="t", adapter_name="a", brand="", operation="run_action",
        )

    def _end_event(self, reason: str, retryable_fallback: bool = False) -> dict:
        c = self._collector()
        c.start("execute")
        c.end("execute", "error", reason, retryable=retryable_fallback)
        return c.events[1]

    def test_execute_failed_retryable_true(self):
        ev = self._end_event(LifecycleReason.EXECUTE_FAILED)
        self.assertTrue(ev["retryable"])

    def test_spot_command_timeout_retryable_true(self):
        ev = self._end_event("spot_command_timeout")
        self.assertTrue(ev["retryable"])

    def test_cyberdog_action_failed_retryable_true(self):
        ev = self._end_event("cyberdog_action_failed")
        self.assertTrue(ev["retryable"])

    def test_preflight_failed_retryable_false(self):
        ev = self._end_event(LifecycleReason.PREFLIGHT_FAILED)
        self.assertFalse(ev["retryable"])

    def test_safety_gate_blocked_retryable_false(self):
        ev = self._end_event(LifecycleReason.SAFETY_GATE_BLOCKED)
        self.assertFalse(ev["retryable"])

    def test_session_open_failed_retryable_false(self):
        ev = self._end_event(LifecycleReason.SESSION_OPEN_FAILED)
        self.assertFalse(ev["retryable"])

    def test_empty_reason_uses_fallback_false(self):
        c = self._collector()
        c.start("execute")
        c.end("execute", "ok", reason="", retryable=False)
        self.assertFalse(c.events[1]["retryable"])

    def test_unknown_reason_uses_fallback_false(self):
        ev = self._end_event("totally_unknown_code_abc")
        self.assertFalse(ev["retryable"])

    def test_start_event_retryable_always_false(self):
        c = self._collector()
        c.start("execute")
        self.assertFalse(c.events[0]["retryable"])


if __name__ == "__main__":
    unittest.main()
