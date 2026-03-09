#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 STAGE-06: Error taxonomy and diagnostics consolidation tests.

Covers:
  A. LifecycleReason.EXECUTE_FAILED constant (no more magic string)
  B. ServiceRouter: adapter_name in every result path
  C. reason_codes module: taxonomy completeness and structure
  D. Consistency between reason_codes registry and adapter constants
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
from system.service.reason_codes import (                                                # noqa: E402
    ALL_REASON_CODES, CANONICAL_REASONS, UNITREE_REASONS, SPOT_REASONS,
    CYBERDOG_REASONS, REASON_MAP, DIAGNOSTIC_KEY_SCHEMA,
)
from system.service.adapters.base_adapter import BaseAdapter                             # noqa: E402


# ── Minimal mock adapter ──────────────────────────────────────────────────────

class _MockAdapter(BaseAdapter):
    def __init__(self):
        self.open_status  = "ok"
        self.pre_status   = "ok"
        self.close_status = "ok"
        self.action_raises = None

    def connect(self, **kw):    return True
    def run_action(self, a, **p):
        if self.action_raises:
            raise self.action_raises
        return True
    def stop(self):             pass
    def get_sensor_data(self):  return {"x": 1}
    def health(self):           return {"connected": True}

    def open_session(self, config=None):
        if self.open_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock failure"},
        ).to_dict()

    def preflight(self, context=None):
        if self.pre_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock failure"},
        ).to_dict()

    def close_session(self):
        if self.close_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock failure"},
        ).to_dict()


def _router_with(adapter, name="myAdapter"):
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


# ══════════════════════════════════════════════════════════════════════════
# A — LifecycleReason.EXECUTE_FAILED constant
# ══════════════════════════════════════════════════════════════════════════

class TestExecuteFailedConstant(unittest.TestCase):

    def test_constant_exists(self):
        self.assertTrue(hasattr(LifecycleReason, "EXECUTE_FAILED"))

    def test_constant_value_is_execute_failed_string(self):
        self.assertEqual(LifecycleReason.EXECUTE_FAILED, "execute_failed")

    def test_constant_is_stable_string(self):
        """Must be a plain string — no class magic, safe for dict comparison."""
        self.assertIsInstance(LifecycleReason.EXECUTE_FAILED, str)

    def test_constant_distinct_from_session_open_failed(self):
        self.assertNotEqual(LifecycleReason.EXECUTE_FAILED, LifecycleReason.SESSION_OPEN_FAILED)

    def test_six_canonical_constants(self):
        """At least 6 canonical constants (Phase 6 adds SAFETY_GATE_BLOCKED + RETRY_BUDGET_EXHAUSTED)."""
        codes = {v for k, v in vars(LifecycleReason).items()
                 if not k.startswith("_")}
        self.assertGreaterEqual(len(codes), 6)

    def test_all_six_are_strings(self):
        codes = {v for k, v in vars(LifecycleReason).items()
                 if not k.startswith("_")}
        for code in codes:
            with self.subTest(code=code):
                self.assertIsInstance(code, str)

    def test_router_uses_constant_not_magic_string(self):
        """Router reason for execute failure must equal the LifecycleReason constant."""
        adapter = _MockAdapter()
        adapter.action_raises = RuntimeError("boom")
        router = _router_with(adapter)
        result = router.execute_with_lifecycle("myAdapter", RouteOp.RUN_ACTION)
        self.assertEqual(result["reason"], LifecycleReason.EXECUTE_FAILED)


# ══════════════════════════════════════════════════════════════════════════
# B — adapter_name in every result path
# ══════════════════════════════════════════════════════════════════════════

class TestAdapterNameInResults(unittest.TestCase):

    _ADAPTER_NAME = "testAdapter"

    def _run(self, adapter=None, op=RouteOp.RUN_ACTION, op_args=None, policy=None):
        if adapter is None:
            adapter = _MockAdapter()
        reg = ServiceRegistry()
        reg.register(self._ADAPTER_NAME, adapter)
        router = ServiceRouter(registry=reg)
        return router.execute_with_lifecycle(
            self._ADAPTER_NAME, op, op_args or {}, policy
        )

    # ── Happy path ───────────────────────────────────────────────────────

    def test_success_path_has_adapter_name(self):
        result = self._run()
        self.assertIn("adapter_name", result)

    def test_success_path_adapter_name_correct(self):
        result = self._run()
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── Adapter not found ────────────────────────────────────────────────

    def test_not_found_path_has_adapter_name(self):
        router = ServiceRouter()  # empty registry
        result = router.execute_with_lifecycle("ghost", RouteOp.RUN_ACTION)
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], "ghost")

    # ── open_session failure ─────────────────────────────────────────────

    def test_open_session_fail_has_adapter_name(self):
        adapter = _MockAdapter()
        adapter.open_status = "error"
        result = self._run(adapter)
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── preflight failure ────────────────────────────────────────────────

    def test_preflight_fail_has_adapter_name(self):
        adapter = _MockAdapter()
        adapter.pre_status = "error"
        result = self._run(adapter, policy=LifecyclePolicy(run_preflight=True))
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── execute failure ──────────────────────────────────────────────────

    def test_execute_fail_has_adapter_name(self):
        adapter = _MockAdapter()
        adapter.action_raises = RuntimeError("crash")
        result = self._run(adapter)
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── close_session failure (success status, but adapter_name still present) ─

    def test_close_fail_result_has_adapter_name(self):
        adapter = _MockAdapter()
        adapter.close_status = "error"
        result = self._run(adapter, policy=LifecyclePolicy(close_after=True))
        # Status is still "ok" (close failure doesn't mask payload)
        self.assertEqual(result["status"], "ok")
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── close_session ok ─────────────────────────────────────────────────

    def test_close_ok_result_has_adapter_name(self):
        result = self._run(policy=LifecyclePolicy(close_after=True))
        self.assertIn("adapter_name", result)
        self.assertEqual(result["adapter_name"], self._ADAPTER_NAME)

    # ── Different adapter names are reflected correctly ───────────────────

    def test_adapter_name_reflects_actual_name_used(self):
        adapter = _MockAdapter()
        reg = ServiceRegistry()
        reg.register("spot_sdk", adapter)
        router = ServiceRouter(registry=reg)
        result = router.execute_with_lifecycle("spot_sdk", RouteOp.GET_SENSOR_DATA)
        self.assertEqual(result["adapter_name"], "spot_sdk")

    def test_adapter_name_reflects_cyberdog_sdk(self):
        adapter = _MockAdapter()
        reg = ServiceRegistry()
        reg.register("cyberdog_sdk", adapter)
        router = ServiceRouter(registry=reg)
        result = router.execute_with_lifecycle("cyberdog_sdk", RouteOp.STOP)
        self.assertEqual(result["adapter_name"], "cyberdog_sdk")


# ══════════════════════════════════════════════════════════════════════════
# C — reason_codes module completeness and structure
# ══════════════════════════════════════════════════════════════════════════

class TestReasonCodesModule(unittest.TestCase):

    def test_all_reason_codes_is_frozenset(self):
        self.assertIsInstance(ALL_REASON_CODES, frozenset)

    def test_all_reason_codes_nonempty(self):
        self.assertGreater(len(ALL_REASON_CODES), 0)

    def test_all_codes_are_strings(self):
        for code in ALL_REASON_CODES:
            with self.subTest(code=code):
                self.assertIsInstance(code, str)

    def test_all_codes_are_nonempty_strings(self):
        for code in ALL_REASON_CODES:
            with self.subTest(code=code):
                self.assertTrue(code, f"Empty reason code found: {code!r}")

    def test_canonical_reasons_subset_of_all(self):
        self.assertTrue(CANONICAL_REASONS.issubset(ALL_REASON_CODES))

    def test_unitree_reasons_subset_of_all(self):
        self.assertTrue(UNITREE_REASONS.issubset(ALL_REASON_CODES))

    def test_spot_reasons_subset_of_all(self):
        self.assertTrue(SPOT_REASONS.issubset(ALL_REASON_CODES))

    def test_cyberdog_reasons_subset_of_all(self):
        self.assertTrue(CYBERDOG_REASONS.issubset(ALL_REASON_CODES))

    def test_tiers_are_disjoint(self):
        """No reason code belongs to more than one adapter tier."""
        tiers = [UNITREE_REASONS, SPOT_REASONS, CYBERDOG_REASONS]
        for i, a in enumerate(tiers):
            for j, b in enumerate(tiers):
                if i != j:
                    overlap = a & b
                    self.assertFalse(overlap, f"Overlap between tiers: {overlap}")

    def test_canonical_plus_adapters_equals_all(self):
        combined = CANONICAL_REASONS | UNITREE_REASONS | SPOT_REASONS | CYBERDOG_REASONS
        self.assertEqual(combined, ALL_REASON_CODES)

    def test_total_reason_count(self):
        """8 canonical + 3 unitree + 8 spot + 8 cyberdog = 27 total.
        Phase 6 STAGE-04 added SAFETY_GATE_BLOCKED + RETRY_BUDGET_EXHAUSTED."""
        self.assertEqual(len(CANONICAL_REASONS), 8)
        self.assertEqual(len(UNITREE_REASONS), 3)
        self.assertEqual(len(SPOT_REASONS), 8)
        self.assertEqual(len(CYBERDOG_REASONS), 8)
        self.assertEqual(len(ALL_REASON_CODES), 27)

    def test_execute_failed_in_canonical(self):
        self.assertIn(LifecycleReason.EXECUTE_FAILED, CANONICAL_REASONS)

    def test_adapter_not_found_in_canonical(self):
        self.assertIn(LifecycleReason.ADAPTER_NOT_FOUND, CANONICAL_REASONS)

    def test_spot_sdk_unavailable_in_spot_reasons(self):
        self.assertIn("spot_sdk_unavailable", SPOT_REASONS)

    def test_cyberdog_ros2_unavailable_in_cyberdog_reasons(self):
        self.assertIn("cyberdog_ros2_unavailable", CYBERDOG_REASONS)

    def test_unitree_model_unavailable_in_unitree_reasons(self):
        self.assertIn("unitree_model_unavailable", UNITREE_REASONS)


# ══════════════════════════════════════════════════════════════════════════
# D — REASON_MAP and DIAGNOSTIC_KEY_SCHEMA structure
# ══════════════════════════════════════════════════════════════════════════

class TestReasonMap(unittest.TestCase):

    def test_reason_map_keys_match_all_reason_codes(self):
        self.assertEqual(frozenset(REASON_MAP.keys()), ALL_REASON_CODES)

    def test_every_entry_has_adapter_stage_tier(self):
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIn("adapter", meta, f"{code} missing 'adapter'")
                self.assertIn("stage",   meta, f"{code} missing 'stage'")
                self.assertIn("tier",    meta, f"{code} missing 'tier'")

    def test_canonical_entries_have_all_or_router_adapter(self):
        for code in CANONICAL_REASONS:
            with self.subTest(code=code):
                adapter = REASON_MAP[code]["adapter"]
                self.assertIn(adapter, ("all", "router"),
                              f"Canonical {code} has unexpected adapter: {adapter}")

    def test_unitree_entries_have_unitree_sdk2_adapter(self):
        for code in UNITREE_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "unitree_sdk2")

    def test_spot_entries_have_spot_sdk_adapter(self):
        for code in SPOT_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "spot_sdk")

    def test_cyberdog_entries_have_cyberdog_sdk_adapter(self):
        for code in CYBERDOG_REASONS:
            with self.subTest(code=code):
                self.assertEqual(REASON_MAP[code]["adapter"], "cyberdog_sdk")

    def test_tier_values_are_valid(self):
        valid_tiers = {"canonical", "adapter"}
        for code, meta in REASON_MAP.items():
            with self.subTest(code=code):
                self.assertIn(meta["tier"], valid_tiers)


class TestDiagnosticKeySchema(unittest.TestCase):

    def test_schema_has_three_adapters(self):
        self.assertIn("unitree_sdk2", DIAGNOSTIC_KEY_SCHEMA)
        self.assertIn("spot_sdk",     DIAGNOSTIC_KEY_SCHEMA)
        self.assertIn("cyberdog_sdk", DIAGNOSTIC_KEY_SCHEMA)

    def test_each_adapter_has_three_stages(self):
        for adapter in ("unitree_sdk2", "spot_sdk", "cyberdog_sdk"):
            with self.subTest(adapter=adapter):
                schema = DIAGNOSTIC_KEY_SCHEMA[adapter]
                self.assertIn("open_session_fail",  schema)
                self.assertIn("preflight_fail",     schema)
                self.assertIn("close_session_fail", schema)

    def test_key_lists_are_lists_of_strings(self):
        for adapter, stages in DIAGNOSTIC_KEY_SCHEMA.items():
            for stage, keys in stages.items():
                with self.subTest(adapter=adapter, stage=stage):
                    self.assertIsInstance(keys, list)
                    for k in keys:
                        self.assertIsInstance(k, str)


if __name__ == "__main__":
    unittest.main()
