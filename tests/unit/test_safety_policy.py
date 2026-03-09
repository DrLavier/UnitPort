"""Phase 6 STAGE-02 — SafetyPolicy and unified preflight gate tests.

Covers:
  A. SafetyPolicy dataclass defaults and construction.
  B. LifecyclePolicy.safety_policy integration.
  C. ServiceRouter preflight gate behaviour (SafetyPolicy vs legacy paths).
  D. SAFETY_GATE_BLOCKED reason constant.

Isolation: no Qt, no real SDK.  All adapter calls use _MockAdapter.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.service.lifecycle import (          # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult, SafetyPolicy,
)
from system.service.adapters.base_adapter import BaseAdapter   # noqa: E402
from system.service.service_registry import ServiceRegistry    # noqa: E402
from system.service.service_router   import RouteOp, ServiceRouter  # noqa: E402


# ── Shared _MockAdapter ─────────────────────────────────────────────────────


class _MockAdapter(BaseAdapter):
    """Lightweight mock used across all STAGE-02 tests."""

    def __init__(self):
        self.open_session_status  = "ok"
        self.preflight_status     = "ok"
        self.close_session_status = "ok"
        self.run_action_raises    = None
        self.calls: list          = []

    def connect(self, **kw: Any) -> bool:
        self.calls.append(("connect", kw)); return True

    def run_action(self, action: str, **p: Any) -> Any:
        self.calls.append(("run_action", action, p))
        if self.run_action_raises:
            raise self.run_action_raises
        return True

    def stop(self) -> None:
        self.calls.append(("stop",))

    def get_sensor_data(self) -> Dict[str, Any]:
        self.calls.append(("get_sensor_data",)); return {"sensor": 1}

    def health(self) -> Dict[str, Any]:
        return {"connected": True}

    def open_session(self, config=None) -> Dict[str, Any]:
        self.calls.append(("open_session", config))
        if self.open_session_status == "ok":
            return LifecycleResult.ok("open_session").to_dict()
        return LifecycleResult.error(
            "open_session", LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "mock failure"},
        ).to_dict()

    def preflight(self, context=None) -> Dict[str, Any]:
        self.calls.append(("preflight", context))
        if self.preflight_status == "ok":
            return LifecycleResult.ok("preflight").to_dict()
        return LifecycleResult.error(
            "preflight", LifecycleReason.PREFLIGHT_FAILED,
            {"message": "mock preflight failure"},
        ).to_dict()

    def close_session(self) -> Dict[str, Any]:
        self.calls.append(("close_session",))
        if self.close_session_status == "ok":
            return LifecycleResult.ok("close_session").to_dict()
        return LifecycleResult.error(
            "close_session", LifecycleReason.SESSION_CLOSE_FAILED,
            {"message": "mock failure"},
        ).to_dict()


# ── Helper ──────────────────────────────────────────────────────────────────


def _router_with(adapter: Any, name: str = "mock") -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


def _run(adapter: _MockAdapter, policy: LifecyclePolicy,
         name: str = "mock") -> Dict[str, Any]:
    router = _router_with(adapter, name)
    return router.execute_with_lifecycle(
        name, RouteOp.RUN_ACTION,
        {"action": "stand", "params": {}},
        policy=policy,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — SafetyPolicy dataclass defaults and construction
# ═══════════════════════════════════════════════════════════════════════════


class TestSafePolicyDefaults(unittest.TestCase):
    """SafetyPolicy dataclass construction and default values."""

    def test_require_preflight_default_is_true(self):
        """SafetyPolicy() defaults require_preflight to True."""
        self.assertTrue(SafetyPolicy().require_preflight)

    def test_allow_execute_without_preflight_default_is_false(self):
        """SafetyPolicy() defaults allow_execute_without_preflight to False."""
        self.assertFalse(SafetyPolicy().allow_execute_without_preflight)

    def test_safety_context_default_is_empty_dict(self):
        """SafetyPolicy() defaults safety_context to empty dict."""
        self.assertEqual(SafetyPolicy().safety_context, {})

    def test_safety_context_is_independent_per_instance(self):
        """Two SafetyPolicy instances do not share the same safety_context dict."""
        a = SafetyPolicy()
        b = SafetyPolicy()
        a.safety_context["key"] = "val"
        self.assertNotIn("key", b.safety_context)

    def test_require_preflight_can_be_set_false(self):
        sp = SafetyPolicy(require_preflight=False)
        self.assertFalse(sp.require_preflight)

    def test_allow_execute_without_preflight_can_be_set_true(self):
        sp = SafetyPolicy(allow_execute_without_preflight=True)
        self.assertTrue(sp.allow_execute_without_preflight)

    def test_safety_context_accepts_dict(self):
        ctx = {"battery_ok": True, "estop": False}
        sp  = SafetyPolicy(safety_context=ctx)
        self.assertEqual(sp.safety_context, ctx)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — LifecyclePolicy.safety_policy integration
# ═══════════════════════════════════════════════════════════════════════════


class TestLifecyclePolicySafetyIntegration(unittest.TestCase):
    """LifecyclePolicy.safety_policy field is additive and backward-compatible."""

    def test_lifecycle_policy_safety_policy_default_is_none(self):
        """LifecyclePolicy() has safety_policy=None by default."""
        self.assertIsNone(LifecyclePolicy().safety_policy)

    def test_lifecycle_policy_accepts_safety_policy(self):
        sp     = SafetyPolicy()
        policy = LifecyclePolicy(safety_policy=sp)
        self.assertIs(policy.safety_policy, sp)

    def test_legacy_run_preflight_field_still_present(self):
        """run_preflight field is not removed from LifecyclePolicy."""
        self.assertFalse(LifecyclePolicy().run_preflight)

    def test_legacy_preflight_context_field_still_present(self):
        """preflight_context field is not removed from LifecyclePolicy."""
        self.assertEqual(LifecyclePolicy().preflight_context, {})

    def test_legacy_close_after_field_still_present(self):
        self.assertFalse(LifecyclePolicy().close_after)

    def test_legacy_session_config_field_still_present(self):
        self.assertEqual(LifecyclePolicy().session_config, {})

    def test_safety_policy_and_run_preflight_coexist(self):
        """Both safety_policy and run_preflight can be set simultaneously."""
        policy = LifecyclePolicy(
            run_preflight=True,
            safety_policy=SafetyPolicy(require_preflight=False),
        )
        self.assertTrue(policy.run_preflight)
        self.assertFalse(policy.safety_policy.require_preflight)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — SafetyPolicy gate in ServiceRouter
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetyGateInRouter(unittest.TestCase):
    """ServiceRouter Phase 6 preflight gate behaviour."""

    # ── Legacy path (safety_policy=None) — must be unchanged ─────────────

    def test_legacy_default_policy_does_not_call_preflight(self):
        """Default LifecyclePolicy() (no safety_policy) skips preflight."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy())
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(preflight_calls), 0)

    def test_legacy_run_preflight_true_calls_preflight(self):
        """Legacy run_preflight=True still calls preflight."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy(run_preflight=True))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(preflight_calls), 1)

    def test_legacy_preflight_failure_returns_preflight_failed(self):
        """Legacy path: preflight failure → PREFLIGHT_FAILED (unchanged)."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(run_preflight=True))
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_legacy_preflight_failure_stage_is_preflight(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(run_preflight=True))
        self.assertEqual(res["stage"], "preflight")

    def test_legacy_context_forwarded_to_preflight(self):
        """Legacy preflight_context is forwarded to adapter.preflight."""
        adapter = _MockAdapter()
        ctx     = {"legacy_key": 42}
        _run(adapter, LifecyclePolicy(run_preflight=True, preflight_context=ctx))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(preflight_calls[0][1], ctx)

    # ── Phase 6 SafetyPolicy path ─────────────────────────────────────────

    def test_safety_policy_require_preflight_true_calls_preflight(self):
        """SafetyPolicy(require_preflight=True) → preflight called."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=True)))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(preflight_calls), 1)

    def test_safety_policy_require_preflight_false_skips_preflight(self):
        """SafetyPolicy(require_preflight=False) → preflight skipped."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy(require_preflight=False)))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(preflight_calls), 0)

    def test_safety_policy_allow_execute_without_preflight_skips_preflight(self):
        """allow_execute_without_preflight=True skips gate even when require_preflight=True."""
        adapter = _MockAdapter()
        sp = SafetyPolicy(require_preflight=True, allow_execute_without_preflight=True)
        _run(adapter, LifecyclePolicy(safety_policy=sp))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(len(preflight_calls), 0)

    def test_safety_policy_failure_returns_safety_gate_blocked(self):
        """SafetyPolicy gate failure → SAFETY_GATE_BLOCKED (not PREFLIGHT_FAILED)."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(res["reason"], LifecycleReason.SAFETY_GATE_BLOCKED)

    def test_safety_policy_failure_not_preflight_failed(self):
        """SafetyPolicy gate failure reason is distinctly NOT PREFLIGHT_FAILED."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertNotEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_safety_policy_failure_stage_is_preflight(self):
        """SafetyPolicy gate failure stage is 'preflight'."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(res["stage"], "preflight")

    def test_safety_policy_failure_status_is_error(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(res["status"], "error")

    def test_safety_policy_failure_payload_is_none(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertIsNone(res["payload"])

    def test_safety_policy_failure_adapter_name_present(self):
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(res.get("adapter_name"), "mock")

    def test_safety_policy_failure_calls_close_session(self):
        """SafetyPolicy gate failure triggers best-effort close_session."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        close_calls = [c for c in adapter.calls if c[0] == "close_session"]
        self.assertEqual(len(close_calls), 1)

    def test_safety_context_forwarded_to_preflight(self):
        """SafetyPolicy.safety_context is forwarded to adapter.preflight."""
        adapter = _MockAdapter()
        ctx = {"estop_ok": True, "battery_pct": 80}
        sp  = SafetyPolicy(require_preflight=True, safety_context=ctx)
        _run(adapter, LifecyclePolicy(safety_policy=sp))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertEqual(preflight_calls[0][1], ctx)

    def test_safety_policy_gate_success_returns_ok(self):
        """SafetyPolicy gate with passing preflight → execution proceeds → ok."""
        adapter = _MockAdapter()
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        self.assertEqual(res["status"], "ok")

    def test_safety_policy_default_instance_calls_preflight_on_default_policy(self):
        """SafetyPolicy() with default require_preflight=True calls preflight."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        preflight_calls = [c for c in adapter.calls if c[0] == "preflight"]
        self.assertGreater(len(preflight_calls), 0)

    def test_safety_policy_gate_open_session_called_before_preflight(self):
        """open_session is called before preflight gate."""
        adapter = _MockAdapter()
        _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        stage_names = [c[0] for c in adapter.calls]
        open_idx     = stage_names.index("open_session")
        preflight_idx = stage_names.index("preflight")
        self.assertLess(open_idx, preflight_idx)

    def test_safety_policy_result_has_full_router_schema(self):
        """SafetyPolicy gate failure result has all 6 router-level keys."""
        adapter = _MockAdapter()
        adapter.preflight_status = "error"
        res = _run(adapter, LifecyclePolicy(safety_policy=SafetyPolicy()))
        for key in ("status", "reason", "stage", "payload", "diagnostics", "adapter_name"):
            with self.subTest(key=key):
                self.assertIn(key, res)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D — SAFETY_GATE_BLOCKED reason constant
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetyGateBlockedConstant(unittest.TestCase):
    """LifecycleReason.SAFETY_GATE_BLOCKED is a stable string constant."""

    def test_safety_gate_blocked_constant_exists(self):
        self.assertTrue(hasattr(LifecycleReason, "SAFETY_GATE_BLOCKED"))

    def test_safety_gate_blocked_value_is_correct(self):
        self.assertEqual(LifecycleReason.SAFETY_GATE_BLOCKED, "safety_gate_blocked")

    def test_safety_gate_blocked_is_string(self):
        self.assertIsInstance(LifecycleReason.SAFETY_GATE_BLOCKED, str)

    def test_safety_gate_blocked_is_distinct_from_preflight_failed(self):
        self.assertNotEqual(
            LifecycleReason.SAFETY_GATE_BLOCKED,
            LifecycleReason.PREFLIGHT_FAILED,
        )

    def test_retry_budget_exhausted_constant_exists(self):
        """RETRY_BUDGET_EXHAUSTED added alongside SAFETY_GATE_BLOCKED in Phase 6."""
        self.assertTrue(hasattr(LifecycleReason, "RETRY_BUDGET_EXHAUSTED"))

    def test_retry_budget_exhausted_value_is_correct(self):
        self.assertEqual(
            LifecycleReason.RETRY_BUDGET_EXHAUSTED, "retry_budget_exhausted"
        )

    def test_retry_budget_exhausted_is_string(self):
        self.assertIsInstance(LifecycleReason.RETRY_BUDGET_EXHAUSTED, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
