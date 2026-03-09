#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Phase 3 STAGE-02: BaseAdapter lifecycle interface.

Covers:
  - LifecycleResult factories and contract invariants
  - LifecycleReason constants
  - LifecyclePolicy defaults and custom values
  - BaseAdapter lifecycle default implementations
  - Backward-compatibility: existing adapter (UnitreeAdapter) still
    imports and instantiates without any code changes

Isolation: no Qt, no SDK, no network — all adapters are pure Python stubs.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Minimal stub for models.Unitree so UnitreeAdapter import works ─────────
_models_mod   = types.ModuleType("models")
_unitree_mod  = types.ModuleType("models.Unitree")

class _FakeUnitreeModel:
    def __init__(self, robot_type="go2"):
        self.robot_type = robot_type
        self.is_available = True
    def run_action(self, action, **params): return True
    def stop(self): pass
    def get_sensor_data(self): return {}
    def get_available_actions(self): return []

_unitree_mod.UnitreeModel = _FakeUnitreeModel
_models_mod.Unitree = _unitree_mod
sys.modules.setdefault("models", _models_mod)
sys.modules.setdefault("models.Unitree", _unitree_mod)

# ── Imports under test ─────────────────────────────────────────────────────
from system.service.lifecycle import (          # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
)
from system.service.adapters.base_adapter import BaseAdapter  # noqa: E402


# ── Concrete stub (implements legacy abstract methods) ────────────────────

class _StubAdapter(BaseAdapter):
    """Minimal concrete adapter for testing BaseAdapter defaults."""

    def __init__(self, connect_return: bool = True, connect_raises=None):
        self._connect_return = connect_return
        self._connect_raises = connect_raises
        self.connect_call_kwargs: Dict[str, Any] = {}
        self.connect_call_count = 0

    def connect(self, **kwargs: Any) -> bool:
        self.connect_call_count += 1
        self.connect_call_kwargs = kwargs
        if self._connect_raises:
            raise self._connect_raises
        return self._connect_return

    def run_action(self, action: str, **params: Any) -> Any:
        return None

    def stop(self) -> None:
        pass

    def get_sensor_data(self) -> Dict[str, Any]:
        return {}

    def health(self) -> Dict[str, Any]:
        return {}


# ══════════════════════════════════════════════════════════════════════════
# LifecycleReason constants
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleReasonConstants(unittest.TestCase):
    """LifecycleReason values are stable, non-empty, distinct strings."""

    _CONSTANTS = [
        ("SESSION_OPEN_FAILED",    "session_open_failed"),
        ("PREFLIGHT_FAILED",       "preflight_failed"),
        ("SESSION_CLOSE_FAILED",   "session_close_failed"),
        ("CAPABILITY_UNAVAILABLE", "capability_unavailable"),
        ("ADAPTER_NOT_FOUND",      "adapter_not_found"),
    ]

    def test_all_constants_exist(self):
        for attr, _ in self._CONSTANTS:
            self.assertTrue(hasattr(LifecycleReason, attr), f"Missing: {attr}")

    def test_all_constants_are_strings(self):
        for attr, _ in self._CONSTANTS:
            self.assertIsInstance(getattr(LifecycleReason, attr), str)

    def test_all_constants_are_nonempty(self):
        for attr, _ in self._CONSTANTS:
            self.assertNotEqual(getattr(LifecycleReason, attr), "")

    def test_constant_values_match_spec(self):
        for attr, expected in self._CONSTANTS:
            self.assertEqual(getattr(LifecycleReason, attr), expected,
                             f"{attr} value changed")

    def test_all_constants_are_distinct(self):
        values = [getattr(LifecycleReason, a) for a, _ in self._CONSTANTS]
        self.assertEqual(len(values), len(set(values)), "Duplicate reason codes")


# ══════════════════════════════════════════════════════════════════════════
# LifecycleResult contract
# ══════════════════════════════════════════════════════════════════════════

class TestLifecycleResultOkFactory(unittest.TestCase):

    def setUp(self):
        self.result = LifecycleResult.ok("open_session")

    def test_status_is_ok(self):
        self.assertEqual(self.result.status, "ok")

    def test_reason_is_empty(self):
        self.assertEqual(self.result.reason, "")

    def test_stage_is_set(self):
        self.assertEqual(self.result.stage, "open_session")

    def test_diagnostics_is_empty_dict(self):
        self.assertEqual(self.result.diagnostics, {})

    def test_is_ok_property_true(self):
        self.assertTrue(self.result.is_ok)

    def test_to_dict_has_required_keys(self):
        d = self.result.to_dict()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, d)

    def test_to_dict_values_match(self):
        d = self.result.to_dict()
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["reason"], "")
        self.assertEqual(d["stage"], "open_session")
        self.assertEqual(d["diagnostics"], {})

    def test_stage_preserved_for_each_lifecycle_stage(self):
        for stage in ("open_session", "preflight", "close_session"):
            r = LifecycleResult.ok(stage)
            self.assertEqual(r.stage, stage)


class TestLifecycleResultErrorFactory(unittest.TestCase):

    def setUp(self):
        self.result = LifecycleResult.error(
            "open_session",
            LifecycleReason.SESSION_OPEN_FAILED,
            {"message": "connect failed"},
        )

    def test_status_is_error(self):
        self.assertEqual(self.result.status, "error")

    def test_reason_is_set(self):
        self.assertEqual(self.result.reason, LifecycleReason.SESSION_OPEN_FAILED)

    def test_stage_is_set(self):
        self.assertEqual(self.result.stage, "open_session")

    def test_diagnostics_from_arg(self):
        self.assertEqual(self.result.diagnostics, {"message": "connect failed"})

    def test_is_ok_property_false(self):
        self.assertFalse(self.result.is_ok)

    def test_to_dict_status_is_error(self):
        self.assertEqual(self.result.to_dict()["status"], "error")

    def test_error_without_diagnostics_defaults_to_empty(self):
        r = LifecycleResult.error("preflight", LifecycleReason.PREFLIGHT_FAILED)
        self.assertEqual(r.diagnostics, {})

    def test_error_diagnostics_none_treated_as_empty(self):
        r = LifecycleResult.error("preflight", LifecycleReason.PREFLIGHT_FAILED, None)
        self.assertEqual(r.diagnostics, {})

    def test_to_dict_is_json_serialisable_types(self):
        d = self.result.to_dict()
        self.assertIsInstance(d["status"], str)
        self.assertIsInstance(d["reason"], str)
        self.assertIsInstance(d["stage"], str)
        self.assertIsInstance(d["diagnostics"], dict)


# ══════════════════════════════════════════════════════════════════════════
# LifecyclePolicy contract
# ══════════════════════════════════════════════════════════════════════════

class TestLifecyclePolicyDefaults(unittest.TestCase):

    def setUp(self):
        self.policy = LifecyclePolicy()

    def test_run_preflight_default_false(self):
        self.assertFalse(self.policy.run_preflight)

    def test_close_after_default_false(self):
        self.assertFalse(self.policy.close_after)

    def test_session_config_default_empty_dict(self):
        self.assertEqual(self.policy.session_config, {})

    def test_preflight_context_default_empty_dict(self):
        self.assertEqual(self.policy.preflight_context, {})

    def test_two_default_policies_independent(self):
        p1, p2 = LifecyclePolicy(), LifecyclePolicy()
        p1.session_config["x"] = 1
        self.assertNotIn("x", p2.session_config)

    def test_custom_values_accepted(self):
        p = LifecyclePolicy(
            run_preflight=True,
            close_after=True,
            session_config={"host": "localhost"},
            preflight_context={"timeout": 5},
        )
        self.assertTrue(p.run_preflight)
        self.assertTrue(p.close_after)
        self.assertEqual(p.session_config["host"], "localhost")
        self.assertEqual(p.preflight_context["timeout"], 5)


# ══════════════════════════════════════════════════════════════════════════
# BaseAdapter lifecycle defaults
# ══════════════════════════════════════════════════════════════════════════

class TestBaseAdapterOpenSession(unittest.TestCase):

    def test_open_session_success_returns_ok_status(self):
        adapter = _StubAdapter(connect_return=True)
        result = adapter.open_session()
        self.assertEqual(result["status"], "ok")

    def test_open_session_success_reason_empty(self):
        adapter = _StubAdapter(connect_return=True)
        result = adapter.open_session()
        self.assertEqual(result["reason"], "")

    def test_open_session_success_stage(self):
        adapter = _StubAdapter(connect_return=True)
        result = adapter.open_session()
        self.assertEqual(result["stage"], "open_session")

    def test_open_session_success_has_diagnostics_key(self):
        adapter = _StubAdapter(connect_return=True)
        result = adapter.open_session()
        self.assertIn("diagnostics", result)

    def test_open_session_delegates_to_connect(self):
        adapter = _StubAdapter(connect_return=True)
        adapter.open_session()
        self.assertEqual(adapter.connect_call_count, 1)

    def test_open_session_passes_config_to_connect(self):
        adapter = _StubAdapter(connect_return=True)
        adapter.open_session(config={"robot_type": "go2"})
        self.assertEqual(adapter.connect_call_kwargs, {"robot_type": "go2"})

    def test_open_session_no_config_passes_empty_kwargs(self):
        adapter = _StubAdapter(connect_return=True)
        adapter.open_session()
        self.assertEqual(adapter.connect_call_kwargs, {})

    def test_open_session_connect_returns_false_gives_error(self):
        adapter = _StubAdapter(connect_return=False)
        result = adapter.open_session()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_open_session_connect_raises_gives_error(self):
        adapter = _StubAdapter(connect_raises=RuntimeError("boom"))
        result = adapter.open_session()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], LifecycleReason.SESSION_OPEN_FAILED)

    def test_open_session_connect_raises_message_in_diagnostics(self):
        adapter = _StubAdapter(connect_raises=RuntimeError("boom"))
        result = adapter.open_session()
        self.assertIn("message", result["diagnostics"])
        self.assertIn("boom", result["diagnostics"]["message"])

    def test_open_session_result_has_all_required_keys(self):
        for return_val in (True, False):
            with self.subTest(return_val=return_val):
                adapter = _StubAdapter(connect_return=return_val)
                result = adapter.open_session()
                for key in ("status", "reason", "stage", "diagnostics"):
                    self.assertIn(key, result)


class TestBaseAdapterPreflight(unittest.TestCase):

    def test_preflight_default_returns_ok(self):
        adapter = _StubAdapter()
        result = adapter.preflight()
        self.assertEqual(result["status"], "ok")

    def test_preflight_default_reason_empty(self):
        adapter = _StubAdapter()
        result = adapter.preflight()
        self.assertEqual(result["reason"], "")

    def test_preflight_default_stage(self):
        adapter = _StubAdapter()
        result = adapter.preflight()
        self.assertEqual(result["stage"], "preflight")

    def test_preflight_with_context_still_ok(self):
        adapter = _StubAdapter()
        result = adapter.preflight(context={"key": "value"})
        self.assertEqual(result["status"], "ok")

    def test_preflight_does_not_call_connect(self):
        adapter = _StubAdapter()
        adapter.preflight()
        self.assertEqual(adapter.connect_call_count, 0)

    def test_preflight_result_has_required_keys(self):
        adapter = _StubAdapter()
        result = adapter.preflight()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, result)


class TestBaseAdapterCloseSession(unittest.TestCase):

    def test_close_session_returns_ok(self):
        adapter = _StubAdapter()
        result = adapter.close_session()
        self.assertEqual(result["status"], "ok")

    def test_close_session_reason_empty(self):
        adapter = _StubAdapter()
        result = adapter.close_session()
        self.assertEqual(result["reason"], "")

    def test_close_session_stage(self):
        adapter = _StubAdapter()
        result = adapter.close_session()
        self.assertEqual(result["stage"], "close_session")

    def test_close_session_does_not_call_connect(self):
        adapter = _StubAdapter()
        adapter.close_session()
        self.assertEqual(adapter.connect_call_count, 0)

    def test_close_session_result_has_required_keys(self):
        adapter = _StubAdapter()
        result = adapter.close_session()
        for key in ("status", "reason", "stage", "diagnostics"):
            self.assertIn(key, result)


class TestBaseAdapterCapabilities(unittest.TestCase):

    def test_capabilities_returns_dict(self):
        adapter = _StubAdapter()
        result = adapter.capabilities()
        self.assertIsInstance(result, dict)

    def test_capabilities_has_actions_key(self):
        adapter = _StubAdapter()
        self.assertIn("actions", adapter.capabilities())

    def test_capabilities_has_sensors_key(self):
        adapter = _StubAdapter()
        self.assertIn("sensors", adapter.capabilities())

    def test_capabilities_has_flags_key(self):
        adapter = _StubAdapter()
        self.assertIn("flags", adapter.capabilities())

    def test_capabilities_actions_is_list(self):
        adapter = _StubAdapter()
        self.assertIsInstance(adapter.capabilities()["actions"], list)

    def test_capabilities_sensors_is_list(self):
        adapter = _StubAdapter()
        self.assertIsInstance(adapter.capabilities()["sensors"], list)

    def test_capabilities_flags_is_dict(self):
        adapter = _StubAdapter()
        self.assertIsInstance(adapter.capabilities()["flags"], dict)

    def test_capabilities_default_actions_empty(self):
        adapter = _StubAdapter()
        self.assertEqual(adapter.capabilities()["actions"], [])

    def test_capabilities_default_sensors_empty(self):
        adapter = _StubAdapter()
        self.assertEqual(adapter.capabilities()["sensors"], [])

    def test_capabilities_default_flags_empty(self):
        adapter = _StubAdapter()
        self.assertEqual(adapter.capabilities()["flags"], {})


# ══════════════════════════════════════════════════════════════════════════
# Legacy interface contract — abstract methods still present
# ══════════════════════════════════════════════════════════════════════════

class TestBaseAdapterLegacyInterface(unittest.TestCase):

    def test_concrete_adapter_implements_connect(self):
        adapter = _StubAdapter()
        self.assertTrue(adapter.connect())

    def test_concrete_adapter_implements_run_action(self):
        adapter = _StubAdapter()
        adapter.run_action("stand")  # must not raise

    def test_concrete_adapter_implements_stop(self):
        adapter = _StubAdapter()
        adapter.stop()  # must not raise

    def test_concrete_adapter_implements_get_sensor_data(self):
        adapter = _StubAdapter()
        result = adapter.get_sensor_data()
        self.assertIsInstance(result, dict)

    def test_concrete_adapter_implements_health(self):
        adapter = _StubAdapter()
        result = adapter.health()
        self.assertIsInstance(result, dict)

    def test_baseadapter_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            BaseAdapter()  # type: ignore

    def test_lifecycle_methods_are_not_abstract(self):
        """Lifecycle methods must have non-abstract defaults."""
        abstract_names = set(getattr(BaseAdapter, "__abstractmethods__", set()))
        for method in ("open_session", "preflight", "close_session", "capabilities"):
            self.assertNotIn(method, abstract_names,
                             f"{method} must not be abstract")

    def test_legacy_methods_are_abstract(self):
        abstract_names = set(getattr(BaseAdapter, "__abstractmethods__", set()))
        for method in ("connect", "run_action", "stop", "get_sensor_data", "health"):
            self.assertIn(method, abstract_names,
                          f"{method} must remain abstract")


# ══════════════════════════════════════════════════════════════════════════
# UnitreeAdapter backward compat (import + instantiate, no SDK needed)
# ══════════════════════════════════════════════════════════════════════════

class TestUnitreeAdapterBackwardCompat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """UnitreeAdapter imports models.Unitree — ensure stub is present."""
        # Already installed at module level; just verify
        assert "models.Unitree" in sys.modules

    def test_unitree_adapter_imports(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        self.assertTrue(callable(UnitreeAdapter))

    def test_unitree_adapter_instantiates(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        self.assertIsNotNone(adapter)

    def test_unitree_adapter_is_baseadapter_subclass(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        self.assertTrue(issubclass(UnitreeAdapter, BaseAdapter))

    def test_unitree_adapter_has_open_session(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        self.assertTrue(callable(adapter.open_session))

    def test_unitree_adapter_open_session_returns_ok(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        result = adapter.open_session()
        self.assertEqual(result["status"], "ok")

    def test_unitree_adapter_preflight_returns_ok_after_open_session(self):
        # Phase 4 v2: preflight requires open_session to initialise the model.
        # Without it the model is None and v2 preflight returns a reason-coded error.
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        adapter.open_session()  # initialise model
        result = adapter.preflight()
        self.assertEqual(result["status"], "ok")

    def test_unitree_adapter_close_session_returns_ok(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        result = adapter.close_session()
        self.assertEqual(result["status"], "ok")

    def test_unitree_adapter_capabilities_structure(self):
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        adapter = UnitreeAdapter("go2")
        caps = adapter.capabilities()
        self.assertIsInstance(caps["actions"], list)
        self.assertIsInstance(caps["sensors"], list)
        self.assertIsInstance(caps["flags"], dict)


if __name__ == "__main__":
    unittest.main()
