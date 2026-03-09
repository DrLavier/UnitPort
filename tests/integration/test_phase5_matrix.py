#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5 Test Matrix — STAGE-06.

Single-document evidence of Phase 5 completeness.
Organised in three sections mirroring the STAGE-06 task list:

  SECTION A — Unit evidence
    A1: Capability schema cross-brand validation (all 3 real adapters).
    A2: Settings schema completeness (all 3 brands).
    A3: Settings validator cross-brand behaviour.

  SECTION B — Integration coverage
    B1: Router validation gating — valid / invalid / legacy paths.
    B2: End-to-end Phase 5 stack (capabilities → schema → validator → router).

  SECTION C — Regression coverage
    C1: Phase 4 deliverables unchanged.
    C2: Phase 5 deliverable gate (new modules importable, callable, wired).

Isolation
---------
No Qt, no real SDK / ROS2.  UnitreeModel is stubbed; SPOT_SDK_AVAILABLE and
ROS2_AVAILABLE are patched to False for SDK-absent paths.  All router tests
use _MockAdapter so no real hardware calls are made.
"""

import copy
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Stub models.* (must precede any adapter / robot_context import) ─────────

_models_pkg          = types.ModuleType("models")
_models_pkg.__path__    = []
_models_pkg.__package__ = "models"

_unitree_pkg = types.ModuleType("models.Unitree")


class _StubUnitreeModel:
    """Stub with SDK flags False so open_session returns graceful error."""
    def __init__(self, robot_type: str = "go2"):
        self.robot_type = robot_type
    is_available     = False
    mujoco_available = False
    running          = False

    def stop(self) -> None:
        self.running = False

    def close_viewer(self) -> None:
        pass


_unitree_pkg.UnitreeModel = _StubUnitreeModel
_models_pkg.Unitree       = _unitree_pkg

_brand_reg_pkg = types.ModuleType("models.brand_registry")


class _StubBrandRegistry:
    @staticmethod
    def get_robot_brand_map() -> dict:
        return {
            "go2": "unitree", "a1": "unitree", "b1": "unitree",
            "b2": "unitree",  "h1": "unitree",
            "spot": "bostiondynamics",
            "cyberdog": "xiaomi", "cyberdog2": "xiaomi",
        }


_brand_reg_pkg.BrandRegistry = _StubBrandRegistry
_models_pkg.brand_registry   = _brand_reg_pkg

sys.modules.setdefault("models",                _models_pkg)
sys.modules.setdefault("models.Unitree",        _unitree_pkg)
sys.modules.setdefault("models.brand_registry", _brand_reg_pkg)

# ── Imports under test ──────────────────────────────────────────────────────

from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter   # noqa: E402
from system.service.adapters.spot_sdk.adapter     import SpotAdapter      # noqa: E402
from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter  # noqa: E402
from system.service.adapters.base_adapter         import BaseAdapter      # noqa: E402
from system.service.lifecycle import (                                     # noqa: E402
    LifecyclePolicy, LifecycleReason, LifecycleResult,
)
from system.service.service_registry import ServiceRegistry               # noqa: E402
from system.service.service_router   import RouteOp, ServiceRouter        # noqa: E402
from system.service.capability_schema import (                            # noqa: E402
    CAPABILITY_SCHEMA, validate_capability, capability_schema_errors,
)
from system.service.settings_schema  import KNOWN_BRANDS, get_settings_schema  # noqa: E402
from system.service.settings_validator import validate_settings           # noqa: E402
from bin.core.robot_context import RobotContext                           # noqa: E402


# ── Shared valid configuration dicts (one per brand) ────────────────────────

_VALID_UNITREE: Dict[str, Any] = {
    "brand":              "unitree",
    "robot_type":         "go2",
    "connection_mode":    "simulation",
    "timeout":            30,
    "stop_policy":        "immediate",
    "safety_enabled":     True,
    "dds_domain_id":      0,
    "network_interface":  "eth0",
    "control_level":      "high",
    "mode_switch_policy": "immediate",
}

_VALID_SPOT: Dict[str, Any] = {
    "brand":             "bostiondynamics",
    "robot_type":        "spot",
    "connection_mode":   "simulation",
    "timeout":           30,
    "stop_policy":       "graceful",
    "safety_enabled":    True,
    "hostname":          "192.168.80.3",
    "auth_token":        "tok-abc123",
    "timesync_required": True,
    "lease_required":    True,
    "safety_channel":    "main",
    "power_policy":      "safe_power_off",
}

_VALID_CYBERDOG: Dict[str, Any] = {
    "brand":            "xiaomi",
    "robot_type":       "cyberdog2",
    "connection_mode":  "simulation",
    "timeout":          30,
    "stop_policy":      "immediate",
    "safety_enabled":   True,
    "ros_domain_id":    1,
    "rmw_impl":         "rmw_fastrtps_cpp",
    "namespace":        "/cyberdog",
    "action_endpoints": ["/stand", "/sit"],
    "timestamp_policy": "strict",
}

# ── Shared: configurable mock adapter for router tests ───────────────────────


class _MockAdapter(BaseAdapter):
    """Lifecycle mock used by router matrix and Phase 5 integration tests."""

    def __init__(self):
        self.open_session_status  = "ok"
        self.preflight_status     = "ok"
        self.close_session_status = "ok"
        self.run_action_return    = True
        self.run_action_raises    = None
        self.calls: list          = []

    def connect(self, **kw: Any) -> bool:
        self.calls.append(("connect", kw)); return True

    def run_action(self, action: str, **p: Any) -> Any:
        self.calls.append(("run_action", action, p))
        if self.run_action_raises:
            raise self.run_action_raises
        return self.run_action_return

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
            {"message": "mock open_session failure"},
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
            {"message": "mock close_session failure"},
        ).to_dict()


# ── Helpers ─────────────────────────────────────────────────────────────────

_ROUTER_SCHEMA = frozenset({"status", "reason", "stage", "payload", "diagnostics", "adapter_name"})

_PATCH_SPOT_OFF  = "system.service.adapters.spot_sdk.adapter.SPOT_SDK_AVAILABLE"
_PATCH_ROS2_OFF  = "system.service.adapters.cyberdog_sdk.adapter.ROS2_AVAILABLE"
_PATCH_UNITREE_MODEL = "system.service.adapters.unitree_sdk2.adapter.UnitreeModel"


def _router_with(adapter: Any, name: str) -> ServiceRouter:
    reg = ServiceRegistry()
    reg.register(name, adapter)
    return ServiceRouter(registry=reg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Unit evidence
# ═══════════════════════════════════════════════════════════════════════════

# ── A1: Capability schema cross-brand validation ────────────────────────────


class TestCapabilitySchemaMatrix(unittest.TestCase):
    """All 3 real adapters' capabilities() conform to CAPABILITY_SCHEMA."""

    _ADAPTER_CONFIGS = [
        (UnitreeAdapter,   "unitree",          "unitree_sdk2"),
        (SpotAdapter,      "bostiondynamics",   "spot_sdk"),
        (CyberDogAdapter,  "xiaomi",            "cyberdog_sdk"),
    ]

    # ── validate_capability / capability_schema_errors ────────────────────

    def test_unitree_capabilities_pass_validate_capability(self):
        """UnitreeAdapter capabilities() passes validate_capability."""
        self.assertTrue(validate_capability(UnitreeAdapter().capabilities()))

    def test_spot_capabilities_pass_validate_capability(self):
        """SpotAdapter capabilities() passes validate_capability."""
        self.assertTrue(validate_capability(SpotAdapter().capabilities()))

    def test_cyberdog_capabilities_pass_validate_capability(self):
        """CyberDogAdapter capabilities() passes validate_capability."""
        self.assertTrue(validate_capability(CyberDogAdapter().capabilities()))

    def test_unitree_capabilities_zero_schema_errors(self):
        """capability_schema_errors returns empty list for UnitreeAdapter."""
        self.assertEqual(capability_schema_errors(UnitreeAdapter().capabilities()), [])

    def test_spot_capabilities_zero_schema_errors(self):
        """capability_schema_errors returns empty list for SpotAdapter."""
        self.assertEqual(capability_schema_errors(SpotAdapter().capabilities()), [])

    def test_cyberdog_capabilities_zero_schema_errors(self):
        """capability_schema_errors returns empty list for CyberDogAdapter."""
        self.assertEqual(capability_schema_errors(CyberDogAdapter().capabilities()), [])

    # ── brand / adapter identity fields ──────────────────────────────────

    def test_all_adapters_have_correct_brand_strings(self):
        """capabilities() brand field matches expected canonical brand name."""
        for cls, expected_brand, _ in self._ADAPTER_CONFIGS:
            with self.subTest(adapter=cls.__name__):
                cap = cls().capabilities()
                self.assertEqual(cap["brand"], expected_brand)

    def test_all_adapters_have_correct_adapter_strings(self):
        """capabilities() adapter field matches expected adapter module name."""
        for cls, _, expected_adapter in self._ADAPTER_CONFIGS:
            with self.subTest(adapter=cls.__name__):
                cap = cls().capabilities()
                self.assertEqual(cap["adapter"], expected_adapter)

    # ── required_settings completeness ───────────────────────────────────

    def test_all_adapters_have_non_empty_required_settings(self):
        """required_settings is a non-empty list for all 3 adapters."""
        for cls, _, _ in self._ADAPTER_CONFIGS:
            with self.subTest(adapter=cls.__name__):
                rs = cls().capabilities()["required_settings"]
                self.assertIsInstance(rs, list)
                self.assertGreater(len(rs), 0)

    def test_all_required_settings_elements_are_strings(self):
        """Every element of required_settings is a plain str."""
        for cls, _, _ in self._ADAPTER_CONFIGS:
            cap = cls().capabilities()
            for i, item in enumerate(cap["required_settings"]):
                with self.subTest(adapter=cls.__name__, index=i):
                    self.assertIsInstance(item, str)

    def test_required_settings_includes_global_keys(self):
        """All 3 adapters include the 6 global required settings."""
        _global_keys = {
            "brand", "robot_type", "connection_mode",
            "timeout", "stop_policy", "safety_enabled",
        }
        for cls, _, _ in self._ADAPTER_CONFIGS:
            with self.subTest(adapter=cls.__name__):
                rs = set(cls().capabilities()["required_settings"])
                self.assertTrue(
                    _global_keys.issubset(rs),
                    f"{cls.__name__} required_settings missing global keys: "
                    f"{_global_keys - rs}",
                )

    # ── CAPABILITY_SCHEMA constant ────────────────────────────────────────

    def test_capability_schema_constant_has_six_keys(self):
        """CAPABILITY_SCHEMA defines the 6 required keys."""
        self.assertEqual(len(CAPABILITY_SCHEMA), 6)

    def test_capability_schema_keys_are_correct(self):
        expected = {"brand", "adapter", "actions", "sensors", "flags", "required_settings"}
        self.assertEqual(set(CAPABILITY_SCHEMA.keys()), expected)

    # ── Non-conformant dicts are rejected ─────────────────────────────────

    def test_non_dict_fails_validate_capability(self):
        """Non-dict is rejected by validate_capability."""
        self.assertFalse(validate_capability("not a dict"))
        self.assertFalse(validate_capability(None))
        self.assertFalse(validate_capability([]))

    def test_missing_required_key_fails_validate_capability(self):
        """Missing a required key → validate_capability returns False."""
        cap = UnitreeAdapter().capabilities()
        del cap["brand"]
        self.assertFalse(validate_capability(cap))

    def test_wrong_type_fails_validate_capability(self):
        """Wrong type for a required key → validate_capability returns False."""
        cap = UnitreeAdapter().capabilities()
        cap["actions"] = "not-a-list"
        self.assertFalse(validate_capability(cap))

    def test_extra_keys_do_not_fail_validation(self):
        """Extra top-level keys in capabilities() dict are tolerated."""
        cap = UnitreeAdapter().capabilities()
        cap["extra_future_key"] = "value"
        self.assertTrue(validate_capability(cap))


# ── A2: Settings schema completeness ───────────────────────────────────────


class TestSettingsSchemaMatrix(unittest.TestCase):
    """settings_schema.py exports complete per-brand schemas."""

    _BRANDS = ("unitree", "bostiondynamics", "xiaomi")

    _GLOBAL_KEYS = frozenset({
        "brand", "robot_type", "connection_mode",
        "timeout", "stop_policy", "safety_enabled",
    })

    # ── KNOWN_BRANDS ──────────────────────────────────────────────────────

    def test_known_brands_is_tuple(self):
        self.assertIsInstance(KNOWN_BRANDS, tuple)

    def test_known_brands_has_exactly_three_entries(self):
        self.assertEqual(len(KNOWN_BRANDS), 3)

    def test_known_brands_contains_all_expected_brands(self):
        for brand in self._BRANDS:
            with self.subTest(brand=brand):
                self.assertIn(brand, KNOWN_BRANDS)

    # ── get_settings_schema return shape ─────────────────────────────────

    def test_get_settings_schema_returns_dict_for_all_brands(self):
        for brand in self._BRANDS:
            with self.subTest(brand=brand):
                schema = get_settings_schema(brand)
                self.assertIsInstance(schema, dict)

    def test_all_brand_schemas_contain_global_keys(self):
        """Every brand schema includes all 6 global settings."""
        for brand in self._BRANDS:
            with self.subTest(brand=brand):
                schema = get_settings_schema(brand)
                missing = self._GLOBAL_KEYS - set(schema.keys())
                self.assertFalse(
                    missing,
                    f"{brand} schema missing global keys: {missing}",
                )

    def test_unitree_schema_has_at_least_ten_keys(self):
        """6 global + 4 unitree-specific = 10 minimum."""
        self.assertGreaterEqual(len(get_settings_schema("unitree")), 10)

    def test_spot_schema_has_at_least_twelve_keys(self):
        """6 global + 6 spot-specific = 12 minimum."""
        self.assertGreaterEqual(len(get_settings_schema("bostiondynamics")), 12)

    def test_cyberdog_schema_has_at_least_eleven_keys(self):
        """6 global + 5 cyberdog-specific = 11 minimum."""
        self.assertGreaterEqual(len(get_settings_schema("xiaomi")), 11)

    # ── Per-entry shape ───────────────────────────────────────────────────

    def test_every_schema_entry_has_required_fields(self):
        """Each schema entry dict has type/required/default/description/choices."""
        _entry_keys = {"type", "required", "default", "description", "choices"}
        for brand in self._BRANDS:
            schema = get_settings_schema(brand)
            for key, entry in schema.items():
                with self.subTest(brand=brand, key=key):
                    missing_fields = _entry_keys - set(entry.keys())
                    self.assertFalse(
                        missing_fields,
                        f"{brand}/{key} entry missing fields: {missing_fields}",
                    )

    def test_every_schema_entry_type_is_a_python_type(self):
        """The 'type' field in every entry is an actual Python type object."""
        for brand in self._BRANDS:
            schema = get_settings_schema(brand)
            for key, entry in schema.items():
                with self.subTest(brand=brand, key=key):
                    self.assertIsInstance(entry["type"], type)

    def test_every_schema_entry_required_is_bool(self):
        """The 'required' field in every entry is a bool."""
        for brand in self._BRANDS:
            schema = get_settings_schema(brand)
            for key, entry in schema.items():
                with self.subTest(brand=brand, key=key):
                    self.assertIsInstance(entry["required"], bool)

    # ── Unknown brand ─────────────────────────────────────────────────────

    def test_unknown_brand_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_settings_schema("mystery_brand")

    def test_error_message_mentions_brand(self):
        try:
            get_settings_schema("bad_brand")
        except ValueError as exc:
            self.assertIn("bad_brand", str(exc))

    # ── Schema isolation ──────────────────────────────────────────────────

    def test_schemas_outer_dict_is_a_new_object(self):
        """Each get_settings_schema call returns a distinct outer dict object."""
        schema_a = get_settings_schema("unitree")
        schema_b = get_settings_schema("unitree")
        self.assertIsNot(schema_a, schema_b)

    def test_schemas_outer_dict_is_independent(self):
        """Adding a key to one returned schema does not affect subsequent calls."""
        schema_a = get_settings_schema("unitree")
        schema_a["__test_sentinel__"] = True
        schema_b = get_settings_schema("unitree")
        self.assertNotIn("__test_sentinel__", schema_b)

    # ── Brand-specific keys present ───────────────────────────────────────

    def test_unitree_schema_has_brand_specific_keys(self):
        schema = get_settings_schema("unitree")
        for key in ("dds_domain_id", "network_interface", "control_level", "mode_switch_policy"):
            with self.subTest(key=key):
                self.assertIn(key, schema)

    def test_spot_schema_has_brand_specific_keys(self):
        schema = get_settings_schema("bostiondynamics")
        for key in ("hostname", "auth_token", "safety_channel", "power_policy"):
            with self.subTest(key=key):
                self.assertIn(key, schema)

    def test_cyberdog_schema_has_brand_specific_keys(self):
        schema = get_settings_schema("xiaomi")
        for key in ("ros_domain_id", "rmw_impl", "action_endpoints", "timestamp_policy"):
            with self.subTest(key=key):
                self.assertIn(key, schema)

    def test_schemas_do_not_cross_contaminate(self):
        """Unitree-specific keys must not appear in Spot or CyberDog schemas."""
        unitree_keys = set(get_settings_schema("unitree")) - self._GLOBAL_KEYS
        for brand in ("bostiondynamics", "xiaomi"):
            with self.subTest(brand=brand):
                other_schema = get_settings_schema(brand)
                overlap = unitree_keys & set(other_schema.keys())
                self.assertFalse(
                    overlap,
                    f"Unitree keys leaked into {brand}: {overlap}",
                )


# ── A3: Settings validator cross-brand behaviour ────────────────────────────


class TestSettingsValidatorMatrix(unittest.TestCase):
    """validate_settings correctly validates configs for all 3 brands."""

    # ── Valid configs → ok ────────────────────────────────────────────────

    def test_valid_unitree_config_returns_ok(self):
        res = validate_settings("unitree", copy.deepcopy(_VALID_UNITREE))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["missing"], [])
        self.assertEqual(res["invalid"], [])

    def test_valid_spot_config_returns_ok(self):
        res = validate_settings("bostiondynamics", copy.deepcopy(_VALID_SPOT))
        self.assertEqual(res["status"], "ok")

    def test_valid_cyberdog_config_returns_ok(self):
        res = validate_settings("xiaomi", copy.deepcopy(_VALID_CYBERDOG))
        self.assertEqual(res["status"], "ok")

    # ── Missing required key ──────────────────────────────────────────────

    def test_missing_global_required_brand_key(self):
        config = copy.deepcopy(_VALID_UNITREE)
        del config["brand"]
        res = validate_settings("unitree", config)
        self.assertEqual(res["status"], "error")
        self.assertIn("brand", res["missing"])

    def test_missing_global_required_timeout(self):
        config = copy.deepcopy(_VALID_UNITREE)
        del config["timeout"]
        res = validate_settings("unitree", config)
        self.assertEqual(res["status"], "error")
        self.assertIn("timeout", res["missing"])

    def test_missing_unitree_specific_dds_domain_id(self):
        config = copy.deepcopy(_VALID_UNITREE)
        del config["dds_domain_id"]
        res = validate_settings("unitree", config)
        self.assertIn("dds_domain_id", res["missing"])

    def test_missing_spot_specific_hostname(self):
        config = copy.deepcopy(_VALID_SPOT)
        del config["hostname"]
        res = validate_settings("bostiondynamics", config)
        self.assertIn("hostname", res["missing"])

    def test_missing_cyberdog_specific_ros_domain_id(self):
        config = copy.deepcopy(_VALID_CYBERDOG)
        del config["ros_domain_id"]
        res = validate_settings("xiaomi", config)
        self.assertIn("ros_domain_id", res["missing"])

    # ── Optional keys absent → not flagged as missing ─────────────────────

    def test_optional_spot_timesync_required_absent_is_ok(self):
        config = copy.deepcopy(_VALID_SPOT)
        del config["timesync_required"]
        res = validate_settings("bostiondynamics", config)
        self.assertEqual(res["status"], "ok")
        self.assertNotIn("timesync_required", res["missing"])

    def test_optional_cyberdog_namespace_absent_is_ok(self):
        config = copy.deepcopy(_VALID_CYBERDOG)
        del config["namespace"]
        res = validate_settings("xiaomi", config)
        self.assertEqual(res["status"], "ok")
        self.assertNotIn("namespace", res["missing"])

    # ── Type violations ───────────────────────────────────────────────────

    def test_timeout_wrong_type_string_is_invalid(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["timeout"] = "thirty"
        res = validate_settings("unitree", config)
        self.assertEqual(res["status"], "error")
        self.assertIn("timeout", res["invalid"])

    def test_safety_enabled_wrong_type_string_is_invalid(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["safety_enabled"] = "yes"
        res = validate_settings("unitree", config)
        self.assertIn("safety_enabled", res["invalid"])

    def test_dds_domain_id_wrong_type_string_is_invalid(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["dds_domain_id"] = "zero"
        res = validate_settings("unitree", config)
        self.assertIn("dds_domain_id", res["invalid"])

    # ── bool-for-int guard ────────────────────────────────────────────────

    def test_bool_true_for_int_timeout_is_invalid(self):
        """True should not pass as int (bool is subclass of int in Python)."""
        config = copy.deepcopy(_VALID_UNITREE)
        config["timeout"] = True
        res = validate_settings("unitree", config)
        self.assertIn("timeout", res["invalid"])

    def test_bool_false_for_int_dds_domain_id_is_invalid(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["dds_domain_id"] = False
        res = validate_settings("unitree", config)
        self.assertIn("dds_domain_id", res["invalid"])

    # ── Choices violations ────────────────────────────────────────────────

    def test_invalid_connection_mode_choice(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["connection_mode"] = "wireless"
        res = validate_settings("unitree", config)
        self.assertIn("connection_mode", res["invalid"])

    def test_invalid_control_level_choice(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["control_level"] = "ultra"
        res = validate_settings("unitree", config)
        self.assertIn("control_level", res["invalid"])

    def test_invalid_power_policy_choice(self):
        config = copy.deepcopy(_VALID_SPOT)
        config["power_policy"] = "pull_plug"
        res = validate_settings("bostiondynamics", config)
        self.assertIn("power_policy", res["invalid"])

    def test_invalid_timestamp_policy_choice(self):
        config = copy.deepcopy(_VALID_CYBERDOG)
        config["timestamp_policy"] = "maybe"
        res = validate_settings("xiaomi", config)
        self.assertIn("timestamp_policy", res["invalid"])

    # ── Multiple violations ───────────────────────────────────────────────

    def test_multiple_violations_all_reported(self):
        config = {"brand": "unitree", "robot_type": "go2"}   # missing many
        res = validate_settings("unitree", config)
        self.assertEqual(res["status"], "error")
        self.assertGreater(len(res["missing"]), 1)

    # ── Guard cases ───────────────────────────────────────────────────────

    def test_non_dict_config_returns_error(self):
        res = validate_settings("unitree", "not-a-dict")
        self.assertEqual(res["status"], "error")
        self.assertGreater(len(res["errors"]), 0)

    def test_none_config_returns_error(self):
        res = validate_settings("unitree", None)
        self.assertEqual(res["status"], "error")

    def test_unknown_brand_returns_error(self):
        res = validate_settings("unknown_brand", {})
        self.assertEqual(res["status"], "error")

    def test_extra_config_keys_are_ignored(self):
        config = copy.deepcopy(_VALID_UNITREE)
        config["extra_future_setting"] = "ignored"
        res = validate_settings("unitree", config)
        self.assertEqual(res["status"], "ok")

    # ── ValidationResult schema ───────────────────────────────────────────

    def test_validation_result_has_all_required_keys(self):
        for brand, config in (
            ("unitree",        _VALID_UNITREE),
            ("bostiondynamics", _VALID_SPOT),
            ("xiaomi",         _VALID_CYBERDOG),
        ):
            with self.subTest(brand=brand):
                res = validate_settings(brand, copy.deepcopy(config))
                for key in ("status", "brand", "errors", "missing", "invalid"):
                    self.assertIn(key, res)

    def test_ok_status_only_when_all_lists_empty(self):
        res = validate_settings("unitree", copy.deepcopy(_VALID_UNITREE))
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["errors"],  [])
        self.assertEqual(res["missing"], [])
        self.assertEqual(res["invalid"], [])


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Integration coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── B1: Router validation gating ───────────────────────────────────────────


class TestRouterValidationGating(unittest.TestCase):
    """execute_with_lifecycle performs brand-gated settings validation at Step 0."""

    # ── Helper ────────────────────────────────────────────────────────────

    def _run(
        self,
        adapter_name: str,
        session_config: Dict[str, Any],
        adapter: Any = None,
    ) -> Dict[str, Any]:
        if adapter is None:
            adapter = _MockAdapter()
        router = _router_with(adapter, adapter_name)
        policy = LifecyclePolicy(session_config=session_config)
        return router.execute_with_lifecycle(
            adapter_name, RouteOp.RUN_ACTION,
            {"action": "stand", "params": {}},
            policy=policy,
        ), adapter

    # ── Invalid config → PREFLIGHT_FAILED at settings_validation ──────────

    def test_unitree_invalid_config_returns_preflight_failed(self):
        """Missing required unitree keys → PREFLIGHT_FAILED."""
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_unitree_invalid_config_stage_is_settings_validation(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        self.assertEqual(res["stage"], "settings_validation")

    def test_spot_invalid_config_returns_preflight_failed(self):
        """Missing required spot keys → PREFLIGHT_FAILED."""
        res, _ = self._run("spot_sdk", {"brand": "bostiondynamics"})
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)

    def test_cyberdog_invalid_config_returns_preflight_failed(self):
        """Missing required cyberdog keys → PREFLIGHT_FAILED."""
        res, _ = self._run("cyberdog_sdk", {"brand": "xiaomi"})
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.PREFLIGHT_FAILED)

    # ── Invalid config → open_session NOT called ──────────────────────────

    def test_invalid_config_open_session_not_called(self):
        """Validation failure short-circuits before open_session."""
        adapter = _MockAdapter()
        res, adapter = self._run("unitree_sdk2", {"brand": "unitree"}, adapter=adapter)
        open_calls = [c for c in adapter.calls if c[0] == "open_session"]
        self.assertEqual(len(open_calls), 0)

    # ── Invalid config → result shape ─────────────────────────────────────

    def test_invalid_config_result_has_adapter_name(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        self.assertEqual(res.get("adapter_name"), "unitree_sdk2")

    def test_invalid_config_result_has_diagnostics_validation_key(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        self.assertIn("validation", res["diagnostics"])

    def test_invalid_config_result_payload_is_none(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        self.assertIsNone(res["payload"])

    def test_invalid_config_result_has_full_router_schema(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        for key in _ROUTER_SCHEMA:
            with self.subTest(key=key):
                self.assertIn(key, res)

    def test_diagnostics_validation_contains_validation_result(self):
        res, _ = self._run("unitree_sdk2", {"brand": "unitree"})
        val = res["diagnostics"]["validation"]
        for key in ("status", "brand", "errors", "missing", "invalid"):
            with self.subTest(key=key):
                self.assertIn(key, val)

    # ── Valid config → proceeds past settings_validation ──────────────────

    def test_valid_unitree_config_proceeds_past_validation(self):
        """Valid unitree config → not blocked at settings_validation stage."""
        res, _ = self._run("unitree_sdk2", copy.deepcopy(_VALID_UNITREE))
        self.assertNotEqual(res.get("stage"), "settings_validation")

    def test_valid_spot_config_proceeds_past_validation(self):
        res, _ = self._run("spot_sdk", copy.deepcopy(_VALID_SPOT))
        self.assertNotEqual(res.get("stage"), "settings_validation")

    def test_valid_cyberdog_config_proceeds_past_validation(self):
        res, _ = self._run("cyberdog_sdk", copy.deepcopy(_VALID_CYBERDOG))
        self.assertNotEqual(res.get("stage"), "settings_validation")

    # ── Legacy (no brand in session_config) → validation skipped ──────────

    def test_legacy_empty_session_config_skips_validation(self):
        """Empty session_config (no brand key) → validation skipped → ok."""
        adapter = _MockAdapter()
        router  = _router_with(adapter, "unitree_sdk2")
        res = router.execute_with_lifecycle(
            "unitree_sdk2", RouteOp.RUN_ACTION,
            {"action": "stand", "params": {}},
        )
        self.assertEqual(res["status"], "ok")

    def test_legacy_session_config_none_skips_validation(self):
        """session_config=None → no brand → validation skipped → ok."""
        adapter = _MockAdapter()
        router  = _router_with(adapter, "unitree_sdk2")
        res = router.execute_with_lifecycle(
            "unitree_sdk2", RouteOp.RUN_ACTION,
            {"action": "stand", "params": {}},
            policy=LifecyclePolicy(session_config=None),
        )
        self.assertEqual(res["status"], "ok")

    def test_session_config_without_brand_skips_validation(self):
        """session_config with non-brand keys but no 'brand' → validation skipped."""
        adapter = _MockAdapter()
        router  = _router_with(adapter, "unitree_sdk2")
        res = router.execute_with_lifecycle(
            "unitree_sdk2", RouteOp.RUN_ACTION,
            {"action": "stand", "params": {}},
            policy=LifecyclePolicy(session_config={"timeout": 30}),
        )
        # No brand key → validation skipped → proceeds normally → ok
        self.assertEqual(res["status"], "ok")


# ── B2: End-to-end Phase 5 stack ───────────────────────────────────────────


class TestPhase5EndToEnd(unittest.TestCase):
    """capabilities() → schema → validator → router form a coherent stack."""

    _ADAPTER_BRAND_CONFIG = [
        (UnitreeAdapter,  "unitree",         "unitree_sdk2",  _VALID_UNITREE),
        (SpotAdapter,     "bostiondynamics",  "spot_sdk",      _VALID_SPOT),
        (CyberDogAdapter, "xiaomi",           "cyberdog_sdk",  _VALID_CYBERDOG),
    ]

    # ── capabilities() → validate_capability ─────────────────────────────

    def test_all_real_adapter_capabilities_pass_schema_in_one_pass(self):
        """All 3 adapters pass schema validation in a single loop."""
        for cls, _, _, _ in self._ADAPTER_BRAND_CONFIG:
            with self.subTest(adapter=cls.__name__):
                self.assertTrue(validate_capability(cls().capabilities()))

    # ── get_settings_schema → validate_settings ───────────────────────────

    def test_valid_config_per_brand_passes_validator(self):
        """A config built from the schema always passes validate_settings."""
        for _, brand, _, valid_config in self._ADAPTER_BRAND_CONFIG:
            with self.subTest(brand=brand):
                res = validate_settings(brand, copy.deepcopy(valid_config))
                self.assertEqual(res["status"], "ok",
                                 f"{brand}: validator rejected valid config: {res['errors']}")

    # ── validate_settings → execute_with_lifecycle ────────────────────────

    def test_valid_config_in_router_does_not_hit_settings_validation_stage(self):
        """End-to-end: valid config passes validation gate in router."""
        for _, brand, adapter_name, valid_config in self._ADAPTER_BRAND_CONFIG:
            with self.subTest(brand=brand):
                adapter = _MockAdapter()
                router  = _router_with(adapter, adapter_name)
                policy  = LifecyclePolicy(session_config=copy.deepcopy(valid_config))
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                    policy=policy,
                )
                self.assertNotEqual(res.get("stage"), "settings_validation",
                                    f"{brand}: valid config blocked at settings_validation")

    def test_invalid_config_in_router_always_hits_settings_validation_stage(self):
        """End-to-end: brand-only config (missing all required) blocked at validation."""
        for _, brand, adapter_name, _ in self._ADAPTER_BRAND_CONFIG:
            with self.subTest(brand=brand):
                adapter = _MockAdapter()
                router  = _router_with(adapter, adapter_name)
                policy  = LifecyclePolicy(session_config={"brand": brand})
                res = router.execute_with_lifecycle(
                    adapter_name, RouteOp.RUN_ACTION,
                    {"action": "stand", "params": {}},
                    policy=policy,
                )
                self.assertEqual(res.get("stage"), "settings_validation",
                                 f"{brand}: invalid config not blocked at validation")

    # ── required_settings ↔ schema alignment ──────────────────────────────

    def test_unitree_required_settings_aligned_with_schema(self):
        """Every key in UnitreeAdapter.required_settings exists in settings_schema."""
        schema = get_settings_schema("unitree")
        rs     = UnitreeAdapter().capabilities()["required_settings"]
        for key in rs:
            with self.subTest(key=key):
                self.assertIn(key, schema,
                              f"unitree required_setting {key!r} missing from schema")

    def test_spot_required_settings_aligned_with_schema(self):
        schema = get_settings_schema("bostiondynamics")
        rs     = SpotAdapter().capabilities()["required_settings"]
        for key in rs:
            with self.subTest(key=key):
                self.assertIn(key, schema)

    def test_cyberdog_required_settings_aligned_with_schema(self):
        schema = get_settings_schema("xiaomi")
        rs     = CyberDogAdapter().capabilities()["required_settings"]
        for key in rs:
            with self.subTest(key=key):
                self.assertIn(key, schema)

    # ── Brand string consistency ───────────────────────────────────────────

    def test_adapter_brand_strings_are_in_known_brands(self):
        """The brand string returned by capabilities() is a KNOWN_BRANDS member."""
        for cls, _, _, _ in self._ADAPTER_BRAND_CONFIG:
            with self.subTest(adapter=cls.__name__):
                brand = cls().capabilities()["brand"]
                self.assertIn(brand, KNOWN_BRANDS)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Regression coverage
# ═══════════════════════════════════════════════════════════════════════════

# ── C1: Phase 4 deliverables unchanged ─────────────────────────────────────


class TestPhase4Regression(unittest.TestCase):
    """Phase 4 adapter lifecycle and routing contracts remain unchanged."""

    _ADAPTER_CLASSES = [UnitreeAdapter, SpotAdapter, CyberDogAdapter]

    # ── capabilities() backward compat ────────────────────────────────────

    def test_all_adapters_capabilities_still_return_dict(self):
        for cls in self._ADAPTER_CLASSES:
            with self.subTest(adapter=cls.__name__):
                self.assertIsInstance(cls().capabilities(), dict)

    def test_all_adapters_capabilities_have_actions_sensors_flags(self):
        """Phase 4 shape of capabilities() unchanged (actions/sensors/flags)."""
        for cls in self._ADAPTER_CLASSES:
            with self.subTest(adapter=cls.__name__):
                cap = cls().capabilities()
                for key in ("actions", "sensors", "flags"):
                    self.assertIn(key, cap)

    # ── BRAND_ADAPTER_MAP unchanged ───────────────────────────────────────

    def test_brand_adapter_map_still_has_three_entries(self):
        self.assertEqual(len(RobotContext.BRAND_ADAPTER_MAP), 3)

    def test_brand_adapter_map_unitree_routes_correctly(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["unitree"], "unitree_sdk2")

    def test_brand_adapter_map_bostiondynamics_routes_correctly(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["bostiondynamics"], "spot_sdk")

    def test_brand_adapter_map_xiaomi_routes_correctly(self):
        self.assertEqual(RobotContext.BRAND_ADAPTER_MAP["xiaomi"], "cyberdog_sdk")

    # ── Legacy router passthrough unchanged ───────────────────────────────

    def test_legacy_run_action_passthrough_still_works(self):
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        result = router.run_action("u", "stand")
        self.assertEqual(result, True)

    def test_legacy_stop_passthrough_still_works(self):
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        router.stop("u")   # must not raise

    def test_legacy_get_sensor_data_passthrough_still_works(self):
        mock = _MockAdapter()
        router = _router_with(mock, "u")
        data = router.get_sensor_data("u")
        self.assertIsInstance(data, dict)

    # ── execute_with_lifecycle contract unchanged ──────────────────────────

    def test_adapter_not_found_still_returns_structured_error(self):
        """Missing adapter → ADAPTER_NOT_FOUND at acquire stage."""
        res = ServiceRouter().execute_with_lifecycle("ghost", RouteOp.STOP)
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["reason"], LifecycleReason.ADAPTER_NOT_FOUND)
        self.assertEqual(res["stage"],  "acquire")

    def test_execute_failed_reason_code_unchanged(self):
        """EXECUTE_FAILED constant is still the correct string value."""
        self.assertEqual(LifecycleReason.EXECUTE_FAILED, "execute_failed")

    def test_execute_failure_still_produces_execute_failed_reason(self):
        adapter = _MockAdapter()
        adapter.run_action_raises = RuntimeError("boom")
        router = _router_with(adapter, "u")
        res = router.execute_with_lifecycle("u", RouteOp.RUN_ACTION)
        self.assertEqual(res["reason"], LifecycleReason.EXECUTE_FAILED)

    def test_adapter_name_still_injected_on_all_result_paths(self):
        """adapter_name key present on every return path (Phase 4 contract)."""
        cases = []

        # not-found
        cases.append(ServiceRouter().execute_with_lifecycle("missing", RouteOp.STOP))

        # success
        a = _MockAdapter()
        cases.append(_router_with(a, "ok_adapter").execute_with_lifecycle(
            "ok_adapter", RouteOp.RUN_ACTION))

        # open_session fail
        a2 = _MockAdapter(); a2.open_session_status = "error"
        cases.append(_router_with(a2, "fail_open").execute_with_lifecycle(
            "fail_open", RouteOp.STOP))

        # execute fail
        a3 = _MockAdapter(); a3.run_action_raises = RuntimeError("x")
        cases.append(_router_with(a3, "fail_exec").execute_with_lifecycle(
            "fail_exec", RouteOp.RUN_ACTION))

        for res in cases:
            with self.subTest(status=res["status"], stage=res["stage"]):
                self.assertIn("adapter_name", res)

    def test_lifecycle_reason_constants_are_all_unchanged_strings(self):
        """All LifecycleReason string constants remain stable strings."""
        expected = {
            "SESSION_OPEN_FAILED":  "session_open_failed",
            "PREFLIGHT_FAILED":     "preflight_failed",
            "SESSION_CLOSE_FAILED": "session_close_failed",
            "CAPABILITY_UNAVAILABLE": "capability_unavailable",
            "ADAPTER_NOT_FOUND":    "adapter_not_found",
            "EXECUTE_FAILED":       "execute_failed",
        }
        for attr, value in expected.items():
            with self.subTest(constant=attr):
                self.assertEqual(getattr(LifecycleReason, attr), value)


# ── C2: Phase 5 deliverable gate ───────────────────────────────────────────


class TestPhase5DeliverableGate(unittest.TestCase):
    """All Phase 5 deliverables are present, importable, and callable."""

    # ── capability_schema module ──────────────────────────────────────────

    def test_capability_schema_module_importable(self):
        import system.service.capability_schema  # noqa: F401

    def test_capability_schema_constant_is_dict(self):
        self.assertIsInstance(CAPABILITY_SCHEMA, dict)

    def test_capability_schema_constant_has_six_keys(self):
        self.assertEqual(len(CAPABILITY_SCHEMA), 6)

    def test_validate_capability_is_callable(self):
        self.assertTrue(callable(validate_capability))

    def test_capability_schema_errors_is_callable(self):
        self.assertTrue(callable(capability_schema_errors))

    # ── settings_schema module ────────────────────────────────────────────

    def test_settings_schema_module_importable(self):
        import system.service.settings_schema  # noqa: F401

    def test_known_brands_is_exported(self):
        self.assertIsNotNone(KNOWN_BRANDS)

    def test_get_settings_schema_is_callable(self):
        self.assertTrue(callable(get_settings_schema))

    # ── settings_validator module ─────────────────────────────────────────

    def test_settings_validator_module_importable(self):
        import system.service.settings_validator  # noqa: F401

    def test_validate_settings_is_callable(self):
        self.assertTrue(callable(validate_settings))

    # ── Validation wired into execute_with_lifecycle ──────────────────────

    def test_settings_validation_stage_appears_in_router_on_invalid_config(self):
        """'settings_validation' stage appears in router result when config is invalid."""
        adapter = _MockAdapter()
        router  = _router_with(adapter, "unitree_sdk2")
        policy  = LifecyclePolicy(session_config={"brand": "unitree"})
        res = router.execute_with_lifecycle(
            "unitree_sdk2", RouteOp.STOP, policy=policy,
        )
        self.assertEqual(res["stage"], "settings_validation")

    def test_settings_validation_stage_absent_when_no_brand(self):
        """No brand in session_config → settings_validation stage never appears."""
        adapter = _MockAdapter()
        router  = _router_with(adapter, "unitree_sdk2")
        res = router.execute_with_lifecycle("unitree_sdk2", RouteOp.STOP)
        self.assertNotEqual(res.get("stage"), "settings_validation")

    def test_three_new_modules_all_importable_in_one_pass(self):
        """All 3 Phase 5 service modules importable without error."""
        try:
            import system.service.capability_schema   # noqa: F401
            import system.service.settings_schema     # noqa: F401
            import system.service.settings_validator  # noqa: F401
        except ImportError as exc:
            self.fail(f"Phase 5 module import failed: {exc}")

    def test_real_adapter_capabilities_schemas_match_known_brands(self):
        """The brand strings from all 3 real adapters are in KNOWN_BRANDS."""
        brands = {
            UnitreeAdapter().capabilities()["brand"],
            SpotAdapter().capabilities()["brand"],
            CyberDogAdapter().capabilities()["brand"],
        }
        self.assertEqual(brands, set(KNOWN_BRANDS))

    def test_validate_capability_false_for_empty_dict(self):
        """validate_capability rejects {} — confirms it is non-trivial."""
        self.assertFalse(validate_capability({}))

    def test_validate_settings_returns_error_for_totally_empty_config(self):
        """validate_settings({}) → error (all required keys missing)."""
        for brand in KNOWN_BRANDS:
            with self.subTest(brand=brand):
                res = validate_settings(brand, {})
                self.assertEqual(res["status"], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
