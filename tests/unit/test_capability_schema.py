"""Tests for system.service.capability_schema — Phase 5 STAGE-03.

Coverage:
    A. validate_capability / capability_schema_errors — contract surface
    B. Conformant dicts — all three real adapter shapes pass
    C. Missing required keys — each omission is caught
    D. Wrong types — each type violation is caught
    E. List element type check — non-str items in list keys are caught
    F. Extra keys are tolerated
    G. Edge cases — non-dict input, empty dict, partially conformant
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from system.service.capability_schema import (
    CAPABILITY_SCHEMA,
    capability_schema_errors,
    validate_capability,
)


# ── Fixture helpers ────────────────────────────────────────────────────────

def _valid_unitree() -> dict:
    """Minimal conformant capability dict matching Unitree adapter output."""
    return {
        "brand":   "unitree",
        "adapter": "unitree_sdk2",
        "actions": ["stand", "sit", "walk", "stop", "lift_right_leg"],
        "sensors": ["battery", "imu", "foot_contact", "qpos", "qvel", "time"],
        "flags": {
            "sdk_available":      False,
            "mujoco_available":   False,
            "high_level_control": True,
            "robot_type":         "go2",
        },
        "required_settings": [
            "brand", "robot_type", "connection_mode", "timeout",
            "stop_policy", "safety_enabled",
            "dds_domain_id", "network_interface", "control_level",
            "mode_switch_policy",
        ],
    }


def _valid_spot() -> dict:
    """Minimal conformant capability dict matching Spot adapter output."""
    return {
        "brand":   "bostiondynamics",
        "adapter": "spot_sdk",
        "actions": ["stand", "sit", "walk", "stop"],
        "sensors": ["battery", "estop", "power_state"],
        "flags": {
            "sdk_available":     False,
            "lease_required":    True,
            "timesync_required": True,
            "session_open":      False,
            "hostname":          "",
        },
        "required_settings": [
            "brand", "robot_type", "connection_mode", "timeout",
            "stop_policy", "safety_enabled",
            "hostname", "auth_token", "timesync_required", "lease_required",
            "safety_channel", "power_policy",
        ],
    }


def _valid_cyberdog() -> dict:
    """Minimal conformant capability dict matching CyberDog adapter output."""
    return {
        "brand":   "xiaomi",
        "adapter": "cyberdog_sdk",
        "actions": ["stand", "sit", "walk", "stop"],
        "sensors": ["battery", "imu", "odometry"],
        "flags": {
            "ros2_available":     False,
            "ros2_required":      True,
            "namespace_isolated": True,
            "session_open":       False,
            "namespace":          "cyberdog",
        },
        "required_settings": [
            "brand", "robot_type", "connection_mode", "timeout",
            "stop_policy", "safety_enabled",
            "ros_domain_id", "rmw_impl", "namespace",
            "action_endpoints", "timestamp_policy",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
# A — Public API contract
# ══════════════════════════════════════════════════════════════════════════

class TestPublicAPI(unittest.TestCase):
    """validate_capability and capability_schema_errors surface contract."""

    def test_schema_constant_is_dict(self):
        """CAPABILITY_SCHEMA is a dict."""
        self.assertIsInstance(CAPABILITY_SCHEMA, dict)

    def test_schema_constant_has_required_keys(self):
        """CAPABILITY_SCHEMA defines all six required keys."""
        for key in ("brand", "adapter", "actions", "sensors",
                    "flags", "required_settings"):
            self.assertIn(key, CAPABILITY_SCHEMA, f"Missing from CAPABILITY_SCHEMA: {key!r}")

    def test_validate_returns_bool(self):
        """validate_capability always returns bool."""
        self.assertIsInstance(validate_capability(_valid_unitree()), bool)
        self.assertIsInstance(validate_capability({}), bool)

    def test_errors_returns_list(self):
        """capability_schema_errors always returns a list."""
        self.assertIsInstance(capability_schema_errors(_valid_unitree()), list)
        self.assertIsInstance(capability_schema_errors({}), list)

    def test_validate_true_iff_errors_empty(self):
        """validate_capability(x) == (len(capability_schema_errors(x)) == 0)."""
        for cap in (_valid_unitree(), _valid_spot(), _valid_cyberdog(), {}):
            self.assertEqual(
                validate_capability(cap),
                len(capability_schema_errors(cap)) == 0,
                f"Mismatch for cap={cap!r}",
            )


# ══════════════════════════════════════════════════════════════════════════
# B — Conformant dicts (all 3 real adapter shapes)
# ══════════════════════════════════════════════════════════════════════════

class TestConformantDicts(unittest.TestCase):
    """Real adapter capability shapes must pass schema validation."""

    def test_unitree_valid(self):
        """Unitree capability dict passes validation."""
        self.assertTrue(validate_capability(_valid_unitree()))
        self.assertEqual(capability_schema_errors(_valid_unitree()), [])

    def test_spot_valid(self):
        """Spot capability dict passes validation."""
        self.assertTrue(validate_capability(_valid_spot()))
        self.assertEqual(capability_schema_errors(_valid_spot()), [])

    def test_cyberdog_valid(self):
        """CyberDog capability dict passes validation."""
        self.assertTrue(validate_capability(_valid_cyberdog()))
        self.assertEqual(capability_schema_errors(_valid_cyberdog()), [])

    def test_real_unitree_adapter_output(self):
        """Real UnitreeAdapter().capabilities() passes schema validation."""
        from system.service.adapters.unitree_sdk2.adapter import UnitreeAdapter
        cap = UnitreeAdapter().capabilities()
        errors = capability_schema_errors(cap)
        self.assertEqual(errors, [], f"Unitree adapter output failed schema: {errors}")

    def test_real_spot_adapter_output(self):
        """Real SpotAdapter().capabilities() passes schema validation."""
        from system.service.adapters.spot_sdk.adapter import SpotAdapter
        cap = SpotAdapter().capabilities()
        errors = capability_schema_errors(cap)
        self.assertEqual(errors, [], f"Spot adapter output failed schema: {errors}")

    def test_real_cyberdog_adapter_output(self):
        """Real CyberDogAdapter().capabilities() passes schema validation."""
        from system.service.adapters.cyberdog_sdk.adapter import CyberDogAdapter
        cap = CyberDogAdapter().capabilities()
        errors = capability_schema_errors(cap)
        self.assertEqual(errors, [], f"CyberDog adapter output failed schema: {errors}")


# ══════════════════════════════════════════════════════════════════════════
# C — Missing required keys
# ══════════════════════════════════════════════════════════════════════════

class TestMissingKeys(unittest.TestCase):
    """Each missing required key must produce an error."""

    def _missing(self, key: str) -> dict:
        cap = _valid_unitree()
        del cap[key]
        return cap

    def test_missing_brand(self):
        errors = capability_schema_errors(self._missing("brand"))
        self.assertTrue(any("brand" in e for e in errors))
        self.assertFalse(validate_capability(self._missing("brand")))

    def test_missing_adapter(self):
        errors = capability_schema_errors(self._missing("adapter"))
        self.assertTrue(any("adapter" in e for e in errors))

    def test_missing_actions(self):
        errors = capability_schema_errors(self._missing("actions"))
        self.assertTrue(any("actions" in e for e in errors))

    def test_missing_sensors(self):
        errors = capability_schema_errors(self._missing("sensors"))
        self.assertTrue(any("sensors" in e for e in errors))

    def test_missing_flags(self):
        errors = capability_schema_errors(self._missing("flags"))
        self.assertTrue(any("flags" in e for e in errors))

    def test_missing_required_settings(self):
        errors = capability_schema_errors(self._missing("required_settings"))
        self.assertTrue(any("required_settings" in e for e in errors))

    def test_missing_all_keys(self):
        """Empty dict produces one error per required key."""
        errors = capability_schema_errors({})
        self.assertEqual(len(errors), len(CAPABILITY_SCHEMA))

    def test_error_count_matches_missing_count(self):
        """Removing N keys produces at least N errors."""
        cap = _valid_spot()
        del cap["brand"]
        del cap["adapter"]
        errors = capability_schema_errors(cap)
        self.assertGreaterEqual(len(errors), 2)


# ══════════════════════════════════════════════════════════════════════════
# D — Wrong types
# ══════════════════════════════════════════════════════════════════════════

class TestWrongTypes(unittest.TestCase):
    """Each type violation on a required key must produce an error."""

    def _wrong_type(self, key: str, value) -> dict:
        cap = _valid_unitree()
        cap[key] = value
        return cap

    def test_brand_not_str(self):
        """brand must be str."""
        errors = capability_schema_errors(self._wrong_type("brand", 42))
        self.assertTrue(any("brand" in e for e in errors))

    def test_adapter_not_str(self):
        """adapter must be str."""
        errors = capability_schema_errors(self._wrong_type("adapter", None))
        self.assertTrue(any("adapter" in e for e in errors))

    def test_actions_not_list(self):
        """actions must be list, not str."""
        errors = capability_schema_errors(self._wrong_type("actions", "stand,sit"))
        self.assertTrue(any("actions" in e for e in errors))
        self.assertFalse(validate_capability(self._wrong_type("actions", "stand,sit")))

    def test_sensors_not_list(self):
        """sensors must be list, not tuple."""
        errors = capability_schema_errors(self._wrong_type("sensors", ("battery",)))
        self.assertTrue(any("sensors" in e for e in errors))

    def test_flags_not_dict(self):
        """flags must be dict, not list."""
        errors = capability_schema_errors(self._wrong_type("flags", []))
        self.assertTrue(any("flags" in e for e in errors))

    def test_flags_not_dict_str(self):
        """flags must be dict, not str."""
        errors = capability_schema_errors(self._wrong_type("flags", "{}"))
        self.assertTrue(any("flags" in e for e in errors))

    def test_required_settings_not_list(self):
        """required_settings must be list, not dict."""
        errors = capability_schema_errors(self._wrong_type("required_settings", {}))
        self.assertTrue(any("required_settings" in e for e in errors))

    def test_multiple_wrong_types(self):
        """Multiple type violations produce multiple errors."""
        cap = _valid_spot()
        cap["brand"] = 123
        cap["actions"] = "stand"
        errors = capability_schema_errors(cap)
        self.assertGreaterEqual(len(errors), 2)


# ══════════════════════════════════════════════════════════════════════════
# E — List element type checks
# ══════════════════════════════════════════════════════════════════════════

class TestListElementTypes(unittest.TestCase):
    """Non-str elements inside actions/sensors/required_settings are caught."""

    def test_actions_contains_int(self):
        cap = _valid_unitree()
        cap["actions"] = ["stand", 42, "stop"]
        errors = capability_schema_errors(cap)
        self.assertTrue(any("actions" in e for e in errors))
        self.assertFalse(validate_capability(cap))

    def test_sensors_contains_none(self):
        cap = _valid_spot()
        cap["sensors"] = ["battery", None]
        errors = capability_schema_errors(cap)
        self.assertTrue(any("sensors" in e for e in errors))

    def test_required_settings_contains_int(self):
        cap = _valid_cyberdog()
        cap["required_settings"] = ["brand", 99]
        errors = capability_schema_errors(cap)
        self.assertTrue(any("required_settings" in e for e in errors))

    def test_empty_lists_are_valid(self):
        """Empty lists for actions/sensors/required_settings are conformant."""
        cap = _valid_unitree()
        cap["actions"] = []
        cap["sensors"] = []
        cap["required_settings"] = []
        self.assertEqual(capability_schema_errors(cap), [])

    def test_all_str_items_are_valid(self):
        """Lists containing only strings pass element type check."""
        cap = _valid_unitree()
        cap["actions"] = ["a", "b", "c"]
        cap["sensors"] = ["x", "y"]
        cap["required_settings"] = ["p", "q"]
        self.assertEqual(capability_schema_errors(cap), [])


# ══════════════════════════════════════════════════════════════════════════
# F — Extra keys are tolerated
# ══════════════════════════════════════════════════════════════════════════

class TestExtraKeysTolerated(unittest.TestCase):
    """Keys beyond the required set must not cause failures."""

    def test_extra_top_level_key_ignored(self):
        cap = _valid_unitree()
        cap["future_extension"] = "some_value"
        self.assertEqual(capability_schema_errors(cap), [])
        self.assertTrue(validate_capability(cap))

    def test_multiple_extra_keys_ignored(self):
        cap = _valid_spot()
        cap["version"] = "2.0"
        cap["deprecated"] = True
        cap["meta"] = {"author": "test"}
        self.assertEqual(capability_schema_errors(cap), [])

    def test_extra_flags_keys_ignored(self):
        """Extra keys inside flags dict are fine (flags is just checked as dict)."""
        cap = _valid_cyberdog()
        cap["flags"]["new_flag"] = "anything"
        self.assertEqual(capability_schema_errors(cap), [])


# ══════════════════════════════════════════════════════════════════════════
# G — Edge cases
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Non-dict input and unusual values."""

    def test_none_input(self):
        """None input returns an error."""
        errors = capability_schema_errors(None)
        self.assertGreater(len(errors), 0)
        self.assertFalse(validate_capability(None))

    def test_list_input(self):
        """List input returns an error."""
        errors = capability_schema_errors(["stand"])
        self.assertGreater(len(errors), 0)

    def test_string_input(self):
        """String input returns an error."""
        errors = capability_schema_errors("capabilities")
        self.assertGreater(len(errors), 0)

    def test_int_input(self):
        """Int input returns an error."""
        self.assertFalse(validate_capability(42))

    def test_empty_dict_fails(self):
        """Empty dict fails because all required keys are missing."""
        self.assertFalse(validate_capability({}))
        self.assertEqual(len(capability_schema_errors({})), len(CAPABILITY_SCHEMA))

    def test_empty_strings_are_valid_for_brand_adapter(self):
        """Empty strings satisfy the str type check (semantic checks are caller's job)."""
        cap = _valid_unitree()
        cap["brand"] = ""
        cap["adapter"] = ""
        self.assertEqual(capability_schema_errors(cap), [])

    def test_flags_can_have_any_value_types(self):
        """flags dict values are not type-checked — any value is allowed."""
        cap = _valid_spot()
        cap["flags"] = {"a": 1, "b": [1, 2], "c": None, "d": {"nested": True}}
        self.assertEqual(capability_schema_errors(cap), [])


if __name__ == "__main__":
    unittest.main()
