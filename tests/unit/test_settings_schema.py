"""Tests for system.service.settings_schema — Phase 5 STAGE-04.

Coverage:
    A. Module-level constants (KNOWN_BRANDS)
    B. Entry shape contract — every entry has the required five fields
    C. get_settings_schema("unitree") — global + Unitree-specific keys present
    D. get_settings_schema("bostiondynamics") — global + Spot-specific keys present
    E. get_settings_schema("xiaomi") — global + CyberDog-specific keys present
    F. Global keys present in all three brand schemas
    G. Brand-specific keys do not bleed across brands
    H. Unknown brand raises ValueError
    I. Return value is a fresh dict (mutation does not affect subsequent calls)
    J. Specific required/default/choices spot-checks (README §8.3–§8.4)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from system.service.settings_schema import (
    KNOWN_BRANDS,
    get_settings_schema,
)


# ── Constants ──────────────────────────────────────────────────────────────

_GLOBAL_KEYS = (
    "brand", "robot_type", "connection_mode",
    "timeout", "stop_policy", "safety_enabled",
)

_UNITREE_SPECIFIC = (
    "dds_domain_id", "network_interface", "control_level", "mode_switch_policy",
)

_SPOT_SPECIFIC = (
    "hostname", "auth_token", "timesync_required",
    "lease_required", "safety_channel", "power_policy",
)

_CYBERDOG_SPECIFIC = (
    "ros_domain_id", "rmw_impl", "namespace",
    "action_endpoints", "timestamp_policy",
)

_ENTRY_FIELDS = ("type", "required", "default", "description", "choices")


# ══════════════════════════════════════════════════════════════════════════
# A — Module-level constants
# ══════════════════════════════════════════════════════════════════════════

class TestKnownBrands(unittest.TestCase):
    """KNOWN_BRANDS constant shape and content."""

    def test_known_brands_is_tuple(self):
        self.assertIsInstance(KNOWN_BRANDS, tuple)

    def test_known_brands_contains_three_brands(self):
        self.assertEqual(len(KNOWN_BRANDS), 3)

    def test_known_brands_contains_unitree(self):
        self.assertIn("unitree", KNOWN_BRANDS)

    def test_known_brands_contains_bostiondynamics(self):
        self.assertIn("bostiondynamics", KNOWN_BRANDS)

    def test_known_brands_contains_xiaomi(self):
        self.assertIn("xiaomi", KNOWN_BRANDS)


# ══════════════════════════════════════════════════════════════════════════
# B — Entry shape contract
# ══════════════════════════════════════════════════════════════════════════

class TestEntryShape(unittest.TestCase):
    """Every entry in every schema has the five required fields."""

    def _assert_entry_shape(self, key: str, entry: dict, brand: str) -> None:
        for field in _ENTRY_FIELDS:
            self.assertIn(
                field, entry,
                f"Entry {key!r} in brand {brand!r} missing field {field!r}",
            )
        self.assertIsInstance(
            entry["type"], type,
            f"Entry {key!r} 'type' must be a Python type",
        )
        self.assertIsInstance(
            entry["required"], bool,
            f"Entry {key!r} 'required' must be bool",
        )
        self.assertIsInstance(
            entry["description"], str,
            f"Entry {key!r} 'description' must be str",
        )
        # choices is either None or a non-empty list
        choices = entry["choices"]
        if choices is not None:
            self.assertIsInstance(
                choices, list,
                f"Entry {key!r} 'choices' must be list or None",
            )

    def test_unitree_entries_all_have_required_fields(self):
        schema = get_settings_schema("unitree")
        for key, entry in schema.items():
            self._assert_entry_shape(key, entry, "unitree")

    def test_spot_entries_all_have_required_fields(self):
        schema = get_settings_schema("bostiondynamics")
        for key, entry in schema.items():
            self._assert_entry_shape(key, entry, "bostiondynamics")

    def test_cyberdog_entries_all_have_required_fields(self):
        schema = get_settings_schema("xiaomi")
        for key, entry in schema.items():
            self._assert_entry_shape(key, entry, "xiaomi")


# ══════════════════════════════════════════════════════════════════════════
# C — Unitree schema
# ══════════════════════════════════════════════════════════════════════════

class TestUnitreeSchema(unittest.TestCase):
    """get_settings_schema('unitree') content."""

    def setUp(self):
        self.schema = get_settings_schema("unitree")

    def test_returns_dict(self):
        self.assertIsInstance(self.schema, dict)

    def test_has_all_global_keys(self):
        for key in _GLOBAL_KEYS:
            self.assertIn(key, self.schema, f"Missing global key: {key!r}")

    def test_has_all_unitree_specific_keys(self):
        for key in _UNITREE_SPECIFIC:
            self.assertIn(key, self.schema, f"Missing Unitree-specific key: {key!r}")

    def test_total_key_count(self):
        """Global (6) + Unitree-specific (4) = 10 keys."""
        self.assertEqual(len(self.schema), len(_GLOBAL_KEYS) + len(_UNITREE_SPECIFIC))

    def test_brand_choices(self):
        self.assertIn("unitree", self.schema["brand"]["choices"])
        self.assertIn("bostiondynamics", self.schema["brand"]["choices"])
        self.assertIn("xiaomi", self.schema["brand"]["choices"])

    def test_robot_type_choices_dynamic_from_registry(self):
        choices = self.schema["robot_type"]["choices"]
        self.assertIsInstance(choices, list)
        self.assertIn("go2", choices)

    def test_connection_mode_choices(self):
        choices = self.schema["connection_mode"]["choices"]
        self.assertIn("simulation", choices)
        self.assertIn("hardware", choices)

    def test_stop_policy_choices(self):
        choices = self.schema["stop_policy"]["choices"]
        self.assertIn("immediate", choices)
        self.assertIn("graceful", choices)

    def test_timeout_type_is_int(self):
        self.assertIs(self.schema["timeout"]["type"], int)

    def test_safety_enabled_type_is_bool(self):
        self.assertIs(self.schema["safety_enabled"]["type"], bool)

    def test_dds_domain_id_type_and_required(self):
        entry = self.schema["dds_domain_id"]
        self.assertIs(entry["type"], int)
        self.assertTrue(entry["required"])
        self.assertIsNone(entry["default"])

    def test_network_interface_type_and_required(self):
        entry = self.schema["network_interface"]
        self.assertIs(entry["type"], str)
        self.assertTrue(entry["required"])

    def test_control_level_choices(self):
        choices = self.schema["control_level"]["choices"]
        self.assertIn("high", choices)
        self.assertIn("low", choices)

    def test_mode_switch_policy_choices(self):
        choices = self.schema["mode_switch_policy"]["choices"]
        self.assertIn("immediate", choices)
        self.assertIn("graceful", choices)

    def test_all_keys_required_true(self):
        """All Unitree settings are required (no defaults)."""
        for key in _UNITREE_SPECIFIC:
            self.assertTrue(
                self.schema[key]["required"],
                f"Expected required=True for {key!r}",
            )


# ══════════════════════════════════════════════════════════════════════════
# D — Spot schema
# ══════════════════════════════════════════════════════════════════════════

class TestSpotSchema(unittest.TestCase):
    """get_settings_schema('bostiondynamics') content."""

    def setUp(self):
        self.schema = get_settings_schema("bostiondynamics")

    def test_returns_dict(self):
        self.assertIsInstance(self.schema, dict)

    def test_has_all_global_keys(self):
        for key in _GLOBAL_KEYS:
            self.assertIn(key, self.schema, f"Missing global key: {key!r}")

    def test_has_all_spot_specific_keys(self):
        for key in _SPOT_SPECIFIC:
            self.assertIn(key, self.schema, f"Missing Spot-specific key: {key!r}")

    def test_total_key_count(self):
        """Global (6) + Spot-specific (6) = 12 keys."""
        self.assertEqual(len(self.schema), len(_GLOBAL_KEYS) + len(_SPOT_SPECIFIC))

    def test_hostname_required(self):
        entry = self.schema["hostname"]
        self.assertIs(entry["type"], str)
        self.assertTrue(entry["required"])
        self.assertIsNone(entry["default"])

    def test_auth_token_required(self):
        entry = self.schema["auth_token"]
        self.assertIs(entry["type"], str)
        self.assertTrue(entry["required"])

    def test_timesync_required_has_default(self):
        """timesync_required is optional with default=True."""
        entry = self.schema["timesync_required"]
        self.assertIs(entry["type"], bool)
        self.assertFalse(entry["required"])
        self.assertTrue(entry["default"])

    def test_lease_required_has_default(self):
        """lease_required is optional with default=True."""
        entry = self.schema["lease_required"]
        self.assertIs(entry["type"], bool)
        self.assertFalse(entry["required"])
        self.assertTrue(entry["default"])

    def test_safety_channel_required(self):
        entry = self.schema["safety_channel"]
        self.assertIs(entry["type"], str)
        self.assertTrue(entry["required"])
        self.assertIsNone(entry["default"])

    def test_power_policy_choices(self):
        choices = self.schema["power_policy"]["choices"]
        self.assertIn("safe_power_off", choices)
        self.assertIn("emergency_cut", choices)

    def test_power_policy_required(self):
        self.assertTrue(self.schema["power_policy"]["required"])

    def test_robot_type_choices_contains_spot(self):
        self.assertIn("spot", self.schema["robot_type"]["choices"])


# ══════════════════════════════════════════════════════════════════════════
# E — CyberDog schema
# ══════════════════════════════════════════════════════════════════════════

class TestCyberDogSchema(unittest.TestCase):
    """get_settings_schema('xiaomi') content."""

    def setUp(self):
        self.schema = get_settings_schema("xiaomi")

    def test_returns_dict(self):
        self.assertIsInstance(self.schema, dict)

    def test_has_all_global_keys(self):
        for key in _GLOBAL_KEYS:
            self.assertIn(key, self.schema, f"Missing global key: {key!r}")

    def test_has_all_cyberdog_specific_keys(self):
        for key in _CYBERDOG_SPECIFIC:
            self.assertIn(key, self.schema, f"Missing CyberDog-specific key: {key!r}")

    def test_total_key_count(self):
        """Global (6) + CyberDog-specific (5) = 11 keys."""
        self.assertEqual(len(self.schema), len(_GLOBAL_KEYS) + len(_CYBERDOG_SPECIFIC))

    def test_ros_domain_id_type_and_required(self):
        entry = self.schema["ros_domain_id"]
        self.assertIs(entry["type"], int)
        self.assertTrue(entry["required"])
        self.assertIsNone(entry["default"])

    def test_rmw_impl_type_and_required(self):
        entry = self.schema["rmw_impl"]
        self.assertIs(entry["type"], str)
        self.assertTrue(entry["required"])

    def test_namespace_is_optional_with_empty_default(self):
        """namespace is optional with default=''."""
        entry = self.schema["namespace"]
        self.assertIs(entry["type"], str)
        self.assertFalse(entry["required"])
        self.assertEqual(entry["default"], "")

    def test_action_endpoints_type_is_list(self):
        entry = self.schema["action_endpoints"]
        self.assertIs(entry["type"], list)
        self.assertTrue(entry["required"])
        self.assertIsNone(entry["default"])

    def test_timestamp_policy_choices(self):
        choices = self.schema["timestamp_policy"]["choices"]
        self.assertIn("strict", choices)
        self.assertIn("relaxed", choices)

    def test_timestamp_policy_required(self):
        self.assertTrue(self.schema["timestamp_policy"]["required"])

    def test_robot_type_choices_contains_cyberdog(self):
        choices = self.schema["robot_type"]["choices"]
        self.assertIn("cyberdog", choices)


# ══════════════════════════════════════════════════════════════════════════
# F — Global keys present in all three brand schemas
# ══════════════════════════════════════════════════════════════════════════

class TestGlobalKeysAcrossBrands(unittest.TestCase):
    """Each global key appears in every brand's merged schema."""

    def test_all_global_keys_in_unitree(self):
        schema = get_settings_schema("unitree")
        for key in _GLOBAL_KEYS:
            self.assertIn(key, schema)

    def test_all_global_keys_in_spot(self):
        schema = get_settings_schema("bostiondynamics")
        for key in _GLOBAL_KEYS:
            self.assertIn(key, schema)

    def test_all_global_keys_in_cyberdog(self):
        schema = get_settings_schema("xiaomi")
        for key in _GLOBAL_KEYS:
            self.assertIn(key, schema)

    def test_global_brand_entry_identical_across_brands(self):
        """The 'brand' entry has the same choices regardless of which brand schema."""
        u = get_settings_schema("unitree")["brand"]
        s = get_settings_schema("bostiondynamics")["brand"]
        c = get_settings_schema("xiaomi")["brand"]
        self.assertEqual(u["choices"], s["choices"])
        self.assertEqual(u["choices"], c["choices"])


# ══════════════════════════════════════════════════════════════════════════
# G — Brand-specific keys do not bleed across brands
# ══════════════════════════════════════════════════════════════════════════

class TestNoKeyCrossContamination(unittest.TestCase):
    """Brand-specific keys must not appear in other brands' schemas."""

    def test_unitree_specific_keys_absent_from_spot(self):
        schema = get_settings_schema("bostiondynamics")
        for key in _UNITREE_SPECIFIC:
            self.assertNotIn(key, schema, f"Unitree key {key!r} leaked into Spot schema")

    def test_unitree_specific_keys_absent_from_cyberdog(self):
        schema = get_settings_schema("xiaomi")
        for key in _UNITREE_SPECIFIC:
            self.assertNotIn(key, schema, f"Unitree key {key!r} leaked into CyberDog schema")

    def test_spot_specific_keys_absent_from_unitree(self):
        schema = get_settings_schema("unitree")
        for key in _SPOT_SPECIFIC:
            self.assertNotIn(key, schema, f"Spot key {key!r} leaked into Unitree schema")

    def test_spot_specific_keys_absent_from_cyberdog(self):
        schema = get_settings_schema("xiaomi")
        for key in _SPOT_SPECIFIC:
            self.assertNotIn(key, schema, f"Spot key {key!r} leaked into CyberDog schema")

    def test_cyberdog_specific_keys_absent_from_unitree(self):
        schema = get_settings_schema("unitree")
        for key in _CYBERDOG_SPECIFIC:
            self.assertNotIn(key, schema, f"CyberDog key {key!r} leaked into Unitree schema")

    def test_cyberdog_specific_keys_absent_from_spot(self):
        schema = get_settings_schema("bostiondynamics")
        for key in _CYBERDOG_SPECIFIC:
            self.assertNotIn(key, schema, f"CyberDog key {key!r} leaked into Spot schema")


# ══════════════════════════════════════════════════════════════════════════
# H — Unknown brand raises ValueError
# ══════════════════════════════════════════════════════════════════════════

class TestUnknownBrand(unittest.TestCase):
    """get_settings_schema raises ValueError for unknown brand strings."""

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            get_settings_schema("")

    def test_unknown_brand_raises(self):
        with self.assertRaises(ValueError):
            get_settings_schema("unknown_brand")

    def test_wrong_case_raises(self):
        """Brand keys are case-sensitive."""
        with self.assertRaises(ValueError):
            get_settings_schema("Unitree")

    def test_partial_brand_name_raises(self):
        with self.assertRaises(ValueError):
            get_settings_schema("uni")

    def test_error_message_mentions_brand(self):
        with self.assertRaises(ValueError) as ctx:
            get_settings_schema("badkey")
        self.assertIn("badkey", str(ctx.exception))

    def test_error_message_mentions_known_brands(self):
        """ValueError message lists the valid brand options."""
        with self.assertRaises(ValueError) as ctx:
            get_settings_schema("nope")
        msg = str(ctx.exception)
        for brand in ("unitree", "bostiondynamics", "xiaomi"):
            self.assertIn(brand, msg)


# ══════════════════════════════════════════════════════════════════════════
# I — Return value is a fresh dict
# ══════════════════════════════════════════════════════════════════════════

class TestReturnIsFreshDict(unittest.TestCase):
    """Mutating the returned schema must not affect subsequent calls."""

    def test_adding_key_does_not_persist(self):
        s1 = get_settings_schema("unitree")
        s1["__test_key__"] = "injected"
        s2 = get_settings_schema("unitree")
        self.assertNotIn("__test_key__", s2)

    def test_deleting_key_does_not_persist(self):
        s1 = get_settings_schema("unitree")
        del s1["brand"]
        s2 = get_settings_schema("unitree")
        self.assertIn("brand", s2)

    def test_two_calls_return_equal_dicts(self):
        s1 = get_settings_schema("bostiondynamics")
        s2 = get_settings_schema("bostiondynamics")
        self.assertEqual(set(s1.keys()), set(s2.keys()))


# ══════════════════════════════════════════════════════════════════════════
# J — Specific required/default/choices spot-checks (§8.3–§8.4)
# ══════════════════════════════════════════════════════════════════════════

class TestSpotChecks(unittest.TestCase):
    """Exact values for specific critical entries from README tables."""

    # Global — all required=True, default=None except where noted
    def test_global_brand_required_no_default(self):
        e = get_settings_schema("unitree")["brand"]
        self.assertTrue(e["required"])
        self.assertIsNone(e["default"])

    def test_global_timeout_type_int(self):
        for brand in KNOWN_BRANDS:
            e = get_settings_schema(brand)["timeout"]
            self.assertIs(e["type"], int)
            self.assertTrue(e["required"])

    def test_global_safety_enabled_type_bool(self):
        for brand in KNOWN_BRANDS:
            e = get_settings_schema(brand)["safety_enabled"]
            self.assertIs(e["type"], bool)
            self.assertTrue(e["required"])

    # Spot — timesync_required and lease_required have defaults
    def test_spot_timesync_default_true(self):
        e = get_settings_schema("bostiondynamics")["timesync_required"]
        self.assertFalse(e["required"])
        self.assertTrue(e["default"])

    def test_spot_lease_default_true(self):
        e = get_settings_schema("bostiondynamics")["lease_required"]
        self.assertFalse(e["required"])
        self.assertTrue(e["default"])

    # CyberDog — namespace has default=""
    def test_cyberdog_namespace_default_empty_str(self):
        e = get_settings_schema("xiaomi")["namespace"]
        self.assertFalse(e["required"])
        self.assertEqual(e["default"], "")

    # CyberDog — action_endpoints type is list
    def test_cyberdog_action_endpoints_type_list(self):
        e = get_settings_schema("xiaomi")["action_endpoints"]
        self.assertIs(e["type"], list)
        self.assertTrue(e["required"])
        self.assertIsNone(e["default"])

    # All truly-required entries have default=None
    def test_all_required_no_default_entries_have_none_default(self):
        for brand in KNOWN_BRANDS:
            schema = get_settings_schema(brand)
            for key, entry in schema.items():
                if entry["required"] and entry["default"] is None:
                    pass  # correct
                elif entry["required"] and entry["default"] is not None:
                    self.fail(
                        f"Brand {brand!r} key {key!r} is required=True "
                        f"but has non-None default {entry['default']!r}"
                    )

    def test_optional_entries_have_non_none_defaults(self):
        """Every optional (required=False) entry must have a usable default."""
        for brand in KNOWN_BRANDS:
            schema = get_settings_schema(brand)
            for key, entry in schema.items():
                if not entry["required"]:
                    # default must not be None for optional entries
                    self.assertIsNotNone(
                        entry["default"],
                        f"Brand {brand!r} key {key!r} is optional "
                        f"but has no default (default=None)",
                    )


if __name__ == "__main__":
    unittest.main()
