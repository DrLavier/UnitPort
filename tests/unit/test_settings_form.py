"""Tests for bin.core.settings_form — Cycle 2 STAGE-02.

Coverage:

    A. schema_to_form_descriptors — unitree brand
    B. schema_to_form_descriptors — bostiondynamics brand
    C. schema_to_form_descriptors — xiaomi brand
    D. Field order policy (global first, then brand-specific)
    E. Deterministic ordering (stable across repeated calls)
    F. Widget type derivation (text / choice / integer / float / bool / list)
    G. Label generation (title-case fallback + override cases)
    H. Descriptor structure (all nine keys present, types correct)
    I. Required/optional group assignment
    J. Schema evolution — additive safety (unknown extra field is passed through)
    K. Unknown brand raises ValueError
    L. Uniqueness — no duplicate field_ids in output
    M. Order values — sequential from 0
    N. WIDGET_TYPES constant completeness
    O. descriptor_for_field — standalone usage
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.core.settings_form import (
    WIDGET_TYPES,
    descriptor_for_field,
    schema_to_form_descriptors,
    _GLOBAL_FIELD_ORDER,
    _LABEL_OVERRIDES,
    _validation_hints,
)
from system.service.settings_schema import KNOWN_BRANDS, get_settings_schema


# ── Helpers ──────────────────────────────────────────────────────────────────

_DESCRIPTOR_KEYS = frozenset(
    {"field_id", "label", "widget_type", "required", "default",
     "choices", "description", "order", "group", "validation_hints"}
)

_GLOBAL_KEY_COUNT = len(_GLOBAL_FIELD_ORDER)  # 6


def _descriptors(brand: str) -> List[Dict[str, Any]]:
    return schema_to_form_descriptors(brand)


def _by_field(descriptors: List[Dict[str, Any]], field_id: str) -> Dict[str, Any]:
    return next(d for d in descriptors if d["field_id"] == field_id)


# ─────────────────────────────────────────────────────────────────────────────
# A. unitree brand
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitreeBrand(unittest.TestCase):
    def setUp(self):
        self.descs = _descriptors("unitree")

    def test_correct_field_count(self):
        expected = len(get_settings_schema("unitree"))
        self.assertEqual(len(self.descs), expected)

    def test_global_fields_present(self):
        ids = [d["field_id"] for d in self.descs]
        for fid in _GLOBAL_FIELD_ORDER:
            self.assertIn(fid, ids)

    def test_unitree_specific_fields_present(self):
        ids = [d["field_id"] for d in self.descs]
        for fid in ("dds_domain_id", "network_interface", "control_level", "mode_switch_policy"):
            self.assertIn(fid, ids)

    def test_global_fields_come_first(self):
        global_ids = [d["field_id"] for d in self.descs if d["field_id"] in _GLOBAL_FIELD_ORDER]
        self.assertEqual(global_ids, list(_GLOBAL_FIELD_ORDER))

    def test_brand_fields_come_after_global(self):
        global_positions = [d["order"] for d in self.descs if d["field_id"] in _GLOBAL_FIELD_ORDER]
        brand_positions  = [d["order"] for d in self.descs if d["field_id"] not in _GLOBAL_FIELD_ORDER]
        self.assertTrue(max(global_positions) < min(brand_positions))


# ─────────────────────────────────────────────────────────────────────────────
# B. bostiondynamics brand
# ─────────────────────────────────────────────────────────────────────────────

class TestBostionDynamicsBrand(unittest.TestCase):
    def setUp(self):
        self.descs = _descriptors("bostiondynamics")

    def test_correct_field_count(self):
        expected = len(get_settings_schema("bostiondynamics"))
        self.assertEqual(len(self.descs), expected)

    def test_spot_specific_fields_present(self):
        ids = [d["field_id"] for d in self.descs]
        for fid in ("hostname", "auth_token", "timesync_required",
                    "lease_required", "safety_channel", "power_policy"):
            self.assertIn(fid, ids)

    def test_optional_fields_have_optional_group(self):
        optionals = ("timesync_required", "lease_required")
        for fid in optionals:
            d = _by_field(self.descs, fid)
            self.assertEqual(d["group"], "optional")
            self.assertFalse(d["required"])

    def test_required_fields_have_required_group(self):
        for fid in ("hostname", "auth_token", "safety_channel", "power_policy"):
            d = _by_field(self.descs, fid)
            self.assertEqual(d["group"], "required")
            self.assertTrue(d["required"])


# ─────────────────────────────────────────────────────────────────────────────
# C. xiaomi brand
# ─────────────────────────────────────────────────────────────────────────────

class TestXiaomiBrand(unittest.TestCase):
    def setUp(self):
        self.descs = _descriptors("xiaomi")

    def test_correct_field_count(self):
        expected = len(get_settings_schema("xiaomi"))
        self.assertEqual(len(self.descs), expected)

    def test_cyberdog_specific_fields_present(self):
        ids = [d["field_id"] for d in self.descs]
        for fid in ("ros_domain_id", "rmw_impl", "namespace",
                    "action_endpoints", "timestamp_policy"):
            self.assertIn(fid, ids)

    def test_optional_namespace_field(self):
        d = _by_field(self.descs, "namespace")
        self.assertEqual(d["group"], "optional")
        self.assertFalse(d["required"])
        self.assertEqual(d["default"], "")


# ─────────────────────────────────────────────────────────────────────────────
# D & E. Field order policy + determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldOrder(unittest.TestCase):
    def test_global_order_matches_canonical_list(self):
        for brand in KNOWN_BRANDS:
            descs = _descriptors(brand)
            global_descs = [d for d in descs if d["field_id"] in _GLOBAL_FIELD_ORDER]
            ids = [d["field_id"] for d in global_descs]
            self.assertEqual(ids, list(_GLOBAL_FIELD_ORDER),
                             f"Global field order wrong for brand={brand!r}")

    def test_order_values_sequential_from_zero(self):
        for brand in KNOWN_BRANDS:
            descs = _descriptors(brand)
            orders = [d["order"] for d in descs]
            self.assertEqual(orders, list(range(len(descs))),
                             f"Non-sequential order for brand={brand!r}")

    def test_deterministic_across_calls(self):
        for brand in KNOWN_BRANDS:
            ids_first  = [d["field_id"] for d in _descriptors(brand)]
            ids_second = [d["field_id"] for d in _descriptors(brand)]
            self.assertEqual(ids_first, ids_second,
                             f"Non-deterministic order for brand={brand!r}")

    def test_no_duplicate_field_ids(self):
        for brand in KNOWN_BRANDS:
            ids = [d["field_id"] for d in _descriptors(brand)]
            self.assertEqual(len(ids), len(set(ids)),
                             f"Duplicate field_ids found for brand={brand!r}")


# ─────────────────────────────────────────────────────────────────────────────
# F. Widget type derivation
# ─────────────────────────────────────────────────────────────────────────────

class TestWidgetTypes(unittest.TestCase):
    def _entry(self, typ, choices=None, required=True, default=None):
        return {"type": typ, "required": required, "default": default,
                "description": "", "choices": choices}

    def test_str_without_choices_is_text(self):
        d = descriptor_for_field("x", self._entry(str))
        self.assertEqual(d["widget_type"], "text")

    def test_str_with_choices_is_choice(self):
        d = descriptor_for_field("x", self._entry(str, choices=["a", "b"]))
        self.assertEqual(d["widget_type"], "choice")

    def test_int_is_integer(self):
        d = descriptor_for_field("x", self._entry(int))
        self.assertEqual(d["widget_type"], "integer")

    def test_float_is_float(self):
        d = descriptor_for_field("x", self._entry(float))
        self.assertEqual(d["widget_type"], "float")

    def test_bool_is_bool(self):
        d = descriptor_for_field("x", self._entry(bool))
        self.assertEqual(d["widget_type"], "bool")

    def test_list_is_list(self):
        d = descriptor_for_field("x", self._entry(list))
        self.assertEqual(d["widget_type"], "list")

    def test_all_widget_types_in_constant(self):
        for wt in ("text", "choice", "integer", "float", "bool", "list"):
            self.assertIn(wt, WIDGET_TYPES)

    def test_real_schema_control_level_is_choice(self):
        descs = _descriptors("unitree")
        d = _by_field(descs, "control_level")
        self.assertEqual(d["widget_type"], "choice")

    def test_real_schema_robot_type_is_choice(self):
        descs = _descriptors("unitree")
        d = _by_field(descs, "robot_type")
        self.assertEqual(d["widget_type"], "choice")
        self.assertIn("go2", d.get("choices") or [])

    def test_real_schema_timeout_is_integer(self):
        descs = _descriptors("unitree")
        d = _by_field(descs, "timeout")
        self.assertEqual(d["widget_type"], "integer")

    def test_real_schema_safety_enabled_is_bool(self):
        descs = _descriptors("unitree")
        d = _by_field(descs, "safety_enabled")
        self.assertEqual(d["widget_type"], "bool")

    def test_real_schema_action_endpoints_is_list(self):
        descs = _descriptors("xiaomi")
        d = _by_field(descs, "action_endpoints")
        self.assertEqual(d["widget_type"], "list")


# ─────────────────────────────────────────────────────────────────────────────
# G. Label generation
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelGeneration(unittest.TestCase):
    def _entry(self):
        return {"type": str, "required": True, "default": None,
                "description": "", "choices": None}

    def test_label_override_dds_domain_id(self):
        d = descriptor_for_field("dds_domain_id", self._entry())
        self.assertEqual(d["label"], "DDS Domain ID")

    def test_label_override_ros_domain_id(self):
        d = descriptor_for_field("ros_domain_id", self._entry())
        self.assertEqual(d["label"], "ROS Domain ID")

    def test_label_override_rmw_impl(self):
        d = descriptor_for_field("rmw_impl", self._entry())
        self.assertEqual(d["label"], "RMW Implementation")

    def test_label_override_safety_enabled(self):
        d = descriptor_for_field("safety_enabled", self._entry())
        self.assertEqual(d["label"], "Safety Enabled")

    def test_label_fallback_hostname(self):
        d = descriptor_for_field("hostname", self._entry())
        self.assertEqual(d["label"], "Hostname")

    def test_label_fallback_connection_mode(self):
        d = descriptor_for_field("connection_mode", self._entry())
        self.assertEqual(d["label"], "Connection Mode")

    def test_label_fallback_stop_policy(self):
        d = descriptor_for_field("stop_policy", self._entry())
        self.assertEqual(d["label"], "Stop Policy")


# ─────────────────────────────────────────────────────────────────────────────
# H. Descriptor structure
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptorStructure(unittest.TestCase):
    def test_all_nine_keys_present(self):
        entry = {"type": str, "required": True, "default": None,
                 "description": "test desc", "choices": ["a"]}
        d = descriptor_for_field("some_field", entry, order=3)
        self.assertEqual(set(d.keys()), _DESCRIPTOR_KEYS)

    def test_order_value_propagated(self):
        entry = {"type": int, "required": False, "default": 5,
                 "description": "", "choices": None}
        d = descriptor_for_field("x", entry, order=7)
        self.assertEqual(d["order"], 7)

    def test_choices_preserved(self):
        choices = ["alpha", "beta", "gamma"]
        entry = {"type": str, "required": True, "default": None,
                 "description": "", "choices": choices}
        d = descriptor_for_field("x", entry)
        self.assertEqual(d["choices"], choices)

    def test_description_preserved(self):
        entry = {"type": str, "required": True, "default": None,
                 "description": "My description.", "choices": None}
        d = descriptor_for_field("x", entry)
        self.assertEqual(d["description"], "My description.")

    def test_all_descriptors_have_non_empty_description(self):
        for brand in KNOWN_BRANDS:
            for d in _descriptors(brand):
                self.assertTrue(
                    d["description"],
                    f"Empty description for field_id={d['field_id']!r} brand={brand!r}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# I. Required/optional group assignment (already partially in B)
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupAssignment(unittest.TestCase):
    def test_required_entry_group_is_required(self):
        entry = {"type": str, "required": True, "default": None,
                 "description": "", "choices": None}
        d = descriptor_for_field("x", entry)
        self.assertEqual(d["group"], "required")
        self.assertTrue(d["required"])

    def test_optional_entry_group_is_optional(self):
        entry = {"type": bool, "required": False, "default": True,
                 "description": "", "choices": None}
        d = descriptor_for_field("x", entry)
        self.assertEqual(d["group"], "optional")
        self.assertFalse(d["required"])

    def test_all_global_fields_are_required_in_all_brands(self):
        for brand in KNOWN_BRANDS:
            descs = _descriptors(brand)
            for fid in _GLOBAL_FIELD_ORDER:
                d = _by_field(descs, fid)
                self.assertTrue(d["required"],
                                f"Global field {fid!r} not required for brand={brand!r}")


# ─────────────────────────────────────────────────────────────────────────────
# J. Additive schema evolution safety
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaEvolution(unittest.TestCase):
    def test_extra_field_in_entry_is_ignored_gracefully(self):
        # A future schema entry may have extra keys — descriptor_for_field must
        # not raise, and the nine standard keys must still be present.
        entry = {
            "type": str, "required": False, "default": "x",
            "description": "future field", "choices": None,
            "future_key": "future_value",  # unknown key
        }
        d = descriptor_for_field("new_field", entry)
        self.assertEqual(set(d.keys()), _DESCRIPTOR_KEYS)

    def test_missing_optional_choices_key_defaults_to_none(self):
        entry = {"type": str, "required": True, "default": None,
                 "description": "no choices"}
        d = descriptor_for_field("x", entry)
        self.assertIsNone(d["choices"])

    def test_missing_optional_description_key_defaults_to_empty(self):
        entry = {"type": int, "required": True, "default": None}
        d = descriptor_for_field("x", entry)
        self.assertEqual(d["description"], "")


# ─────────────────────────────────────────────────────────────────────────────
# K. Unknown brand
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownBrand(unittest.TestCase):
    def test_unknown_brand_raises_value_error(self):
        with self.assertRaises(ValueError):
            schema_to_form_descriptors("unknown_brand")

    def test_empty_brand_raises_value_error(self):
        with self.assertRaises(ValueError):
            schema_to_form_descriptors("")


# ─────────────────────────────────────────────────────────────────────────────
# P. Validation hints
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationHints(unittest.TestCase):
    """Explicit validation_hints key is present in every descriptor and
    carries the correct derived content."""

    def _entry(self, typ, choices=None, required=True, default=None):
        return {"type": typ, "required": required, "default": default,
                "description": "", "choices": choices}

    # ── _validation_hints helper ─────────────────────────────────────────────

    def test_choices_hint_lists_all_values(self):
        entry = self._entry(str, choices=["a", "b", "c"])
        self.assertEqual(_validation_hints(entry), "One of: a, b, c")

    def test_choices_hint_single_value(self):
        entry = self._entry(str, choices=["only"])
        self.assertEqual(_validation_hints(entry), "One of: only")

    def test_int_hint_is_integer(self):
        self.assertEqual(_validation_hints(self._entry(int)), "Integer")

    def test_float_hint_is_decimal(self):
        self.assertEqual(_validation_hints(self._entry(float)), "Decimal number")

    def test_bool_hint_is_true_false(self):
        self.assertEqual(_validation_hints(self._entry(bool)), "true / false")

    def test_list_hint_is_list_of_values(self):
        self.assertEqual(_validation_hints(self._entry(list)), "List of values")

    def test_str_no_choices_hint_is_empty(self):
        self.assertEqual(_validation_hints(self._entry(str)), "")

    def test_choices_takes_priority_over_type(self):
        # choices present on an int field → list hint, not "Integer"
        entry = self._entry(int, choices=[0, 1, 2])
        self.assertIn("One of:", _validation_hints(entry))

    # ── descriptor_for_field includes validation_hints key ───────────────────

    def test_descriptor_has_validation_hints_key(self):
        entry = self._entry(str, choices=["x", "y"])
        d = descriptor_for_field("field", entry)
        self.assertIn("validation_hints", d)

    def test_descriptor_validation_hints_correct_for_choice(self):
        entry = self._entry(str, choices=["simulation", "hardware"])
        d = descriptor_for_field("connection_mode", entry)
        self.assertEqual(d["validation_hints"], "One of: simulation, hardware")

    def test_descriptor_validation_hints_correct_for_int(self):
        entry = self._entry(int)
        d = descriptor_for_field("timeout", entry)
        self.assertEqual(d["validation_hints"], "Integer")

    def test_descriptor_validation_hints_correct_for_bool(self):
        entry = self._entry(bool)
        d = descriptor_for_field("safety_enabled", entry)
        self.assertEqual(d["validation_hints"], "true / false")

    # ── Real schema spot-checks ───────────────────────────────────────────────

    def test_real_connection_mode_hint(self):
        descs = schema_to_form_descriptors("unitree")
        d = _by_field(descs, "connection_mode")
        self.assertEqual(d["validation_hints"], "One of: simulation, hardware")

    def test_real_timeout_hint(self):
        descs = schema_to_form_descriptors("unitree")
        d = _by_field(descs, "timeout")
        self.assertEqual(d["validation_hints"], "Integer")

    def test_real_safety_enabled_hint(self):
        descs = schema_to_form_descriptors("unitree")
        d = _by_field(descs, "safety_enabled")
        self.assertEqual(d["validation_hints"], "true / false")

    def test_real_action_endpoints_hint(self):
        descs = schema_to_form_descriptors("xiaomi")
        d = _by_field(descs, "action_endpoints")
        self.assertEqual(d["validation_hints"], "List of values")

    def test_real_hostname_hint_is_empty(self):
        # str field with no choices → unrestricted, hint is ""
        descs = schema_to_form_descriptors("bostiondynamics")
        d = _by_field(descs, "hostname")
        self.assertEqual(d["validation_hints"], "")

    def test_all_descriptors_have_string_validation_hints(self):
        for brand in KNOWN_BRANDS:
            for d in schema_to_form_descriptors(brand):
                self.assertIsInstance(
                    d["validation_hints"], str,
                    f"validation_hints not str for field_id={d['field_id']!r} brand={brand!r}",
                )


if __name__ == "__main__":
    unittest.main()
