#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/behavior/hb_node_catalog.py — Circle 1 Step 1.4.

Coverage
--------
HBNodeAvailability
  - constants exist and are distinct strings
  - ALL tuple contains all four states

HBNodeCatalogEntry
  - field access (kind, label, availability, reason)
  - predicate methods: is_available, is_limited, is_unsupported, is_unknown
  - default reason is empty string

HBNodeCatalog.unknown()
  - returns catalog with all 5 HBNodeKind entries
  - every entry has UNKNOWN_CAPABILITY availability
  - deterministic reason string "No capability data loaded"
  - brand is empty string

HBNodeCatalog.from_capability_dict()
  - non-dict input → unknown catalog (no raise)
  - None → unknown catalog
  - empty dict → deterministic states (AVAILABLE/UNSUPPORTED/UNKNOWN_CAPABILITY)
  - Unitree caps (imu + high_level_control) → full AVAILABLE nodes
  - Spot caps (no imu, no high_level_control) → LIMITED/UNKNOWN_CAPABILITY
  - CyberDog caps (imu, no high_level_control) → AVAILABLE sensors, UNKNOWN_CAPABILITY blend
  - non-list sensors → treated as empty list (no raise)
  - non-dict flags → treated as empty dict (no raise)
  - brand key extracted correctly

HBNodeCatalog.for_kind()
  - known kind → correct entry
  - unknown kind → UNKNOWN_CAPABILITY fallback, no raise

HBNodeCatalog.all_entries()
  - returns 5 entries in canonical order
  - each entry has correct kind

Specific node rules
  - heartbeat_source always AVAILABLE regardless of sensors/flags
  - safety_gate always AVAILABLE regardless of sensors/flags
  - sensor_read: AVAILABLE with imu, LIMITED without imu but with other sensors, UNSUPPORTED with no sensors
  - sensor_transform: AVAILABLE with imu (transform-compatible), LIMITED with non-compatible sensors,
    UNSUPPORTED with no sensors
  - movement_blend: AVAILABLE with high_level_control=True, LIMITED with =False, UNKNOWN_CAPABILITY missing

Isolation
---------
- No QApplication. No Qt imports. Pure Python only.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.hb_node_catalog import (  # noqa: E402
    HBNodeAvailability,
    HBNodeCatalog,
    HBNodeCatalogEntry,
)


# ===========================================================================
# Capability dict fixtures
# ===========================================================================

def _unitree_caps() -> Dict[str, Any]:
    """Unitree adapter capability dict (imu present, high_level_control=True)."""
    return {
        "brand":   "unitree",
        "adapter": "unitree_sdk2",
        "actions": ["stand", "sit", "walk", "stop"],
        "sensors": ["battery", "imu", "foot_contact", "qpos", "qvel", "time"],
        "flags": {
            "sdk_available":      True,
            "mujoco_available":   False,
            "high_level_control": True,
            "robot_type":         "go2",
        },
        "required_settings": ["brand", "robot_type", "connection_mode"],
    }


def _spot_caps() -> Dict[str, Any]:
    """Spot adapter capability dict (no imu, no high_level_control flag)."""
    return {
        "brand":   "bostiondynamics",
        "adapter": "spot_sdk",
        "actions": ["stand", "sit", "walk", "stop"],
        "sensors": ["battery", "estop", "power_state"],
        "flags": {
            "sdk_available":     True,
            "lease_required":    True,
            "timesync_required": True,
            "session_open":      False,
            "hostname":          "192.168.1.1",
        },
        "required_settings": ["brand", "robot_type", "hostname"],
    }


def _cyberdog_caps() -> Dict[str, Any]:
    """CyberDog adapter capability dict (imu present, no high_level_control flag)."""
    return {
        "brand":   "xiaomi",
        "adapter": "cyberdog_sdk",
        "actions": ["stand", "sit", "walk", "stop"],
        "sensors": ["battery", "imu", "odometry"],
        "flags": {
            "ros2_available":     True,
            "ros2_required":      True,
            "namespace_isolated": True,
            "session_open":       False,
            "namespace":          "cyberdog",
        },
        "required_settings": ["brand", "robot_type", "ros_domain_id"],
    }


def _no_sensor_caps() -> Dict[str, Any]:
    """Minimal capability dict with no sensors declared."""
    return {
        "brand":   "generic",
        "adapter": "generic_adapter",
        "actions": [],
        "sensors": [],
        "flags":   {"high_level_control": True},
        "required_settings": [],
    }


def _no_imu_but_sensors_caps() -> Dict[str, Any]:
    """Capability dict with sensors but no IMU and no high_level_control."""
    return {
        "brand":   "legacy",
        "adapter": "legacy_adapter",
        "actions": ["stand", "walk"],
        "sensors": ["battery", "odometry"],
        "flags":   {},
        "required_settings": [],
    }


# ===========================================================================
# HBNodeAvailability
# ===========================================================================

class TestHBNodeAvailability(unittest.TestCase):

    def test_available_constant(self):
        self.assertEqual(HBNodeAvailability.AVAILABLE, "available")

    def test_limited_constant(self):
        self.assertEqual(HBNodeAvailability.LIMITED, "limited")

    def test_unsupported_constant(self):
        self.assertEqual(HBNodeAvailability.UNSUPPORTED, "unsupported")

    def test_unknown_capability_constant(self):
        self.assertEqual(HBNodeAvailability.UNKNOWN_CAPABILITY, "unknown_capability")

    def test_all_tuple_has_four_states(self):
        self.assertEqual(len(HBNodeAvailability.ALL), 4)

    def test_all_tuple_contains_all_states(self):
        expected = {
            HBNodeAvailability.AVAILABLE,
            HBNodeAvailability.LIMITED,
            HBNodeAvailability.UNSUPPORTED,
            HBNodeAvailability.UNKNOWN_CAPABILITY,
        }
        self.assertEqual(set(HBNodeAvailability.ALL), expected)

    def test_all_states_are_distinct_strings(self):
        states = list(HBNodeAvailability.ALL)
        self.assertEqual(len(states), len(set(states)))

    def test_all_states_are_lowercase_snake_case(self):
        for state in HBNodeAvailability.ALL:
            self.assertRegex(state, r"^[a-z][a-z0-9_]*$",
                             msg=f"State '{state}' is not lowercase snake_case")


# ===========================================================================
# HBNodeCatalogEntry
# ===========================================================================

class TestHBNodeCatalogEntry(unittest.TestCase):

    def test_fields_accessible(self):
        e = HBNodeCatalogEntry(
            kind="sensor_read",
            label="Sensor Read",
            availability=HBNodeAvailability.AVAILABLE,
            reason="",
        )
        self.assertEqual(e.kind, "sensor_read")
        self.assertEqual(e.label, "Sensor Read")
        self.assertEqual(e.availability, HBNodeAvailability.AVAILABLE)
        self.assertEqual(e.reason, "")

    def test_reason_default_empty_string(self):
        e = HBNodeCatalogEntry(
            kind="heartbeat_source",
            label="Source",
            availability=HBNodeAvailability.AVAILABLE,
        )
        self.assertEqual(e.reason, "")

    def test_is_available_true(self):
        e = HBNodeCatalogEntry("k", "L", HBNodeAvailability.AVAILABLE)
        self.assertTrue(e.is_available())
        self.assertFalse(e.is_limited())
        self.assertFalse(e.is_unsupported())
        self.assertFalse(e.is_unknown())

    def test_is_limited_true(self):
        e = HBNodeCatalogEntry("k", "L", HBNodeAvailability.LIMITED)
        self.assertFalse(e.is_available())
        self.assertTrue(e.is_limited())
        self.assertFalse(e.is_unsupported())
        self.assertFalse(e.is_unknown())

    def test_is_unsupported_true(self):
        e = HBNodeCatalogEntry("k", "L", HBNodeAvailability.UNSUPPORTED)
        self.assertFalse(e.is_available())
        self.assertFalse(e.is_limited())
        self.assertTrue(e.is_unsupported())
        self.assertFalse(e.is_unknown())

    def test_is_unknown_true(self):
        e = HBNodeCatalogEntry("k", "L", HBNodeAvailability.UNKNOWN_CAPABILITY)
        self.assertFalse(e.is_available())
        self.assertFalse(e.is_limited())
        self.assertFalse(e.is_unsupported())
        self.assertTrue(e.is_unknown())

    def test_reason_stored_correctly(self):
        e = HBNodeCatalogEntry("k", "L", HBNodeAvailability.UNSUPPORTED, "IMU missing")
        self.assertEqual(e.reason, "IMU missing")


# ===========================================================================
# HBNodeCatalog.unknown()
# ===========================================================================

class TestHBNodeCatalogUnknown(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.unknown()

    def test_returns_catalog_instance(self):
        self.assertIsInstance(self.catalog, HBNodeCatalog)

    def test_has_five_entries(self):
        self.assertEqual(len(self.catalog.all_entries()), 5)

    def test_every_entry_is_unknown_capability(self):
        for entry in self.catalog.all_entries():
            self.assertEqual(
                entry.availability,
                HBNodeAvailability.UNKNOWN_CAPABILITY,
                msg=f"Entry '{entry.kind}' should be UNKNOWN_CAPABILITY",
            )

    def test_all_expected_kinds_present(self):
        kinds = {e.kind for e in self.catalog.all_entries()}
        expected = {
            "heartbeat_source", "sensor_read", "sensor_transform",
            "movement_blend", "safety_gate",
        }
        self.assertEqual(kinds, expected)

    def test_brand_is_empty_string(self):
        self.assertEqual(self.catalog.brand, "")

    def test_every_entry_has_non_empty_reason(self):
        for entry in self.catalog.all_entries():
            self.assertTrue(
                entry.reason,
                msg=f"Entry '{entry.kind}' should have a reason in unknown catalog",
            )

    def test_multiple_calls_return_independent_catalogs(self):
        c1 = HBNodeCatalog.unknown()
        c2 = HBNodeCatalog.unknown()
        # Modifying c1's internal entries should not affect c2
        c1.all_entries()[0].reason = "MUTATED"
        self.assertNotEqual(c2.all_entries()[0].reason, "MUTATED")


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — edge cases
# ===========================================================================

class TestHBNodeCatalogFromDictEdgeCases(unittest.TestCase):

    def test_none_input_returns_unknown(self):
        catalog = HBNodeCatalog.from_capability_dict(None)
        self.assertIsInstance(catalog, HBNodeCatalog)
        for entry in catalog.all_entries():
            self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_string_input_returns_unknown(self):
        catalog = HBNodeCatalog.from_capability_dict("bad_input")
        for entry in catalog.all_entries():
            self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_list_input_returns_unknown(self):
        catalog = HBNodeCatalog.from_capability_dict(["sensors", "imu"])
        for entry in catalog.all_entries():
            self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_empty_dict_does_not_raise(self):
        catalog = HBNodeCatalog.from_capability_dict({})
        self.assertIsInstance(catalog, HBNodeCatalog)

    def test_empty_dict_has_five_entries(self):
        catalog = HBNodeCatalog.from_capability_dict({})
        self.assertEqual(len(catalog.all_entries()), 5)

    def test_non_list_sensors_treated_as_empty(self):
        catalog = HBNodeCatalog.from_capability_dict({"sensors": "imu", "flags": {}})
        # sensors treated as empty → sensor_read should be UNSUPPORTED
        entry = catalog.for_kind("sensor_read")
        self.assertEqual(entry.availability, HBNodeAvailability.UNSUPPORTED)

    def test_non_dict_flags_treated_as_empty(self):
        catalog = HBNodeCatalog.from_capability_dict({
            "sensors": ["imu"],
            "flags": "bad",
        })
        # flags is bad → movement_blend should be UNKNOWN_CAPABILITY
        entry = catalog.for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_brand_extracted_from_dict(self):
        catalog = HBNodeCatalog.from_capability_dict({"brand": "unitree", "sensors": []})
        self.assertEqual(catalog.brand, "unitree")

    def test_missing_brand_key_defaults_to_empty_string(self):
        catalog = HBNodeCatalog.from_capability_dict({"sensors": ["imu"]})
        self.assertEqual(catalog.brand, "")

    def test_extra_keys_are_ignored(self):
        catalog = HBNodeCatalog.from_capability_dict({
            "sensors": ["imu"],
            "flags": {"high_level_control": True},
            "unknown_future_key": "some_value",
        })
        # Should not raise; catalog built normally
        self.assertEqual(len(catalog.all_entries()), 5)


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — Unitree (full support)
# ===========================================================================

class TestHBNodeCatalogUnitree(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.from_capability_dict(_unitree_caps())

    def test_heartbeat_source_available(self):
        self.assertEqual(
            self.catalog.for_kind("heartbeat_source").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_read_available(self):
        # Unitree has IMU → fully available
        self.assertEqual(
            self.catalog.for_kind("sensor_read").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_transform_available(self):
        # Unitree has imu → transform-compatible
        self.assertEqual(
            self.catalog.for_kind("sensor_transform").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_movement_blend_available(self):
        # Unitree flags high_level_control=True
        self.assertEqual(
            self.catalog.for_kind("movement_blend").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_safety_gate_available(self):
        self.assertEqual(
            self.catalog.for_kind("safety_gate").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_brand_extracted(self):
        self.assertEqual(self.catalog.brand, "unitree")

    def test_available_entries_have_empty_reason(self):
        for entry in self.catalog.all_entries():
            if entry.availability == HBNodeAvailability.AVAILABLE:
                self.assertEqual(
                    entry.reason, "",
                    msg=f"Entry '{entry.kind}' is AVAILABLE but has non-empty reason: {entry.reason!r}",
                )


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — Spot (no IMU, no high_level_control)
# ===========================================================================

class TestHBNodeCatalogSpot(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.from_capability_dict(_spot_caps())

    def test_heartbeat_source_available(self):
        # Always available regardless of sensors
        self.assertEqual(
            self.catalog.for_kind("heartbeat_source").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_read_limited(self):
        # Spot has sensors but no IMU → LIMITED
        entry = self.catalog.for_kind("sensor_read")
        self.assertEqual(entry.availability, HBNodeAvailability.LIMITED)
        self.assertTrue(entry.reason, "LIMITED entry should have a reason")

    def test_sensor_transform_limited(self):
        # Spot has no transform-compatible sensors but has some sensors → LIMITED
        entry = self.catalog.for_kind("sensor_transform")
        self.assertEqual(entry.availability, HBNodeAvailability.LIMITED)

    def test_movement_blend_unknown_capability(self):
        # Spot flags have no high_level_control key
        entry = self.catalog.for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)
        self.assertIn("high_level_control", entry.reason)

    def test_safety_gate_available(self):
        self.assertEqual(
            self.catalog.for_kind("safety_gate").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_brand_extracted(self):
        self.assertEqual(self.catalog.brand, "bostiondynamics")


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — CyberDog (imu, no high_level_control)
# ===========================================================================

class TestHBNodeCatalogCyberDog(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.from_capability_dict(_cyberdog_caps())

    def test_heartbeat_source_available(self):
        self.assertEqual(
            self.catalog.for_kind("heartbeat_source").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_read_available(self):
        # CyberDog has IMU → fully available
        self.assertEqual(
            self.catalog.for_kind("sensor_read").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_transform_available(self):
        # CyberDog has IMU → transform-compatible
        self.assertEqual(
            self.catalog.for_kind("sensor_transform").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_movement_blend_unknown_capability(self):
        # CyberDog flags have no high_level_control key
        entry = self.catalog.for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_safety_gate_available(self):
        self.assertEqual(
            self.catalog.for_kind("safety_gate").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_brand_extracted(self):
        self.assertEqual(self.catalog.brand, "xiaomi")


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — no sensors at all
# ===========================================================================

class TestHBNodeCatalogNoSensors(unittest.TestCase):

    def setUp(self):
        # high_level_control=True but no sensors
        self.catalog = HBNodeCatalog.from_capability_dict(_no_sensor_caps())

    def test_heartbeat_source_available(self):
        self.assertEqual(
            self.catalog.for_kind("heartbeat_source").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_sensor_read_unsupported(self):
        self.assertEqual(
            self.catalog.for_kind("sensor_read").availability,
            HBNodeAvailability.UNSUPPORTED,
        )

    def test_sensor_transform_unsupported(self):
        self.assertEqual(
            self.catalog.for_kind("sensor_transform").availability,
            HBNodeAvailability.UNSUPPORTED,
        )

    def test_movement_blend_available_despite_no_sensors(self):
        # high_level_control=True in flags → AVAILABLE (movement blend is flag-based)
        self.assertEqual(
            self.catalog.for_kind("movement_blend").availability,
            HBNodeAvailability.AVAILABLE,
        )

    def test_safety_gate_available(self):
        self.assertEqual(
            self.catalog.for_kind("safety_gate").availability,
            HBNodeAvailability.AVAILABLE,
        )


# ===========================================================================
# HBNodeCatalog.from_capability_dict() — sensors without IMU, no control flag
# ===========================================================================

class TestHBNodeCatalogPartialSensors(unittest.TestCase):

    def setUp(self):
        # sensors = ["battery", "odometry"]; flags = {} (no high_level_control)
        self.catalog = HBNodeCatalog.from_capability_dict(_no_imu_but_sensors_caps())

    def test_sensor_read_limited(self):
        entry = self.catalog.for_kind("sensor_read")
        self.assertEqual(entry.availability, HBNodeAvailability.LIMITED)
        self.assertTrue(entry.reason)

    def test_sensor_transform_available_via_odometry(self):
        # odometry IS in _SENSOR_TRANSFORM_COMPATIBLE → AVAILABLE even without imu
        entry = self.catalog.for_kind("sensor_transform")
        self.assertEqual(entry.availability, HBNodeAvailability.AVAILABLE)

    def test_movement_blend_unknown(self):
        entry = self.catalog.for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)


# ===========================================================================
# HBNodeCatalog.for_kind()
# ===========================================================================

class TestHBNodeCatalogForKind(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.from_capability_dict(_unitree_caps())

    def test_known_kind_returns_correct_entry(self):
        entry = self.catalog.for_kind("sensor_read")
        self.assertEqual(entry.kind, "sensor_read")

    def test_known_kind_has_correct_label(self):
        entry = self.catalog.for_kind("heartbeat_source")
        self.assertEqual(entry.label, "Heartbeat Source")

    def test_unknown_kind_returns_unknown_capability(self):
        entry = self.catalog.for_kind("nonexistent_kind_xyz")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_unknown_kind_does_not_raise(self):
        # Must not raise for any string input
        try:
            entry = self.catalog.for_kind("")
            self.assertIsInstance(entry, HBNodeCatalogEntry)
        except Exception as exc:
            self.fail(f"for_kind raised {exc!r} for empty string input")

    def test_unknown_kind_returns_entry_with_input_kind(self):
        entry = self.catalog.for_kind("mystery_node")
        self.assertEqual(entry.kind, "mystery_node")

    def test_all_five_known_kinds_return_entries(self):
        known = ["heartbeat_source", "sensor_read", "sensor_transform",
                 "movement_blend", "safety_gate"]
        for kind in known:
            entry = self.catalog.for_kind(kind)
            self.assertEqual(entry.kind, kind,
                             msg=f"for_kind('{kind}') returned wrong kind")


# ===========================================================================
# HBNodeCatalog.all_entries()
# ===========================================================================

class TestHBNodeCatalogAllEntries(unittest.TestCase):

    def setUp(self):
        self.catalog = HBNodeCatalog.from_capability_dict(_unitree_caps())

    def test_returns_five_entries(self):
        self.assertEqual(len(self.catalog.all_entries()), 5)

    def test_canonical_order_first_is_heartbeat_source(self):
        self.assertEqual(self.catalog.all_entries()[0].kind, "heartbeat_source")

    def test_canonical_order_last_is_safety_gate(self):
        self.assertEqual(self.catalog.all_entries()[-1].kind, "safety_gate")

    def test_all_expected_kinds_present(self):
        kinds = {e.kind for e in self.catalog.all_entries()}
        expected = {
            "heartbeat_source", "sensor_read", "sensor_transform",
            "movement_blend", "safety_gate",
        }
        self.assertEqual(kinds, expected)

    def test_every_entry_has_kind_string(self):
        for entry in self.catalog.all_entries():
            self.assertIsInstance(entry.kind, str)
            self.assertTrue(entry.kind)

    def test_every_entry_has_label_string(self):
        for entry in self.catalog.all_entries():
            self.assertIsInstance(entry.label, str)
            self.assertTrue(entry.label)

    def test_every_entry_has_valid_availability(self):
        valid = set(HBNodeAvailability.ALL)
        for entry in self.catalog.all_entries():
            self.assertIn(
                entry.availability, valid,
                msg=f"Entry '{entry.kind}' has invalid availability: {entry.availability!r}",
            )

    def test_unknown_catalog_canonical_order(self):
        catalog = HBNodeCatalog.unknown()
        expected_order = [
            "heartbeat_source", "sensor_read", "sensor_transform",
            "movement_blend", "safety_gate",
        ]
        actual_order = [e.kind for e in catalog.all_entries()]
        self.assertEqual(actual_order, expected_order)


# ===========================================================================
# Invariant: heartbeat_source and safety_gate always AVAILABLE
# ===========================================================================

class TestHBNodeCatalogInvariants(unittest.TestCase):
    """heartbeat_source and safety_gate must be AVAILABLE for every capability profile."""

    _PROFILES = [
        ("unitree", _unitree_caps()),
        ("spot", _spot_caps()),
        ("cyberdog", _cyberdog_caps()),
        ("no_sensors", _no_sensor_caps()),
        ("partial_sensors", _no_imu_but_sensors_caps()),
        ("empty_dict", {}),
    ]

    def test_heartbeat_source_always_available(self):
        for name, cap in self._PROFILES:
            catalog = HBNodeCatalog.from_capability_dict(cap)
            entry = catalog.for_kind("heartbeat_source")
            self.assertEqual(
                entry.availability, HBNodeAvailability.AVAILABLE,
                msg=f"heartbeat_source not AVAILABLE for profile '{name}'",
            )

    def test_safety_gate_always_available(self):
        for name, cap in self._PROFILES:
            catalog = HBNodeCatalog.from_capability_dict(cap)
            entry = catalog.for_kind("safety_gate")
            self.assertEqual(
                entry.availability, HBNodeAvailability.AVAILABLE,
                msg=f"safety_gate not AVAILABLE for profile '{name}'",
            )


# ===========================================================================
# Deterministic fallback: no silent hide
# ===========================================================================

class TestHBNodeCatalogNoSilentHide(unittest.TestCase):
    """Nodes must never be absent from the catalog, even for unsupported profiles."""

    _BAD_INPUTS = [None, "string", 42, [], {}, {"sensors": []}, {"sensors": [], "flags": {}}]

    def test_all_entries_always_present_for_any_input(self):
        expected_kinds = {
            "heartbeat_source", "sensor_read", "sensor_transform",
            "movement_blend", "safety_gate",
        }
        for inp in self._BAD_INPUTS:
            catalog = HBNodeCatalog.from_capability_dict(inp)
            kinds = {e.kind for e in catalog.all_entries()}
            self.assertEqual(
                kinds, expected_kinds,
                msg=f"Missing kinds for input {inp!r}: {expected_kinds - kinds}",
            )

    def test_from_dict_never_raises(self):
        bad_inputs = [None, "x", [], 0, {}, {"sensors": "bad"}, {"flags": []},
                      {"sensors": [1, 2, 3]}, {"flags": {"key": None}}]
        for inp in bad_inputs:
            try:
                HBNodeCatalog.from_capability_dict(inp)
            except Exception as exc:
                self.fail(f"from_capability_dict raised {exc!r} for input {inp!r}")


# ===========================================================================
# movement_blend flag variations
# ===========================================================================

class TestMovementBlendFlagVariations(unittest.TestCase):

    def test_high_level_control_true_yields_available(self):
        cap = {"sensors": [], "flags": {"high_level_control": True}}
        entry = HBNodeCatalog.from_capability_dict(cap).for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.AVAILABLE)

    def test_high_level_control_false_yields_limited(self):
        cap = {"sensors": [], "flags": {"high_level_control": False}}
        entry = HBNodeCatalog.from_capability_dict(cap).for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.LIMITED)

    def test_missing_high_level_control_yields_unknown(self):
        cap = {"sensors": ["imu"], "flags": {"other_flag": True}}
        entry = HBNodeCatalog.from_capability_dict(cap).for_kind("movement_blend")
        self.assertEqual(entry.availability, HBNodeAvailability.UNKNOWN_CAPABILITY)

    def test_limited_entry_has_reason(self):
        cap = {"sensors": [], "flags": {"high_level_control": False}}
        entry = HBNodeCatalog.from_capability_dict(cap).for_kind("movement_blend")
        self.assertTrue(entry.reason, "LIMITED entry should have a reason")


# ===========================================================================
# robot_type extraction — compliance gap fix
# ===========================================================================

class TestHBNodeCatalogRobotType(unittest.TestCase):
    """HBNodeCatalog.robot_type extraction: top-level → flags → empty string."""

    def test_unknown_catalog_has_empty_robot_type(self):
        """unknown() sets robot_type to empty string."""
        cat = HBNodeCatalog.unknown()
        self.assertEqual(cat.robot_type, "")

    def test_robot_type_from_top_level_key(self):
        """Top-level cap['robot_type'] is used directly."""
        cap = {"sensors": [], "flags": {}, "robot_type": "go2"}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "go2")

    def test_robot_type_from_flags_fallback(self):
        """When top-level robot_type absent, flags['robot_type'] is used."""
        cap = {"sensors": [], "flags": {"robot_type": "h1"}}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "h1")

    def test_top_level_robot_type_takes_priority_over_flags(self):
        """Explicit top-level robot_type wins over flags['robot_type']."""
        cap = {
            "sensors": [],
            "flags": {"robot_type": "flags_type"},
            "robot_type": "top_type",
        }
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "top_type")

    def test_robot_type_empty_when_both_absent(self):
        """Empty string when neither top-level nor flags['robot_type'] is present."""
        cap = {"sensors": [], "flags": {}}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "")

    def test_robot_type_empty_on_non_dict_input(self):
        """Non-dict input falls back to unknown() — robot_type is empty."""
        cat = HBNodeCatalog.from_capability_dict(None)
        self.assertEqual(cat.robot_type, "")

    def test_robot_type_empty_string_value_falls_through_to_flags(self):
        """Empty string top-level robot_type falls through to flags value."""
        cap = {"sensors": [], "flags": {"robot_type": "spot"}, "robot_type": ""}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "spot")

    def test_robot_type_none_value_falls_through_to_flags(self):
        """None top-level robot_type falls through to flags value."""
        cap = {"sensors": [], "flags": {"robot_type": "cyberdog"}, "robot_type": None}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "cyberdog")

    def test_robot_type_non_dict_flags_yields_empty(self):
        """Non-dict flags → no flags extraction; robot_type is empty."""
        cap = {"sensors": [], "flags": "bad", "robot_type": None}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "")

    def test_robot_type_and_brand_both_populated(self):
        """robot_type and brand are stored independently."""
        cap = {
            "brand": "unitree",
            "robot_type": "go2",
            "sensors": ["imu"],
            "flags": {"high_level_control": True},
        }
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.brand, "unitree")
        self.assertEqual(cat.robot_type, "go2")

    def test_build_entries_sets_robot_type_empty(self):
        """Internal _build_entries sets robot_type to empty string (set by from_capability_dict)."""
        # Verify via the public factory that robot_type defaults to "" when not in cap
        cap = {"sensors": ["imu"], "flags": {"high_level_control": True}}
        cat = HBNodeCatalog.from_capability_dict(cap)
        self.assertEqual(cat.robot_type, "")


if __name__ == "__main__":
    unittest.main()
