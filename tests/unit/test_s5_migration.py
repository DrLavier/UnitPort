#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 5 — Migration handling and backward compatibility.

Coverage (pure-Python; no Qt):

  migrate_mission_payload():
  - Missing schema_version         → needs_protocol_upgrade=True
  - schema_version "unknown"       → needs_protocol_upgrade=True
  - schema_version "1.0"           → needs_protocol_upgrade=True
  - schema_version "1.3"           → needs_protocol_upgrade=True (< 1.4)
  - schema_version "1.4"           → needs_protocol_upgrade=False (up to date)
  - schema_version "1.5"           → needs_protocol_upgrade=False (newer)
  - prior_schema_version reflects actual file value ("unknown" when absent)
  - warnings list non-empty when upgrade needed; empty when current
  - warnings contain "protocol" and "condition" context
  - non-dict input → safe no-op (needs_protocol_upgrade=False, no raise)
  - returns dict with exactly three keys

  MISSION_SCHEMA_VERSION:
  - is "1.4" (Step 5 bump)

  validate_mission_schema() backward compat:
  - Old files without schema_version still pass (not a required key)
  - Old files with "nodes" + "connections" as lists still pass

  _version_lt() helper (internal — tested via migrate_mission_payload outcomes):
  - 1.3 < 1.4 → True
  - 1.4 < 1.4 → False
  - 1.4 < 1.5 → True
  - 1.10 > 1.9 → True (numeric segment comparison, not lexicographic)
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bin.core.mission_persistence import (
    MISSION_SCHEMA_VERSION,
    migrate_mission_payload,
    validate_mission_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_file(schema_version=None) -> Dict[str, Any]:
    d = {"nodes": [], "connections": []}
    if schema_version is not None:
        d["schema_version"] = schema_version
    return d


# ---------------------------------------------------------------------------
# Tests: MISSION_SCHEMA_VERSION
# ---------------------------------------------------------------------------

class TestSchemaVersionConstant(unittest.TestCase):

    def test_current_version_is_1_4(self):
        self.assertEqual(MISSION_SCHEMA_VERSION, "1.4")


# ---------------------------------------------------------------------------
# Tests: migrate_mission_payload — return shape
# ---------------------------------------------------------------------------

class TestMigrateMissionPayloadShape(unittest.TestCase):

    def test_returns_dict(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertGreaterEqual(
            set(result.keys()),
            {"needs_protocol_upgrade", "prior_schema_version", "warnings"},
        )

    def test_needs_protocol_upgrade_is_bool(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertIsInstance(result["needs_protocol_upgrade"], bool)

    def test_prior_schema_version_is_str(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertIsInstance(result["prior_schema_version"], str)

    def test_warnings_is_list(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertIsInstance(result["warnings"], list)

    def test_non_dict_input_does_not_raise(self):
        for bad in (None, [], "string", 42):
            result = migrate_mission_payload(bad)
            self.assertIsInstance(result, dict)
            self.assertFalse(result["needs_protocol_upgrade"])


# ---------------------------------------------------------------------------
# Tests: migrate_mission_payload — upgrade detection
# ---------------------------------------------------------------------------

class TestMigrateMissionPayloadDetection(unittest.TestCase):

    def test_missing_schema_version_needs_upgrade(self):
        data = {"nodes": [], "connections": []}
        result = migrate_mission_payload(data)
        self.assertTrue(result["needs_protocol_upgrade"])

    def test_missing_schema_version_prior_is_unknown(self):
        data = {"nodes": [], "connections": []}
        result = migrate_mission_payload(data)
        self.assertEqual(result["prior_schema_version"], "unknown")

    def test_version_1_0_needs_upgrade(self):
        result = migrate_mission_payload(_minimal_file("1.0"))
        self.assertTrue(result["needs_protocol_upgrade"])

    def test_version_1_1_needs_upgrade(self):
        result = migrate_mission_payload(_minimal_file("1.1"))
        self.assertTrue(result["needs_protocol_upgrade"])

    def test_version_1_2_needs_upgrade(self):
        result = migrate_mission_payload(_minimal_file("1.2"))
        self.assertTrue(result["needs_protocol_upgrade"])

    def test_version_1_3_needs_upgrade(self):
        """1.3 is the last pre-protocol version."""
        result = migrate_mission_payload(_minimal_file("1.3"))
        self.assertTrue(result["needs_protocol_upgrade"])

    def test_version_1_4_no_upgrade_needed(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertFalse(result["needs_protocol_upgrade"])

    def test_version_1_5_no_upgrade_needed(self):
        result = migrate_mission_payload(_minimal_file("1.5"))
        self.assertFalse(result["needs_protocol_upgrade"])

    def test_version_2_0_no_upgrade_needed(self):
        result = migrate_mission_payload(_minimal_file("2.0"))
        self.assertFalse(result["needs_protocol_upgrade"])

    def test_version_numeric_segment_comparison(self):
        """1.10 must be treated as greater than 1.4 (numeric, not lexicographic)."""
        result = migrate_mission_payload(_minimal_file("1.10"))
        self.assertFalse(result["needs_protocol_upgrade"])

    def test_prior_schema_version_reflects_file(self):
        result = migrate_mission_payload(_minimal_file("1.3"))
        self.assertEqual(result["prior_schema_version"], "1.3")

    def test_prior_schema_version_reflects_current(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertEqual(result["prior_schema_version"], "1.4")


# ---------------------------------------------------------------------------
# Tests: migrate_mission_payload — warnings content
# ---------------------------------------------------------------------------

class TestMigrateMissionPayloadWarnings(unittest.TestCase):

    def test_old_file_has_non_empty_warnings(self):
        result = migrate_mission_payload(_minimal_file("1.3"))
        self.assertTrue(len(result["warnings"]) > 0)

    def test_current_file_has_empty_warnings(self):
        result = migrate_mission_payload(_minimal_file("1.4"))
        self.assertEqual(result["warnings"], [])

    def test_warning_mentions_protocol(self):
        result = migrate_mission_payload(_minimal_file("1.3"))
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("protocol", combined)

    def test_warning_mentions_condition_port(self):
        result = migrate_mission_payload(_minimal_file("1.3"))
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("condition", combined)

    def test_warning_mentions_schema_version(self):
        result = migrate_mission_payload(_minimal_file("1.3"))
        combined = " ".join(result["warnings"])
        self.assertIn("1.3", combined)

    def test_missing_schema_version_warning_mentions_unknown(self):
        data = {"nodes": [], "connections": []}
        result = migrate_mission_payload(data)
        combined = " ".join(result["warnings"]).lower()
        self.assertIn("unknown", combined)


# ---------------------------------------------------------------------------
# Tests: validate_mission_schema backward compat (old files still load)
# ---------------------------------------------------------------------------

class TestValidateMissionSchemaBackwardCompat(unittest.TestCase):

    def test_old_file_without_schema_version_is_valid(self):
        data = {"nodes": [], "connections": []}
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, reason)

    def test_old_file_with_v1_0_is_valid(self):
        data = {"nodes": [], "connections": [], "schema_version": "1.0"}
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, reason)

    def test_current_version_file_is_valid(self):
        data = {"nodes": [], "connections": [], "schema_version": "1.4"}
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, reason)

    def test_extra_keys_tolerated(self):
        data = {
            "nodes": [],
            "connections": [],
            "schema_version": "1.4",
            "behavior_drafts": {},
            "behavior_timelines": {},
            "settings": {"brand": "go2", "config": {}},
            "protocol_migration_flag": True,  # hypothetical future key
        }
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, reason)

    def test_missing_nodes_is_invalid(self):
        ok, _ = validate_mission_schema({"connections": []})
        self.assertFalse(ok)

    def test_missing_connections_is_invalid(self):
        ok, _ = validate_mission_schema({"nodes": []})
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
