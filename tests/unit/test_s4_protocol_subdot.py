#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 4 — Behavior Node sub_dot protocol contract.

Coverage (pure-Python model only; no Qt):

  validate_protocol_structure():
  - None / non-dict → invalid, appropriate error
  - Missing required top-level fields → invalid
  - Wrong protocol_version → invalid
  - Empty trace_id → invalid
  - Negative timestamp_ms → invalid
  - Bad mode → invalid
  - Non-positive max_age_ms → invalid
  - Empty targets list → invalid

  Runtime → border state mapping (_protocol_state_from_node_result):
  - protocol_status "invalid" → protocol_invalid
  - protocol_status "stale"   → protocol_invalid
  - reason "INVALID_PROTOCOL" → protocol_invalid (regardless of protocol_status)
  - protocol_status "valid"   → protocol_valid
  - protocol_status "absent"  → protocol_none
  - missing protocol_status   → protocol_none
  - empty node result dict    → protocol_none

  _PROTOCOL_BORDER_COLORS contract:
  - all three keys present and non-empty hex strings
  - protocol_invalid distinct from protocol_valid
  - protocol_none is neither green nor amber
  - Valid minimal payload → (True, [])
  - Per-target required field missing → invalid + message contains field name
  - Invalid target dtype → invalid
  - Invalid target shape → invalid
  - Invalid target source kind → invalid
  - Constant source missing constant field → invalid
  - External source missing external_key → invalid
  - Multiple targets, all valid → (True, [])
  - Does NOT check staleness (timestamp may be 0) → still valid if structure OK

  ProtocolDiagCode constants:
  - INVALID_SCHEMA, STALE, UNKNOWN_ADDRESS, CLAMPED_VALUE, VALID all defined

  Regression guard:
  - validate_protocol_structure is exported from motor_weight_protocol module
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.motor_weight_protocol import (
    ProtocolDiagCode,
    validate_protocol_structure,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal valid payload
# ---------------------------------------------------------------------------

def _minimal_valid() -> Dict[str, Any]:
    return {
        "protocol_version": "1.0",
        "trace_id": "test-trace-001",
        "timestamp_ms": 0,
        "mode": "constant",
        "max_age_ms": 5000,
        "targets": [
            {
                "motor_key": "leg_fl",
                "param_key": "amplitude",
                "address": 0,
                "dtype": "float32",
                "shape": [1],
                "source": {"kind": "constant", "constant": 0.5},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests: ProtocolDiagCode constants
# ---------------------------------------------------------------------------

class TestProtocolDiagCodeConstants(unittest.TestCase):

    def test_invalid_schema(self):
        self.assertEqual(ProtocolDiagCode.INVALID_SCHEMA, "protocol.invalid_schema")

    def test_stale(self):
        self.assertEqual(ProtocolDiagCode.STALE, "protocol.stale")

    def test_unknown_address(self):
        self.assertEqual(ProtocolDiagCode.UNKNOWN_ADDRESS, "protocol.unknown_address")

    def test_clamped_value(self):
        self.assertEqual(ProtocolDiagCode.CLAMPED_VALUE, "protocol.clamped_value")

    def test_valid(self):
        self.assertEqual(ProtocolDiagCode.VALID, "protocol.valid")


# ---------------------------------------------------------------------------
# Tests: validate_protocol_structure — bad types / None
# ---------------------------------------------------------------------------

class TestValidateProtocolStructureRejections(unittest.TestCase):

    def test_none_is_invalid(self):
        ok, errors = validate_protocol_structure(None)
        self.assertFalse(ok)
        self.assertTrue(any("JSON object" in e or "NoneType" in e for e in errors),
                        msg=f"Expected type error, got: {errors}")

    def test_list_is_invalid(self):
        ok, errors = validate_protocol_structure([])
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_string_is_invalid(self):
        ok, errors = validate_protocol_structure("payload")
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_int_is_invalid(self):
        ok, errors = validate_protocol_structure(42)
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_empty_dict_missing_all_fields(self):
        ok, errors = validate_protocol_structure({})
        self.assertFalse(ok)
        self.assertTrue(len(errors) >= 1)

    def test_missing_protocol_version(self):
        p = _minimal_valid()
        del p["protocol_version"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("protocol_version" in e for e in errors))

    def test_missing_trace_id(self):
        p = _minimal_valid()
        del p["trace_id"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("trace_id" in e for e in errors))

    def test_missing_timestamp_ms(self):
        p = _minimal_valid()
        del p["timestamp_ms"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("timestamp_ms" in e for e in errors))

    def test_missing_mode(self):
        p = _minimal_valid()
        del p["mode"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("mode" in e for e in errors))

    def test_missing_max_age_ms(self):
        p = _minimal_valid()
        del p["max_age_ms"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("max_age_ms" in e for e in errors))

    def test_missing_targets(self):
        p = _minimal_valid()
        del p["targets"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("targets" in e for e in errors))

    def test_wrong_protocol_version(self):
        p = _minimal_valid()
        p["protocol_version"] = "2.0"
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("protocol_version" in e or "unsupported" in e for e in errors))

    def test_empty_trace_id(self):
        p = _minimal_valid()
        p["trace_id"] = ""
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("trace_id" in e for e in errors))

    def test_negative_timestamp_ms(self):
        p = _minimal_valid()
        p["timestamp_ms"] = -1
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("timestamp_ms" in e for e in errors))

    def test_invalid_mode(self):
        p = _minimal_valid()
        p["mode"] = "continuous"
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("mode" in e for e in errors))

    def test_zero_max_age_ms(self):
        p = _minimal_valid()
        p["max_age_ms"] = 0
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("max_age_ms" in e for e in errors))

    def test_empty_targets_list(self):
        p = _minimal_valid()
        p["targets"] = []
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("targets" in e for e in errors))

    def test_targets_not_a_list(self):
        p = _minimal_valid()
        p["targets"] = "leg_fl"
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(errors)


# ---------------------------------------------------------------------------
# Tests: validate_protocol_structure — target-level validation
# ---------------------------------------------------------------------------

class TestValidateProtocolStructureTargets(unittest.TestCase):

    def test_missing_motor_key(self):
        p = _minimal_valid()
        del p["targets"][0]["motor_key"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("motor_key" in e for e in errors))

    def test_missing_param_key(self):
        p = _minimal_valid()
        del p["targets"][0]["param_key"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("param_key" in e for e in errors))

    def test_missing_dtype(self):
        p = _minimal_valid()
        del p["targets"][0]["dtype"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("dtype" in e for e in errors))

    def test_invalid_dtype(self):
        p = _minimal_valid()
        p["targets"][0]["dtype"] = "int32"
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("dtype" in e for e in errors))

    def test_invalid_shape_empty(self):
        p = _minimal_valid()
        p["targets"][0]["shape"] = []
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("shape" in e for e in errors))

    def test_invalid_shape_zero_dim(self):
        p = _minimal_valid()
        p["targets"][0]["shape"] = [0]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("shape" in e for e in errors))

    def test_missing_source_field(self):
        p = _minimal_valid()
        del p["targets"][0]["source"]
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("source" in e for e in errors))

    def test_invalid_source_kind(self):
        p = _minimal_valid()
        p["targets"][0]["source"] = {"kind": "live"}
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("kind" in e or "source" in e for e in errors))

    def test_constant_source_missing_constant(self):
        p = _minimal_valid()
        p["targets"][0]["source"] = {"kind": "constant"}
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("constant" in e for e in errors))

    def test_external_source_missing_external_key(self):
        p = _minimal_valid()
        p["targets"][0]["source"] = {
            "kind": "external",
            "external_value": 0.5,
        }
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("external_key" in e for e in errors))

    def test_external_source_missing_external_value(self):
        p = _minimal_valid()
        p["targets"][0]["source"] = {
            "kind": "external",
            "external_key": "imu.pitch",
        }
        ok, errors = validate_protocol_structure(p)
        self.assertFalse(ok)
        self.assertTrue(any("external_value" in e for e in errors))


# ---------------------------------------------------------------------------
# Tests: validate_protocol_structure — valid payloads
# ---------------------------------------------------------------------------

class TestValidateProtocolStructureValid(unittest.TestCase):

    def test_minimal_valid_payload(self):
        ok, errors = validate_protocol_structure(_minimal_valid())
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_mode_external_valid(self):
        p = _minimal_valid()
        p["mode"] = "external"
        p["targets"][0]["source"] = {
            "kind": "external",
            "external_key": "imu.pitch",
            "external_value": 0.3,
        }
        ok, errors = validate_protocol_structure(p)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_multiple_targets_all_valid(self):
        p = _minimal_valid()
        p["targets"].append({
            "motor_key": "leg_fr",
            "param_key": "gain",
            "address": 1,
            "dtype": "float64",
            "shape": [1],
            "source": {"kind": "constant", "constant": 1.0},
        })
        ok, errors = validate_protocol_structure(p)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_timestamp_zero_not_stale_error(self):
        """Structural validator must NOT check staleness; timestamp=0 is OK."""
        p = _minimal_valid()
        p["timestamp_ms"] = 0
        ok, errors = validate_protocol_structure(p)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_target_with_limits_valid(self):
        p = _minimal_valid()
        p["targets"][0]["limits"] = {"min": 0.0, "max": 1.0, "clamp": True}
        ok, errors = validate_protocol_structure(p)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_returns_tuple(self):
        result = validate_protocol_structure(_minimal_valid())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        ok, errors = result
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(errors, list)


# ---------------------------------------------------------------------------
# Tests: module export contract
# ---------------------------------------------------------------------------

class TestModuleExports(unittest.TestCase):

    def test_validate_protocol_structure_callable(self):
        self.assertTrue(callable(validate_protocol_structure))

    def test_protocol_diag_code_constants_exist(self):
        required = ["INVALID_SCHEMA", "STALE", "UNKNOWN_ADDRESS", "CLAMPED_VALUE", "VALID"]
        for attr in required:
            self.assertTrue(hasattr(ProtocolDiagCode, attr), f"Missing ProtocolDiagCode.{attr}")


# ---------------------------------------------------------------------------
# Tests: runtime → border state mapping (pure-Python logic extracted for testing)
# ---------------------------------------------------------------------------

def _protocol_state_from_node_result(node_result: dict) -> str:
    """Mirror the mapping logic in apply_behavior_protocol_states_from_run_result.

    This pure-Python helper lets us unit-test the priority rules without Qt.
    """
    if not isinstance(node_result, dict):
        return "protocol_none"
    proto_status = node_result.get("protocol_status", "absent")
    reason = node_result.get("reason", "")
    if proto_status in ("invalid", "stale") or reason == "INVALID_PROTOCOL":
        return "protocol_invalid"
    if proto_status == "valid":
        return "protocol_valid"
    return "protocol_none"


class TestProtocolStateMapping(unittest.TestCase):
    """Verify runtime → border state priority rules."""

    def test_invalid_status_gives_protocol_invalid(self):
        self.assertEqual(_protocol_state_from_node_result({"protocol_status": "invalid"}), "protocol_invalid")

    def test_stale_status_gives_protocol_invalid(self):
        self.assertEqual(_protocol_state_from_node_result({"protocol_status": "stale"}), "protocol_invalid")

    def test_invalid_protocol_reason_gives_protocol_invalid(self):
        """INVALID_PROTOCOL reason overrides absent protocol_status."""
        result = {"reason": "INVALID_PROTOCOL", "protocol_status": "absent"}
        self.assertEqual(_protocol_state_from_node_result(result), "protocol_invalid")

    def test_invalid_protocol_reason_overrides_even_valid_status(self):
        """If reason=INVALID_PROTOCOL, state must be invalid regardless of other fields."""
        result = {"reason": "INVALID_PROTOCOL", "protocol_status": "valid"}
        self.assertEqual(_protocol_state_from_node_result(result), "protocol_invalid")

    def test_valid_status_gives_protocol_valid(self):
        self.assertEqual(_protocol_state_from_node_result({"protocol_status": "valid"}), "protocol_valid")

    def test_absent_status_gives_protocol_none(self):
        self.assertEqual(_protocol_state_from_node_result({"protocol_status": "absent"}), "protocol_none")

    def test_missing_protocol_status_gives_protocol_none(self):
        self.assertEqual(_protocol_state_from_node_result({}), "protocol_none")

    def test_empty_dict_gives_protocol_none(self):
        self.assertEqual(_protocol_state_from_node_result({}), "protocol_none")

    def test_non_dict_gives_protocol_none(self):
        self.assertEqual(_protocol_state_from_node_result(None), "protocol_none")

    def test_unknown_status_gives_protocol_none(self):
        self.assertEqual(_protocol_state_from_node_result({"protocol_status": "pending"}), "protocol_none")

    def test_valid_with_no_reason_gives_protocol_valid(self):
        result = {"protocol_status": "valid", "reason": ""}
        self.assertEqual(_protocol_state_from_node_result(result), "protocol_valid")


# ---------------------------------------------------------------------------
# Tests: _PROTOCOL_BORDER_COLORS contract (model-layer; no Qt)
# ---------------------------------------------------------------------------

class TestProtocolBorderColorsContract(unittest.TestCase):
    """Verify the color dict exposed by graph_scene without importing Qt."""

    # Extract the dict at module import time via ast/parse to avoid Qt dependency.
    # Simpler: just re-declare the expected contract and verify it matches the source.

    EXPECTED_KEYS = {"protocol_valid", "protocol_invalid", "protocol_none"}

    def _get_colors_from_source(self):
        """Extract _PROTOCOL_BORDER_COLORS dict from graph_scene.py via regex."""
        import re
        src = Path(PROJECT_ROOT / "bin" / "components" / "graph_scene.py").read_text(encoding="utf-8-sig")
        # Match the dict block: _PROTOCOL_BORDER_COLORS: Dict[str, str] = { ... }
        m = re.search(r"_PROTOCOL_BORDER_COLORS[^=]*=\s*\{([^}]+)\}", src, re.DOTALL)
        if not m:
            return None
        colors = {}
        for pair in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)):
            colors[pair.group(1)] = pair.group(2)
        return colors if colors else None

    def test_all_three_keys_present(self):
        colors = self._get_colors_from_source()
        self.assertIsNotNone(colors, "Could not parse _PROTOCOL_BORDER_COLORS from source")
        self.assertEqual(set(colors.keys()), self.EXPECTED_KEYS)

    def test_all_values_are_hex_strings(self):
        colors = self._get_colors_from_source()
        self.assertIsNotNone(colors)
        for key, val in colors.items():
            self.assertIsInstance(val, str, f"{key} value must be a string")
            self.assertTrue(val.startswith("#"), f"{key} must be a hex color starting with '#'")
            self.assertIn(len(val), (4, 7, 9), f"{key} has unexpected hex length: {val}")

    def test_valid_and_invalid_are_distinct_colors(self):
        colors = self._get_colors_from_source()
        self.assertIsNotNone(colors)
        self.assertNotEqual(colors["protocol_valid"], colors["protocol_invalid"])

    def test_none_is_distinct_from_valid_and_invalid(self):
        colors = self._get_colors_from_source()
        self.assertIsNotNone(colors)
        self.assertNotEqual(colors["protocol_none"], colors["protocol_valid"])
        self.assertNotEqual(colors["protocol_none"], colors["protocol_invalid"])


if __name__ == "__main__":
    unittest.main()
