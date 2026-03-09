"""Tests for system.service.settings_validator — Phase 5 STAGE-05.

Coverage:
    A. Return shape — ValidationResult keys always present
    B. Valid configs — all three brands pass with "ok" status
    C. Missing required keys — each absent required key appears in "missing"
    D. Optional keys — absent optional keys are NOT flagged as missing
    E. Type violations — wrong-type values appear in "invalid"
    F. bool-for-int guard — True/False rejected for int-typed settings
    G. Choices violations — disallowed values appear in "invalid"
    H. Multiple violations — all errors accumulate
    I. Unknown brand — returns "error" status
    J. Non-dict config — returns "error" status
    K. Extra keys in config — silently ignored
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from system.service.settings_validator import validate_settings


# ── Fixture helpers ────────────────────────────────────────────────────────

def _valid_unitree() -> dict:
    return {
        "brand":              "unitree",
        "robot_type":         "go2",
        "connection_mode":    "simulation",
        "timeout":            30,
        "stop_policy":        "graceful",
        "safety_enabled":     True,
        "dds_domain_id":      0,
        "network_interface":  "eth0",
        "control_level":      "high",
        "mode_switch_policy": "immediate",
    }


def _valid_spot() -> dict:
    return {
        "brand":             "bostiondynamics",
        "robot_type":        "spot",
        "connection_mode":   "hardware",
        "timeout":           60,
        "stop_policy":       "graceful",
        "safety_enabled":    True,
        "hostname":          "192.168.80.3",
        "auth_token":        "secret",
        "safety_channel":    "estop_channel_0",
        "power_policy":      "safe_power_off",
        # timesync_required and lease_required are optional — omitted here
    }


def _valid_cyberdog() -> dict:
    return {
        "brand":             "xiaomi",
        "robot_type":        "cyberdog2",
        "connection_mode":   "hardware",
        "timeout":           20,
        "stop_policy":       "immediate",
        "safety_enabled":    False,
        "ros_domain_id":     42,
        "rmw_impl":          "rmw_fastrtps_cpp",
        "action_endpoints":  ["/cyberdog/stand", "/cyberdog/walk"],
        "timestamp_policy":  "relaxed",
        # namespace is optional — omitted here
    }


# ══════════════════════════════════════════════════════════════════════════
# A — Return shape
# ══════════════════════════════════════════════════════════════════════════

class TestReturnShape(unittest.TestCase):
    """ValidationResult always has the five required keys."""

    _REQUIRED = ("status", "brand", "errors", "missing", "invalid")

    def _assert_shape(self, result, brand):
        for key in self._REQUIRED:
            self.assertIn(key, result, f"Missing key {key!r} in result")
        self.assertIsInstance(result["errors"],  list)
        self.assertIsInstance(result["missing"], list)
        self.assertIsInstance(result["invalid"], list)
        self.assertIn(result["status"], ("ok", "error"))
        self.assertEqual(result["brand"], brand)

    def test_valid_unitree_shape(self):
        self._assert_shape(validate_settings("unitree", _valid_unitree()), "unitree")

    def test_valid_spot_shape(self):
        self._assert_shape(validate_settings("bostiondynamics", _valid_spot()), "bostiondynamics")

    def test_valid_cyberdog_shape(self):
        self._assert_shape(validate_settings("xiaomi", _valid_cyberdog()), "xiaomi")

    def test_failing_config_shape(self):
        self._assert_shape(validate_settings("unitree", {}), "unitree")

    def test_unknown_brand_shape(self):
        self._assert_shape(validate_settings("no_such_brand", {}), "no_such_brand")

    def test_non_dict_config_shape(self):
        self._assert_shape(validate_settings("unitree", None), "unitree")


# ══════════════════════════════════════════════════════════════════════════
# B — Valid configs
# ══════════════════════════════════════════════════════════════════════════

class TestValidConfigs(unittest.TestCase):
    """Complete, correct configs produce status=ok with empty lists."""

    def test_unitree_valid_status_ok(self):
        r = validate_settings("unitree", _valid_unitree())
        self.assertEqual(r["status"], "ok")

    def test_unitree_valid_empty_lists(self):
        r = validate_settings("unitree", _valid_unitree())
        self.assertEqual(r["errors"],  [])
        self.assertEqual(r["missing"], [])
        self.assertEqual(r["invalid"], [])

    def test_spot_valid_status_ok(self):
        r = validate_settings("bostiondynamics", _valid_spot())
        self.assertEqual(r["status"], "ok")

    def test_spot_valid_empty_lists(self):
        r = validate_settings("bostiondynamics", _valid_spot())
        self.assertEqual(r["errors"],  [])
        self.assertEqual(r["missing"], [])
        self.assertEqual(r["invalid"], [])

    def test_cyberdog_valid_status_ok(self):
        r = validate_settings("xiaomi", _valid_cyberdog())
        self.assertEqual(r["status"], "ok")

    def test_cyberdog_valid_empty_lists(self):
        r = validate_settings("xiaomi", _valid_cyberdog())
        self.assertEqual(r["errors"],  [])
        self.assertEqual(r["missing"], [])
        self.assertEqual(r["invalid"], [])


# ══════════════════════════════════════════════════════════════════════════
# C — Missing required keys
# ══════════════════════════════════════════════════════════════════════════

class TestMissingRequired(unittest.TestCase):
    """Absent required keys appear in 'missing'; status is 'error'."""

    def _drop(self, config: dict, key: str) -> dict:
        c = dict(config)
        del c[key]
        return c

    def test_missing_brand_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "brand"))
        self.assertIn("brand", r["missing"])
        self.assertEqual(r["status"], "error")

    def test_missing_robot_type_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "robot_type"))
        self.assertIn("robot_type", r["missing"])

    def test_missing_connection_mode_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "connection_mode"))
        self.assertIn("connection_mode", r["missing"])

    def test_missing_timeout_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "timeout"))
        self.assertIn("timeout", r["missing"])

    def test_missing_stop_policy_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "stop_policy"))
        self.assertIn("stop_policy", r["missing"])

    def test_missing_safety_enabled_flagged(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "safety_enabled"))
        self.assertIn("safety_enabled", r["missing"])

    def test_missing_unitree_specific_key(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "dds_domain_id"))
        self.assertIn("dds_domain_id", r["missing"])

    def test_missing_spot_hostname_flagged(self):
        r = validate_settings("bostiondynamics", self._drop(_valid_spot(), "hostname"))
        self.assertIn("hostname", r["missing"])

    def test_missing_spot_auth_token_flagged(self):
        r = validate_settings("bostiondynamics", self._drop(_valid_spot(), "auth_token"))
        self.assertIn("auth_token", r["missing"])

    def test_missing_cyberdog_action_endpoints_flagged(self):
        r = validate_settings("xiaomi", self._drop(_valid_cyberdog(), "action_endpoints"))
        self.assertIn("action_endpoints", r["missing"])

    def test_missing_key_also_in_errors(self):
        r = validate_settings("unitree", self._drop(_valid_unitree(), "timeout"))
        self.assertTrue(any("timeout" in e for e in r["errors"]))

    def test_empty_config_all_required_missing(self):
        """Empty config → all truly required keys appear in missing."""
        r = validate_settings("unitree", {})
        self.assertEqual(r["status"], "error")
        # 6 global + 4 unitree-specific = 10 required keys (no optionals for unitree)
        for key in ("brand", "robot_type", "connection_mode", "timeout",
                    "stop_policy", "safety_enabled", "dds_domain_id",
                    "network_interface", "control_level", "mode_switch_policy"):
            self.assertIn(key, r["missing"], f"Expected {key!r} in missing")


# ══════════════════════════════════════════════════════════════════════════
# D — Optional keys are not flagged as missing when absent
# ══════════════════════════════════════════════════════════════════════════

class TestOptionalKeysNotMissing(unittest.TestCase):
    """Optional entries (required=False) must NOT appear in 'missing'."""

    def test_spot_timesync_required_optional(self):
        """timesync_required absent → not flagged as missing."""
        config = dict(_valid_spot())
        config.pop("timesync_required", None)
        r = validate_settings("bostiondynamics", config)
        self.assertNotIn("timesync_required", r["missing"])
        self.assertEqual(r["status"], "ok")

    def test_spot_lease_required_optional(self):
        """lease_required absent → not flagged as missing."""
        config = dict(_valid_spot())
        config.pop("lease_required", None)
        r = validate_settings("bostiondynamics", config)
        self.assertNotIn("lease_required", r["missing"])

    def test_cyberdog_namespace_optional(self):
        """namespace absent → not flagged as missing."""
        config = dict(_valid_cyberdog())
        config.pop("namespace", None)
        r = validate_settings("xiaomi", config)
        self.assertNotIn("namespace", r["missing"])
        self.assertEqual(r["status"], "ok")


# ══════════════════════════════════════════════════════════════════════════
# E — Type violations
# ══════════════════════════════════════════════════════════════════════════

class TestTypeViolations(unittest.TestCase):
    """Wrong-type values appear in 'invalid'; status is 'error'."""

    def _with(self, config: dict, key: str, value) -> dict:
        c = dict(config)
        c[key] = value
        return c

    def test_brand_wrong_type(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "brand", 42))
        self.assertIn("brand", r["invalid"])
        self.assertEqual(r["status"], "error")

    def test_timeout_wrong_type_str(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "timeout", "30"))
        self.assertIn("timeout", r["invalid"])

    def test_safety_enabled_wrong_type_int(self):
        """safety_enabled=1 (int) must be rejected — type is bool."""
        r = validate_settings("unitree", self._with(_valid_unitree(), "safety_enabled", 1))
        self.assertIn("safety_enabled", r["invalid"])

    def test_dds_domain_id_wrong_type_str(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "dds_domain_id", "0"))
        self.assertIn("dds_domain_id", r["invalid"])

    def test_control_level_wrong_type_int(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "control_level", 1))
        self.assertIn("control_level", r["invalid"])

    def test_spot_timesync_required_wrong_type(self):
        config = self._with(_valid_spot(), "timesync_required", "yes")
        r = validate_settings("bostiondynamics", config)
        self.assertIn("timesync_required", r["invalid"])

    def test_cyberdog_action_endpoints_wrong_type(self):
        config = self._with(_valid_cyberdog(), "action_endpoints", "endpoint_a")
        r = validate_settings("xiaomi", config)
        self.assertIn("action_endpoints", r["invalid"])

    def test_type_error_also_in_errors_list(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "timeout", "30"))
        self.assertTrue(any("timeout" in e for e in r["errors"]))


# ══════════════════════════════════════════════════════════════════════════
# F — bool-for-int guard
# ══════════════════════════════════════════════════════════════════════════

class TestBoolForIntGuard(unittest.TestCase):
    """True/False must be rejected for int-typed settings."""

    def _with(self, config: dict, key: str, value) -> dict:
        c = dict(config)
        c[key] = value
        return c

    def test_dds_domain_id_true_rejected(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "dds_domain_id", True))
        self.assertIn("dds_domain_id", r["invalid"])

    def test_dds_domain_id_false_rejected(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "dds_domain_id", False))
        self.assertIn("dds_domain_id", r["invalid"])

    def test_timeout_true_rejected(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "timeout", True))
        self.assertIn("timeout", r["invalid"])

    def test_ros_domain_id_bool_rejected(self):
        config = self._with(_valid_cyberdog(), "ros_domain_id", True)
        r = validate_settings("xiaomi", config)
        self.assertIn("ros_domain_id", r["invalid"])

    def test_int_zero_is_valid_for_int_fields(self):
        """0 (int) must pass for int-typed fields."""
        config = self._with(_valid_unitree(), "dds_domain_id", 0)
        r = validate_settings("unitree", config)
        self.assertNotIn("dds_domain_id", r["invalid"])

    def test_bool_true_valid_for_bool_fields(self):
        """True must pass for bool-typed fields."""
        r = validate_settings("unitree", _valid_unitree())
        self.assertNotIn("safety_enabled", r["invalid"])

    def test_bool_false_valid_for_bool_fields(self):
        """False must pass for bool-typed fields."""
        config = dict(_valid_unitree())
        config["safety_enabled"] = False
        r = validate_settings("unitree", config)
        self.assertNotIn("safety_enabled", r["invalid"])


# ══════════════════════════════════════════════════════════════════════════
# G — Choices violations
# ══════════════════════════════════════════════════════════════════════════

class TestChoicesViolations(unittest.TestCase):
    """Values not in the allowed choices appear in 'invalid'."""

    def _with(self, config: dict, key: str, value) -> dict:
        c = dict(config)
        c[key] = value
        return c

    def test_brand_disallowed_value(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "brand", "acme"))
        self.assertIn("brand", r["invalid"])
        self.assertEqual(r["status"], "error")

    def test_connection_mode_disallowed(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "connection_mode", "cloud"))
        self.assertIn("connection_mode", r["invalid"])

    def test_stop_policy_disallowed(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "stop_policy", "force"))
        self.assertIn("stop_policy", r["invalid"])

    def test_control_level_disallowed(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "control_level", "mid"))
        self.assertIn("control_level", r["invalid"])

    def test_power_policy_disallowed(self):
        config = self._with(_valid_spot(), "power_policy", "fast_off")
        r = validate_settings("bostiondynamics", config)
        self.assertIn("power_policy", r["invalid"])

    def test_timestamp_policy_disallowed(self):
        config = self._with(_valid_cyberdog(), "timestamp_policy", "loose")
        r = validate_settings("xiaomi", config)
        self.assertIn("timestamp_policy", r["invalid"])

    def test_choices_error_in_errors_list(self):
        r = validate_settings("unitree", self._with(_valid_unitree(), "brand", "acme"))
        self.assertTrue(any("brand" in e for e in r["errors"]))


# ══════════════════════════════════════════════════════════════════════════
# H — Multiple violations accumulate
# ══════════════════════════════════════════════════════════════════════════

class TestMultipleViolations(unittest.TestCase):

    def test_missing_and_invalid_both_reported(self):
        config = {
            "brand":           "unitree",
            "robot_type":      "go2",
            "connection_mode": "bad_mode",  # invalid choice
            "timeout":         "not_an_int", # wrong type
            # missing: stop_policy, safety_enabled, dds_domain_id,
            #          network_interface, control_level, mode_switch_policy
        }
        r = validate_settings("unitree", config)
        self.assertEqual(r["status"], "error")
        self.assertGreater(len(r["missing"]), 0)
        self.assertGreater(len(r["invalid"]), 0)

    def test_two_missing_keys_both_in_list(self):
        config = dict(_valid_unitree())
        del config["dds_domain_id"]
        del config["network_interface"]
        r = validate_settings("unitree", config)
        self.assertIn("dds_domain_id",    r["missing"])
        self.assertIn("network_interface", r["missing"])

    def test_two_invalid_keys_both_in_list(self):
        config = dict(_valid_unitree())
        config["connection_mode"] = "cloud"
        config["control_level"]   = "medium"
        r = validate_settings("unitree", config)
        self.assertIn("connection_mode", r["invalid"])
        self.assertIn("control_level",   r["invalid"])

    def test_errors_count_matches_violations(self):
        config = dict(_valid_unitree())
        del config["timeout"]          # 1 missing
        config["control_level"] = 99   # 1 invalid
        r = validate_settings("unitree", config)
        self.assertEqual(len(r["missing"]), 1)
        self.assertEqual(len(r["invalid"]), 1)
        self.assertEqual(len(r["errors"]),  2)


# ══════════════════════════════════════════════════════════════════════════
# I — Unknown brand
# ══════════════════════════════════════════════════════════════════════════

class TestUnknownBrand(unittest.TestCase):

    def test_unknown_brand_status_error(self):
        r = validate_settings("no_such_brand", {})
        self.assertEqual(r["status"], "error")

    def test_unknown_brand_errors_non_empty(self):
        r = validate_settings("no_such_brand", {})
        self.assertGreater(len(r["errors"]), 0)

    def test_unknown_brand_missing_empty(self):
        """Unknown brand → missing list is empty (no schema to check against)."""
        r = validate_settings("no_such_brand", {})
        self.assertEqual(r["missing"], [])

    def test_unknown_brand_preserved_in_result(self):
        r = validate_settings("unknown_brand_xyz", {})
        self.assertEqual(r["brand"], "unknown_brand_xyz")


# ══════════════════════════════════════════════════════════════════════════
# J — Non-dict config
# ══════════════════════════════════════════════════════════════════════════

class TestNonDictConfig(unittest.TestCase):

    def test_none_config_status_error(self):
        self.assertEqual(validate_settings("unitree", None)["status"], "error")

    def test_list_config_status_error(self):
        self.assertEqual(validate_settings("unitree", [])["status"], "error")

    def test_string_config_status_error(self):
        self.assertEqual(validate_settings("unitree", "config")["status"], "error")

    def test_int_config_status_error(self):
        self.assertEqual(validate_settings("unitree", 42)["status"], "error")

    def test_non_dict_errors_non_empty(self):
        r = validate_settings("unitree", None)
        self.assertGreater(len(r["errors"]), 0)


# ══════════════════════════════════════════════════════════════════════════
# K — Extra keys in config are silently ignored
# ══════════════════════════════════════════════════════════════════════════

class TestExtraKeysIgnored(unittest.TestCase):

    def test_extra_key_does_not_cause_error(self):
        config = dict(_valid_unitree())
        config["unexpected_future_setting"] = "value"
        r = validate_settings("unitree", config)
        self.assertEqual(r["status"], "ok")

    def test_multiple_extra_keys_ignored(self):
        config = dict(_valid_spot())
        config["alpha"] = 1
        config["beta"]  = [2, 3]
        r = validate_settings("bostiondynamics", config)
        self.assertEqual(r["status"], "ok")


if __name__ == "__main__":
    unittest.main()
