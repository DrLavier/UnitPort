"""Cycle 3 STAGE-02 — Mission settings persistence unit tests.

Covers:
  - build_settings_payload / extract_settings_payload helpers
  - MISSION_SCHEMA_VERSION constant (bumped to "1.1")
  - inject_snapshot_metadata non-destructive to settings payload
  - validate_mission_schema backward-compat (missing "settings" key is not an error)
  - JSON save/load roundtrip preserves settings payload

All tests are pure Python — no Qt dependency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bin.core.mission_persistence import (
    MISSION_SCHEMA_VERSION,
    UNITPORT_VERSION,
    build_settings_payload,
    extract_settings_payload,
    inject_snapshot_metadata,
    validate_mission_schema,
)


# ── Suite A: MISSION_SCHEMA_VERSION and UNITPORT_VERSION ──────────────────────


class TestVersionConstants(unittest.TestCase):

    def test_schema_version_is_1_4(self):
        self.assertEqual(MISSION_SCHEMA_VERSION, "1.4")

    def test_unitport_version_is_cycle3(self):
        self.assertEqual(UNITPORT_VERSION, "cycle3")


# ── Suite B: build_settings_payload ──────────────────────────────────────────


class TestBuildSettingsPayload(unittest.TestCase):

    def test_basic_structure(self):
        result = build_settings_payload("unitree", {"ip": "192.168.1.1", "port": 8080})
        self.assertEqual(result["brand"], "unitree")
        self.assertEqual(result["config"]["ip"], "192.168.1.1")
        self.assertEqual(result["config"]["port"], 8080)

    def test_empty_config_produces_empty_dict(self):
        result = build_settings_payload("bostiondynamics", {})
        self.assertEqual(result["config"], {})

    def test_none_config_produces_empty_dict(self):
        result = build_settings_payload("xiaomi", None)  # type: ignore[arg-type]
        self.assertEqual(result["config"], {})

    def test_does_not_mutate_input_config(self):
        original = {"host": "10.0.0.1"}
        result = build_settings_payload("unitree", original)
        result["config"]["host"] = "MUTATED"
        self.assertEqual(original["host"], "10.0.0.1")

    def test_brand_preserved_exactly(self):
        result = build_settings_payload("bostiondynamics", {"token": "abc"})
        self.assertEqual(result["brand"], "bostiondynamics")

    def test_keys_are_brand_and_config(self):
        result = build_settings_payload("xiaomi", {"model": "cyberdog2"})
        self.assertIn("brand", result)
        self.assertIn("config", result)

    def test_config_values_preserved_types(self):
        cfg = {"enabled": True, "retries": 3, "host": "1.2.3.4", "tags": ["a", "b"]}
        result = build_settings_payload("unitree", cfg)
        self.assertIs(result["config"]["enabled"], True)
        self.assertEqual(result["config"]["retries"], 3)
        self.assertEqual(result["config"]["tags"], ["a", "b"])


# ── Suite C: extract_settings_payload ────────────────────────────────────────


class TestExtractSettingsPayload(unittest.TestCase):

    def _mission(self, **settings_kwargs):
        """Build a minimal mission dict with optional settings block."""
        d = {"nodes": [], "connections": []}
        if settings_kwargs:
            d["settings"] = settings_kwargs
        return d

    def test_returns_none_when_no_settings_key(self):
        data = {"nodes": [], "connections": []}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_settings_is_none(self):
        data = {"nodes": [], "connections": [], "settings": None}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_settings_is_not_dict(self):
        data = {"nodes": [], "connections": [], "settings": "bad"}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_brand_missing(self):
        data = {"nodes": [], "connections": [], "settings": {"config": {}}}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_brand_is_empty_string(self):
        data = {"nodes": [], "connections": [], "settings": {"brand": "", "config": {}}}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_brand_is_whitespace_only(self):
        data = {"nodes": [], "connections": [], "settings": {"brand": "   ", "config": {}}}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_brand_is_not_str(self):
        data = {"nodes": [], "connections": [], "settings": {"brand": 42, "config": {}}}
        self.assertIsNone(extract_settings_payload(data))

    def test_returns_none_when_input_is_not_dict(self):
        self.assertIsNone(extract_settings_payload(None))
        self.assertIsNone(extract_settings_payload("bad"))
        self.assertIsNone(extract_settings_payload(123))

    def test_valid_payload_returns_tuple(self):
        data = {
            "nodes": [], "connections": [],
            "settings": {"brand": "unitree", "config": {"ip": "10.0.0.1"}},
        }
        result = extract_settings_payload(data)
        self.assertIsNotNone(result)
        brand, config = result
        self.assertEqual(brand, "unitree")
        self.assertEqual(config["ip"], "10.0.0.1")

    def test_missing_config_inside_settings_treated_as_empty_dict(self):
        data = {"nodes": [], "connections": [], "settings": {"brand": "bostiondynamics"}}
        result = extract_settings_payload(data)
        self.assertIsNotNone(result)
        brand, config = result
        self.assertEqual(brand, "bostiondynamics")
        self.assertEqual(config, {})

    def test_config_not_dict_treated_as_empty_dict(self):
        data = {
            "nodes": [], "connections": [],
            "settings": {"brand": "xiaomi", "config": "not-a-dict"},
        }
        result = extract_settings_payload(data)
        self.assertIsNotNone(result)
        brand, config = result
        self.assertEqual(brand, "xiaomi")
        self.assertEqual(config, {})

    def test_full_roundtrip_build_then_extract(self):
        payload = build_settings_payload("unitree", {"port": 9090, "mode": "real"})
        data = {"nodes": [], "connections": [], "settings": payload}
        result = extract_settings_payload(data)
        self.assertIsNotNone(result)
        brand, config = result
        self.assertEqual(brand, "unitree")
        self.assertEqual(config["port"], 9090)
        self.assertEqual(config["mode"], "real")


# ── Suite D: validate_mission_schema backward-compat ─────────────────────────


class TestValidateMissionSchemaBackwardCompat(unittest.TestCase):

    def test_no_settings_key_passes_validation(self):
        data = {"nodes": [], "connections": []}
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, f"Unexpected rejection: {reason}")

    def test_with_settings_key_passes_validation(self):
        data = {
            "nodes": [], "connections": [],
            "settings": {"brand": "unitree", "config": {}},
        }
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, f"Unexpected rejection: {reason}")

    def test_old_file_schema_version_1_0_passes_validation(self):
        data = {
            "schema_version": "1.0",
            "nodes": [], "connections": [],
        }
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, f"Old schema_version 1.0 incorrectly rejected: {reason}")

    def test_new_file_schema_version_1_1_passes_validation(self):
        data = {
            "schema_version": "1.1",
            "nodes": [], "connections": [],
            "settings": {"brand": "unitree", "config": {}},
        }
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, f"New schema_version 1.1 incorrectly rejected: {reason}")


# ── Suite E: inject_snapshot_metadata non-destructive ────────────────────────


class TestInjectSnapshotMetadataPreservesSettings(unittest.TestCase):

    def test_settings_payload_not_overwritten(self):
        data = {
            "nodes": [], "connections": [],
            "settings": {"brand": "unitree", "config": {"host": "1.2.3.4"}},
        }
        inject_snapshot_metadata(data)
        # inject must not clobber the "settings" key
        self.assertEqual(data["settings"]["brand"], "unitree")
        self.assertEqual(data["settings"]["config"]["host"], "1.2.3.4")

    def test_schema_version_set_to_1_4(self):
        data = {"nodes": [], "connections": []}
        inject_snapshot_metadata(data)
        self.assertEqual(data["schema_version"], "1.4")

    def test_unitport_version_set_to_cycle3(self):
        data = {"nodes": [], "connections": []}
        inject_snapshot_metadata(data)
        self.assertEqual(data["unitport_version"], "cycle3")

    def test_saved_at_key_present(self):
        data = {"nodes": [], "connections": []}
        inject_snapshot_metadata(data)
        self.assertIn("saved_at", data)

    def test_inject_does_not_add_settings_when_absent(self):
        data = {"nodes": [], "connections": []}
        inject_snapshot_metadata(data)
        self.assertNotIn("settings", data)


# ── Suite F: JSON roundtrip preserves settings ───────────────────────────────


class TestJsonRoundtripWithSettings(unittest.TestCase):

    def _roundtrip(self, data: dict) -> dict:
        with tempfile.NamedTemporaryFile(
            suffix=".unitport", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name
        try:
            with open(fname, "r", encoding="utf-8") as fh:
                return json.load(fh)
        finally:
            os.unlink(fname)

    def test_settings_survive_json_roundtrip(self):
        payload = build_settings_payload("unitree", {"host": "10.0.0.2", "port": 8080})
        data = {
            "nodes": [], "connections": [],
            "settings": payload,
        }
        inject_snapshot_metadata(data)
        loaded = self._roundtrip(data)
        result = extract_settings_payload(loaded)
        self.assertIsNotNone(result)
        brand, config = result
        self.assertEqual(brand, "unitree")
        self.assertEqual(config["host"], "10.0.0.2")
        self.assertEqual(config["port"], 8080)

    def test_old_file_without_settings_roundtrip(self):
        """Pre-Cycle-3 file survives roundtrip with extract returning None."""
        data = {"nodes": [], "connections": [], "schema_version": "1.0"}
        loaded = self._roundtrip(data)
        ok, _ = validate_mission_schema(loaded)
        self.assertTrue(ok)
        self.assertIsNone(extract_settings_payload(loaded))

    def test_schema_version_in_roundtripped_file(self):
        data = {"nodes": [], "connections": []}
        inject_snapshot_metadata(data)
        loaded = self._roundtrip(data)
        self.assertEqual(loaded.get("schema_version"), MISSION_SCHEMA_VERSION)

    def test_settings_brand_preserved_in_roundtrip(self):
        for brand in ("unitree", "bostiondynamics", "xiaomi"):
            with self.subTest(brand=brand):
                payload = build_settings_payload(brand, {})
                data = {"nodes": [], "connections": [], "settings": payload}
                loaded = self._roundtrip(data)
                result = extract_settings_payload(loaded)
                self.assertIsNotNone(result)
                self.assertEqual(result[0], brand)

    def test_complex_config_values_survive_roundtrip(self):
        cfg = {
            "host": "10.0.0.3",
            "port": 9090,
            "enabled": True,
            "retries": 3,
            "tags": ["alpha", "beta"],
        }
        payload = build_settings_payload("unitree", cfg)
        data = {"nodes": [], "connections": [], "settings": payload}
        loaded = self._roundtrip(data)
        result = extract_settings_payload(loaded)
        self.assertIsNotNone(result)
        _, config = result
        self.assertEqual(config["host"], "10.0.0.3")
        self.assertEqual(config["port"], 9090)
        self.assertIs(config["enabled"], True)
        self.assertEqual(config["retries"], 3)
        self.assertEqual(config["tags"], ["alpha", "beta"])


from bin.core.mission_persistence import (  # noqa: E402
    build_behavior_timeline_payload,
    extract_behavior_timeline_payload,
)


# ── Suite G: build_behavior_timeline_payload / extract_behavior_timeline_payload


class TestBehaviorTimelinePayload(unittest.TestCase):

    def _sample_timeline_dict(self) -> dict:
        return {
            "version": "1.0",
            "action_segments": [
                {"action_id": "a1", "name": "walk", "start_time": 0.0, "duration": 4.0,
                 "params": {}, "kind": "movement", "locked": False},
            ],
            "motor_overlays": [],
            "robot_type": "go2",
            "active_motor_tracks": ["leg_group_left"],
        }

    def test_build_stringifies_int_keys(self):
        timelines = {1: self._sample_timeline_dict(), 2: self._sample_timeline_dict()}
        result = build_behavior_timeline_payload(timelines)
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertNotIn(1, result)

    def test_build_shallow_copy(self):
        orig = self._sample_timeline_dict()
        timelines = {"5::walk": orig}
        result = build_behavior_timeline_payload(timelines)
        result["5::walk"]["extra"] = "injected"
        self.assertNotIn("extra", orig)

    def test_build_skips_non_dict_values(self):
        timelines = {"ok": self._sample_timeline_dict(), "bad": "not_a_dict"}
        result = build_behavior_timeline_payload(timelines)
        self.assertIn("ok", result)
        self.assertNotIn("bad", result)

    def test_extract_returns_none_for_missing_key(self):
        data = {"nodes": [], "connections": []}
        self.assertIsNone(extract_behavior_timeline_payload(data))

    def test_extract_returns_none_for_non_dict_value(self):
        data = {"nodes": [], "connections": [], "behavior_timelines": "bad"}
        self.assertIsNone(extract_behavior_timeline_payload(data))

    def test_extract_returns_none_for_non_dict_input(self):
        for bad in [None, "string", 42, []]:
            self.assertIsNone(extract_behavior_timeline_payload(bad))

    def test_extract_returns_dict_when_present(self):
        tl_dict = {"5::walk": self._sample_timeline_dict()}
        data = {"nodes": [], "connections": [], "behavior_timelines": tl_dict}
        result = extract_behavior_timeline_payload(data)
        self.assertIsNotNone(result)
        self.assertIn("5::walk", result)

    def test_roundtrip_json(self):
        import json
        import tempfile
        import os
        tl_dict = {"5::walk": self._sample_timeline_dict()}
        payload = build_behavior_timeline_payload(tl_dict)
        data = {"nodes": [], "connections": [], "behavior_timelines": payload}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            json.dump(data, fh)
            fname = fh.name
        try:
            with open(fname) as fh:
                loaded = json.load(fh)
        finally:
            os.unlink(fname)
        result = extract_behavior_timeline_payload(loaded)
        self.assertIsNotNone(result)
        self.assertIn("5::walk", result)
        entry = result["5::walk"]
        self.assertEqual(entry["robot_type"], "go2")
        self.assertEqual(entry["active_motor_tracks"], ["leg_group_left"])


if __name__ == "__main__":
    unittest.main()
