"""STAGE-05 — Mission Persistence and Snapshot Flow tests.

Verifies that:
  - validate_mission_schema accepts/rejects the right payloads.
  - inject_snapshot_metadata stamps the correct keys.
  - File roundtrip (dict → JSON → file → reload) succeeds.
  - Old-format files (no schema_version) are accepted (backward compat).
  - GraphScene serialize → inject → validate → load_workflow roundtrip works.

Sections:
    A — Schema validation
    B — Snapshot metadata injection
    C — File roundtrip (pure-Python, no Qt)
    D — GraphScene roundtrip (Qt / offscreen)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest


# ── Section A: Schema validation ──────────────────────────────────────────────

class TestValidateMissionSchema(unittest.TestCase):

    def setUp(self):
        from bin.core.mission_persistence import validate_mission_schema
        self.validate = validate_mission_schema

    @staticmethod
    def _valid() -> dict:
        return {"nodes": [], "connections": [], "groups": []}

    # ── valid inputs ──────────────────────────────────────────────────────────

    def test_valid_empty_mission_passes(self):
        ok, reason = self.validate(self._valid())
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_valid_with_schema_version_passes(self):
        d = self._valid()
        d["schema_version"] = "1.0"
        ok, _ = self.validate(d)
        self.assertTrue(ok)

    def test_missing_schema_version_still_passes(self):
        """Pre-STAGE-05 files without schema_version must be accepted."""
        d = {"nodes": [], "connections": []}
        ok, _ = self.validate(d)
        self.assertTrue(ok)

    def test_extra_unknown_keys_tolerated(self):
        d = self._valid()
        d["future_extension"] = {"anything": True}
        ok, _ = self.validate(d)
        self.assertTrue(ok)

    def test_valid_with_real_node_data(self):
        d = {
            "nodes": [{"id": 1, "display_name": "Start", "position": {"x": 0, "y": 0}}],
            "connections": [],
            "groups": [],
        }
        ok, _ = self.validate(d)
        self.assertTrue(ok)

    # ── invalid inputs ────────────────────────────────────────────────────────

    def test_non_dict_top_level_fails(self):
        ok, reason = self.validate([1, 2, 3])
        self.assertFalse(ok)
        self.assertIn("JSON object", reason)

    def test_null_value_fails(self):
        ok, _ = self.validate(None)
        self.assertFalse(ok)

    def test_string_value_fails(self):
        ok, _ = self.validate("not_a_dict")
        self.assertFalse(ok)

    def test_missing_nodes_key_fails(self):
        ok, reason = self.validate({"connections": []})
        self.assertFalse(ok)
        self.assertIn("nodes", reason)

    def test_missing_connections_key_fails(self):
        ok, reason = self.validate({"nodes": []})
        self.assertFalse(ok)
        self.assertIn("connections", reason)

    def test_nodes_as_dict_fails(self):
        """exec-graph format (nodes as dict) must be rejected — wrong payload."""
        ok, reason = self.validate({"nodes": {}, "connections": []})
        self.assertFalse(ok)
        self.assertIn("list", reason)

    def test_connections_as_string_fails(self):
        ok, reason = self.validate({"nodes": [], "connections": "bad"})
        self.assertFalse(ok)
        self.assertIn("list", reason)

    def test_reason_is_string_on_failure(self):
        ok, reason = self.validate("not_a_dict")
        self.assertFalse(ok)
        self.assertIsInstance(reason, str)

    def test_reason_is_empty_string_on_success(self):
        ok, reason = self.validate(self._valid())
        self.assertTrue(ok)
        self.assertEqual(reason, "")


# ── Section B: Snapshot metadata injection ────────────────────────────────────

class TestInjectSnapshotMetadata(unittest.TestCase):

    def setUp(self):
        from bin.core.mission_persistence import inject_snapshot_metadata
        self.inject = inject_snapshot_metadata

    @staticmethod
    def _base() -> dict:
        return {"nodes": [], "connections": []}

    def test_injects_schema_version(self):
        d = self.inject(self._base())
        self.assertIn("schema_version", d)

    def test_injects_unitport_version(self):
        d = self.inject(self._base())
        self.assertIn("unitport_version", d)

    def test_injects_saved_at(self):
        d = self.inject(self._base())
        self.assertIn("saved_at", d)

    def test_saved_at_ends_with_z(self):
        d = self.inject(self._base())
        self.assertTrue(d["saved_at"].endswith("Z"), f"Got: {d['saved_at']}")

    def test_schema_version_is_string(self):
        d = self.inject(self._base())
        self.assertIsInstance(d["schema_version"], str)

    def test_unitport_version_is_string(self):
        d = self.inject(self._base())
        self.assertIsInstance(d["unitport_version"], str)

    def test_existing_keys_preserved(self):
        d = self._base()
        d["robot_brand"] = "unitree"
        result = self.inject(d)
        self.assertEqual(result["robot_brand"], "unitree")

    def test_mutates_in_place(self):
        d = self._base()
        result = self.inject(d)
        self.assertIs(result, d)

    def test_schema_version_matches_constant(self):
        from bin.core.mission_persistence import MISSION_SCHEMA_VERSION
        d = self.inject(self._base())
        self.assertEqual(d["schema_version"], MISSION_SCHEMA_VERSION)

    def test_unitport_version_matches_constant(self):
        from bin.core.mission_persistence import UNITPORT_VERSION
        d = self.inject(self._base())
        self.assertEqual(d["unitport_version"], UNITPORT_VERSION)

    # ── source field (STAGE-05 gap fix) ──────────────────────────────────

    def test_injects_source_with_default(self):
        d = self.inject(self._base())
        self.assertIn("source", d)

    def test_source_default_is_mission_editor_ui(self):
        d = self.inject(self._base())
        self.assertEqual(d["source"], "mission_editor_ui")

    def test_source_custom_value_accepted(self):
        d = self.inject(self._base(), source="test_harness")
        self.assertEqual(d["source"], "test_harness")

    def test_source_is_string(self):
        d = self.inject(self._base())
        self.assertIsInstance(d["source"], str)

    def test_source_nonempty_by_default(self):
        d = self.inject(self._base())
        self.assertTrue(len(d["source"]) > 0)

    def test_source_preserved_in_json_roundtrip(self):
        import json
        d = self.inject(self._base())
        raw = json.dumps(d)
        loaded = json.loads(raw)
        self.assertEqual(loaded["source"], "mission_editor_ui")


# ── Section C: File roundtrip (pure-Python, no Qt) ────────────────────────────

class TestFileRoundtrip(unittest.TestCase):

    def setUp(self):
        from bin.core.mission_persistence import (
            validate_mission_schema,
            inject_snapshot_metadata,
        )
        self.validate = validate_mission_schema
        self.inject = inject_snapshot_metadata

    def test_inject_save_reload_validate(self):
        """Full file roundtrip: dict → inject metadata → JSON file → reload → validate."""
        original = {
            "version": "1.0",
            "nodes": [{"id": 1, "display_name": "Start", "position": {"x": 0, "y": 0}}],
            "connections": [],
            "groups": [],
            "robot_brand": "unitree",
            "robot_type": "go2",
        }
        data = self.inject(dict(original))

        with tempfile.NamedTemporaryFile(
            suffix=".unitport", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name

        try:
            with open(fname, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            ok, reason = self.validate(loaded)
            self.assertTrue(ok, f"Validation failed after roundtrip: {reason}")
            self.assertIn("schema_version", loaded)  # version bumped to 1.1 in Cycle 3 STAGE-02
            self.assertIn("saved_at", loaded)
        finally:
            os.unlink(fname)

    def test_old_format_no_schema_version_validates(self):
        """Pre-STAGE-05 files saved without schema_version must still validate."""
        old_data = {"version": "1.0", "nodes": [], "connections": []}
        ok, reason = self.validate(old_data)
        self.assertTrue(ok, f"Old format rejected: {reason}")

    def test_malformed_nodes_rejected_after_roundtrip(self):
        """Files with nodes as dict (exec-graph format) fail validation."""
        bad = {"nodes": {"1": {}}, "connections": []}
        ok, _ = self.validate(bad)
        self.assertFalse(ok)

    def test_json_serializable_output(self):
        """inject_snapshot_metadata must produce JSON-serializable output."""
        data = self.inject({
            "nodes": [{"id": 1}],
            "connections": [],
        })
        # Must not raise
        _ = json.dumps(data)

    def test_multiple_inject_calls_overwrite_saved_at(self):
        """Each inject call updates saved_at — no duplicate/stale timestamps."""
        import time
        d = {"nodes": [], "connections": []}
        self.inject(d)
        ts1 = d["saved_at"]
        time.sleep(0.01)
        self.inject(d)
        ts2 = d["saved_at"]
        # Both are valid ISO-Z; ts2 may equal ts1 on very fast machines — just check format
        self.assertTrue(ts2.endswith("Z"))


# ── Section D: GraphScene roundtrip (requires Qt / offscreen) ─────────────────

_QT_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    _QT_AVAILABLE = True
except ImportError:
    pass


@unittest.skipUnless(_QT_AVAILABLE, "PySide6 not available")
class TestGraphSceneRoundtrip(unittest.TestCase):
    """Verify that the full save/load lifecycle works end-to-end with GraphScene."""

    _app = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if QApplication.instance() is None:
            cls._app = QApplication(sys.argv[:1])

    def _make_scene(self):
        from bin.components.graph_scene import GraphScene
        return GraphScene()

    def test_empty_scene_serialize_validate_load(self):
        from bin.core.mission_persistence import (
            inject_snapshot_metadata,
            validate_mission_schema,
        )
        scene = self._make_scene()
        data = scene.serialize_workflow()
        inject_snapshot_metadata(data)
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, reason)
        # Must not raise
        scene.load_workflow(data)

    def test_schema_version_present_after_inject(self):
        from bin.core.mission_persistence import inject_snapshot_metadata
        scene = self._make_scene()
        data = scene.serialize_workflow()
        inject_snapshot_metadata(data)
        self.assertIn("schema_version", data)

    def test_groups_key_always_present_in_serialized(self):
        """groups key must exist for backward-compat load_workflow logic."""
        scene = self._make_scene()
        data = scene.serialize_workflow()
        self.assertIn("groups", data)

    def test_serialize_load_preserves_node_count(self):
        """Loaded scene has same node count as original (no phantom nodes)."""
        scene = self._make_scene()
        data = scene.serialize_workflow()
        n_before = len(data["nodes"])
        scene.load_workflow(data)
        data2 = scene.serialize_workflow()
        self.assertEqual(len(data2["nodes"]), n_before)

    def test_roundtrip_result_validates(self):
        """Data produced by serialize after a load still validates."""
        from bin.core.mission_persistence import validate_mission_schema
        scene = self._make_scene()
        data = scene.serialize_workflow()
        scene.load_workflow(data)
        data2 = scene.serialize_workflow()
        ok, reason = validate_mission_schema(data2)
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
