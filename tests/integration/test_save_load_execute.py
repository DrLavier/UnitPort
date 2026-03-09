"""STAGE-05 — Save → Load → Execute end-to-end roundtrip tests.

Verifies that the full persistence lifecycle is non-regressive:

    GraphScene.serialize_workflow()
      → inject_snapshot_metadata()
      → validate_mission_schema()
      → JSON file I/O (write + read)
      → validate_mission_schema()
      → GraphScene.load_workflow()
      → GraphScene.get_execution_graph()
      → RuntimeEngine.execute()
      → assert run_result structure

Done criteria (STAGE-05):
  "Missions roundtrip (save→load→execute) without structural or runtime
  regressions."

All tests use QT_QPA_PLATFORM=offscreen so they run headlessly in CI.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# ── Qt availability guard ─────────────────────────────────────────────────────

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_QT_AVAILABLE = False
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    _QT_AVAILABLE = True
except ImportError:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app():
    if QApplication.instance() is None:
        return QApplication(sys.argv[:1])
    return QApplication.instance()


def _make_scene():
    from bin.components.graph_scene import GraphScene
    return GraphScene()


def _run_engine(exec_graph):
    from system.runtime import RuntimeEngine
    engine = RuntimeEngine()
    return engine.execute(exec_graph, {})


# ── Suite A: In-memory roundtrip (no file I/O) ────────────────────────────────

@unittest.skipUnless(_QT_AVAILABLE, "PySide6 not available")
class TestInMemoryRoundtrip(unittest.TestCase):
    """Serialize → validate → load → execute without touching the filesystem."""

    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = _make_app()

    def test_serialize_then_execute_returns_dict(self):
        scene = _make_scene()
        exec_graph = scene.get_execution_graph()
        run_result = _run_engine(exec_graph)
        self.assertIsInstance(run_result, dict)

    def test_serialize_validate_load_execute_returns_dict(self):
        from bin.core.mission_persistence import (
            inject_snapshot_metadata,
            validate_mission_schema,
        )
        scene = _make_scene()
        data = inject_snapshot_metadata(scene.serialize_workflow())
        ok, reason = validate_mission_schema(data)
        self.assertTrue(ok, f"Validation failed: {reason}")

        scene2 = _make_scene()
        scene2.load_workflow(data)
        exec_graph = scene2.get_execution_graph()
        run_result = _run_engine(exec_graph)
        self.assertIsInstance(run_result, dict)

    def test_run_result_has_status_key(self):
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        run_result = _run_engine(scene2.get_execution_graph())
        self.assertIn("status", run_result)

    def test_run_result_has_diagnostics_key(self):
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        run_result = _run_engine(scene2.get_execution_graph())
        self.assertIn("diagnostics", run_result)

    def test_run_result_status_is_valid_string(self):
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        run_result = _run_engine(scene2.get_execution_graph())
        self.assertIn(run_result["status"], {"success", "failed", "blocked"})

    def test_run_result_not_blocked_for_default_scene(self):
        """A default canvas should not be blocked by compile/safety checks."""
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        run_result = _run_engine(scene2.get_execution_graph())
        # The default scene has no malformed IR and no safety triggers; it
        # must not be blocked.  (It may succeed or fail depending on node count.)
        self.assertNotEqual(run_result["status"], "blocked")

    def test_run_result_has_results_key(self):
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        run_result = _run_engine(scene2.get_execution_graph())
        self.assertIn("results", run_result)

    def test_no_exception_raised_during_execute(self):
        scene = _make_scene()
        data = scene.serialize_workflow()
        scene2 = _make_scene()
        scene2.load_workflow(data)
        try:
            _run_engine(scene2.get_execution_graph())
        except Exception as exc:  # noqa: BLE001
            self.fail(f"RuntimeEngine.execute() raised unexpectedly: {exc}")


# ── Suite B: File I/O roundtrip (save to disk, reload, execute) ───────────────

@unittest.skipUnless(_QT_AVAILABLE, "PySide6 not available")
class TestFileRoundtripExecute(unittest.TestCase):
    """Full save→load→execute pipeline with real file I/O."""

    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = _make_app()

    def _full_roundtrip(self):
        """Helper: save → file → reload → load scene → execute.

        Returns (run_result, tmp_path).
        Caller is responsible for deleting tmp_path.
        """
        from bin.core.mission_persistence import (
            inject_snapshot_metadata,
            validate_mission_schema,
        )

        scene = _make_scene()
        data = inject_snapshot_metadata(scene.serialize_workflow(), source="test_harness")

        ok, reason = validate_mission_schema(data)
        assert ok, f"Pre-save validation failed: {reason}"

        with tempfile.NamedTemporaryFile(
            suffix=".unitport", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fname = fh.name

        with open(fname, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)

        ok, reason = validate_mission_schema(loaded)
        assert ok, f"Post-reload validation failed: {reason}"

        scene2 = _make_scene()
        scene2.load_workflow(loaded)
        exec_graph = scene2.get_execution_graph()
        run_result = _run_engine(exec_graph)
        return run_result, fname

    def test_file_roundtrip_returns_dict(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertIsInstance(run_result, dict)

    def test_file_roundtrip_status_present(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertIn("status", run_result)

    def test_file_roundtrip_diagnostics_present(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertIn("diagnostics", run_result)

    def test_file_roundtrip_results_present(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertIn("results", run_result)

    def test_file_roundtrip_node_count_key_present(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertIn("node_count", run_result)

    def test_file_roundtrip_not_blocked(self):
        run_result, fname = self._full_roundtrip()
        os.unlink(fname)
        self.assertNotEqual(run_result["status"], "blocked")

    def test_file_roundtrip_no_exception(self):
        try:
            run_result, fname = self._full_roundtrip()
            os.unlink(fname)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"Full roundtrip raised unexpectedly: {exc}")

    def test_saved_file_contains_source_field(self):
        """Saved file must carry the source metadata field."""
        from bin.core.mission_persistence import inject_snapshot_metadata

        scene = _make_scene()
        data = inject_snapshot_metadata(scene.serialize_workflow(), source="test_harness")

        with tempfile.NamedTemporaryFile(
            suffix=".unitport", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name

        try:
            with open(fname, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded.get("source"), "test_harness")
        finally:
            os.unlink(fname)

    def test_saved_file_contains_schema_version(self):
        from bin.core.mission_persistence import inject_snapshot_metadata, MISSION_SCHEMA_VERSION

        scene = _make_scene()
        data = inject_snapshot_metadata(scene.serialize_workflow())

        with tempfile.NamedTemporaryFile(
            suffix=".unitport", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            json.dump(data, fh)
            fname = fh.name

        try:
            with open(fname, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded.get("schema_version"), MISSION_SCHEMA_VERSION)
        finally:
            os.unlink(fname)

    def test_old_mission_file_without_source_still_executes(self):
        """Files saved before source was added must still load and execute."""
        from bin.core.mission_persistence import validate_mission_schema

        # Simulate a pre-source mission file (no source key)
        old_data = {
            "version": "1.0",
            "schema_version": "1.0",
            "nodes": [],
            "connections": [],
            "groups": [],
            "robot_brand": "",
            "robot_type": "",
        }
        ok, reason = validate_mission_schema(old_data)
        self.assertTrue(ok, f"Old file rejected by validator: {reason}")

        scene2 = _make_scene()
        scene2.load_workflow(old_data)
        run_result = _run_engine(scene2.get_execution_graph())
        self.assertIn("status", run_result)
        self.assertNotEqual(run_result["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
