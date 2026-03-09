#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for BehaviorPanel compile helpers — Phase 2 STAGE-04.

Coverage
--------
- _format_compile_output: valid artifact, invalid artifact with diagnostics,
  empty diagnostics, source_hash absent, truncation of artifact_id/source_hash,
  diagnostic location present/absent, multi-diagnostic list

Isolation
---------
- No QApplication, no Qt imports.
- Tests only the module-level _format_compile_output() function.
- BehaviorArtifact and BehaviorDiagnostic are imported directly from system.behavior.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the Qt-free module-level helper directly.
from bin.layout.behavior_panel import _format_compile_output  # noqa: E402
from system.behavior.behavior_artifact import (               # noqa: E402
    BehaviorArtifact,
    BehaviorDiagnostic,
)
from system.ir.workflow_ir import IRNode, NodeKind, WorkflowIR  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_ir() -> WorkflowIR:
    return WorkflowIR(nodes=[], edges=[])


def _one_node_ir() -> WorkflowIR:
    return WorkflowIR(
        nodes=[IRNode(id="n0", kind=NodeKind.ACTION, schema_id="stand", params={})],
        edges=[],
    )


def _valid_artifact(behavior_ref: str = "stand") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_one_node_ir(),
    )


def _invalid_artifact(behavior_ref: str = "broken") -> BehaviorArtifact:
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_empty_ir(),
        diagnostics=[BehaviorDiagnostic.error("E001", "forced failure")],
    )


def _artifact_with_location(behavior_ref: str = "loctest") -> BehaviorArtifact:
    diag = BehaviorDiagnostic(
        level="error",
        code="E002",
        message="bad call",
        location="line:4",
    )
    return BehaviorArtifact.create(
        behavior_ref=behavior_ref,
        behavior_ir=_empty_ir(),
        diagnostics=[diag],
    )


# ===========================================================================
# TestFormatCompileOutputValid
# ===========================================================================

class TestFormatCompileOutputValid(unittest.TestCase):
    """_format_compile_output with a valid, clean artifact."""

    def setUp(self):
        self.artifact = _valid_artifact("stand")
        self.output = _format_compile_output(self.artifact)

    def test_status_tag_ok(self):
        self.assertIn("[OK]", self.output)

    def test_behavior_ref_present(self):
        self.assertIn("behavior_ref='stand'", self.output)

    def test_artifact_id_prefix_present(self):
        # First 8 chars of artifact_id followed by ellipsis
        prefix = self.artifact.artifact_id[:8]
        self.assertIn(prefix, self.output)

    def test_artifact_id_truncated_with_ellipsis(self):
        self.assertTrue(("..." in self.output) or ("…" in self.output))

    def test_node_count_present(self):
        self.assertIn("nodes=1", self.output)

    def test_edge_count_present(self):
        self.assertIn("edges=0", self.output)

    def test_compiled_at_present(self):
        self.assertIn("compiled_at=", self.output)

    def test_zero_diagnostics_line(self):
        self.assertIn("0 diagnostics", self.output)

    def test_no_error_count_line(self):
        # "error(s)" only appears when diagnostics exist
        self.assertNotIn("error(s)", self.output)

    def test_source_hash_present_when_set(self):
        # compile() generates a source_hash via _source_hash(); our fixture
        # uses BehaviorArtifact.create() which sets source_hash="" unless
        # explicitly supplied.  The helper should not crash either way.
        if self.artifact.source_hash:
            self.assertIn("source_hash=", self.output)

    def test_returns_string(self):
        self.assertIsInstance(self.output, str)

    def test_multiline(self):
        self.assertGreater(self.output.count("\n"), 0)


# ===========================================================================
# TestFormatCompileOutputInvalid
# ===========================================================================

class TestFormatCompileOutputInvalid(unittest.TestCase):
    """_format_compile_output with an invalid artifact (compile errors)."""

    def setUp(self):
        self.artifact = _invalid_artifact("broken")
        self.output = _format_compile_output(self.artifact)

    def test_status_tag_error(self):
        self.assertIn("[ERROR]", self.output)

    def test_behavior_ref_present(self):
        self.assertIn("behavior_ref='broken'", self.output)

    def test_error_count_line_present(self):
        self.assertIn("error(s)", self.output)

    def test_error_count_is_one(self):
        self.assertIn("1 error(s)", self.output)

    def test_warning_count_is_zero(self):
        self.assertIn("0 warning(s)", self.output)

    def test_diagnostic_code_present(self):
        self.assertIn("E001", self.output)

    def test_diagnostic_message_present(self):
        self.assertIn("forced failure", self.output)

    def test_diagnostic_level_uppercased(self):
        self.assertIn("[ERROR]", self.output)

    def test_no_zero_diagnostics_line(self):
        self.assertNotIn("0 diagnostics", self.output)

    def test_nodes_is_zero(self):
        self.assertIn("nodes=0", self.output)


# ===========================================================================
# TestFormatCompileOutputDiagnosticLocation
# ===========================================================================

class TestFormatCompileOutputDiagnosticLocation(unittest.TestCase):
    """_format_compile_output: location field in diagnostic output."""

    def test_location_present_when_set(self):
        artifact = _artifact_with_location("loctest")
        output = _format_compile_output(artifact)
        self.assertIn("[line:4]", output)

    def test_no_location_brackets_when_empty(self):
        artifact = _invalid_artifact("no_loc")
        output = _format_compile_output(artifact)
        # The diagnostic has no location, so no "[]" appended to that line
        diag_lines = [ln for ln in output.splitlines() if "E001" in ln]
        self.assertTrue(diag_lines, "diagnostic line not found")
        self.assertNotIn("[", diag_lines[0].split("E001")[1])


# ===========================================================================
# TestFormatCompileOutputMultipleDiagnostics
# ===========================================================================

class TestFormatCompileOutputMultipleDiagnostics(unittest.TestCase):
    """_format_compile_output: multiple diagnostics render all entries."""

    def setUp(self):
        diags = [
            BehaviorDiagnostic.error("E001", "first error"),
            BehaviorDiagnostic.warning("W001", "first warning"),
            BehaviorDiagnostic.error("E002", "second error"),
        ]
        self.artifact = BehaviorArtifact.create(
            behavior_ref="multi",
            behavior_ir=_empty_ir(),
            diagnostics=diags,
        )
        self.output = _format_compile_output(self.artifact)

    def test_error_count_is_two(self):
        self.assertIn("2 error(s)", self.output)

    def test_warning_count_is_one(self):
        self.assertIn("1 warning(s)", self.output)

    def test_all_codes_present(self):
        self.assertIn("E001", self.output)
        self.assertIn("W001", self.output)
        self.assertIn("E002", self.output)

    def test_all_messages_present(self):
        self.assertIn("first error", self.output)
        self.assertIn("first warning", self.output)
        self.assertIn("second error", self.output)

    def test_warning_level_uppercased(self):
        self.assertIn("[WARNING]", self.output)


# ===========================================================================
# TestFormatCompileOutputNoSourceHash
# ===========================================================================

class TestFormatCompileOutputNoSourceHash(unittest.TestCase):
    """_format_compile_output: absent source_hash does not appear in output."""

    def test_no_source_hash_line_when_empty(self):
        artifact = BehaviorArtifact.create(
            behavior_ref="nohash",
            behavior_ir=_one_node_ir(),
            source_hash="",
        )
        output = _format_compile_output(artifact)
        self.assertNotIn("source_hash=", output)

    def test_source_hash_line_present_when_supplied(self):
        artifact = BehaviorArtifact.create(
            behavior_ref="hashtest",
            behavior_ir=_one_node_ir(),
            source_hash="abcdef1234567890abcdef1234567890",
        )
        output = _format_compile_output(artifact)
        self.assertIn("source_hash=", output)
        # First 16 chars should appear
        self.assertIn("abcdef1234567890", output)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 1 — BehaviorCompileWorker timeline / is_simulation wiring
# ===========================================================================

class TestCompileWorkerTimelineWiring(unittest.TestCase):
    """Verify BehaviorCompileWorker passes timeline and is_simulation to bridge.

    Uses a mock bridge so no Qt main loop is required; the worker's run()
    method is called synchronously in the test thread.
    """

    def _make_timeline(self):
        from system.behavior.action_profile import BehaviorTimeline
        return BehaviorTimeline()

    def test_worker_passes_timeline_to_bridge(self):
        """run() must forward timeline= to bridge.compile()."""
        from unittest.mock import MagicMock
        from bin.layout.behavior_panel import BehaviorCompileWorker

        bridge = MagicMock()
        bridge.compile.return_value = _valid_artifact("ref1")
        tl = self._make_timeline()

        worker = BehaviorCompileWorker(
            bridge=bridge,
            source="stand()",
            behavior_ref="ref1",
            robot_type="go2",
            timeline=tl,
            is_simulation=False,
        )
        # Call run() directly (no QThread.start needed)
        worker.run()

        bridge.compile.assert_called_once()
        _, kwargs = bridge.compile.call_args
        self.assertIs(kwargs.get("timeline"), tl)

    def test_worker_passes_is_simulation_false(self):
        from unittest.mock import MagicMock
        from bin.layout.behavior_panel import BehaviorCompileWorker

        bridge = MagicMock()
        bridge.compile.return_value = _valid_artifact("ref2")

        worker = BehaviorCompileWorker(
            bridge=bridge,
            source="stand()",
            behavior_ref="ref2",
            is_simulation=False,
        )
        worker.run()

        _, kwargs = bridge.compile.call_args
        self.assertFalse(kwargs.get("is_simulation"))

    def test_worker_passes_is_simulation_true(self):
        from unittest.mock import MagicMock
        from bin.layout.behavior_panel import BehaviorCompileWorker

        bridge = MagicMock()
        bridge.compile.return_value = _valid_artifact("ref3")

        worker = BehaviorCompileWorker(
            bridge=bridge,
            source="stand()",
            behavior_ref="ref3",
            is_simulation=True,
        )
        worker.run()

        _, kwargs = bridge.compile.call_args
        self.assertTrue(kwargs.get("is_simulation"))

    def test_worker_none_timeline_forwards_none(self):
        from unittest.mock import MagicMock
        from bin.layout.behavior_panel import BehaviorCompileWorker

        bridge = MagicMock()
        bridge.compile.return_value = _valid_artifact("ref4")

        worker = BehaviorCompileWorker(
            bridge=bridge,
            source="stand()",
            behavior_ref="ref4",
            timeline=None,
        )
        worker.run()

        _, kwargs = bridge.compile.call_args
        self.assertIsNone(kwargs.get("timeline"))


if __name__ == "__main__":
    unittest.main()
