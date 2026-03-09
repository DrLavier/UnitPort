#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Behavior panel workflow helpers (Cycle 4 STAGE-02)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bin.layout.behavior_panel import (  # noqa: E402
    _default_core_source_for_behavior_ref,
    _extract_steps_from_source,
    _render_steps_to_source,
)


class TestExtractStepsFromSource(unittest.TestCase):
    def test_extracts_call_like_lines(self):
        src = """
# comment
stand(duration=2.0)
walk(speed=0.3, duration=4.0)
invalid_line
sit()
"""
        steps = _extract_steps_from_source(src)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["action"], "stand")
        self.assertEqual(steps[0]["args"], "duration=2.0")
        self.assertEqual(steps[2]["action"], "sit")
        self.assertEqual(steps[2]["args"], "")

    def test_empty_source_returns_empty_list(self):
        self.assertEqual(_extract_steps_from_source(""), [])


class TestRenderStepsToSource(unittest.TestCase):
    def test_renders_steps_as_function_calls(self):
        steps = [
            {"action": "stand", "args": "duration=1.5"},
            {"action": "sit", "args": ""},
        ]
        out = _render_steps_to_source(steps)
        self.assertIn("stand(duration=1.5)", out)
        self.assertIn("sit()", out)

    def test_blank_action_falls_back_to_noop(self):
        out = _render_steps_to_source([{"action": "", "args": ""}])
        self.assertEqual(out.strip(), "noop()")

    def test_empty_steps_returns_empty_source(self):
        self.assertEqual(_render_steps_to_source([]), "")


class TestDefaultCoreSourceForBehaviorRef(unittest.TestCase):
    def test_seeded_from_selected_action(self):
        out = _default_core_source_for_behavior_ref("lift_right_leg")
        self.assertEqual(out, "lift_right_leg()")

    def test_walk_has_parameterized_default(self):
        out = _default_core_source_for_behavior_ref("walk")
        self.assertEqual(out, "walk(speed=0.3, duration=4.0)")

    def test_invalid_ref_falls_back_to_legacy_template(self):
        out = _default_core_source_for_behavior_ref("bad-ref")
        self.assertIn("stand(duration=2.0)", out)
        self.assertIn("walk(speed=0.3, duration=4.0)", out)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 6 — motor_track_expanded persisted in draft state
# (pure-Python; no Qt required)
# ===========================================================================

class TestMotorTrackExpandedPersistence(unittest.TestCase):
    """Verify get/set_behavior_drafts_state round-trips motor_track_expanded.

    Uses a minimal stub approach: we patch Qt so the module can import,
    then exercise only the draft-state helpers in isolation.
    """

    def _make_stub_panel(self):
        """Return a minimal object that mimics the relevant BehaviorPanel API."""
        import types, os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(sys.argv)

        from bin.core.config_manager import ConfigManager
        from bin.layout.behavior_panel import BehaviorPanel

        # Build a real BehaviorPanel (offscreen)
        panel = BehaviorPanel()
        return panel

    def test_motor_track_expanded_present_in_draft_after_set(self):
        panel = self._make_stub_panel()
        # Simulate having a node context
        panel._node_id = 42
        panel._node_name = "walk_node"
        # Set some expand state on the timeline view
        panel._timeline._motor_track_expanded["leg_group_left"] = False
        panel._timeline._motor_track_expanded["leg_group_right"] = True

        drafts = panel.get_behavior_drafts_state()
        # The current node's entry should have motor_track_expanded
        node_key = f"42::walk_node"
        self.assertIn(node_key, drafts)
        self.assertIn("motor_track_expanded", drafts[node_key])
        self.assertFalse(drafts[node_key]["motor_track_expanded"]["leg_group_left"])
        self.assertTrue(drafts[node_key]["motor_track_expanded"]["leg_group_right"])

    def test_motor_track_expanded_restored_from_draft(self):
        panel = self._make_stub_panel()
        panel._node_id = 10
        panel._node_name = "stand_node"

        drafts = {
            "10::stand_node": {
                "core": "stand()",
                "hb": {},
                "motor_track_expanded": {"leg_group_left": False, "body_posture": True},
            }
        }
        panel.set_behavior_drafts_state(drafts)

        # After restore, timeline should have updated expanded state
        expanded = panel._timeline._motor_track_expanded
        self.assertFalse(expanded.get("leg_group_left", True))
        self.assertTrue(expanded.get("body_posture", False))

    def test_missing_motor_track_expanded_key_is_tolerated(self):
        panel = self._make_stub_panel()
        panel._node_id = 99
        panel._node_name = "noop_node"
        # Draft without motor_track_expanded key — should not raise
        drafts = {
            "99::noop_node": {
                "core": "noop()",
                "hb": {},
            }
        }
        try:
            panel.set_behavior_drafts_state(drafts)
        except Exception as exc:
            self.fail(f"set_behavior_drafts_state raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
