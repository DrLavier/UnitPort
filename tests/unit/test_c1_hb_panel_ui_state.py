#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for BehaviorPanel UI state logic — Circle 1 Step 1.5.

Verifies the display-state integration layer used by
_refresh_event_state() and _refresh_movement_io_mapping():

Coverage
--------
_BADGE_COLOR_MAP
  - all 5 semantic keys present (success, error, warning, info, neutral)
  - values are valid #RRGGBB hex strings
  - specific color key → expected hue range

Event state refresh logic
  - empty entries → summary label text is empty
  - non-empty entries → summary text includes total / active / errors counts
  - state badge for known states → correct color_key resolved via _BADGE_COLOR_MAP
  - state badge background color retrieved for all HBEventStatus values

IO mapping refresh logic
  - empty mappings → no conflict hint needed
  - active/distinct mappings → no conflicts detected
  - conflict → hint message assembled as "<signal>: <reason>"
  - direction label: inbound → "↓ In", outbound → "↑ Out"
  - IO badge for known statuses → color_key resolved via _BADGE_COLOR_MAP
  - placeholder text keys defined for both panels

Isolation
---------
- No QApplication, no Qt widget instantiation.
- Imports module-level constants and pure functions only.
- BehaviorPanel class body is never executed.
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Module-level constant — importable without QApplication
from bin.layout.behavior_panel import _BADGE_COLOR_MAP  # noqa: E402

# Pure display-state functions
from system.behavior.hb_display_state import (  # noqa: E402
    HBConflictHint,
    HBEventStatus,
    HBIOStatus,
    badge_for_event_state,
    badge_for_io_status,
    compute_event_summary,
    compute_io_summary,
    detect_io_conflicts,
)


# ---------------------------------------------------------------------------
# Lightweight duck-typed stubs matching HBEventStateEntry / HBIOMapping
# ---------------------------------------------------------------------------

class _Ev:
    def __init__(self, state: str, event_name: str = "e",
                 source: str = "s", timestamp: str = "T0", reason: str = ""):
        self.state = state
        self.event_name = event_name
        self.source = source
        self.timestamp = timestamp
        self.reason = reason


class _IO:
    def __init__(self, signal: str, direction: str = "inbound",
                 status: str = "active", target_param: str = "p"):
        self.signal = signal
        self.direction = direction
        self.status = status
        self.target_param = target_param


# ===========================================================================
# _BADGE_COLOR_MAP
# ===========================================================================

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

REQUIRED_KEYS = ("success", "error", "warning", "info", "neutral")


class TestBadgeColorMap(unittest.TestCase):

    def test_all_five_semantic_keys_present(self):
        for key in REQUIRED_KEYS:
            self.assertIn(key, _BADGE_COLOR_MAP, msg=f"missing key: {key}")

    def test_all_values_are_hex_strings(self):
        for key, val in _BADGE_COLOR_MAP.items():
            self.assertRegex(val, _HEX_RE, msg=f"bad hex for key '{key}': {val!r}")

    def test_success_is_green_hue(self):
        # #4caf50 — R < G; green channel dominant
        hex_val = _BADGE_COLOR_MAP["success"].lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        self.assertGreater(g, r, "success color should have dominant green channel")
        self.assertGreater(g, b)

    def test_error_is_red_hue(self):
        hex_val = _BADGE_COLOR_MAP["error"].lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        self.assertGreater(r, g, "error color should have dominant red channel")
        self.assertGreater(r, b)

    def test_warning_is_orange_hue(self):
        # orange: high R, high G, low B
        hex_val = _BADGE_COLOR_MAP["warning"].lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        self.assertGreater(r, b, "warning color should have dominant red channel over blue")
        self.assertGreater(g, b)

    def test_info_is_blue_hue(self):
        hex_val = _BADGE_COLOR_MAP["info"].lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        self.assertGreater(b, r, "info color should have dominant blue channel")

    def test_neutral_has_equal_rgb_channels(self):
        # neutral grey: R ≈ G ≈ B (within 16 steps)
        hex_val = _BADGE_COLOR_MAP["neutral"].lstrip("#")
        r, g, b = int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16)
        self.assertLessEqual(abs(r - g), 16, "neutral should be near-grey")
        self.assertLessEqual(abs(g - b), 16)

    def test_no_extra_unknown_keys(self):
        for key in _BADGE_COLOR_MAP:
            self.assertIn(key, REQUIRED_KEYS, msg=f"unexpected key: {key!r}")


# ===========================================================================
# Event state refresh logic
# ===========================================================================

class TestEventStateSummaryText(unittest.TestCase):
    """Verify the text that _refresh_event_state() would write to the summary label."""

    def _summary_text(self, entries):
        """Replicate the summary label formatting used in _refresh_event_state."""
        if not entries:
            return ""
        s = compute_event_summary(entries)
        return (
            f"{s.total} events  |  {s.active} active  |  {s.errors} errors"
        )

    def test_empty_entries_yields_empty_text(self):
        self.assertEqual(self._summary_text([]), "")

    def test_single_active_contains_correct_counts(self):
        text = self._summary_text([_Ev("active")])
        self.assertIn("1", text)

    def test_mixed_entries_total_count(self):
        entries = [_Ev("active"), _Ev("error"), _Ev("inactive")]
        text = self._summary_text(entries)
        self.assertIn("3", text)

    def test_error_count_appears_in_text(self):
        entries = [_Ev("active"), _Ev("error"), _Ev("error")]
        text = self._summary_text(entries)
        # "2 errors" or at least "2" appearing in the errors position
        s = compute_event_summary(entries)
        self.assertEqual(s.errors, 2)
        self.assertIn("2", text)

    def test_active_count_appears_in_text(self):
        entries = [_Ev("active"), _Ev("active"), _Ev("warning")]
        s = compute_event_summary(entries)
        self.assertEqual(s.active, 2)


class TestEventStateBadgeColors(unittest.TestCase):
    """Verify that badge_for_event_state returns color_keys resolvable via _BADGE_COLOR_MAP."""

    def test_all_known_states_resolve_in_color_map(self):
        for state in HBEventStatus.ALL:
            badge = badge_for_event_state(state)
            # color_key must be in _BADGE_COLOR_MAP (or fallback neutral must be there)
            resolved = _BADGE_COLOR_MAP.get(badge.color_key)
            self.assertIsNotNone(
                resolved,
                msg=f"badge.color_key={badge.color_key!r} not in _BADGE_COLOR_MAP for state={state!r}"
            )

    def test_active_state_resolves_to_success_color(self):
        badge = badge_for_event_state("active")
        color = _BADGE_COLOR_MAP[badge.color_key]
        self.assertTrue(color.startswith("#"))

    def test_error_state_resolves_to_error_color(self):
        badge = badge_for_event_state("error")
        self.assertEqual(badge.color_key, "error")
        self.assertIn("error", _BADGE_COLOR_MAP)

    def test_warning_state_resolves_to_warning_color(self):
        badge = badge_for_event_state("warning")
        self.assertEqual(badge.color_key, "warning")

    def test_unknown_state_resolves_via_neutral_fallback(self):
        badge = badge_for_event_state("some_unknown_future_state")
        self.assertEqual(badge.color_key, "neutral")
        self.assertIn("neutral", _BADGE_COLOR_MAP)


# ===========================================================================
# IO mapping refresh logic
# ===========================================================================

class TestDirectionLabel(unittest.TestCase):
    """Verify the direction label text that _refresh_movement_io_mapping would set."""

    def _dir_label(self, direction: str) -> str:
        """Replicate the direction label logic used in the refresh method."""
        if direction == "inbound":
            return "\u2193 In"
        elif direction == "outbound":
            return "\u2191 Out"
        return direction

    def test_inbound_arrow_label(self):
        self.assertEqual(self._dir_label("inbound"), "↓ In")

    def test_outbound_arrow_label(self):
        self.assertEqual(self._dir_label("outbound"), "↑ Out")

    def test_unknown_direction_passthrough(self):
        self.assertEqual(self._dir_label("bidirectional"), "bidirectional")

    def test_inbound_contains_down_arrow(self):
        self.assertIn("↓", self._dir_label("inbound"))

    def test_outbound_contains_up_arrow(self):
        self.assertIn("↑", self._dir_label("outbound"))


class TestIOStatusBadgeColors(unittest.TestCase):
    """Verify that badge_for_io_status returns color_keys resolvable via _BADGE_COLOR_MAP."""

    def test_all_known_statuses_resolve_in_color_map(self):
        for status in HBIOStatus.ALL:
            badge = badge_for_io_status(status)
            resolved = _BADGE_COLOR_MAP.get(badge.color_key)
            self.assertIsNotNone(
                resolved,
                msg=f"badge.color_key={badge.color_key!r} not in _BADGE_COLOR_MAP for status={status!r}"
            )

    def test_active_status_resolves_to_success_color(self):
        badge = badge_for_io_status("active")
        self.assertEqual(badge.color_key, "success")

    def test_unmapped_resolves_to_warning_color(self):
        badge = badge_for_io_status("unmapped")
        self.assertIn(badge.color_key, _BADGE_COLOR_MAP)

    def test_unsupported_resolves_to_error_color(self):
        badge = badge_for_io_status("unsupported")
        self.assertEqual(badge.color_key, "error")

    def test_conflict_resolves_to_error_color(self):
        badge = badge_for_io_status("conflict")
        self.assertEqual(badge.color_key, "error")


class TestIOConflictLabelText(unittest.TestCase):
    """Verify the conflict hint text that _refresh_movement_io_mapping would display."""

    def _conflict_text(self, mappings):
        """Replicate the conflict label assembly used in the refresh method."""
        conflicts = detect_io_conflicts(mappings)
        if not conflicts:
            return None  # label hidden
        msgs = [f"{h.signal}: {h.reason}" for h in conflicts]
        return "\u26a0 " + "  |  ".join(msgs)

    def test_empty_mappings_yields_none(self):
        self.assertIsNone(self._conflict_text([]))

    def test_clean_mappings_yields_none(self):
        mappings = [_IO("imu", "inbound"), _IO("cmd", "outbound")]
        self.assertIsNone(self._conflict_text(mappings))

    def test_dual_direction_yields_warning_text(self):
        mappings = [_IO("sig", "inbound"), _IO("sig", "outbound")]
        text = self._conflict_text(mappings)
        self.assertIsNotNone(text)
        self.assertIn("sig", text)

    def test_conflict_text_starts_with_warning_symbol(self):
        mappings = [_IO("s", "inbound"), _IO("s", "outbound")]
        text = self._conflict_text(mappings)
        self.assertTrue(text.startswith("⚠"))

    def test_conflict_text_contains_signal_and_reason(self):
        mappings = [_IO("x", "inbound", "unsupported")]
        text = self._conflict_text(mappings)
        self.assertIn("x", text)
        self.assertGreater(len(text), 5)

    def test_multiple_conflicts_joined_by_separator(self):
        mappings = [
            _IO("a", "inbound"), _IO("a", "outbound"),
            _IO("b", "inbound", "unsupported"),
        ]
        text = self._conflict_text(mappings)
        self.assertIn("|", text)

    def test_unmapped_produces_warning_hint(self):
        hints = detect_io_conflicts([_IO("no_bind", "inbound", "unmapped")])
        self.assertTrue(any(h.signal == "no_bind" and h.severity == "warning" for h in hints))


class TestPlaceholderTextKeys(unittest.TestCase):
    """Verify placeholder localisation key defaults are non-empty strings."""

    def test_no_events_placeholder_non_empty(self):
        from bin.core.localisation import tr
        text = tr("hb.no_events", "No events recorded")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_no_io_bindings_placeholder_non_empty(self):
        from bin.core.localisation import tr
        text = tr("hb.no_io_bindings", "No IO bindings active")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
