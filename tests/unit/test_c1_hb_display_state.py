#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system/behavior/hb_display_state.py — Circle 1 Step 1.5.

Coverage
--------
HBEventStatus / HBIOStatus
  - all constants are distinct non-empty strings
  - ALL tuples contain all defined constants

HBEventBadge / HBIOBadge / HBConflictHint / HBEventSummary / HBIOSummary
  - dataclass fields correct types

badge_for_event_state()
  - known states return correct label and color_key
  - unknown string → neutral badge, no raise

badge_for_io_status()
  - known statuses return correct label and color_key
  - unknown string → neutral badge, no raise

detect_io_conflicts()
  - empty list → []
  - all distinct, all "active" → []
  - same signal inbound AND outbound → error conflict
  - multiple dual-direction conflicts → all returned
  - status="unsupported" → error conflict
  - status="unmapped" → warning conflict
  - mixed: duplicate + unsupported → both hints

compute_event_summary()
  - empty → all zeros
  - mixed states → correct counts

compute_io_summary()
  - empty → all zeros
  - inbound/outbound split; conflicts from hint list

Isolation
---------
- No QApplication. No Qt. Pure-Python only.
- HBEventStateEntry and HBIOMapping are duck-typed stubs defined in this file;
  no dependency on bin.layout.behavior_panel (which imports Qt at module level).
"""

import sys
import unittest
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# hb_display_state is a pure Python file with no Qt/SDK imports — import directly.
from system.behavior.hb_display_state import (  # noqa: E402
    HBConflictHint,
    HBEventBadge,
    HBEventStatus,
    HBEventSummary,
    HBIOBadge,
    HBIOStatus,
    HBIOSummary,
    badge_for_event_state,
    badge_for_io_status,
    compute_event_summary,
    compute_io_summary,
    detect_io_conflicts,
)


# ---------------------------------------------------------------------------
# Lightweight duck-typed stubs matching HBEventStateEntry / HBIOMapping fields
# ---------------------------------------------------------------------------

class _EventEntry:
    """Minimal stand-in for HBEventStateEntry (only fields used by display_state)."""
    def __init__(self, state: str, event_name: str = "e",
                 source: str = "s", timestamp: str = "T0", reason: str = ""):
        self.event_name = event_name
        self.state = state
        self.source = source
        self.timestamp = timestamp
        self.reason = reason


class _IOMapping:
    """Minimal stand-in for HBIOMapping (only fields used by display_state)."""
    def __init__(self, signal: str, direction: str = "inbound",
                 status: str = "active", target_param: str = ""):
        self.signal = signal
        self.target_param = target_param or signal
        self.direction = direction
        self.status = status


def _ev(state: str) -> _EventEntry:
    return _EventEntry(state=state)


def _io(signal: str, direction: str = "inbound", status: str = "active") -> _IOMapping:
    return _IOMapping(signal=signal, direction=direction, status=status)


# ===========================================================================
# HBEventStatus
# ===========================================================================

class TestHBEventStatus(unittest.TestCase):

    def test_all_constants_are_strings(self):
        for name in ("ACTIVE", "INACTIVE", "PENDING", "ERROR", "WARNING", "UNKNOWN"):
            val = getattr(HBEventStatus, name)
            self.assertIsInstance(val, str, msg=f"HBEventStatus.{name}")

    def test_all_constants_are_non_empty(self):
        for name in ("ACTIVE", "INACTIVE", "PENDING", "ERROR", "WARNING", "UNKNOWN"):
            self.assertTrue(getattr(HBEventStatus, name))

    def test_all_constants_are_distinct(self):
        vals = [HBEventStatus.ACTIVE, HBEventStatus.INACTIVE, HBEventStatus.PENDING,
                HBEventStatus.ERROR, HBEventStatus.WARNING, HBEventStatus.UNKNOWN]
        self.assertEqual(len(vals), len(set(vals)))

    def test_all_tuple_contains_all_constants(self):
        for c in (HBEventStatus.ACTIVE, HBEventStatus.INACTIVE, HBEventStatus.PENDING,
                  HBEventStatus.ERROR, HBEventStatus.WARNING, HBEventStatus.UNKNOWN):
            self.assertIn(c, HBEventStatus.ALL)

    def test_all_tuple_has_six_entries(self):
        self.assertEqual(len(HBEventStatus.ALL), 6)


# ===========================================================================
# HBIOStatus
# ===========================================================================

class TestHBIOStatus(unittest.TestCase):

    def test_all_constants_are_strings(self):
        for name in ("ACTIVE", "UNMAPPED", "UNSUPPORTED", "CONFLICT", "UNKNOWN"):
            val = getattr(HBIOStatus, name)
            self.assertIsInstance(val, str, msg=f"HBIOStatus.{name}")

    def test_all_constants_are_distinct(self):
        vals = [HBIOStatus.ACTIVE, HBIOStatus.UNMAPPED, HBIOStatus.UNSUPPORTED,
                HBIOStatus.CONFLICT, HBIOStatus.UNKNOWN]
        self.assertEqual(len(vals), len(set(vals)))

    def test_all_tuple_contains_all_constants(self):
        for c in (HBIOStatus.ACTIVE, HBIOStatus.UNMAPPED, HBIOStatus.UNSUPPORTED,
                  HBIOStatus.CONFLICT, HBIOStatus.UNKNOWN):
            self.assertIn(c, HBIOStatus.ALL)

    def test_all_tuple_has_five_entries(self):
        self.assertEqual(len(HBIOStatus.ALL), 5)


# ===========================================================================
# HBEventBadge / HBIOBadge
# ===========================================================================

class TestHBEventBadge(unittest.TestCase):

    def test_label_is_str(self):
        b = HBEventBadge(label="Active", color_key="success")
        self.assertIsInstance(b.label, str)

    def test_color_key_is_str(self):
        b = HBEventBadge(label="Active", color_key="success")
        self.assertIsInstance(b.color_key, str)

    def test_fields_stored_correctly(self):
        b = HBEventBadge(label="My Label", color_key="error")
        self.assertEqual(b.label, "My Label")
        self.assertEqual(b.color_key, "error")


class TestHBIOBadge(unittest.TestCase):

    def test_label_is_str(self):
        b = HBIOBadge(label="OK", color_key="success")
        self.assertIsInstance(b.label, str)

    def test_color_key_is_str(self):
        b = HBIOBadge(label="OK", color_key="success")
        self.assertIsInstance(b.color_key, str)


# ===========================================================================
# HBConflictHint
# ===========================================================================

class TestHBConflictHint(unittest.TestCase):

    def test_fields_exist(self):
        h = HBConflictHint(signal="imu.pitch", reason="dual direction", severity="error")
        self.assertEqual(h.signal, "imu.pitch")
        self.assertEqual(h.reason, "dual direction")
        self.assertEqual(h.severity, "error")

    def test_severity_warning(self):
        h = HBConflictHint(signal="battery", reason="unmapped", severity="warning")
        self.assertEqual(h.severity, "warning")


# ===========================================================================
# HBEventSummary / HBIOSummary
# ===========================================================================

class TestHBEventSummary(unittest.TestCase):

    def test_fields_exist(self):
        s = HBEventSummary(total=5, active=3, errors=1, warnings=1)
        self.assertEqual(s.total, 5)
        self.assertEqual(s.active, 3)
        self.assertEqual(s.errors, 1)
        self.assertEqual(s.warnings, 1)


class TestHBIOSummary(unittest.TestCase):

    def test_fields_exist(self):
        s = HBIOSummary(total=6, inbound=4, outbound=2, conflicts=1)
        self.assertEqual(s.total, 6)
        self.assertEqual(s.inbound, 4)
        self.assertEqual(s.outbound, 2)
        self.assertEqual(s.conflicts, 1)


# ===========================================================================
# badge_for_event_state()
# ===========================================================================

class TestBadgeForEventState(unittest.TestCase):

    def test_active_yields_success(self):
        b = badge_for_event_state("active")
        self.assertEqual(b.color_key, "success")

    def test_active_label_non_empty(self):
        b = badge_for_event_state("active")
        self.assertGreater(len(b.label), 0)

    def test_inactive_yields_neutral(self):
        b = badge_for_event_state("inactive")
        self.assertEqual(b.color_key, "neutral")

    def test_error_yields_error(self):
        b = badge_for_event_state("error")
        self.assertEqual(b.color_key, "error")

    def test_warning_yields_warning(self):
        b = badge_for_event_state("warning")
        self.assertEqual(b.color_key, "warning")

    def test_pending_yields_info(self):
        b = badge_for_event_state("pending")
        self.assertEqual(b.color_key, "info")

    def test_unknown_string_does_not_raise(self):
        try:
            badge_for_event_state("some_future_state")
        except Exception as exc:
            self.fail(f"badge_for_event_state raised: {exc}")

    def test_unknown_string_returns_badge_type(self):
        b = badge_for_event_state("some_future_state")
        self.assertIsInstance(b, HBEventBadge)

    def test_unknown_string_yields_neutral(self):
        b = badge_for_event_state("some_future_state")
        self.assertEqual(b.color_key, "neutral")

    def test_empty_string_does_not_raise(self):
        badge_for_event_state("")

    def test_unknown_state_returns_state_as_label(self):
        b = badge_for_event_state("weird_state")
        self.assertEqual(b.label, "weird_state")


# ===========================================================================
# badge_for_io_status()
# ===========================================================================

class TestBadgeForIOStatus(unittest.TestCase):

    def test_active_label_ok(self):
        b = badge_for_io_status("active")
        self.assertEqual(b.label, "OK")

    def test_active_yields_success(self):
        b = badge_for_io_status("active")
        self.assertEqual(b.color_key, "success")

    def test_unmapped_yields_warning(self):
        b = badge_for_io_status("unmapped")
        self.assertEqual(b.color_key, "warning")

    def test_unsupported_yields_error(self):
        b = badge_for_io_status("unsupported")
        self.assertEqual(b.color_key, "error")

    def test_conflict_yields_error(self):
        b = badge_for_io_status("conflict")
        self.assertEqual(b.color_key, "error")

    def test_unknown_string_does_not_raise(self):
        try:
            badge_for_io_status("future_status")
        except Exception as exc:
            self.fail(f"badge_for_io_status raised: {exc}")

    def test_unknown_string_returns_badge_type(self):
        b = badge_for_io_status("future_status")
        self.assertIsInstance(b, HBIOBadge)

    def test_unknown_string_yields_neutral(self):
        b = badge_for_io_status("future_status")
        self.assertEqual(b.color_key, "neutral")

    def test_unknown_status_label_is_status_string(self):
        b = badge_for_io_status("weird_status")
        self.assertEqual(b.label, "weird_status")


# ===========================================================================
# detect_io_conflicts()
# ===========================================================================

class TestDetectIOConflicts(unittest.TestCase):

    def test_empty_list_returns_empty(self):
        self.assertEqual(detect_io_conflicts([]), [])

    def test_no_conflict_all_active_distinct(self):
        mappings = [
            _io("imu.pitch", "inbound"),
            _io("imu.roll", "inbound"),
            _io("base_action", "outbound"),
        ]
        self.assertEqual(detect_io_conflicts(mappings), [])

    def test_dual_direction_yields_error_conflict(self):
        mappings = [_io("imu.pitch", "inbound"), _io("imu.pitch", "outbound")]
        hints = detect_io_conflicts(mappings)
        self.assertTrue(any(h.signal == "imu.pitch" for h in hints))
        dup = next(h for h in hints if h.signal == "imu.pitch")
        self.assertEqual(dup.severity, "error")

    def test_multiple_dual_direction_conflicts(self):
        mappings = [
            _io("a", "inbound"), _io("a", "outbound"),
            _io("b", "inbound"), _io("b", "outbound"),
        ]
        signals = {h.signal for h in detect_io_conflicts(mappings)}
        self.assertIn("a", signals)
        self.assertIn("b", signals)

    def test_unsupported_status_yields_error_conflict(self):
        mappings = [_io("bad_sig", "inbound", "unsupported")]
        hints = detect_io_conflicts(mappings)
        self.assertTrue(any(h.signal == "bad_sig" and h.severity == "error" for h in hints))

    def test_unmapped_status_yields_warning_conflict(self):
        mappings = [_io("no_target", "inbound", "unmapped")]
        hints = detect_io_conflicts(mappings)
        self.assertTrue(any(h.signal == "no_target" and h.severity == "warning" for h in hints))

    def test_active_status_no_conflict(self):
        self.assertEqual(detect_io_conflicts([_io("ok_sig", "inbound", "active")]), [])

    def test_mixed_duplicate_and_unsupported(self):
        mappings = [
            _io("dup_sig", "inbound"),
            _io("dup_sig", "outbound"),
            _io("bad_sig", "inbound", "unsupported"),
        ]
        sigs = {h.signal for h in detect_io_conflicts(mappings)}
        self.assertIn("dup_sig", sigs)
        self.assertIn("bad_sig", sigs)

    def test_conflict_hint_reason_non_empty(self):
        mappings = [_io("sig", "inbound"), _io("sig", "outbound")]
        for h in detect_io_conflicts(mappings):
            if h.signal == "sig":
                self.assertGreater(len(h.reason), 0)
                break

    def test_return_type_is_list(self):
        self.assertIsInstance(detect_io_conflicts([]), list)

    def test_does_not_raise_on_arbitrary_status(self):
        try:
            detect_io_conflicts([_io("x", "inbound", "future_status")])
        except Exception as exc:
            self.fail(f"detect_io_conflicts raised: {exc}")

    def test_no_false_positive_for_two_inbound_same_signal(self):
        """Two inbound bindings of the same signal is NOT a conflict in current rules."""
        # Current rules only flag inbound+outbound pair; duplicate inbound is allowed.
        mappings = [_io("same", "inbound"), _io("same", "inbound")]
        hints = detect_io_conflicts(mappings)
        dual = [h for h in hints if "both inbound and outbound" in h.reason]
        self.assertEqual(len(dual), 0)


# ===========================================================================
# compute_event_summary()
# ===========================================================================

class TestComputeEventSummary(unittest.TestCase):

    def test_empty_list(self):
        s = compute_event_summary([])
        self.assertEqual(s.total, 0)
        self.assertEqual(s.active, 0)
        self.assertEqual(s.errors, 0)
        self.assertEqual(s.warnings, 0)

    def test_single_active(self):
        s = compute_event_summary([_ev("active")])
        self.assertEqual(s.total, 1)
        self.assertEqual(s.active, 1)
        self.assertEqual(s.errors, 0)

    def test_single_error(self):
        s = compute_event_summary([_ev("error")])
        self.assertEqual(s.errors, 1)
        self.assertEqual(s.active, 0)

    def test_single_warning(self):
        s = compute_event_summary([_ev("warning")])
        self.assertEqual(s.warnings, 1)

    def test_mixed_states(self):
        entries = [
            _ev("active"), _ev("active"),
            _ev("error"), _ev("warning"), _ev("inactive"),
        ]
        s = compute_event_summary(entries)
        self.assertEqual(s.total, 5)
        self.assertEqual(s.active, 2)
        self.assertEqual(s.errors, 1)
        self.assertEqual(s.warnings, 1)

    def test_unknown_state_not_counted_in_active_or_errors(self):
        s = compute_event_summary([_ev("pending"), _ev("inactive")])
        self.assertEqual(s.active, 0)
        self.assertEqual(s.errors, 0)
        self.assertEqual(s.total, 2)

    def test_returns_event_summary_type(self):
        self.assertIsInstance(compute_event_summary([]), HBEventSummary)


# ===========================================================================
# compute_io_summary()
# ===========================================================================

class TestComputeIOSummary(unittest.TestCase):

    def test_empty_list(self):
        s = compute_io_summary([], [])
        self.assertEqual(s.total, 0)
        self.assertEqual(s.inbound, 0)
        self.assertEqual(s.outbound, 0)
        self.assertEqual(s.conflicts, 0)

    def test_inbound_outbound_split(self):
        mappings = [_io("a", "inbound"), _io("b", "inbound"), _io("c", "outbound")]
        s = compute_io_summary(mappings, [])
        self.assertEqual(s.total, 3)
        self.assertEqual(s.inbound, 2)
        self.assertEqual(s.outbound, 1)
        self.assertEqual(s.conflicts, 0)

    def test_conflicts_from_provided_hint_list(self):
        hints = [
            HBConflictHint(signal="x", reason="r", severity="error"),
            HBConflictHint(signal="y", reason="r", severity="warning"),
        ]
        s = compute_io_summary([], hints)
        self.assertEqual(s.conflicts, 2)

    def test_total_includes_all_directions(self):
        mappings = [_io("a", "inbound"), _io("b", "outbound")]
        s = compute_io_summary(mappings, [])
        self.assertEqual(s.total, 2)

    def test_returns_io_summary_type(self):
        self.assertIsInstance(compute_io_summary([], []), HBIOSummary)

    def test_conflicts_zero_with_empty_hints(self):
        s = compute_io_summary([_io("x")], [])
        self.assertEqual(s.conflicts, 0)


if __name__ == "__main__":
    unittest.main()
