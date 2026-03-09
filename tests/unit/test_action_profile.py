#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for system.behavior.action_profile — Phase 1 data model.

Coverage
--------
- ActionSegment: create, to_dict, from_dict, field defaults
- MotorSegment: create, to_dict, from_dict, field defaults
- ActionMotorOverlay: to_dict, from_dict, get_segments_for_track
- BehaviorTimeline: to_dict, from_dict, empty/non-empty,
  total_duration, get_motor_segments_for_track, reorder_action,
  set_overlay_expanded, from_dict with invalid input
- MotorTrackDef: clamp_safe, clamp_sim, is_in_safe_range, is_in_sim_range
- ActionPhaseTemplate.decompose(): walk template, track filtering
- validate_motor_segment(): safe range, sim range, None track_def, non-numeric
- validate_timeline(): aggregation across overlays
- build_timeline_from_modules(): empty list, Unitree auto-decompose,
  param preservation, non-Unitree robot_type, SequenceModule-like objects
- UNITREE_ACTION_PROFILES / UNITREE_MOTOR_TRACKS: catalog completeness
- Roundtrip: to_dict → from_dict identity

Isolation: pure Python, no Qt, no SDK.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.action_profile import (  # noqa: E402
    BEHAVIOR_TIMELINE_VERSION,
    ActionMotorOverlay,
    ActionPhaseTemplate,
    ActionSegment,
    BehaviorTimeline,
    MotorSegment,
    MotorSegmentDiagnostic,
    MotorTrackDef,
    UNITREE_ACTION_PROFILES,
    UNITREE_MOTOR_TRACK_MAP,
    UNITREE_MOTOR_TRACKS,
    build_timeline_from_modules,
    validate_motor_segment,
    validate_timeline,
)


# ---------------------------------------------------------------------------
# ActionSegment
# ---------------------------------------------------------------------------

class TestActionSegment(unittest.TestCase):
    def test_create_generates_uuid(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        self.assertTrue(len(seg.action_id) > 0)
        self.assertEqual(seg.name, "walk")
        self.assertAlmostEqual(seg.start_time, 0.0)
        self.assertAlmostEqual(seg.duration, 4.0)

    def test_create_clamps_duration_min(self):
        seg = ActionSegment.create("noop", 0.0, 0.0)
        self.assertGreaterEqual(seg.duration, 0.1)

    def test_create_with_params(self):
        seg = ActionSegment.create("walk", 0.0, 4.0, params={"speed": 0.3})
        self.assertEqual(seg.params["speed"], 0.3)

    def test_to_dict_roundtrip(self):
        seg = ActionSegment.create("stand", 1.0, 2.0, params={"gain": 1.2}, kind="movement")
        d = seg.to_dict()
        seg2 = ActionSegment.from_dict(d)
        self.assertEqual(seg.action_id, seg2.action_id)
        self.assertEqual(seg.name, seg2.name)
        self.assertAlmostEqual(seg.start_time, seg2.start_time)
        self.assertAlmostEqual(seg.duration, seg2.duration)
        self.assertEqual(seg.params, seg2.params)
        self.assertEqual(seg.kind, seg2.kind)

    def test_from_dict_defaults(self):
        seg = ActionSegment.from_dict({})
        self.assertEqual(seg.name, "noop")
        self.assertAlmostEqual(seg.start_time, 0.0)
        self.assertAlmostEqual(seg.duration, 1.0)
        self.assertEqual(seg.kind, "movement")
        self.assertFalse(seg.locked)

    def test_from_dict_generates_uuid_when_missing(self):
        seg = ActionSegment.from_dict({"name": "sit"})
        self.assertTrue(len(seg.action_id) > 0)

    def test_kind_default(self):
        seg = ActionSegment.create("walk", 0.0, 1.0)
        self.assertEqual(seg.kind, "movement")


# ---------------------------------------------------------------------------
# MotorSegment
# ---------------------------------------------------------------------------

class TestMotorSegment(unittest.TestCase):
    def test_create_generates_uuid(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0)
        self.assertTrue(len(seg.motor_id) > 0)
        self.assertEqual(seg.track_name, "leg_group_left")

    def test_create_with_params_and_parent(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 2.0,
                                  params={"amplitude": 0.3}, parent_action_id="abc")
        self.assertEqual(seg.params["amplitude"], 0.3)
        self.assertEqual(seg.parent_action_id, "abc")

    def test_to_dict_roundtrip(self):
        seg = MotorSegment.create("body_posture", 0.5, 1.5, params={"gain": 1.1})
        d = seg.to_dict()
        seg2 = MotorSegment.from_dict(d)
        self.assertEqual(seg.motor_id, seg2.motor_id)
        self.assertEqual(seg.track_name, seg2.track_name)
        self.assertAlmostEqual(seg.start_time, seg2.start_time)
        self.assertAlmostEqual(seg.duration, seg2.duration)
        self.assertEqual(seg.params, seg2.params)

    def test_from_dict_defaults(self):
        seg = MotorSegment.from_dict({})
        self.assertEqual(seg.track_name, "unknown")
        self.assertAlmostEqual(seg.start_time, 0.0)
        self.assertAlmostEqual(seg.duration, 1.0)
        self.assertIsNone(seg.parent_action_id)

    def test_duration_minimum_clamp(self):
        seg = MotorSegment.create("leg_group_left", 0.0, -5.0)
        self.assertGreaterEqual(seg.duration, 0.1)


# ---------------------------------------------------------------------------
# ActionMotorOverlay
# ---------------------------------------------------------------------------

class TestActionMotorOverlay(unittest.TestCase):
    def _make_overlay(self):
        segs = [
            MotorSegment.create("leg_group_left", 0.0, 1.0),
            MotorSegment.create("leg_group_right", 0.0, 1.0),
        ]
        return ActionMotorOverlay(action_id="act1", motor_segments=segs, expanded=True)

    def test_get_segments_for_track_filters(self):
        overlay = self._make_overlay()
        left = overlay.get_segments_for_track("leg_group_left")
        self.assertEqual(len(left), 1)
        self.assertEqual(left[0].track_name, "leg_group_left")

    def test_get_segments_for_missing_track(self):
        overlay = self._make_overlay()
        result = overlay.get_segments_for_track("head_pitch")
        self.assertEqual(result, [])

    def test_to_dict_roundtrip(self):
        overlay = self._make_overlay()
        d = overlay.to_dict()
        overlay2 = ActionMotorOverlay.from_dict(d)
        self.assertEqual(overlay.action_id, overlay2.action_id)
        self.assertEqual(len(overlay.motor_segments), len(overlay2.motor_segments))
        self.assertTrue(overlay2.expanded)

    def test_from_dict_empty(self):
        overlay = ActionMotorOverlay.from_dict({"action_id": "x"})
        self.assertEqual(overlay.action_id, "x")
        self.assertEqual(overlay.motor_segments, [])
        self.assertTrue(overlay.expanded)


# ---------------------------------------------------------------------------
# BehaviorTimeline
# ---------------------------------------------------------------------------

class TestBehaviorTimeline(unittest.TestCase):
    def _make_timeline(self):
        segs = [
            ActionSegment.create("stand", 0.0, 2.0),
            ActionSegment.create("walk", 2.0, 4.0),
        ]
        overlay0 = ActionMotorOverlay(
            action_id=segs[0].action_id,
            motor_segments=[MotorSegment.create("leg_group_left", 0.0, 2.0)],
        )
        overlay1 = ActionMotorOverlay(
            action_id=segs[1].action_id,
            motor_segments=[MotorSegment.create("leg_group_left", 2.0, 4.0)],
        )
        return BehaviorTimeline(
            action_segments=segs,
            motor_overlays=[overlay0, overlay1],
            active_motor_tracks=["leg_group_left"],
        )

    def test_is_empty_false(self):
        tl = self._make_timeline()
        self.assertFalse(tl.is_empty())

    def test_is_empty_true(self):
        self.assertTrue(BehaviorTimeline().is_empty())

    def test_total_duration(self):
        tl = self._make_timeline()
        self.assertAlmostEqual(tl.total_duration(), 6.0)

    def test_total_duration_empty(self):
        self.assertAlmostEqual(BehaviorTimeline().total_duration(), 0.0)

    def test_get_overlay_for_action(self):
        tl = self._make_timeline()
        seg = tl.action_segments[0]
        overlay = tl.get_overlay_for_action(seg.action_id)
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.action_id, seg.action_id)

    def test_get_overlay_for_missing_action(self):
        tl = self._make_timeline()
        self.assertIsNone(tl.get_overlay_for_action("nonexistent"))

    def test_get_motor_segments_for_track(self):
        tl = self._make_timeline()
        segs = tl.get_motor_segments_for_track("leg_group_left")
        self.assertEqual(len(segs), 2)

    def test_get_motor_segments_for_missing_track(self):
        tl = self._make_timeline()
        self.assertEqual(tl.get_motor_segments_for_track("head_pitch"), [])

    def test_to_dict_roundtrip(self):
        tl = self._make_timeline()
        d = tl.to_dict()
        tl2 = BehaviorTimeline.from_dict(d)
        self.assertEqual(len(tl.action_segments), len(tl2.action_segments))
        self.assertEqual(len(tl.motor_overlays), len(tl2.motor_overlays))
        self.assertEqual(tl.active_motor_tracks, tl2.active_motor_tracks)
        self.assertEqual(tl.version, tl2.version)

    def test_from_dict_invalid_input_returns_empty(self):
        for bad in [None, 42, "string", []]:
            tl = BehaviorTimeline.from_dict(bad)
            self.assertTrue(tl.is_empty())

    def test_from_dict_tolerates_missing_keys(self):
        tl = BehaviorTimeline.from_dict({"robot_type": "go2"})
        self.assertEqual(tl.robot_type, "go2")
        self.assertTrue(tl.is_empty())

    def test_set_overlay_expanded(self):
        tl = self._make_timeline()
        action_id = tl.action_segments[0].action_id
        tl.set_overlay_expanded(action_id, False)
        overlay = tl.get_overlay_for_action(action_id)
        self.assertFalse(overlay.expanded)

    def test_set_overlay_expanded_missing_noop(self):
        tl = self._make_timeline()
        tl.set_overlay_expanded("nonexistent", False)  # should not raise

    def test_reorder_action(self):
        tl = self._make_timeline()
        name0 = tl.action_segments[0].name
        name1 = tl.action_segments[1].name
        tl.reorder_action(0, 1)
        self.assertEqual(tl.action_segments[0].name, name1)
        self.assertEqual(tl.action_segments[1].name, name0)

    def test_reorder_action_recalculates_start_times(self):
        tl = self._make_timeline()
        tl.reorder_action(0, 1)
        self.assertAlmostEqual(tl.action_segments[0].start_time, 0.0)
        # Second segment starts after first
        self.assertAlmostEqual(
            tl.action_segments[1].start_time,
            tl.action_segments[0].duration,
        )

    def test_reorder_action_out_of_bounds_noop(self):
        tl = self._make_timeline()
        names_before = [s.name for s in tl.action_segments]
        tl.reorder_action(0, 99)
        names_after = [s.name for s in tl.action_segments]
        self.assertEqual(names_before, names_after)

    def test_version_field(self):
        tl = BehaviorTimeline()
        self.assertEqual(tl.version, BEHAVIOR_TIMELINE_VERSION)


# ---------------------------------------------------------------------------
# MotorTrackDef
# ---------------------------------------------------------------------------

class TestMotorTrackDef(unittest.TestCase):
    def setUp(self):
        self.tdef = MotorTrackDef(
            track_name="test_track",
            label="Test",
            safe_min=0.0,
            safe_max=0.5,
            sim_min=-1.0,
            sim_max=2.0,
        )

    def test_clamp_safe_within(self):
        self.assertAlmostEqual(self.tdef.clamp_safe(0.3), 0.3)

    def test_clamp_safe_below(self):
        self.assertAlmostEqual(self.tdef.clamp_safe(-1.0), 0.0)

    def test_clamp_safe_above(self):
        self.assertAlmostEqual(self.tdef.clamp_safe(1.0), 0.5)

    def test_clamp_sim_below(self):
        self.assertAlmostEqual(self.tdef.clamp_sim(-5.0), -1.0)

    def test_clamp_sim_above(self):
        self.assertAlmostEqual(self.tdef.clamp_sim(5.0), 2.0)

    def test_is_in_safe_range_boundary(self):
        self.assertTrue(self.tdef.is_in_safe_range(0.0))
        self.assertTrue(self.tdef.is_in_safe_range(0.5))
        self.assertFalse(self.tdef.is_in_safe_range(0.51))
        self.assertFalse(self.tdef.is_in_safe_range(-0.01))

    def test_is_in_sim_range(self):
        self.assertTrue(self.tdef.is_in_sim_range(1.9))
        self.assertFalse(self.tdef.is_in_sim_range(2.01))
        self.assertFalse(self.tdef.is_in_sim_range(-1.01))


# ---------------------------------------------------------------------------
# ActionPhaseTemplate.decompose
# ---------------------------------------------------------------------------

class TestActionPhaseTemplateDecompose(unittest.TestCase):
    def setUp(self):
        self.walk_tpl = UNITREE_ACTION_PROFILES["walk"]
        self.stand_tpl = UNITREE_ACTION_PROFILES["stand"]

    def test_decompose_walk_produces_segments(self):
        seg = ActionSegment.create("walk", 0.0, 10.0, params={"speed": 0.4})
        motor_segs = self.walk_tpl.decompose(seg)
        self.assertGreater(len(motor_segs), 0)

    def test_decompose_walk_covers_full_duration(self):
        seg = ActionSegment.create("walk", 0.0, 10.0)
        motor_segs = self.walk_tpl.decompose(seg, available_tracks=["leg_fl"])
        total = sum(s.duration for s in motor_segs)
        self.assertAlmostEqual(total, 10.0, places=5)

    def test_decompose_walk_aligned_to_parent_start(self):
        seg = ActionSegment.create("walk", 5.0, 4.0)
        segs = self.walk_tpl.decompose(seg, available_tracks=["leg_fl"])
        self.assertAlmostEqual(segs[0].start_time, 5.0)

    def test_decompose_sets_parent_action_id(self):
        seg = ActionSegment.create("stand", 0.0, 2.0)
        segs = self.stand_tpl.decompose(seg, available_tracks=["leg_fl"])
        for s in segs:
            self.assertEqual(s.parent_action_id, seg.action_id)

    def test_decompose_with_track_filter(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        segs = self.walk_tpl.decompose(seg, available_tracks=["leg_fl"])
        tracks = {s.track_name for s in segs}
        self.assertEqual(tracks, {"leg_fl"})

    def test_decompose_empty_phases_returns_empty(self):
        tpl = ActionPhaseTemplate(action_name="noop", label="Noop", phases=[], motor_tracks=["leg_group_left"])
        seg = ActionSegment.create("noop", 0.0, 1.0)
        self.assertEqual(tpl.decompose(seg), [])

    def test_decompose_no_matching_tracks_returns_empty(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        segs = self.walk_tpl.decompose(seg, available_tracks=["head_pitch"])
        self.assertEqual(segs, [])

    def test_decompose_user_override_applied(self):
        seg = ActionSegment.create("walk", 0.0, 4.0, params={"amplitude": 0.45})
        segs = self.walk_tpl.decompose(seg, available_tracks=["leg_fl"])
        # All segments with default amplitude override should have 0.45
        for s in segs:
            if "amplitude" in s.params:
                self.assertAlmostEqual(s.params["amplitude"], 0.45)


# ---------------------------------------------------------------------------
# validate_motor_segment
# ---------------------------------------------------------------------------

class TestValidateMotorSegment(unittest.TestCase):
    def setUp(self):
        self.tdef = UNITREE_MOTOR_TRACK_MAP["leg_group_left"]  # safe 0–0.5

    def test_within_safe_range_no_diag(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 0.3})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(diags, [])

    def test_exceeds_safe_range_gives_error(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 0.9})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].level, "error")
        self.assertIn("safe hardware", diags[0].message)

    def test_within_sim_range_no_diag(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 0.8})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=True)
        self.assertEqual(diags, [])

    def test_exceeds_sim_range_gives_warning(self):
        # Fix 3: sim violations are now "warning" (non-blocking), not "error"
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 1.5})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=True)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].level, "warning")

    def test_none_track_def_returns_empty(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 999.0})
        diags = validate_motor_segment(seg, None)
        self.assertEqual(diags, [])

    def test_missing_param_key_returns_empty(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"gain": 0.9})
        # param_key is "amplitude" for this track; "gain" is not checked
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(diags, [])

    def test_non_numeric_param_returns_empty(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": "fast"})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(diags, [])

    def test_diagnostic_to_dict_schema(self):
        seg = MotorSegment.create("leg_group_left", 0.0, 1.0, params={"amplitude": 0.9})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        d = diags[0].to_dict()
        for key in ("motor_id", "track_name", "param_key", "value", "level", "message"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
# validate_timeline
# ---------------------------------------------------------------------------

class TestValidateTimeline(unittest.TestCase):
    def test_empty_timeline_no_diags(self):
        tl = BehaviorTimeline()
        self.assertEqual(validate_timeline(tl), [])

    def test_valid_overlay_no_diags(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        mseg = MotorSegment.create("leg_group_left", 0.0, 4.0, params={"amplitude": 0.3},
                                   parent_action_id=seg.action_id)
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=[mseg])
        tl = BehaviorTimeline(action_segments=[seg], motor_overlays=[overlay])
        diags = validate_timeline(tl, is_simulation=False)
        self.assertEqual(diags, [])

    def test_invalid_overlay_gives_diag(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        mseg = MotorSegment.create("leg_group_left", 0.0, 4.0, params={"amplitude": 0.99},
                                   parent_action_id=seg.action_id)
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=[mseg])
        tl = BehaviorTimeline(action_segments=[seg], motor_overlays=[overlay])
        diags = validate_timeline(tl, is_simulation=False)
        self.assertEqual(len(diags), 1)

    def test_multiple_invalid_segments_all_reported(self):
        seg = ActionSegment.create("walk", 0.0, 4.0)
        bad_segs = [
            MotorSegment.create("leg_group_left", 0.0, 2.0, params={"amplitude": 0.9}),
            MotorSegment.create("leg_group_right", 2.0, 2.0, params={"amplitude": 0.8}),
        ]
        overlay = ActionMotorOverlay(action_id=seg.action_id, motor_segments=bad_segs)
        tl = BehaviorTimeline(action_segments=[seg], motor_overlays=[overlay])
        diags = validate_timeline(tl, is_simulation=False)
        self.assertGreaterEqual(len(diags), 2)


# ---------------------------------------------------------------------------
# build_timeline_from_modules
# ---------------------------------------------------------------------------

class TestBuildTimelineFromModules(unittest.TestCase):
    def _make_module(self, name, duration=1.0, kind="movement", args=""):
        """Return a simple dict acting as a SequenceModule."""
        return {"name": name, "duration": duration, "kind": kind, "args": args}

    def test_empty_modules_returns_empty_timeline(self):
        tl = build_timeline_from_modules([])
        self.assertTrue(tl.is_empty())

    def test_builds_action_segments_in_order(self):
        modules = [
            self._make_module("stand", 2.0),
            self._make_module("walk", 4.0),
            self._make_module("sit", 1.5),
        ]
        tl = build_timeline_from_modules(modules)
        self.assertEqual(len(tl.action_segments), 3)
        self.assertEqual(tl.action_segments[0].name, "stand")
        self.assertEqual(tl.action_segments[1].name, "walk")
        self.assertEqual(tl.action_segments[2].name, "sit")

    def test_start_times_are_sequential(self):
        modules = [self._make_module("stand", 2.0), self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules)
        self.assertAlmostEqual(tl.action_segments[0].start_time, 0.0)
        self.assertAlmostEqual(tl.action_segments[1].start_time, 2.0)

    def test_total_duration_matches_sum(self):
        modules = [self._make_module("stand", 2.0), self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules)
        self.assertAlmostEqual(tl.total_duration(), 6.0)

    def test_unitree_auto_decompose_creates_motor_overlays(self):
        modules = [self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules, robot_type="go2")
        self.assertEqual(len(tl.motor_overlays), 1)
        self.assertGreater(len(tl.motor_overlays[0].motor_segments), 0)

    def test_unitree_auto_decompose_populates_active_tracks(self):
        modules = [self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules, robot_type="go2")
        self.assertGreater(len(tl.active_motor_tracks), 0)
        for track in tl.active_motor_tracks:
            self.assertIn(track, UNITREE_MOTOR_TRACK_MAP)

    def test_non_unitree_robot_type_no_decompose(self):
        modules = [self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules, robot_type="spot")
        self.assertEqual(tl.motor_overlays, [])
        self.assertEqual(tl.active_motor_tracks, [])

    def test_auto_decompose_false_no_overlays(self):
        modules = [self._make_module("walk", 4.0)]
        tl = build_timeline_from_modules(modules, robot_type="go2", auto_decompose=False)
        self.assertEqual(tl.motor_overlays, [])

    def test_args_parsed_into_params(self):
        modules = [self._make_module("walk", 4.0, args="speed=0.3, duration=4.0")]
        tl = build_timeline_from_modules(modules)
        self.assertAlmostEqual(tl.action_segments[0].params.get("speed"), 0.3)

    def test_unknown_action_gets_empty_overlay(self):
        modules = [self._make_module("unknown_action", 1.0)]
        tl = build_timeline_from_modules(modules, robot_type="go2")
        self.assertEqual(len(tl.motor_overlays), 1)
        # No motor segments for unknown action
        self.assertEqual(tl.motor_overlays[0].motor_segments, [])

    def test_sequencemodule_like_object_accepted(self):
        """Objects with __dict__ (e.g. SequenceModule instances) are accepted."""
        class FakeModule:
            def __init__(self, name, duration, kind="movement", args=""):
                self.name = name
                self.duration = duration
                self.kind = kind
                self.args = args

        modules = [FakeModule("stand", 2.0), FakeModule("walk", 4.0)]
        tl = build_timeline_from_modules(modules)
        self.assertEqual(len(tl.action_segments), 2)

    def test_invalid_duration_coerced(self):
        modules = [{"name": "stand", "duration": "bad", "kind": "movement", "args": ""}]
        tl = build_timeline_from_modules(modules)
        self.assertGreaterEqual(tl.action_segments[0].duration, 0.1)

    def test_robot_type_stored(self):
        tl = build_timeline_from_modules([], robot_type="go2")
        self.assertEqual(tl.robot_type, "go2")


# ---------------------------------------------------------------------------
# Unitree catalog completeness
# ---------------------------------------------------------------------------

class TestUnitreeCatalog(unittest.TestCase):
    EXPECTED_TRACKS = ["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture", "head_pitch"]
    EXPECTED_ACTIONS = ["walk", "stand", "sit", "wait"]

    def test_motor_tracks_present(self):
        track_names = [t.track_name for t in UNITREE_MOTOR_TRACKS]
        for name in self.EXPECTED_TRACKS:
            self.assertIn(name, track_names)

    def test_motor_track_map_keys(self):
        for name in self.EXPECTED_TRACKS:
            self.assertIn(name, UNITREE_MOTOR_TRACK_MAP)

    def test_action_profiles_present(self):
        for name in self.EXPECTED_ACTIONS:
            self.assertIn(name, UNITREE_ACTION_PROFILES)

    def test_all_tracks_have_safe_range_above_zero(self):
        for t in UNITREE_MOTOR_TRACKS:
            # sim range should be >= safe range
            self.assertLessEqual(t.safe_max, t.sim_max)

    def test_walk_profile_has_three_phases(self):
        self.assertEqual(len(UNITREE_ACTION_PROFILES["walk"].phases), 3)

    def test_stand_profile_has_two_phases(self):
        self.assertEqual(len(UNITREE_ACTION_PROFILES["stand"].phases), 2)

    def test_track_def_fields(self):
        for t in UNITREE_MOTOR_TRACKS:
            self.assertIsInstance(t.track_name, str)
            self.assertIsInstance(t.label, str)
            self.assertIsInstance(t.color, str)
            self.assertTrue(t.color.startswith("#"))


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Fix 3 — validate_motor_segment level: sim → warning, hardware → error
# ===========================================================================

class TestValidateMotorSegmentLevels(unittest.TestCase):
    """Verify the level field on MotorSegmentDiagnostic is correct per mode."""

    def setUp(self):
        self.tdef = UNITREE_MOTOR_TRACK_MAP["leg_fl"]
        # amplitude=0.99 exceeds safe_max=0.5; check within/above sim range
        self.over_safe = 0.99  # always > safe_max

    def test_hardware_violation_level_is_error(self):
        seg = MotorSegment.create("leg_fl", 0.0, 1.0,
                                  params={"amplitude": self.over_safe})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].level, "error")

    def test_sim_violation_level_is_warning(self):
        over_sim = self.tdef.sim_max + 1.0
        seg = MotorSegment.create("leg_fl", 0.0, 1.0,
                                  params={"amplitude": over_sim})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=True)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].level, "warning")

    def test_within_sim_but_over_safe_hardware_is_error(self):
        # amplitude=0.99: over safe(0.5) but within sim(~1.0)
        # In hardware mode → error
        seg = MotorSegment.create("leg_fl", 0.0, 1.0,
                                  params={"amplitude": self.over_safe})
        hw_diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(hw_diags[0].level, "error")

    def test_within_sim_but_over_safe_sim_mode_is_no_diag(self):
        # amplitude=0.99: over safe(0.5) but within sim(~1.0)
        # In sim mode → no diag because it's within sim range
        seg = MotorSegment.create("leg_fl", 0.0, 1.0,
                                  params={"amplitude": self.over_safe})
        sim_diags = validate_motor_segment(seg, self.tdef, is_simulation=True)
        self.assertEqual(sim_diags, [])

    def test_no_diag_when_within_safe_range_hardware(self):
        seg = MotorSegment.create("leg_fl", 0.0, 1.0,
                                  params={"amplitude": 0.3})
        diags = validate_motor_segment(seg, self.tdef, is_simulation=False)
        self.assertEqual(diags, [])


class TestActionPackageExpansion(unittest.TestCase):
    def _make_module(self, name, duration=1.0, kind="movement", args=""):
        return {"name": name, "duration": duration, "kind": kind, "args": args}

    def test_lift_left_leg_expands_on_action_track(self):
        tl = build_timeline_from_modules(
            [self._make_module("lift_left_leg", 4.0)],
            robot_type="go2",
        )
        names = [s.name for s in tl.action_segments]
        self.assertEqual(
            names,
            [
                "lift_left_leg_prepare",
                "lift_left_leg_raise",
                "lift_left_leg_hold",
                "lift_left_leg_recover",
            ],
        )

    def test_lift_left_leg_expansion_generates_leg_specific_tracks(self):
        tl = build_timeline_from_modules(
            [self._make_module("lift_left_leg", 4.0)],
            robot_type="go2",
        )
        for expected in ("leg_fl", "leg_fr", "leg_rl", "leg_rr"):
            self.assertIn(expected, tl.active_motor_tracks)


# ===========================================================================
# Fix 5 — build_timeline_from_modules auto_decompose=False / robot_type guard
# ===========================================================================

class TestBuildTimelineAutoDecomposeFalse(unittest.TestCase):
    def _mod(self, name="walk", duration=4.0):
        return {"name": name, "duration": duration, "kind": "movement", "args": ""}

    def test_auto_decompose_false_skips_unitree_overlays(self):
        tl = build_timeline_from_modules(
            [self._mod()], robot_type="go2", auto_decompose=False
        )
        self.assertEqual(tl.motor_overlays, [])
        self.assertEqual(tl.active_motor_tracks, [])

    def test_auto_decompose_false_unknown_robot_still_no_overlays(self):
        tl = build_timeline_from_modules(
            [self._mod()], robot_type="spot", auto_decompose=False
        )
        self.assertEqual(tl.motor_overlays, [])

    def test_auto_decompose_true_empty_robot_type_uses_unitree(self):
        # Empty robot_type with auto_decompose=True → Unitree decomposition
        tl = build_timeline_from_modules(
            [self._mod()], robot_type="", auto_decompose=True
        )
        self.assertGreater(len(tl.motor_overlays), 0)

    def test_auto_decompose_false_action_segments_still_created(self):
        tl = build_timeline_from_modules(
            [self._mod("stand", 2.0), self._mod("walk", 4.0)],
            robot_type="go2",
            auto_decompose=False,
        )
        self.assertEqual(len(tl.action_segments), 2)


if __name__ == "__main__":
    unittest.main()
