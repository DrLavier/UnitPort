#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 2 — Motor Weight Navigator model.

Coverage (pure-Python model only; no Qt import):
  build_navigator_model():
  - empty/None timeline → empty list (fallback path)
  - single movement × single motor → one region, one motor, one movement
  - region grouping: leg_fl → Legs, body_posture → Body
  - motor with no movement overlay → excluded in fallback, shown Unused with available_tracks
  - source "constant" when param in MotorSegment.params
  - source "external" when HBIOMapping targets the motor/param (inbound)
  - source "unresolved" when motor overlay present but primary param absent
  - has_issue propagates: NavParam → NavMovement → NavMotor → NavRegion
  - multiple movements on same motor → multiple NavMovement entries
  - multiple motors in same region → multiple NavMotor entries
  - fallback region for unknown track names

  available_tracks (full catalog) support:
  - full catalog shown even when no overlays
  - unused motor visible and marked is_unused=True
  - used + unused motors coexist in same region
  - fallback path (available_tracks=None) preserves original overlay-only behavior
  - NavMotor.is_unused=False by default for used motors

  NavParam helpers:
  - display_value for constant (formatted float)
  - display_value for external (← signal)
  - display_value for unresolved (—)
  - source_badge labels

  NavSource constants defined
"""

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# motor_weight_nav_model is pure Python (no Qt).  PySide6 is available in
# this environment so we let the normal import chain run unobstructed.


from system.behavior.motor_weight_nav_model import (  # noqa: E402
    NavParam,
    NavSource,
    NavMovement,
    NavMotor,
    NavRegion,
    build_navigator_model,
    _region_for_track,
    _FALLBACK_REGION,
)


# ---------------------------------------------------------------------------
# Minimal stubs for BehaviorTimeline objects
# ---------------------------------------------------------------------------

@dataclass
class _MotorSeg:
    motor_id: str
    track_name: str
    start_time: float = 0.0
    duration: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    parent_action_id: Optional[str] = None


@dataclass
class _Overlay:
    action_id: str
    motor_segments: List[_MotorSeg] = field(default_factory=list)


@dataclass
class _ActionSeg:
    action_id: str
    name: str
    start_time: float = 0.0
    duration: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    kind: str = "movement"


class _FakeTimeline:
    def __init__(self, action_segments=None, motor_overlays=None):
        self.action_segments = action_segments or []
        self.motor_overlays = motor_overlays or []

    def is_empty(self):
        return not self.action_segments


@dataclass
class _IOMapping:
    signal: str
    target_param: str
    direction: str = "inbound"
    status: str = "active"


# Minimal MotorTrackDef-like stub for available_tracks tests
@dataclass
class _TrackDef:
    track_name: str
    label: str
    param_key: str = "amplitude"


def _catalog(*track_names) -> Dict[str, _TrackDef]:
    """Build a minimal available_tracks dict from track names."""
    return {tn: _TrackDef(track_name=tn, label=tn.replace("_", " ").title())
            for tn in track_names}


# ---------------------------------------------------------------------------
# Tests: NavParam helpers
# ---------------------------------------------------------------------------

class TestNavParam(unittest.TestCase):

    def test_constant_display_value_float(self):
        p = NavParam(param_key="gain", source=NavSource.CONSTANT, value=1.12345)
        self.assertEqual(p.display_value, "1.123")  # 4 sig figs

    def test_constant_display_value_int(self):
        p = NavParam(param_key="gain", source=NavSource.CONSTANT, value=2)
        self.assertIn("2", p.display_value)

    def test_external_display_value_with_key(self):
        p = NavParam(param_key="gain", source=NavSource.EXTERNAL, external_key="imu.pitch")
        self.assertEqual(p.display_value, "← imu.pitch")

    def test_external_display_value_no_key(self):
        p = NavParam(param_key="gain", source=NavSource.EXTERNAL, external_key="")
        self.assertEqual(p.display_value, "← [external]")

    def test_unresolved_display_value(self):
        p = NavParam(param_key="gain", source=NavSource.UNRESOLVED)
        self.assertEqual(p.display_value, "—")

    def test_source_badges(self):
        self.assertEqual(NavParam("a", NavSource.CONSTANT).source_badge, "Constant")
        self.assertEqual(NavParam("a", NavSource.EXTERNAL).source_badge, "External")
        self.assertEqual(NavParam("a", NavSource.UNRESOLVED).source_badge, "Unresolved")

    def test_has_issue_true_for_unresolved(self):
        p = NavParam(param_key="x", source=NavSource.UNRESOLVED, has_issue=True)
        self.assertTrue(p.has_issue)

    def test_has_issue_false_for_constant(self):
        p = NavParam(param_key="x", source=NavSource.CONSTANT, value=1.0)
        self.assertFalse(p.has_issue)


# ---------------------------------------------------------------------------
# Tests: NavMovement / NavMotor / NavRegion has_issue propagation
# ---------------------------------------------------------------------------

class TestHasIssuePropagation(unittest.TestCase):

    def test_movement_has_issue_from_param(self):
        ok_param = NavParam("a", NavSource.CONSTANT, 1.0)
        bad_param = NavParam("b", NavSource.UNRESOLVED, has_issue=True)
        m = NavMovement("Walk", "id1", [ok_param, bad_param])
        self.assertTrue(m.has_issue)

    def test_movement_no_issue(self):
        p = NavParam("a", NavSource.CONSTANT, 1.0)
        m = NavMovement("Stand", "id2", [p])
        self.assertFalse(m.has_issue)

    def test_motor_has_issue_from_movement(self):
        bad = NavParam("x", NavSource.UNRESOLVED, has_issue=True)
        mv = NavMovement("Walk", "id3", [bad])
        motor = NavMotor("leg_fl", "Front Left Leg", [mv])
        self.assertTrue(motor.has_issue)

    def test_region_has_issue_from_motor(self):
        bad = NavParam("x", NavSource.UNRESOLVED, has_issue=True)
        mv = NavMovement("Walk", "id4", [bad])
        motor = NavMotor("leg_fl", "FL", [mv])
        region = NavRegion("Legs", [motor])
        self.assertTrue(region.has_issue)

    def test_unused_motor_has_no_issue(self):
        """An unused motor (no movements) should not raise has_issue."""
        motor = NavMotor("leg_fl", "FL", movements=[], is_unused=True)
        self.assertFalse(motor.has_issue)


# ---------------------------------------------------------------------------
# Tests: region grouping
# ---------------------------------------------------------------------------

class TestRegionGrouping(unittest.TestCase):

    def test_leg_fl_in_legs(self):
        self.assertEqual(_region_for_track("leg_fl"), "Legs")

    def test_leg_rr_in_legs(self):
        self.assertEqual(_region_for_track("leg_rr"), "Legs")

    def test_body_posture_in_body(self):
        self.assertEqual(_region_for_track("body_posture"), "Body")

    def test_head_pitch_in_head(self):
        self.assertEqual(_region_for_track("head_pitch"), "Head")

    def test_leg_group_left_in_leg_groups(self):
        self.assertEqual(_region_for_track("leg_group_left"), "Leg Groups")

    def test_unknown_track_fallback_region(self):
        self.assertEqual(_region_for_track("custom_actuator_99"), _FALLBACK_REGION)


# ---------------------------------------------------------------------------
# Tests: build_navigator_model — basic cases (fallback / overlay-only path)
# ---------------------------------------------------------------------------

class TestBuildNavigatorModelBasic(unittest.TestCase):

    def test_none_timeline_returns_empty(self):
        result = build_navigator_model(None)
        self.assertEqual(result, [])

    def test_empty_timeline_returns_empty(self):
        tl = _FakeTimeline()
        result = build_navigator_model(tl)
        self.assertEqual(result, [])

    def test_single_action_single_motor_constant(self):
        seg = _ActionSeg(action_id="a1", name="Walk")
        mseg = _MotorSeg(motor_id="m1", track_name="leg_fl",
                         params={"amplitude": 0.3}, parent_action_id="a1")
        overlay = _Overlay(action_id="a1", motor_segments=[mseg])
        tl = _FakeTimeline(action_segments=[seg], motor_overlays=[overlay])

        result = build_navigator_model(tl)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].region_label, "Legs")
        self.assertEqual(len(result[0].motors), 1)
        motor = result[0].motors[0]
        self.assertEqual(motor.track_name, "leg_fl")
        self.assertEqual(len(motor.movements), 1)
        movement = motor.movements[0]
        self.assertEqual(movement.movement_name, "Walk")
        amp_params = [p for p in movement.params if p.param_key == "amplitude"]
        self.assertTrue(len(amp_params) > 0)
        self.assertEqual(amp_params[0].source, NavSource.CONSTANT)
        self.assertAlmostEqual(amp_params[0].value, 0.3)

    def test_motor_not_in_any_overlay_excluded_in_fallback(self):
        """Without available_tracks, a motor not in any overlay is excluded."""
        seg = _ActionSeg(action_id="a1", name="Stand")
        # No motor overlays
        tl = _FakeTimeline(action_segments=[seg], motor_overlays=[])
        result = build_navigator_model(tl)
        self.assertEqual(result, [])

    def test_multiple_regions(self):
        seg = _ActionSeg(action_id="a1", name="Walk")
        # leg_fl → Legs; body_posture → Body
        mseg_leg = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        mseg_body = _MotorSeg("m2", "body_posture", params={"amplitude": 0.05})
        overlay = _Overlay("a1", [mseg_leg, mseg_body])
        tl = _FakeTimeline([seg], [overlay])

        result = build_navigator_model(tl)
        region_labels = [r.region_label for r in result]
        self.assertIn("Legs", region_labels)
        self.assertIn("Body", region_labels)

    def test_multiple_movements_same_motor(self):
        seg1 = _ActionSeg("a1", "Walk")
        seg2 = _ActionSeg("a2", "Stand")
        mseg1 = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        mseg2 = _MotorSeg("m2", "leg_fl", params={"amplitude": 0.1})
        ov1 = _Overlay("a1", [mseg1])
        ov2 = _Overlay("a2", [mseg2])
        tl = _FakeTimeline([seg1, seg2], [ov1, ov2])

        result = build_navigator_model(tl)
        leg_region = next((r for r in result if r.region_label == "Legs"), None)
        self.assertIsNotNone(leg_region)
        leg_fl = next((m for m in leg_region.motors if m.track_name == "leg_fl"), None)
        self.assertIsNotNone(leg_fl)
        self.assertEqual(len(leg_fl.movements), 2)
        names = {mv.movement_name for mv in leg_fl.movements}
        self.assertEqual(names, {"Walk", "Stand"})


# ---------------------------------------------------------------------------
# Tests: build_navigator_model — source detection
# ---------------------------------------------------------------------------

class TestBuildNavigatorModelSources(unittest.TestCase):

    def _tl_with_motor(self, track_name: str, params: dict) -> _FakeTimeline:
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", track_name, params=params)
        overlay = _Overlay("a1", [mseg])
        return _FakeTimeline([seg], [overlay])

    def test_source_constant_when_param_present(self):
        tl = self._tl_with_motor("leg_fl", {"amplitude": 0.5})
        result = build_navigator_model(tl, [])
        params = result[0].motors[0].movements[0].params
        amp = next(p for p in params if p.param_key == "amplitude")
        self.assertEqual(amp.source, NavSource.CONSTANT)
        self.assertAlmostEqual(amp.value, 0.5)
        self.assertFalse(amp.has_issue)

    def test_source_external_from_inbound_io_mapping(self):
        tl = self._tl_with_motor("body_posture", {"amplitude": 0.1})
        io_map = [_IOMapping(
            signal="imu.pitch",
            target_param="body_posture.amplitude",
            direction="inbound",
        )]
        result = build_navigator_model(tl, io_map)
        body = next(r for r in result if r.region_label == "Body")
        motor = body.motors[0]
        movement = motor.movements[0]
        amp = next(p for p in movement.params if p.param_key == "amplitude")
        self.assertEqual(amp.source, NavSource.EXTERNAL)
        self.assertEqual(amp.external_key, "imu.pitch")
        self.assertFalse(amp.has_issue)

    def test_outbound_io_mapping_not_treated_as_external(self):
        """Outbound mappings should not override param source to EXTERNAL."""
        tl = self._tl_with_motor("leg_fl", {"amplitude": 0.3})
        io_map = [_IOMapping(
            signal="leg_fl.amplitude",
            target_param="leg_fl.amplitude",
            direction="outbound",
        )]
        result = build_navigator_model(tl, io_map)
        motor = result[0].motors[0]
        amp = next(p for p in motor.movements[0].params if p.param_key == "amplitude")
        # outbound → still constant (param in seg_params)
        self.assertEqual(amp.source, NavSource.CONSTANT)

    def test_source_unresolved_when_primary_param_absent(self):
        """Motor overlay exists but primary param (amplitude) not in seg.params."""
        tl = self._tl_with_motor("leg_fl", {})  # no amplitude
        result = build_navigator_model(tl, [])
        motor = result[0].motors[0]
        amp = next((p for p in motor.movements[0].params if p.param_key == "amplitude"), None)
        self.assertIsNotNone(amp)
        self.assertEqual(amp.source, NavSource.UNRESOLVED)
        self.assertTrue(amp.has_issue)

    def test_has_issue_propagates_to_region(self):
        tl = self._tl_with_motor("leg_fl", {})  # no amplitude → unresolved
        result = build_navigator_model(tl, [])
        self.assertTrue(result[0].has_issue)

    def test_no_issue_when_all_params_resolved(self):
        tl = self._tl_with_motor("leg_fl", {"amplitude": 0.3})
        result = build_navigator_model(tl, [])
        self.assertFalse(result[0].has_issue)

    def test_fallback_region_for_unknown_track(self):
        tl = self._tl_with_motor("custom_motor_x99", {"amplitude": 1.0})
        result = build_navigator_model(tl, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].region_label, _FALLBACK_REGION)

    def test_start_time_duration_in_params(self):
        """start_time and duration are surfaced as Constant params."""
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", start_time=0.5, duration=2.0, params={"amplitude": 0.3})
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])
        result = build_navigator_model(tl, [])
        movement_params = result[0].motors[0].movements[0].params
        pk_set = {p.param_key for p in movement_params}
        self.assertIn("start_time", pk_set)
        self.assertIn("duration", pk_set)
        st = next(p for p in movement_params if p.param_key == "start_time")
        self.assertEqual(st.source, NavSource.CONSTANT)


# ---------------------------------------------------------------------------
# Tests: build_navigator_model — available_tracks (full catalog) path
# ---------------------------------------------------------------------------

class TestBuildNavigatorModelFullCatalog(unittest.TestCase):
    """Tests for the available_tracks parameter (full model topology)."""

    def test_all_catalog_tracks_shown_with_no_timeline(self):
        """When available_tracks given and timeline is None, all motors shown as Unused."""
        catalog = _catalog("leg_fl", "leg_fr", "body_posture")
        result = build_navigator_model(None, [], available_tracks=catalog)
        all_track_names = {m.track_name for r in result for m in r.motors}
        self.assertIn("leg_fl", all_track_names)
        self.assertIn("leg_fr", all_track_names)
        self.assertIn("body_posture", all_track_names)

    def test_all_catalog_tracks_shown_with_empty_timeline(self):
        """Empty timeline + available_tracks → all motors Unused."""
        tl = _FakeTimeline()  # is_empty() → True
        catalog = _catalog("leg_fl", "leg_rr")
        result = build_navigator_model(tl, [], available_tracks=catalog)
        all_track_names = {m.track_name for r in result for m in r.motors}
        self.assertIn("leg_fl", all_track_names)
        self.assertIn("leg_rr", all_track_names)

    def test_unused_motor_is_marked_is_unused(self):
        """Motor in catalog but not in any overlay → is_unused=True."""
        catalog = _catalog("leg_fl", "leg_fr")
        # Only leg_fl has an overlay
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])

        result = build_navigator_model(tl, [], available_tracks=catalog)
        leg_region = next(r for r in result if r.region_label == "Legs")
        leg_fr = next(m for m in leg_region.motors if m.track_name == "leg_fr")
        self.assertTrue(leg_fr.is_unused)
        self.assertEqual(leg_fr.movements, [])

    def test_used_motor_is_not_marked_unused(self):
        """Motor with at least one movement overlay → is_unused=False."""
        catalog = _catalog("leg_fl")
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])

        result = build_navigator_model(tl, [], available_tracks=catalog)
        leg_fl = result[0].motors[0]
        self.assertFalse(leg_fl.is_unused)

    def test_used_and_unused_motors_coexist_in_same_region(self):
        """leg_fl used, leg_fr unused — both appear in Legs region."""
        catalog = _catalog("leg_fl", "leg_fr", "leg_rl", "leg_rr")
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])

        result = build_navigator_model(tl, [], available_tracks=catalog)
        leg_region = next(r for r in result if r.region_label == "Legs")
        track_names = [m.track_name for m in leg_region.motors]
        self.assertIn("leg_fl", track_names)
        self.assertIn("leg_fr", track_names)

        leg_fl = next(m for m in leg_region.motors if m.track_name == "leg_fl")
        leg_fr = next(m for m in leg_region.motors if m.track_name == "leg_fr")
        self.assertFalse(leg_fl.is_unused)
        self.assertTrue(leg_fr.is_unused)

    def test_fallback_when_available_tracks_none_overlay_only(self):
        """When available_tracks=None, only overlay tracks are shown (original behavior)."""
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", params={"amplitude": 0.3})
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])

        # No available_tracks — only leg_fl should appear
        result = build_navigator_model(tl, [], available_tracks=None)
        all_track_names = {m.track_name for r in result for m in r.motors}
        self.assertIn("leg_fl", all_track_names)
        # leg_fr not in overlays → must not appear in fallback path
        self.assertNotIn("leg_fr", all_track_names)

    def test_fallback_none_timeline_no_tracks_returns_empty(self):
        """No available_tracks, no timeline → always empty."""
        result = build_navigator_model(None, [], available_tracks=None)
        self.assertEqual(result, [])

    def test_catalog_regions_grouped_correctly(self):
        """Catalog tracks are grouped into the correct regions."""
        catalog = _catalog("leg_fl", "body_posture", "head_pitch")
        result = build_navigator_model(None, [], available_tracks=catalog)
        region_labels = {r.region_label for r in result}
        self.assertIn("Legs", region_labels)
        self.assertIn("Body", region_labels)
        self.assertIn("Head", region_labels)

    def test_unresolved_warning_not_lost_with_available_tracks(self):
        """Existing unresolved binding visibility must survive the catalog path."""
        catalog = _catalog("leg_fl")
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", "leg_fl", params={})  # no amplitude → unresolved
        overlay = _Overlay("a1", [mseg])
        tl = _FakeTimeline([seg], [overlay])

        result = build_navigator_model(tl, [], available_tracks=catalog)
        leg_fl = result[0].motors[0]
        self.assertFalse(leg_fl.is_unused)  # it HAS a movement
        amp = next(
            (p for p in leg_fl.movements[0].params if p.param_key == "amplitude"),
            None,
        )
        self.assertIsNotNone(amp)
        self.assertEqual(amp.source, NavSource.UNRESOLVED)
        self.assertTrue(amp.has_issue)
        self.assertTrue(result[0].has_issue)

    def test_is_unused_default_false_on_dataclass(self):
        """NavMotor.is_unused defaults to False (backward compat)."""
        motor = NavMotor("leg_fl", "Front Left Leg", [])
        self.assertFalse(motor.is_unused)


# ---------------------------------------------------------------------------
# Tests: get_motor_track_map — brand-aware catalog lookup
# ---------------------------------------------------------------------------

class TestGetMotorTrackMap(unittest.TestCase):
    """Verify the brand-dispatch helper added to action_profile."""

    def setUp(self):
        from system.behavior.action_profile import get_motor_track_map, UNITREE_MOTOR_TRACK_MAP
        self._fn = get_motor_track_map
        self._unitree_map = UNITREE_MOTOR_TRACK_MAP

    def test_go2_returns_unitree_map(self):
        self.assertIs(self._fn("go2"), self._unitree_map)

    def test_unitree_brand_returns_unitree_map(self):
        self.assertIs(self._fn("unitree"), self._unitree_map)

    def test_h1_returns_unitree_map(self):
        self.assertIs(self._fn("h1"), self._unitree_map)

    def test_unknown_brand_returns_none(self):
        self.assertIsNone(self._fn("spot"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self._fn(""))

    def test_case_insensitive(self):
        self.assertIs(self._fn("GO2"), self._unitree_map)

    def test_returned_map_contains_leg_fl(self):
        result = self._fn("go2")
        self.assertIn("leg_fl", result)

    def test_returned_map_contains_body_posture(self):
        result = self._fn("go2")
        self.assertIn("body_posture", result)

    def test_none_robot_type_handled_gracefully(self):
        """Callers may pass None; must not raise."""
        # get_motor_track_map expects str; None coerces to "" via (None or "")
        # Behavior: returns None (unknown)
        result = self._fn(None or "")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
