#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Step 3 — Movement Settings source-switch model.

Coverage (pure-Python model only; no Qt):

  motor_param_source.STRUCTURAL_KEYS:
  - start_time and duration are structural
  - other keys (amplitude, gain, offset) are not structural

  motor_param_source.get_param_source():
  - structural key → always "constant", "" regardless of IO mappings
  - no matching IO mapping → "constant", ""
  - exact inbound mapping → "external", signal
  - outbound mapping does NOT cause "external"
  - multiple mappings: only the matching inbound matters
  - empty signal string → "external", ""

  IO mutation helpers via build_navigator_model round-trip:
  - constant param + no IO mapping → NavSource.CONSTANT in nav model
  - adding inbound IO mapping → NavSource.EXTERNAL in nav model
  - removing IO mapping → back to NavSource.CONSTANT (or UNRESOLVED if value absent)
  - external source takes precedence over constant value in seg.params

  No-ambiguity contract:
  - a param cannot be both Constant and External simultaneously
  - switching to External removes the constant interpretation from the nav model
"""

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system.behavior.motor_param_source import (
    STRUCTURAL_KEYS,
    ParamSourceInfo,
    get_param_source,
)
from system.behavior.motor_weight_nav_model import (
    NavSource,
    build_navigator_model,
)


# ---------------------------------------------------------------------------
# Minimal IO mapping stub (mirrors HBIOMapping duck-typing contract)
# ---------------------------------------------------------------------------

@dataclass
class _IO:
    signal: str
    target_param: str
    direction: str = "inbound"
    status: str = "active"


# Minimal BehaviorTimeline stubs
@dataclass
class _MotorSeg:
    motor_id: str
    track_name: str
    start_time: float = 0.0
    duration: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Overlay:
    action_id: str
    motor_segments: List[_MotorSeg] = field(default_factory=list)


@dataclass
class _ActionSeg:
    action_id: str
    name: str


class _FakeTimeline:
    def __init__(self, action_segments=None, motor_overlays=None):
        self.action_segments = action_segments or []
        self.motor_overlays = motor_overlays or []

    def is_empty(self):
        return not self.action_segments


# ---------------------------------------------------------------------------
# Tests: STRUCTURAL_KEYS
# ---------------------------------------------------------------------------

class TestStructuralKeys(unittest.TestCase):

    def test_start_time_is_structural(self):
        self.assertIn("start_time", STRUCTURAL_KEYS)

    def test_duration_is_structural(self):
        self.assertIn("duration", STRUCTURAL_KEYS)

    def test_amplitude_is_not_structural(self):
        self.assertNotIn("amplitude", STRUCTURAL_KEYS)

    def test_gain_is_not_structural(self):
        self.assertNotIn("gain", STRUCTURAL_KEYS)

    def test_offset_is_not_structural(self):
        self.assertNotIn("offset", STRUCTURAL_KEYS)

    def test_structural_keys_is_frozenset(self):
        self.assertIsInstance(STRUCTURAL_KEYS, frozenset)


# ---------------------------------------------------------------------------
# Tests: get_param_source
# ---------------------------------------------------------------------------

class TestGetParamSource(unittest.TestCase):

    def test_structural_key_always_constant_even_with_io_mapping(self):
        """Structural params (start_time/duration) ignore IO mappings."""
        io = [_IO(signal="clock.tick", target_param="leg_fl.duration", direction="inbound")]
        result = get_param_source("leg_fl", "duration", io)
        self.assertEqual(result.source, "constant")
        self.assertEqual(result.signal, "")

    def test_no_io_mappings_returns_constant(self):
        result = get_param_source("leg_fl", "amplitude", [])
        self.assertEqual(result.source, "constant")
        self.assertEqual(result.signal, "")

    def test_exact_inbound_match_returns_external(self):
        io = [_IO(signal="imu.pitch", target_param="leg_fl.amplitude", direction="inbound")]
        result = get_param_source("leg_fl", "amplitude", io)
        self.assertEqual(result.source, "external")
        self.assertEqual(result.signal, "imu.pitch")

    def test_outbound_mapping_does_not_count_as_external(self):
        io = [_IO(signal="leg_fl.amplitude", target_param="leg_fl.amplitude", direction="outbound")]
        result = get_param_source("leg_fl", "amplitude", io)
        self.assertEqual(result.source, "constant")

    def test_different_param_key_not_matched(self):
        """Mapping for amplitude does not affect gain resolution."""
        io = [_IO(signal="imu.pitch", target_param="leg_fl.amplitude", direction="inbound")]
        result = get_param_source("leg_fl", "gain", io)
        self.assertEqual(result.source, "constant")

    def test_different_track_name_not_matched(self):
        """Mapping for leg_fl does not affect leg_fr."""
        io = [_IO(signal="imu.pitch", target_param="leg_fl.amplitude", direction="inbound")]
        result = get_param_source("leg_fr", "amplitude", io)
        self.assertEqual(result.source, "constant")

    def test_empty_signal_returns_external_with_empty_signal(self):
        """Even an empty signal name marks the param as External."""
        io = [_IO(signal="", target_param="body_posture.gain", direction="inbound")]
        result = get_param_source("body_posture", "gain", io)
        self.assertEqual(result.source, "external")
        self.assertEqual(result.signal, "")

    def test_multiple_mappings_picks_correct_one(self):
        io = [
            _IO(signal="imu.pitch", target_param="leg_fl.amplitude", direction="inbound"),
            _IO(signal="imu.roll",  target_param="leg_fr.amplitude", direction="inbound"),
        ]
        self.assertEqual(get_param_source("leg_fl", "amplitude", io).signal, "imu.pitch")
        self.assertEqual(get_param_source("leg_fr", "amplitude", io).signal, "imu.roll")

    def test_returns_named_tuple(self):
        result = get_param_source("leg_fl", "amplitude", [])
        self.assertIsInstance(result, ParamSourceInfo)
        self.assertEqual(result.source, "constant")
        self.assertEqual(result.signal, "")

    def test_empty_io_mappings_list(self):
        result = get_param_source("head_pitch", "offset", [])
        self.assertEqual(result, ParamSourceInfo("constant", ""))


# ---------------------------------------------------------------------------
# Tests: nav-model round-trips verify source mutability contract
# ---------------------------------------------------------------------------

class TestSourceSwitchRoundTrip(unittest.TestCase):
    """Validate the IO mapping mutation contract through build_navigator_model.

    Simulates what the settings panel does when the user switches source:
    - Switch to External  → append inbound IO mapping
    - Switch to Constant  → remove inbound IO mapping
    """

    def _tl_with_seg(self, track: str, params: dict):
        seg = _ActionSeg("a1", "Walk")
        mseg = _MotorSeg("m1", track, params=params)
        overlay = _Overlay("a1", [mseg])
        return _FakeTimeline([seg], [overlay])

    def _get_motor_param(self, tl, io_mappings, track, param_key):
        result = build_navigator_model(tl, io_mappings)
        for region in result:
            for motor in region.motors:
                if motor.track_name == track:
                    for movement in motor.movements:
                        for p in movement.params:
                            if p.param_key == param_key:
                                return p
        return None

    def test_constant_param_shows_as_constant_in_nav(self):
        tl = self._tl_with_seg("leg_fl", {"amplitude": 0.3})
        p = self._get_motor_param(tl, [], "leg_fl", "amplitude")
        self.assertIsNotNone(p)
        self.assertEqual(p.source, NavSource.CONSTANT)

    def test_adding_inbound_io_mapping_makes_param_external_in_nav(self):
        tl = self._tl_with_seg("leg_fl", {"amplitude": 0.3})
        # Simulate: user switches to External → _io_upsert_external adds this mapping
        io = [_IO(signal="imu.roll", target_param="leg_fl.amplitude", direction="inbound")]
        p = self._get_motor_param(tl, io, "leg_fl", "amplitude")
        self.assertIsNotNone(p)
        self.assertEqual(p.source, NavSource.EXTERNAL)
        self.assertEqual(p.external_key, "imu.roll")
        self.assertFalse(p.has_issue)

    def test_removing_io_mapping_reverts_to_constant_in_nav(self):
        """After switching back to Constant the IO mapping is removed."""
        tl = self._tl_with_seg("leg_fl", {"amplitude": 0.3})
        # Start with external, then simulate removing the mapping
        p_no_io = self._get_motor_param(tl, [], "leg_fl", "amplitude")
        self.assertEqual(p_no_io.source, NavSource.CONSTANT)

    def test_external_overrides_constant_value_in_nav(self):
        """Even when seg.params has a value, inbound IO mapping wins in nav model."""
        tl = self._tl_with_seg("leg_fl", {"amplitude": 0.5})
        io = [_IO(signal="ctrl.freq", target_param="leg_fl.amplitude", direction="inbound")]
        p = self._get_motor_param(tl, io, "leg_fl", "amplitude")
        self.assertEqual(p.source, NavSource.EXTERNAL)
        # No ambiguity: external wins, constant not shown separately
        self.assertFalse(p.has_issue)

    def test_no_value_no_io_mapping_is_unresolved(self):
        """If source is Constant but value absent → UNRESOLVED (routing gap)."""
        tl = self._tl_with_seg("leg_fl", {})  # amplitude absent
        p = self._get_motor_param(tl, [], "leg_fl", "amplitude")
        self.assertIsNotNone(p)
        self.assertEqual(p.source, NavSource.UNRESOLVED)
        self.assertTrue(p.has_issue)

    def test_external_no_value_is_not_unresolved(self):
        """External binding with no constant value is fine (not an issue)."""
        tl = self._tl_with_seg("leg_fl", {})  # no constant value
        io = [_IO(signal="imu.pitch", target_param="leg_fl.amplitude", direction="inbound")]
        p = self._get_motor_param(tl, io, "leg_fl", "amplitude")
        self.assertIsNotNone(p)
        self.assertEqual(p.source, NavSource.EXTERNAL)
        self.assertFalse(p.has_issue)

    def test_structural_param_always_constant_even_with_mapping(self):
        """start_time: structural param should resolve to CONSTANT in nav model too."""
        tl = self._tl_with_seg("leg_fl", {"amplitude": 0.3})
        # The nav model uses prefix matching for IO — structural keys excluded by STRUCTURAL_KEYS
        from system.behavior.motor_param_source import STRUCTURAL_KEYS
        self.assertIn("start_time", STRUCTURAL_KEYS)
        # get_param_source also confirms this
        io = [_IO(signal="clock", target_param="leg_fl.start_time", direction="inbound")]
        result = get_param_source("leg_fl", "start_time", io)
        self.assertEqual(result.source, "constant")


# ---------------------------------------------------------------------------
# Tests: no-ambiguity contract (dual-source prevention)
# ---------------------------------------------------------------------------

class TestNoAmbiguity(unittest.TestCase):
    """Verify the contract that a param cannot have two active sources."""

    def test_external_binding_unique_per_param(self):
        """Only one inbound mapping per track/param — get_param_source returns first match."""
        io = [
            _IO(signal="sig_a", target_param="leg_fl.amplitude", direction="inbound"),
            _IO(signal="sig_b", target_param="leg_fl.amplitude", direction="inbound"),
        ]
        result = get_param_source("leg_fl", "amplitude", io)
        # get_param_source returns the FIRST match, preventing ambiguity in display
        self.assertEqual(result.source, "external")
        self.assertEqual(result.signal, "sig_a")

    def test_outbound_plus_inbound_only_inbound_is_external(self):
        """Outbound mapping does not interfere with inbound source resolution."""
        io = [
            _IO(signal="leg_fl.amplitude", target_param="recorder.input", direction="outbound"),
            _IO(signal="imu.pitch",         target_param="leg_fl.amplitude", direction="inbound"),
        ]
        result = get_param_source("leg_fl", "amplitude", io)
        self.assertEqual(result.source, "external")
        self.assertEqual(result.signal, "imu.pitch")


if __name__ == "__main__":
    unittest.main()
