#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior action profile — Phase 1 structured timeline data model.

Defines the canonical data structures for the multi-track behavior timeline:

  ActionSegment         — one high-level action on the main Action Track
  MotorSegment          — one motor/joint control segment on a sub-track
  ActionMotorOverlay    — one action + its aligned motor sub-track segments
  BehaviorTimeline      — full structured timeline (action_segments + overlays)
  MotorTrackDef         — motor track definition (name, label, safe/sim bounds)
  ActionPhaseTemplate   — SDK-aware action decomposition template
  MotorSegmentDiagnostic— diagnostic entry from motor parameter validation

  UNITREE_MOTOR_TRACKS    — motor track catalog for Unitree Go2/Go2W
  UNITREE_ACTION_PROFILES — action phase templates for Unitree SDK2

  build_timeline_from_modules()  — migrate old SequenceModule list to BehaviorTimeline
  validate_motor_segment()       — safe-range enforcement (error/warning)

Design constraints
------------------
- Pure Python; no Qt imports.
- All dataclass fields are optional-tolerant on load (from_dict never raises).
- Safe ranges are enforced on real hardware path; sim path allows sim bounds.
- BEHAVIOR_TIMELINE_VERSION enables forward-compat migration in future iterations.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

BEHAVIOR_TIMELINE_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# ActionSegment — main Action Track entry
# ---------------------------------------------------------------------------

@dataclass
class ActionSegment:
    """One high-level action on the main Action Track.

    Fields
    ------
    action_id   : Unique ID within this timeline (auto-generated UUID).
    name        : Action name (e.g. "walk", "stand", "sit").
    start_time  : Start offset in seconds from timeline origin (>= 0).
    duration    : Duration in seconds (> 0).
    params      : Key-value param overrides (e.g. {"speed": 0.3, "duration": 4.0}).
    kind        : "movement" | "behavior" — mirrors SequenceModule.kind.
    locked      : When True, dragging/resizing this segment is disabled.
    """

    action_id: str
    name: str
    start_time: float
    duration: float
    params: Dict[str, Any] = field(default_factory=dict)
    kind: str = "movement"
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "start_time": self.start_time,
            "duration": self.duration,
            "params": dict(self.params),
            "kind": self.kind,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionSegment":
        return cls(
            action_id=str(d.get("action_id") or str(uuid.uuid4())),
            name=str(d.get("name", "noop")),
            start_time=float(d.get("start_time", 0.0)),
            duration=max(0.1, float(d.get("duration", 1.0))),
            params=dict(d.get("params") or {}),
            kind=str(d.get("kind", "movement")),
            locked=bool(d.get("locked", False)),
        )

    @classmethod
    def create(
        cls,
        name: str,
        start_time: float,
        duration: float,
        params: Optional[Dict[str, Any]] = None,
        kind: str = "movement",
    ) -> "ActionSegment":
        return cls(
            action_id=str(uuid.uuid4()),
            name=name,
            start_time=start_time,
            duration=max(0.1, duration),
            params=dict(params or {}),
            kind=kind,
        )


# ---------------------------------------------------------------------------
# MotorSegment — sub-track entry (joint/leg-group level)
# ---------------------------------------------------------------------------

@dataclass
class MotorSegment:
    """One motor-level control segment on a named sub-track.

    Fields
    ------
    motor_id        : Unique ID within this timeline.
    track_name      : Sub-track identifier (e.g. "FL_hip", "leg_group_left").
    start_time      : Start offset in seconds, aligned to Action Track origin.
    duration        : Duration in seconds.
    params          : Motor-level parameters (e.g. {"amplitude": 0.1, "gain": 1.2}).
    parent_action_id: Optional action_id this segment is aligned to.
    locked          : When True, editing is disabled for this segment.
    """

    motor_id: str
    track_name: str
    start_time: float
    duration: float
    params: Dict[str, Any] = field(default_factory=dict)
    parent_action_id: Optional[str] = None
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motor_id": self.motor_id,
            "track_name": self.track_name,
            "start_time": self.start_time,
            "duration": self.duration,
            "params": dict(self.params),
            "parent_action_id": self.parent_action_id,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotorSegment":
        return cls(
            motor_id=str(d.get("motor_id") or str(uuid.uuid4())),
            track_name=str(d.get("track_name", "unknown")),
            start_time=float(d.get("start_time", 0.0)),
            duration=max(0.1, float(d.get("duration", 1.0))),
            params=dict(d.get("params") or {}),
            parent_action_id=d.get("parent_action_id") or None,
            locked=bool(d.get("locked", False)),
        )

    @classmethod
    def create(
        cls,
        track_name: str,
        start_time: float,
        duration: float,
        params: Optional[Dict[str, Any]] = None,
        parent_action_id: Optional[str] = None,
    ) -> "MotorSegment":
        return cls(
            motor_id=str(uuid.uuid4()),
            track_name=track_name,
            start_time=start_time,
            duration=max(0.1, duration),
            params=dict(params or {}),
            parent_action_id=parent_action_id,
        )


# ---------------------------------------------------------------------------
# ActionMotorOverlay — action + aligned motor sub-track segments
# ---------------------------------------------------------------------------

@dataclass
class ActionMotorOverlay:
    """Pairs an ActionSegment with its aligned motor sub-track segments.

    Motor segments are aligned to the same time axis as ActionSegments;
    start_time offsets are relative to the global timeline origin (same
    as ActionSegment.start_time).

    Fields
    ------
    action_id       : Must match the parent ActionSegment.action_id.
    motor_segments  : All MotorSegments belonging to this action.
    expanded        : Whether motor sub-tracks are visible in the UI.
    """

    action_id: str
    motor_segments: List[MotorSegment] = field(default_factory=list)
    expanded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "motor_segments": [s.to_dict() for s in self.motor_segments],
            "expanded": self.expanded,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionMotorOverlay":
        return cls(
            action_id=str(d.get("action_id", "")),
            motor_segments=[
                MotorSegment.from_dict(s)
                for s in (d.get("motor_segments") or [])
            ],
            expanded=bool(d.get("expanded", True)),
        )

    def get_segments_for_track(self, track_name: str) -> List[MotorSegment]:
        """Return only the motor segments belonging to the given track."""
        return [s for s in self.motor_segments if s.track_name == track_name]


# ---------------------------------------------------------------------------
# BehaviorTimeline — full structured timeline model
# ---------------------------------------------------------------------------

@dataclass
class BehaviorTimeline:
    """Full structured timeline for one behavior/action node.

    Fields
    ------
    version              : Schema version string (BEHAVIOR_TIMELINE_VERSION).
    action_segments      : Ordered ActionSegments on the main Action Track.
    motor_overlays       : ActionMotorOverlay list, one per action_segment
                           that has motor-level data.
    robot_type           : Robot type hint used during decomposition
                           (e.g. "go2").  Empty string means unknown.
    active_motor_tracks  : Ordered list of track names shown in the UI.
    """

    version: str = BEHAVIOR_TIMELINE_VERSION
    action_segments: List[ActionSegment] = field(default_factory=list)
    motor_overlays: List[ActionMotorOverlay] = field(default_factory=list)
    robot_type: str = ""
    active_motor_tracks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "action_segments": [s.to_dict() for s in self.action_segments],
            "motor_overlays": [o.to_dict() for o in self.motor_overlays],
            "robot_type": self.robot_type,
            "active_motor_tracks": list(self.active_motor_tracks),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BehaviorTimeline":
        """Deserialize from dict; returns empty timeline if d is None/invalid."""
        if not isinstance(d, dict):
            return cls()
        return cls(
            version=str(d.get("version", BEHAVIOR_TIMELINE_VERSION)),
            action_segments=[
                ActionSegment.from_dict(s)
                for s in (d.get("action_segments") or [])
            ],
            motor_overlays=[
                ActionMotorOverlay.from_dict(o)
                for o in (d.get("motor_overlays") or [])
            ],
            robot_type=str(d.get("robot_type", "")),
            active_motor_tracks=list(d.get("active_motor_tracks") or []),
        )

    def get_overlay_for_action(self, action_id: str) -> Optional[ActionMotorOverlay]:
        """Return the ActionMotorOverlay for the given action_id, or None."""
        for o in self.motor_overlays:
            if o.action_id == action_id:
                return o
        return None

    def is_empty(self) -> bool:
        return len(self.action_segments) == 0

    def total_duration(self) -> float:
        if not self.action_segments:
            return 0.0
        return max(s.start_time + s.duration for s in self.action_segments)

    def get_motor_segments_for_track(self, track_name: str) -> List[MotorSegment]:
        """Return all motor segments across all overlays for a given track."""
        result: List[MotorSegment] = []
        for overlay in self.motor_overlays:
            result.extend(overlay.get_segments_for_track(track_name))
        return result

    def set_overlay_expanded(self, action_id: str, expanded: bool) -> None:
        """Toggle expanded state for an action's motor overlay (no-op if missing)."""
        overlay = self.get_overlay_for_action(action_id)
        if overlay is not None:
            overlay.expanded = expanded

    def reorder_action(self, from_index: int, to_index: int) -> None:
        """Move ActionSegment from_index to to_index, updating start_times."""
        n = len(self.action_segments)
        if not (0 <= from_index < n and 0 <= to_index < n):
            return
        seg = self.action_segments.pop(from_index)
        self.action_segments.insert(to_index, seg)
        self._recalculate_start_times()

    def _recalculate_start_times(self) -> None:
        """Recompute start_time for every ActionSegment in order."""
        cursor = 0.0
        for seg in self.action_segments:
            seg.start_time = cursor
            cursor += seg.duration


# ---------------------------------------------------------------------------
# MotorTrackDef — motor track definition with safe/sim bounds
# ---------------------------------------------------------------------------

@dataclass
class MotorTrackDef:
    """Defines a motor sub-track: name, label, bounds, and rendering hints.

    Fields
    ------
    track_name  : Unique identifier (e.g. "FL_hip", "leg_group_left").
    label       : Human-readable label shown in the UI.
    color       : Hex color string for the track UI (#rrggbb).
    param_key   : The primary parameter this track modulates (e.g. "amplitude").
    safe_min    : Safe minimum value for real hardware path.
    safe_max    : Safe maximum value for real hardware path.
    sim_min     : Extended minimum value for simulation path.
    sim_max     : Extended maximum value for simulation path.
    unit        : Unit label for display (e.g. "m/s", "rad", "N·m").
    default_val : Default parameter value within safe range.
    """

    track_name: str
    label: str
    color: str = "#5f7fbf"
    param_key: str = "amplitude"
    safe_min: float = 0.0
    safe_max: float = 1.0
    sim_min: float = 0.0
    sim_max: float = 2.0
    unit: str = ""
    default_val: float = 0.5

    def clamp_safe(self, value: float) -> float:
        """Clamp value to safe hardware range."""
        return max(self.safe_min, min(self.safe_max, value))

    def clamp_sim(self, value: float) -> float:
        """Clamp value to simulation range."""
        return max(self.sim_min, min(self.sim_max, value))

    def is_in_safe_range(self, value: float) -> bool:
        return self.safe_min <= value <= self.safe_max

    def is_in_sim_range(self, value: float) -> bool:
        return self.sim_min <= value <= self.sim_max


# ---------------------------------------------------------------------------
# ActionPhaseTemplate — SDK-aware action decomposition template
# ---------------------------------------------------------------------------

@dataclass
class ActionPhaseTemplate:
    """SDK-aware template for decomposing a high-level action into motor phases.

    Fields
    ------
    action_name  : Action identifier (e.g. "walk", "stand").
    label        : Human-readable label.
    phases       : List of phase dicts — each has:
                     name            : str
                     duration_weight : float (proportional weight for duration)
                     motor_params    : Dict[str, Any] default param values
    motor_tracks : List of MotorTrackDef track_names this template populates.
    """

    action_name: str
    label: str
    phases: List[Dict[str, Any]] = field(default_factory=list)
    motor_tracks: List[str] = field(default_factory=list)

    def decompose(
        self,
        action_segment: ActionSegment,
        available_tracks: Optional[List[str]] = None,
    ) -> List[MotorSegment]:
        """Decompose an ActionSegment into MotorSegments using this template.

        Each phase produces one MotorSegment per applicable motor_track.
        Phase durations are proportional to action_segment.duration, with
        each phase's duration_weight acting as a relative weight.

        Parameters
        ----------
        action_segment   : The parent ActionSegment to decompose.
        available_tracks : Optional filter — only produce segments for these
                           tracks.  None means produce for all motor_tracks.
        """
        tracks = available_tracks if available_tracks is not None else self.motor_tracks
        tracks = [t for t in tracks if t in self.motor_tracks]
        if not self.phases or not tracks:
            return []

        total_weight = sum(float(p.get("duration_weight", 1.0)) for p in self.phases)
        if total_weight <= 0.0:
            total_weight = float(len(self.phases))

        segments: List[MotorSegment] = []
        phase_start = action_segment.start_time
        for phase in self.phases:
            weight = float(phase.get("duration_weight", 1.0))
            phase_dur = max(0.01, (weight / total_weight) * action_segment.duration)
            base_params = dict(phase.get("motor_params") or {})
            # Apply any matching user overrides from action_segment.params
            for k, v in action_segment.params.items():
                if k in base_params:
                    base_params[k] = v
            for track_name in tracks:
                seg = MotorSegment.create(
                    track_name=track_name,
                    start_time=phase_start,
                    duration=phase_dur,
                    params=dict(base_params),
                    parent_action_id=action_segment.action_id,
                )
                segments.append(seg)
            phase_start += phase_dur

        return segments


# ---------------------------------------------------------------------------
# Unitree motor tracks
# ---------------------------------------------------------------------------

UNITREE_MOTOR_TRACKS: List[MotorTrackDef] = [
    MotorTrackDef(
        track_name="leg_fl",
        label="Front Left Leg",
        color="#4d84c4",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
    MotorTrackDef(
        track_name="leg_fr",
        label="Front Right Leg",
        color="#7a5db6",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
    MotorTrackDef(
        track_name="leg_rl",
        label="Rear Left Leg",
        color="#4ca88b",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
    MotorTrackDef(
        track_name="leg_rr",
        label="Rear Right Leg",
        color="#b06a3d",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
    MotorTrackDef(
        track_name="body_posture",
        label="Body Posture",
        color="#5a9a5a",
        param_key="gain",
        safe_min=0.5, safe_max=1.5,
        sim_min=0.0,  sim_max=3.0,
        unit="",
        default_val=1.0,
    ),
    MotorTrackDef(
        track_name="head_pitch",
        label="Head Pitch",
        color="#9a5a1a",
        param_key="offset",
        safe_min=-0.2, safe_max=0.2,
        sim_min=-0.5,  sim_max=0.5,
        unit="rad",
        default_val=0.0,
    ),
    # Legacy compatibility tracks (old grouped layout).
    MotorTrackDef(
        track_name="leg_group_left",
        label="Left Legs (Legacy)",
        color="#4d84c4",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
    MotorTrackDef(
        track_name="leg_group_right",
        label="Right Legs (Legacy)",
        color="#7a5db6",
        param_key="amplitude",
        safe_min=0.0, safe_max=0.5,
        sim_min=0.0,  sim_max=1.0,
        unit="m/s",
        default_val=0.3,
    ),
]

# Map track_name → MotorTrackDef for fast lookup
UNITREE_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in UNITREE_MOTOR_TRACKS
}

# Robot types whose motor topology is covered by UNITREE_MOTOR_TRACK_MAP.
_UNITREE_ROBOT_TYPES = frozenset({
    "go2", "go2w", "h1", "h1_2", "b2", "b2w", "unitree",
})


def get_motor_track_map(robot_type: str) -> Optional[Dict[str, MotorTrackDef]]:
    """Return the motor-track catalog for *robot_type*, or None if unknown.

    None signals callers (e.g. MotorWeightNavigator) to fall back to
    overlay-derived tracks rather than showing a mis-matched catalog.
    """
    if (robot_type or "").lower() in _UNITREE_ROBOT_TYPES:
        return UNITREE_MOTOR_TRACK_MAP
    return None



# ---------------------------------------------------------------------------
# Unitree action phase templates
# ---------------------------------------------------------------------------

UNITREE_ACTION_PROFILES: Dict[str, ActionPhaseTemplate] = {
    "walk": ActionPhaseTemplate(
        action_name="walk",
        label="Walk",
        phases=[
            {
                "name": "prepare",
                "duration_weight": 0.1,
                "motor_params": {"amplitude": 0.1, "gain": 1.0},
            },
            {
                "name": "swing",
                "duration_weight": 0.6,
                "motor_params": {"amplitude": 0.3, "gain": 1.0},
            },
            {
                "name": "stance",
                "duration_weight": 0.3,
                "motor_params": {"amplitude": 0.15, "gain": 0.8},
            },
        ],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "stand": ActionPhaseTemplate(
        action_name="stand",
        label="Stand",
        phases=[
            {
                "name": "lift",
                "duration_weight": 0.3,
                "motor_params": {"amplitude": 0.2, "gain": 1.2},
            },
            {
                "name": "hold",
                "duration_weight": 0.7,
                "motor_params": {"amplitude": 0.05, "gain": 1.0},
            },
        ],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "sit": ActionPhaseTemplate(
        action_name="sit",
        label="Sit",
        phases=[
            {
                "name": "lower",
                "duration_weight": 0.5,
                "motor_params": {"amplitude": 0.3, "gain": 0.9},
            },
            {
                "name": "settle",
                "duration_weight": 0.5,
                "motor_params": {"amplitude": 0.05, "gain": 0.8},
            },
        ],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "wait": ActionPhaseTemplate(
        action_name="wait",
        label="Wait",
        phases=[
            {
                "name": "idle",
                "duration_weight": 1.0,
                "motor_params": {"amplitude": 0.02, "gain": 1.0},
            },
        ],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "lift_left_leg_prepare": ActionPhaseTemplate(
        action_name="lift_left_leg_prepare",
        label="Lift Left Leg / Prepare",
        phases=[{"name": "prepare", "duration_weight": 1.0, "motor_params": {"amplitude": 0.08, "gain": 1.1}}],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "lift_left_leg_raise": ActionPhaseTemplate(
        action_name="lift_left_leg_raise",
        label="Lift Left Leg / Raise",
        phases=[{"name": "raise", "duration_weight": 1.0, "motor_params": {"amplitude": 0.35, "gain": 1.2}}],
        motor_tracks=["leg_fl", "body_posture"],
    ),
    "lift_left_leg_hold": ActionPhaseTemplate(
        action_name="lift_left_leg_hold",
        label="Lift Left Leg / Hold",
        phases=[{"name": "hold", "duration_weight": 1.0, "motor_params": {"amplitude": 0.28, "gain": 1.15}}],
        motor_tracks=["leg_fl", "body_posture"],
    ),
    "lift_left_leg_recover": ActionPhaseTemplate(
        action_name="lift_left_leg_recover",
        label="Lift Left Leg / Recover",
        phases=[{"name": "recover", "duration_weight": 1.0, "motor_params": {"amplitude": 0.1, "gain": 1.0}}],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "lift_right_leg_prepare": ActionPhaseTemplate(
        action_name="lift_right_leg_prepare",
        label="Lift Right Leg / Prepare",
        phases=[{"name": "prepare", "duration_weight": 1.0, "motor_params": {"amplitude": 0.08, "gain": 1.1}}],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
    "lift_right_leg_raise": ActionPhaseTemplate(
        action_name="lift_right_leg_raise",
        label="Lift Right Leg / Raise",
        phases=[{"name": "raise", "duration_weight": 1.0, "motor_params": {"amplitude": 0.35, "gain": 1.2}}],
        motor_tracks=["leg_fr", "body_posture"],
    ),
    "lift_right_leg_hold": ActionPhaseTemplate(
        action_name="lift_right_leg_hold",
        label="Lift Right Leg / Hold",
        phases=[{"name": "hold", "duration_weight": 1.0, "motor_params": {"amplitude": 0.28, "gain": 1.15}}],
        motor_tracks=["leg_fr", "body_posture"],
    ),
    "lift_right_leg_recover": ActionPhaseTemplate(
        action_name="lift_right_leg_recover",
        label="Lift Right Leg / Recover",
        phases=[{"name": "recover", "duration_weight": 1.0, "motor_params": {"amplitude": 0.1, "gain": 1.0}}],
        motor_tracks=["leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"],
    ),
}


# Pre-installed action packages expanded into multiple Action Track segments.
# These names are also present in UNITREE_ACTION_PROFILES above so motor overlays
# are generated for each expanded internal action.
UNITREE_ACTION_PACKAGE_EXPANSIONS: Dict[str, List[Dict[str, Any]]] = {
    "lift_left_leg": [
        {"name": "lift_left_leg_prepare", "duration_weight": 0.20},
        {"name": "lift_left_leg_raise", "duration_weight": 0.30},
        {"name": "lift_left_leg_hold", "duration_weight": 0.30},
        {"name": "lift_left_leg_recover", "duration_weight": 0.20},
    ],
    "lift_right_leg": [
        {"name": "lift_right_leg_prepare", "duration_weight": 0.20},
        {"name": "lift_right_leg_raise", "duration_weight": 0.30},
        {"name": "lift_right_leg_hold", "duration_weight": 0.30},
        {"name": "lift_right_leg_recover", "duration_weight": 0.20},
    ],
}


# ---------------------------------------------------------------------------
# Motor segment validation
# ---------------------------------------------------------------------------

@dataclass
class MotorSegmentDiagnostic:
    """Diagnostic entry for an invalid motor segment parameter value.

    level : "error" on hardware-safe violation, "warning" on soft violation.
    """

    motor_id: str
    track_name: str
    param_key: str
    value: float
    level: str  # "error" | "warning"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motor_id": self.motor_id,
            "track_name": self.track_name,
            "param_key": self.param_key,
            "value": self.value,
            "level": self.level,
            "message": self.message,
        }


def validate_motor_segment(
    segment: MotorSegment,
    track_def: Optional[MotorTrackDef],
    is_simulation: bool = False,
) -> List[MotorSegmentDiagnostic]:
    """Validate motor segment parameter values against track bounds.

    Parameters
    ----------
    segment      : MotorSegment to validate.
    track_def    : MotorTrackDef providing bounds.  None → returns empty list.
    is_simulation: When True, sim_min/sim_max bounds are used; otherwise
                   safe_min/safe_max (hardware-safe bounds).

    Returns
    -------
    List of MotorSegmentDiagnostic entries; empty list means all OK.
    """
    if track_def is None:
        return []

    diags: List[MotorSegmentDiagnostic] = []
    param_key = track_def.param_key
    raw = segment.params.get(param_key)
    if raw is None:
        return diags

    try:
        val = float(raw)
    except (TypeError, ValueError):
        return diags

    if is_simulation:
        if not track_def.is_in_sim_range(val):
            diags.append(MotorSegmentDiagnostic(
                motor_id=segment.motor_id,
                track_name=segment.track_name,
                param_key=param_key,
                value=val,
                level="warning",
                message=(
                    f"{track_def.label} {param_key}={val} is outside simulation "
                    f"range [{track_def.sim_min}, {track_def.sim_max}]."
                ),
            ))
    else:
        if not track_def.is_in_safe_range(val):
            diags.append(MotorSegmentDiagnostic(
                motor_id=segment.motor_id,
                track_name=segment.track_name,
                param_key=param_key,
                value=val,
                level="error",
                message=(
                    f"{track_def.label} {param_key}={val} exceeds safe hardware "
                    f"range [{track_def.safe_min}, {track_def.safe_max}]. "
                    f"Use simulation mode for extended tuning."
                ),
            ))
    return diags


def validate_timeline(
    timeline: BehaviorTimeline,
    is_simulation: bool = False,
    track_map: Optional[Dict[str, MotorTrackDef]] = None,
) -> List[MotorSegmentDiagnostic]:
    """Validate all motor segments in a BehaviorTimeline.

    Parameters
    ----------
    timeline      : BehaviorTimeline to validate.
    is_simulation : Passed through to validate_motor_segment().
    track_map     : Map of track_name → MotorTrackDef for bounds lookup.
                    Defaults to UNITREE_MOTOR_TRACK_MAP when None.

    Returns
    -------
    List of MotorSegmentDiagnostic entries for all violations.
    """
    if track_map is None:
        track_map = UNITREE_MOTOR_TRACK_MAP
    diags: List[MotorSegmentDiagnostic] = []
    for overlay in timeline.motor_overlays:
        for seg in overlay.motor_segments:
            tdef = track_map.get(seg.track_name)
            diags.extend(validate_motor_segment(seg, tdef, is_simulation))
    return diags


# ---------------------------------------------------------------------------
# Timeline migration from legacy SequenceModule list
# ---------------------------------------------------------------------------

def build_timeline_from_modules(
    modules: List[Any],
    robot_type: str = "",
    auto_decompose: bool = True,
) -> BehaviorTimeline:
    """Migrate a legacy SequenceModule list into a BehaviorTimeline.

    Parameters
    ----------
    modules       : List of SequenceModule instances or raw dicts.
                    Dicts must have "name", "duration", "kind", "args" keys.
    robot_type    : Robot type hint for decomposition ("go2", "unitree", etc.).
                    Empty string triggers Unitree decomposition by default.
    auto_decompose: When True, auto-populate motor overlays from Unitree
                    profiles when robot_type is empty/"go2"/"unitree".

    Returns
    -------
    BehaviorTimeline — always valid; never raises.
    """
    action_segments: List[ActionSegment] = []
    cursor = 0.0

    for mod in modules:
        if hasattr(mod, "__dict__"):
            d = mod.__dict__
        elif isinstance(mod, dict):
            d = mod
        else:
            continue

        name = str(d.get("name", "noop"))
        raw_dur = d.get("duration", 1.0)
        try:
            duration = max(0.1, float(raw_dur))
        except (TypeError, ValueError):
            duration = 1.0
        kind = str(d.get("kind", "movement"))

        # Parse args string into params dict for preservation
        args_str = str(d.get("args", "")).strip()
        params: Dict[str, Any] = {}
        if args_str:
            for part in args_str.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v

        pkg_steps = UNITREE_ACTION_PACKAGE_EXPANSIONS.get(name)
        if not pkg_steps:
            seg = ActionSegment.create(
                name=name,
                start_time=cursor,
                duration=duration,
                params=params,
                kind=kind,
            )
            action_segments.append(seg)
            cursor += duration
            continue

        total_w = sum(float(s.get("duration_weight", 1.0)) for s in pkg_steps)
        if total_w <= 0.0:
            total_w = float(len(pkg_steps))
        seg_cursor = cursor
        for idx, step in enumerate(pkg_steps):
            step_name = str(step.get("name", name))
            if idx == len(pkg_steps) - 1:
                step_duration = max(0.1, (cursor + duration) - seg_cursor)
            else:
                weight = float(step.get("duration_weight", 1.0))
                step_duration = max(0.1, (weight / total_w) * duration)
            step_seg = ActionSegment.create(
                name=step_name,
                start_time=seg_cursor,
                duration=step_duration,
                params=dict(params),
                kind=kind,
            )
            action_segments.append(step_seg)
            seg_cursor += step_duration
        cursor = seg_cursor

    # Decide whether to apply Unitree decomposition
    use_unitree = auto_decompose and (
        robot_type in ("", "go2", "unitree")
        or "go2" in robot_type
        or "unitree" in robot_type
    )

    motor_overlays: List[ActionMotorOverlay] = []
    active_tracks: List[str] = []
    tracks_seen: set = set()

    if use_unitree:
        for seg in action_segments:
            template = UNITREE_ACTION_PROFILES.get(seg.name)
            if template is None:
                overlay = ActionMotorOverlay(action_id=seg.action_id)
            else:
                motor_segs = template.decompose(seg)
                overlay = ActionMotorOverlay(
                    action_id=seg.action_id,
                    motor_segments=motor_segs,
                )
                for track_name in template.motor_tracks:
                    if track_name not in tracks_seen:
                        active_tracks.append(track_name)
                        tracks_seen.add(track_name)
            motor_overlays.append(overlay)

    return BehaviorTimeline(
        version=BEHAVIOR_TIMELINE_VERSION,
        action_segments=action_segments,
        motor_overlays=motor_overlays,
        robot_type=robot_type,
        active_motor_tracks=active_tracks,
    )
