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
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

BEHAVIOR_TIMELINE_VERSION: str = "1.1"
# v1.0 — initial timeline schema
# v1.1 — ActionSegment gains intent_id as the durable semantic authoring field;
#         raw action (name) is retained as derived/display metadata.


# ---------------------------------------------------------------------------
# ActionSegment — main Action Track entry
# ---------------------------------------------------------------------------

@dataclass
class ActionSegment:
    """One high-level action on the main Action Track.

    Fields
    ------
    action_id   : Unique ID within this timeline (auto-generated UUID).
    name        : Raw action / display name (e.g. "walk", "stand", "sit").
                  Retained as derived/display metadata.  **Not** the primary
                  authoring identity — use ``intent_id`` for that.
    start_time  : Start offset in seconds from timeline origin (>= 0).
    duration    : Duration in seconds (> 0).
    params      : Key-value param overrides (e.g. {"speed": 0.3, "duration": 4.0}).
    kind        : "movement" | "behavior" — mirrors SequenceModule.kind.
    locked      : When True, dragging/resizing this segment is disabled.
    intent_id   : **Durable semantic authoring key** (e.g. "locomotion.forward").
                  This is the stable identifier stored in behavior drafts and
                  used for cross-model compatibility.  Empty string means the
                  segment was created before Circle 4 migration (legacy format);
                  call ``migrate_timeline()`` to resolve it.
    """

    action_id: str
    name: str
    start_time: float
    duration: float
    params: Dict[str, Any] = field(default_factory=dict)
    kind: str = "movement"
    locked: bool = False
    intent_id: str = ""  # v1.1: durable semantic ID; "" = legacy / not yet resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "start_time": self.start_time,
            "duration": self.duration,
            "params": dict(self.params),
            "kind": self.kind,
            "locked": self.locked,
            "intent_id": self.intent_id,  # v1.1: always serialised
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
            intent_id=str(d.get("intent_id") or ""),
        )

    @classmethod
    def create(
        cls,
        name: str,
        start_time: float,
        duration: float,
        params: Optional[Dict[str, Any]] = None,
        kind: str = "movement",
        intent_id: str = "",
    ) -> "ActionSegment":
        return cls(
            action_id=str(uuid.uuid4()),
            name=name,
            start_time=start_time,
            duration=max(0.1, duration),
            params=dict(params or {}),
            kind=kind,
            intent_id=intent_id,
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
# MotorTrackDef — motor track definition with safe/sim bounds and real motor mapping
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
    
    Real motor mapping fields (v2.0):
    ------------------
    motor_ids   : List of real motor IDs this track maps to (from motor_registry).
                  Empty list means this is a composite/virtual track.
    joint_names : List of joint names for direct motor control.
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
    # Real motor mapping (v2.0)
    motor_ids: Tuple[str, ...] = field(default_factory=tuple)
    joint_names: Tuple[str, ...] = field(default_factory=tuple)

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
        motor_ids=("go2_fl_hip", "go2_fl_thigh", "go2_fl_calf"),
        joint_names=("FL_hip", "FL_thigh", "FL_calf"),
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
        motor_ids=("go2_fr_hip", "go2_fr_thigh", "go2_fr_calf"),
        joint_names=("FR_hip", "FR_thigh", "FR_calf"),
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
        motor_ids=("go2_rl_hip", "go2_rl_thigh", "go2_rl_calf"),
        joint_names=("RL_hip", "RL_thigh", "RL_calf"),
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
        motor_ids=("go2_rr_hip", "go2_rr_thigh", "go2_rr_calf"),
        joint_names=("RR_hip", "RR_thigh", "RR_calf"),
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
        motor_ids=(),
        joint_names=(),
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
        motor_ids=(),
        joint_names=(),
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
        motor_ids=("go2_fl_hip", "go2_fl_thigh", "go2_fl_calf", "go2_rl_hip", "go2_rl_thigh", "go2_rl_calf"),
        joint_names=("FL_hip", "FL_thigh", "FL_calf", "RL_hip", "RL_thigh", "RL_calf"),
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
        motor_ids=("go2_fr_hip", "go2_fr_thigh", "go2_fr_calf", "go2_rr_hip", "go2_rr_thigh", "go2_rr_calf"),
        joint_names=("FR_hip", "FR_thigh", "FR_calf", "RR_hip", "RR_thigh", "RR_calf"),
    ),
]

# Map track_name → MotorTrackDef for fast lookup
UNITREE_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in UNITREE_MOTOR_TRACKS
}

# Robot types whose motor topology is covered by UNITREE_MOTOR_TRACK_MAP.
_UNITREE_ROBOT_TYPES = frozenset({
    "go2", "go2w", "a1", "b1", "h1", "h1_2", "b2", "b2w", "unitree",
})


# ---------------------------------------------------------------------------
# A1 Motor Tracks — 4 legs, 12 motors total (3 joints per leg)
# ---------------------------------------------------------------------------
A1_MOTOR_TRACKS: List[MotorTrackDef] = [
    # Front Left Leg
    MotorTrackDef(track_name="FL_hip", label="Front Left Hip", color="#4d84c4",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FL_thigh", label="Front Left Thigh", color="#5a9ad4",
                 param_key="position", safe_min=0.0, safe_max=2.6, sim_min=-0.5, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="FL_calf", label="Front Left Calf", color="#6ab0e4",
                 param_key="position", safe_min=-2.7, safe_max=-0.5, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.5),
    # Front Right Leg
    MotorTrackDef(track_name="FR_hip", label="Front Right Hip", color="#7a5db6",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FR_thigh", label="Front Right Thigh", color="#8a6dc6",
                 param_key="position", safe_min=0.0, safe_max=2.6, sim_min=-0.5, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="FR_calf", label="Front Right Calf", color="#9a7dd6",
                 param_key="position", safe_min=-2.7, safe_max=-0.5, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.5),
    # Rear Left Leg
    MotorTrackDef(track_name="RL_hip", label="Rear Left Hip", color="#4ca88b",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RL_thigh", label="Rear Left Thigh", color="#5cb89b",
                 param_key="position", safe_min=0.0, safe_max=2.6, sim_min=-0.5, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="RL_calf", label="Rear Left Calf", color="#6cc8ab",
                 param_key="position", safe_min=-2.7, safe_max=-0.5, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.5),
    # Rear Right Leg
    MotorTrackDef(track_name="RR_hip", label="Rear Right Hip", color="#b06a3d",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RR_thigh", label="Rear Right Thigh", color="#c07a4d",
                 param_key="position", safe_min=0.0, safe_max=2.6, sim_min=-0.5, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="RR_calf", label="Rear Right Calf", color="#d08a5d",
                 param_key="position", safe_min=-2.7, safe_max=-0.5, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.5),
    # Body posture
    MotorTrackDef(track_name="body_posture", label="Body Posture", color="#5a9a5a",
                 param_key="gain", safe_min=0.5, safe_max=1.5, sim_min=0.0, sim_max=3.0, unit="", default_val=1.0),
]

A1_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in A1_MOTOR_TRACKS
}

# ---------------------------------------------------------------------------
# B1 Motor Tracks — 4 legs, 12 motors + additional joints
# ---------------------------------------------------------------------------
B1_MOTOR_TRACKS: List[MotorTrackDef] = [
    # Front Left Leg
    MotorTrackDef(track_name="FL_hip", label="Front Left Hip", color="#4d84c4",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FL_thigh", label="Front Left Thigh", color="#5a9ad4",
                 param_key="position", safe_min=0.0, safe_max=2.8, sim_min=-0.5, sim_max=3.2, unit="rad", default_val=1.1),
    MotorTrackDef(track_name="FL_calf", label="Front Left Calf", color="#6ab0e4",
                 param_key="position", safe_min=-2.9, safe_max=-0.6, sim_min=-3.2, sim_max=0.0, unit="rad", default_val=-1.6),
    # Front Right Leg
    MotorTrackDef(track_name="FR_hip", label="Front Right Hip", color="#7a5db6",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FR_thigh", label="Front Right Thigh", color="#8a6dc6",
                 param_key="position", safe_min=0.0, safe_max=2.8, sim_min=-0.5, sim_max=3.2, unit="rad", default_val=1.1),
    MotorTrackDef(track_name="FR_calf", label="Front Right Calf", color="#9a7dd6",
                 param_key="position", safe_min=-2.9, safe_max=-0.6, sim_min=-3.2, sim_max=0.0, unit="rad", default_val=-1.6),
    # Rear Left Leg
    MotorTrackDef(track_name="RL_hip", label="Rear Left Hip", color="#4ca88b",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RL_thigh", label="Rear Left Thigh", color="#5cb89b",
                 param_key="position", safe_min=0.0, safe_max=2.8, sim_min=-0.5, sim_max=3.2, unit="rad", default_val=1.1),
    MotorTrackDef(track_name="RL_calf", label="Rear Left Calf", color="#6cc8ab",
                 param_key="position", safe_min=-2.9, safe_max=-0.6, sim_min=-3.2, sim_max=0.0, unit="rad", default_val=-1.6),
    # Rear Right Leg
    MotorTrackDef(track_name="RR_hip", label="Rear Right Hip", color="#b06a3d",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RR_thigh", label="Rear Right Thigh", color="#c07a4d",
                 param_key="position", safe_min=0.0, safe_max=2.8, sim_min=-0.5, sim_max=3.2, unit="rad", default_val=1.1),
    MotorTrackDef(track_name="RR_calf", label="Rear Right Calf", color="#d08a5d",
                 param_key="position", safe_min=-2.9, safe_max=-0.6, sim_min=-3.2, sim_max=0.0, unit="rad", default_val=-1.6),
    # Body posture
    MotorTrackDef(track_name="body_posture", label="Body Posture", color="#5a9a5a",
                 param_key="gain", safe_min=0.5, safe_max=1.5, sim_min=0.0, sim_max=3.0, unit="", default_val=1.0),
    # Waist (B1 specific)
    MotorTrackDef(track_name="waist", label="Waist", color="#9a5a1a",
                 param_key="position", safe_min=-0.5, safe_max=0.5, sim_min=-1.0, sim_max=1.0, unit="rad", default_val=0.0),
]

B1_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in B1_MOTOR_TRACKS
}

# ---------------------------------------------------------------------------
# G1 Motor Tracks — Humanoid robot with more joints
# ---------------------------------------------------------------------------
G1_MOTOR_TRACKS: List[MotorTrackDef] = [
    # Left Leg
    MotorTrackDef(track_name="L_hip_yaw", label="Left Hip Yaw", color="#4d84c4",
                 param_key="position", safe_min=-1.0, safe_max=1.0, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="L_hip_roll", label="Left Hip Roll", color="#5a9ad4",
                 param_key="position", safe_min=-0.5, safe_max=0.5, sim_min=-1.0, sim_max=1.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="L_hip_pitch", label="Left Hip Pitch", color="#6ab0e4",
                 param_key="position", safe_min=-1.5, safe_max=0.2, sim_min=-2.0, sim_max=0.5, unit="rad", default_val=-0.5),
    MotorTrackDef(track_name="L_knee", label="Left Knee", color="#7ac6f4",
                 param_key="position", safe_min=0.0, safe_max=2.5, sim_min=-0.2, sim_max=2.8, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="L_ankle", label="Left Ankle", color="#8adcfc",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    # Right Leg
    MotorTrackDef(track_name="R_hip_yaw", label="Right Hip Yaw", color="#b06a3d",
                 param_key="position", safe_min=-1.0, safe_max=1.0, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="R_hip_roll", label="Right Hip Roll", color="#c07a4d",
                 param_key="position", safe_min=-0.5, safe_max=0.5, sim_min=-1.0, sim_max=1.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="R_hip_pitch", label="Right Hip Pitch", color="#d08a5d",
                 param_key="position", safe_min=-1.5, safe_max=0.2, sim_min=-2.0, sim_max=0.5, unit="rad", default_val=-0.5),
    MotorTrackDef(track_name="R_knee", label="Right Knee", color="#e09a6d",
                 param_key="position", safe_min=0.0, safe_max=2.5, sim_min=-0.2, sim_max=2.8, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="R_ankle", label="Right Ankle", color="#f0aa7d",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    # Left Arm
    MotorTrackDef(track_name="L_shoulder_pitch", label="Left Shoulder Pitch", color="#4ca88b",
                 param_key="position", safe_min=-2.8, safe_max=2.8, sim_min=-3.0, sim_max=3.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="L_shoulder_roll", label="Left Shoulder Roll", color="#5cb89b",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="L_elbow", label="Left Elbow", color="#6cc8ab",
                 param_key="position", safe_min=-2.6, safe_max=0.0, sim_min=-3.0, sim_max=0.2, unit="rad", default_val=-1.0),
    # Right Arm
    MotorTrackDef(track_name="R_shoulder_pitch", label="Right Shoulder Pitch", color="#7a5db6",
                 param_key="position", safe_min=-2.8, safe_max=2.8, sim_min=-3.0, sim_max=3.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="R_shoulder_roll", label="Right Shoulder Roll", color="#8a6dc6",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="R_elbow", label="Right Elbow", color="#9a7dd6",
                 param_key="position", safe_min=-2.6, safe_max=0.0, sim_min=-3.0, sim_max=0.2, unit="rad", default_val=-1.0),
    # Torso
    MotorTrackDef(track_name="torso_yaw", label="Torso Yaw", color="#5a9a5a",
                 param_key="position", safe_min=-1.0, safe_max=1.0, sim_min=-1.5, sim_max=1.5, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="torso_pitch", label="Torso Pitch", color="#6aaa6a",
                 param_key="position", safe_min=-0.5, safe_max=0.5, sim_min=-0.8, sim_max=0.8, unit="rad", default_val=0.0),
]

G1_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in G1_MOTOR_TRACKS
}

# ---------------------------------------------------------------------------
# Spot Motor Tracks (BostionDynamics) — 4 legs, 12 motors
# ---------------------------------------------------------------------------
SPOT_MOTOR_TRACKS: List[MotorTrackDef] = [
    # Front Left Leg
    MotorTrackDef(track_name="FL_hip_x", label="FL Hip X (Abduction)", color="#4d84c4",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FL_hip_y", label="FL Hip Y (Flexion)", color="#5a9ad4",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.5),
    MotorTrackDef(track_name="FL_knee", label="FL Knee", color="#6ab0e4",
                 param_key="position", safe_min=-2.6, safe_max=-0.2, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.2),
    # Front Right Leg
    MotorTrackDef(track_name="FR_hip_x", label="FR Hip X (Abduction)", color="#7a5db6",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FR_hip_y", label="FR Hip Y (Flexion)", color="#8a6dc6",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.5),
    MotorTrackDef(track_name="FR_knee", label="FR Knee", color="#9a7dd6",
                 param_key="position", safe_min=-2.6, safe_max=-0.2, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.2),
    # Rear Left Leg
    MotorTrackDef(track_name="RL_hip_x", label="RL Hip X (Abduction)", color="#4ca88b",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RL_hip_y", label="RL Hip Y (Flexion)", color="#5cb89b",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.5),
    MotorTrackDef(track_name="RL_knee", label="RL Knee", color="#6cc8ab",
                 param_key="position", safe_min=-2.6, safe_max=-0.2, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.2),
    # Rear Right Leg
    MotorTrackDef(track_name="RR_hip_x", label="RR Hip X (Abduction)", color="#b06a3d",
                 param_key="position", safe_min=-0.8, safe_max=0.8, sim_min=-1.2, sim_max=1.2, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RR_hip_y", label="RR Hip Y (Flexion)", color="#c07a4d",
                 param_key="position", safe_min=-1.5, safe_max=1.5, sim_min=-2.0, sim_max=2.0, unit="rad", default_val=0.5),
    MotorTrackDef(track_name="RR_knee", label="RR Knee", color="#d08a5d",
                 param_key="position", safe_min=-2.6, safe_max=-0.2, sim_min=-3.0, sim_max=0.0, unit="rad", default_val=-1.2),
    # Body posture
    MotorTrackDef(track_name="body_posture", label="Body Posture", color="#5a9a5a",
                 param_key="gain", safe_min=0.5, safe_max=1.5, sim_min=0.0, sim_max=3.0, unit="", default_val=1.0),
]

SPOT_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in SPOT_MOTOR_TRACKS
}

# ---------------------------------------------------------------------------
# Cyberdog Motor Tracks (Xiaomi) — 4 legs, 12 motors
# ---------------------------------------------------------------------------
CYBERDOG_MOTOR_TRACKS: List[MotorTrackDef] = [
    # Front Left Leg
    MotorTrackDef(track_name="FL_hip", label="Front Left Hip", color="#4d84c4",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.4, sim_max=1.4, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FL_thigh", label="Front Left Thigh", color="#5a9ad4",
                 param_key="position", safe_min=0.0, safe_max=2.7, sim_min=-0.4, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="FL_calf", label="Front Left Calf", color="#6ab0e4",
                 param_key="position", safe_min=-2.8, safe_max=-0.5, sim_min=-3.1, sim_max=0.0, unit="rad", default_val=-1.4),
    # Front Right Leg
    MotorTrackDef(track_name="FR_hip", label="Front Right Hip", color="#7a5db6",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.4, sim_max=1.4, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="FR_thigh", label="Front Right Thigh", color="#8a6dc6",
                 param_key="position", safe_min=0.0, safe_max=2.7, sim_min=-0.4, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="FR_calf", label="Front Right Calf", color="#9a7dd6",
                 param_key="position", safe_min=-2.8, safe_max=-0.5, sim_min=-3.1, sim_max=0.0, unit="rad", default_val=-1.4),
    # Rear Left Leg
    MotorTrackDef(track_name="RL_hip", label="Rear Left Hip", color="#4ca88b",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.4, sim_max=1.4, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RL_thigh", label="Rear Left Thigh", color="#5cb89b",
                 param_key="position", safe_min=0.0, safe_max=2.7, sim_min=-0.4, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="RL_calf", label="Rear Left Calf", color="#6cc8ab",
                 param_key="position", safe_min=-2.8, safe_max=-0.5, sim_min=-3.1, sim_max=0.0, unit="rad", default_val=-1.4),
    # Rear Right Leg
    MotorTrackDef(track_name="RR_hip", label="Rear Right Hip", color="#b06a3d",
                 param_key="position", safe_min=-0.9, safe_max=0.9, sim_min=-1.4, sim_max=1.4, unit="rad", default_val=0.0),
    MotorTrackDef(track_name="RR_thigh", label="Rear Right Thigh", color="#c07a4d",
                 param_key="position", safe_min=0.0, safe_max=2.7, sim_min=-0.4, sim_max=3.0, unit="rad", default_val=1.0),
    MotorTrackDef(track_name="RR_calf", label="Rear Right Calf", color="#d08a5d",
                 param_key="position", safe_min=-2.8, safe_max=-0.5, sim_min=-3.1, sim_max=0.0, unit="rad", default_val=-1.4),
    # Body posture
    MotorTrackDef(track_name="body_posture", label="Body Posture", color="#5a9a5a",
                 param_key="gain", safe_min=0.5, safe_max=1.5, sim_min=0.0, sim_max=3.0, unit="", default_val=1.0),
]

CYBERDOG_MOTOR_TRACK_MAP: Dict[str, MotorTrackDef] = {
    t.track_name: t for t in CYBERDOG_MOTOR_TRACKS
}

# ---------------------------------------------------------------------------
# Motor track map registry by brand and robot type
# ---------------------------------------------------------------------------
_MOTOR_TRACK_REGISTRY: Dict[Tuple[str, str], Dict[str, MotorTrackDef]] = {
    # Unitree robots
    ("unitree", "go2"): UNITREE_MOTOR_TRACK_MAP,
    ("unitree", "go2w"): UNITREE_MOTOR_TRACK_MAP,
    ("unitree", "a1"): A1_MOTOR_TRACK_MAP,
    ("unitree", "b1"): B1_MOTOR_TRACK_MAP,
    # B2: Uses same motor configuration as B1 (approximation - verify against actual B2 specs if needed)
    # H1: Uses same motor configuration as G1 (approximation - verify against actual H1 specs if needed)
    ("unitree", "b2"): B1_MOTOR_TRACK_MAP,
    ("unitree", "g1"): G1_MOTOR_TRACK_MAP,
    ("unitree", "h1"): G1_MOTOR_TRACK_MAP,
    ("unitree", "h1_2"): G1_MOTOR_TRACK_MAP,
    # BostionDynamics robots
    ("bostiondynamics", "spot"): SPOT_MOTOR_TRACK_MAP,
    # Xiaomi robots
    ("xiaomi", "cyberdog"): CYBERDOG_MOTOR_TRACK_MAP,
    ("xiaomi", "cyberdog2"): CYBERDOG_MOTOR_TRACK_MAP,
}


def get_motor_track_map(robot_type: str, brand: str = "unitree") -> Optional[Dict[str, MotorTrackDef]]:
    """Return the motor-track catalog for *robot_type*, or None if unknown.

    Parameters
    ----------
    robot_type : str
        Robot type identifier (e.g. "go2", "a1", "spot", "cyberdog")
    brand : str
        Robot brand (e.g. "unitree", "bostiondynamics", "xiaomi")
        Defaults to "unitree" for backward compatibility.

    Returns
    -------
    Optional[Dict[str, MotorTrackDef]]
        Motor track map for the specified robot, or None if unknown.

    None signals callers (e.g. MotorWeightNavigator) to fall back to
    overlay-derived tracks rather than showing a mis-matched catalog.
    """
    key = (str(brand or "").lower(), str(robot_type or "").lower())
    track_map = _MOTOR_TRACK_REGISTRY.get(key)
    if track_map is not None:
        return track_map

    # Fallback: check if robot_type is in UNITREE_ROBOT_TYPES (backward compat)
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

    # Decide whether to apply Unitree decomposition.
    # Covers all known Unitree models; Go2/Go2W use abstract tracks directly,
    # other models have abstract tracks expanded to joint-level below.
    _norm_rt = (robot_type or "").lower()
    use_unitree = auto_decompose and (
        not _norm_rt
        or _norm_rt in ("go2", "go2w", "unitree")
        or _norm_rt in ("a1", "b1", "b2", "g1", "h1", "h1_2")
        or "go2" in _norm_rt
        or "unitree" in _norm_rt
    )

    # For joint-addressed models (A1, B1, H1, G1 …) the canonical topology
    # uses individual joint names as track keys (FL_hip, L_hip_yaw, …) rather
    # than the Go2 abstract keys (leg_fl, …) that UNITREE_ACTION_PROFILES emit.
    # Build an abstract→[joint_name] expansion map from the motor registry so
    # the decomposed segments can be re-keyed to match the topology.
    #
    # Humanoid models (H1, G1) use "leg_l"/"leg_r" as abstract tracks while
    # Go2 profiles use "leg_fl"/"leg_rl" (front/rear-left) and
    # "leg_fr"/"leg_rr" (front/rear-right).  Map these onto the single
    # humanoid abstract track to avoid missing expansion; deduplicate so that
    # the left-leg joints are only emitted once (from leg_fl, not leg_rl too).
    _GO2_TO_HUMANOID: Dict[str, str] = {
        "leg_fl": "leg_l",
        "leg_rl": "",        # collapsed into leg_l via leg_fl — skip to avoid dups
        "leg_fr": "leg_r",
        "leg_rr": "",        # collapsed into leg_r via leg_fr — skip
        "body_posture": "",  # no direct equivalent; omit for now
    }

    _is_go2_family = _norm_rt in ("", "go2", "go2w") or "go2" in _norm_rt
    _abstract_to_joints: Dict[str, List[str]] = {}
    if use_unitree and not _is_go2_family:
        try:
            from system.behavior.motor_registry import get_motors_for_robot  # type: ignore
            _model_motors = get_motors_for_robot("unitree", _norm_rt)
            # Build model-native abstract→joints map
            _native: Dict[str, List[str]] = {}
            for _ms in _model_motors:
                _native.setdefault(_ms.abstract_track, []).append(_ms.joint_name)

            # Check if Go2 abstract tracks (leg_fl …) exist directly in the
            # model's registry (A1/B1 use same abstract names as Go2).
            _go2_tracks = {"leg_fl", "leg_fr", "leg_rl", "leg_rr", "body_posture"}
            if _go2_tracks & set(_native):
                # Direct match (A1, B1) — use as-is.
                _abstract_to_joints = _native
            else:
                # Humanoid remap: map Go2 names through _GO2_TO_HUMANOID first.
                for _go2_key, _humanoid_key in _GO2_TO_HUMANOID.items():
                    if _humanoid_key and _humanoid_key in _native:
                        _abstract_to_joints[_go2_key] = _native[_humanoid_key]
        except Exception:
            pass

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
                # Expand abstract tracks → joint-level for non-Go2 models.
                if _abstract_to_joints:
                    expanded: List[MotorSegment] = []
                    for mseg in motor_segs:
                        joint_names = _abstract_to_joints.get(mseg.track_name)
                        if joint_names:
                            for jn in joint_names:
                                eseg = MotorSegment.create(
                                    track_name=jn,
                                    start_time=mseg.start_time,
                                    duration=mseg.duration,
                                    params=dict(mseg.params),
                                    parent_action_id=mseg.parent_action_id,
                                )
                                expanded.append(eseg)
                                if jn not in tracks_seen:
                                    active_tracks.append(jn)
                                    tracks_seen.add(jn)
                        else:
                            # In expansion mode: unmapped tracks are dropped.
                            # They either have no equivalent on this model (e.g.
                            # quadruped rear-legs merged into humanoid leg) or are
                            # semantically irrelevant (body_posture for humanoid).
                            pass
                    motor_segs = expanded
                else:
                    for track_name in template.motor_tracks:
                        if track_name not in tracks_seen:
                            active_tracks.append(track_name)
                            tracks_seen.add(track_name)
                overlay = ActionMotorOverlay(
                    action_id=seg.action_id,
                    motor_segments=motor_segs,
                )
            motor_overlays.append(overlay)

    return BehaviorTimeline(
        version=BEHAVIOR_TIMELINE_VERSION,
        action_segments=action_segments,
        motor_overlays=motor_overlays,
        robot_type=robot_type,
        active_motor_tracks=active_tracks,
    )
