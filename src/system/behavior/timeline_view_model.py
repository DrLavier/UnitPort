#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior Timeline View Model — Step 4.

Pure-Python; no Qt dependency.

Provides a structured view model for the Behavior Timeline so the rendering
layer can present:

    Action axis
      └─ Body-part sub-axes  (from canonical topology)
           └─ Motor rows      (one per joint/motor, with authored MotorSegment)

This module translates a ``BehaviorTimeline`` (raw data) + a canonical
``ModelTopology`` into a hierarchy the timeline widget renders directly.

Hierarchy
---------
TimelineViewModel
  ActionRow          — one per ActionSegment on the action axis
    BodyPartRow      — one per LimbTopology (canonical ordering)
      MotorRow       — one per MotorEntry inside the limb

Selection contract
------------------
TimelineViewModel carries a ``TimelineSelection`` that can be at one of
four levels:

  NONE       — nothing selected
  ACTION     — one ActionRow selected (action_id)
  BODY_PART  — one BodyPartRow selected within an action (action_id + limb_id)
  MOTOR      — one MotorRow selected within a body part (action_id + limb_id + track_name)

Selection mutations are pure (no side-effects); callers apply the returned
new model or directly mutate ``TimelineViewModel.selection``.

Degraded mode
-------------
When brand/robot_type are unknown the canonical topology is unavailable.
  - ``TimelineViewModel.is_degraded = True``
  - Body-part rows fall back to heuristic grouping with degraded labels
  - The widget should show a visible notice

Public API
----------
build_timeline_view_model(timeline, brand, robot_type, available_tracks)
    -> TimelineViewModel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.system.behavior.topology_fallback import heuristic_limb_for_track  # shared degraded fallback


# ---------------------------------------------------------------------------
# Selection model
# ---------------------------------------------------------------------------

class SelectionLevel(Enum):
    NONE       = "none"
    ACTION     = "action"
    BODY_PART  = "body_part"
    MOTOR      = "motor"


@dataclass
class TimelineSelection:
    """Immutable-ish selection state for the timeline view.

    All fields are optional; set only the fields that identify the
    selection level.  Higher-level fields are always set alongside
    lower-level ones (e.g. ``limb_id`` is always set together with
    ``action_id``).
    """
    level:      SelectionLevel = SelectionLevel.NONE
    action_id:  Optional[str]  = None   # set at ACTION / BODY_PART / MOTOR
    limb_id:    Optional[str]  = None   # set at BODY_PART / MOTOR
    track_name: Optional[str]  = None   # set at MOTOR
    motor_id:   Optional[str]  = None   # set at MOTOR — MotorSegment.motor_id (UUID)

    # --- factories ---

    @classmethod
    def none(cls) -> "TimelineSelection":
        return cls(level=SelectionLevel.NONE)

    @classmethod
    def action(cls, action_id: str) -> "TimelineSelection":
        return cls(level=SelectionLevel.ACTION, action_id=action_id)

    @classmethod
    def body_part(cls, action_id: str, limb_id: str) -> "TimelineSelection":
        return cls(level=SelectionLevel.BODY_PART,
                   action_id=action_id, limb_id=limb_id)

    @classmethod
    def motor(
        cls,
        action_id: str,
        limb_id: str,
        track_name: str,
        motor_id: Optional[str] = None,
    ) -> "TimelineSelection":
        return cls(
            level=SelectionLevel.MOTOR,
            action_id=action_id,
            limb_id=limb_id,
            track_name=track_name,
            motor_id=motor_id,
        )

    # --- helpers ---

    def matches_action(self, action_id: str) -> bool:
        return self.action_id == action_id and self.level != SelectionLevel.NONE

    def matches_body_part(self, action_id: str, limb_id: str) -> bool:
        return (self.action_id == action_id and self.limb_id == limb_id
                and self.level in (SelectionLevel.BODY_PART, SelectionLevel.MOTOR))

    def matches_motor(self, action_id: str, limb_id: str, track_name: str) -> bool:
        return (self.action_id == action_id and self.limb_id == limb_id
                and self.track_name == track_name
                and self.level == SelectionLevel.MOTOR)


# ---------------------------------------------------------------------------
# View model nodes
# ---------------------------------------------------------------------------

@dataclass
class MotorRow:
    """One motor / joint row under a body-part row.

    Fields
    ------
    track_name  : Timeline/UI row identifier.
    lookup_track_name : Authored MotorSegment lookup key in BehaviorTimeline.
    motor_id    : Hardware motor ID from MotorEntry; "" when unavailable.
    joint_name  : Joint name from MotorEntry; "" when unavailable.
    param_key   : Primary editable parameter for this motor.
    track_label : Human-readable track/joint label.
    segment     : Authored MotorSegment for the parent action; None if not yet
                  authored.
    is_authored : True when ``segment`` is not None.
    is_stale    : True when this motor comes from a stale / unrecognised track
                  (not in canonical topology).
    """
    track_name:  str
    motor_id:    str
    joint_name:  str
    param_key:   str
    track_label: str
    lookup_track_name: str = ""
    segment:     Optional[Any] = None    # MotorSegment | None
    is_authored: bool = False
    is_stale:    bool = False

    @property
    def display_label(self) -> str:
        """Preferred label for UI: joint_name > track_label > track_name."""
        return self.joint_name or self.track_label or self.track_name

    @property
    def safe_range(self) -> Optional[Tuple[float, float]]:
        return getattr(self, "_safe_range", None)

    @property
    def sim_range(self) -> Optional[Tuple[float, float]]:
        return getattr(self, "_sim_range", None)


@dataclass
class BodyPartRow:
    """One body-part / limb row under an action row.

    Fields
    ------
    limb_id        : Stable limb identifier from LimbTopology.
    limb_label     : Human-readable label (e.g. "Front Left Leg").
    motors         : Ordered list of MotorRows within this limb.
    is_expanded    : Whether motor rows are visible (UI expand/collapse state).
    is_stale_group : True when this row holds stale/unrecognised motors.
    """
    limb_id:        str
    limb_label:     str
    motors:         List[MotorRow] = field(default_factory=list)
    is_expanded:    bool = True
    is_stale_group: bool = False

    @property
    def has_authored_segments(self) -> bool:
        return any(m.is_authored for m in self.motors)

    @property
    def authored_count(self) -> int:
        return sum(1 for m in self.motors if m.is_authored)

    @property
    def motor_count(self) -> int:
        return len(self.motors)


@dataclass
class ActionRow:
    """One action row on the main action axis.

    Fields
    ------
    action_id   : ActionSegment.action_id.
    action_name : Display name (ActionSegment.name).
    intent_id   : Durable semantic ID (ActionSegment.intent_id).
    start_time  : Start offset in seconds.
    duration    : Duration in seconds.
    body_parts  : Ordered BodyPartRows, one per canonical limb.
    is_expanded : Whether body-part sub-rows are visible.
    """
    action_id:   str
    action_name: str
    intent_id:   str
    start_time:  float
    duration:    float
    body_parts:  List[BodyPartRow] = field(default_factory=list)
    is_expanded: bool = True

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def total_authored_segments(self) -> int:
        return sum(bp.authored_count for bp in self.body_parts)

    def get_body_part(self, limb_id: str) -> Optional[BodyPartRow]:
        for bp in self.body_parts:
            if bp.limb_id == limb_id:
                return bp
        return None

    def get_motor_row(self, limb_id: str, track_name: str) -> Optional[MotorRow]:
        bp = self.get_body_part(limb_id)
        if bp is None:
            return None
        for m in bp.motors:
            if m.track_name == track_name:
                return m
        return None


@dataclass
class TimelineViewModel:
    """Full view model for the Behavior Timeline.

    Holds the structured action/body-part/motor hierarchy and a
    ``TimelineSelection`` tracking what is currently selected.

    Fields
    ------
    brand           : Active robot brand.
    robot_type      : Active robot type.
    actions         : Ordered ActionRows (action axis).
    is_degraded     : True when canonical topology is unavailable.
    degraded_reason : Human-readable reason for degraded state.
    selection       : Current selection state.
    """
    brand:          str
    robot_type:     str
    actions:        List[ActionRow] = field(default_factory=list)
    is_degraded:    bool = False
    degraded_reason: str = ""
    selection:      TimelineSelection = field(
        default_factory=TimelineSelection.none
    )

    # --- timeline-level properties ---

    @property
    def total_duration(self) -> float:
        if not self.actions:
            return 0.0
        return max(a.end_time for a in self.actions)

    @property
    def is_empty(self) -> bool:
        return len(self.actions) == 0

    @property
    def layout_body_parts(self) -> "List[BodyPartRow]":
        """Stable body-part track layout for timeline rendering.

        Returns the union of body-part rows across all actions, preserving
        canonical limb ordering from the first action encountered.  This is
        the global track structure the timeline widget renders as fixed
        vertical tracks — independent of which action is currently selected.

        Replaces the previous ``vm.actions[0].body_parts`` anti-pattern so
        that multi-action timelines with different motor coverage still expose
        the full body-part structure as a single coherent layout.
        """
        if not self.actions:
            return []
        seen: Dict[str, "BodyPartRow"] = {}
        for ar in self.actions:
            for bp in ar.body_parts:
                if bp.limb_id not in seen:
                    seen[bp.limb_id] = bp
        return list(seen.values())

    # --- action lookup ---

    def get_action(self, action_id: str) -> Optional[ActionRow]:
        for a in self.actions:
            if a.action_id == action_id:
                return a
        return None

    # --- selection mutators (return new selection; do NOT mutate in place) ---

    def select_action(self, action_id: str) -> None:
        """Set selection to ACTION level for the given action_id."""
        self.selection = TimelineSelection.action(action_id)

    def select_body_part(self, action_id: str, limb_id: str) -> None:
        """Set selection to BODY_PART level."""
        self.selection = TimelineSelection.body_part(action_id, limb_id)

    def select_motor(
        self,
        action_id: str,
        limb_id: str,
        track_name: str,
        motor_id: Optional[str] = None,
    ) -> None:
        """Set selection to MOTOR level."""
        self.selection = TimelineSelection.motor(action_id, limb_id, track_name, motor_id=motor_id)

    def clear_selection(self) -> None:
        self.selection = TimelineSelection.none()

    # --- convenience queries ---

    def selected_action_row(self) -> Optional[ActionRow]:
        if self.selection.action_id:
            return self.get_action(self.selection.action_id)
        return None

    def selected_body_part_row(self) -> Optional[BodyPartRow]:
        ar = self.selected_action_row()
        if ar and self.selection.limb_id:
            return ar.get_body_part(self.selection.limb_id)
        return None

    def selected_motor_row(self) -> Optional[MotorRow]:
        bp = self.selected_body_part_row()
        if bp and self.selection.track_name:
            for m in bp.motors:
                if m.track_name == self.selection.track_name:
                    return m
        return None


# ---------------------------------------------------------------------------
# Heuristic region grouping — fallback only.
# Delegated to topology_fallback (single source shared with motor_weight_nav_model).
# ---------------------------------------------------------------------------

# _heuristic_limb_for_track imported at module top from topology_fallback.


# ---------------------------------------------------------------------------
# Internal build helpers
# ---------------------------------------------------------------------------

def _motor_row_from_entry(
    entry: Any,                             # MotorEntry
    overlay_seg_map: Dict[str, Any],        # track_name -> MotorSegment
    available_tracks: Dict[str, Any],       # track_name -> MotorTrackDef
) -> MotorRow:
    """Build a MotorRow from a canonical MotorEntry."""
    track_name = entry.track_name
    lookup_track_name = getattr(entry, "authored_track_key", track_name) or track_name
    track_def  = available_tracks.get(track_name)
    track_label = (
        track_def.label if track_def and hasattr(track_def, "label")
        else track_name.replace("_", " ").title()
    )
    param_key = (
        getattr(entry, "param_key", None)
        or (track_def.param_key if track_def and hasattr(track_def, "param_key") else None)
        or "amplitude"
    )
    seg = overlay_seg_map.get(lookup_track_name)
    row = MotorRow(
        track_name  = track_name,
        lookup_track_name = lookup_track_name,
        motor_id    = getattr(entry, "motor_id", ""),
        joint_name  = getattr(entry, "joint_name", ""),
        param_key   = param_key,
        track_label = track_label,
        segment     = seg,
        is_authored = seg is not None,
    )
    # Attach range metadata (referenced by settings panel)
    row._safe_range = getattr(entry, "safe_range", None)
    row._sim_range  = getattr(entry, "sim_range",  None)
    return row


def _motor_row_from_track(
    track_name: str,
    overlay_seg_map: Dict[str, Any],
    available_tracks: Dict[str, Any],
    is_stale: bool = False,
) -> MotorRow:
    """Build a MotorRow from a plain track name (degraded / stale path)."""
    track_def = available_tracks.get(track_name)
    track_label = (
        track_def.label if track_def and hasattr(track_def, "label")
        else track_name.replace("_", " ").title()
    )
    param_key = (
        track_def.param_key if track_def and hasattr(track_def, "param_key")
        else "amplitude"
    )
    seg = overlay_seg_map.get(track_name)
    return MotorRow(
        track_name  = track_name,
        lookup_track_name = track_name,
        motor_id    = "",
        joint_name  = "",
        param_key   = param_key,
        track_label = track_label,
        segment     = seg,
        is_authored = seg is not None,
        is_stale    = is_stale,
    )


def _overlay_seg_map(timeline: Any, action_id: str) -> Dict[str, Any]:
    """Build track_name -> MotorSegment map for a given action_id."""
    result: Dict[str, Any] = {}
    for overlay in (getattr(timeline, "motor_overlays", []) or []):
        if getattr(overlay, "action_id", None) == action_id:
            for seg in (getattr(overlay, "motor_segments", []) or []):
                tn = getattr(seg, "track_name", "")
                if tn:
                    result[tn] = seg
    return result


def _build_body_parts_canonical(
    topology: Any,                           # ModelTopology
    overlay_seg_map: Dict[str, Any],
    available_tracks: Dict[str, Any],
) -> Tuple[List[BodyPartRow], set]:
    """Build BodyPartRows from canonical topology. Returns (rows, tracks_placed)."""
    body_parts: List[BodyPartRow] = []
    tracks_placed: set = set()

    for region in topology.regions:
        for limb in region.limbs:
            motors: List[MotorRow] = []
            for entry in limb.motors:
                tracks_placed.add(getattr(entry, "authored_track_key", entry.track_name))
                motors.append(_motor_row_from_entry(
                    entry, overlay_seg_map, available_tracks
                ))
            if motors:
                body_parts.append(BodyPartRow(
                    limb_id    = limb.limb_id,
                    limb_label = limb.label,
                    motors     = motors,
                ))

    return body_parts, tracks_placed


def _build_body_parts_degraded(
    available_tracks: Dict[str, Any],
    overlay_seg_map: Dict[str, Any],
) -> List[BodyPartRow]:
    """Build BodyPartRows via heuristic grouping — degraded fallback."""
    # Group tracks by heuristic limb
    limb_tracks: Dict[str, Tuple[str, str, List[str]]] = {}  # limb_id -> (id, label, tracks)
    heuristic_order: List[str] = []

    for track_name in available_tracks:
        lid, llabel = _heuristic_limb_for_track(track_name)
        if lid not in limb_tracks:
            limb_tracks[lid] = (lid, llabel, [])
            heuristic_order.append(lid)
        limb_tracks[lid][2].append(track_name)

    body_parts: List[BodyPartRow] = []
    for lid in heuristic_order:
        _, llabel, tracks = limb_tracks[lid]
        motors = [
            _motor_row_from_track(t, overlay_seg_map, available_tracks)
            for t in tracks
        ]
        if motors:
            body_parts.append(BodyPartRow(
                limb_id    = lid,
                limb_label = llabel,
                motors     = motors,
            ))
    return body_parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_timeline_view_model(
    timeline: Optional[Any],               # BehaviorTimeline | None
    brand: str = "",
    robot_type: str = "",
    available_tracks: Optional[Dict[str, Any]] = None,  # track_name -> MotorTrackDef
) -> TimelineViewModel:
    """Build a TimelineViewModel from a BehaviorTimeline.

    Parameters
    ----------
    timeline         : BehaviorTimeline (may be None or empty).
    brand            : Robot brand (e.g. "unitree", "boston_dynamics").
    robot_type       : Robot type (e.g. "go2", "spot", "h1").
    available_tracks : Full motor track catalog (track_name -> MotorTrackDef).
                       Resolved from action_profile when None.

    Returns
    -------
    TimelineViewModel
        Canonical mode: ActionRow.body_parts driven by motor_topology limbs.
        Degraded mode:  ActionRow.body_parts grouped heuristically;
                        TimelineViewModel.is_degraded = True.
    """
    # ------------------------------------------------------------------
    # Resolve canonical topology.
    # ------------------------------------------------------------------
    topology = None
    use_canonical = False

    if brand or robot_type:
        try:
            from src.system.behavior.motor_topology import resolve_topology  # type: ignore
            topology = resolve_topology(brand or "unitree", robot_type or "")
            use_canonical = not topology.is_degraded
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Resolve track catalog.
    # ------------------------------------------------------------------
    if available_tracks is None:
        try:
            from src.system.behavior.action_profile import get_motor_track_map  # type: ignore
            available_tracks = get_motor_track_map(robot_type, brand=brand) or {}
        except Exception:
            available_tracks = {}
        # No Unitree fallback — using Unitree metadata for a non-Unitree model
        # would surface wrong labels.  The degraded path handles the empty
        # available_tracks case explicitly below.

    # ------------------------------------------------------------------
    # Degraded reason.
    # ------------------------------------------------------------------
    is_degraded = not use_canonical
    if is_degraded:
        if topology is not None and topology.is_degraded:
            degraded_reason = topology.degraded_reason
        elif not brand and not robot_type:
            degraded_reason = "No robot model selected"
        else:
            degraded_reason = (
                f"Topology unavailable for {brand}/{robot_type} — using fallback grouping"
            )
    else:
        degraded_reason = ""

    # ------------------------------------------------------------------
    # Empty / None timeline.
    # ------------------------------------------------------------------
    if timeline is None or getattr(timeline, "is_empty", lambda: True)():
        return TimelineViewModel(
            brand           = brand,
            robot_type      = robot_type,
            actions         = [],
            is_degraded     = is_degraded,
            degraded_reason = degraded_reason,
        )

    # ------------------------------------------------------------------
    # Build one ActionRow per ActionSegment.
    # ------------------------------------------------------------------
    action_rows: List[ActionRow] = []
    action_segments = getattr(timeline, "action_segments", []) or []

    for seg in action_segments:
        action_id = getattr(seg, "action_id", "")
        seg_map = _overlay_seg_map(timeline, action_id)

        if use_canonical and topology is not None:
            body_parts, tracks_placed = _build_body_parts_canonical(
                topology, seg_map, available_tracks
            )
            # Stale tracks: in overlay but not in canonical topology
            stale_motors: List[MotorRow] = []
            for track_name in seg_map:
                if track_name not in tracks_placed:
                    stale_motors.append(_motor_row_from_track(
                        track_name, seg_map, available_tracks, is_stale=True
                    ))
            if stale_motors:
                body_parts.append(BodyPartRow(
                    limb_id        = "stale",
                    limb_label     = "Unknown / Stale Tracks",
                    motors         = stale_motors,
                    is_stale_group = True,
                ))
        else:
            body_parts = _build_body_parts_degraded(available_tracks, seg_map)

        action_rows.append(ActionRow(
            action_id   = action_id,
            action_name = getattr(seg, "name", action_id) or action_id,
            intent_id   = getattr(seg, "intent_id", "") or "",
            start_time  = float(getattr(seg, "start_time", 0.0)),
            duration    = float(getattr(seg, "duration", 1.0)),
            body_parts  = body_parts,
        ))

    return TimelineViewModel(
        brand           = brand,
        robot_type      = robot_type,
        actions         = action_rows,
        is_degraded     = is_degraded,
        degraded_reason = degraded_reason,
    )
