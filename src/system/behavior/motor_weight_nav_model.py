#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor Weight Navigator data model — Step 2 / Step 3 refactor.

Pure-Python; no Qt dependency.  Converts a BehaviorTimeline + IO-mapping
list into a hierarchical model that the MotorWeightNavigator widget renders.

Hierarchy (canonical path)
--------------------------
NavRegion   — body region    ("Legs", "Arms", "Torso", …)
  NavLimb   — body part/limb ("Front Left Leg", "Left Arm", …)
    NavMotor  — actual motor/joint (motor_id, joint_name from MotorEntry)
      NavMovement — movement calling this motor
        NavParam  — one param with source badge

Degraded / fallback path
------------------------
When brand/robot_type are absent or unknown the canonical topology is
unavailable.  In that case:
  - NavRegion.limbs is empty; NavRegion.motors is populated (heuristic grouping)
  - An explicit is_degraded=True notice region is prepended so the UI never
    silently pretends the grouping is authoritative.

Public API
----------
build_navigator_model(timeline, io_mappings, available_tracks,
                      brand, robot_type) → List[NavRegion]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.system.behavior.topology_fallback import heuristic_region_label_for_track  # shared degraded fallback


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class NavSource:
    """Source-status vocabulary constants."""
    CONSTANT   = "constant"
    EXTERNAL   = "external"
    UNRESOLVED = "unresolved"


@dataclass
class NavParam:
    """One motor parameter with its source badge."""

    param_key: str
    source: str                        # NavSource.*
    value: Any = None                  # numeric value for constant; None otherwise
    external_key: str = ""             # HBIOMapping.signal when source="external"
    has_issue: bool = False            # True when source == UNRESOLVED

    @property
    def display_value(self) -> str:
        if self.source == NavSource.CONSTANT and self.value is not None:
            if isinstance(self.value, float):
                return f"{self.value:.4g}"
            return str(self.value)
        if self.source == NavSource.EXTERNAL:
            return f"← {self.external_key}" if self.external_key else "← [external]"
        return "—"

    @property
    def source_badge(self) -> str:
        return {
            NavSource.CONSTANT:   "Constant",
            NavSource.EXTERNAL:   "Script",
            NavSource.UNRESOLVED: "Unresolved",
        }.get(self.source, self.source)


@dataclass
class NavMovement:
    """One action (movement) that calls a particular motor track."""

    movement_name: str
    action_id: str
    params: List[NavParam] = field(default_factory=list)

    @property
    def has_issue(self) -> bool:
        return any(p.has_issue for p in self.params)


@dataclass
class NavMotor:
    """One physical motor / joint with its calling movements.

    In the canonical path this corresponds to a single MotorEntry from
    LimbTopology.motors, carrying the actual hardware motor_id and joint_name.

    In the degraded path motor_id / joint_name may be empty; track_name and
    track_label are the only reliable fields.
    """

    track_name: str
    track_label: str
    lookup_track_name: str = ""
    motor_id:   str = ""               # hardware motor ID  (MotorEntry.motor_id)
    joint_name: str = ""               # joint name          (MotorEntry.joint_name)
    movements: List[NavMovement] = field(default_factory=list)
    is_unused: bool = False            # True when no movement calls this motor

    @property
    def display_label(self) -> str:
        """Label shown in the widget (joint name preferred, fallback to track)."""
        return self.joint_name or self.track_label or self.track_name

    @property
    def has_issue(self) -> bool:
        return any(m.has_issue for m in self.movements)


@dataclass
class NavLimb:
    """A body-part / limb grouping one or more NavMotors.

    Corresponds directly to LimbTopology in the canonical path.
    Only populated in canonical mode; absent in degraded mode.
    """

    limb_id:    str
    limb_label: str
    motors: List[NavMotor] = field(default_factory=list)

    @property
    def has_issue(self) -> bool:
        return any(m.has_issue for m in self.motors)


@dataclass
class NavRegion:
    """A body region grouping one or more limbs (canonical) or motors (degraded).

    Fields
    ------
    region_label : Human-readable label.
    limbs        : NavLimb list — populated in canonical mode only.
    motors       : NavMotor list — populated in degraded mode only (flat).
    is_degraded  : True when this region is a degraded-mode notice rather than
                   a real body-part group.  The UI renders it with a warning
                   style and no interactive children.
    """
    region_label: str
    limbs:  List[NavLimb]  = field(default_factory=list)
    motors: List[NavMotor] = field(default_factory=list)
    is_degraded: bool = False

    @property
    def has_issue(self) -> bool:
        return (
            any(l.has_issue for l in self.limbs) or
            any(m.has_issue for m in self.motors)
        )


# ---------------------------------------------------------------------------
# Heuristic region grouping — degraded fallback only.
# Delegated to topology_fallback (single source shared with timeline_view_model).
# ---------------------------------------------------------------------------

# heuristic_region_label_for_track imported at module top from topology_fallback.


# ---------------------------------------------------------------------------
# IO-mapping helpers
# ---------------------------------------------------------------------------

def _external_key_for_param(
    track_name: str,
    param_key: str,
    io_mappings: "List[Any]",
) -> Optional[str]:
    exact = f"{track_name}.{param_key}"
    prefix = f"{track_name}."
    for m in io_mappings:
        if getattr(m, "direction", "") != "inbound":
            continue
        tp = getattr(m, "target_param", "")
        if tp == exact or tp.startswith(prefix):
            return getattr(m, "signal", "")
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_timeline_maps(
    timeline: Optional[Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Extract action_map and action_track_map from a BehaviorTimeline."""
    action_map: Dict[str, Any] = {}
    action_track_map: Dict[str, Dict[str, Any]] = {}
    if timeline is not None and not getattr(timeline, "is_empty", lambda: True)():
        for seg in (getattr(timeline, "action_segments", []) or []):
            action_map[seg.action_id] = seg
        for overlay in (getattr(timeline, "motor_overlays", []) or []):
            aid = overlay.action_id
            action_track_map.setdefault(aid, {})
            for mseg in overlay.motor_segments:
                action_track_map[aid][mseg.track_name] = mseg
    return action_map, action_track_map


def _build_params_for_cell(
    track_name: str,
    primary_param: str,
    motor_seg: Optional[Any],
    io_mappings: List[Any],
) -> List[NavParam]:
    """Build NavParam list for a movement × track_name cell.

    Returns [] when motor_seg is None (movement does not call this track).
    """
    if motor_seg is None:
        return []

    seg_params: Dict[str, Any] = getattr(motor_seg, "params", {}) or {}
    nav_params: List[NavParam] = []

    param_keys_ordered = [primary_param] + [
        k for k in seg_params if k != primary_param
    ]
    for extra in ("start_time", "duration"):
        if extra not in param_keys_ordered:
            param_keys_ordered.append(extra)

    for pk in param_keys_ordered:
        ext_signal = _external_key_for_param(track_name, pk, io_mappings)
        if ext_signal is not None:
            nav_params.append(NavParam(
                param_key=pk,
                source=NavSource.EXTERNAL,
                value=seg_params.get(pk),
                external_key=ext_signal,
                has_issue=False,
            ))
            continue

        if pk in seg_params:
            nav_params.append(NavParam(
                param_key=pk,
                source=NavSource.CONSTANT,
                value=seg_params[pk],
                has_issue=False,
            ))
        elif pk in ("start_time", "duration"):
            v = getattr(motor_seg, pk, None)
            if v is not None:
                nav_params.append(NavParam(
                    param_key=pk,
                    source=NavSource.CONSTANT,
                    value=v,
                    has_issue=False,
                ))
        else:
            if pk == primary_param:
                nav_params.append(NavParam(
                    param_key=pk,
                    source=NavSource.UNRESOLVED,
                    has_issue=True,
                ))

    return nav_params


def _build_movements_for_track(
    track_name: str,
    primary_param: str,
    action_map: Dict[str, Any],
    action_track_map: Dict[str, Dict[str, Any]],
    io_mappings: List[Any],
) -> List[NavMovement]:
    """Build the NavMovement list for one track_name across all actions."""
    nav_movements: List[NavMovement] = []
    for action_id, action_seg in action_map.items():
        motor_seg = action_track_map.get(action_id, {}).get(track_name)
        movement_name = getattr(action_seg, "name", action_id) or action_id
        nav_params = _build_params_for_cell(
            track_name, primary_param, motor_seg, io_mappings
        )
        if not nav_params:
            continue
        nav_movements.append(NavMovement(
            movement_name=movement_name,
            action_id=action_id,
            params=nav_params,
        ))
    return nav_movements


def _nav_motor_from_entry(
    entry: Any,                          # MotorEntry from motor_topology
    available_tracks: Dict[str, Any],
    action_map: Dict[str, Any],
    action_track_map: Dict[str, Dict[str, Any]],
    io_mappings: List[Any],
) -> NavMotor:
    """Build a NavMotor from a canonical MotorEntry.

    Uses MotorEntry.param_key as the primary parameter; falls back to
    MotorTrackDef.param_key from available_tracks when available.
    """
    track_name = entry.track_name
    lookup_track_name = getattr(entry, "authored_track_key", track_name) or track_name
    track_def  = available_tracks.get(track_name)

    # Prefer MotorEntry param_key; fall back to track map; last resort "amplitude"
    primary_param = (
        getattr(entry, "param_key", None)
        or (track_def.param_key if track_def and hasattr(track_def, "param_key") else None)
        or "amplitude"
    )
    track_label = (
        track_def.label if track_def and hasattr(track_def, "label")
        else track_name.replace("_", " ").title()
    )

    nav_movements = _build_movements_for_track(
        lookup_track_name, primary_param, action_map, action_track_map, io_mappings
    )
    return NavMotor(
        track_name  = track_name,
        lookup_track_name = lookup_track_name,
        track_label = track_label,
        motor_id    = getattr(entry, "motor_id", ""),
        joint_name  = getattr(entry, "joint_name", ""),
        movements   = nav_movements,
        is_unused   = (len(nav_movements) == 0),
    )


def _nav_motor_for_track(
    track_name: str,
    available_tracks: Dict[str, Any],
    action_map: Dict[str, Any],
    action_track_map: Dict[str, Dict[str, Any]],
    io_mappings: List[Any],
) -> NavMotor:
    """Build a NavMotor for one track_name (degraded / stale path)."""
    track_def = available_tracks.get(track_name)
    track_label = (
        track_def.label if track_def and hasattr(track_def, "label")
        else track_name.replace("_", " ").title()
    )
    primary_param = (
        track_def.param_key if track_def and hasattr(track_def, "param_key")
        else "amplitude"
    )
    nav_movements = _build_movements_for_track(
        track_name, primary_param, action_map, action_track_map, io_mappings
    )
    return NavMotor(
        track_name  = track_name,
        lookup_track_name = track_name,
        track_label = track_label,
        movements   = nav_movements,
        is_unused   = (len(nav_movements) == 0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_navigator_model(
    timeline: Optional[Any],                   # BehaviorTimeline | None
    io_mappings: Optional[List[Any]] = None,   # List[HBIOMapping]
    available_tracks: Optional[Dict[str, Any]] = None,  # track_name → MotorTrackDef
    brand: str = "",
    robot_type: str = "",
) -> List[NavRegion]:
    """Build the hierarchical navigator model from a BehaviorTimeline.

    Parameters
    ----------
    timeline         : BehaviorTimeline (may be None or empty).
    io_mappings      : List of HBIOMapping objects (may be None or empty).
    available_tracks : Full motor track catalog for the active robot model
                       (track_name → MotorTrackDef).  When None the function
                       resolves it from action_profile.get_motor_track_map().
    brand            : Robot brand (e.g. "unitree", "boston_dynamics").
                       When provided together with robot_type, the canonical
                       motor_topology is used for region/limb/motor hierarchy.
    robot_type       : Robot type (e.g. "go2", "spot", "h1").

    Returns
    -------
    List[NavRegion]
        Canonical mode: NavRegion.limbs populated; each NavLimb.motors
        contains NavMotor instances built from MotorEntry (real joints).
        Degraded mode: NavRegion.motors populated (flat heuristic grouping);
        a leading is_degraded=True notice region is prepended.
    """
    if io_mappings is None:
        io_mappings = []

    # ------------------------------------------------------------------
    # Build timeline lookup tables.
    # ------------------------------------------------------------------
    action_map, action_track_map = _build_timeline_maps(timeline)

    # ------------------------------------------------------------------
    # Resolve canonical topology (Step 3).
    # ------------------------------------------------------------------
    topology = None
    use_canonical = False

    if brand or robot_type:
        try:
            from src.system.behavior.motor_topology import resolve_topology  # type: ignore
            topology = resolve_topology(brand or "unitree", robot_type or "")
            use_canonical = not topology.is_degraded
        except Exception:
            topology = None
            use_canonical = False

    # ------------------------------------------------------------------
    # Resolve master track catalog.
    # ------------------------------------------------------------------
    if available_tracks is None:
        try:
            from src.system.behavior.action_profile import get_motor_track_map  # type: ignore
            available_tracks = get_motor_track_map(robot_type, brand=brand) or {}
        except Exception:
            available_tracks = {}

        # No Unitree fallback — using Unitree metadata for a non-Unitree model
        # would surface wrong labels and param keys.  The degraded path handles
        # the empty-available_tracks case explicitly below.

    # ------------------------------------------------------------------
    # CANONICAL PATH — topology is known and authoritative.
    #
    # Hierarchy: NavRegion → NavLimb → NavMotor (MotorEntry) → NavMovement
    # ------------------------------------------------------------------
    if use_canonical and topology is not None:
        nav_regions: List[NavRegion] = []
        tracks_placed: set = set()

        for region in topology.regions:
            nav_limbs: List[NavLimb] = []
            for limb in region.limbs:
                nav_motors: List[NavMotor] = []
                for entry in limb.motors:
                    # Each MotorEntry knows its track_name.
                    tracks_placed.add(getattr(entry, "authored_track_key", entry.track_name))
                    nav_motors.append(_nav_motor_from_entry(
                        entry, available_tracks,
                        action_map, action_track_map, io_mappings,
                    ))

                if nav_motors:
                    nav_limbs.append(NavLimb(
                        limb_id    = limb.limb_id,
                        limb_label = limb.label,
                        motors     = nav_motors,
                    ))

            if nav_limbs:
                nav_regions.append(NavRegion(
                    region_label = region.label,
                    limbs        = nav_limbs,
                ))

        # Stale tracks: present in timeline but absent from canonical topology.
        stale_motors: List[NavMotor] = []
        seen_stale: set = set()
        for aid_tracks in action_track_map.values():
            for track_name in aid_tracks:
                if track_name not in tracks_placed and track_name not in seen_stale:
                    seen_stale.add(track_name)
                    motor = _nav_motor_for_track(
                        track_name, available_tracks,
                        action_map, action_track_map, io_mappings,
                    )
                    if not motor.is_unused:
                        stale_motors.append(motor)

        if stale_motors:
            stale_limb = NavLimb(
                limb_id    = "stale",
                limb_label = "Stale Tracks",
                motors     = stale_motors,
            )
            nav_regions.append(NavRegion(
                region_label = "Unknown / Stale Tracks",
                limbs        = [stale_limb],
            ))

        return nav_regions

    # ------------------------------------------------------------------
    # DEGRADED/FALLBACK PATH — heuristic region grouping.
    # Inserts an explicit notice so the UI never silently pretends
    # this grouping is authoritative.
    # ------------------------------------------------------------------
    nav_regions_fallback: List[NavRegion] = []

    # Determine degraded reason for the notice.
    if topology is not None and topology.is_degraded:
        degraded_reason = topology.degraded_reason
    elif not brand and not robot_type:
        degraded_reason = "No robot model selected — showing available tracks"
    else:
        degraded_reason = (
            f"Model topology unavailable for {brand}/{robot_type} — "
            "showing fallback track grouping"
        )

    nav_regions_fallback.append(NavRegion(
        region_label = degraded_reason,
        motors       = [],
        is_degraded  = True,
    ))

    # Determine master track set (same as legacy logic).
    if available_tracks:
        master_tracks: Dict[str, Any] = dict(available_tracks)
    else:
        master_tracks = {}
        for aid_tracks in action_track_map.values():
            for tn in aid_tracks:
                if tn not in master_tracks:
                    master_tracks[tn] = None
        # Also surface tracks the timeline reports as active (even if no overlays exist yet).
        for tn in getattr(timeline, 'active_motor_tracks', None) or []:
            if tn not in master_tracks:
                master_tracks[tn] = None
        if not master_tracks:
            return nav_regions_fallback

    # Group tracks into heuristic regions.
    region_dict: Dict[str, List[str]] = {}
    for track_name in master_tracks:
        region_label = heuristic_region_label_for_track(track_name)
        region_dict.setdefault(region_label, []).append(track_name)

    from src.system.behavior.topology_fallback import FALLBACK_LIMBS, FALLBACK_REGION
    ordered_regions: List[str] = []
    for _, region_label, _ in FALLBACK_LIMBS:
        if region_label in region_dict and region_label not in ordered_regions:
            ordered_regions.append(region_label)
    if FALLBACK_REGION in region_dict:
        ordered_regions.append(FALLBACK_REGION)

    for region_label in ordered_regions:
        track_names_in_region = region_dict[region_label]
        canonical_order = next(
            (members for _, rlabel, members in FALLBACK_LIMBS if rlabel == region_label),
            track_names_in_region,
        )
        ordered_tracks = [t for t in canonical_order if t in track_names_in_region]
        ordered_tracks += [t for t in track_names_in_region if t not in ordered_tracks]

        nav_motors_fb: List[NavMotor] = []
        for track_name in ordered_tracks:
            nav_motors_fb.append(_nav_motor_for_track(
                track_name, master_tracks,
                action_map, action_track_map, io_mappings,
            ))

        if nav_motors_fb:
            nav_regions_fallback.append(NavRegion(
                region_label = region_label,
                motors       = nav_motors_fb,
            ))

    return nav_regions_fallback
