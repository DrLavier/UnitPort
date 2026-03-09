#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor Weight Navigator data model — Step 2.

Pure-Python; no Qt dependency.  Converts a BehaviorTimeline + IO-mapping
list into a flat hierarchical model that the MotorWeightNavigator widget
renders for quick correctness inspection.

Hierarchy
---------
NavRegion  ("Legs", "Body", "Head", …)
  NavMotor   (one motor track: leg_fl, body_posture, …)
    NavMovement  (one ActionSegment calling this motor: Walk, Stand, …)
      NavParam   (one param: amplitude, gain, …  with source badge)

Source status vocabulary
------------------------
"constant"   — value is provided inline in the MotorSegment params dict.
"external"   — an inbound HBIOMapping targets this motor/param combination.
"unresolved" — the movement uses this motor track but the param has no
               source; this indicates a routing gap that needs attention.

Public API
----------
build_navigator_model(timeline, io_mappings, available_tracks) → List[NavRegion]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
            NavSource.EXTERNAL:   "External",
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
    """One motor track with all its calling movements."""

    track_name: str
    track_label: str
    movements: List[NavMovement] = field(default_factory=list)
    # True when the motor exists in the model registry but no movement calls it.
    is_unused: bool = False

    @property
    def has_issue(self) -> bool:
        return any(m.has_issue for m in self.movements)


@dataclass
class NavRegion:
    """A body region grouping one or more motor tracks."""

    region_label: str
    motors: List[NavMotor] = field(default_factory=list)

    @property
    def has_issue(self) -> bool:
        return any(m.has_issue for m in self.motors)


# ---------------------------------------------------------------------------
# Region grouping — derived from UNITREE_MOTOR_TRACKS naming convention.
# Order matters: entries appear in the navigator in this order.
# ---------------------------------------------------------------------------

_REGION_GROUPS: List[Tuple[str, List[str]]] = [
    ("Legs",           ["leg_fl", "leg_fr", "leg_rl", "leg_rr"]),
    ("Leg Groups",     ["leg_group_left", "leg_group_right"]),
    ("Body",           ["body_posture"]),
    ("Head",           ["head_pitch"]),
]

# Fallback for tracks not covered by the predefined groups.
_FALLBACK_REGION = "Other"


def _region_for_track(track_name: str) -> str:
    for region_label, members in _REGION_GROUPS:
        if track_name in members:
            return region_label
    return _FALLBACK_REGION


# ---------------------------------------------------------------------------
# IO-mapping helpers
# ---------------------------------------------------------------------------

def _external_key_for_param(
    track_name: str,
    param_key: str,
    io_mappings: "List[Any]",  # List[HBIOMapping]
) -> Optional[str]:
    """Return the signal name if an inbound IO mapping targets track/param.

    Matching convention:
      HBIOMapping.target_param == "{track_name}.{param_key}"  (exact)
      OR  HBIOMapping.target_param starts with "{track_name}"  (prefix)

    Returns the signal name if found, else None.
    """
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
# Public API
# ---------------------------------------------------------------------------

def build_navigator_model(
    timeline: Optional[Any],                  # BehaviorTimeline | None
    io_mappings: Optional[List[Any]] = None,  # List[HBIOMapping]
    available_tracks: Optional[Dict[str, Any]] = None,  # track_name → MotorTrackDef
) -> List[NavRegion]:
    """Build the hierarchical navigator model from a BehaviorTimeline.

    Parameters
    ----------
    timeline         : BehaviorTimeline (may be None or empty).
    io_mappings      : List of HBIOMapping objects (may be None or empty).
    available_tracks : Full motor track catalog for the loaded robot model
                       (track_name → MotorTrackDef).  When provided, every
                       motor in the catalog is shown — motors not called by any
                       movement are rendered with ``is_unused=True``.
                       When None, falls back to showing only tracks that appear
                       in the timeline overlays (original behaviour).

    Returns
    -------
    List[NavRegion]  — always returned; empty list when no tracks can be shown.
    """
    if io_mappings is None:
        io_mappings = []

    # ------------------------------------------------------------------
    # Build action/overlay lookup tables (non-empty timeline only).
    # ------------------------------------------------------------------
    action_map: Dict[str, Any] = {}
    action_track_map: Dict[str, Dict[str, Any]] = {}  # aid → {track_name: MotorSeg}

    if timeline is not None and not getattr(timeline, "is_empty", lambda: True)():
        action_segments = getattr(timeline, "action_segments", []) or []
        motor_overlays  = getattr(timeline, "motor_overlays",  []) or []
        action_map = {seg.action_id: seg for seg in action_segments}
        for overlay in motor_overlays:
            aid = overlay.action_id
            action_track_map.setdefault(aid, {})
            for mseg in overlay.motor_segments:
                action_track_map[aid][mseg.track_name] = mseg

    # ------------------------------------------------------------------
    # Determine the master set of tracks to display.
    # ------------------------------------------------------------------
    if available_tracks:
        # Full catalog provided — always show every motor in the registry.
        master_tracks: Dict[str, Any] = dict(available_tracks)
    else:
        # Fallback: derive tracks from overlays only.
        # Lazy-import the built-in Unitree catalog for label/param_key lookups.
        try:
            from system.behavior.action_profile import UNITREE_MOTOR_TRACK_MAP
            _fallback_catalog: Dict[str, Any] = UNITREE_MOTOR_TRACK_MAP
        except Exception:
            _fallback_catalog = {}

        master_tracks = {}
        for aid_tracks in action_track_map.values():
            for tn in aid_tracks:
                if tn not in master_tracks:
                    master_tracks[tn] = _fallback_catalog.get(tn)

        if not master_tracks:
            return []

    # ------------------------------------------------------------------
    # Group tracks into regions.
    # ------------------------------------------------------------------
    region_dict: Dict[str, List[str]] = {}
    for track_name in master_tracks:
        region_label = _region_for_track(track_name)
        region_dict.setdefault(region_label, []).append(track_name)

    # Build ordered region list.
    ordered_regions: List[str] = []
    for region_label, _ in _REGION_GROUPS:
        if region_label in region_dict:
            ordered_regions.append(region_label)
    if _FALLBACK_REGION in region_dict:
        ordered_regions.append(_FALLBACK_REGION)

    # ------------------------------------------------------------------
    # Build NavRegion → NavMotor → NavMovement → NavParam tree.
    # ------------------------------------------------------------------
    nav_regions: List[NavRegion] = []

    for region_label in ordered_regions:
        track_names_in_region = region_dict[region_label]
        nav_motors: List[NavMotor] = []

        # Order motors within region by _REGION_GROUPS list order where possible.
        canonical_order = next(
            (members for rlabel, members in _REGION_GROUPS if rlabel == region_label),
            track_names_in_region,
        )
        ordered_tracks = [t for t in canonical_order if t in track_names_in_region]
        ordered_tracks += [t for t in track_names_in_region if t not in ordered_tracks]

        for track_name in ordered_tracks:
            track_def = master_tracks[track_name]
            track_label = (
                track_def.label if track_def and hasattr(track_def, "label")
                else track_name.replace("_", " ").title()
            )
            primary_param = (
                track_def.param_key if track_def and hasattr(track_def, "param_key")
                else "amplitude"
            )

            nav_movements: List[NavMovement] = []

            for action_id, action_seg in action_map.items():
                motor_seg = action_track_map.get(action_id, {}).get(track_name)
                movement_name = getattr(action_seg, "name", action_id) or action_id

                nav_params = _build_params_for_cell(
                    track_name=track_name,
                    primary_param=primary_param,
                    motor_seg=motor_seg,
                    io_mappings=io_mappings,
                )

                if not nav_params:
                    # Movement doesn't involve this motor — skip it.
                    continue

                nav_movements.append(NavMovement(
                    movement_name=movement_name,
                    action_id=action_id,
                    params=nav_params,
                ))

            # Always add motor — mark unused when no movement calls it.
            nav_motors.append(NavMotor(
                track_name=track_name,
                track_label=track_label,
                movements=nav_movements,
                is_unused=(len(nav_movements) == 0),
            ))

        if nav_motors:
            nav_regions.append(NavRegion(
                region_label=region_label,
                motors=nav_motors,
            ))

    return nav_regions


def _build_params_for_cell(
    track_name: str,
    primary_param: str,
    motor_seg: Optional[Any],  # MotorSegment | None
    io_mappings: List[Any],
) -> List[NavParam]:
    """Build NavParam list for a single movement × motor_track cell.

    When motor_seg is None the movement doesn't call this motor — return [].
    """
    if motor_seg is None:
        return []

    seg_params: Dict[str, Any] = getattr(motor_seg, "params", {}) or {}
    nav_params: List[NavParam] = []

    # Always show primary param first.
    param_keys_ordered = [primary_param] + [
        k for k in seg_params if k != primary_param
    ]
    # Also add start_time / duration as informational params.
    for extra in ("start_time", "duration"):
        if extra not in param_keys_ordered:
            param_keys_ordered.append(extra)

    for pk in param_keys_ordered:
        # Check IO mapping (external source) first — it overrides constant.
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

        # Check if param value exists in the segment.
        if pk in seg_params:
            nav_params.append(NavParam(
                param_key=pk,
                source=NavSource.CONSTANT,
                value=seg_params[pk],
                has_issue=False,
            ))
        elif pk in ("start_time", "duration"):
            # Structural params — use direct attrs if available.
            v = getattr(motor_seg, pk, None)
            if v is not None:
                nav_params.append(NavParam(
                    param_key=pk,
                    source=NavSource.CONSTANT,
                    value=v,
                    has_issue=False,
                ))
        else:
            # Primary param missing and no external binding → unresolved.
            if pk == primary_param:
                nav_params.append(NavParam(
                    param_key=pk,
                    source=NavSource.UNRESOLVED,
                    has_issue=True,
                ))

    return nav_params
