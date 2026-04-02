#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical motor topology — system/behavior/motor_topology.py

Single authoritative source of truth for body-part / limb / motor hierarchy
per robot model.  All Behavior-side consumers (navigator, timeline, Movement
Settings, compile, runtime) must resolve body-part and motor metadata through
this module instead of maintaining their own incompatible mapping tables.

Hierarchy
---------
ModelTopology  (brand + robot_type)
  RegionTopology   ("Legs", "Arms", "Torso", ...)
    LimbTopology   ("Front Left Leg", "Left Arm", ...)
      MotorEntry   (one motor / joint with param bounds)

Design constraints
------------------
- Pure Python; no Qt imports.
- No circular imports: lazily imports action_profile + motor_registry at
  build time, not at module level.
- Topology objects are built once and cached; rebuild only on cache_clear().
- Degraded mode is explicit: callers always receive a ModelTopology, never
  None.  is_degraded=True signals that no canonical data exists.
- Aliases (e.g. go2w == go2, h1 == g1) are resolved transparently.

Public API
----------
get_topology(brand, robot_type)             -> Optional[ModelTopology]
resolve_topology(brand, robot_type)         -> ModelTopology  (degraded if unknown)
is_topology_known(brand, robot_type)        -> bool
get_all_limbs(brand, robot_type)            -> List[LimbTopology]
get_limb_for_track(brand, robot_type, track_name) -> Optional[LimbTopology]
get_region_for_track(brand, robot_type, track_name) -> Optional[RegionTopology]
get_all_track_names(brand, robot_type)      -> List[str]
cache_clear()                               -> None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorEntry:
    """One physical motor / joint in the canonical topology.

    Fields
    ------
    motor_id   : Registry motor ID (e.g. "go2_fl_hip"); "" when unknown.
    joint_name : Joint name (e.g. "FL_hip"); falls back to track_name.
    track_name : Canonical authored timeline track key for this motor entry.
    authored_track_key : Durable lookup key used to resolve MotorSegment rows
                         in BehaviorTimeline.motor_overlays. Kept explicit so
                         UI display labels and authoring lookup are not
                         conflated.
    param_key  : Primary editable parameter (e.g. "amplitude", "position").
    safe_range : (min, max) for real hardware path.
    sim_range  : (min, max) for simulation path.
    """
    motor_id:   str
    joint_name: str
    track_name: str
    authored_track_key: str
    param_key:  str
    safe_range: Tuple[float, float]
    sim_range:  Tuple[float, float]


@dataclass
class LimbTopology:
    """One body-part / limb group within a region.

    Fields
    ------
    limb_id     : Stable identifier (e.g. "leg_fl", "l_arm", "torso").
    label       : Human-readable label (e.g. "Front Left Leg").
    track_names : Ordered timeline track names belonging to this limb.
    motors      : MotorEntry list; may span multiple track_names (e.g. Go2
                  abstract tracks) or one entry per joint (A1/humanoid).
    """
    limb_id:     str
    label:       str
    track_names: List[str] = field(default_factory=list)
    motors:      List[MotorEntry] = field(default_factory=list)

    @property
    def has_motors(self) -> bool:
        return bool(self.motors)

    @property
    def primary_track(self) -> str:
        """First track name, or limb_id if track_names is empty."""
        return self.track_names[0] if self.track_names else self.limb_id


@dataclass
class RegionTopology:
    """One body region containing one or more limbs.

    Fields
    ------
    region_id : Stable identifier (e.g. "legs", "front_legs", "torso").
    label     : Human-readable label (e.g. "Legs", "Torso").
    limbs     : Ordered list of limbs in this region.
    """
    region_id: str
    label:     str
    limbs:     List[LimbTopology] = field(default_factory=list)

    @property
    def all_track_names(self) -> List[str]:
        seen: set = set()
        result: List[str] = []
        for limb in self.limbs:
            for t in limb.track_names:
                if t not in seen:
                    seen.add(t)
                    result.append(t)
        return result


@dataclass
class ModelTopology:
    """Complete motor topology for one brand + robot_type combination.

    Fields
    ------
    brand           : Robot brand (e.g. "unitree", "boston_dynamics", "xiaomi").
    robot_type      : Robot model (e.g. "go2", "spot", "g1").
    regions         : Ordered list of body regions.
    is_degraded     : True when no canonical data exists; callers should show
                      an explicit degraded-state indicator in the UI.
    degraded_reason : Human-readable explanation (non-empty iff is_degraded).
    """
    brand:           str
    robot_type:      str
    regions:         List[RegionTopology] = field(default_factory=list)
    is_degraded:     bool = False
    degraded_reason: str = ""

    # Internal lookup caches — built lazily, not part of equality / repr
    _limb_by_track:   Optional[Dict[str, LimbTopology]]   = field(
        default=None, repr=False, compare=False)
    _region_by_track: Optional[Dict[str, RegionTopology]] = field(
        default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ props

    @property
    def all_limbs(self) -> List[LimbTopology]:
        result: List[LimbTopology] = []
        for region in self.regions:
            result.extend(region.limbs)
        return result

    @property
    def all_track_names(self) -> List[str]:
        seen: set = set()
        result: List[str] = []
        for limb in self.all_limbs:
            for t in limb.track_names:
                if t not in seen:
                    seen.add(t)
                    result.append(t)
        return result

    # ------------------------------------------------------------------ lookup

    def _ensure_track_cache(self) -> None:
        if self._limb_by_track is not None:
            return
        self._limb_by_track   = {}
        self._region_by_track = {}
        for region in self.regions:
            for limb in region.limbs:
                for t in limb.track_names:
                    self._limb_by_track[t]   = limb
                    self._region_by_track[t] = region

    def limb_for_track(self, track_name: str) -> Optional[LimbTopology]:
        """Return the LimbTopology that owns *track_name*, or None."""
        self._ensure_track_cache()
        assert self._limb_by_track is not None
        return self._limb_by_track.get(track_name)

    def region_for_track(self, track_name: str) -> Optional[RegionTopology]:
        """Return the RegionTopology that owns *track_name*, or None."""
        self._ensure_track_cache()
        assert self._region_by_track is not None
        return self._region_by_track.get(track_name)

    def is_track_known(self, track_name: str) -> bool:
        """Return True iff *track_name* appears in the canonical topology."""
        return self.limb_for_track(track_name) is not None


# ---------------------------------------------------------------------------
# Static topology specifications
# ---------------------------------------------------------------------------
# Format per model:
#   List[ (region_id, region_label,
#          [(limb_id, limb_label, [track_names]), ...]) ]
#
# Only canonical tracks are listed.  Legacy / alias tracks (leg_group_left,
# etc.) are intentionally omitted — they are handled by the degraded path.

_RawSpec = List[Tuple[str, str, List[Tuple[str, str, List[str]]]]]

# ---- Unitree Go2 / Go2W (abstract leg tracks, 3 motors per leg) ------------
_SPEC_GO2: _RawSpec = [
    ("legs", "Legs", [
        ("leg_fl", "Front Left Leg",  ["leg_fl"]),
        ("leg_fr", "Front Right Leg", ["leg_fr"]),
        ("leg_rl", "Rear Left Leg",   ["leg_rl"]),
        ("leg_rr", "Rear Right Leg",  ["leg_rr"]),
    ]),
    ("body", "Body", [
        ("body_posture", "Body Posture", ["body_posture"]),
    ]),
    ("head", "Head", [
        ("head_pitch", "Head Pitch", ["head_pitch"]),
    ]),
]

# ---- Unitree A1 (joint-level tracks: hip / thigh / calf per leg) -----------
_SPEC_A1: _RawSpec = [
    ("fl_leg", "Front Left Leg", [
        ("fl_leg", "Front Left Leg", ["FL_hip", "FL_thigh", "FL_calf"]),
    ]),
    ("fr_leg", "Front Right Leg", [
        ("fr_leg", "Front Right Leg", ["FR_hip", "FR_thigh", "FR_calf"]),
    ]),
    ("rl_leg", "Rear Left Leg", [
        ("rl_leg", "Rear Left Leg", ["RL_hip", "RL_thigh", "RL_calf"]),
    ]),
    ("rr_leg", "Rear Right Leg", [
        ("rr_leg", "Rear Right Leg", ["RR_hip", "RR_thigh", "RR_calf"]),
    ]),
    ("body", "Body", [
        ("body_posture", "Body Posture", ["body_posture"]),
    ]),
]

# ---- Unitree B1 / B2 (same as A1 plus waist joint) ------------------------
_SPEC_B1: _RawSpec = [
    ("fl_leg", "Front Left Leg", [
        ("fl_leg", "Front Left Leg", ["FL_hip", "FL_thigh", "FL_calf"]),
    ]),
    ("fr_leg", "Front Right Leg", [
        ("fr_leg", "Front Right Leg", ["FR_hip", "FR_thigh", "FR_calf"]),
    ]),
    ("rl_leg", "Rear Left Leg", [
        ("rl_leg", "Rear Left Leg", ["RL_hip", "RL_thigh", "RL_calf"]),
    ]),
    ("rr_leg", "Rear Right Leg", [
        ("rr_leg", "Rear Right Leg", ["RR_hip", "RR_thigh", "RR_calf"]),
    ]),
    ("body", "Body", [
        ("body_posture", "Body Posture", ["body_posture"]),
    ]),
    ("waist", "Waist", [
        ("waist", "Waist", ["waist"]),
    ]),
]

# ---- Unitree G1 / H1 / H1-2 (humanoid) ------------------------------------
_SPEC_G1: _RawSpec = [
    ("l_leg", "Left Leg", [
        ("l_leg", "Left Leg", [
            "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
        ]),
    ]),
    ("r_leg", "Right Leg", [
        ("r_leg", "Right Leg", [
            "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
        ]),
    ]),
    ("l_arm", "Left Arm", [
        ("l_arm", "Left Arm", [
            "L_shoulder_pitch", "L_shoulder_roll", "L_elbow",
        ]),
    ]),
    ("r_arm", "Right Arm", [
        ("r_arm", "Right Arm", [
            "R_shoulder_pitch", "R_shoulder_roll", "R_elbow",
        ]),
    ]),
    ("torso", "Torso", [
        ("torso", "Torso", ["torso_yaw", "torso_pitch"]),
    ]),
]

# ---- Boston Dynamics Spot (hip_x / hip_y / knee per leg) ------------------
_SPEC_SPOT: _RawSpec = [
    ("fl_leg", "Front Left Leg", [
        ("fl_leg", "Front Left Leg", ["FL_hip_x", "FL_hip_y", "FL_knee"]),
    ]),
    ("fr_leg", "Front Right Leg", [
        ("fr_leg", "Front Right Leg", ["FR_hip_x", "FR_hip_y", "FR_knee"]),
    ]),
    ("rl_leg", "Rear Left Leg", [
        ("rl_leg", "Rear Left Leg", ["RL_hip_x", "RL_hip_y", "RL_knee"]),
    ]),
    ("rr_leg", "Rear Right Leg", [
        ("rr_leg", "Rear Right Leg", ["RR_hip_x", "RR_hip_y", "RR_knee"]),
    ]),
    ("body", "Body", [
        ("body_posture", "Body Posture", ["body_posture"]),
    ]),
]

# ---- Xiaomi Cyberdog / Cyberdog2 (joint-level, same structure as A1) -------
_SPEC_CYBERDOG: _RawSpec = [
    ("fl_leg", "Front Left Leg", [
        ("fl_leg", "Front Left Leg", ["FL_hip", "FL_thigh", "FL_calf"]),
    ]),
    ("fr_leg", "Front Right Leg", [
        ("fr_leg", "Front Right Leg", ["FR_hip", "FR_thigh", "FR_calf"]),
    ]),
    ("rl_leg", "Rear Left Leg", [
        ("rl_leg", "Rear Left Leg", ["RL_hip", "RL_thigh", "RL_calf"]),
    ]),
    ("rr_leg", "Rear Right Leg", [
        ("rr_leg", "Rear Right Leg", ["RR_hip", "RR_thigh", "RR_calf"]),
    ]),
    ("body", "Body", [
        ("body_posture", "Body Posture", ["body_posture"]),
    ]),
]

# Primary spec registry — (brand, robot_type) -> _RawSpec
_TOPOLOGY_SPECS: Dict[Tuple[str, str], _RawSpec] = {
    ("unitree",         "go2"):       _SPEC_GO2,
    ("unitree",         "go2w"):      _SPEC_GO2,
    ("unitree",         "a1"):        _SPEC_A1,
    ("unitree",         "b1"):        _SPEC_B1,
    ("unitree",         "b2"):        _SPEC_B1,
    ("unitree",         "g1"):        _SPEC_G1,
    ("unitree",         "h1"):        _SPEC_G1,
    ("unitree",         "h1_2"):      _SPEC_G1,
    ("boston_dynamics", "spot"):      _SPEC_SPOT,
    ("xiaomi",          "cyberdog"):  _SPEC_CYBERDOG,
    ("xiaomi",          "cyberdog2"): _SPEC_CYBERDOG,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _lookup_key(brand: str, robot_type: str) -> Tuple[str, str]:
    return (_norm(brand), _norm(robot_type))


def _get_track_def(track_map: Dict[str, Any], track_name: str) -> Any:
    """Return MotorTrackDef for *track_name* or None."""
    return track_map.get(track_name)


def _motor_entry_for_track(
    track_name: str,
    track_map: Dict[str, Any],
    by_joint: Dict[str, Any],
    by_abstract: Dict[str, List[Any]],
) -> List[MotorEntry]:
    """Build MotorEntry list for one track_name.

    Matching priority:
    1. joint_name == track_name  (A1, G1, Cyberdog — one-to-one)
    2. abstract_track == track_name  (Go2 — one abstract → N physical motors)
    3. No hardware match — populate from MotorTrackDef only (param bounds known)
    """
    td = _get_track_def(track_map, track_name)
    param_key  = td.param_key  if td else "amplitude"
    safe_range = (td.safe_min, td.safe_max) if td else (0.0, 1.0)
    sim_range  = (td.sim_min,  td.sim_max)  if td else (0.0, 2.0)

    # Priority 1: joint-name exact match
    ms = by_joint.get(track_name)
    if ms is not None:
        return [MotorEntry(
            motor_id=ms.motor_id,
            joint_name=ms.joint_name,
            track_name=track_name,
            authored_track_key=track_name,
            param_key=param_key,
            safe_range=safe_range,
            sim_range=sim_range,
        )]

    # Priority 2: abstract-track match (may yield multiple motors)
    abstract_list = by_abstract.get(track_name, [])
    if abstract_list:
        return [
            MotorEntry(
                motor_id=ms2.motor_id,
                joint_name=ms2.joint_name,
                track_name=track_name,
                authored_track_key=track_name,
                param_key=param_key,
                safe_range=safe_range,
                sim_range=sim_range,
            )
            for ms2 in abstract_list
        ]

    # Priority 3: no hardware match — track data only
    return [MotorEntry(
        motor_id="",
        joint_name=track_name,
        track_name=track_name,
        authored_track_key=track_name,
        param_key=param_key,
        safe_range=safe_range,
        sim_range=sim_range,
    )]


def _build_topology(brand: str, robot_type: str) -> ModelTopology:
    """Build a ModelTopology from the static spec + live action_profile data."""
    key = _lookup_key(brand, robot_type)
    spec = _TOPOLOGY_SPECS.get(key)

    if spec is None:
        return ModelTopology(
            brand=brand,
            robot_type=robot_type,
            regions=[],
            is_degraded=True,
            degraded_reason=(
                f"Model topology unavailable: {brand}/{robot_type}. "
                "Showing fallback track grouping."
            ),
        )

    # --- Lazily load track map and motor registry --------------------------
    track_map: Dict[str, Any] = {}
    try:
        from src.system.behavior.action_profile import get_motor_track_map  # type: ignore
        tm = get_motor_track_map(robot_type, brand=brand)
        if tm:
            track_map = tm
    except Exception:
        pass

    by_joint:    Dict[str, Any]       = {}
    by_abstract: Dict[str, List[Any]] = {}
    try:
        from src.system.behavior.motor_registry import get_motors_for_robot  # type: ignore
        for ms in get_motors_for_robot(brand, robot_type):
            by_joint[ms.joint_name] = ms
            by_abstract.setdefault(ms.abstract_track, []).append(ms)
    except Exception:
        pass

    # --- Build topology objects from spec ----------------------------------
    regions: List[RegionTopology] = []

    for (region_id, region_label, limb_specs) in spec:
        limbs: List[LimbTopology] = []

        for (limb_id, limb_label, track_names) in limb_specs:
            motors: List[MotorEntry] = []
            for track_name in track_names:
                motors.extend(
                    _motor_entry_for_track(
                        track_name, track_map, by_joint, by_abstract)
                )

            limbs.append(LimbTopology(
                limb_id=limb_id,
                label=limb_label,
                track_names=list(track_names),
                motors=motors,
            ))

        regions.append(RegionTopology(
            region_id=region_id,
            label=region_label,
            limbs=limbs,
        ))

    return ModelTopology(
        brand=brand,
        robot_type=robot_type,
        regions=regions,
        is_degraded=False,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE: Dict[Tuple[str, str], ModelTopology] = {}


def cache_clear() -> None:
    """Flush the topology cache (useful after hot-reload or testing)."""
    _CACHE.clear()


def _get_cached(brand: str, robot_type: str) -> ModelTopology:
    key = _lookup_key(brand, robot_type)
    if key not in _CACHE:
        _CACHE[key] = _build_topology(brand, robot_type)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_topology_known(brand: str, robot_type: str) -> bool:
    """Return True iff a canonical topology spec exists for brand/robot_type."""
    return _lookup_key(brand, robot_type) in _TOPOLOGY_SPECS


def get_topology(brand: str, robot_type: str) -> Optional[ModelTopology]:
    """Return the canonical ModelTopology, or None if unknown.

    Prefer ``resolve_topology`` when you need a guaranteed non-None result.
    """
    if not is_topology_known(brand, robot_type):
        return None
    return _get_cached(brand, robot_type)


def resolve_topology(brand: str, robot_type: str) -> ModelTopology:
    """Return the canonical ModelTopology; always non-None.

    When no canonical spec exists, returns a degraded ModelTopology with
    ``is_degraded=True`` and an explanatory ``degraded_reason``.  Callers
    should surface this to the user rather than silently falling back.
    """
    return _get_cached(brand, robot_type)


def get_all_limbs(brand: str, robot_type: str) -> List[LimbTopology]:
    """Return all limbs for brand/robot_type in canonical body-part order."""
    return _get_cached(brand, robot_type).all_limbs


def get_limb_for_track(
    brand: str, robot_type: str, track_name: str
) -> Optional[LimbTopology]:
    """Return the LimbTopology that owns *track_name*, or None if unknown."""
    return _get_cached(brand, robot_type).limb_for_track(track_name)


def get_region_for_track(
    brand: str, robot_type: str, track_name: str
) -> Optional[RegionTopology]:
    """Return the RegionTopology that owns *track_name*, or None if unknown."""
    return _get_cached(brand, robot_type).region_for_track(track_name)


def get_all_track_names(brand: str, robot_type: str) -> List[str]:
    """Return all canonical track names in body-part order for brand/robot_type."""
    return _get_cached(brand, robot_type).all_track_names
