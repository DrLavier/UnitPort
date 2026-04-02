#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared heuristic fallback for degraded topology grouping.

Used by both ``motor_weight_nav_model`` and ``timeline_view_model`` when
canonical topology (``motor_topology``) is unavailable.  Single source for
degraded-mode track-to-group heuristics so that the navigator and timeline
always agree when in degraded mode.

Do NOT add new consumers of this logic; use ``motor_topology`` for canonical paths.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Track name lists — shared constants
# ---------------------------------------------------------------------------

LEG_TRACKS_QUADRUPED: List[str] = ["leg_fl", "leg_fr", "leg_rl", "leg_rr"]
LEG_GROUPS: List[str]            = ["leg_group_left", "leg_group_right"]
BODY_TRACKS: List[str]           = ["body_posture", "waist", "torso_yaw", "torso_pitch"]
HEAD_TRACKS: List[str]           = ["head_pitch", "head_yaw"]

QUAD_JOINT_LEGS: List[str] = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
SPOT_LEGS: List[str] = [
    "FL_hip_x", "FL_hip_y", "FL_knee",
    "FR_hip_x", "FR_hip_y", "FR_knee",
    "RL_hip_x", "RL_hip_y", "RL_knee",
    "RR_hip_x", "RR_hip_y", "RR_knee",
]
HUMANOID_LEGS: List[str] = [
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
]
HUMANOID_ARMS: List[str] = [
    "L_shoulder_pitch", "L_shoulder_roll", "L_elbow",
    "R_shoulder_pitch", "R_shoulder_roll", "R_elbow",
]

# ---------------------------------------------------------------------------
# Fallback limb table — (limb_id, limb_label, track_names)
# ---------------------------------------------------------------------------

FALLBACK_LIMBS: List[Tuple[str, str, List[str]]] = [
    ("legs",          "Legs",             LEG_TRACKS_QUADRUPED),
    ("leg_groups",    "Leg Groups",       LEG_GROUPS),
    ("body",          "Body",             BODY_TRACKS),
    ("head",          "Head",             HEAD_TRACKS),
    ("fl_leg",        "Front Left Leg",   ["FL_hip", "FL_thigh", "FL_calf",
                                           "FL_hip_x", "FL_hip_y", "FL_knee"]),
    ("fr_leg",        "Front Right Leg",  ["FR_hip", "FR_thigh", "FR_calf",
                                           "FR_hip_x", "FR_hip_y", "FR_knee"]),
    ("rl_leg",        "Rear Left Leg",    ["RL_hip", "RL_thigh", "RL_calf",
                                           "RL_hip_x", "RL_hip_y", "RL_knee"]),
    ("rr_leg",        "Rear Right Leg",   ["RR_hip", "RR_thigh", "RR_calf",
                                           "RR_hip_x", "RR_hip_y", "RR_knee"]),
    ("left_leg",      "Left Leg",         list(HUMANOID_LEGS[:5])),
    ("right_leg",     "Right Leg",        list(HUMANOID_LEGS[5:])),
    ("left_arm",      "Left Arm",         list(HUMANOID_ARMS[:3])),
    ("right_arm",     "Right Arm",        list(HUMANOID_ARMS[3:])),
    ("torso",         "Torso",            ["torso_yaw", "torso_pitch"]),
]

# Fast lookup: track_name → (limb_id, limb_label); first-write wins.
_TRACK_TO_LIMB: Dict[str, Tuple[str, str]] = {}
for _lid, _llabel, _members in FALLBACK_LIMBS:
    for _t in _members:
        if _t not in _TRACK_TO_LIMB:
            _TRACK_TO_LIMB[_t] = (_lid, _llabel)

FALLBACK_REGION = "Other"
_FALLBACK_LIMB: Tuple[str, str] = ("other", "Other")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def heuristic_limb_for_track(track_name: str) -> Tuple[str, str]:
    """Return ``(limb_id, limb_label)`` via heuristic — degraded fallback only.

    Used by ``timeline_view_model`` for body-part grouping rows.
    """
    if track_name in _TRACK_TO_LIMB:
        return _TRACK_TO_LIMB[track_name]
    tn = track_name.lower()
    if tn.startswith("fl_"):
        return "fl_leg", "Front Left Leg"
    if tn.startswith("fr_"):
        return "fr_leg", "Front Right Leg"
    if tn.startswith("rl_"):
        return "rl_leg", "Rear Left Leg"
    if tn.startswith("rr_"):
        return "rr_leg", "Rear Right Leg"
    if tn.startswith("l_"):
        if any(k in tn for k in ["hip", "knee", "ankle"]):
            return "left_leg", "Left Leg"
        if any(k in tn for k in ["shoulder", "elbow"]):
            return "left_arm", "Left Arm"
    if tn.startswith("r_"):
        if any(k in tn for k in ["hip", "knee", "ankle"]):
            return "right_leg", "Right Leg"
        if any(k in tn for k in ["shoulder", "elbow"]):
            return "right_arm", "Right Arm"
    if any(k in tn for k in ["torso", "waist", "body"]):
        return "body", "Body"
    if "head" in tn:
        return "head", "Head"
    return _FALLBACK_LIMB


def heuristic_region_label_for_track(track_name: str) -> str:
    """Return region label string for track — degraded fallback only.

    Used by ``motor_weight_nav_model`` for region grouping.
    Returns the limb_label (e.g. ``"Front Left Leg"``) or ``"Other"``.
    """
    _, llabel = heuristic_limb_for_track(track_name)
    return llabel
