#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical semantic intent vocabulary — Circle 1 STEP 1.2.

This module defines the initial stable set of semantic intent IDs for
movement and posture actions that are present in the project today.

Intent ID Naming Rules
----------------------
Intent IDs follow the pattern ``{category}.{action}`` where:

- **category** is a high-level semantic group (ASCII lowercase):
    - ``posture``    — static body configuration (stand, sit, crouch…)
    - ``locomotion`` — directed movement (forward, backward, turn, halt…)
    - ``gesture``    — expressive or manipulation actions (wave, lift_right_leg…)

- **action** is a vendor-neutral semantic verb (ASCII lowercase, underscores
  allowed, no brand-specific terms).  Use the *behaviour* being requested,
  not the SDK call that implements it.

Stability contract:
    Once defined, an intent ID is **STABLE and IMMUTABLE** across versions.
    - Adding new intents is additive and safe.
    - Renaming or removing an intent ID is a **breaking change** and requires
      a deprecation cycle with a legacy-alias shim.

Scope:
    This module defines only intents for which at least one adapter has a
    confirmed executable mapping.  Speculative future intents must NOT be
    added here until a mapping exists.

Reserved categories for future expansion (do not use yet):
    - ``manipulation`` — gripper / arm gestures
    - ``navigation``   — waypoint / map-relative movement
    - ``interaction``  — social / human-facing behaviors
"""

from __future__ import annotations

from typing import Dict, List

from src.system.behavior.semantic_action import (
    ActionAvailability,
    SemanticActionDescriptor,
)


# ---------------------------------------------------------------------------
# Canonical intent ID constants
#
# Use these constants everywhere in behavior authoring, compatibility audits,
# and runtime resolution.  Do NOT compare against raw string literals like
# "posture.stand" — import and use the constant so renames are caught by
# the Python linter.
# ---------------------------------------------------------------------------

#: Robot is upright and stationary in a stable standing posture.
POSTURE_STAND = "posture.stand"

#: Robot lowers itself into a seated/resting posture.
POSTURE_SIT = "posture.sit"

#: Robot walks forward at a configurable speed.
#: Maps to "walk" in current adapter implementations.
LOCOMOTION_FORWARD = "locomotion.forward"

#: Robot decelerates to a full stop and holds position.
#: Maps to "stop" in current adapter implementations.
LOCOMOTION_HALT = "locomotion.halt"

#: Robot lifts its right front leg (Unitree Go2/Go2W gesture action).
#: Model-specific: not universally supported across all brands.
GESTURE_LIFT_RIGHT_LEG = "gesture.lift_right_leg"

# ---------------------------------------------------------------------------
# Complete canonical intent set
#
# ALL_CANONICAL_INTENTS is the exhaustive set of intent IDs defined in this
# module.  Use it for enumeration, catalog validation, and test assertions.
# ---------------------------------------------------------------------------

ALL_CANONICAL_INTENTS: frozenset = frozenset({
    POSTURE_STAND,
    POSTURE_SIT,
    LOCOMOTION_FORWARD,
    LOCOMOTION_HALT,
    GESTURE_LIFT_RIGHT_LEG,
})


# ---------------------------------------------------------------------------
# Base catalog
#
# Brand-independent default descriptors for each canonical intent.
# brand="" and robot_type="" means "applicable to all models by default".
#
# Adapter-specific modules (Circle 2) will extend these with brand-bound
# descriptors that carry the correct raw_action mapping and model-specific
# availability/parameter overrides.
# ---------------------------------------------------------------------------

_BASE_CATALOG: List[SemanticActionDescriptor] = [
    SemanticActionDescriptor(
        intent_id=POSTURE_STAND,
        display_name="Stand",
        category="posture",
        raw_action="stand",
        brand="",
        robot_type="",
        parameters={
            "duration": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 300.0,
                "description": "Duration in seconds",
            },
        },
        availability=ActionAvailability.AVAILABLE,
        tags=("posture", "safe"),
    ),
    SemanticActionDescriptor(
        intent_id=POSTURE_SIT,
        display_name="Sit",
        category="posture",
        raw_action="sit",
        brand="",
        robot_type="",
        parameters={
            "duration": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 300.0,
                "description": "Duration in seconds",
            },
        },
        availability=ActionAvailability.AVAILABLE,
        tags=("posture", "safe"),
    ),
    SemanticActionDescriptor(
        intent_id=LOCOMOTION_FORWARD,
        display_name="Move - Forward",
        category="locomotion",
        raw_action="walk",
        brand="",
        robot_type="",
        parameters={
            "speed": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Forward speed (m/s)",
            },
            "duration": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "description": "Duration in seconds",
            },
        },
        availability=ActionAvailability.AVAILABLE,
        tags=("locomotion",),
    ),
    SemanticActionDescriptor(
        intent_id=LOCOMOTION_HALT,
        display_name="Stop",
        category="locomotion",
        raw_action="stop",
        brand="",
        robot_type="",
        parameters={},
        availability=ActionAvailability.AVAILABLE,
        tags=("locomotion", "safe"),
    ),
    SemanticActionDescriptor(
        intent_id=GESTURE_LIFT_RIGHT_LEG,
        display_name="Lift Right Leg",
        category="gesture",
        raw_action="lift_right_leg",
        brand="",
        robot_type="",
        parameters={
            "duration": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 30.0,
                "description": "Duration in seconds",
            },
        },
        availability=ActionAvailability.AVAILABLE,
        tags=("gesture",),
    ),
]


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_base_catalog() -> List[SemanticActionDescriptor]:
    """Return a copy of the brand-independent base catalog.

    The returned list is a shallow copy; the descriptors themselves are
    immutable (frozen dataclasses).
    """
    return list(_BASE_CATALOG)


def get_base_catalog_by_intent() -> Dict[str, SemanticActionDescriptor]:
    """Return the base catalog indexed by ``intent_id``.

    Useful for O(1) lookup when resolving a single intent.
    """
    return {d.intent_id: d for d in _BASE_CATALOG}
