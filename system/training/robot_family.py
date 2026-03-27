from __future__ import annotations

from typing import Dict


GENERIC_LOCOMOTION_FAMILY = "generic_locomotion"


# Project-level robot family grouping used by Training Ground template logic.
# Keep this mapping focused on stable robot_type identifiers already present
# in the codebase and fall back safely for anything unknown.
ROBOT_FAMILY_BY_TYPE: Dict[str, str] = {
    "go1": "quadruped",
    "go2": "quadruped",
    "a1": "quadruped",
    "b1": "quadruped",
    "b2": "quadruped",
    "spot": "quadruped",
    "go2w": "wheeled",
    "b2w": "wheeled",
    "g1": "biped",
    "h1": "biped",
    "h1_2": "biped",
    "humanoid": "biped",
}


def normalize_robot_type(robot_type: str) -> str:
    """Return a stable, lower-cased robot_type key for family lookup."""
    return str(robot_type or "").strip().lower()


def resolve_robot_family(robot_type: str) -> str:
    """
    Resolve a project robot_type into a template family key.

    Unknown or empty robot types must never fail; they fall back to the
    generic locomotion family so later template resolution remains safe.
    """
    normalized = normalize_robot_type(robot_type)
    return ROBOT_FAMILY_BY_TYPE.get(normalized, GENERIC_LOCOMOTION_FAMILY)

