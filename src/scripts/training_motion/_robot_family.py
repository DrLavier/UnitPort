"""Minimal robot-family lookup vendored from DEMO.

DEMO's ``src.system.training.robot_family`` was a thin wrapper around
the unified Robot Asset registry (``robot_assets.list_unified_assets``)
with a static seed table as fallback. RELEASE has neither the registry
nor the live overlay yet, so this module ships only the seed table.

# TODO: migrate to ``registers/robots.py`` once that module lands;
#       this is a temporary vendoring (see plan §3.2).
"""

from __future__ import annotations

from typing import Dict


GENERIC_LOCOMOTION_FAMILY = "generic_locomotion"


# Seed table — matches ``DEMO/src/system/training/robot_assets/seed_data.py``
# (DEFAULT_FAMILY_BY_MODEL). Hand-edited; do not auto-extend from URDF
# imports (RELEASE CLAUDE.md §1.2).
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
    "mini_pupper_v2": "quadruped",
}


def normalize_robot_type(robot_type: str) -> str:
    """Return a stable, lower-cased robot_type key for family lookup."""
    return str(robot_type or "").strip().lower()


def resolve_robot_family(robot_type: str) -> str:
    """Resolve a project robot_type into a template family key.

    Unknown or empty robot types fall back to the generic locomotion
    family so later template resolution remains safe.
    """
    return ROBOT_FAMILY_BY_TYPE.get(
        normalize_robot_type(robot_type),
        GENERIC_LOCOMOTION_FAMILY,
    )


__all__ = [
    "GENERIC_LOCOMOTION_FAMILY",
    "ROBOT_FAMILY_BY_TYPE",
    "normalize_robot_type",
    "resolve_robot_family",
]
