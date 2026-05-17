"""Observation preset registry — Isaac Lab.

Migrated from ``DEMO/src/system/training/task_module_registry.py``
(``IL_OBS_REGISTRY`` portion). Each preset is a :class:`TaskModuleItem`
carrying just metadata (title / desc / family applicability) — there is
no inline Python source here, because Isaac Lab observation functions
are picked from ``isaaclab.envs.mdp.observations`` by **key name** at
compile time.

The IL config compiler (``isaac_lab_config_compiler._obs_term_func``,
to be migrated alongside the engine) maps each enabled key to the
corresponding ``mdp.<func_name>`` reference. Adding a brand-new
observation preset here is **not** sufficient on its own — the compiler
must also know how to bind the new key.

DEMO has no SB3 observation registry; only Isaac Lab is covered today.
The AMP-only ``amp_obs_terms.py`` (per-clip discriminator inputs) is a
separate concern and is intentionally NOT part of this preset library —
those terms are dispatched per-motion-clip at AMP runtime, not picked
by the user from a Canvas dropdown.
"""

from __future__ import annotations

from typing import Dict

from scripts.task_module import (
    ALL_FAMILIES,
    LEGGED_FAMILIES,
    LOCOMOTION_FAMILIES,
    TaskModuleItem,
    observation_item,
)


IL_OBS_REGISTRY: Dict[str, TaskModuleItem] = {
    "base_lin_vel": observation_item(
        key="base_lin_vel",
        title="Base Lin Vel",
        desc="Base linear velocity in body frame (3D).",
        applicable_families=ALL_FAMILIES,
    ),
    "base_ang_vel": observation_item(
        key="base_ang_vel",
        title="Base Ang Vel",
        desc="Base angular velocity in body frame (3D).",
        applicable_families=ALL_FAMILIES,
    ),
    "projected_gravity": observation_item(
        key="projected_gravity",
        title="Projected Gravity",
        desc="Gravity vector projected into body frame (3D).",
        applicable_families=ALL_FAMILIES,
    ),
    "velocity_command": observation_item(
        key="velocity_command",
        title="Velocity Command",
        desc="Commanded velocity target (3D: vx, vy, wz).",
        applicable_families=LOCOMOTION_FAMILIES,
    ),
    "joint_pos": observation_item(
        key="joint_pos",
        title="Joint Positions",
        desc="Joint position readings, relative to default pose (N-dim).",
        applicable_families=ALL_FAMILIES,
    ),
    "joint_vel": observation_item(
        key="joint_vel",
        title="Joint Velocities",
        desc="Joint velocity readings (N-dim).",
        applicable_families=ALL_FAMILIES,
    ),
    "last_action": observation_item(
        key="last_action",
        title="Last Action",
        desc="Previous policy output / action applied (N-dim).",
        applicable_families=ALL_FAMILIES,
    ),
    "height_scan": observation_item(
        key="height_scan",
        title="Height Scan",
        desc="Terrain height scanner readings around each foot (M-dim). "
             "Requires a height_scanner sensor on the canvas (Play Ground "
             "Setting → height_scan_enabled=true).",
        applicable_families=LEGGED_FAMILIES,
    ),
}


def il_obs_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_OBS_REGISTRY)


def default_il_obs_terms() -> Dict[str, float]:
    """Default IL observation terms for a fresh canvas — every standard
    proprioceptive input enabled, plus the velocity command. Excludes
    ``height_scan`` because it requires a ray-caster sensor on the canvas
    (the IL compiler skips the term when no scanner is present, but
    starting it disabled avoids a silent surprise).
    """
    return {
        k: IL_OBS_REGISTRY[k].default
        for k in (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_command",
            "joint_pos",
            "joint_vel",
            "last_action",
        )
    }


__all__ = [
    "IL_OBS_REGISTRY",
    "il_obs_registry",
    "default_il_obs_terms",
]
