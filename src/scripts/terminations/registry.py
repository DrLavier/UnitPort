"""Termination preset registry — SB3 + Isaac Lab.

Migrated from ``DEMO/src/system/training/task_module_registry.py``
(termination portion only). Each preset is a :class:`TaskModuleItem`
carrying a single float threshold so it fits the unified
``_RegistryModuleEditor`` row layout.
"""

from __future__ import annotations

from typing import Dict

from scripts.task_module import (
    ALL_FAMILIES,
    BACKEND_ISAAC,
    BACKEND_SB3,
    LEGGED_FAMILIES,
    LOCOMOTION_FAMILIES,
    TaskModuleItem,
    termination_item,
)


# SB3-only termination presets. The default ``backends=ALL_BACKENDS`` would
# let these surface in the Isaac Lab termination editor and corrupt the
# canvas with SB3-only keys (e.g. ``"timeout"`` instead of the IL
# ``"time_out"`` that env_cfg_compiler actually consumes). Scope every
# entry to ``{sb3}`` so query_registry(backend="isaac_lab") excludes them.
_SB3_ONLY: frozenset = frozenset({BACKEND_SB3})

TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = {
    "fall_threshold_roll": termination_item(
        key="fall_threshold_roll",
        title="Roll Threshold",
        desc="Terminate when roll exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_SB3_ONLY,
    ),
    "fall_threshold_pitch": termination_item(
        key="fall_threshold_pitch",
        title="Pitch Threshold",
        desc="Terminate when pitch exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=LEGGED_FAMILIES,
        backends=_SB3_ONLY,
    ),
    "min_height": termination_item(
        key="min_height",
        title="Minimum Height",
        desc="Terminate when the robot base drops below this height.",
        default=0.15,
        min_value=0.05,
        max_value=0.5,
        step=0.01,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_SB3_ONLY,
    ),
    "max_contact_impulse": termination_item(
        key="max_contact_impulse",
        title="Contact Impulse",
        desc="Terminate on excessive impact indicating unstable collapse.",
        default=250.0,
        min_value=50.0,
        max_value=500.0,
        step=5.0,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_SB3_ONLY,
    ),
    "timeout": termination_item(
        key="timeout",
        title="Timeout",
        desc="Terminate when the task exceeds its allowed duration.",
        default=1000.0,
        min_value=10.0,
        max_value=100000.0,
        step=10.0,
        backends=_SB3_ONLY,
    ),
    "joint_limit_violation": termination_item(
        key="joint_limit_violation",
        title="Joint Limit",
        desc="Terminate when joints exceed safe configured limits. "
             "Threshold is a count: 3 = terminate when 3 or more joints exceed their range.",
        default=3.0,
        min_value=1.0,
        max_value=12.0,
        step=1.0,
        applicable_families=ALL_FAMILIES,
        backends=_SB3_ONLY,
    ),
    "self_collision": termination_item(
        key="self_collision",
        title="Self Collision",
        desc="Terminate when forbidden self-collision is detected.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        applicable_families=frozenset({"manipulator"}),
        backends=_SB3_ONLY,
    ),
}


def termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(TERMINATION_REGISTRY)


def default_termination_conditions() -> Dict[str, float]:
    return {
        key: TERMINATION_REGISTRY[key].default
        for key in (
            "fall_threshold_roll",
            "fall_threshold_pitch",
            "min_height",
            "max_contact_impulse",
            "joint_limit_violation",
        )
    }


# ─── Isaac Lab termination registry ─────────────────────────────────────────
# These mirror the DoneTerm entries the Isaac Lab compiler used to build from
# standalone toggle params (enable_timeout / enable_illegal_contact /
# enable_base_height). Each item carries a single float threshold so it fits
# the unified _RegistryModuleEditor row layout, exactly like the SB3
# termination registry. illegal_contact's bodies regex list is kept as a
# separate hidden node parameter (illegal_contact_bodies) — the items list
# only owns the contact-force threshold.

# Isaac Lab-only termination presets. Scoping these to ``{isaac_lab}``
# keeps SB3 canvases from picking up keys (``time_out`` / ``base_height``)
# that env_cfg_compiler interprets exclusively. ``base_height`` also exists
# as a reward in SB3 with different semantics, but here the key is owned
# by Isaac Lab's DoneTerm path.
_IL_ONLY: frozenset = frozenset({BACKEND_ISAAC})

IL_TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = {
    "time_out": termination_item(
        key="time_out",
        title="Episode Timeout",
        desc="Terminate when episode wall-clock duration (s) exceeds this limit. "
             "Mapped to mdp.time_out in the compiled Isaac Lab task.",
        default=20.0,
        min_value=1.0,
        max_value=300.0,
        step=0.5,
        applicable_families=ALL_FAMILIES,
        backends=_IL_ONLY,
    ),
    "illegal_contact": termination_item(
        key="illegal_contact",
        title="Illegal Contact",
        desc="Terminate when net contact force on the configured bodies "
             "exceeds this Newton threshold. Body regex list is configured "
             "via the node's illegal_contact_bodies parameter.",
        default=1.0,
        min_value=0.1,
        max_value=200.0,
        step=0.1,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL_ONLY,
    ),
    "base_height": termination_item(
        key="base_height",
        title="Base Height",
        desc="Terminate when the robot base drops below this minimum height (m).",
        default=0.2,
        min_value=0.05,
        max_value=1.0,
        step=0.01,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL_ONLY,
    ),
    "bad_orientation": termination_item(
        key="bad_orientation",
        title="Bad Orientation",
        desc="Terminate when the base tilts beyond this projected-gravity "
             "deviation (rad). Mapped to mdp.bad_orientation in the compiled "
             "Isaac Lab task.",
        default=0.7,
        min_value=0.1,
        max_value=1.5,
        step=0.05,
        applicable_families=LOCOMOTION_FAMILIES,
        backends=_IL_ONLY,
    ),
}


def il_termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_TERMINATION_REGISTRY)


def default_il_termination_conditions() -> Dict[str, float]:
    """Default IL termination items for a fresh canvas — keep timeout and
    illegal-contact on by default to mirror the previous toggle defaults
    (enable_timeout=true, enable_illegal_contact=true, enable_base_height=false).
    """
    return {
        "time_out": IL_TERMINATION_REGISTRY["time_out"].default,
        "illegal_contact": IL_TERMINATION_REGISTRY["illegal_contact"].default,
    }


__all__ = [
    "TERMINATION_REGISTRY",
    "IL_TERMINATION_REGISTRY",
    "termination_registry",
    "il_termination_registry",
    "default_termination_conditions",
    "default_il_termination_conditions",
]
