# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Task-module preset shared types.

Common dataclass and constants used by both reward presets
(``scripts.rewards.registry``) and termination presets
(``scripts.terminations.registry``).

Migrated from ``DEMO/src/system/training/task_module_registry.py`` —
the dataclass + factories were lifted out so the two preset registries
can live in their own packages without duplicating type definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


# ── Backend / algorithm compatibility constants ───────────────────────
# These tags let the UI filter which preset functions are applicable to
# the user's current training context (engine x algorithm).

BACKEND_SB3 = "sb3_mujoco"   # the SB3 training-engine id (single source; NOT "sb3", which is the stable_baselines3 *library* component probe)
BACKEND_ISAAC = "isaac_lab"
BACKEND_NEWTON = "newton"
ALL_BACKENDS: FrozenSet[str] = frozenset({BACKEND_SB3, BACKEND_ISAAC, BACKEND_NEWTON})

ALG_PPO = "PPO"
ALG_SAC = "SAC"
ALG_AMP = "AMP"
ALG_TD3 = "TD3"
ALG_ALL = "ALL"
ALL_ALGORITHMS: FrozenSet[str] = frozenset({ALG_PPO, ALG_SAC, ALG_AMP, ALG_TD3, ALG_ALL})

# Isaac Lab module routing constants
IL_MOD_MDP = "mdp"               # isaaclab.envs.mdp (core)
IL_MOD_VEL = "velocity_mdp"      # isaaclab_tasks...velocity.mdp
IL_MOD_INLINE = ""               # emitted inline in compiled config


# ── Robot family sets ─────────────────────────────────────────────────

LOCOMOTION_FAMILIES = frozenset({"quadruped", "biped", "wheeled", "generic_locomotion"})
LEGGED_FAMILIES = frozenset({"quadruped", "biped", "generic_locomotion"})
ALL_FAMILIES = frozenset(
    {"quadruped", "biped", "wheeled", "manipulator", "generic_locomotion"}
)


@dataclass(frozen=True)
class TaskModuleItem:
    key: str
    kind: str
    polarity: str
    title: str
    desc: str
    default: float
    min_value: float
    max_value: float
    step: float
    applicable_families: FrozenSet[str] = field(default_factory=frozenset)
    # Two-layer compatibility filters:
    #   backends   — which training engines support this function
    #                (sb3 / isaac_lab / newton)
    #   algorithms — which RL algorithms it applies to
    #                (PPO / SAC / AMP / TD3 / ALL)
    # "ALL" in algorithms means universally applicable.
    backends: FrozenSet[str] = field(default_factory=lambda: ALL_BACKENDS)
    algorithms: FrozenSet[str] = field(default_factory=lambda: frozenset({ALG_ALL}))

    # ── Isaac Lab compiler metadata ───────────────────────────────────
    # Populated only for IL rewards. The compiler reads these instead of
    # maintaining parallel hardcoded dicts.
    #
    #   il_func   — Isaac Lab function name (e.g. "track_lin_vel_xy_exp")
    #   il_module — module alias: "mdp" | "velocity_mdp" | "" (inline)
    #   il_params — extra RewTerm params template string (may contain
    #               ``{node_std}`` / ``{node_threshold}`` placeholders
    #               that the compiler substitutes from the Rewards node)
    #   il_inline — Python source for an inline function when the reward
    #               is not available in standard Isaac Lab packages.
    #               Empty string -> no inline needed.
    il_func: str = ""
    il_module: str = ""
    il_params: str = ""
    il_inline: str = ""

    # ── "Value" column metadata (function-internal tunable param) ─────
    # The Rewards node exposes ONE tunable scalar a reward function takes
    # internally (e.g. ``base_height(target_height=…)``) via a per-item
    # "Value" chip. Non-empty ``il_value_label`` ⇒ the UI renders that
    # chip and the IL ``il_params`` template carries the ``{item_value}``
    # placeholder the compiler substitutes from the canvas payload.
    #
    #   il_value_label   — column-header / chip label ("" = no Value chip)
    #   il_value_default — value when unset (0.0 = "auto", reward decides)
    #   il_value_min/max — popup bounds
    #   il_value_step    — popup step
    #   il_value_unit    — UI unit suffix (e.g. "m")
    #
    # NOTE: this is NOT for ``std`` / ``threshold`` reward-shaping knobs —
    # those get dedicated handling elsewhere.
    il_value_label: str = ""
    il_value_default: float = 0.0
    il_value_min: float = 0.0
    il_value_max: float = 0.0
    il_value_step: float = 0.01
    il_value_unit: str = ""

    # ── Per-joint-subset partitioning (缺口③) ─────────────────────────
    # When non-empty, this reward can be split into multiple per-joint-subset
    # instances from ONE canvas term — the legged_gym hip/arm/waist_dof_deviation
    # pattern. The value names the partition taxonomy the UI + compiler draw
    # from; today the only source is ``"pd_groups"`` (the family-keyed PD joint
    # groups: hip_x/hip_y/knee/shoulder_*/elbow/wrist_*/waist/...). The Rewards
    # node then stores a ``partitions`` map ``{pd_group_id: weight}`` (see
    # term_payload.parse_partitions); the compiler fans it out into one RewTerm
    # per partition, each ``SceneEntityCfg("robot", joint_names=[...])`` resolved
    # via JointIRResolver from that partition's IR-role regex. Empty ⇒ the term
    # runs on all joints (legacy scalar-weight path, byte-identical).
    il_partition_source: str = ""


def reward_item(
    *,
    key: str,
    polarity: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = ALL_FAMILIES,
    backends: FrozenSet[str] = ALL_BACKENDS,
    algorithms: FrozenSet[str] = frozenset({ALG_ALL}),
    il_func: str = "",
    il_module: str = "",
    il_params: str = "",
    il_inline: str = "",
    il_value_label: str = "",
    il_value_default: float = 0.0,
    il_value_min: float = 0.0,
    il_value_max: float = 0.0,
    il_value_step: float = 0.01,
    il_value_unit: str = "",
    il_partition_source: str = "",
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="reward",
        polarity=polarity,
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
        backends=backends,
        algorithms=algorithms,
        il_func=il_func,
        il_module=il_module,
        il_params=il_params,
        il_inline=il_inline,
        il_value_label=il_value_label,
        il_value_default=il_value_default,
        il_value_min=il_value_min,
        il_value_max=il_value_max,
        il_value_step=il_value_step,
        il_value_unit=il_value_unit,
        il_partition_source=il_partition_source,
    )


def termination_item(
    *,
    key: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = ALL_FAMILIES,
    backends: FrozenSet[str] = ALL_BACKENDS,
    algorithms: FrozenSet[str] = frozenset({ALG_ALL}),
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="termination",
        polarity="",
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
        backends=backends,
        algorithms=algorithms,
    )


def observation_item(
    *,
    key: str,
    title: str,
    desc: str,
    default: float = 1.0,
    applicable_families: FrozenSet[str] = ALL_FAMILIES,
    backends: FrozenSet[str] = frozenset({BACKEND_ISAAC}),
    algorithms: FrozenSet[str] = frozenset({ALG_ALL}),
) -> TaskModuleItem:
    """Build an ``observation``-kind preset item.

    Defaults reflect IL observation usage today: weight = 1.0 (binary
    enabled / disabled rather than a graded scalar), Isaac Lab backend
    (DEMO ships no SB3 obs registry).
    """
    return TaskModuleItem(
        key=key,
        kind="observation",
        polarity="",
        title=title,
        desc=desc,
        default=default,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        applicable_families=applicable_families,
        backends=backends,
        algorithms=algorithms,
    )


def discriminator_item(
    *,
    key: str,
    title: str,
    desc: str,
    il_inline: str,
) -> TaskModuleItem:
    """Build a ``discriminator``-kind preset item (AMP only).

    Discriminator entries carry no scalar weight — they are slot-bound
    to ``AMPDiscriminator._OVERRIDE_SLOTS`` and override class methods
    via ``il_inline`` Python source. The numeric fields are sentinel
    zeros so the unified RegistryModuleEditor row layout stays happy.

    The slot keys are part of the AMPDiscriminator contract; do NOT
    invent new keys without first extending ``_OVERRIDE_SLOTS`` in
    ``amp/algorithms/discriminator.py``.
    """
    return TaskModuleItem(
        key=key,
        kind="discriminator",
        polarity="",
        title=title,
        desc=desc,
        default=0.0,
        min_value=0.0,
        max_value=0.0,
        step=0.0,
        applicable_families=ALL_FAMILIES,
        backends=frozenset({BACKEND_ISAAC}),
        algorithms=frozenset({ALG_AMP}),
        il_inline=il_inline,
    )


__all__ = [
    "BACKEND_SB3",
    "BACKEND_ISAAC",
    "BACKEND_NEWTON",
    "ALL_BACKENDS",
    "ALG_PPO",
    "ALG_SAC",
    "ALG_AMP",
    "ALG_TD3",
    "ALG_ALL",
    "ALL_ALGORITHMS",
    "IL_MOD_MDP",
    "IL_MOD_VEL",
    "IL_MOD_INLINE",
    "LOCOMOTION_FAMILIES",
    "LEGGED_FAMILIES",
    "ALL_FAMILIES",
    "TaskModuleItem",
    "reward_item",
    "termination_item",
    "observation_item",
    "discriminator_item",
]
