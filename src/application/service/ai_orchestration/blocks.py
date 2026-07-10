# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Canvas blocks — the unit of agent work for AI Build v2.

A training canvas is conceptually five **blocks**: Robot selection, Actor &
Environment, Rewards, Training, Export. The :class:`~application.service.
ai_orchestration.harness.scheduler.Scheduler` walks these blocks in order,
giving the Orchestrator one small, verifiable scope at a time (fewer tokens,
higher accuracy) instead of one monolithic pass over the whole graph.

This module is **pure data + pure functions** (no Qt, no registry I/O at import
time): the block catalog, the deterministic tier→number tables for the
Requirements card, and the helpers that map a node back to its block (used by
the bounded repair loop to re-dispatch an issue to the block that owns it).

The block membership is keyed by node ``schema_id`` (a node's
``manifest.id``). New node types that join the canvas must be added to exactly
one block here — a schema_id with no block is reported by
:func:`unmapped_schema_ids` so it cannot silently fall outside agent scope (§8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

__all__ = [
    "CanvasBlock",
    "BLOCKS",
    "block_for_schema",
    "block_for_node_id",
    "unmapped_schema_ids",
    "LENGTH_TIERS",
    "INTENSITY_TIERS",
    "resolve_length",
    "resolve_intensity",
]


@dataclass(frozen=True)
class CanvasBlock:
    """One logical region of the canvas the agent processes as a unit."""

    id: str
    title_key: str
    default_title: str
    schema_ids: FrozenSet[str]


# Ordered: the scheduler processes blocks top-to-bottom. Membership is by node
# schema_id (== NodeManifest.id). Keep this the single source of block→schema
# mapping; do not branch on schema_id strings elsewhere.
BLOCKS: Tuple[CanvasBlock, ...] = (
    CanvasBlock(
        id="robot",
        title_key="canvas.ai_build.block_robot",
        default_title="Robot",
        schema_ids=frozenset({"robot"}),
    ),
    CanvasBlock(
        id="actor_env",
        title_key="canvas.ai_build.block_actor_env",
        default_title="Actor & Environment",
        schema_ids=frozenset(
            {
                "actor_setting",
                "physics_config",
                "play_ground_setting",
                "obs_action_config",
                "domain_rand",
                "env_assembler",
                "il_observation",
                "il_policy_network",
            }
        ),
    ),
    CanvasBlock(
        id="rewards",
        title_key="canvas.ai_build.block_rewards",
        default_title="Rewards",
        schema_ids=frozenset({"rewards", "multigated_reward", "terminations"}),
    ),
    CanvasBlock(
        id="training",
        title_key="canvas.ai_build.block_training",
        default_title="Training",
        schema_ids=frozenset(
            {
                "training_motion",
                "algorithm_config",
                "il_ppo_trainer",
                "amp_trainer",
                "discriminator",
                "task_config",
                "base_asset",
                "train",
                "stage_switch",
            }
        ),
    ),
    CanvasBlock(
        id="export",
        title_key="canvas.ai_build.block_export",
        default_title="Export",
        schema_ids=frozenset({"export", "eval_config", "vis_check"}),
    ),
)


def block_for_schema(schema_id: str) -> Optional[CanvasBlock]:
    """Return the block owning ``schema_id``, or None if it is unmapped.

    ``note`` (annotation) is intentionally unmapped — it is not training config.
    """
    sid = str(schema_id or "")
    for block in BLOCKS:
        if sid in block.schema_ids:
            return block
    return None


def block_for_node_id(snapshot: Dict[str, Any], node_id: str) -> Optional[CanvasBlock]:
    """Resolve a node instance id to its block via the canvas snapshot.

    ``snapshot`` is a ``to_workflow_dict()``-shape dict; nodes carry ``id`` and
    ``schema_id``. Returns None if the node or its block can't be resolved (the
    repair loop then falls back to the generic dispatch).
    """
    nid = str(node_id or "")
    if not nid:
        return None
    for node in snapshot.get("nodes", []) or []:
        if str(node.get("id")) == nid:
            return block_for_schema(str(node.get("schema_id") or ""))
    return None


def unmapped_schema_ids(all_schema_ids: FrozenSet[str]) -> FrozenSet[str]:
    """Schema ids present in the registry but not assigned to any block.

    Excludes ``note`` (a non-config annotation node). Used by tests to keep the
    block catalog exhaustive as the node catalog grows.
    """
    mapped: set = set()
    for block in BLOCKS:
        mapped |= block.schema_ids
    mapped.add("note")
    return frozenset(s for s in all_schema_ids if s not in mapped)


# =============================================================================
# Requirements tier → concrete number tables (the card's no-token decisions)
# =============================================================================
#
# Backend-aware because the units differ: IsaacLab counts PPO *iterations* and
# parallel GPU *envs*; SB3/MuJoCo counts *total_timesteps* and CPU *n_envs*.
# A blank tier with no numeric override resolves to None → the apply step leaves
# the node's own default untouched (it does NOT fabricate a value, §8).

#: Training length tier → trainer iteration/timestep budget.
LENGTH_TIERS: Dict[str, Dict[str, int]] = {
    "isaac_lab": {"short": 500, "medium": 1500, "long": 3000},
    "sb3_mujoco": {"short": 2_000_000, "medium": 10_000_000, "long": 30_000_000},
}

#: Compute-intensity tier → parallel environment count.
INTENSITY_TIERS: Dict[str, Dict[str, int]] = {
    "isaac_lab": {"light": 1024, "medium": 4096, "heavy": 8192},
    "sb3_mujoco": {"light": 4, "medium": 16, "heavy": 32},
}


def _resolve_tier(
    table: Dict[str, Dict[str, int]],
    backend: str,
    tier: str,
    override: Optional[int],
) -> Optional[int]:
    """Override wins; else the tier maps via the table; else None (leave default)."""
    if override is not None:
        return int(override)
    key = (tier or "").strip().lower()
    if not key:
        return None
    return table.get(str(backend or ""), {}).get(key)


def resolve_length(
    backend: str, tier: str, override: Optional[int] = None
) -> Optional[int]:
    """Resolve the training-length tier (or numeric override) to a budget."""
    return _resolve_tier(LENGTH_TIERS, backend, tier, override)


def resolve_intensity(
    backend: str, tier: str, override: Optional[int] = None
) -> Optional[int]:
    """Resolve the compute-intensity tier (or numeric override) to an env count."""
    return _resolve_tier(INTENSITY_TIERS, backend, tier, override)
