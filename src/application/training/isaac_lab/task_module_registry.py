# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""task_module_registry — compatibility shim. Canonical source: ``src/scripts/``.

This module used to hand-maintain its own ~1600-line copy of the reward /
termination / observation / discriminator registries plus the query machinery
(``lookup`` / ``query_registry`` / ``ALL_REGISTRIES`` / …). That made it a
second source of truth alongside the canonical per-file ``src/scripts/`` tree,
and the two drifted — ``termination_penalty`` lived only here, ``bad_orientation``
only in scripts, and several ``il_inline`` / ``il_params`` bodies (base_height,
gait, joint_deviation_l1, disc_*, feet_*) diverged. The compiler emitted *this*
module's versions while the UI listed scripts', so the registries silently
disagreed.

The literals + machinery were collapsed into ``src/scripts/`` (the declared
canonical home). The compiler (`env_cfg_compiler`) and validator
(`spec_validator`) now import from ``scripts`` directly; an env_cfg byte-diff
across the shipped sample canvases confirmed the switch is behaviour-preserving.

Everything below simply re-exports ``scripts`` so any remaining or dynamic
importer keeps working. **New code must import from ``scripts`` directly.**
This shim has no importers left in-tree and may be deleted in a follow-up.
"""

from __future__ import annotations

from scripts import (  # noqa: F401  — re-export, canonical source is scripts/
    ALG_ALL,
    ALG_AMP,
    ALG_PPO,
    ALG_SAC,
    ALG_TD3,
    ALL_ALGORITHMS,
    ALL_BACKENDS,
    BACKEND_ISAAC,
    BACKEND_NEWTON,
    BACKEND_SB3,
    IL_DISC_REGISTRY,
    IL_MOD_INLINE,
    IL_MOD_MDP,
    IL_MOD_VEL,
    IL_OBS_REGISTRY,
    IL_RECOMMENDED_LOCOMOTION_REWARD_TERMS,
    IL_REWARD_REGISTRY,
    IL_TERMINATION_REGISTRY,
    RECOMMENDED_LOCOMOTION_REWARD_TERMS,
    REWARD_REGISTRY,
    TERMINATION_REGISTRY,
    TaskModuleItem,
    lookup,
    recommended_reward_terms_for_backend,
)
from scripts.query import (  # noqa: F401
    ALL_REGISTRIES,
    iter_all_items,
    query_registry,
    validate_keys,
    writable_registry,
)

__all__ = [
    "TaskModuleItem",
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
    "REWARD_REGISTRY",
    "IL_REWARD_REGISTRY",
    "TERMINATION_REGISTRY",
    "IL_TERMINATION_REGISTRY",
    "IL_OBS_REGISTRY",
    "IL_DISC_REGISTRY",
    "ALL_REGISTRIES",
    "lookup",
    "query_registry",
    "writable_registry",
    "iter_all_items",
    "validate_keys",
    "RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "IL_RECOMMENDED_LOCOMOTION_REWARD_TERMS",
    "recommended_reward_terms_for_backend",
]
